# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""A typical-LLM-shaped compile smoke test.

Every other file in this directory targets one fuser edge case in
isolation, under the new MAP-dialect fusion system. This one instead
checks that a naively-written, typical LLM -- sliding-window attention
with QK norm and RoPE, SwiGLU MLP, several blocks -- compiles at all,
with mixed static/dynamic shapes (symbolic ``batch``/``seq_len``, static
``window_size``/``num_heads``/``head_size``) in the same graph.
Compile-only, no correctness/numeric checks and no benchmarking.

Not env-gated: this graph shape aborts the process
(``MAP::SliceSpecAttr::verifyInvariants``) under
``MAX_GC_USE_ADV_FUSION=1`` today -- a known new-system gap, not
addressed here -- so this runs under the legacy pipeline instead.

``mo.mha.no_cache`` (the attention op `_AttentionBlock` calls through
``F.custom``) has only a GPU kernel registration
(``builtin_kernels/attention.mojo``'s ``FlashAttentionGPU``), so this test
is skipped without one, like `test_tile_based_fusion.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module, ModuleList, RMSNorm
from max.experimental.tensor import Tensor, TensorType
from max.graph import DeviceRef


@dataclass
class _Config:
    hidden_size: int
    projection_size: int
    num_heads: int
    window_size: int
    num_blocks: int


_SNIFFTEST_CONFIG = _Config(
    hidden_size=64,
    projection_size=128,
    num_heads=4,
    window_size=32,
    num_blocks=2,
)


class _RopeInit(Module[..., Tensor]):
    pos_inv_freqs: Tensor

    def __init__(self, num_pos_embeddings: int, base: float = 10000.0) -> None:
        super().__init__()
        exponent = Tensor.arange(
            num_pos_embeddings, step=2, dtype=DType.float32
        )
        self.pos_inv_freqs = base ** (-exponent / num_pos_embeddings)

    @F.functional
    def forward(self, num_events: int, device: DeviceRef) -> Tensor:
        positions = Tensor.arange(
            num_events,
            device=device,  # type: ignore[arg-type]
            dtype=DType.float32,
        )
        theta = F.outer(positions, self.pos_inv_freqs).unsqueeze(0)
        theta = F.concat([theta, theta], axis=-1)
        cos = F.cos(theta).unsqueeze(axis=2)
        sin = F.sin(theta).unsqueeze(axis=2)
        return F.stack([cos, sin], axis=-1)


class _LayerNorm(Module[[Tensor], Tensor]):
    weight: Tensor
    bias: Tensor
    eps: float

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = Tensor.ones([hidden_size])
        self.bias = Tensor.zeros([hidden_size])
        self.eps = eps

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class _AttentionBlock(Module[..., tuple[Tensor, Tensor, Tensor]]):
    norm: _LayerNorm
    qkv_proj: Linear
    out_proj: Linear
    qk_norm: RMSNorm
    hidden_size: int
    window_size: int
    num_heads: int
    head_size: int

    def __init__(
        self, hidden_size: int, num_heads: int, window_size: int
    ) -> None:
        super().__init__()
        self.norm = _LayerNorm(hidden_size)
        self.qkv_proj = Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qk_norm = RMSNorm(self.head_size)

    def _apply_rope(self, x: Tensor, rope: Tensor) -> Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = F.concat([-x2, x1], axis=-1).cast(DType.float32)
        return (x * rope[..., 0] + rotated * rope[..., 1]).cast(x.dtype)

    @F.functional
    def forward(
        self, x: Tensor, rope: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_tokens, _ = x.shape
        y = self.norm(x)
        qkv = self.qkv_proj(y)
        query = qkv[..., : self.hidden_size]
        key = qkv[..., self.hidden_size : 2 * self.hidden_size]
        value = qkv[..., 2 * self.hidden_size :]

        query = query.reshape(
            (batch_size, num_tokens, self.num_heads, self.head_size)
        )
        key = key.reshape(
            (batch_size, num_tokens, self.num_heads, self.head_size)
        )
        value = value.reshape(
            (batch_size, num_tokens, self.num_heads, self.head_size)
        )

        query = self.qk_norm(query)
        key = self.qk_norm(key)
        query = self._apply_rope(query, rope)
        key = self._apply_rope(key, rope)

        key = F.concat([k_cache, key], axis=1)
        value = F.concat([v_cache, value], axis=1)
        new_k_cache = key[:, -self.window_size :]
        new_v_cache = value[:, -self.window_size :]

        attn_out = F.custom(
            "mo.mha.no_cache",
            query.device,
            [
                query,
                key,
                value,
                F.constant(
                    1 / math.sqrt(self.head_size),
                    dtype=DType.float32,
                    device=DeviceRef.CPU(),
                ),
            ],
            [query.type],
            parameters={
                "mask_str": "sliding_window_causal",
                "local_window_size": self.window_size,
            },
        )[0]

        attn_out = attn_out.reshape([batch_size, num_tokens, self.hidden_size])
        return self.out_proj(attn_out) + x, new_k_cache, new_v_cache

    def state_types(
        self, batch_size: int, device: DeviceRef, precision: DType
    ) -> list[TensorType]:
        shape = (batch_size, self.window_size, self.num_heads, self.head_size)
        return [
            TensorType(shape=shape, device=device, dtype=precision),
            TensorType(shape=shape, device=device, dtype=precision),
        ]


class _GatedMlpBlock(Module[[Tensor], Tensor]):
    norm: _LayerNorm
    up_and_gate: Linear
    down: Linear
    projection_size: int

    def __init__(self, hidden_size: int, projection_size: int) -> None:
        super().__init__()
        self.norm = _LayerNorm(hidden_size)
        self.up_and_gate = Linear(hidden_size, 2 * projection_size)
        self.down = Linear(projection_size, hidden_size)
        self.projection_size = projection_size

    @F.functional
    def forward(self, x: Tensor) -> Tensor:
        y = self.up_and_gate(self.norm(x))
        up, gate = (
            y[..., : self.projection_size],
            y[..., self.projection_size :],
        )
        return self.down(F.silu(up) * gate) + x


class _Block(Module[..., tuple[Tensor, Tensor, Tensor]]):
    attn: _AttentionBlock
    mlp: _GatedMlpBlock

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        projection_size: int,
    ) -> None:
        super().__init__()
        self.attn = _AttentionBlock(hidden_size, num_heads, window_size)
        self.mlp = _GatedMlpBlock(hidden_size, projection_size)

    @F.functional
    def forward(
        self, x: Tensor, rope: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        x, new_k, new_v = self.attn(x, rope, k_cache, v_cache)
        return self.mlp(x), new_k, new_v

    def state_types(
        self, batch_size: int, device: DeviceRef, precision: DType
    ) -> list[TensorType]:
        return self.attn.state_types(batch_size, device, precision)


class _Model(Module[..., tuple[Tensor, ...]]):
    rope_init: _RopeInit
    blocks: ModuleList

    def __init__(self, config: _Config) -> None:
        super().__init__()
        self.config = config
        self.rope_init = _RopeInit(config.hidden_size // config.num_heads)
        self.blocks = ModuleList(
            [
                _Block(
                    config.hidden_size,
                    config.num_heads,
                    config.window_size,
                    config.projection_size,
                )
                for _ in range(config.num_blocks)
            ]
        )

    @F.functional
    def forward(self, x: Tensor, *kv_cache: Tensor) -> tuple[Tensor, ...]:
        _, num_tokens, _ = x.shape
        rope = self.rope_init(num_tokens, x.device)
        new_kv: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            x, new_k, new_v = block(
                x, rope, kv_cache[i * 2], kv_cache[i * 2 + 1]
            )
            new_kv.extend([new_k, new_v])
        return x, *new_kv

    def state_types(
        self, batch_size: int, device: DeviceRef, precision: DType
    ) -> list[TensorType]:
        types: list[TensorType] = []
        for block in self.blocks:
            types.extend(
                cast(_Block, block).state_types(batch_size, device, precision)
            )
        return types


def test_typical_llm_compiles_with_mixed_static_dynamic_shapes() -> None:
    device_ref = DeviceRef.GPU()
    cfg = _SNIFFTEST_CONFIG
    precision = DType.bfloat16

    with F.lazy():
        max_model = _Model(cfg).to(Accelerator())

    head_size = cfg.hidden_size // cfg.num_heads
    kv_cache_type = TensorType(
        shape=("batch", cfg.window_size, cfg.num_heads, head_size),
        device=device_ref,
        dtype=precision,
    )
    input_types = [
        TensorType(
            shape=("batch", "seq_len", cfg.hidden_size),
            device=device_ref,
            dtype=precision,
        ),
        *([kv_cache_type] * (2 * cfg.num_blocks)),
    ]

    compiled = max_model.compile(
        *input_types, weights=dict(max_model.parameters)
    )
    assert compiled is not None
