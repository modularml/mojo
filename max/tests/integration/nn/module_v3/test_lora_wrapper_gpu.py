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
"""Multi-adapter SGMV LoRA on a fused-QKV QKVLinear.

The serving swap: a transformer block's ``qkv`` is a callable ``QKVLinear``
(GQA: q_dim != kv_dim); wrapping it with ``LoRA`` adds a per-projection
multi-adapter delta (one adapter for q, k, v each -- PEFT semantics), computed
with the SGMV kernels and concatenated back onto the fused qkv output. Adapters
(per projection) and routing (shared) arrive as extra graph inputs, distributed
via ``lora_parameters`` / ``lora_layers``. Verified against a PyTorch reference.
GPU-only.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.nn import (
    LoRA,
    Module,
    lora_layers,
    lora_parameters,
)
from max.experimental.nn.common_layers.linear import QKVLinear
from max.experimental.tensor import Tensor, TensorType

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU"
)

IN_DIM, Q_DIM, KV_DIM = 128, 128, 32  # GQA: q_dim != kv_dim
OUT_DIM = Q_DIM + 2 * KV_DIM
PROJ_DIMS = (Q_DIM, KV_DIM, KV_DIM)
MAX_LORAS, RANK, MAX_LORA_SEQ_LEN = 3, 8, 64
RTOL, ATOL = 1e-3, 5e-3


def rand(*shape: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.bfloat16) * 0.02


class Block(Module[[Tensor], Tensor]):
    qkv: Module[[Tensor], Tensor]

    def __init__(self) -> None:
        self.qkv = QKVLinear(IN_DIM, Q_DIM, KV_DIM, stacked=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.qkv(x)


class MyModel(Module[..., Tensor]):
    block: Block

    def __init__(self) -> None:
        self.block = Block()

    def forward(
        self,
        x: Tensor,
        lora_ids: Tensor,
        grouped_offsets: Tensor,
        end_idx: Tensor,
        *adapters: Tensor,
    ) -> Tensor:
        for _, layer in lora_layers(self):
            layer.set_lora_batch_info(lora_ids, grouped_offsets, end_idx)
        for value, (_, slot) in zip(
            adapters, lora_parameters(self), strict=True
        ):
            slot.set(value)
        return self.block(x)


def _end_carrier(lora_end: int) -> npt.NDArray[np.int64]:
    carrier = np.zeros(lora_end, dtype=np.int64)
    if lora_end:
        carrier[0] = lora_end
    return carrier


def test_wrap_stacked_qkv_per_projection_sgmv() -> None:
    device = Accelerator()
    model = MyModel().to(device)

    # Serving swap: wrap the fused-QKV QKVLinear in place.
    assert isinstance(model.block.qkv, QKVLinear)
    model.block.qkv = LoRA(
        model.block.qkv,
        max_num_loras=MAX_LORAS,
        max_lora_rank=RANK,
        max_lora_seq_len=MAX_LORA_SEQ_LEN,
    )
    assert len(list(lora_layers(model))) == 1
    # Fused qkv -> two adapter slots (fused A, fused B), not six.
    assert len(list(lora_parameters(model))) == 2

    # The wrapper is invisible to naming: the base weight keeps the exact path
    # it had unwrapped (``block.qkv.weight``), with neither the wrapper's own
    # attribute nor its ``module`` holder introduced into the path. Adapters
    # ride in as inputs, not parameters.
    param_names = set(dict(model.parameters))
    assert "block.qkv.weight" in param_names
    assert not any(".module" in name for name in param_names)

    wrapped = model.block.qkv
    assert isinstance(wrapped, LoRA)
    base = wrapped.module
    assert isinstance(base, QKVLinear)
    w = torch.from_dlpack(base.weight.to(CPU()))

    compiled = model.compile(
        TensorType(DType.bfloat16, ["tokens", IN_DIM], device=device),
        TensorType(DType.int32, ["lora_ids"], device=device),
        TensorType(DType.uint32, ["grouped_offsets"], device=device),
        TensorType(DType.int64, ["end_idx"], device=CPU()),
        *(slot.type for _, slot in lora_parameters(model)),
    )

    # Ragged batch: two adapter groups (12 + 8 rows), then a base-only tail.
    tokens = 30
    x = rand(tokens, IN_DIM, seed=1)
    ids = np.array([1, 2], dtype=np.int32)
    offsets = np.array([0, 12, 20], dtype=np.uint32)
    lora_end = 20

    # Per-projection (a, b): a is [loras, rank, in]; b is [loras, proj_dim, rank].
    a_list = [rand(MAX_LORAS, RANK, IN_DIM, seed=10 + p) for p in range(3)]
    b_list = [
        rand(MAX_LORAS, PROJ_DIMS[p], RANK, seed=20 + p) for p in range(3)
    ]
    # The fused qkv slots are the per-projection weights stacked: A on the rank
    # axis -> [loras, 3*rank, in]; B on the output axis -> [loras, q+2*kv, rank].
    a_fused = torch.cat(a_list, dim=1)
    b_fused = torch.cat(b_list, dim=1)
    adapter_args = [a_fused, b_fused]

    out = compiled(
        Tensor(x, device=device),
        Tensor(ids, device=device),
        Tensor(offsets, device=device),
        Tensor(_end_carrier(lora_end), device=CPU()),
        *(Tensor(t, device=device) for t in adapter_args),
    )
    actual = torch.from_dlpack(out.to(CPU()))

    expected = x @ w.T  # [tokens, q + 2*kv]
    for g in range(len(ids)):
        start, end = int(offsets[g]), min(int(offsets[g + 1]), lora_end)
        if end > start:
            xg = x[start:end]
            deltas = [
                xg @ a_list[p][ids[g]].T @ b_list[p][ids[g]].T for p in range(3)
            ]
            expected[start:end] += torch.cat(deltas, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
