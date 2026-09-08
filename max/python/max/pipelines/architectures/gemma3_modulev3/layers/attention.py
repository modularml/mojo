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

"""Gemma3 Attention Layer for the ModuleV3 API."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams

from .rms_norm import Gemma3RMSNorm
from .rotary_embedding import Llama3RotaryEmbedding


class Gemma3Attention(Module[..., Tensor]):
    """Gemma3 attention with QK normalization and sliding window support."""

    def __init__(
        self,
        *,
        rope_global: Llama3RotaryEmbedding,
        rope_local: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        is_sliding: bool,
        sliding_window_pattern: int = 6,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.has_bias = has_bias
        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.qk_norm_eps = qk_norm_eps
        self.kv_params = kv_params

        self.q_norm = Gemma3RMSNorm(kv_params.head_dim, eps=qk_norm_eps)
        self.k_norm = Gemma3RMSNorm(kv_params.head_dim, eps=qk_norm_eps)

        q_weight_dim = kv_params.head_dim * num_attention_heads
        kv_weight_dim = kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_weight_dim
        self.kv_weight_dim = kv_weight_dim

        self.q_proj = ColumnParallelLinear(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            bias=has_bias,
        )
        self.k_proj = ColumnParallelLinear(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            bias=has_bias,
        )
        self.v_proj = ColumnParallelLinear(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            bias=has_bias,
        )
        self.o_proj = RowParallelLinear(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            bias=False,
        )

    @property
    def wqkv(self) -> Tensor:
        """The concatenation of q, k, and v weight vectors."""
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        n_devices = kv_collection.n_devices
        per_device_n_heads = self.n_heads // n_devices

        head_dim = self.kv_params.head_dim
        q_dim = self.q_weight_dim
        kv_dim = self.kv_weight_dim
        num_kv_heads = kv_dim // head_dim

        # QKV matmul: graph-level weight concat, then a single matmul.
        wqkv = self.wqkv
        qkv = x @ wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        # Split into Q, K, V and apply per-head QK norm. The reshape rule
        # resolves `-1` per-shard via the placement's `local_dim`, so
        # plain global sizes here produce correct per-shard targets.
        x_q, x_k, x_v = qkv.split([q_dim, kv_dim, kv_dim], axis=-1)
        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )

        # Re-concat and apply RoPE + KV cache store.
        qkv = F.concat([x_q, x_k, x_v], axis=-1)

        rope = self.rope_local if self.is_sliding else self.rope_global

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=per_device_n_heads,
            interleaved=rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, head_dim))

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.is_sliding
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = attn_out.reshape((-1, self.n_heads * head_dim))
        return self.o_proj(attn_out)
