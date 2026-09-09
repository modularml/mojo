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

"""Gemma4 attention layer for the ModuleV3 API."""

from __future__ import annotations

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
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams

from .rms_norm import Gemma4RMSNorm


class Gemma4Attention(Module[..., Tensor]):
    """Gemma4 attention: QKV norms, k==v on global layers, dual geometry."""

    def __init__(
        self,
        *,
        rope_global: RotaryEmbedding,
        rope_local: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_global_key_value_heads: int,
        attention_k_eq_v: bool,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx_in_cache: int,
        is_sliding: bool,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        # Attention scaling is absorbed by the qk-norms (graph arch does the
        # same); do not apply 1/sqrt(head_dim).
        self.scale = 1.0
        self.use_local = is_sliding
        self.layer_idx_in_cache = layer_idx_in_cache
        self.local_window_size = local_window_size

        # MultiKVCacheParams leaf carries this layer type's head_dim
        # (256 sliding / 512 global for 31B).
        self.head_dim = kv_params.head_dim
        num_kv = (
            num_key_value_heads if is_sliding else num_global_key_value_heads
        )
        self.num_key_value_heads = num_kv
        self.q_weight_dim = self.head_dim * num_attention_heads
        self.kv_weight_dim = self.head_dim * num_kv

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=qk_norm_eps, with_weight=False
        )

        # Global layers with attention_k_eq_v share K weights for V: the
        # checkpoint has no v_proj there.
        self._has_v_proj = not (attention_k_eq_v and not is_sliding)
        self.q_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )
        if self._has_v_proj:
            self.v_proj = ColumnParallelLinear(
                in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
            )
        self.o_proj = RowParallelLinear(
            in_dim=self.q_weight_dim, out_dim=hidden_size, bias=False
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        *,
        input_row_offsets: Tensor,
    ) -> Tensor:
        layer_idx = F.constant(
            self.layer_idx_in_cache, DType.uint32, device=CPU()
        )
        per_device_n_heads = self.n_heads // kv_collection.n_devices
        head_dim = self.head_dim
        q_dim, kv_dim = self.q_weight_dim, self.kv_weight_dim
        num_kv_heads = self.num_key_value_heads

        x_q = self.q_proj(x)
        x_k = self.k_proj(x)
        x_v = self.v_proj(x) if self._has_v_proj else x_k

        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )
        x_v = self.v_norm(x_v.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )

        qkv = F.concat([x_q, x_k, x_v], axis=-1)
        rope = self.rope_local if self.use_local else self.rope_global

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=per_device_n_heads,
            interleaved=rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, head_dim))

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size if self.use_local else -1,
        )
        attn_out = attn_out.reshape((-1, self.n_heads * head_dim))
        return self.o_proj(attn_out)
