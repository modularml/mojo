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

"""MiniCPM multi-head attention with standard split-half RoPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..minicpm import MiniCPMRope
import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


class MiniCPMAttention(Module[..., Tensor]):
    """Plain multi-head attention for MiniCPM (no GQA)."""

    def __init__(
        self,
        *,
        rope: MiniCPMRope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float | None = None,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        q_dim = self.kv_params.head_dim * num_attention_heads
        kv_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_dim

        self.q_proj = Linear(in_dim=hidden_size, out_dim=q_dim, bias=has_bias)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.o_proj = Linear(in_dim=q_dim, out_dim=hidden_size, bias=False)

    def forward(
        self, x: Tensor, kv_collection: PagedCacheValues, **kwargs
    ) -> Tensor:
        """Compute MHA attention for a ragged batch using split-half RoPE.

        Steps:
            1. Project Q/K/V.
            2. Apply RoPE and store K/V via ``rope_split_store_ragged``.
            3. Run flash attention with a causal mask.
            4. Project the attention output back to hidden dimension.

        Args:
            x: Input hidden states, shape ``(total_seq_len, hidden_size)``.
            kv_collection: Paged KV cache for the current layer.
            **kwargs: Must contain ``input_row_offsets`` — uint32 tensor of
                ragged-batch row delimiters.

        Returns:
            Attended output, shape ``(total_seq_len, hidden_size)``.
        """

        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qkv = F.concat([q, k, v], axis=-1)

        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)

        # interleaved=False → split-half (GPT-NeoX/Llama) rotation.
        # This matches MiniCPM's `rotate_half`-based apply_rotary_pos_emb.
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=False,  # MiniCPM-specific — standard RoPE
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)
