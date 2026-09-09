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

"""Attention layer for Kimi K2.5 vision tower."""

from __future__ import annotations

import math
from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.sharding.action import ActionSet
from max.experimental.sharding.cost import force_replicated_action_set
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged_gpu as _flash_attention_ragged_gpu,
)


def _flash_attention_ragged_gpu_rule(
    q: TensorLayout,
    k: TensorLayout,
    v: TensorLayout,
    input_row_offsets: TensorLayout,
    max_seq_len: TensorLayout,
    *extras: Any,
) -> ActionSet:
    """Replicated on every operand: the vision tower is not sharded."""
    return force_replicated_action_set(q, k, v, input_row_offsets, max_seq_len)


flash_attention_ragged_gpu = F.functional(
    _flash_attention_ragged_gpu, rule=_flash_attention_ragged_gpu_rule
)


class Attention(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """QKV-packed self-attention for the Kimi K2.5 vision encoder.

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension.
        has_bias: Whether linear projections include bias terms.
        head_dim: Dimension per attention head. Defaults to
            ``hidden_dim // num_heads``.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        has_bias: bool = False,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = (
            head_dim if head_dim is not None else hidden_dim // num_heads
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.wqkv = Linear(
            in_dim=hidden_dim, out_dim=hidden_dim * 3, bias=has_bias
        )
        self.wo = Linear(in_dim=hidden_dim, out_dim=hidden_dim, bias=has_bias)

    @staticmethod
    def _apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
        """Applies 2-D rotary position embedding to a query or key tensor.

        Args:
            x: Tensor of shape (tot_seqlens, num_heads, head_dim).
            freqs_cis: Precomputed [cos, sin] pairs of shape
                (tot_seqlens, head_dim // 2, 2).

        Returns:
            Tensor with the same shape as *x* after RoPE rotation.
        """
        orig_dtype = x.dtype
        x = x.cast(DType.float32)

        complex_x = F.as_interleaved_complex(x)
        x_re = complex_x[..., 0]
        x_im = complex_x[..., 1]

        freqs = freqs_cis.unsqueeze(1).cast(DType.float32)
        freqs_re = freqs[..., 0]
        freqs_im = freqs[..., 1]

        rope_re = x_re * freqs_re - x_im * freqs_im
        rope_im = x_re * freqs_im + x_im * freqs_re

        result = F.stack([rope_re, rope_im], axis=-1)
        result = F.reshape(result, x.shape)
        return result.cast(orig_dtype)

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        rope_freqs_cis: Tensor,
    ) -> Tensor:
        """Compute self-attention with packed QKV projection.

        Args:
            x: Input tensor of shape (n_patches, hidden_dim).
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            rope_freqs_cis: Precomputed [cos, sin] pairs of shape
                (n_patches, head_dim // 2, 2).

        Returns:
            Output tensor of shape (n_patches, hidden_dim).
        """
        n_patches = x.shape[0]

        xqkv = self.wqkv(x)

        # Reshape to [n_patches, 3, num_heads, head_dim] and split Q, K, V.
        xqkv = xqkv.reshape([n_patches, 3, self.num_heads, self.head_dim])
        xq, xk, xv = xqkv.split([1, 1, 1], axis=1)
        xq = xq.squeeze(1)
        xk = xk.squeeze(1)
        xv = xv.squeeze(1)

        xq = self._apply_rope(xq, rope_freqs_cis)
        xk = self._apply_rope(xk, rope_freqs_cis)

        attn_out = flash_attention_ragged_gpu(
            q=xq,
            k=xk,
            v=xv,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )

        attn_out = attn_out.reshape([n_patches, -1])

        return self.wo(attn_out)
