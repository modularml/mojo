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

"""Encoder-only attention without KV cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu

if TYPE_CHECKING:
    from max.experimental.nn.common_layers.rotary_embedding import (
        RotaryEmbedding,
    )

flash_attention_gpu = F.functional(_flash_attention_gpu)


class EncoderAttention(Module[..., Tensor]):
    """Encoder-only attention without KV cache."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = scale

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(hidden_size, q_dim, bias=False)
        self.k_proj = Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, hidden_size, bias=False)

    def _apply_rope(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        """Apply rotary position embedding (non-interleaved format).

        Args:
            x: Input tensor with shape [seq_len, n_heads, head_dim].
            rope: RotaryEmbedding module.

        Returns:
            Tensor with RoPE applied, same shape as input.
        """
        seq_len = x.shape[0]
        head_dim = x.shape[2]
        half_dim = head_dim // 2

        # ``rope.freqs_cis`` is precomputed on CPU (see the encoder's
        # ``__init__``) to match the legacy graph API's ``max.nn``
        # RotaryEmbedding precision; transfer the slice to ``x.device``
        # so the bf16 multiplication below stays on-device.
        freqs_cis = rope.freqs_cis.to(x.device)
        freqs = freqs_cis[:seq_len, :]

        if len(freqs.shape) == 2:
            d0, d1 = freqs.shape
            freqs = F.reshape(freqs, (d0, d1 // 2, 2))

        freqs = F.cast(freqs, x.dtype)

        cos = F.unsqueeze(freqs[:, :, 0], 1)
        sin = F.unsqueeze(freqs[:, :, 1], 1)

        x_re = x[:, :, :half_dim]
        x_im = x[:, :, half_dim:]

        rope_re = (x_re * cos) - (x_im * sin)
        rope_im = (x_re * sin) + (x_im * cos)

        return F.concat((rope_re, rope_im), axis=-1)

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat KV heads for GQA (Grouped Query Attention).

        Args:
            x: Input tensor with shape [seq_len, n_kv_heads, head_dim]
            n_rep: Number of times to repeat each head

        Returns:
            Tensor with shape [seq_len, n_kv_heads * n_rep, head_dim]
        """
        if n_rep == 1:
            return x

        seq_len = x.shape[0]
        n_kv_heads = x.shape[1]
        head_dim = x.shape[2]

        # [S, H_kv, D] -> [S, H_kv, 1, D] -> [S, H_kv, n_rep, D] -> [S, H, D]
        # Use concat instead of tile: tile has no GPU implementation and forces
        # a CPU round-trip (DtoH + tile + HtoD) for every layer.
        x = F.unsqueeze(x, 2)
        x = F.concat([x] * n_rep, axis=2)
        x = F.reshape(x, (seq_len, n_kv_heads * n_rep, head_dim))

        return x

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        """Forward pass computing causal self-attention.

        Args:
            x: Input tensor with shape [total_seq_len, hidden_dim]
            rope: RotaryEmbedding module

        Returns:
            Output tensor with shape [total_seq_len, hidden_dim]
        """
        total_seq_len = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = F.reshape(q, (total_seq_len, self.n_heads, self.head_dim))
        k = F.reshape(k, (total_seq_len, self.n_kv_heads, self.head_dim))
        v = F.reshape(v, (total_seq_len, self.n_kv_heads, self.head_dim))

        q = self._apply_rope(q, rope)
        k = self._apply_rope(k, rope)

        # GQA: expand K, V if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        # flash_attention_gpu expects [B, S, heads, head_dim]
        q = F.unsqueeze(q, 0)
        k = F.unsqueeze(k, 0)
        v = F.unsqueeze(v, 0)

        attn_out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = F.squeeze(attn_out, 0)
        attn_out = F.reshape(attn_out, (total_seq_len, -1))
        return self.o_proj(attn_out)
