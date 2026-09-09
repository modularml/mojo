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

from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.kernels import (
    masked_flash_attention_gpu as _masked_flash_attention_gpu,
)

from .rotary_embedding import RotaryEmbedding

masked_flash_attention_gpu = F.functional(_masked_flash_attention_gpu)


class EncoderAttention(Module[..., Tensor]):
    """Encoder-only attention without KV cache (Qwen3: interleaved RoPE via rope.forward)."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        rms_norm_eps: float = 1e-6,
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

        # Qwen3: Q/K norm over head_dim before RoPE (eps matches config.rms_norm_eps)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

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

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        attention_bias: Tensor,
    ) -> Tensor:
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

        # Qwen3: norm over head_dim (per-head), then RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # module_v3.common_layers RotaryEmbedding.forward expects 4D (B, S, H, D); add batch dim
        q = F.squeeze(rope(F.unsqueeze(q, 0)), 0)
        k = F.squeeze(rope(F.unsqueeze(k, 0)), 0)

        # GQA: expand K, V if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        q = F.unsqueeze(q, 0)
        k = F.unsqueeze(k, 0)
        v = F.unsqueeze(v, 0)
        attn_out = masked_flash_attention_gpu(
            q,
            k,
            v,
            mask=F.squeeze(attention_bias, axis=1),
            scale=self.scale,
        )
        attn_out = F.squeeze(attn_out, 0)
        attn_out = F.reshape(attn_out, (total_seq_len, -1))
        return self.o_proj(attn_out)
