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

import math

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.kernels import masked_flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


class EncoderAttention(Module):
    """Encoder-only attention without KV cache (Qwen3: interleaved RoPE via rope.forward)."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        dtype: DType,
        device: DeviceRef,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = scale if scale is not None else math.sqrt(1.0 / head_dim)

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(hidden_size, q_dim, dtype, device, has_bias=False)
        self.k_proj = Linear(hidden_size, kv_dim, dtype, device, has_bias=False)
        self.v_proj = Linear(hidden_size, kv_dim, dtype, device, has_bias=False)
        self.o_proj = Linear(q_dim, hidden_size, dtype, device, has_bias=False)

        self.q_norm = RMSNorm(
            head_dim,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            head_dim,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )

    def _repeat_kv(self, x: TensorValue, n_rep: int) -> TensorValue:
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
        x = ops.unsqueeze(x, 2)
        x = ops.concat([x] * n_rep, axis=2)
        return ops.reshape(x, (seq_len, n_kv_heads * n_rep, head_dim))

    def __call__(
        self,
        x: TensorValue,
        rope: RotaryEmbedding,
        attention_bias: TensorValue,
    ) -> TensorValue:
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

        q = ops.reshape(q, (total_seq_len, self.n_heads, self.head_dim))
        k = ops.reshape(k, (total_seq_len, self.n_kv_heads, self.head_dim))
        v = ops.reshape(v, (total_seq_len, self.n_kv_heads, self.head_dim))

        # Qwen3: norm over head_dim (per-head), then RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # module_v3.common_layers RotaryEmbedding.forward expects 4D (B, S, H, D); add batch dim
        q = ops.squeeze(
            rope(
                ops.unsqueeze(q, 0),
                start_pos=Dim(0),
                seq_len=total_seq_len,
            ),
            0,
        )
        k = ops.squeeze(
            rope(
                ops.unsqueeze(k, 0),
                start_pos=Dim(0),
                seq_len=total_seq_len,
            ),
            0,
        )

        # GQA: expand K, V if needed
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        q = ops.unsqueeze(q, 0)
        k = ops.unsqueeze(k, 0)
        v = ops.unsqueeze(v, 0)

        attn_out = masked_flash_attention_gpu(
            q,
            k,
            v,
            mask=ops.squeeze(attention_bias, axis=1),
            scale=self.scale,
        )
        attn_out = ops.squeeze(attn_out, 0)
        attn_out = ops.reshape(attn_out, (total_seq_len, -1))
        return self.o_proj(attn_out)
