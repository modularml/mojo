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
"""Single-stream attention with fused QKV, QK-RMSNorm, and 3D MRoPE."""

from __future__ import annotations

import math

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu

from .embeddings import apply_mrope
from .fp8_linear import Ideogram4FP8Linear

flash_attention_gpu = F.functional(_flash_attention_gpu)


class Ideogram4Attention(Module[..., Tensor]):
    """Fused-QKV attention matching ``Ideogram4Attention``.

    For single-sample (unpadded) inference the per-sample block-diagonal mask
    degenerates to full bidirectional attention, so a ``NULL_MASK`` flash
    attention is exact.
    """

    def __init__(self, hidden_size: int, num_heads: int, eps: float) -> None:
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = Ideogram4FP8Linear(hidden_size, hidden_size * 3)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.o = Ideogram4FP8Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        qkv = self.qkv(x)
        qkv = F.reshape(
            qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim]
        )
        q = qkv[:, :, 0]  # (B, L, H, Dh)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = apply_mrope(q, cos, sin)
        k = apply_mrope(k, cos, sin)
        q = q.cast(v.dtype)
        k = k.cast(v.dtype)

        out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=math.sqrt(1.0 / float(self.head_dim)),
        )

        out = F.reshape(out, [batch_size, seq_len, self.hidden_size])
        return self.o(out)
