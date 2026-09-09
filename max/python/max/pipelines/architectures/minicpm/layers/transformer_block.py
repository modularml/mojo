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

"""MiniCPM transformer block — pre-norm with depth-scaled residuals."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues

from .attention import MiniCPMAttention


class MiniCPMTransformerBlock(Module[..., Tensor]):
    """Pre-norm decoder block: norm → attn → scaled residual → norm → mlp → scaled residual."""

    def __init__(
        self,
        attention: MiniCPMAttention,
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.residual_multiplier = residual_multiplier

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        **kwargs,
    ) -> Tensor:
        """Apply one pre-norm decoder block: Attention → residual → MLP → residual.

        Args:
            layer_idx: Scalar uint32 tensor identifying this layer in the KV cache.
            x: Input hidden states, shape ``(total_seq_len, hidden_size)``.
            kv_collection: Paged KV cache (read/written by attention).
            input_row_offsets: uint32 ragged-batch row delimiters passed
                through to attention.
            **kwargs: Forwarded to the attention layer.

        Returns:
            Updated hidden states, same shape as ``x``.
        """

        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )
        if self.residual_multiplier != 1.0:
            m = F.constant(self.residual_multiplier, x.dtype, device=x.device)
            attn_out = attn_out * m
        h = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(h))
        if self.residual_multiplier != 1.0:
            m = F.constant(self.residual_multiplier, h.dtype, device=h.device)
            mlp_out = mlp_out * m
        return h + mlp_out
