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

"""GritLM transformer block — standard pre-norm decoder block."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues

from .attention import GritLMAttention


class GritLMTransformerBlock(Module[..., Tensor]):
    """Pre-norm decoder block: Attention → residual → MLP → residual."""

    def __init__(
        self,
        attention: GritLMAttention,
        mlp: Module[[Tensor], Tensor],
        input_layernorm: Module[[Tensor], Tensor],
        post_attention_layernorm: Module[[Tensor], Tensor],
    ) -> None:
        """Initialize a pre-norm GritLM transformer block.

        Args:
            attention: ``GritLMAttention`` instance for this layer.
            mlp: SwiGLU MLP module.
            input_layernorm: RMSNorm applied before the attention sub-layer.
            post_attention_layernorm: RMSNorm applied before the MLP sub-layer.
        """

        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

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

        h = x + self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )
        return h + self.mlp(self.post_attention_layernorm(h))
