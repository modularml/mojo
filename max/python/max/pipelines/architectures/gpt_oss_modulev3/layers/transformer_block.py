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

"""Implements the GPT OSS transformer block."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor

from .attention import GptOssAttention
from .moe import GptOssMoE


class GptOssTransformerBlock(Module[..., Tensor]):
    """Stack of Attention, MoE, and RMSNorm layers for GPT OSS.

    This is a distributed transformer block that uses a Mixture of Experts (MoE)
    layer instead of a standard feedforward network.
    Block's attention type (full or window) is specified in the model config.
    """

    def __init__(
        self,
        attention: GptOssAttention,
        mlp: GptOssMoE,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
    ) -> None:
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
        residual = x
        norm_xs = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_xs,
            kv_collection,
            input_row_offsets=input_row_offsets,
            **kwargs,
        )

        # Add residual connection after attention
        hidden_states = residual + attn_out

        # Apply post-attention layer norm and then MoE
        residual = hidden_states
        norm_xs = self.post_attention_layernorm(hidden_states)

        # Apply MoE - it returns (output, router_logits)
        mlp_outputs = self.mlp(norm_xs)

        # Add residual connection
        hidden_states = residual + mlp_outputs
        return hidden_states
