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

"""DeepSeek-V2 Transformer block (ModuleV3)."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.nn.common_layers.multi_latent_attention import (
    LatentAttentionWithRope,
)
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.nn.kv_cache import PagedCacheValues


class DeepseekV2TransformerBlock(Module[..., Tensor]):
    """Stack of LatentAttentionWithRope, MoE/MLP, and RMSNorm for DeepSeek V2."""

    def __init__(
        self,
        *,
        attention: LatentAttentionWithRope,
        mlp: Module[[Tensor], Tensor],
        attention_norm: RMSNorm,
        mlp_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
    ) -> Tensor:
        residual = x
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_x,
            kv_collection,
            freqs_cis,
            input_row_offsets,
        )

        hidden_states = residual + attn_out

        residual = hidden_states
        norm_h = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(norm_h)
        return residual + mlp_out
