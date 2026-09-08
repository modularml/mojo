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

"""Olmo3 transformer block (experimental KV types)."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor


class Olmo3TransformerBlock(
    Module[[Tensor, Tensor, PagedCacheValues, Tensor], Tensor]
):
    """Post-norm Attention -> Norm -> MLP -> Norm block for Olmo3."""

    def __init__(
        self,
        attention: Module[[Tensor, PagedCacheValues, Tensor], Tensor],
        mlp: MLP,
        post_attention_layernorm: RMSNorm,
        post_feedforward_layernorm: RMSNorm,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.post_attention_layernorm = post_attention_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm
        self.residual_multiplier = residual_multiplier

    def forward(
        self,
        layer_idx: Tensor,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        attn_out = self.self_attn(
            x,
            kv_collection,
            input_row_offsets,
        )
        attn_out = self.post_attention_layernorm(attn_out)

        if self.residual_multiplier != 1.0:
            multiplier = F.constant(
                self.residual_multiplier, x.dtype, device=x.device
            )
            attn_out = attn_out * multiplier

        h = x + attn_out
        residual = h

        mlp_out = self.mlp(h)
        mlp_out = self.post_feedforward_layernorm(mlp_out)

        if self.residual_multiplier != 1.0:
            multiplier = F.constant(
                self.residual_multiplier, x.dtype, device=x.device
            )
            mlp_out = mlp_out * multiplier

        return mlp_out + residual
