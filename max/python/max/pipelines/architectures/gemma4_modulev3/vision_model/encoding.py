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

"""Gemma4 vision encoder stack for the ModuleV3 API."""

from __future__ import annotations

from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)

from ..layers.rms_norm import Gemma4RMSNorm
from .attention import Gemma4VisionAttention


class Gemma4VisionEncoderLayer(Module[..., Tensor]):
    """Full-attention transformer block, 4 norms, no layer scalar / MoE.

    Port of the graph arch's Gemma4VisionEncoderLayer.
    """

    def __init__(
        self, config: Gemma4ForConditionalGenerationConfig, layer_idx: int
    ) -> None:
        super().__init__()
        vision_config = config.vision_config
        assert vision_config is not None
        hidden = vision_config.hidden_size
        eps = vision_config.rms_norm_eps

        self.input_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)
        self.mlp = MLP(
            hidden_dim=hidden,
            feed_forward_length=vision_config.intermediate_size,
            activation_function=vision_config.hidden_activation,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden, eps=eps)

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, freqs_cis, cu_seqlens, max_seq_len)
        h = residual + self.post_attention_layernorm(h)

        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        return residual + self.post_feedforward_layernorm(h)


class Gemma4VisionEncoder(Module[..., Tensor]):
    """Stack of Gemma4VisionEncoderLayer blocks."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        assert config.vision_config is not None
        self.layers = ModuleList(
            [
                Gemma4VisionEncoderLayer(config, layer_idx)
                for layer_idx in range(config.vision_config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_state: Tensor,
        freqs_cis: Tensor,
        cu_seqlens: Tensor,
        max_seq_len: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            hidden_state = layer(
                hidden_state, freqs_cis, cu_seqlens, max_seq_len
            )
        return hidden_state
