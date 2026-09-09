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

from dataclasses import dataclass
from typing import ClassVar, Literal

from max.graph.weights import WeightData
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.llama3_modulev3.model_config import (
    Llama3Config,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig


@dataclass(kw_only=True)
class OlmoConfig(Llama3Config):
    """Model configuration for Olmo graph construction/execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    # OLMo uses non-parametric LayerNorm (no learnable weight/bias).
    norm_elementwise_affine: bool = False

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        super().finalize(
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            norm_method="layer_norm",  # Olmo uses layer norm
            attention_bias=attention_bias,
        )
