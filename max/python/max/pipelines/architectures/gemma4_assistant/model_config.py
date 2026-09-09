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
"""Config for Gemma4 Assistant (Multi-Token Prediction draft) models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.transformer.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self

# Use the native config if available once transformers ships support,
# otherwise fall back to our shim for older versions.
try:
    from transformers import Gemma4AssistantConfig as Gemma4AssistantHFConfig
except ImportError:

    class Gemma4AssistantHFConfig(PretrainedConfig):  # type: ignore[no-redef]
        model_type = "gemma4_assistant"

    try:
        AutoConfig.register("gemma4_assistant", Gemma4AssistantHFConfig)
    except ValueError:
        pass


@dataclass
class Gemma4AssistantConfig:
    """Configuration for the Gemma4 Assistant draft model.

    The assistant model is a lightweight decoder that performs cross-attention
    against the target (backbone) model's KV cache. It has no K/V projection
    weights of its own.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    backbone_hidden_size: int = 5376
    """Hidden dimension of the target (backbone) model."""

    hidden_size: int = 1024
    """Hidden dimension of the assistant model."""

    num_hidden_layers: int = 4
    """Number of decoder layers in the assistant model."""

    num_attention_heads: int = 32
    """Number of query attention heads."""

    num_key_value_heads: int = 16
    """Number of key/value heads for sliding window attention in the target."""

    num_global_key_value_heads: int = 4
    """Number of key/value heads for global attention in the target."""

    head_dim: int = 256
    """Per-head dimension for sliding window attention."""

    global_head_dim: int = 512
    """Per-head dimension for global attention."""

    intermediate_size: int = 8192
    """Feed-forward intermediate dimension."""

    vocab_size: int = 262144
    """Vocabulary size."""

    rms_norm_eps: float = 1e-6
    """Epsilon for RMS normalization."""

    hidden_activation: str = "gelu_pytorch_tanh"
    """Activation function for the MLP."""

    layer_types: list[str] = field(
        default_factory=lambda: [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
    )
    """Per-layer attention type specification."""

    sliding_window: int = 1024
    """Sliding window size for local attention layers."""

    sliding_window_rope_theta: float = 10000.0
    """RoPE theta for sliding window attention."""

    global_rope_theta: float = 1000000.0
    """RoPE theta for global attention."""

    global_rope_scaling: ProportionalScalingParams | None = None
    """Proportional scaling config for global RoPE."""

    attention_k_eq_v: bool = True
    """Whether K and V projections are shared in the target model."""

    num_kv_shared_layers: int = 4
    """Number of KV-shared layers."""

    max_position_embeddings: int = 262144
    """Maximum sequence length supported by position embeddings."""

    devices: list[DeviceRef] = field(default_factory=lambda: [DeviceRef.GPU()])
    """Devices to place weights and run computation on."""

    dtype: DType = DType.bfloat16
    """Data type for model weights and activations."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Which logits to return from the model."""

    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    """Which hidden states to return from the model."""

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for Gemma4AssistantConfig"
            )
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        if isinstance(text_config, dict):
            max_pos = text_config["max_position_embeddings"]
        else:
            max_pos = text_config.max_position_embeddings
        return cls(max_position_embeddings=max_pos)

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del model_config
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        if isinstance(text_config, dict):
            return int(text_config["max_position_embeddings"])
        return int(text_config.max_position_embeddings)
