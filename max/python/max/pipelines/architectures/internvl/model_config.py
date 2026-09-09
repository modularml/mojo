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
"""Config for InternVL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.llama3.model_config import (
    Llama3Config as Qwen2Config,
)
from max.pipelines.architectures.qwen3.model_config import Qwen3Config
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self, override


def _select_llm_config_class(
    hf_llm_cfg: AutoConfig,
) -> type[Qwen2Config | Qwen3Config]:
    """Choose the correct config class based on parameters in the HuggingFace
    config. Qwen2 is a wrapper around Llama3 and doesn't have its own config, so
    we alias Llama3Config as Qwen2Config for clarity."""
    mt = getattr(hf_llm_cfg, "model_type", None)
    archs = getattr(hf_llm_cfg, "architectures", None) or []
    if mt == "qwen3" or "Qwen3ForCausalLM" in archs:
        return Qwen3Config
    return Qwen2Config


@dataclass
class VisionConfig:
    """Base configuration for InternVL models with required fields."""

    hidden_size: int
    """Hidden size of the vision encoder."""

    intermediate_size: int
    """Intermediate size in the vision encoder's feed-forward layers."""

    norm_type: Literal["rms_norm", "layer_norm"]
    """Type of normalization used in the vision encoder."""

    image_size: int
    """Input image size."""

    patch_size: int
    """Vision transformer patch size."""

    num_attention_heads: int
    """Number of attention heads in the vision encoder."""

    head_dim: int
    """Dimension of each attention head."""

    layer_norm_eps: float
    """Epsilon for layer normalization."""

    qk_normalization: bool
    """Whether to use QK normalization in attention."""

    qkv_bias: bool
    """Whether to use bias in the QKV projection. Default: False."""

    num_hidden_layers: int
    """Number of hidden layers in the vision encoder."""

    # Fields that will be set during finalize
    dtype: DType = DType.bfloat16
    """DType of the InternVL vision model weights."""

    o_proj_bias: bool = False
    """Whether to use bias in the out projection."""

    @classmethod
    def initialize_from_config(
        cls, hf_vision_config: AutoConfig
    ) -> VisionConfig:
        """Initialize VisionConfig from HuggingFace vision config.

        Note: dtype and o_proj_bias fields will be set to defaults and should be
        updated via finalize() once state_dict is available.
        """
        num_attention_heads = hf_vision_config.num_attention_heads
        hidden_size = hf_vision_config.hidden_size
        head_dim = hidden_size // num_attention_heads

        return cls(
            hidden_size=hidden_size,
            intermediate_size=hf_vision_config.intermediate_size,
            norm_type=getattr(hf_vision_config, "norm_type", "rms_norm"),
            image_size=hf_vision_config.image_size,
            patch_size=hf_vision_config.patch_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            layer_norm_eps=getattr(hf_vision_config, "layer_norm_eps", 1e-6),
            qk_normalization=getattr(
                hf_vision_config, "qk_normalization", True
            ),
            qkv_bias=getattr(hf_vision_config, "qkv_bias", False),
            num_hidden_layers=getattr(
                hf_vision_config, "num_hidden_layers", 32
            ),
            # Note: these fields will be overridden in finalize
            dtype=DType.bfloat16,
            o_proj_bias=False,
        )

    def finalize(self, dtype: DType, state_dict: dict[str, WeightData]) -> None:
        """Finalize VisionConfig with state_dict dependent fields."""
        # InternVL o_proj_bias is not in the config, check checkpoint.
        # Check for the presence of the o_proj.bias key dynamically across all layers
        o_proj_bias = any(
            key.endswith(".attn.o_proj.bias") for key in state_dict
        )

        self.dtype = dtype
        self.o_proj_bias = o_proj_bias


@dataclass(kw_only=True)
class InternVLConfig(ArchConfigWithKVCache):
    """Configuration for InternVL models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    devices: list[DeviceRef]
    """Devices that the InternVL model is parallelized over."""

    # Multimodal options.
    downsample_ratio: float
    """Downsample ratio for vision features."""

    num_image_token: int
    """Number of image tokens per patch."""

    # Vision encoder configuration.
    vision_config: VisionConfig
    """Vision encoder configuration."""

    # Composed language model configuration.
    llm_config: Qwen2Config | Qwen3Config
    """Language model configuration (Qwen2 or Qwen3)."""

    quantization_encoding: SupportedEncoding | None = None

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters from the embedded LLM config."""
        return self.llm_config.get_kv_params()

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length from the embedded LLM config."""
        return self.llm_config.get_max_seq_len()

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        # Delegate to the selected decoder family for language model parameters.
        llm_hf_cfg = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(llm_hf_cfg)
        return ConfigCls.construct_kv_params(
            huggingface_config=llm_hf_cfg,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to the selected decoder family for language model parameters.
        llm_hf_cfg = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(llm_hf_cfg)
        return ConfigCls.get_num_layers(llm_hf_cfg)

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Calculate maximum sequence length for InternVL."""
        # Delegate to the selected decoder family for language model parameters.
        llm_hf_cfg = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(llm_hf_cfg)
        return ConfigCls.calculate_max_seq_len(
            huggingface_config=llm_hf_cfg,
            model_config=model_config,
        )

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes an InternVLConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An InternVLConfig instance with fields initialized from config.
        """
        model_config = model_config or pipeline_config.model
        return cls.initialize_from_config(
            pipeline_config,
            model_config.huggingface_config,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes an InternVLConfig from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be
        determined from the pipeline and HuggingFace configurations, without
        needing the state_dict. Fields that depend on the state_dict should
        be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            An InternVLConfig instance ready for finalization.
        """
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")

        # Create VisionConfig from the vision config
        vision_config = VisionConfig.initialize_from_config(hf_vision_config)

        # Select decoder family (Qwen2/Qwen3) from HF llm_config
        hf_llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(hf_llm_config)
        llm_config: Qwen2Config | Qwen3Config
        if ConfigCls is Qwen3Config:
            llm_config = Qwen3Config.initialize_from_config(
                pipeline_config, hf_llm_config, max_seq_len=max_seq_len
            )
        else:
            # Qwen2 semantics (delegates to Llama3-style config under the hood)
            llm_config = Qwen2Config.initialize_from_config(
                pipeline_config, hf_llm_config, max_seq_len=max_seq_len
            )

        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )

        return cls(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            # Multimodal parameters
            downsample_ratio=getattr(
                huggingface_config, "downsample_ratio", 0.5
            ),
            num_image_token=getattr(huggingface_config, "num_image_token", 256),
            # Vision configuration
            vision_config=vision_config,
            # Composed language model configuration
            llm_config=llm_config,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        vision_state_dict: dict[str, WeightData],
        dtype: DType,
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
    ) -> None:
        """Finalize the InternVLConfig instance with state_dict dependent fields.

        Args:
            huggingface_config: HuggingFace model configuration.
            llm_state_dict: Language model weights dictionary.
            vision_state_dict: Vision encoder weights dictionary.
            dtype: Data type for model parameters.
            return_logits: Return logits configuration.
            norm_method: Normalization method.
        """
        # Finalize vision config
        self.vision_config.finalize(
            dtype=dtype,
            state_dict=vision_state_dict,
        )

        # Finalize llm config
        hf_llm_config = getattr(
            huggingface_config, "llm_config", huggingface_config
        )
        ConfigCls = _select_llm_config_class(hf_llm_config)
        if ConfigCls is Qwen3Config:
            self.llm_config.finalize(
                huggingface_config=hf_llm_config,
                state_dict=llm_state_dict,
                return_logits=return_logits,
                norm_method=norm_method,
                attention_bias=False,  # Qwen3 removes QKV biases
            )
        else:
            # Qwen2 semantics
            self.llm_config.finalize(
                huggingface_config=hf_llm_config,
                state_dict=llm_state_dict,
                norm_method=norm_method,
                attention_bias=True,  # Qwen2 uses attention bias
                return_logits=return_logits,
            )
