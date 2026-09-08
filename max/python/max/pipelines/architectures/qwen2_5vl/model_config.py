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
"""Config for Qwen2.5VL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
    parse_quant_config,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchVLConfigWithTextSubconfig,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass
class VisionConfig:
    """Base configuration for Qwen2.5VL models with required fields."""

    dtype: DType
    """DType of the Qwen2.5VL vision model weights."""

    llm_dtype: DType
    """DType of the Qwen2.5VL language model weights."""

    devices: list[DeviceRef]
    """Devices that the Qwen2.5VL vision encoder model is parallelized over."""

    patch_size: int
    """Vision transformer patch size."""

    temporal_patch_size: int
    """Vision transformer temporal patch size."""

    in_channels: int
    """Vision transformer number of input channels."""

    hidden_size: int
    """Hidden size of the vision encoder."""

    num_attention_heads: int
    """Number of attention heads in the vision encoder."""

    depth: int
    """Number of vision transformer layers."""

    intermediate_size: int
    """Intermediate size in the vision encoder's feed-forward layers."""

    out_hidden_size: int
    """Output hidden size of the vision encoder. Also the hidden size of the language model."""

    fullatt_block_indexes: list[int]
    """Indexes of the full attention blocks in the vision encoder."""

    rms_norm_eps: float
    """Epsilon for layer normalization."""

    window_size: int
    """Window size for the vision encoder."""

    spatial_merge_size: int
    """Spatial merge size for the vision encoder."""

    quant_config: QuantConfig | None = None
    """Scaled quantization configuration for the vision encoder."""

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        hf_vision_config: AutoConfig,
    ) -> VisionConfig:
        """Initialize VisionConfig from HuggingFace vision config.

        Note: dtype fields will be set to defaults and should be updated
        via finalize() once state_dict is available.
        """

        return cls(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            patch_size=hf_vision_config.patch_size,
            temporal_patch_size=hf_vision_config.temporal_patch_size,
            in_channels=hf_vision_config.in_channels,
            hidden_size=hf_vision_config.hidden_size,
            num_attention_heads=hf_vision_config.num_heads,
            depth=hf_vision_config.depth,
            intermediate_size=hf_vision_config.intermediate_size,
            out_hidden_size=hf_vision_config.out_hidden_size,
            fullatt_block_indexes=hf_vision_config.fullatt_block_indexes,
            window_size=hf_vision_config.window_size,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            # Note: these fields will be overridden in finalize
            dtype=DType.bfloat16,
            llm_dtype=DType.bfloat16,
            rms_norm_eps=1e-06,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        vision_state_dict: dict[str, WeightData],
        vision_dtype: DType,
        llm_dtype: DType,
    ) -> None:
        """Finalize VisionConfig with state_dict dependent fields."""
        # Parse (if present) a float8 configuration for the vision path.
        v_float8 = parse_quant_config(
            huggingface_config,
            vision_state_dict,
            vision_dtype,
            state_dict_name_prefix="vision_encoder.",
            ignored_modules_prefix="vision_encoder.",
        )

        self.dtype = vision_dtype
        self.llm_dtype = llm_dtype
        self.quant_config = v_float8


@dataclass(kw_only=True)
class Qwen2_5VLConfig(ArchVLConfigWithTextSubconfig, ArchConfigWithKVCache):
    """Configuration for Qwen2.5VL models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
        "float8_e4m3fn",
    }

    devices: list[DeviceRef]
    """Devices that the Qwen2.5VL model is parallelized over."""

    # Multimodal parameters
    image_token_id: int
    """Token ID used for image placeholders in the input sequence."""

    video_token_id: int
    """Token ID used for video placeholders in the input sequence."""

    vision_start_token_id: int
    """Token ID that marks the start of vision content."""

    spatial_merge_size: int
    """Size parameter for spatial merging of vision features."""

    tokens_per_second: int
    """Number of tokens per second."""

    mrope_section: list[int]
    """List of indices for the mrope section."""

    # Vision encoder configuration.
    vision_config: VisionConfig
    """Vision encoder configuration."""

    # Composed language model configuration.
    llm_config: Llama3Config
    """Language model configuration using Llama3 architecture."""

    quantization_encoding: SupportedEncoding | None = None

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters from the embedded LLM config."""
        return self.llm_config.get_kv_params()

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to Llama3Config for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return Llama3Config.get_num_layers(llm_config)

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a Qwen2_5VLConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            A Qwen2_5VLConfig instance with fields initialized from config.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls.initialize_from_config(
            pipeline_config, huggingface_config, max_seq_len=max_seq_len
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a Qwen2_5VLConfig from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be
        determined from the pipeline and HuggingFace configurations, without
        needing the state_dict. Fields that depend on the state_dict should
        be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            A Qwen2_5VLConfig instance ready for finalization.
        """
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")

        # Create VisionConfig from the vision config
        vision_config = VisionConfig.initialize_from_config(
            pipeline_config, hf_vision_config
        )

        # Create Llama3Config for the language model using the text sub-config
        # so RoPE attributes (rope_scaling / rope_parameters) resolve correctly
        # under both transformers v4 and v5.
        text_config = huggingface_config.text_config
        llm_config = Llama3Config.initialize_from_config(
            pipeline_config, text_config, max_seq_len=max_seq_len
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
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            vision_start_token_id=huggingface_config.vision_start_token_id,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            tokens_per_second=hf_vision_config.tokens_per_second,
            mrope_section=text_config.rope_scaling["mrope_section"],
            # Vision configuration
            vision_config=vision_config,
            # Composed language model configuration
            llm_config=llm_config,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        llm_state_dict: dict[str, WeightData],
        vision_state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
    ) -> None:
        """Finalize the Qwen2_5VLConfig instance with state_dict dependent fields.

        Args:
            huggingface_config: HuggingFace model configuration.
            pipeline_config: The MAX Engine pipeline configuration.
            llm_state_dict: Language model weights dictionary.
            vision_state_dict: Vision encoder weights dictionary.
            return_logits: Return logits configuration.
            norm_method: Normalization method.
        """
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")

        # Determine dtypes from state_dict
        vision_dtype = vision_state_dict[
            "vision_encoder.patch_embed.proj.weight"
        ].dtype
        llm_dtype = llm_state_dict["language_model.embed_tokens.weight"].dtype

        # Finalize vision config
        self.vision_config.finalize(
            huggingface_config=huggingface_config,
            vision_state_dict=vision_state_dict,
            vision_dtype=vision_dtype,
            llm_dtype=llm_dtype,
        )

        # quantization_config lives at the top level of the HF config, not
        # under text_config. Propagate it so parse_quant_config() finds it.
        if hasattr(huggingface_config, "quantization_config"):
            huggingface_config.text_config.quantization_config = (
                huggingface_config.quantization_config
            )

        # Finalize llm config (with Qwen2 attention_bias=True)
        self.llm_config.finalize(
            huggingface_config=huggingface_config.text_config,
            state_dict=llm_state_dict,
            return_logits=return_logits,
            norm_method=norm_method,
            attention_bias=True,
        )
