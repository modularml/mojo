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
"""Config for Qwen3VL models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchVLConfigWithTextSubconfig,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass
class VisionConfig:
    """Base configuration for Qwen3VL models with required fields."""

    dtype: DType
    """DType of the Qwen3VL vision model weights."""

    llm_dtype: DType
    """DType of the Qwen3VL language model weights."""

    devices: list[DeviceRef]
    """Devices that the Qwen3VL vision encoder model is parallelized over."""

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

    deepstack_visual_indexes: list[int]
    """Indexes of the full attention blocks in the vision encoder."""

    rms_norm_eps: float
    """Epsilon for layer normalization."""

    spatial_merge_size: int
    """Spatial merge size for the vision encoder."""

    num_position_embeddings: int
    """Number of position embeddings for the vision encoder."""

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
        from max.dtype import DType as MaxDType

        return cls(
            # Defaults that will be overridden in finalize
            dtype=MaxDType.bfloat16,
            llm_dtype=MaxDType.bfloat16,
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
            deepstack_visual_indexes=hf_vision_config.deepstack_visual_indexes,
            rms_norm_eps=1e-06,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            num_position_embeddings=hf_vision_config.num_position_embeddings,
        )

    def finalize(
        self,
        vision_dtype: DType,
        llm_dtype: DType,
    ) -> None:
        """Finalize VisionConfig with state_dict dependent fields."""
        self.dtype = vision_dtype
        self.llm_dtype = llm_dtype


@dataclass(kw_only=True)
class Qwen3VLConfig(ArchVLConfigWithTextSubconfig, ArchConfigWithKVCache):
    """Configuration for Qwen3VL models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
        "float8_e4m3fn",
    }

    devices: list[DeviceRef]
    """Devices that the Qwen3VL model is parallelized over."""

    dtype: DType
    """DType of the Qwen3VL model weights."""

    # Multimodal parameters
    image_token_id: int
    """Token ID used for image placeholders in the input sequence."""

    video_token_id: int
    """Token ID used for video placeholders in the input sequence."""

    vision_start_token_id: int
    """Token ID that marks the start of vision content."""

    spatial_merge_size: int
    """Size parameter for spatial merging of vision features."""

    mrope_section: list[int]
    """List of indices for the mrope section."""

    num_experts: int
    """Number of experts in the MoE layer."""

    num_experts_per_tok: int
    """Number of experts per token in the MoE layer."""

    moe_intermediate_size: int
    """Intermediate size in the MoE layer."""

    mlp_only_layers: list[int]
    """List of indices for the MLP only layers."""

    norm_topk_prob: bool
    """Whether to use top-k probability normalization in the MoE layer."""

    decoder_sparse_step: int
    """Sparse step for the decoder."""

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
        """Initializes a Qwen3VLConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            A Qwen3VLConfig instance with fields initialized from config.
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
        """Initializes a Qwen3VLConfig from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be
        determined from the pipeline and HuggingFace configurations, without
        needing the state_dict. Fields that depend on the state_dict should
        be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            A Qwen3VLConfig instance ready for finalization.
        """
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")

        text_config = huggingface_config.text_config

        # Get quantization encoding for dtype
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)

        # Create VisionConfig from the vision config

        # Propagate quantization_config to vision_config if present on main
        # config but not vision_config
        if hasattr(huggingface_config, "quantization_config") and not hasattr(
            hf_vision_config, "quantization_config"
        ):
            hf_vision_config.quantization_config = (
                huggingface_config.quantization_config
            )

        vision_config = VisionConfig.initialize_from_config(
            pipeline_config, hf_vision_config
        )

        # Create Llama3Config for the language model

        # Propagate quantization_config to text_config if present on main config
        # but not text_config
        if hasattr(huggingface_config, "quantization_config") and not hasattr(
            huggingface_config.text_config, "quantization_config"
        ):
            huggingface_config.text_config.quantization_config = (
                huggingface_config.quantization_config
            )

        # For VLM models, tie_word_embeddings on the top-level config determines
        # whether lm_head.weight exists in the checkpoint. The text_config may
        # have a different value (e.g., Qwen3-VL-4B-FP8 has top-level=false but
        # text_config=true). Use top-level value to match actual checkpoint.
        if hasattr(huggingface_config, "tie_word_embeddings"):
            huggingface_config.text_config.tie_word_embeddings = (
                huggingface_config.tie_word_embeddings
            )

        llm_config = Llama3Config.initialize_from_config(
            pipeline_config, text_config, max_seq_len=max_seq_len
        )

        # Handle both MoE (e.g., 30B) and dense (e.g., VL 2B 4B etc) variants.
        # For dense models, num_experts=0 ensures the decoder always uses MLP layers
        num_experts = getattr(text_config, "num_experts", 0)
        num_experts_per_tok = getattr(text_config, "num_experts_per_tok", 1)
        moe_intermediate_size = getattr(
            text_config, "moe_intermediate_size", text_config.intermediate_size
        )
        mlp_only_layers = getattr(text_config, "mlp_only_layers", [])
        norm_topk_prob = getattr(text_config, "norm_topk_prob", False)
        decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1)

        return cls(
            dtype=dtype,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            # Multimodal parameters
            image_token_id=huggingface_config.image_token_id,
            video_token_id=huggingface_config.video_token_id,
            vision_start_token_id=huggingface_config.vision_start_token_id,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            mrope_section=text_config.rope_scaling["mrope_section"],
            # MoE parameters
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            mlp_only_layers=mlp_only_layers,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
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
        return_logits: ReturnLogits,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
    ) -> None:
        """Finalize the Qwen3VLConfig instance with state_dict dependent fields.

        Args:
            huggingface_config: HuggingFace model configuration.
            llm_state_dict: Language model weights dictionary.
            vision_state_dict: Vision encoder weights dictionary.
            return_logits: Return logits configuration.
            norm_method: Normalization method.
        """
        # Determine dtypes from state_dict
        vision_dtype = vision_state_dict[
            "vision_encoder.patch_embed.proj.weight"
        ].dtype
        llm_dtype = llm_state_dict["language_model.embed_tokens.weight"].dtype

        # Finalize vision config
        self.vision_config.finalize(
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
            attention_bias=True,  # Qwen3VL uses Qwen2 which has attention_bias=True
        )
        self.llm_config.interleaved_rope_weights = False
