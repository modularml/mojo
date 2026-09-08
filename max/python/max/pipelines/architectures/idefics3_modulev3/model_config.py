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
"""Config for Idefics3 models (ModuleV3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchVLConfigWithTextSubconfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

# Reuse the vision config from the V2 implementation (no V3-specific changes).
from ..idefics3.model_config import Idefics3VisionConfig

# Use the V3 Llama3Config for the text component.
from ..llama3_modulev3.model_config import Llama3Config


@dataclass(kw_only=True)
class Idefics3Config(ArchVLConfigWithTextSubconfig, ArchConfigWithKVCache):
    """Configuration for Idefics3 models (ModuleV3)."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    devices: list[DeviceRef]
    """Devices that the Idefics3 model is parallelized over."""

    # Multimodal options.
    scale_factor: int
    """Scale factor for pixel shuffle operation in the connector."""

    image_token_id: int
    """Token ID used to represent image tokens in the text sequence."""

    # Vision encoder configuration.
    vision_config: Idefics3VisionConfig
    """Vision encoder configuration (SigLIP-based)."""

    # Text model configuration - using V3 Llama3Config directly
    text_config: Llama3Config
    """Text model configuration (Llama3-based)."""

    quantization_encoding: SupportedEncoding | None = None

    @property
    def image_seq_len(self) -> int:
        """Calculate the number of image tokens after connector processing."""
        patches_per_side = (
            self.vision_config.image_size // self.vision_config.patch_size
        )
        total_patches = patches_per_side * patches_per_side
        return total_patches // (self.scale_factor * self.scale_factor)

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters from the embedded text config."""
        return self.text_config.get_kv_params()

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Get number of layers in the language model."""
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return text_config.num_hidden_layers

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes an Idefics3Config instance from pipeline configuration."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )

        # Build a V3 Llama3Config for the text component.
        text_config = _create_llama3_text_config(
            pipeline_config, hf_text_config, max_seq_len
        )

        vision_config = Idefics3VisionConfig.initialize_from_config(
            pipeline_config, huggingface_config, text_config.hidden_size
        )

        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )

        return cls(
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in model_config.device_specs
            ],
            scale_factor=getattr(huggingface_config, "scale_factor", 2),
            image_token_id=getattr(
                huggingface_config, "image_token_id", 128257
            ),
            vision_config=vision_config,
            text_config=text_config,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        llm_state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
    ) -> None:
        """Finalize the Idefics3Config with state_dict-dependent fields."""
        hf_text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        self.text_config.finalize(
            huggingface_config=hf_text_config,
            state_dict=llm_state_dict,
            norm_method=norm_method,
            attention_bias=False,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
        )


def _create_llama3_text_config(
    pipeline_config: PipelineConfig,
    hf_text_config: AutoConfig,
    max_seq_len: int,
) -> Llama3Config:
    """Create a V3 Llama3Config from a text sub-config.

    This replicates the initialization logic of Llama3Config.initialize(max_seq_len=max_seq_len)
    but accepts a specific HuggingFace text sub-config rather than reading
    from pipeline_config.model.huggingface_config.
    """
    kv_cache_config = pipeline_config.model.kv_cache
    quantization_encoding = _select_quantization_encoding(
        pipeline_config.model, Idefics3Config.DEFAULT_ENCODING
    )
    dtype = supported_encoding_dtype(quantization_encoding)
    cache_dtype = cache_dtype_for_encoding(
        quantization_encoding, pipeline_config.model.kv_cache.kv_cache_format
    )

    interleaved_rope_weights = _interleaved_rope_weights(pipeline_config.model)

    device_refs = [
        DeviceRef(spec.device_type, spec.id)
        for spec in pipeline_config.model.device_specs
    ]

    embedding_multiplier = getattr(hf_text_config, "embedding_multiplier", 1.0)
    residual_multiplier = getattr(hf_text_config, "residual_multiplier", 1.0)
    rope_scaling_params: Llama3RopeScalingParams | None = None
    rope_scaling = getattr(hf_text_config, "rope_scaling", None)

    if rope_scaling is not None:
        rope_type = rope_scaling.get("type")
        rope_type_alt = rope_scaling.get("rope_type")
        if rope_type == "llama3" or rope_type_alt == "llama3":
            rope_scaling_params = Llama3RopeScalingParams(
                factor=rope_scaling["factor"],
                low_freq_factor=rope_scaling["low_freq_factor"],
                high_freq_factor=rope_scaling["high_freq_factor"],
                orig_max_position=rope_scaling[
                    "original_max_position_embeddings"
                ],
            )

    attention_multiplier = Llama3Config.calculate_attention_multiplier(
        hf_text_config
    )

    return Llama3Config(
        hidden_size=hf_text_config.hidden_size,
        num_attention_heads=hf_text_config.num_attention_heads,
        num_key_value_heads=hf_text_config.num_key_value_heads,
        num_hidden_layers=hf_text_config.num_hidden_layers,
        rope_theta=get_rope_theta(hf_text_config),
        rope_scaling_params=rope_scaling_params,
        longrope_scaling_params=None,
        intermediate_size=hf_text_config.intermediate_size,
        interleaved_rope_weights=interleaved_rope_weights,
        vocab_size=hf_text_config.vocab_size,
        dtype=dtype,
        max_seq_len=max_seq_len,
        kv_params=Llama3Config.construct_kv_params(
            huggingface_config=hf_text_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        ),
        attention_multiplier=attention_multiplier,
        embedding_multiplier=embedding_multiplier,
        residual_multiplier=residual_multiplier,
        devices=device_refs,
        clip_qkv=getattr(hf_text_config, "clip_qkv", None),
        logits_scaling=getattr(hf_text_config, "logits_scaling", 1.0),
        quantization_encoding=quantization_encoding,
    )
