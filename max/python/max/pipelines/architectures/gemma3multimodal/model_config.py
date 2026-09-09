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

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gemma3.model_config import Gemma3Config
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    parse_quant_config,
)
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ArchVLConfigWithTextSubconfig,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass
class Gemma3VisionConfig:
    """
    The vision-specific config for Gemma3
    More info at: https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json
    """

    hidden_act: str
    """The non-linear activation function (function or string) in the encoder and pooler.
    `"gelu"`, `"gelu_tanh"`, `"relu"`, `"sigmoid"`, `"silu"`, and `"tanh"`
    are supported."""

    hidden_size: int
    """Dimensionality of the encoder layers and the pooler layer"""

    image_size: int
    """The size (resolution) of each image"""

    intermediate_size: int
    """Dimension of the MLP representations"""

    layer_norm_eps: float
    """The epsilon used by the layer normalization layers."""

    num_attention_heads: int
    """Number of attention heads for each attention layer in the Transformer encoder"""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer encoder"""

    num_channels: int
    """Number of channels in the input images."""

    patch_size: int
    """The size (resolution) of each patch"""

    attention_bias: bool = True

    attention_dropout: float = 0.0
    """The dropout ratio for the attention probabilities"""

    vision_use_head: bool = False
    """Flag whether to use attention heads for vision"""

    _HIDDEN_ACTIVATION_MAP = {
        "gelu_pytorch_tanh": "tanh",
        "swish": "silu",
    }

    @classmethod
    def initialize_from_config(
        cls, hf_vision_config: AutoConfig
    ) -> Gemma3VisionConfig:
        """Initialize Gemma3VisionConfig from HuggingFace vision config."""
        hidden_act = hf_vision_config.hidden_act
        if hidden_act in cls._HIDDEN_ACTIVATION_MAP:
            hidden_act = cls._HIDDEN_ACTIVATION_MAP[hidden_act]

        return cls(
            hidden_size=hf_vision_config.hidden_size,
            image_size=hf_vision_config.image_size,
            intermediate_size=hf_vision_config.intermediate_size,
            num_attention_heads=hf_vision_config.num_attention_heads,
            num_hidden_layers=hf_vision_config.num_hidden_layers,
            patch_size=hf_vision_config.patch_size,
            num_channels=hf_vision_config.num_channels,
            hidden_act=hidden_act,
            layer_norm_eps=hf_vision_config.layer_norm_eps,
        )


@dataclass(kw_only=True)
class Gemma3ForConditionalGenerationConfig(
    ArchVLConfigWithTextSubconfig,
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
):
    """Base configuration for Gemma 3 models.

    Contains parameters specific to the Gemma 3 architecture, typically
    extracted from a HuggingFace configuration object's text config.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float8_e4m3fn",
    }

    boi_token_index: int
    """The begin-of-image token index to wrap the image prompt"""

    eoi_token_index: int
    """The end-of-image token index to wrap the image prompt"""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    dtype: DType
    """DType of the model weights and input."""

    kv_params: KVCacheParamInterface
    """KV cache parameters."""

    image_token_index: int
    """The image token index to encode the image prompt"""

    initializer_range: float
    """Standard deviation for weight initialization."""

    interleaved_rope_weights: bool
    """True if the rope weights are in interleaved complex format."""

    mm_tokens_per_image: int
    """The number of tokens per image embedding"""

    return_logits: ReturnLogits
    """Whether to return the last token, all logits, or a variable number of logits."""

    tie_word_embeddings: bool
    """Whether to tie weight embeddings. When true, the output linear layer
    uses the same
    weight as the embedding layer."""

    text_config: Gemma3Config
    """The config object of the text backbone"""

    vision_config: Gemma3VisionConfig
    """Custom vision config or dict"""

    attention_bias: bool = False
    """Whether to use a bias in the query, key, value and output projection layers during self-attention."""

    quant_config: QuantConfig | None = None
    """Scaled quantization configuration."""

    quantization_encoding: SupportedEncoding | None = None
    """The resolved quantization encoding the model runs with."""

    head_dim: int = 256
    """The attention head dimension."""

    num_key_value_heads: int = 4
    """
    This is the number of key_value heads that should be used to implement Grouped Query Attention. If
    `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
    `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
    converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed"
    """

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.text_config.num_hidden_layers

    @classmethod
    def construct_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        # ArchVLConfigWithTextSubconfig reads text_config from annotations,
        # which stay strings under from __future__ import annotations, so it
        # would otherwise build a single full-attention leaf.
        return Gemma3Config.construct_kv_params(
            huggingface_config=huggingface_config.text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
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
        """Initializes a Gemma3ForConditionalGenerationConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            A Gemma3ForConditionalGenerationConfig instance with fields initialized from config.
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
        """Initializes a Gemma3ForConditionalGenerationConfig from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be
        determined from the pipeline and HuggingFace configurations, without
        needing the state_dict. Fields that depend on the state_dict should
        be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            A Gemma3ForConditionalGenerationConfig instance ready for finalization.
        """
        interleaved_rope_weights = _interleaved_rope_weights(
            pipeline_config.model
        )
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]

        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )

        # Generate the vision config from HuggingFace config
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        vision_config = Gemma3VisionConfig.initialize_from_config(
            hf_vision_config
        )

        # Generate the text config from HuggingFace config
        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        text_config = Gemma3Config.initialize_from_config(
            pipeline_config=pipeline_config,
            huggingface_config=hf_text_config,
            max_seq_len=max_seq_len,
        )

        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=pipeline_config.model.kv_cache,
            cache_dtype=cache_dtype,
        )

        return cls(
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            return_logits=ReturnLogits.LAST_TOKEN,  # Default, will be updated in finalize
            kv_params=kv_params,
            vision_config=vision_config,
            text_config=text_config,
            mm_tokens_per_image=huggingface_config.mm_tokens_per_image,
            boi_token_index=huggingface_config.boi_token_index,
            eoi_token_index=huggingface_config.eoi_token_index,
            image_token_index=huggingface_config.image_token_index,
            initializer_range=0.0,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Finalize the Gemma3ForConditionalGenerationConfig instance with state_dict dependent fields.

        Args:
            huggingface_config: HuggingFace model configuration.
            state_dict: Model weights dictionary.
            return_logits: Return logits configuration.
        """
        # Parse the float8 config from compressed-tensors
        layer_name_prefix = "language_model.model."
        quant_config = parse_quant_config(
            huggingface_config,
            state_dict,
            self.dtype,
            state_dict_name_prefix=layer_name_prefix,
            ignored_modules_prefix=layer_name_prefix,
        )

        self.quant_config = quant_config
        self.return_logits = return_logits

        # Finalize text config
        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        self.text_config.finalize(
            huggingface_config=hf_text_config,
            state_dict=state_dict,
            return_logits=return_logits,
            quant_config=quant_config,
        )
