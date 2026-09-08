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
from max.experimental.sharding import DeviceMesh
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import LinearScalingParams
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithPermissiveMaxSeqLen,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class Gemma3Config(
    ArchConfigWithPermissiveMaxSeqLen,
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
):
    """Represents the MAX Engine configuration for Gemma 3 models.

    Contains parameters specific to the Gemma 3 architecture (typically extracted
    from HuggingFace configs), plus MAX-specific runtime settings and helpers.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    # Gemma 3 specific parameters (taken from Transformer's `configuration_gemma3.py`)
    vocab_size: int
    """Vocabulary size of the Gemma3Text model."""

    hidden_size: int
    """Dimension of the hidden representations."""

    intermediate_size: int
    """Dimension of the MLP representations."""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer decoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer in the Transformer
    decoder."""

    num_key_value_heads: int
    """Number of key_value heads that should be used to implement Grouped Query
    Attention."""

    head_dim: int
    """The attention head dimension."""

    hidden_activation: str
    """The non-linear activation function (function or string) in the decoder.
    Will default to `"gelu_tanh"` if not specified. `"gelu_tanh"`
    uses an approximation of the `"gelu"` activation function."""

    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""

    rms_norm_eps: float
    """The epsilon used by the rms normalization layers."""

    rope_theta: float
    """The base period of the RoPE embeddings."""

    attention_bias: bool
    """Whether to use a bias in the query, key, value and output projection
    layers during self-attention."""

    query_pre_attn_scalar: float | None
    """Scaling factor used on the attention scores."""

    sliding_window: int
    """In the Gemma3 language model, every other layer uses sliding window
    attention. This is the size of the sliding window."""

    final_logit_softcapping: float | None
    """Scaling factor when applying tanh softcapping on the logits."""

    attn_logit_softcapping: int | None
    """Scaling factor when applying tanh softcapping on the attention scores."""

    rope_scaling: LinearScalingParams | None
    """Scaling configuration for the RoPE embeddings used in global attention."""

    rope_local_base_freq: float
    """The base period of the RoPE embeddings for local attention."""

    sliding_window_pattern: int
    """Pattern for the sliding window attention."""

    # Max-specific config parameters.
    dtype: DType
    """DType of the model weights and input."""

    mesh: DeviceMesh
    """Device mesh to run the model with."""

    interleaved_rope_weights: bool
    """True if the rope weights are in interleaved complex format."""

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    """Whether to return the last token, all logits, or a variable number of logits."""

    kv_params: KVCacheParamInterface
    """KV cache parameters."""

    tie_word_embeddings: bool = False
    """Whether to tie weight embeddings. When true, the output linear layer
    uses the same weight as the embedding layer."""

    quant_config: QuantConfig | None = None
    """Scaled quantization configuration."""

    quantization_encoding: SupportedEncoding | None = None
    """The resolved quantization encoding the model runs with."""

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Retrieves the number of hidden layers from the HuggingFace configuration.

        Args:
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).

        Returns:
            The number of hidden layers specified in the configuration's text config.
        """
        return huggingface_config.num_hidden_layers

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
        """Constructor for hybrid sliding + full KV tree."""
        pattern = huggingface_config._sliding_window_pattern
        layer_types = [
            (
                "full_attention"
                if (i + 1) % pattern == 0
                else "sliding_attention"
            )
            for i in range(huggingface_config.num_hidden_layers)
        ]
        return hybrid_swa_full_kv_params(
            layer_types=layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
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
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        # If this is a multimodal config (has text_config), extract the text config
        # for text-only Gemma3Config initialization
        if hasattr(huggingface_config, "text_config"):
            huggingface_config = huggingface_config.text_config
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
        """Initializes a Gemma3Config instance from pipeline and HuggingFace configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline and HuggingFace configuration, without needing the state_dict.
        Fields that depend on the state_dict (like tie_word_embeddings, quant_config)
        should be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration object.

        Returns:
            An initialized :obj:`Gemma3Config` instance.
        """
        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )

        interleaved_rope_weights = _interleaved_rope_weights(
            pipeline_config.model
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs
        ]
        device_mesh = DeviceMesh(
            tuple(device.to_device() for device in device_refs),
            (len(device_refs),),
            ("tp",),
        )

        # transformers >= 5.0 moves rope config into nested rope_parameters;
        # transformers 4.x uses direct attributes.
        rope_params = getattr(huggingface_config, "rope_parameters", None)
        if isinstance(rope_params, dict) and "full_attention" in rope_params:
            rope_theta = rope_params["full_attention"]["rope_theta"]
            rope_local_base_freq = rope_params["sliding_attention"][
                "rope_theta"
            ]
            full_attn_params = rope_params["full_attention"]
            rope_type = full_attn_params.get(
                "rope_type", full_attn_params.get("type")
            )
            rope_scaling_params = None
            if rope_type == "linear":
                rope_scaling_params = LinearScalingParams(
                    factor=full_attn_params["factor"]
                )
        else:
            rope_theta = huggingface_config.rope_theta
            rope_local_base_freq = huggingface_config.rope_local_base_freq
            rope_scaling_params = None
            rope_scaling = huggingface_config.rope_scaling
            if rope_scaling is not None:
                # Since "rope_type" huggingface config is not standardized,
                # we need to check for both "type" and "rope_type" keys.
                rope_type = rope_scaling.get("type")
                rope_type_alt = rope_scaling.get("rope_type")
                if rope_type is None and rope_type_alt is None:
                    raise ValueError(
                        "Neither 'type' nor 'rope_type' found in"
                        " rope_scaling huggingface config"
                    )
                if rope_type == "linear" or rope_type_alt == "linear":
                    rope_scaling_params = LinearScalingParams(
                        factor=rope_scaling["factor"]
                    )

        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            huggingface_config.hidden_activation,
            huggingface_config.hidden_activation,
        )

        return cls(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=max_seq_len,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_theta=rope_theta,
            attention_bias=huggingface_config.attention_bias,
            query_pre_attn_scalar=huggingface_config.query_pre_attn_scalar,
            sliding_window=huggingface_config.sliding_window,
            final_logit_softcapping=huggingface_config.final_logit_softcapping,
            attn_logit_softcapping=huggingface_config.attn_logit_softcapping,
            rope_scaling=rope_scaling_params,
            rope_local_base_freq=rope_local_base_freq,
            sliding_window_pattern=huggingface_config._sliding_window_pattern,
            dtype=dtype,
            mesh=device_mesh,
            interleaved_rope_weights=interleaved_rope_weights,
            kv_params=Gemma3Config.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config.

        This method sets fields that require introspection of the model weights
        (state_dict), such as tie_word_embeddings and quant_config.

        Args:
            huggingface_config: The HuggingFace model configuration object.
            state_dict: The model's state dictionary containing weights.
            return_logits: Whether to return the last token, all tokens or a
                variable number of logits.
            quant_config: Scaled quantization configuration (optional).
        """
        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )

        self.tie_word_embeddings = tie_word_embeddings
        self.quant_config = quant_config
        self.return_logits = return_logits


_HIDDEN_ACTIVATION_MAP = {
    "gelu_pytorch_tanh": "gelu_tanh",
    "swish": "silu",
}
