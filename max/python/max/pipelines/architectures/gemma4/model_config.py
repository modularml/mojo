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
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma3.model_config import (
    _HIDDEN_ACTIVATION_MAP,
    Gemma3Config,
)
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config.model_config import (
    _interleaved_rope_weights,
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ArchConfigWithVisionCache,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights.quant import parse_quant_config
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self, override

from .layers.rotary_embedding import ProportionalScalingParams


def _resolve_num_global_kv_heads(text_config: PretrainedConfig) -> int:
    """Returns the number of KV heads used by full-attention layers.

    The Gemma 4 E*B checkpoints ship ``"num_global_key_value_heads": null``;
    the transformers ``configuration_gemma4.py`` docstring defines null/absent
    as "defaults to ``num_key_value_heads``". Note transformers never applies
    that fallback in code -- its modeling only consults the field when
    ``attention_k_eq_v`` is true (false on E*B), so resolving here matches the
    HF runtime behavior.
    """
    num_global_kv_heads = getattr(
        text_config, "num_global_key_value_heads", None
    )
    if num_global_kv_heads is None:
        return text_config.num_key_value_heads
    return num_global_kv_heads


@dataclass(kw_only=True)
class Gemma4TextConfig(Gemma3Config):
    """Text configuration for Gemma 4 models.

    Defaults are from gemma-4-31b-it.

    The following parameters are ignored:
      - routed_layer_pattern
      - stream_and_decode_in_f32
    """

    vocab_size_per_layer_input: int = 262144
    """Vocab size used in the per-layer input embedding (used in smaller architectures)."""

    hidden_size_per_layer_input: int = 0
    """Hidden size output of the per-layer input embedding. When this is 0, the
    per-layer input embedding is not used."""

    num_global_key_value_heads: int = 4
    """Number of key value heads used in full attention layers."""

    global_head_dim: int = 512
    """Head dimension used in full attention layers."""

    attention_k_eq_v: bool = True
    """If the key and value projections are the same.

    When true, the checkpoint will not contain `v_proj` and `v_norm` weights.
    """

    num_kv_shared_layers: int = 0
    """An optimization used in smaller models to share the kv cache across layers."""

    fused_projection_weights: bool = False
    """Load MLP gate/up and attention qkv/qk projections pre-fused (DISTINF-194).

    When true (single-device, non-quantized only), each decoder layer's MLP uses
    :class:`~max.nn.FusedMLP` (one ``gate_up_proj_fused`` weight) and attention
    uses ``StackedLinear(stacked=True)`` (one ``qkv_proj``/``qk_proj`` weight),
    and the weight adapter concatenates the checkpoint projections accordingly.
    Avoids the in-graph concat the compiler would otherwise hoist to model init
    and materialize as a second on-device copy. The gen-mef build tool sets
    this after config finalization when ``--engine-owned-weights`` is enabled."""

    enable_moe_block: bool = False
    """If the model uses MOE."""

    use_double_wide_mlp: bool = False
    """If the model uses a double wide MLP."""

    num_experts: int = 0
    """Total number of MoE experts."""

    top_k_experts: int = 0
    """Number of experts selected per token by the router."""

    moe_intermediate_size: int = 0
    """Hidden dimension of each MoE expert's feed-forward block."""

    global_rope_scaling: ProportionalScalingParams | None = None
    """Scaling configuration for the RoPE embeddings used in global attention."""

    global_rope_theta: float = 1000000.0
    """Rope theta used for the RoPE embeddings used in global attention."""

    sliding_window_rope_theta: float = 10000.0
    """Rope theta used for the RoPE embeddings used in sliding window attention."""

    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    """Which hidden states the text model returns alongside logits."""

    target_layer_ids: list[int] | None = None
    """For ``ReturnHiddenStates.SELECTED_LAYERS``, the zero-based layer
    indices whose post-block hidden states are captured and concatenated
    along the feature dimension (used by spec-decode drafters)."""

    layer_types: list[str]

    max_seq_len: int
    """actual max seq length determined by calculate_max_seq_len"""

    # Removed from HF config
    query_pre_attn_scalar: float | None = None
    attn_logit_softcapping: int | None = None

    @property  # type: ignore[misc]
    def rope_theta(self) -> float:
        raise ValueError(
            "rope_theta is not supported for Gemma4TextConfig. Use"
            " global_rope_theta or sliding_window_rope_theta instead."
        )

    @rope_theta.setter
    def rope_theta(self, value: float) -> None:
        pass

    @property  # type: ignore[misc, override]
    def rope_scaling(self) -> ProportionalScalingParams | None:
        raise ValueError(
            "rope_scaling is not supported for Gemma4TextConfig. Use"
            " global_rope_scaling or sliding_window_rope_scaling instead."
        )

    @rope_scaling.setter
    def rope_scaling(self, value: ProportionalScalingParams | None) -> None:
        pass

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        # Gemma3Config (parent) is permissive; Gemma4 text uses upper-bounded
        # max_length semantics instead.
        return ArchConfigWithStoredKVParams.calculate_max_seq_len(
            huggingface_config, model_config
        )

    @override
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
        """Constructor for hybrid sliding + full KV tree.

        Gemma 4 HF text configs list attention type per layer and may use a
        wider full-attention head (``global_head_dim``,
        ``num_global_key_value_heads``) than the sliding-window leaf.
        """
        return hybrid_swa_full_kv_params(
            layer_types=huggingface_config.layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            allow_kv_head_replication=allow_kv_head_replication,
            full_n_kv_heads=_resolve_num_global_kv_heads(huggingface_config),
            full_head_dim=huggingface_config.global_head_dim,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initialize Gemma4TextConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace text model configuration.

        Returns:
            An initialized Gemma4TextConfig instance.
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

        # Extract global rope scaling parameters.
        global_rope_params = huggingface_config.rope_parameters[
            "full_attention"
        ]
        global_rope_type = global_rope_params.get("rope_type")
        if global_rope_type != "proportional":
            raise ValueError(
                f"Global rope type {global_rope_type} not supported"
            )
        partial_rotary_factor = global_rope_params.get("partial_rotary_factor")
        global_rope_scaling = ProportionalScalingParams(
            partial_rotary_factor=partial_rotary_factor,
        )
        global_rope_theta = global_rope_params["rope_theta"]

        # Extract sliding window rope scaling parameters.
        sliding_window_rope_params = huggingface_config.rope_parameters[
            "sliding_attention"
        ]
        sliding_window_rope_type = sliding_window_rope_params.get("rope_type")
        if sliding_window_rope_type != "default":
            raise ValueError(
                f"Sliding window rope type {sliding_window_rope_type} not supported"
            )
        sliding_window_rope_theta = sliding_window_rope_params["rope_theta"]

        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            huggingface_config.hidden_activation,
            huggingface_config.hidden_activation,
        )

        return cls(
            # Parent Gemma3Config fields
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            hidden_activation=hidden_activation,
            max_position_embeddings=huggingface_config.max_position_embeddings,
            max_seq_len=max_seq_len,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            # Gemma4 uses different ropes for global and sliding window attention
            rope_theta=-1,
            rope_scaling=None,
            attention_bias=huggingface_config.attention_bias,
            sliding_window=huggingface_config.sliding_window,
            final_logit_softcapping=huggingface_config.final_logit_softcapping,
            rope_local_base_freq=sliding_window_rope_theta,
            sliding_window_pattern=-1,  # unused in Gemma4.
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            # Gemma4-specific fields
            vocab_size_per_layer_input=huggingface_config.vocab_size_per_layer_input,
            hidden_size_per_layer_input=huggingface_config.hidden_size_per_layer_input,
            num_global_key_value_heads=_resolve_num_global_kv_heads(
                huggingface_config
            ),
            global_head_dim=huggingface_config.global_head_dim,
            attention_k_eq_v=huggingface_config.attention_k_eq_v,
            num_kv_shared_layers=huggingface_config.num_kv_shared_layers,
            enable_moe_block=huggingface_config.enable_moe_block,
            use_double_wide_mlp=huggingface_config.use_double_wide_mlp,
            num_experts=huggingface_config.num_experts or 0,
            top_k_experts=huggingface_config.top_k_experts or 0,
            moe_intermediate_size=getattr(
                huggingface_config, "moe_intermediate_size", 0
            ),
            global_rope_scaling=global_rope_scaling,
            global_rope_theta=global_rope_theta,
            sliding_window_rope_theta=sliding_window_rope_theta,
            layer_types=huggingface_config.layer_types,
            quantization_encoding=quantization_encoding,
        )


@dataclass
class Gemma4VisionConfig:
    """Vision-specific configuration for Gemma 4 models."""

    hidden_size: int
    """Dimensionality of the encoder layers."""

    intermediate_size: int
    """Dimension of the MLP representations."""

    num_hidden_layers: int
    """Number of hidden layers in the vision Transformer encoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer."""

    num_key_value_heads: int
    """Number of key-value heads for grouped-query attention."""

    head_dim: int
    """Dimension of each attention head."""

    hidden_activation: str
    """The non-linear activation function in the encoder."""

    rms_norm_eps: float
    """The epsilon used by the RMS normalization layers."""

    max_position_embeddings: int
    """The maximum sequence length supported by position embeddings."""

    patch_size: int
    """The size (resolution) of each patch."""

    position_embedding_size: int
    """Size of the position embedding table."""

    pooling_kernel_size: int
    """Kernel size for spatial pooling."""

    standardize: bool = False
    """Whether to standardize the image features."""

    attention_bias: bool = False
    """Whether to use bias in attention projection layers."""

    attention_dropout: float = 0.0
    """The dropout ratio for the attention probabilities."""

    use_bidirectional_attention: str | None = "vision"
    """Controls bidirectional attention scope. ``"all"``, ``"vision"``, or ``None``."""

    layer_types: list[str] | None = None
    """Per-layer attention type specification (e.g. ``"full_attention"``)."""

    use_clipped_linears: bool = False
    """Whether to use clipped linear layers."""

    rope_theta: float = 100.0

    @classmethod
    def initialize_from_config(
        cls, hf_vision_config: AutoConfig
    ) -> Gemma4VisionConfig:
        """Initialize Gemma4VisionConfig from a HuggingFace vision config.

        Args:
            hf_vision_config: The HuggingFace vision configuration object.

        Returns:
            An initialized Gemma4VisionConfig instance.
        """
        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            hf_vision_config.hidden_activation,
            hf_vision_config.hidden_activation,
        )

        layer_types = getattr(hf_vision_config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "full_attention"
            ] * hf_vision_config.num_hidden_layers

        rope_parameters = hf_vision_config.rope_parameters
        if "rope_theta" in rope_parameters:
            rope_theta = rope_parameters["rope_theta"]
        elif "full_attention" in rope_parameters:
            rope_theta = rope_parameters["full_attention"]["rope_theta"]
        else:
            raise ValueError(f"Unknown rope parameters: {rope_parameters}")

        return cls(
            hidden_size=hf_vision_config.hidden_size,
            intermediate_size=hf_vision_config.intermediate_size,
            num_hidden_layers=hf_vision_config.num_hidden_layers,
            num_attention_heads=hf_vision_config.num_attention_heads,
            num_key_value_heads=hf_vision_config.num_key_value_heads,
            head_dim=hf_vision_config.head_dim,
            hidden_activation=hidden_activation,
            rms_norm_eps=hf_vision_config.rms_norm_eps,
            max_position_embeddings=hf_vision_config.max_position_embeddings,
            patch_size=hf_vision_config.patch_size,
            position_embedding_size=hf_vision_config.position_embedding_size,
            pooling_kernel_size=hf_vision_config.pooling_kernel_size,
            standardize=getattr(hf_vision_config, "standardize", False),
            attention_bias=getattr(hf_vision_config, "attention_bias", False),
            attention_dropout=getattr(
                hf_vision_config, "attention_dropout", 0.0
            ),
            use_bidirectional_attention=getattr(
                hf_vision_config, "use_bidirectional_attention", "vision"
            ),
            layer_types=layer_types,
            use_clipped_linears=getattr(
                hf_vision_config, "use_clipped_linears", False
            ),
            rope_theta=rope_theta,
        )


@dataclass(kw_only=True)
class Gemma4ForConditionalGenerationConfig(
    ArchConfigWithKVCache, ArchConfigWithVisionCache
):
    """Base configuration for Gemma 4 multimodal models.

    This is the top-level config that composes text and vision sub-configs.
    Model-specific parameters live in the respective sub-configs.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float16",
        "float4_e2m1fnx2",
    }

    devices: list[DeviceRef]
    """Devices to run the model with."""

    dtype: DType
    """DType of the model weights and input."""

    unquantized_dtype: DType = DType.bfloat16
    """DType of unquantized weights."""

    kv_params: MultiKVCacheParams
    """KV cache parameters."""

    image_token_index: int
    """The image token index to encode the image prompt."""

    video_token_index: int = 262_144
    """The video token index to encode the video prompt."""

    text_config: Gemma4TextConfig
    """The config object of the text backbone."""

    vision_config: Gemma4VisionConfig | None
    """The config object of the vision encoder, or ``None`` for checkpoints
    served text-only (the ``gemma4_unified`` line)."""

    tie_word_embeddings: bool = False
    """Whether to tie weight embeddings. When true, the output linear layer
    uses the same weight as the embedding layer."""

    quantization_encoding: SupportedEncoding | None = None
    """The resolved quantization encoding the model runs with."""

    def get_kv_params(self) -> MultiKVCacheParams:
        """Returns the KV cache parameters."""
        return self.kv_params

    def get_max_seq_len(self) -> int:
        """Returns the maximum sequence length from the embedded text config."""
        return self.text_config.get_max_seq_len()

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        """Constructs KV cache parameters from the top-level HuggingFace config.

        Args:
            huggingface_config: The top-level HuggingFace config (with ``text_config``).
            pipeline_config: The MAX Engine pipeline configuration.
            devices: Target devices for the model.
            kv_cache_config: KV cache configuration settings.
            cache_dtype: Data type for the KV cache.

        Returns:
            Configured KV cache parameters.
        """
        return Gemma4TextConfig.construct_kv_params(
            huggingface_config=huggingface_config.text_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Calculates the maximum sequence length for the Gemma 4 model."""
        return Gemma4TextConfig.calculate_max_seq_len(
            huggingface_config.text_config, model_config
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
        """Initializes from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: Optional model config override.

        Returns:
            An initialized config instance.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for"
                f" '{model_config.model_path}', but config could not be loaded."
                " Please ensure the model repository contains a valid"
                " config.json file."
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
        """Initializes from pipeline and HuggingFace configs.

        Fields that depend on the state_dict should be set via ``finalize()``.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: Top-level HuggingFace model configuration.

        Returns:
            A config instance ready for finalization.
        """
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

        tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )

        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")
        vision_config: Gemma4VisionConfig | None
        if getattr(huggingface_config, "model_type", None) == "gemma4_unified":
            # These checkpoints (e.g. google/gemma-4-12B-it) carry a
            # lightweight vision_embedder with a different schema that is not
            # implemented yet; serve text-only. Keyed on model_type so a
            # genuinely malformed full-vision config fails loudly below
            # instead of silently degrading to text-only.
            vision_config = None
        else:
            vision_config = Gemma4VisionConfig.initialize_from_config(
                hf_vision_config
            )

        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")
        text_config = Gemma4TextConfig.initialize_from_config(
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
            kv_params=kv_params,
            vision_config=vision_config,
            text_config=text_config,
            image_token_index=huggingface_config.image_token_id,
            video_token_index=getattr(
                huggingface_config, "video_token_id", 262_144
            ),
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Finalize with state_dict-dependent fields.

        Parses quantization config from the weights and finalizes the text
        sub-config.

        Args:
            huggingface_config: HuggingFace model configuration.
            state_dict: Model weights dictionary.
            return_logits: Return logits configuration.
        """

        hf_text_config = getattr(huggingface_config, "text_config", None)
        if hf_text_config is None:
            raise ValueError("text_config not found in huggingface_config")

        quant_config = parse_quant_config(
            huggingface_config,
            state_dict,
            self.dtype,
            # Gemma 4 checkpoints nest the language tower under
            # "model.language_model."; without these prefixes the per-layer
            # quantized/ignored classification looks up "model.layers.*"
            # keys that never exist, so ignore-listed (BF16) attention in
            # modelopt 12B quants was never recognized as ignored.
            state_dict_name_prefix="model.language_model.",
            ignored_modules_prefix="model.language_model.",
        )

        for k, v in state_dict.items():
            if k.endswith("layers.0.input_layernorm.weight"):
                self.unquantized_dtype = v.dtype
                break

        self.text_config.finalize(
            huggingface_config=hf_text_config,
            state_dict=state_dict,
            return_logits=return_logits,
            quant_config=quant_config,
        )

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates per-entry bytes for the Gemma4 vision encoder cache.

        Worst-case tokens per image is
        ``position_embedding_size / pooling_kernel_size²``, stored at the text
        hidden size in bfloat16.

        Args:
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated bytes per vision cache entry.

        Raises:
            ValueError: If the required vision or text config is absent.
        """
        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "Gemma4 requires a vision_config in the HuggingFace config"
            )
        text_config = getattr(huggingface_config, "text_config", None)
        if text_config is None:
            raise ValueError(
                "Gemma4 requires a text_config in the HuggingFace config"
            )
        if getattr(huggingface_config, "model_type", None) == "gemma4_unified":
            # These checkpoints are served text-only (different vision
            # schema); no vision cache is needed.
            return 0
        k = vision_config.pooling_kernel_size
        max_tokens = vision_config.position_embedding_size // (k * k)
        spec = cls.get_vision_cache_row_spec(huggingface_config)
        if spec is None:
            return 0
        hidden, dtype = spec
        return max_tokens * hidden * dtype.size_in_bytes

    @classmethod
    def get_vision_cache_row_spec(
        cls,
        huggingface_config: AutoConfig,
    ) -> tuple[int, DType] | None:
        """One embedding row per merged vision token: text hidden, bfloat16.

        ``None`` for the text-only ``gemma4_unified`` checkpoints, which
        have no vision cache.
        """
        if getattr(huggingface_config, "model_type", None) == "gemma4_unified":
            return None
        text_config = getattr(huggingface_config, "text_config", None)
        if text_config is None:
            raise ValueError(
                "Gemma4 requires a text_config in the HuggingFace config"
            )
        return (text_config.hidden_size, DType.bfloat16)
