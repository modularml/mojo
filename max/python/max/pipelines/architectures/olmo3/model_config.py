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
from max.nn import ReturnLogits, YarnScalingParams
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
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
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class Olmo3Config(
    ArchConfigWithPermissiveMaxSeqLen,
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
):
    """Configuration for Olmo3 models.

    Contains parameters specific to the Olmo3 architecture, typically
    extracted from a HuggingFace configuration object's text config.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    vocab_size: int
    """Vocabulary size of the Olmo3 model."""

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
    """Dimension of each attention head."""

    hidden_activation: str
    """The non-linear activation function (function or string) in the decoder.
    Will default to `"silu"` if not specified."""

    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""

    rms_norm_eps: float
    """The epsilon used by the rms normalization layers."""

    tie_word_embeddings: bool
    """Whether to tie weight embeddings. When true, the output linear layer
    uses the same weight as the embedding layer."""

    rope_theta: float
    """The base period of the RoPE embeddings."""

    attention_bias: bool
    """Whether to use a bias in the query, key, value and output projection
    layers during self-attention."""

    sliding_window: int
    """In the Olmo3 language model, specific layers use sliding window
    attention. This is the size of the sliding window."""

    layer_types: list[str]
    """Type of attention for each layer ('full_attention' or 'sliding_attention')."""

    attention_dropout: float
    """Dropout probability for attention weights."""

    rope_scaling: YarnScalingParams | None
    """Scaling configuration for the RoPE embeddings used in global attention."""

    rope_scaling_type: str | None
    """Type of RoPE scaling (e.g., 'yarn', 'linear', etc.)."""

    query_pre_attn_scalar: float | None = None
    """Scalar applied to queries before attention computation."""

    final_logit_softcapping: float | None = None
    """Softcapping value for final logits."""

    attn_logit_softcapping: float | None = None
    """Softcapping value for attention logits."""

    qk_norm_eps: float
    """Epsilon value for Q and K normalization layers."""

    use_qk_norm: bool
    """Whether to use Q and K normalization."""

    use_cache: bool
    """Whether to use a cache."""

    # Max-specific config parameters.
    dtype: DType
    """DType of the model weights and input."""

    devices: list[DeviceRef]
    """Devices to run the model with."""

    interleaved_rope_weights: bool
    """True if the rope weights are in interleaved complex format."""

    return_logits: ReturnLogits
    """Whether to return the last token, all logits, or a variable number of logits."""

    kv_params: KVCacheParamInterface
    """KV cache parameters (sliding-window and full-attention groups)."""

    quantization_encoding: SupportedEncoding | None = None

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
        layer_types = getattr(
            huggingface_config,
            "layer_types",
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ]
            * (huggingface_config.num_hidden_layers // 4),
        )
        return hybrid_swa_full_kv_params(
            layer_types=layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
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
        """Initializes a Olmo3Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing to state_dict.
        Fields that depend on state_dict (like tie_word_embeddings)
        should be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized Olmo3Config instance.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        assert huggingface_config is not None
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )

        interleaved_rope_weights = _interleaved_rope_weights(model_config)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        rope_scaling_params: YarnScalingParams | None = None
        rope_scaling = huggingface_config.rope_scaling
        rope_scaling_type = None

        if rope_scaling is not None:
            rope_scaling_type = rope_scaling.get("type") or rope_scaling.get(
                "rope_type"
            )
            if rope_scaling_type is None:
                raise ValueError(
                    "Neither 'type' nor 'rope_type' found in rope_scaling huggingface config"
                )
            if rope_scaling_type == "linear":
                raise ValueError(
                    "Linear scaling is not supported for Olmo3 models"
                )
            elif rope_scaling_type == "yarn":
                rope_scaling_params = YarnScalingParams(
                    factor=rope_scaling.get("factor", 32.0),
                    beta_fast=rope_scaling.get("beta_fast", 32.0),
                    beta_slow=rope_scaling.get("beta_slow", 1.0),
                    original_max_position_embeddings=rope_scaling.get(
                        "original_max_position_embeddings", 4096
                    ),
                    truncate=rope_scaling.get("truncate", False),
                )
            else:
                raise ValueError(
                    f"Unknown rope scaling type: {rope_scaling_type}"
                )

        hidden_activation = _HIDDEN_ACTIVATION_MAP.get(
            huggingface_config.hidden_act,
            huggingface_config.hidden_act,
        )
        if hidden_activation is None:
            hidden_activation = huggingface_config.hidden_act

        layer_types = getattr(
            huggingface_config,
            "layer_types",
            [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ]
            * (huggingface_config.num_hidden_layers // 4),
        )

        query_pre_attn_scalar = getattr(
            huggingface_config, "query_pre_attn_scalar", None
        )
        final_logit_softcapping = getattr(
            huggingface_config, "final_logit_softcapping", None
        )
        attn_logit_softcapping = getattr(
            huggingface_config, "attn_logit_softcapping", None
        )

        use_qk_norm = getattr(huggingface_config, "use_qk_norm", True)
        qk_norm_eps = getattr(
            huggingface_config, "qk_norm_eps", huggingface_config.rms_norm_eps
        )

        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        return cls(
            vocab_size=huggingface_config.vocab_size,
            hidden_size=huggingface_config.hidden_size,
            intermediate_size=huggingface_config.intermediate_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            head_dim=Olmo3Config.get_head_dim(huggingface_config),
            hidden_activation=hidden_activation,
            max_position_embeddings=max_seq_len,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_theta=get_rope_theta(huggingface_config),
            attention_bias=huggingface_config.attention_bias,
            sliding_window=huggingface_config.sliding_window,
            rope_scaling=rope_scaling_params,
            rope_scaling_type=rope_scaling_type,
            layer_types=layer_types,
            attention_dropout=getattr(
                huggingface_config, "attention_dropout", 0.0
            ),
            query_pre_attn_scalar=query_pre_attn_scalar,
            final_logit_softcapping=final_logit_softcapping,
            attn_logit_softcapping=attn_logit_softcapping,
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_cache=getattr(huggingface_config, "use_cache", True),
            dtype=dtype,
            devices=device_refs,
            interleaved_rope_weights=interleaved_rope_weights,
            kv_params=kv_params,
            tie_word_embeddings=False,
            return_logits=ReturnLogits.LAST_TOKEN,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config.

        Args:
            huggingface_config: The HuggingFace model configuration object.
            state_dict: The model's state dictionary containing weights.
            return_logits: Whether to return the last token, all tokens or a variable number of logits.
        """
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )

        self.tie_word_embeddings = tie_word_embeddings
        self.return_logits = return_logits

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Retrieves the number of hidden layers from the HuggingFace configuration.

        Args:
            huggingface_config: The HuggingFace model configuration object (:obj:`transformers.AutoConfig`).

        Returns:
            The number of hidden layers specified in the configuration.
        """
        return huggingface_config.num_hidden_layers


_HIDDEN_ACTIVATION_MAP = {
    "gelu_pytorch_tanh": "gelu_tanh",
    "swish": "silu",
}
