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
"""Config for Olmo2 models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
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
class Olmo2Config(
    ArchConfigWithPermissiveMaxSeqLen,
    ArchConfigWithStoredKVParams,
    ArchConfigWithKVCache,
):
    """Configuration for Olmo2 models.

    Contains parameters specific to the Olmo2 architecture, typically
    extracted from a HuggingFace configuration object.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
    }

    quantization_encoding: SupportedEncoding | None = None

    vocab_size: int
    """Vocabulary size of the Olmo2 model."""

    hidden_size: int
    """Dimension of the hidden representations."""

    intermediate_size: int
    """Dimension of the MLP representations."""

    num_hidden_layers: int
    """Number of hidden layers in the Transformer decoder."""

    num_attention_heads: int
    """Number of attention heads for each attention layer."""

    num_key_value_heads: int
    """Number of key_value heads for Grouped Query Attention."""

    head_dim: int
    """Dimension of each attention head."""

    max_position_embeddings: int
    """The maximum sequence length that this model might ever be used with."""

    rms_norm_eps: float
    """The epsilon used by the rms normalization layers."""

    rope_theta: float
    """The base period of the RoPE embeddings."""

    attention_bias: bool
    """Whether to use a bias in the attention projection layers."""

    tie_word_embeddings: bool
    """Whether to tie weight embeddings."""

    attention_multiplier: float
    """Scalar applied to attention scores."""

    embedding_multiplier: float
    """Scalar applied to embeddings."""

    residual_multiplier: float
    """Scalar applied to residual connections."""

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
    """KV cache parameters."""

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
    ) -> KVCacheParamInterface:
        """Olmo2 does not support data parallelism; use default grouped KV (no EAGLE)."""
        if pipeline_config.model.data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported for Olmo2 models"
            )
        return ArchConfigWithStoredKVParams.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(Olmo2Config.get_head_dim(huggingface_config))
            ),
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
            head_dim=Olmo2Config.get_head_dim(huggingface_config),
            max_position_embeddings=max_seq_len,
            rms_norm_eps=huggingface_config.rms_norm_eps,
            rope_theta=get_rope_theta(huggingface_config),
            attention_bias=getattr(huggingface_config, "attention_bias", False),
            attention_multiplier=Olmo2Config.calculate_attention_multiplier(
                huggingface_config
            ),
            embedding_multiplier=getattr(
                huggingface_config, "embedding_multiplier", 1.0
            ),
            residual_multiplier=getattr(
                huggingface_config, "residual_multiplier", 1.0
            ),
            dtype=dtype,
            devices=device_refs,
            quantization_encoding=quantization_encoding,
            interleaved_rope_weights=interleaved_rope_weights,
            kv_params=kv_params,
            # Placeholder values; finalize() sets the real values once
            # we have the HuggingFace config and the weight state dict.
            tie_word_embeddings=False,
            return_logits=ReturnLogits.LAST_TOKEN,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config."""
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "language_model.lm_head.weight" not in state_dict
        )

        self.tie_word_embeddings = tie_word_embeddings
        self.return_logits = return_logits
