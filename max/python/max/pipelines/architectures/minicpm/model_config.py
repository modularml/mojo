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

"""Config for MiniCPMForCausalLM models (ModuleV3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    upper_bounded_default,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class MiniCPMConfig(ArchConfigWithKVCache):
    """Model configuration for MiniCPM (MiniCPMForCausalLM)."""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    max_seq_len: int
    intermediate_size: int
    vocab_size: int
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]

    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0

    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    norm_method: Literal["rms_norm"] = "rms_norm"
    rms_norm_eps: float | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    attention_multiplier: float = 0.0

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        if (
            hasattr(huggingface_config, "head_dim")
            and huggingface_config.head_dim
        ):
            return huggingface_config.head_dim
        return (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_head_dim_from_config(config: MiniCPMConfig) -> int:
        return config.kv_params.head_dim

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(MiniCPMConfig.get_head_dim(huggingface_config))
            ),
        )

    @staticmethod
    def get_rope_theta(huggingface_config: AutoConfig) -> float:
        """Extract rope_theta, defaulting to MiniCPM's 10000.0 base."""
        rope_params = getattr(huggingface_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            if "rope_theta" in rope_params:
                return float(rope_params["rope_theta"])
            if "base" in rope_params:
                return float(rope_params["base"])
        return float(getattr(huggingface_config, "rope_theta", 10000.0))

    @staticmethod
    def calculate_embedding_multiplier(huggingface_config: AutoConfig) -> float:
        return float(getattr(huggingface_config, "scale_emb", 1.0))

    @staticmethod
    def calculate_residual_multiplier(huggingface_config: AutoConfig) -> float:
        scale_depth = getattr(huggingface_config, "scale_depth", None)
        if scale_depth is None:
            return 1.0
        num_layers = MiniCPMConfig.get_num_layers(huggingface_config)
        return float(scale_depth) / math.sqrt(float(num_layers))

    @staticmethod
    def calculate_logits_scaling(huggingface_config: AutoConfig) -> float:
        dim_model_base = getattr(huggingface_config, "dim_model_base", None)
        if dim_model_base is None:
            return 1.0
        return float(huggingface_config.hidden_size) / float(dim_model_base)

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=MiniCPMConfig.get_head_dim(huggingface_config),
            num_layers=MiniCPMConfig.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        hf_max = getattr(huggingface_config, "max_position_embeddings", 4096)
        try:
            return upper_bounded_default(
                upper_bound=hf_max,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                f"max_length ({pipeline_config.model.max_length}) exceeds "
                f"model max_position_embeddings ({hf_max})."
            ) from e

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config required for '{model_config.model_path}'"
            )

        kv_cache_config = model_config.kv_cache
        quantization_encoding = model_config.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")

        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = model_config.kv_cache.cache_dtype

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=cls.get_rope_theta(huggingface_config),
            intermediate_size=huggingface_config.intermediate_size,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config=huggingface_config
            ),
            kv_params=cls.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            attention_multiplier=cls.calculate_attention_multiplier(
                huggingface_config
            ),
            embedding_multiplier=cls.calculate_embedding_multiplier(
                huggingface_config
            ),
            residual_multiplier=cls.calculate_residual_multiplier(
                huggingface_config
            ),
            logits_scaling=cls.calculate_logits_scaling(huggingface_config),
            devices=device_refs,
            tie_word_embeddings=getattr(
                huggingface_config, "tie_word_embeddings", True
            ),
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        def _strip(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in state_dict)

        if has_lm_prefix:
            normalized = {
                _strip(k, "language_model."): v
                for k, v in state_dict.items()
                if k.startswith("language_model.")
            }
        elif has_model_prefix:
            normalized = {
                _strip(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized = dict(state_dict)
        tie_word_embeddings = (
            getattr(huggingface_config, "tie_word_embeddings", True)
            or "lm_head.weight" not in normalized
        )
        rms_norm_eps = (
            huggingface_config.rms_norm_eps
            if norm_method == "rms_norm"
            else None
        )

        self.norm_method = norm_method
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
