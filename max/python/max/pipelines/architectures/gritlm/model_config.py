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


"""Config for GritLM models (ModuleV3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, WeightsFormat, weights_format
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    upper_bounded_default,
)
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class GritLMConfig(ArchConfigWithKVCache):
    """Model configuration for GritLM graph construction/execution.

    GritLM is architecturally identical to Mistral-7B for the causal LM path,
    with sliding window attention on every layer (window=4096).
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    kv_params: KVCacheParams
    devices: list[DeviceRef]

    # Sliding window size — from config.json: sliding_window=4096
    # GritLM applies SWA on every layer (no alternating pattern).
    sliding_window: int | None = 4096

    # Standard optional fields
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    norm_method: Literal["rms_norm"] = "rms_norm"
    rms_norm_eps: float | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float = 0.0
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    clip_qkv: float | None = None
    logits_scaling: float = 1.0

    # ------------------------------------------------------------------ #
    # ArchConfigWithKVCache interface
    # ------------------------------------------------------------------ #

    def get_kv_params(self) -> KVCacheParams:
        return self.kv_params

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_head_dim(huggingface_config: AutoConfig) -> int:
        """Return the per-head dimension for attention.

        Uses the explicit ``head_dim`` field when present; otherwise derives
        it as ``hidden_size // num_attention_heads``.

        Args:
            huggingface_config: HuggingFace model configuration.

        Returns:
            Integer head dimension.
        """

        if hasattr(huggingface_config, "head_dim"):
            return huggingface_config.head_dim
        return (
            huggingface_config.hidden_size
            // huggingface_config.num_attention_heads
        )

    @staticmethod
    def get_head_dim_from_config(config: GritLMConfig) -> int:
        """Return the per-head dimension stored in a ``GritLMConfig``.

        Args:
            config: An initialized ``GritLMConfig`` instance.

        Returns:
            Integer head dimension from ``kv_params``.
        """

        return config.kv_params.head_dim

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Return the number of hidden layers from a HuggingFace config.

        Args:
            huggingface_config: HuggingFace model configuration.

        Returns:
            Number of transformer layers.
        """

        return huggingface_config.num_hidden_layers

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """Compute the QK scaling factor for GritLM attention.

        Uses ``attention_multiplier`` when present in the config; otherwise
        falls back to ``1 / sqrt(head_dim)`` (standard Mistral scaling).

        Args:
            huggingface_config: HuggingFace model configuration.

        Returns:
            Float scaling factor applied to query-key dot products.
        """

        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(GritLMConfig.get_head_dim(huggingface_config))
            ),
        )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Build ``KVCacheParams`` for GritLM from pipeline and model config.

        Args:
            huggingface_config: HuggingFace model configuration.
            pipeline_config: MAX Engine pipeline configuration.
            devices: Device references for cache placement.
            kv_cache_config: KV cache configuration (page size, dtype, etc.).
            cache_dtype: Data type for cached key/value tensors.

        Returns:
            Configured ``KVCacheParams`` instance.
        """

        return kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=GritLMConfig.get_head_dim(huggingface_config),
            num_layers=GritLMConfig.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Resolve the effective maximum sequence length for GritLM.

        Clamps ``pipeline_config.model.max_length`` to the model's
        ``max_position_embeddings`` (default 32 768).

        Args:
            pipeline_config: MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Effective maximum sequence length as an integer.

        Raises:
            ValueError: If the requested ``max_length`` exceeds
                ``max_position_embeddings``.
        """

        hf_max = getattr(huggingface_config, "max_position_embeddings", 32768)
        try:
            return upper_bounded_default(
                upper_bound=hf_max,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for GritLM; provided "
                f"max_length ({pipeline_config.model.max_length}) exceeds "
                f"model max_position_embeddings ({hf_max})."
            ) from e

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

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
                f"HuggingFace config is required for '{model_config.model_path}'"
            )

        kv_cache_config = model_config.kv_cache
        quantization_encoding = model_config.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")

        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = model_config.kv_cache.cache_dtype

        _weights_format = weights_format(model_config.weight_path)
        interleaved_rope_weights = (
            _weights_format == WeightsFormat.gguf
            and model_config.rope_type == "normal"
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        # GritLM sliding_window: all layers use SWA with window=4096
        raw_sliding_window: int | None = getattr(
            huggingface_config, "sliding_window", None
        )

        sliding_window: int | None = (
            raw_sliding_window if raw_sliding_window else None
        )

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=get_rope_theta(huggingface_config),
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            sliding_window=sliding_window,
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
            devices=device_refs,
            tie_word_embeddings=getattr(
                huggingface_config, "tie_word_embeddings", False
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
        """Populate runtime fields that require the loaded state dict.

        Called after ``initialize`` once the weight names are known.
        Detects stacked MLP/QKV layouts, resolves ``tie_word_embeddings``,
        and stores norm/logit settings.

        Args:
            huggingface_config: HuggingFace model configuration.
            state_dict: Loaded (and remapped) weight dictionary.
            return_logits: Which logits the model should return.
            return_hidden_states: Which hidden states to return.
            norm_method: Normalization method (always ``"rms_norm"`` for GritLM).
            attention_bias: Whether attention projections include bias terms.
        """

        def _strip(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_model = any(k.startswith("model.") for k in state_dict)
        normalized = (
            {
                _strip(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            if has_model
            else dict(state_dict)
        )

        tie = (
            getattr(huggingface_config, "tie_word_embeddings", False)
            or "lm_head.weight" not in normalized
        )

        self.norm_method = norm_method
        self.rms_norm_eps = (
            huggingface_config.rms_norm_eps
            if norm_method == "rms_norm"
            else None
        )
        self.tie_word_embeddings = tie
        self.stacked_mlp = "layers.0.mlp.gate_up_proj.weight" in normalized
        self.stacked_qkv = "layers.0.self_attn.qkv_proj.weight" in normalized
        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
