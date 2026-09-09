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
"""Config for Llama3 models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
    LongRoPERotaryEmbedding,
    LongRoPEScalingParams,
    RotaryEmbedding,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
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
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
    supported_encoding_quantization,
)
from max.pipelines.weights import gptq_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override


def create_rope_embedding(
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float,
    max_seq_len: int,
    interleaved_rope_weights: bool,
    rope_scaling_params: Llama3RopeScalingParams | None,
    longrope_scaling_params: LongRoPEScalingParams | None,
    device: DeviceRef,
) -> RotaryEmbedding:
    """Create appropriate RoPE embedding based on scaling parameters.

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Number of attention heads
        rope_theta: RoPE theta parameter (typically 10000.0)
        max_seq_len: Maximum sequence length
        interleaved_rope_weights: Whether to use interleaved RoPE weights
        rope_scaling_params: Llama3 RoPE scaling parameters (if any)
        longrope_scaling_params: LongRoPE scaling parameters (if any)
        device: Device to place tensors on

    Returns:
        Configured RoPE embedding instance
    """
    if longrope_scaling_params is not None:
        return LongRoPERotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            interleaved=interleaved_rope_weights,
            scaling_params=longrope_scaling_params,
        )
    else:
        return Llama3RotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            interleaved=interleaved_rope_weights,
            scaling_params=rope_scaling_params,
        )


@dataclass(kw_only=True)
class Llama3Config(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Model configuration for Llama3 graph construction/execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "q4_k"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "gptq",
        "q4_k",
        "q4_0",
        "q6_k",
        "float32",
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    }

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rope_theta: float
    rope_scaling_params: Llama3RopeScalingParams | None
    max_seq_len: int
    intermediate_size: int
    interleaved_rope_weights: bool
    vocab_size: int
    dtype: DType
    model_quantization_encoding: QuantizationEncoding | None
    quantization_config: QuantizationConfig | None
    kv_params: KVCacheParams
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    norm_dtype: DType | None = None
    attention_bias: bool = False
    rms_norm_eps: float | None = None
    tie_word_embeddings: bool = False
    stacked_mlp: bool = False
    stacked_qkv: bool = False
    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    devices: list[DeviceRef]
    clip_qkv: float | None
    quant_config: QuantConfig | None = None
    longrope_scaling_params: LongRoPEScalingParams | None = None
    logits_scaling: float = 1.0
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    target_layer_ids: list[int] | None = None
    use_subgraphs: bool = True
    data_parallel_degree: int = 1
    sliding_window: int | None = None
    quantization_encoding: SupportedEncoding | None = None

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
    ) -> float:
        """The attention multiplier is a scalar that scales the attention scores.
        It is used to control the variance of the attention scores.

        This function is used to get the attention multiplier from the
        huggingface config. If the attention multiplier is not set, it will be
        calculated as the square root of 1.0 divided by the head dimension.
        """
        # Base attention multiplier
        base_multiplier = getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(
                1.0 / float(Llama3Config.get_head_dim(huggingface_config))
            ),
        )
        # TODO(zheng): Figure out a scalable abstract method for all MAXModelConfigs.

        return base_multiplier

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
    ) -> KVCacheParams:
        """Grouped-attention KV with EAGLE draft-token count when speculative is on.

        ``allow_kv_head_replication`` lets subclasses with fewer KV heads
        than the tensor-parallel width replicate each head across a device
        group.
        """
        return kv_cache_config.to_params(
            dtype=cache_dtype,
            allow_kv_head_replication=allow_kv_head_replication,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=cls.get_head_dim(huggingface_config),
            num_layers=cls.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            speculative_method=pipeline_config.speculative.speculative_method
            if pipeline_config.speculative
            else None,
            num_draft_tokens=(
                pipeline_config.speculative.num_speculative_tokens or 0
            )
            if pipeline_config.speculative
            else 0,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

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
        return cls.initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )
        device_specs = model_config.device_specs
        n_devices = len(device_specs)

        interleaved_rope_weights = _interleaved_rope_weights(model_config)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in device_specs[:n_devices]
        ]

        embedding_multiplier = getattr(
            huggingface_config, "embedding_multiplier", 1.0
        )
        residual_multiplier = getattr(
            huggingface_config, "residual_multiplier", 1.0
        )
        rope_scaling_params: Llama3RopeScalingParams | None = None
        longrope_scaling_params: LongRoPEScalingParams | None = None
        rope_scaling = getattr(huggingface_config, "rope_scaling", None)

        if rope_scaling is not None:
            # Only the *flat* HuggingFace rope_scaling shape is supported
            # here, with either "type" or "rope_type" as the discriminator
            # (e.g. {"rope_type": "llama3", "factor": 8.0, ...}). The
            # nested per-layer-type shape used by some modern HF configs —
            # where `rope_parameters` is keyed by layer type such as
            # "full_attention" or "sliding_attention" — is not handled
            # here. Architectures using that shape need a config subclass
            # that pre-flattens the dominant layer-type's parameters
            # before delegating to this method.
            # Since "rope_type" huggingface config is not standardized, we need
            # to check for both "type" and "rope_type" keys.
            # TODO: A better solution would be for those family of models to
            # create their own subclass of MAXModelConfig or Llama3Config, then
            # parts of it like rope_scaling to account for such differences.
            rope_type = rope_scaling.get("type")
            rope_type_alt = rope_scaling.get("rope_type")
            if rope_type is None and rope_type_alt is None:
                raise ValueError(
                    "Neither 'type' nor 'rope_type' found in rope_scaling huggingface config"
                )
            if rope_type == "llama3" or rope_type_alt == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            elif rope_type == "longrope" or rope_type_alt == "longrope":
                longrope_scaling_params = LongRoPEScalingParams(
                    short_factor=rope_scaling["short_factor"],
                    long_factor=rope_scaling["long_factor"],
                    original_max_position=huggingface_config.original_max_position_embeddings,
                    max_position_embeddings=huggingface_config.max_position_embeddings,
                )
                rope_scaling_params = None

        # Calculate base attention multiplier
        base_attention_multiplier = Llama3Config.calculate_attention_multiplier(
            huggingface_config
        )

        # Apply LongRoPE attention scaling if needed
        attention_multiplier = base_attention_multiplier
        if longrope_scaling_params is not None:
            # Create temporary RoPE embedding to get proper attention scale
            rope_embedding = create_rope_embedding(
                hidden_size=huggingface_config.hidden_size,
                num_attention_heads=huggingface_config.num_attention_heads,
                rope_theta=get_rope_theta(huggingface_config),
                max_seq_len=max_seq_len,
                interleaved_rope_weights=interleaved_rope_weights,
                rope_scaling_params=rope_scaling_params,
                longrope_scaling_params=longrope_scaling_params,
                device=DeviceRef.CPU(),  # temporary device, not used for scale computation
            )
            attention_multiplier = rope_embedding.compute_scale()

        return cls(
            hidden_size=huggingface_config.hidden_size,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            rope_theta=get_rope_theta(huggingface_config),
            rope_scaling_params=rope_scaling_params,
            longrope_scaling_params=longrope_scaling_params,
            max_seq_len=max_seq_len,
            intermediate_size=huggingface_config.intermediate_size,
            interleaved_rope_weights=interleaved_rope_weights,
            vocab_size=huggingface_config.vocab_size,
            dtype=dtype,
            model_quantization_encoding=supported_encoding_quantization(
                quantization_encoding
            ),
            quantization_config=gptq_quant_config(
                quantization_encoding,
                pipeline_config.model.huggingface_config,
            ),
            kv_params=Llama3Config.construct_kv_params(
                huggingface_config=huggingface_config,
                pipeline_config=pipeline_config,
                devices=device_refs,
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            attention_multiplier=attention_multiplier,
            embedding_multiplier=embedding_multiplier,
            residual_multiplier=residual_multiplier,
            devices=device_refs,
            clip_qkv=getattr(huggingface_config, "clip_qkv", None),
            use_subgraphs=pipeline_config.model.use_subgraphs,
            logits_scaling=getattr(huggingface_config, "logits_scaling", 1.0),
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            quantization_encoding=quantization_encoding,
        )

    def _parse_quant_config(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
    ) -> QuantConfig | None:
        """Parses the checkpoint's quantization config during ``finalize``.

        An extension point for architectures whose quantization metadata
        ``parse_quant_config`` cannot read: it is a single uniform format keyed
        off one ``quant_algo``, resolved against the config object handed to
        ``finalize``. Qwen3.5 overrides this because its checkpoint mixes NVFP4
        and FP8 and keeps ``quantization_config`` beside ``text_config``.
        """
        return parse_quant_config(huggingface_config, state_dict, self.dtype)

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

        # Normalize the LLM state dict so downstream introspection sees canonical
        # Llama-style keys (no "language_model." or "model." prefix). This keeps
        # float8 parsing and feature detection resilient to pack variations.
        def _strip_prefix(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in state_dict)

        if has_lm_prefix:
            normalized_state_dict: dict[str, WeightData] = {
                _strip_prefix(k, "language_model."): v
                for k, v in state_dict.items()
                if k.startswith("language_model.")
            }
        elif has_model_prefix:
            normalized_state_dict = {
                _strip_prefix(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized_state_dict = dict(state_dict)

        quant_config = self._parse_quant_config(
            huggingface_config, normalized_state_dict
        )

        # Determine norm_dtype.
        # Note: due to automatic weight dtype casting, norm dtype is not always
        # correct. To avoid any issue, only set norm_dtype for float8 models
        # for now.
        norm_dtype = None
        if "layers.0.input_layernorm.weight" in normalized_state_dict:
            norm_dtype = normalized_state_dict[
                "layers.0.input_layernorm.weight"
            ].dtype

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        if "tie_word_embeddings" in huggingface_config:
            tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            tie_word_embeddings = (
                getattr(huggingface_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in normalized_state_dict
            )

        rms_norm_eps = None
        if norm_method == "rms_norm":
            rms_norm_eps = huggingface_config.rms_norm_eps

        self.norm_method = norm_method
        self.norm_dtype = norm_dtype
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.quant_config = quant_config
        self.stacked_mlp = (
            "layers.0.mlp.gate_up_proj.weight" in normalized_state_dict
        )
        self.stacked_qkv = (
            "layers.0.self_attn.qkv_proj.weight" in normalized_state_dict
        )

        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
