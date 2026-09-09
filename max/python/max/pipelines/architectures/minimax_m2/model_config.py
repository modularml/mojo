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
"""Config for MiniMax-M2 models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config


@dataclass(kw_only=True)
class MiniMaxM2Config(Llama3Config):
    """Configuration for MiniMax-M2 MoE models.

    Extends Llama3Config with MoE-specific parameters including sigmoid
    routing with expert score correction bias.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    }

    # MoE parameters
    num_local_experts: int = 256
    """Number of local experts in each MoE layer."""

    num_experts_per_tok: int = 8
    """Number of experts selected per token."""

    norm_topk_prob: bool = True
    """Whether to normalize top-k expert probabilities to sum to 1."""

    # Sigmoid routing parameters
    correction_bias_dtype: DType | None = None
    """Data type of the e_score_correction_bias weight. Detected from
    state dict during finalize()."""

    gate_dtype: DType | None = None
    """Data type for the gate linear layer. Detected from state dict
    during finalize()."""

    attn_dtype: DType | None = None
    """Data type for attention weights. Detected from state dict
    during finalize()."""

    ep_config: EPConfig | None = None
    """Expert parallelism configuration. None means no EP (single-GPU)."""

    # Partial RoPE
    partial_rotary_factor: float = 1.0
    """Fraction of head_dim used for rotary embeddings.
    For MiniMax-M2: rotary_dim/head_dim = 64/128 = 0.5."""

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """The attention multiplier for MiniMax-M2 models.

        Uses the explicit head_dim from the config.

        Args:
            huggingface_config: The HuggingFace configuration object.

        Returns:
            The attention multiplier value.
        """
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(huggingface_config.head_dim)),
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
        """Initializes a MiniMaxM2Config from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized MiniMaxM2Config instance.
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

    @override
    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a MiniMaxM2Config from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration.
            model_config: The MAX Engine model configuration.

        Returns:
            An initialized MiniMaxM2Config instance.
        """
        # Get base config from Llama3Config
        base_config = Llama3Config.initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )

        kv_cache_config = pipeline_config.model.kv_cache
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )
        n_devices = len(pipeline_config.model.device_specs)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]

        # Override KV params and attention multiplier with MiniMax-specific values
        kv_params = MiniMaxM2Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        attention_multiplier = MiniMaxM2Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # MoE parameters
        num_local_experts = getattr(
            huggingface_config, "num_local_experts", 256
        )
        num_experts_per_tok = getattr(
            huggingface_config, "num_experts_per_tok", 8
        )

        # Partial RoPE factor
        head_dim = huggingface_config.head_dim
        rotary_dim = getattr(huggingface_config, "rotary_dim", head_dim)
        partial_rotary_factor = getattr(
            huggingface_config,
            "partial_rotary_factor",
            rotary_dim / head_dim,
        )

        return cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            rope_theta=base_config.rope_theta,
            rope_scaling_params=base_config.rope_scaling_params,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            interleaved_rope_weights=base_config.interleaved_rope_weights,
            vocab_size=base_config.vocab_size,
            dtype=base_config.dtype,
            model_quantization_encoding=base_config.model_quantization_encoding,
            quantization_config=base_config.quantization_config,
            max_seq_len=base_config.max_seq_len,
            kv_params=kv_params,
            attention_multiplier=attention_multiplier,
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            data_parallel_degree=base_config.data_parallel_degree,
            # MoE parameters
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            partial_rotary_factor=partial_rotary_factor,
            quantization_encoding=quantization_encoding,
        )
