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
"""Config for Qwen3 models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config


@dataclass(kw_only=True)
class Qwen3Config(Llama3Config):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
        "float8_e4m3fn",
    }

    # MoE parameters - these are optional and only used for Qwen3-MOE models
    num_experts: int = 0
    """Number of experts in the MoE layer. 0 means dense model (no MoE)."""

    num_experts_per_tok: int = 1
    """Number of experts per token in the MoE layer."""

    moe_intermediate_size: int = 0
    """Intermediate size in the MoE layer. If 0, uses intermediate_size."""

    mlp_only_layers: list[int] = field(default_factory=list)
    """List of layer indices that use MLP instead of MoE."""

    norm_topk_prob: bool = False
    """Whether to use top-k probability normalization in the MoE layer."""

    decoder_sparse_step: int = 1
    """Sparse step for the decoder. Controls which layers use MoE."""

    ep_config: EPConfig | None = None
    """Expert parallelism configuration. None means no EP."""

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """The attention multiplier for Qwen3 models.

        Uses the explicit head_dim from the config instead of calculating it.

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
        """Initializes a Qwen3Config instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized Qwen3Config instance.
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
        """Initializes a Qwen3Config instance from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration.
            model_config: The MAX Engine model configuration.

        Returns:
            An initialized Qwen3Config instance.
        """
        # Get base config from Llama3Config
        base_config = Llama3Config.initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )

        kv_cache_config = pipeline_config.model.kv_cache
        cache_dtype = cache_dtype_for_encoding(
            base_config.quantization_encoding,
            pipeline_config.model.kv_cache.kv_cache_format,
        )
        n_devices = len(pipeline_config.model.device_specs)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in pipeline_config.model.device_specs[:n_devices]
        ]

        # Override the KV parameters and attention multiplier with Qwen3-specific calculations
        qwen3_kv_params = Qwen3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        qwen3_attention_multiplier = Qwen3Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # Handle both MoE (e.g., Qwen3-30B-A3B) and dense (e.g., Qwen3-8B) variants.
        # For dense models, num_experts=0 ensures the decoder always uses MLP layers
        num_experts = getattr(huggingface_config, "num_experts", 0)
        num_experts_per_tok = getattr(
            huggingface_config, "num_experts_per_tok", 1
        )
        moe_intermediate_size = getattr(
            huggingface_config,
            "moe_intermediate_size",
            base_config.intermediate_size,
        )
        mlp_only_layers = getattr(huggingface_config, "mlp_only_layers", [])
        norm_topk_prob = getattr(huggingface_config, "norm_topk_prob", False)
        decoder_sparse_step = getattr(
            huggingface_config, "decoder_sparse_step", 1
        )

        # Return a new Qwen3Config with the corrected parameters
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
            kv_params=qwen3_kv_params,
            attention_multiplier=qwen3_attention_multiplier,
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            data_parallel_degree=base_config.data_parallel_degree,
            quantization_encoding=base_config.quantization_encoding,
            # MoE parameters
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            mlp_only_layers=mlp_only_layers,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
        )
