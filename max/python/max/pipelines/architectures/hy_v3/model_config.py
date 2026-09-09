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
"""Config for Hy3-preview."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override


def hyv3_num_experts_from_config(huggingface_config: AutoConfig) -> int:
    """Return the expert count from the HuggingFace config.

    Hy3 checkpoints use ``num_experts``; other MoE variants may use
    ``num_local_experts`` instead. Fail fast if neither is set.
    """
    num_experts = getattr(huggingface_config, "num_experts", None)
    if num_experts is not None:
        return int(num_experts)
    num_local_experts = getattr(huggingface_config, "num_local_experts", None)
    if num_local_experts is not None:
        return int(num_local_experts)
    raise ValueError(
        "Hy3 HuggingFace config must define num_experts or num_local_experts"
    )


def _required_config_int(huggingface_config: AutoConfig, name: str) -> int:
    if not hasattr(huggingface_config, name):
        raise ValueError(
            f"Hy3 HuggingFace config is missing required field {name!r}"
        )
    value = getattr(huggingface_config, name)
    if value is None:
        raise ValueError(
            f"Hy3 HuggingFace config field {name!r} must not be None"
        )
    return int(value)


def _required_config_float(huggingface_config: AutoConfig, name: str) -> float:
    if not hasattr(huggingface_config, name):
        raise ValueError(
            f"Hy3 HuggingFace config is missing required field {name!r}"
        )
    value = getattr(huggingface_config, name)
    if value is None:
        raise ValueError(
            f"Hy3 HuggingFace config field {name!r} must not be None"
        )
    return float(value)


def _required_config_bool(huggingface_config: AutoConfig, name: str) -> bool:
    if not hasattr(huggingface_config, name):
        raise ValueError(
            f"Hy3 HuggingFace config is missing required field {name!r}"
        )
    value = getattr(huggingface_config, name)
    if value is None:
        raise ValueError(
            f"Hy3 HuggingFace config field {name!r} must not be None"
        )
    return bool(value)


@dataclass(kw_only=True)
class HYV3Config(Llama3Config):
    """Hy3-preview decoder-only MoE config."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    # MoE
    num_local_experts: int = 192
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    num_shared_experts: int = 1
    router_scaling_factor: float = 2.826
    route_norm: bool = True
    # The first ``first_k_dense_replace`` layers are dense SwiGLU MLPs;
    # later layers are sparse MoE.
    first_k_dense_replace: int = 1
    intermediate_size_dense: int = 13312

    # Detected from state_dict during finalize().
    correction_bias_dtype: DType | None = None
    gate_dtype: DType | None = None
    ep_config: EPConfig | None = None

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams:
        """Construct KV cache params using the explicit head_dim.

        With ``data_parallel_degree=1`` and ``n_devices>1``,
        ``KVCacheParams.__post_init__`` tensor-parallel-shards the paged KV
        cache to ``n_kv_heads_per_device = num_key_value_heads // n_devices``
        (e.g. 8 // 4 = 2). Hy3 attention is TP-sharded to match: each
        device's shard owns ``num_attention_heads // n_devices`` Q heads and
        ``num_key_value_heads // n_devices`` KV heads — exactly the KV heads
        resident in that device's slice of the cache, so
        ``flash_attention_ragged`` reads device-local KV heads consistently
        (see ``layers/attention.py``).

        Constraint: ``num_key_value_heads`` (8) must be divisible by
        ``n_devices`` for the per-device KV-head count to be a whole number,
        so valid attention TP for Hy3 is 1, 2, 4, or 8.
        """
        configured_dp = getattr(
            pipeline_config.model, "data_parallel_degree", 1
        )
        data_parallel_degree = max(1, int(configured_dp))
        return kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=HYV3Config.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
        )

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """``1 / sqrt(head_dim)`` — standard scaled-dot-product."""
        return getattr(
            huggingface_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(huggingface_config.head_dim)),
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # HF reports 80; the trailing MTP layer at index 80 is dropped
        # in the weight adapter (matches HF reference).
        return int(huggingface_config.num_hidden_layers)

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
        return cls.initialize_from_config(
            pipeline_config,
            model_config.huggingface_config,
            max_seq_len=max_seq_len,
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
        # Hy3 stores RoPE under ``rope_parameters`` (a flat dict).
        # Llama3Config reads ``rope_theta`` / ``rope_scaling`` directly,
        # so stub the flat fields, initialise, then restore.
        rope_parameters = (
            getattr(huggingface_config, "rope_parameters", {}) or {}
        )
        _orig_rope_scaling = getattr(huggingface_config, "rope_scaling", None)
        _orig_rope_theta = getattr(huggingface_config, "rope_theta", None)
        huggingface_config.rope_scaling = None
        huggingface_config.rope_theta = float(
            rope_parameters.get("rope_theta", 10000.0)
        )
        try:
            base_config = Llama3Config.initialize_from_config(
                pipeline_config,
                huggingface_config,
                model_config,
                max_seq_len=max_seq_len,
            )
        finally:
            huggingface_config.rope_scaling = _orig_rope_scaling
            if _orig_rope_theta is not None:
                huggingface_config.rope_theta = _orig_rope_theta

        # Subgraphs ON: the 79 sparse MoE layers share one compiled subgraph
        # body and the dense layer(s) another. The per-layer module bodies
        # are identical within each group, so dedup is safe and essential for
        # tractable compile time.
        base_config.use_subgraphs = True

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

        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        attention_multiplier = cls.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        num_local_experts = hyv3_num_experts_from_config(huggingface_config)
        num_experts_per_tok = _required_config_int(
            huggingface_config, "num_experts_per_tok"
        )
        moe_intermediate_size = _required_config_int(
            huggingface_config, "moe_intermediate_size"
        )
        num_shared_experts = _required_config_int(
            huggingface_config, "num_shared_experts"
        )
        router_scaling_factor = _required_config_float(
            huggingface_config, "router_scaling_factor"
        )
        route_norm = _required_config_bool(huggingface_config, "route_norm")
        first_k_dense_replace = _required_config_int(
            huggingface_config, "first_k_dense_replace"
        )
        intermediate_size_dense = _required_config_int(
            huggingface_config, "intermediate_size"
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
            quantization_encoding=base_config.quantization_encoding,
            # MoE
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            num_shared_experts=num_shared_experts,
            router_scaling_factor=router_scaling_factor,
            route_norm=route_norm,
            first_k_dense_replace=first_k_dense_replace,
            intermediate_size_dense=intermediate_size_dense,
        )
