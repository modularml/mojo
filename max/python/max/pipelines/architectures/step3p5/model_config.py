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
"""Config for Step-3.5 models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config


@dataclass(kw_only=True)
class Step3p5Config(Llama3Config):
    """Model configuration for Step-3.5-Flash."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    # Attention parameters
    num_attention_groups: int = 8
    """Number of KV head groups (same as num_key_value_heads for full attn)."""

    head_dim: int = 128
    """Dimension of each attention head."""

    # Sliding window attention
    sliding_window: int = 512
    """Sliding window size for local attention layers."""

    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention type: 'full_attention' or 'sliding_attention'."""

    # Sliding attention uses different num_heads/kv_heads
    sliding_num_attention_heads: int = 96
    """Number of attention heads for sliding attention layers."""

    sliding_num_attention_groups: int = 8
    """Number of KV head groups for sliding attention layers."""

    # Per-layer RoPE
    per_layer_rope_theta: list[float] = field(default_factory=list)
    """Per-layer RoPE theta values. If empty, uses a single rope_theta."""

    partial_rotary_factors: list[float] = field(default_factory=list)
    """Per-layer partial rotary factors (0.5 for full attn, 1.0 for sliding)."""

    # Yarn-only types: rope_scaling only applies to these layer types
    yarn_only_types: list[str] = field(default_factory=list)
    """Layer types that use rope_scaling (e.g. ['full_attention'])."""

    # Head-wise attention gate
    use_head_wise_attn_gate: bool = True
    """Whether to use per-head sigmoid attention gating (g_proj)."""

    # MoE parameters
    moe_num_experts: int = 288
    """Number of routed experts in MoE layers."""

    moe_top_k: int = 8
    """Number of experts activated per token."""

    moe_intermediate_size: int = 1280
    """Intermediate dimension of each MoE expert MLP."""

    share_expert_dim: int = 1280
    """Intermediate dimension of the shared expert MLP."""

    moe_layers: set[int] = field(default_factory=set)
    """Set of layer indices that use MoE (vs dense MLP)."""

    moe_router_scaling_factor: float = 3.0
    """Scaling factor applied to routed expert weights."""

    norm_expert_weight: bool = True
    """Whether to normalize top-k expert weights to sum to 1."""

    swiglu_limits: list[float] = field(default_factory=list)
    """Per-layer SwiGLU activation clipping thresholds for routed experts.
    0.0 means no clipping. Non-zero values clamp intermediate activations."""

    swiglu_limits_shared: list[float] = field(default_factory=list)
    """Per-layer SwiGLU activation clipping thresholds for shared experts."""

    @staticmethod
    def construct_kv_params(  # type: ignore[override]
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> MultiKVCacheParams:
        """Constructor for hybrid sliding + full KV tree."""
        layer_types = huggingface_config.layer_types
        assert len(layer_types) == huggingface_config.num_hidden_layers
        return hybrid_swa_full_kv_params(
            layer_types=layer_types,
            sliding_window=huggingface_config.sliding_window,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_attention_groups,
            head_dim=huggingface_config.head_dim,
            allow_kv_head_replication=allow_kv_head_replication,
            sliding_n_kv_heads=huggingface_config.attention_other_setting[
                "num_attention_groups"
            ],
            full_n_kv_heads=huggingface_config.num_attention_groups,
        )

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
    ) -> float:
        """Compute the attention scale for Step-3.5.

        Args:
            huggingface_config: The HuggingFace configuration object.

        Returns:
            The attention multiplier value.
        """
        head_dim = getattr(huggingface_config, "head_dim", 128)
        return math.sqrt(1.0 / float(head_dim))

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a Step3p5Config instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            model_config: Optional MAX model configuration override.

        Returns:
            An initialized Step3p5Config instance.
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
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )

    @staticmethod
    def _ensure_hf_config_aliases(
        huggingface_config: AutoConfig,
    ) -> list[float]:
        """Ensure standard field aliases that Llama3Config.initialize_from_config expects.

        When trust_remote_code=True, the HF repo's own config class is loaded
        instead of our Step3p5PretrainedConfig, and may lack standard aliases.
        This method covers the aliases that Llama3Config.initialize_from_config
        requires so the downstream code works regardless of which config class
        was used.  See also Step3p5PretrainedConfig.__init__ which sets a
        superset of these aliases at construction time.

        Returns:
            The original per-layer rope_theta list (empty if rope_theta is scalar).
        """
        if not hasattr(huggingface_config, "num_key_value_heads"):
            huggingface_config.num_key_value_heads = getattr(
                huggingface_config, "num_attention_groups", 8
            )
        if not hasattr(huggingface_config, "rms_norm_eps"):
            huggingface_config.rms_norm_eps = 1e-5
        if not hasattr(huggingface_config, "rope_scaling"):
            huggingface_config.rope_scaling = None
        if not hasattr(huggingface_config, "hidden_act"):
            huggingface_config.hidden_act = "silu"

        # If Step3p5PretrainedConfig already ran, per_layer_rope_theta is set
        # and rope_theta is already a scalar — nothing more to do.
        per_layer_rope_theta: list[float] = getattr(
            huggingface_config, "per_layer_rope_theta", []
        )
        if per_layer_rope_theta:
            return [float(t) for t in per_layer_rope_theta]

        # trust_remote_code path: rope_theta may still be a raw list.
        rope_theta_raw = getattr(huggingface_config, "rope_theta", 10000.0)
        if isinstance(rope_theta_raw, list):
            per_layer_rope_theta = [float(t) for t in rope_theta_raw]
            scalar_theta = rope_theta_raw[0] if rope_theta_raw else 10000.0
            huggingface_config.rope_theta = scalar_theta
            # Save the list so a second init recovers it after the collapse.
            huggingface_config.per_layer_rope_theta = per_layer_rope_theta
            # Transformers v5 copies rope_theta into rope_parameters; patch
            # there too so get_rope_theta() returns the scalar.
            rope_params = getattr(huggingface_config, "rope_parameters", None)
            if isinstance(rope_params, dict) and "rope_theta" in rope_params:
                rope_params["rope_theta"] = scalar_theta

        return per_layer_rope_theta

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
        """Initializes a Step3p5Config instance from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration.
            model_config: Optional MAX model configuration override.

        Returns:
            An initialized Step3p5Config instance.
        """
        per_layer_rope_theta_raw = cls._ensure_hf_config_aliases(
            huggingface_config
        )

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

        kv_params = Step3p5Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        attention_multiplier = Step3p5Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # Layer types
        layer_types = getattr(huggingface_config, "layer_types", [])
        num_hidden_layers = huggingface_config.num_hidden_layers
        # Truncate to num_hidden_layers (config may have extra for MTP layers)
        layer_types = list(layer_types[:num_hidden_layers])

        # Truncate per-layer rope theta to actual model layers.
        per_layer_rope_theta = per_layer_rope_theta_raw[:num_hidden_layers]

        # Partial rotary factors
        prf = getattr(huggingface_config, "partial_rotary_factors", [])
        partial_rotary_factors = [float(f) for f in prf[:num_hidden_layers]]

        # Yarn-only types
        yarn_only_types = getattr(huggingface_config, "yarn_only_types", [])

        # Sliding attention head config
        other = getattr(huggingface_config, "attention_other_setting", None)
        num_attention_groups = getattr(
            huggingface_config, "num_attention_groups", 8
        )
        head_dim = getattr(huggingface_config, "head_dim", 128)

        if other:
            sliding_num_attention_heads = other.get("num_attention_heads", 96)
            sliding_num_attention_groups = other.get(
                "num_attention_groups", num_attention_groups
            )
        else:
            sliding_num_attention_heads = huggingface_config.num_attention_heads
            sliding_num_attention_groups = num_attention_groups

        # MoE layer indices
        moe_layers_enum = getattr(huggingface_config, "moe_layers_enum", "")
        if moe_layers_enum:
            moe_layers = {int(i) for i in moe_layers_enum.strip().split(",")}
        else:
            # Default: all layers except first are MoE
            moe_layers = set(range(1, num_hidden_layers))

        moe_num_experts = getattr(huggingface_config, "moe_num_experts", 288)
        moe_top_k = getattr(huggingface_config, "moe_top_k", 8)
        moe_intermediate_size = getattr(
            huggingface_config, "moe_intermediate_size", 1280
        )
        share_expert_dim = getattr(huggingface_config, "share_expert_dim", 1280)
        moe_router_scaling_factor = getattr(
            huggingface_config, "moe_router_scaling_factor", 3.0
        )
        norm_expert_weight = getattr(
            huggingface_config, "norm_expert_weight", True
        )

        # Per-layer SwiGLU activation clipping (prevents NaN at long contexts)
        raw_swiglu = getattr(huggingface_config, "swiglu_limits", [])
        swiglu_limits = [float(v) for v in raw_swiglu[:num_hidden_layers]]
        raw_swiglu_shared = getattr(
            huggingface_config, "swiglu_limits_shared", []
        )
        swiglu_limits_shared = [
            float(v) for v in raw_swiglu_shared[:num_hidden_layers]
        ]

        sliding_window = getattr(huggingface_config, "sliding_window", 512)
        use_head_wise_attn_gate = getattr(
            huggingface_config, "use_head_wise_attn_gate", True
        )

        return cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=num_attention_groups,
            num_hidden_layers=num_hidden_layers,
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
            kv_params=kv_params,  # type: ignore[arg-type]
            attention_multiplier=attention_multiplier,
            embedding_multiplier=base_config.embedding_multiplier,
            residual_multiplier=base_config.residual_multiplier,
            devices=base_config.devices,
            clip_qkv=base_config.clip_qkv,
            use_subgraphs=base_config.use_subgraphs,
            data_parallel_degree=base_config.data_parallel_degree,
            quantization_encoding=base_config.quantization_encoding,
            # Step3p5-specific
            num_attention_groups=num_attention_groups,
            head_dim=head_dim,
            sliding_window=sliding_window,
            layer_types=layer_types,
            sliding_num_attention_heads=sliding_num_attention_heads,
            sliding_num_attention_groups=sliding_num_attention_groups,
            per_layer_rope_theta=per_layer_rope_theta,
            partial_rotary_factors=partial_rotary_factors,
            yarn_only_types=yarn_only_types,
            use_head_wise_attn_gate=use_head_wise_attn_gate,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_intermediate_size=moe_intermediate_size,
            share_expert_dim=share_expert_dim,
            moe_layers=moe_layers,
            moe_router_scaling_factor=moe_router_scaling_factor,
            norm_expert_weight=norm_expert_weight,
            swiglu_limits=swiglu_limits,
            swiglu_limits_shared=swiglu_limits_shared,
        )
