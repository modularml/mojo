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
"""Config for Laguna (``poolside/Laguna-M.1-NVFP4``).

This module exposes every Laguna-specific HF config field that the
graph in ``laguna.py`` consumes: per-layer attention shape and type,
per-layer MLP type, the RoPE tables, sigmoid MoE routing parameters,
shared-expert sizing, and the softplus attention-output gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.comm.ep import EPConfig
from max.nn.kv_cache import KVCacheParams, KVCacheQuantizationConfig
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class LagunaConfig(Llama3Config):
    """Configuration for Laguna decoder-only MoE models.

    Extends ``Llama3Config`` with Laguna-specific fields:

    - **Uniform GQA**: 64 query / 8 KV heads, full causal attention, a single
      RoPE table (``rope_theta`` + ``partial_rotary_factor``; M.1 is
      full-rotary, 1.0).
    - **Per-layer MLP type**: ``mlp_layer_types`` is ``dense`` (the dense
      prefix) or ``sparse`` (the rest). The decoder block dispatches between
      the dense MLP and the sparse MoE block.
    - **Sigmoid + correction-bias routing**: not softmax. See
      ``LagunaTopKRouter`` for the per-token routing math.
    - **Routed scaling factor + shared experts + softplus attention
      output gate**: applied per token / per element as Laguna requires.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    # --- MoE (routed expert collection) ---
    num_local_experts: int = 256
    """Number of routed experts per sparse layer."""

    num_experts_per_tok: int = 8
    """Top-k experts selected per token."""

    moe_intermediate_size: int = 512
    """Per-expert intermediate dim (each routed expert is a SwiGLU MLP
    with this intermediate size)."""

    shared_expert_intermediate_size: int = 512
    """Intermediate dim of the always-on shared expert MLP added
    alongside the routed-expert output (sum, not gated)."""

    moe_routed_scaling_factor: float = 2.5
    """Scalar applied to routed-expert output before adding the shared
    expert output. Laguna-specific (default 1.0 in donors that have it
    at all)."""

    moe_router_logit_softcapping: float = 0.0
    """If > 0, router logits are passed through
    ``softcap * tanh(logits/softcap)`` before sigmoid. Disabled (0.0)
    for ``poolside/Laguna-M.1-NVFP4``."""

    norm_topk_prob: bool = True
    """Whether to L1-normalise the selected top-k routing weights so
    they sum to 1. True for Laguna (matches HF reference)."""

    # --- Sigmoid routing parameters ---
    correction_bias_dtype: DType | None = None
    """Data type of the e_score_correction_bias weight. Detected from
    state dict during finalize()."""

    gate_dtype: DType | None = None
    """Data type for the routed-expert gate Linear. Detected from state
    dict during finalize()."""

    attn_dtype: DType | None = None
    """Data type for attention weights. Detected from state dict during
    finalize()."""

    ep_config: EPConfig | None = None
    """Expert parallelism configuration. None means no EP (single-GPU)."""

    # --- Per-layer MLP type ---
    mlp_layer_types: list[str] | None = None
    """Per-layer MLP type. Each entry is ``"dense"`` or ``"sparse"``.
    Length equals ``num_hidden_layers``. Typically layer 0 is dense
    and the rest are sparse MoE."""

    intermediate_size_dense: int = 8192
    """Intermediate dim of the dense-layer MLPs (the one or more layers
    where ``mlp_layer_types[i] == "dense"``). Distinct from
    ``intermediate_size`` (which Llama3Config uses for the active
    branch) and from ``moe_intermediate_size``."""

    # --- RoPE ---
    partial_rotary_factor: float = 0.5
    """Fraction of head_dim used for rotary embeddings, read from the HF
    config. ``poolside/Laguna-M.1-NVFP4`` is full-rotary (1.0)."""

    # --- Attention output gate ---
    gating: bool = True
    """When True, attention applies ``softplus(g_proj(hidden)) *
    attn_out`` per-head before ``o_proj``. ``g_proj.out_features =
    num_heads`` (one scalar per head); broadcast over ``head_dim``.
    Always True for ``poolside/Laguna-M.1-NVFP4``."""

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
        """Constructs KV cache parameters using explicit head_dim from config.

        Args:
            huggingface_config: The HuggingFace configuration object.
            pipeline_config: The MAX Engine pipeline configuration.
            devices: Devices to use for the KV cache.
            kv_cache_config: Configuration for KV cache.
            cache_dtype: Data type for the cache.

        Returns:
            KVCacheParams object with the correct head_dim from config.
        """
        # The MiniMax-M2 donor defaulted to
        # ``data_parallel_degree = len(devices)``; this couples DP
        # degree to device count and requires N batched-per-request
        # tokens at inference. Laguna instead respects the
        # pipeline_config's explicit DP setting (default 1) so a single
        # prompt through ``/v1/completions`` works on any device count.
        # The pipeline owner can opt into DP=N via
        # ``pipeline_config.model.data_parallel_degree=N``.
        configured_dp = getattr(
            pipeline_config.model, "data_parallel_degree", 1
        )
        data_parallel_degree = max(1, int(configured_dp))
        # FP8 KV cache: EXPERIMENTAL and not yet accuracy-validated. Opt in via
        # ``--kv-cache-format float8_e4m3fn``; the default bf16 KV cache is the
        # validated configuration. When the resolved cache dtype is FP8, MAX
        # needs a quantization config to lay out the cache; mirror deepseekV3.
        # MAX scales KV dynamically at runtime — the checkpoint's static
        # ``self_attn.{k,v}_scale`` are dropped in the weight adapter (MAX never
        # consumes them). ``scale_dtype=int8`` selects the *unscaled*
        # ``is_fp8_kv`` cast path (full per-block scaling / SnapMLA is still WIP
        # upstream), so this can degrade quality on longer contexts and has NOT
        # been validated against the bf16 path. ``LagunaAttention`` uses the
        # fused ``rope_split_store_ragged`` store, which converts bf16 -> FP8 at
        # store time so the path at least runs.
        kvcache_quant_config = None
        if cache_dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
            kvcache_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.int8, quantization_granularity=32
            )
        return kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=huggingface_config.head_dim,
            num_layers=LagunaConfig.get_num_layers(huggingface_config),
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            kvcache_quant_config=kvcache_quant_config,
        )

    @staticmethod
    def calculate_attention_multiplier(huggingface_config: AutoConfig) -> float:
        """Computes the attention multiplier from the config's head_dim.

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
        """Initializes a LagunaConfig from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized LagunaConfig instance.
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
        """Initializes a LagunaConfig from pipeline and HuggingFace configs.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: The HuggingFace model configuration.
            model_config: The MAX Engine model configuration.

        Returns:
            An initialized LagunaConfig instance.
        """
        # Llama3Config expects flat HF fields (``rope_scaling`` with a
        # ``type`` / ``rope_type`` key, and ``rope_theta``). Laguna nests
        # those under ``rope_parameters[layer_type]``. Stub flat values
        # from the full-attention table so the parent can initialise; we
        # build the actual RoPE wiring later from the original dict.
        rope_parameters = (
            getattr(huggingface_config, "rope_parameters", {}) or {}
        )
        full_rope = rope_parameters.get("full_attention", {})

        _orig_rope_scaling = getattr(huggingface_config, "rope_scaling", None)
        _orig_rope_theta = getattr(huggingface_config, "rope_theta", None)
        try:
            huggingface_config.rope_scaling = None
            huggingface_config.rope_theta = float(
                full_rope.get("rope_theta", 10000.0)
            )
        except (AttributeError, TypeError):
            pass

        try:
            base_config = Llama3Config.initialize_from_config(
                pipeline_config,
                huggingface_config,
                model_config,
                max_seq_len=max_seq_len,
            )
        finally:
            # Restore the originals so downstream readers see the
            # unmodified HF config.
            try:
                huggingface_config.rope_scaling = _orig_rope_scaling
                if _orig_rope_theta is not None:
                    huggingface_config.rope_theta = _orig_rope_theta
            except (AttributeError, TypeError):
                pass

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

        kv_params = LagunaConfig.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        attention_multiplier = LagunaConfig.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # --- MoE ---
        # Laguna uses ``num_experts`` (HF field). MiniMax called it
        # ``num_local_experts``. Accept both.
        num_local_experts = getattr(
            huggingface_config,
            "num_experts",
            getattr(huggingface_config, "num_local_experts", 256),
        )
        num_experts_per_tok = getattr(
            huggingface_config, "num_experts_per_tok", 8
        )
        moe_intermediate_size = getattr(
            huggingface_config, "moe_intermediate_size", 512
        )
        shared_expert_intermediate_size = getattr(
            huggingface_config, "shared_expert_intermediate_size", 512
        )
        moe_routed_scaling_factor = float(
            getattr(huggingface_config, "moe_routed_scaling_factor", 1.0)
        )
        moe_router_logit_softcapping = float(
            getattr(huggingface_config, "moe_router_logit_softcapping", 0.0)
        )

        # --- Per-layer MLP shape ---
        num_hidden_layers = base_config.num_hidden_layers
        mlp_layer_types = getattr(huggingface_config, "mlp_layer_types", None)
        if mlp_layer_types is None:
            # Laguna default: first layer dense, rest sparse.
            mlp_layer_types = ["dense"] + ["sparse"] * (num_hidden_layers - 1)
        intermediate_size_dense = getattr(
            huggingface_config, "intermediate_size", 8192
        )

        # --- RoPE wiring ---
        # rope_parameters on the HF config is keyed by layer type; Laguna uses
        # the full_attention table. M.1 is full-rotary and runs at contexts
        # below original_max_position_embeddings, where YaRN is the identity,
        # so only rope_theta + partial_rotary_factor feed the RoPE table.
        rope_parameters = (
            getattr(huggingface_config, "rope_parameters", {}) or {}
        )
        full_rope = rope_parameters.get("full_attention", {})
        partial_rotary_factor = float(
            full_rope.get("partial_rotary_factor", 0.5)
        )

        gating = bool(getattr(huggingface_config, "gating", True))

        # NVFP4 convention (matches deepseekV3/gemma4): config.dtype IS the
        # packed-fp4 quant dtype (uint8), so the MoE/dense quant Linears and the
        # grouped-quantize strategy use the right out_type. Non-quant layers
        # (embed/lm_head) are handled via quant_config.embedding_output_dtype;
        # attention/norm dtypes are auto-detected from the checkpoint.
        resolved_dtype = base_config.dtype

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
            dtype=resolved_dtype,
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
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            moe_routed_scaling_factor=moe_routed_scaling_factor,
            moe_router_logit_softcapping=moe_router_logit_softcapping,
            mlp_layer_types=list(mlp_layer_types),
            intermediate_size_dense=intermediate_size_dense,
            partial_rotary_factor=partial_rotary_factor,
            gating=gating,
        )
