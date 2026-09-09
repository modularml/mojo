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
"""Config for Qwen3.5 models (hybrid linear/full attention)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import QuantConfig
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithVisionCache
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.weights import resolve_hf_quant_config
from transformers.models.auto.configuration_auto import AutoConfig
from typing_extensions import Self, override

from ..llama3.model_config import Llama3Config
from ..qwen3vl_moe.model_config import VisionConfig
from ..qwen3vl_moe.nn.data_processing import QWEN3VL_MAX_PIXELS
from .quantization import Qwen3_5QuantScheme, parse_quant_scheme

__all__ = ["Qwen3_5Config", "VisionConfig"]

_DECLARED_DTYPES: dict[str, DType] = {
    "bfloat16": DType.bfloat16,
    "float16": DType.float16,
    "float32": DType.float32,
}


def _declared_dtype(text_config: AutoConfig) -> DType | None:
    """The dtype the checkpoint declares for its unquantized tensors.

    A quantized checkpoint still declares the dtype of everything it left
    alone -- norms, embeddings, the GDN conv, the state pools. Returns None
    when the field is absent or names a dtype outside the set above, leaving
    the caller on its existing fallback.
    """
    for attr in ("dtype", "torch_dtype"):
        declared = getattr(text_config, attr, None)
        if isinstance(declared, str) and declared in _DECLARED_DTYPES:
            return _DECLARED_DTYPES[declared]
    return None


@dataclass(kw_only=True)
class Qwen3_5Config(Llama3Config, ArchConfigWithVisionCache):
    """Configuration for Qwen3.5 hybrid attention models.

    Qwen3.5 uses a hybrid architecture with both full (standard) attention
    and linear attention (Gated DeltaNet) layers. Every full_attention_interval-th
    layer uses full attention, and the rest use linear attention.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    }

    # Hybrid attention parameters
    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention type: 'full_attention' or 'linear_attention'."""

    full_attention_interval: int = 4
    """Every N-th layer uses full attention."""

    # Linear attention (Gated DeltaNet) parameters
    linear_key_head_dim: int = 128
    """Key head dimension for linear attention layers."""

    linear_value_head_dim: int = 128
    """Value head dimension for linear attention layers."""

    linear_num_key_heads: int = 16
    """Number of key heads for linear attention layers."""

    linear_num_value_heads: int = 48
    """Number of value heads for linear attention layers."""

    linear_conv_kernel_dim: int = 4
    """Causal conv1d kernel size for linear attention layers."""

    # Qwen3.5-specific full attention parameters
    partial_rotary_factor: float = 0.25
    """Fraction of head_dim that gets rotary position embedding."""

    attn_output_gate: bool = True
    """Whether full attention layers use a sigmoid output gate."""

    mamba_ssm_dtype: DType = DType.float32
    """Dtype for SSM (state space model) computations in linear attention layers."""

    state_pool_dtype: DType | None = None
    """Storage dtype override for the linear-attention state pools.

    ``None`` (the default) stores both pools at :attr:`compute_dtype`
    (bfloat16), the configuration every exported artifact and the Mach
    registry declare. ``float32`` makes a speculated generation follow the
    exact state trajectory of an unspeculated one — the recurrence rounds to
    the pool dtype only at a call boundary, so a lossy pool makes the
    trajectory depend on speculation's chunking — at roughly double the
    per-request state memory (74.8 to 149.6 MiB for Qwen3.8-27B). Set via the
    ``state_pool_dtype`` KV-cache config knob; read through
    :attr:`state_dtype`."""

    # Vision encoder (optional - text-only models leave these None)
    vision_config: VisionConfig | None = None
    """Vision encoder configuration; None for text-only models."""

    image_token_id: int | None = None
    """Token ID used for image placeholders in the input sequence."""

    video_token_id: int | None = None
    """Token ID used for video placeholders in the input sequence."""

    vision_start_token_id: int | None = None
    """Token ID that marks the start of vision content."""

    mrope_section: list[int] | None = None
    """MRoPE section lengths for multimodal rotary position encoding."""

    hf_quantization_config: dict[str, Any] | None = None
    """The checkpoint's resolved Hugging Face quantization config.

    Captured at ``initialize_from_config`` because it lives on the top-level
    multimodal config, while ``finalize`` is handed ``text_config``."""

    quant_scheme: Qwen3_5QuantScheme | None = None
    """Which modules are quantized and how; set by :meth:`_parse_quant_config`."""

    declared_dtype: DType | None = None
    """The dtype the checkpoint declares for its unquantized tensors.

    Captured at ``initialize_from_config`` so :attr:`compute_dtype` is right
    before ``finalize`` resolves :attr:`quant_scheme`. Memory planning runs in
    that window."""

    @property
    def compute_dtype(self) -> DType:
        """Dtype of activations and every unquantized weight.

        ``dtype`` is the *storage* dtype the resolved encoding implies
        (``uint8`` for packed NVFP4), which is not what the norms, embeddings,
        conv1d or linear-attention state pools use.

        :attr:`quant_scheme` is authoritative but only exists after
        ``finalize``; :attr:`declared_dtype` covers the earlier window so a
        pre-``finalize`` caller does not silently read the storage dtype.
        """
        if self.quant_scheme is not None:
            return self.quant_scheme.compute_dtype
        if self.declared_dtype is not None:
            return self.declared_dtype
        return self.dtype

    @property
    def state_dtype(self) -> DType:
        """Storage dtype of the linear-attention state pools.

        Every declarer of a pool buffer — the base graph, the fused
        speculative graph, and the serving-side state cache — must read this
        one property: the two graphs share one pool allocation at serve time,
        so a disagreement is an unserveable artifact pair.
        """
        if self.state_pool_dtype is not None:
            return self.state_pool_dtype
        return self.compute_dtype

    @override
    def _parse_quant_config(
        self,
        huggingface_config: AutoConfig,
        state_dict: Mapping[str, WeightData],
    ) -> QuantConfig | None:
        """Parses the mixed NVFP4/FP8 scheme from the top-level config.

        ``huggingface_config`` here is ``text_config``, which carries no
        ``quantization_config``; the base implementation would fall back to
        sniffing ``weight_scale_2`` out of the state dict and describe the
        whole model as uniform NVFP4. The captured top-level config is used
        instead, and the returned :class:`QuantConfig` -- the NVFP4 one -- is
        only the MLP half. Consumers must go through :attr:`quant_scheme`.
        """
        self.quant_scheme = parse_quant_scheme(
            self.hf_quantization_config, state_dict, self.num_hidden_layers
        )
        return self.quant_scheme.mlp if self.quant_scheme else None

    @staticmethod
    def _get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
        """Extract text config, handling both multimodal and text-only models."""
        return getattr(huggingface_config, "text_config", huggingface_config)

    @staticmethod
    def _get_vision_config(huggingface_config: AutoConfig) -> AutoConfig | None:
        """The checkpoint's vision tower config, or None for text-only ones.

        Same shape check ``initialize_from_config`` uses to decide whether to
        build a :class:`VisionConfig`, so the vision-cache facts below and the
        compiled graph agree on whether this checkpoint has a tower.
        """
        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None or not hasattr(vision_config, "patch_size"):
            return None
        return vision_config

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        """Estimates per-entry bytes for the Qwen3.5 vision encoder cache.

        Qwen3.5's tower is NaViT-style: an image keeps its aspect ratio and
        yields one token per merged patch, so — unlike Gemma4's fixed pooled
        patch count — no per-image token count exists in the config to read.
        The bound that does exist is the image processor's post-resize pixel
        ceiling, which every served image is smart-resized under, so the
        largest entry the cache can be asked to hold is that ceiling divided
        by the patch area and the spatial merge.

        Returns:
            Estimated bytes per vision cache entry, or 0 for a checkpoint
            with no vision tower.
        """
        vision_config = cls._get_vision_config(huggingface_config)
        if vision_config is None:
            return 0
        patch_area = vision_config.patch_size**2
        merge = getattr(vision_config, "spatial_merge_size", 2)
        max_tokens = QWEN3VL_MAX_PIXELS // (patch_area * merge**2)
        spec = cls.get_vision_cache_row_spec(huggingface_config)
        assert spec is not None
        hidden, dtype = spec
        return max_tokens * hidden * dtype.size_in_bytes

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        """One embedding row per merged vision token, at the LM hidden size.

        The dtype is the checkpoint's declared one rather than a fixed
        bfloat16, because the block pool has to match the buffers the encoder
        hands it: those are cast to :attr:`compute_dtype`, which for every
        encoding this architecture supports resolves to the declared dtype.
        """
        vision_config = cls._get_vision_config(huggingface_config)
        if vision_config is None:
            return None
        declared = _declared_dtype(cls._get_text_config(huggingface_config))
        return (vision_config.out_hidden_size, declared or DType.bfloat16)

    @override
    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        """Bounds against the text config's ``max_position_embeddings``."""
        return super().calculate_max_seq_len(
            cls._get_text_config(huggingface_config),
            model_config,
        )

    @staticmethod
    def _get_layer_types(text_config: AutoConfig) -> list[str]:
        """Return the per-layer attention type list for the model.

        Uses `layer_types` from the config when present; otherwise generates
        it from `full_attention_interval`.
        """
        layer_types = getattr(text_config, "layer_types", [])
        if layer_types:
            return list(layer_types)
        full_interval = getattr(text_config, "full_attention_interval", 4)
        num_layers = text_config.num_hidden_layers
        return [
            "full_attention"
            if (i + 1) % full_interval == 0
            else "linear_attention"
            for i in range(num_layers)
        ]

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
        """Construct KV cache parameters for full attention layers only.

        Only allocates KV cache entries for full-attention layers; linear
        attention layers use separate conv/recurrent state buffers instead.
        The forward pass maps each full-attention layer to a sequential KV
        cache index (0, 1, 2, ...) independent of the absolute layer index.
        """
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        data_parallel_degree = pipeline_config.model.data_parallel_degree
        if data_parallel_degree > 1:
            raise ValueError(
                "Data parallelism is not supported for Qwen3.5 models"
            )
        layer_types = Qwen3_5Config._get_layer_types(text_config)
        num_full_attention_layers = sum(
            1 for lt in layer_types if lt == "full_attention"
        )
        # The MHA kernel selects tile_size == head_dim. The KV cache
        # page_size must be >= tile_size. Qwen3.5 has head_dim=256.
        page_size = kv_cache_config.kv_cache_page_size
        if text_config.head_dim > 128:
            page_size = max(page_size, text_config.head_dim)
        return kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            num_layers=num_full_attention_layers,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            page_size=page_size,
        )

    @staticmethod
    def calculate_attention_multiplier(
        huggingface_config: AutoConfig,
    ) -> float:
        """Compute attention scaling factor using explicit head_dim."""
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        return getattr(
            text_config,
            "attention_multiplier",
            math.sqrt(1.0 / float(text_config.head_dim)),
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        return text_config.num_hidden_layers

    def _per_request_state_bytes(self) -> int:
        """Return GPU bytes for one request's linear-attention state (all linear layers).

        Each linear-attention layer stores two state arrays per active request:
        - Conv state:       `(1, conv_dim, kernel-1)` :attr:`state_dtype`
        - Recurrent state:  `(1, nv, kd, vd)`         :attr:`state_dtype`

        Computation is promoted to float32 inside GatedDeltaNet.__call__().
        These buffers are NOT included in the KV-cache budget.
        """
        num_linear = sum(
            1 for lt in self.layer_types if lt == "linear_attention"
        )
        if num_linear == 0:
            return 0
        conv_dim = (
            2 * self.linear_key_head_dim * self.linear_num_key_heads
            + self.linear_value_head_dim * self.linear_num_value_heads
        )
        # `state_dtype` is the one property every pool declarer reads, so the
        # cost this feeds to `infer_optimal_batch_size` tracks the override
        # too -- a float32 pool is 2x a bf16 one, and 4x what the encoding's
        # `uint8` storage dtype would have implied.
        dtype_bytes = self.state_dtype.size_in_bytes
        bytes_per_layer = (
            # conv state: (conv_dim * (kernel-1)) elements
            conv_dim * (self.linear_conv_kernel_dim - 1) * dtype_bytes
            # recurrent state: (nv * kd * vd) elements
            + self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * dtype_bytes
        )
        return num_linear * bytes_per_layer

    def infer_optimal_batch_size(
        self,
        devices: list[Device],
        *,
        weights_size: int,
        device_memory_utilization: float,
    ) -> int:
        """Return a memory-safe default `max_batch_size` for this architecture.

        Qwen3.5 stores GatedDeltaNet conv and recurrent state in a single
        ``max_batch x per_req`` pool that the slot-indexed SSM kernels
        mutate in place. There are no working copies, so peak footprint is
        ``max_batch x per_req`` bytes.

        We split the post-weights utilization budget evenly: the state pool
        gets up to half, the KV cache absorbs the rest. This uses the same
        ``device_memory_utilization`` headroom factor as the rest of the
        pipeline, and matches the ``estimate_activation_memory()`` reservation.

        Falls back to 32—safe for the 27B model on H100/A100 (80 GB)—when
        the device query fails.
        """
        per_req = self._per_request_state_bytes()
        try:
            free_bytes = int(
                sum(d.stats.get("free_memory", 0) for d in devices)
            )
        except Exception:
            free_bytes = 0
        if free_bytes <= 0:
            # Conservative fallback: safe for Qwen3.5-27B on H100/A100 (80 GB).
            return 32
        budget = int(free_bytes * device_memory_utilization) - weights_size
        if budget <= 0:
            return 1
        # Single in-place pool: divide half the budget by per_req.
        max_batch = max(1, (budget // 2) // per_req)
        return min(512, max_batch)

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
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', "
                "but config could not be loaded."
            )
        return cls.initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
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
        """Initialize config from pipeline and HuggingFace configurations.

        Handles both multimodal (Qwen3_5ForConditionalGeneration) and
        text-only (Qwen3_5ForCausalLM) configs by extracting the text config.
        """
        model_config = model_config or pipeline_config.model
        text_config = Qwen3_5Config._get_text_config(huggingface_config)

        # Get base Llama3Config from the text config
        base_config = Llama3Config.initialize_from_config(
            pipeline_config, text_config, max_seq_len=max_seq_len
        )

        kv_cache_config = model_config.kv_cache
        cache_dtype = cache_dtype_for_encoding(
            base_config.quantization_encoding,
            model_config.kv_cache.kv_cache_format,
        )
        n_devices = len(model_config.device_specs)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs[:n_devices]
        ]

        # Override KV params and attention multiplier
        kv_params = Qwen3_5Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )
        attention_multiplier = Qwen3_5Config.calculate_attention_multiplier(
            huggingface_config=huggingface_config,
        )

        # Extract rope_theta and partial_rotary_factor.
        # Priority: top-level text_config.rope_theta > rope_parameters.rope_theta
        # > base config default.  Qwen3.5 stores these inside rope_parameters;
        # some fine-tuned variants may promote rope_theta to the top level.
        rope_theta = base_config.rope_theta
        partial_rotary_factor = 0.25
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is not None:
            if isinstance(rope_params, dict):
                rope_theta = rope_params.get("rope_theta", rope_theta)
                partial_rotary_factor = rope_params.get(
                    "partial_rotary_factor", partial_rotary_factor
                )
            else:
                rope_theta = getattr(rope_params, "rope_theta", rope_theta)
                partial_rotary_factor = getattr(
                    rope_params, "partial_rotary_factor", partial_rotary_factor
                )

        # Top-level text_config.rope_theta takes explicit priority when present.
        if hasattr(text_config, "rope_theta"):
            rope_theta = text_config.rope_theta

        # Hybrid attention parameters
        layer_types = Qwen3_5Config._get_layer_types(text_config)

        # Linear attention parameters
        linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        linear_value_head_dim = getattr(
            text_config, "linear_value_head_dim", 128
        )
        linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        linear_num_value_heads = getattr(
            text_config, "linear_num_value_heads", 48
        )
        linear_conv_kernel_dim = getattr(
            text_config, "linear_conv_kernel_dim", 4
        )
        attn_output_gate = getattr(text_config, "attn_output_gate", True)

        _mamba_dtype_map: dict[str, DType] = {
            "float32": DType.float32,
            "bfloat16": DType.bfloat16,
            "float16": DType.float16,
        }
        mamba_ssm_dtype_str = getattr(text_config, "mamba_ssm_dtype", "float32")
        mamba_ssm_dtype = _mamba_dtype_map.get(
            mamba_ssm_dtype_str, DType.float32
        )

        # State-pool storage knob (None = compute dtype). Unlike
        # mamba_ssm_dtype this is a user setting, so an unknown value is
        # rejected rather than defaulted.
        state_pool_dtype: DType | None = None
        state_pool_dtype_str = model_config.kv_cache.state_pool_dtype
        if state_pool_dtype_str is not None:
            _pool_dtype_map = {
                "bfloat16": DType.bfloat16,
                "float32": DType.float32,
            }
            if state_pool_dtype_str not in _pool_dtype_map:
                raise ValueError(
                    "state_pool_dtype must be 'bfloat16' or 'float32', got"
                    f" {state_pool_dtype_str!r}"
                )
            state_pool_dtype = _pool_dtype_map[state_pool_dtype_str]

        # Handle tie_word_embeddings from top-level config
        tie_word_embeddings = getattr(
            huggingface_config, "tie_word_embeddings", False
        )

        # Vision encoder (only present in multimodal checkpoints)
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        vision_cfg: VisionConfig | None = None
        if hf_vision_config is not None and hasattr(
            hf_vision_config, "patch_size"
        ):
            vision_cfg = VisionConfig.initialize_from_config(
                pipeline_config, hf_vision_config
            )

        # Multimodal token IDs and MRoPE section
        image_token_id = getattr(huggingface_config, "image_token_id", None)
        video_token_id = getattr(huggingface_config, "video_token_id", None)
        vision_start_token_id = getattr(
            huggingface_config, "vision_start_token_id", None
        )
        mrope_section: list[int] | None = None
        rope_params = getattr(text_config, "rope_parameters", None)
        if rope_params is not None:
            raw_section = (
                rope_params.get("mrope_section")
                if isinstance(rope_params, dict)
                else getattr(rope_params, "mrope_section", None)
            )
            if raw_section is not None:
                mrope_section = list(raw_section)

        config_instance = cls(
            hidden_size=base_config.hidden_size,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            num_hidden_layers=base_config.num_hidden_layers,
            rope_theta=rope_theta,
            rope_scaling_params=base_config.rope_scaling_params,
            rms_norm_eps=base_config.rms_norm_eps,
            intermediate_size=base_config.intermediate_size,
            # Partial RoPE requires interleaved pattern in the kernel
            interleaved_rope_weights=True,
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
            tie_word_embeddings=tie_word_embeddings,
            quantization_encoding=base_config.quantization_encoding,
            # Hybrid attention parameters
            layer_types=layer_types,
            full_attention_interval=getattr(
                text_config, "full_attention_interval", 4
            ),
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            partial_rotary_factor=partial_rotary_factor,
            attn_output_gate=attn_output_gate,
            mamba_ssm_dtype=mamba_ssm_dtype,
            state_pool_dtype=state_pool_dtype,
            # Vision (optional)
            vision_config=vision_cfg,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            mrope_section=mrope_section,
            hf_quantization_config=resolve_hf_quant_config(
                huggingface_config, {}
            ),
            declared_dtype=_declared_dtype(text_config),
        )

        return config_instance
