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
"""Config for GLM-5.3-Flash (``glm5_next``).

GLM-5.3-Flash combines Linear Attention and Sparse MLA attention, so
the config is derived from DeepSeek-V3.2 for the Sparse MLA fields and adds
KDA, mHC, k-pool and vision fields on top.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    KVCacheParams,
    KVCacheQuantizationConfig,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.weights import resolve_hf_quant_config
from transformers import AutoConfig
from typing_extensions import Self, override

from .quantization import Glm5NextQuantScheme, parse_quant_scheme

__all__ = ["Glm5NextConfig", "Glm5NextVisionConfig"]

#: ``layer_types`` value for a KDA linear-attention layer.
LINEAR_ATTENTION = "linear_attention"
#: ``layer_types`` value for a sparse-MLA + DSA-indexer layer.
SPARSE_ATTENTION = "deepseek_sparse_attention"

_DECLARED_DTYPES: dict[str, DType] = {
    "bfloat16": DType.bfloat16,
    "float16": DType.float16,
    "float32": DType.float32,
}


# TODO(KERN-3520): Pad to this width until we have kernel support for 512.
MLA_KERNEL_LATENT_WIDTH = 576


def _resolve_mla_latent_width(text_config: AutoConfig) -> int:
    native = text_config.kv_lora_rank + text_config.qk_rope_head_dim
    return max(native, MLA_KERNEL_LATENT_WIDTH)


def _declared_dtype(text_config: AutoConfig) -> DType | None:
    """The dtype the checkpoint declares for its unquantized tensors."""
    for attr in ("dtype", "torch_dtype"):
        declared = getattr(text_config, attr, None)
        if isinstance(declared, str) and declared in _DECLARED_DTYPES:
            return _DECLARED_DTYPES[declared]
    return None


@dataclass(kw_only=True)
class Glm5NextVisionConfig:
    """Vision tower configuration for GLM-5.3-Flash."""

    depth: int = 24
    """Number of transformer blocks."""

    hidden_size: int = 1024
    """Block width."""

    num_heads: int = 16
    """Attention heads per block; ``head_dim`` is ``hidden_size // num_heads``."""

    intermediate_size: int = 4096
    """Width of each **block's** MLP. Not the merger's."""

    projection_intermediate_size: int = 10240
    """Width of the **patch merger's** clamped SwiGLU."""

    out_hidden_size: int = 4096
    """Tower output width. Equal to the decoder's ``hidden_size``, so the
    merger feeds the decoder directly with no bridge projector."""

    patch_size: int = 14
    """Spatial patch edge."""

    temporal_patch_size: int = 2
    """Temporal patch depth of the 3D patch embedding."""

    spatial_merge_size: int = 2
    """Edge of the strided-conv spatial merge."""

    in_channels: int = 3
    image_size: int = 448
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    attention_bias: bool = True
    """Every projection in the tower carries a bias, unlike MAX's GLM-4.6V
    tower, which sets ``has_bias=False`` on both ``qkv`` and ``proj``."""

    swiglu_limit: float = 10.0
    """Clamp applied to the block MLP and merger gate/up projections."""

    dtype: DType = DType.bfloat16
    """The vision tower is BF16 throughout; it carries no FP8 weight."""

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def initialize_from_config(
        cls, hf_vision_config: AutoConfig, *, dtype: DType
    ) -> Self:
        """Reads the tower's fields off the checkpoint's ``vision_config``."""
        return cls(
            depth=hf_vision_config.depth,
            hidden_size=hf_vision_config.hidden_size,
            num_heads=hf_vision_config.num_heads,
            intermediate_size=hf_vision_config.intermediate_size,
            projection_intermediate_size=(
                hf_vision_config.projection_intermediate_size
            ),
            out_hidden_size=hf_vision_config.out_hidden_size,
            patch_size=hf_vision_config.patch_size,
            temporal_patch_size=hf_vision_config.temporal_patch_size,
            spatial_merge_size=hf_vision_config.spatial_merge_size,
            in_channels=getattr(hf_vision_config, "in_channels", 3),
            image_size=getattr(hf_vision_config, "image_size", 448),
            rms_norm_eps=getattr(hf_vision_config, "rms_norm_eps", 1e-5),
            hidden_act=getattr(hf_vision_config, "hidden_act", "silu"),
            attention_bias=getattr(hf_vision_config, "attention_bias", True),
            swiglu_limit=getattr(hf_vision_config, "swiglu_limit", 10.0),
            dtype=dtype,
        )


@dataclass(kw_only=True)
class Glm5NextConfig(DeepseekV3_2Config):
    """Configuration for GLM-5.3-Flash."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float8_e4m3fn",
        "bfloat16",
    }

    # ---------------------------------------------------------------- schedule

    layer_types: list[str] = field(default_factory=list)
    """Per-layer attention family: `LINEAR_ATTENTION` or `SPARSE_ATTENTION`."""

    # ------------------------------------------------------------------- KDA

    linear_num_heads: int = 64
    """KDA heads. One count for q, k and v alike, instead of separate key and
    value head counts."""

    linear_head_dim: int = 128
    """KDA head dimension, shared by q, k and v. The recurrent state is
    ``[linear_num_heads, linear_head_dim, linear_head_dim]``."""

    linear_conv_kernel_dim: int = 4
    """Depthwise causal conv1d kernel size over the concatenated QKV."""

    linear_lower_bound: float = -5.0
    """Lower bound of the bounded forget gate."""

    state_pool_dtype: DType | None = DType.float32
    """Storage dtype of the KDA conv and recurrent pools."""

    # ------------------------------------------------------------------- mHC

    mhc: bool = True
    """Whether the decoder uses manifold-constrained hyper-connections. False
    for the MTP layer, which has a plain residual add."""

    hc_mult: int = 4
    """Number of parallel residual streams."""

    hc_sinkhorn_iters: int = 20
    """Alternating row/column normalisation steps that project the stream-mix
    matrix towards doubly stochastic."""

    hc_eps: float = 1e-6

    # ----------------------------------------------------------- DSA k-pool

    index_kpool: int = 4
    """Tokens per indexer scoring pool."""

    index_kpool_compress: bool = True
    """Whether pooled keys are a learned weighted average over each pool's
    tokens (gate logits plus a learned intra-pool position embedding) rather
    than a plain mean."""

    index_kpool_always_select_tail: bool = True
    """Whether the incomplete trailing pool is always visible."""

    mla_use_nope: bool = True
    """NoPE. There is no rotary embedding anywhere in the text decoder."""

    mla_latent_pad_to: int | None = None
    """Temporarily required padding for MLA kernel."""

    # ---------------------------------------------------------- MoE and MLP

    swiglu_limit: float = 10.0
    """Clamp applied **before** the activation:
    ``silu(min(gate, limit)) * clamp(up, -limit, limit)``."""

    # ------------------------------------------------------------------- MTP

    num_nextn_predict_layers: int = 1
    """MTP draft layers, at indices ``num_hidden_layers`` onward."""

    index_share_for_mtp_iteration: bool = True
    """Whether the MTP draft layer reuses the target's top-k selection."""

    # ---------------------------------------------------------------- vision

    vision_config: Glm5NextVisionConfig | None = None
    """Vision tower configuration; ``None`` for a text-only checkpoint."""

    image_token_id: int | None = None
    video_token_id: int | None = None
    image_start_token_id: int | None = None
    image_end_token_id: int | None = None
    video_start_token_id: int | None = None
    video_end_token_id: int | None = None

    # ------------------------------------------------------------ precision

    hf_quantization_config: dict[str, Any] | None = None
    quant_scheme: Glm5NextQuantScheme | None = None
    declared_dtype: DType | None = None
    """The dtype the checkpoint declares for unquantized tensors."""

    def __post_init__(self) -> None:
        """Validates the invariants the rest of the architecture assumes."""
        super_post_init = getattr(super(), "__post_init__", None)
        if super_post_init is not None:
            super_post_init()

        if self.qk_rope_head_dim != 0:
            raise ValueError(
                "GLM-5.3-Flash is NoPE: qk_rope_head_dim must be 0, got "
                f"{self.qk_rope_head_dim}. The reference config rejects any "
                "non-zero value for the same reason."
            )
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(
                f"index_topk ({self.index_topk}) must be divisible by "
                f"index_kpool ({self.index_kpool})."
            )
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}."
            )
        unknown = set(self.layer_types) - {LINEAR_ATTENTION, SPARSE_ATTENTION}
        if unknown:
            raise ValueError(
                f"Unrecognised layer_types entries: {sorted(unknown)}"
            )

    @property
    def compute_dtype(self) -> DType:
        """Dtype of activations and every unquantized weight."""
        if self.quant_scheme is not None:
            return self.quant_scheme.compute_dtype
        if self.declared_dtype is not None:
            return self.declared_dtype
        return self.dtype

    @property
    def state_dtype(self) -> DType:
        """Storage dtype of the KDA conv and recurrent pools."""
        if self.state_pool_dtype is not None:
            return self.state_pool_dtype
        return self.compute_dtype

    @property
    def kda_layers(self) -> tuple[int, ...]:
        """Indices of the KDA linear-attention layers."""
        return tuple(
            i for i, t in enumerate(self.layer_types) if t == LINEAR_ATTENTION
        )

    @property
    def sparse_attention_layers(self) -> tuple[int, ...]:
        """Indices of the sparse-MLA layers, excluding the MTP draft layer."""
        return tuple(
            i for i, t in enumerate(self.layer_types) if t == SPARSE_ATTENTION
        )

    @property
    def conv_dim(self) -> int:
        """Width of the concatenated-QKV causal convolution."""
        return 3 * self.linear_num_heads * self.linear_head_dim

    @property
    def mla_head_dim(self) -> int:
        """Latent width the MLA KV cache stores per token."""
        return self.mla_latent_pad_to or self.kv_lora_rank

    # ------------------------------------------------------- config plumbing

    @staticmethod
    def _get_text_config(huggingface_config: AutoConfig) -> AutoConfig:
        """Extracts the text sub-config, tolerating a flat text-only config."""
        return getattr(huggingface_config, "text_config", huggingface_config)

    @staticmethod
    def resolve_layer_types(text_config: AutoConfig) -> list[str]:
        """Returns the per-layer attention family list.

        Prefers the checkpoint's explicit ``layer_types``. Falls back to
        ``linear_attn_config.full_attn_layers``, then to the period-4 rule the
        family uses (sparse where ``idx % 4 == 3``).
        """
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types:
            return list(layer_types)

        num_layers = text_config.num_hidden_layers
        linear_attn_config = (
            getattr(text_config, "linear_attn_config", None) or {}
        )
        if not isinstance(linear_attn_config, Mapping):
            linear_attn_config = vars(linear_attn_config)
        full_attn_layers = linear_attn_config.get("full_attn_layers")
        if full_attn_layers:
            full = set(full_attn_layers)
            return [
                SPARSE_ATTENTION if i in full else LINEAR_ATTENTION
                for i in range(num_layers)
            ]
        return [
            SPARSE_ATTENTION if (i % 4) == 3 else LINEAR_ATTENTION
            for i in range(num_layers)
        ]

    @staticmethod
    def _linear_attn_field(
        text_config: AutoConfig, name: str, default: Any
    ) -> Any:
        """Reads one field out of the nested ``linear_attn_config``."""
        linear_attn_config = (
            getattr(text_config, "linear_attn_config", None) or {}
        )
        if not isinstance(linear_attn_config, Mapping):
            linear_attn_config = vars(linear_attn_config)
        return linear_attn_config.get(name, default)

    @override
    @classmethod
    def calculate_max_seq_len(
        cls, huggingface_config: AutoConfig, model_config: MAXModelConfig
    ) -> int:
        """Bounds against the *text* config's ``max_position_embeddings``."""
        return super().calculate_max_seq_len(
            cls._get_text_config(huggingface_config), model_config
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        text_config = Glm5NextConfig._get_text_config(huggingface_config)
        return text_config.num_hidden_layers

    def resolve_quant_scheme(
        self, state_dict: Mapping[str, WeightData]
    ) -> QuantConfig | None:
        """Builds the mixed-precision map from the checkpoint's weight scales."""
        self.quant_scheme = parse_quant_scheme(
            self.hf_quantization_config,
            state_dict,
            self.declared_dtype or DType.bfloat16,
        )
        return self.quant_scheme.config if self.quant_scheme else None

    # ------------------------------------------------------------- KV caches

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Builds the MLA and indexer caches over the sparse-attention subset."""
        text_config = Glm5NextConfig._get_text_config(huggingface_config)
        layer_types = Glm5NextConfig.resolve_layer_types(text_config)
        num_cached_layers = sum(1 for t in layer_types if t == SPARSE_ATTENTION)
        if num_cached_layers == 0:
            raise ValueError(
                "GLM-5.3-Flash needs at least one sparse-attention layer; "
                f"layer_types resolved to {len(layer_types)} entries with none."
            )
        if pipeline_config.speculative:
            # The MTP draft layer is itself sparse MLA and owns a latent.
            num_cached_layers += getattr(
                text_config, "num_nextn_predict_layers", 1
            )

        speculative_method = None
        num_draft_tokens = 0
        if pipeline_config.speculative:
            speculative_method = pipeline_config.speculative.speculative_method
            num_draft_tokens = (
                pipeline_config.speculative.num_speculative_tokens or 0
            )

        mla_quant_config = None
        if cache_dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
            mla_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.int8, quantization_granularity=32
            )

        # NoPE: the latent is `kv_lora_rank` wide with no rotary tail, but the
        # SM100 kernels comptime-assert 576, so it is stored padded. See
        # `Glm5NextConfig.mla_latent_pad_to`.
        latent_width = _resolve_mla_latent_width(text_config)
        mla_kv_params = kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=latent_width,
            num_layers=num_cached_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            is_mla=True,
            num_q_heads=text_config.num_attention_heads,
            kvcache_quant_config=mla_quant_config,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        assert isinstance(mla_kv_params, KVCacheParams)

        # The indexer's K cache holds one *pooled* key per `index_kpool` tokens.
        index_kpool = getattr(text_config, "index_kpool", 4)
        pooled_head_dim, remainder = divmod(
            text_config.index_head_dim, index_kpool
        )
        if remainder:
            raise ValueError(
                f"index_head_dim ({text_config.index_head_dim}) must be "
                f"divisible by index_kpool ({index_kpool}) to declare the "
                "pooled key cache."
            )
        if mla_kv_params.page_size % index_kpool != 0:
            raise ValueError(
                f"kv_cache_page_size ({mla_kv_params.page_size}) must be a "
                f"multiple of index_kpool ({index_kpool}): a page that ends "
                "mid-pool would split a pooled key across two pages, and every "
                "leaf of the multi-cache shares one page table."
            )
        indexer_kv_params = kv_cache_config.to_params(
            dtype=DType.float8_e4m3fn,
            n_kv_heads=1,
            head_dim=pooled_head_dim,
            num_layers=mla_kv_params.num_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            is_mla=True,
            num_q_heads=text_config.num_attention_heads,
            kvcache_quant_config=KVCacheQuantizationConfig(
                scale_dtype=DType.float32,
                quantization_granularity=pooled_head_dim,
            ),
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        assert isinstance(indexer_kv_params, KVCacheParams)
        return MultiKVCacheParams.from_params(
            {"mla": mla_kv_params, "indexer": indexer_kv_params}
        )

    # --------------------------------------------------------- state sizing

    def per_request_state_bytes(self) -> int:
        """Approximates bytes a single request's KDA state occupies across all
        KDA layers.

        Per KDA layer, at :attr:`state_dtype`:

        * recurrent state ``[linear_num_heads, linear_head_dim, linear_head_dim]``
        * conv state ``[conv_dim, linear_conv_kernel_dim - 1]``

        At float32 and the real dimensions that is 4 MiB + 288 KiB per layer,
        or **146 MiB per sequence across 34 layers, independent of context
        length**.
        """
        num_linear = len(self.kda_layers)
        if num_linear == 0:
            return 0
        dtype_bytes = self.state_dtype.size_in_bytes
        recurrent = (
            self.linear_num_heads * self.linear_head_dim * self.linear_head_dim
        )
        conv = self.conv_dim * (self.linear_conv_kernel_dim - 1)
        return num_linear * (recurrent + conv) * dtype_bytes

    def activation_bytes_per_token(self) -> int:
        """The number of bytes a single live residual tensor occupies per token."""
        return (
            self.hc_mult * self.hidden_size * self.compute_dtype.size_in_bytes
        )

    def infer_optimal_batch_size(
        self,
        devices: Sequence[Device],
        *,
        weights_size: int,
        device_memory_utilization: float,
    ) -> int:
        """Returns a memory-safe default ``max_batch_size``.

        The KDA pools are a single ``max_batch x per_request`` allocation that
        the slot-indexed kernels mutate in place, so peak footprint is that
        allocation with no working copies. Half the post-weights budget goes to
        the state pools and the KV cache absorbs the rest.
        """
        per_request = self.per_request_state_bytes()
        if per_request == 0:
            return 32
        try:
            free_bytes = int(
                sum(d.stats.get("free_memory", 0) for d in devices)
            )
        except Exception:
            free_bytes = 0
        if free_bytes <= 0:
            # GLM-5.3-Flash needs 8 GPUs; this fallback only covers the
            # minimized bringup fixture on one.
            return 8
        budget = int(free_bytes * device_memory_utilization) - weights_size
        if budget <= 0:
            return 1
        return min(512, max(1, (budget // 2) // per_request))

    # -------------------------------------------------------- initialization

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Builds the config from the checkpoint's nested `text_config`."""
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for "
                f"'{model_config.model_path}', but config could not be loaded. "
                "Please ensure the model repository contains a valid "
                "config.json file."
            )
        text_config = cls._get_text_config(huggingface_config)

        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, kv_cache_config.kv_cache_format
        )
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

        # Use float32 for pooling unless the user asks otherwise.
        state_pool_dtype: DType | None = DType.float32
        state_pool_dtype_str = kv_cache_config.state_pool_dtype
        if state_pool_dtype_str is not None:
            pool_dtypes = {
                "bfloat16": DType.bfloat16,
                "float32": DType.float32,
            }
            if state_pool_dtype_str not in pool_dtypes:
                raise ValueError(
                    "state_pool_dtype must be 'bfloat16' or 'float32', got "
                    f"{state_pool_dtype_str!r}"
                )
            state_pool_dtype = pool_dtypes[state_pool_dtype_str]

        declared_dtype = _declared_dtype(text_config)
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        vision_config: Glm5NextVisionConfig | None = None
        if hf_vision_config is not None and hasattr(
            hf_vision_config, "patch_size"
        ):
            vision_config = Glm5NextVisionConfig.initialize_from_config(
                hf_vision_config, dtype=declared_dtype or DType.bfloat16
            )

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            use_subgraphs=model_config.use_subgraphs,
            data_parallel_degree=model_config.data_parallel_degree,
            quantization_encoding=quantization_encoding,
            # --- text spine, all off `text_config` ---
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            moe_intermediate_size=text_config.moe_intermediate_size,
            num_hidden_layers=text_config.num_hidden_layers,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            n_shared_experts=text_config.n_shared_experts,
            n_routed_experts=text_config.n_routed_experts,
            routed_scaling_factor=text_config.routed_scaling_factor,
            kv_lora_rank=text_config.kv_lora_rank,
            q_lora_rank=text_config.q_lora_rank,
            qk_rope_head_dim=text_config.qk_rope_head_dim,
            qk_nope_head_dim=text_config.qk_nope_head_dim,
            v_head_dim=text_config.v_head_dim,
            topk_method=text_config.topk_method,
            n_group=text_config.n_group,
            topk_group=text_config.topk_group,
            num_experts_per_tok=text_config.num_experts_per_tok,
            first_k_dense_replace=text_config.first_k_dense_replace,
            norm_topk_prob=text_config.norm_topk_prob,
            hidden_act=text_config.hidden_act,
            max_position_embeddings=text_config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            max_seq_len=max_seq_len,
            rms_norm_eps=text_config.rms_norm_eps,
            tie_word_embeddings=getattr(
                text_config, "tie_word_embeddings", False
            ),
            scoring_func=text_config.scoring_func,
            attention_bias=text_config.attention_bias,
            attention_dropout=text_config.attention_dropout,
            # --- DSA indexer ---
            index_head_dim=text_config.index_head_dim,
            index_n_heads=text_config.index_n_heads,
            index_topk=text_config.index_topk,
            # The checkpoint sets "full" on all 45 layers explicitly. Read it
            # rather than recomputing a frequency schedule.
            indexer_types=list(
                getattr(text_config, "indexer_types", None)
                or ["full"] * text_config.num_hidden_layers
            ),
            index_kpool=getattr(text_config, "index_kpool", 4),
            index_kpool_compress=getattr(
                text_config, "index_kpool_compress", True
            ),
            index_kpool_always_select_tail=getattr(
                text_config, "index_kpool_always_select_tail", True
            ),
            mla_use_nope=getattr(text_config, "mla_use_nope", True),
            # `kv_b_proj` is [32768, 512] BF16 while `q_a_proj`, `q_b_proj`,
            # `kv_a_proj_with_mqa` and `o_proj` are all FP8.
            kv_b_proj_dtype=declared_dtype or DType.bfloat16,
            # TODO(KERN-3520): Remove this when fixed.
            mla_latent_pad_to=_resolve_mla_latent_width(text_config),
            # --- hybrid schedule ---
            layer_types=cls.resolve_layer_types(text_config),
            # --- KDA ---
            linear_num_heads=cls._linear_attn_field(
                text_config, "num_heads", 64
            ),
            linear_head_dim=cls._linear_attn_field(
                text_config, "head_dim", 128
            ),
            linear_conv_kernel_dim=cls._linear_attn_field(
                text_config, "short_conv_kernel_size", 4
            ),
            linear_lower_bound=cls._linear_attn_field(
                text_config, "gate_lower_bound", -5.0
            ),
            state_pool_dtype=state_pool_dtype,
            # --- mHC ---
            mhc=getattr(text_config, "mhc", True),
            hc_mult=getattr(text_config, "hc_mult", 4),
            hc_sinkhorn_iters=getattr(text_config, "hc_sinkhorn_iters", 20),
            hc_eps=getattr(text_config, "hc_eps", 1e-6),
            # --- activation ---
            swiglu_limit=getattr(text_config, "swiglu_limit", 10.0),
            # --- MTP ---
            num_nextn_predict_layers=getattr(
                text_config, "num_nextn_predict_layers", 1
            ),
            index_share_for_mtp_iteration=getattr(
                text_config, "index_share_for_mtp_iteration", True
            ),
            # --- vision ---
            vision_config=vision_config,
            image_token_id=getattr(huggingface_config, "image_token_id", None),
            video_token_id=getattr(huggingface_config, "video_token_id", None),
            image_start_token_id=getattr(
                huggingface_config, "image_start_token_id", None
            ),
            image_end_token_id=getattr(
                huggingface_config, "image_end_token_id", None
            ),
            video_start_token_id=getattr(
                huggingface_config, "video_start_token_id", None
            ),
            video_end_token_id=getattr(
                huggingface_config, "video_end_token_id", None
            ),
            # --- precision ---
            hf_quantization_config=resolve_hf_quant_config(
                huggingface_config, {}
            ),
            declared_dtype=declared_dtype,
        )
