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
"""Config for Kimi-K2.5 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
    ArchConfigWithVisionCache,
    ArchVLConfigWithTextSubconfig,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

from ..deepseekV3_modulev3.model_config import DeepseekV3Config


class _KimiK2_5VisionCacheConfig:
    """Vision-cache facts shared by both registered Kimi K2.5 arch configs."""

    @classmethod
    def get_vision_cache_row_spec(
        cls,
        huggingface_config: AutoConfig,
    ) -> tuple[int, DType] | None:
        """One embedding row per merged vision token: text hidden, bfloat16."""
        text_config = getattr(huggingface_config, "text_config", None)
        if text_config is None:
            raise ValueError(
                "KimiK2.5 requires a text_config in the HuggingFace config"
            )
        hidden = getattr(text_config, "hidden_size", 0)
        if hidden <= 0:
            raise ValueError(
                "KimiK2.5 text_config.hidden_size must be positive"
            )
        return (hidden, DType.bfloat16)

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates per-entry bytes for the Kimi K2.5 vision encoder cache.

        Max tokens per image = pos_emb_height * pos_emb_width / merge_sq,
        multiplied by the text hidden size and 2 bytes (bfloat16).

        Args:
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated bytes per vision cache entry.

        Raises:
            ValueError: If required vision or text config fields are absent or
                invalid.
        """
        vision_config = getattr(huggingface_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "KimiK2.5 requires a vision_config in the HuggingFace config"
            )
        text_config = getattr(huggingface_config, "text_config", None)
        if text_config is None:
            raise ValueError(
                "KimiK2.5 requires a text_config in the HuggingFace config"
            )
        hidden = getattr(text_config, "hidden_size", 0)
        if hidden <= 0:
            raise ValueError(
                "KimiK2.5 text_config.hidden_size must be positive"
            )
        merge_kernel_size = getattr(vision_config, "merge_kernel_size", [2, 2])
        merge_sq = 1
        for k in (
            merge_kernel_size
            if isinstance(merge_kernel_size, (list, tuple))
            else [merge_kernel_size]
        ):
            merge_sq *= k
        pos_h = getattr(vision_config, "init_pos_emb_height", 0)
        pos_w = getattr(vision_config, "init_pos_emb_width", 0)
        if pos_h <= 0 or pos_w <= 0:
            raise ValueError(
                "KimiK2.5 vision_config must provide "
                "init_pos_emb_height and init_pos_emb_width"
            )
        max_tokens = (pos_h * pos_w) // merge_sq
        return max_tokens * hidden * 2


class KimiK2_5TextConfig(
    _KimiK2_5VisionCacheConfig, DeepseekV3Config, ArchConfigWithVisionCache
):
    """DeepseekV3 (ModuleV3) text config for the Kimi K2.5 language tower.

    Reads the language parameters from ``huggingface_config.text_config`` while
    reusing the DeepseekV3 ModuleV3 config machinery (mesh, EP, YaRN RoPE).
    """

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a DeepseekV3Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.
        Fields that depend on the state_dict (like norm_dtype, quant_config, etc.)
        should be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized DeepseekV3Config instance.
        """
        model_config = model_config or pipeline_config.model
        assert model_config.huggingface_config is not None
        hf_config = model_config.huggingface_config
        config = getattr(hf_config, "text_config", hf_config)
        if config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        kv_cache_config = model_config.kv_cache
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        cache_dtype = cache_dtype_for_encoding(
            quantization_encoding, model_config.kv_cache.kv_cache_format
        )

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]

        kv_params = cls.construct_kv_params(
            huggingface_config=config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        if pipeline_config.runtime.pipeline_role == "prefill_only":
            graph_mode = "prefill"
        elif pipeline_config.runtime.pipeline_role == "decode_only":
            graph_mode = "decode"
        else:
            graph_mode = "auto"

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            data_parallel_degree=model_config.data_parallel_degree,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            moe_layer_freq=config.moe_layer_freq,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            n_shared_experts=config.n_shared_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.routed_scaling_factor,
            kv_lora_rank=config.kv_lora_rank,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            topk_method=config.topk_method,
            n_group=config.n_group,
            topk_group=config.topk_group,
            num_experts_per_tok=config.num_experts_per_tok,
            first_k_dense_replace=config.first_k_dense_replace,
            norm_topk_prob=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=get_rope_theta(config),
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            graph_mode=graph_mode,
            quantization_encoding=quantization_encoding,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        # DeepseekV3Config does not inherit ArchConfigWithStoredKVParams, so the
        # VLM mixin cannot delegate max_seq_len to this class directly.
        return ArchConfigWithStoredKVParams.calculate_max_seq_len(
            huggingface_config, model_config
        )


@dataclass
class VisionConfig:
    """Vision configuration for Kimi-K2.5 models with required fields."""

    # Required fields (no defaults); order must precede any field with a default.
    dtype: DType
    """DType of the Kimi-K2.5 vision model weights."""
    devices: list[DeviceRef]
    """Devices that the Kimi-K2.5 vision encoder model is parallelized over."""

    init_pos_emb_height: int
    """Height of the initial position embedding."""
    init_pos_emb_time: int
    """Time of the initial position embedding."""
    init_pos_emb_width: int
    """Width of the initial position embedding."""
    merge_kernel_size: list[int]
    """Kernel size for the merge operation."""
    mm_hidden_size: int
    """Hidden size of the multi-modal hidden layer."""
    patch_size: int
    """Size of the patch."""
    projector_ln_eps: float
    """Epsilon for the layer normalization."""
    text_hidden_size: int
    """Hidden size of the text hidden layer."""
    vt_hidden_size: int
    """Hidden size of the video hidden layer."""
    vt_intermediate_size: int
    """Intermediate size of the video hidden layer."""
    vt_num_attention_heads: int
    """Number of attention heads of the video hidden layer."""
    vt_num_hidden_layers: int
    """Number of hidden layers of the video hidden layer."""

    # Optional fields (from HF config or defaults).
    merge_type: str | None = None
    """Type of the merge operation."""
    mm_projector_type: str | None = None
    """Type of the multi-modal projector."""
    model_type: str = ""
    """Type of the model."""
    pos_emb_type: str | None = None
    """Type of the position embedding."""
    projector_hidden_act: str | None = None
    """Activation function for the projector."""
    video_attn_type: str | None = None
    """Type of the video attention."""

    # Fields with hardcoded defaults (not present in HF config.json).
    has_bias: bool = True
    """Whether linear projections in the vision transformer include bias terms."""
    in_channels: int = 3
    """Number of input image channels (3 for RGB).
    """
    rope_max_height: int = 512
    """Maximum grid height for RoPE frequency precomputation. Hardcoded to 512 in
    https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/modeling_kimi_k25.py#L571
    """
    rope_max_width: int = 512
    """Maximum grid width for RoPE frequency precomputation. Hardcoded to 512 in
    https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/modeling_kimi_k25.py#L571
    """
    rope_theta: float = 10000.0
    """Base for the RoPE inverse-frequency exponent. Hardcoded to 10000 in
    https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/modeling_kimi_k25.py#L379
    """

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        hf_vision_config: AutoConfig,
        huggingface_config: AutoConfig | None = None,
    ) -> VisionConfig:
        """Initialize VisionConfig from HuggingFace vision config.

        Args:
            pipeline_config: MAX Engine pipeline configuration.
            hf_vision_config: HuggingFace vision sub-config.
            huggingface_config: Full HuggingFace model config, used to derive
                ``text_hidden_size`` from ``text_config.hidden_size`` when
                ``hf_vision_config`` does not carry the attribute directly
                (e.g. moonshotai/Kimi-VL-A3B-Instruct vs nvidia/Kimi-K2.5-NVFP4).

        Note: dtype fields will be set to defaults and should be updated
        via finalize() once state_dict is available.
        """
        # text_hidden_size is the patch-merger output dim, which must match the
        # LLM hidden size.  Prefer the explicit attribute on the vision config;
        # fall back to the LLM text_config hidden_size; then a last-resort default.
        text_hidden_size = getattr(hf_vision_config, "text_hidden_size", None)
        if text_hidden_size is None and huggingface_config is not None:
            llm_cfg = getattr(
                huggingface_config, "text_config", huggingface_config
            )
            text_hidden_size = llm_cfg.hidden_size
        if text_hidden_size is None:
            text_hidden_size = 7168  # last-resort default (Kimi-K2.5 value)

        return cls(
            dtype=DType.bfloat16,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            init_pos_emb_height=hf_vision_config.init_pos_emb_height,
            init_pos_emb_time=getattr(hf_vision_config, "init_pos_emb_time", 4),
            init_pos_emb_width=hf_vision_config.init_pos_emb_width,
            merge_kernel_size=hf_vision_config.merge_kernel_size,
            merge_type=getattr(hf_vision_config, "merge_type", None),
            mm_hidden_size=getattr(hf_vision_config, "mm_hidden_size", 1152),
            mm_projector_type=getattr(
                hf_vision_config, "mm_projector_type", None
            ),
            model_type=hf_vision_config.model_type,
            patch_size=hf_vision_config.patch_size,
            pos_emb_type=getattr(hf_vision_config, "pos_emb_type", None),
            projector_hidden_act=getattr(
                hf_vision_config, "projector_hidden_act", None
            ),
            projector_ln_eps=getattr(
                hf_vision_config, "projector_ln_eps", 1e-05
            ),
            text_hidden_size=text_hidden_size,
            video_attn_type=getattr(hf_vision_config, "video_attn_type", None),
            vt_hidden_size=hf_vision_config.vt_hidden_size
            if hasattr(hf_vision_config, "vt_hidden_size")
            else hf_vision_config.hidden_size,
            vt_intermediate_size=hf_vision_config.vt_intermediate_size
            if hasattr(hf_vision_config, "vt_intermediate_size")
            else hf_vision_config.intermediate_size,
            vt_num_attention_heads=hf_vision_config.vt_num_attention_heads
            if hasattr(hf_vision_config, "vt_num_attention_heads")
            else hf_vision_config.num_attention_heads,
            vt_num_hidden_layers=hf_vision_config.vt_num_hidden_layers
            if hasattr(hf_vision_config, "vt_num_hidden_layers")
            else hf_vision_config.num_hidden_layers,
        )

    def finalize(self, vision_dtype: DType) -> None:
        """Finalize VisionConfig with state_dict dependent fields."""
        self.dtype = vision_dtype


@dataclass(kw_only=True)
class KimiK2_5Config(
    _KimiK2_5VisionCacheConfig,
    ArchVLConfigWithTextSubconfig,
    ArchConfigWithKVCache,
    ArchConfigWithVisionCache,
):
    """Configuration for Kimi-K2.5 models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float8_e4m3fn",
        "float4_e2m1fnx2",
    }

    devices: list[DeviceRef]
    """Devices that the Kimi-K2.5 model is parallelized over."""

    dtype: DType
    """DType of the Kimi-K2.5 model weights."""

    quantization_encoding: SupportedEncoding | None = None
    """The resolved weight encoding the model runs with."""

    bos_token_id: int
    """ID of the beginning-of-sequence (BOS) token."""

    eos_token_id: int
    """ID of the end-of-sequence (EOS) token."""

    ignore_index: int
    """Index that should be ignored when calculating loss (e.g., for padding)."""

    media_placeholder_token_id: int
    """Token ID used as a placeholder for media (e.g., images, video frames) within sequences."""

    pad_token_id: int
    """Token ID used for padding sequences to uniform length."""

    tie_word_embeddings: bool
    """Whether to share (tie) the input and output word embeddings in the language model."""

    use_unified_vision_chunk: bool | None
    """Whether to use a unified chunk for vision inputs."""

    video_placeholder: str | None
    """Placeholder string used to represent video segments in input text."""

    # Vision encoder configuration.
    vision_config: VisionConfig
    """Vision encoder configuration."""

    llm_config: KimiK2_5TextConfig
    """Language model configuration using DeepseekV3 architecture."""

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters from the embedded LLM config."""
        return self.llm_config.get_kv_params()

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        # Delegate to KimiK2_5TextConfig for language model parameters.
        llm_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return KimiK2_5TextConfig.get_num_layers(llm_config)

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a Qwen3VLConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            A Qwen3VLConfig instance with fields initialized from config.
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

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        llm_config: KimiK2_5TextConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a KimiK2_5Config from pipeline and HuggingFace configs.

        This method creates a config instance with all fields that can be
        determined from the pipeline and HuggingFace configurations, without
        needing the state_dict. Fields that depend on the state_dict should
        be set via the `finalize()` method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.
            huggingface_config: HuggingFace model configuration.
            llm_config: Pre-initialized DeepseekV3 configuration.

        Returns:
            A KimiK2_5Config instance ready for finalization.
        """
        hf_vision_config = getattr(huggingface_config, "vision_config", None)
        if hf_vision_config is None:
            raise ValueError("vision_config not found in huggingface_config")

        # Get quantization encoding for dtype
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)

        # Create VisionConfig from the vision config

        # Propagate quantization_config to vision_config if present on main
        # config but not vision_config
        if hasattr(huggingface_config, "quantization_config") and not hasattr(
            hf_vision_config, "quantization_config"
        ):
            hf_vision_config.quantization_config = (
                huggingface_config.quantization_config
            )

        vision_config = VisionConfig.initialize_from_config(
            pipeline_config,
            hf_vision_config,
            huggingface_config=huggingface_config,
        )

        # Create KimiK2_5TextConfig for the language model

        # Propagate quantization_config to text_config if present on main config
        # but not text_config
        if hasattr(huggingface_config, "quantization_config") and not hasattr(
            huggingface_config.text_config, "quantization_config"
        ):
            huggingface_config.text_config.quantization_config = (
                huggingface_config.quantization_config
            )

        # For VLM models, tie_word_embeddings on the top-level config determines
        # whether lm_head.weight exists in the checkpoint. The text_config may
        # have a different value (e.g., Qwen3-VL-4B-FP8 has top-level=false but
        # text_config=true). Use top-level value to match actual checkpoint.
        if hasattr(huggingface_config, "tie_word_embeddings"):
            huggingface_config.text_config.tie_word_embeddings = (
                huggingface_config.tie_word_embeddings
            )

        if llm_config is None:
            llm_config = KimiK2_5TextConfig.initialize(
                pipeline_config, max_seq_len=max_seq_len
            )

        return cls(
            dtype=dtype,
            devices=[
                DeviceRef(spec.device_type, spec.id)
                for spec in pipeline_config.model.device_specs
            ],
            # Multimodal parameters
            bos_token_id=huggingface_config.bos_token_id,
            eos_token_id=huggingface_config.eos_token_id,
            ignore_index=huggingface_config.ignore_index,
            media_placeholder_token_id=huggingface_config.media_placeholder_token_id,
            pad_token_id=huggingface_config.pad_token_id,
            tie_word_embeddings=huggingface_config.tie_word_embeddings,
            use_unified_vision_chunk=getattr(
                huggingface_config, "use_unified_vision_chunk", None
            ),
            video_placeholder=getattr(
                huggingface_config, "video_placeholder", None
            ),
            # Vision configuration
            vision_config=vision_config,
            # Composed language model configuration
            llm_config=llm_config,
            quantization_encoding=quantization_encoding,
        )
