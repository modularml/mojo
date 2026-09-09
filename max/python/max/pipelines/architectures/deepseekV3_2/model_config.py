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
"""Config for DeepseekV3.2 models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    KVCacheParams,
    KVCacheQuantizationConfig,
    MultiKVCacheParams,
    spec_decode_cache_slack,
)
from max.pipelines.architectures.deepseekV3.model_config import DeepseekV3Config
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.pipelines.speculative.config import SpeculativeMethod
from transformers import AutoConfig
from typing_extensions import Self, override


def resolve_indexer_types(
    huggingface_config: AutoConfig, num_hidden_layers: int
) -> list[str]:
    """Resolve the per-layer indexer schedule for DeepSeek Sparse Attention.

    Each layer is either ``"full"`` (runs its own lightning indexer top-k) or
    ``"shared"`` (reuses the top-k selection from the most recent full layer).

    If the model doesn't provide an indexer schedule, it will default to all
    ``"full"`` layers.
    """
    types = getattr(huggingface_config, "indexer_types", None)
    if types is not None:
        return list(types)

    pattern = getattr(huggingface_config, "index_topk_pattern", None)
    if pattern is not None:
        if isinstance(pattern, str):
            return [{"F": "full", "S": "shared"}[c] for c in pattern]
        return list(pattern)

    freq = max(getattr(huggingface_config, "index_topk_freq", 1) or 1, 1)
    offset = getattr(huggingface_config, "index_skip_topk_offset", 2)
    return [
        "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
        for i in range(num_hidden_layers)
    ]


@dataclass(kw_only=True)
class DeepseekV3_2Config(DeepseekV3Config):
    """Configuration for DeepseekV3.2 models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"float8_e4m3fn"}

    unpadded_vocab_size: int | None = None

    # Added parameters for the Indexer used in DeepSeek Sparse Attention.
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
    indexer_types: list[str] = field(default_factory=list)
    # GLM-5.x sets indexer_rope_interleave=true.
    indexer_rope_interleave: bool = False

    kv_b_proj_dtype: DType | None = None
    """Storage dtype of ``kv_b_proj`` when it differs from the rest of the
    attention block.

    ``None`` keeps it quantized with the other three sparse-MLA projections,
    which is what DeepSeek-V3.2 and GLM-5.2 ship. A checkpoint that leaves this
    one projection unquantized sets it here: the absorb then reads the weight
    directly instead of dequantizing, and declares no
    ``kv_b_proj.weight_scale``."""

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        mla_kv_params = DeepseekV3Config.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=devices,
            kv_cache_config=kv_cache_config,
            cache_dtype=cache_dtype,
        )

        # Always store the indexer's K cache in float8_e4m3fn.
        indexer_cache_dtype = DType.float8_e4m3fn
        # Per-token FP8 KV scales use float32 storage; required for
        # ``KVCacheParams.quantized_kv_cache``, runtime ``kv_scales`` buffers,
        # and ``store_k_scale_cache`` in the indexer path.
        indexer_kvcache_quant_config = KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=32
        )
        assert isinstance(mla_kv_params, KVCacheParams)

        speculative_method: SpeculativeMethod | None = None
        num_draft_tokens: int = 0
        if pipeline_config.speculative:
            speculative_method = pipeline_config.speculative.speculative_method
            num_draft_tokens = (
                pipeline_config.speculative.num_speculative_tokens or 0
            )

        indexer_kv_params = kv_cache_config.to_params(
            dtype=indexer_cache_dtype,
            # Similar to MLA, the indexer's k-cache uses a single KV head.
            n_kv_heads=1,
            head_dim=huggingface_config.index_head_dim,
            num_layers=mla_kv_params.num_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            # Set to True because there is only a key-cache, and one KV head.
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            kvcache_quant_config=indexer_kvcache_quant_config,
            speculative_method=speculative_method,
            num_draft_tokens=num_draft_tokens,
        )
        assert isinstance(indexer_kv_params, KVCacheParams)
        return MultiKVCacheParams.from_params(
            {"mla": mla_kv_params, "indexer": indexer_kv_params}
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
        """Initializes a DeepseekV3_2Config instance from pipeline configuration.

        This method creates a config instance with all fields that can be determined
        from the pipeline configuration, without needing the state_dict.
        Fields that depend on the state_dict (like norm_dtype, quant_config, etc.)
        should be set directly after calling this method.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized DeepseekV3_2Config instance.
        """
        model_config = model_config or pipeline_config.model
        config = model_config.huggingface_config
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

        return cls(
            dtype=dtype,
            kv_params=kv_params,
            devices=device_refs,
            use_subgraphs=model_config.use_subgraphs,
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
            max_position_embeddings=config.max_position_embeddings
            + spec_decode_cache_slack(kv_params),
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=get_rope_theta(config),
            rope_scaling=config.rope_scaling,
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            # DeepseekV3.2 specific fields
            index_head_dim=config.index_head_dim,
            index_n_heads=config.index_n_heads,
            index_topk=config.index_topk,
            indexer_types=resolve_indexer_types(
                config, config.num_hidden_layers
            ),
            indexer_rope_interleave=getattr(
                config, "indexer_rope_interleave", False
            ),
            quantization_encoding=quantization_encoding,
        )
