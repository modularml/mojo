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
"""Config for GLM-5.1 (GlmMoeDsa) models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
    resolve_indexer_types,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

logger = logging.getLogger("max.pipelines")


def _glm_unpadded_vocab_size(
    pipeline_config: PipelineConfig,
) -> int | None:
    """Returns the tokenizer's token count, or ``None`` to skip tail masking."""
    try:
        tokenizer = PIPELINE_REGISTRY.get_active_tokenizer(
            pipeline_config.model.huggingface_model_repo
        )
    except Exception as e:
        # Skipping the mask leaves the untrained tail sampleable, so say so
        # rather than degrading silently.
        logger.warning(
            "GLM-5.x: could not read the tokenizer vocab size (%s); the "
            "padded vocab tail will not be masked.",
            e,
        )
        return None

    return len(tokenizer)


def _glm_rope_scaling(huggingface_config: AutoConfig) -> dict[str, Any] | None:
    """Return YaRN rope_scaling for MAX, or None for standard RoPE.

    GLM-5.x repos declare ``rope_parameters`` with ``rope_type: "default"``
    (standard RoPE). Transformers may mirror that into ``rope_scaling``; MAX
    DeepSeek-V3.2 paths only accept YaRN or no scaling.
    """
    rope_scaling = getattr(huggingface_config, "rope_scaling", None)
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type in (None, "default"):
            return None
        return rope_scaling

    rope_parameters = getattr(huggingface_config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        rope_type = rope_parameters.get("rope_type")
        if rope_type in (None, "default"):
            return None

    return rope_scaling


@dataclass(kw_only=True)
class Glm5_1Config(DeepseekV3_2Config):
    """Configuration for GLM-5.1 models.

    Skeleton alias of :class:`~max.pipelines.architectures.deepseekV3_2.model_config.DeepseekV3_2Config`
    until GLM-specific bring-up diverges from DeepSeek-V3.2.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float8_e4m3fn"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float4_e2m1fnx2",
        "float8_e4m3fn",
        "bfloat16",
    }

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initialize config, mapping GLM default RoPE to ``rope_scaling=None``."""
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
            max_position_embeddings=config.max_position_embeddings,
            max_seq_len=max_seq_len,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=get_rope_theta(config),
            rope_scaling=_glm_rope_scaling(config),
            rope_interleave=getattr(config, "rope_interleave", True),
            scoring_func=config.scoring_func,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
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
            unpadded_vocab_size=_glm_unpadded_vocab_size(pipeline_config),
        )
