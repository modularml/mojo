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

"""Memory planner for the DeepseekV3 NextN architecture."""

from __future__ import annotations

import logging

from max.dtype import DType
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    is_float4_encoding,
    supported_encoding_dtype,
)
from max.support.human_readable_formatter import to_human_readable_bytes

from .model_config import DeepseekV3NextNConfig

logger = logging.getLogger(__name__)


class DeepseekV3NextNMemoryPlanner(PagedMemoryPlanner):
    """Memory planner for DeepseekV3 NextN (speculative-decoding draft) models.

    Accounts for the single-decoder-layer NextN structure and
    expert-parallel sharding.
    """

    _always_signal_buffers = True

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Calculates the estimated memory consumption of the DeepseekV3 NextN model.

        The NextN model consists of:
        - embed_tokens: VocabParallelEmbedding (shared in EAGLE/MTP mode)
        - lm_head: ColumnParallelLinear (shared in EAGLE/MTP mode)
        - enorm, hnorm, shared_head_norm: RMSNorm layers
        - eh_proj: Linear layer (hidden_size * 2 -> hidden_size)
        - decoder_layer: Single DeepseekV3DecoderLayer (MoE layer)

        Args:
            pipeline_config: The pipeline configuration containing model settings.

        Returns:
            Estimated weight memory in bytes.
        """
        draft_model_config = pipeline_config.draft_model
        assert draft_model_config is not None, (
            "draft_model must be set for NextN"
        )
        encoding = _select_quantization_encoding(
            draft_model_config, DeepseekV3NextNConfig.DEFAULT_ENCODING
        )
        # NextN weights are always BF16 even when the pipeline encoding is FP4,
        # because the NextN checkpoint is not quantized.
        if is_float4_encoding(encoding):
            dtype_bytes = DType.bfloat16.size_in_bytes
        else:
            dtype_bytes = supported_encoding_dtype(encoding).size_in_bytes
        config = draft_model_config.huggingface_config
        assert config is not None
        n_gpus_per_node = len(draft_model_config.device_specs)
        data_parallel_degree = pipeline_config.model.data_parallel_degree

        total_size = 0

        sharing_enabled = pipeline_config.speculative is not None and (
            pipeline_config.speculative.is_eagle()
            or pipeline_config.speculative.is_mtp()
        )

        # 1. Embedding and LM head (always in BF16 unless shared with target)
        # In EAGLE/MTP, embedding and lm_head are shared with the target model.
        embedding_size = (
            config.vocab_size
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        lm_head_size = embedding_size
        if not sharing_enabled:
            total_size += embedding_size + lm_head_size

        # 2-4. Non-expert weights: norms, eh_proj, attention, router.
        # In DP mode these are replicated per DP rank; in TP mode they are
        # sharded across devices. Multiply by data_parallel_degree to account
        # for this, matching the pattern in DeepseekV3Model.estimate_weights_size.
        non_expert_size = 0

        # 2. NextN-specific norms (enorm, hnorm, shared_head_norm) - always BF16
        norm_size = config.hidden_size * DType.bfloat16.size_in_bytes
        non_expert_size += 2 * norm_size
        if not sharing_enabled:
            non_expert_size += norm_size

        # 3. eh_proj: Linear(hidden_size * 2, hidden_size)
        eh_proj_size = config.hidden_size * 2 * config.hidden_size * dtype_bytes
        non_expert_size += eh_proj_size

        # 4. Single decoder layer components

        # 4a. Layer norms (input_layernorm, post_attention_layernorm)
        non_expert_size += 2 * norm_size

        # 4b. MLA attention weights
        num_heads = config.num_attention_heads
        # kv_a_proj: hidden_size -> kv_lora_rank + qk_rope_head_dim
        kv_a_proj_size = (
            config.hidden_size
            * (config.kv_lora_rank + config.qk_rope_head_dim)
            * dtype_bytes
        )
        # kv_a_layernorm: kv_lora_rank
        kv_a_layernorm_size = config.kv_lora_rank * DType.bfloat16.size_in_bytes
        # kv_b_proj: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        kv_b_proj_size = (
            config.kv_lora_rank
            * num_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
            * dtype_bytes
        )
        # q_proj: hidden_size -> num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        q_proj_size = (
            config.hidden_size
            * num_heads
            * (config.qk_nope_head_dim + config.qk_rope_head_dim)
            * dtype_bytes
        )
        # o_proj: num_heads * v_head_dim -> hidden_size
        o_proj_size = (
            num_heads * config.v_head_dim * config.hidden_size * dtype_bytes
        )

        attn_size = (
            kv_a_proj_size
            + kv_a_layernorm_size
            + kv_b_proj_size
            + q_proj_size
            + o_proj_size
        )
        non_expert_size += attn_size

        # 4c. MoE weights (single layer)
        # Expert FFN: gate_proj, up_proj, down_proj
        expert_size = (
            config.moe_intermediate_size * config.hidden_size * 3 * dtype_bytes
        )
        routing_experts_size = config.n_routed_experts * expert_size
        shared_experts_size = config.n_shared_experts * expert_size

        # Router gate weights
        router_size = config.hidden_size * config.n_routed_experts * dtype_bytes
        non_expert_size += router_size

        total_size += non_expert_size * data_parallel_degree

        # Handle expert parallelism
        ep_size = max(pipeline_config.runtime.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # Routing experts are sharded across nodes
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Shared experts are replicated on each device
        total_size += shared_experts_size * n_gpus_per_node

        logger.info(
            f"Estimated NextN weights size: {to_human_readable_bytes(total_size)}"
        )

        return total_size
