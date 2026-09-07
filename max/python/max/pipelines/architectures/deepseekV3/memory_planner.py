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

"""Memory planner for the DeepseekV3 architecture."""

from __future__ import annotations

import logging

from max.dtype import DType
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import (
    is_float4_encoding,
    supported_encoding_dtype,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

from .model_config import DeepseekV3Config

logger = logging.getLogger("max.pipelines")

_GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE = 8 * 1024**3


def _get_mtp_draft_ep_dispatch_dtype(
    pipeline_config: PipelineConfig,
) -> DType | None:
    """Returns the draft model's EP dispatch dtype for MTP with FP4 target.

    When MTP speculative decoding is used with an FP4 target model, EP
    buffers must be sized for the draft model's (larger) dispatch dtype.
    Returns ``None`` if this override is not needed.
    """
    spec_config = pipeline_config.speculative
    if spec_config is None or not spec_config.is_mtp():
        return None

    encoding = _select_quantization_encoding(
        pipeline_config.model, DeepseekV3Config.DEFAULT_ENCODING
    )
    if not is_float4_encoding(encoding):
        return None

    draft_encoding = (
        _select_quantization_encoding(
            pipeline_config.draft_model, DeepseekV3Config.DEFAULT_ENCODING
        )
        if pipeline_config.draft_model is not None
        else None
    )
    if draft_encoding is None:
        return None

    return supported_encoding_dtype(draft_encoding)


def _ep_max_rank_send_tokens_for_pipeline(
    pipeline_config: PipelineConfig,
) -> int:
    """Upper bound on EP dispatch tokens held on one rank for this pipeline."""
    return calculate_ep_max_tokens_per_rank(
        max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
        ep_size=pipeline_config.runtime.ep_size,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        use_allreduce=pipeline_config.runtime.ep_use_allreduce,
    )


class DeepseekV3MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for DeepseekV3 models.

    Accounts for MLA up-projection buffers, expert-parallel routing memory,
    and EP SHMEM communication buffers.
    """

    _always_signal_buffers = True

    def _ep_max_rank_send_tokens(self, pipeline_config: PipelineConfig) -> int:
        """Upper bound on EP dispatch tokens held on one rank.

        Delegates to the module-level helper by default. Subclasses (e.g.
        ``DeepseekV3_2MemoryPlanner``) may override for architecture-specific
        EP token sizing.
        """
        return _ep_max_rank_send_tokens_for_pipeline(pipeline_config)

    def estimate_weights_size(self, pipeline_config: PipelineConfig) -> int:
        """Estimates weight memory for DeepseekV3 models.

        Adjusts the raw weight file size to account for expert-parallel
        sharding, shared expert replication, and DP attention duplication.

        Args:
            pipeline_config: Pipeline configuration.

        Returns:
            Estimated weight memory in bytes.
        """
        model_config = pipeline_config.model
        weights_size = model_config.weights_size()
        n_gpus_per_node = len(model_config.device_specs)

        encoding = _select_quantization_encoding(
            pipeline_config.model, DeepseekV3Config.DEFAULT_ENCODING
        )

        def _n_elems_to_bytes(n_elems: int) -> int:
            dtype = supported_encoding_dtype(encoding).size_in_bytes
            if is_float4_encoding(encoding):
                # Account for the scales. For NVFP4 format, every 16 FP4 elements
                # share one FP8 scale factor. The size of the scales is one
                # eighth of the size of the FP4 quants (8 bits / (16 * 4 bits)).
                return int(n_elems // 2 * dtype * 1.125)
            else:
                return n_elems * dtype

        config = model_config.huggingface_config
        assert config is not None
        n_sparse_layers = (
            config.num_hidden_layers - config.first_k_dense_replace
        )
        n_mtp_layers = config.num_nextn_predict_layers

        # Note: All the following calculations are not exact, but they are
        # better than directly using the raw weights size.

        # First, Calculate the lm_head/embed_tokens size.
        # These are always in BF16.
        lm_head_size = (
            config.vocab_size
            * config.hidden_size
            * DType.bfloat16.size_in_bytes
        )
        embed_tokens_size = lm_head_size

        # Subtract the lm_head/embed_tokens size from the weights size
        weights_size -= lm_head_size + embed_tokens_size
        weights_size -= (lm_head_size + embed_tokens_size) * n_mtp_layers

        # We don't use the MTP module for now, so subtract the MTP attn/moe size.
        # Estimate the MTP module size by assuming the MTP layer is of the same
        # size as a sparse model layer.
        weights_size = int(
            weights_size * n_sparse_layers / (n_sparse_layers + n_mtp_layers)
        )

        # Calculate the routing experts and the shared experts size.
        expert_elems = (
            config.moe_intermediate_size * config.hidden_size * 3
        )  # A factor of 3 accounts for the gate/up/down proj weights.
        expert_size = _n_elems_to_bytes(expert_elems)
        routing_experts_size = (
            n_sparse_layers * config.n_routed_experts * expert_size
        )
        shared_experts_size = (
            n_sparse_layers * config.n_shared_experts * expert_size
        )

        # Estimate the size of the attention weights.
        attn_weights_size = (
            weights_size - routing_experts_size - shared_experts_size
        )

        # If we use DP attention, attention weights are duplicated on each DP rank.
        total_size = attn_weights_size * model_config.data_parallel_degree

        # The shared experts are duplicated on each device.
        total_size += shared_experts_size * n_gpus_per_node

        ep_size = max(pipeline_config.runtime.ep_size, 1)
        if ep_size == 1:
            total_size += routing_experts_size
        else:
            # we don't support mixing EP and TP strategies yet.
            # ep_size must be equal to n_gpus_per_node * n_nodes
            assert ep_size % n_gpus_per_node == 0
            n_nodes = ep_size // n_gpus_per_node
            total_size += routing_experts_size // n_nodes

        # Add back the lm_head/embed_tokens size, they will never be duplicated.
        total_size += lm_head_size + embed_tokens_size

        return total_size

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates activation memory for DeepseekV3 models.

        Accounts for MLA up-projection buffers during prefill, expert-parallel
        routing buffers, EP SHMEM communication buffers, and optional CUDA
        graph capture headroom.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated activation memory in bytes.
        """
        encoding = _select_quantization_encoding(
            pipeline_config.model, DeepseekV3Config.DEFAULT_ENCODING
        )
        mla_activation_memory: int = 0
        moe_activation_memory: int = 0
        ep_buffer_memory = 0

        # During the prefill, we need to up-project all the KV cache for
        # current requests. The total context length of requests in a batch
        # should be limited by max_batch_total_tokens.
        if pipeline_config.runtime.pipeline_role != "decode_only":
            max_kv_length: int = 0

            if pipeline_config.runtime.max_batch_total_tokens is None:
                # If max_batch_total_tokens is not set, we use max_length.
                max_kv_length = pipeline_config.model.max_length or 0
            else:
                max_kv_length = pipeline_config.runtime.max_batch_total_tokens

            cache_dtype = cache_dtype_for_encoding(
                encoding,
                pipeline_config.model.kv_cache.kv_cache_format,
            )
            mla_activation_memory += (
                pipeline_config.model.data_parallel_degree
                * 2  # 2 for K and V
                * max_kv_length
                * huggingface_config.num_attention_heads
                * huggingface_config.qk_nope_head_dim
                * cache_dtype.size_in_bytes
            )

        # Estimate buffer and activation memory during Expert Parallel MoE.
        if pipeline_config.runtime.ep_size > 1:
            n_gpus_per_node = len(pipeline_config.model.device_specs)

            ep_max_rank_send_tokens = self._ep_max_rank_send_tokens(
                pipeline_config
            )

            # Calculate the maximum number of tokens a rank may receive during
            # all-to-all routing. Each token selects top_k experts, and in the
            # worst case all selections land on one rank.
            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                huggingface_config.n_routed_experts,
                pipeline_config.runtime.ep_size
                * huggingface_config.num_experts_per_tok,
            )

            if pipeline_config.runtime.ep_use_allreduce:
                max_recv_tokens_per_rank = (
                    pipeline_config.runtime.max_batch_input_tokens
                    * min(
                        huggingface_config.n_routed_experts // n_gpus_per_node,
                        huggingface_config.num_experts_per_tok,
                    )
                )

            # The maximal activation memory usage happens at the second
            # grouped_matmul in the MoE layer. The input for that matmul would
            # of shape [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.moe_intermediate_size
                * supported_encoding_dtype(encoding).size_in_bytes
            )

            # The output would be of shape [max_recv_tokens_per_rank, hidden_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * huggingface_config.hidden_size
                * DType.bfloat16.size_in_bytes  # output is always bfloat16.
            )

            # Adding 256MB per GPU to account for misc items (e.g. FP8 scalars).
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            # EP SHMEM communication buffers are persistent (allocated once at
            # model init, not freed between layers).
            n_nodes = pipeline_config.runtime.ep_size // n_gpus_per_node

            ep_dispatch_dtype = supported_encoding_dtype(encoding)
            draft_ep_dtype = _get_mtp_draft_ep_dispatch_dtype(pipeline_config)
            if draft_ep_dtype is not None:
                ep_dispatch_dtype = draft_ep_dtype

            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=huggingface_config.hidden_size,
                dispatch_dtype=ep_dispatch_dtype,
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=huggingface_config.n_routed_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=huggingface_config.num_experts_per_tok,
                use_allreduce=pipeline_config.runtime.ep_use_allreduce,
            )
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node

            logger.info(
                "Estimated EP SHMEM buffer memory: "
                f"{to_human_readable_bytes(ep_buffer_memory)}"
            )

        # We only need to consider the maximum of the MLA and MoE activation
        # memories, because the MLA and MoE layers are executed sequentially.
        activation_memory = max(mla_activation_memory, moe_activation_memory)
        activation_memory += ep_buffer_memory

        if pipeline_config.runtime.device_graph_capture:
            graph_capture_headroom = (
                _GRAPH_CAPTURE_HEADROOM_BYTES_PER_DEVICE
                * len(pipeline_config.model.device_specs)
            )
            activation_memory += graph_capture_headroom
            logger.info(
                "Added graph capture headroom to activation memory: %s",
                to_human_readable_bytes(graph_capture_headroom),
            )

        if activation_memory != 0:
            logger.info(
                f"Estimated activation memory: {to_human_readable_bytes(activation_memory)}"
            )

        return activation_memory
