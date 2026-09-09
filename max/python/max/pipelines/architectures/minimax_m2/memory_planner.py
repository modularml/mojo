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

"""Memory planner for the MiniMax M2 architecture."""

from __future__ import annotations

import logging

from max.dtype import DType
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from max.support.human_readable_formatter import to_human_readable_bytes
from transformers import AutoConfig

from .model_config import MiniMaxM2Config

logger = logging.getLogger(__name__)


class MiniMaxM2MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for MiniMax M2 MoE models.

    Accounts for expert-parallel routing buffers (with double-buffering).
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        encoding = _select_quantization_encoding(
            pipeline_config.model, MiniMaxM2Config.DEFAULT_ENCODING
        )
        n_gpus_per_node = len(pipeline_config.model.device_specs)
        num_experts = getattr(huggingface_config, "num_local_experts", 256)
        moe_dim = getattr(huggingface_config, "intermediate_size", 1536)
        hidden_size = getattr(huggingface_config, "hidden_size", 3072)
        top_k = getattr(huggingface_config, "num_experts_per_tok", 8)

        ep_buffer_memory = 0
        moe_activation_memory = 0
        ep_size = pipeline_config.runtime.ep_size
        ep_use_allreduce = pipeline_config.runtime.ep_use_allreduce
        if ep_size > 1 and encoding is not None:
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
                use_allreduce=ep_use_allreduce,
            )
            ep_dispatch_dtype = supported_encoding_dtype(encoding)

            # Worst-case tokens received per rank during all-to-all routing.
            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                num_experts,
                ep_size * top_k,
            )

            # Peak MoE activation: input to second grouped_matmul has shape
            # [max_recv_tokens_per_rank, moe_intermediate_size].
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * moe_dim
                * ep_dispatch_dtype.size_in_bytes
            )
            # Output has shape [max_recv_tokens_per_rank, hidden_size] in
            # bfloat16.
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * hidden_size
                * DType.bfloat16.size_in_bytes
            )
            # 256MB per GPU for misc scalar buffers.
            moe_activation_memory += 256 * 1024 * 1024
            moe_activation_memory *= n_gpus_per_node

            n_nodes = max(ep_size // n_gpus_per_node, 1)
            per_device_ep_memory = estimate_ep_memory_usage(
                hidden_size=hidden_size,
                dispatch_dtype=ep_dispatch_dtype,
                combine_dtype=DType.bfloat16,
                max_tokens_per_rank=ep_max_rank_send_tokens,
                n_experts=num_experts,
                n_nodes=n_nodes,
                n_gpus_per_node=n_gpus_per_node,
                top_k=top_k,
                use_allreduce=ep_use_allreduce,
            )
            # EPCommInitializer double-buffers (NUM_GROUPS=2) the SHMEM
            # dispatch/combine buffers.
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node * 2

        activation_memory = moe_activation_memory + ep_buffer_memory

        if activation_memory != 0:
            logger.info(
                "Estimated activation memory: %s "
                "(ep_buffers=%s, moe_activation=%s)",
                to_human_readable_bytes(activation_memory),
                to_human_readable_bytes(ep_buffer_memory),
                to_human_readable_bytes(moe_activation_memory),
            )

        return activation_memory
