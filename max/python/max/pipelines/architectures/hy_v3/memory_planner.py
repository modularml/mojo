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

"""Memory planner for the Hunyuan-Video V3 architecture."""

from __future__ import annotations

from max.dtype import DType
from max.nn.comm.ep.ep_config import (
    calculate_ep_max_tokens_per_rank,
    estimate_ep_memory_usage,
)
from max.pipelines.architectures.hy_v3.model_config import (
    HYV3Config,
    hyv3_num_experts_from_config,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig


class HyV3MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for HY-V3 (Hunyuan) MoE models.

    Accounts for expert-parallel routing buffers.
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        encoding = _select_quantization_encoding(
            pipeline_config.model, HYV3Config.DEFAULT_ENCODING
        )
        n_gpus_per_node = len(pipeline_config.model.device_specs)
        # Use moe_intermediate_size for EP buffer math, not the
        # dense-layer intermediate_size (the latter is ~9x larger on
        # Hy3 and would over-reserve).
        num_experts = hyv3_num_experts_from_config(huggingface_config)
        moe_dim = int(huggingface_config.moe_intermediate_size)
        hidden_size = int(huggingface_config.hidden_size)
        top_k = int(huggingface_config.num_experts_per_tok)

        ep_buffer_memory = 0
        moe_activation_memory = 0
        ep_size = pipeline_config.runtime.ep_size
        if ep_size > 1:
            ep_max_rank_send_tokens = calculate_ep_max_tokens_per_rank(
                max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
                ep_size=ep_size,
                data_parallel_degree=pipeline_config.model.data_parallel_degree,
            )
            ep_dispatch_dtype = supported_encoding_dtype(encoding)

            max_recv_tokens_per_rank = ep_max_rank_send_tokens * min(
                num_experts,
                ep_size * top_k,
            )

            moe_activation_memory += (
                max_recv_tokens_per_rank
                * moe_dim
                * ep_dispatch_dtype.size_in_bytes
            )
            moe_activation_memory += (
                max_recv_tokens_per_rank
                * hidden_size
                * DType.bfloat16.size_in_bytes
            )
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
            )
            ep_buffer_memory = per_device_ep_memory * n_gpus_per_node * 2

        activation_memory = moe_activation_memory + ep_buffer_memory

        return activation_memory
