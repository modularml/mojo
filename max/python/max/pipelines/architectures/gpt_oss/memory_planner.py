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

"""Memory planner for the GPT OSS architecture."""

from __future__ import annotations

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from transformers import AutoConfig

from .model_config import GptOssConfig


class GptOssMemoryPlanner(PagedMemoryPlanner):
    """Memory planner for GPT OSS MoE models.

    Adds a fixed 6 GiB base reservation plus an optional MXFP4 dequant buffer
    to work around MemoryManager fragmentation (see GEX-3248).
    """

    _always_signal_buffers = True

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Estimates activation memory for GPT OSS models.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated activation memory in bytes.
        """
        # FIXME GEX-3248: This is a workaround for a MemoryManager
        # fragmentation issue. In #77700 we swapped the order of model weight
        # loading and KV cache loading. This affected memory fragmentation and
        # led to CUDA OOM when running
        # `br smoke-test -- unsloth/gpt-oss-20b-BF16` on 1xH100.
        # We reduce the KV cache size slightly to avoid this.
        base = 6 * 1024 * 1024 * 1024  # 6 GiB

        # MXFP4 dequant materializes full BF16 weight buffers on GPU.
        # 3 projections (gate, up, down), each num_experts * hidden * moe_dim
        # at 2 bytes (BF16). The extra 15 GiB covers compilation workspace
        # and memory fragmentation.
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, GptOssConfig.DEFAULT_ENCODING
        )
        if quantization_encoding == "float4_e2m1fnx2":
            num_experts = getattr(huggingface_config, "num_local_experts", 32)
            moe_dim = getattr(huggingface_config, "intermediate_size", 2880)
            hidden_size = getattr(huggingface_config, "hidden_size", 2880)
            # 3 projections (gate, up, down) * 2 bytes per BF16 element.
            dequant_bytes = num_experts * hidden_size * 3 * moe_dim * 2
            base += dequant_bytes + 15 * 1024 * 1024 * 1024

        return base
