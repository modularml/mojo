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

"""Memory planner for the DeepseekV3-2 architecture."""

from __future__ import annotations

from max.pipelines.lib.config import PipelineConfig

from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner


class DeepseekV3_2MemoryPlanner(DeepseekV3MemoryPlanner):
    """Memory planner for DeepseekV3-2 models.

    Inherits all estimation logic from :class:`DeepseekV3MemoryPlanner` but
    overrides EP token sizing: DeepseekV3-2 holds full-length activations
    before EP MoE (no ring-scatter like V3 TP+EP), so the per-rank token
    budget is simply ``max_batch_input_tokens`` rather than the result of
    ``calculate_ep_max_tokens_per_rank``.
    """

    def _ep_max_rank_send_tokens(self, pipeline_config: PipelineConfig) -> int:
        """Each rank holds full-length activations before EP MoE."""
        return pipeline_config.runtime.max_batch_input_tokens
