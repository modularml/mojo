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
"""Memory planning for the fused DFlash2 graph.

The recurrent-state reservation is the target's, unchanged -- the drafter adds
no state pool of its own, only a KV leaf the planner already prices through
the KV tree. Two things do differ from the base Qwen3.5 planner, and both are
handled here rather than in it: the arch config is a wrapper holding the
target and the drafter side by side, and the drafter's weights come from a
second checkpoint that ``pipeline_config.model`` does not describe.
"""

from __future__ import annotations

from typing import Any

from max.pipelines.lib.config import PipelineConfig
from typing_extensions import override

from ..qwen3_5.memory_planner import Qwen3_5MemoryPlanner
from .model_config import UnifiedDflash2Qwen3_5Config


class UnifiedDflash2Qwen3_5MemoryPlanner(Qwen3_5MemoryPlanner):
    """The Qwen3.5 planner, driven by the fused config's target half."""

    def __init__(self, config: Any) -> None:
        assert isinstance(config, UnifiedDflash2Qwen3_5Config)
        super().__init__(config.target)

    @override
    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Counts the drafter's checkpoint alongside the target's.

        Both live on the same device, and the drafter is ~4 GB against a 27B
        target -- small, but omitting it makes an inferred ``max_batch_size``
        optimistic by exactly the amount the pool cannot then have.
        """
        total = super().estimate_weights_size(pipeline_config)
        assert isinstance(pipeline_config, PipelineConfig)
        if pipeline_config.draft_model is not None:
            total += pipeline_config.draft_model.weights_size()
        return total
