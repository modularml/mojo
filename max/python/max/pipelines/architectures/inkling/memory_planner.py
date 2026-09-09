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
"""Memory planner for the Inkling architecture."""

from __future__ import annotations

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.memory_estimation import _DEFAULT_BATCH_SIZE
from transformers import AutoConfig

from .model_config import InklingConfig
from .state_cache import InklingConvStateLayout


class InklingMemoryPlanner(PagedMemoryPlanner):
    """Memory planner that reserves Inkling's short-convolution slot pool."""

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserves the convolution state pool, summed across devices."""
        del huggingface_config
        config = self._config
        assert isinstance(config, InklingConfig)
        layout = InklingConvStateLayout.from_config(config.text_config)
        # Runs before the estimator resolves the batch size, so fall back to
        # the value it will infer.
        max_batch_size = (
            pipeline_config.runtime.max_batch_size or _DEFAULT_BATCH_SIZE
        )
        return max_batch_size * layout.bytes_per_request()
