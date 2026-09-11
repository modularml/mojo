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

"""Memory planner for the Qwen3.5 (GatedDeltaNet) architecture."""

from __future__ import annotations

from max.driver import Device
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib.config import PipelineConfig

from .model_config import Qwen3_5Config


class Qwen3_5MemoryPlanner(PagedMemoryPlanner):
    """Memory planner for Qwen3.5 GatedDeltaNet SSM models.

    The recurrent state occupies pages of the KV pool, so it is reserved
    nowhere here. Only the batch-size inference differs from the base.
    """

    _always_signal_buffers = True

    def shadow_state_bytes(self) -> int:
        """Per-request bytes in a *second* copy of the state pools, or 0.

        An unspeculated arch allocates one pool set, so nothing here. A
        speculative one verifies on a shadow set and must price it: the bytes
        are per request like the live pools, and they are invisible to the
        reconciliation assert in ``load_model`` because the shadow is
        allocated after it runs.
        """
        return 0

    def infer_max_batch_size(
        self,
        pipeline_config: PipelineConfig,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        """Infers a memory-safe default ``max_batch_size``.

        The base default counts only KV, so it promises concurrency the pool
        cannot also hold a recurrent state per request for.
        """
        assert isinstance(self._config, Qwen3_5Config)
        inferred = self._config.infer_optimal_batch_size(
            devices,
            weights_size=weights_size,
            device_memory_utilization=(
                pipeline_config.model.kv_cache.device_memory_utilization
            ),
            extra_per_request_bytes=self.shadow_state_bytes(),
        )
        return inferred
