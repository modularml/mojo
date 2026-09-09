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
"""Memory planner for unified Inkling MTP: target plus draft conv pools."""

from __future__ import annotations

from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.memory_estimation import _DEFAULT_BATCH_SIZE
from transformers import AutoConfig

from ..inkling.memory_planner import InklingMemoryPlanner
from ..inkling.model_config import InklingConfig, parse_inkling_mtp_config
from ..inkling.state_cache import InklingConvStateLayout


class UnifiedMTPInklingMemoryPlanner(InklingMemoryPlanner):
    """Reserves target conv pools plus one row of sites per MTP depth."""

    def estimate_activation_memory(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        target_bytes = super().estimate_activation_memory(
            pipeline_config, huggingface_config
        )
        config = self._config
        assert isinstance(config, InklingConfig)
        mtp = config.mtp or parse_inkling_mtp_config(huggingface_config)
        if mtp is None:
            return target_bytes
        spec = pipeline_config.speculative
        assert spec is not None
        n_depths = mtp.num_depths_for(spec)
        layout = InklingConvStateLayout.from_local_flags(
            config.text_config, mtp.local_flags(n_depths)
        )
        max_batch_size = (
            pipeline_config.runtime.max_batch_size or _DEFAULT_BATCH_SIZE
        )
        return target_bytes + max_batch_size * layout.bytes_per_request()
