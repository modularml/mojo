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
"""Input batching for the unified speculators-DSpark Gemma4 pipeline model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    UnifiedSpecDecodeBatchProcessor,
)

if TYPE_CHECKING:
    from .model import UnifiedDSparkGemma4_31BInputs
    from .model_config import UnifiedDSparkGemma4_31BConfig


class UnifiedDSparkGemma4_31BBatchProcessor(
    UnifiedSpecDecodeBatchProcessor["UnifiedDSparkGemma4_31BInputs"]
):
    """Ragged batching with persistent buffers and seed for DSpark Gemma4.

    Identical to the DFlash batching except that the graph binds signal
    buffers (Gemma4's embedding/lm_head layers use collectives even on a
    single device), sourced from the model's always-on allocation.
    """

    def __init__(
        self,
        config: UnifiedDSparkGemma4_31BConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> UnifiedDSparkGemma4_31BInputs:
        from .model import UnifiedDSparkGemma4_31BInputs

        return UnifiedDSparkGemma4_31BInputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
        )
