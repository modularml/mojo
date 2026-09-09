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
"""Input batching for Step-3.5 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
    Llama3EpBatchProcessor,
)
from max.pipelines.context import TextContext

from .step3p5 import ParallelismMode

if TYPE_CHECKING:
    from ..llama3.model import Llama3Inputs
    from .model import Step3p5Inputs


class Step3p5BatchProcessor(Llama3EpBatchProcessor):
    """Ragged batching for TP, TP+EP, and DP+EP Step-3.5 parallelism modes."""

    _mode: ParallelismMode | None = None

    def bind_parallelism_mode(self, mode: ParallelismMode) -> None:
        """Wires the mode selected during graph build."""
        self._mode = mode

    def _host_input_row_offsets_for_dp(
        self, host_row_offsets: Buffer, dp: int
    ) -> Buffer | None:
        if dp <= 1:
            return None
        return Buffer.from_numpy(host_row_offsets.to_numpy().copy())

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs | Step3p5Inputs:
        from .model import Step3p5Inputs

        assert self._mode is not None, (
            "Step3p5 parallelism mode must be bound before prepare_initial_token_inputs()"
        )

        if self._mode == ParallelismMode.TP_TP:
            return Llama3BatchProcessor.prepare_initial_token_inputs(
                self,
                replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )

        ep_inputs = self._ep_inputs()

        if self._mode == ParallelismMode.TP_EP:
            base = Llama3BatchProcessor.prepare_initial_token_inputs(
                self,
                replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )
            return Step3p5Inputs(
                tokens=base.tokens,
                input_row_offsets=base.input_row_offsets,
                return_n_logits=base.return_n_logits,
                signal_buffers=base.signal_buffers,
                kv_cache_inputs=base.kv_cache_inputs,
                host_input_row_offsets=None,
                data_parallel_splits=None,
                ep_inputs=ep_inputs,
            )

        (
            device_tokens,
            device_row_offsets,
            return_n_logits_tensor,
            data_parallel_splits,
            ep_inputs,
            host_input_row_offsets,
        ) = self._prepare_ep_moe_token_inputs(replica_batches, return_n_logits)

        return Step3p5Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_tensor,
            host_input_row_offsets=host_input_row_offsets,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            ep_inputs=ep_inputs,
        )
