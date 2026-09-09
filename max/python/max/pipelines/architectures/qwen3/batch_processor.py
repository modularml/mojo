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
"""Input batching for Qwen3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3EpBatchProcessor,
)
from max.pipelines.context import TextContext

if TYPE_CHECKING:
    from ..llama3.model import Llama3Inputs
    from .model import Qwen3Inputs


class Qwen3BatchProcessor(Llama3EpBatchProcessor):
    """Ragged batching with optional DP attention and EP MoE for Qwen3."""

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
    ) -> Llama3Inputs | Qwen3Inputs:
        from .model import Qwen3Inputs

        dp = self.runtime.pipeline_config.model.data_parallel_degree
        if dp <= 1:
            return super().prepare_initial_token_inputs(
                replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )

        (
            device_tokens,
            device_row_offsets,
            return_n_logits_tensor,
            data_parallel_splits,
            ep_inputs,
            host_input_row_offsets,
        ) = self._prepare_ep_moe_token_inputs(replica_batches, return_n_logits)

        return Qwen3Inputs(
            tokens=device_tokens,
            input_row_offsets=device_row_offsets,
            return_n_logits=return_n_logits_tensor,
            host_input_row_offsets=host_input_row_offsets,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            ep_inputs=ep_inputs,
        )
