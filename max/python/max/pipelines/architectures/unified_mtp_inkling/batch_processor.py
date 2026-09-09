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
"""Input batching for unified Inkling MTP."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer
from max.engine import Model
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime

from ..inkling.batch_processor import InklingBatchProcessor
from ..inkling.state_cache import InklingConvStateCache

if TYPE_CHECKING:
    from .model import UnifiedMTPInklingInputs


class UnifiedMTPInklingBatchProcessor(InklingBatchProcessor):
    """Ragged batching for unified Inkling MTP.

    Adds a host copy of the row offsets and the draft conv pools;
    ``draft_tokens`` is left unset for the caller to fill.
    """

    def __init__(
        self, config: ArchConfig, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._draft_state_cache: InklingConvStateCache | None = None
        self._draft_conv_pools: list[Buffer] = []

    def bind_runtime_state(
        self,
        state_cache: InklingConvStateCache | None,
        vision_model: Model,
        draft_state_cache: InklingConvStateCache | None = None,
    ) -> None:
        super().bind_runtime_state(state_cache, vision_model)
        self._draft_state_cache = draft_state_cache
        if draft_state_cache is None:
            self._draft_conv_pools = []
            return
        self._draft_conv_pools = [
            pool
            for device_idx in range(len(self.runtime.devices))
            for pool in draft_state_cache.pools(device_idx)
        ]

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPInklingInputs:
        from .model import UnifiedMTPInklingInputs

        base = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        draft_cache = self._draft_state_cache
        assert draft_cache is not None
        for context in context_batch:
            draft_cache.claim(context.request_id)

        lengths = np.fromiter(
            (context.tokens.active_length for context in context_batch),
            dtype=np.int64,
            count=len(context_batch),
        )
        host_input_row_offsets = Buffer.from_numpy(
            np.cumsum([0, *lengths], dtype=np.uint32)
        )
        return UnifiedMTPInklingInputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            positions=base.positions,
            return_n_logits=base.return_n_logits,
            host_input_row_offsets=host_input_row_offsets,
            image_embeddings=base.image_embeddings,
            image_indices=base.image_indices,
            signal_buffers=base.signal_buffers,
            slot_idx=base.slot_idx,
            conv_pools=base.conv_pools,
            draft_conv_pools=self._draft_conv_pools,
            kv_cache_inputs=base.kv_cache_inputs,
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
