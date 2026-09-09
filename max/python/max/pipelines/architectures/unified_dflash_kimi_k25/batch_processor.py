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
"""Input batching for the unified DFlash Kimi K2.5 pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface

from ..kimik2_5.batch_processor import KimiK2_5BatchProcessor
from ..kimik2_5.context import KimiK2_5TextAndVisionContext

if TYPE_CHECKING:
    from .model import UnifiedDflashKimiK25Inputs


class UnifiedDflashKimiK25BatchProcessor(KimiK2_5BatchProcessor):
    """Extends KimiK2_5BatchProcessor with DFlash seed and draft token wiring.

    The parent class handles all vision-encoder and token-tensor preparation.
    This subclass wraps the resulting :class:`KimiK2_5ModelInputs` into the
    wider :class:`UnifiedDflashKimiK25Inputs` dataclass by appending a
    monotonically-increasing seed buffer.  ``draft_tokens`` are left *None*
    here and populated downstream by the speculative-decoding driver.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        """Return a monotonically-increasing 64-bit seed on device0."""
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.runtime.devices[0])

    def prepare_initial_token_inputs(  # type: ignore[override]
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        **kwargs: Any,
    ) -> UnifiedDflashKimiK25Inputs:
        """Build base KimiK2.5 inputs then wrap with DFlash draft fields.

        Args:
            replica_batches: Per-DP-rank context batches.
            kv_cache_inputs: KV cache state for this forward pass.
            return_n_logits: How many top-k logit positions to return.
            draft_tokens: Draft token buffer from the spec-decode driver,
                or *None* on the very first prefill step.
            **kwargs: Forwarded to the parent for forward-compatibility.

        Returns:
            :class:`UnifiedDflashKimiK25Inputs` ready for ``execute``.
        """
        base = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        from .model import UnifiedDflashKimiK25Inputs

        return UnifiedDflashKimiK25Inputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
            draft_tokens=draft_tokens,
            seed=self._next_seed(),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
