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
"""Input batching for the unified MTP GLM-5.2 pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessor,
)
from max.pipelines.context import TextContext

if TYPE_CHECKING:
    from .model import UnifiedMTPGlm5_2Inputs


class UnifiedMTPGlm5_2BatchProcessor(DeepseekV3BatchProcessor):
    """Ragged batching for unified MTP GLM-5.2.

    Extends :class:`DeepseekV3BatchProcessor` to return
    :class:`UnifiedMTPGlm5_2Inputs` which carries the extra ``draft_tokens``
    slot required for speculative decoding. The slot is left ``None`` here; the
    overlap pipeline fills it in after this call returns.
    """

    def prepare_initial_token_inputs(  # type: ignore[override]
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedMTPGlm5_2Inputs:
        """Prepare batch inputs for a UnifiedMTPGlm5_2 forward pass.

        Delegates to :class:`DeepseekV3BatchProcessor` for the standard token
        and KV inputs, then wraps the result in
        :class:`UnifiedMTPGlm5_2Inputs`. ``draft_tokens`` is left as ``None``;
        the overlap pipeline assigns it after this method returns.

        Args:
            replica_batches: One inner list per DP replica containing the
                :class:`TextContext` objects for that shard.
            kv_cache_inputs: Optional KV cache inputs (may be ``None`` during
                compilation warm-up).
            return_n_logits: Number of per-token logit rows to return.

        Returns:
            :class:`UnifiedMTPGlm5_2Inputs` with all standard fields populated
            and ``draft_tokens=None``.
        """
        from .model import UnifiedMTPGlm5_2Inputs

        base = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

        return UnifiedMTPGlm5_2Inputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
            draft_tokens=None,
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )
