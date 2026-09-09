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
"""Input batching for DeepseekV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from max.driver import Buffer, DevicePinnedBuffer
from max.dtype import DType
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3EpBatchProcessor,
)
from max.pipelines.context import TextContext
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.support.algorithm import flatten2d

if TYPE_CHECKING:
    from .model import DeepseekV3Inputs


class DeepseekV3BatchProcessor(Llama3EpBatchProcessor):
    """Ragged batching for DeepseekV3 with MLA context lengths and EP MoE.

    Extends :class:`Llama3EpBatchProcessor` with:

    - Per-device preallocated ``batch_context_lengths`` buffers that track
      page-aligned KV context length for MLA prefill.
    - Unconditional ``host_input_row_offsets`` emission (required for MLA
      latent decode regardless of data-parallel degree).
    - Unconditional ``data_parallel_splits`` emission (always present in the
      DeepseekV3 graph ABI, even when DP == 1).
    """

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        self._batch_context_lengths = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(runtime.devices))
        ]
        self._kv_cache_page_size = (
            runtime.pipeline_config.model.kv_cache.kv_cache_page_size
        )

    def _host_input_row_offsets_for_dp(
        self, host_row_offsets: Buffer, dp: int
    ) -> Buffer | None:
        # DeepseekV3 always emits host offsets (needed for MLA latent decode,
        # not only for DP); override the parent which returns None when dp <= 1.
        return host_row_offsets

    def _update_batch_context_lengths(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
    ) -> None:
        """Update per-device batch context lengths for MLA prefill.

        Computes the page-aligned sum of ``current_position`` for each DP
        replica and writes it into the preallocated CPU buffers in-place.
        When DP < num_devices (i.e. TP only), broadcasts replica 0's value
        to all remaining device slots.

        Args:
            replica_batches: Batches of text contexts per DP replica.
        """
        dp = self.runtime.pipeline_config.model.data_parallel_degree
        page_size = self._kv_cache_page_size

        def align_length(length: int) -> int:
            return (length + page_size - 1) // page_size * page_size

        for i, batch in enumerate(replica_batches):
            curr_length = sum(
                align_length(ctx.tokens.current_position) for ctx in batch
            )
            self._batch_context_lengths[i][0] = curr_length

        if dp != len(self.runtime.devices):
            assert dp == 1
            for dev_idx in range(1, len(self.runtime.devices)):
                self._batch_context_lengths[dev_idx][0] = (
                    self._batch_context_lengths[0][0].item()
                )

    def prepare_initial_token_inputs(  # type: ignore[override]
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> DeepseekV3Inputs:
        """Prepare batch inputs for a DeepseekV3 forward pass.

        Args:
            replica_batches: One inner list per DP replica containing the
                :class:`TextContext` objects for that shard.
            kv_cache_inputs: Optional KV cache inputs (may be ``None`` during
                compilation warm-up).
            return_n_logits: Number of per-token logit rows to return.

        Returns:
            :class:`DeepseekV3Inputs` with tokens, row offsets, host offsets,
            batch context lengths, DP splits, signal buffers, and EP buffers.
        """
        from .model import DeepseekV3Inputs

        dp = self.runtime.pipeline_config.model.data_parallel_degree
        if len(replica_batches) != dp:
            raise ValueError(
                "Number of replica batches must match data parallel degree"
            )

        device0 = self.runtime.devices[0]
        pinned = not device0.is_host

        if self.runtime.pipeline_config.runtime.pipeline_role != "decode_only":
            self._update_batch_context_lengths(replica_batches)

        context_batch = flatten2d(replica_batches)

        tokens: Buffer
        device_input_row_offsets: Buffer
        host_input_row_offsets: Buffer

        if len(context_batch) == 0:
            if pinned:
                tokens = DevicePinnedBuffer(
                    shape=[0], dtype=DType.int64, device=device0
                )
            else:
                tokens = Buffer(shape=[0], dtype=DType.int64, device=device0)
            host_input_row_offsets = Buffer.zeros(shape=[1], dtype=DType.uint32)
            if pinned:
                pinned_offsets: Buffer = DevicePinnedBuffer.zeros(
                    shape=[1], dtype=DType.uint32, device=device0
                )
            else:
                pinned_offsets = Buffer.zeros(
                    shape=[1], dtype=DType.uint32, device=device0
                )
            device_input_row_offsets = pinned_offsets.to(device0)
        else:
            num_tokens = sum(ctx.tokens.active_length for ctx in context_batch)
            if pinned:
                tokens_host: Buffer = DevicePinnedBuffer(
                    shape=(num_tokens,), dtype=DType.int64, device=device0
                )
            else:
                tokens_host = Buffer(
                    shape=(num_tokens,), dtype=DType.int64, device=device0
                )
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=tokens_host.to_numpy(),
            )
            tokens = tokens_host.to(device0)

            input_row_offsets_np = np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )

            # FIXME GEX-3121: Using a pinned buffer as a graph CPU input
            # triggers a device-mismatch error.  Use a separate non-pinned
            # CPU buffer for the graph input slot.
            host_input_row_offsets = Buffer(
                shape=(len(context_batch) + 1,), dtype=DType.uint32
            )
            host_input_row_offsets.to_numpy()[:] = input_row_offsets_np[:]

            if pinned:
                pinned_offsets = DevicePinnedBuffer(
                    shape=(len(context_batch) + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            else:
                pinned_offsets = Buffer(
                    shape=(len(context_batch) + 1,),
                    dtype=DType.uint32,
                    device=device0,
                )
            pinned_offsets.to_numpy()[:] = input_row_offsets_np[:]
            device_input_row_offsets = pinned_offsets.to(device0)

        data_parallel_splits = Buffer.from_numpy(
            compute_data_parallel_splits(replica_batches)
        )

        return DeepseekV3Inputs(
            tokens=tokens,
            input_row_offsets=device_input_row_offsets,
            host_input_row_offsets=host_input_row_offsets,
            batch_context_lengths=self._batch_context_lengths,
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            data_parallel_splits=data_parallel_splits,
            ep_inputs=self._ep_inputs(),
        )
