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
"""Input batching for Qwen3.5 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from max.driver import Buffer
from max.nn.kv_cache import KVCacheInputsInterface
from max.pipelines.architectures.llama3.batch_processor import (
    Llama3BatchProcessor,
)
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.context import TextContext

if TYPE_CHECKING:
    from .model import Qwen3_5Inputs
    from .state_cache import GatedDeltaNetStateCache


_MROPE_AXES = 3
"""Temporal, height and width, the three axes M-RoPE positions carry."""


class Qwen3_5BatchProcessor(Llama3BatchProcessor):
    """Ragged batching with linear-attention state pools and optional vision inputs."""

    _state_cache: GatedDeltaNetStateCache | None = None
    _slot_idx_prealloc: list[Buffer] | None = None
    _mrope_enabled: bool = False

    def bind_prepare_state(
        self,
        *,
        state_cache: GatedDeltaNetStateCache,
        slot_idx_prealloc: list[Buffer],
        mrope_enabled: bool = False,
    ) -> None:
        """Wires the state pools created during ``load_model``."""
        self._state_cache = state_cache
        self._slot_idx_prealloc = slot_idx_prealloc
        self._mrope_enabled = mrope_enabled

    def _decoder_position_ids(self, contexts: Sequence[TextContext]) -> Buffer:
        """Returns this step's ``[3, total_seq_len]`` M-RoPE positions.

        A prompt whose images are still to be encoded takes the slice of the
        positions the tokenizer precomputed for the whole prompt. Everything
        else -- decode steps, and continuations of a prompt whose images are
        already behind it -- counts on from the processed length, offset by
        the request's rope delta. Past the last image those two agree, which
        is what lets a decode step extend the corrected positions without
        recomputing them.
        """
        rows: list[npt.NDArray[np.int64]] = []
        for ctx in contexts:
            rope_delta = 0
            if isinstance(ctx, Qwen3VLTextAndVisionContext):
                precomputed = ctx.decoder_position_ids
                if ctx.needs_vision_encoding and precomputed.shape[1] == len(
                    ctx.tokens
                ):
                    rows.append(
                        precomputed[
                            :,
                            ctx.tokens.processed_length : ctx.tokens.current_position,
                        ]
                    )
                    continue
                rope_delta = ctx.rope_delta
            flat = np.arange(ctx.tokens.active_length, dtype=np.int64)
            rows.append(
                np.tile(flat, (_MROPE_AXES, 1))
                + ctx.tokens.processed_length
                + rope_delta
            )
        return Buffer.from_numpy(
            np.concatenate(rows, axis=1).astype(np.int64)
        ).to(self.runtime.devices[0])

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3_5Inputs:
        from .model import Qwen3_5Inputs

        base_inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

        all_contexts = [ctx for batch in replica_batches for ctx in batch]
        request_ids = [ctx.request_id for ctx in all_contexts]

        assert self._state_cache is not None, (
            "Qwen3.5 always has linear-attention layers; state cache must "
            "be initialised by load_model()"
        )
        assert self._slot_idx_prealloc is not None
        for rid in request_ids:
            self._state_cache.claim(rid)
        slot_idx = self._state_cache.slot_idx_for(
            request_ids, self._slot_idx_prealloc
        )
        conv_pools = self._state_cache.conv_pools
        recurrent_pools = self._state_cache.rec_pools

        # TODO(kevinbi): nothing between here and the model worker's main
        # loop catches this, so it ends the worker process rather than the
        # one request that asked for the impossible. Failing just the
        # request needs the scheduler's per-request path
        # (`SchedulerResult.failed`) to cover batch preparation, not only
        # batch construction.
        if not self._mrope_enabled and any(
            isinstance(ctx, Qwen3VLTextAndVisionContext)
            and ctx.needs_vision_encoding
            for ctx in all_contexts
        ):
            raise ValueError(
                "Qwen3.5 cannot serve image prompts for this checkpoint: "
                "it declares no vision config or no mrope_section, so "
                "M-RoPE positions are not wired into the compiled graph "
                "and every token after an image would get a flat position."
            )

        return Qwen3_5Inputs(
            tokens=base_inputs.tokens,
            input_row_offsets=base_inputs.input_row_offsets,
            signal_buffers=base_inputs.signal_buffers,
            kv_cache_inputs=base_inputs.kv_cache_inputs,
            return_n_logits=base_inputs.return_n_logits,
            slot_idx=slot_idx,
            conv_pools=conv_pools,
            recurrent_pools=recurrent_pools,
            request_ids=request_ids,
            decoder_position_ids=(
                self._decoder_position_ids(all_contexts)
                if self._mrope_enabled
                else None
            ),
        )
