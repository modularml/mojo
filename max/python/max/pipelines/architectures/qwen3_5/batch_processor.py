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


_MROPE_AXES = 3
"""Temporal, height and width, the three axes M-RoPE positions carry."""


class Qwen3_5BatchProcessor(Llama3BatchProcessor):
    """Ragged batching with linear-attention state pools and optional vision inputs."""

    mrope_enabled: bool = False
    """Whether the compiled graph takes M-RoPE positions. Set by the model."""

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

        # TODO(kevinbi): nothing between here and the model worker's main
        # loop catches this, so it ends the worker process rather than the
        # one request that asked for the impossible. Failing just the
        # request needs the scheduler's per-request path
        # (`SchedulerResult.failed`) to cover batch preparation, not only
        # batch construction.
        if not self.mrope_enabled and any(
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
            request_ids=request_ids,
            decoder_position_ids=(
                self._decoder_position_ids(all_contexts)
                if self.mrope_enabled
                else None
            ),
        )
