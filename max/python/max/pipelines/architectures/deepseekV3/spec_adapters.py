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
"""Speculative-decoding adapters for a DeepseekV3-family target.

The driver in :mod:`max.pipelines.speculative.driver` owns the seven phases of
a sequential spec-decode iteration. What is left per model is the pair below:
how to call the target and read its output tuple, and how to call the draft.

Both DeepseekV3 speculators: MTP (NextN head in the target
checkpoint) and Eagle3 (separate draft checkpoint) -- share the same draft call
signature, so they share :class:`DeepseekV3MLAProposer` and differ only in the
class attributes at the top of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from max.graph import BufferType, DimLike, TensorType, TensorValue, Value
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative.driver import (
    DecodeKVSwap,
    DraftStepInput,
    Proposed,
    SequentialBatch,
)
from max.pipelines.speculative.spec_target import Verified

from .deepseekV3 import DeepseekV3
from .model_config import DeepseekV3Config

__all__ = ["DeepseekV3MLAProposer", "DeepseekV3Target", "MLADraft"]


class MLADraft(Protocol):
    """The draft-side surface :class:`DeepseekV3MLAProposer` drives."""

    return_hidden_states: ReturnHiddenStates
    return_logits: ReturnLogits

    @property
    def config(self) -> DeepseekV3Config: ...

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[TensorValue, ...]: ...


class DeepseekV3Target:
    """The DeepseekV3 target entry point and its output layout."""

    def __init__(self, target: DeepseekV3) -> None:
        self.target = target

    def verify(self, batch: SequentialBatch) -> Verified:
        outputs = self.target(
            batch.merged_tokens,
            batch.signal_buffers,
            batch.kv_collections,
            batch.return_n_logits,
            batch.merged_offsets_per_dev,
            batch.host_merged_offsets,
            batch.data_parallel_splits,
            batch.batch_context_lengths,
            batch.ep_inputs,
        )
        # DeepseekV3 has no ``emit_last_token_logits`` attribute, so the tuple
        # is [last_logits, logits, offsets, hidden per device...].
        return Verified(
            logits=outputs[1],
            hidden=list(outputs[3 : 3 + batch.n_devs]),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        if self.target.ep_manager is None:
            return ()
        return self.target.ep_manager.input_types()


class DeepseekV3MLAProposer:
    """Shared body of the DeepseekV3-family sequential drafts."""

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
        DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS,
    )
    split_prefix: str
    carry_dim_prefix: str
    step_hidden_mode: ReturnHiddenStates
    uses_thinking_phase: bool = False
    draft_takes_ep_inputs: bool = False
    """Whether the draft's ``__call__`` accepts the target's EP inputs."""

    def __init__(self, draft: MLADraft) -> None:
        self.draft = draft
        self.hidden_dim: DimLike = draft.config.hidden_size

    def _ep(
        self, batch: SequentialBatch
    ) -> tuple[list[Value[Any]] | None, ...]:
        return (batch.ep_inputs,) if self.draft_takes_ep_inputs else ()

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        # Step 0 always uses ALL hidden states (for per-batch-element gather
        # at accepted positions) + VARIABLE logits (for draft argmax).
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        outputs = self.draft(
            tokens,
            target_hidden,
            batch.signal_buffers,  # reuse target signal buffers for draft
            batch.draft_kv_collections,
            batch.return_n_logits,
            batch.merged_offsets_per_dev,
            batch.host_merged_offsets,
            batch.data_parallel_splits,
            batch.batch_context_lengths,
            *self._ep(batch),
        )
        self.draft.return_hidden_states = self.step_hidden_mode
        self.draft.return_logits = ReturnLogits.LAST_TOKEN
        return Proposed(
            logits=outputs[1],
            hidden=list(outputs[3 : 3 + batch.n_devs]),
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        outputs = self.draft(
            draft_input.tokens,
            draft_input.hidden,
            batch.signal_buffers,
            batch.draft_kv_collections,
            batch.return_n_logits,
            batch.merged_offsets_per_dev,
            batch.host_merged_offsets,
            batch.data_parallel_splits,
            batch.batch_context_lengths,
            *self._ep(batch),
            split_prefix=f"{self.split_prefix}_draft_step{index}",
        )
        return Proposed(
            logits=outputs[0],
            hidden=list(outputs[1 : 1 + batch.n_devs]),
        )
