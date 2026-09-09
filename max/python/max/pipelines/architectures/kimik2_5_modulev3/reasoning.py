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

"""Kimi K2.5 reasoning parser for <think>...</think> sections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from max.pipelines.lib.reasoning import register
from max.pipelines.lib.tokenizer import convert_token_to_id
from max.pipelines.modeling.types import (
    ParsedReasoningDelta,
    PipelineTokenizer,
    ReasoningParser,
    ReasoningSpan,
)


@register("kimik2_5")
class KimiK2_5ReasoningParser(ReasoningParser):
    """Kimi K2.5 reasoning parser for <think>...</think> sections.

    Per Moonshot's "Interleaved Thinking" design (see
    https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model and
    https://huggingface.co/moonshotai/Kimi-K2.5), a single assistant turn
    can interleave multiple ``<think>...</think>`` blocks with
    ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>`` blocks.

    A reasoning span ends on ``</think>`` or ``<|tool_calls_section_begin|>``;
    the model may open the tool-call section directly from inside the
    prefilled ``<think>`` block without a closing ``</think>``. The section
    marker is left as content rather than consumed as a delimiter, so the
    tool parser (which only sees content) receives the whole section.

    Reasoning may begin implicitly, without an explicit ``<think>``
    token, when the chat template prefilled the assistant turn already
    inside a thinking block.

    Reasoning can be disabled through the chat template by including a
    ``</think>`` token at the end of the prompt; this is detected by
    :meth:`will_reason_after_prompt`.
    """

    REASONING_START: ClassVar[str] = "<think>"
    REASONING_END: ClassVar[str] = "</think>"

    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
        tool_section_start_token_id: int | None = None,
    ) -> None:
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id
        # ``<|tool_calls_section_begin|>`` implicitly ends reasoning when the
        # model skips the closing ``</think>``. Kimi tool calls always open
        # with this section marker (the inner ``<|tool_call_begin|>`` only
        # ever appears inside a section), so it is the single tool terminator.
        self.tool_section_start_token_id = tool_section_start_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identify a reasoning span within a streaming delta chunk.

        When ``is_currently_reasoning=False`` and the chunk contains no
        ``<think>`` opener, returns an empty span so non-reasoning chunks
        (turns where the chat template prefilled ``</think>``, or any
        chunk after reasoning ended in a prior chunk) aren't misclassified
        as reasoning.
        """
        # Reasoning ends on ``</think>`` or, when the model skips it, on the
        # tool-call section opener. Whether the end delimiter is consumed
        # (``</think>``) or kept as content (the section marker, which the
        # tool parser needs) is decided after the loop.
        end_token_ids = (
            (self.think_end_token_id, self.tool_section_start_token_id)
            if self.tool_section_start_token_id is not None
            else (self.think_end_token_id,)
        )

        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.think_start_token_id
            ):
                # Take the earliest start token
                start_token_idx = i
            elif token_id in end_token_ids:
                # Only end on an active span — either pre-seeded via
                # ``is_currently_reasoning`` or opened by a ``<think>``
                # earlier in this chunk — so a stray marker from prior
                # content doesn't pull content into the reasoning region.
                if is_currently_reasoning or start_token_idx is not None:
                    end_token_idx = i
                    break

        if start_token_idx is None and not is_currently_reasoning:
            # No reasoning section in this chunk and we weren't already
            # inside one — empty span, all tokens are content.
            empty_span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0),
                reasoning=(0, 0),
            )
            return ParsedReasoningDelta(
                span=empty_span,
                is_still_reasoning=False,
            )

        if start_token_idx is None:
            start_reasoning = 0
            start_reasoning_with_delimiters = 0
        else:
            start_reasoning = start_token_idx + 1
            start_reasoning_with_delimiters = start_token_idx

        if end_token_idx is None:
            end_reasoning = len(delta_token_ids)
            end_reasoning_with_delimiters = len(delta_token_ids)
        elif delta_token_ids[end_token_idx] == self.think_end_token_id:
            # ``</think>`` is consumed as a delimiter (dropped from output).
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # A tool-call marker is not consumed — it stays in the content
            # region so the downstream tool parser sees the whole section.
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx

        span = ReasoningSpan(
            reasoning_with_delimiters=(
                start_reasoning_with_delimiters,
                end_reasoning_with_delimiters,
            ),
            reasoning=(start_reasoning, end_reasoning),
        )
        is_still_reasoning = end_token_idx is None
        return ParsedReasoningDelta(
            span=span,
            is_still_reasoning=is_still_reasoning,
        )

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Predicts whether the model will emit reasoning after this prompt.

        Kimi K2.5 chat templates emit ``<think>`` to open the new assistant
        turn's reasoning section, and ``</think>`` to close the prior
        assistant turn's reasoning section.

        Scan right-to-left and return based on the first delimiter seen:

        * ``<think>`` → reasoning is currently open → ``True``.
        * ``</think>`` (or ``<|tool_calls_section_begin|>``) → reasoning is
          currently closed → ``False``.
        * No delimiters at all → reasoning is not in use → ``False``.

        Uses the same end-of-reasoning delimiters as :meth:`stream` so both
        agree on where reasoning ends.
        """
        end_token_ids = (
            (self.think_end_token_id, self.tool_section_start_token_id)
            if self.tool_section_start_token_id is not None
            else (self.think_end_token_id,)
        )
        for token_id in reversed(prompt_token_ids):
            if token_id == self.think_start_token_id:
                return True
            if token_id in end_token_ids:
                return False
        return False

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> KimiK2_5ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        think_start_id = await convert_token_to_id(
            tokenizer, cls.REASONING_START
        )
        think_end_id = await convert_token_to_id(tokenizer, cls.REASONING_END)

        if think_start_id is None or think_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate think start/end tokens in the tokenizer"
            )

        tool_section_start_id = await convert_token_to_id(
            tokenizer, "<|tool_calls_section_begin|>"
        )

        return cls(
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
            tool_section_start_token_id=tool_section_start_id,
        )

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``</think>`` token id."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
