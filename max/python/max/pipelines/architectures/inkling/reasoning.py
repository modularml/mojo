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

"""Inkling reasoning parser for ``<|content_thinking|>...<|end_message|>`` sections."""

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


@register("inkling")
class InklingReasoningParser(ReasoningParser):
    """Inkling reasoning parser for ``<|content_thinking|>...<|end_message|>``.

    Inkling frames every message as
    ``<role_token>[name]<content_type_token>...<|end_message|>``. The
    generation prompt ends at ``<|message_model|>`` with no content-type
    marker, so the model picks the content type as its first generated
    token and reasoning opens only on an explicit ``<|content_thinking|>``.
    There is no implicit start.

    ``<|end_message|>`` closes reasoning and is consumed. It terminates
    every Inkling message type, so it is honored only while a span is
    open. A ``<|content_invoke_tool_json|>`` opener also ends reasoning,
    left unconsumed so the tool parser receives it.
    ``<|content_model_end_sampling|>`` terminates the whole assistant turn
    and acts as a stop token rather than a delimiter.
    """

    REASONING_START: ClassVar[str] = "<|content_thinking|>"
    REASONING_END: ClassVar[str] = "<|end_message|>"

    def __init__(
        self,
        thinking_start_token_id: int,
        end_message_token_id: int,
        tool_call_start_token_id: int | None = None,
    ) -> None:
        self.thinking_start_token_id = thinking_start_token_id
        self.end_message_token_id = end_message_token_id
        self.tool_call_start_token_id = tool_call_start_token_id
        self._end_token_ids = (
            (end_message_token_id, tool_call_start_token_id)
            if tool_call_start_token_id is not None
            else (end_message_token_id,)
        )

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk.

        When ``is_currently_reasoning=False`` and the chunk contains no
        ``<|content_thinking|>`` opener, returns an empty span so the whole
        chunk routes to content: Inkling prefills no thinking marker, so a
        turn that answers directly is not reasoning.
        """
        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.thinking_start_token_id
            ):
                start_token_idx = i
            elif token_id in self._end_token_ids:
                # Only honor an end delimiter while a span is active, either
                # pre-seeded via ``is_currently_reasoning`` or opened earlier
                # in this chunk. ``<|end_message|>`` closes every Inkling
                # message type, so a closer from a content or tool message
                # must not pull content into the reasoning span.
                if is_currently_reasoning or start_token_idx is not None:
                    end_token_idx = i
                    break

        if start_token_idx is None and not is_currently_reasoning:
            # No opener and not already reasoning: all content. Inkling
            # prefills no thinking marker, so there is no implicit start.
            empty_span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0),
                reasoning=(0, 0),
            )
            return ParsedReasoningDelta(
                span=empty_span, is_still_reasoning=False
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
        elif delta_token_ids[end_token_idx] == self.end_message_token_id:
            # <|end_message|> is consumed as a delimiter.
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # <|content_invoke_tool_json|> is not consumed: it belongs to
            # the content region where downstream tool parsing handles it.
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
            span=span, is_still_reasoning=is_still_reasoning
        )

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Predicts whether generation starts inside a reasoning span.

        Inkling's generation prompt ends at ``<|message_model|>`` with no
        content-type marker, so a delimiter-free prompt is ``False``: the
        model, not the template, decides whether to think. A ``True``
        suspends grammar enforcement for constrained decoding until
        ``<|end_message|>`` fires, which never comes if the model answers
        directly.

        The scan runs right-to-left because the most recent delimiter
        decides, and it covers the one reachable ``True``: a caller
        prefilling a partial assistant turn whose tail is an open
        ``<|content_thinking|>`` block.
        """
        for token_id in reversed(prompt_token_ids):
            if token_id == self.thinking_start_token_id:
                return True
            if token_id in self._end_token_ids:
                return False
        return False

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> InklingReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        thinking_start_id = await convert_token_to_id(
            tokenizer, cls.REASONING_START
        )
        end_message_id = await convert_token_to_id(tokenizer, cls.REASONING_END)

        if thinking_start_id is None or end_message_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate thinking start/end message"
                " tokens in the tokenizer"
            )

        tool_call_start_id = await convert_token_to_id(
            tokenizer, "<|content_invoke_tool_json|>"
        )

        return cls(
            thinking_start_token_id=thinking_start_id,
            end_message_token_id=end_message_id,
            tool_call_start_token_id=tool_call_start_id,
        )

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``<|end_message|>`` token id that closes reasoning."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
