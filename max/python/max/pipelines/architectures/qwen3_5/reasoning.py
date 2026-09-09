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

"""Qwen 3.5 / 3.6 reasoning parser for ``<think>...</think>`` sections."""

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


@register("qwen3_5")
class Qwen3_5ReasoningParser(ReasoningParser):
    """Qwen 3.5 / 3.6 reasoning parser for ``<think>...</think>`` sections.

    Qwen 3.5/3.6's chat template prepends ``<think>\\n`` to every assistant
    turn when ``enable_thinking`` is true (the default), so reasoning
    begins implicitly without an explicit ``<think>`` token in the model
    output stream. Reasoning ends explicitly at ``</think>``, or
    implicitly when a tool call begins (``<tool_call>``) — the tool-call
    marker is left in the content region for the tool parser to consume.
    """

    REASONING_START: ClassVar[str] = "<think>"
    REASONING_END: ClassVar[str] = "</think>"

    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
        tool_call_start_token_id: int | None = None,
    ) -> None:
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id
        self.tool_call_start_token_id = tool_call_start_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identify a reasoning span within a streaming delta chunk.

        When ``is_currently_reasoning=False`` and the chunk contains no
        ``<think>`` opener, returns an empty span so post-reasoning content
        chunks aren't misclassified as reasoning.
        """
        end_token_ids: tuple[int, ...]
        if self.tool_call_start_token_id is not None:
            end_token_ids = (
                self.think_end_token_id,
                self.tool_call_start_token_id,
            )
        else:
            end_token_ids = (self.think_end_token_id,)

        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.think_start_token_id
            ):
                start_token_idx = i
            elif token_id in end_token_ids:
                # Only consume an end delimiter if a reasoning span is active
                # (pre-seeded via ``is_currently_reasoning`` or opened by a
                # ``<think>`` earlier in this chunk); a stray delimiter from
                # prior content must not pull content into the reasoning span.
                if is_currently_reasoning or start_token_idx is not None:
                    end_token_idx = i
                    break

        if start_token_idx is None and not is_currently_reasoning:
            # No reasoning opener and not already reasoning — all content.
            empty_span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0),
                reasoning=(0, 0),
            )
            return ParsedReasoningDelta(
                span=empty_span, is_still_reasoning=False
            )

        if start_token_idx is None:
            # Implicit start: chat template pre-fills <think>, so reasoning
            # begins at index 0 when no explicit <think> token is present.
            start_reasoning = 0
            start_reasoning_with_delimiters = 0
        else:
            start_reasoning = start_token_idx + 1
            start_reasoning_with_delimiters = start_token_idx

        if end_token_idx is None:
            end_reasoning = len(delta_token_ids)
            end_reasoning_with_delimiters = len(delta_token_ids)
        elif delta_token_ids[end_token_idx] == self.think_end_token_id:
            # </think> is consumed as a delimiter.
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # <tool_call> is not consumed — it belongs to the content
            # region where downstream tool parsing handles it.
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
        """Decide whether the next generated token continues a reasoning span.

        Overrides the ABC default (which delegates to ``stream`` scanning
        left-to-right). That default is wrong for Qwen: the chat template
        embeds a literal ``<tool_call>`` *example* in the tool instructions,
        and ``<tool_call>`` is a reasoning-end delimiter — so a left-to-right
        scan hits the example and falsely concludes reasoning already ended,
        leaking the model's ``<think>`` block into content.

        Multi-turn prompts can also contain ``<think>``/``</think>`` tokens
        from prior assistant turns; only the most-recently-emitted delimiter
        describes the *current* state. Scan right-to-left: the last delimiter
        before generation is the chat template's prefilled ``<think>``.
        """
        end_token_ids: tuple[int, ...]
        if self.tool_call_start_token_id is not None:
            end_token_ids = (
                self.think_end_token_id,
                self.tool_call_start_token_id,
            )
        else:
            end_token_ids = (self.think_end_token_id,)

        for token_id in reversed(prompt_token_ids):
            if token_id == self.think_start_token_id:
                return True
            if token_id in end_token_ids:
                return False
        # No delimiters at all: matches the implicit pre-fill seeding.
        return True

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> Qwen3_5ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        think_start_id = await convert_token_to_id(
            tokenizer, cls.REASONING_START
        )
        think_end_id = await convert_token_to_id(tokenizer, cls.REASONING_END)

        if think_start_id is None or think_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate think start/end tokens in the tokenizer"
            )

        tool_call_start_id = await convert_token_to_id(tokenizer, "<tool_call>")

        return cls(
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
            tool_call_start_token_id=tool_call_start_id,
        )

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``</think>`` token id that closes a reasoning span."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
