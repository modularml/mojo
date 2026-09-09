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

"""Gemma 4 reasoning parser for <|channel>...<channel|> sections."""

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

from .tokenizer import REASONING_OPEN, SpecialToken


@register("gemma4")
class Gemma4ReasoningParser(ReasoningParser):
    """Gemma 4 reasoning parser for ``<|channel>``...``<channel|>`` sections.

    When thinking is enabled, the chat template injects a ``<|think|>``
    token in the system message. The model then wraps reasoning output in
    ``<|channel>thought\\n...\\n<channel|>`` blocks. This parser identifies
    those blocks at the token-ID level.

    Reasoning may end implicitly when a tool call begins (``<|tool_call>``).
    The tool-call token is *not* consumed as a delimiter — it stays in the
    content region for downstream tool parsing.
    """

    REASONING_START: ClassVar[str] = "<|channel>"
    REASONING_END: ClassVar[str] = "<channel|>"

    # The literal text that immediately follows the <|channel> token to
    # identify a thinking block (not a special token). Derived from the same
    # opener the tokenizer prefills, so the two never drift.
    # See https://ai.google.dev/gemma/docs/core/model_card_4#2_thinking_mode_configuration
    reasoning_prefix = REASONING_OPEN.removeprefix("<|channel>")

    def __init__(
        self,
        channel_start_token_id: int,
        channel_end_token_id: int,
        tool_call_start_token_id: int | None = None,
        think_token_id: int | None = None,
    ) -> None:
        self.channel_start_token_id = channel_start_token_id
        self.channel_end_token_id = channel_end_token_id
        self.tool_call_start_token_id = tool_call_start_token_id
        self.think_token_id = think_token_id
        self._prefix_cursor = 0
        self._channel_started = False
        # Tokens that close a reasoning span: the channel-end delimiter, plus a
        # tool-call start (reasoning ends implicitly when a tool call begins).
        self._end_token_ids = (
            (channel_end_token_id, tool_call_start_token_id)
            if tool_call_start_token_id is not None
            else (channel_end_token_id,)
        )

    def reset(self) -> None:
        self._prefix_cursor = 0
        self._channel_started = False

    def _format_reasoning_text(self, reasoning: str) -> str | None:
        if self._prefix_cursor >= len(self.reasoning_prefix):
            return reasoning

        for i, ch in enumerate(reasoning):
            if self._prefix_cursor >= len(self.reasoning_prefix):
                return reasoning[i:]
            if self.reasoning_prefix[self._prefix_cursor] == ch:
                self._prefix_cursor += 1
            else:
                result = (
                    self.reasoning_prefix[: self._prefix_cursor] + reasoning[i:]
                )
                self._prefix_cursor = len(self.reasoning_prefix)
                return result

        return None

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk.

        Returns a :class:`ParsedReasoningDelta` containing:

        - ``span``: a :class:`ReasoningSpan` with two index pairs into
          *delta_token_ids* — ``reasoning`` (content only) and
          ``reasoning_with_delimiters`` (includes boundary tokens).
        - ``is_still_reasoning``: ``True`` when no end delimiter was found
          in this chunk *and* the chunk contained a reasoning section.
        - ``reasoning_text_formatter``: callback that strips the
          ``"thought\\n"`` prefix from decoded reasoning text.

        When ``is_currently_reasoning=False`` and no ``<|channel>`` start
        delimiter appears in the chunk, the parser returns an empty
        reasoning span — Gemma 4 emits ``<|channel>thought\\n...<channel|>``
        even when ``enable_thinking`` is off, so callers should pass every
        chunk through here and let the parser dynamically detect mid-stream
        reasoning sections (mirroring vLLM's behavior).
        """
        end_token_ids = self._end_token_ids

        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.channel_start_token_id
            ):
                start_token_idx = i
            elif token_id in end_token_ids:
                # Only treat this as a reasoning-end if we have an active
                # span — either pre-seeded via is_currently_reasoning or
                # opened by a <|channel> in this chunk. Otherwise it's a
                # stray closer from prior content and should not pull
                # content tokens into the reasoning region.
                if is_currently_reasoning or start_token_idx is not None:
                    end_token_idx = i
                    break

        if start_token_idx is not None:
            self._channel_started = True

        # Fall through to the main reasoning logic only for confirmed
        # mid-reasoning continuations: the caller says we're inside a
        # reasoning span AND we've actually seen ``<|channel>`` open
        # the block.  Everything else — not reasoning, or pre-seeded
        # but the model never emitted ``<|channel>`` (skipped thinking)
        # — returns an empty reasoning span so tokens route to content.
        if start_token_idx is None and not (
            is_currently_reasoning and self._channel_started
        ):
            empty_span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0),
                reasoning=(0, 0),
            )
            return ParsedReasoningDelta(
                span=empty_span,
                is_still_reasoning=(
                    is_currently_reasoning and not delta_token_ids
                ),
                reasoning_text_formatter=self._format_reasoning_text,
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
        elif delta_token_ids[end_token_idx] == self.channel_end_token_id:
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # <|tool_call> is not consumed — stays in content region.
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
            reasoning_text_formatter=self._format_reasoning_text,
        )

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Predict whether the model will emit reasoning after this prompt.

        When thinking is on, ``apply_chat_template`` prefills the
        ``<|channel>thought`` opener on the generation turn, so the reasoning
        block is already open at the tail of the prompt. Detect that and mark
        the channel started so ``stream`` parses the model's output as
        reasoning. Otherwise fall back to the ``<|think|>`` hint (the model
        may open a block itself; CENG-249).
        """
        # Is a prefilled <|channel> still open at the tail? Scan back to the
        # last closing delimiter (channel-end, or a tool call); finding the
        # opener first means reasoning is already open.
        end_token_ids = self._end_token_ids
        for token_id in reversed(prompt_token_ids):
            if token_id == self.channel_start_token_id:
                self._channel_started = True
                return True
            if token_id in end_token_ids:
                break

        # Only the <|think|> hint: leave _channel_started unset so stream()
        # routes to content if the model doesn't open a block (CENG-249).
        if self.think_token_id is None:
            return False
        return self.think_token_id in prompt_token_ids

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> Gemma4ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        channel_start_id = await convert_token_to_id(
            tokenizer, cls.REASONING_START
        )
        channel_end_id = await convert_token_to_id(tokenizer, cls.REASONING_END)

        if channel_start_id is None or channel_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate channel start/end"
                " tokens in the tokenizer"
            )

        tool_call_start_id = await convert_token_to_id(
            tokenizer, SpecialToken.TOOL_CALL_START
        )
        think_id = await convert_token_to_id(tokenizer, "<|think|>")

        return cls(
            channel_start_token_id=channel_start_id,
            channel_end_token_id=channel_end_id,
            tool_call_start_token_id=tool_call_start_id,
            think_token_id=think_id,
        )

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``<channel|>`` token id."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
