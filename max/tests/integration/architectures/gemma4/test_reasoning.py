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

from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.architectures.gemma4.reasoning import (
    Gemma4ReasoningParser,
)


def _mock_tokenizer(token_map: dict[str, int | None]) -> Mock:
    """Create a mock tokenizer whose encode() returns single-element arrays."""

    async def mock_encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        token_id = token_map.get(token)
        if token_id is None:
            return np.array([0, 0])
        return np.array([token_id])

    mock = Mock()
    mock.encode = mock_encode
    return mock


CHANNEL_START_TOKEN_ID = 1
CHANNEL_END_TOKEN_ID = 2
TOOL_CALL_START_TOKEN_ID = 3
THINK_TOKEN_ID = 4


def _make_parser(
    tool_call_start_token_id: int | None = TOOL_CALL_START_TOKEN_ID,
    think_token_id: int | None = THINK_TOKEN_ID,
) -> Gemma4ReasoningParser:
    return Gemma4ReasoningParser(
        channel_start_token_id=CHANNEL_START_TOKEN_ID,
        channel_end_token_id=CHANNEL_END_TOKEN_ID,
        tool_call_start_token_id=tool_call_start_token_id,
        think_token_id=think_token_id,
    )


@pytest.mark.parametrize(
    "tokens,expected_reasoning,expected_content,expected_is_still_reasoning",
    [
        pytest.param(
            [CHANNEL_START_TOKEN_ID, 10, 20, CHANNEL_END_TOKEN_ID, 30],
            [10, 20],
            [30],
            False,
            id="complete_channel_block",
        ),
        pytest.param(
            [CHANNEL_START_TOKEN_ID, 10, 20],
            [10, 20],
            [],
            True,
            id="channel_start_only",
        ),
        pytest.param(
            [10, 20, 30],
            [],
            [10, 20, 30],
            False,
            id="no_channel_tokens_skips_reasoning",
        ),
        pytest.param(
            [],
            [],
            [],
            True,
            id="empty_chunk",
        ),
        pytest.param(
            [
                CHANNEL_START_TOKEN_ID,
                10,
                CHANNEL_START_TOKEN_ID,
                20,
                CHANNEL_END_TOKEN_ID,
            ],
            [10, CHANNEL_START_TOKEN_ID, 20],
            [],
            False,
            id="multiple_channel_starts",
        ),
        pytest.param(
            [10, CHANNEL_END_TOKEN_ID, CHANNEL_START_TOKEN_ID, 20],
            [],
            [10, CHANNEL_END_TOKEN_ID, CHANNEL_START_TOKEN_ID, 20],
            False,
            id="channel_end_before_channel_start_skips_reasoning",
        ),
    ],
)
def test_stream_parsing(
    tokens: list[int],
    expected_reasoning: list[int],
    expected_content: list[int],
    expected_is_still_reasoning: bool,
) -> None:
    parser = _make_parser()
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == expected_reasoning
    assert delta.span.extract_content(tokens) == expected_content
    assert delta.is_still_reasoning is expected_is_still_reasoning


def test_stream_tool_call_ends_reasoning() -> None:
    parser = _make_parser()
    tokens = [CHANNEL_START_TOKEN_ID, 11, TOOL_CALL_START_TOKEN_ID, 77, 78]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [11]
    # <|tool_call> is NOT consumed — stays in content region.
    assert delta.span.extract_content(tokens) == [
        TOOL_CALL_START_TOKEN_ID,
        77,
        78,
    ]


def test_stream_channel_end_wins_over_tool_call() -> None:
    parser = _make_parser()
    # Establish mid-reasoning state from a prior chunk.
    parser.stream([CHANNEL_START_TOKEN_ID, 50])
    tokens = [
        10,
        CHANNEL_END_TOKEN_ID,
        30,
        TOOL_CALL_START_TOKEN_ID,
        40,
    ]
    delta = parser.stream(tokens)
    assert delta.is_still_reasoning is False
    assert delta.span.extract_reasoning(tokens) == [10]
    assert delta.span.extract_content(tokens) == [
        30,
        TOOL_CALL_START_TOKEN_ID,
        40,
    ]


def test_stream_no_tool_call_support() -> None:
    parser = _make_parser(tool_call_start_token_id=None)
    # Establish mid-reasoning state from a prior chunk.
    parser.stream([CHANNEL_START_TOKEN_ID, 50])
    tokens = [10, TOOL_CALL_START_TOKEN_ID, 30]
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == [
        10,
        TOOL_CALL_START_TOKEN_ID,
        30,
    ]
    assert delta.span.extract_content(tokens) == []
    assert delta.is_still_reasoning


# ---- Continuation scenarios (channel already started) --------------------


def test_stream_continuation_no_delimiters() -> None:
    """Mid-reasoning continuation: ``<|channel>`` seen in prior chunk, so
    subsequent tokens without delimiters are still reasoning."""
    parser = _make_parser()
    parser.stream([CHANNEL_START_TOKEN_ID, 50])
    tokens = [10, 20, 30]
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == [10, 20, 30]
    assert delta.span.extract_content(tokens) == []
    assert delta.is_still_reasoning is True


def test_stream_continuation_channel_end_only() -> None:
    """Mid-reasoning continuation: ``<channel|>`` closes the block."""
    parser = _make_parser()
    parser.stream([CHANNEL_START_TOKEN_ID, 50])
    tokens = [10, CHANNEL_END_TOKEN_ID, 30]
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == [10]
    assert delta.span.extract_content(tokens) == [30]
    assert delta.is_still_reasoning is False


def test_stream_continuation_channel_end_before_channel_start() -> None:
    """Mid-reasoning: ``<channel|>`` ends the block, subsequent
    ``<|channel>`` falls into the content region."""
    parser = _make_parser()
    parser.stream([CHANNEL_START_TOKEN_ID, 50])
    tokens = [10, CHANNEL_END_TOKEN_ID, CHANNEL_START_TOKEN_ID, 20]
    delta = parser.stream(tokens)
    assert delta.span.extract_reasoning(tokens) == [10]
    assert delta.span.extract_content(tokens) == [CHANNEL_START_TOKEN_ID, 20]
    assert delta.is_still_reasoning is False


# ---- Model skips thinking (CENG-249 regression) -------------------------


def test_stream_skips_reasoning_routes_to_content() -> None:
    """When ``enable_thinking`` is on but the model decides to skip
    the thinking phase (no ``<|channel>`` emitted), tokens must be
    routed to content, not reasoning."""
    parser = _make_parser()
    # Chunk 1: pre-seeded into reasoning but no <|channel> — model
    # skipped thinking.
    tokens1 = [10, 20, 30]
    delta1 = parser.stream(tokens1, is_currently_reasoning=True)
    assert delta1.span.extract_reasoning(tokens1) == []
    assert delta1.span.extract_content(tokens1) == [10, 20, 30]
    assert delta1.is_still_reasoning is False

    # Chunk 2: caller now passes is_currently_reasoning=False.
    tokens2 = [40, 50]
    delta2 = parser.stream(tokens2, is_currently_reasoning=False)
    assert delta2.span.extract_reasoning(tokens2) == []
    assert delta2.span.extract_content(tokens2) == [40, 50]
    assert delta2.is_still_reasoning is False


def test_stream_skips_reasoning_empty_chunk_stays_reasoning() -> None:
    """An empty first chunk is not enough to conclude the model skipped
    thinking — we need to see actual non-channel tokens first."""
    parser = _make_parser()
    delta = parser.stream([], is_currently_reasoning=True)
    assert delta.is_still_reasoning is True


# ---- is_currently_reasoning=False (dynamic mid-stream detection) ----------
# Regression coverage for the streaming-layer fix that always runs the
# reasoning parser, gated on current state, so Gemma 4 emitting
# ``<|channel>thought\n...<channel|>`` mid-stream (even when not pre-
# seeded into reasoning) gets captured as reasoning instead of leaking
# into content.


def test_stream_not_currently_reasoning_no_channel_returns_empty() -> None:
    parser = _make_parser()
    tokens = [10, 20, 30]
    delta = parser.stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_reasoning(tokens) == []
    assert delta.span.extract_content(tokens) == tokens
    assert delta.is_still_reasoning is False


def test_stream_not_currently_reasoning_with_channel_enters_reasoning() -> None:
    parser = _make_parser()
    tokens = [10, CHANNEL_START_TOKEN_ID, 20, 30, CHANNEL_END_TOKEN_ID, 40]
    delta = parser.stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_reasoning(tokens) == [20, 30]
    assert delta.span.extract_content(tokens) == [10, 40]
    assert delta.is_still_reasoning is False


def test_stream_not_currently_reasoning_open_only_stays_reasoning() -> None:
    parser = _make_parser()
    tokens = [10, CHANNEL_START_TOKEN_ID, 20, 30]
    delta = parser.stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_reasoning(tokens) == [20, 30]
    assert delta.span.extract_content(tokens) == [10]
    assert delta.is_still_reasoning is True


def test_stream_not_currently_reasoning_stray_end_token_ignored() -> None:
    """When we weren't in a reasoning span, a stray ``<channel|>`` (e.g.
    from a prior turn already in the prompt) must not pull content into
    the reasoning region."""
    parser = _make_parser()
    tokens = [10, CHANNEL_END_TOKEN_ID, 20]
    delta = parser.stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_reasoning(tokens) == []
    assert delta.span.extract_content(tokens) == tokens
    assert delta.is_still_reasoning is False


def test_stream_not_currently_reasoning_stray_tool_call_token_ignored() -> None:
    """Same as above but with ``<|tool_call>`` — only treated as a
    reasoning-end when we were actually inside a reasoning span."""
    parser = _make_parser()
    tokens = [10, TOOL_CALL_START_TOKEN_ID, 20]
    delta = parser.stream(tokens, is_currently_reasoning=False)
    assert delta.span.extract_reasoning(tokens) == []
    assert delta.span.extract_content(tokens) == tokens
    assert delta.is_still_reasoning is False


def test_will_reason_after_prompt_think_present() -> None:
    """``<|think|>`` in the prompt means the model will start reasoning."""
    parser = _make_parser()
    prompt = [THINK_TOKEN_ID, 10, 20]
    assert parser.will_reason_after_prompt(prompt) is True


def test_will_reason_after_prompt_no_markers_returns_false() -> None:
    parser = _make_parser()
    prompt = [10, 20, 30]
    assert parser.will_reason_after_prompt(prompt) is False


def test_will_reason_after_prompt_empty_prompt_defaults_false() -> None:
    parser = _make_parser()
    assert parser.will_reason_after_prompt([]) is False


def test_will_reason_after_prompt_multi_turn_with_think() -> None:
    """Even after a closed channel block, ``<|think|>`` means the model
    will open a new thinking block on the next assistant turn."""
    parser = _make_parser()
    prompt = [
        THINK_TOKEN_ID,
        CHANNEL_START_TOKEN_ID,
        10,
        CHANNEL_END_TOKEN_ID,
        20,
    ]
    assert parser.will_reason_after_prompt(prompt) is True


def test_will_reason_after_prompt_no_think_token_returns_false() -> None:
    """Without ``<|think|>`` and with the prior reasoning block already
    closed, the model won't reason on the next turn."""
    parser = _make_parser()
    prompt = [10, CHANNEL_START_TOKEN_ID, 20, CHANNEL_END_TOKEN_ID, 30]
    assert parser.will_reason_after_prompt(prompt) is False


def test_will_reason_after_prompt_prefilled_channel_open_returns_true() -> None:
    """A still-open ``<|channel>`` at the tail of the prompt (the chat
    template prefills ``<|channel>thought`` on the generation turn) means
    reasoning is already open -- even without ``<|think|>``. This is what
    lets Gemma reason on the turn after a ``tool`` result."""
    parser = _make_parser(think_token_id=None)
    prompt = [10, 20, CHANNEL_START_TOKEN_ID]
    assert parser.will_reason_after_prompt(prompt) is True


def test_prefilled_open_stream_captures_reasoning_after_tool_turn() -> None:
    """Regression for OpenRouter ``reasoning-enabled-tool-call-step-5``.

    After a tool result, the chat template prefills ``<|channel>thought`` so
    the prompt ends with an open ``<|channel>``. ``will_reason_after_prompt``
    detects that and seeds the parser, so the model's output -- which has no
    ``<|channel>`` opener (it lives in the prompt) -- is still parsed as
    reasoning instead of leaking into content (which made OR see zero
    reasoning and auto-disable tools)."""
    parser = _make_parser()
    # Prompt: ...tool result... then the prefilled, still-open opener.
    prompt = [TOOL_CALL_START_TOKEN_ID, 99, CHANNEL_START_TOKEN_ID]
    assert parser.will_reason_after_prompt(prompt) is True

    # Model output: reasoning text, the closing delimiter, then the answer.
    tokens = [10, 20, CHANNEL_END_TOKEN_ID, 30]
    delta = parser.stream(tokens, is_currently_reasoning=True)
    assert delta.span.extract_reasoning(tokens) == [10, 20]
    assert delta.span.extract_content(tokens) == [30]
    assert delta.is_still_reasoning is False


def test_will_reason_after_prompt_no_think_token_id_configured() -> None:
    parser = _make_parser(think_token_id=None)
    prompt = [10, 20, 30]
    assert parser.will_reason_after_prompt(prompt) is False


def test_format_reasoning_full_prefix_in_one_chunk() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10, 20])
    assert delta.reasoning_text_formatter is not None
    result = delta.reasoning_text_formatter("thought\nactual reasoning")
    assert result == "actual reasoning"


def test_format_reasoning_split_across_chunks() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    result1 = delta.reasoning_text_formatter("thou")
    assert result1 is None

    delta2 = parser.stream([20, 30])
    assert delta2.reasoning_text_formatter is not None
    result2 = delta2.reasoning_text_formatter("ght\nactual")
    assert result2 == "actual"


def test_format_reasoning_exact_prefix() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    result = delta.reasoning_text_formatter("thought\n")
    assert result is None


def test_format_reasoning_mismatch() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    result = delta.reasoning_text_formatter("unexpected")
    assert result == "unexpected"


def test_reset_clears_prefix_state() -> None:
    parser = _make_parser()
    delta = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta.reasoning_text_formatter is not None
    delta.reasoning_text_formatter("thou")

    parser.reset()

    delta2 = parser.stream([CHANNEL_START_TOKEN_ID, 10])
    assert delta2.reasoning_text_formatter is not None
    result = delta2.reasoning_text_formatter("thought\nactual")
    assert result == "actual"


def test_reset_clears_channel_started_state() -> None:
    """After reset, the parser should again detect skipped thinking."""
    parser = _make_parser()
    parser.stream([CHANNEL_START_TOKEN_ID, 10])

    parser.reset()

    tokens = [10, 20, 30]
    delta = parser.stream(tokens, is_currently_reasoning=True)
    assert delta.span.extract_reasoning(tokens) == []
    assert delta.span.extract_content(tokens) == [10, 20, 30]
    assert delta.is_still_reasoning is False


@pytest.mark.asyncio
async def test_from_tokenizer_missing_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<|channel>": None,
            "<channel|>": None,
            "<|tool_call>": None,
            "<|think|>": None,
        }
    )
    with pytest.raises(
        ValueError,
        match="could not locate channel start/end tokens in the tokenizer",
    ):
        await Gemma4ReasoningParser.from_tokenizer(mock)


@pytest.mark.asyncio
async def test_from_tokenizer_optional_tokens() -> None:
    mock = _mock_tokenizer(
        {
            "<|channel>": 100,
            "<channel|>": 200,
            "<|tool_call>": None,
            "<|think|>": None,
        }
    )
    parser = await Gemma4ReasoningParser.from_tokenizer(mock)
    assert parser.channel_start_token_id == 100
    assert parser.channel_end_token_id == 200
    assert parser.tool_call_start_token_id is None
    assert parser.think_token_id is None


@pytest.mark.asyncio
async def test_from_tokenizer_with_all_tokens() -> None:
    mock = _mock_tokenizer(
        {
            "<|channel>": 100,
            "<channel|>": 200,
            "<|tool_call>": 300,
            "<|think|>": 400,
        }
    )
    parser = await Gemma4ReasoningParser.from_tokenizer(mock)
    assert parser.channel_start_token_id == 100
    assert parser.channel_end_token_id == 200
    assert parser.tool_call_start_token_id == 300
    assert parser.think_token_id == 400
