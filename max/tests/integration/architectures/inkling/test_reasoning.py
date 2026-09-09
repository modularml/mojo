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
from max.pipelines.architectures.inkling.reasoning import (
    InklingReasoningParser,
)

# Synthetic ids for the parser's delimiters.
THINKING = 100  # <|content_thinking|>
END_MESSAGE = 200  # <|end_message|>
TOOL_JSON = 300  # <|content_invoke_tool_json|>
# Structural tokens the parser must treat as ordinary content.
MESSAGE_MODEL = 50  # <|message_model|>
CONTENT_TEXT = 60  # <|content_text|>


def _mock_tokenizer(token_map: dict[str, int | None]) -> Mock:
    """Create a mock tokenizer whose encode() returns single-element arrays."""

    async def mock_encode(
        token: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        token_id = token_map.get(token)
        if token_id is None:
            # Simulate unrecognized token: encode produces multiple IDs
            return np.array([0, 0])
        return np.array([token_id])

    mock = Mock()
    mock.encode = mock_encode
    return mock


def _make_parser() -> InklingReasoningParser:
    return InklingReasoningParser(
        thinking_start_token_id=THINKING,
        end_message_token_id=END_MESSAGE,
        tool_call_start_token_id=TOOL_JSON,
    )


def test_stream_thinking_then_text_in_one_chunk() -> None:
    """Full turn in one chunk: thinking block closes, answer follows."""
    parser = _make_parser()
    tokens = [THINKING, 11, 12, END_MESSAGE, MESSAGE_MODEL, CONTENT_TEXT, 13]
    parsed = parser.stream(tokens, is_currently_reasoning=False)
    assert parsed.is_still_reasoning is False
    assert parsed.span.extract_reasoning(tokens) == [11, 12]
    # <|end_message|> is consumed; the structural tokens for the answer
    # message stay in content (dropped later by skip_special_tokens).
    assert parsed.span.extract_content(tokens) == [
        MESSAGE_MODEL,
        CONTENT_TEXT,
        13,
    ]


def test_stream_straight_to_text_is_all_content() -> None:
    """A turn with no thinking is all content: there is no implicit start."""
    parser = _make_parser()
    tokens = [CONTENT_TEXT, 11, 12, END_MESSAGE]
    parsed = parser.stream(tokens, is_currently_reasoning=False)
    assert parsed.is_still_reasoning is False
    assert parsed.span.extract_reasoning(tokens) == []
    assert parsed.span.extract_content(tokens) == tokens


def test_stream_reasoning_continues_across_chunks() -> None:
    parser = _make_parser()
    chunk1 = [THINKING, 11]
    parsed1 = parser.stream(chunk1, is_currently_reasoning=False)
    assert parsed1.is_still_reasoning is True
    assert parsed1.span.extract_reasoning(chunk1) == [11]
    assert parsed1.span.extract_content(chunk1) == []

    chunk2 = [12, 13]
    parsed2 = parser.stream(chunk2, is_currently_reasoning=True)
    assert parsed2.is_still_reasoning is True
    assert parsed2.span.extract_reasoning(chunk2) == [12, 13]
    assert parsed2.span.extract_content(chunk2) == []

    chunk3 = [14, END_MESSAGE, MESSAGE_MODEL, CONTENT_TEXT, 15]
    parsed3 = parser.stream(chunk3, is_currently_reasoning=True)
    assert parsed3.is_still_reasoning is False
    assert parsed3.span.extract_reasoning(chunk3) == [14]
    assert parsed3.span.extract_content(chunk3) == [
        MESSAGE_MODEL,
        CONTENT_TEXT,
        15,
    ]


def test_stream_second_thinking_block_after_close() -> None:
    """An Inkling turn is a sequence of sub-messages, so it can run think ->
    tool -> think -> answer. The parser is stateless, so each fresh
    <|content_thinking|> opens a new span."""
    parser = _make_parser()

    chunk1 = [THINKING, 11, END_MESSAGE]
    parsed1 = parser.stream(chunk1, is_currently_reasoning=False)
    assert parsed1.is_still_reasoning is False
    assert parsed1.span.extract_reasoning(chunk1) == [11]

    chunk2 = [MESSAGE_MODEL, TOOL_JSON, 77, END_MESSAGE]
    parsed2 = parser.stream(chunk2, is_currently_reasoning=False)
    assert parsed2.is_still_reasoning is False
    assert parsed2.span.extract_reasoning(chunk2) == []
    assert parsed2.span.extract_content(chunk2) == chunk2

    chunk3 = [MESSAGE_MODEL, THINKING, 12, END_MESSAGE, CONTENT_TEXT, 13]
    parsed3 = parser.stream(chunk3, is_currently_reasoning=False)
    assert parsed3.is_still_reasoning is False
    assert parsed3.span.extract_reasoning(chunk3) == [12]
    assert parsed3.span.extract_content(chunk3) == [
        MESSAGE_MODEL,
        CONTENT_TEXT,
        13,
    ]


def test_stream_empty_chunk() -> None:
    """A chunk with no tokens preserves the current reasoning state."""
    parser = _make_parser()

    parsed_reasoning = parser.stream([], is_currently_reasoning=True)
    assert parsed_reasoning.is_still_reasoning is True
    assert parsed_reasoning.span.extract_reasoning([]) == []
    assert parsed_reasoning.span.extract_content([]) == []

    parsed_content = parser.stream([], is_currently_reasoning=False)
    assert parsed_content.is_still_reasoning is False
    assert parsed_content.span.extract_reasoning([]) == []
    assert parsed_content.span.extract_content([]) == []


def test_stream_stray_end_message_not_consumed() -> None:
    """<|end_message|> closes every message type, so with no span active it
    stays in content instead of acting as a reasoning delimiter."""
    parser = _make_parser()
    tokens = [CONTENT_TEXT, 11, END_MESSAGE, 12]
    parsed = parser.stream(tokens, is_currently_reasoning=False)
    assert parsed.is_still_reasoning is False
    assert parsed.span.extract_reasoning(tokens) == []
    assert parsed.span.extract_content(tokens) == tokens


def test_stream_tool_call_opener_ends_reasoning_not_consumed() -> None:
    """A tool-json opener inside an unclosed thinking block ends reasoning
    and stays in content for the tool parser."""
    parser = _make_parser()
    tokens = [THINKING, 11, TOOL_JSON, 77, 78]
    parsed = parser.stream(tokens, is_currently_reasoning=False)
    assert parsed.is_still_reasoning is False
    assert parsed.span.extract_reasoning(tokens) == [11]
    assert parsed.span.extract_content(tokens) == [TOOL_JSON, 77, 78]


def test_will_reason_fresh_generation_prompt_false() -> None:
    """The generation prompt ends at <|message_model|> with no content-type
    marker, so the model decides whether to think and the answer is False."""
    parser = _make_parser()
    prompt = [10, 11, END_MESSAGE, MESSAGE_MODEL]
    assert parser.will_reason_after_prompt(prompt) is False


def test_will_reason_no_delimiters_false() -> None:
    """A delimiter-free prompt does not mean reasoning is open."""
    parser = _make_parser()
    assert parser.will_reason_after_prompt([10, 11, 12]) is False
    assert parser.will_reason_after_prompt([]) is False


def test_will_reason_multi_turn_closed_thinking_false() -> None:
    """Multi-turn prompts replay prior closed thinking blocks; the most
    recent delimiter (the closer) wins."""
    parser = _make_parser()
    prompt = [
        MESSAGE_MODEL,
        THINKING,
        11,
        END_MESSAGE,
        MESSAGE_MODEL,
        CONTENT_TEXT,
        12,
        END_MESSAGE,
        MESSAGE_MODEL,
    ]
    assert parser.will_reason_after_prompt(prompt) is False


def test_will_reason_open_prefilled_thinking_true() -> None:
    """A prompt whose tail is an open <|content_thinking|> block (partial
    assistant prefill) resumes reasoning."""
    parser = _make_parser()
    prompt = [10, END_MESSAGE, MESSAGE_MODEL, THINKING, 11, 12]
    assert parser.will_reason_after_prompt(prompt) is True


@pytest.mark.asyncio
async def test_from_tokenizer_resolves_all_tokens() -> None:
    mock = _mock_tokenizer(
        {
            "<|content_thinking|>": 200008,
            "<|end_message|>": 200010,
            "<|content_invoke_tool_json|>": 200049,
        }
    )
    parser = await InklingReasoningParser.from_tokenizer(mock)
    assert parser.thinking_start_token_id == 200008
    assert parser.end_message_token_id == 200010
    assert parser.tool_call_start_token_id == 200049
    assert await InklingReasoningParser.reasoning_end_token_id(mock) == 200010


@pytest.mark.asyncio
async def test_from_tokenizer_optional_tool_token() -> None:
    mock = _mock_tokenizer(
        {
            "<|content_thinking|>": 200008,
            "<|end_message|>": 200010,
            "<|content_invoke_tool_json|>": None,
        }
    )
    parser = await InklingReasoningParser.from_tokenizer(mock)
    assert parser.tool_call_start_token_id is None


@pytest.mark.asyncio
async def test_from_tokenizer_missing_required_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<|content_thinking|>": None,
            "<|end_message|>": 200010,
            "<|content_invoke_tool_json|>": 200049,
        }
    )
    with pytest.raises(ValueError, match="InklingReasoningParser"):
        await InklingReasoningParser.from_tokenizer(mock)
