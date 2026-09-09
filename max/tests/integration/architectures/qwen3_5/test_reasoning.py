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
from max.pipelines.architectures.qwen3_5.reasoning import (
    Qwen3_5ReasoningParser,
)


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


def _make_parser(
    tool_call_start_token_id: int | None = 300,
) -> Qwen3_5ReasoningParser:
    return Qwen3_5ReasoningParser(
        think_start_token_id=100,
        think_end_token_id=200,
        tool_call_start_token_id=tool_call_start_token_id,
    )


def test_stream_finds_think_boundaries() -> None:
    parser = _make_parser()
    # [prefix, <think>, r1, r2, </think>, suffix]
    tokens = [10, 100, 11, 12, 200, 13]
    parsed = parser.stream(tokens)
    span, is_still_reasoning = parsed.span, parsed.is_still_reasoning
    assert is_still_reasoning is False
    assert span.extract_reasoning(tokens) == [11, 12]
    assert span.extract_content(tokens) == [10, 13]


def test_stream_implicit_start() -> None:
    """Chat template prepends <think>\\n, so model output begins inside reasoning."""
    parser = _make_parser()
    # [r1, r2, </think>, answer]
    tokens = [11, 12, 200, 42]
    parsed = parser.stream(tokens)
    span, is_still_reasoning = parsed.span, parsed.is_still_reasoning
    assert is_still_reasoning is False
    assert span.extract_reasoning(tokens) == [11, 12]
    assert span.extract_content(tokens) == [42]


def test_stream_tool_call_ends_reasoning() -> None:
    """<tool_call> implicitly ends reasoning but stays in the content region."""
    parser = _make_parser()
    # [<think>, r1, <tool_call>, tc1, tc2]
    tokens = [100, 11, 300, 77, 78]
    parsed = parser.stream(tokens)
    span, is_still_reasoning = parsed.span, parsed.is_still_reasoning
    assert is_still_reasoning is False
    assert span.extract_reasoning(tokens) == [11]
    # <tool_call> is NOT consumed — stays in content for the tool parser.
    assert span.extract_content(tokens) == [300, 77, 78]


def test_stream_no_end_still_reasoning() -> None:
    parser = _make_parser()
    tokens = [11, 12, 13]
    parsed = parser.stream(tokens)
    span, is_still_reasoning = parsed.span, parsed.is_still_reasoning
    assert is_still_reasoning is True
    assert span.extract_reasoning(tokens) == [11, 12, 13]
    assert span.extract_content(tokens) == []


def test_stream_without_tool_call_token_only_uses_think_end() -> None:
    parser = _make_parser(tool_call_start_token_id=None)
    # 300 should now NOT terminate reasoning.
    tokens = [11, 300, 12, 200, 42]
    parsed = parser.stream(tokens)
    span, is_still_reasoning = parsed.span, parsed.is_still_reasoning
    assert is_still_reasoning is False
    assert span.extract_reasoning(tokens) == [11, 300, 12]
    assert span.extract_content(tokens) == [42]


def test_will_reason_after_prompt_new_turn_open() -> None:
    """Chat template prepended <think> for the new turn — most recent delimiter wins."""
    parser = _make_parser()
    # Prior turn: <think> r1 </think> content. New turn: <think> at the end.
    prompt = [100, 11, 200, 12, 100]
    assert parser.will_reason_after_prompt(prompt) is True


def test_will_reason_after_prompt_disabled_thinking() -> None:
    """No new <think> prepended (enable_thinking=false): most recent </think> closes."""
    parser = _make_parser()
    prompt = [100, 11, 200, 12, 13]
    assert parser.will_reason_after_prompt(prompt) is False


def test_will_reason_after_prompt_empty_prompt_open() -> None:
    parser = _make_parser()
    assert parser.will_reason_after_prompt([]) is True


def test_will_reason_after_prompt_no_delimiters_open() -> None:
    """No delimiters at all: matches the implicit pre-fill seeding."""
    parser = _make_parser()
    assert parser.will_reason_after_prompt([10, 11, 12]) is True


@pytest.mark.asyncio
async def test_from_tokenizer_missing_tokens_raises() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": None,
            "</think>": 200,
            "<tool_call>": 300,
        }
    )
    with pytest.raises(ValueError, match="Qwen3_5ReasoningParser"):
        await Qwen3_5ReasoningParser.from_tokenizer(mock)


@pytest.mark.asyncio
async def test_from_tokenizer_optional_tool_token() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<tool_call>": None,
        }
    )
    parser = await Qwen3_5ReasoningParser.from_tokenizer(mock)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_call_start_token_id is None


@pytest.mark.asyncio
async def test_from_tokenizer_with_tool_token() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<tool_call>": 300,
        }
    )
    parser = await Qwen3_5ReasoningParser.from_tokenizer(mock)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_call_start_token_id == 300
