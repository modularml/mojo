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

import asyncio
from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.architectures.kimik2_5.reasoning import (
    KimiK2_5ReasoningParser,
)
from max.pipelines.architectures.minimax_m2.reasoning import (
    MiniMaxM2ReasoningParser,
)
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    _resolve_thinking_token_ids,
)
from max.pipelines.lib.reasoning import create, register
from max.pipelines.lib.registry import _ThinkingRegionNewContext
from max.pipelines.lib.tokenizer import resolve_single_special_token
from max.pipelines.modeling.types.reasoning import (
    ParsedReasoningDelta,
    ReasoningParser,
    ReasoningPipelineTokenizer,
    ReasoningSpan,
)


def test_reasoning_span_extract_content_removes_delimited_span() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 4), reasoning=(1, 3))
    result = span.extract_content([10, 20, 30, 40, 50])
    assert result == [50]


def test_reasoning_span_extract_reasoning_excludes_delimiters() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 4), reasoning=(1, 3))
    result = span.extract_reasoning([10, 20, 30, 40, 50])
    assert result == [20, 30]


def test_reasoning_span_extract_content_returns_empty_when_empty() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 5), reasoning=(1, 4))
    result = span.extract_content([10, 20, 30, 40, 50])
    assert result == []


def test_reasoning_span_extract_reasoning_returns_empty_when_empty() -> None:
    """Zero-width reasoning within delimiters returns empty list."""
    span = ReasoningSpan(reasoning_with_delimiters=(0, 2), reasoning=(1, 1))
    result = span.extract_reasoning([10, 20, 30])
    assert result == []


def test_reasoning_span_zero_width() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(0, 0), reasoning=(0, 0))
    content = span.extract_content([10, 20])
    reasoning = span.extract_reasoning([10, 20])
    assert content == [10, 20]
    assert reasoning == []


def test_reasoning_span_content_before_and_after() -> None:
    span = ReasoningSpan(reasoning_with_delimiters=(1, 4), reasoning=(2, 3))
    tokens = [10, 20, 30, 40, 50]
    assert span.extract_content(tokens) == [10, 50]
    assert span.extract_reasoning(tokens) == [30]


def test_reasoning_span_invalid_ranges_raise() -> None:
    with pytest.raises(AssertionError):
        # reasoning range extends beyond delimited range
        ReasoningSpan(reasoning_with_delimiters=(2, 4), reasoning=(1, 3))


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


@pytest.mark.asyncio
async def test_register_and_create() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<|tool_calls_section_begin|>": None,
        }
    )
    parser = await create("kimik2_5", mock)
    assert isinstance(parser, KimiK2_5ReasoningParser)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_section_start_token_id is None


@pytest.mark.asyncio
async def test_create_unknown_parser_raises() -> None:
    mock = Mock()
    with pytest.raises(ValueError, match="Unknown reasoning parser"):
        await create("nonexistent", mock)


@pytest.mark.asyncio
async def test_minimax_m2_register_and_create() -> None:
    mock = _mock_tokenizer(
        {
            "<think>": 100,
            "</think>": 200,
            "<minimax:tool_call>": 300,
        }
    )
    parser = await create("minimax_m2", mock)
    assert isinstance(parser, MiniMaxM2ReasoningParser)
    assert parser.think_start_token_id == 100
    assert parser.think_end_token_id == 200
    assert parser.tool_call_start_token_id == 300


# ---------------------------------------------------------------------------
# ReasoningPipelineTokenizer protocol + _resolve_thinking_token_ids
# ---------------------------------------------------------------------------


def _stub_reasoning_tokenizer(start_id: int, end_id: int) -> Mock:
    """Build a Mock that satisfies the full ``ReasoningPipelineTokenizer``."""
    tok = Mock()
    tok.eos_token_ids = set()
    tok.expects_content_wrapping = False
    tok.reasoning_start_token_id = start_id
    tok.reasoning_end_token_id = end_id
    return tok


def test_protocol_isinstance_check_negative_for_bare_object() -> None:
    """An object without the protocol surface does not satisfy the protocol."""
    assert not isinstance(object(), ReasoningPipelineTokenizer)


def test_resolve_thinking_token_ids_reads_protocol_properties() -> None:
    """``_resolve_thinking_token_ids`` returns the two property values
    declared by a :class:`ReasoningPipelineTokenizer`."""
    tok = _stub_reasoning_tokenizer(start_id=100, end_id=101)
    assert _resolve_thinking_token_ids(
        cast(ReasoningPipelineTokenizer[Any, Any, Any], tok)
    ) == (100, 101)


# ---------------------------------------------------------------------------
# resolve_single_special_token helper
# ---------------------------------------------------------------------------


def _stub_delegate(vocab: dict[str, int | list[int]], unk_id: int = 0) -> Mock:
    """Build a Mock HF-style delegate with ``convert_tokens_to_ids``."""
    delegate = Mock()
    delegate.unk_token_id = unk_id
    delegate.convert_tokens_to_ids = lambda token: vocab.get(token, unk_id)
    return delegate


def test_resolve_single_special_token_returns_id() -> None:
    delegate = _stub_delegate({"<think>": 42})
    assert resolve_single_special_token(delegate, "<think>") == 42


def test_resolve_single_special_token_raises_when_missing() -> None:
    delegate = _stub_delegate(vocab={}, unk_id=3)
    with pytest.raises(ValueError, match="not found in tokenizer vocabulary"):
        resolve_single_special_token(delegate, "<think>")


def test_resolve_single_special_token_raises_when_multi_id() -> None:
    delegate = _stub_delegate({"<think>": [1, 2, 3]})
    with pytest.raises(ValueError, match="resolved to multiple ids"):
        resolve_single_special_token(delegate, "<think>")


# ---------------------------------------------------------------------------
# _ThinkingRegionNewContext lazy-resolution race
# ---------------------------------------------------------------------------


class _GatedParser(ReasoningParser):
    """Parser whose async construction blocks until the test releases it."""

    gate: asyncio.Event

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        raise NotImplementedError

    def will_reason_after_prompt(self, prompt_token_ids: Sequence[int]) -> bool:
        return True

    @classmethod
    async def from_tokenizer(cls, tokenizer: Any) -> "_GatedParser":
        await cls.gate.wait()
        return cls()

    @classmethod
    async def reasoning_end_token_id(cls, tokenizer: Any) -> int | None:
        return 7


def _constrained_context() -> Mock:
    context = Mock()
    context.grammar = None
    context.json_schema = '{"type": "object"}'
    context.tokens.prompt = [1, 2, 3]
    context.grammar_state = Mock()
    context.grammar_state.grammar_enforced = True
    return context


@pytest.mark.asyncio
async def test_thinking_region_suspends_grammar_for_racing_first_requests() -> (
    None
):
    """Concurrent first requests must all get the thinking-region suspension.

    Regression: resolution of the reasoning parser is lazy, and publishing
    the resolved flag before the awaited construction finished let a request
    that raced the first one skip the suspension — the grammar bitmask was
    then enforced from token 0, inside the model's reasoning span.
    """
    register("gated-race-test")(_GatedParser)
    _GatedParser.gate = asyncio.Event()

    contexts = [_constrained_context(), _constrained_context()]

    async def original_new_context(request: Any) -> Mock:
        return contexts[request]

    wrapper = _ThinkingRegionNewContext(
        tokenizer=Mock(),
        original_new_context=original_new_context,
        reasoning_parser_name="gated-race-test",
    )

    first = asyncio.create_task(wrapper(0))
    second = asyncio.create_task(wrapper(1))
    # Let both tasks reach the (blocked) resolution before releasing it.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    _GatedParser.gate.set()
    await asyncio.gather(first, second)

    for context in contexts:
        assert context.grammar_state.grammar_enforced is False
        assert context.grammar_state._in_thinking_region is True
        context.set_thinking_region.assert_called_once_with(None, [7])
