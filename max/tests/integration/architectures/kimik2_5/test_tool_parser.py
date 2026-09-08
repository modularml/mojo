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

import json
import uuid
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from max import _xgrammar as xgrammar
from max.pipelines.architectures.kimik2_5.tokenizer import (
    IM_END,
    THINK_END,
    THINK_START,
)
from max.pipelines.architectures.kimik2_5.tool_parser import (
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    KimiToolParser,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    XgrammarBackend,
)
from max.pipelines.lib.tool_parsing import StreamingToolCallState
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
    PipelineTokenizer,
)


class _MinimalTokenizer:
    """Byte tokenizer extended with Kimi K2.5 special tokens.

    Maps byte values 0-255 to token IDs 0-255, then assigns dedicated IDs to
    each Kimi structural / reasoning / turn-terminator token so that the
    xgrammar matcher sees them as single tokens — matching how the real Kimi
    tokenizer encodes these markers. This is what lets the grammar frame each
    tool-call body atomically between its structural markers.
    """

    _SPECIAL_TOKENS: dict[str, int] = {
        TOOL_CALLS_SECTION_BEGIN: 256,
        TOOL_CALLS_SECTION_END: 257,
        TOOL_CALL_BEGIN: 258,
        TOOL_CALL_END: 259,
        TOOL_CALL_ARGUMENT_BEGIN: 260,
        THINK_START: 261,
        THINK_END: 262,
        IM_END: 263,
    }
    _N_VOCAB: int = 264

    eos_token_id: int = 0
    bos_token_id: int | None = None
    unk_token_id: int | None = None

    def __init__(self) -> None:
        self.tokens: list[bytes] = [bytes([i]) for i in range(256)]
        self.tokens.extend(t.encode("utf-8") for t in self._SPECIAL_TOKENS)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._SPECIAL_TOKENS.get(token)

    def __call__(self, s: bytes | str) -> list[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        result: list[int] = []
        i = 0
        while i < len(s):
            for text, tid in sorted(
                self._SPECIAL_TOKENS.items(), key=lambda x: -len(x[0])
            ):
                encoded = text.encode("utf-8")
                if s[i : i + len(encoded)] == encoded:
                    result.append(tid)
                    i += len(encoded)
                    break
            else:
                result.append(s[i])
                i += 1
        return result


@pytest.fixture(scope="module")
def minimal_tokenizer() -> _MinimalTokenizer:
    """Raw byte+special-token tokenizer for grammar validation tests."""
    return _MinimalTokenizer()


@pytest.fixture(scope="module")
def mock_tokenizer(
    minimal_tokenizer: _MinimalTokenizer,
) -> PipelineTokenizer[Any, Any, Any]:
    """PipelineTokenizer stub whose ``.delegate`` is the minimal tokenizer."""
    stub = cast(PipelineTokenizer[Any, Any, Any], MagicMock())
    stub.delegate = minimal_tokenizer  # type: ignore[attr-defined]
    return stub


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call with Kimi structural tags."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "New York", "unit": "fahrenheit"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.id.startswith("get_weather:")
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


def test_multiple_tool_calls_parsing() -> None:
    """Test parsing multiple tool calls from Kimi response."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "New York"}
<|tool_call_end|>
<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>
{"timezone": "EST"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 2

    # Check first tool call
    tool_call1 = result.tool_calls[0]
    assert tool_call1.name == "get_weather"
    assert json.loads(tool_call1.arguments) == {"location": "New York"}

    # Check second tool call
    tool_call2 = result.tool_calls[1]
    assert tool_call2.name == "get_time"
    assert json.loads(tool_call2.arguments) == {"timezone": "EST"}

    # Ensure IDs are different
    assert tool_call1.id != tool_call2.id


def test_response_without_tool_calls() -> None:
    """Test parsing a response without tool calls section."""
    parser = KimiToolParser()

    response = "This is just a regular response with no tool calls."

    result = parser.parse_complete(response)

    assert result.content == response
    assert len(result.tool_calls) == 0


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = KimiToolParser()

    response = ""

    result = parser.parse_complete(response)

    assert result.content == ""
    assert len(result.tool_calls) == 0


def test_content_before_tool_calls() -> None:
    """Test parsing response with content before tool calls section."""
    parser = KimiToolParser()

    response = """I'll help you check the weather.

<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "Boston"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert result.content == "I'll help you check the weather."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_function_id_without_prefix() -> None:
    """Test parsing function ID without 'functions.' prefix (fallback format)."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>search:2<|tool_call_argument_begin|>
{"query": "python tutorials"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search"
    assert tool_call.id.startswith("search:")


def test_function_id_without_index() -> None:
    """Test parsing function ID without index suffix."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.calculate<|tool_call_argument_begin|>
{"expression": "2 + 2"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("calculate:")


def test_plain_function_name() -> None:
    """Test parsing plain function name without prefix or index."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>get_random_fact<|tool_call_argument_begin|>
{}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"


def test_complex_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = KimiToolParser()

    complex_params = {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["ai", "tech"],
            "min_score": 0.8,
        },
        "options": {"limit": 10, "sort": "relevance", "include_metadata": True},
    }

    response = f"""<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search_articles:0<|tool_call_argument_begin|>
{json.dumps(complex_params)}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == complex_params


def test_empty_parameters() -> None:
    """Test parsing tool call with empty parameters."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_random_fact:0<|tool_call_argument_begin|>
{}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_tool_calls_section_without_end_tag() -> None:
    """Test parsing when end tag is missing (should still parse)."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>
{"key": "value"}
<|tool_call_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "test"


def test_empty_tool_calls_section_raises_error() -> None:
    """Test that empty tool calls section raises ValueError."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_calls_section_end|>"""

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>
{"param": "value"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        tool_call_id = result.tool_calls[0].id
        ids.add(tool_call_id)

    # All IDs should be unique
    assert len(ids) == 10


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:5<|tool_call_argument_begin|>
{"param": "value"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    assert tool_call_id.startswith("test:")


def test_response_structure() -> None:
    """Test that the response structure matches expected ParsedToolResponse format."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.calculate:0<|tool_call_argument_begin|>
{"expression": "2 + 2"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    # Should return a ParsedToolResponse object
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("calculate:")


def test_whitespace_handling() -> None:
    """Test that whitespace in arguments is handled correctly."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>

    {
        "key": "value with spaces",
        "nested": {
            "inner": "data"
        }
    }

<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].arguments)
    assert args == {"key": "value with spaces", "nested": {"inner": "data"}}


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer and streaming state."""

    parser = KimiToolParser()

    # Simulate accumulating some data
    parser._buffer = "some accumulated data"
    parser._state.sent_content_idx = 10
    parser._state.tool_calls.append(StreamingToolCallState())

    parser.reset()

    assert parser._buffer == ""
    assert parser._state.sent_content_idx == 0
    assert len(parser._state.tool_calls) == 0


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = KimiToolParser()

    # parse_delta should accumulate tokens; return [] to indicate parser is actively buffering and raw tokens shouldn't be used yet.
    result1 = parser.parse_delta("<|tool_calls")
    assert result1 == []

    # Once the section-begin marker is complete, returns [] (not None) so the
    # streaming path knows to suppress structural tokens even with no deltas yet
    result2 = parser.parse_delta("_section_begin|>")
    assert result2 == []
    assert parser._buffer == "<|tool_calls_section_begin|>"


def test_parse_delta_returns_empty_list_inside_tool_section() -> None:
    """Test that parse_delta returns [] (not None) inside the tool-calls section.

    [] signals the caller to suppress raw structural tokens from being emitted
    as content, even when there are no tool-call deltas ready to stream yet.
    """
    parser = KimiToolParser()

    result_pre = parser.parse_delta("<|tool_calls")
    assert result_pre == []

    # Once the section-begin marker completes, returns [] even with no deltas
    result_in_section = parser.parse_delta("_section_begin|>")
    assert result_in_section == []

    # Structural tokens mid-section also return [] while parsing
    result_mid = parser.parse_delta("<|tool_call_begin|>")
    assert result_mid == []


def test_parse_delta_single_tool_call_streaming() -> None:
    """Test streaming a single tool call token by token."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.architectures.kimik2_5.tool_parser.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = KimiToolParser()

        # Simulate streaming a complete tool call
        chunks = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "functions.get_weather:0",
            "<|tool_call_argument_begin|>",
            '{"loc',
            'ation": "',
            'New York"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                id="get_weather:12345678",
                name="get_weather",
            ),
            ParsedToolCallDelta(index=0, arguments='{"loc'),
            ParsedToolCallDelta(index=0, arguments='ation": "'),
            ParsedToolCallDelta(index=0, arguments='New York"}'),
        ]


def test_parse_delta_multiple_tool_calls_streaming() -> None:
    """Test streaming multiple tool calls."""

    uuid_first = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uuid_second = uuid.UUID("22222222-2222-2222-2222-222222222222")
    with patch(
        "max.pipelines.architectures.kimik2_5.tool_parser.uuid.uuid4",
        side_effect=[uuid_first, uuid_second],
    ):
        parser = KimiToolParser()

        # Complete response with two tool calls
        response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "NYC"}
<|tool_call_end|>
<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>
{"zone": "EST"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

        result = parser.parse_delta(response)

        assert result == [
            ParsedToolCallDelta(
                index=0,
                id="get_weather:11111111",
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='\n{"location": "NYC"}\n',
            ),
            ParsedToolCallDelta(
                index=1,
                id="get_time:22222222",
                name="get_time",
            ),
            ParsedToolCallDelta(
                index=1,
                arguments='\n{"zone": "EST"}\n',
            ),
        ]


def test_parse_delta_with_content_before_tools() -> None:
    """Test streaming when there's content before tool calls section."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.architectures.kimik2_5.tool_parser.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = KimiToolParser()

        # Stream content then tool call
        chunks = [
            "I'll check the weather for you.\n\n",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
            '{"location": "Boston"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                content="I'll check the weather for you.\n\n",
            ),
            ParsedToolCallDelta(
                index=0,
                id="get_weather:12345678",
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='{"location": "Boston"}',
            ),
        ]


def test_parse_delta_argument_diffing() -> None:
    """Test that argument deltas are properly diffed."""

    parser = KimiToolParser()

    # Start the tool call
    parser.parse_delta("<|tool_calls_section_begin|>")
    parser.parse_delta(
        "<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>"
    )

    # Send arguments in small chunks
    result1 = parser.parse_delta('{"key')
    result2 = parser.parse_delta('": "val')
    result3 = parser.parse_delta('ue"}')

    # Each result should only contain the new portion
    all_args = []
    for r in [result1, result2, result3]:
        if r:
            for delta in r:
                if delta.arguments:
                    all_args.append(delta.arguments)

    # Concatenated should form the full arguments
    full = "".join(all_args)
    assert '{"key": "value"}' in full or full == '{"key": "value"}'


def test_parse_delta_reset_clears_state() -> None:
    """Test that reset() clears all streaming state."""
    parser = KimiToolParser()

    # Accumulate some state
    parser.parse_delta("<|tool_calls_section_begin|>")
    parser.parse_delta(
        "<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>"
    )
    parser.parse_delta('{"key": "value"}')

    # Verify state exists
    assert parser._buffer != ""
    assert len(parser._state.tool_calls) > 0

    # Reset
    parser.reset()

    # Verify state is cleared
    assert parser._buffer == ""
    assert len(parser._state.tool_calls) == 0
    assert parser._state.sent_content_idx == 0


def test_parse_delta_partial_marker_handling() -> None:
    """Test that partial markers at buffer end are held back."""
    parser = KimiToolParser()

    # Send content that ends with partial marker
    result1 = parser.parse_delta("Hello world<|tool")

    # Should not emit the partial marker as content
    if result1:
        for delta in result1:
            if delta.content:
                assert "<|tool" not in delta.content

    # Complete the marker
    parser.parse_delta("_calls_section_begin|>")

    # Buffer should now contain the complete marker
    assert "<|tool_calls_section_begin|>" in parser._buffer


def test_multiple_tool_calls_same_function() -> None:
    """Test parsing multiple calls to the same function."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>
{"query": "first query"}
<|tool_call_end|>
<|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>
{"query": "second query"}
<|tool_call_end|>
<|tool_call_begin|>functions.search:2<|tool_call_argument_begin|>
{"query": "third query"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 3

    # All should have the same function name
    for tc in result.tool_calls:
        assert tc.name == "search"

    # But different IDs
    ids = [tc.id for tc in result.tool_calls]
    assert len(set(ids)) == 3

    # And different arguments
    queries = [json.loads(tc.arguments)["query"] for tc in result.tool_calls]
    assert queries == ["first query", "second query", "third query"]


@pytest.mark.xfail(
    strict=True,
    reason="TODO(CENG-769): the section scan takes the first SECTION_END, so a "
    "lookalike inside a string argument strands its call and drops later ones",
)
def test_section_end_lookalike_in_argument_value() -> None:
    """Test a JSON string argument that contains the section-end text.

    With tool-call constrained decoding off the model can emit
    ``<|tool_calls_section_end|>`` inside an argument value. Treating that
    as the real section end strands the call it sits in and drops every
    later call in the response.
    """
    parser = KimiToolParser()

    response = f"""<|tool_calls_section_begin|>
<|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>
{json.dumps({"content": "the marker is <|tool_calls_section_end|> here"})}
<|tool_call_end|>
<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>
{{"timezone": "EST"}}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["write_file", "get_time"]
    assert json.loads(result.tool_calls[0].arguments) == {
        "content": "the marker is <|tool_calls_section_end|> here"
    }
    assert json.loads(result.tool_calls[1].arguments) == {"timezone": "EST"}


def test_special_characters_in_arguments() -> None:
    """Test handling of special characters in tool arguments."""
    parser = KimiToolParser()

    special_args = {
        "code": 'print("Hello, World!")',
        "regex": r"\d+\.\d+",
        "unicode": "Hello \u4e16\u754c",
        "newlines": "line1\nline2\nline3",
    }

    response = f"""<|tool_calls_section_begin|>
<|tool_call_begin|>functions.execute:0<|tool_call_argument_begin|>
{json.dumps(special_args)}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == special_args


def _tools(*names: str) -> list[dict[str, Any]]:
    """Build a minimal OpenAI-style tools list from function names."""
    return [{"type": "function", "function": {"name": n}} for n in names]


def _section(name: str, idx: int, args: str) -> str:
    """Builds one ``<|tool_calls_section_begin|>...end|>`` block string."""
    return (
        f"{TOOL_CALLS_SECTION_BEGIN}"
        f"{TOOL_CALL_BEGIN}functions.{name}:{idx}{TOOL_CALL_ARGUMENT_BEGIN}"
        f"{args}{TOOL_CALL_END}"
        f"{TOOL_CALLS_SECTION_END}"
    )


def _make_grammar_matcher(
    grammar: str,
    minimal_tokenizer: _MinimalTokenizer,
) -> Any:
    """Compile ``grammar`` on the xgrammar backend and return a stepping matcher.

    The returned matcher satisfies the ``GrammarMatcher`` interface used by the
    decode path (``try_consume_tokens`` / ``is_accepting``). Compilation raising
    is itself the signal that the grammar is invalid.
    """
    # Build a RAW-vocab tokenizer info from the byte+special vocab, then compile
    # the structural tag through the production backend path
    # (str -> StructuralTag -> compile_structural_tag).
    tokenizer_info = xgrammar.TokenizerInfo(
        minimal_tokenizer.tokens,
        vocab_type=xgrammar.VocabType.RAW,
        vocab_size=_MinimalTokenizer._N_VOCAB,
        stop_token_ids=[minimal_tokenizer.eos_token_id],
    )
    compiler = xgrammar.GrammarCompiler(tokenizer_info)
    return XgrammarBackend(compiler).create_matcher(grammar)


def test_generate_tool_call_grammar_with_tool_names(
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Test generating an xgrammar StructuralTag for constrained decoding."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"),
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Compiling the structural tag raises if it is invalid.
    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)
    assert matcher is not None


def test_generate_tool_call_grammar_without_tool_names(
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Test generating a grammar that accepts any valid identifier."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=None,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)
    assert matcher is not None


def test_generate_tool_call_grammar_rejects_non_xgrammar_backend(
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Grammar generation must reject any backend other than xgrammar."""
    with pytest.raises(InputError, match=r"xgrammar"):
        KimiToolParser.generate_tool_call_grammar(
            tools=_tools("get_weather"),
            tokenizer=mock_tokenizer,
            backend="some_other_backend",
        )


# --- Combined tool-call + response_format on the xgrammar backend ---
#
# xgrammar must support serving tool-calling and response_format=json_schema in
# one request: an ``OrFormat`` structural tag around the Kimi tool-call
# envelope. The behavioral assertions below drive it via ``_make_grammar_matcher``.

_COMBINED_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def test_combined_tool_and_response_format_grammar_compiles(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """The combined grammar compiles on xgrammar.

    Compilation raising is the failure signal: the xgrammar path must produce a
    valid ``OrFormat`` structural tag.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"),
        response_format_schema=_COMBINED_RESPONSE_SCHEMA,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )
    assert isinstance(grammar, str) and grammar

    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)
    assert matcher is not None


def test_combined_grammar_accepts_conforming_json_response(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """The response_format branch accepts a schema-conforming JSON response.

    The grammar pins JSON to a compact form (no inter-token whitespace,
    separators ``","`` / ``":"``), so the payload is emitted with matching
    ``separators`` and xgrammar accepts it.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=_COMBINED_RESPONSE_SCHEMA,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )
    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)

    tokens = minimal_tokenizer(
        json.dumps({"answer": "sunny"}, separators=(",", ":"))
    )
    consumed = matcher.try_consume_tokens(tokens)
    assert consumed == len(tokens), (
        f"rejected a conforming JSON response at offset "
        f"{consumed} of {len(tokens)}; error: {matcher.get_error()}"
    )
    assert matcher.is_accepting(), (
        "matcher not at an accepting state after a complete "
        "schema-conforming JSON response"
    )


def test_combined_grammar_enforces_response_schema(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """The response_format branch enforces the schema.

    A JSON object missing the required ``answer`` field must not be a complete,
    accepted output. The combined grammar is "a full tool call OR a
    schema-conforming JSON" with no free-text branch, so ``{}`` reaches no
    accepting state.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=_COMBINED_RESPONSE_SCHEMA,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )
    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)

    matcher.try_consume_tokens(minimal_tokenizer("{}"))  # missing "answer"
    assert not matcher.is_accepting(), (
        "accepted a JSON response missing a required field — "
        "the response schema was not enforced"
    )


def test_combined_grammar_still_accepts_tool_call(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Adding response_format must not break the tool-call branch.

    Reasoning is handled by the runtime tool/thinking region mechanism, not
    the grammar, so the grammar starts at the tool-call section itself and must
    accept a complete tool call.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=_COMBINED_RESPONSE_SCHEMA,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )
    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)

    section = _section("get_weather", 0, json.dumps({"location": "NYC"}))
    tokens = minimal_tokenizer(section)
    consumed = matcher.try_consume_tokens(tokens)
    assert consumed == len(tokens), (
        f"rejected a tool call in the combined grammar at offset "
        f"{consumed} of {len(tokens)}; error: {matcher.get_error()}"
    )
    assert matcher.is_accepting(), (
        "matcher not accepting after a complete tool call"
    )


def test_combined_grammar_xgrammar_structural_tag_shape(
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """The combined grammar is a plain alternation of a tool call and a
    schema-conforming JSON response.

    Verifies the serialized StructuralTag has that shape: an ``or`` between
    the tool section and the response ``json_schema``, with no reasoning
    prefix (reasoning is handled by the runtime region mechanism).
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=_COMBINED_RESPONSE_SCHEMA,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )
    tag = xgrammar.StructuralTag.model_validate_json(grammar)
    or_format = tag.format
    assert or_format.type == "or"
    element_types = {element.type for element in or_format.elements}
    assert "json_schema" in element_types
    json_branch = next(
        element
        for element in or_format.elements
        if element.type == "json_schema"
    )
    assert json_branch.json_schema == _COMBINED_RESPONSE_SCHEMA


def test_combined_grammar_xgrammar_rejects_reasoning_prefix(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """xgrammar: the grammar admits no reasoning preamble.

    Reasoning is handled by the runtime thinking-region mechanism, which
    suspends enforcement until ``</think>``. The grammar itself must start at
    the tool call or JSON response — a ``</think>`` token is not grammar
    content and is rejected outright.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=_COMBINED_RESPONSE_SCHEMA,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
    )
    matcher = _make_grammar_matcher(grammar, minimal_tokenizer)

    tokens = minimal_tokenizer(
        THINK_END + json.dumps({"answer": "sunny"}, separators=(",", ":"))
    )
    consumed = matcher.try_consume_tokens(tokens)
    assert consumed == 0, (
        f"grammar consumed {consumed} tokens of a reasoning-prefixed "
        f"response; reasoning must not be grammar content"
    )


def test_xgrammar_enforces_tool_argument_schema(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """xgrammar constrains each tool call's arguments to the tool's JSON schema.

    A call whose arguments violate the tool's parameter schema (missing a
    required field, wrong value type) never reaches an accepting state.
    ``required`` tool choice forces the section from the first token, so no
    reasoning prefix is needed. (The combined-grammar tests above only exercise
    the *response* schema; this pins the *tool-argument* schema.)
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=tools,
        tokenizer=mock_tokenizer,
        backend="xgrammar",
        tool_choice="required",
    )

    def accepts(args: str) -> bool:
        matcher = _make_grammar_matcher(grammar, minimal_tokenizer)
        tokens = minimal_tokenizer(_section("get_weather", 0, args))
        return (
            matcher.try_consume_tokens(tokens) == len(tokens)
            and matcher.is_accepting()
        )

    assert accepts('{"location": "NYC"}'), "schema-valid args must be accepted"
    assert not accepts("{}"), "missing required field must be rejected"
    assert not accepts('{"location": 42}'), "wrong value type must be rejected"


def test_xgrammar_rejects_nonjson_tool_argument_body(
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """xgrammar frames the tool-argument body as a JSON value, so trailing
    non-JSON garbage is rejected.

    The JSON value ends at ``}`` and a trailing ``< extra`` has no continuation,
    so ``{"done": true} < extra`` is rejected while ``{"done": true}`` alone is
    accepted.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        tokenizer=mock_tokenizer,
        backend="xgrammar",
        tool_choice="required",
    )

    def accepts(args: str) -> bool:
        matcher = _make_grammar_matcher(grammar, minimal_tokenizer)
        tokens = minimal_tokenizer(_section("get_weather", 0, args))
        return (
            matcher.try_consume_tokens(tokens) == len(tokens)
            and matcher.is_accepting()
        )

    assert not accepts('{"done": true} < extra')
    # Sanity: the same body without the trailing garbage IS accepted.
    assert accepts('{"done": true}')


def test_parse_complete_multiple_sections() -> None:
    """parse_complete aggregates tool calls across multiple sections.

    Kimi emits multiple ``<|tool_calls_section_begin|>...end|>`` blocks per
    turn. The parser must return every call across all sections and must not
    leak inter-section text (here a reasoning block) into a tool call.
    """
    parser = KimiToolParser()

    response = (
        _section("get_weather", 0, '{"location": "NYC"}')
        + f"{THINK_START}now the time{THINK_END}"
        + _section("get_time", 1, '{"zone": "EST"}')
    )

    result = parser.parse_complete(response)

    assert result.content is None
    assert [tc.name for tc in result.tool_calls] == ["get_weather", "get_time"]
    assert json.loads(result.tool_calls[0].arguments) == {"location": "NYC"}
    assert json.loads(result.tool_calls[1].arguments) == {"zone": "EST"}
    # The inter-section reasoning must not have leaked into either call.
    for tc in result.tool_calls:
        assert "now the time" not in tc.arguments
        assert "think" not in tc.arguments


def test_parser_handles_json_content_when_no_tool_calls() -> None:
    """Test that parser returns content as-is when no tool call markers present."""
    parser = KimiToolParser()

    # This is what the model would output when choosing JSON content
    # instead of tool calls (with combined grammar)
    json_response = '{"answer": "The weather is sunny", "confidence": 0.95}'

    result = parser.parse_complete(json_response)

    # No tool calls should be parsed
    assert len(result.tool_calls) == 0
    # Content should be returned as-is
    assert result.content == json_response
