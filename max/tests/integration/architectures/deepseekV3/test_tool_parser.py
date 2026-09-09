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
# ruff: noqa: RUF001

import json
import pathlib
import uuid
from unittest.mock import patch

import pytest
from max.pipelines.architectures.deepseekV3.tool_parser import (
    DeepseekV3_1ToolParser,
    DeepseekV3ToolParser,
    resolve_deepseekv3_tool_parser,
)
from max.pipelines.lib.tool_parsing import StreamingToolCallState
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)
from max.pipelines.weights.hf_utils import HuggingFaceRepo


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call with DeepSeek V3 structural tags."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "New York", "unit": "fahrenheit"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.id.startswith("call_")
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


def test_multiple_tool_calls_parsing() -> None:
    """Test parsing multiple tool calls from DeepSeek V3 response."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "New York"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "EST"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 2

    tool_call1 = result.tool_calls[0]
    assert tool_call1.name == "get_weather"
    assert json.loads(tool_call1.arguments) == {"location": "New York"}

    tool_call2 = result.tool_calls[1]
    assert tool_call2.name == "get_time"
    assert json.loads(tool_call2.arguments) == {"timezone": "EST"}

    assert tool_call1.id != tool_call2.id


def test_response_without_tool_calls() -> None:
    """Test parsing a response without tool calls section."""
    parser = DeepseekV3_1ToolParser()

    response = "This is just a regular response with no tool calls."

    result = parser.parse_complete(response)

    assert result.content == response
    assert len(result.tool_calls) == 0


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = DeepseekV3_1ToolParser()

    response = ""

    result = parser.parse_complete(response)

    assert result.content == ""
    assert len(result.tool_calls) == 0


def test_content_before_tool_calls() -> None:
    """Test parsing response with content before tool calls section."""
    parser = DeepseekV3_1ToolParser()

    response = """I'll help you check the weather.

<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "Boston"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert result.content == "I'll help you check the weather."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_function_name_with_special_chars() -> None:
    """Test parsing a function name with underscores and digits."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search_v2_results<｜tool▁sep｜>{"query": "test"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_v2_results"


def test_complex_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = DeepseekV3_1ToolParser()

    complex_params = {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["ai", "tech"],
            "min_score": 0.8,
        },
        "options": {"limit": 10, "sort": "relevance", "include_metadata": True},
    }

    response = f"""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search_articles<｜tool▁sep｜>{json.dumps(complex_params)}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == complex_params


def test_empty_parameters() -> None:
    """Test parsing tool call with empty parameters."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_random_fact<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_tool_calls_section_without_end_tag() -> None:
    """Test parsing when end tag is missing (should still parse)."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>{"key": "value"}<｜tool▁call▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "test"


def test_empty_tool_calls_section_raises_error() -> None:
    """Test that empty tool calls section raises ValueError."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>"""

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>{"param": "value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        ids.add(result.tool_calls[0].id)

    assert len(ids) == 10


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>{"param": "value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    assert tool_call_id.startswith("call_")
    # The suffix should be hex characters of length 24.
    suffix = tool_call_id[len("call_") :]
    assert len(suffix) == 24
    assert all(c in "0123456789abcdef" for c in suffix)


def test_response_structure() -> None:
    """Test that the response structure matches expected ParsedToolResponse format."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>calculate<｜tool▁sep｜>{"expression": "2 + 2"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("call_")


def test_whitespace_handling() -> None:
    """Test that whitespace in arguments is handled correctly."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>

    {
        "key": "value with spaces",
        "nested": {
            "inner": "data"
        }
    }

<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].arguments)
    assert args == {"key": "value with spaces", "nested": {"inner": "data"}}


def test_special_characters_in_arguments() -> None:
    """Test handling of special characters in tool arguments."""
    parser = DeepseekV3_1ToolParser()

    special_args = {
        "code": 'print("Hello, World!")',
        "regex": r"\d+\.\d+",
        "unicode": "Hello 世界",
        "newlines": "line1\nline2\nline3",
    }

    response = f"""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>execute<｜tool▁sep｜>{json.dumps(special_args)}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == special_args


def test_multiple_tool_calls_same_function() -> None:
    """Test parsing multiple calls to the same function."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"query": "first query"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"query": "second query"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"query": "third query"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 3

    for tc in result.tool_calls:
        assert tc.name == "search"

    ids = [tc.id for tc in result.tool_calls]
    assert len(set(ids)) == 3

    queries = [json.loads(tc.arguments)["query"] for tc in result.tool_calls]
    assert queries == ["first query", "second query", "third query"]


def test_parse_complete_skips_empty_name() -> None:
    """Test that tool calls with empty function names are skipped."""
    parser = DeepseekV3_1ToolParser()

    response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜><｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁call▁begin｜>valid<｜tool▁sep｜>{"ok": true}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "valid"


def test_parse_complete_whitespace_only_before_tool_calls() -> None:
    """Whitespace-only content before the section is coerced to None."""
    parser = DeepseekV3_1ToolParser()

    response = """\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>{"x": 1}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

    result = parser.parse_complete(response)

    assert result.content is None
    assert len(result.tool_calls) == 1


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer and streaming state."""
    parser = DeepseekV3_1ToolParser()

    parser._buffer = "some accumulated data"
    parser._state.sent_content_idx = 10
    parser._state.tool_calls.append(StreamingToolCallState())

    parser.reset()

    assert parser._buffer == ""
    assert parser._state.sent_content_idx == 0
    assert len(parser._state.tool_calls) == 0


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = DeepseekV3_1ToolParser()

    # parse_delta should accumulate tokens; return [] to indicate parser is actively buffering and raw tokens shouldn't be used yet.
    result1 = parser.parse_delta("<｜tool▁calls")
    assert result1 == []

    # Once the marker completes, the parser returns [] (not None) so the
    # streaming path knows to suppress raw structural tokens even with
    # no deltas yet.
    result2 = parser.parse_delta("▁begin｜>")
    assert result2 == []
    assert parser._buffer == "<｜tool▁calls▁begin｜>"


def test_parse_delta_single_tool_call_streaming() -> None:
    """Test streaming a single tool call token by token."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = DeepseekV3_1ToolParser()

        chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "get_weather",
            "<｜tool▁sep｜>",
            '{"loc',
            'ation": "',
            'New York"}',
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:24]}"
        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                id=expected_id,
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
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        side_effect=[uuid_first, uuid_second],
    ):
        parser = DeepseekV3_1ToolParser()

        response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "NYC"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"zone": "EST"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

        result = parser.parse_delta(response)

        expected_id_first = f"call_{uuid_first.hex[:24]}"
        expected_id_second = f"call_{uuid_second.hex[:24]}"
        assert result == [
            ParsedToolCallDelta(
                index=0,
                id=expected_id_first,
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='{"location": "NYC"}',
            ),
            ParsedToolCallDelta(
                index=1,
                id=expected_id_second,
                name="get_time",
            ),
            ParsedToolCallDelta(
                index=1,
                arguments='{"zone": "EST"}',
            ),
        ]


def test_parse_delta_with_content_before_tools() -> None:
    """Test streaming when there's content before tool calls section."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = DeepseekV3_1ToolParser()

        chunks = [
            "I'll check the weather for you.\n\n",
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>",
            '{"location": "Boston"}',
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:24]}"
        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                content="I'll check the weather for you.\n\n",
            ),
            ParsedToolCallDelta(
                index=0,
                id=expected_id,
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='{"location": "Boston"}',
            ),
        ]


def test_parse_delta_argument_diffing() -> None:
    """Test that argument deltas are properly diffed."""
    parser = DeepseekV3_1ToolParser()

    parser.parse_delta("<｜tool▁calls▁begin｜>")
    parser.parse_delta("<｜tool▁call▁begin｜>test<｜tool▁sep｜>")

    result1 = parser.parse_delta('{"key')
    result2 = parser.parse_delta('": "val')
    result3 = parser.parse_delta('ue"}')

    all_args = []
    for r in [result1, result2, result3]:
        if r:
            for delta in r:
                if delta.arguments:
                    all_args.append(delta.arguments)

    full = "".join(all_args)
    assert full == '{"key": "value"}'


def test_parse_delta_reset_clears_state() -> None:
    """Test that reset() clears all streaming state."""
    parser = DeepseekV3_1ToolParser()

    parser.parse_delta("<｜tool▁calls▁begin｜>")
    parser.parse_delta("<｜tool▁call▁begin｜>test<｜tool▁sep｜>")
    parser.parse_delta('{"key": "value"}')

    assert parser._buffer != ""
    assert len(parser._state.tool_calls) > 0

    parser.reset()

    assert parser._buffer == ""
    assert len(parser._state.tool_calls) == 0
    assert parser._state.sent_content_idx == 0


def test_parse_delta_partial_marker_handling() -> None:
    """Test that partial markers at buffer end are held back."""
    parser = DeepseekV3_1ToolParser()

    result = parser.parse_delta("Hello world<｜tool")

    assert result is not None
    assert len(result) == 1
    assert result[0].content == "Hello world"

    parser.parse_delta("▁calls▁begin｜>")

    assert "<｜tool▁calls▁begin｜>" in parser._buffer


def test_parse_delta_mid_token_splits() -> None:
    """Test streaming with chunks that split mid-tag."""
    fixed_uuid = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = DeepseekV3_1ToolParser()

        chunks = [
            "<｜tool▁calls▁beg",
            "in｜><｜tool▁call▁begin｜>",
            "do_thing<｜tool▁s",
            'ep｜>{"x": 1}',
            "<｜tool▁call▁en",
            "d｜><｜tool▁calls▁end｜>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:24]}"
        # Order matters: name first, then arguments.
        name_deltas = [d for d in all_deltas if d.name]
        arg_deltas = [d for d in all_deltas if d.arguments]
        assert name_deltas == [
            ParsedToolCallDelta(index=0, id=expected_id, name="do_thing"),
        ]
        # All argument fragments concatenate to the full JSON.
        assert "".join(d.arguments or "" for d in arg_deltas) == '{"x": 1}'


def test_parse_delta_ignores_invoke_after_end_tag() -> None:
    """A tool call after the section end tag must not be parsed."""
    parser = DeepseekV3_1ToolParser()

    response = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>real<｜tool▁sep｜>{"a": 1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
        # This bogus tool call after the end tag must be ignored.
        '<｜tool▁call▁begin｜>ghost<｜tool▁sep｜>{"b": 2}<｜tool▁call▁end｜>'
    )

    result = parser.parse_delta(response)

    assert result is not None
    names = [d.name for d in result if d.name]
    assert names == ["real"]


def test_parse_delta_multiple_tool_call_blocks() -> None:
    """Streaming supports multiple complete section blocks."""
    uuid_first = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uuid_second = uuid.UUID("22222222-2222-2222-2222-222222222222")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        side_effect=[uuid_first, uuid_second],
    ):
        parser = DeepseekV3_1ToolParser()

        response = (
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"a": 1}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"b": 2}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )

        result = parser.parse_delta(response)

        assert result is not None
        names = [d.name for d in result if d.name]
        assert names == ["foo", "bar"]


def test_parse_delta_no_tool_calls() -> None:
    """Streaming a multi-chunk plain text response yields only content."""
    parser = DeepseekV3_1ToolParser()

    chunks = ["Hello ", "world ", "without tools."]

    all_deltas: list[ParsedToolCallDelta] = []
    for chunk in chunks:
        result = parser.parse_delta(chunk)
        if result:
            all_deltas.extend(result)

    # No name or argument deltas.
    assert all(d.name is None and d.arguments is None for d in all_deltas)
    # Content reassembles to the full text.
    full = "".join(d.content or "" for d in all_deltas)
    assert full == "Hello world without tools."


def test_parse_delta_no_content_after_end_tag() -> None:
    """No content deltas are emitted for text after the section end tag."""
    parser = DeepseekV3_1ToolParser()

    response = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>test<｜tool▁sep｜>{"x": 1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )
    parser.parse_delta(response)

    result = parser.parse_delta("trailing content")

    if result is not None:
        for d in result:
            assert d.content is None


def test_parse_delta_streaming_empty_arguments() -> None:
    """Stream a tool call with empty JSON arguments."""
    fixed_uuid = uuid.UUID("33333333-3333-3333-3333-333333333333")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = DeepseekV3_1ToolParser()

        response = """<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>ping<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"""

        result = parser.parse_delta(response)

        assert result is not None
        arg_deltas = [d for d in result if d.arguments is not None]
        assert len(arg_deltas) == 1
        assert arg_deltas[0].arguments == "{}"


# ---------------------------------------------------------------------------
# DeepSeek V3 (markdown-wrapped) parser tests
# ---------------------------------------------------------------------------

_V3_SINGLE_CALL = (
    "<｜tool▁calls▁begin｜>"
    "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
    '```json\n{"location": "NYC"}\n```'
    "<｜tool▁call▁end｜>"
    "<｜tool▁calls▁end｜>"
)


def test_v3_markdown_single_tool_call_parsing() -> None:
    """V3 markdown grammar: single tool call with ```json fences."""
    parser = DeepseekV3ToolParser()

    result = parser.parse_complete(_V3_SINGLE_CALL)

    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert json.loads(tc.arguments) == {"location": "NYC"}
    assert tc.id.startswith("call_")


def test_v3_markdown_multiple_tool_calls_parsing() -> None:
    """V3 markdown grammar: multiple tool calls in one section."""
    parser = DeepseekV3ToolParser()

    response = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        '```json\n{"location": "NYC"}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n"
        '```json\n{"timezone": "EST"}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    result = parser.parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(result.tool_calls[0].arguments) == {"location": "NYC"}
    assert json.loads(result.tool_calls[1].arguments) == {"timezone": "EST"}


def test_v3_markdown_empty_parameters() -> None:
    """V3 markdown grammar: ``{}`` arguments parse correctly."""
    parser = DeepseekV3ToolParser()

    response = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>ping\n"
        "```json\n{}\n```"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments) == {}


def test_v3_markdown_streaming_single_tool_call() -> None:
    """V3 markdown grammar streams name then chunked arguments."""
    fixed_uuid = uuid.UUID("44444444-4444-4444-4444-444444444444")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = DeepseekV3ToolParser()

        chunks = [
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>",
            "get_weather",
            "\n```json\n",
            '{"loc',
            'ation": "NYC"}',
            "\n```",
            "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:24]}"
        name_deltas = [d for d in all_deltas if d.name]
        assert name_deltas == [
            ParsedToolCallDelta(index=0, id=expected_id, name="get_weather"),
        ]
        # Argument fragments must concatenate to the full JSON, never
        # leaking the closing markdown fence.
        full_args = "".join(d.arguments or "" for d in all_deltas)
        assert full_args == '{"location": "NYC"}'


def test_v3_markdown_streaming_holds_back_closing_fence() -> None:
    """The trailing ``\\n``` `` suffix is held back while streaming."""
    parser = DeepseekV3ToolParser()

    parser.parse_delta("<｜tool▁calls▁begin｜>")
    parser.parse_delta("<｜tool▁call▁begin｜>function<｜tool▁sep｜>")
    parser.parse_delta("foo\n```json\n")
    result = parser.parse_delta('{"x": 1}\n')

    if result is not None:
        emitted = "".join(d.arguments or "" for d in result)
        # The trailing newline is the start of the closing fence —
        # must not be emitted yet.
        assert not emitted.endswith("\n")


def test_v3_markdown_rejects_v31_format() -> None:
    """The V3 parser does not match the V3.1 (raw JSON) grammar."""
    parser = DeepseekV3ToolParser()

    v31_response = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"x": 1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(v31_response)


# ---------------------------------------------------------------------------
# Repo-based resolver tests
# ---------------------------------------------------------------------------


def _write_tokenizer_config(
    dir_path: pathlib.Path, chat_template: object
) -> None:
    """Write a minimal ``tokenizer_config.json`` to ``dir_path``."""
    config: dict[str, object] = {}
    if chat_template is not None:
        config["chat_template"] = chat_template
    (dir_path / "tokenizer_config.json").write_text(json.dumps(config))


def _local_repo(
    path: pathlib.Path, subfolder: str | None = None
) -> HuggingFaceRepo:
    """Construct a local HuggingFaceRepo handle pointing at ``path``."""
    return HuggingFaceRepo(repo_id=str(path), subfolder=subfolder)


def test_resolver_picks_v31_for_raw_json_template(
    tmp_path: pathlib.Path,
) -> None:
    """A template without ``\\`\\`\\`json`` resolves to the V3.1 parser."""
    template = (
        "{{'<｜tool▁call▁begin｜>'+ tool['function']['name'] + "
        "'<｜tool▁sep｜>' + tool['function']['arguments'] + "
        "'<｜tool▁call▁end｜>'}}"
    )
    _write_tokenizer_config(tmp_path, template)

    assert (
        resolve_deepseekv3_tool_parser(_local_repo(tmp_path)) == "deepseekv3_1"
    )


def test_resolver_picks_v3_for_markdown_template(
    tmp_path: pathlib.Path,
) -> None:
    """A template containing ``\\`\\`\\`json`` resolves to the V3 parser."""
    template = (
        "{{'<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + "
        "tool['function']['name'] + '\\n' + '```json' + '\\n' + "
        "tool['function']['arguments'] + '\\n' + '```' + "
        "'<｜tool▁call▁end｜>'}}"
    )
    _write_tokenizer_config(tmp_path, template)

    assert resolve_deepseekv3_tool_parser(_local_repo(tmp_path)) == "deepseekv3"


def test_resolver_defaults_to_v31_without_template(
    tmp_path: pathlib.Path,
) -> None:
    """Missing chat template defaults to the V3.1 parser."""
    _write_tokenizer_config(tmp_path, chat_template=None)

    assert (
        resolve_deepseekv3_tool_parser(_local_repo(tmp_path)) == "deepseekv3_1"
    )


def test_resolver_handles_list_form_template(
    tmp_path: pathlib.Path,
) -> None:
    """``chat_template`` may be a list of ``{name, template}`` entries."""
    template = (
        "{{'<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + "
        "tool['function']['name'] + '\\n' + '```json' + '\\n' + "
        "tool['function']['arguments'] + '\\n' + '```' + "
        "'<｜tool▁call▁end｜>'}}"
    )
    _write_tokenizer_config(
        tmp_path, [{"name": "default", "template": template}]
    )

    assert resolve_deepseekv3_tool_parser(_local_repo(tmp_path)) == "deepseekv3"


def test_resolver_honors_subfolder(tmp_path: pathlib.Path) -> None:
    """Resolver reads ``tokenizer_config.json`` from ``repo.subfolder``."""
    sub = tmp_path / "language_model"
    sub.mkdir()
    # Root-level config points at V3.1; subfolder points at V3. The
    # resolver must pick the subfolder's config.
    _write_tokenizer_config(
        tmp_path, "raw json grammar"
    )  # no fence -> would be V3.1
    _write_tokenizer_config(sub, "wraps args in ```json fences")  # would be V3

    repo = _local_repo(tmp_path, subfolder="language_model")
    assert resolve_deepseekv3_tool_parser(repo) == "deepseekv3"


def test_resolver_returns_default_when_subfolder_missing(
    tmp_path: pathlib.Path,
) -> None:
    """If the subfolder has no tokenizer config, defaults to V3.1."""
    _write_tokenizer_config(tmp_path, "wraps args in ```json fences")
    repo = _local_repo(tmp_path, subfolder="does_not_exist")

    assert resolve_deepseekv3_tool_parser(repo) == "deepseekv3_1"


def test_create_returns_separate_parsers_by_name() -> None:
    """``tool_parsing.create`` produces V3 vs V3.1 from their names."""
    from max.pipelines.lib.tool_parsing import create

    assert isinstance(create("deepseekv3"), DeepseekV3ToolParser)
    assert isinstance(create("deepseekv3_1"), DeepseekV3_1ToolParser)


# ---------------------------------------------------------------------------
# parse_complete / parse_delta parity for multiple section blocks
# ---------------------------------------------------------------------------


def test_parse_complete_v31_multiple_section_blocks() -> None:
    """V3.1 parse_complete must walk every section block, not just the first."""
    parser = DeepseekV3_1ToolParser()

    response = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"a": 1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"b": 2}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    result = parser.parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["foo", "bar"]
    assert json.loads(result.tool_calls[0].arguments) == {"a": 1}
    assert json.loads(result.tool_calls[1].arguments) == {"b": 2}


def test_parse_complete_v3_multiple_section_blocks() -> None:
    """V3 markdown parse_complete must walk every section block too."""
    parser = DeepseekV3ToolParser()

    response = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>foo\n"
        '```json\n{"a": 1}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>bar\n"
        '```json\n{"b": 2}\n```'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    result = parser.parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["foo", "bar"]
    assert json.loads(result.tool_calls[0].arguments) == {"a": 1}
    assert json.loads(result.tool_calls[1].arguments) == {"b": 2}


def test_parse_complete_matches_streaming_for_multiple_blocks() -> None:
    """parse_complete and parse_delta must agree on multi-section responses."""
    response = (
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>foo<｜tool▁sep｜>{"a": 1}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
        "<｜tool▁calls▁begin｜>"
        '<｜tool▁call▁begin｜>bar<｜tool▁sep｜>{"b": 2}<｜tool▁call▁end｜>'
        "<｜tool▁calls▁end｜>"
    )

    completed = DeepseekV3_1ToolParser().parse_complete(response)
    streaming = DeepseekV3_1ToolParser().parse_delta(response)

    assert streaming is not None
    streamed_names = [d.name for d in streaming if d.name]
    assert streamed_names == [tc.name for tc in completed.tool_calls]
