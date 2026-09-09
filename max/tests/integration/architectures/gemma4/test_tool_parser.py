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
from typing import Any

import pytest
from max import _xgrammar as xgr
from max.pipelines.architectures.gemma4.tool_parser import Gemma4ToolParser
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolResponse,
)


def _tools(*names: str) -> list[dict[str, Any]]:
    """Build a minimal OpenAI-style tools list from function names."""
    return [{"type": "function", "function": {"name": n}} for n in names]


def _tools_with_schemas(
    schemas: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build tools list with parameter schemas attached."""
    return [
        {
            "type": "function",
            "function": {"name": n, "parameters": s},
        }
        for n, s in schemas.items()
    ]


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call."""
    parser = Gemma4ToolParser()

    response = (
        '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>'
    )

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {"location": "Paris"}


def test_multiple_tool_calls_parsing() -> None:
    """Test parsing multiple tool calls."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:get_weather{location:<|"|>New York<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>Asia/Tokyo<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 2

    tool_call1 = result.tool_calls[0]
    assert tool_call1.name == "get_weather"
    assert json.loads(tool_call1.arguments) == {"location": "New York"}

    tool_call2 = result.tool_calls[1]
    assert tool_call2.name == "get_time"
    assert json.loads(tool_call2.arguments) == {"timezone": "Asia/Tokyo"}

    assert tool_call1.id != tool_call2.id


def test_response_without_tool_calls() -> None:
    """Test parsing a response without tool calls."""
    parser = Gemma4ToolParser()

    response = "This is just a regular response with no tool calls."

    result = parser.parse_complete(response)

    assert result.content == response
    assert len(result.tool_calls) == 0


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = Gemma4ToolParser()

    response = ""

    result = parser.parse_complete(response)

    assert result.content == ""
    assert len(result.tool_calls) == 0


def test_multiple_parameters() -> None:
    """Test parsing tool call with multiple parameters."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:get_weather{location:<|"|>Boston<|"|>,unit:<|"|>fahrenheit<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {
        "location": "Boston",
        "unit": "fahrenheit",
    }


def test_complex_nested_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:search_articles{filters:{categories:[<|"|>AI<|"|>],date_range:{end:<|"|>2023-12-31<|"|>,start:<|"|>2023-01-01<|"|>}},options:{limit:10,sort:<|"|>relevance<|"|>},query:<|"|>machine learning<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["AI"],
        },
        "options": {"limit": 10, "sort": "relevance"},
    }


def test_empty_parameters() -> None:
    """Test parsing tool call with empty parameters."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:get_random_fact{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_multiple_calls_same_function() -> None:
    """Test parsing multiple calls to the same function."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>Berlin<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 3

    for tc in result.tool_calls:
        assert tc.name == "get_weather"

    ids = [tc.id for tc in result.tool_calls]
    assert len(set(ids)) == 3

    locations = [
        json.loads(tc.arguments)["location"] for tc in result.tool_calls
    ]
    assert locations == ["London", "Paris", "Berlin"]


def test_special_characters_in_arguments() -> None:
    """Test handling of special characters in tool arguments."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:execute_code{code:<|"|>print("Hello, World!")<|"|>,language:<|"|>python<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {
        "code": 'print("Hello, World!")',
        "language": "python",
    }


def test_json_escaped_tab_in_delimited_string() -> None:
    """A grammar-escaped backslash-t in a delimited value decodes to a tab."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:f{val:<|"|>' + "\\t" + '<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments)["val"] == "\t"


def test_json_escaped_newline_in_delimited_string() -> None:
    """A grammar-escaped backslash-n in a delimited value decodes to a newline."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:f{val:<|"|>' + "\\n" + '<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments)["val"] == "\n"


def test_json_escaped_backslash_in_delimited_string() -> None:
    """A grammar-escaped double-backslash decodes to one backslash."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:f{val:<|"|>' + "\\\\" + '<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments)["val"] == "\\"


def test_json_escaped_unicode_in_delimited_string() -> None:
    """A grammar-escaped backslash-u escape decodes to the char."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:f{val:<|"|>' + "\\u00e9" + '<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments)["val"] == "\u00e9"


def test_plain_char_in_delimited_string() -> None:
    """A plain char in a delimited value stays unchanged."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:f{val:<|"|>A<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments)["val"] == "A"


def test_array_parameters() -> None:
    """Test parsing tool call with array parameters."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:calculate_sum{numbers:[1,2,3,4,5]}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "calculate_sum"
    assert json.loads(tool_call.arguments) == {"numbers": [1, 2, 3, 4, 5]}


def test_boolean_parameters() -> None:
    """Test parsing tool call with boolean parameters."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:send_notification{message:<|"|>Notification message<|"|>,priority:true}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "send_notification"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == {"message": "Notification message", "priority": True}


def test_tool_call_without_end_tag_raises_error() -> None:
    """Test that a tool call missing its close tag raises ValueError."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:test{key:<|"|>value<|"|>}'

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_empty_tool_calls_section_raises_error() -> None:
    """Test that tool call start without actual calls raises ValueError."""
    parser = Gemma4ToolParser()

    response = "<|tool_call><tool_call|>"

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:test{param:<|"|>value<|"|>}<tool_call|>'

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        tool_call_id = result.tool_calls[0].id
        ids.add(tool_call_id)

    assert len(ids) == 10


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:test{param:<|"|>value<|"|>}<tool_call|>'

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    assert isinstance(tool_call_id, str)
    assert tool_call_id.startswith("call_")
    assert len(tool_call_id) == 29  # "call_" + 24 hex chars


def test_response_structure() -> None:
    """Test that the response structure matches expected format."""
    parser = Gemma4ToolParser()

    response = (
        '<|tool_call>call:calculate{expression:<|"|>2 + 2<|"|>}<tool_call|>'
    )

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert isinstance(tool_call.id, str)


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer."""
    parser = Gemma4ToolParser()

    parser._buffer = "some accumulated data"

    parser.reset()

    assert parser._buffer == ""


def test_parse_delta_accumulates() -> None:
    """Test that streamed deltas accumulate into the right id/name/arguments."""
    parser = Gemma4ToolParser()

    # First chunk opens the tool call — no name yet, so suppression only.
    result1 = parser.parse_delta("<|tool_call>")
    assert result1 == []

    # Second chunk delivers the header (call:test{). Under the deferred-opener
    # contract the opener (id + name) is withheld until the first non-empty
    # args delta, so a header-only chunk (no args yet) yields no opener.
    result2 = parser.parse_delta("call:test{")
    assert result2 == []

    assert parser._buffer == "<|tool_call>call:test{"

    # Feeding enough to complete the call surfaces the opener (id + name)
    # together with the arguments in the same flush.
    result3 = parser.parse_delta('key:<|"|>value<|"|>}<tool_call|>')
    assert result3 is not None

    id_name_deltas = [d for d in result3 if d.name is not None]
    assert len(id_name_deltas) == 1
    assert id_name_deltas[0].name == "test"
    assert id_name_deltas[0].id is not None

    args = "".join(d.arguments for d in result3 if d.arguments is not None)
    assert json.loads(args) == {"key": "value"}


def test_parse_delta_returns_none_outside_tool_section() -> None:
    """parse_delta returns None for plain content with no tool markers."""
    parser = Gemma4ToolParser()

    result = parser.parse_delta("Just some text.")

    # No tool-call markers anywhere — the chunk is plain content. We emit
    # it via a ParsedToolCallDelta(content=...) so the streaming layer
    # routes it to the assistant ``content`` field.
    assert result is not None
    assert len(result) == 1
    assert result[0].content == "Just some text."
    assert result[0].id is None
    assert result[0].name is None


def test_parse_delta_emits_complete_tool_call() -> None:
    """parse_delta surfaces the opener (id + name) and arguments for a
    complete call. Under the deferred-opener contract the opener rides with
    the first non-empty args delta, so we assemble the deltas rather than
    asserting name arrives strictly before any arguments."""
    parser = Gemma4ToolParser()

    assert parser.parse_delta("<|tool_call>") == []

    # Header parses (name known) but arguments stay withheld until the close
    # marker, so no opener is surfaced yet.
    assert parser.parse_delta('call:get_weather{location:<|"|>Paris') == []

    # The close marker completes the call: opener (id + name) and arguments
    # surface together.
    result = parser.parse_delta('<|"|>}<tool_call|>')
    assert result is not None
    assert all(d.index == 0 for d in result)

    id_name_deltas = [d for d in result if d.name is not None]
    assert len(id_name_deltas) == 1
    assert id_name_deltas[0].name == "get_weather"
    assert id_name_deltas[0].id is not None

    args = "".join(d.arguments for d in result if d.arguments is not None)
    assert json.loads(args) == {"location": "Paris"}


def test_parse_delta_emits_content_before_tool_call() -> None:
    """parse_delta emits leading plain content, then surfaces a complete call.

    Gemma-4 treats a call as complete only at the ``<tool_call|>`` close
    marker, so the call body must include it for the opener + arguments to
    surface.
    """
    parser = Gemma4ToolParser()

    result = parser.parse_delta("preamble<|tool_call>")

    assert result is not None
    assert len(result) == 1
    assert result[0].content == "preamble"

    # A complete empty-args call (closed with <tool_call|>) surfaces the
    # opener (id + name) and the empty-object arguments together.
    call_result = parser.parse_delta("call:f{}<tool_call|>")
    assert call_result is not None
    assert any(d.name == "f" for d in call_result)

    args = "".join(d.arguments for d in call_result if d.arguments is not None)
    assert json.loads(args) == {}


def test_streaming_args_openai_contract_full_stream() -> None:
    """OpenAI streaming contract: accumulated arguments across all deltas
    must form a single valid JSON object.

    This is the end-to-end contract the fuzz scenario ``stream_tool_schema``
    validates.  Splits the response token-by-token (as xgrammar constrained
    decoding would emit), collects all ``arguments`` deltas, concatenates them,
    and asserts the result is valid JSON matching the expected args object.
    """
    # Simulate xgrammar token-by-token output: special tokens are single tokens.
    token_chunks = [
        "<|tool_call>",  # CALL_BEGIN — single token
        "call:get_weather{location:",  # header + brace + key prefix
        '<|"|>',  # STRING_DELIM — single token
        "Chicago",  # string value chars
        '<|"|>',  # STRING_DELIM — single token
        ",unit:",  # next key
        '<|"|>',  # STRING_DELIM — single token
        "fahrenheit",  # string value chars
        '<|"|>',  # STRING_DELIM — single token
        "}",  # close args brace
        "<tool_call|>",  # CALL_END — single token
    ]

    parser = Gemma4ToolParser()
    all_args: list[str] = []
    for chunk in token_chunks:
        result = parser.parse_delta(chunk)
        if result:
            for delta in result:
                if delta.arguments is not None:
                    all_args.append(delta.arguments)

    accumulated = "".join(all_args)
    assert accumulated, "accumulated arguments must be non-empty"
    parsed = json.loads(accumulated)
    assert parsed == {"location": "Chicago", "unit": "fahrenheit"}, (
        f"accumulated arguments do not match expected: {accumulated!r}"
    )


def _assemble_streamed_tool_calls(
    parser: Gemma4ToolParser, token_chunks: list[str]
) -> list[dict[str, str]]:
    """Feed ``token_chunks`` through ``parse_delta`` and reconstruct the
    per-index tool calls a streaming client would see.

    Mirrors how the serving layer assembles streamed tool calls: an opener
    delta carries ``id``/``name``; subsequent deltas append ``arguments``.
    Returns one dict per surfaced tool call with keys ``id``, ``name``,
    ``arguments`` (arguments is the concatenation of all argument deltas,
    ``""`` if none arrived).
    """
    calls: dict[int, dict[str, str]] = {}
    for chunk in token_chunks:
        result = parser.parse_delta(chunk)
        if not result:
            continue
        for delta in result:
            if (
                delta.id is None
                and delta.name is None
                and delta.content is None
            ):
                if delta.arguments is None:
                    continue
            if delta.content is not None:
                continue
            call = calls.setdefault(
                delta.index, {"id": "", "name": "", "arguments": ""}
            )
            if delta.id is not None:
                call["id"] = delta.id
            if delta.name is not None:
                call["name"] = delta.name
            if delta.arguments is not None:
                call["arguments"] += delta.arguments
    return [calls[i] for i in sorted(calls)]


def test_streaming_incomplete_call_not_surfaced() -> None:
    """A tool call that opens but is cut off before ``<tool_call|>`` must not
    surface a dangling call with empty ``arguments``.

    Simulates generation halting after the opener: the header ``call:emit{``
    parses (name known) but the close marker never arrives, so no arguments are
    ever emitted. Non-streaming discards such incomplete calls; streaming must
    not leak a ``{id, name, arguments: ""}`` entry to the client.
    """
    parser = Gemma4ToolParser()

    # Runaway with multiple emit calls; the LAST one opens then is cut off.
    token_chunks = [
        "<|tool_call>",
        'call:emit{value:<|"|>a<|"|>}',
        "<tool_call|>",
        "<|tool_call>",
        'call:emit{value:<|"|>b<|"|>}',
        "<tool_call|>",
        "<|tool_call>",
        "call:emit{",  # opens but never closes — generation cut off here
    ]

    streamed = _assemble_streamed_tool_calls(parser, token_chunks)

    dangling = [c for c in streamed if c["arguments"] == ""]
    assert not dangling, (
        "streaming surfaced tool call(s) with empty arguments (dangling "
        f"incomplete call leaked): {dangling}"
    )


def test_streaming_complete_call_surfaces_name_and_args() -> None:
    """A complete tool call streams its opener (id + name) AND arguments.

    Guards that suppressing dangling incomplete calls does not suppress
    legitimate, fully-closed tool calls.
    """
    parser = Gemma4ToolParser()

    token_chunks = [
        "<|tool_call>",
        'call:emit{value:<|"|>hi<|"|>}',
        "<tool_call|>",
    ]

    streamed = _assemble_streamed_tool_calls(parser, token_chunks)

    assert len(streamed) == 1
    call = streamed[0]
    assert call["name"] == "emit"
    assert call["id"]
    assert json.loads(call["arguments"]) == {"value": "hi"}


def test_number_parameters() -> None:
    """Test parsing tool call with integer and float parameters."""
    parser = Gemma4ToolParser()

    response = (
        "<|tool_call>call:set_temperature{value:22,precision:0.5}<tool_call|>"
    )

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {"value": 22, "precision": 0.5}


def test_false_boolean_parameter() -> None:
    """Test parsing tool call with false boolean value."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:configure{enabled:false}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {"enabled": False}


def test_nested_arrays() -> None:
    """Test parsing tool call with nested arrays."""
    parser = Gemma4ToolParser()

    response = (
        "<|tool_call>call:process_matrix{matrix:[[1,2,3],[4,5,6]]}<tool_call|>"
    )

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {"matrix": [[1, 2, 3], [4, 5, 6]]}


def test_function_name_with_underscores() -> None:
    """Test parsing function names with underscores."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:get_current_time{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_current_time"


def test_function_name_with_dots() -> None:
    """Test parsing function names with dots."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:api.v2.search{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "api.v2.search"


def test_function_name_with_hyphens() -> None:
    """Test parsing function names with hyphens."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:get-user-info{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get-user-info"


# Vocab for stepping the matcher. The Gemma delimiters <|"|>, <|tool_call> and
# <tool_call|> are SINGLE tokens (as the real tokenizer emits them) — the
# "gemma" style references <|"|> by token ID with no byte-literal fallback, so
# they must be single vocab entries, not spelled out as separate chars.
_GEMMA_VOCAB = (
    [chr(c) for c in range(32, 127)]
    + ["<|tool_call>", "<tool_call|>", '<|"|>']
    + ["<eos>"]
)
_GEMMA_SPECIALS = ("<|tool_call>", "<tool_call|>", '<|"|>')
_CHAR_ID = {tok: i for i, tok in enumerate(_GEMMA_VOCAB)}


def _gemma_encode(text: str) -> list[int]:
    """Tokenize text, emitting the multi-char Gemma specials as single tokens."""
    out: list[int] = []
    i = 0
    while i < len(text):
        for sp in _GEMMA_SPECIALS:
            if text.startswith(sp, i):
                out.append(_CHAR_ID[sp])
                i += len(sp)
                break
        else:
            out.append(_CHAR_ID[text[i]])
            i += 1
    return out


_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "days": {"type": "integer"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["location"],
}


def _gemma_matcher(
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "required",
) -> xgr.GrammarMatcher:
    """Compile the Gemma 4 tool-call tag and return a fresh matcher."""
    tag = xgr.get_builtin_structural_tag(
        "gemma_4", tools=tools, tool_choice=tool_choice, reasoning=False
    )
    info = xgr.TokenizerInfo(
        _GEMMA_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_CHAR_ID["<eos>"]],
        special_token_ids=[_CHAR_ID[tok] for tok in _GEMMA_SPECIALS],
    )
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)
    return xgr.GrammarMatcher(compiled)


def _accepts(matcher: xgr.GrammarMatcher, text: str) -> bool:
    """Feed text token by token (specials as single tokens); False at first reject."""
    return all(matcher.accept_token(tid) for tid in _gemma_encode(text))


def test_xgrammar_accepts_string_arg() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    good = '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_string_rejects_tool_call_end_marker() -> None:
    """A string value must not admit the ``<tool_call|>`` end marker as content.

    Regression for the runaway-string fail-open: the string-content wildcard
    excluded only the ``<|"|>`` delimiter, so the model could emit the section
    markers inside an unterminated string. The parser breaks at the first
    ``<tool_call|>`` regardless of string state, so it then sees an unterminated
    string and returns no tool calls ("no tool_calls in response"). The grammar
    must forbid the section markers inside a string, forcing the closing
    ``<|"|>`` before the tool call can end.
    """
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    # Open the `location` string value.
    assert _accepts(matcher, '<|tool_call>call:get_weather{location:<|"|>Paris')
    # The tool-call END marker is not string content: it must be rejected while
    # the string is open (the model must emit the closing <|"|> first).
    assert not matcher.accept_token(_CHAR_ID["<tool_call|>"])


def test_xgrammar_string_rejects_tool_call_start_marker() -> None:
    """A string value must not admit the ``<|tool_call>`` start marker either."""
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    assert _accepts(matcher, '<|tool_call>call:get_weather{location:<|"|>Paris')
    assert not matcher.accept_token(_CHAR_ID["<|tool_call>"])


def test_xgrammar_minlength_string_rejects_tool_call_end_marker() -> None:
    """A ``minLength`` string's unbounded tail must not admit the end marker.

    The ``minLength`` path builds a separate string-content body that counts
    characters via a byte char-class rather than an ``ExcludeToken(...)`` tail:
    an exact ``{min}`` char prefix followed by an unbounded byte char-class
    tail. Excluding the section markers is delegated to the xgrammar
    special-token machinery (the markers are registered as ``special_token_ids``
    on ``TokenizerInfo``), so a runaway string past ``minLength`` cannot swallow
    the end marker.
    """
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "minLength": 3}},
        "required": ["s"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    # Open the string and satisfy minLength (3 chars), landing in the unbounded
    # tail. The end marker must be rejected while the string is still open.
    assert _accepts(matcher, '<|tool_call>call:f{s:<|"|>abcd')
    assert not matcher.accept_token(_CHAR_ID["<tool_call|>"])


def test_xgrammar_rejects_json_quoted_key() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    # Gemma keys are bare; a JSON-quoted key must be rejected.
    assert not _accepts(matcher, '<|tool_call>call:get_weather{"location"')


def test_xgrammar_rejects_json_quoted_value() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    # String values must use <|"|> delimiters, not JSON double quotes.
    assert not _accepts(
        matcher, '<|tool_call>call:get_weather{location:"Paris"'
    )


def test_xgrammar_accepts_integer_arg() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    good = (
        '<|tool_call>call:get_weather{location:<|"|>P<|"|>,days:42}<tool_call|>'
    )
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_accepts_enum_value() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    good = (
        '<|tool_call>call:get_weather{location:<|"|>P<|"|>,'
        'unit:<|"|>celsius<|"|>}<tool_call|>'
    )
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_rejects_out_of_enum_value() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA})
    )
    # "kelvin" is not in the enum {celsius, fahrenheit}.
    assert not _accepts(
        matcher,
        '<|tool_call>call:get_weather{location:<|"|>P<|"|>,unit:<|"|>k',
    )


def test_xgrammar_accepts_nested_object() -> None:
    schema = {
        "type": "object",
        "properties": {
            "opts": {
                "type": "object",
                "properties": {"verbose": {"type": "boolean"}},
                "required": ["verbose"],
            }
        },
        "required": ["opts"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"configure": schema}))
    good = "<|tool_call>call:configure{opts:{verbose:true}}<tool_call|>"
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_accepts_array_arg() -> None:
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["tags"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"label": schema}))
    good = '<|tool_call>call:label{tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_accepts_named_tool_choice() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get_weather": _WEATHER_SCHEMA}),
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
    )
    good = '<|tool_call>call:get_weather{location:<|"|>P<|"|>}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_accepts_function_name_with_special_chars() -> None:
    matcher = _gemma_matcher(
        _tools_with_schemas({"get-weather.v2": _WEATHER_SCHEMA})
    )
    good = '<|tool_call>call:get-weather.v2{location:<|"|>P<|"|>}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_generate_tool_call_grammar_requires_xgrammar_backend() -> None:
    with pytest.raises(InputError, match="xgrammar"):
        Gemma4ToolParser.generate_tool_call_grammar(
            tools=_tools("f"), backend="llguidance"
        )


def test_generate_tool_call_grammar_allows_response_format() -> None:
    """With a ``response_format`` schema (tool_choice=auto), the grammar wraps
    the tool-call envelope and the response schema in an ``OrFormat`` so the
    model may emit either a tool call or a schema-conforming JSON response
    (mirrors the Kimi xgrammar path)."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        response_format_schema=_WEATHER_SCHEMA,
        tools=_tools_with_schemas({"get_weather": _WEATHER_SCHEMA}),
        backend="xgrammar",
        tool_choice="auto",
    )
    tag = xgr.StructuralTag.model_validate_json(grammar)
    assert isinstance(tag, xgr.StructuralTag)
    # The response_format branch is present: an OrFormat alternation between the
    # tool-call envelope and the response schema.
    assert '"type":"or"' in grammar.replace(" ", "")


def test_generate_tool_call_grammar_returns_compilable_tag() -> None:
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas({"get_weather": _WEATHER_SCHEMA}),
        backend="xgrammar",
        tool_choice="required",
    )
    tag = xgr.StructuralTag.model_validate_json(grammar)
    assert isinstance(tag, xgr.StructuralTag)
    info = xgr.TokenizerInfo(
        _GEMMA_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_CHAR_ID["<eos>"]],
    )
    compiled = xgr.GrammarCompiler(info).compile_structural_tag(tag)
    assert isinstance(compiled, xgr.CompiledGrammar)


_FREEFORM_SCHEMA = {
    "type": "object",
    "properties": {"meta": {"type": "object", "additionalProperties": True}},
    "required": ["meta"],
}


def test_xgrammar_freeform_object_uses_gemma_string() -> None:
    # additionalProperties:true falls back to the freeform "any" rule; its
    # nested string value must still use <|"|> (gemma_string), not JSON quotes.
    matcher = _gemma_matcher(_tools_with_schemas({"f": _FREEFORM_SCHEMA}))
    good = '<|tool_call>call:f{meta:{k:<|"|>v<|"|>}}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_freeform_object_rejects_json_quoted_value() -> None:
    matcher = _gemma_matcher(_tools_with_schemas({"f": _FREEFORM_SCHEMA}))
    # A JSON-quoted freeform value must be rejected (gemma_any has no "..." arm).
    assert not _accepts(matcher, '<|tool_call>call:f{meta:{k:"v"')


def test_xgrammar_freeform_array_uses_gemma_string() -> None:
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array"}},
        "required": ["items"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    good = '<|tool_call>call:f{items:[<|"|>a<|"|>,1,true]}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_string_pattern_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "[0-9]+"}},
        "required": ["code"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    good = '<|tool_call>call:f{code:<|"|>123<|"|>}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_string_pattern_rejects_nonmatching() -> None:
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "[0-9]+"}},
        "required": ["code"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    # 'a' violates the digit pattern.
    assert not _accepts(matcher, '<|tool_call>call:f{code:<|"|>12a')


def test_xgrammar_string_maxlength_enforced() -> None:
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "maxLength": 3}},
        "required": ["s"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    good = '<|tool_call>call:f{s:<|"|>abc<|"|>}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


def test_xgrammar_string_maxlength_rejects_overlong() -> None:
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "maxLength": 3}},
        "required": ["s"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    # A 4th content character exceeds maxLength=3 and must be rejected.
    assert not _accepts(matcher, '<|tool_call>call:f{s:<|"|>abcd')


def _compile_grammar(tools: list[dict[str, Any]]) -> str:
    """Build and return the Gemma4 tool-call grammar string for the given tools."""
    return Gemma4ToolParser.generate_tool_call_grammar(
        tools=tools,
        backend="xgrammar",
        tool_choice="required",
    )


def _gemma_compiler_for_tests() -> xgr.GrammarCompiler:
    info = xgr.TokenizerInfo(
        _GEMMA_VOCAB,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[_CHAR_ID["<eos>"]],
    )
    return xgr.GrammarCompiler(info)


def _compile_structural_tag(grammar: str) -> xgr.CompiledGrammar:
    """Compile the StructuralTag JSON string produced by generate_tool_call_grammar."""
    tag = xgr.StructuralTag.model_validate_json(grammar)
    return _gemma_compiler_for_tests().compile_structural_tag(tag)


# (a) Non-object roots raise at compile.


def test_non_object_root_string_type_raises() -> None:
    """A tool with parameters type:string raises at xgrammar compile time."""
    grammar = _compile_grammar(_tools_with_schemas({"f": {"type": "string"}}))
    with pytest.raises(Exception):
        _compile_structural_tag(grammar)


def test_non_object_root_scalar_const_raises() -> None:
    """A tool with parameters const:2 (non-object) raises at xgrammar compile time."""
    grammar = _compile_grammar(_tools_with_schemas({"f": {"const": 2}}))
    with pytest.raises(Exception):
        _compile_structural_tag(grammar)


def test_non_object_root_oneof_raises() -> None:
    """A tool with oneOf (unsupported) raises at xgrammar compile time."""
    grammar = _compile_grammar(
        _tools_with_schemas(
            {
                "f": {
                    "oneOf": [
                        {"type": "object"},
                        {"type": "object"},
                    ]
                }
            }
        )
    )
    with pytest.raises(Exception):
        _compile_structural_tag(grammar)


# (b) Empty schema and typed-object schemas compile without error.


def test_empty_schema_compiles() -> None:
    """A tool with an empty parameters schema ({}) compiles successfully."""
    grammar = _compile_grammar(_tools_with_schemas({"f": {}}))
    compiled = _compile_structural_tag(grammar)
    assert isinstance(compiled, xgr.CompiledGrammar)


def test_typed_object_schema_compiles() -> None:
    """A standard typed-object parameters schema compiles successfully."""
    grammar = _compile_grammar(
        _tools_with_schemas(
            {
                "get_weather": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                }
            }
        )
    )
    compiled = _compile_structural_tag(grammar)
    assert isinstance(compiled, xgr.CompiledGrammar)


# (c) An unsupported keyword (multipleOf) raises at compile.


def test_unsupported_keyword_multipleof_raises() -> None:
    """A tool schema containing multipleOf raises at xgrammar compile time."""
    grammar = _compile_grammar(
        _tools_with_schemas(
            {
                "f": {
                    "type": "object",
                    "properties": {"n": {"type": "integer", "multipleOf": 2}},
                }
            }
        )
    )
    with pytest.raises(Exception):
        _compile_structural_tag(grammar)


# (d) minLength bounds the string body.


def test_minlength_rejects_short_string() -> None:
    """A string shorter than minLength is rejected at decode time."""
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "minLength": 3}},
        "required": ["s"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    # "ab" is 2 chars — too short, the close delimiter must be rejected.
    assert not _accepts(
        matcher, '<|tool_call>call:f{s:<|"|>ab<|"|>}<tool_call|>'
    )


def test_minlength_accepts_exact_length() -> None:
    """A string of exactly minLength is accepted."""
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "minLength": 3}},
        "required": ["s"],
    }
    matcher = _gemma_matcher(_tools_with_schemas({"f": schema}))
    good = '<|tool_call>call:f{s:<|"|>abc<|"|>}<tool_call|>'
    assert _accepts(matcher, good)
    assert matcher.is_completed()


# (e) Non-scalar const is enforced in gemma form; scalar const compiles.


def test_non_scalar_const_object_enforced() -> None:
    """A non-scalar (object) const value is rendered in gemma form and enforced:
    the conforming rendering is accepted and a divergent value is rejected."""
    schema = {"type": "object", "properties": {"c": {"const": {"a": 1}}}}
    m = _gemma_matcher(_tools_with_schemas({"f": schema}))
    assert _accepts(m, "<|tool_call>call:f{c:{a:1}}<tool_call|>")
    m2 = _gemma_matcher(_tools_with_schemas({"f": schema}))
    assert not _accepts(m2, "<|tool_call>call:f{c:{a:2")


def test_scalar_const_compiles() -> None:
    """A tool schema with a scalar string const compiles successfully."""
    grammar = _compile_grammar(
        _tools_with_schemas(
            {
                "f": {
                    "type": "object",
                    "properties": {"mode": {"const": "fast"}},
                }
            }
        )
    )
    compiled = _compile_structural_tag(grammar)
    assert isinstance(compiled, xgr.CompiledGrammar)
