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

import pytest
from max.pipelines.architectures.qwen3_5.tool_parser import Qwen3_5ToolParser
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

# ---------------------------------------------------------------------------
# parse_complete
# ---------------------------------------------------------------------------


def test_parse_complete_single_call() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    result = parser.parse_complete(response)
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    call = result.tool_calls[0]
    assert isinstance(call, ParsedToolCall)
    assert call.id.startswith("call_")
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {
        "location": "Paris",
        "unit": "celsius",
    }


def test_parse_complete_complex_args() -> None:
    """Non-string values (numbers, bools, lists, objects) are JSON-encoded
    by the chat template and should round-trip through the parser."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        '<parameter=locations>\n[{"country": "France", "city": "Paris"}]\n</parameter>\n'
        "<parameter=temp_units>\ncelsius\n</parameter>\n"
        "<parameter=detail>\n3\n</parameter>\n"
        "<parameter=verbose>\ntrue\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    result = parser.parse_complete(response)
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].arguments)
    assert args == {
        "locations": [{"country": "France", "city": "Paris"}],
        "temp_units": "celsius",
        "detail": 3,
        "verbose": True,
    }


def test_parse_complete_multiple_calls() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n<parameter=y>\nhello\n</parameter>\n</function>\n</tool_call>"
    )

    result = parser.parse_complete(response)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "a"
    assert json.loads(result.tool_calls[0].arguments) == {"x": 1}
    assert result.tool_calls[1].name == "b"
    assert json.loads(result.tool_calls[1].arguments) == {"y": "hello"}
    assert result.tool_calls[0].id != result.tool_calls[1].id


def test_parse_complete_with_preamble() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "I should check the weather first.\n\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBerlin\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    result = parser.parse_complete(response)
    assert result.content == "I should check the weather first."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_parse_complete_no_tool_call() -> None:
    parser = Qwen3_5ToolParser()
    result = parser.parse_complete("just a regular response")
    assert result.content == "just a regular response"
    assert result.tool_calls == []


def test_parse_complete_empty_args() -> None:
    parser = Qwen3_5ToolParser()
    response = "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
    result = parser.parse_complete(response)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "ping"
    assert json.loads(result.tool_calls[0].arguments) == {}


def test_parse_complete_python_style_scalars() -> None:
    """The chat template renders bools/None with Python ``str()`` (``True``/
    ``False``/``None``), not JSON. They must decode to real JSON scalars."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=configure>\n"
        "<parameter=enabled>\nTrue\n</parameter>\n"
        "<parameter=verbose>\nFalse\n</parameter>\n"
        "<parameter=fallback>\nNone\n</parameter>\n"
        "<parameter=retries>\n5\n</parameter>\n"
        "<parameter=label>\nhello\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = parser.parse_complete(response)
    assert json.loads(result.tool_calls[0].arguments) == {
        "enabled": True,
        "verbose": False,
        "fallback": None,
        "retries": 5,
        "label": "hello",
    }


def test_parse_complete_multiline_string_value() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=write_note>\n"
        "<parameter=text>\nline one\nline two\nline three\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = parser.parse_complete(response)
    assert json.loads(result.tool_calls[0].arguments) == {
        "text": "line one\nline two\nline three"
    }


def test_parse_complete_invalid_block_raises() -> None:
    parser = Qwen3_5ToolParser()
    # Open tag but no <function=> inside.
    with pytest.raises(ValueError, match=r"no valid calls parsed"):
        parser.parse_complete("<tool_call>\nbroken\n</tool_call>")


# ---------------------------------------------------------------------------
# parse_delta (streaming)
# ---------------------------------------------------------------------------


def _feed(parser: Qwen3_5ToolParser, *deltas: str) -> list[ParsedToolCallDelta]:
    """Feed deltas one at a time and return the flat list of all emissions."""
    out: list[ParsedToolCallDelta] = []
    for d in deltas:
        result = parser.parse_delta(d)
        if result:
            out.extend(result)
    return out


def _split_content_and_calls(
    deltas: list[ParsedToolCallDelta],
) -> tuple[str | None, list[ParsedToolCallDelta]]:
    content_parts: list[str] = []
    tool_deltas: list[ParsedToolCallDelta] = []
    for d in deltas:
        if d.content is not None:
            content_parts.append(d.content)
        else:
            tool_deltas.append(d)
    content = "".join(content_parts) if content_parts else None
    return content, tool_deltas


def test_parse_delta_single_call_one_shot() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(_feed(parser, response))
    assert content is None

    head = tool_deltas[0]
    assert head.id is not None
    assert head.name == "get_weather"
    assert head.arguments == "{"

    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"city": "Paris", "unit": "celsius"}
    assert all(d.index == 0 for d in tool_deltas)


def test_parse_delta_single_char_deltas() -> None:
    """The state machine has to survive sentinels split across deltas."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=fn>\n"
        "<parameter=key>\nvalue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(
        _feed(parser, *list(response))
    )
    assert content is None
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"key": "value"}


def test_parse_delta_pre_call_content_streamed() -> None:
    parser = Qwen3_5ToolParser()
    deltas = _feed(
        parser,
        "Let me think. ",
        "Calling tool now.\n",
        "<tool_call>\n<function=fn>\n",
        "<parameter=x>\n1\n</parameter>\n",
        "</function>\n</tool_call>",
    )
    content, tool_deltas = _split_content_and_calls(deltas)
    assert content is not None
    assert "Let me think. Calling tool now." in content
    assert tool_deltas[0].name == "fn"
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"x": 1}


def test_parse_delta_partial_tag_held_back() -> None:
    """A trailing partial '<tool_' must not leak as content."""
    parser = Qwen3_5ToolParser()

    out1 = parser.parse_delta("hello ")
    assert out1 is not None
    assert len(out1) == 1
    assert out1[0].content == "hello "

    # parse_delta should accumulate tokens; return [] to indicate parser is actively buffering and raw tokens shouldn't be used yet.
    out2 = parser.parse_delta("<tool_")
    assert out2 == []

    out3 = parser.parse_delta(
        "call>\n<function=fn>\n<parameter=k>\nv\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert out3 is not None
    content, tool_deltas = _split_content_and_calls(out3)
    assert content is None
    assert tool_deltas[0].name == "fn"


def test_parse_delta_multiple_calls() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=a>\n"
        "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n"
        "<parameter=y>\nhello\n</parameter>\n</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(_feed(parser, response))
    # The "\n" separating back-to-back calls is structural and must not leak
    # into assistant content.
    assert content is None
    indices = sorted({d.index for d in tool_deltas})
    assert indices == [0, 1]

    by_index: dict[int, str] = {0: "", 1: ""}
    for d in tool_deltas:
        if d.arguments:
            by_index[d.index] += d.arguments
    assert json.loads(by_index[0]) == {"x": 1}
    assert json.loads(by_index[1]) == {"y": "hello"}

    name_deltas = [d for d in tool_deltas if d.name is not None]
    assert [d.name for d in name_deltas] == ["a", "b"]
    assert name_deltas[0].id != name_deltas[1].id


def test_parse_delta_complex_arg_value() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=search>\n"
        '<parameter=filters>\n{"min": 1, "max": 10}\n</parameter>\n'
        '<parameter=tags>\n["a", "b"]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    _, tool_deltas = _split_content_and_calls(_feed(parser, response))
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {
        "filters": {"min": 1, "max": 10},
        "tags": ["a", "b"],
    }


def test_parse_delta_id_emitted_only_on_first_chunk() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=fn>\n"
        "<parameter=a>\n1\n</parameter>\n"
        "<parameter=b>\n2\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, tool_deltas = _split_content_and_calls(_feed(parser, response))
    head = tool_deltas[0]
    rest = tool_deltas[1:]
    assert head.id is not None and head.name == "fn"
    assert all(d.id is None and d.name is None for d in rest)


def test_parse_delta_no_tool_calls_streams_content() -> None:
    parser = Qwen3_5ToolParser()
    deltas = _feed(parser, "Hello, ", "this is ", "a plain response.")
    content, tool_deltas = _split_content_and_calls(deltas)
    assert content == "Hello, this is a plain response."
    assert tool_deltas == []


def test_parse_delta_python_style_scalars() -> None:
    """Streaming decodes Python-style ``True``/``False``/``None`` the same as
    ``parse_complete``."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=configure>\n"
        "<parameter=enabled>\nTrue\n</parameter>\n"
        "<parameter=fallback>\nNone\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, tool_deltas = _split_content_and_calls(_feed(parser, response))
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"enabled": True, "fallback": None}


def test_parse_delta_multiple_calls_no_separator_leak_char_by_char() -> None:
    """Fed one char at a time, the inter-call ``\\n`` separator must not leak
    as content (the parser returns ``[]`` to suppress it, not ``None``)."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=a>\n"
        "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n"
        "<parameter=y>\nhello\n</parameter>\n</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(
        _feed(parser, *list(response))
    )
    assert content is None
    by_index: dict[int, str] = {0: "", 1: ""}
    for d in tool_deltas:
        if d.arguments:
            by_index[d.index] += d.arguments
    assert json.loads(by_index[0]) == {"x": 1}
    assert json.loads(by_index[1]) == {"y": "hello"}


def test_parse_delta_trailing_text_after_call_suppressed() -> None:
    """Per the template, no content follows a function call; any trailing
    text after the final ``</tool_call>`` is suppressed rather than streamed."""
    parser = Qwen3_5ToolParser()
    deltas = _feed(
        parser,
        "<tool_call>\n<function=fn>\n"
        "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
        "\n",
        "trailing junk",
    )
    content, tool_deltas = _split_content_and_calls(deltas)
    assert content is None
    assert tool_deltas[0].name == "fn"


def test_reset_clears_state() -> None:
    parser = Qwen3_5ToolParser()
    parser.parse_delta("<tool_call>\n<function=fn>\n")
    parser.reset()
    out = parser.parse_delta("hi")
    assert out is not None
    content, tool_deltas = _split_content_and_calls(out)
    assert content == "hi"
    assert tool_deltas == []
