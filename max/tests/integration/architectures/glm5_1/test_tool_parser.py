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

"""Tests for the GLM-4.5+ tool-call parser and constrained-decoding grammar."""

from __future__ import annotations

import json
from typing import Any

import pytest
from max import _xgrammar as xgr
from max.pipelines.architectures.glm5_1.tool_parser import GlmToolParser
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import ParsedToolResponse

# ---------------------------------------------------------------------------
# Complete parsing
# ---------------------------------------------------------------------------


def test_single_tool_call() -> None:
    parser = GlmToolParser()
    response = (
        "<tool_call>get_weather"
        "<arg_key>location</arg_key><arg_value>San Francisco</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>"
    )
    result = parser.parse_complete(response)
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.id.startswith("call_")
    # Bare strings stay strings.
    assert json.loads(tc.arguments) == {
        "location": "San Francisco",
        "unit": "celsius",
    }


def test_non_string_values_decode_as_json() -> None:
    parser = GlmToolParser()
    response = (
        "<tool_call>calc"
        "<arg_key>n</arg_key><arg_value>42</arg_value>"
        "<arg_key>flag</arg_key><arg_value>true</arg_value>"
        "<arg_key>items</arg_key><arg_value>[1, 2, 3]</arg_value>"
        "</tool_call>"
    )
    result = parser.parse_complete(response)
    assert json.loads(result.tool_calls[0].arguments) == {
        "n": 42,
        "flag": True,
        "items": [1, 2, 3],
    }


def test_content_before_tool_call_is_preserved() -> None:
    parser = GlmToolParser()
    response = "Let me check.<tool_call>ping</tool_call>"
    result = parser.parse_complete(response)
    assert result.content == "Let me check."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "ping"
    assert json.loads(result.tool_calls[0].arguments) == {}


def test_multiple_tool_calls() -> None:
    parser = GlmToolParser()
    response = (
        "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
        "<tool_call>b<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>"
    )
    result = parser.parse_complete(response)
    assert [tc.name for tc in result.tool_calls] == ["a", "b"]
    assert json.loads(result.tool_calls[0].arguments) == {"x": 1}
    assert json.loads(result.tool_calls[1].arguments) == {"y": 2}


def test_quoted_string_value_keeps_quotes() -> None:
    """A JSON-quoted ``<arg_value>`` payload is literal content, not a string
    to un-quote.

    GLM emits string values bare, and the grammar's ``minLength`` bound counts
    the raw payload bytes. Un-quoting a quoted form (``"b"`` -> ``b``,
    ``""`` -> the empty string) would drop it below its length bound, so the
    raw payload must be preserved. Regression for the multi-call value slip
    where a later call wrapped its value in quotes and decoded sub-minLength.
    """
    parser = GlmToolParser()
    result = parser.parse_complete(
        '<tool_call>emit<arg_key>value</arg_key><arg_value>"b"</arg_value></tool_call>'
    )
    assert json.loads(result.tool_calls[0].arguments) == {"value": '"b"'}

    result = parser.parse_complete(
        '<tool_call>emit<arg_key>value</arg_key><arg_value>""</arg_value></tool_call>'
    )
    assert json.loads(result.tool_calls[0].arguments) == {"value": '""'}


def test_string_facet_only_schema_coerces_bare_value_to_string() -> None:
    """A property constrained only by string facets (``minLength`` etc.) with no
    explicit ``"type": "string"`` still coerces a bare non-string value back to
    its string form, matching the grammar's own is-string treatment.
    """
    parser = GlmToolParser()
    schema = {
        "type": "object",
        "properties": {"value": {"minLength": 2}},
        "required": ["value"],
        "additionalProperties": False,
    }
    coerced = parser.coerce_arguments({"value": 1}, schema)
    assert coerced == {"value": "1"}


def test_plain_text_has_no_tool_calls() -> None:
    parser = GlmToolParser()
    result = parser.parse_complete("just a normal answer")
    assert result.content == "just a normal answer"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _collect_stream(
    parser: GlmToolParser, chunks: list[str]
) -> tuple[str, str, str]:
    content = ""
    name = ""
    args = ""
    for chunk in chunks:
        deltas = parser.parse_delta(chunk)
        if not deltas:
            continue
        for d in deltas:
            if d.content:
                content += d.content
            if d.name:
                name = d.name
            if d.arguments:
                args += d.arguments
    return content, name, args


def test_streaming_reassembles_call() -> None:
    parser = GlmToolParser()
    # Split mid-marker to exercise partial-token holdback.
    chunks = [
        "hi <tool_",
        "call>get_weather<arg_key>loc",
        "ation</arg_key><arg_value>Paris",
        "</arg_value></tool_call>",
    ]
    content, name, args = _collect_stream(parser, chunks)
    assert content.strip() == "hi"
    assert name == "get_weather"
    assert json.loads(args) == {"location": "Paris"}


def _string_arg_schema(prop: str = "expression") -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {prop: {"type": "string"}},
        "required": [prop],
    }


@pytest.mark.parametrize(
    "raw, expected",
    [("2", "2"), ("null", "null"), ("true", "true"), ("3.14", "3.14")],
)
def test_streaming_coerces_bare_scalar_to_string(
    raw: str, expected: str
) -> None:
    # A bare value whose text is a JSON scalar must be coerced back to the
    # schema's string type, mirroring the non-streaming coerce_arguments path.
    parser = GlmToolParser()
    parser.set_streaming_tool_schemas({"calc": _string_arg_schema()})
    chunks = [
        f"<tool_call>calc<arg_key>expression</arg_key><arg_value>{raw}",
        "</arg_value></tool_call>",
    ]
    _, name, args = _collect_stream(parser, chunks)
    assert name == "calc"
    assert json.loads(args) == {"expression": expected}


def test_streaming_leaves_non_json_string_untouched() -> None:
    parser = GlmToolParser()
    parser.set_streaming_tool_schemas({"calc": _string_arg_schema()})
    chunks = [
        "<tool_call>calc<arg_key>expression</arg_key><arg_value>2 + 2",
        "</arg_value></tool_call>",
    ]
    _, _, args = _collect_stream(parser, chunks)
    assert json.loads(args) == {"expression": "2 + 2"}


def test_streaming_without_schema_does_not_coerce() -> None:
    # No set_streaming_tool_schemas: bare scalars decode as JSON (unchanged).
    parser = GlmToolParser()
    chunks = [
        "<tool_call>calc<arg_key>expression</arg_key><arg_value>2",
        "</arg_value></tool_call>",
    ]
    _, _, args = _collect_stream(parser, chunks)
    assert json.loads(args) == {"expression": 2}


def test_streaming_coercion_is_prefix_stable_across_deltas() -> None:
    # The value's tokens straddle deltas; the emitted diffs must still
    # concatenate into valid JSON carrying the coerced string.
    parser = GlmToolParser()
    parser.set_streaming_tool_schemas({"calc": _string_arg_schema()})
    chunks = [
        "<tool_call>calc<arg_key>expr",
        "ession</arg_key><arg_value>2",
        "</arg_value></tool_call>",
    ]
    _, _, args = _collect_stream(parser, chunks)
    assert json.loads(args) == {"expression": "2"}


# ---------------------------------------------------------------------------
# Constrained-decoding grammar (xgrammar StructuralTag)
# ---------------------------------------------------------------------------


def _tools(*names: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
        for n in names
    ]


def test_generate_tool_call_grammar_requires_xgrammar_backend() -> None:
    with pytest.raises(InputError, match="xgrammar"):
        GlmToolParser.generate_tool_call_grammar(
            tools=_tools("get_weather"), backend="llguidance"
        )


def test_generate_tool_call_grammar_returns_structural_tag() -> None:
    grammar = GlmToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        backend="xgrammar",
        tool_choice="required",
    )
    tag = xgr.StructuralTag.model_validate_json(grammar)
    assert isinstance(tag, xgr.StructuralTag)


def test_generate_tool_call_grammar_allows_response_format() -> None:
    grammar = GlmToolParser.generate_tool_call_grammar(
        response_format_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        tools=_tools("get_weather"),
        backend="xgrammar",
        tool_choice="auto",
    )
    tag = xgr.StructuralTag.model_validate_json(grammar)
    assert isinstance(tag, xgr.StructuralTag)
    assert '"type":"or"' in grammar.replace(" ", "")


def test_parsers_registered_under_glm45() -> None:
    """The arch wires ``tool_parser="glm45"`` / ``reasoning_parser="glm45"``."""
    # Importing the arch triggers the @register side effects.
    import max.pipelines.architectures.glm5_1.arch  # noqa: F401
    from max.pipelines.architectures.glm5_1.reasoning import GlmReasoningParser
    from max.pipelines.lib.reasoning import (
        get_parser_cls as get_reasoning_cls,
    )
    from max.pipelines.lib.tool_parsing import get_parser_cls as get_tool_cls

    assert get_tool_cls("glm45") is GlmToolParser
    assert get_reasoning_cls("glm45") is GlmReasoningParser
