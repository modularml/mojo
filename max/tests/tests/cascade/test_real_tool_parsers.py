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
"""The real MAX per-model tool parsers plug into the cascade parser worker.

Cascade selects a tool parser by its registry name, the same way the MAX Serve
route does, and drives it over decoded text via ``parse_delta``. This test wires
the *real* Kimi K2.5 and MiniMax M2 parsers (not the echo dummy) through
:class:`ChatChunkParser` on those models' actual wire formats, so the framework
is proven to compose with production parsers.

The second half covers Kimi's span rules -- reasoning and tool-call spans that
terminate each other, and turns that interleave several of both. Those are the
text-domain counterparts of ``KimiK2_5ReasoningParser``'s own tests
(``max/tests/integration/architectures/kimik2_5/test_reasoning.py``), and they
mark where splitting reasoning ahead of the tool parser loses information MAX
Serve keeps.
"""

from __future__ import annotations

import json

import pytest

# Importing each architecture's tool_parser module runs its @register, exactly
# as register_all_models() does in a running server.
from max.experimental.cascade.interfaces.gen_ai import (
    GenAIChunk,
    GenAIReasoningChunk,
    GenAITextChunk,
    GenAIToolCall,
)
from max.experimental.cascade.pipelines.chat_parser import (
    ChatChunkParser,
    ChatParserConfig,
)
from max.pipelines.architectures.kimik2_5.tool_parser import (  # noqa: F401
    KimiToolParser,
)
from max.pipelines.architectures.laguna.tool_parser import (  # noqa: F401
    LagunaToolParser,
)
from max.pipelines.architectures.minimax_m2.tool_parser import (  # noqa: F401
    MinimaxM2ToolParser,
)


def _kimi_section(*calls: tuple[str, str]) -> str:
    """Build one Kimi tool-call section from ``(name, json arguments)`` pairs."""
    body = "".join(
        f"<|tool_call_begin|>functions.{name}:{index}"
        f"<|tool_call_argument_begin|>\n{arguments}\n<|tool_call_end|>\n"
        for index, (name, arguments) in enumerate(calls)
    )
    return f"<|tool_calls_section_begin|>\n{body}<|tool_calls_section_end|>"


KIMI_RESPONSE = _kimi_section(
    ("get_weather", '{"location": "New York", "unit": "fahrenheit"}'),
    ("get_time", '{"timezone": "EST"}'),
)

# Kimi's section markers are 26 and 28 characters, so every size below one
# splits them across feeds, and each case also exercises marker holdback.
CHUNK_SIZES = [1000, 13, 5, 1]

MINIMAX_RESPONSE = (
    "<minimax:tool_call>\n"
    '<invoke name="get_weather">\n'
    '<parameter name="location">New York</parameter>\n'
    '<parameter name="unit">fahrenheit</parameter>\n'
    "</invoke>\n"
    "</minimax:tool_call>"
)

KIMI_CONFIG = ChatParserConfig(
    reasoning_start="<think>",
    reasoning_end="</think>",
    tool_parser="kimik2_5",
)

# A Kimi chat template can prefill the opening ``<think>``, so the response
# itself begins inside reasoning with no opening delimiter of its own.
KIMI_PREFILLED_CONFIG = ChatParserConfig(
    reasoning_end="</think>",
    starts_in_reasoning=True,
    tool_parser="kimik2_5",
)


def _parse(
    text: str, config: ChatParserConfig, chunk_size: int = 1000
) -> list[GenAIChunk]:
    """Feed ``text`` through the parser in ``chunk_size`` slices."""
    parser = ChatChunkParser(config)
    chunks: list[GenAIChunk] = []
    for i in range(0, len(text), chunk_size):
        chunks.extend(parser.feed(text[i : i + chunk_size]))
    chunks.extend(parser.finish())
    return chunks


def _reasoning(chunks: list[GenAIChunk]) -> str:
    return "".join(c.text for c in chunks if isinstance(c, GenAIReasoningChunk))


def _text(chunks: list[GenAIChunk]) -> str:
    return "".join(c.text for c in chunks if isinstance(c, GenAITextChunk))


def _transcript(chunks: list[GenAIChunk]) -> list[tuple[str, str]]:
    """Collapse chunks into the ordered ``(kind, text)`` runs they represent.

    Chunk boundaries follow arrival, not structure -- one span emits many chunks
    when fed a character at a time -- so adjacent chunks of a kind merge back
    together. What survives is the order the spans came in, which is what
    concatenating by kind cannot show. A tool call is keyed by its index so
    separate calls stay separate runs, and carries its function name; arguments
    are asserted separately by :func:`_tool_calls`.
    """
    runs: list[list[str]] = []
    for chunk in chunks:
        if isinstance(chunk, GenAIReasoningChunk):
            kind, text = "reasoning", chunk.text
        elif isinstance(chunk, GenAITextChunk):
            kind, text = "text", chunk.text
        else:
            assert isinstance(chunk, GenAIToolCall)
            kind, text = f"tool:{chunk.index}", chunk.name or ""
        if runs and runs[-1][0] == kind:
            runs[-1][1] += text
        else:
            runs.append([kind, text])
    return [(kind, text) for kind, text in runs]


def _tool_calls(chunks: list[GenAIChunk]) -> dict[int, dict[str, str]]:
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        if not isinstance(chunk, GenAIToolCall):
            continue
        call = calls.setdefault(chunk.index, {"name": "", "arguments": ""})
        if chunk.name:
            call["name"] = chunk.name
        if chunk.arguments:
            call["arguments"] += chunk.arguments
    return calls


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_parser(chunk_size: int) -> None:
    calls = _tool_calls(
        _parse(
            KIMI_RESPONSE, ChatParserConfig(tool_parser="kimik2_5"), chunk_size
        )
    )
    assert calls[0]["name"] == "get_weather"
    assert calls[1]["name"] == "get_time"
    assert json.loads(calls[0]["arguments"]) == {
        "location": "New York",
        "unit": "fahrenheit",
    }
    assert json.loads(calls[1]["arguments"]) == {"timezone": "EST"}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_minimax_parser(chunk_size: int) -> None:
    calls = _tool_calls(
        _parse(
            MINIMAX_RESPONSE,
            ChatParserConfig(tool_parser="minimax_m2"),
            chunk_size,
        )
    )
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_reasoning_and_tool_calls_compose(chunk_size: int) -> None:
    # Kimi emits <think>...</think> reasoning ahead of the tool-call section;
    # the cascade text-level reasoning split feeds the content region to the
    # real Kimi tool parser.
    chunks = _parse(
        "<think>pick tools</think>" + KIMI_RESPONSE, KIMI_CONFIG, chunk_size
    )
    assert _reasoning(chunks) == "pick tools"
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert calls[1]["name"] == "get_time"


# --------------------------------------------------------------------------- #
# Span termination and interleaving
#
# A Kimi turn is a sequence of spans that terminate each other: a reasoning
# span ends at ``</think>`` *or* at ``<|tool_calls_section_begin|>``, the
# section marker stays with the content so the tool parser sees the whole
# section, and one turn may interleave several reasoning and tool-call spans
# ("interleaved thinking"). MAX Serve implements those rules in the token
# domain in ``KimiK2_5ReasoningParser``; the cases below are the text-domain
# equivalents of its own tests, in the same order, and pin cascade's span
# scanner against them so the two routes cannot drift apart.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_bare_call_begin_does_not_end_reasoning(chunk_size: int) -> None:
    # Only the section marker terminates reasoning. ``<|tool_call_begin|>``
    # appears solely inside a section, so on its own it is reasoning text.
    chunks = _parse(
        "<think>maybe <|tool_call_begin|> then</think>Done.",
        KIMI_CONFIG,
        chunk_size,
    )
    assert _reasoning(chunks) == "maybe <|tool_call_begin|> then"
    assert _text(chunks) == "Done."
    assert _tool_calls(chunks) == {}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_think_end_wins_over_tool_section(chunk_size: int) -> None:
    # Both terminators are present; the earlier ``</think>`` ends reasoning,
    # leaving the text before the section as ordinary assistant content.
    chunks = _parse(
        "<think>plan</think>On it." + KIMI_RESPONSE, KIMI_CONFIG, chunk_size
    )
    assert _reasoning(chunks) == "plan"
    assert _text(chunks) == "On it."
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert calls[1]["name"] == "get_time"


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_unterminated_tool_section_streams_partial_call(
    chunk_size: int,
) -> None:
    # A turn cut off mid-section still streams the call it had started: the
    # name is known and the argument bytes seen so far are emitted, with no
    # marker text leaking into assistant content.
    truncated = (
        "<|tool_calls_section_begin|>\n"
        "<|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>\n"
        '{"location": "New'
    )
    chunks = _parse("<think>plan</think>" + truncated, KIMI_CONFIG, chunk_size)
    assert _reasoning(chunks) == "plan"
    assert _text(chunks) == ""
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"].strip() == '{"location": "New'


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_tool_section_ends_unterminated_reasoning(
    chunk_size: int,
) -> None:
    chunks = _parse(
        "<think>need the weather" + KIMI_RESPONSE, KIMI_CONFIG, chunk_size
    )
    assert _reasoning(chunks) == "need the weather"
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert calls[1]["name"] == "get_time"
    assert json.loads(calls[0]["arguments"]) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_tool_section_ends_prefilled_reasoning(chunk_size: int) -> None:
    chunks = _parse(
        "need the weather" + KIMI_RESPONSE,
        KIMI_PREFILLED_CONFIG,
        chunk_size,
    )
    assert _reasoning(chunks) == "need the weather"
    assert len(_tool_calls(chunks)) == 2


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_interleaved_reasoning_between_tool_sections(
    chunk_size: int,
) -> None:
    chunks = _parse(
        "<think>first</think>"
        + KIMI_RESPONSE
        + "<think>second</think>"
        + KIMI_RESPONSE,
        KIMI_CONFIG,
        chunk_size,
    )
    # The spans concatenate: at chunk_size=1 each character is its own chunk,
    # so span boundaries are not recoverable from the chunk list.
    assert _reasoning(chunks) == "firstsecond"
    assert len(_tool_calls(chunks)) == 4


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_spans_come_out_in_any_order(chunk_size: int) -> None:
    # Nothing privileges reasoning as the first span or a tool region as the
    # last, so a turn can run think -> tool -> chat -> think -> tool -> chat.
    # Tool-call indices keep counting across regions.
    chunks = _parse(
        "<think>weather first</think>"
        + KIMI_RESPONSE
        + "Both are in."
        + "<think>now the forecast</think>"
        + _kimi_section(("get_forecast", '{"days": 3}'))
        + "Rain Thursday.",
        KIMI_CONFIG,
        chunk_size,
    )
    assert _transcript(chunks) == [
        ("reasoning", "weather first"),
        ("tool:0", "get_weather"),
        ("tool:1", "get_time"),
        ("text", "Both are in."),
        ("reasoning", "now the forecast"),
        ("tool:2", "get_forecast"),
        ("text", "Rain Thursday."),
    ]
    assert json.loads(_tool_calls(chunks)[2]["arguments"]) == {"days": 3}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_reasoning_reopens_directly_after_tool_section(
    chunk_size: int,
) -> None:
    # Spans can abut with nothing between them: reasoning opens on the very
    # next character after a section closes, and the turn ends mid-thought.
    chunks = _parse(
        "<think>first</think>" + KIMI_RESPONSE + "<think>still thinking",
        KIMI_CONFIG,
        chunk_size,
    )
    assert _transcript(chunks) == [
        ("reasoning", "first"),
        ("tool:0", "get_weather"),
        ("tool:1", "get_time"),
        ("reasoning", "still thinking"),
    ]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_reasoning_delimiters_inside_tool_arguments(
    chunk_size: int,
) -> None:
    # Only the section closer ends a tool region, so argument text that looks
    # like a reasoning delimiter is argument text, not a span boundary.
    arguments = '{"quote": "he wrote </think> then <think> again"}'
    chunks = _parse(
        "<think>quoting</think>" + _kimi_section(("save_note", arguments)),
        KIMI_CONFIG,
        chunk_size,
    )
    assert _transcript(chunks) == [
        ("reasoning", "quoting"),
        ("tool:0", "save_note"),
    ]
    assert json.loads(_tool_calls(chunks)[0]["arguments"]) == json.loads(
        arguments
    )


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_incomplete_section_marker_stays_content(chunk_size: int) -> None:
    # Text that starts down a marker and diverges is content. Held-back bytes
    # have to be released once they cannot complete, in their original place.
    chunks = _parse(
        "A <|tool_calls_section_begi is not a marker." + KIMI_RESPONSE,
        KIMI_CONFIG,
        chunk_size,
    )
    assert _transcript(chunks) == [
        ("text", "A <|tool_calls_section_begi is not a marker."),
        ("tool:0", "get_weather"),
        ("tool:1", "get_time"),
    ]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_empty_spans_emit_nothing(chunk_size: int) -> None:
    # Degenerate spans carry no text, and a zero-length chunk would surface as
    # an empty streamed delta, so they must produce no chunk at all.
    chunks = _parse(
        "<think></think>Hi."
        + "<|tool_calls_section_begin|><|tool_calls_section_end|>"
        + "Bye.",
        KIMI_CONFIG,
        chunk_size,
    )
    assert _transcript(chunks) == [("text", "Hi.Bye.")]
    assert all(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, (GenAITextChunk, GenAIReasoningChunk))
    )


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_parser_without_region_markers_owns_content(chunk_size: int) -> None:
    # Laguna's parser declares no region markers, so its span cannot be
    # delimited from the outside: it receives the whole non-reasoning stream
    # and decides itself what is content, as it does on the MAX Serve route.
    # Reasoning still splits ahead of it, and its calls still parse.
    config = ChatParserConfig(
        reasoning_start="<think>",
        reasoning_end="</think>",
        tool_parser="laguna",
    )
    chunks = _parse(
        "<think>plan</think>On it."
        "<tool_call>get_weather\n"
        "<arg_key>location</arg_key>\n"
        "<arg_value>New York</arg_value>\n"
        "</tool_call>",
        config,
        chunk_size,
    )
    assert _reasoning(chunks) == "plan"
    assert _text(chunks) == "On it."
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"location": "New York"}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_kimi_content_after_tool_section_is_preserved(
    chunk_size: int,
) -> None:
    chunks = _parse(
        "<think>plan</think>Looking those up." + KIMI_RESPONSE + "All set.",
        KIMI_CONFIG,
        chunk_size,
    )
    assert _reasoning(chunks) == "plan"
    assert _text(chunks) == "Looking those up.All set."
    assert len(_tool_calls(chunks)) == 2
