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
"""Unit tests for parsing a model's text stream into structured chunks.

These exercise the reasoning/text/tool-call split directly (no runtime). There
is only one parse path -- incremental -- so every case is fed at several chunk
sizes, from the whole string at once down to one character at a time, which is
what covers partial-marker holdback across chunk boundaries.
"""

from __future__ import annotations

import json

import pytest
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

REASONING_CONFIG = ChatParserConfig(
    reasoning_start="<think>", reasoning_end="</think>"
)
TOOL_CONFIG = ChatParserConfig(tool_parser="echo")
FULL_CONFIG = ChatParserConfig(
    reasoning_start="<think>",
    reasoning_end="</think>",
    tool_parser="echo",
)

# Whole string, then progressively finer splits down to one character.
CHUNK_SIZES = [1000, 7, 5, 3, 2, 1]


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


def _text(chunks: list[GenAIChunk]) -> str:
    return "".join(c.text for c in chunks if isinstance(c, GenAITextChunk))


def _reasoning(chunks: list[GenAIChunk]) -> str:
    return "".join(c.text for c in chunks if isinstance(c, GenAIReasoningChunk))


def _tool_calls(chunks: list[GenAIChunk]) -> dict[int, dict[str, str]]:
    """Reassemble streamed tool-call fragments into ``{index: {name, args}}``."""
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


# --------------------------------------------------------------------------- #
# Reasoning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_reasoning_split(chunk_size: int) -> None:
    chunks = _parse(
        "<think>weigh options</think>The answer is 42.",
        REASONING_CONFIG,
        chunk_size,
    )
    assert _reasoning(chunks) == "weigh options"
    assert _text(chunks) == "The answer is 42."
    assert _tool_calls(chunks) == {}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_reasoning_without_opening_tag(chunk_size: int) -> None:
    config = ChatParserConfig(
        reasoning_end="</think>", starts_in_reasoning=True
    )
    chunks = _parse("hidden thoughts</think>hello", config, chunk_size)
    assert _reasoning(chunks) == "hidden thoughts"
    assert _text(chunks) == "hello"


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_no_reasoning_markers_is_all_text(chunk_size: int) -> None:
    chunks = _parse("just content", REASONING_CONFIG, chunk_size)
    assert _reasoning(chunks) == ""
    assert _text(chunks) == "just content"


def test_unterminated_reasoning_flushes_as_reasoning() -> None:
    chunks = _parse("<think>still thinking", REASONING_CONFIG)
    assert _reasoning(chunks) == "still thinking"
    assert _text(chunks) == ""


# --------------------------------------------------------------------------- #
# Tool calls
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_single_tool_call(chunk_size: int) -> None:
    chunks = _parse(
        'Sure!<tool_call>get_weather\n{"city": "SF"}</tool_call>',
        TOOL_CONFIG,
        chunk_size,
    )
    assert _text(chunks) == "Sure!"
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "SF"}


def test_tool_call_opener_carries_id_once() -> None:
    chunks = _parse(
        '<tool_call>fn\n{"a": 1}</tool_call>', TOOL_CONFIG, chunk_size=1
    )
    tool_chunks = [c for c in chunks if isinstance(c, GenAIToolCall)]
    ids = [c.id for c in tool_chunks if c.id is not None]
    assert len(ids) == 1
    assert ids[0].startswith("call_")
    # Only the opener names the tool; the rest are argument fragments.
    assert [c.name for c in tool_chunks if c.name is not None] == ["fn"]


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_multiple_tool_calls(chunk_size: int) -> None:
    chunks = _parse(
        '<tool_call>a\n{"x": 1}</tool_call><tool_call>b\n{"y": 2}</tool_call>',
        TOOL_CONFIG,
        chunk_size,
    )
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "a"
    assert calls[1]["name"] == "b"
    assert json.loads(calls[0]["arguments"]) == {"x": 1}
    assert json.loads(calls[1]["arguments"]) == {"y": 2}


def test_tool_markers_are_not_parsed_without_tool_parser() -> None:
    chunks = _parse("<tool_call>a\n{}</tool_call>", ChatParserConfig())
    assert _text(chunks) == "<tool_call>a\n{}</tool_call>"
    assert _tool_calls(chunks) == {}


def test_tool_markers_are_not_parsed_when_no_tools_offered() -> None:
    # A request that offers no tools disables the parser, so a stray marker in
    # ordinary prose stays ordinary prose.
    parser = ChatChunkParser(TOOL_CONFIG, tools_enabled=False)
    chunks = parser.feed("<tool_call>a\n{}</tool_call>") + parser.finish()
    assert _text(chunks) == "<tool_call>a\n{}</tool_call>"
    assert _tool_calls(chunks) == {}


# --------------------------------------------------------------------------- #
# Reasoning + tool calls together
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_reasoning_then_tool_calls(chunk_size: int) -> None:
    text = (
        "<think>need the weather</think>"
        'On it.<tool_call>get_weather\n{"city": "SF"}</tool_call>'
        '<tool_call>get_time\n{"tz": "PST"}</tool_call>'
    )
    chunks = _parse(text, FULL_CONFIG, chunk_size)
    assert _reasoning(chunks) == "need the weather"
    assert _text(chunks) == "On it."
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "get_weather"
    assert calls[1]["name"] == "get_time"
    assert json.loads(calls[0]["arguments"]) == {"city": "SF"}
    assert json.loads(calls[1]["arguments"]) == {"tz": "PST"}


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
def test_nested_json_arguments_survive_chunking(chunk_size: int) -> None:
    text = (
        "<think>reason</think>prefix "
        '<tool_call>fn\n{"a": [1, 2, 3], "b": {"c": true}}</tool_call>'
    )
    chunks = _parse(text, FULL_CONFIG, chunk_size)
    assert _reasoning(chunks) == "reason"
    assert _text(chunks) == "prefix "
    calls = _tool_calls(chunks)
    assert calls[0]["name"] == "fn"
    assert json.loads(calls[0]["arguments"]) == {
        "a": [1, 2, 3],
        "b": {"c": True},
    }
