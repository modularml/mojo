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
"""Turn a model's raw text stream into structured generative-AI chunks.

A text generator emits plain characters. Reasoning models wrap their
chain-of-thought in delimiters and tool-calling models emit marker grammars, so
recovering assistant text, reasoning, and tool calls means parsing that
character stream back apart. That parsing belongs here, next to the pipeline
that knows which model it is serving, rather than in the serving layer.

A response is a sequence of *spans* -- reasoning, assistant text, tool-call
regions -- each bounded by markers in the text, and spans terminate each other:
a tool region opening inside a thinking block ends that block, which is how
Kimi K2.5 leaves reasoning it never closed. Spans also repeat, so one turn can
interleave several of each ("interleaved thinking"). One scanner therefore walks
the stream tracking the span it is inside and the markers that can end it,
holding back partial marker matches across chunk boundaries.

Splitting spans is cascade's job because it knows the model; deciding what a
tool-call region *means* is not. Each region goes verbatim, markers included, to
the model's parser from the MAX pipelines tool-parser registry
(:mod:`max.pipelines.lib.tool_parsing`), so cascade shares the exact call
parsing and argument diffing the main MAX Serve path uses and per-model tool
grammars live in exactly one place.

Everything is parsed incrementally, including for a client that asked for a
whole response rather than a stream -- a non-streaming caller just accumulates
the same chunks. One parse path means the streaming and non-streaming responses
cannot disagree.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from enum import Enum
from typing import Annotated

from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.interfaces.gen_ai import (
    GenAIChunk,
    GenAIReasoningChunk,
    GenAITextChunk,
    GenAIToolCall,
)

# Importing the module registers the "echo" tool parser (used by the echo
# pipeline) so it resolves in whatever process runs this parsing -- including
# the parser worker, which imports this module but not the pipeline.
from max.experimental.cascade.pipelines import echo_tool_parser  # noqa: F401
from max.pipelines.lib.tool_parsing import create as resolve_tool_parser
from max.pipelines.lib.tool_parsing import partial_tag_overlap
from max.pipelines.modeling.types import ToolParser
from pydantic import BaseModel, StringConstraints

# Every delimiter is a marker the scanner searches for, and an empty needle
# matches at every position, so absence is spelled ``None`` rather than "".
_Delimiter = Annotated[str, StringConstraints(min_length=1)]


class ChatParserConfig(BaseModel):
    """How to split one model's raw text into reasoning, content, and tool calls.

    A default-constructed config describes a plain text generator: no reasoning,
    no tool calls, so every character becomes assistant text. A pipeline builds
    the config for its model once, at construction, and hands it to the parser
    worker; it is never visible above the pipeline.
    """

    model_config = {"frozen": True}

    reasoning_start: _Delimiter | None = None
    """Delimiter that opens a reasoning region (e.g. ``"<think>"``).

    Leave ``None`` when the model omits the opening tag because the chat
    template already emitted it; pair that with ``starts_in_reasoning=True``.
    """

    reasoning_end: _Delimiter | None = None
    """Delimiter that closes a reasoning region (e.g. ``"</think>"``).

    Reasoning parsing is enabled only when this is set: without a closing
    delimiter an opened reasoning region could never end, so a model that
    emits one would have its whole turn swallowed as reasoning. A tool region
    opening also ends reasoning, so a model that skips this delimiter on its
    way into a tool call still parses.
    """

    starts_in_reasoning: bool = False
    """Whether the response begins inside a reasoning region (no opening tag)."""

    tool_parser: str | None = None
    """Name of a registered :class:`~max.pipelines.modeling.types.ToolParser`
    (see :mod:`max.pipelines.lib.tool_parsing`) used to extract tool calls.
    ``None`` disables tool parsing.

    The named parser also declares the markers bounding a tool-call region,
    which is how the scanner knows where that span starts and ends."""

    @property
    def reasoning_enabled(self) -> bool:
        """Whether reasoning splitting is configured and can start."""
        return self.reasoning_end is not None and (
            self.starts_in_reasoning or self.reasoning_start is not None
        )


class _Span(Enum):
    """One region of a response, distinguished by how its text is delivered."""

    REASONING = "reasoning"
    CONTENT = "content"
    TOOL = "tool"


# Markers that end a span, each paired with the span it opens.
_Transitions = Mapping[_Span, tuple[tuple[str, _Span], ...]]


def _tool_region(parser: ToolParser) -> tuple[str | None, str | None]:
    """Return the markers bounding one tool-call region, as ``parser`` declares.

    Section-wrapped grammars (Kimi K2.5, MiniMax M2) declare an outer pair that
    brackets every call in a region. Flat grammars (Gemma 4, GLM) declare only
    per-call markers, and a per-call closer is not a region boundary because
    calls repeat back to back, so those regions run to the end of the turn.

    Markers are read by attribute rather than by class so that any parser
    declaring them takes part, mirroring how the serving layer discovers the
    same tags for grammar enforcement. A parser declaring none gets
    ``(None, None)``: its region cannot be delimited from the outside.
    """
    section_begin = getattr(parser, "SECTION_BEGIN", "")
    section_end = getattr(parser, "SECTION_END", "")
    if section_begin and section_end:
        return section_begin, section_end
    return getattr(parser, "CALL_BEGIN", "") or None, None


def _span_transitions(
    config: ChatParserConfig,
    region_start: str | None,
    region_end: str | None,
) -> tuple[_Transitions, _Span]:
    """Build one model's span grammar: how to leave each span, and where to start.

    Every rule the models need falls out of this table. A span ends at its own
    closing delimiter, and equally at the marker that *begins* another span --
    the two together are what let a model open a tool region straight out of a
    thinking block it never closed. Because the table is a graph rather than a
    sequence, spans repeat, so no span is privileged as the first one and
    interleaved turns need no separate rule.
    """
    into_tool = ((region_start, _Span.TOOL),) if region_start else ()
    transitions: dict[_Span, tuple[tuple[str, _Span], ...]] = {
        _Span.CONTENT: into_tool,
        _Span.REASONING: (),
        # A region with no declared closer runs to the end of the turn.
        _Span.TOOL: ((region_end, _Span.CONTENT),) if region_end else (),
    }
    if not config.reasoning_enabled:
        return transitions, _Span.CONTENT

    assert config.reasoning_end is not None
    transitions[_Span.REASONING] = (
        (config.reasoning_end, _Span.CONTENT),
    ) + into_tool
    if config.reasoning_start:
        transitions[_Span.CONTENT] += (
            (config.reasoning_start, _Span.REASONING),
        )
    initial = _Span.REASONING if config.starts_in_reasoning else _Span.CONTENT
    return transitions, initial


class ChatChunkParser:
    """Incrementally parse a text stream into :data:`GenAIChunk` values.

    Walks the stream once, tracking the span it is inside and the markers that
    can end it, so reasoning, assistant text, and tool calls all come out of a
    single pass over a single buffer.
    """

    def __init__(
        self,
        config: ChatParserConfig,
        tool_schemas: Mapping[str, dict[str, object]] | None = None,
        tools_enabled: bool = True,
    ) -> None:
        """Build a parser for one response.

        Args:
            config: The model's response format.
            tool_schemas: Per-tool parameter JSON Schemas, keyed by tool name.
                Schema-driven parsers (XML-style tag grammars) need these to
                stream typed argument bytes; raw-JSON parsers ignore them.
            tools_enabled: Whether the request offered any tools. Tool parsing
                is skipped when it did not, so a stray marker in ordinary prose
                is not misread as a call.
        """
        self._tool_parser = (
            resolve_tool_parser(config.tool_parser)
            if config.tool_parser and tools_enabled
            else None
        )
        if self._tool_parser is not None and tool_schemas:
            self._tool_parser.set_streaming_tool_schemas(dict(tool_schemas))

        region_start, region_end = (
            _tool_region(self._tool_parser)
            if self._tool_parser is not None
            else (None, None)
        )
        # A parser that declares no region start cannot have a span delimited
        # for it, so it owns the whole non-reasoning stream and decides for
        # itself which of that is content -- what it does on the MAX Serve
        # route. Such a model gets no interleaving; nothing marks its regions.
        self._emit_content = (
            self._emit_tool
            if self._tool_parser is not None and region_start is None
            else self._emit_text
        )
        self._transitions, self._span = _span_transitions(
            config, region_start, region_end
        )
        self._buffer = ""

    def feed(self, text: str) -> list[GenAIChunk]:
        """Parse one text delta into its reasoning / text / tool-call chunks."""
        self._buffer += text
        return self._drain(flush=False)

    def finish(self) -> list[GenAIChunk]:
        """Flush text held back across chunk boundaries, ending the open span."""
        return self._drain(flush=True)

    def _drain(self, flush: bool) -> list[GenAIChunk]:
        chunks: list[GenAIChunk] = []
        while (transition := self._next_transition()) is not None:
            index, marker, target = transition
            self._emit(self._buffer[:index], chunks)
            self._buffer = self._buffer[index:]
            # A tool region's own markers belong to that region: the parser
            # matches on the opener, and needs the closer to know the last call
            # finished. So an opener stays at the head of the buffer for the
            # span it introduces, a closer is handed over before the span ends,
            # and a reasoning delimiter -- meaningless downstream -- is dropped.
            if target is not _Span.TOOL:
                if self._span is _Span.TOOL:
                    self._emit_tool(marker, chunks)
                self._buffer = self._buffer[len(marker) :]
            self._span = target

        if flush:
            self._emit(self._buffer, chunks)
            self._buffer = ""
            return chunks

        # No marker can complete in what is buffered, but a suffix of it may
        # still grow into one, so hold that much back rather than emit marker
        # bytes as text.
        holdback = max(
            (
                partial_tag_overlap(self._buffer, marker)
                for marker, _ in self._transitions[self._span]
            ),
            default=0,
        )
        sendable = len(self._buffer) - holdback
        self._emit(self._buffer[:sendable], chunks)
        self._buffer = self._buffer[sendable:]
        return chunks

    def _next_transition(self) -> tuple[int, str, _Span] | None:
        """Find the earliest marker in the buffer that ends the current span."""
        return min(
            (
                (index, marker, target)
                for marker, target in self._transitions[self._span]
                if (index := self._buffer.find(marker)) != -1
            ),
            key=lambda transition: transition[0],
            default=None,
        )

    def _emit(self, text: str, chunks: list[GenAIChunk]) -> None:
        """Deliver one span's text as the chunks that span produces."""
        if not text:
            return
        if self._span is _Span.REASONING:
            chunks.append(GenAIReasoningChunk(text=text))
        elif self._span is _Span.TOOL:
            self._emit_tool(text, chunks)
        else:
            self._emit_content(text, chunks)

    def _emit_text(self, text: str, chunks: list[GenAIChunk]) -> None:
        chunks.append(GenAITextChunk(text=text))

    def _emit_tool(self, text: str, chunks: list[GenAIChunk]) -> None:
        """Stream one piece of a tool region through the model's tool parser."""
        assert self._tool_parser is not None
        tool_deltas = self._tool_parser.parse_delta(text)
        if tool_deltas is None:
            # Plain text: pass through.
            chunks.append(GenAITextChunk(text=text))
            return
        # A non-None result means the parser owns this chunk: [] suppresses the
        # raw text (buffering inside/around a marker); otherwise each delta is
        # either pre-tool content or a tool-call fragment.
        for tool_delta in tool_deltas:
            if tool_delta.content is not None:
                chunks.append(GenAITextChunk(text=tool_delta.content))
            else:
                chunks.append(
                    GenAIToolCall(
                        index=tool_delta.index,
                        id=tool_delta.id,
                        name=tool_delta.name,
                        arguments=tool_delta.arguments,
                    )
                )


class ChatParserWorker(Worker):
    """Parse a model's text stream into structured chunks on the worker pool.

    Chained after detokenization, so the text stream flows worker-to-worker and
    the orchestrator never sits in the middle doing per-token parsing. The
    model's response format is fixed at construction; only the per-request tool
    information varies.
    """

    def __init__(self, config: ChatParserConfig) -> None:
        super().__init__(deploy_hints=["cpu"])
        self.config = config

    @worker_method()
    async def parse_stream(
        self,
        text_iter: AsyncIterable[str],
        tools_enabled: bool,
        tool_schemas: dict[str, dict[str, object]],
    ) -> AsyncIterator[GenAIChunk]:
        """Stream structured chunks parsed from a stream of text deltas."""
        parser = ChatChunkParser(self.config, tool_schemas, tools_enabled)
        async for text in text_iter:
            for chunk in parser.feed(text):
                yield chunk
        for chunk in parser.finish():
            yield chunk
