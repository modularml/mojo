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

"""Registry and reusable base class for tool-call parsers.

Two pieces live here:

* :func:`register` / :func:`create` — the registry mapping names to
  :class:`max.pipelines.modeling.types.ToolParser` implementations.
* :class:`StructuralTagToolParser` — a base class that captures the
  shared structure of "section-marker" tool-call grammars.
"""

from __future__ import annotations

import logging
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
    ToolParser,
)

__all__ = ["get_parser_cls"]

logger = logging.getLogger(__name__)


def name_from_tool(tool: Mapping[str, Any]) -> str:
    """Extracts the function name from an OpenAI-style tool dict."""
    return tool["function"]["name"]


def maybe_name_from_tool(tool: Mapping[str, Any]) -> str | None:
    """Extracts the function name from an OpenAI-style tool dict, or ``None``."""
    func = tool.get("function")
    if isinstance(func, dict):
        name = func.get("name")
        if isinstance(name, str) and name:
            return name
    return None


def names_from_tools(
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[str] | None:
    """Extracts function names from an OpenAI-style tools list.

    Returns ``None`` when *tools* is ``None`` or empty, matching the
    semantics expected by ``generate_tool_call_grammar`` implementations.
    """
    if not tools:
        return None
    names = [
        name for t in tools if (name := maybe_name_from_tool(t)) is not None
    ]
    return names or None


_JSON_TYPE_TO_GRAMMAR_RULE: dict[str, str] = {
    "string": "string_val",
    "number": "number_val",
    "integer": "integer_val",
    "boolean": "bool_val",
    "array": "array_val",
    "object": "object_val",
    "null": "null_val",
}


def grammar_rule_for_json_type(json_type: str, default: str = "value") -> str:
    """Maps a JSON Schema type name to its Lark grammar rule name."""
    return _JSON_TYPE_TO_GRAMMAR_RULE.get(json_type, default)


_TOOL_PARSERS: dict[str, type[ToolParser]] = {}


def register(name: str) -> Callable[[type[ToolParser]], type[ToolParser]]:
    """Class decorator that registers a ToolParser under the given name."""

    def decorator(cls: type[ToolParser]) -> type[ToolParser]:
        _TOOL_PARSERS[name] = cls
        return cls

    return decorator


def create(name: str) -> ToolParser:
    """Look up a registered parser by name and instantiate it."""
    cls = _TOOL_PARSERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tool parser: {name!r}. Available: {sorted(_TOOL_PARSERS)}"
        )
    return cls()


def get_parser_cls(name: str | None) -> type[ToolParser] | None:
    """Look up a registered parser *class* by name without instantiating it.

    Returns ``None`` if no parser is registered under that name, or if
    ``name`` is ``None``. Useful for callers that need to read class-level
    attributes (e.g., structural tag delimiters).
    """
    if name is None:
        return None
    return _TOOL_PARSERS.get(name)


# ---------------------------------------------------------------------------
# Shared base class for "section-marker" tool-call grammars
# ---------------------------------------------------------------------------


def partial_tag_overlap(text: str, tag: str) -> int:
    """Returns the length of partial overlap between end of text and start of tag.

    Detects when the end of accumulated text might be the beginning of a
    marker tag, so callers can hold back those bytes to avoid leaking
    partial markers into emitted content.

    Args:
        text: The accumulated text to check.
        tag: The marker tag to check for partial overlap.

    Returns:
        Number of trailing characters in ``text`` that match the start of ``tag``.
    """
    max_overlap = min(len(text), len(tag) - 1)
    for i in range(max_overlap, 0, -1):
        if text[-i:] == tag[:i]:
            return i
    return 0


_TOOL_CALL_ID_LENGTH = 24


def generate_call_id() -> str:
    """Generates a unique ``call_``-prefixed tool call ID."""
    return f"call_{uuid.uuid4().hex[:_TOOL_CALL_ID_LENGTH]}"


def escape_for_lark_string(s: str) -> str:
    """Escapes a string for use inside a Lark double-quoted terminal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def get_token_id(tokenizer: Any, token: str) -> int | None:
    """Resolve a single token string to its ID, or ``None`` if unknown."""
    delegate = getattr(tokenizer, "delegate", tokenizer)
    convert = getattr(delegate, "convert_tokens_to_ids", None)
    if convert is None:
        return None

    unk_id = getattr(delegate, "unk_token_id", None)
    tid = convert(token)
    if isinstance(tid, int) and tid != unk_id:
        return tid
    return None


def resolve_lark_token_reference(token_id: int) -> str:
    """Format a token ID as llguidance's ``<[N]>`` reference."""
    return f"<[{token_id}]>"


def canonicalize_lark_rule_name(s: str) -> str:
    """Convert *s* to a string safe for use as a Lark rule name.

    Lark rule names must be lowercase (uppercase starts a terminal).
    """
    return re.sub(r"[^a-z0-9_]", "_", s.lower())


@dataclass
class StreamingToolCallState:
    """State for a single tool call being streamed.

    Tracks the identifier, function name, and the arguments string that
    has already been emitted for one tool call so that successive
    ``parse_delta`` calls only yield new content.
    """

    id: str = ""
    name: str = ""
    arguments_sent: str = ""
    opener_sent: bool = False


@dataclass
class StreamingState:
    """Internal state for streaming tool call parsing.

    Accumulated across successive ``parse_delta`` calls within a single
    streaming response. Call ``reset`` on the parser to clear this state
    before starting a new response.
    """

    sent_content_idx: int = 0
    tool_calls: list[StreamingToolCallState] = field(default_factory=list)


class StructuralTagToolParser(ABC):
    """Abstract base for tool parsers that use structural tag markers.

    Supports two layouts:

    * **Section-wrapped** (e.g. Kimi K2.5, DeepSeek V3, MiniMax M2):
      an outer ``SECTION_BEGIN``/``SECTION_END`` pair wrapping one or more
      inner ``CALL_BEGIN``/``CALL_END`` pairs.
    * **Flat** (e.g. Gemma 4): only ``CALL_BEGIN``/``CALL_END`` are set;
      ``SECTION_BEGIN``/``SECTION_END`` are left empty (the default).
      The base class scans for call pairs directly with no nesting.

    Subclasses implement a small number of hooks for grammar-specific
    body splitting, header parsing, and argument formatting. Everything
    else — buffer accumulation, content-delta extraction with
    partial-marker holdback, argument diffing, and ``reset`` — is shared.
    """

    # Marker constants — subclasses override.
    ENFORCE_TOOL_REGION_TO_EOS: ClassVar[bool] = False
    """Keep grammar enforcement on after ``SECTION_END``.

    For models whose chat template terminates a tool-call turn immediately
    after its single section (e.g. MiniMax-M3), leaving enforcement on lets
    the completed grammar mask everything but EOS, so the turn cannot
    continue past the section. The default ``False`` keeps the
    section-bounded region: enforcement flips off at ``SECTION_END`` and
    the model may emit free text afterwards.
    """

    SECTION_BEGIN: ClassVar[str] = ""
    SECTION_END: ClassVar[str] = ""
    CALL_BEGIN: ClassVar[str] = ""
    CALL_END: ClassVar[str] = ""

    def __init__(self) -> None:
        self._buffer: str = ""
        self._state: StreamingState = StreamingState()

    @property
    def _start_marker(self) -> str:
        """The marker that opens the tool-call region (section or call)."""
        return self.SECTION_BEGIN or self.CALL_BEGIN

    # ----- Public ToolParser protocol -----------------------------------

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls.

        In section-wrapped mode, walks every ``SECTION_BEGIN`` …
        ``SECTION_END`` pair. In flat mode (no section markers), passes
        everything from the first ``CALL_BEGIN`` onward to
        :meth:`_parse_complete_section`. Content before the first marker
        is preserved.
        """
        start_marker = self._start_marker
        first_marker_idx = response.find(start_marker)
        if first_marker_idx == -1:
            return ParsedToolResponse(content=response, tool_calls=[])

        content_before: str | None = None
        if first_marker_idx > 0:
            content_before = response[:first_marker_idx].strip() or None

        tool_calls: list[ParsedToolCall] = []

        if not self.SECTION_BEGIN:
            # Flat mode: everything from the first CALL_BEGIN onward is
            # the tool-call region (call markers included).
            tool_calls.extend(
                self._parse_complete_section(response[first_marker_idx:])
            )
        else:
            cursor = first_marker_idx
            while True:
                section_start = response.find(self.SECTION_BEGIN, cursor)
                if section_start == -1:
                    break
                body_start = section_start + len(self.SECTION_BEGIN)
                section_end = response.find(self.SECTION_END, body_start)
                if section_end == -1:
                    tool_calls.extend(
                        self._parse_complete_section(response[body_start:])
                    )
                    break
                tool_calls.extend(
                    self._parse_complete_section(
                        response[body_start:section_end]
                    )
                )
                cursor = section_end + len(self.SECTION_END)

        if not tool_calls:
            # Do not embed the raw model output -- it may contain PII and is
            # surfaced to logs. Report only the length for debugging.
            raise ValueError(
                "Tool calls section found but no valid tool calls parsed "
                f"(section length: {len(response) - first_marker_idx} chars)"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses incremental deltas for streaming tool calls.

        Accumulates tokens in an internal buffer and emits tool call
        deltas as complete or partial calls become extractable. Uses
        argument diffing so each chunk carries only newly-arrived bytes.

        Returns:
            - A non-empty list of :class:`ParsedToolCallDelta` when new
              content (tool name, id, or argument bytes) is ready to
              stream.
            - An empty list ``[]`` when the parser is actively handling
              this chunk — either inside the tool-calls section or
              holding back bytes that partially match a section marker.
              The caller must suppress raw tokens so they don't leak as
              assistant content.
            - ``None`` when the chunk is plain text with no marker
              activity; the caller should pass raw decoded tokens
              through as assistant content.
        """
        self._buffer += delta
        deltas: list[ParsedToolCallDelta] = []

        try:
            marker_pos = self._buffer.find(self._start_marker)

            content_delta = self._extract_content_delta(marker_pos)
            if content_delta:
                deltas.append(
                    ParsedToolCallDelta(index=0, content=content_delta)
                )

            tool_call_bodies = self._extract_tool_call_bodies(marker_pos)

            for i, (body, is_complete) in enumerate(tool_call_bodies):
                while i >= len(self._state.tool_calls):
                    self._state.tool_calls.append(StreamingToolCallState())

                tc_state = self._state.tool_calls[i]

                header, args = self._split_tool_call_body(body, is_complete)

                if not tc_state.name and header is not None:
                    tool_id, tool_name = self._extract_tool_id_and_name(header)
                    if tool_id and tool_name:
                        tc_state.id = tool_id
                        tc_state.name = tool_name

                if args is not None:
                    args_str = self._format_args_for_streaming(
                        args, is_complete
                    )
                    if args_str:
                        args_diff = self._compute_args_diff(i, args_str)
                        if args_diff:
                            if (
                                not tc_state.opener_sent
                                and tc_state.id
                                and tc_state.name
                            ):
                                deltas.append(
                                    ParsedToolCallDelta(
                                        index=i,
                                        id=tc_state.id,
                                        name=tc_state.name,
                                    )
                                )
                                tc_state.opener_sent = True
                            deltas.append(
                                ParsedToolCallDelta(
                                    index=i, arguments=args_diff
                                )
                            )

            # Returning None indicates nothing happening; router passes raw tokens as content.
            # Return [] to indicate this chunk is actively buffering (e.g. partial marker); suppress raw
            # tokens so they don't leak as content.
            in_tool_section = marker_pos != -1
            has_holdback = (
                not in_tool_section
                and len(self._buffer) > self._state.sent_content_idx
            )
            return (
                deltas if (deltas or in_tool_section or has_holdback) else None
            )

        except Exception:
            logger.exception("Error parsing streaming tool call delta")
            raise

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""
        self._state = StreamingState()

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """No-op: these parsers do not need schema-driven streaming.

        Schema-driven parsers (e.g. MiniMax-M3) override this; see
        ``ToolParser.set_streaming_tool_schemas`` for when that is required.
        """
        return

    # ----- Hooks (subclasses override) ----------------------------------

    @abstractmethod
    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        """Returns the ``ParsedToolCall``s found in a single section body.

        Called for each ``SECTION_BEGIN`` ... ``SECTION_END`` block. The
        ``tool_section`` argument is the text *between* the markers.
        Implementations typically run a grammar-specific regex over the
        section, extract name+args from each match, and produce one
        ``ParsedToolCall`` per valid match (skipping invalid entries with
        empty names rather than raising).
        """

    @abstractmethod
    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits one ``CALL_BEGIN``/``CALL_END`` body into (header, args).

        ``body`` is the text between the call's start and end markers
        (or up to the buffer end if the call is still streaming, with
        any partial closing marker already trimmed). ``is_complete`` is
        ``True`` when the closing marker has been seen.

        Return ``(None, None)`` when the body does not yet contain enough
        information to split — for example, when the separator hasn't
        landed yet. The header should be the substring that
        :meth:`_extract_tool_id_and_name` can parse into a function name;
        the args should be the (possibly still-growing) argument text.
        """

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        """Parses a header string into ``(tool_id, function_name)``.

        Default implementation treats the entire header (after stripping
        whitespace) as the function name and generates a fresh
        ``call_``-prefixed id. Subclasses override for grammars that
        embed an index or wrap the name in attribute quoting.
        """
        name = header.strip()
        if not name:
            return None, None
        return generate_call_id(), name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Returns the argument representation to diff for streaming.

        Default implementation passes ``args_text`` through unchanged,
        which is correct when the grammar already emits arguments as raw
        JSON (Kimi, DeepSeek). Subclasses whose grammar emits structured
        arguments (for example, XML ``<parameter>`` blocks) override
        this to assemble a monotonically growing JSON string from the
        completed parameter elements seen so far.
        """
        return args_text

    # ----- Shared internals --------------------------------------------

    def _extract_content_delta(self, marker_pos: int) -> str | None:
        """Returns unsent text before the first tool-call marker, if any.

        Holds back any trailing suffix that partially matches the start
        marker so that incremental tokens never leak marker bytes as
        assistant content.
        """
        if marker_pos == -1:
            overlap = partial_tag_overlap(self._buffer, self._start_marker)
            sendable_idx = len(self._buffer) - overlap
        else:
            sendable_idx = marker_pos

        if sendable_idx > self._state.sent_content_idx:
            content = self._buffer[self._state.sent_content_idx : sendable_idx]
            self._state.sent_content_idx = sendable_idx
            return content
        return None

    def _extract_tool_call_bodies(
        self, marker_pos: int
    ) -> list[tuple[str, bool]]:
        """Extracts call bodies from the buffer.

        In section-wrapped mode, walks every ``SECTION_BEGIN`` block,
        then within each block iterates over ``CALL_BEGIN`` …
        ``CALL_END`` pairs. In flat mode (no section markers), scans
        for ``CALL_BEGIN`` … ``CALL_END`` pairs directly.

        Returns ``(body, is_complete)`` tuples where ``is_complete``
        indicates whether the closing call marker has been seen.
        """
        if marker_pos == -1:
            return []

        if not self.SECTION_BEGIN:
            return self._extract_flat_call_bodies(marker_pos)

        results: list[tuple[str, bool]] = []
        search_pos = marker_pos + len(self.SECTION_BEGIN)

        while True:
            section_end_pos = self._buffer.find(self.SECTION_END, search_pos)
            scan_end = (
                section_end_pos if section_end_pos != -1 else len(self._buffer)
            )

            found_incomplete = False
            while True:
                start = self._buffer.find(self.CALL_BEGIN, search_pos, scan_end)
                if start == -1:
                    break

                body_start = start + len(self.CALL_BEGIN)
                end = self._buffer.find(self.CALL_END, body_start, scan_end)

                if end != -1:
                    results.append((self._buffer[body_start:end], True))
                    search_pos = end + len(self.CALL_END)
                else:
                    body = self._buffer[body_start:scan_end]
                    overlap = partial_tag_overlap(body, self.CALL_END)
                    if overlap:
                        body = body[:-overlap]
                    results.append((body, False))
                    found_incomplete = True
                    break

            if found_incomplete or section_end_pos == -1:
                break

            next_section_pos = self._buffer.find(
                self.SECTION_BEGIN, section_end_pos + len(self.SECTION_END)
            )
            if next_section_pos == -1:
                break
            search_pos = next_section_pos + len(self.SECTION_BEGIN)

        return results

    def _extract_flat_call_bodies(
        self, marker_pos: int
    ) -> list[tuple[str, bool]]:
        """Extracts call bodies without section wrappers.

        Scans for ``CALL_BEGIN`` … ``CALL_END`` pairs directly starting
        from ``marker_pos``. Holds back any partial ``CALL_END`` suffix
        on the last incomplete call.
        """
        results: list[tuple[str, bool]] = []
        search_pos = marker_pos

        while True:
            start = self._buffer.find(self.CALL_BEGIN, search_pos)
            if start == -1:
                break

            body_start = start + len(self.CALL_BEGIN)
            end = self._buffer.find(self.CALL_END, body_start)

            if end != -1:
                results.append((self._buffer[body_start:end], True))
                search_pos = end + len(self.CALL_END)
            else:
                body = self._buffer[body_start:]
                overlap = partial_tag_overlap(body, self.CALL_END)
                if overlap:
                    body = body[:-overlap]
                results.append((body, False))
                break

        return results

    def _compute_args_diff(self, index: int, args: str) -> str | None:
        """Returns the unsent suffix of ``args`` for tool call ``index``.

        Advances the per-call ``arguments_sent`` cursor so successive
        calls only return newly-appended bytes. Returns ``None`` when
        the cumulative args string has not grown since the last call.
        """
        tc_state = self._state.tool_calls[index]
        prev_len = len(tc_state.arguments_sent)

        if len(args) <= prev_len:
            return None

        diff = args[prev_len:]
        tc_state.arguments_sent = args
        return diff
