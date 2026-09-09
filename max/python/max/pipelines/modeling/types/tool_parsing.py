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

"""Generic types for tool call parsing.

These types provide a server-agnostic representation of parsed tool calls
that can be translated to specific API schemas (for example, OpenAI) by the
serving layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ParsedToolCall:
    """A parsed tool/function call extracted from model output."""

    id: str
    """The unique identifier for this tool call."""

    name: str
    """The name of the function to call."""

    arguments: str
    """The function arguments as a JSON string."""


@dataclass
class ParsedToolCallDelta:
    """Incremental tool call data for streaming responses.

    Used during streaming to send partial tool call information
    as it becomes available.
    """

    index: int
    """The index of this tool call in the list of tool calls."""

    id: str | None = None
    """The tool call ID, typically sent with the first chunk."""

    name: str | None = None
    """The function name, typically sent with the first chunk."""

    arguments: str | None = None
    """The partial arguments string, streamed incrementally."""

    content: str | None = None
    """Assistant message text before the tool-calls section (streaming only).

    When present, this is normal assistant output and must not be interpreted
    as tool-call arguments. The serving layer maps this to the chat completion
    ``content`` field, not to ``tool_calls``.
    """


@dataclass
class ParsedToolResponse:
    """Result of parsing a complete model response for tool calls."""

    content: str | None = None
    """The text content from the response (before/after tool calls)."""

    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    """The list of :class:`ParsedToolCall` objects extracted from the response."""


class ToolParser(Protocol):
    """Protocol for parsing tool calls from model responses.

    Implementations parse model-specific tool calling formats into generic
    tool call structures. Supports both complete (non-streaming) and
    incremental (streaming) parsing modes.

    Different model architectures use different tool calling formats:

    - Llama models use JSON-based tool calls.
    - Kimi K2.5 uses structural tags like ``<|tool_call_begin|>``.

    The serving layer is responsible for translating these generic types
    to API-specific schemas (for example, OpenAI).
    """

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls.

        Args:
            response: The full model response text.

        Returns:
            A ParsedToolResponse containing any text content and parsed
            tool calls.

        Raises:
            ValueError: If the response cannot be parsed.
        """
        ...

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses an incremental token delta for streaming tool calls.

        Accumulates tokens internally and returns tool call deltas when
        complete or partial tool calls can be extracted.

        Args:
            delta: The incremental token(s) to process.

        Returns:
            - A non-empty list of :class:`ParsedToolCallDelta` when new
              content (tool name, id, or argument bytes) is ready to stream.
            - An empty list ``[]`` when the parser has consumed the token and
              is inside the tool-calls section but has no deltas to emit yet;
              the caller must suppress the raw token from flowing as text
              content.
            - ``None`` when more tokens are needed before anything can be
              emitted (e.g. buffering a potential section-begin marker).

        Note:
            The empty-list suppression state is currently only implemented by
            :class:`~max.pipelines.architectures.kimik2_5.tool_parser.KimiToolParser`.
            Other model tool parsers return only non-empty lists or ``None``
            and should be updated to adopt this convention.
        """
        ...

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        ...

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Provides per-tool parameter schemas for schema-driven arg streaming.

        The router injects each tool's JSON-schema ``parameters`` (keyed by
        tool name) before streaming begins, letting a parser make schema-driven
        decisions about how to emit argument bytes incrementally.

        - Override this (schema required) for parsers whose models emit
          argument values as type-ambiguous tag bodies (XML-style
          ``<name>value</name>``, where a bare scalar's type is not recoverable
          from the wire — ``<n>42</n>`` could be the integer ``42`` or the
          string ``"42"``). The schema decides which parameters stream
          character-by-character (strings) versus are buffered and typed when
          the element closes (numbers, booleans, …). This matters only on the
          streaming path — argument coercion (``coerce_arguments``) runs only
          on the non-streaming complete parse — so the schema is the sole type
          signal available at the moment the parser must commit bytes to the
          wire, and the emitted value must stay a monotonically-growing
          valid-JSON prefix.
        - Leave the null implementation (no schema needed) for raw-JSON
          argument formats: the JSON already carries types and streams
          incrementally through the base byte-diffing.

        Args:
            schemas: Mapping of tool name to its JSON-schema ``parameters``
                object.
        """
        ...
