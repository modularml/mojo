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

"""Tool call parser for Laguna models.

Laguna emits tool calls as XML-ish blocks, with the function name immediately
after the opening tag and arguments as alternating key/value blocks::

    <tool_call>function-name
    <arg_key>argument-key</arg_key>
    <arg_value>value-of-argument-key</arg_value>
    </tool_call>

(This is the format the chat template advertises in the system prompt.)
String argument values are emitted bare; numbers, booleans, arrays, and
objects are emitted as JSON (the template renders ``value | tojson`` for
non-strings). Multiple ``<tool_call>`` blocks may appear back-to-back, and
reasoning/content text may precede the first one.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

from max.pipelines.lib.tool_parsing import register
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
ARG_KEY_OPEN = "<arg_key>"
ARG_KEY_CLOSE = "</arg_key>"
ARG_VALUE_OPEN = "<arg_value>"
ARG_VALUE_CLOSE = "</arg_value>"

_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_ARG_KEY_RE = re.compile(
    re.escape(ARG_KEY_OPEN) + r"(.*?)" + re.escape(ARG_KEY_CLOSE), re.DOTALL
)
_ARG_VALUE_RE = re.compile(
    re.escape(ARG_VALUE_OPEN) + r"(.*?)" + re.escape(ARG_VALUE_CLOSE),
    re.DOTALL,
)


def _decode_value(raw: str) -> object:
    """Decodes a Laguna ``<arg_value>`` payload.

    Mirrors the chat template's value rendering (``value | tojson`` for
    non-strings, bare text for strings): JSON scalars/arrays/objects round-trip
    via ``json.loads``, and the Python-style ``True``/``False``/``None``
    literals the template can emit map back to real values. Anything else is
    treated as a bare string. Without the tool's parameter schema, a string
    that happens to look like one of these literals is decoded as the literal.
    """
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    if stripped == "True":
        return True
    if stripped == "False":
        return False
    if stripped == "None":
        return None
    return stripped


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


def _parse_block(body: str) -> ParsedToolCall | None:
    """Parses one ``<tool_call>...</tool_call>`` body into a tool call.

    The function name is the text between the opening tag and the first
    ``<arg_key>`` (or the whole body when there are no arguments); arguments
    are the paired ``<arg_key>``/``<arg_value>`` blocks in order.
    """
    key_start = body.find(ARG_KEY_OPEN)
    name = (body if key_start < 0 else body[:key_start]).strip()
    if not name:
        return None

    keys = [m.group(1).strip() for m in _ARG_KEY_RE.finditer(body)]
    values = [m.group(1) for m in _ARG_VALUE_RE.finditer(body)]
    args = {
        key: _decode_value(value)
        for key, value in zip(keys, values, strict=False)
    }
    return ParsedToolCall(
        id=_new_call_id(),
        name=name,
        arguments=json.dumps(args, ensure_ascii=False),
    )


def _max_holdback(buffer: str, sentinel: str) -> int:
    """Returns the length of the longest suffix of ``buffer`` that is a proper prefix of ``sentinel``.

    This is how many trailing characters to hold back so a sentinel split
    across deltas is still recognized.
    """
    max_keep = min(len(buffer), len(sentinel) - 1)
    for n in range(max_keep, 0, -1):
        if sentinel.startswith(buffer[-n:]):
            return n
    return 0


@register("laguna")
class LagunaToolParser:
    """Parser for Laguna ``<tool_call>`` blocks.

    Streaming is emitted at per-call granularity: each complete
    ``<tool_call>...</tool_call>`` block produces one delta carrying the
    function name and the full JSON arguments. That is coarser than per-token
    but still valid for OpenAI-style clients, which concatenate argument
    fragments back into a JSON string.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._tool_index: int = -1
        self._seen_tool_call: bool = False

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete model response into tool calls."""
        if TOOL_CALL_OPEN not in response:
            return ParsedToolResponse(content=response, tool_calls=[])

        first_open = response.find(TOOL_CALL_OPEN)
        content_before = response[:first_open].strip() or None

        tool_calls: list[ParsedToolCall] = []
        for block in _TOOL_CALL_BLOCK_RE.finditer(response):
            call = _parse_block(block.group(1))
            if call is not None:
                tool_calls.append(call)

        if not tool_calls:
            raise ValueError(
                "<tool_call> block(s) found but no valid calls parsed from: "
                f"{response}"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Processes one decoded-token delta incrementally.

        Buffers until a full ``<tool_call>...</tool_call>`` block is available,
        then emits it as a single tool-call delta. Content before the first
        block is forwarded; structural text between/after blocks is suppressed.
        """
        if delta:
            self._buffer += delta

        deltas: list[ParsedToolCallDelta] = []
        progressed = True
        while progressed:
            progressed = False
            open_idx = self._buffer.find(TOOL_CALL_OPEN)

            if open_idx < 0:
                # No opener yet. Forward content only before the first call;
                # hold back a possible partial opener at the tail.
                keep = _max_holdback(self._buffer, TOOL_CALL_OPEN)
                emit = self._buffer[: len(self._buffer) - keep]
                if emit and not self._seen_tool_call:
                    deltas.append(ParsedToolCallDelta(index=0, content=emit))
                self._buffer = self._buffer[len(self._buffer) - keep :]
                break

            # Content before the first opener is real assistant content.
            if open_idx > 0 and not self._seen_tool_call:
                deltas.append(
                    ParsedToolCallDelta(
                        index=0, content=self._buffer[:open_idx]
                    )
                )
            self._buffer = self._buffer[open_idx:]

            close_idx = self._buffer.find(TOOL_CALL_CLOSE)
            if close_idx < 0:
                break  # incomplete block; wait for more

            block_end = close_idx + len(TOOL_CALL_CLOSE)
            body = self._buffer[len(TOOL_CALL_OPEN) : close_idx]
            self._buffer = self._buffer[block_end:]
            self._seen_tool_call = True

            call = _parse_block(body)
            if call is not None:
                self._tool_index += 1
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._tool_index,
                        id=call.id,
                        name=call.name,
                        arguments=call.arguments,
                    )
                )
            progressed = True

        if deltas:
            return deltas
        # Suppress raw structural tokens once a tool call has started (or while
        # holding a partial sentinel); otherwise let the router pass content
        # through.
        return [] if self._seen_tool_call or self._buffer else None

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""
        self._tool_index = -1
        self._seen_tool_call = False

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """No-op: this format does not need schema-driven streaming.

        See ``ToolParser.set_streaming_tool_schemas``.
        """
        return
