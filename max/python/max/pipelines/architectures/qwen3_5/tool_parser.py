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

"""Tool call parser for Qwen 3.5 / 3.6 models.

Qwen 3.5/3.6 emits tool calls as XML-ish blocks::

    <tool_call>
    <function=function_name>
    <parameter=arg_name>
    arg_value
    </parameter>
    <parameter=arg2_name>
    {"key": "value"}
    </parameter>
    </function>
    </tool_call>

Strings are emitted bare; numbers, booleans, arrays, and objects are emitted
as JSON. Multiple ``<tool_call>`` blocks may appear back-to-back. Reasoning
text may precede the first tool call.

Streaming emits per-parameter argument fragments (``{"k":v``, ``,"k":v``,
``}``) — finer than per-call, coarser than per-token, and good enough for
OpenAI-style clients that just concatenate the fragments back into a JSON
string.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

from max.pipelines.lib.tool_parsing import register
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN_PREFIX = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN_PREFIX = "<parameter="
PARAMETER_CLOSE = "</parameter>"

_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_FUNCTION_NAME_RE = re.compile(
    re.escape(FUNCTION_OPEN_PREFIX) + r"([^>]+)>",
)
_PARAMETER_RE = re.compile(
    re.escape(PARAMETER_OPEN_PREFIX)
    + r"([^>]+)>\n?(.*?)\n?"
    + re.escape(PARAMETER_CLOSE),
    re.DOTALL,
)


def _decode_value(raw: str) -> object:
    """Decode a Qwen parameter value.

    Mirrors the chat template's value rendering (``args_value | tojson`` for
    mappings/sequences, ``args_value | string`` for everything else): lists
    and objects arrive as JSON, while numbers, booleans, and ``None`` arrive
    as Python ``str()`` output (``3``, ``True``, ``False``, ``None``) — *not*
    JSON (``true``/``false``/``null``).

    Try ``json.loads`` first (handles numbers, lists, objects, and any
    JSON-style scalars), then recognize the Python-style ``True``/``False``/
    ``None`` literals the template emits, and finally fall back to the raw
    string. This parser has no access to the tool's parameter schema, so a
    string value that happens to look like one of these literals is
    indistinguishable from the literal itself and is decoded as the literal.
    """
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # The chat template serializes bools/None with Python ``str()``, so the
    # model emits ``True``/``False``/``None`` rather than JSON's
    # ``true``/``false``/``null``. Map those back to real values.
    if stripped == "True":
        return True
    if stripped == "False":
        return False
    if stripped == "None":
        return None
    return stripped


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:16]}"


# Streaming state machine states.
_STATE_INIT = 0  # Outside any <tool_call>; emit content.
_STATE_AFTER_TOOL_CALL_OPEN = 1  # Inside <tool_call>, before <function=...>.
_STATE_IN_FUNCTION = 2  # After <function=NAME>, between parameters.
_STATE_IN_PARAMETER = 3  # After <parameter=KEY>, accumulating value.
_STATE_AFTER_FUNCTION_CLOSE = 4  # After </function>, before </tool_call>.


def _max_holdback(buffer: str, sentinel: str) -> int:
    """Return the length of the longest suffix of ``buffer`` that is a
    proper prefix of ``sentinel``. We hold back exactly that many
    characters so a sentinel split across deltas is still recognized."""
    max_keep = min(len(buffer), len(sentinel) - 1)
    for n in range(max_keep, 0, -1):
        if sentinel.startswith(buffer[-n:]):
            return n
    return 0


@register("qwen3_5")
class Qwen3_5ToolParser:
    """Parser for Qwen 3.5 / 3.6 tool calls."""

    # Structural start marker, exposed under the same name as
    # ``StructuralTagToolParser.CALL_BEGIN`` so the serving layer's
    # marker-leak guard covers this parser too.
    CALL_BEGIN: ClassVar[str] = TOOL_CALL_OPEN

    def __init__(self) -> None:
        self._buffer: str = ""
        self._state: int = _STATE_INIT
        self._tool_index: int = -1
        self._param_count_in_call: int = 0
        self._current_param_key: str = ""

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parse a complete model response into tool calls."""
        if TOOL_CALL_OPEN not in response:
            return ParsedToolResponse(content=response, tool_calls=[])

        first_open = response.find(TOOL_CALL_OPEN)
        content_before = response[:first_open].strip() or None

        tool_calls: list[ParsedToolCall] = []
        for block in _TOOL_CALL_BLOCK_RE.finditer(response):
            body = block.group(1)
            name_match = _FUNCTION_NAME_RE.search(body)
            if not name_match:
                continue
            name = name_match.group(1).strip()

            args: dict[str, object] = {}
            for param in _PARAMETER_RE.finditer(body):
                key = param.group(1).strip()
                value = _decode_value(param.group(2))
                args[key] = value

            tool_calls.append(
                ParsedToolCall(
                    id=_new_call_id(),
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                )
            )

        if not tool_calls:
            raise ValueError(
                f"<tool_call> block(s) found but no valid calls parsed from: {response}"
            )

        return ParsedToolResponse(content=content_before, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Incrementally process one decoded-token delta.

        Returns content text to forward to the client and any tool-call
        increments to emit, in the order they were produced. Content
        deltas have ``content`` set; tool-call deltas have one or more of
        ``id`` / ``name`` / ``arguments`` set.
        """
        if delta:
            self._buffer += delta

        deltas: list[ParsedToolCallDelta] = []

        progressed = True
        while progressed:
            progressed = False

            if self._state == _STATE_INIT:
                idx = self._buffer.find(TOOL_CALL_OPEN)
                if idx >= 0:
                    # Only text *before the first* tool call is assistant
                    # content. Text between back-to-back calls (the template's
                    # "\n" separator) is structural and must be suppressed, not
                    # streamed as content.
                    if idx > 0 and self._tool_index == -1:
                        deltas.append(
                            ParsedToolCallDelta(
                                index=0,
                                content=self._buffer[:idx],
                            )
                        )
                    self._buffer = self._buffer[idx + len(TOOL_CALL_OPEN) :]
                    self._state = _STATE_AFTER_TOOL_CALL_OPEN
                    self._tool_index += 1
                    self._param_count_in_call = 0
                    progressed = True
                else:
                    # Hold back any tail that could be a partial sentinel.
                    keep = _max_holdback(self._buffer, TOOL_CALL_OPEN)
                    if keep < len(self._buffer):
                        emit = self._buffer[: len(self._buffer) - keep]
                        # Suppress trailing text once a tool call has started;
                        # the template forbids content after a function call.
                        if self._tool_index == -1:
                            deltas.append(
                                ParsedToolCallDelta(index=0, content=emit)
                            )
                        self._buffer = self._buffer[len(self._buffer) - keep :]

            elif self._state == _STATE_AFTER_TOOL_CALL_OPEN:
                # Looking for "<function=NAME>".
                start = self._buffer.find(FUNCTION_OPEN_PREFIX)
                if start < 0:
                    break  # need more
                end = self._buffer.find(">", start + len(FUNCTION_OPEN_PREFIX))
                if end < 0:
                    break  # need more
                name = self._buffer[
                    start + len(FUNCTION_OPEN_PREFIX) : end
                ].strip()
                self._buffer = self._buffer[end + 1 :]
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._tool_index,
                        id=_new_call_id(),
                        name=name,
                        arguments="{",
                    )
                )
                self._state = _STATE_IN_FUNCTION
                progressed = True

            elif self._state == _STATE_IN_FUNCTION:
                # Either "<parameter=KEY>" or "</function>".
                param_start = self._buffer.find(PARAMETER_OPEN_PREFIX)
                func_close = self._buffer.find(FUNCTION_CLOSE)

                next_param = param_start if param_start >= 0 else None
                next_close = func_close if func_close >= 0 else None
                if next_param is not None and (
                    next_close is None or next_param < next_close
                ):
                    end = self._buffer.find(
                        ">", next_param + len(PARAMETER_OPEN_PREFIX)
                    )
                    if end < 0:
                        break  # need more
                    self._current_param_key = self._buffer[
                        next_param + len(PARAMETER_OPEN_PREFIX) : end
                    ].strip()
                    after_open = end + 1
                    if (
                        after_open < len(self._buffer)
                        and self._buffer[after_open] == "\n"
                    ):
                        after_open += 1
                    self._buffer = self._buffer[after_open:]
                    self._state = _STATE_IN_PARAMETER
                    progressed = True
                elif next_close is not None:
                    self._buffer = self._buffer[
                        next_close + len(FUNCTION_CLOSE) :
                    ]
                    deltas.append(
                        ParsedToolCallDelta(
                            index=self._tool_index,
                            arguments="}",
                        )
                    )
                    self._state = _STATE_AFTER_FUNCTION_CLOSE
                    progressed = True
                else:
                    break  # need more

            elif self._state == _STATE_IN_PARAMETER:
                close = self._buffer.find(PARAMETER_CLOSE)
                if close < 0:
                    break  # need more
                raw_value = self._buffer[:close]
                raw_value = raw_value.removesuffix("\n")
                self._buffer = self._buffer[close + len(PARAMETER_CLOSE) :]

                value = _decode_value(raw_value)
                fragment = (
                    ("," if self._param_count_in_call > 0 else "")
                    + json.dumps(self._current_param_key)
                    + ":"
                    + json.dumps(value, ensure_ascii=False)
                )
                deltas.append(
                    ParsedToolCallDelta(
                        index=self._tool_index,
                        arguments=fragment,
                    )
                )
                self._param_count_in_call += 1
                self._state = _STATE_IN_FUNCTION
                progressed = True

            elif self._state == _STATE_AFTER_FUNCTION_CLOSE:
                close = self._buffer.find(TOOL_CALL_CLOSE)
                if close < 0:
                    break  # need more
                self._buffer = self._buffer[close + len(TOOL_CALL_CLOSE) :]
                self._state = _STATE_INIT
                progressed = True

        if deltas:
            return deltas
        # No deltas emitted this chunk. If we're mid-tool-call (any state
        # other than INIT), we've consumed the chunk's structural tokens into
        # the buffer and will emit them later — return [] (not None) so the
        # router SUPPRESSES the raw tokens rather than leaking them as content
        # (e.g. a partial "<parameter=" / unterminated "</parameter" held while
        # awaiting the rest of the tag).
        if self._state != _STATE_INIT:
            return []
        # In INIT: once a tool call has started, any buffered text is an
        # inter-call separator or trailing token — return [] to suppress it
        # (the router passes raw tokens through as content on None). Only
        # before the first call do we defer to that raw-content passthrough —
        # BUT if we're holding back bytes that partially match a sentinel,
        # return [] to suppress the raw tokens so they don't leak as content.
        return [] if self._tool_index >= 0 or self._buffer else None

    def reset(self) -> None:
        """Reset internal state for a new streaming session."""
        self._buffer = ""
        self._state = _STATE_INIT
        self._tool_index = -1
        self._param_count_in_call = 0
        self._current_param_key = ""

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """No-op: this format does not need schema-driven streaming.

        See ``ToolParser.set_streaming_tool_schemas``.
        """
        return
