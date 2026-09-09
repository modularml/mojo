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

"""Tool-call parser for the Inkling architecture."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    build_xgrammar_tool_grammar,
)
from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    register,
)
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolResponse,
    PipelineTokenizer,
)

from .tokenizer import TOOL_CALL_JSON_MARKER

# Payload shape: {"name":...,"args":{...}}. The tool-call grammar pins that
# frame to a const string, but the parser also runs on unconstrained output:
# with ``enable_tool_call_constrained_decode=False``, or after a rejected token
# fails enforcement open for the rest of a request. So neither the key order nor
# ``args`` being the payload's last key is guaranteed here.
_ARGS_KEY_RE = re.compile(r'[,{]\s*"args"\s*:\s*')

_DECODER = json.JSONDecoder()

# Closing quote included, so a half-streamed name never parses as a complete one.
_NAME_VALUE_RE = re.compile(r'"name"\s*:\s*("(?:[^"\\]|\\.)*")')

# OpenAI's tool-name charset, anchored with ``\Z`` so a trailing newline does not
# read as the end of a name run.
_TRAILING_NAME_RUN_RE = re.compile(r"[A-Za-z0-9_-]*\Z")


def _scan_json_object(text: str, start: int) -> tuple[int, bool]:
    """Returns the exclusive ``(end, is_complete)`` extent of a JSON object.

    There is no closing marker to find: the next call's bare function name
    butts straight against the previous call's final ``}``. ``end`` is
    ``len(text)`` while the object is still arriving.
    """
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                # Unbalanced close before any open: malformed, give up.
                break
            depth -= 1
            if depth == 0:
                return i + 1, True

    return len(text), False


def _args_text(body: str, is_complete: bool) -> str | None:
    """Returns the payload's ``args`` value exactly as the model wrote it.

    Both parse paths read arguments through here, so a call's ``arguments`` are
    the model's own bytes either way. Re-serializing one side would normalize
    the interior whitespace the tool-call grammar admits, leaving streaming and
    non-streaming to disagree on output the grammar allows.

    The extent is bounded by the value itself whenever it decodes, so a key
    following ``args`` cannot bleed into it -- mid-stream either, where the
    argument deltas already sent cannot be taken back.
    """
    match = _ARGS_KEY_RE.search(body)
    if match is None:
        return None
    start = match.end()
    try:
        _, end = _DECODER.raw_decode(body, start)
    except json.JSONDecodeError:
        # A value still arriving streams as the raw tail, which the base class
        # diffs into deltas. In a closed payload it is malformed, so withhold.
        return None if is_complete else body[start:]
    return body[start:end]


def _iter_call_payloads(
    text: str, start: int = 0
) -> Iterator[tuple[str, bool]]:
    """Yields ``(body, is_complete)`` for each tool call at or after ``start``."""
    search_pos = start
    while (marker := text.find(TOOL_CALL_JSON_MARKER, search_pos)) != -1:
        body_start = marker + len(TOOL_CALL_JSON_MARKER)
        end, is_complete = _scan_json_object(text, body_start)
        yield text[body_start:end], is_complete
        if not is_complete:
            return
        search_pos = end


@register("inkling")
class InklingToolParser(StructuralTagToolParser):
    """Parses Inkling tool calls, which reach this parser detokenized as
    ``NAME<|content_invoke_tool_json|>{"name":...,"args":{...}}``.

    Two hooks depart from the base class: bodies are bounded by brace balancing
    because there is no visible closing marker, and content deltas must drop the
    trailing function name, which precedes ``CALL_BEGIN``.
    """

    CALL_BEGIN: ClassVar[str] = TOOL_CALL_JSON_MARKER

    XGRAMMAR_FORMAT: ClassVar[str] = "inkling"

    _declared_tool_names: tuple[str, ...] = ()

    def parse_complete(self, response: str) -> ParsedToolResponse:
        parsed = super().parse_complete(response)
        if parsed.content is None or not parsed.tool_calls:
            return parsed

        # Content runs up to the first marker, so it ends with the bare
        # function name, which the payload repeats. Trimming that exact suffix
        # recovers the real content without consulting tool schemas.
        content = parsed.content.removesuffix(parsed.tool_calls[0].name).strip()
        return ParsedToolResponse(
            content=content or None, tool_calls=parsed.tool_calls
        )

    def reset(self) -> None:
        super().reset()
        # The router resets before re-supplying schemas, so stale names must not
        # survive into a request that declares none.
        self._declared_tool_names = ()

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Records declared names for the streaming holdback, longest first.

        The router omits tools without ``parameters``, so their names surface
        once in streamed content. Cosmetic; non-streaming is unaffected.
        """
        self._declared_tool_names = tuple(
            sorted(schemas, key=len, reverse=True)
        )

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for body, is_complete in _iter_call_payloads(tool_section):
            if not is_complete:
                continue
            try:
                payload = json.loads(body)
                name = payload["name"]
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if not isinstance(name, str) or not name:
                continue
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=name,
                    arguments=_args_text(body, True) or "{}",
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits the payload into the tool name and its ``args`` text.

        Withholds both until the name has landed, because the base class
        streams argument deltas even for a call it cannot yet open.
        """
        args_match = _ARGS_KEY_RE.search(body)
        if args_match is None:
            return (None, None)

        # Bounded to the text ahead of the key so a ``name`` property nested
        # inside ``args`` cannot pass for the tool's own name.
        match = _NAME_VALUE_RE.search(body[: args_match.start()])
        if match is None:
            return (None, None)
        try:
            name = json.loads(match.group(1))
        except json.JSONDecodeError:
            # An invalid escape in the name must not abort the stream.
            return (None, None)

        args = _args_text(body, is_complete)
        if args is None:
            return (None, None)
        return (name, args)

    def _extract_flat_call_bodies(
        self, marker_pos: int
    ) -> list[tuple[str, bool]]:
        """Extracts call bodies by brace balancing instead of by ``CALL_END``."""
        return list(_iter_call_payloads(self._buffer, marker_pos))

    def _extract_content_delta(self, marker_pos: int) -> str | None:
        content = super()._extract_content_delta(marker_pos)
        if content is None:
            return None

        holdback = self._name_holdback_len(content, marker_pos)
        if holdback == 0:
            return content

        self._state.sent_content_idx -= holdback
        return content[:-holdback] or None

    def _name_holdback_len(self, content: str, marker_pos: int) -> int:
        """Returns how many trailing bytes of ``content`` may be a tool name.

        The name sits immediately before ``CALL_BEGIN``, so once the marker is
        in the buffer the tail definitely is one. Before it arrives the tail is
        only a candidate: held back while it still prefixes a declared tool,
        released as soon as a character breaks the prefix, and dropped if the
        stream ends first.
        """
        run_match = _TRAILING_NAME_RUN_RE.search(content)
        run = run_match.group(0) if run_match is not None else ""
        if not run:
            return 0

        if marker_pos == -1:
            return (
                len(run)
                if any(n.startswith(run) for n in self._declared_tool_names)
                else 0
            )

        # Prefer an exact declared name so content fused to it
        # ("checkget_weather") is not swallowed whole.
        for name in self._declared_tool_names:
            if content.endswith(name):
                return len(name)
        return len(run)

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Builds the decode-time grammar that constrains tool calls."""
        if backend != "xgrammar":
            raise InputError(
                "Inkling constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        normalized_choice = tool_choice if tool_choice is not None else "auto"
        return build_xgrammar_tool_grammar(
            InklingToolParser.XGRAMMAR_FORMAT,
            tools or [],
            normalized_choice,
            response_format_schema=response_format_schema,
        )
