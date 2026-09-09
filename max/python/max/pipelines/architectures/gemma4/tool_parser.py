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
import re
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
    PipelineTokenizer,
)

from .tokenizer import SpecialToken

TOOL_CALL_PATTERN = re.compile(
    re.escape(SpecialToken.TOOL_CALL_START)
    + r"call:([\w\-\.]+)\{(.*?)\}"
    + re.escape(SpecialToken.TOOL_CALL_END),
    re.DOTALL,
)


def _json_loads_gemma4_string(body: str) -> str:
    """Decode a ``<|"|>``-delimited Gemma4 string body as a JSON string body.

    The grammar emits the body JSON-escaped (e.g. ``\\t`` for a tab) except a
    literal ``"``, which is emitted raw. Backslashes are always doubled, so no
    ``"`` is ever already-escaped: escape every ``"`` to ``\\"`` and decode
    with :func:`json.loads`. Falls open (returns ``body`` unchanged) on
    malformed input.
    """
    try:
        return json.loads('"' + body.replace('"', '\\"') + '"')
    except json.JSONDecodeError:
        return body


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    if value_str == "null":
        return None

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Number (int or float)
    try:
        if "." in value_str or "e" in value_str or "E" in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(
    args_str: str, *, partial: bool = False
) -> dict[str, Any]:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict[str, Any] = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(SpecialToken.STRING_DELIM):
            i += len(SpecialToken.STRING_DELIM)
            val_start = i
            end_pos = args_str.find(SpecialToken.STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = _json_loads_gemma4_string(args_str[val_start:])
                break
            result[key] = _json_loads_gemma4_string(args_str[val_start:end_pos])
            i = end_pos + len(SpecialToken.STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(SpecialToken.STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(SpecialToken.STRING_DELIM)
                    next_delim = args_str.find(SpecialToken.STRING_DELIM, i)
                    i = (
                        n
                        if next_delim == -1
                        else next_delim + len(SpecialToken.STRING_DELIM)
                    )
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(
                    args_str[obj_start:i], partial=True
                )
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(SpecialToken.STRING_DELIM):
                    i += len(SpecialToken.STRING_DELIM)
                    next_delim = args_str.find(SpecialToken.STRING_DELIM, i)
                    i = (
                        n
                        if next_delim == -1
                        else next_delim + len(SpecialToken.STRING_DELIM)
                    )
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(
                    args_str[arr_start:i], partial=True
                )
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list[Any]:
    """Parse a Gemma4 array content string into a Python list."""
    items: list[Any] = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(SpecialToken.STRING_DELIM):
            i += len(SpecialToken.STRING_DELIM)
            end_pos = arr_str.find(SpecialToken.STRING_DELIM, i)
            if end_pos == -1:
                items.append(_json_loads_gemma4_string(arr_str[i:]))
                break
            items.append(_json_loads_gemma4_string(arr_str[i:end_pos]))
            i = end_pos + len(SpecialToken.STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(SpecialToken.STRING_DELIM):
                    i += len(SpecialToken.STRING_DELIM)
                    nd = arr_str.find(SpecialToken.STRING_DELIM, i)
                    i = nd + len(SpecialToken.STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(
                    _parse_gemma4_args(arr_str[obj_start:i], partial=True)
                )
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(
                    _parse_gemma4_array(arr_str[sub_start:i], partial=True)
                )
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


@register("gemma4")
class Gemma4ToolParser(StructuralTagToolParser):
    """Gemma 4 tool parser using flat ``<|tool_call>`` … ``<tool_call|>`` pairs.

    Uses the flat (no-section-wrapper) mode of :class:`StructuralTagToolParser`:
    only ``CALL_BEGIN``/``CALL_END`` are set. Arguments are emitted atomically
    (withheld until the close marker) because Gemma4's ``<|"|>`` string
    delimiters make incremental JSON conversion non-monotonic.
    """

    CALL_BEGIN: ClassVar[str] = SpecialToken.TOOL_CALL_START
    CALL_END: ClassVar[str] = SpecialToken.TOOL_CALL_END

    # ----- StructuralTagToolParser hooks ----------------------------------

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for match in TOOL_CALL_PATTERN.finditer(tool_section):
            func_name = match.group(1)
            args_str = match.group(2)
            args_obj = _parse_gemma4_args(args_str)
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=func_name,
                    arguments=json.dumps(args_obj, ensure_ascii=False),
                )
            )
        return tool_calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        prefix = "call:"
        if not body.startswith(prefix):
            return (None, None)
        brace_pos = body.find("{")
        if brace_pos == -1:
            return (None, None)
        header = body[:brace_pos]
        if is_complete and body.endswith("}"):
            args = body[brace_pos + 1 : -1]
        else:
            args = body[brace_pos + 1 :]
        return (header, args)

    def _extract_tool_id_and_name(
        self, header: str
    ) -> tuple[str | None, str | None]:
        prefix = "call:"
        if not header.startswith(prefix):
            return (None, None)
        name = header[len(prefix) :].strip()
        if not name:
            return (None, None)
        return generate_call_id(), name

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        if not is_complete:
            return ""
        try:
            args_obj = _parse_gemma4_args(args_text)
            return json.dumps(args_obj, ensure_ascii=False)
        except Exception:
            return "{}"

    # ----- Constrained decoding grammar (xgrammar StructuralTag) ---------

    XGRAMMAR_FORMAT = "gemma_4"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a constrained-decoding grammar for Gemma 4 tool calls.

        Returns a serialized xgrammar ``StructuralTag`` that frames the
        ``<|tool_call>call:func{...}<tool_call|>`` envelope and constrains each
        call's arguments to that tool's JSON schema with bare keys and
        ``<|"|>`` string delimiters (a ``"json"``-style ``JSONSchemaFormat``
        configured via its bare-key and string-delimiter options). The full
        JSON schema spec is enforced by xgrammar's native converter.

        When ``response_format_schema`` is provided (tool_choice=auto), the
        grammar also accepts a JSON response conforming to that schema as an
        alternative to a tool call -- the model's first tokens select the
        branch (mirrors the Kimi xgrammar path).

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a schema-conforming JSON response as an
                alternative to a tool call.
            tools: OpenAI-style tool dicts.
            tokenizer: Unused (the xgrammar tag references literal markers).
            backend: Structured-output backend; must be ``"xgrammar"``.
            tool_choice: ``"auto"``, ``"required"``, or a named choice.
            **kwargs: Ignored; accepts future kwargs.

        Returns:
            The StructuralTag serialized as a JSON string.
        """
        if backend != "xgrammar":
            raise InputError(
                "Gemma 4 constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        normalized_choice = tool_choice if tool_choice is not None else "auto"
        return build_xgrammar_tool_grammar(
            Gemma4ToolParser.XGRAMMAR_FORMAT,
            tools or [],
            normalized_choice,
            response_format_schema=response_format_schema,
        )
