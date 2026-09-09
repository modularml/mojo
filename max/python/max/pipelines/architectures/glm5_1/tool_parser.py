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

"""Tool call parser for GLM-4.5+ models (GLM-5.1 / GLM-5.2).

GLM emits tool calls as a flat sequence of ``<tool_call>`` blocks (no outer
section wrapper). Each block opens with the bare function name, followed by
alternating ``<arg_key>`` / ``<arg_value>`` pairs::

    <tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value><arg_key>units</arg_key><arg_value>celsius</arg_value></tool_call>

All of ``<tool_call>``, ``</tool_call>``, ``<arg_key>``, ``</arg_key>``,
``<arg_value>``, ``</arg_value>`` are single special tokens in the vocab.

Value encoding mirrors the chat template
(``{{ v | tojson(ensure_ascii=False) if v is not string else v }}``): string
arguments are emitted *bare* (no surrounding quotes), while numbers, booleans,
arrays, and objects are emitted as JSON. Decoding therefore tries ``json.loads``
first and falls back to the raw string, exactly as the MiniMax M2 / Qwen 3.5
parsers do.

Constraint level for tool calls (see ``generate_tool_call_grammar``): a
serialized xgrammar ``StructuralTag`` (xgrammar's builtin ``glm_4_7`` format,
``glm_xml`` style) frames the ``<tool_call>`` envelope and the
``<arg_key>``/``<arg_value>`` framing, restricts the function name to the
provided tools, and constrains each call's arguments to that tool's
``parameters`` schema — string values bare, every other type as JSON.
``response_format`` schemas are accepted as a JSON alternative to a tool call.

Reference: ``architectures/gemma4/tool_parser.py`` (flat-mode base + xgrammar
grammar).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
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
from max.pipelines.modeling.types import ParsedToolCall, PipelineTokenizer

# Special-token surface forms (all single tokens in the GLM vocab).
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_ARG_KEY_OPEN = "<arg_key>"
_ARG_KEY_CLOSE = "</arg_key>"
_ARG_VALUE_OPEN = "<arg_value>"
_ARG_VALUE_CLOSE = "</arg_value>"

# Complete-parse regexes. Non-greedy so a value never swallows the next marker
# (markers are special tokens, so they never appear inside a value anyway).
_TOOL_CALL_BLOCK_RE = re.compile(
    re.escape(_TOOL_CALL_OPEN) + r"(.*?)" + re.escape(_TOOL_CALL_CLOSE),
    re.DOTALL,
)
_ARG_PAIR_RE = re.compile(
    re.escape(_ARG_KEY_OPEN)
    + r"(.*?)"
    + re.escape(_ARG_KEY_CLOSE)
    + r"\s*"
    + re.escape(_ARG_VALUE_OPEN)
    + r"(.*?)"
    + re.escape(_ARG_VALUE_CLOSE),
    re.DOTALL,
)


def _decode_value(raw: str) -> object:
    """Decode a GLM ``<arg_value>`` payload.

    Strings arrive bare; numbers, booleans, arrays, and objects arrive as JSON.
    Try ``json.loads`` first (covers every non-string case plus quoted strings)
    and fall back to the raw text verbatim for bare strings. The grammar frames
    the value with no surrounding whitespace, so the raw payload is exactly the
    value: it must be returned unstripped, or space padding a length-bounded
    value used to reach its ``minLength`` (spaces are valid content bytes) would
    be dropped below the bound.
    """
    if not raw:
        return ""
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    # GLM strings are emitted bare; a payload that JSON-decodes to a *string*
    # is a quoted-string form whose surrounding quotes are literal content.
    # The grammar's length bound counts the raw bytes, so returning the
    # un-quoted form would drop a value below its ``minLength`` (e.g.
    # ``"b"`` -> ``b``, ``""`` -> ``""``). Keep the raw payload for the string
    # case; decode only non-string JSON (numbers, booleans, null, arrays,
    # objects).
    if isinstance(decoded, str):
        return raw
    return decoded


def _parse_args(args_body: str) -> dict[str, object]:
    """Parse the ``<arg_key>``/``<arg_value>`` pairs in a call body to a dict."""
    args: dict[str, object] = {}
    for match in _ARG_PAIR_RE.finditer(args_body):
        key = match.group(1).strip()
        if not key:
            continue
        args[key] = _decode_value(match.group(2))
    return args


def _json_type(value: object) -> str:
    """Return the JSON Schema primitive type name for a decoded value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    return "object"


_STRING_FACET_KEYS = ("minLength", "maxLength", "pattern", "format")


def _schema_types(schema: dict[str, Any]) -> set[str]:
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    # A schema that declares string facets but omits "type" still constrains
    # a string (JSON Schema applies these keywords only to strings), matching
    # the grammar's own is_string test. Infer "string" so a bare non-string
    # value is coerced back to its string form.
    if any(k in schema for k in _STRING_FACET_KEYS):
        return {"string"}
    return set()


def _value_matches(value: object, types: set[str]) -> bool:
    if not types:
        return True
    jt = _json_type(value)
    if jt in types:
        return True
    # An integer also satisfies a ``number`` type.
    return jt == "integer" and "number" in types


def _as_string(value: object) -> str:
    """Return the on-wire bare-string form of a decoded value."""
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False)
    )


def _ref_index(root: object) -> dict[str, Any]:
    """Index a schema's referenceable subschemas for local ``$ref`` resolution.

    Covers ``#/$defs`` / ``#/definitions`` pointers plus ``$id``-anchored
    subschemas (indexed by full ``$id``, by URI path, and by filename) so the
    draft-7 base-URI ``$ref`` forms the suite exercises resolve without a full
    URI resolver.
    """
    index: dict[str, Any] = {}
    for defs_key in ("$defs", "definitions"):
        d = root.get(defs_key) if isinstance(root, dict) else None
        if isinstance(d, dict):
            for name, sub in d.items():
                index.setdefault(f"#/{defs_key}/{name}", sub)

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        nid = node.get("$id")
        if isinstance(nid, str) and nid:
            index.setdefault(nid, node)
            path = nid.split("://", 1)[-1]
            if "/" in path:
                index.setdefault("/" + path.split("/", 1)[1], node)
            index.setdefault(nid.rstrip("/").rsplit("/", 1)[-1], node)
        for value in node.values():
            walk(value)

    walk(root)
    return index


def _resolve_local_ref(schema: object, index: dict[str, Any]) -> object:
    """Follow local ``$ref`` chains via the precomputed :func:`_ref_index`."""
    seen: set[str] = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in seen:
            break
        seen.add(ref)
        target = index.get(ref)
        if target is None and not ref.startswith("#"):
            # Base-URI relative ref: match on trailing path / filename.
            target = index.get(ref.lstrip("/").rsplit("/", 1)[-1])
        if not isinstance(target, dict):
            break
        schema = target
    return schema


def _coerce_to_schema(
    value: object, schema: object, defs: dict[str, Any]
) -> object:
    """Coerce a decoded ``<arg_value>`` back toward its schema-declared type.

    GLM emits string values *bare* (see ``_decode_value``), so a value whose
    text happens to be a JSON scalar/container (``123``, ``true``,
    ``{"a":1}``) decodes to that type even though the grammar constrained the
    field to a string. Wherever the (ref-resolved) schema admits ``string`` but
    the decoded value matches none of the declared types, restore the on-wire
    string form so the round-tripped arguments match the schema the grammar
    enforced. Recurses through ``object`` / ``array`` / ``anyOf`` / ``oneOf``
    schemas; leaves values that already satisfy the schema untouched.
    """
    schema = _resolve_local_ref(schema, defs)
    if not isinstance(schema, dict):
        return value

    # allOf: a value must satisfy every subschema, so coerce through each. A
    # single-element allOf is the common ``{"allOf": [{"$ref": ...}]}`` wrapper.
    allof = schema.get("allOf")
    if isinstance(allof, list) and allof:
        for sub in allof:
            value = _coerce_to_schema(value, sub, defs)

    types = _schema_types(schema)
    # The schema admits a string here but the value decoded to another type:
    # the grammar only produced a bare string, so treat it as that string.
    # Checked before ``anyOf`` so a ``{"type": "string", "anyOf": [...]}`` base
    # type is still applied.
    if "string" in types and not _value_matches(value, types):
        return _as_string(value)

    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            resolved = [_resolve_local_ref(b, defs) for b in branches]
            for b in resolved:
                bt = _schema_types(b) if isinstance(b, dict) else set()
                if bt and _value_matches(value, bt):
                    return _coerce_to_schema(value, b, defs)
            # Object branches usually omit an explicit ``"type": "object"`` and
            # only declare ``properties``/``required``, so the type check above
            # can't pick one. For an object value, coerce through the branch
            # whose required keys are all present (else one whose properties it
            # touches), so a bare scalar under a string-typed property (e.g.
            # ``{"foo": 123}`` where ``foo`` is a string) is restored.
            if isinstance(value, dict):
                for b in resolved:
                    if not isinstance(b, dict):
                        continue
                    req = b.get("required")
                    if (
                        isinstance(req, list)
                        and req
                        and all(k in value for k in req)
                    ):
                        return _coerce_to_schema(value, b, defs)
                for b in resolved:
                    if not isinstance(b, dict):
                        continue
                    props = b.get("properties")
                    if isinstance(props, dict) and any(
                        k in props for k in value
                    ):
                        return _coerce_to_schema(value, b, defs)
            allows_string = any(
                isinstance(b, dict) and "string" in _schema_types(b)
                for b in resolved
            )
            if allows_string and not isinstance(value, str):
                return _as_string(value)
            return value

    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return {
                k: (_coerce_to_schema(v, props[k], defs) if k in props else v)
                for k, v in value.items()
            }
        return value
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [_coerce_to_schema(v, items, defs) for v in value]
        return value
    return value


@register("glm45")
class GlmToolParser(StructuralTagToolParser):
    """Parses GLM-4.5+ (GLM-5.1 / GLM-5.2) tool calls.

    Flat layout: only ``CALL_BEGIN``/``CALL_END`` are set, so the base class
    scans for ``<tool_call>`` … ``</tool_call>`` pairs directly. Within each
    call the function name precedes the first ``<arg_key>``; the remainder is
    parameter XML that we convert to growing JSON for streaming.
    """

    CALL_BEGIN: ClassVar[str] = _TOOL_CALL_OPEN
    CALL_END: ClassVar[str] = _TOOL_CALL_CLOSE

    def __init__(self) -> None:
        super().__init__()
        self._stream_tool_schemas: dict[
            str, tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        self._active_params_schema: dict[str, Any] | None = None
        self._active_defs: dict[str, Any] = {}
        self._coerce_schema: dict[str, Any] | None = None
        self._coerced: dict[str, tuple[object, object]] = {}

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """Stores per-tool ``parameters`` schemas for streaming coercion."""
        self._stream_tool_schemas = {
            name: (schema, _ref_index(schema))
            for name, schema in schemas.items()
        }

    def reset(self) -> None:
        super().reset()
        self._active_params_schema = None
        self._active_defs = {}
        self._coerce_schema = None
        self._coerced = {}

    # ----- Complete parsing --------------------------------------------

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        tool_calls: list[ParsedToolCall] = []
        for block in _TOOL_CALL_BLOCK_RE.finditer(tool_section):
            body = block.group(1)
            name, args_body = self._split_tool_call_body(body, is_complete=True)
            if not name:
                continue
            args = _parse_args(args_body or "")
            tool_calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                )
            )
        return tool_calls

    # ----- Streaming hooks ---------------------------------------------

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        """Splits a ``<tool_call>`` body into (function-name, parameter-XML).

        The name is everything before the first ``<arg_key>``. Until either an
        ``<arg_key>`` marker or the closing ``</tool_call>`` has arrived the name
        boundary is unknown, so return ``(None, None)`` to defer.
        """
        key_idx = body.find(_ARG_KEY_OPEN)
        if key_idx >= 0:
            name = body[:key_idx].strip() or None
            self._activate_schema(name)
            return name, body[key_idx:]
        if is_complete:
            name = body.strip() or None
            self._activate_schema(name)
            return name, ""
        return None, None

    def _activate_schema(self, name: str | None) -> None:
        """Selects the schema and defs to coerce the in-progress call against."""
        self._active_params_schema, self._active_defs = (
            self._stream_tool_schemas.get(name or "", (None, {}))
        )

    def _format_args_for_streaming(
        self, args_text: str, is_complete: bool
    ) -> str:
        """Builds a growing JSON string from complete ``<arg_key>`` pairs.

        Omits the closing brace while the call is still streaming so that
        successive argument diffs concatenate into valid JSON once the final
        pair lands.
        """
        args = _parse_args(args_text)
        schema = self._active_params_schema
        if schema is not None and args:
            if schema is not self._coerce_schema:
                self._coerce_schema = schema
                self._coerced = {}
            coerced_args: dict[str, object] = {}
            for key, value in args.items():
                cached = self._coerced.get(key)
                if (
                    cached is not None
                    and type(cached[0]) is type(value)
                    and cached[0] == value
                ):
                    coerced_args[key] = cached[1]
                    continue
                out = _coerce_to_schema({key: value}, schema, self._active_defs)
                coerced_value = (
                    out[key] if isinstance(out, dict) and key in out else value
                )
                self._coerced[key] = (value, coerced_value)
                coerced_args[key] = coerced_value
            args = coerced_args
        if not args:
            return "{}" if is_complete else ""
        inner = ", ".join(
            f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}"
            for k, v in args.items()
        )
        return "{" + inner + ("}" if is_complete else "")

    # ----- Schema-aware argument coercion ------------------------------

    def coerce_arguments(
        self, args: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Coerce parsed arguments toward their tool ``parameters`` schema.

        GLM's bare string encoding is type-ambiguous (``123`` is the string
        ``"123"`` for a string-typed field, but decodes to the integer ``123``).
        The router invokes this on the non-streaming complete parse with the
        tool's ``parameters`` schema so a string field the grammar constrained
        is reported as the string it constrained.
        """
        index = _ref_index(schema)
        coerced = _coerce_to_schema(args, schema, index)
        return coerced if isinstance(coerced, dict) else args

    # ----- Constrained-decoding grammar --------------------------------

    XGRAMMAR_FORMAT = "glm_4_7"

    @staticmethod
    def generate_tool_call_grammar(
        response_format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tokenizer: PipelineTokenizer[Any, Any, Any] | None = None,
        backend: str = "xgrammar",
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generates a tool-call constrained-decoding grammar for GLM.

        Returns a serialized xgrammar ``StructuralTag``. It frames the
        ``<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
        envelope and constrains each call's arguments to that tool's JSON schema
        using xgrammar's native ``glm_xml`` style (bare string values, JSON for
        every other type). When ``response_format_schema`` is provided the tag
        also accepts a schema-conforming JSON response as an alternative to a
        tool call (mirroring the gemma4/kimi xgrammar paths).

        Args:
            response_format_schema: Optional JSON schema dict. When provided,
                the grammar also accepts a JSON response matching the schema.
            tools: Optional list of OpenAI-style tool dicts. ``None`` accepts
                any tool name.
            tokenizer: Unused (the xgrammar tag references literal markers).
            backend: Structured-output backend; must be ``"xgrammar"``.
            tool_choice: ``"auto"``, ``"required"``, or a named choice.
            **kwargs: Ignored (accepts future kwargs).

        Returns:
            The StructuralTag serialized as a JSON string.
        """
        if backend != "xgrammar":
            raise InputError(
                "GLM constrained tool calling requires the xgrammar "
                "backend; run with --structured-output-backend=xgrammar."
            )
        normalized_choice = tool_choice if tool_choice is not None else "auto"
        return build_xgrammar_tool_grammar(
            GlmToolParser.XGRAMMAR_FORMAT,
            tools or [],
            normalized_choice,
            response_format_schema=response_format_schema,
        )
