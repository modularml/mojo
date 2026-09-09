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
"""Shared helpers for normalizing OpenAI-style tool-call payloads."""

from __future__ import annotations

import json
from typing import Any

from max.pipelines.lib.json_schema import (
    SUBSCHEMA_KEYS,
    SUBSCHEMA_LIST_KEYS,
    SUBSCHEMA_MAP_KEYS,
)


def normalize_tool_call_arguments(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns ``tool_calls`` with ``function.arguments`` coerced to a mapping.

    OpenAI's chat completion schema renders
    ``tool_calls[*].function.arguments`` as a JSON-encoded string, while
    most tool-use chat templates iterate the arguments as a mapping.

    - Non-string ``arguments`` (already a dict/list) are passed through.
    - Missing, ``None``, or empty-string ``arguments`` become ``{}``.
    - JSON strings are decoded; malformed JSON is passed through untouched
      so client-side encoding errors surface at the template layer instead
      of being silently swallowed here.

    The input list and its dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        out = dict(tc)
        fn = out.get("function")
        if isinstance(fn, dict):
            args = fn.get("arguments")
            fn = dict(fn)
            if isinstance(args, (dict, list)):
                # Already a mapping/sequence; leave as-is.
                pass
            elif args is None or args == "":
                fn["arguments"] = {}
            elif isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    # Pass malformed JSON through so the template layer
                    # surfaces the original encoding error.
                    pass
            out["function"] = fn
        normalized.append(out)
    return normalized


# JSON Schema keywords that imply an object type when ``type`` is omitted.
# Mirrors xgrammar's ``json_schema_converter`` inference, which is the reason
# vLLM and SGLang (both on xgrammar) anchor such schemas to ``{`` and never
# hit the runaway.
_OBJECT_IMPLYING_KEYWORDS = frozenset(
    {"properties", "required", "additionalProperties", "patternProperties"}
)

# ``propertyNames`` holds a subschema validated against property names, which
# are always strings. Inferring an object type inside it would reject every
# name and make an otherwise satisfiable object schema unsatisfiable.
_TYPE_INFERRED_SUBSCHEMA_KEYS = SUBSCHEMA_KEYS - {"propertyNames"}


def normalize_response_format_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Returns ``schema`` with object-shaped untyped (sub)schemas given a type.

    A JSON Schema (sub)schema that omits ``type`` but carries an
    object-implying keyword (``properties``, ``required``,
    ``additionalProperties``, ``patternProperties``) is valid JSON Schema
    ("accept any type") but compiles, against the llguidance grammar backend,
    to a production that permits a bare, unbounded value -- including a JSON
    string. A model that loops inside that string can never emit its closing
    quote, so generation runs to ``max_length`` with ``finish_reason="length"``
    (the runaway-output incident).

    This pass mirrors xgrammar's type inference (the reason xgrammar-backed
    engines never hit this): where ``type`` is absent but an object-implying
    keyword is present, inject ``"type": "object"`` so the grammar anchors on
    ``{`` and stays terminating. The pass recurses through nested subschemas
    (property values, array items, ``$defs``, ``allOf``/``anyOf``/``oneOf``,
    etc.) so nested object-shaped schemas are anchored too.

    Deliberately left untouched:

    - A genuinely empty ``{}`` (no object-implying keyword): this is "any
      value" and is preserved for parity with xgrammar/SGLang and MAX's
      ``json_object`` semantics.
    - A ``type`` that is already present (a single type or a union list).

    The input is not mutated; a normalized copy is returned only where a
    change is needed.

    Args:
        schema: The ``response_format.json_schema.schema`` mapping (or any
            nested subschema).

    Returns:
        The schema with object types inferred where needed; the same object
        when no change applies.
    """
    return _normalize_subschema(schema)


def _normalize_subschema(node: Any) -> Any:
    """Recursively infer ``type: object`` for object-shaped untyped schemas."""
    if not isinstance(node, dict):
        return node

    changed = False
    out: dict[str, Any] = node

    # Infer an object type where it is omitted but implied.
    if "type" not in node and any(k in node for k in _OBJECT_IMPLYING_KEYWORDS):
        out = dict(node)
        out["type"] = "object"
        changed = True

    # Recurse into nested subschemas so inner object-shaped schemas are
    # anchored too (parity with xgrammar's recursive conversion).
    for key in _TYPE_INFERRED_SUBSCHEMA_KEYS:
        if key in out and isinstance(out[key], dict):
            new_child = _normalize_subschema(out[key])
            if new_child is not out[key]:
                if not changed:
                    out = dict(out)
                    changed = True
                out[key] = new_child

    for key in SUBSCHEMA_MAP_KEYS:
        mapping = out.get(key)
        if isinstance(mapping, dict):
            new_mapping: dict[str, Any] | None = None
            for name, child in mapping.items():
                new_child = _normalize_subschema(child)
                if new_child is not child:
                    if new_mapping is None:
                        new_mapping = dict(mapping)
                    new_mapping[name] = new_child
            if new_mapping is not None:
                if not changed:
                    out = dict(out)
                    changed = True
                out[key] = new_mapping

    for key in SUBSCHEMA_LIST_KEYS:
        seq = out.get(key)
        if isinstance(seq, list):
            new_seq: list[Any] | None = None
            for i, child in enumerate(seq):
                new_child = _normalize_subschema(child)
                if new_child is not child:
                    if new_seq is None:
                        new_seq = list(seq)
                    new_seq[i] = new_child
            if new_seq is not None:
                if not changed:
                    out = dict(out)
                    changed = True
                out[key] = new_seq

    return out


def _normalize_tools_parameters(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Returns ``tools`` with ``function.parameters`` coerced to a dict.

    OpenAI's API normalizes ``tools[*].function.parameters: null`` to an
    empty parameter list (equivalent to omitting the field) and returns
    200. MAX should match.

    - Dict ``parameters`` are passed through unchanged.
    - ``None`` or missing ``parameters`` becomes ``{}``.
    - Other values pass through unchanged (downstream validation handles
      type errors).
    - Tool entries without a ``function`` dict pass through unchanged.

    The input list and its dicts are not mutated.
    """
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        out = dict(tool)
        fn = out.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            params = fn.get("parameters")
            if params is None:
                fn["parameters"] = {}
            out["function"] = fn
        normalized.append(out)
    return normalized
