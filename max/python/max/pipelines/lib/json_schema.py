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
"""Structural traversal of JSON Schema documents.

Shared by the serve-side schema normalizer and the pipelines-side compile
diagnostics, which must agree on what counts as a nested subschema.
"""

from __future__ import annotations

import json
from typing import Any

# Keys whose value is a single subschema.
SUBSCHEMA_KEYS = frozenset(
    {
        "items",
        "additionalItems",
        "unevaluatedItems",
        "contains",
        "additionalProperties",
        "unevaluatedProperties",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    }
)

# Keys whose value maps property/definition names to subschemas.
SUBSCHEMA_MAP_KEYS = frozenset(
    {
        "properties",
        "patternProperties",
        "dependentSchemas",
        "$defs",
        "definitions",
    }
)

# Keys whose value is a list of subschemas.
SUBSCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})


def child_schemas(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns the subschemas nested directly under ``node``.

    Walks the node's own keys rather than probing for each keyword, so cost
    tracks the schema rather than the size of the vocabulary above.
    """
    children: list[dict[str, Any]] = []
    for key, value in node.items():
        if key in SUBSCHEMA_KEYS:
            if isinstance(value, dict):
                children.append(value)
        elif key in SUBSCHEMA_MAP_KEYS:
            if isinstance(value, dict):
                children.extend(
                    c for c in value.values() if isinstance(c, dict)
                )
        elif key in SUBSCHEMA_LIST_KEYS:
            if isinstance(value, list):
                children.extend(c for c in value if isinstance(c, dict))
    return children


def schema_shape(schema: Any) -> tuple[int, int] | None:
    """Returns ``(max_depth, subschema_count)``, or None if not a schema.

    Accepts a mapping or its JSON serialization; anything else returns None.
    Counts object-form subschemas only: a boolean is a valid schema but adds
    no nesting. A ``$ref`` is counted as the leaf it is written as and never
    resolved, so depth under-reports for ``$defs``-built schemas.

    Iterative because the input is client-supplied: recursion would raise
    ``RecursionError`` on a deep schema rather than measuring it.
    """
    if isinstance(schema, (str, bytes)):
        try:
            schema = json.loads(schema)
        except (ValueError, TypeError):
            return None
    if not isinstance(schema, dict):
        return None

    nodes = 0
    max_depth = 0
    stack: list[tuple[dict[str, Any], int]] = [(schema, 1)]
    while stack:
        node, depth = stack.pop()
        nodes += 1
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in child_schemas(node))
    return max_depth, nodes
