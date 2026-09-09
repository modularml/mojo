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
import sys
from typing import Any

import pytest
from max.pipelines.lib.json_schema import (
    SUBSCHEMA_KEYS,
    SUBSCHEMA_LIST_KEYS,
    SUBSCHEMA_MAP_KEYS,
    child_schemas,
    schema_shape,
)

LEAF: dict[str, Any] = {"type": "string"}


def spine(levels: int) -> dict[str, Any]:
    """A schema whose deepest subschema sits at ``levels``, built iteratively."""
    node: dict[str, Any] = dict(LEAF)
    for _ in range(levels - 1):
        node = {"type": "object", "properties": {"a": node}}
    return node


@pytest.mark.parametrize(
    "unmeasurable",
    [None, 42, "plain string", b"\xff\xfe", [], ["a"], object()],
)
def test_shape_returns_none_when_there_is_no_object_to_measure(
    unmeasurable: Any,
) -> None:
    assert schema_shape(unmeasurable) is None


@pytest.mark.parametrize("boolean_schema", [True, False])
def test_shape_returns_none_for_a_boolean_schema(boolean_schema: bool) -> None:
    """A boolean is a valid schema but has no nesting, so it is not counted."""
    assert schema_shape(boolean_schema) is None


def test_shape_returns_none_for_malformed_json() -> None:
    assert schema_shape("{not json") is None
    assert schema_shape('["a", "b"]') is None, "valid JSON, but not a mapping"


def test_shape_of_empty_schema_is_one_node() -> None:
    assert schema_shape({}) == (1, 1)


def test_shape_ignores_non_subschema_keywords() -> None:
    """``required``, ``enum`` and ``type`` are keywords, not subschemas."""
    assert schema_shape(
        {
            "type": "object",
            "required": ["a", "b"],
            "enum": [{"a": 1}, {"b": 2}],
            "description": "x",
        }
    ) == (1, 1)


def test_shape_accepts_str_bytes_and_mapping_alike() -> None:
    schema = spine(4)
    expected = (4, 4)
    assert schema_shape(schema) == expected
    assert schema_shape(json.dumps(schema)) == expected
    assert schema_shape(json.dumps(schema).encode()) == expected


@pytest.mark.parametrize("key", sorted(SUBSCHEMA_KEYS))
def test_shape_descends_single_subschema_keys(key: str) -> None:
    assert schema_shape({key: dict(LEAF)}) == (2, 2)


@pytest.mark.parametrize("key", sorted(SUBSCHEMA_MAP_KEYS))
def test_shape_descends_mapping_keys(key: str) -> None:
    assert schema_shape({key: {"one": dict(LEAF), "two": dict(LEAF)}}) == (2, 3)


@pytest.mark.parametrize("key", sorted(SUBSCHEMA_LIST_KEYS))
def test_shape_descends_list_keys(key: str) -> None:
    assert schema_shape({key: [dict(LEAF), dict(LEAF)]}) == (2, 3)


def test_shape_skips_boolean_and_scalar_children() -> None:
    """Boolean and scalar values under a container are not counted."""
    assert schema_shape(
        {
            "properties": {"a": True, "b": dict(LEAF)},
            "anyOf": [False, "junk", dict(LEAF)],
            "items": True,
        }
    ) == (2, 3)


@pytest.mark.parametrize(
    "key",
    [
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "additionalItems",
        "contains",
    ],
)
def test_shape_descends_the_converters_other_single_containers(
    key: str,
) -> None:
    """Each of these holds one subschema, and nesting below it counts.

    Named explicitly rather than derived from the set, so dropping a keyword
    from the set fails here.
    """
    assert schema_shape({"type": "object", key: spine(3)}) == (4, 4)


def test_shape_descends_dependent_schemas() -> None:
    assert schema_shape(
        {"dependentSchemas": {"a": dict(LEAF), "b": spine(2)}}
    ) == (3, 4)


@pytest.mark.parametrize("levels", [1, 2, 5, 64])
def test_shape_depth_tracks_nesting(levels: int) -> None:
    assert schema_shape(spine(levels)) == (levels, levels)


def test_shape_counts_breadth_without_adding_depth() -> None:
    wide = {"properties": {f"k{i}": dict(LEAF) for i in range(500)}}
    assert schema_shape(wide) == (2, 501)


def test_shape_takes_the_deepest_branch_not_the_last() -> None:
    """Depth is the maximum over branches, not the last one walked."""
    assert schema_shape(
        {"properties": {"shallow": dict(LEAF), "deep": spine(10)}}
    ) == (11, 12)


def test_shape_combines_container_kinds_at_one_level() -> None:
    assert schema_shape(
        {
            "properties": {"a": dict(LEAF)},
            "allOf": [dict(LEAF)],
            "items": dict(LEAF),
            "$defs": {"S": dict(LEAF)},
        }
    ) == (2, 5)


def test_shape_counts_ref_as_a_leaf_without_resolving_it() -> None:
    """A ``$ref`` is never followed; each use site counts as one leaf.

    A definition is counted once, where it is defined, so reuse grows the
    node count and leaves the depth alone.
    """
    once = {
        "properties": {"a": {"$ref": "#/$defs/Item"}},
        "$defs": {"Item": {"properties": {"x": dict(LEAF)}}},
    }
    assert schema_shape(once) == (3, 4)

    fifty = {
        "properties": {f"k{i}": {"$ref": "#/$defs/Item"} for i in range(50)},
        "$defs": {"Item": {"properties": {"x": dict(LEAF)}}},
    }
    assert schema_shape(fifty) == (3, 53)


def test_shape_terminates_on_a_self_referential_ref() -> None:
    """A self-referential ``$ref`` terminates: refs are values, not edges.

    Counted: the root, ``Node``, ``children`` and its ``items``.
    """
    assert schema_shape(
        {
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "properties": {
                        "children": {"items": {"$ref": "#/$defs/Node"}}
                    }
                }
            },
        }
    ) == (4, 4)


def test_shape_survives_a_schema_deeper_than_the_recursion_limit() -> None:
    """Depth is bounded by the input, not by the interpreter stack."""
    levels = sys.getrecursionlimit() * 2
    assert schema_shape(spine(levels)) == (levels, levels)


def test_child_schemas_returns_only_mappings() -> None:
    node = {
        "properties": {"a": dict(LEAF), "b": True},
        "anyOf": [dict(LEAF), "junk"],
        "not": dict(LEAF),
        "type": "object",
    }
    assert all(isinstance(c, dict) for c in child_schemas(node))
    assert len(child_schemas(node)) == 3


def test_child_schemas_of_a_leaf_is_empty() -> None:
    assert child_schemas(dict(LEAF)) == []
