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
"""Tests for max.experimental.tree_utils.

One test per contract, with the facets of a contract as asserts inside it. The
module is value-agnostic, so these use plain Python leaves and stand-in node
classes. The last section drives the utilities the way ``nn.Module`` does --
naming weights by path, loading a checkpoint, moving a model, tied weights and
parent back-references -- because that is what the module layer depends on.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, NamedTuple

import pytest
from max.experimental.tree_utils import (
    TreeDef,
    as_predicate,
    extend_path,
    flatten,
    flatten_one_level,
    is_node,
    leaves,
    nodes,
    paths,
    unflatten,
    update,
)
from max.experimental.tree_utils import map as tree_map

# ═══ Node fixtures: one per shape the protocol allows ═════════════════════════

Point = namedtuple("Point", ["x", "y"])


class TypedPoint(NamedTuple):
    x: int
    y: int


class Pair:
    """Mapping children with metadata, rebuilt from its children."""

    def __init__(self, left: Any, right: Any, label: str = "p") -> None:
        self.left, self.right, self.label = left, right, label

    def __tree_flatten__(self) -> tuple[dict[str, Any], str]:
        return {"left": self.left, "right": self.right}, self.label

    @classmethod
    def __tree_unflatten__(cls, meta: str, children: Mapping[str, Any]) -> Pair:
        return cls(children["left"], children["right"], meta)


class Row:
    """Sequence children, so they are numbered by position."""

    def __init__(self, *items: Any) -> None:
        self.items = list(items)

    def __tree_flatten__(self) -> tuple[tuple[Any, ...], None]:
        return tuple(self.items), None

    @classmethod
    def __tree_unflatten__(cls, meta: None, children: Any) -> Row:
        del meta
        return cls(*children)


class Cell:
    """Created before its children, so it can sit on a cycle."""

    def __init__(self, value: Any = None, link: Any = None) -> None:
        self.value, self.link = value, link

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {"value": self.value, "link": self.link}, None

    @classmethod
    def __tree_empty__(cls, meta: None) -> Cell:
        del meta
        return object.__new__(cls)


class Rigid:
    """Rebuilt from its children, so a cycle cannot close through it."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {"value": self.value}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Rigid:
        del meta
        return cls(children["value"])


class Bank:
    """Children under keys that are not attribute names, so it needs a setter.

    Records every write, which is how the tests check that ``update`` touches
    only what changed.
    """

    def __init__(self, **entries: Any) -> None:
        self.entries = dict(entries)
        self.writes: list[str] = []

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return dict(self.entries), None

    @classmethod
    def __tree_empty__(cls, meta: None) -> Bank:
        del meta
        bank = object.__new__(cls)
        bank.entries, bank.writes = {}, []
        return bank

    def __tree_setattr__(self, key: str, value: Any) -> None:
        self.writes.append(key)
        self.entries[key] = value


class Roster(list[Any]):
    """A node that is also a ``list``, holding entries and an attribute.

    Its children come from two places, so placing one is the class's business
    rather than the container's.
    """

    norm: Any

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {
            **{str(i): entry for i, entry in enumerate(self)},
            **vars(self),
        }, None

    @classmethod
    def __tree_empty__(cls, meta: None) -> Roster:
        del meta
        return list.__new__(cls)

    def __tree_setattr__(self, key: str, value: Any) -> None:
        if not key.isdigit():
            setattr(self, key, value)
        elif (index := int(key)) < len(self):
            self[index] = value
        else:
            self.append(value)


class Unwritable:
    """Fillable, but its children are positional, so a fill has no name."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __tree_flatten__(self) -> tuple[tuple[Any, ...], None]:
        return (self.value,), None

    @classmethod
    def __tree_empty__(cls, meta: None) -> Unwritable:
        del meta
        return object.__new__(cls)


class Transparent:
    """Names its child with an empty key, so it adds no path segment."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {"": self.inner}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Transparent:
        del meta
        return cls(children[""])


@dataclass(frozen=True)
class Frozen:
    """A frozen dataclass node, spelled out rather than generated."""

    a: Any
    b: Any

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}, None

    @classmethod
    def __tree_unflatten__(
        cls, meta: None, children: Mapping[str, Any]
    ) -> Frozen:
        del meta
        return cls(**children)


class Ambiguous:
    """Metadata whose ``==`` answers elementwise, as an array does."""

    class _Elementwise:
        def __bool__(self) -> bool:
            raise ValueError("truth value of an array is ambiguous")

    def __eq__(self, other: object) -> Any:
        return self._Elementwise()

    def __hash__(self) -> int:
        return 0


class Explosive:
    """Metadata whose ``==`` raises, which structure comparison must absorb."""

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("no comparing me")

    def __hash__(self) -> int:
        return 0


class Weight:
    """Stands in for a tensor: the thing a typed ``leaf`` selects."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __repr__(self) -> str:
        return f"Weight({self.tag!r})"


def _flat(tree: Any, **knobs: Any) -> tuple[TreeDef, list[Any]]:
    """Flattens, returning the pair in ``unflatten``'s argument order."""
    flat, treedef = flatten(tree, **knobs)
    return treedef, flat


def _struct(tree: Any, **knobs: Any) -> TreeDef:
    """The structure alone, discarding the leaves.

    This is the deferred public ``structure()``. It lives here rather than in
    the module because no production caller discards the leaves; every one wants
    both halves of ``flatten``'s pair. Its use count here is not the trigger to
    promote it.
    """
    return flatten(tree, **knobs)[1]


# ═══ Round trips and ordering ═════════════════════════════════════════════════


@pytest.mark.parametrize(
    "tree",
    [
        1,
        None,
        [],
        {},
        (),
        [1, 2, 3],
        {"a": 1, "b": {"c": 2}},
        (1, [2, {"three": 3}]),
        Point(1, 2),
        TypedPoint(1, 2),
        Pair(1, Pair(2, 3)),
        Row(1, Row(2)),
        Frozen(a=1, b=[2, 3]),
        Transparent({"x": 1}),
        {"n": [Point(1, 2), Row(3)], "m": Pair(4, 5)},
    ],
)
def test_round_trip(tree: Any) -> None:
    flat, treedef = flatten(tree)
    rebuilt = unflatten(treedef, flat)
    assert flatten(rebuilt) == (flat, treedef)


def test_dict_structure_depends_on_the_key_set_not_insertion_order() -> None:
    """The reason keys are sorted: otherwise two dicts differing only in
    insertion order would pair the wrong leaves in a multi-tree map."""
    assert _struct({"a": 1, "b": 2}) == _struct({"b": 2, "a": 1})
    assert tree_map(
        lambda a, b: (a, b), {"a": 1, "b": 2}, {"b": 20, "a": 10}
    ) == {"a": (1, 10), "b": (2, 20)}
    # Rebuilt in sorted order, since that is what the structure records.
    rebuilt = unflatten(*_flat({"b": 1, "a": 2}))
    assert rebuilt == {"b": 1, "a": 2} and list(rebuilt) == ["a", "b"]


def test_unsortable_dict_keys_raise() -> None:
    with pytest.raises(TypeError, match="mutually sortable"):
        flatten({1: "a", "b": 2})


def test_a_child_is_named_by_a_mapping_and_numbered_by_a_sequence() -> None:
    assert list(paths(Pair(7, 8))) == ["left", "right"]
    assert list(paths(Row(7, 8))) == ["0", "1"]
    # A namedtuple names them from its fields, and stays its concrete type --
    # so it is a different structure from the plain tuple of the same arity.
    assert list(paths(Point(1, 2))) == ["x", "y"]
    assert isinstance(unflatten(*_flat(Point(1, 2))), Point)
    assert _struct(Point(1, 2)) != _struct((1, 2))
    # A dict value is sorted; a node's mapping keeps the order the node chose.

    class Ordered:
        def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
            return {"z": 1, "a": 2}, None

    assert list(paths(Ordered())) == ["z", "a"]

    # Any Mapping names them, not only a dict; a non-mapping is positional.
    class Proxied:
        def __tree_flatten__(self) -> tuple[Mapping[str, Any], None]:
            return MappingProxyType({"z": 1, "a": 2}), None

        @classmethod
        def __tree_unflatten__(
            cls, meta: None, children: Mapping[str, Any]
        ) -> dict[str, Any]:
            del meta
            return dict(children)

    assert list(paths(Proxied())) == ["z", "a"]
    assert unflatten(*_flat(Proxied())) == {"z": 1, "a": 2}
    # And metadata survives the round trip.
    assert unflatten(*_flat(Pair(1, 2, label="kept"))).label == "kept"


# ═══ Knobs ════════════════════════════════════════════════════════════════════


def test_a_structure_records_what_the_knobs_did_not_which_were_set() -> None:
    """Two walks that decompose a value the same way produce one structure, and
    either rebuilds the other's leaves -- which is what equality promises."""
    w = Weight("w")
    # `Weight` is not a node, so it is already a leaf: naming it changes
    # nothing about the decomposition, so the structures are the same one.
    default_flat, default_def = flatten({"w": w})
    named_flat, named_def = flatten({"w": w}, leaf=Weight)
    assert default_def == named_def and hash(default_def) == hash(named_def)
    assert unflatten(default_def, named_flat) == {"w": w}
    assert unflatten(named_def, default_flat) == {"w": w}
    # A predicate is not compared, so two spellings of one rule agree too.
    assert _struct([w], leaf=lambda v: True) == _struct(
        [w], leaf=lambda v: True
    )
    # And where a knob *does* change the decomposition, the shape shows it: an
    # alias is two leaves by default and a leaf plus a reference when shared.
    tied = {"a": w, "b": w}
    assert _struct(tied, leaf=Weight) != _struct(tied, leaf=Weight, shared=True)


# ═══ leaf= ════════════════════════════════════════════════════════════════════


def test_leaf_selects_by_type_tuple_of_types_or_predicate() -> None:
    w = Weight("w")
    assert leaves({"a": 1, "b": "s", "c": None}) == [1, "s", None]
    assert leaves({"w": w, "n": 5}, leaf=Weight) == [w]
    assert leaves({"w": w, "n": 5, "s": "x"}, leaf=(Weight, int)) == [5, w]
    assert leaves(
        [1, 2, 3, 4], leaf=lambda v: isinstance(v, int) and v % 2 == 0
    ) == [2, 4]
    # A node that is also a leaf is not entered; other nodes still recurse.
    inner = Row(1, 2)
    assert leaves({"r": inner}, leaf=Row) == [inner]
    w1, w2 = Weight("a"), Weight("b")
    assert leaves({"o": [Pair(w1, {"d": w2})]}, leaf=Weight) == [w1, w2]


def test_a_leaf_predicate_is_asked_about_interior_nodes_too() -> None:
    """So it must be written to survive being handed a container."""
    seen: list[Any] = []

    def probing_leaf(value: Any) -> bool:
        seen.append(value)
        return not is_node(value)

    leaves({"a": [1]}, leaf=probing_leaf)
    assert seen == [{"a": [1]}, [1], 1]


# ═══ Static structure ═════════════════════════════════════════════════════════


def test_a_value_that_is_neither_a_leaf_nor_a_node_rides_along() -> None:
    """The only thing that happens to an unclaimed value: it is carried in the
    structure, restored untouched, and is part of what the structure compares."""
    w = Weight("w")
    flat, treedef = flatten({"w": w, "n": 5}, leaf=Weight)
    assert flat == [w]
    # Being static is what narrowing `leaf` leaves behind, not a property of the
    # value: the same 5, and the same Weight, are leaves under the default walk.
    assert leaves({"w": w, "n": 5}) == [5, w]
    assert unflatten(treedef, [w])["n"] == 5
    assert treedef != _struct({"w": w, "n": 6}, leaf=Weight)
    # Untouched, not copied: the rebuilt tree holds the original object, so
    # mutating it is visible on both sides.
    stray = Ambiguous()
    rebuilt = unflatten(*_flat({"w": w, "cfg": stray}, leaf=Weight))
    assert rebuilt["cfg"] is stray


# ═══ shared= ══════════════════════════════════════════════════════════════════


def test_shared_decides_whether_an_alias_is_one_value_or_two() -> None:
    w = Weight("w")
    tree = {"a": w, "b": w}
    # Off, the default: one leaf per path, which is what a checkpoint wants.
    assert leaves(tree, leaf=Weight) == [w, w]
    # On: one leaf, mapped once, aliasing restored -- and part of the
    # structure, so shared and duplicated trees compare unequal.
    flat, shared_def = flatten(tree, leaf=Weight, shared=True)
    assert flat == [w]
    rebuilt = unflatten(shared_def, [Weight("new")])
    assert rebuilt["a"] is rebuilt["b"]
    assert shared_def != _struct(tree, leaf=Weight)
    assert shared_def.num_leaves == 1
    seen: list[Weight] = []

    def observe(value: Weight) -> Weight:
        seen.append(value)
        return value

    tree_map(observe, tree, leaf=Weight, shared=True)
    assert seen == [w]

    # A container and a custom node alias the same way a leaf does.
    inner = [1]
    rebuilt = unflatten(_struct({"a": inner, "b": inner}, shared=True), [9])
    assert rebuilt["a"] is rebuilt["b"] == [9]
    node = Pair(1, 2)
    rebuilt = unflatten(*_flat({"a": node, "b": node}, shared=True))
    assert rebuilt["a"] is rebuilt["b"]


def test_identity_is_only_consulted_where_it_means_something() -> None:
    """Interned values alias at the interpreter's whim, and a tuple is rebuilt
    whole rather than filled, so nothing can reference it -- though a value
    inside one still can be shared."""
    for value in (1, "s", None, True, 2.5, b"x"):
        assert leaves({"a": value, "b": value}, shared=True) == [
            value,
            value,
        ], value
    pair = (1, 2)
    assert len(leaves({"a": pair, "b": pair}, shared=True)) == 4
    shared = [1]
    assert len(leaves({"a": (shared,), "b": (shared,)}, shared=True)) == 1


# ═══ Cycles ═══════════════════════════════════════════════════════════════════


def test_a_cycle_round_trips_through_anything_that_exists_early() -> None:
    """One rule covers all of it: a cycle can only close through a value that
    exists before its children -- a list, a dict, or a ``__tree_empty__`` node.
    A tuple never can, but a cycle may pass *through* one on its way back."""
    cyclic_dict: dict[str, Any] = {"leaf": 1}
    cyclic_dict["self"] = cyclic_dict
    rebuilt = unflatten(_struct(cyclic_dict, shared=True), [2])
    assert rebuilt["self"] is rebuilt and rebuilt["leaf"] == 2

    items: list[Any] = [1]
    items.append(items)
    rebuilt = unflatten(*_flat(items, shared=True))
    assert rebuilt[1] is rebuilt and rebuilt[0] == 1

    cell = Cell(value=1)
    cell.link = cell
    rebuilt = unflatten(*_flat(cell, shared=True))
    assert rebuilt.link is rebuilt and rebuilt.value == 1

    # Around a three-node ring, and reaching back from any depth.
    a, b, c = Cell("a"), Cell("b"), Cell("c")
    a.link, b.link, c.link = b, c, a
    rebuilt = unflatten(*_flat(a, shared=True))
    assert rebuilt.link.link.link is rebuilt
    assert (rebuilt.value, rebuilt.link.value) == ("a", "b")

    deep = Cell(value=Cell(value=Cell(value=1)))
    deep.value.value.link = deep
    rebuilt = unflatten(*_flat(deep, shared=True))
    assert rebuilt.value.value.link is rebuilt

    # Through a tuple, which claims no reference of its own but is on the route.
    via_tuple: list[Any] = [1]
    via_tuple.append((via_tuple,))
    rebuilt = unflatten(*_flat(via_tuple, shared=True))
    assert rebuilt[1][0] is rebuilt

    # And the walk visits a cycle once rather than forever: one leaf, one call.
    calls: list[Any] = []

    def observe(value: Any) -> Any:
        calls.append(value)
        return value * 10

    mapped = tree_map(observe, cyclic_dict, shared=True)
    assert calls == [1]
    assert mapped["leaf"] == 10 and mapped["self"] is mapped
    ring_a, ring_b = Cell("a"), Cell("b")
    ring_a.link, ring_b.link = ring_b, ring_a
    assert list(nodes(ring_a, Cell, shared=True)) == ["", "link"]


def test_a_cycle_that_cannot_close_names_the_fix() -> None:
    """The two ways a cycle fails: nothing mutable on it to close through, and a
    node that has to be built from children it does not have yet."""
    inner: list[Any] = [None]
    inner[0] = (inner,)
    with pytest.raises(ValueError, match="closes through a tuple"):
        flatten(inner[0], shared=True)

    rigid = Rigid(None)
    rigid.value = rigid
    with pytest.raises(TypeError, match="__tree_empty__"):
        unflatten(*_flat(rigid, shared=True))

    # Without ``shared`` there is no reference to spell a cycle with at all.
    self_ref: dict[str, Any] = {}
    self_ref["self"] = self_ref
    with pytest.raises(ValueError, match=r"cycle at 'self'.*shared=True"):
        flatten(self_ref)


# ═══ Paths ════════════════════════════════════════════════════════════════════


def test_paths_read_like_checkpoint_keys() -> None:
    tree = {"blocks": [{"bias": 1}, {"bias": 2}]}
    assert list(paths(tree)) == ["blocks.0.bias", "blocks.1.bias"]
    # Paths come from the structure alone, and stopping the walk early with
    # `leaf` addresses a whole subtree.
    assert _struct(tree).leaf_paths == ("blocks.0.bias", "blocks.1.bias")
    inner = Row(1, 2)
    assert paths({"deep": {"r": inner}}, leaf=Row) == {"deep.r": inner}


def test_an_empty_key_adds_no_segment_and_collisions_raise() -> None:
    assert list(paths({"w": Transparent({"inner": 1})})) == ["w.inner"]
    with pytest.raises(ValueError, match="two leaves share a path"):
        paths({"a": Transparent({"b": 1}), "a.b": 2})


# ═══ nodes ════════════════════════════════════════════════════════════════════


def test_nodes_reports_matching_nodes_by_path() -> None:
    root = Pair(Pair(1, 2), {"deep": Pair(3, 4)})
    assert list(nodes(root, Pair)) == ["", "left", "right.deep"]
    assert nodes(root, Row) == {}
    assert list(nodes({"p": Pair(1, 2), "r": Row(3)}, (Pair, Row))) == [
        "p",
        "r",
    ]
    # A match that is not itself a node is reported without being entered.
    w = Weight("w")
    assert nodes({"a": w}, Weight) == {"a": w}


def test_nodes_honours_leaf_and_shared() -> None:
    # `leaf` stops the walk, so a node behind one is never reached -- which is
    # what keeps `nodes` out of a value whose children are not tree structure.
    hidden = Pair(1, 2)
    assert nodes({"a": Row(hidden)}, Pair, leaf=Row) == {}
    assert list(nodes({"a": Row(hidden)}, Pair)) == ["a.0"]
    shared = Pair(1, 2)
    both = {"a": shared, "b": shared}
    assert list(nodes(both, Pair, shared=True)) == ["a"]
    assert list(nodes(both, Pair)) == ["a", "b"]


# ═══ map ══════════════════════════════════════════════════════════════════════


def test_map_rebuilds_with_the_mapped_leaves() -> None:
    tree = {"a": 1, "b": [2]}
    assert tree_map(lambda v: v * 2, tree) == {"a": 2, "b": [4]}
    assert tree == {"a": 1, "b": [2]}, "the original is untouched"
    assert tree_map(
        lambda a, b, c: a + b + c, {"x": 1}, {"x": 2}, {"x": 3}
    ) == {"x": 6}


def test_map_structure_mismatch_names_the_paths() -> None:
    with pytest.raises(ValueError) as excinfo:
        tree_map(lambda a, b: None, {"a": 1}, {"b": 1})
    message = str(excinfo.value)
    assert "tree argument 2" in message
    assert "expected leaves at ['a']" in message
    assert "got leaves at ['b']" in message
    # When the paths agree, the structures are shown instead; the offending
    # tree is named by position; a static payload is part of the structure.
    with pytest.raises(ValueError, match="same paths"):
        tree_map(lambda a, b: None, {"a": (1,)}, {"a": [1]})
    with pytest.raises(ValueError, match="tree argument 3"):
        tree_map(lambda a, b, c: None, {"a": 1}, {"a": 2}, {"b": 3})
    with pytest.raises(ValueError, match="same paths"):
        tree_map(
            lambda a, b: None,
            {"w": Weight("w"), "n": 1},
            {"w": Weight("w"), "n": 2},
            leaf=Weight,
        )


# ═══ update ═══════════════════════════════════════════════════════════════════


def test_update_writes_leaves_in_place_through_the_node_protocol() -> None:
    inner = [1, 2]
    tree = {"k": inner}
    assert update(tree, {"k.0": 10, "k.1": 20}) is tree
    assert tree["k"] is inner and inner == [10, 20], "the same list, refilled"

    cell = Cell(value=1, link=2)
    update(cell, {"value": 10, "link": 20})
    assert (cell.value, cell.link) == (10, 20)
    # A node's own setter sees only the children that actually changed.
    bank = Bank(a=1, b=2)
    update(bank, {"b": 20})
    assert bank.writes == ["b"]
    assert bank.entries == {"a": 1, "b": 20}


def test_update_is_partial_but_every_entry_must_land() -> None:
    """A leaf the state does not name is left alone, so half a checkpoint loads
    in one call; a name that matches no leaf is a typo, not a no-op."""
    tree = {"a": 1, "b": 2}
    update(tree, {"a": 10})
    assert tree == {"a": 10, "b": 2}
    with pytest.raises(ValueError, match=r"no leaf at.*'c'"):
        update(tree, {"c": 30})
    # Naming an interior node is the same mistake: only leaves are written.
    with pytest.raises(ValueError, match="no leaf at"):
        update({"outer": {"inner": 1}}, {"outer": 5})
    # And so is naming a static position, which `paths` would not name either.
    with pytest.raises(ValueError, match="no leaf at"):
        update({"w": Weight("w"), "eps": 1e-5}, {"eps": 1e-3}, leaf=Weight)


def test_update_rebuilds_only_what_cannot_be_written_into() -> None:
    """A tuple is immutable, so it is rebuilt and placed in its parent; the
    parent, and everything above it, keeps its identity."""
    inner = {"t": (1, 2)}
    tree = {"held": inner}
    assert update(tree, {"held.t.0": 10}) is tree
    assert tree["held"] is inner, "the mutable parent survives"
    assert inner["t"] == (10, 2)
    # A namedtuple keeps its concrete type through the rebuild.
    point = {"p": Point(1, 2)}
    update(point, {"p.x": 9})
    assert point["p"] == Point(9, 2) and isinstance(point["p"], Point)
    # At the root there is no parent to place it in, so it comes back instead.
    assert update((1, 2), {"0": 10}) == (10, 2)


def test_update_stops_where_leaf_says_and_reports_an_unwritable_key() -> None:
    w = Weight("old")
    model: dict[str, Any] = {"w": w, "eps": 1e-5}
    update(model, {"w": Weight("new")}, leaf=Weight)
    assert model["w"].tag == "new" and model["eps"] == 1e-5
    # `leaf` decides what a writable position is, so a value behind one is not
    # addressable -- the path names the subtree, not its insides.
    with pytest.raises(ValueError, match="no leaf at"):
        update({"r": Row(1)}, {"r.0": 9}, leaf=Row)
    # A positional key needs the node's own setter.
    with pytest.raises(TypeError, match="__tree_setattr__"):
        update(Unwritable(1), {"0": 2})


def test_a_node_subclassing_a_container_places_its_own_children() -> None:
    """Both writers ask the node where a child goes, not the builtin it
    subclasses, so an attribute stays an attribute."""
    roster = Roster([Weight("a"), Weight("b")])
    roster.norm = Weight("norm")
    assert list(paths(roster, leaf=Weight)) == ["0", "1", "norm"]

    treedef, _ = _flat(roster, leaf=Weight)
    rebuilt = unflatten(treedef, [Weight("A"), Weight("B"), Weight("N")])
    assert [w.tag for w in rebuilt] == ["A", "B"]
    assert rebuilt.norm.tag == "N"

    update(roster, {"0": Weight("x"), "norm": Weight("y")}, leaf=Weight)
    assert [w.tag for w in roster] == ["x", "b"]
    assert roster.norm.tag == "y"


def test_a_mapping_rebuilds_as_its_own_type() -> None:
    """An ``OrderedDict`` is ordered by contract, so its own order is its
    structure; every other mapping is keyed by its sorted keys. Either way the
    type, and what an empty one needs, survives."""
    ordered = OrderedDict([("z", 1), ("a", 2)])
    assert list(paths(ordered, leaf=int)) == ["z", "a"]
    rebuilt = unflatten(*_flat(ordered, leaf=int))
    assert type(rebuilt) is OrderedDict and list(rebuilt) == ["z", "a"]
    assert _struct(ordered, leaf=int) != _struct(
        OrderedDict([("a", 2), ("z", 1)]), leaf=int
    )
    # Not interchangeable with a plain dict of the same keys, and free of the
    # sortability requirement, since it never sorts.
    assert _struct(ordered) != _struct(dict(ordered))
    assert list(paths(OrderedDict([(1, "x"), ("b", "y")]))) == ["1", "b"]

    counts: defaultdict[str, int] = defaultdict(int, {"z": 1, "a": 2})
    assert list(paths(counts, leaf=int)) == ["a", "z"]
    restored = unflatten(*_flat(counts, leaf=int))
    assert type(restored) is defaultdict and restored.default_factory is int
    bare = unflatten(*_flat(defaultdict(None, {"k": 1})))
    assert type(bare) is defaultdict and bare.default_factory is None
    # The factory is structure too.
    assert _struct(counts, leaf=int) != _struct(
        defaultdict(list, {"z": 1, "a": 2}), leaf=int
    )


def test_an_unknown_builtin_subclass_is_a_leaf() -> None:
    """Builtins are matched by exact type, so an unknown subclass passes
    through whole. Declaring the protocol is what opts one in."""

    class Config(dict[str, Any]):
        def __init__(self, *args: Any, tag: str = "", **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.tag = tag

    config = Config({"w": 1}, tag="keep")
    assert not is_node(config)
    rebuilt = unflatten(*_flat(config, leaf=Config))
    assert type(rebuilt) is Config and rebuilt.tag == "keep"

    class Vec(list[Any]):
        pass

    assert not is_node(Vec([1, 2]))
    assert leaves({"v": Vec([1, 2])}, leaf=Vec) == [Vec([1, 2])]

    roster = Roster([Weight("a")])
    roster.norm = Weight("norm")
    assert (
        is_node(roster)
        and type(unflatten(*_flat(roster, leaf=Weight))) is Roster
    )


def test_update_preserves_a_tie_when_shared_is_set() -> None:
    """`update` takes `paths`' knobs, so under `shared` the one path named
    stands for every alias."""
    w = Weight("tied")
    tree: dict[str, Any] = {"a": w, "b": w}
    assert list(paths(tree, leaf=Weight, shared=True)) == ["a"]

    update(tree, {"a": Weight("new")}, leaf=Weight, shared=True)
    assert tree["a"] is tree["b"] and tree["a"].tag == "new"

    # Only the first path names a tied leaf, exactly as `paths` reports it.
    with pytest.raises(ValueError, match=r"no leaf at.*'b'"):
        update(tree, {"b": Weight("x")}, leaf=Weight, shared=True)


def test_update_writes_each_tied_path_separately() -> None:
    """Two paths, two writes: `update` is path-keyed like `paths`, so a tie
    does not survive it. `shared=True` plus `unflatten` is what preserves one."""
    w = Weight("tied")
    tree = {"a": w, "b": w}
    assert list(paths(tree, leaf=Weight)) == ["a", "b"]
    update(tree, {"a": Weight("x"), "b": Weight("y")}, leaf=Weight)
    assert tree["a"].tag == "x" and tree["b"].tag == "y"
    assert tree["a"] is not tree["b"]

    tied = {"a": w, "b": w}
    rebuilt = unflatten(*_flat(tied, leaf=Weight, shared=True))
    assert rebuilt["a"] is rebuilt["b"]


# ═══ TreeDef ══════════════════════════════════════════════════════════════════


def test_structures_compare_by_shape_not_leaf_values() -> None:
    assert _struct([1, 2]) == _struct(["a", "b"])
    assert _struct([1]) != _struct((1,))
    assert _struct([1]) != [1]
    # Hashable, so a structure can key a compilation cache.
    cache = {_struct({"x": 1}): "hit"}
    assert cache[_struct({"x": 2})] == "hit"
    # child_keys falls back to positions when children are unnamed.
    assert _struct([1, 2]).child_keys == (0, 1)
    assert _struct({"b": 1, "a": 2}).child_keys == ("a", "b")


def test_misbehaving_metadata_comparisons_are_absorbed() -> None:
    """A structure comparison must answer, not leak an array's ``__eq__``."""
    for payload_type in (Ambiguous, Explosive):
        a = _struct({"n": payload_type()}, leaf=Weight)
        b = _struct({"n": payload_type()}, leaf=Weight)
        assert a != b, payload_type
    # Identity short-circuits, so the same payload never gets compared.
    payload = Explosive()
    assert _struct({"n": payload}, leaf=Weight) == _struct(
        {"n": payload}, leaf=Weight
    )


# ═══ Errors from unflatten ════════════════════════════════════════════════════


def test_unflatten_rejects_leaves_and_structures_it_cannot_use() -> None:
    treedef, _ = _flat([1, 2, 3])
    with pytest.raises(ValueError, match="ran out of leaves"):
        unflatten(treedef, [1, 2])
    with pytest.raises(ValueError, match="too many leaves"):
        unflatten(treedef, iter([1, 2, 3, 4]))
    # A structure that did not come from `flatten`, and half a protocol.
    with pytest.raises(ValueError, match="unknown TreeDef kind"):
        unflatten(TreeDef("banana", (TreeDef("leaf"),)), [1])
    with pytest.raises(ValueError, match="points at nothing"):
        unflatten(TreeDef("list", (TreeDef("ref", meta=99),)), [])

    class Halfway:
        def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
            return {"a": 1}, None

    with pytest.raises(TypeError, match="no way to rebuild"):
        unflatten(*_flat(Halfway()))


# ═══ Writing your own walk ════════════════════════════════════════════════════


def test_is_node_reads_the_type_without_taking_it_apart() -> None:
    containers: tuple[Any, ...] = (
        [],
        (),
        {},
        Point(1, 2),
        Pair(1, 2),
        Row(1),
        Frozen(1, 2),
    )
    for value in containers:
        assert is_node(value), value
    for value in (1, "s", None, Weight("w"), object()):
        assert not is_node(value), value

    class Loud:
        def __tree_flatten__(self) -> tuple[tuple[Any, ...], None]:
            raise AssertionError("must not be called")

    assert is_node(Loud())
    # The protocol lives on the type: the class object itself is a value,
    # so a class stored in a tree travels as a leaf instead of crashing.
    assert not is_node(Pair)
    assert leaves({"activation": Pair}) == [Pair]


def test_flatten_one_level_decomposes_exactly_one_level() -> None:
    children, keys, meta = flatten_one_level({"b": [1], "a": 2})
    assert children == (2, [1]) and keys == ("a", "b") and meta is None
    children, keys, meta = flatten_one_level(Pair(1, 2, label="tag"))
    assert children == (1, 2) and keys == ("left", "right") and meta == "tag"
    assert flatten_one_level(Row(1, 2))[1] == (), "positional children unnamed"
    with pytest.raises(TypeError, match="is not a tree node"):
        flatten_one_level(5)


def test_extend_path_is_the_rule_paths_are_spelled_by() -> None:
    """The two rules a hand-written f-string gets wrong: no leading dot at the
    root, and an empty key adds no segment."""
    assert extend_path("blocks.3", "bias") == "blocks.3.bias"
    assert extend_path("", "blocks") == "blocks"
    assert extend_path("blocks.3", "") == "blocks.3"
    assert extend_path("", "") == ""
    # A key is rendered, so an index needs no conversion by the caller.
    assert extend_path("blocks", 3) == "blocks.3"

    # A walk built from the public pieces names positions exactly as `paths`
    # does -- which is the point of publishing the rule rather than the join.
    model = {"embed": Pair(Weight("e"), 1), "blocks": [Weight("b0")]}

    def walk(value: Any, path: str) -> dict[str, Any]:
        if not is_node(value):
            return {path: value} if isinstance(value, Weight) else {}
        children, keys, _ = flatten_one_level(value)
        found: dict[str, Any] = {}
        for key, child in zip(
            keys or range(len(children)), children, strict=False
        ):
            found.update(walk(child, extend_path(path, key)))
        return found

    assert walk(model, "") == paths(model, leaf=Weight)


def test_as_predicate_falls_back_to_the_default_only_for_none() -> None:
    """The three ``Selector`` spellings are exercised through ``leaf=``; what is
    unique here is that ``None`` means "no selector given" and hands over to the
    default, rather than being a selector that matches nothing."""

    def never(value: Any) -> bool:
        return False

    assert as_predicate(None, never) is never
    assert as_predicate(Weight, never)(Weight("w"))
    assert as_predicate((Weight, int), never)(5)
    assert as_predicate(lambda value: value == 5, never)(5)


# ═══ How nn.Module drives these utilities ═════════════════════════════════════
#
# Stand-ins for Module, matching its protocol exactly: attributes are the
# children, and it is created before them so it can sit on a cycle.


class Layer:
    """A module-shaped node: its attributes are its children."""

    def __init__(self, **attributes: Any) -> None:
        self.__dict__.update(attributes)

    def __getattr__(self, name: str) -> Any:
        # Attributes are dynamic (they arrive through ``__init__`` or a tree
        # rebuild), so declare that rather than annotating each test's set.
        raise AttributeError(name)

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return dict(vars(self)), None

    @classmethod
    def __tree_empty__(cls, meta: None) -> Layer:
        del meta
        return object.__new__(cls)


class Owned(Layer):
    """A module that keeps a parent pointer out of its own children.

    The recommended handling for a back-reference, and the only one needing no
    special mode: it is not part of the subtree, so the node does not report
    it. ``node`` cannot do this job, because it is handed the value and not
    the path, so excluding the parent by identity would exclude it at the root
    too.
    """

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        return {
            name: value
            for name, value in vars(self).items()
            if name != "parent"
        }, None


def _model() -> Layer:
    return Layer(
        embed=Layer(weight=Weight("embed")),
        blocks=[
            Layer(up=Weight("up0"), down=Weight("down0")),
            Layer(up=Weight("up1"), down=Weight("down1")),
        ],
        eps=1e-5,
    )


def test_a_models_weights_are_named_by_their_path() -> None:
    model = _model()
    assert list(paths(model, leaf=Weight)) == [
        "embed.weight",
        "blocks.0.up",
        "blocks.0.down",
        "blocks.1.up",
        "blocks.1.down",
    ]
    # A non-weight attribute is not named, but survives the round trip.
    flat, treedef = flatten(model, leaf=Weight)
    assert unflatten(treedef, flat).eps == 1e-5


def test_loading_a_checkpoint_is_an_update() -> None:
    """What ``Module.load_state_dict`` needs: read the names out, write them
    back. The two halves are ``paths`` and ``update`` over the same names."""
    model = _model()
    checkpoint = {
        path: Weight(f"loaded:{path}") for path in paths(model, leaf=Weight)
    }

    update(model, checkpoint, leaf=Weight)

    assert model.embed.weight.tag == "loaded:embed.weight"
    assert model.blocks[1].down.tag == "loaded:blocks.1.down"


def test_moving_a_model_keeps_every_submodule_identity() -> None:
    """What ``Module.to`` needs: the weights change, the modules do not."""
    model = _model()
    embed, blocks, first = model.embed, model.blocks, model.blocks[0]

    update(
        model,
        {
            path: Weight(f"moved:{w.tag}")
            for path, w in paths(model, leaf=Weight).items()
        },
        leaf=Weight,
    )

    assert model.embed is embed
    assert model.blocks is blocks and model.blocks[0] is first
    assert model.embed.weight.tag == "moved:embed"


def test_declaring_a_model_replaces_every_weight_and_keeps_the_shape() -> None:
    """What ``_all_tensors_to_parameters`` needs: a fresh copy of holes."""
    model = _model()
    declared = tree_map(lambda w: Weight("hole"), model, leaf=Weight)

    assert declared is not model
    assert list(paths(declared, leaf=Weight)) == list(paths(model, leaf=Weight))
    assert all(w.tag == "hole" for w in leaves(declared, leaf=Weight))
    assert model.embed.weight.tag == "embed", "the original is untouched"


def test_a_parent_back_reference_is_the_nodes_own_business() -> None:
    # Kept out of the tree by the node itself: no special mode needed.
    root = Owned(child=Owned(weight=Weight("w")))
    root.child.parent = root
    assert list(paths(root, leaf=Weight)) == ["child.weight"]
    assert list(nodes(root, Owned)) == ["", "child"]

    # Kept in the tree: positional flattening refuses, and shared=True
    # carries it through with the wiring intact.
    looped = Layer(child=Layer(weight=Weight("w")))
    looped.child.parent = looped
    with pytest.raises(ValueError, match="cycle at"):
        flatten(looped, leaf=Weight)
    rebuilt = unflatten(*_flat(looped, leaf=Weight, shared=True))
    assert rebuilt.child.parent is rebuilt
    assert rebuilt.child.weight.tag == "w"
    # update() skips the back-edge instead of recursing into it, so a
    # checkpoint loads into the looped model with no special mode.
    update(looped, {"child.weight": Weight("loaded")}, leaf=Weight)
    assert looped.child.weight.tag == "loaded"
    assert looped.child.parent is looped


def test_a_models_structure_is_a_compilation_cache_key() -> None:
    """Same shape agrees, different shape or hyperparameter does not."""
    a = _struct(_model(), leaf=Weight)
    b = _struct(_model(), leaf=Weight)
    assert a == b and hash(a) == hash(b)
    smaller = Layer(embed=Layer(weight=Weight("e")), blocks=[], eps=1e-5)
    assert _struct(smaller, leaf=Weight) != a
    # A static payload is part of the key, so it cannot be silently ignored.
    assert _struct(Layer(w=Weight("w"), eps=1e-5), leaf=Weight) != _struct(
        Layer(w=Weight("w"), eps=1e-3), leaf=Weight
    )


# The transforms tree_utils leaves unbuilt on purpose (listed at the foot of
# tree_utils.py) have their contracts recorded in the design doc alongside the
# behaviour they would pin, so the tests for one come from there if a caller
# ever appears for it.
