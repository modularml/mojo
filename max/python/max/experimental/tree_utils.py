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

"""Provides utilities for taking nested values apart and putting them back.

Import the module as a namespace. Interior nodes are ``list``, ``tuple``,
``dict``, and any class declaring the protocol; everything else is a leaf.
``__tree_flatten__`` returns ``(children, meta)`` and ``__tree_unflatten__``
rebuilds.

.. code-block:: python

    from max.experimental import tree_utils as tree

    class Linear:
        def __init__(self, weight, eps=1e-5):
            self.weight, self.eps = weight, eps

        def __tree_flatten__(self):
            return {"weight": self.weight}, self.eps

        @classmethod
        def __tree_unflatten__(cls, eps, children):
            return cls(children["weight"], eps)

    model = [Linear("w0"), Linear("w1")]

    tree.paths(model, leaf=str)               # {"0.weight": "w0", "1.weight": "w1"}
    tree.map(str.upper, model, leaf=str)      # a fresh model, weights mapped
    tree.update(model, {"0.weight": "new"}, leaf=str)   # written in place
    flat, treedef = tree.flatten(model, leaf=str)
    tree.unflatten(treedef, flat)             # a rebuilt model, eps intact

Declare ``__tree_empty__(meta)`` instead of ``__tree_unflatten__`` when the node
must exist before its children, and optionally ``__tree_setattr__(key, value)``.
Every walk takes ``leaf``, saying where it stops, and ``shared``, saying whether
a value reachable by two paths is one object or two.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

__all__ = [
    "Selector",
    "TreeDef",
    "as_predicate",
    "extend_path",
    "flatten",
    "flatten_one_level",
    "is_node",
    "leaves",
    "map",
    "nodes",
    "paths",
    "unflatten",
    "update",
]

_T = TypeVar("_T")

#: What ``leaf`` accepts: a type, a tuple of types, or a predicate.
Selector = type | tuple[type, ...] | Callable[[Any], bool]

#: Interned at the interpreter's whim, so ``shared`` never tracks them.
_INTERNED_ATOMS = (type(None), bool, int, float, complex, str, bytes)

#: Reserves a slot while its node's children are still being built.
_BUILDING = object()

#: "Nothing was passed", which a caller's own ``None`` is not.
_MISSING = object()


def as_predicate(
    selector: Selector | None, default: Callable[[Any], bool]
) -> Callable[[Any], bool]:
    """Resolves a :obj:`Selector` into a predicate.

    Args:
        selector: A type, a tuple of types, a predicate, or ``None``.
        default: The predicate to use when ``selector`` is ``None``.

    Returns:
        A predicate over one value.
    """
    if selector is None:
        return default
    if isinstance(selector, (type, tuple)):
        types = selector
        return lambda value: isinstance(value, types)
    return selector


def extend_path(path: str, key: Any) -> str:
    """Extends a dotted path by one key.

    An empty key adds no segment, and the root takes no leading dot:

    .. code-block:: python

        from max.experimental import tree_utils as tree

        assert tree.extend_path("blocks.3", "bias") == "blocks.3.bias"
        assert tree.extend_path("", "blocks") == "blocks"
        assert tree.extend_path("blocks.3", "") == "blocks.3"

    Args:
        path: The path so far, or ``""`` at the root.
        key: This position's key, rendered with :func:`str`.

    Returns:
        The extended path.
    """
    if (segment := str(key)) == "":
        return path
    return f"{path}.{segment}" if path else segment


def _path_or_root(path: str) -> str:
    """Names a path in an error message; the root has no name of its own."""
    return repr(path) if path else "<root>"


def _meta_eq(a: Any, b: Any) -> bool:
    """Compares two payloads without trusting their ``__eq__``."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _place_child(node: Any, key: Any, value: Any) -> None:
    """Puts one child on a node: its own setter first, then ``setattr``."""
    if hasattr(type(node), "__tree_setattr__"):
        node.__tree_setattr__(key, value)
    elif isinstance(key, str):
        setattr(node, key, value)
    else:
        raise TypeError(
            f"cannot write child {key!r} into a {type(node).__name__}: a "
            "positional child has no attribute name, so the node must declare "
            "__tree_setattr__(key, value) to say where it goes."
        )


#: By exact type: another subclass is a leaf unless it declares the protocol.
_BUILTIN_KINDS: dict[type, str] = {
    list: "list",
    tuple: "tuple",
    dict: "dict",
    OrderedDict: "dict",
    defaultdict: "dict",
}


def _node_kind(value: Any) -> str | None:
    """Returns the kind ``value`` flattens to, or ``None`` if it is not a node."""
    # On the type, so a class object stored in a tree is a value, not a node.
    if hasattr(type(value), "__tree_flatten__"):
        return "node"
    # A namedtuple rebuilds positionally, not from an iterable.
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return "namedtuple"
    return _BUILTIN_KINDS.get(type(value))


def is_node(value: Any) -> bool:
    """Returns whether ``value`` is an interior node rather than a leaf.

    Reads the type without taking ``value`` apart.

    Args:
        value: Any value.
    """
    return _node_kind(value) is not None


def flatten_one_level(
    value: Any,
) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
    """Takes one interior node apart, one level deep.

    Args:
        value: An interior node, as :func:`is_node` judges it.

    Returns:
        ``(children, keys, meta)``; ``keys`` is empty when the children are
        positional.

    Raises:
        TypeError: If ``value`` is not an interior node, or a ``dict``'s keys
            are not mutually sortable.
    """
    kind = _node_kind(value)
    if kind == "node":
        children, meta = value.__tree_flatten__()
        if isinstance(children, Mapping):
            return tuple(children.values()), tuple(children), meta
        return tuple(children), (), meta
    if kind == "namedtuple":
        return tuple(value), tuple(value._fields), type(value)
    if kind in ("list", "tuple"):
        return tuple(value), (), None
    if kind == "dict":
        if type(value) is OrderedDict:
            # Ordered by contract, so its own order is its structure.
            keys = tuple(value)
        else:
            try:
                keys = tuple(sorted(value))
            except TypeError:
                raise TypeError(
                    "dict keys must be mutually sortable to flatten a dict, "
                    "because a tree's structure is defined by its sorted keys; "
                    f"got keys of types "
                    f"{sorted({type(k).__name__ for k in value})}."
                ) from None
        return tuple(value[key] for key in keys), keys, _mapping_meta(value)
    raise TypeError(f"{type(value).__name__} is not a tree node.")


def _mapping_meta(value: Any) -> tuple[type, tuple[Any, ...]] | None:
    """Returns what rebuilds an empty ``value``, or ``None`` for a plain dict."""
    if type(value) is dict:
        return None
    args = (value.default_factory,) if type(value) is defaultdict else ()
    return type(value), args


@dataclass(frozen=True, eq=False)
class TreeDef:
    """The shape of a tree, with its leaves abstracted away.

    Equal structures rebuild each other's leaves. Hashing skips the payloads.
    """

    kind: str
    """What this position holds: ``"leaf"`` (a leaf slot), ``"static"`` (a
    value carried as-is in :attr:`meta`), ``"ref"`` (a repeat of an earlier
    value), or one of the container tags ``"list"``, ``"tuple"``,
    ``"namedtuple"``, ``"dict"`` and ``"node"``."""

    children: tuple[TreeDef, ...] = ()
    """The structure of each child, in leaf order."""

    keys: tuple[Any, ...] = ()
    """One key per child, or empty when the children are positional."""

    meta: Any = None
    """Whatever the kind needs to rebuild: the value itself for ``"static"``,
    the referenced slot number for ``"ref"``, the concrete type for
    ``"namedtuple"``, and a ``(type, metadata)`` pair for ``"node"``."""

    @property
    def num_leaves(self) -> int:
        """The number of leaf slots in this structure; a ``"ref"`` counts none."""
        if self.kind == "leaf":
            return 1
        return sum(child.num_leaves for child in self.children)

    @property
    def child_keys(self) -> tuple[Any, ...]:
        """The key of each child: :attr:`keys`, or positions if empty."""
        return self.keys or tuple(range(len(self.children)))

    @property
    def leaf_paths(self) -> tuple[str, ...]:
        """The dotted path to each leaf, in leaf order."""
        found: list[str] = []

        def walk(treedef: TreeDef, path: str) -> None:
            if treedef.kind == "leaf":
                found.append(path)
                return
            for key, child in zip(
                treedef.child_keys, treedef.children, strict=True
            ):
                walk(child, extend_path(path, key))

        walk(self, "")
        return tuple(found)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeDef):
            return False
        if (self.kind, self.keys) != (other.kind, other.keys):
            return False
        return (
            _meta_eq(self.meta, other.meta) and self.children == other.children
        )

    def __hash__(self) -> int:
        return hash((self.kind, self.children, self.keys))


@dataclass
class _RefTracker:
    """The ``shared`` knob's memory, for one traversal."""

    shared: bool
    #: ``id(value) -> (slot, first path, the value pinning the id)``.
    first_visits: dict[int, tuple[int, str, Any]] = field(default_factory=dict)
    #: The ids being descended into right now, which is how a cycle is caught.
    descending: set[int] = field(default_factory=set)
    #: How many slots were claimed so far, numbering every position.
    slots_claimed: int = 0

    def claim_slot(self) -> int:
        """Numbers this position, so a ``"ref"`` can name it."""
        # Mode-independent, so unflatten replays the numbering from structure.
        self.slots_claimed += 1
        return self.slots_claimed - 1

    def tracks_identity_of(self, value: Any) -> bool:
        """Returns whether this walk consults ``value``'s identity."""
        if not self.shared:
            return False
        if hasattr(type(value), "__tree_flatten__"):
            return True
        # Interned atoms and folded tuple literals alias at CPython's whim.
        return not isinstance(value, (*_INTERNED_ATOMS, tuple))

    def recall_first_visit(self, value: Any) -> int | None:
        """The slot ``value`` first appeared in, or ``None`` if it is new."""
        if not self.tracks_identity_of(value):
            return None
        if (first := self.first_visits.get(id(value))) is None:
            return None
        return first[0]

    def record_first_visit(self, value: Any, path: str, slot: int) -> None:
        """Records where ``value`` was first found."""
        if self.tracks_identity_of(value):
            # Holding the value pins its id against recycling.
            self.first_visits[id(value)] = (slot, path, value)

    def begin_descent(self, value: Any, path: str) -> None:
        """Marks a descent, so a cycle is an error and not a stack overflow.

        Raises:
            ValueError: If the walk is already inside ``value`` and cannot
                express the cycle it just found.
        """
        if id(value) in self.descending:
            if not self.shared:
                raise ValueError(
                    f"cycle at {_path_or_root(path)}: the value contains "
                    "itself, which this walk cannot spell. Pass shared=True to "
                    "keep the cycle as a reference."
                )
            if not self.tracks_identity_of(value):
                # Rebuilding would close the cycle on a copy: a rewired tree.
                raise ValueError(
                    f"cycle at {_path_or_root(path)}: the cycle closes "
                    "through a tuple, which is rebuilt whole rather than "
                    "filled, so no reference can reach it. Route the cycle "
                    "through a list, a dict, or a node with __tree_empty__."
                )
        self.descending.add(id(value))

    def end_descent(self, value: Any) -> None:
        """Unmarks a descent, for a ``finally``."""
        self.descending.discard(id(value))


def flatten(
    tree: Any, *, leaf: Selector | None = None, shared: bool = False
) -> tuple[list[Any], TreeDef]:
    """Splits ``tree`` into its leaves and the structure around them.

    Args:
        tree: The value to take apart.
        leaf: Where the walk stops.
        shared: Whether a value reachable by two paths is one object or two.

    Returns:
        The leaves, left to right, and the :class:`TreeDef` that rebuilds them.

    Raises:
        TypeError: If a ``dict`` in ``tree`` has keys that are not mutually
            sortable.
        ValueError: If ``tree`` holds a cycle this walk cannot express.
    """
    is_leaf = as_predicate(leaf, lambda value: not is_node(value))
    tracker = _RefTracker(shared)
    flat: list[Any] = []

    def walk(value: Any, path: str) -> TreeDef:
        if is_leaf(value):
            if (slot := tracker.recall_first_visit(value)) is not None:
                return TreeDef("ref", meta=slot)
            tracker.record_first_visit(value, path, tracker.claim_slot())
            flat.append(value)
            return TreeDef("leaf")

        if (kind := _node_kind(value)) is None:
            # Neither leaf nor node: carried as the same object, untouched.
            tracker.claim_slot()
            return TreeDef("static", meta=value)

        if (slot := tracker.recall_first_visit(value)) is not None:
            return TreeDef("ref", meta=slot)
        tracker.record_first_visit(value, path, tracker.claim_slot())
        tracker.begin_descent(value, path)
        try:
            children, keys, meta = flatten_one_level(value)
            return TreeDef(
                kind,
                tuple(
                    walk(child, extend_path(path, key))
                    for key, child in zip(
                        keys or range(len(children)), children, strict=True
                    )
                ),
                keys,
                (type(value), meta) if kind == "node" else meta,
            )
        finally:
            tracker.end_descent(value)

    return flat, walk(tree, "")


def unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
    """Rebuilds a tree from a structure and its leaves.

    The inverse of :func:`flatten`, always building a fresh tree.

    Args:
        treedef: The structure, as :func:`flatten` reported it.
        leaves: Exactly one value per leaf slot, left to right.

    Returns:
        The rebuilt tree.

    Raises:
        TypeError: If a cycle closes through a node whose class declares no
            ``__tree_empty__``.
        ValueError: If the leaf count does not match the structure, or if
            ``treedef`` did not come from :func:`flatten`.
    """
    remaining = iter(leaves)
    filled: list[Any] = []

    def next_leaf() -> Any:
        # A node's own StopIteration must not be mistaken for a shortfall.
        try:
            return next(remaining)
        except StopIteration:
            raise ValueError(
                f"ran out of leaves: the structure has {treedef.num_leaves} "
                "leaf slots."
            ) from None

    def resolve(slot: Any) -> Any:
        if not isinstance(slot, int) or not 0 <= slot < len(filled):
            raise ValueError(
                f"reference to slot {slot!r} points at nothing, so this "
                "structure did not come from flatten."
            )
        if (value := filled[slot]) is _BUILDING:
            raise TypeError(
                "a cycle closes through a node built by __tree_unflatten__, "
                "which needs its children first. Declare __tree_empty__(meta) "
                "on its class so the node can exist before them."
            )
        return value

    def build(treedef: TreeDef) -> Any:
        kind = treedef.kind
        if kind == "ref":
            return resolve(treedef.meta)
        if kind == "leaf":
            filled.append(value := next_leaf())
            return value
        if kind == "static":
            filled.append(treedef.meta)
            return treedef.meta

        # Claimed before the children, so a back-reference finds something.
        slot = len(filled)
        filled.append(_BUILDING)

        # These exist before their children, so a cycle through them closes.
        cls = treedef.meta[0] if kind == "node" else None
        empty = getattr(cls, "__tree_empty__", None)
        if kind in ("list", "dict") or empty is not None:
            if kind == "list":
                node: Any = []
            elif kind == "dict":
                mapping = treedef.meta
                node = {} if mapping is None else mapping[0](*mapping[1])
            else:
                assert empty is not None
                node = empty(treedef.meta[1])
            filled[slot] = node
            # On the kind, so a node subclassing list still uses the protocol.
            if kind == "list":
                node.extend(build(child) for child in treedef.children)
            else:
                for key, child in zip(
                    treedef.child_keys, treedef.children, strict=True
                ):
                    value = build(child)
                    if kind == "dict":
                        node[key] = value
                    else:
                        _place_child(node, key, value)
            return node

        built = [build(child) for child in treedef.children]
        if kind == "namedtuple":
            value = treedef.meta(*built)
        elif kind == "tuple":
            value = tuple(built)
        elif kind == "node":
            node_cls, node_meta = treedef.meta
            if (
                rebuild := getattr(node_cls, "__tree_unflatten__", None)
            ) is None:
                raise TypeError(
                    f"{node_cls.__name__} declares __tree_flatten__ but no "
                    "way to rebuild: add a classmethod __tree_unflatten__("
                    "meta, children) to build it from its children, or "
                    "__tree_empty__(meta) to create it before them."
                )
            value = rebuild(
                node_meta,
                dict(zip(treedef.keys, built, strict=True))
                if treedef.keys
                else built,
            )
        else:
            raise ValueError(f"unknown TreeDef kind: {kind!r}")
        filled[slot] = value
        return value

    result = build(treedef)
    if next(remaining, _MISSING) is not _MISSING:
        raise ValueError(
            f"too many leaves: the structure has {treedef.num_leaves} leaf "
            "slots."
        )
    return result


# ─── reads ──────────────────────────────────────────────────────────────────


def leaves(
    tree: Any, *, leaf: Selector | None = None, shared: bool = False
) -> list[Any]:
    """Returns ``tree``'s leaves, left to right, dropping the structure.

    Args:
        tree: The value to read.
        leaf: Where the walk stops.
        shared: Whether an aliased value is one leaf or one per path.
    """
    return flatten(tree, leaf=leaf, shared=shared)[0]


def paths(
    tree: Any, *, leaf: Selector | None = None, shared: bool = False
) -> dict[str, Any]:
    """Returns ``tree``'s leaves keyed by their dotted path.

    The result is what :func:`update` consumes.

    Args:
        tree: The value to read.
        leaf: Where the walk stops.
        shared: Whether an aliased value is one leaf or one per path.

    Returns:
        One entry per leaf, in leaf order.

    Raises:
        ValueError: If two leaves share a path, so it names neither.
    """
    flat, treedef = flatten(tree, leaf=leaf, shared=shared)
    leaf_paths = treedef.leaf_paths
    found = dict(zip(leaf_paths, flat, strict=True))
    if len(found) != len(leaf_paths):
        clashing = sorted(
            {path for path in leaf_paths if leaf_paths.count(path) > 1}
        )
        raise ValueError(
            f"two leaves share a path, so it names neither: {clashing}"
        )
    return found


def nodes(
    tree: Any,
    node_type: type[_T] | tuple[type[_T], ...],
    *,
    leaf: Selector | None = None,
    shared: bool = False,
) -> dict[str, _T]:
    """Returns every ``node_type`` value inside ``tree``, keyed by its path.

    The root matches at path ``""``.

    Args:
        tree: The value to walk.
        node_type: The type, or tuple of types, to report.
        leaf: Where the walk stops.
        shared: Whether a value reachable twice is reported once.

    Returns:
        ``{path: node}`` in the order the walk found them.
    """
    is_leaf = as_predicate(leaf, lambda value: not is_node(value))
    tracker = _RefTracker(shared)
    found: dict[str, _T] = {}

    def visit(value: Any, path: str) -> None:
        if tracker.recall_first_visit(value) is not None:
            return
        tracker.record_first_visit(value, path, tracker.claim_slot())
        if isinstance(value, node_type):
            found[path] = value
        if is_leaf(value) or not is_node(value):
            return
        tracker.begin_descent(value, path)
        try:
            children, keys, _ = flatten_one_level(value)
            for key, child in zip(
                keys or range(len(children)), children, strict=True
            ):
                visit(child, extend_path(path, key))
        finally:
            tracker.end_descent(value)

    visit(tree, "")
    return found


# ─── transform ──────────────────────────────────────────────────────────────


def map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    leaf: Selector | None = None,
    shared: bool = False,
) -> Any:
    """Builds a new tree with each leaf replaced by what ``f`` returns.

    With extra trees in ``rest``, ``f`` receives one leaf from each.

    Args:
        f: Applied to one leaf from ``tree`` and one from each of ``rest``.
        tree: The structure whose leaves are mapped.
        rest: More trees, each with ``tree``'s structure.
        leaf: Where the walk stops.
        shared: Whether an aliased value is one leaf or one per path. Set,
            ``f`` runs once per distinct leaf and ``rest`` must alias in the
            same places.

    Returns:
        A new tree with ``tree``'s structure, holding the mapped leaves.

    Raises:
        ValueError: If a tree in ``rest`` does not match ``tree``'s structure.
    """
    flat, treedef = flatten(tree, leaf=leaf, shared=shared)
    columns = [flat]
    for index, other in enumerate(rest):
        other_flat, other_def = flatten(other, leaf=leaf, shared=shared)
        if other_def != treedef:
            # Reported as leaf paths, which are what a caller can act on.
            want, got = treedef.leaf_paths, other_def.leaf_paths
            lines = [
                f"tree structure mismatch: tree argument {index + 2} does "
                "not match the structure of tree argument 1.",
                f"  expected leaves at {list(want)}",
                f"  got leaves at {list(got)}",
            ]
            if want == got:
                lines.append(
                    "  same paths, so the structures differ: "
                    f"{treedef} != {other_def}"
                )
            raise ValueError("\n".join(lines))
        columns.append(other_flat)
    mapped = [f(*row) for row in zip(*columns, strict=True)]
    return unflatten(treedef, mapped)


def update(
    tree: Any,
    state: Mapping[str, Any],
    *,
    leaf: Selector | None = None,
    shared: bool = False,
) -> Any:
    """Writes path-keyed values into ``tree``, in place.

    The counterpart of :func:`paths`, taking the same knobs. Only leaves are
    written, and an unnamed leaf is left alone.

    Args:
        tree: The tree to write into. Mutated.
        state: The values to write, keyed by dotted path as :func:`paths`
            returns them.
        leaf: Where the walk stops, and so what counts as a writable position.
        shared: Whether a value reachable by two paths is one leaf or two. Set,
            one write reaches every path to it, and the tie survives.

    Returns:
        ``tree`` itself, or a rebuilt root when the root is a ``tuple``.

    Raises:
        TypeError: If a value must be written under a positional key and its
            node declares no ``__tree_setattr__``.
        ValueError: If some entry in ``state`` names no leaf in ``tree``.
    """
    is_leaf = as_predicate(leaf, lambda value: not is_node(value))
    unwritten = dict(state)
    descending: set[int] = set()
    #: What each tied leaf was replaced by, so every path agrees.
    written: dict[int, Any] = {}

    def visit(value: Any, path: str) -> Any:
        if is_leaf(value):
            if shared and id(value) in written:
                return written[id(value)]
            # Popped, so whatever is left at the end names what never landed.
            new = unwritten.pop(path, value)
            if shared:
                written[id(value)] = new
            return new
        kind = _node_kind(value)
        if kind is None or id(value) in descending:
            # Static structure has no leaf path; a back-edge closes a cycle.
            return value
        children, keys, _ = flatten_one_level(value)
        child_keys = keys or tuple(range(len(children)))
        descending.add(id(value))
        try:
            updated = [
                visit(child, extend_path(path, key))
                for key, child in zip(child_keys, children, strict=True)
            ]
        finally:
            descending.discard(id(value))
        changed = [
            (key, new)
            for key, old, new in zip(child_keys, children, updated, strict=True)
            if new is not old
        ]
        if not changed:
            return value
        if kind == "list":
            value[:] = updated
        elif kind == "dict":
            for key, new in changed:
                value[key] = new
        elif kind in ("tuple", "namedtuple"):
            # Immutable, so the replacement goes back to the parent to place.
            return (
                type(value)(*updated)
                if kind == "namedtuple"
                else tuple(updated)
            )
        else:
            for key, new in changed:
                _place_child(value, key, new)
        return value

    result = visit(tree, "")
    if unwritten:
        raise ValueError(
            "update: no leaf at "
            f"{sorted(_path_or_root(path) for path in unwritten)}, so these "
            "values were not written. Check them against paths(tree)."
        )
    return result
