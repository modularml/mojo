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

"""Per-shard :class:`~max.graph.Dim` wrapper carrying one value per device.

A :class:`PerShardDim` bundles one :class:`~max.graph.Dim` per mesh shard
(device index along a :class:`Sharded` mesh axis). Arithmetic distributes
element-wise. :func:`make_per_shard_dim` collapses to a plain :class:`Dim`
when every cell is equal. Wrappers must never reach MLIR; they are projected
per shard by shape rules before any :meth:`Dim.to_mlir` call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NoReturn, TypeGuard

from max.graph.dim import Dim, StaticDim, SymbolicDim
from max.graph.shape import Shape

__all__ = [
    "PerShardDim",
    "global_dim",
    "global_shape",
    "is_per_shard_dim",
    "is_static",
    "is_symbolic",
    "local_dim_at",
    "local_shape_at",
    "make_per_shard_dim",
]


def _all_equal(items: Sequence[object]) -> bool:
    if not items:
        return True
    first = items[0]
    return all(x == first for x in items[1:])


class PerShardDim(Dim):
    """A :class:`~max.graph.Dim` whose ``per_shard`` tuple lists one cell per mesh shard.

    Used on :class:`Sharded` axes when shards hold different per-device
    sizes (uneven static splits, dynamic axes minted per-shard). The
    wrapper must be projected per shard via :func:`local_dim_at` before
    reaching MLIR.
    """

    __slots__ = ("_global", "per_shard")

    per_shard: tuple[Dim, ...]
    _global: Dim | None

    def __new__(
        cls,
        per_shard: Iterable[Dim] | PerShardDim = (),
        *,
        global_dim: Dim | None = None,
    ) -> PerShardDim:
        """Allocates the wrapper, returning ``per_shard`` itself on a plain re-wrap."""
        if isinstance(per_shard, PerShardDim) and global_dim is None:
            return per_shard
        return object.__new__(cls)

    def __init__(
        self,
        per_shard: Iterable[Dim] | PerShardDim,
        *,
        global_dim: Dim | None = None,
    ) -> None:
        if isinstance(per_shard, PerShardDim):
            if per_shard is self:
                return
            if global_dim is None:
                global_dim = per_shard._global
            per_shard = per_shard.per_shard
        object.__setattr__(self, "per_shard", tuple(per_shard))
        object.__setattr__(self, "_global", global_dim)

    def to_mlir(self) -> NoReturn:
        """Raises; wrappers must be projected per shard before reaching MLIR."""
        raise TypeError(
            f"PerShardDim cannot be lowered to MLIR: {self!r}. "
            "Project per shard via the shape rule before reaching graph IR."
        )

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Distinct symbolic-dim parameters referenced across all cells."""
        seen: set[str] = set()
        for d in self.per_shard:
            for p in d.parameters:
                if p.name not in seen:
                    seen.add(p.name)
                    yield p

    def __hash__(self) -> int:
        if self._global is not None:
            return hash(self._global)
        return hash(self.per_shard)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PerShardDim):
            return self.per_shard == other.per_shard
        if self._global is not None and isinstance(other, (Dim, int, str)):
            return Dim(self._global) == Dim(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        return NotImplemented if eq is NotImplemented else not eq

    def __str__(self) -> str:
        if self._global is not None:
            return str(self._global)
        return "(" + " | ".join(str(d) for d in self.per_shard) + ")"

    def __repr__(self) -> str:
        if self._global is not None:
            return f"PerShardDim({self.per_shard!r}, global={self._global!r})"
        return f"PerShardDim({self.per_shard!r})"

    def __add__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a + b)

    def __radd__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a + b)

    def __mul__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a * b)

    def __rmul__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a * b)

    def __floordiv__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a // b)

    def __rfloordiv__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a // b)

    def __neg__(self) -> Dim:
        return make_per_shard_dim(tuple(-d for d in self.per_shard))

    def __sub__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a - b)

    def __rsub__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a - b)

    def __int__(self) -> int:
        if self._global is not None:
            return int(self._global)
        if len(self.per_shard) == 1:
            return int(self.per_shard[0])
        raise TypeError(
            f"int(PerShardDim) is undefined for per-shard cells "
            f"{self.per_shard!r}; use symbolic Dim arithmetic instead."
        )

    def __index__(self) -> int:
        return self.__int__()

    @property
    def is_static(self) -> bool:
        """``True`` if this axis's global extent is a static size.

        Folds to the global dim first (see :func:`global_dim`), so a sharded
        axis whose global is static reports ``True`` even though
        ``isinstance(self, StaticDim)`` is ``False``.
        """
        return is_static(self)

    @property
    def is_symbolic(self) -> bool:
        """``True`` if this axis's global extent is a symbolic (named) dim."""
        return is_symbolic(self)


def _cells_of(x: object) -> tuple[Dim, ...] | None:
    """Returns ``x.per_shard`` if ``x`` is a :class:`PerShardDim`, else ``None``."""
    if isinstance(x, PerShardDim):
        return x.per_shard
    return None


def _global_or_none(x: object) -> Dim | None:
    """Returns the global :class:`Dim` for ``x``, or ``None`` if unavailable.

    Unlike :func:`global_dim`, this never folds per-shard cells: it returns the
    recorded global of a :class:`PerShardDim` (which may be ``None``) or the dim
    itself for a plain :class:`~max.graph.Dim`. Used to decide whether a global
    can be propagated through :func:`_binop`.
    """
    if isinstance(x, PerShardDim):
        return x._global
    if isinstance(x, (int, str, Dim)):
        return Dim(x)
    return None


def _binop(
    lhs: object,
    rhs: object,
    op: Callable[[Dim, Dim], Dim],
) -> Dim:
    lhs_cells = _cells_of(lhs)
    rhs_cells = _cells_of(rhs)
    if lhs_cells is not None and rhs_cells is not None:
        if len(lhs_cells) != len(rhs_cells):
            raise ValueError(
                f"PerShardDim arity mismatch: "
                f"{len(lhs_cells)} vs {len(rhs_cells)}."
            )
        cells = tuple(
            op(a, b) for a, b in zip(lhs_cells, rhs_cells, strict=False)
        )
    elif lhs_cells is not None:
        assert isinstance(rhs, (int, str, Dim))
        rhs_dim = Dim(rhs)
        cells = tuple(op(a, rhs_dim) for a in lhs_cells)
    elif rhs_cells is not None:
        assert isinstance(lhs, (int, str, Dim))
        lhs_dim = Dim(lhs)
        cells = tuple(op(lhs_dim, b) for b in rhs_cells)
    else:
        raise TypeError(
            f"_binop called without per-shard operand: {lhs!r}, {rhs!r}"
        )
    g_lhs, g_rhs = _global_or_none(lhs), _global_or_none(rhs)
    result_global = (
        op(g_lhs, g_rhs) if g_lhs is not None and g_rhs is not None else None
    )
    return make_per_shard_dim(cells, global_dim=result_global, force_wrap=True)


def make_per_shard_dim(
    per_shard: Sequence[Dim],
    *,
    global_dim: Dim | None = None,
    force_wrap: bool = False,
) -> Dim:
    """Wraps ``per_shard`` in a :class:`PerShardDim`, or collapses to a plain :class:`Dim`.

    Collapses to ``per_shard[0]`` when every entry is equal and no
    ``global_dim`` is attached. Pass ``force_wrap=True`` to keep the
    wrapper even when entries happen to be equal (used on Sharded axes so
    :func:`global_dim` recovers the global extent), or pass ``global_dim``
    to record the axis's global size for ``int()``/display.
    """
    if not per_shard:
        raise ValueError("make_per_shard_dim: empty per_shard tuple.")
    cells = tuple(Dim(d) for d in per_shard)
    if not force_wrap and global_dim is None and _all_equal(cells):
        return cells[0]
    return PerShardDim(cells, global_dim=global_dim)


def global_dim(d: Dim) -> Dim:
    """Folds a :class:`PerShardDim` into a single global :class:`Dim`.

    Returns the wrapper's recorded global when present, else sums the
    per-shard cells (the common single-mesh-axis :class:`Sharded` case;
    multi-mesh-axis sharding requires the placement's
    :meth:`Placement.global_dim`).
    """
    if not is_per_shard_dim(d):
        return d
    if d._global is not None:
        return d._global
    total: Dim = d.per_shard[0]
    for x in d.per_shard[1:]:
        total = total + x
    return total


def is_per_shard_dim(d: object) -> TypeGuard[PerShardDim]:
    """``True`` if ``d`` is a :class:`PerShardDim`."""
    return isinstance(d, PerShardDim)


def is_static(d: Dim) -> bool:
    """``True`` if ``d``'s global extent is a static (compile-time) size.

    A :class:`PerShardDim` is judged by its global dim (see :func:`global_dim`),
    so a sharded axis whose global is static reports ``True`` even though
    ``isinstance(d, StaticDim)`` would not. Prefer this over
    ``isinstance(dim, StaticDim)`` on a possibly-sharded shape.
    """
    return isinstance(global_dim(d), StaticDim)


def is_symbolic(d: Dim) -> bool:
    """``True`` if ``d``'s global extent is a symbolic (named) dim.

    Folds a :class:`PerShardDim` to its global first, mirroring
    :func:`is_static`.
    """
    return isinstance(global_dim(d), SymbolicDim)


def local_dim_at(d: Dim, shard: int) -> Dim:
    """Returns the dimension value at ``shard``.

    If ``d`` is a regular (non-distributed) :class:`Dim`, then ``d`` is
    returned unchanged.
    """
    if is_per_shard_dim(d):
        return d.per_shard[shard]
    return d


def local_shape_at(shape: Sequence[Dim], shard: int) -> Shape:
    """Projects every axis of ``shape`` onto a single ``shard`` index."""
    return Shape([local_dim_at(Dim(d), shard) for d in shape])


def global_shape(shape: Sequence[Dim]) -> Shape:
    """Returns the global shape by summing every :class:`PerShardDim` axis."""
    return Shape([global_dim(Dim(d)) for d in shape])
