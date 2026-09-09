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

"""Placement rules for shape-family ops (``reshape``, ``transpose``, ``split``, ``stack``, ``gather``, ...)."""

from __future__ import annotations

import builtins
import functools
import itertools
import operator
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Placement,
    PlacementMapping,
    Sharded,
)
from max.experimental.sharding.per_shard_dim import (
    global_dim,
    is_per_shard_dim,
    local_dim_at,
    make_per_shard_dim,
)
from max.experimental.sharding.placements import _shard_sizes_along_axis
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim, DimLike, StaticDim
from max.graph.ops.slice_tensor import SliceIndex, SliceIndices
from max.graph.shape import Shape

from ..action import Action, ActionSet, AxisAssignment, PerShard
from ..cost import P, R, build_action_set

# ── split / slice helpers (only consumers of these are split_rule / slice_tensor_rule) ──


def _localize_sizes(
    sizes: Sequence[DimLike],
    axis: int,
    ndim: int,
    placements: tuple[Placement, ...],
    mesh: DeviceMesh,
) -> list[Dim] | PerShard:
    """Adaptive per-rank size kwarg for :func:`split`.

    When the split axis is sharded, divisible static sizes return one list
    of plain Dims; non-divisible static sizes are split unevenly via
    :func:`_shard_sizes_along_axis` and returned as :class:`PerShard`.
    """
    norm = axis % ndim
    sharded = [
        (ax, mesh.mesh_shape[ax])
        for ax, p in enumerate(placements)
        if p.localized_axis() == norm
    ]
    if not sharded:
        return [Dim(s) for s in sizes]

    diverges = any(
        not isinstance(Dim(s), StaticDim)
        or any(int(Dim(s)) % msz != 0 for _, msz in sharded)
        for s in sizes
    )
    if not diverges:
        local = [Dim(s) for s in sizes]
        for _, msz in sharded:
            local = [d // msz for d in local]
        return local

    n_devices = mesh.num_devices
    msh_ax, msz = sharded[0]
    stride = 1
    for k in builtins.range(msh_ax + 1, mesh.ndim):
        stride *= mesh.mesh_shape[k]
    per_rank: list[list[Dim]] = [[] for _ in builtins.range(n_devices)]
    for s in sizes:
        sd = Dim(s)
        if isinstance(sd, StaticDim):
            chunks = _shard_sizes_along_axis(int(sd), msz)
            for d in builtins.range(n_devices):
                per_rank[d].append(StaticDim(chunks[(d // stride) % msz]))
        else:
            divided = sd // msz
            for d in builtins.range(n_devices):
                per_rank[d].append(divided)
    return PerShard(per_rank)


def _expand_ellipsis(
    indices: Sequence[SliceIndex], ndim: int
) -> list[SliceIndex]:
    n_explicit = builtins.sum(1 for i in indices if i is not Ellipsis)
    out: list[SliceIndex] = []
    for idx in indices:
        if idx is Ellipsis:
            out.extend([slice(None)] * (ndim - n_explicit))
        else:
            out.append(idx)
    while len(out) < ndim:
        out.append(slice(None))
    return out


def _slice_modifies_axis(indices: SliceIndices, axis: int, ndim: int) -> bool:
    """``True`` if the slice expression touches ``axis``."""
    if not isinstance(indices, (list, tuple)):
        return True
    expanded = _expand_ellipsis(indices, ndim)
    return bool(axis < len(expanded) and expanded[axis] != slice(None))


def _is_minus_one(d: Dim) -> bool:
    return isinstance(d, StaticDim) and d.dim == -1


def _resolve_minus_one(
    src_dims: Sequence[Dim], tgt_dims: Sequence[Dim]
) -> list[Dim]:
    """Replaces a single ``-1`` in ``tgt_dims`` with ``prod(src) // prod(others)``.

    Returns ``tgt_dims`` unchanged when no ``-1`` is present or more
    than one is present.
    """
    pos: int | None = None
    for i, d in enumerate(tgt_dims):
        if _is_minus_one(d):
            if pos is not None:
                return list(tgt_dims)
            pos = i
    if pos is None:
        return list(tgt_dims)
    src_prod = functools.reduce(
        operator.mul, (Dim(d) for d in src_dims), Dim(1)
    )
    other_prod = functools.reduce(
        operator.mul,
        (Dim(d) for i, d in enumerate(tgt_dims) if i != pos),
        Dim(1),
    )
    out = list(tgt_dims)
    # Zero-cell denominator (e.g. static 1 sharded by mesh size 4):
    # leave ``-1`` unresolved so the IR rejects the action cleanly.
    if isinstance(other_prod, StaticDim) and other_prod.dim == 0:
        return list(tgt_dims)
    out[pos] = src_prod // other_prod
    return out


def _per_rank_target(
    target: Sequence[DimLike],
    src_shape: Sequence[DimLike],
    out_placements: tuple[Placement, ...],
    mesh: DeviceMesh,
) -> PerShard:
    """Returns the per-rank local target shape.

    Wrappers and ``-1`` pass through; other dims are lifted via the
    output placements (rejecting bare symbolic targets), then projected
    per rank and any remaining ``-1`` is resolved.
    """
    n = mesh.num_devices
    target_dims = [Dim(d) for d in target]

    lifted: list[Dim] = []
    for ti, td in enumerate(target_dims):
        if is_per_shard_dim(td) or _is_minus_one(td):
            lifted.append(td)
            continue
        d: Dim = td
        for mesh_axis, p in enumerate(out_placements):
            if p.localized_axis() == ti:
                d = p.local_dim(d, mesh, mesh_axis, allow_symbolic_mint=False)
        lifted.append(d)

    per_rank_shapes: list[Shape] = []
    for r in range(n):
        src_r = [local_dim_at(Dim(d), r) for d in src_shape]
        tgt_r: list[Dim] = []
        for d in lifted:
            if is_per_shard_dim(d):
                assert isinstance(d, Dim)
                tgt_r.append(local_dim_at(d, r))
            else:
                tgt_r.append(d)
        if any(_is_minus_one(d) for d in tgt_r):
            tgt_r = _resolve_minus_one(src_r, tgt_r)
        per_rank_shapes.append(Shape(tgt_r))
    return PerShard(per_rank_shapes)


def _repack_finalize(
    n: int, axis: int, is_tuple: bool
) -> Callable[[Action], Action]:
    """Builds the ``concat`` / ``stack`` finalize, pre-bound to its context.

    Args:
        n: Number of variadic tensor inputs.
        axis: The concat/stack axis kwarg.
        is_tuple: ``True`` when the user passed a ``tuple`` (vs. ``list``).
    """

    def finalize(action: Action) -> Action:
        mappings = action.inputs[:n]
        container: list[DeviceMapping] | tuple[DeviceMapping, ...] = (
            tuple(mappings) if is_tuple else list(mappings)
        )
        return Action(inputs=(container, axis), outputs=action.outputs)

    return finalize


def passthrough_rule(x: TensorLayout, linear: bool = False) -> ActionSet:
    """Strategies for ops that preserve every input axis (such as activation, cast)."""
    rows = [
        AxisAssignment((R,), R),
        *(AxisAssignment((Sharded(d),), Sharded(d)) for d in range(x.rank)),
    ]
    if linear:
        rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(linear,))


def tile_rule(x: TensorLayout, repeats: Iterable[DimLike]) -> ActionSet:
    """Strategies for ``tile``: any input sharding preserved on output axes."""
    rows = [
        AxisAssignment((R,), R),
        *(AxisAssignment((Sharded(d),), Sharded(d)) for d in range(x.rank)),
        AxisAssignment((P,), P),
    ]
    return build_action_set(rows, layouts=(x,), extras=(repeats,))


def permute_rule(x: TensorLayout, dims: Sequence[int]) -> ActionSet:
    """Strategies for ``permute``: sharding follows the axis permutation."""
    dims_list = list(dims)
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for in_ax in range(x.rank):
        out_ax = dims_list.index(in_ax)
        rows.append(AxisAssignment((Sharded(in_ax),), Sharded(out_ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(dims,))


def transpose_rule(x: TensorLayout, axis_1: int, axis_2: int) -> ActionSet:
    """Strategies for ``transpose``: sharding swaps along with the two axes."""
    n = x.rank
    a1, a2 = axis_1 % n, axis_2 % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for in_ax in range(n):
        out_ax = a2 if in_ax == a1 else a1 if in_ax == a2 else in_ax
        rows.append(AxisAssignment((Sharded(in_ax),), Sharded(out_ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(axis_1, axis_2))


def unsqueeze_rule(x: TensorLayout, axis: int) -> ActionSet:
    """Strategies for ``unsqueeze``: inserts a size-1 axis; sharding shifts."""
    n = x.rank
    norm = axis if axis >= 0 else axis + n + 1
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for in_ax in range(n):
        out_ax = in_ax if in_ax < norm else in_ax + 1
        rows.append(AxisAssignment((Sharded(in_ax),), Sharded(out_ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(axis,))


def squeeze_rule(x: TensorLayout, axis: int) -> ActionSet:
    """Strategies for ``squeeze``: removes a size-1 axis; sharding shifts."""
    n = x.rank
    norm = axis % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for in_ax in range(n):
        if in_ax == norm:
            continue
        out_ax = in_ax if in_ax < norm else in_ax - 1
        rows.append(AxisAssignment((Sharded(in_ax),), Sharded(out_ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(axis,))


def flatten_rule(
    x: TensorLayout, start_dim: int = 0, end_dim: int = -1
) -> ActionSet:
    """Strategies for ``flatten``: sharding only on axes outside the flattened range."""
    n = x.rank
    sd = start_dim if start_dim >= 0 else start_dim + n
    ed = end_dim if end_dim >= 0 else end_dim + n
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for in_ax in range(n):
        if sd <= in_ax <= ed:
            continue
        out_ax = in_ax if in_ax < sd else in_ax - (ed - sd)
        rows.append(AxisAssignment((Sharded(in_ax),), Sharded(out_ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(start_dim, end_dim))


def _reshape_finalize(
    x: TensorLayout, shape: Shape
) -> Callable[[Action], Action]:
    """Builds the ``reshape`` finalize (per-rank projection), pre-bound."""

    def finalize(action: Action) -> Action:
        in_mapping = action.inputs[0]
        assert isinstance(in_mapping, PlacementMapping)
        out_placements = action.outputs[0].placements
        local = _per_rank_target(shape, x.shape, out_placements, x.mapping.mesh)
        return Action(inputs=(in_mapping, local), outputs=action.outputs)

    return finalize


def _cells_match_product(target: Dim, src_dims: Sequence[Dim]) -> bool:
    """True iff ``target``'s per-rank cells equal the cell-wise product of ``src_dims``."""
    if not is_per_shard_dim(target):
        return False
    target_cells = target.per_shard
    n = len(target_cells)
    products: list[Dim] = [Dim(1)] * n
    for src in src_dims:
        src_cells: tuple[Dim, ...] | list[Dim]
        if is_per_shard_dim(src):
            src_cells = src.per_shard
            if len(src_cells) != n:
                return False
        else:
            src_cells = [Dim(src)] * n
        products = [
            Dim(p) * Dim(c) for p, c in zip(products, src_cells, strict=False)
        ]
    return all(
        Dim(t) == Dim(p) for t, p in zip(target_cells, products, strict=False)
    )


def _find_wrapper_landings(
    sharded_src_axes: Sequence[int],
    src_shape: Sequence[Dim],
    target_shape: Sequence[Dim],
) -> dict[int, int]:
    """Maps each sharded source axis to the target axis it lands on.

    For each target axis carrying a per-rank wrapper, finds the largest
    subset of source axes whose cell-wise product matches the wrapper's
    cells.
    """
    if not sharded_src_axes:
        return {}
    sharded_set = set(sharded_src_axes)
    landings: dict[int, int] = {}
    target_wrappers = [
        (t, Dim(td))
        for t, td in enumerate(target_shape)
        if is_per_shard_dim(Dim(td))
    ]
    all_src_axes = list(range(len(src_shape)))
    for t, td in target_wrappers:
        matched_subset: tuple[int, ...] | None = None
        for size in range(len(all_src_axes), 0, -1):
            for subset in itertools.combinations(all_src_axes, size):
                if not any(k in sharded_set for k in subset):
                    continue
                if _cells_match_product(
                    td, [Dim(src_shape[k]) for k in subset]
                ):
                    matched_subset = subset
                    break
            if matched_subset is not None:
                break
        if matched_subset is None:
            continue
        for k in matched_subset:
            if k in sharded_set and k not in landings:
                landings[k] = t
    return landings


def _structural_split_route(
    in_ax: int, x: TensorLayout, new_shape: Sequence[Dim]
) -> int | None:
    """Returns the target axis a ``Sharded(in_ax)`` lands on.

    Compares cumulative-product boundaries on source and target axes.
    Source axis ``in_ax`` covers ``[src_cum[in_ax], src_cum[in_ax+1]]``;
    a target axis ``j`` is a landing iff its range either lies inside
    (split) or contains (merge / 1-1) the source axis's range. Symbolic
    dims on both sides contribute ``1`` (matching :func:`_global_axis_size`
    for sources); a single ``-1`` in the target is resolved against the
    source total before alignment. Returns :data:`None` for ambiguous
    correspondences (boundaries cross).
    """
    from ..cost import _global_axis_size

    # Identity passthrough: same rank and same per-axis size on both
    # sides => axis ``in_ax`` lands at ``in_ax``. Catches reshape-to-self
    # cases (including uneven static splits) where ``_global_axis_size``
    # would misreport the global due to its even-split assumption.
    if x.rank == len(new_shape):
        match = True
        for i in range(x.rank):
            dim = x.shape[i]
            if is_per_shard_dim(dim) and all(
                isinstance(c, StaticDim) for c in dim.per_shard
            ):
                src_i = sum(
                    c.dim for c in dim.per_shard if isinstance(c, StaticDim)
                )
            else:
                src_i = _global_axis_size(x, i)
            # A per-shard target dim contributes its global size here.
            tgt = global_dim(Dim(new_shape[i]))
            if not isinstance(tgt, StaticDim) or tgt.dim != src_i:
                match = False
                break
        if match:
            return in_ax

    in_sizes = [_global_axis_size(x, i) for i in range(x.rank)]
    if any(s <= 0 for s in in_sizes):
        return None
    src_total = 1
    for s in in_sizes:
        src_total *= s

    target_sizes: list[int] = []
    minus_one_idx: int | None = None
    other_prod = 1
    for d in new_shape:
        dd = global_dim(Dim(d))
        if _is_minus_one(dd):
            if minus_one_idx is not None:
                return None
            minus_one_idx = len(target_sizes)
            target_sizes.append(1)  # placeholder
        elif isinstance(dd, StaticDim):
            target_sizes.append(dd.dim)
            other_prod *= dd.dim
        else:
            target_sizes.append(1)  # symbolic → opaque size 1
    if minus_one_idx is not None:
        if other_prod <= 0 or src_total % other_prod != 0:
            return None
        target_sizes[minus_one_idx] = src_total // other_prod

    src_cum = [1]
    for s in in_sizes:
        src_cum.append(src_cum[-1] * s)
    tgt_cum = [1]
    for t in target_sizes:
        tgt_cum.append(tgt_cum[-1] * t)
    if src_cum[-1] != tgt_cum[-1]:
        return None

    src_left, src_right = src_cum[in_ax], src_cum[in_ax + 1]
    for j in range(len(target_sizes)):
        tl, tr = tgt_cum[j], tgt_cum[j + 1]
        if tr <= src_left:
            continue
        if tl >= src_right:
            break
        if src_left <= tl and tr <= src_right:
            return j  # split: target axis lies inside source's range
        if tl <= src_left and src_right <= tr:
            return j  # merge / 1-1: target range contains source's range
        return None  # boundaries cross — ambiguous
    return None


def reshape_rule(x: TensorLayout, shape: Any) -> ActionSet:
    """Strategies for ``reshape``.

    Wrapper landings (:func:`_find_wrapper_landings`) take precedence
    for per-rank-wrapper target axes; otherwise the structural split
    (:func:`_structural_split_route`) routes by left-to-right size
    correspondence. Iteration is per mesh axis so two placements on
    the same tensor axis can route to two different targets. A target
    taken from ``Tensor.shape`` already carries per-device cells on its
    :class:`PerShardDim` axes, so those land directly.
    """
    new_shape = Shape(shape)
    if sum(1 for d in new_shape if _is_minus_one(d)) > 1:
        raise ValueError(
            f"reshape: at most one -1 dimension is allowed (target {new_shape})."
        )
    sharded_src_axes = sorted(
        {
            ax
            for p in x.mapping.to_placements()
            if (ax := p.localized_axis()) is not None
        }
    )
    landings = _find_wrapper_landings(sharded_src_axes, x.shape, new_shape)
    # ``-1`` shorthand: when target has exactly one ``-1`` and exactly one
    # sharded source axis remains un-landed, land it on the ``-1`` slot.
    minus_one_axes = [
        i for i, d in enumerate(new_shape) if _is_minus_one(Dim(d))
    ]

    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    placements = x.mapping.to_placements()
    seen_placements: set[Sharded] = set()
    for p in placements:
        if not isinstance(p, Sharded) or p in seen_placements:
            continue
        seen_placements.add(p)
        k = p.axis
        # Wrapper landing handles per-rank-wrapper target axes.
        if k in landings:
            rows.append(
                AxisAssignment(
                    (p,),
                    Sharded(landings[k]),
                )
            )
            continue
        # Structural split: handles un-merger and contiguous splits.
        # Single landing per sharded source axis; no fan-out fallback
        # so the picker cannot guess a wrong target axis.
        route = _structural_split_route(k, x, new_shape)
        if route is not None:
            rows.append(AxisAssignment((p,), Sharded(route)))
            continue
        # ``-1`` shorthand: unique sharded source + unique ``-1`` in
        # target ⇒ land on the ``-1`` slot.
        if len(minus_one_axes) == 1 and len(sharded_src_axes) == 1:
            rows.append(AxisAssignment((p,), Sharded(minus_one_axes[0])))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(
        rows,
        layouts=(x,),
        extras=(shape,),
        result_shape=new_shape,
        finalize=_reshape_finalize(x=x, shape=new_shape),
    )


def _rebind_finalize(
    shape: Shape, message: str, layout: object
) -> Callable[[Action], Action]:
    """Builds the ``rebind`` finalize, pre-bound to its context.

    Projects the rebind target shape per-rank under the input's placement.
    For each target axis, the per-shard size is:

    - the cell from a :class:`PerShardDim` wrapper if the caller passed one;
    - a :class:`PerShardDim` carrying load-balanced cells
      ``((r + 1) * target[i]) // n - (r * target[i]) // n`` when the
      input is :class:`Sharded` on axis ``i`` via a mesh axis of group
      ``n``, so the per-shard form matches what
      :func:`_even_split_along_axis` produces on the other side of a
      residual add;
    - the target dim unchanged otherwise.

    Static dims on a sharded axis use :meth:`Sharded.local_dim`'s
    uneven divmod (so e.g. ``Sharded(0)`` of static ``5`` on ``n=3``
    yields per-shard ``[2, 2, 1]``); symbolic and algebraic dims use
    the same load-balanced formula so dynamic dims not divisible by
    the mesh axis size are preserved exactly.
    """

    def finalize(action: Action) -> Action:
        in_mapping = action.inputs[0]
        assert isinstance(in_mapping, PlacementMapping)
        mesh = in_mapping.mesh

        def _local(ti: int, td: DimLike) -> Dim:
            d = Dim(td)
            if is_per_shard_dim(d):
                return d
            for mesh_axis, p in enumerate(in_mapping.placements):
                if not isinstance(p, Sharded) or p.localized_axis() != ti:
                    continue
                group = mesh.mesh_shape[mesh_axis]
                if isinstance(d, StaticDim):
                    d = p.local_dim(
                        d, mesh, mesh_axis, allow_symbolic_mint=False
                    )
                else:
                    cells = tuple(
                        ((r + 1) * d) // group - (r * d) // group
                        for r in range(group)
                    )
                    d = make_per_shard_dim(cells)
            return d

        lifted = [_local(i, td) for i, td in enumerate(shape)]
        per_rank = PerShard(
            [
                Shape(
                    local_dim_at(d, r) if is_per_shard_dim(d) else d
                    for d in lifted
                )
                for r in range(mesh.num_devices)
            ]
        )
        return Action(
            inputs=(in_mapping, per_rank, message, layout),
            outputs=action.outputs,
        )

    return finalize


def rebind_rule(
    x: TensorLayout,
    shape: Any,
    message: str = "",
    layout: object = None,
) -> ActionSet:
    """Strategies for ``rebind``: strict identity on placement.

    Rebind is a runtime shape assertion. It must never insert a
    collective. The rule emits exactly one passthrough row per unique
    placement on the input mesh axes, so the picker's only feasible
    choice is the input's own placement, independent of solver. The
    finalize hook projects the user-supplied target shape per-rank
    against that placement.
    """
    target_shape = Shape(shape)
    placements = x.mapping.to_placements()
    rows: list[AxisAssignment] = []
    seen: set[Placement] = set()
    for p in placements:
        if p in seen:
            continue
        seen.add(p)
        rows.append(AxisAssignment((p,), p))
    return build_action_set(
        rows,
        layouts=(x,),
        extras=(shape, message, layout),
        finalize=_rebind_finalize(
            shape=target_shape, message=message, layout=layout
        ),
    )


def _broadcast_to_finalize(
    x: TensorLayout, shape: list[Dim], out_dims: Iterable[DimLike] | None
) -> Callable[[Action], Action]:
    """Builds the ``broadcast_to`` finalize (per-rank projection), pre-bound."""

    def finalize(action: Action) -> Action:
        in_mapping = action.inputs[0]
        assert isinstance(in_mapping, PlacementMapping)
        out_placements = action.outputs[0].placements
        local = _per_rank_target(shape, x.shape, out_placements, x.mapping.mesh)
        return Action(
            inputs=(in_mapping, local, out_dims),
            outputs=action.outputs,
        )

    return finalize


def broadcast_to_rule(
    x: TensorLayout,
    shape: Any,
    out_dims: Iterable[DimLike] | None = None,
) -> ActionSet:
    """Strategies for ``broadcast_to`` with right-aligned 1:1 axis landing."""
    if isinstance(shape, TensorLayout):
        raise NotImplementedError(
            "broadcast_to does not support a tensor-valued shape; pass a "
            "ShapeLike (list of DimLike)."
        )
    target_shape = list(shape)
    from max.experimental.tensor import _fold_sharded_shape

    src_global = _fold_sharded_shape(x.shape, x.mapping)
    for i in range(1, min(len(src_global), len(target_shape)) + 1):
        s_global = Dim(src_global[-i])
        t_dim = Dim(target_shape[-i])
        if is_per_shard_dim(t_dim):
            continue
        if s_global == t_dim or s_global == 1:
            continue
        raise ValueError(
            f"broadcast_to: input dimension {-i} (size {s_global}) must be "
            f"either 1 or equal to the target size {t_dim}."
        )

    placements = x.mapping.to_placements()
    sharded_src_axes = sorted(
        {ax for p in placements if (ax := p.localized_axis()) is not None}
    )
    n_src, n_tgt = x.rank, len(target_shape)
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for k in sharded_src_axes:
        t = k + (n_tgt - n_src)
        if t < 0 or t >= n_tgt:
            continue
        if Dim(src_global[k]) == Dim(1):
            continue
        rows.append(AxisAssignment((Sharded(k),), Sharded(t)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(
        rows,
        layouts=(x,),
        extras=(shape, out_dims),
        finalize=_broadcast_to_finalize(
            x=x, shape=target_shape, out_dims=out_dims
        ),
    )


def _concat_stack_rows(
    layouts: tuple[TensorLayout, ...],
    out_axis_for_in: int | None,
    norm: int | None,
) -> list[AxisAssignment]:
    """Rows shared by ``concat``/``stack``: every input wears the same placement."""
    n = len(layouts)
    rank = layouts[0].rank
    rows = [AxisAssignment((R,) * n, R)]
    for ax in range(rank):
        if norm is None:
            out_ax = ax
        else:
            out_ax = ax if ax < norm else ax + 1
        rows.append(AxisAssignment((Sharded(ax),) * n, Sharded(out_ax)))
    rows.append(AxisAssignment((P,) * n, P))
    return rows


def concat_rule(
    original_vals: Iterable[TensorLayout], axis: int = 0
) -> ActionSet:
    """Concat rule — all inputs share placement; finalize repacks list/tuple."""
    layouts = tuple(original_vals)
    if not layouts:
        raise ValueError("concat: no tensor inputs.")
    rows = _concat_stack_rows(layouts, out_axis_for_in=None, norm=None)
    return build_action_set(
        rows,
        layouts=layouts,
        extras=(axis,),
        finalize=_repack_finalize(
            n=len(layouts), axis=axis, is_tuple=isinstance(original_vals, tuple)
        ),
    )


def stack_rule(values: Iterable[TensorLayout], axis: int = 0) -> ActionSet:
    """Stack rule — inserts a new axis; finalize repacks the input container."""
    layouts = tuple(values)
    if not layouts:
        raise ValueError("stack: no tensor inputs.")
    rank = layouts[0].rank
    norm = axis if axis >= 0 else axis + rank + 1
    rows = _concat_stack_rows(layouts, out_axis_for_in=None, norm=norm)
    return build_action_set(
        rows,
        layouts=layouts,
        extras=(axis,),
        finalize=_repack_finalize(
            n=len(layouts), axis=axis, is_tuple=isinstance(values, tuple)
        ),
    )


def chunk_rule(x: TensorLayout, chunks: int, axis: int = 0) -> ActionSet:
    """Strategies for ``chunk``: sharding on any axis except the chunk axis."""
    n = x.rank
    norm = axis % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for ax in range(n):
        if ax == norm:
            continue
        rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(chunks, axis))


def top_k_rule(input: TensorLayout, k: int, axis: int = -1) -> ActionSet:
    """Strategies for ``top_k``: sharding on any axis except the reduction axis."""
    n = input.rank
    norm = axis % n
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for ax in range(n):
        if ax == norm:
            continue
        rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
    return build_action_set(rows, layouts=(input,), extras=(k, axis))


def argsort_rule(x: TensorLayout, ascending: bool = True) -> ActionSet:
    """Strategies for ``argsort``: Replicated only (sort needs full view)."""
    return build_action_set(
        [AxisAssignment((R,), R)], layouts=(x,), extras=(ascending,)
    )


def nonzero_rule(x: TensorLayout, out_dim: DimLike) -> ActionSet:
    """Strategies for ``nonzero``: Replicated only (data-dependent output shape)."""
    return build_action_set(
        [AxisAssignment((R,), R)], layouts=(x,), extras=(out_dim,)
    )


def repeat_interleave_rule(
    x: TensorLayout,
    repeats: int | TensorLayout,
    axis: int | None = None,
    out_dim: DimLike | None = None,
) -> ActionSet:
    """Strategies for ``repeat_interleave``: sharding on any non-repeat axis."""
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    if axis is not None:
        n = x.rank
        norm = axis % n
        for ax in range(n):
            if ax == norm:
                continue
            rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
        rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(repeats, axis, out_dim))


def pad_rule(
    input: TensorLayout,
    paddings: Iterable[int],
    mode: str = "constant",
    value: TensorLayout | float = 0,
) -> ActionSet:
    """Strategies for ``pad``: sharding allowed on unpadded axes only."""
    pads = tuple(paddings)
    padded = {
        i // 2
        for i in range(0, len(pads) - 1, 2)
        if pads[i] != 0 or pads[i + 1] != 0
    }
    linear = mode != "constant" or (
        isinstance(value, (int, float)) and value == 0
    )
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for ax in range(input.rank):
        if ax in padded:
            continue
        rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
    if linear:
        rows.append(AxisAssignment((P,), P))
    return build_action_set(
        rows, layouts=(input,), extras=(paddings, mode, value)
    )


def slice_tensor_rule(x: TensorLayout, indices: SliceIndices) -> ActionSet:
    """Strategies for ``slice_tensor``: sharding allowed on non-sliced axes only."""
    sliced = (
        {
            ax
            for ax in range(x.rank)
            if _slice_modifies_axis(indices, ax, x.rank)
        }
        if indices is not None
        else set()
    )
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for ax in range(x.rank):
        if ax in sliced:
            continue
        rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(rows, layouts=(x,), extras=(indices,))


def gather_rule(
    input: TensorLayout, indices: TensorLayout, axis: int
) -> ActionSet:
    """Strategies for ``gather``: sharding follows either input or indices axes.

    Deliberately does not emit the expert-parallel
    ``(Sharded(a_axis), R) -> Partial(SUM)`` row. That row treats a
    local gather on each rank as if the missing cross-rank entries
    were zero and then sums — only correct when the caller has masked
    indices to each rank's owned slice. Letting the picker pick it
    silently produces wrong results in the common case (e.g. gathering
    a few token positions out of a seq-sharded residual stream). When
    the gather axis is sharded under this rule, the picker must
    ``allgather`` to :class:`Replicated` first. Callers that genuinely
    want EP semantics override ``gather.rule`` with their own rule
    (see :doc:`rules/README` "Adding rows for a custom placement type").
    """
    in_r, idx_r = input.rank, indices.rank
    a_axis = axis % in_r
    rows: list[AxisAssignment] = [AxisAssignment((R, R), R)]
    for in_ax in range(in_r):
        if in_ax == a_axis:
            continue
        out_ax = in_ax if in_ax < a_axis else in_ax + idx_r - 1
        rows.append(AxisAssignment((Sharded(in_ax), R), Sharded(out_ax)))
    for idx_ax in range(idx_r):
        rows.append(
            AxisAssignment((R, Sharded(idx_ax)), Sharded(a_axis + idx_ax))
        )
    # Indices' sharding wins on the cross.
    for in_ax in range(in_r):
        if in_ax == a_axis:
            continue
        for idx_ax in range(idx_r):
            rows.append(
                AxisAssignment(
                    (Sharded(in_ax), Sharded(idx_ax)),
                    Sharded(a_axis + idx_ax),
                )
            )
    rows.append(AxisAssignment((P, R), P))
    return build_action_set(rows, layouts=(input, indices), extras=(axis,))


def gather_nd_rule(
    input: TensorLayout, indices: TensorLayout, batch_dims: int = 0
) -> ActionSet:
    """Strategies for ``gather_nd``: sharding on batch dims only."""
    rows: list[AxisAssignment] = [AxisAssignment((R, R), R)]
    for ax in range(batch_dims):
        rows.append(AxisAssignment((Sharded(ax), Sharded(ax)), Sharded(ax)))
    rows.append(AxisAssignment((P, R), P))
    return build_action_set(
        rows, layouts=(input, indices), extras=(batch_dims,)
    )


def _scatter_rows(input: TensorLayout, axis: int) -> list[AxisAssignment]:
    """Builds the menu for ``scatter`` / ``scatter_add``.

    Non-scatter axes are batch-like (``updates``/``indices`` share the
    input's extent there), so all three operands shard together on them.
    Replicating ``updates``/``indices`` against a sharded input would give
    each device the full batch extent against a partial input shard, and the
    scatter kernel would index outside the shard.
    """
    in_r = input.rank
    a_axis = axis % in_r
    rows: list[AxisAssignment] = [AxisAssignment((R, R, R), R)]
    for ax in range(in_r):
        if ax == a_axis:
            continue
        rows.append(
            AxisAssignment((Sharded(ax), Sharded(ax), Sharded(ax)), Sharded(ax))
        )
    return rows


def scatter_rule(
    input: TensorLayout,
    updates: TensorLayout,
    indices: TensorLayout,
    axis: int = -1,
) -> ActionSet:
    """Strategies for ``scatter``: sharding on any axis except the scatter axis."""
    return build_action_set(
        _scatter_rows(input, axis),
        layouts=(input, updates, indices),
        extras=(axis,),
    )


def scatter_add_rule(
    input: TensorLayout,
    updates: TensorLayout,
    indices: TensorLayout,
    axis: int = -1,
) -> ActionSet:
    """Strategies for ``scatter_add``: same as ``scatter`` (accumulating variant)."""
    return build_action_set(
        _scatter_rows(input, axis),
        layouts=(input, updates, indices),
        extras=(axis,),
    )


def _split_finalize(
    x: TensorLayout, split_sizes: Sequence[DimLike], axis: int
) -> Callable[[Action], Action]:
    """Builds the ``split`` finalize, pre-bound to its context."""

    def finalize(action: Action) -> Action:
        chosen = action.inputs[0]
        assert isinstance(chosen, PlacementMapping)
        local_sizes = _localize_sizes(
            [Dim(sz) for sz in split_sizes],
            axis,
            x.rank,
            chosen.to_placements(),
            x.mapping.mesh,
        )
        return Action(
            inputs=(chosen, local_sizes, axis), outputs=action.outputs
        )

    return finalize


def split_rule(
    x: TensorLayout, split_sizes: Sequence[DimLike], axis: int = 0
) -> ActionSet:
    """Strategies for ``split``: sharding on any axis except the split axis."""
    n = x.rank
    norm = axis % n
    mesh = x.mapping.mesh
    max_group = max(mesh.mesh_shape) if mesh.mesh_shape else 1
    normalized = [Dim(sz) for sz in split_sizes]
    split_axis_ok = all(
        not isinstance(sz, StaticDim) or sz.dim % max_group == 0
        for sz in normalized
    )
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for ax in range(n):
        if ax == norm and not split_axis_ok:
            continue
        rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
    rows.append(AxisAssignment((P,), P))
    return build_action_set(
        rows,
        layouts=(x,),
        extras=(split_sizes, axis),
        finalize=_split_finalize(x=x, split_sizes=split_sizes, axis=axis),
    )
