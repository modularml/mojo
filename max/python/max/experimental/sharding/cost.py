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

"""SPMD redistribution cost model and :class:`ActionSet` builders."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from max.graph.dim import Dim, StaticDim

from .action import Action, ActionSet, AxisAssignment
from .mesh import DeviceMesh
from .per_shard_dim import is_per_shard_dim
from .placements import (
    Collective,
    Partial,
    Placement,
    ReduceOp,
    Replicated,
    Sharded,
)
from .types import TensorLayout

R: Placement = Replicated()
"""The :class:`Replicated` placement singleton."""

P: Placement = Partial(ReduceOp.SUM)
"""The default :class:`Partial` (``SUM``) singleton."""


# ─── Per-collective ring formulas ────────────────────────────────────


def _ring_allgather(
    message_bytes: float, mesh: DeviceMesh, axis_index: int
) -> float:
    """Ring-allgather cost: ``bytes * (N-1)/N`` on ``axis_index``."""
    n = mesh.mesh_shape[axis_index]
    factor = (n - 1) / n if n > 1 else 0.0
    return message_bytes * factor


def _ring_allreduce(
    message_bytes: float, mesh: DeviceMesh, axis_index: int
) -> float:
    """Ring-allreduce cost: 2x the allgather term (reduce-scatter + allgather)."""
    n = mesh.mesh_shape[axis_index]
    factor = (n - 1) / n if n > 1 else 0.0
    return 2.0 * message_bytes * factor


def _ring_reduce_scatter(
    message_bytes: float, mesh: DeviceMesh, axis_index: int
) -> float:
    """Ring reduce-scatter cost: same volume class as allgather."""
    n = mesh.mesh_shape[axis_index]
    factor = (n - 1) / n if n > 1 else 0.0
    return message_bytes * factor


def transition_cost(
    source: Placement,
    dest: Placement,
    *,
    message_bytes: float,
    mesh: DeviceMesh,
    axis_index: int,
) -> float:
    """Returns the cost of redistributing ``source`` to ``dest`` on ``axis_index``.

    Dispatches on :meth:`Placement.transition_to`, so custom
    :class:`Placement` subclasses participate as long as they return a
    :class:`Collective` member. Anything else is reported as infeasible
    (``+inf``) so solvers reject it.
    """
    if source == dest:
        return 0.0
    name: Collective = source.transition_to(dest)
    if name in (Collective.NOOP, Collective.LOCAL_SLICE):
        return 0.0
    if name in (Collective.ALLGATHER, Collective.ALL_TO_ALL):
        return _ring_allgather(message_bytes, mesh, axis_index)
    if name == Collective.ALLREDUCE:
        return _ring_allreduce(message_bytes, mesh, axis_index)
    if name == Collective.REDUCE_SCATTER:
        return _ring_reduce_scatter(message_bytes, mesh, axis_index)
    return float("inf")


def rank_axis_assignments(
    actuals: tuple[Placement, ...],
    candidates: Iterable[AxisAssignment],
    *,
    mesh: DeviceMesh,
    axis_index: int,
    tensor_bytes: tuple[float, ...] | None = None,
) -> tuple[tuple[AxisAssignment, float], ...]:
    """Sorts arity-matching strategies ascending by cost (stable on ties)."""
    n = len(actuals)
    bpi = tensor_bytes or (1.0,) * n
    if len(bpi) != n:
        raise ValueError(f"tensor_bytes has length {len(bpi)}, expected {n}.")
    scored = [
        (
            axs,
            _action_cost(
                actuals,
                axs.needed_inputs,
                bpi,
                mesh=mesh,
                axis_index=axis_index,
            ),
        )
        for axs in candidates
        if len(axs.needed_inputs) == n
    ]
    return tuple(sorted(scored, key=lambda x: x[1]))


def _action_cost(
    actuals: tuple[Placement, ...],
    needed: tuple[Placement, ...],
    tensor_bytes: tuple[float, ...],
    *,
    mesh: DeviceMesh,
    axis_index: int,
) -> float:
    """Sum of per-input transition costs from ``actuals`` to ``needed`` on one axis."""
    return sum(
        transition_cost(
            a, p, message_bytes=tb, mesh=mesh, axis_index=axis_index
        )
        for a, p, tb in zip(actuals, needed, tensor_bytes, strict=True)
    )


# ─── Feasibility ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FeasibilityContext:
    """The state one feasibility check needs about *one* mesh axis.

    Internal to :mod:`cost`. Feasibility is decided per mesh axis: given
    the actual per-input placements along that axis and the candidate
    :class:`AxisAssignment`, does the candidate produce non-empty shards
    on every tensor input, and does its output dim fit the user-requested
    result shape (if any)? This context carries everything the helpers in
    :func:`action_is_feasible` consult. The per-axis size is derived from
    ``mesh`` and ``mesh_axis_idx`` via :attr:`group_size`.
    """

    layouts: tuple[TensorLayout, ...]
    """Per-tensor input layouts. Indexed positionally; one entry per
    tensor argument the rule received."""
    mesh: DeviceMesh
    """The device mesh; only the per-axis size is read."""
    mesh_axis_idx: int
    """The mesh axis under evaluation. Indexes ``mesh.mesh_shape`` and the
    per-axis tuple of every input's placement."""
    result_shape: Sequence[Any] | None = None
    """Output shape constraint set by reshape-style rules (rules where the
    output shape is a rule argument, not derived). Used to reject
    :class:`Sharded` output rows whose target dim is smaller than the
    cumulative shard factor."""
    input_placements: Sequence[tuple[Placement, ...]] | None = None
    """Current per-input placements (one tuple per tensor input). ``None``
    falls back to reading ``layouts[i].mapping``."""

    @property
    def group_size(self) -> int:
        """The size of the mesh axis under evaluation."""
        return self.mesh.mesh_shape[self.mesh_axis_idx]


def action_is_feasible(
    axs: AxisAssignment,
    actuals: tuple[Placement, ...],
    ctx: _FeasibilityContext,
) -> bool:
    """Rejects strategies whose new :class:`Sharded` axes would create empty shards."""
    in_use = {ax for a in actuals if (ax := a.localized_axis()) is not None}
    for i, p in enumerate(axs.needed_inputs):
        if not _input_passes_empty_shard(i, p, actuals, in_use, ctx):
            return False
    return _output_dim_fits(axs.output, ctx)


def _input_passes_empty_shard(
    i: int,
    p: Placement,
    actuals: tuple[Placement, ...],
    in_use: set[int],
    ctx: _FeasibilityContext,
) -> bool:
    """True if input ``i`` under placement ``p`` would not produce an empty shard."""
    p_axis = p.localized_axis()
    if p_axis is None or i >= len(ctx.layouts):
        return True
    if i < len(actuals) and actuals[i] == p:
        return True
    shape = ctx.layouts[i].shape
    if not 0 <= p_axis < len(shape):
        return False
    dim = shape[p_axis]
    if is_per_shard_dim(dim):
        if any(isinstance(c, StaticDim) and c.dim == 0 for c in dim.per_shard):
            return False
        static_cells = [c for c in dim.per_shard if isinstance(c, StaticDim)]
        if len(static_cells) != len(dim.per_shard):
            return True
        min_cell = min(c.dim for c in static_cells)
        if min_cell == 1 and ctx.group_size > 1:
            return False
        if p_axis in in_use:
            return True
        return min_cell >= ctx.group_size
    if isinstance(dim, StaticDim) and dim.dim == 1:
        return False
    if p_axis in in_use:
        return True
    effective = _cumulative_axis_group_size(i, p_axis, ctx)
    return not (isinstance(dim, StaticDim) and dim.dim < effective)


def _cumulative_axis_group_size(
    i: int, p_axis: int, ctx: _FeasibilityContext
) -> int:
    """Product of mesh-axis sizes already sharding tensor axis ``p_axis``."""
    placements = (
        ctx.input_placements[i]
        if ctx.input_placements is not None
        else ctx.layouts[i].mapping.to_placements()
    )
    return ctx.group_size * math.prod(
        ctx.mesh.mesh_shape[j]
        for j, other in enumerate(placements)
        if j != ctx.mesh_axis_idx and other.localized_axis() == p_axis
    )


def _output_dim_fits(output: Placement, ctx: _FeasibilityContext) -> bool:
    """True if ``output``'s result-shape dim is large enough to shard on this mesh axis."""
    if ctx.result_shape is None:
        return True
    out_axis = output.localized_axis()
    if out_axis is None or not 0 <= out_axis < len(ctx.result_shape):
        return True
    d = ctx.result_shape[out_axis]
    d = d if isinstance(d, Dim) else Dim(d)
    if is_per_shard_dim(d):
        return True
    if not isinstance(d, StaticDim):
        return False
    if d.dim == -1:
        return True
    return d.dim >= ctx.group_size


# ─── Byte counts ──────────────────────────────────────────────────────


def _global_axis_size(layout: TensorLayout, tensor_axis: int) -> int:
    """Best-effort global static extent of tensor axis ``tensor_axis``."""
    dim = layout.shape[tensor_axis]
    mesh = layout.mapping.mesh
    placements = layout.mapping.to_placements()
    factor = 1
    for mesh_axis, p in enumerate(placements):
        if isinstance(p, Sharded) and p.localized_axis() == tensor_axis:
            factor *= mesh.mesh_shape[mesh_axis]
    if is_per_shard_dim(dim):
        cell = dim.per_shard[0]
        if not isinstance(cell, StaticDim):
            return 1
        return cell.dim * factor
    if isinstance(dim, StaticDim):
        return dim.dim * factor
    return 1


def tensor_byte_count(layout: TensorLayout) -> float:
    """Returns the global byte count for ``layout``.

    Symbolic axes fall back to ``1``.
    """
    total = float(layout.dtype.size_in_bytes)
    for tensor_axis in range(len(layout.shape)):
        total *= _global_axis_size(layout, tensor_axis)
    return total


def input_placements_at_axis(
    per_input_placements: Sequence[tuple[Placement, ...]], mesh_axis: int
) -> tuple[Placement, ...]:
    """Per-input placement at ``mesh_axis``, defaulting to ``Replicated``."""
    return tuple(
        p[mesh_axis] if len(p) > mesh_axis else Replicated()
        for p in per_input_placements
    )


def feasible_rows_at_axis(
    menu: ActionSet,
    mesh_axis: int,
    per_input_placements: Sequence[tuple[Placement, ...]],
) -> tuple[AxisAssignment, ...]:
    """Rows of ``menu`` feasible at ``mesh_axis``."""
    actuals = input_placements_at_axis(per_input_placements, mesh_axis)
    ctx = _FeasibilityContext(
        layouts=menu.layouts,
        mesh=menu.mesh,
        mesh_axis_idx=mesh_axis,
        result_shape=menu.result_shape,
        input_placements=per_input_placements,
    )
    return tuple(
        row
        for row in menu.axis_assignments
        if action_is_feasible(row, actuals, ctx)
    )


# ─── Menu builders ────────────────────────────────────────────────────


def build_action_set(
    rows: Iterable[AxisAssignment],
    *,
    layouts: tuple[TensorLayout, ...],
    extras: tuple[Any, ...] = (),
    result_shape: Sequence[Dim] | None = None,
    finalize: Callable[[Action], Action] | None = None,
) -> ActionSet:
    """Wraps rule-emitted rows into a feasibility-filtered :class:`ActionSet`.

    Filters ``rows`` to entries feasible on at least one mesh axis and
    appends the universal ``(R,...,R) -> R`` fallback if absent.

    Args:
        rows: Per-axis decisions the rule wants on the menu. Order is
            preserved as the cost-model tie-breaker.
        layouts: Tensor input layouts, one per tensor positional arg.
        extras: Non-tensor positional args forwarded to the picked
            :class:`Action`.
        result_shape: Output shape constraint for reshape-style rules.
        finalize: Optional post-pick transform of shape
            ``action -> action`` (pre-bound by the rule).
    """
    if not layouts:
        raise ValueError("build_action_set: at least one layout required.")

    mesh = layouts[0].mapping.mesh
    placements_per_input = [l.mapping.to_placements() for l in layouts]
    rows = tuple(rows)
    feasible = tuple(
        r
        for r in rows
        if _feasible_at_any_axis(
            r, layouts, mesh, placements_per_input, result_shape
        )
    )
    fallback = AxisAssignment(needed_inputs=(R,) * len(layouts), output=R)
    if not feasible or feasible[-1] != fallback:
        feasible = (*feasible, fallback)

    return ActionSet(
        axis_assignments=feasible,
        layouts=layouts,
        mesh=mesh,
        extras=extras,
        result_shape=result_shape,
        finalize=finalize,
    )


def force_replicated_action_set(
    *layouts: TensorLayout, extras: tuple[Any, ...] = ()
) -> ActionSet:
    """Single-row ``(R,…,R) -> R`` :class:`ActionSet` for ops that do not expose sharding."""
    if not layouts:
        raise ValueError(
            "force_replicated_action_set: at least one layout required."
        )
    n = len(layouts)
    return ActionSet(
        axis_assignments=(AxisAssignment(needed_inputs=(R,) * n, output=R),),
        layouts=layouts,
        mesh=layouts[0].mapping.mesh,
        extras=extras,
    )


def _feasible_at_any_axis(
    row: AxisAssignment,
    layouts: tuple[TensorLayout, ...],
    mesh: DeviceMesh,
    placements_per_input: list[tuple[Placement, ...]],
    result_shape: Sequence[Dim] | None,
) -> bool:
    """``True`` if ``row`` is feasible on at least one mesh axis."""
    return any(
        action_is_feasible(
            row,
            input_placements_at_axis(placements_per_input, mesh_axis),
            _FeasibilityContext(
                layouts=layouts,
                mesh=mesh,
                mesh_axis_idx=mesh_axis,
                result_shape=result_shape,
                input_placements=placements_per_input,
            ),
        )
        for mesh_axis in range(mesh.ndim)
    )


def pair_transition_cost(
    producer_placements: Sequence[Placement],
    consumer_placements: Sequence[Placement],
    tensor_bytes: float,
    mesh: DeviceMesh,
) -> float:
    """Per-mesh-axis transition cost summed for one producer-to-consumer pair."""
    return sum(
        transition_cost(
            old,
            new,
            message_bytes=tensor_bytes,
            mesh=mesh,
            axis_index=axis,
        )
        for axis, (old, new) in enumerate(
            zip(producer_placements, consumer_placements, strict=False)
        )
    )
