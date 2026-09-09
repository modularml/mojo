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

"""Reshard diagnostic-message formatting for ``on_reshard='warn'`` / ``'raise'``."""

from __future__ import annotations

import math
import sys
import warnings
from types import FrameType
from typing import TYPE_CHECKING, Any

from .action import Action, ActionSet
from .cost import (
    feasible_rows_at_axis,
    input_placements_at_axis,
    rank_axis_assignments,
    tensor_byte_count,
)
from .mappings import DeviceMapping
from .mesh import DeviceMesh
from .placements import Placement, ShardingError

if TYPE_CHECKING:
    from .picker import ReshardBehavior, Solver

_FRAMEWORK_PREFIXES = (
    "max.experimental.functional.",
    "max.experimental.tensor",
    "max.experimental.realization_context",
    "max.experimental.nn.",
    "max.experimental.sharding.",
    "contextlib",
)

_MODE_LABELS: dict[ReshardBehavior, str] = {
    "silent": "silent insertion",
    "warn": "this warning",
    "raise": "raise ShardingError",
}


def build_reshard_message(
    op_name: str,
    layout_args: tuple[Any, ...],
    strategy_inputs: tuple[Any, ...],
    menu: ActionSet | None,
    on_reshard: ReshardBehavior,
) -> str | None:
    """Returns the reshard diagnostic message, or ``None`` if no reshard.

    ``None`` when every layout already matches its picked input mapping.
    The string is the multi-line message used by the ``warn`` / ``raise``
    paths; ``warnings.warn`` dedupes downstream by ``(message, category,
    filename, lineno)``.
    """
    blocks: list[str] = []
    suggestions: list[str] = []
    for i, (lay, sugg) in enumerate(
        zip(layout_args, strategy_inputs, strict=True)
    ):
        if not isinstance(sugg, DeviceMapping) or not hasattr(lay, "mapping"):
            continue
        if lay.mapping == sugg:
            continue
        blocks.append(_format_arg_block(i, lay, sugg))
        suggestions.append(f"      arg{i} = transfer_to(arg{i}, {sugg!r})")
    if not blocks:
        return None

    parts = [
        f"  op:        {op_name}",
        f"  call site: {_caller_site()}",
        _format_meshes(layout_args, strategy_inputs, menu),
        "  rule proposes redistribution:",
        "\n".join(blocks),
    ]
    if menu is not None:
        parts.append(_format_alternatives(menu))
    parts.append(
        "  to make explicit, insert before this op:\n" + "\n".join(suggestions)
    )
    parts.append(_mode_footer(on_reshard))

    return "\n" + "\n".join(parts)


def report_reshard(
    solver: Solver,
    op_name: str,
    layout_args: tuple[Any, ...],
    menu: ActionSet,
    action: Action,
) -> None:
    """Honors ``solver.on_reshard`` after a picker has picked.

    Reads ``getattr(solver, "on_reshard", "silent")``: solvers without the
    attribute (such as :class:`NoReshard` and :class:`PartialsOnly`) are
    treated as ``"silent"`` and this is a no-op for them. Only
    :class:`GreedyReshard` carries a meaningful policy today.
    """
    on_reshard: ReshardBehavior = getattr(solver, "on_reshard", "silent")
    if on_reshard == "silent":
        return
    message = build_reshard_message(
        op_name, layout_args, action.inputs, menu, on_reshard
    )
    if message is None:
        return
    if on_reshard == "warn":
        warnings.warn(message, UserWarning, stacklevel=4)
        return
    raise ShardingError(message)


def _caller_site() -> str:
    """Returns ``"<file>:<line> in <func>"`` for the first non-framework frame."""
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        mod = frame.f_globals.get("__name__", "") or ""
        if not any(mod.startswith(p) for p in _FRAMEWORK_PREFIXES):
            return (
                f"{frame.f_code.co_filename}:{frame.f_lineno} in "
                f"{frame.f_code.co_name}()"
            )
        frame = frame.f_back
    return "<unknown call site>"


def _format_meshes(
    layout_args: tuple[Any, ...],
    suggested: tuple[Any, ...],
    menu: ActionSet | None,
) -> str:
    """Lists every distinct mesh involved in this op's strategy decision."""
    seen: dict[int, DeviceMesh] = {}
    sources: list[Any] = [*layout_args, *suggested]
    if menu is not None:
        sources.append(menu)
    for s in sources:
        m = getattr(s, "mesh", None) or getattr(
            getattr(s, "mapping", None), "mesh", None
        )
        if isinstance(m, DeviceMesh):
            seen.setdefault(id(m), m)
    return (
        "  meshes:    " + ", ".join(repr(m) for m in seen.values())
        if seen
        else ""
    )


def _format_arg_block(i: int, lay: Any, sugg: DeviceMapping) -> str:
    """One mismatched argument: full mappings + per-axis breakdown."""
    shape = list(lay.shape) if hasattr(lay, "shape") else "?"
    head = (
        f"    arg[{i}] shape={shape}\n"
        f"      actual:    {lay.mapping!r}\n"
        f"      suggested: {sugg!r}"
    )
    if lay.mapping.mesh != sugg.mesh:
        return head + "\n      collective: scatter (cross-mesh)"
    diffs = [
        f"        mesh axis {name!r}: {a!r} → {n!r}  [{_classify_transition(a, n)}]"
        for name, a, n in zip(
            sugg.mesh.axis_names,
            lay.mapping.to_placements(),
            sugg.to_placements(),
            strict=True,
        )
        if a != n
    ]
    return head + (
        "\n      per-axis transitions:\n" + "\n".join(diffs) if diffs else ""
    )


def _classify_transition(actual: Placement, needed: Placement) -> str:
    """Names the collective inserted to redistribute ``actual -> needed``.

    Dispatches through :meth:`Placement.transition_to` and reports the
    resulting :class:`Collective` by its ``.value`` -- the same value the
    cost model scored. ``transition_to`` returns
    :attr:`Collective.NOOP` when ``actual == needed``, so no equality
    special-case is needed here.
    """
    return actual.transition_to(needed).value


def _format_alternatives(menu: ActionSet, top_n: int = 3) -> str:
    """Top-N axis strategies the cost model considered, per mesh axis."""
    lines = ["  alternatives the cost model considered:"]
    mesh = menu.mesh
    bytes_per_input = tuple(tensor_byte_count(l) for l in menu.layouts)
    per_input = [l.mapping.to_placements() for l in menu.layouts]
    for ax in range(mesh.ndim):
        gs = mesh.mesh_shape[ax]
        actuals = input_placements_at_axis(per_input, ax)
        valid = feasible_rows_at_axis(menu, ax, per_input)
        ranked = [
            (r, c)
            for r, c in rank_axis_assignments(
                actuals,
                valid,
                mesh=mesh,
                axis_index=ax,
                tensor_bytes=bytes_per_input,
            )
            if c != math.inf
        ]
        lines.append(
            f"    mesh axis {mesh.axis_names[ax]!r} (group_size={gs}):"
        )
        for rank, (row, cost) in enumerate(ranked[:top_n]):
            tag = "picked" if rank == 0 else f"#{rank + 1}"
            ins = ", ".join(repr(p) for p in row.needed_inputs)
            lines.append(
                f"      [{tag} cost={cost:g}] inputs=({ins}) -> output={row.output!r}"
            )
        if len(ranked) > top_n:
            lines.append(f"      ... ({len(ranked) - top_n} more)")
    return "\n".join(lines)


def _mode_footer(cur: ReshardBehavior) -> str:
    """One-line footer naming all three observers and how to switch."""
    parts = [
        f"'{m}' [{l}]" + (" ← current" if m == cur else "")
        for m, l in _MODE_LABELS.items()
    ]
    return (
        "  on_reshard: "
        + ", ".join(parts)
        + ". Set on_reshard on the solver, e.g. "
        + "`mode(GreedyReshard(on_reshard='silent'|'warn'|'raise'))`."
    )
