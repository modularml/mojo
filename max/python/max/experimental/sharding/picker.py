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

"""Per-op :class:`Action` selection from an :class:`ActionSet`."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from max.experimental.sharding.action import Action, ActionSet
from max.experimental.sharding.cost import (
    feasible_rows_at_axis,
    pair_transition_cost,
    tensor_byte_count,
)
from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.mesh import DeviceMesh
from max.experimental.sharding.placements import (
    Partial,
    Replicated,
    Sharded,
    ShardingError,
)
from max.experimental.sharding.types import TensorLayout

__all__ = [
    "GreedyReshard",
    "NoReshard",
    "PartialsOnly",
    "ReshardBehavior",
    "Solver",
    "action_input_for_slot",
    "cheapest_action",
    "enumerate_feasible_actions",
]


#: A per-op action picker. Receives the feasible :class:`ActionSet` and
#: the actual input :class:`TensorLayout`\ s; returns the picked
#: :class:`Action`. Plug your own to override the default.
Solver = Callable[[ActionSet, Sequence[TensorLayout]], Action]


#: ``"silent"`` accepts any picked action. ``"warn"`` emits a
#: :class:`UserWarning` when the picked action requires a reshard on any
#: input. ``"raise"`` raises
#: :class:`~max.experimental.sharding.mode.ShardingError` instead.
ReshardBehavior = Literal["silent", "warn", "raise"]


def enumerate_feasible_actions(
    menu: ActionSet, mesh: DeviceMesh
) -> list[Action]:
    """Cartesian product of per-axis admissible rows from ``menu``."""
    placements = [l.mapping.to_placements() for l in menu.layouts]
    combos = list(
        itertools.product(
            *(
                feasible_rows_at_axis(menu, mesh_axis, placements)
                for mesh_axis in range(mesh.ndim)
            )
        )
    )
    return [
        Action(
            inputs=(
                *tuple(
                    PlacementMapping(
                        mesh, tuple(row.needed_inputs[i] for row in combo)
                    )
                    for i in range(len(menu.layouts))
                ),
                *menu.extras,
            ),
            outputs=(
                PlacementMapping(mesh, tuple(row.output for row in combo)),
            ),
        )
        for combo in combos
    ]


def _finalize(menu: ActionSet, action: Action) -> Action:
    """Applies the rule's post-pick ``finalize`` to the chosen action, if any.

    Pickers select among the raw (un-finalized) actions from
    :func:`enumerate_feasible_actions` so the cost model and passthrough
    checks see plain input mappings, then call this on the single winner.
    """
    if menu.finalize is None:
        return action
    return menu.finalize(action)


def action_input_for_slot(action: Action, slot_idx: int) -> Any:
    """Flat-index lookup into an action's per-slot expected placement.

    Returns whatever the action stored at that slot (typically
    :class:`PlacementMapping`, :class:`PerShard`, a bare scalar, or
    ``None``). Callers are expected to ``isinstance``-check before use.
    """
    flat: list[Any] = []
    for entry in action.inputs:
        if isinstance(entry, (list, tuple)):
            flat.extend(entry)
        else:
            flat.append(entry)
    return flat[slot_idx] if slot_idx < len(flat) else None


def cheapest_action(
    actions: Sequence[Action],
    in_layouts: Sequence[TensorLayout],
    mesh: DeviceMesh,
) -> Action:
    """Returns the action whose summed input-side reshard cost is lowest."""
    best = actions[0]
    best_cost = float("inf")
    for action in actions:
        total = 0.0
        for slot, in_layout in enumerate(in_layouts):
            consumer = action_input_for_slot(action, slot)
            if not isinstance(consumer, PlacementMapping):
                continue
            tensor_bytes = tensor_byte_count(in_layout)
            total += pair_transition_cost(
                in_layout.mapping.to_placements(),
                consumer.placements,
                tensor_bytes,
                mesh,
            )
        if total < best_cost:
            best_cost = total
            best = action
    return best


@dataclass
class GreedyReshard:
    """Default per-op picker: enumerate → cheapest.

    The shipped default. Plug your own callable matching the
    :data:`Solver` protocol to override.
    """

    on_reshard: ReshardBehavior = "silent"
    """Diagnostic policy when the picked action requires a reshard on any
    input. ``"silent"`` (default), ``"warn"``, or ``"raise"``. Inspected by
    ``_local_dispatch`` after the picker picks."""
    allow_partial_to_sharded: bool = False
    """When ``True``, the picker may resolve a ``Partial`` input by
    ``reduce_scatter`` to ``Sharded(d)`` when that is the locally cheapest
    collective. On Megatron-style transformers this tends to land in
    sequence-parallel + TP: same byte volume as pure TP, but a sharded
    residual stream, two extra collectives per block, and per-rank
    symbolic-dim drift (the cost the SP activation-memory win pays for,
    which only matters for long-sequence training). When ``False`` (the
    default), the picker may not redistribute a ``Partial`` input to
    ``Sharded``: it either keeps the ``Partial`` (linear passthrough) or
    resolves it to ``Replicated`` via allreduce, on exactly the mesh axes
    that are ``Partial`` — other axes (for example a ``dp`` batch shard)
    are left untouched. A nonlinear consumer such as ``rms_norm`` has no
    ``Partial`` row, so its input lands on ``Replicated``: the textbook
    Megatron pure-TP layout and the right default for inference (fewer
    collectives per block, a replicated residual stream, no ``rebind``
    needed in the model). Set ``True`` to opt into sequence-parallel
    discovery."""

    def __call__(
        self,
        menu: ActionSet,
        in_layouts: Sequence[TensorLayout],
    ) -> Action:
        """Picks the cheapest feasible action."""
        mesh = menu.mesh
        actions = enumerate_feasible_actions(menu, mesh)
        if not actions:
            raise RuntimeError(
                "GreedyReshard: rule returned no feasible actions."
            )
        if not self.allow_partial_to_sharded:
            actions = _reject_partial_to_sharded(actions, in_layouts)
            if not actions:
                raise RuntimeError(
                    "GreedyReshard(allow_partial_to_sharded=False): every "
                    "candidate action requires a Partial -> Sharded "
                    "(reduce_scatter) transition. Pass "
                    "allow_partial_to_sharded=True to permit "
                    "sequence-parallel resharding here."
                )
        return _finalize(menu, cheapest_action(actions, in_layouts, mesh))


def _reject_partial_to_sharded(
    actions: Sequence[Action],
    in_layouts: Sequence[TensorLayout],
) -> list[Action]:
    """Drops actions that redistribute a ``Partial`` input to ``Sharded``.

    For each tensor input slot, rejects the action if any mesh axis
    carries a :class:`Partial` placement that the action would
    redistribute to :class:`Sharded` (a ``reduce_scatter``). Actions
    that keep the ``Partial`` (linear passthrough) or resolve it to
    :class:`Replicated` (allreduce) are kept, as are all other slots and
    axes — in particular a ``dp`` batch shard on another mesh axis is
    left untouched. This suppresses the sequence-parallel layout while
    leaving legitimate ``Partial`` passthrough chains and the pure-TP
    allreduce intact.
    """
    kept: list[Action] = []
    for action in actions:
        ok = True
        for slot, in_layout in enumerate(in_layouts):
            consumer = action_input_for_slot(action, slot)
            if not isinstance(consumer, PlacementMapping):
                continue
            for src, dst in zip(
                in_layout.mapping.to_placements(),
                consumer.placements,
                strict=True,
            ):
                if isinstance(src, Partial) and isinstance(dst, Sharded):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            kept.append(action)
    return kept


# ─── Cost-free pickers ────────────────────────────────────────────────


def _zero_reshard(action: Action, in_layouts: Sequence[TensorLayout]) -> bool:
    """``True`` if every tensor input slot already matches the action."""
    for slot, layout in enumerate(in_layouts):
        consumer = action_input_for_slot(action, slot)
        if not isinstance(consumer, PlacementMapping):
            continue
        if consumer != layout.mapping:
            return False
    return True


def _only_partial_to_replicated(
    action: Action, in_layouts: Sequence[TensorLayout]
) -> bool:
    """``True`` if every mismatch on inputs is a per-axis ``Partial → Replicated``.

    Each tensor input is compared placement-by-placement against the
    action's expected consumer mapping. An axis is allowed to differ
    only when the input carries a :class:`Partial` and the consumer
    expects a :class:`Replicated`; any other transition makes the
    action ineligible.
    """
    for slot, layout in enumerate(in_layouts):
        consumer = action_input_for_slot(action, slot)
        if not isinstance(consumer, PlacementMapping):
            continue
        producer = layout.mapping
        if producer == consumer:
            continue
        prod_ps = producer.to_placements()
        cons_ps = consumer.to_placements()
        if len(prod_ps) != len(cons_ps):
            return False
        for p_in, p_out in zip(prod_ps, cons_ps, strict=False):
            if p_in == p_out:
                continue
            if isinstance(p_in, Partial) and isinstance(p_out, Replicated):
                continue
            return False
    return True


class NoReshard:
    """Passthrough-only picker: returns the first zero-reshard action.

    Refuses any reshard, including a ``Partial → Replicated`` allreduce. If
    no input-matching action exists, raises
    :class:`~max.experimental.sharding.placements.ShardingError` instead of
    silently inserting a collective. Never computes a transition cost.
    """

    def __call__(
        self,
        menu: ActionSet,
        in_layouts: Sequence[TensorLayout],
    ) -> Action:
        """Picks the first zero-reshard action; raises if none exists."""
        actions = enumerate_feasible_actions(menu, menu.mesh)
        if not actions:
            raise RuntimeError("NoReshard: rule returned no feasible actions.")
        for action in actions:
            if _zero_reshard(action, in_layouts):
                return _finalize(menu, action)
        raise ShardingError(
            "NoReshard: no zero-reshard action for input layouts "
            f"{[layout.mapping for layout in in_layouts]}; an input would need "
            "resharding. Use PartialsOnly or GreedyReshard to allow it."
        )


class PartialsOnly:
    """Cost-model-free picker that only resolves ``Partial → Replicated``.

    Picks a feasible action under these rules, applied in order:

    1. The first action whose input mappings already match the layouts
       exactly (no reshard).
    2. The first action whose only per-axis input differences are
       ``Partial → Replicated`` transitions (allreduce-only).
    3. Otherwise: raises
       :class:`~max.experimental.sharding.mode.ShardingError`.

    Never computes a transition cost; never permits S↔R, S(d)↔S(d'),
    or R→S reshards on inputs. Useful when you've already pinned the
    placement of every tensor and just want the picker to allreduce
    Partials when an op can't consume them directly. No ``on_reshard``
    diagnostic — this picker raises directly when it can't satisfy the
    P→R-only contract.
    """

    def __call__(
        self,
        menu: ActionSet,
        in_layouts: Sequence[TensorLayout],
    ) -> Action:
        """Picks the strictest feasible action: passthrough → P→R only → error."""
        actions = enumerate_feasible_actions(menu, menu.mesh)
        if not actions:
            raise RuntimeError(
                "PartialsOnly: rule returned no feasible actions."
            )
        for action in actions:
            if _zero_reshard(action, in_layouts):
                return _finalize(menu, action)
        for action in actions:
            if _only_partial_to_replicated(action, in_layouts):
                return _finalize(menu, action)
        raise ShardingError(
            "PartialsOnly: no feasible action that only resolves "
            "Partial → Replicated. Pin input placements upstream or switch "
            "to a more permissive picker."
        )
