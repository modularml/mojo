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

"""Shared fixtures for pure-metadata placement rule tests.

These tests never create Tensors or graph ops directly — they call
rule functions on :class:`TensorLayout` inputs and exercise the
production :class:`GreedyReshard` to check what the picker selects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from max.driver import CPU
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.sharding.action import Action, ActionSet

# ── Convenience aliases ──────────────────────────────────────────────

R = Replicated()
P = Partial()


def S(d: int) -> Sharded:
    return Sharded(d)


# ── Standard test meshes ────────────────────────────────────────────

MESH_1D = DeviceMesh(
    devices=tuple(CPU() for _ in range(4)),
    mesh_shape=(4,),
    axis_names=("tp",),
)

MESH_2D = DeviceMesh(
    devices=tuple(CPU() for _ in range(4)),
    mesh_shape=(2, 2),
    axis_names=("dp", "tp"),
)

MESH_2 = DeviceMesh(
    devices=(CPU(), CPU()),
    mesh_shape=(2,),
    axis_names=("tp",),
)

# ── Mapping builders ────────────────────────────────────────────────


def M(
    mesh: DeviceMesh, *placements: Replicated | Sharded | Partial
) -> PlacementMapping:
    """Shorthand: M(MESH_1D, S(0)) -> PlacementMapping(MESH_1D, (S(0),))."""
    return PlacementMapping(mesh, tuple(placements))


# ── Solver-driven picker for rule tests ─────────────────────────────


def pick(rule: Callable[..., ActionSet], *args: Any, **kwargs: Any) -> Action:
    """Picks the cheapest :class:`Action` for ``rule(*args, **kwargs)``.

    Test-only helper that bypasses the source-graph trace: invokes ``rule``
    directly with the supplied :class:`TensorLayout`\\ s, enumerates the
    feasible actions over the layouts' shared mesh, and returns the cheapest
    by :func:`pair_transition_cost`. Equivalent to one step of the
    :class:`~max.experimental.sharding.picker.GreedyReshard`
    but without going through a graph.
    """
    from max.experimental.sharding.picker import (
        _finalize,
        cheapest_action,
        enumerate_feasible_actions,
    )

    def _flatten_layouts(value: Any) -> Any:
        if isinstance(value, TensorLayout):
            yield value
        elif isinstance(value, (list, tuple)):
            for v in value:
                yield from _flatten_layouts(v)

    in_layouts: list[TensorLayout] = []
    for a in args:
        in_layouts.extend(_flatten_layouts(a))
    for v in kwargs.values():
        in_layouts.extend(_flatten_layouts(v))
    menu = rule(*args, **kwargs)
    mesh = menu.mesh
    actions = enumerate_feasible_actions(menu, mesh)
    if not actions:
        raise ValueError(f"pick: rule {rule.__name__!r} returned no actions.")
    return _finalize(menu, cheapest_action(actions, in_layouts, mesh))
