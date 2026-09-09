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
"""Tests for :mod:`max.experimental.sharding.picker`."""

from __future__ import annotations

import warnings

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    Action,
    ActionSet,
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    GreedyReshard,
    NoReshard,
    Partial,
    PartialsOnly,
    Replicated,
    Sharded,
    ShardingError,
    TensorLayout,
    build_action_set,
)
from max.experimental.sharding._diagnostics import report_reshard
from max.experimental.sharding.picker import enumerate_feasible_actions
from max.experimental.sharding.placements import Placement


def mesh_1d(n: int) -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), ("tp",))


def layout(
    mesh: DeviceMesh,
    shape: tuple[int, ...],
    placement: tuple[Placement, ...],
) -> TensorLayout:
    return TensorLayout(DType.float32, shape, DeviceMapping(mesh, placement))


def output_placements(action: Action) -> tuple[Placement, ...]:
    return action.outputs[0].placements


def input_placements(action: Action) -> tuple[Placement, ...]:
    return action.inputs[0].placements


# ─── Shared scenario: input is Partial, two non-passthrough resolutions ───


PartialInputMenu = tuple[TensorLayout, ActionSet]


@pytest.fixture
def partial_input_menu() -> PartialInputMenu:
    mesh = mesh_1d(4)
    lay = layout(mesh, (1024,), (Partial(),))
    rows = [
        AxisAssignment((Replicated(),), Replicated()),
        AxisAssignment((Sharded(0),), Sharded(0)),
    ]
    return lay, build_action_set(rows, layouts=(lay,))


class TestSolverDiscrimination:
    def test_greedy_default_resolves_partial_to_replicated(
        self, partial_input_menu: PartialInputMenu
    ) -> None:
        lay, menu = partial_input_menu
        picked = GreedyReshard()(menu, (lay,))
        assert input_placements(picked) == (Replicated(),)

    def test_greedy_opt_in_picks_reduce_scatter_over_allreduce(
        self, partial_input_menu: PartialInputMenu
    ) -> None:
        lay, menu = partial_input_menu
        picked = GreedyReshard(allow_partial_to_sharded=True)(menu, (lay,))
        assert input_placements(picked) == (Sharded(0),)

    def test_no_cost_refuses_partial_to_replicated_reshard(
        self, partial_input_menu: PartialInputMenu
    ) -> None:
        lay, menu = partial_input_menu
        with pytest.raises(ShardingError):
            NoReshard()(menu, (lay,))

    def test_partial_resolve_takes_allreduce_only_path(
        self, partial_input_menu: PartialInputMenu
    ) -> None:
        lay, menu = partial_input_menu
        picked = PartialsOnly()(menu, (lay,))
        assert input_placements(picked) == (Replicated(),)


class TestPassthroughPreference:
    def test_no_cost_prefers_zero_reshard_even_if_listed_last(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Sharded(0),))
        rows = [
            AxisAssignment((Replicated(),), Replicated()),
            AxisAssignment((Sharded(0),), Sharded(0)),
        ]
        menu = build_action_set(rows, layouts=(lay,))
        picked = NoReshard()(menu, (lay,))
        assert input_placements(picked) == (Sharded(0),)

    def test_partial_resolve_prefers_passthrough(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Sharded(0),))
        rows = [
            AxisAssignment((Replicated(),), Replicated()),
            AxisAssignment((Sharded(0),), Sharded(0)),
        ]
        menu = build_action_set(rows, layouts=(lay,))
        picked = PartialsOnly()(menu, (lay,))
        assert input_placements(picked) == (Sharded(0),)


class TestPartialResolveRefusal:
    def test_raises_when_only_non_partial_reshards_available(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Sharded(0),))
        rows = [AxisAssignment((Replicated(),), Replicated())]
        menu = build_action_set(rows, layouts=(lay,))
        with pytest.raises(ShardingError):
            PartialsOnly()(menu, (lay,))


class TestNoReshardRefusal:
    def test_raises_when_only_shard_reshard_available(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Sharded(0),))
        rows = [AxisAssignment((Replicated(),), Replicated())]
        menu = build_action_set(rows, layouts=(lay,))
        with pytest.raises(ShardingError):
            NoReshard()(menu, (lay,))

    def test_error_names_the_input_layouts(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (1024,), (Partial(),))
        rows = [AxisAssignment((Replicated(),), Replicated())]
        menu = build_action_set(rows, layouts=(lay,))
        with pytest.raises(ShardingError, match="no zero-reshard action"):
            NoReshard()(menu, (lay,))


class TestEnumerateFeasibleActions:
    def test_filters_infeasible_combinations(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (1, 8), (Replicated(),))
        rows = [
            AxisAssignment((Sharded(0),), Sharded(0)),
            AxisAssignment((Sharded(1),), Sharded(1)),
        ]
        menu = build_action_set(rows, layouts=(lay,))
        actions = enumerate_feasible_actions(menu, mesh)
        produced = {output_placements(a) for a in actions}
        assert (Sharded(0),) not in produced
        assert any(p == (Sharded(1),) or p == (Replicated(),) for p in produced)

    def test_2d_mesh_yields_cartesian_product(self) -> None:
        mesh = DeviceMesh(
            tuple(CPU() for _ in range(2 * 2)), (2, 2), ("dp", "tp")
        )
        lay = layout(mesh, (16, 16), (Replicated(), Replicated()))
        rows = [
            AxisAssignment((Sharded(0),), Sharded(0)),
            AxisAssignment((Sharded(1),), Sharded(1)),
        ]
        menu = build_action_set(rows, layouts=(lay,))
        actions = enumerate_feasible_actions(menu, mesh)
        produced = {output_placements(a) for a in actions}
        assert (Sharded(0), Sharded(0)) in produced
        assert (Sharded(0), Sharded(1)) in produced
        assert (Sharded(1), Sharded(0)) in produced
        assert (Sharded(1), Sharded(1)) in produced


class TestReportReshard:
    def _picked_with_reshard(self) -> tuple[TensorLayout, ActionSet]:
        mesh = mesh_1d(4)
        lay = layout(mesh, (1024,), (Partial(),))
        rows = [AxisAssignment((Replicated(),), Replicated())]
        menu = build_action_set(rows, layouts=(lay,))
        return lay, menu

    def test_silent_emits_nothing_and_does_not_raise(self) -> None:
        lay, menu = self._picked_with_reshard()
        solver = GreedyReshard(on_reshard="silent")
        picked = solver(menu, (lay,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report_reshard(solver, "op_name", (lay,), menu, picked)
        assert caught == []

    def test_warn_emits_user_warning(self) -> None:
        lay, menu = self._picked_with_reshard()
        solver = GreedyReshard(on_reshard="warn")
        picked = solver(menu, (lay,))
        with pytest.warns(UserWarning):
            report_reshard(solver, "op_name", (lay,), menu, picked)

    def test_raise_raises_sharding_error(self) -> None:
        lay, menu = self._picked_with_reshard()
        solver = GreedyReshard(on_reshard="raise")
        picked = solver(menu, (lay,))
        with pytest.raises(ShardingError):
            report_reshard(solver, "op_name", (lay,), menu, picked)

    def test_warn_is_silent_when_already_matching(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Replicated(),))
        menu = build_action_set([], layouts=(lay,))
        solver = GreedyReshard(on_reshard="warn")
        picked = solver(menu, (lay,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report_reshard(solver, "op_name", (lay,), menu, picked)
        assert caught == []

    def test_no_reshard_solver_refuses_instead_of_reporting(self) -> None:
        lay, menu = self._picked_with_reshard()
        with pytest.raises(ShardingError):
            NoReshard()(menu, (lay,))

    def test_no_reshard_solver_is_silent_on_passthrough(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Replicated(),))
        menu = build_action_set([], layouts=(lay,))
        solver = NoReshard()
        picked = solver(menu, (lay,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report_reshard(solver, "op_name", (lay,), menu, picked)
        assert caught == []
