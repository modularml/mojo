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
"""Tests for :mod:`max.experimental.sharding.cost`."""

from __future__ import annotations

import math

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    Action,
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    P,
    Partial,
    R,
    Replicated,
    Sharded,
    TensorLayout,
    build_action_set,
    force_replicated_action_set,
)
from max.experimental.sharding.cost import (
    pair_transition_cost,
    tensor_byte_count,
    transition_cost,
)
from max.experimental.sharding.placements import Placement


def mesh_1d(n: int) -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), ("tp",))


def layout(
    mesh: DeviceMesh,
    shape: tuple[int, ...],
    placement: tuple[Placement, ...],
) -> TensorLayout:
    return TensorLayout(DType.float32, shape, DeviceMapping(mesh, placement))


class TestSingletons:
    def test_r_and_p_are_canonical_instances(self) -> None:
        assert R == Replicated()
        assert P == Partial()


class TestTensorByteCount:
    def test_product_of_shape_times_dtype(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (4, 8, 3), (Replicated(),))
        assert tensor_byte_count(lay) == 4 * 8 * 3 * 4.0

    def test_scalar_is_one_element(self) -> None:
        mesh = mesh_1d(2)
        assert tensor_byte_count(layout(mesh, (), (Replicated(),))) == 4.0


class TestBuildActionSet:
    def test_appends_missing_fallback(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (4,), (Replicated(),))
        s = build_action_set(
            [AxisAssignment((Sharded(0),), Sharded(0))], layouts=(lay,)
        )
        assert s.axis_assignments[-1] == AxisAssignment(
            (Replicated(),), Replicated()
        )

    def test_does_not_duplicate_existing_fallback(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (4,), (Replicated(),))
        s = build_action_set(
            [AxisAssignment((Replicated(),), Replicated())], layouts=(lay,)
        )
        assert s.axis_assignments == (
            AxisAssignment((Replicated(),), Replicated()),
        )

    def test_filters_sharded_row_on_size_one_axis(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (1, 8), (Replicated(),))
        s = build_action_set(
            [AxisAssignment((Sharded(0),), Sharded(0))], layouts=(lay,)
        )
        assert (
            AxisAssignment((Sharded(0),), Sharded(0)) not in s.axis_assignments
        )

    def test_threads_layouts_mesh_extras_finalize(self) -> None:
        mesh = mesh_1d(2)
        lay = layout(mesh, (4,), (Replicated(),))
        extras = (uniform_sentinel := object(),)

        def finalize(action: Action) -> Action:
            return action

        s = build_action_set(
            [], layouts=(lay,), extras=extras, finalize=finalize
        )
        assert s.layouts == (lay,)
        assert s.mesh is mesh
        assert s.extras == (uniform_sentinel,)
        assert s.finalize is finalize


class TestForceReplicatedActionSet:
    def test_emits_a_single_all_replicated_row(self) -> None:
        mesh = mesh_1d(2)
        a = layout(mesh, (4,), (Replicated(),))
        b = layout(mesh, (4,), (Replicated(),))
        s = force_replicated_action_set(a, b)
        assert s.axis_assignments == (
            AxisAssignment((Replicated(), Replicated()), Replicated()),
        )

    def test_requires_at_least_one_layout(self) -> None:
        with pytest.raises(ValueError):
            force_replicated_action_set()


class TestPairTransitionCost:
    def test_matching_placements_are_free(self) -> None:
        mesh = mesh_1d(4)
        cost = pair_transition_cost(
            producer_placements=[Sharded(0)],
            consumer_placements=[Sharded(0)],
            tensor_bytes=1024.0,
            mesh=mesh,
        )
        assert cost == 0.0

    def test_replicated_to_sharded_uses_local_slice(self) -> None:
        mesh = mesh_1d(4)
        assert (
            pair_transition_cost(
                [Replicated()],
                [Sharded(0)],
                tensor_bytes=1024.0,
                mesh=mesh,
            )
            == 0.0
        )

    def test_partial_to_replicated_is_twice_partial_to_sharded(self) -> None:
        mesh = mesh_1d(4)
        ar = pair_transition_cost(
            [Partial()],
            [Replicated()],
            tensor_bytes=1024.0,
            mesh=mesh,
        )
        rs = pair_transition_cost(
            [Partial()],
            [Sharded(0)],
            tensor_bytes=1024.0,
            mesh=mesh,
        )
        assert math.isclose(ar, 2.0 * rs)

    def test_infeasible_transition_is_infinite(self) -> None:
        mesh = mesh_1d(4)
        cost = pair_transition_cost(
            [Replicated()],
            [Partial()],
            tensor_bytes=1024.0,
            mesh=mesh,
        )
        assert math.isinf(cost)


def _ring_factor(n: int) -> float:
    return 0.0 if n <= 1 else (n - 1) / n


class TestTransitionCost:
    """Ring-collective arithmetic, exercised via :func:`transition_cost`."""

    def test_self_transition_is_free(self) -> None:
        mesh = mesh_1d(4)
        assert (
            transition_cost(
                Sharded(0),
                Sharded(0),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
            == 0.0
        )

    def test_replicated_to_sharded_is_local_slice(self) -> None:
        mesh = mesh_1d(4)
        assert (
            transition_cost(
                Replicated(),
                Sharded(0),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
            == 0.0
        )

    def test_sharded_to_replicated_matches_ring_allgather(self) -> None:
        for n in (2, 4, 8):
            mesh = mesh_1d(n)
            c = transition_cost(
                Sharded(0),
                Replicated(),
                message_bytes=1024.0,
                mesh=mesh,
                axis_index=0,
            )
            assert math.isclose(c, 1024.0 * _ring_factor(n))

    def test_partial_to_replicated_is_twice_partial_to_sharded(self) -> None:
        mesh = mesh_1d(8)
        ar = transition_cost(
            Partial(),
            Replicated(),
            message_bytes=4096.0,
            mesh=mesh,
            axis_index=0,
        )
        rs = transition_cost(
            Partial(),
            Sharded(0),
            message_bytes=4096.0,
            mesh=mesh,
            axis_index=0,
        )
        assert math.isclose(ar, 2.0 * rs)

    def test_single_device_rings_are_free(self) -> None:
        mesh = mesh_1d(1)
        for src, dst in (
            (Sharded(0), Replicated()),
            (Partial(), Replicated()),
            (Partial(), Sharded(0)),
        ):
            assert (
                transition_cost(
                    src,
                    dst,
                    message_bytes=1024.0,
                    mesh=mesh,
                    axis_index=0,
                )
                == 0.0
            )

    def test_replicated_to_partial_is_infeasible(self) -> None:
        mesh = mesh_1d(4)
        assert math.isinf(
            transition_cost(
                Replicated(),
                Partial(),
                message_bytes=1.0,
                mesh=mesh,
                axis_index=0,
            )
        )

    def test_2d_mesh_uses_requested_axis_size(self) -> None:
        mesh = DeviceMesh(
            tuple(CPU() for _ in range(2 * 8)), (2, 8), ("dp", "tp")
        )
        on_dp = transition_cost(
            Sharded(0),
            Replicated(),
            message_bytes=1024.0,
            mesh=mesh,
            axis_index=0,
        )
        on_tp = transition_cost(
            Sharded(0),
            Replicated(),
            message_bytes=1024.0,
            mesh=mesh,
            axis_index=1,
        )
        assert math.isclose(on_dp, 1024.0 * _ring_factor(2))
        assert math.isclose(on_tp, 1024.0 * _ring_factor(8))
        assert on_tp > on_dp
