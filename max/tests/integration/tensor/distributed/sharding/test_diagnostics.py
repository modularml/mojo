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
"""Tests for :mod:`max.experimental.sharding._diagnostics`."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    ActionSet,
    AxisAssignment,
    DeviceMapping,
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    TensorLayout,
    build_action_set,
)
from max.experimental.sharding._diagnostics import build_reshard_message
from max.experimental.sharding.placements import Placement


def mesh_1d(n: int) -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), ("tp",))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(
        tuple(CPU() for _ in range(rows * cols)),
        (rows, cols),
        ("dp", "tp"),
    )


def layout(
    mesh: DeviceMesh,
    shape: tuple[int, ...],
    placement: tuple[Placement, ...],
) -> TensorLayout:
    return TensorLayout(DType.float32, shape, DeviceMapping(mesh, placement))


def partial_input_menu() -> tuple[TensorLayout, ActionSet]:
    mesh = mesh_1d(4)
    lay = layout(mesh, (1024,), (Partial(),))
    rows = [
        AxisAssignment((Replicated(),), Replicated()),
        AxisAssignment((Sharded(0),), Sharded(0)),
    ]
    return lay, build_action_set(rows, layouts=(lay,))


class TestNoMismatch:
    def test_returns_none_when_layout_already_matches(self) -> None:
        mesh = mesh_1d(4)
        lay = layout(mesh, (16,), (Replicated(),))
        menu = build_action_set([], layouts=(lay,))
        suggested = DeviceMapping(mesh, (Replicated(),))
        assert (
            build_reshard_message(
                "op",
                layout_args=(lay,),
                strategy_inputs=(suggested,),
                menu=menu,
                on_reshard="warn",
            )
            is None
        )


class TestMessageContents:
    def test_mentions_op_and_both_endpoints(self) -> None:
        lay, menu = partial_input_menu()
        suggested = DeviceMapping(lay.mesh, (Replicated(),))
        result = build_reshard_message(
            "custom_op",
            layout_args=(lay,),
            strategy_inputs=(suggested,),
            menu=menu,
            on_reshard="raise",
        )
        assert result is not None
        message = result
        assert "custom_op" in message
        assert "Partial" in message
        assert "Replicated" in message

    def test_lists_cost_ranked_alternatives_from_menu(self) -> None:
        lay, menu = partial_input_menu()
        suggested = DeviceMapping(lay.mesh, (Replicated(),))
        result = build_reshard_message(
            "op",
            layout_args=(lay,),
            strategy_inputs=(suggested,),
            menu=menu,
            on_reshard="warn",
        )
        assert result is not None
        message = result
        assert "Sharded" in message

    def test_multi_axis_mesh_labels_both_axes(self) -> None:
        mesh = mesh_2d(2, 2)
        lay = layout(mesh, (16, 16), (Replicated(), Replicated()))
        menu = build_action_set([], layouts=(lay,))
        suggested = DeviceMapping(mesh, (Sharded(0), Sharded(1)))
        result = build_reshard_message(
            "op",
            layout_args=(lay,),
            strategy_inputs=(suggested,),
            menu=menu,
            on_reshard="warn",
        )
        assert result is not None
        message = result
        assert "dp" in message
        assert "tp" in message
