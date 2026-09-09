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

"""Placement rules for buffer-mutation ops."""

from __future__ import annotations

from max.experimental.sharding import PlacementMapping, Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.ops.slice_tensor import SliceIndices

from ..action import Action, ActionSet, AxisAssignment
from ..cost import P, R, build_action_set


def _output_mirrors_first_input(action: Action) -> Action:
    """Forces the output mapping to mirror ``inputs[0]`` (the destination)."""
    dest = action.inputs[0]
    assert isinstance(dest, PlacementMapping)
    return Action(inputs=action.inputs, outputs=(dest,))


def _shared_axis_rows(
    destination: TensorLayout, source: TensorLayout
) -> list[AxisAssignment]:
    return [
        AxisAssignment((R, R), R),
        *(
            AxisAssignment((Sharded(d), Sharded(d)), Sharded(d))
            for d in range(destination.rank)
        ),
        AxisAssignment((P, P), P),
    ]


def buffer_store_rule(
    destination: TensorLayout, source: TensorLayout
) -> ActionSet:
    """Strategies for ``buffer_store``: destination and source share placement."""
    return build_action_set(
        _shared_axis_rows(destination, source),
        layouts=(destination, source),
        finalize=_output_mirrors_first_input,
    )


def buffer_store_slice_rule(
    destination: TensorLayout,
    source: TensorLayout,
    indices: SliceIndices,
) -> ActionSet:
    """Strategies for ``buffer_store_slice``: same shape table as ``buffer_store``."""
    return build_action_set(
        _shared_axis_rows(destination, source),
        layouts=(destination, source),
        extras=(indices,),
        finalize=_output_mirrors_first_input,
    )
