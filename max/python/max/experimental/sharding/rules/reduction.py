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

"""Placement rules for reduction ops (``reduce``, ``softmax``, ``sum``, ``mean``)."""

from __future__ import annotations

from typing import Any

from max.experimental.sharding import Partial, ReduceOp, Sharded
from max.experimental.sharding.types import TensorLayout

from ..action import ActionSet, AxisAssignment
from ..cost import P, R, build_action_set

_AVG = Partial(ReduceOp.AVG)


def _non_reduction_axis_rows(
    x: TensorLayout, axis: int
) -> list[AxisAssignment]:
    """``Sharded(d) -> Sharded(d)`` rows over every non-reduction axis."""
    norm = axis % x.rank
    return [
        AxisAssignment((Sharded(d),), Sharded(d))
        for d in range(x.rank)
        if d != norm
    ]


def reduce_rule(x: TensorLayout, axis: int = -1, *extra: Any) -> ActionSet:
    """Non-linear reduction: shard any non-reduction axis.

    Shared by ``prod``, ``argmax``, ``argmin``, ``max``, ``min``;
    ``*extra`` absorbs op-specific trailing args.
    """
    rows = [AxisAssignment((R,), R), *_non_reduction_axis_rows(x, axis)]
    return build_action_set(rows, layouts=(x,), extras=(axis, *extra))


def softmax_rule(value: TensorLayout, axis: int = -1) -> ActionSet:
    """Strategies for ``softmax`` / ``logsoftmax``: shard any non-softmax axis."""
    rows = [AxisAssignment((R,), R), *_non_reduction_axis_rows(value, axis)]
    return build_action_set(rows, layouts=(value,), extras=(axis,))


def linear_reduce_rule(
    x: TensorLayout, axis: int = -1, *extra: Any
) -> ActionSet:
    """Linear reduction: ``S(reduced_axis) -> Partial(SUM)``; ``P -> P``.

    Shared by ``sum`` and ``cumsum``.
    """
    norm = axis % x.rank
    rows = [
        AxisAssignment((R,), R),
        *_non_reduction_axis_rows(x, axis),
        AxisAssignment((Sharded(norm),), P),
        AxisAssignment((P,), P),
    ]
    return build_action_set(rows, layouts=(x,), extras=(axis, *extra))


def mean_rule(x: TensorLayout, axis: int = -1) -> ActionSet:
    """Mean is linear with reduction op AVG."""
    norm = axis % x.rank
    rows = [
        AxisAssignment((R,), R),
        *_non_reduction_axis_rows(x, axis),
        AxisAssignment((Sharded(norm),), _AVG),
        AxisAssignment((_AVG,), _AVG),
    ]
    return build_action_set(rows, layouts=(x,), extras=(axis,))
