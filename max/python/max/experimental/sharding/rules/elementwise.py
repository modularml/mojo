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

"""Placement rules for unary, binary, and ternary elementwise ops."""

from __future__ import annotations

from typing import Any

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim, StaticDim

from ..action import ActionSet, AxisAssignment
from ..cost import P, R, build_action_set


def _is_size_one(dim: Dim) -> bool:
    """``True`` for a static size-1 dim (a broadcast axis)."""
    return isinstance(dim, StaticDim) and dim.dim == 1


def _aligned_axis(layout: TensorLayout, out_axis: int, out_rank: int) -> int:
    """Input tensor axis that trailing-aligns to ``out_axis``, or ``-1`` if absent."""
    return out_axis - (out_rank - layout.rank)


def _elementwise_rows(
    layouts: tuple[TensorLayout, ...], *, linear: bool
) -> list[AxisAssignment]:
    """Candidate sharding rows for an elementwise op, aligned by trailing axis.

    For each output axis, offers a symmetric row (all inputs carrying the axis
    at full extent sharded together) plus one one-sided row per such input
    (that input sharded alone, the rest Replicated). Inputs that broadcast the
    axis (absent or size 1) stay Replicated. Prepends the ``(R, ...) -> R``
    fallback; ``linear`` ops add a ``(P, ...) -> P`` row.
    """
    n_in = len(layouts)
    out_rank = max(layout.rank for layout in layouts)
    rows: list[AxisAssignment] = [AxisAssignment((R,) * n_in, R)]
    for out_axis in range(out_rank):
        aligned = [
            _aligned_axis(layout, out_axis, out_rank) for layout in layouts
        ]
        shardable = [
            i
            for i, (k, layout) in enumerate(zip(aligned, layouts, strict=True))
            if k >= 0 and not _is_size_one(layout.shape[k])
        ]
        if not shardable:
            continue
        # Symmetric row (all shardable inputs together), then one one-sided
        # row per shardable input (that input alone, the rest Replicated).
        active_sets = [shardable]
        if len(shardable) > 1:
            active_sets.extend([i] for i in shardable)
        for active in active_sets:
            needed = tuple(
                Sharded(aligned[i]) if i in active else R for i in range(n_in)
            )
            rows.append(AxisAssignment(needed, Sharded(out_axis)))
    if linear:
        rows.append(AxisAssignment((P,) * n_in, P))
    return rows


def unary_rule(x: TensorLayout, *extra: Any) -> ActionSet:
    """Strategies for nonlinear unary ops (relu, exp, ...): Replicated or Sharded."""
    rows = [
        AxisAssignment((R,), R),
        *(AxisAssignment((Sharded(d),), Sharded(d)) for d in range(x.rank)),
    ]
    return build_action_set(rows, layouts=(x,), extras=extra)


def linear_unary_rule(x: TensorLayout, *extra: Any) -> ActionSet:
    """Strategies for linear unary ops (negate, cast, ...): adds Partial passthrough."""
    rows = [
        AxisAssignment((R,), R),
        *(AxisAssignment((Sharded(d),), Sharded(d)) for d in range(x.rank)),
        AxisAssignment((P,), P),
    ]
    return build_action_set(rows, layouts=(x,), extras=extra)


def binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> ActionSet:
    """Strategies for elementwise binary ops (mul, div, ...): no Partial passthrough."""
    rows = _elementwise_rows((lhs, rhs), linear=False)
    return build_action_set(rows, layouts=(lhs, rhs))


def linear_binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> ActionSet:
    """Strategies for linear binary ops (add, sub): Partial passthrough when both Partial."""
    rows = _elementwise_rows((lhs, rhs), linear=True)
    return build_action_set(rows, layouts=(lhs, rhs))


def ternary_rule(
    condition: TensorLayout, x: TensorLayout, y: TensorLayout
) -> ActionSet:
    """Strategies for elementwise ternary (``where``): trailing-aligned Sharded."""
    rows = _elementwise_rows((condition, x, y), linear=False)
    return build_action_set(rows, layouts=(condition, x, y))
