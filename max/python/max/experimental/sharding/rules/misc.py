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

"""Placement rules for miscellaneous ops (resize, irfft, qmatmul, scatter_nd, ...)."""

from __future__ import annotations

import dataclasses
from typing import Any

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import DimLike

from ..action import ActionSet, AxisAssignment
from ..cost import P, R, build_action_set, force_replicated_action_set
from .elementwise import linear_unary_rule


def _block_axes_rows(
    x: TensorLayout, blocked: set[int]
) -> list[AxisAssignment]:
    """Linear-block helper: shard any non-blocked axis; Partial pass-through."""
    rows: list[AxisAssignment] = [AxisAssignment((R,), R)]
    for ax in range(x.rank):
        if ax in blocked:
            continue
        rows.append(AxisAssignment((Sharded(ax),), Sharded(ax)))
    rows.append(AxisAssignment((P,), P))
    return rows


def band_part_rule(x: TensorLayout, *extra: Any, **kwargs: Any) -> ActionSet:
    """Strategies for ``band_part``: sharding only outside the last-two matrix axes."""
    return build_action_set(
        _block_axes_rows(x, {max(0, x.rank - 2), x.rank - 1}),
        layouts=(x,),
        extras=extra,
    )


def fold_rule(input: TensorLayout, *extra: Any, **kwargs: Any) -> ActionSet:
    """Strategies for ``fold``: sharding only outside axes 1 and 2."""
    return build_action_set(
        _block_axes_rows(input, {1, 2}), layouts=(input,), extras=extra
    )


def as_interleaved_complex_rule(
    x: TensorLayout, *extra: Any, **kwargs: Any
) -> ActionSet:
    """Strategies for ``as_interleaved_complex``: sharding only outside the last axis."""
    return build_action_set(
        _block_axes_rows(x, {x.rank - 1}), layouts=(x,), extras=extra
    )


def irfft_rule(
    input_tensor: TensorLayout, *extra: Any, **kwargs: Any
) -> ActionSet:
    """Strategies for ``irfft``: sharding only outside the last axis."""
    return build_action_set(
        _block_axes_rows(input_tensor, {input_tensor.rank - 1}),
        layouts=(input_tensor,),
        extras=extra,
    )


def _resize_action_set(
    input: TensorLayout, extras: tuple[object, ...]
) -> ActionSet:
    return build_action_set(
        _block_axes_rows(input, set(range(1, input.rank))),
        layouts=(input,),
        extras=extras,
    )


def resize_rule(input: TensorLayout, *extra: Any, **kwargs: Any) -> ActionSet:
    """Strategies for ``resize``: sharding only on the batch axis (0)."""
    return _resize_action_set(input, extras=extra)


def resize_linear_rule(
    input: TensorLayout, *extra: Any, **kwargs: Any
) -> ActionSet:
    """Strategies for ``resize_linear``: same batch-only family as ``resize``."""
    return _resize_action_set(input, extras=extra)


def resize_nearest_rule(
    input: TensorLayout, *extra: Any, **kwargs: Any
) -> ActionSet:
    """Strategies for ``resize_nearest``: same batch-only family as ``resize``."""
    return _resize_action_set(input, extras=extra)


def resize_bicubic_rule(
    input: TensorLayout, *extra: Any, **kwargs: Any
) -> ActionSet:
    """Strategies for ``resize_bicubic``: same batch-only family as ``resize``."""
    return _resize_action_set(input, extras=extra)


def dequantize_rule(encoding: Any, quantized: TensorLayout) -> ActionSet:
    """Linear unary on ``quantized``; ``encoding`` is non-tensor metadata."""
    return dataclasses.replace(linear_unary_rule(quantized), extras=(encoding,))


def qmatmul_rule(
    encoding: Any, config: Any, lhs: TensorLayout, *rhs: TensorLayout
) -> ActionSet:
    """Quantized matmul: every tensor Replicated."""
    return force_replicated_action_set(lhs, *rhs, extras=(encoding, config))


def masked_scatter_rule(
    input: TensorLayout,
    mask: TensorLayout,
    updates: TensorLayout,
    out_dim: DimLike,
) -> ActionSet:
    """Forces every input to Replicated (mask uses absolute positions)."""
    return force_replicated_action_set(input, mask, updates, extras=(out_dim,))


def scatter_nd_rule(
    input: TensorLayout, updates: TensorLayout, indices: TensorLayout
) -> ActionSet:
    """N-D scatter: everything Replicated."""
    return force_replicated_action_set(input, updates, indices)


def scatter_nd_add_rule(
    input: TensorLayout, updates: TensorLayout, indices: TensorLayout
) -> ActionSet:
    """``scatter_nd_add`` shares its strategy with :func:`scatter_nd_rule`."""
    return force_replicated_action_set(input, updates, indices)


def reject_distributed_rule(x: TensorLayout, op_name: str = "") -> ActionSet:
    """Auto-gathers to fully :class:`Replicated`."""
    return force_replicated_action_set(x)
