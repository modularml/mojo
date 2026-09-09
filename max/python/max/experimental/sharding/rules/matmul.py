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

"""Placement rules for ``matmul`` and ``outer``."""

from __future__ import annotations

from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim, StaticDim

from ..action import ActionSet, AxisAssignment
from ..cost import P, R, build_action_set


def _is_size_one(dim: Dim) -> bool:
    return isinstance(dim, StaticDim) and dim.dim == 1


def _shared_batch(lhs: TensorLayout, rhs: TensorLayout) -> list[int]:
    """Batch axes where BOTH operands are non-broadcast (size > 1).

    Right-aligned: lhs[d] aligns with rhs[d - (lhs.rank - rhs.rank)]
    when ranks differ, treating leading missing dims as size-1.
    """
    n = min(lhs.rank - 2, rhs.rank - 2)
    return [
        d
        for d in range(n)
        if not _is_size_one(lhs.shape[d]) and not _is_size_one(rhs.shape[d])
    ]


def _lhs_batch(lhs: TensorLayout, rhs: TensorLayout) -> list[int]:
    """Batch axes where lhs is non-broadcast but rhs is broadcast / absent."""
    n_shared = min(lhs.rank - 2, rhs.rank - 2)
    out: list[int] = list(range(n_shared, lhs.rank - 2))
    out.extend(
        d
        for d in range(n_shared)
        if not _is_size_one(lhs.shape[d]) and _is_size_one(rhs.shape[d])
    )
    return out


def _rhs_batch(lhs: TensorLayout, rhs: TensorLayout) -> list[int]:
    """Batch axes where rhs is non-broadcast but lhs is broadcast / absent."""
    n_shared = min(lhs.rank - 2, rhs.rank - 2)
    out: list[int] = list(range(n_shared, rhs.rank - 2))
    out.extend(
        d
        for d in range(n_shared)
        if _is_size_one(lhs.shape[d]) and not _is_size_one(rhs.shape[d])
    )
    return out


def _mm_rows(lhs: TensorLayout, rhs: TensorLayout) -> list[AxisAssignment]:
    """Matrix x matrix (and any batched variant).

    Right-aligned axis indices: M and K on lhs, K and N on rhs, M and N
    on output. Output rank equals ``max(lhs.rank, rhs.rank)`` for
    batched cases.
    """
    M_lhs, K_lhs = lhs.rank - 2, lhs.rank - 1
    K_rhs, N_rhs = rhs.rank - 2, rhs.rank - 1
    out_rank = max(lhs.rank, rhs.rank)
    M_out, N_out = out_rank - 2, out_rank - 1

    rows: list[AxisAssignment] = [AxisAssignment((R, R), R)]
    for d in _shared_batch(lhs, rhs):
        rows.append(AxisAssignment((Sharded(d), Sharded(d)), Sharded(d)))
    for d in _lhs_batch(lhs, rhs):
        rows.append(AxisAssignment((Sharded(d), R), Sharded(d)))
    for d in _rhs_batch(lhs, rhs):
        rows.append(AxisAssignment((R, Sharded(d)), Sharded(d)))
    rows.extend(
        [
            AxisAssignment((Sharded(M_lhs), R), Sharded(M_out)),
            AxisAssignment((R, Sharded(N_rhs)), Sharded(N_out)),
            AxisAssignment((Sharded(K_lhs), Sharded(K_rhs)), P),
            AxisAssignment((P, R), P),
            AxisAssignment((R, P), P),
        ]
    )
    return rows


def _vv_rows() -> list[AxisAssignment]:
    """Vector x vector: (K,) @ (K,) -> scalar; both contract on K."""
    return [
        AxisAssignment((R, R), R),
        AxisAssignment((Sharded(0), Sharded(0)), P),
        AxisAssignment((P, R), P),
        AxisAssignment((R, P), P),
    ]


def _vm_rows(rhs: TensorLayout) -> list[AxisAssignment]:
    """Vector x matrix: (K,) @ (..., K, N) -> (..., N)."""
    rows: list[AxisAssignment] = [AxisAssignment((R, R), R)]
    for d in range(rhs.rank - 2):
        rows.append(AxisAssignment((R, Sharded(d)), Sharded(d)))
    rows.append(
        AxisAssignment((R, Sharded(rhs.rank - 1)), Sharded(rhs.rank - 2))
    )
    rows.append(AxisAssignment((Sharded(0), Sharded(rhs.rank - 2)), P))
    rows.extend([AxisAssignment((P, R), P), AxisAssignment((R, P), P)])
    return rows


def _mv_rows(lhs: TensorLayout) -> list[AxisAssignment]:
    """Matrix x vector: (..., M, K) @ (K,) -> (..., M)."""
    rows: list[AxisAssignment] = [AxisAssignment((R, R), R)]
    for d in range(lhs.rank - 2):
        rows.append(AxisAssignment((Sharded(d), R), Sharded(d)))
    rows.append(
        AxisAssignment((Sharded(lhs.rank - 2), R), Sharded(lhs.rank - 2))
    )
    rows.append(AxisAssignment((Sharded(lhs.rank - 1), Sharded(0)), P))
    rows.extend([AxisAssignment((P, R), P), AxisAssignment((R, P), P)])
    return rows


def matmul_rule(lhs: TensorLayout, rhs: TensorLayout) -> ActionSet:
    """Strategies for ``matmul``: vector-vector / vector-matrix / matrix-matrix variants."""
    if lhs.rank == 1 and rhs.rank == 1:
        rows = _vv_rows()
    elif lhs.rank == 1:
        rows = _vm_rows(rhs)
    elif rhs.rank == 1:
        rows = _mv_rows(lhs)
    else:
        rows = _mm_rows(lhs, rhs)
    return build_action_set(rows, layouts=(lhs, rhs))


def outer_rule(lhs: TensorLayout, rhs: TensorLayout) -> ActionSet:
    """Strategies for ``outer``: 1-D x 1-D -> 2-D outer product."""
    rows = [
        AxisAssignment((R, R), R),
        AxisAssignment((Sharded(0), R), Sharded(0)),
        AxisAssignment((R, Sharded(0)), Sharded(1)),
        AxisAssignment((P, R), P),
        AxisAssignment((R, P), P),
    ]
    return build_action_set(rows, layouts=(lhs, rhs))
