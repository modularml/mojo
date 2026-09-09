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

"""Pure-metadata tests for elementwise placement rules."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.sharding import DeviceMapping, TensorLayout
from max.experimental.sharding.rules import (
    binary_rule,
    ternary_rule,
    unary_rule,
)
from max.graph import Shape

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S, pick


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, Shape(shape), mapping)


# ═════════════════════════════════════════════════════════════════════════
#  Unary passthrough
# ═════════════════════════════════════════════════════════════════════════


class TestUnaryPassthrough:
    def test_replicated(self) -> None:
        layout = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(unary_rule, layout)
        assert out.to_placements() == (R,)

    def test_sharded(self) -> None:
        layout = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(unary_rule, layout)
        assert out.to_placements() == (S(0),)

    def test_sharded_axis1(self) -> None:
        layout = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(unary_rule, layout)
        assert out.to_placements() == (S(1),)

    def test_partial(self) -> None:
        """P input: cost model resolves to Sharded (reduce_scatter, 1x) over R (allreduce, 2x)."""
        layout = _layout(M(MESH_1D, P), (4, 8))
        _, (out,) = pick(unary_rule, layout)
        assert out.to_placements() == (S(0),)

    def test_2d_mesh(self) -> None:
        layout = _layout(M(MESH_2D, S(0), R), (4, 8))
        _, (out,) = pick(unary_rule, layout)
        assert out.to_placements() == (S(0), R)

    def test_2d_mesh_both_sharded(self) -> None:
        layout = _layout(M(MESH_2D, S(0), S(1)), (4, 8))
        _, (out,) = pick(unary_rule, layout)
        assert out.to_placements() == (S(0), S(1))


# ═════════════════════════════════════════════════════════════════════════
#  Binary elementwise
# ═════════════════════════════════════════════════════════════════════════


class TestBinaryElementwise:
    def test_both_replicated(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (R,)

    def test_both_sharded_same(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_sharded_plus_replicated(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_replicated_plus_sharded(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(1),)

    def test_incompatible_sharded_aligns_to_first(self) -> None:
        """Mismatched shards: cost model aligns rhs to lhs's S(0) via cheapest plan."""
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_broadcast_rhs_lower_rank(self) -> None:
        """(4, 8) S(1) + (8,): rhs's only axis trailing-aligns to output axis 1."""
        # No rank padding in Python: the rule sees genuinely unequal-rank
        # inputs and trailing-aligns rhs axis 0 to output axis 1.
        lhs = _layout(M(MESH_1D, S(1)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8,))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(1),)

    def test_broadcast_lhs_lower_rank(self) -> None:
        """(8,) + (4, 8) S(1): lhs's only axis trailing-aligns to output axis 1."""
        lhs = _layout(M(MESH_1D, R), (8,))
        rhs = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(1),)

    def test_2d_mesh(self) -> None:
        lhs = _layout(M(MESH_2D, S(0), R), (4, 8))
        rhs = _layout(M(MESH_2D, R, S(1)), (4, 8))
        _, (out,) = pick(binary_rule, lhs, rhs)
        assert out.to_placements() == (S(0), S(1))


# ═════════════════════════════════════════════════════════════════════════
#  Ternary elementwise
# ═════════════════════════════════════════════════════════════════════════


class TestTernaryElementwise:
    def test_all_replicated(self) -> None:
        a = _layout(M(MESH_1D, R), (4, 8))
        b = _layout(M(MESH_1D, R), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(ternary_rule, a, b, c)
        assert out.to_placements() == (R,)

    def test_all_sharded_same(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        c = _layout(M(MESH_1D, S(0)), (4, 8))
        _, (out,) = pick(ternary_rule, a, b, c)
        assert out.to_placements() == (S(0),)

    def test_cond_sharded_values_replicated(self) -> None:
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, R), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(ternary_rule, a, b, c)
        assert out.to_placements() == (S(0),)

    def test_values_sharded_cond_replicated(self) -> None:
        a = _layout(M(MESH_1D, R), (4, 8))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        c = _layout(M(MESH_1D, S(1)), (4, 8))
        _, (out,) = pick(ternary_rule, a, b, c)
        assert out.to_placements() == (S(1),)

    def test_one_value_sharded(self) -> None:
        a = _layout(M(MESH_1D, R), (4, 8))
        b = _layout(M(MESH_1D, S(0)), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(ternary_rule, a, b, c)
        assert out.to_placements() == (S(0),)

    def test_incompatible_aligns_to_first(self) -> None:
        """Mismatched shards: cost model aligns to cheapest valid plan (S(0))."""
        a = _layout(M(MESH_1D, S(0)), (4, 8))
        b = _layout(M(MESH_1D, S(1)), (4, 8))
        c = _layout(M(MESH_1D, R), (4, 8))
        _, (out,) = pick(ternary_rule, a, b, c)
        assert out.to_placements() == (S(0),)
