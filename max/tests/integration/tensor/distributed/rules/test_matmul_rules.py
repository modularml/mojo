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

"""Pure-metadata tests for matmul placement rules."""

from __future__ import annotations

from max.dtype import DType
from max.experimental.sharding import DeviceMapping, Partial, TensorLayout
from max.experimental.sharding.rules import matmul_rule
from max.graph import Shape

from rules._fixtures import MESH_1D, MESH_2D, M, P, R, S, pick


def _layout(
    mapping: DeviceMapping, shape: tuple[int, ...], dtype: DType = DType.float32
) -> TensorLayout:
    return TensorLayout(dtype, Shape(shape), mapping)


# ═════════════════════════════════════════════════════════════════════════
#  Matmul rule -- [M, K] x [K, N] -> [M, N]
# ═════════════════════════════════════════════════════════════════════════


class TestMatmulRule:
    """Tests for matmul_rule on 1D and 2D meshes.

    For [M=4, K=8] x [K=8, N=6]:
      lhs: M=axis0, K=axis1
      rhs: K=axis0, N=axis1
    """

    # -- Trivial ----------------------------------------------------------

    def test_both_replicated(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (R,)

    # -- Data parallel: S(M) x R -> S(M) ---------------------------------

    def test_data_parallel(self) -> None:
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)

    # -- Column TP: R x S(N) -> S(N) -------------------------------------

    def test_column_tp(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, S(1)), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(1),)

    # -- Row TP: S(K_lhs) x S(K_rhs) -> Partial --------------------------

    def test_row_tp(self) -> None:
        """S(K=1) on lhs x S(K=0) on rhs -> Partial."""
        lhs = _layout(M(MESH_1D, S(1)), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (P,)

    # -- Batch parallel ---------------------------------------------------

    def test_batch_parallel_3d(self) -> None:
        """[B, M, K] x [B, K, N] with S(batch=0) on both."""
        lhs = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (2, 8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)

    def test_batch_sharded_lhs_only(self) -> None:
        """[B, M, K] x [B, K, N] with S(batch=0) on lhs, R on rhs."""
        lhs = _layout(M(MESH_1D, S(0)), (2, 4, 8))
        rhs = _layout(M(MESH_1D, R), (2, 8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)

    # -- Bilinear: P x R -> P, R x P -> P --------------------------------

    def test_partial_lhs_replicated_rhs(self) -> None:
        lhs = _layout(M(MESH_1D, P), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (P,)

    def test_replicated_lhs_partial_rhs(self) -> None:
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, P), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (P,)

    # -- Error cases ------------------------------------------------------

    def test_partial_partial_preserves_partial_on_rhs(self) -> None:
        """(P, P): cost model picks (R, P, P) -- byte-weighted cheapest plan."""
        lhs = _layout(M(MESH_1D, P), (4, 8))
        rhs = _layout(M(MESH_1D, P), (8, 6))
        args, (out,) = pick(matmul_rule, lhs, rhs)
        assert args[0].to_placements() == (R,)
        assert args[1].to_placements() == (P,)
        assert out.to_placements() == (P,)

    def test_partial_sharded_picks_row_tp(self) -> None:
        """(P, S(K=0)): cost model picks row-TP via P->S(K=1) reduce_scatter."""
        lhs = _layout(M(MESH_1D, P), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8, 6))
        args, (out,) = pick(matmul_rule, lhs, rhs)
        assert args[0].to_placements() == (S(1),)
        assert args[1].to_placements() == (S(0),)
        assert out.to_placements() == (P,)

    def test_sharded_partial_picks_dp_with_allreduce(self) -> None:
        """S(M=0) x P: picker keeps lhs sharded, allreduces P->R on rhs (cheaper than allgather lhs)."""
        lhs = _layout(M(MESH_1D, S(0)), (4, 8))
        rhs = _layout(M(MESH_1D, P), (8, 6))
        args, (out,) = pick(matmul_rule, lhs, rhs)
        assert args[0].to_placements() == (S(0),)
        assert args[1].to_placements() == (R,)
        assert out.to_placements() == (S(0),)

    def test_s_k_lhs_replicated_rhs_picks_row_tp(self) -> None:
        """S(K) x R: cost model picks row-TP for free (R->S(K_rhs=0) is local slice)."""
        lhs = _layout(M(MESH_1D, S(1)), (4, 8))
        rhs = _layout(M(MESH_1D, R), (8, 6))
        args, (out,) = pick(matmul_rule, lhs, rhs)
        assert args[0].to_placements() == (S(1),)
        assert args[1].to_placements() == (S(0),)
        assert out.to_placements() == (P,)

    def test_replicated_lhs_s_k_rhs_picks_row_tp(self) -> None:
        """R x S(K): cost model picks row-TP for free (R->S(K_lhs=1) is local slice)."""
        lhs = _layout(M(MESH_1D, R), (4, 8))
        rhs = _layout(M(MESH_1D, S(0)), (8, 6))
        args, (out,) = pick(matmul_rule, lhs, rhs)
        assert args[0].to_placements() == (S(1),)
        assert args[1].to_placements() == (S(0),)
        assert out.to_placements() == (P,)

    # -- 2D mesh: combined DP + TP ----------------------------------------

    def test_2d_mesh_dp_plus_column_tp(self) -> None:
        """dp=S(M=0), tp=S(N=1): data parallel x column TP."""
        lhs = _layout(M(MESH_2D, S(0), R), (4, 8))
        rhs = _layout(M(MESH_2D, R, S(1)), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(0), S(1))

    def test_2d_mesh_dp_plus_row_tp(self) -> None:
        """dp=S(batch=0), tp=S(K_lhs=2) x S(K_rhs=1): batch parallel + row TP -> Partial.

        For [B=2, M=4, K=8] x [B=2, K=8, N=6]:
          K_lhs=axis2, K_rhs=axis1
        """
        lhs = _layout(M(MESH_2D, S(0), S(2)), (2, 4, 8))
        rhs = _layout(M(MESH_2D, S(0), S(1)), (2, 8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(0), Partial())

    # -- Vector matmul ----------------------------------------------------

    def test_vec_mat(self) -> None:
        """[K] x [K, N] -> [N]: result is rank 1, so N is at output axis 0."""
        lhs = _layout(M(MESH_1D, R), (8,))
        rhs = _layout(M(MESH_1D, S(1)), (8, 6))
        _, (out,) = pick(matmul_rule, lhs, rhs)
        assert out.to_placements() == (S(0),)
