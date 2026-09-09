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
"""Tests for DistributedTensorType and DistributedBufferType.

These types live in ``max.experimental.sharding`` and describe how
a tensor's global shape maps to per-device local types for graph compilation.
"""

from __future__ import annotations

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.sharding import (
    DeviceMesh,
    DistributedBufferType,
    DistributedTensorType,
    Replicated,
    Sharded,
)
from max.graph import BufferType, SymbolicDim, TensorType

# ── Inline mesh helpers (no conftest dependency) ──────────────────────


def mesh_1d(n: int, name: str = "tp") -> DeviceMesh:
    return DeviceMesh(tuple(CPU() for _ in range(n)), (n,), (name,))


def mesh_2d(rows: int, cols: int) -> DeviceMesh:
    return DeviceMesh(
        tuple(CPU() for _ in range(rows * cols)), (rows, cols), ("dp", "tp")
    )


class TestDistributedTensorType:
    def test_basic_construction(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        assert dt.dtype == DType.float32
        assert list(dt.shape) == [8, 16]
        assert dt.rank == 2

    def test_local_types(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        local = dt.local_types
        assert len(local) == 4
        for lt in local:
            assert isinstance(lt, TensorType)
            assert lt.dtype == DType.float32
            assert list(lt.shape) == [2, 16]

    def test_replicated_local_types(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Replicated()])
        for lt in dt.local_types:
            assert list(lt.shape) == [8, 16]

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DistributedTensorType(
            DType.float32, [8, 16], mesh, [Sharded(0), Sharded(1)]
        )
        local = dt.local_types
        assert len(local) == 8
        for lt in local:
            assert list(lt.shape) == [4, 4]

    def test_symbolic_dim_produces_per_rank_distinct_names(self) -> None:
        """Each rank gets a per-rank distinct symbolic name so uneven
        runtime values can bind independently per rank."""
        mesh = mesh_1d(4)
        dt = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Sharded(0)]
        )
        local = dt.local_types
        for rank, lt in enumerate(local):
            assert isinstance(lt.shape[0], SymbolicDim)
            assert lt.shape[0].name == f"batch_tp_{rank}"

    def test_symbolic_dim_non_sharded_unchanged(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Replicated()]
        )
        local = dt.local_types
        assert isinstance(local[0].shape[0], SymbolicDim)
        assert local[0].shape[0].name == "batch"

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            DistributedTensorType(
                DType.float32, [8, 16], mesh, [Sharded(0), Replicated()]
            )

    def test_out_of_range_shard_axis_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="out of range"):
            DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(5)])

    def test_uneven_static_dim_yields_per_rank_distinct_sizes(self) -> None:
        """Uneven static dim is supported: each rank gets its own size
        from ``_shard_sizes_along_axis`` (e.g. 7 on 4 ranks -> [2,2,2,1])."""
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [7, 16], mesh, [Sharded(0)])
        local = dt.local_types
        assert [int(lt.shape[0]) for lt in local] == [2, 2, 2, 1]
        for lt in local:
            assert int(lt.shape[1]) == 16

    def test_repr(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedTensorType(DType.float32, [8, 16], mesh, [Sharded(0)])
        r = repr(dt)
        assert "DistributedTensorType" in r
        assert "float32" in r
        assert "Sharded" in r

    def test_local_types_count_matches_device_count(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedTensorType(DType.float32, [8, 4], mesh, [Sharded(0)])
        local = dt.local_types
        assert len(local) == 2
        for lt in local:
            assert lt.device is not None


class TestDistributedBufferType:
    def test_basic_construction(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        assert dt.dtype == DType.float32
        assert dt.rank == 2

    def test_local_types_are_buffer_types(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        local = dt.local_types
        assert len(local) == 2
        for lt in local:
            assert isinstance(lt, BufferType)
            assert list(lt.shape) == [4, 4]

    def test_replicated_local_types(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedBufferType(DType.float32, [8, 16], mesh, [Replicated()])
        local = dt.local_types
        assert len(local) == 4
        for lt in local:
            assert isinstance(lt, BufferType)
            assert list(lt.shape) == [8, 16]

    def test_2d_mesh(self) -> None:
        mesh = mesh_2d(2, 4)
        dt = DistributedBufferType(
            DType.float32, [8, 16], mesh, [Sharded(0), Sharded(1)]
        )
        local = dt.local_types
        assert len(local) == 8
        for lt in local:
            assert isinstance(lt, BufferType)
            assert list(lt.shape) == [4, 4]

    def test_wrong_placement_count_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="one placement per mesh axis"):
            DistributedBufferType(
                DType.float32, [8, 16], mesh, [Sharded(0), Replicated()]
            )

    def test_out_of_range_shard_axis_raises(self) -> None:
        mesh = mesh_1d(4)
        with pytest.raises(ValueError, match="out of range"):
            DistributedBufferType(DType.float32, [8, 16], mesh, [Sharded(5)])

    def test_uneven_static_dim_yields_per_rank_distinct_sizes(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedBufferType(DType.float32, [7, 16], mesh, [Sharded(0)])
        local = dt.local_types
        assert [int(lt.shape[0]) for lt in local] == [2, 2, 2, 1]

    def test_symbolic_dim_produces_per_rank_distinct_names(self) -> None:
        mesh = mesh_1d(4)
        dt = DistributedBufferType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Sharded(0)]
        )
        local = dt.local_types
        for rank, lt in enumerate(local):
            assert isinstance(lt.shape[0], SymbolicDim)
            assert lt.shape[0].name == f"batch_tp_{rank}"

    def test_rank_property(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4, 3], mesh, [Sharded(0)])
        assert dt.rank == 3

    def test_dtype_preserved(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.bfloat16, [8, 4], mesh, [Sharded(0)])
        assert dt.dtype == DType.bfloat16

    def test_repr(self) -> None:
        mesh = mesh_1d(2)
        dt = DistributedBufferType(DType.float32, [8, 4], mesh, [Sharded(0)])
        assert "DistributedBufferType" in repr(dt)


# ─── Cross-tensor / cross-construction consistency ────────────────────


class TestSymbolConsistency:
    """Two :class:`DistributedTensorType` with identical global shape
    and placements on the same mesh must produce identical per-rank
    symbolic names — this is what lets the engine identify two
    compiled inputs as sharing the same per-rank dim."""

    @staticmethod
    def _name(dim: object) -> str:
        assert isinstance(dim, SymbolicDim)
        return dim.name

    def test_two_tensors_same_global_dim_share_per_rank_names(
        self,
    ) -> None:
        mesh = mesh_1d(4)
        a = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 16], mesh, [Sharded(0)]
        )
        b = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 32], mesh, [Sharded(0)]
        )
        a_local = a.local_types
        b_local = b.local_types
        for rank in range(mesh.num_devices):
            assert self._name(a_local[rank].shape[0]) == self._name(
                b_local[rank].shape[0]
            )
            assert self._name(a_local[rank].shape[0]) == f"batch_tp_{rank}"

    def test_buffer_and_tensor_share_per_rank_names(self) -> None:
        """A :class:`DistributedBufferType` and
        :class:`DistributedTensorType` over the same global symbolic
        dim and mesh produce the same per-rank names."""
        mesh = mesh_1d(4)
        t = DistributedTensorType(
            DType.float32, [SymbolicDim("seq"), 8], mesh, [Sharded(0)]
        )
        b = DistributedBufferType(
            DType.float32, [SymbolicDim("seq"), 8], mesh, [Sharded(0)]
        )
        for rank in range(mesh.num_devices):
            assert self._name(t.local_types[rank].shape[0]) == self._name(
                b.local_types[rank].shape[0]
            )

    def test_distinct_symbolic_dims_get_distinct_per_rank_names(
        self,
    ) -> None:
        """Different global names should NOT collide at any rank
        (e.g. ``batch_tp_0`` and ``seq_tp_0`` must remain distinct)."""
        mesh = mesh_1d(2)
        a = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 4], mesh, [Sharded(0)]
        )
        b = DistributedTensorType(
            DType.float32, [SymbolicDim("seq"), 4], mesh, [Sharded(0)]
        )
        for rank in range(mesh.num_devices):
            assert self._name(a.local_types[rank].shape[0]) != self._name(
                b.local_types[rank].shape[0]
            )


# ─── Symbol-count scaling sanity at larger mesh sizes ─────────────────


class TestSymbolScalingSanity:
    """Sanity check that per-rank distinct naming doesn't collapse or
    explode beyond ``num_devices`` symbols at larger mesh sizes.

    These are not perf benchmarks — just structural correctness for
    N >= 8.  Each rank produces exactly one fresh symbolic name and
    they're all pairwise distinct.
    """

    @staticmethod
    def _name(dim: object) -> str:
        assert isinstance(dim, SymbolicDim)
        return dim.name

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_n_ranks_yield_n_distinct_symbol_names(self, n: int) -> None:
        mesh = mesh_1d(n)
        dt = DistributedTensorType(
            DType.float32, [SymbolicDim("batch"), 4], mesh, [Sharded(0)]
        )
        local = dt.local_types
        assert len(local) == n
        names = {self._name(lt.shape[0]) for lt in local}
        # Each rank gets one unique name.
        assert len(names) == n
        # Every name follows the per-rank coord template.
        for rank in range(n):
            assert f"batch_tp_{rank}" in names

    def test_2d_mesh_per_rank_names_factor_axes(self) -> None:
        """On a 2D ``(dp=2, tp=4)`` mesh sharding axis 0 on dp, the
        per-rank name uses the dp axis name and dp coord (NOT the
        flat rank) — ranks (0,0)/(0,1)/(0,2)/(0,3) all share
        ``batch_dp_0``; ranks (1,*) all share ``batch_dp_1``."""
        mesh = mesh_2d(2, 4)
        dt = DistributedTensorType(
            DType.float32,
            [SymbolicDim("batch"), 4],
            mesh,
            [Sharded(0), Replicated()],
        )
        local = dt.local_types
        assert len(local) == 8
        # First 4 ranks live on dp=0; last 4 on dp=1.
        for rank in range(4):
            assert self._name(local[rank].shape[0]) == "batch_dp_0"
        for rank in range(4, 8):
            assert self._name(local[rank].shape[0]) == "batch_dp_1"
