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
"""Shared test logic for matmul ops.

DO NOT run this file directly — it contains base classes that are
subclassed by functional/test_matmul_simulated.py and
functional/test_matmul_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
    MESH_2:  DeviceMesh   — 2 devices, shape (2,), axis_names=("tp",)
    partial_fn: Callable   — make_partial (CPU) or make_partial (GPU)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from max.dtype import DType
from max.experimental.functional import (
    full,
    layer_norm,
    matmul,
    ones,
    relu,
    transfer_to,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    ReduceOp,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor

_F32 = DType.float32

# ── Row Tensor Parallelism ───────────────────────────────────────────


class _RowTP:
    """Row tensor parallelism: S(K) x S(K_rhs) -> Partial."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_produces_partial(self) -> None:
        """MESH_1D: S(1) x S(0) -> Partial (contracting dim match)."""
        lhs = transfer_to(
            full([2, 8], 1.0, dtype=_F32, device=self.MESH_1D.devices[0]),
            PlacementMapping(self.MESH_1D, (Sharded(1),)),
        )
        rhs = transfer_to(
            full([8, 4], 1.0, dtype=_F32, device=self.MESH_1D.devices[0]),
            PlacementMapping(self.MESH_1D, (Sharded(0),)),
        )
        out = matmul(lhs, rhs)
        assert out.placements == (Partial(ReduceOp.SUM),)
        # Row-TP: S(1) x S(0) -> Partial. Gather produces correct matmul.
        # ones(2,8) @ ones(8,4) = 8 * ones(2,4)
        np.testing.assert_allclose(
            out.to_numpy(), np.full((2, 4), 8.0), rtol=1e-4
        )

    def test_2d_mesh(self) -> None:
        """2D mesh with S(1)xS(0) -> Partial on tp axis, values correct."""
        lhs = transfer_to(
            full([2, 8], 1.0, dtype=_F32, device=self.MESH_2D.devices[0]),
            PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1))),
        )
        rhs = transfer_to(
            full([8, 4], 1.0, dtype=_F32, device=self.MESH_2D.devices[0]),
            PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0))),
        )
        out = matmul(lhs, rhs)
        assert any(isinstance(p, Partial) for p in out.placements)
        # ones(2,8) @ ones(8,4) = 8 * ones(2,4)
        np.testing.assert_allclose(
            out.to_numpy(), np.full((2, 4), 8.0), rtol=1e-4
        )


# ── Column Tensor Parallelism ────────────────────────────────────────


class _ColTP:
    """Column tensor parallelism: R x S(N) -> S(N)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_produces_sharded(self) -> None:
        """MESH_2: R x S(1) -> S(1) (column parallel), values correct."""
        m2_r = PlacementMapping(self.MESH_2, (Replicated(),))
        m2_s1 = PlacementMapping(self.MESH_2, (Sharded(1),))
        lhs = ones([3, 5], dtype=_F32, device=m2_r)
        rhs = ones([5, 6], dtype=_F32, device=m2_s1)
        result = matmul(lhs, rhs)
        assert result.placements == (Sharded(1),)
        assert list(result.shape) == [3, 6]
        # ones[3,5] @ ones[5,6] = 5*ones[3,6]
        np.testing.assert_allclose(
            result.to_numpy(), np.full((3, 6), 5.0), rtol=1e-4
        )

    def test_col_tp_3d(self) -> None:
        """MESH_2: batched R x S(2) -> S(2), values correct."""
        lhs = ones(
            [2, 3, 5],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        rhs = ones(
            [2, 5, 6],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(2),)),
        )
        out = matmul(lhs, rhs)
        assert out.placements == (Sharded(2),)
        assert list(out.shape) == [2, 3, 6]
        # ones[2,3,5] @ ones[2,5,6] = 5*ones[2,3,6]
        np.testing.assert_allclose(
            out.to_numpy(), np.full((2, 3, 6), 5.0), rtol=1e-4
        )


# ── Data Parallel ────────────────────────────────────────────────────


class _DataParallel:
    """Data parallelism: S(M) x R -> S(M)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_dp_chain(self) -> None:
        """MESH_1D: S(0) x R -> S(0), two matmuls chained."""
        x_np = np.ones((8, 4), dtype=np.float32)
        w1_np = np.full((4, 4), 0.5, dtype=np.float32)
        w2_np = np.full((4, 4), 0.25, dtype=np.float32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        w1 = transfer_to(
            Tensor(w1_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        w2 = transfer_to(
            Tensor(w2_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        h = matmul(x, w1)
        assert isinstance(h, Tensor)
        assert h.placements == (Sharded(0),)
        out = matmul(h, w2)
        assert isinstance(out, Tensor)
        assert out.placements == (Sharded(0),)
        arr = out.to_numpy()
        expected = x_np @ w1_np @ w2_np
        np.testing.assert_allclose(arr, expected, rtol=1e-5)

    def test_dp_uneven_batch(self) -> None:
        """MESH_1D: uneven batch split across 4 devices."""
        # 5 rows cannot be evenly split across 4 devices: 2,1,1,1
        x_np = np.ones((5, 4), dtype=np.float32)
        w_np = np.full((4, 4), 2.0, dtype=np.float32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        result = matmul(x, w)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        arr = result.to_numpy()
        np.testing.assert_allclose(arr, x_np @ w_np, rtol=1e-5)


# ── Partial Passthrough ──────────────────────────────────────────────


class _PartialPassthrough:
    """Partial x R -> Partial (bilinear passthrough)."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_partial_replicated(self) -> None:
        """P x R -> P via self.partial_fn, reduce_op preserved."""
        a_np = np.ones((2, 4), dtype=np.float32)
        b_np = np.ones((4, 3), dtype=np.float32)
        a = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        result = matmul(a, b)
        assert any(isinstance(p, Partial) for p in result.placements)
        assert result.placements == (Partial(),)

    def test_partial_correctness(self) -> None:
        """P x R -> P values are correct after resolve.

        Uses transfer_to(Tensor(...)) for rhs to test the CPU->GPU path.
        """
        a_np = np.ones((2, 4), dtype=np.float32)
        b_np = np.ones((4, 4), dtype=np.float32)
        a = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        result = matmul(a, b)
        assert result.placements == (Partial(),)
        # each shard: ones[2,4] @ ones[4,4] = 4*ones[2,4]
        # allreduce across 4 devices = 16*ones[2,4]
        np.testing.assert_allclose(
            result.to_numpy(), np.full((2, 4), 16.0), rtol=1e-4
        )

    @pytest.mark.skip(
        reason="Allreduce kernel bug on odd element counts (e.g. 6 floats)"
    )
    def test_partial_correctness_odd_shape(self) -> None:
        """P x R with non-power-of-2 output shape.

        Exercises allreduce on (2, 3) = 6 elements. This catches
        alignment bugs in the allreduce kernel.
        """
        a_np = np.ones((2, 4), dtype=np.float32)
        b_np = np.ones((4, 3), dtype=np.float32)
        a = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        result = matmul(a, b)
        assert result.placements == (Partial(),)
        # each shard: ones[2,4] @ ones[4,3] = 4*ones[2,3]
        # allreduce across 4 devices = 16*ones[2,3]
        np.testing.assert_allclose(
            result.to_numpy(), np.full((2, 3), 16.0), rtol=1e-4
        )


# ── Batched Matmul ───────────────────────────────────────────────────


class _BatchedMatmul:
    """Batched matmul with various placement strategies."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_batch_parallel(self) -> None:
        """MESH_2: S(batch) x S(batch) -> S(batch)."""
        lhs = ones(
            [4, 3, 5],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        rhs = ones(
            [4, 5, 6],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        out = matmul(lhs, rhs)
        assert isinstance(out, Tensor)
        assert out.placements == (Sharded(0),)
        assert list(out.shape) == [4, 3, 6]

    def test_batch_sharded_lhs_only(self) -> None:
        """MESH_2: S(batch) x R -> S(batch) (2D rhs broadcasts), values correct."""
        lhs = ones(
            [4, 3, 5],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(0),)),
        )
        rhs = ones(
            [5, 6],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        out = matmul(lhs, rhs)
        assert out.placements == (Sharded(0),)
        assert list(out.shape) == [4, 3, 6]
        np.testing.assert_allclose(
            out.to_numpy(), np.full((4, 3, 6), 5.0), rtol=1e-4
        )

    def test_row_sharded_3d(self) -> None:
        """MESH_2: S(1) x R -> S(1), values correct."""
        lhs = ones(
            [2, 4, 5],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(1),)),
        )
        rhs = ones(
            [2, 5, 6],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Replicated(),)),
        )
        out = matmul(lhs, rhs)
        assert out.placements == (Sharded(1),)
        assert list(out.shape) == [2, 4, 6]
        np.testing.assert_allclose(
            out.to_numpy(), np.full((2, 4, 6), 5.0), rtol=1e-4
        )

    def test_row_tp_3d_partial(self) -> None:
        """MESH_2: 3D matmul S(K) x S(K_rhs) -> Partial, per-shard values correct."""
        lhs = ones(
            [2, 3, 6],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(2),)),
        )
        rhs = ones(
            [2, 6, 4],
            dtype=_F32,
            device=PlacementMapping(self.MESH_2, (Sharded(1),)),
        )
        out = matmul(lhs, rhs)
        assert any(isinstance(p, Partial) for p in out.placements)
        # ones(2,3,6) @ ones(2,6,4) = 6*ones(2,3,4), gathered from Partial
        np.testing.assert_allclose(
            out.to_numpy(), np.full((2, 3, 4), 6.0), rtol=1e-4
        )

    def test_numerics_batch_parallel(self) -> None:
        """Verify actual matmul values for batched case."""
        lhs_np = np.ones((4, 3, 5), dtype=np.float32)
        rhs_np = np.ones((4, 5, 6), dtype=np.float32)
        expected = lhs_np @ rhs_np  # [4, 3, 6] filled with 5.0

        _M2_S0 = PlacementMapping(self.MESH_2, (Sharded(0),))
        lhs = transfer_to(
            full([4, 3, 5], 1.0, dtype=_F32, device=self.MESH_2.devices[0]),
            _M2_S0,
        )
        rhs = transfer_to(
            full([4, 5, 6], 1.0, dtype=_F32, device=self.MESH_2.devices[0]),
            _M2_S0,
        )
        out = matmul(lhs, rhs)
        assert out.placements == (Sharded(0),)
        np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-4)


# ── TP Chain (MLP pattern) ──────────────────────────────────────────


class _TPChain:
    """End-to-end TP MLP chain: ColTP -> RowTP -> AllReduce."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    @staticmethod
    def _tp_mlp_static(
        mesh: DeviceMesh,
        x_shape: list[int],
        d: int,
    ) -> Tensor:
        """Column TP -> relu -> Row TP -> full_tensor."""
        _m = PlacementMapping
        x = ones(x_shape, dtype=_F32, device=_m(mesh, (Replicated(),)))
        W1 = ones([d, d], dtype=_F32, device=_m(mesh, (Sharded(1),)))
        W2 = ones([d, d], dtype=_F32, device=_m(mesh, (Sharded(0),)))
        return matmul(relu(matmul(x, W1)), W2)

    @staticmethod
    def _dp_mlp_static(
        mesh: DeviceMesh,
        x_shape: list[int],
        d: int,
    ) -> Tensor:
        """Data parallel MLP: S(0) x R -> S(0)."""
        _m = PlacementMapping
        x = ones(x_shape, dtype=_F32, device=_m(mesh, (Sharded(0),)))
        W1 = ones([d, d], dtype=_F32, device=_m(mesh, (Replicated(),)))
        W2 = ones([d, d], dtype=_F32, device=_m(mesh, (Replicated(),)))
        return matmul(relu(matmul(x, W1)), W2)

    def test_tp_mlp_chain(self) -> None:
        """MESH_1D: full TP MLP chain — check intermediates and output."""
        _D = 4
        _m = PlacementMapping
        x = ones([4, _D], dtype=_F32, device=_m(self.MESH_1D, (Replicated(),)))
        W1 = ones([_D, _D], dtype=_F32, device=_m(self.MESH_1D, (Sharded(1),)))
        W2 = ones([_D, _D], dtype=_F32, device=_m(self.MESH_1D, (Sharded(0),)))

        h = matmul(x, W1)  # Col-TP -> Sharded(1)
        assert any(isinstance(p, Sharded) for p in h.placements)
        h = relu(h)
        out = matmul(h, W2)  # Row-TP -> Partial
        assert any(isinstance(p, Partial) for p in out.placements)

        result = out.materialize()
        # ones @ ones = D, relu(D) = D, D @ ones = D*D = 16
        np.testing.assert_allclose(
            result.to_numpy(), np.full((4, _D), 16.0), rtol=1e-4
        )

    def test_tp_vs_dp_equivalent(self) -> None:
        """TP and DP produce the same numerical result."""
        _D = 4
        tp_raw = self._tp_mlp_static(self.MESH_1D, [4, _D], _D)
        assert any(isinstance(p, Partial) for p in tp_raw.placements)
        dp_raw = self._dp_mlp_static(self.MESH_1D, [4, _D], _D)
        assert dp_raw.placements == (Sharded(0),)

        tp = tp_raw.to_numpy()
        dp = dp_raw.to_numpy()
        np.testing.assert_allclose(tp, dp, rtol=1e-4)


# ── Error Cases ──────────────────────────────────────────────────────


class _Errors:
    """Unsupported placement combinations that should raise."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]


# ── Combined base class ─────────────────────────────────────────────

# ── LayerNorm ───────────────────────────────────────────────────────


class _LayerNorm:
    """Tests for distributed layer_norm dispatch."""

    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_layer_norm_batch_sharded(self) -> None:
        """layer_norm on batch-sharded tensor — placement preserved."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)
        b_np = np.zeros(8, dtype=np.float32)

        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = layer_norm(x, w, b, epsilon=1e-5)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 8]

    def test_layer_norm_partial_auto_reduces(self) -> None:
        """layer_norm on Partial input auto-reduces before normalizing.

        This is the critical matmul → layer_norm pattern in row-TP:
        matmul produces Partial, layer_norm must all-reduce first because
        LN(p0) + LN(p1) ≠ LN(p0 + p1).
        """
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)
        b_np = np.zeros(8, dtype=np.float32)

        x = self.partial_fn(x_np, self.MESH_2, (Partial(),))
        w = transfer_to(
            Tensor(w_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        b = transfer_to(
            Tensor(b_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = layer_norm(x, w, b, epsilon=1e-5)
        assert result.placements == (Replicated(),)

        # Verify numerics: the result should match layer_norm on the
        # FULL tensor (x_np * num_devices, since make_partial replicates).
        n = self.MESH_2.num_devices
        full_x = x_np * n  # make_partial replicates → all-reduce sums
        mean = full_x.mean(axis=-1, keepdims=True)
        var = full_x.var(axis=-1, keepdims=True)
        expected = (full_x - mean) / np.sqrt(var + 1e-5) * w_np + b_np
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


class MatmulTests(
    _RowTP,
    _ColTP,
    _DataParallel,
    _PartialPassthrough,
    _BatchedMatmul,
    _TPChain,
    _Errors,
    _LayerNorm,
):
    """Aggregates all matmul/linalg test classes for thin subclassing."""
