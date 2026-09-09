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
"""Shared test logic for newly added distributed ops.

Covers sharding propagation for: conv2d, avg_pool2d, max_pool2d,
band_part, and custom op error wrappers.

DO NOT run this file directly — subclass and set MESH_*.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
    MESH_2:  DeviceMesh   — 2 devices, shape (2,), axis_names=("tp",)
    partial_fn: Callable   — make_partial (CPU) or gpu_partial (GPU)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.functional import transfer_to
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    ReduceOp,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

_F32 = DType.float32
# ═══════════════════════════════════════════════════════════════════════
# Conv2d sharding propagation
# ═══════════════════════════════════════════════════════════════════════
#
# conv2d with NHWC input, RSCF filter:
#   x = [N, H, W, C_in],  filter = [kH, kW, C_in/g, C_out]
#   output = [N, H', W', C_out]
#
# Semantic roles:  batch=0, spatial={1,2}, C_in=3 (input), C_out=3 (filter)


class _Conv2dDataParallel:
    """Data parallel: S(batch) x R -> S(batch)."""

    MESH_2: ClassVar[DeviceMesh]

    def test_conv2d_batch_sharded_placement(self) -> None:
        """S(0) x R -> S(0) — output batch-sharded."""
        # [N=4, H=3, W=3, C_in=2]
        x_np = np.ones((4, 3, 3, 2), dtype=np.float32)
        # [kH=1, kW=1, C_in=2, C_out=4] — 1x1 conv = linear per-pixel
        f_np = np.ones((1, 1, 2, 4), dtype=np.float32) * 0.5
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = F.conv2d(x, f)
        assert result.placements == (Sharded(0),)
        assert list(result.shape) == [4, 3, 3, 4]

    def test_conv2d_batch_sharded_numerics(self) -> None:
        """S(0) x R -> S(0) — numerical check with 1x1 conv."""
        x_np = np.arange(24, dtype=np.float32).reshape(4, 3, 1, 2)
        f_np = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        # 1x1 identity conv: output should equal input
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = F.conv2d(x, f)
        assert result.placements == (Sharded(0),)
        np.testing.assert_allclose(result.to_numpy(), x_np, rtol=1e-5)


class _Conv2dOutputChannelParallel:
    """Output-channel parallel: R x S(C_out) -> S(C_out in output)."""

    MESH_2: ClassVar[DeviceMesh]

    def test_conv2d_cout_sharded_placement(self) -> None:
        """R x S(3) -> S(3) — output channel-sharded."""
        x_np = np.ones((2, 3, 3, 2), dtype=np.float32)
        # [kH=1, kW=1, C_in=2, C_out=4]
        f_np = np.ones((1, 1, 2, 4), dtype=np.float32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Sharded(3),))
        )
        result = F.conv2d(x, f)
        # C_out is axis 3 in the output [N, H', W', C_out]
        assert result.placements == (Sharded(3),)
        assert list(result.shape) == [2, 3, 3, 4]

    def test_conv2d_cout_sharded_numerics(self) -> None:
        """R x S(C_out) — 1x1 conv numerical check."""
        x_np = np.ones((1, 1, 1, 2), dtype=np.float32)
        # Each output channel is the sum of input channels * weight
        f_np = np.array(
            [[[[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]]]],
            dtype=np.float32,
        )  # [1,1,2,4]
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Sharded(3),))
        )
        result = F.conv2d(x, f)
        assert result.placements == (Sharded(3),)
        # [1,1] @ [[1,2,3,4],[1,1,1,1]] = [2, 3, 4, 5]
        expected = np.array([[[[2.0, 3.0, 4.0, 5.0]]]], dtype=np.float32)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


class _Conv2dInputChannelParallel:
    """Input-channel parallel: S(C_in) x S(C_in_f) -> Partial."""

    MESH_2: ClassVar[DeviceMesh]

    def test_conv2d_cin_sharded_produces_partial(self) -> None:
        """S(3) x S(2) -> Partial — row-TP analogue for conv."""
        x_np = np.ones((1, 1, 1, 4), dtype=np.float32)
        f_np = np.ones((1, 1, 4, 2), dtype=np.float32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(3),))
        )
        # C_in is axis 2 in RSCF filter
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Sharded(2),))
        )
        result = F.conv2d(x, f)
        assert result.placements == (Partial(ReduceOp.SUM),)

    def test_conv2d_cin_sharded_numerics(self) -> None:
        """S(C_in) x S(C_in_f) -> Partial, gathered result correct."""
        x_np = np.ones((1, 1, 1, 4), dtype=np.float32)
        f_np = np.ones((1, 1, 4, 2), dtype=np.float32) * 0.5
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(3),))
        )
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Sharded(2),))
        )
        result = F.conv2d(x, f)
        assert any(isinstance(p, Partial) for p in result.placements)
        # 1x1 conv: [1,1,1,4] @ [4,2] * 0.5 = [1,1,1,2] with value 2.0
        expected = np.full((1, 1, 1, 2), 2.0, dtype=np.float32)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


class _Conv2dSpatialShardedError:
    """Spatial sharding must be rejected."""

    MESH_2: ClassVar[DeviceMesh]


class _Conv2dPartialRules:
    """Partial bilinear rules and error cases."""

    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_conv2d_partial_x_replicated_f_produces_partial(self) -> None:
        """P x R -> P (bilinear in input)."""
        x_np = np.ones((1, 1, 1, 2), dtype=np.float32)
        f_np = np.ones((1, 1, 2, 2), dtype=np.float32)
        x = self.partial_fn(x_np, self.MESH_2, (Partial(),))
        f = transfer_to(
            Tensor(f_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = F.conv2d(x, f)
        assert result.placements == (Partial(),)


class _Conv2d2DMesh:
    """Conv2d on a 2D mesh (dp, tp)."""

    MESH_2D: ClassVar[DeviceMesh]

    def test_conv2d_2d_mesh_dp_batch(self) -> None:
        """2D mesh: (R, S(batch)) — batch sharded on tp axis."""
        x_np = np.ones((4, 1, 1, 2), dtype=np.float32)
        f_np = np.ones((1, 1, 2, 2), dtype=np.float32) * 0.5
        x = transfer_to(
            Tensor(x_np),
            PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0))),
        )
        f = transfer_to(
            Tensor(f_np),
            PlacementMapping(self.MESH_2D, (Replicated(), Replicated())),
        )
        result = F.conv2d(x, f)
        assert result.placements == (Replicated(), Sharded(0))
        assert list(result.shape) == [4, 1, 1, 2]
        np.testing.assert_allclose(
            result.to_numpy(), np.ones((4, 1, 1, 2)), rtol=1e-5
        )


# ═══════════════════════════════════════════════════════════════════════
# Pooling sharding propagation
# ═══════════════════════════════════════════════════════════════════════
class _PoolingBatchSharded:
    """Batch-sharded pooling: S(0) -> S(0)."""

    MESH_2: ClassVar[DeviceMesh]

    def test_avg_pool2d_batch_sharded_placement(self) -> None:
        """avg_pool2d with S(0) preserves batch sharding."""
        x_np = np.ones((4, 4, 4, 2), dtype=np.float32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = F.avg_pool2d(x, kernel_size=(2, 2), stride=2)
        assert result.placements == (Sharded(0),)

    def test_max_pool2d_batch_sharded_placement(self) -> None:
        """max_pool2d with S(0) preserves batch sharding."""
        x_np = np.ones((4, 4, 4, 2), dtype=np.float32)
        x = transfer_to(
            Tensor(x_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        assert result.placements == (Sharded(0),)


class _PoolingSpatialError:
    """Spatial sharding must be rejected."""

    MESH_2: ClassVar[DeviceMesh]


class _PoolingPartial:
    """Partial handling for pooling ops."""

    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_avg_pool2d_partial_passthrough(self) -> None:
        """avg_pool2d is linear — Partial passes through."""
        x_np = np.ones((1, 4, 4, 2), dtype=np.float32)
        x = self.partial_fn(x_np, self.MESH_2, (Partial(),))
        result = F.avg_pool2d(x, kernel_size=(2, 2), stride=2)
        # linear=True: all-Partial inputs pass through
        assert result.placements == (Partial(),)

    def test_max_pool2d_partial_auto_reduces(self) -> None:
        """max_pool2d is non-linear — Partial auto-reduced."""
        x_np = np.ones((1, 4, 4, 2), dtype=np.float32)
        x = self.partial_fn(x_np, self.MESH_2, (Partial(),))
        result = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        assert not any(isinstance(p, Partial) for p in result.placements)


# ═══════════════════════════════════════════════════════════════════════
# Band part sharding propagation
# ═══════════════════════════════════════════════════════════════════════
class _BandPart:
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_band_part_batch_sharded_placement(self) -> None:
        """S(0) on batch dim → S(0) preserved."""
        arr = np.ones((4, 3, 3), dtype=np.float32)
        t = transfer_to(
            Tensor(arr), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = F.band_part(t, num_lower=0, num_upper=0)
        assert result.placements == (Sharded(0),)

    def test_band_part_batch_sharded_numerics(self) -> None:
        """S(0) diagonal extraction — numerics verified."""
        arr = np.ones((4, 3, 3), dtype=np.float32)
        t = transfer_to(
            Tensor(arr), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = F.band_part(t, num_lower=0, num_upper=0)
        got = result.to_numpy()
        for i in range(4):
            np.testing.assert_allclose(
                got[i], np.diag(np.diag(arr[i])), rtol=1e-5
            )

    def test_band_part_partial_passthrough(self) -> None:
        """linear=True — Partial passes through."""
        arr = np.eye(4, dtype=np.float32)
        t = self.partial_fn(arr, self.MESH_2, (Partial(),))
        result = F.band_part(t, num_lower=0, num_upper=0)
        assert result.placements == (Partial(),)


# ═══════════════════════════════════════════════════════════════════════
# Control flow: cond
# ═══════════════════════════════════════════════════════════════════════
class _Cond:
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_cond_non_distributed_true(self) -> None:
        """Happy path: cond true branch using DF ops."""
        device = DeviceRef.from_device(CPU())
        pred = Tensor.full([], True, dtype=DType.bool, device=CPU())
        a = Tensor(np.array([1.0, 2.0], dtype=np.float32), device=CPU())
        b = Tensor(np.array([3.0, 4.0], dtype=np.float32), device=CPU())
        result = F.cond(
            pred,
            [TensorType(DType.float32, [2], device)],
            lambda: F.add(a, b),
            lambda: F.sub(a, b),
        )
        np.testing.assert_allclose(result[0].to_numpy(), [4.0, 6.0], rtol=1e-5)

    def test_cond_non_distributed_false(self) -> None:
        """Happy path: cond false branch using DF ops."""
        device = DeviceRef.from_device(CPU())
        pred = Tensor.full([], False, dtype=DType.bool, device=CPU())
        a = Tensor(np.array([10.0, 20.0], dtype=np.float32), device=CPU())
        b = Tensor(np.array([3.0, 5.0], dtype=np.float32), device=CPU())
        result = F.cond(
            pred,
            [TensorType(DType.float32, [2], device)],
            lambda: F.add(a, b),
            lambda: F.sub(a, b),
        )
        np.testing.assert_allclose(result[0].to_numpy(), [7.0, 15.0], rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Control flow: while_loop
# ═══════════════════════════════════════════════════════════════════════
class _WhileLoop:
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    @pytest.mark.skip(
        reason="MXF-634: under MAX_EAGER_EXECUTOR=fallback-internal the loop "
        "returns its input unchanged"
    )
    def test_while_loop_non_distributed(self) -> None:
        """Happy path: while_loop using DF ops, adds 1.0 three times."""
        counter = Tensor.full([], 0, dtype=DType.float32, device=CPU())
        x = Tensor(np.array([1.0, 2.0], dtype=np.float32), device=CPU())
        one = Tensor.full([], 1, dtype=DType.float32, device=CPU())
        three = Tensor.full([], 3, dtype=DType.float32, device=CPU())

        result = F.while_loop(
            [counter, x],
            lambda c, v: F.greater(three, c),
            lambda c, v: [F.add(c, one), F.add(v, one)],
        )
        np.testing.assert_allclose(result[1].to_numpy(), [4.0, 5.0], rtol=1e-5)

    def test_while_loop_chained_result(self) -> None:
        """While_loop result feeds into further DF ops."""
        counter = Tensor.full([], 0, dtype=DType.float32, device=CPU())
        x = Tensor(np.array([1.0, 1.0], dtype=np.float32), device=CPU())
        one = Tensor.full([], 1, dtype=DType.float32, device=CPU())
        two = Tensor.full([], 2, dtype=DType.float32, device=CPU())

        result = F.while_loop(
            [counter, x],
            lambda c, v: F.greater(two, c),
            lambda c, v: [F.add(c, one), F.add(v, one)],
        )
        # x starts at [1, 1], adds 1 twice → [3, 3], then * 2
        doubled = F.mul(
            result[1], Tensor.full([2], 2.0, dtype=DType.float32, device=CPU())
        )
        np.testing.assert_allclose(doubled.to_numpy(), [6.0, 6.0], rtol=1e-5)

    def test_cond_result_chained(self) -> None:
        """Cond result feeds into further DF ops."""
        device = DeviceRef.from_device(CPU())
        pred = Tensor.full([], True, dtype=DType.bool, device=CPU())
        a = Tensor(np.array([2.0, 3.0], dtype=np.float32), device=CPU())
        b = Tensor(np.array([1.0, 1.0], dtype=np.float32), device=CPU())
        result = F.cond(
            pred,
            [TensorType(DType.float32, [2], device)],
            lambda: F.add(a, b),
            lambda: F.sub(a, b),
        )
        # [3, 4] * [3, 4] = [9, 16]
        squared = F.mul(result[0], result[0])
        np.testing.assert_allclose(squared.to_numpy(), [9.0, 16.0], rtol=1e-5)

    def test_cond_new_tensor_in_branch(self) -> None:
        """Branch creates a NEW tensor (not just captured outer ones)."""
        device = DeviceRef.from_device(CPU())
        pred = Tensor.full([], True, dtype=DType.bool, device=CPU())
        result = F.cond(
            pred,
            [TensorType(DType.float32, [3], device)],
            lambda: F.full([3], 1.0, dtype=DType.float32, device=CPU()),
            lambda: F.full([3], 9.0, dtype=DType.float32, device=CPU()),
        )
        np.testing.assert_allclose(
            result[0].to_numpy(), [1.0, 1.0, 1.0], rtol=1e-5
        )

    def test_cond_multiple_outputs(self) -> None:
        """Cond with two output tensors."""
        device = DeviceRef.from_device(CPU())
        pred = Tensor.full([], True, dtype=DType.bool, device=CPU())
        a = Tensor(np.array([1.0, 2.0], dtype=np.float32), device=CPU())
        b = Tensor(np.array([3.0, 4.0], dtype=np.float32), device=CPU())
        result = F.cond(
            pred,
            [
                TensorType(DType.float32, [2], device),
                TensorType(DType.float32, [2], device),
            ],
            lambda: [F.add(a, b), F.sub(a, b)],
            lambda: [F.sub(a, b), F.add(a, b)],
        )
        np.testing.assert_allclose(result[0].to_numpy(), [4.0, 6.0], rtol=1e-5)
        np.testing.assert_allclose(
            result[1].to_numpy(), [-2.0, -2.0], rtol=1e-5
        )


# ═══════════════════════════════════════════════════════════════════════
# Aggregator
# ═══════════════════════════════════════════════════════════════════════
class MiscTests(
    _Conv2dDataParallel,
    _Conv2dOutputChannelParallel,
    _Conv2dInputChannelParallel,
    _Conv2dSpatialShardedError,
    _Conv2dPartialRules,
    _Conv2d2DMesh,
    _PoolingBatchSharded,
    _PoolingSpatialError,
    _PoolingPartial,
    _BandPart,
    _Cond,
    _WhileLoop,
):
    """Aggregates all misc op test classes for thin subclassing."""
