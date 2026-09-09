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
"""Shared test logic for reduction ops.

DO NOT run this file directly — it contains base classes that are
subclassed by functional/test_reduction_simulated.py and
functional/test_reduction_multi_gpu.py.

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
from max.experimental.functional import (
    cumsum,
    mean,
    softmax,
    sum,
    transfer_to,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor

# ── Sum ──────────────────────────────────────────────────────────────


class _Sum:
    """Tests for distributed sum reduction."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_non_sharded_axis(self) -> None:
        """MESH_1D: Sharded(0) + sum along axis=1 preserves placement."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = sum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        # Graph ops keep the reduced dim as size 1.
        expected = t_np.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_numerics(self) -> None:
        """MESH_1D with 4 shards: sum along non-sharded axis is correct."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = sum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        expected = t_np.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_partial_passthrough(self) -> None:
        """MESH_1D: Partial placement passes through sum (linear op)."""
        a_np = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        result = sum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Partial(),)
        # to_numpy() materializes (allreduces) the Partial across N devices,
        # so each partial sum of 4.0 becomes 4.0 * N.
        n = self.MESH_1D.num_devices
        result_np = result.to_numpy()
        assert result_np.shape == (4, 1)
        expected = a_np.sum(axis=1, keepdims=True) * n
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)

    def test_axis_none_replicated_ok(self) -> None:
        """MESH_2: sum(Replicated, axis=None) works — all shards identical."""
        t_np = np.arange(8, dtype=np.float32).reshape(2, 4)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Replicated(),))
        )
        result = sum(t, axis=None)
        # axis=None flattens then reduces → scalar-like (1,) shape
        expected = t_np.sum()
        np.testing.assert_allclose(
            result.to_numpy().item(), expected, rtol=1e-5
        )


# ── Mean ─────────────────────────────────────────────────────────────


class _Mean:
    """Tests for distributed mean reduction."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_non_sharded_axis(self) -> None:
        """MESH_2: Sharded(0) + mean along axis=-1 preserves placement."""
        t_np = np.arange(24, dtype=np.float32).reshape(4, 6)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = mean(t, axis=-1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        expected = t_np.mean(axis=-1, keepdims=True)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


# ── Softmax ──────────────────────────────────────────────────────────


class _Softmax:
    """Tests for distributed softmax."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_non_sharded_axis(self) -> None:
        """MESH_1D: softmax along non-sharded axis preserves placement."""
        a_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = transfer_to(
            Tensor(a_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = softmax(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        arr = result.to_numpy()
        # Compute expected softmax along axis=1
        e = np.exp(a_np - a_np.max(axis=1, keepdims=True))
        expected = e / e.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(arr, expected, rtol=1e-5)


# ── Cumsum ──────────────────────────────────────────────────────────


class _Cumsum:
    """Tests for distributed cumsum."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_non_sharded_axis(self) -> None:
        """MESH_2: cumsum along non-sharded axis preserves placement."""
        t_np = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_2, (Sharded(0),))
        )
        result = cumsum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Sharded(0),)
        expected = np.cumsum(t_np, axis=1)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_partial_passthrough(self) -> None:
        """MESH_1D: Partial passes through cumsum (linear op).

        cumsum is linear: cumsum(a+b) = cumsum(a) + cumsum(b).
        Partial fragments should pass through without all-reduce.
        """
        a_np = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a_np, self.MESH_1D, (Partial(),))
        result = cumsum(t, axis=1)
        assert isinstance(result, Tensor)
        assert result.placements == (Partial(),)
        # to_numpy() materializes (allreduces) the Partial across N devices,
        # so each partial cumsum becomes cumsum(a) * N.
        n = self.MESH_1D.num_devices
        result_np = result.to_numpy()
        expected = np.cumsum(a_np, axis=1) * n
        np.testing.assert_allclose(result_np, expected, rtol=1e-5)


# ── Combined base class ─────────────────────────────────────────────


class ReductionTests(_Sum, _Mean, _Softmax, _Cumsum):
    """Aggregates all reduction test classes for thin subclassing."""
