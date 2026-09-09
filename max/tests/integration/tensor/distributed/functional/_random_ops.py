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
"""Shared test logic for random distributed ops.

DO NOT run this file directly — it contains base classes that are
subclassed by functional/test_random_simulated.py and
functional/test_random_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from max.dtype import DType
from max.experimental.functional import gaussian, uniform
from max.experimental.sharding import (
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
)

_F32 = DType.float32

# ── TestUniform ──────────────────────────────────────────────────────────


class _Uniform:
    """Tests for the distributed uniform random op."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]

    def test_uniform_replicated(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        t = uniform([4, 8], range=(0.0, 1.0), dtype=_F32, device=mapping)
        assert t.is_distributed
        assert t.placements == (Replicated(), Replicated())
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        assert result.shape == (4, 8)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_uniform_sharded(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        t = uniform([4, 8], range=(2.0, 5.0), dtype=_F32, device=mapping)
        assert t.placements == (Replicated(), Sharded(1))
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        assert result.shape == (4, 8)
        assert np.all(result >= 2.0) and np.all(result <= 5.0)

    def test_uniform_custom_range(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        t = uniform([8, 4], range=(-1.0, 1.0), dtype=_F32, device=mapping)
        assert t.placements == (Sharded(0), Replicated())
        assert tuple(t.shape) == (8, 4)
        result = t.to_numpy()
        assert result.shape == (8, 4)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)


# ── TestGaussian ─────────────────────────────────────────────────────────


class _Gaussian:
    """Tests for the distributed gaussian random op."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]

    def test_gaussian_replicated(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        t = gaussian([8, 8], mean=0.0, std=1.0, dtype=_F32, device=mapping)
        assert t.is_distributed
        assert t.placements == (Replicated(), Replicated())
        assert tuple(t.shape) == (8, 8)
        result = t.to_numpy()
        assert result.shape == (8, 8)
        # Loose statistical check: mean should be within a reasonable range.
        np.testing.assert_allclose(np.mean(result), 0.0, atol=2.0)

    def test_gaussian_sharded(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        t = gaussian([4, 8], mean=5.0, std=0.01, dtype=_F32, device=mapping)
        assert t.placements == (Replicated(), Sharded(1))
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        assert result.shape == (4, 8)
        # With std=0.01, values should be tightly clustered around mean=5.0.
        np.testing.assert_allclose(result, 5.0, atol=0.5)


class RandomTests(_Uniform, _Gaussian):
    """Aggregates all random test classes for thin subclassing."""
