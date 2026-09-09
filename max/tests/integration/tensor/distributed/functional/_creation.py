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
"""Shared test logic for creation distributed ops.

DO NOT run this file directly — it contains base classes that are
subclassed by functional/test_creation_simulated.py and
functional/test_creation_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
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
    ones,
    transfer_to,
    zeros,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor

_F32 = DType.float32

# ── TestFull ─────────────────────────────────────────────────────────────


class _Full:
    """Tests for the distributed full creation op."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_full_replicated(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        t = full([4, 8], 3.0, dtype=_F32, device=mapping)
        assert t.is_distributed
        assert t.placements == (Replicated(), Replicated())
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        np.testing.assert_allclose(result, np.full((4, 8), 3.0), rtol=1e-5)

    def test_full_sharded_axis0(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        t = full([4, 8], 5.0, dtype=_F32, device=mapping)
        assert t.placements == (Sharded(0), Replicated())
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        np.testing.assert_allclose(result, np.full((4, 8), 5.0), rtol=1e-5)

    def test_full_sharded_axis1(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        t = full([4, 8], 7.0, dtype=_F32, device=mapping)
        assert t.placements == (Replicated(), Sharded(1))
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        np.testing.assert_allclose(result, np.full((4, 8), 7.0), rtol=1e-5)

    def test_full_1d_mesh(self) -> None:
        mapping = PlacementMapping(self.MESH_1D, (Sharded(0),))
        t = full([8, 4], 2.0, dtype=_F32, device=mapping)
        assert t.placements == (Sharded(0),)
        assert tuple(t.shape) == (8, 4)
        result = t.to_numpy()
        np.testing.assert_allclose(result, np.full((8, 4), 2.0), rtol=1e-5)

    def test_ones(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        t = ones([4, 8], dtype=_F32, device=mapping)
        assert t.placements == (Replicated(), Sharded(0))
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        np.testing.assert_allclose(result, np.ones((4, 8)), rtol=1e-5)

    def test_zeros(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        t = zeros([4, 8], dtype=_F32, device=mapping)
        assert t.placements == (Sharded(0), Replicated())
        assert tuple(t.shape) == (4, 8)
        result = t.to_numpy()
        np.testing.assert_allclose(result, np.zeros((4, 8)), rtol=1e-5)


# ── TestDistribute ───────────────────────────────────────────────────────


class _Distribute:
    """Tests for the distribute (shard) op."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_distribute_sharded_2d(self) -> None:
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        result = transfer_to(Tensor(arr), mapping)
        assert result.placements == (Replicated(), Sharded(1))
        assert tuple(result.shape) == (4, 8)
        np.testing.assert_allclose(result.to_numpy(), arr, rtol=1e-5)

    def test_distribute_replicated_2d(self) -> None:
        arr = np.ones((2, 4), dtype=np.float32) * 5
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(Tensor(arr), mapping)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), arr, rtol=1e-5
            )

    def test_distribute_both_axes_sharded(self) -> None:
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        result = transfer_to(Tensor(arr), mapping)
        assert result.placements == (Sharded(0), Sharded(1))
        assert tuple(result.shape) == (4, 8)
        np.testing.assert_allclose(result.to_numpy(), arr, rtol=1e-5)

    def test_distribute_rejects_partial(self) -> None:
        mapping = PlacementMapping(self.MESH_1D, (Partial(),))
        with pytest.raises(ValueError, match="Partial"):
            transfer_to(Tensor(np.ones((4, 4), dtype=np.float32)), mapping)


# ── TestProperties ───────────────────────────────────────────────────────


class _Properties:
    """Tests for metadata properties of distributed tensors."""

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    def test_is_distributed(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        t = full([4, 8], 1.0, dtype=_F32, device=mapping)
        assert t.is_distributed
        np.testing.assert_allclose(
            t.to_numpy(), np.full((4, 8), 1.0), rtol=1e-5
        )

    def test_non_distributed(self) -> None:
        t = Tensor(np.ones((2, 4), dtype=np.float32))
        assert not t.is_distributed
        np.testing.assert_allclose(t.to_numpy(), np.ones((2, 4)), rtol=1e-5)

    def test_mesh_property(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        t = full([4, 8], 1.0, dtype=_F32, device=mapping)
        assert t.mesh == self.MESH_2D

    def test_placements_property(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        t = full([4, 8], 1.0, dtype=_F32, device=mapping)
        assert t.placements == (Sharded(0), Replicated())

    def test_global_shape_sharded(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        t = full([8, 4], 1.0, dtype=_F32, device=mapping)
        assert tuple(t.shape) == (8, 4)

    def test_num_shards(self) -> None:
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        t = full([4, 8], 1.0, dtype=_F32, device=mapping)
        assert len(t.local_shards) == 4

    def test_shard_shapes_sharded_axis0(self) -> None:
        # 2D mesh (2,2), sharded on axis 0 (dp axis, size 2).
        # Global shape (8,4) -> each dp group gets (4,4).
        mapping = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        t = full([8, 4], 1.0, dtype=_F32, device=mapping)
        for s in t.local_shards:
            arr = s.to_numpy()
            assert arr.shape == (4, 4)
            np.testing.assert_allclose(arr, np.ones((4, 4)), rtol=1e-5)

    def test_shard_shapes_sharded_axis1(self) -> None:
        # 2D mesh (2,2), sharded on axis 1 (tp axis, size 2).
        # Global shape (4,8) -> each tp group gets (4,4).
        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        t = full([4, 8], 2.0, dtype=_F32, device=mapping)
        for s in t.local_shards:
            arr = s.to_numpy()
            assert arr.shape == (4, 4)
            np.testing.assert_allclose(arr, np.full((4, 4), 2.0), rtol=1e-5)

    def test_dtype_preserved(self) -> None:
        from max.experimental.functional import cast

        mapping = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        t = full([2, 4], 1.0, dtype=DType.bfloat16, device=mapping)
        assert t.dtype == DType.bfloat16
        # Cast to float32 before materializing (bfloat16 not supported by dlpack)
        t_f32 = cast(t, DType.float32)
        np.testing.assert_allclose(
            t_f32.to_numpy(), np.full((2, 4), 1.0), rtol=1e-2
        )


class CreationTests(_Full, _Distribute, _Properties):
    """Aggregates all creation test classes for thin subclassing."""
