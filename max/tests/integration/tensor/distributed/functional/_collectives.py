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
"""Shared test logic for collective ops and transfer_to.

DO NOT run this file directly — it contains base classes that are
subclassed by test_collectives_simulated.py and test_collectives_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh   — 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh   — 4 devices, shape (2,2), axis_names=("dp","tp")
    partial_fn: Callable   — make_partial (CPU) or make_partial (GPU)

Redistribute transition table (same-mesh):
============================================

    Source → Target          | Action
    -------------------------|------------------------------------------
    Same → Same              | no-op
    Partial → Replicated     | allreduce within group
    Partial → Sharded(dim)   | allreduce + local split (reduce-scatter)
    Sharded → Replicated     | allgather within group
    Replicated → Sharded     | local split (no communication)
    Sharded(i) → Sharded(j)  | allgather(i) + local split(j)
    Partial → Partial        | no-op if same, else not supported

Cross-mesh transitions:
    1. Resolve to Replicated on source mesh
    2. Transfer shard[0] to target mesh primary device
    3. scatter to target placements on target mesh
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pytest
from max.experimental.functional import (
    allgather,
    allreduce_sum,
    reduce_scatter,
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


class CollectivesTests:
    """Base class with collective test methods.

    Subclass must set MESH_1D, MESH_2D, and partial_fn.
    """

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    partial_fn: ClassVar[Callable[..., Tensor]]

    # ─── Helper: validate ALL shards numerically ─────────────────────

    @staticmethod
    def _compute_shard(
        full: np.ndarray,
        device_idx: int,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> np.ndarray:
        """Compute what shard device_idx should hold given placements."""
        result = full
        coords: list[int] = []
        remaining = device_idx
        for ax in range(len(mesh.mesh_shape) - 1, -1, -1):
            coords.insert(0, remaining % mesh.mesh_shape[ax])
            remaining //= mesh.mesh_shape[ax]

        for ax, p in enumerate(placements):
            if isinstance(p, Sharded):
                n = mesh.mesh_shape[ax]
                coord = coords[ax]
                dim_size = result.shape[p.axis]
                chunk = dim_size // n
                start = coord * chunk
                end = start + chunk
                slices = [slice(None)] * result.ndim
                slices[p.axis] = slice(start, end)
                result = result[tuple(slices)]
        return result

    def _assert_shards(
        self,
        result: Tensor,
        expected_full: np.ndarray,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> None:
        """Validate placements AND all shards numerically."""
        assert result.placements == placements
        for i in range(mesh.num_devices):
            actual = result.local_shards[i].to_numpy()
            expected = self._compute_shard(expected_full, i, mesh, placements)
            np.testing.assert_allclose(actual, expected, rtol=1e-5)

    # ─── Helper: create mixed Partial+Sharded tensor ────────────────

    def _make_partial_sharded(
        self,
        data: np.ndarray,
        mesh: DeviceMesh,
        placements: tuple[Any, ...],
    ) -> Tensor:
        """Create a tensor with mixed Partial+Sharded placements.

        Shards along any Sharded axes first, then relabels Partial axes.
        (make_partial only handles pure-Partial placements.)
        """
        # Build a shard placement replacing Partial with Replicated.
        shard_placements = tuple(
            Replicated() if isinstance(p, Partial) else p for p in placements
        )
        sharded = transfer_to(
            Tensor(data), PlacementMapping(mesh, tuple(shard_placements))
        )
        return Tensor._from_shards(
            tuple(s.driver_tensor for s in sharded.local_shards),
            mesh,
            placements,
            data.shape,
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 1: Atomic collective ops (allreduce, allgather, reduce_scatter)
    # ═════════════════════════════════════════════════════════════════════

    # ── allreduce_sum ────────────────────────────────────────────────

    def test_allreduce_1d(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = allreduce_sum(t, mesh_axis=0)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    def test_allreduce_2d_tp_axis(self) -> None:
        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        t = self.partial_fn(a, self.MESH_2D, (Replicated(), Partial()))
        result = allreduce_sum(t, mesh_axis=1)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 2, rtol=1e-5
            )

    def test_allreduce_2d_dp_axis(self) -> None:
        a = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Replicated()))
        result = allreduce_sum(t, mesh_axis=0)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 2, rtol=1e-5
            )

    def test_allreduce_shard_values(self) -> None:
        """All 4 shards hold the reduced value after allreduce."""
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = allreduce_sum(t, mesh_axis=0)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    # ── allgather ────────────────────────────────────────────────────

    def test_allgather_1d_axis0(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=0)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    def test_allgather_1d_axis1(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(2, 16)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(1),))
        )
        result = allgather(sharded, tensor_axis=1, mesh_axis=0)
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    def test_allgather_2d_tp_axis(self) -> None:
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        sharded = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(0),
                ),
            ),
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=1)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    def test_allgather_2d_dp_axis(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        sharded = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=0)
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), t_np, rtol=1e-5
            )

    # ── reduce_scatter ───────────────────────────────────────────────

    def test_reduce_scatter_1d(self) -> None:
        a = np.ones((8, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = reduce_scatter(t, scatter_axis=0, mesh_axis=0)
        assert result.placements == (Sharded(0),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(),
                np.full((2, 4), 4.0, dtype=np.float32),
                rtol=1e-5,
            )

    def test_reduce_scatter_roundtrip(self) -> None:
        a = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        scattered = reduce_scatter(t, scatter_axis=0, mesh_axis=0)
        gathered = allgather(scattered, tensor_axis=0, mesh_axis=0)
        np.testing.assert_allclose(
            gathered.local_shards[0].to_numpy(), a * 4, rtol=1e-5
        )

    def test_reduce_scatter_shard_values(self) -> None:
        a = np.arange(8, dtype=np.float32).reshape(4, 2)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = reduce_scatter(t, scatter_axis=0, mesh_axis=0)
        assert result.placements == (Sharded(0),)
        expected_full = a * 4
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(),
                expected_full[i : i + 1],
                rtol=1e-5,
            )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 2: allreduce_sum / allgather multi-axis tests
    # ═════════════════════════════════════════════════════════════════════

    def test_allreduce_sum_1d_partial(self) -> None:
        """Partial on a 1D mesh — single allreduce resolves to Replicated."""
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = allreduce_sum(t, mesh_axis=0)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            result.local_shards[0].to_numpy(), a * 4, rtol=1e-5
        )

    def test_allgather_1d_sharded(self) -> None:
        """Sharded on a 1D mesh — single allgather resolves to Replicated."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = allgather(sharded, tensor_axis=0, mesh_axis=0)
        assert result.placements == (Replicated(),)
        np.testing.assert_allclose(
            result.local_shards[0].to_numpy(), t_np, rtol=1e-5
        )

    def test_allreduce_sum_2d_both_axes(self) -> None:
        """Partial on both axes — two allreduces, one per axis."""
        a = np.ones((2, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        t = allreduce_sum(t, mesh_axis=0)
        t = allreduce_sum(t, mesh_axis=1)
        assert t.placements == (Replicated(), Replicated())
        np.testing.assert_allclose(
            t.local_shards[0].to_numpy(), a * 4, rtol=1e-5
        )

    def test_allreduce_sum_preserves_sharded_axis(self) -> None:
        """(Sharded, Partial) — allreduce on axis 1 leaves axis 0 Sharded."""
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Sharded(0), Partial()))
        result = allreduce_sum(t, mesh_axis=1)
        assert result.placements[0] == Sharded(0)
        assert result.placements[1] == Replicated()

    # ═════════════════════════════════════════════════════════════════════
    #  Section 3: scatter (distribute) — non-distributed → distributed
    # ═════════════════════════════════════════════════════════════════════

    def test_scatter_sharded_1d(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        result = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    def test_scatter_replicated_1d(self) -> None:
        t_np = np.ones((2, 4), dtype=np.float32)
        result = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    def test_scatter_2d_mesh(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = transfer_to(
            Tensor(t_np),
            PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1))),
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    def test_scatter_2d_both_sharded(self) -> None:
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        result = transfer_to(
            Tensor(t_np),
            PlacementMapping(self.MESH_2D, (Sharded(0), Sharded(1))),
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Sharded(1))
        )

    def test_scatter_roundtrip(self) -> None:
        original = np.arange(32, dtype=np.float32).reshape(8, 4)
        distributed = transfer_to(
            Tensor(original), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        gathered = allgather(distributed, tensor_axis=0, mesh_axis=0)
        np.testing.assert_allclose(
            gathered.local_shards[0].to_numpy(), original, rtol=1e-5
        )

    def test_scatter_rejects_partial(self) -> None:
        with pytest.raises(ValueError, match="Partial"):
            transfer_to(
                Tensor(np.ones((2, 4), dtype=np.float32)),
                PlacementMapping(self.MESH_1D, (Partial(),)),
            )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 4: materialize / to_numpy
    # ═════════════════════════════════════════════════════════════════════

    def test_materialize_sharded(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = sharded.materialize()
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), t_np, rtol=1e-5)

    def test_materialize_partial(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = t.materialize()
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), a * 4, rtol=1e-5)

    def test_materialize_noop(self) -> None:
        t = Tensor(np.ones((2, 4), dtype=np.float32))
        result = t.materialize()
        assert not result.is_distributed

    def test_to_numpy_distributed(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        np.testing.assert_allclose(sharded.to_numpy(), t_np, rtol=1e-5)

    def test_to_numpy_non_distributed(self) -> None:
        t_np = np.ones((2, 4), dtype=np.float32)
        np.testing.assert_allclose(Tensor(t_np).to_numpy(), t_np, rtol=1e-5)

    # ═════════════════════════════════════════════════════════════════════
    #  Section 5: transfer_to — same mesh, 1D
    # ═════════════════════════════════════════════════════════════════════

    def test_redistribute_noop(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = transfer_to(
            sharded, PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    def test_redistribute_sharded_to_replicated(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = transfer_to(
            sharded, PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    def test_redistribute_replicated_to_sharded(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        replicated = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        result = transfer_to(
            replicated, PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    def test_redistribute_partial_to_replicated(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = transfer_to(t, PlacementMapping(self.MESH_1D, (Replicated(),)))
        assert result.placements == (Replicated(),)
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    def test_redistribute_partial_to_sharded(self) -> None:
        a = np.ones((8, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = transfer_to(t, PlacementMapping(self.MESH_1D, (Sharded(0),)))
        assert result.placements == (Sharded(0),)
        expected = a * 4
        self._assert_shards(result, expected, self.MESH_1D, (Sharded(0),))

    def test_redistribute_sharded_to_sharded_different_axis(self) -> None:
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = transfer_to(
            sharded, PlacementMapping(self.MESH_1D, (Sharded(1),))
        )
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(1),))

    def test_redistribute_3d_tensor_sharded_axis0_to_axis2(self) -> None:
        t_np = np.arange(96, dtype=np.float32).reshape(4, 6, 4)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        result = transfer_to(t, PlacementMapping(self.MESH_1D, (Sharded(2),)))
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(2),))

    def test_redistribute_3d_tensor_sharded_to_replicated(self) -> None:
        t_np = np.arange(96, dtype=np.float32).reshape(2, 8, 6)
        t = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(1),))
        )
        result = transfer_to(t, PlacementMapping(self.MESH_1D, (Replicated(),)))
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    def test_redistribute_partial_same_is_noop(self) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        result = transfer_to(t, PlacementMapping(self.MESH_1D, (Partial(),)))
        assert result.placements == (Partial(),)

    # ═════════════════════════════════════════════════════════════════════
    #  Section 6: transfer_to — same mesh, 2D, single-axis change
    # ═════════════════════════════════════════════════════════════════════

    def test_redistribute_2d_partial_partial_to_replicated_replicated(
        self,
    ) -> None:
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_2D, (Partial(), Partial()))
        result = transfer_to(
            t, PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        )
        assert result.placements == (Replicated(), Replicated())
        for i in range(4):
            np.testing.assert_allclose(
                result.local_shards[i].to_numpy(), a * 4, rtol=1e-5
            )

    def test_redistribute_2d_replicated_sharded_to_sharded_replicated(
        self,
    ) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(1),
                ),
            ),
        )
        result = transfer_to(
            t, PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Replicated())
        )

    def test_redistribute_2d_partial_sharded_to_replicated_sharded(
        self,
    ) -> None:
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Partial(), Sharded(0)))
        result = transfer_to(
            t, PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        )
        expected = a * 2  # allreduce on dp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Replicated(), Sharded(0))
        )

    def test_redistribute_2d_sharded_to_sharded_different_axis(
        self,
    ) -> None:
        t_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Replicated(),
                    Sharded(0),
                ),
            ),
        )
        result = transfer_to(
            t, PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        )
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 7: transfer_to — same mesh, 2D, BOTH axes change
    # ═════════════════════════════════════════════════════════════════════

    def test_redistribute_2d_S0_S1_to_R_R(self) -> None:
        """(Sharded(0), Sharded(1)) → (Replicated, Replicated)."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Replicated())
        )

    def test_redistribute_2d_S0_S1_to_S1_S0(self) -> None:
        """(Sharded(0), Sharded(1)) → (Sharded(1), Sharded(0))."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_2D, (Sharded(1), Sharded(0)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(1), Sharded(0))
        )

    def test_redistribute_2d_S0_S1_to_R_S0(self) -> None:
        """(Sharded(0), Sharded(1)) → (Replicated, Sharded(0))."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(0))
        )

    def test_redistribute_2d_S0_R_to_R_S0(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(0)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(0))
        )

    def test_redistribute_2d_S0_R_to_R_S1(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_2D, (Replicated(), Sharded(1)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Replicated(), Sharded(1))
        )

    def test_redistribute_2d_S0_R_to_S0_S1(self) -> None:
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        result = transfer_to(t, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Sharded(1))
        )

    # ── Mixed Partial+Sharded source, both axes change ───────────────

    def test_redistribute_2d_P_S0_to_R_R(self) -> None:
        """(Partial, Sharded(0)) → (Replicated, Replicated)."""
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Partial(), Sharded(0)))
        target = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on dp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Replicated(), Replicated())
        )

    def test_redistribute_2d_P_S0_to_S0_R(self) -> None:
        """(Partial, Sharded(0)) → (Sharded(0), Replicated).

        Partial on dp can't reduce-scatter to Sharded(0) because tp already
        shards axis 0. Falls back to allreduce + allgather + local_split.
        """
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Partial(), Sharded(0)))
        target = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on dp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Sharded(0), Replicated())
        )

    def test_redistribute_2d_S0_P_to_R_R(self) -> None:
        """(Sharded(0), Partial) → (Replicated, Replicated)."""
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        t = self._make_partial_sharded(a, self.MESH_2D, (Sharded(0), Partial()))
        target = PlacementMapping(self.MESH_2D, (Replicated(), Replicated()))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on tp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Replicated(), Replicated())
        )

    def test_redistribute_2d_S0_P_to_S0_S1(self) -> None:
        """(Sharded(0), Partial) → (Sharded(0), Sharded(1))."""
        a = np.arange(64, dtype=np.float32).reshape(4, 16)
        t = self._make_partial_sharded(a, self.MESH_2D, (Sharded(0), Partial()))
        target = PlacementMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        result = transfer_to(t, target)
        expected = a * 2  # allreduce on tp (size 2)
        self._assert_shards(
            result, expected, self.MESH_2D, (Sharded(0), Sharded(1))
        )

    # ═════════════════════════════════════════════════════════════════════
    #  Section 8: transfer_to — cross-mesh transfers
    # ═════════════════════════════════════════════════════════════════════

    def test_cross_mesh_sharded_to_single_device(self) -> None:
        """4-device sharded → 1-device (materialize)."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        single_mesh = DeviceMesh.single(self.MESH_1D.devices[0])
        target = PlacementMapping(single_mesh, (Replicated(),))
        result = transfer_to(sharded, target)
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), t_np, rtol=1e-5)

    def test_cross_mesh_single_to_sharded(self) -> None:
        """1-device → 4-device sharded (weight loading)."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = Tensor(t_np)
        target = PlacementMapping(self.MESH_1D, (Sharded(0),))
        result = transfer_to(t, target)
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    def test_cross_mesh_single_to_replicated(self) -> None:
        """1-device → 4-device replicated."""
        t_np = np.ones((4, 4), dtype=np.float32)
        t = Tensor(t_np)
        target = PlacementMapping(self.MESH_1D, (Replicated(),))
        result = transfer_to(t, target)
        self._assert_shards(result, t_np, self.MESH_1D, (Replicated(),))

    def test_cross_mesh_partial_to_single_device(self) -> None:
        """4-device partial → 1-device (allreduce + materialize)."""
        a = np.ones((4, 4), dtype=np.float32)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        single_mesh = DeviceMesh.single(self.MESH_1D.devices[0])
        target = PlacementMapping(single_mesh, (Replicated(),))
        result = transfer_to(t, target)
        assert not result.is_distributed
        np.testing.assert_allclose(result.to_numpy(), a * 4, rtol=1e-5)

    def test_cross_mesh_1d_to_2d(self) -> None:
        """1D sharded(4) → 2D (2,2) with (Sharded(0), Replicated)."""
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded_1d = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        target = PlacementMapping(self.MESH_2D, (Sharded(0), Replicated()))
        result = transfer_to(sharded_1d, target)
        self._assert_shards(
            result, t_np, self.MESH_2D, (Sharded(0), Replicated())
        )

    def test_cross_mesh_2d_to_1d(self) -> None:
        """2D (Sharded(0), Replicated) → 1D Sharded(0)."""
        t_np = np.arange(32, dtype=np.float32).reshape(4, 8)
        sharded_2d = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Replicated(),
                ),
            ),
        )
        target = PlacementMapping(self.MESH_1D, (Sharded(0),))
        result = transfer_to(sharded_2d, target)
        self._assert_shards(result, t_np, self.MESH_1D, (Sharded(0),))

    # ═════════════════════════════════════════════════════════════════════
    #  Section 9: roundtrip tests — transfer_to A→B→A = identity
    # ═════════════════════════════════════════════════════════════════════

    def test_roundtrip_sharded_replicated_sharded(self) -> None:
        t_np = np.arange(32, dtype=np.float32).reshape(8, 4)
        sharded = transfer_to(
            Tensor(t_np), PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        replicated = transfer_to(
            sharded, PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        back = transfer_to(
            replicated, PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        self._assert_shards(back, t_np, self.MESH_1D, (Sharded(0),))

    def test_roundtrip_partial_to_replicated_to_sharded(self) -> None:
        a = np.arange(32, dtype=np.float32).reshape(8, 4)
        t = self.partial_fn(a, self.MESH_1D, (Partial(),))
        replicated = transfer_to(
            t, PlacementMapping(self.MESH_1D, (Replicated(),))
        )
        sharded = transfer_to(
            replicated, PlacementMapping(self.MESH_1D, (Sharded(0),))
        )
        expected = a * 4
        self._assert_shards(sharded, expected, self.MESH_1D, (Sharded(0),))

    def test_roundtrip_2d_sharded_swap_axes(self) -> None:
        """(S0, S1) → (S1, S0) → (S0, S1) = identity."""
        t_np = np.arange(64, dtype=np.float32).reshape(4, 16)
        original = transfer_to(
            Tensor(t_np),
            PlacementMapping(
                self.MESH_2D,
                (
                    Sharded(0),
                    Sharded(1),
                ),
            ),
        )
        swapped = transfer_to(
            original, PlacementMapping(self.MESH_2D, (Sharded(1), Sharded(0)))
        )
        back = transfer_to(
            swapped, PlacementMapping(self.MESH_2D, (Sharded(0), Sharded(1)))
        )
        self._assert_shards(back, t_np, self.MESH_2D, (Sharded(0), Sharded(1)))
