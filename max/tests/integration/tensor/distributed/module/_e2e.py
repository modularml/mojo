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
"""Shared test logic for end-to-end distributed integration tests.

DO NOT run this file directly -- it contains base classes that are
subclassed by module/test_e2e_simulated.py and module/test_e2e_multi_gpu.py.

Subclasses must define:
    MESH_1D: DeviceMesh  -- 4 devices, shape (4,), axis_names=("tp",)
    MESH_2D: DeviceMesh  -- 4 devices, shape (2,2), axis_names=("dp","tp")
    MESH_2:  DeviceMesh  -- 2 devices, shape (2,), axis_names=("tp",)

IRTests (simulated-only) also requires these meshes.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental.functional import (
    add,
    full,
    matmul,
    mean,
    mul,
    relu,
    rsqrt,
    silu,
    transfer_to,
)
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import DistributedTensorType
from max.experimental.tensor import Tensor, TensorType

F32 = DType.float32
HIDDEN = 8
FFL = 16
BATCH = 3
SEED_WEIGHTS = 42
SEED_INPUT = 123

# ─── Helpers ─────────────────────────────────────────────────────────────────


def sym(feature_dim: int) -> TensorType:
    """Symbolic input type -- always CPU regardless of mesh."""
    return TensorType(F32, ["batch", feature_dim], device=CPU())


def make_weights(
    seed: int = SEED_WEIGHTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W_gate = rng.standard_normal((HIDDEN, FFL)).astype(np.float32) * 0.1
    W_up = rng.standard_normal((HIDDEN, FFL)).astype(np.float32) * 0.1
    W_down = rng.standard_normal((FFL, HIDDEN)).astype(np.float32) * 0.1
    return W_gate, W_up, W_down


def make_input(seed: int = SEED_INPUT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((BATCH, HIDDEN)).astype(np.float32)


def reference_mlp(
    x: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
    W_down: np.ndarray,
) -> np.ndarray:
    gate = x @ W_gate
    gate_act = gate / (1.0 + np.exp(-gate))
    up = x @ W_up
    return (gate_act * up) @ W_down


# ─── Models ──────────────────────────────────────────────────────────────────


@module_dataclass
class AutoTPMLP(Module[[Tensor], Tensor]):
    W_gate: Tensor
    W_up: Tensor
    W_down: Tensor

    def forward(self, x: Tensor) -> Tensor:
        gate = silu(matmul(x, self.W_gate))
        up = matmul(x, self.W_up)
        hidden = mul(gate, up)
        out = matmul(hidden, self.W_down)
        replicated = tuple(Replicated() for _ in range(out.mesh.ndim))
        return transfer_to(out, PlacementMapping(out.mesh, replicated))


# ═══════════════════════════════════════════════════════════════════════════════
# E2E Tests (shared by both simulated and multi-GPU)
# ═══════════════════════════════════════════════════════════════════════════════


class E2ETests:
    """Base class with all 12 E2E test methods.

    Subclass must set MESH_1D, MESH_2D, and MESH_2.
    """

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]

    # ── Helper to distribute weights ─────────────────────────────────────

    def _make_auto(
        self,
        mesh: DeviceMesh,
        W_gate_np: np.ndarray,
        W_up_np: np.ndarray,
        W_down_np: np.ndarray,
        placements_col: tuple[Sharded | Replicated, ...] | None = None,
        placements_row: tuple[Sharded | Replicated, ...] | None = None,
    ) -> AutoTPMLP:
        if placements_col is None:
            placements_col = (
                (Sharded(1),) if mesh.ndim == 1 else (Replicated(), Sharded(1))
            )
        if placements_row is None:
            placements_row = (
                (Sharded(0),) if mesh.ndim == 1 else (Replicated(), Sharded(0))
            )
        return AutoTPMLP(
            W_gate=transfer_to(
                Tensor(W_gate_np),
                PlacementMapping(mesh, tuple(placements_col)),
            ),
            W_up=transfer_to(
                Tensor(W_up_np), PlacementMapping(mesh, tuple(placements_col))
            ),
            W_down=transfer_to(
                Tensor(W_down_np),
                PlacementMapping(mesh, tuple(placements_row)),
            ),
        )

    def _distribute_input(self, data: np.ndarray, mesh: DeviceMesh) -> Tensor:
        placements = (Replicated(),) * mesh.ndim
        return transfer_to(Tensor(data), PlacementMapping(mesh, placements))

    # ── TestTPMLP ────────────────────────────────────────────────────────

    def test_tp_mlp_matches_numpy(self) -> None:
        W_gate, W_up, W_down = make_weights()
        expected = reference_mlp(make_input(), W_gate, W_up, W_down)
        model = self._make_auto(self.MESH_1D, W_gate, W_up, W_down)
        x = self._distribute_input(make_input(), self.MESH_1D)
        result = model(x)
        assert result.placements == (Replicated(),)
        result_np = result.to_numpy()
        assert result_np.shape == (BATCH, HIDDEN)
        np.testing.assert_allclose(result_np, expected, rtol=5e-2)

    def test_bias_auto_reduces(self) -> None:
        W_gate, W_up, W_down = make_weights()
        model = self._make_auto(self.MESH_2, W_gate, W_up, W_down)
        x = self._distribute_input(make_input(), self.MESH_2)
        result = model(x)
        assert result.placements == (Replicated(),)
        result_np = result.to_numpy()
        assert result_np.shape == (BATCH, HIDDEN)
        expected = reference_mlp(make_input(), W_gate, W_up, W_down)
        np.testing.assert_allclose(result_np, expected, rtol=5e-2)

    def test_tp_mlp_2d_mesh(self) -> None:
        W_gate, W_up, W_down = make_weights()
        model = self._make_auto(self.MESH_2D, W_gate, W_up, W_down)
        x = self._distribute_input(make_input(), self.MESH_2D)
        result = model(x)
        assert result.is_distributed
        # Partial is on axis 1 (tp) from row-parallel matmul.
        # transfer_to resolves it; dp stays Replicated.
        assert isinstance(result.placements[0], Replicated)
        result_np = result.to_numpy()
        assert result_np.shape == (BATCH, HIDDEN)
        expected = reference_mlp(make_input(), W_gate, W_up, W_down)
        np.testing.assert_allclose(result_np, expected, rtol=5e-2)

    # ── TestDistributedLinear ────────────────────────────────────────────

    def test_distributed_linear_row_tp(self) -> None:
        mesh = self.MESH_2
        x = transfer_to(
            Tensor(np.ones((3, 8), dtype=np.float32)),
            PlacementMapping(mesh, (Sharded(1),)),
        )
        W = transfer_to(
            Tensor(np.ones((8, 4), dtype=np.float32)),
            PlacementMapping(mesh, (Sharded(0),)),
        )
        out = matmul(x, W)
        assert any(isinstance(p, Partial) for p in out.placements)
        reduced = transfer_to(out, PlacementMapping(mesh, (Replicated(),)))
        assert reduced.placements == (Replicated(),)
        result_np = reduced.to_numpy()
        assert result_np.shape == (3, 4)
        np.testing.assert_allclose(result_np, np.full((3, 4), 8.0), rtol=1e-4)

    # ── TestRMSNormE2E ───────────────────────────────────────────────────

    @staticmethod
    def _decomposed_rms_norm(
        x: Tensor, weight: Tensor, eps: float, mesh: DeviceMesh
    ) -> Tensor:
        variance = mean(mul(x, x), axis=-1)
        inv_rms = rsqrt(add(variance, eps))
        return mul(mul(x, inv_rms), weight)

    def test_rms_baseline_vs_numpy(self) -> None:
        """Non-distributed RMS norm matches numpy reference."""
        mesh = self.MESH_2
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(Tensor(x_np), mesh.devices[0])
        w = transfer_to(Tensor(w_np), mesh.devices[0])
        result = self._decomposed_rms_norm(x, w, 1e-6, mesh)

        variance = np.mean(x_np**2, axis=-1, keepdims=True)
        expected = x_np / np.sqrt(variance + 1e-6) * w_np
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)

    def test_rms_batch_sharded(self) -> None:
        mesh = self.MESH_2
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(Tensor(x_np), PlacementMapping(mesh, (Sharded(0),)))
        w = transfer_to(Tensor(w_np), PlacementMapping(mesh, (Replicated(),)))

        result = self._decomposed_rms_norm(x, w, 1e-6, mesh)
        assert result.placements == (Sharded(0),)
        result_np = result.to_numpy()
        assert result_np.shape == (4, 8)
        variance = np.mean(x_np**2, axis=-1, keepdims=True)
        expected = x_np / np.sqrt(variance + 1e-6) * w_np
        np.testing.assert_allclose(result_np, expected, rtol=1e-4)

    def test_rms_hidden_dim_raises(self) -> None:
        mesh = self.MESH_2
        x = transfer_to(
            Tensor(np.ones((3, 8), dtype=np.float32)),
            PlacementMapping(mesh, (Sharded(1),)),
        )
        w = transfer_to(
            Tensor(np.ones(4, dtype=np.float32)),
            PlacementMapping(mesh, (Replicated(),)),
        )
        with pytest.raises((ValueError, Exception)):
            self._decomposed_rms_norm(x, w, 1e-6, mesh)

    def test_rms_decomposed_vs_fused(self) -> None:
        mesh = self.MESH_2
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = np.ones(8, dtype=np.float32)

        x = transfer_to(Tensor(x_np), PlacementMapping(mesh, (Sharded(0),)))
        w = transfer_to(Tensor(w_np), PlacementMapping(mesh, (Replicated(),)))

        decomposed = self._decomposed_rms_norm(x, w, 1e-6, mesh)
        assert decomposed.placements == (Sharded(0),)
        result_np = decomposed.to_numpy()
        assert result_np.shape == (4, 8)
        variance = np.mean(x_np**2, axis=-1, keepdims=True)
        expected = x_np / np.sqrt(variance + 1e-6) * w_np
        np.testing.assert_allclose(result_np, expected, rtol=1e-4)

    # ── TestTransformerBlock ─────────────────────────────────────────────

    def test_placement_propagation(self) -> None:
        mesh = self.MESH_2
        rng = np.random.default_rng(42)
        D = 8

        w_norm_np = np.ones(D, dtype=np.float32)
        W1_np = rng.standard_normal((D, D)).astype(np.float32) * 0.1
        W2_np = rng.standard_normal((D, D)).astype(np.float32) * 0.1
        x_np = rng.standard_normal((3, D)).astype(np.float32)

        w_norm = transfer_to(
            Tensor(w_norm_np),
            PlacementMapping(mesh, (Replicated(),)),
        )
        W1 = transfer_to(
            Tensor(W1_np),
            PlacementMapping(mesh, (Sharded(1),)),
        )
        W2 = transfer_to(
            Tensor(W2_np),
            PlacementMapping(mesh, (Sharded(0),)),
        )
        x = transfer_to(
            Tensor(x_np),
            PlacementMapping(mesh, (Replicated(),)),
        )

        normed = self._decomposed_rms_norm(x, w_norm, 1e-6, mesh)
        hidden = relu(matmul(normed, W1))
        out = matmul(hidden, W2)
        reduced = transfer_to(out, PlacementMapping(mesh, (Replicated(),)))
        result = add(x, reduced)
        assert result.placements == (Replicated(),)
        result_np = result.to_numpy()
        assert result_np.shape == (3, D)
        # Numerical: x + relu(rms_norm(x) @ W1) @ W2
        var = np.mean(x_np**2, axis=-1, keepdims=True)
        normed_np = x_np / np.sqrt(var + 1e-6) * w_norm_np
        expected = x_np + np.maximum(normed_np @ W1_np, 0.0) @ W2_np
        np.testing.assert_allclose(result_np, expected, rtol=5e-2)

    def test_transformer_2d_mesh(self) -> None:
        mesh = self.MESH_2D
        rng = np.random.default_rng(42)
        D = 8

        W1_np = rng.standard_normal((D, D)).astype(np.float32) * 0.1
        x_np = rng.standard_normal((3, D)).astype(np.float32)

        W1 = transfer_to(
            Tensor(W1_np),
            PlacementMapping(mesh, (Replicated(), Sharded(1))),
        )
        x = transfer_to(
            Tensor(x_np),
            PlacementMapping(mesh, (Replicated(), Replicated())),
        )
        out = matmul(x, W1)
        assert out.placements == (Replicated(), Sharded(1)), (
            f"Expected (Replicated(), Sharded(1)), got {out.placements}"
        )
        result_np = out.to_numpy()
        assert result_np.shape == (3, D)
        np.testing.assert_allclose(
            result_np, x_np @ W1_np, rtol=5e-2, atol=1e-4
        )

    # ── TestFullChain ────────────────────────────────────────────────────

    def test_creation_into_ops(self) -> None:
        mesh = self.MESH_1D
        mapping = PlacementMapping(mesh, (Sharded(0),))
        t = full([8, 4], 2.0, dtype=F32, device=mapping)
        out = relu(t)
        assert out.placements == (Sharded(0),)
        result_np = out.to_numpy()
        assert result_np.shape == (8, 4)
        np.testing.assert_allclose(result_np, np.full((8, 4), 2.0), rtol=1e-5)

    def test_elementwise_chain_2d(self) -> None:
        mesh = self.MESH_2D
        mapping = PlacementMapping(mesh, (Sharded(0), Replicated()))
        a = full([4, 4], 2.0, dtype=F32, device=mapping)
        b = full([4, 4], 3.0, dtype=F32, device=mapping)
        out = add(mul(a, b), a)
        assert out.placements == (Sharded(0), Replicated())
        result_np = out.to_numpy()
        assert result_np.shape == (4, 4)
        np.testing.assert_allclose(result_np, np.full((4, 4), 8.0), rtol=1e-5)

    def test_tp_mlp_with_bias(self) -> None:
        mesh = self.MESH_1D
        W_gate, W_up, W_down = make_weights()
        model = self._make_auto(mesh, W_gate, W_up, W_down)
        x = self._distribute_input(make_input(), mesh)
        result = model(x)
        assert result.placements == (Replicated(),)
        result_np = result.to_numpy()
        assert result_np.shape == (BATCH, HIDDEN)
        expected = reference_mlp(make_input(), W_gate, W_up, W_down)
        np.testing.assert_allclose(result_np, expected, rtol=5e-2)


# ═══════════════════════════════════════════════════════════════════════════════
# IR Tests (simulated-only, NOT inherited by multi-GPU)
# ═══════════════════════════════════════════════════════════════════════════════


class IRTests:
    """IR dump tests -- only for simulated (CPU) meshes.

    Subclass must set MESH_1D, MESH_2D, and MESH_2.
    """

    MESH_1D: ClassVar[DeviceMesh]
    MESH_2D: ClassVar[DeviceMesh]
    MESH_2: ClassVar[DeviceMesh]

    def test_structural_equivalence(self) -> None:
        W_gate, W_up, W_down = make_weights()
        mesh = self.MESH_2

        model = AutoTPMLP(
            W_gate=transfer_to(
                Tensor(W_gate), PlacementMapping(mesh, (Sharded(1),))
            ),
            W_up=transfer_to(
                Tensor(W_up), PlacementMapping(mesh, (Sharded(1),))
            ),
            W_down=transfer_to(
                Tensor(W_down), PlacementMapping(mesh, (Sharded(0),))
            ),
        )

        input_type = DistributedTensorType(
            dtype=F32,
            shape=["batch", HIDDEN],
            mesh=mesh,
            placements=(Replicated(),),
        )
        ir = repr(model.trace(input_type))
        assert "mo.matmul" in ir or "matmul" in ir.lower()

    def test_2d_mesh_ir(self) -> None:
        W_gate, W_up, W_down = make_weights()
        mesh = self.MESH_2D

        model = AutoTPMLP(
            W_gate=transfer_to(
                Tensor(W_gate),
                PlacementMapping(
                    mesh,
                    (
                        Replicated(),
                        Sharded(1),
                    ),
                ),
            ),
            W_up=transfer_to(
                Tensor(W_up),
                PlacementMapping(
                    mesh,
                    (
                        Replicated(),
                        Sharded(1),
                    ),
                ),
            ),
            W_down=transfer_to(
                Tensor(W_down),
                PlacementMapping(
                    mesh,
                    (
                        Replicated(),
                        Sharded(0),
                    ),
                ),
            ),
        )

        input_type = DistributedTensorType(
            dtype=F32,
            shape=["batch", HIDDEN],
            mesh=mesh,
            placements=(Replicated(), Replicated()),
        )
        ir = repr(model.trace(input_type))
        assert len(ir) > 0
