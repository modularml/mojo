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
"""Data-parallel tests with a symbolic batch dimension.

Each test stresses one potential ``int(Dim)`` coercion site in the sharding
+ collective stack: eager scatter, pre-sharded compile, view ops on
non-sharded axes, graph-traced ``transfer_to`` scatter, and the full
scatter → MLP → gather loop.

Subclasses set ``MESH_DP`` (2 devices, ``axis_names=("dp",)``).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental.functional import (
    add,
    matmul,
    relu,
    reshape,
    squeeze,
    transfer_to,
    transpose,
    unsqueeze,
)
from max.experimental.nn.module import Module, module_dataclass
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
    TensorLayout,
)
from max.experimental.tensor import Tensor

F32 = DType.float32
H = 4


def _sym_cpu_input() -> TensorLayout:
    return TensorLayout(F32, ["batch", H], device=CPU())


def _sym_dp_input(mesh: DeviceMesh) -> TensorLayout:
    return TensorLayout(F32, ["batch", H], DeviceMapping(mesh, (Sharded(0),)))


def _replicated_w(mesh: DeviceMesh, data: np.ndarray) -> Tensor:
    return transfer_to(Tensor(data), PlacementMapping(mesh, (Replicated(),)))


def _presharded_input(mesh: DeviceMesh, batch_np: np.ndarray) -> Tensor:
    return transfer_to(Tensor(batch_np), PlacementMapping(mesh, (Sharded(0),)))


class DPDynamicTests:
    MESH_DP: ClassVar[DeviceMesh]

    def test_eager_scatter_static_batch(self) -> None:
        """Eager DP baseline — static batch."""
        mesh = self.MESH_DP
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((6, H)).astype(np.float32)
        w_np = rng.standard_normal((H, H)).astype(np.float32)

        x = _presharded_input(mesh, x_np)
        w = _replicated_w(mesh, w_np)

        y = add(relu(matmul(x, w)), 0.5)
        assert y.placements == (Sharded(0),)

        expected = np.maximum(x_np @ w_np, 0.0) + 0.5
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=1e-2, atol=1e-3)

    def test_presharded_matmul_varying_batch(self) -> None:
        """Compile once with symbolic-batch Sharded(0); call with varying batches."""
        mesh = self.MESH_DP

        @module_dataclass
        class Linear(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                # S(0) @ R  →  S(0)
                return matmul(x, self.W)

        rng = np.random.default_rng(1)
        w_np = rng.standard_normal((H, H)).astype(np.float32)
        model = Linear(W=_replicated_w(mesh, w_np))
        compiled = model.compile(_sym_dp_input(mesh))

        for batch in (4, 8, 6):  # 6 is intentionally not 2x of a power-of-two
            x_np = rng.standard_normal((batch, H)).astype(np.float32)
            result = compiled(_presharded_input(mesh, x_np))
            assert result.placements == (Sharded(0),)
            np.testing.assert_allclose(
                result.to_numpy(), x_np @ w_np, rtol=1e-2, atol=1e-3
            )

    def test_presharded_view_ops_varying_batch(self) -> None:
        """View ops on non-sharded axes while batch is symbolic."""
        mesh = self.MESH_DP

        @module_dataclass
        class ViewOps(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                y = unsqueeze(x, 1)
                y = transpose(y, 1, 2)
                y = squeeze(y, 2)
                y = reshape(y, [-1, H // 2, 2])
                y = reshape(y, [-1, H])
                return matmul(y, self.W)

        rng = np.random.default_rng(2)
        w_np = rng.standard_normal((H, H)).astype(np.float32)
        model = ViewOps(W=_replicated_w(mesh, w_np))
        compiled = model.compile(_sym_dp_input(mesh))

        for batch in (4, 10):
            x_np = rng.standard_normal((batch, H)).astype(np.float32)
            result = compiled(_presharded_input(mesh, x_np))
            assert result.placements == (Sharded(0),)
            np.testing.assert_allclose(
                result.to_numpy(), x_np @ w_np, rtol=1e-2, atol=1e-3
            )

    def test_graph_scatter_varying_batch(self) -> None:
        """``transfer_to`` scatter baked into the compiled graph with symbolic batch."""
        mesh = self.MESH_DP

        @module_dataclass
        class ScatterMM(Module[[Tensor], Tensor]):
            W: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dp = transfer_to(x, PlacementMapping(mesh, (Sharded(0),)))
                return matmul(x_dp, self.W)

        rng = np.random.default_rng(3)
        w_np = rng.standard_normal((H, H)).astype(np.float32)
        model = ScatterMM(W=_replicated_w(mesh, w_np))
        compiled = model.compile(_sym_cpu_input())

        for batch in (4, 8):
            x_np = rng.standard_normal((batch, H)).astype(np.float32)
            result = compiled(Tensor(x_np, device=CPU()))
            np.testing.assert_allclose(
                result.to_numpy(), x_np @ w_np, rtol=1e-2, atol=1e-3
            )

    def test_dp_mlp_full_loop(self) -> None:
        """Full DP step: CPU input → scatter → MLP → gather."""
        mesh = self.MESH_DP

        @module_dataclass
        class DPMLP(Module[[Tensor], Tensor]):
            W1: Tensor
            W2: Tensor

            def forward(self, x: Tensor) -> Tensor:
                x_dp = transfer_to(x, PlacementMapping(mesh, (Sharded(0),)))
                h = relu(matmul(x_dp, self.W1))
                out = matmul(h, self.W2)
                return transfer_to(out, PlacementMapping(mesh, (Replicated(),)))

        rng = np.random.default_rng(4)
        w1_np = rng.standard_normal((H, H)).astype(np.float32) * 0.1
        w2_np = rng.standard_normal((H, H)).astype(np.float32) * 0.1
        model = DPMLP(
            W1=_replicated_w(mesh, w1_np), W2=_replicated_w(mesh, w2_np)
        )
        compiled = model.compile(_sym_cpu_input())

        for batch in (4, 12):
            x_np = rng.standard_normal((batch, H)).astype(np.float32)
            result = compiled(Tensor(x_np, device=CPU()))
            assert result.placements == (Replicated(),)
            expected = np.maximum(x_np @ w1_np, 0.0) @ w2_np
            np.testing.assert_allclose(
                result.to_numpy(), expected, rtol=1e-2, atol=1e-3
            )

    def test_dp_mlp_eager_static_batch(self) -> None:
        """Eager DP MLP sanity check with a static batch."""
        mesh = self.MESH_DP
        rng = np.random.default_rng(5)
        x_np = rng.standard_normal((8, H)).astype(np.float32)
        w1_np = rng.standard_normal((H, H)).astype(np.float32) * 0.1
        w2_np = rng.standard_normal((H, H)).astype(np.float32) * 0.1

        x = transfer_to(Tensor(x_np), PlacementMapping(mesh, (Sharded(0),)))
        w1 = _replicated_w(mesh, w1_np)
        w2 = _replicated_w(mesh, w2_np)

        h = relu(matmul(x, w1))
        out = matmul(h, w2)
        result = transfer_to(out, PlacementMapping(mesh, (Replicated(),)))

        expected = np.maximum(x_np @ w1_np, 0.0) @ w2_np
        np.testing.assert_allclose(
            result.to_numpy(), expected, rtol=1e-2, atol=1e-3
        )
