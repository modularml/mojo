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
"""End-to-end tests for the tile_based_fusion graph compile option.

Compiling with ``session.compile(graph, tile_based_fusion=True)`` selects the
tile-based programming model: elementwise kernels operate on ``TileTensor``
values (driven by ``foreach_fusion_tile``) instead of SIMD. These tests
exercise the whole path -- graph build, tile-based-fusion compile, init, and
GPU execution -- and check the result against numpy for a standalone
``mo.add``, a fused ``mo.add`` + ``mo.mul`` chain, a bare ``mo.matmul``, and a
fused ``mo.matmul`` + bias ``mo.add`` epilogue. The ``mo.add``/``mo.matmul``
cases are each covered both with runtime inputs and with a compile-time-constant
operand.

The ``mo.reduce.rms_norm`` cases exercise the STORE variant of tile-based
epilogue fusion (the fused epilogue owns the store), as opposed to the COMPUTE
variant used by the matmul epilogue (the epilogue returns a value the primary
kernel stores): a bare ``rms_norm`` (no epilogue) and a fused ``rms_norm`` +
broadcast ``mo.mul`` store epilogue.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_SHAPE = [256, 256]


def _run_tile_based_fusion(
    session: InferenceSession, graph: Graph, *inputs: np.ndarray
) -> np.ndarray:
    """Compile ``graph`` with tile_based_fusion, execute on GPU, return output."""
    compiled = session.compile(graph, tile_based_fusion=True)
    model = session.init(compiled)
    device_inputs = [
        Buffer.from_numpy(arr).to(model.input_devices[i])
        for i, arr in enumerate(inputs)
    ]
    (result,) = model.execute(*device_inputs)
    return result.to_numpy()


def test_tile_based_fusion_add() -> None:
    """A standalone ``mo.add`` graph runs under the tile programming model."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_add",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs = (v.tensor for v in graph.inputs)
        graph.output(ops.add(lhs, rhs))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b)

    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_add_mul() -> None:
    """A fused ``mo.add`` + ``mo.mul`` chain runs under the tile programming model."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_add_mul",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs, scale = (v.tensor for v in graph.inputs)
        graph.output(ops.mul(ops.add(lhs, rhs), scale))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    c = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b, c)

    np.testing.assert_allclose(out, (a + b) * c, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_matmul() -> None:
    """A standalone ``mo.matmul`` graph runs under the tile programming model."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_matmul",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs = (v.tensor for v in graph.inputs)
        graph.output(ops.matmul(lhs, rhs))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b)

    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_matmul_add() -> None:
    """A fused ``mo.matmul`` + bias ``mo.add`` runs under the tile programming model."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    bias_shape = [1, _SHAPE[1]]
    with Graph(
        "tile_based_fusion_matmul_add",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, bias_shape, device=gpu),
        ],
    ) as graph:
        lhs, rhs, bias = (v.tensor for v in graph.inputs)
        graph.output(ops.add(ops.matmul(lhs, rhs), bias))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    bias_np = np.random.randn(*bias_shape).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b, bias_np)

    np.testing.assert_allclose(out, a @ b + bias_np, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_matmul_residual_add() -> None:
    """A fused ``mo.matmul`` + full-tensor residual ``mo.add`` (no broadcast).

    Unlike ``test_tile_based_fusion_matmul_add`` (broadcast ``[1, N]`` bias),
    the residual is a full ``[M, N]`` runtime tensor, so the ``ComputeTile``
    fusion captures a full-rank aux input as a ``TileTensor``.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_matmul_residual_add",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs, residual = (v.tensor for v in graph.inputs)
        graph.output(ops.add(ops.matmul(lhs, rhs), residual))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    residual_np = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b, residual_np)

    np.testing.assert_allclose(out, a @ b + residual_np, rtol=1e-5, atol=1e-5)


def _rms_norm_ref(
    x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def test_tile_based_fusion_rms_norm() -> None:
    """A standalone ``mo.reduce.rms_norm`` runs under the tile programming model.

    Exercises the tile store kernel's no-epilogue path (the driver stores
    directly; no ``OutputFusionTile`` is bound).
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_rms_norm",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, [_SHAPE[1]], device=gpu),
        ],
    ) as graph:
        x, gamma = (v.tensor for v in graph.inputs)
        graph.output(ops.rms_norm(x, gamma, epsilon=1e-6))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    g = np.random.randn(_SHAPE[1]).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, g)

    np.testing.assert_allclose(out, _rms_norm_ref(a, g), rtol=1e-3, atol=1e-3)


def test_tile_based_fusion_rms_norm_mul() -> None:
    """A fused ``mo.reduce.rms_norm`` + broadcast ``mo.mul`` store epilogue.

    Exercises the STORE variant of tile-based epilogue fusion: the graph
    compiler binds the ``mo.mul`` as an ``OutputFusionTile`` and the fusion
    (not the primary kernel) owns the store.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    scale_shape = [1, _SHAPE[1]]
    with Graph(
        "tile_based_fusion_rms_norm_mul",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, [_SHAPE[1]], device=gpu),
            TensorType(DType.float32, scale_shape, device=gpu),
        ],
    ) as graph:
        x, gamma, scale = (v.tensor for v in graph.inputs)
        graph.output(ops.mul(ops.rms_norm(x, gamma, epsilon=1e-6), scale))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    g = np.random.randn(_SHAPE[1]).astype(np.float32)
    scale_np = np.random.randn(*scale_shape).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, g, scale_np)

    np.testing.assert_allclose(
        out, _rms_norm_ref(a, g) * scale_np, rtol=1e-3, atol=1e-3
    )


def test_tile_based_fusion_rms_norm_residual_add() -> None:
    """A fused ``mo.reduce.rms_norm`` + full-tensor residual ``mo.add`` store epilogue.

    Exercises the STORE variant (``OutputFusionTile``) capturing a full ``[M, N]``
    residual as a ``TileTensor`` aux input (vs the broadcast ``[1, N]`` scale in
    ``test_tile_based_fusion_rms_norm_mul``); the fusion owns the store.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    with Graph(
        "tile_based_fusion_rms_norm_residual_add",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, [_SHAPE[1]], device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        x, gamma, residual = (v.tensor for v in graph.inputs)
        graph.output(ops.add(ops.rms_norm(x, gamma, epsilon=1e-6), residual))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    g = np.random.randn(_SHAPE[1]).astype(np.float32)
    residual_np = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, g, residual_np)

    np.testing.assert_allclose(
        out, _rms_norm_ref(a, g) + residual_np, rtol=1e-3, atol=1e-3
    )


def test_tile_based_fusion_add_constant() -> None:
    """A standalone ``mo.add`` with a compile-time-constant operand."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    with Graph(
        "tile_based_fusion_add_constant",
        input_types=[TensorType(DType.float32, _SHAPE, device=gpu)],
    ) as graph:
        (lhs,) = (v.tensor for v in graph.inputs)
        rhs = ops.constant(b, DType.float32, device=gpu)
        graph.output(ops.add(lhs, rhs))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a)

    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_add_mul_constant() -> None:
    """A fused ``mo.add`` + ``mo.mul`` where the scale is a compile-time constant."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    c = np.random.randn(*_SHAPE).astype(np.float32)
    with Graph(
        "tile_based_fusion_add_mul_constant",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs = (v.tensor for v in graph.inputs)
        scale = ops.constant(c, DType.float32, device=gpu)
        graph.output(ops.mul(ops.add(lhs, rhs), scale))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b)

    np.testing.assert_allclose(out, (a + b) * c, rtol=1e-5, atol=1e-5)


def test_tile_based_fusion_matmul_add_constant() -> None:
    """A fused ``mo.matmul`` + bias ``mo.add`` where the bias is a constant."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    session = InferenceSession(devices=[Accelerator()])
    gpu = DeviceRef.GPU(0)
    bias_shape = [1, _SHAPE[1]]
    bias_np = np.random.randn(*bias_shape).astype(np.float32)
    with Graph(
        "tile_based_fusion_matmul_add_constant",
        input_types=[
            TensorType(DType.float32, _SHAPE, device=gpu),
            TensorType(DType.float32, _SHAPE, device=gpu),
        ],
    ) as graph:
        lhs, rhs = (v.tensor for v in graph.inputs)
        bias = ops.constant(bias_np, DType.float32, device=gpu)
        graph.output(ops.add(ops.matmul(lhs, rhs), bias))

    a = np.random.randn(*_SHAPE).astype(np.float32)
    b = np.random.randn(*_SHAPE).astype(np.float32)
    out = _run_tile_based_fusion(session, graph, a, b)

    np.testing.assert_allclose(out, a @ b + bias_np, rtol=1e-5, atol=1e-5)
