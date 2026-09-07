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

"""End-to-end tests for the new MAP-dialect fusion system's ElementwiseFuser.

Every graph here is compiled with ``MAX_GC_USE_ADV_FUSION`` set, exercising
``MOToMAP`` + ``ElementwiseFuser`` end to end. These edge cases (duplicate
operands, diamond sharing, a dtype-changing op mid-chain) are ported from
``ElementwiseFuser``'s own MLIR test suite -- the legacy MOGG-based fuser
handles the same *shape* of graphs, but through unrelated code
(``FuseElementwiseKernels.cpp``), so running these without the env var would
just exercise that different fuser instead of the one under test. Whether the
graph actually fused into one kernel is covered by that MLIR test suite, not
here -- these are e2e correctness tests, checked against numpy as the oracle.
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_add_static(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A standalone `mo.add` compiles and runs correctly under adv fusion."""
    with Graph(
        "add_static",
        input_types=[
            TensorType(DType.float32, [8], device=DeviceRef.CPU()),
            TensorType(DType.float32, [8], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(x + y)

    a = np.random.randn(8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a, b)
    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


def test_add_dynamic(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A standalone `mo.add` over a symbolic shape compiles and runs correctly."""
    with Graph(
        "add_dynamic",
        input_types=[
            TensorType(DType.float32, ["n"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["n"], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(x + y)

    a = np.random.randn(4096).astype(np.float32)
    b = np.random.randn(4096).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a, b)
    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-5)


def test_two_op_chain_fuses(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`mul(add(x, y), z)` over a rank-2, partly-dynamic shape fuses correctly."""
    with Graph(
        "two_op_chain",
        input_types=[
            TensorType(DType.float32, ["batch", 64], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["batch", 64], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["batch", 64], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y, z = (v.tensor for v in graph.inputs)
        graph.output((x + y) * z)

    a = np.random.randn(12, 64).astype(np.float32)
    b = np.random.randn(12, 64).astype(np.float32)
    c = np.random.randn(12, 64).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a, b, c, fused=r"mo\.add.*mo\.mul"
    )
    np.testing.assert_allclose(out, (a + b) * c, rtol=1e-5, atol=1e-5)


def test_diamond_fuses(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`add(mul(t0, t0), relu(t0))` where `t0 = add(x, y)` fuses into one kernel.

    `t0` feeds both branches of the diamond, so the fused kernel computes it
    twice rather than materializing it -- the same duplication behavior
    `basic_pointwise_fusion.mlir`'s `fusion_diamond` case checks at the IR
    level. Uses a larger static rank-2 shape.
    """
    with Graph(
        "diamond",
        input_types=[
            TensorType(DType.float32, [32, 128], device=DeviceRef.CPU()),
            TensorType(DType.float32, [32, 128], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        t0 = x + y
        t1 = t0 * t0
        t2 = ops.relu(t0)
        graph.output(t1 + t2)

    a = np.random.randn(32, 128).astype(np.float32)
    b = np.random.randn(32, 128).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a, b, fused=r"mo\.relu.*mo\.mul"
    )
    t0_ref = a + b
    ref = t0_ref * t0_ref + np.maximum(t0_ref, 0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_duplicate_operand_chain_fuses(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`add(x, mul(add(x, x), x))` over a dynamic rank-1 shape fuses correctly.

    `x` feeds the fused kernel through several independent loads (mirrors
    `basic_pointwise_fusion.mlir`'s `add_mul_duplicate_operands`), rather than
    being loaded once and shared.
    """
    with Graph(
        "duplicate_operand_chain",
        input_types=[TensorType(DType.float32, ["n"], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        t0 = x + x
        t1 = t0 * x
        graph.output(x + t1)

    a = np.random.randn(257).astype(np.float32)
    # `x + x` canonicalizes to `x * 2` before fusion, so the fused sequence is
    # two multiplies (the canonicalized `t0` and `t1 = t0 * x`) then the final
    # add, not literally `add, mul, add`.
    (out,) = run_and_verify_fusion(
        session, graph, a, fused=r"mo\.mul.*mo\.mul.*mo\.add"
    )
    t0_ref = a + a
    ref = a + t0_ref * a
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_cast_in_chain_fuses(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A dtype-changing `mo.cast` inside an elementwise chain fuses in.

    `relu(cast(add(x, y), float64))` over a rank-2 shape with one static and
    one dynamic dimension -- the cast changes dtype mid-chain, and the fused
    kernel's output dtype tracks the cast rather than the inputs'.
    """
    with Graph(
        "cast_in_chain",
        input_types=[
            TensorType(DType.float32, [16, "n"], device=DeviceRef.CPU()),
            TensorType(DType.float32, [16, "n"], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(ops.relu((x + y).cast(DType.float64)))

    a = np.random.randn(16, 20).astype(np.float32)
    b = np.random.randn(16, 20).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a, b, fused=r"mo\.add.*mo\.cast.*mo\.relu"
    )
    assert out.dtype == np.float64
    ref = np.maximum((a + b).astype(np.float64), 0)
    np.testing.assert_allclose(out, ref, rtol=1e-9, atol=1e-9)
