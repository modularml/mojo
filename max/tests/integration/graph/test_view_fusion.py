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

"""End-to-end tests for the new MAP-dialect fusion system's view handling.

Every graph here is compiled with ``MAX_GC_USE_ADV_FUSION`` set, exercising
``MOToMAP``'s per-view-kind lowering (slice/broadcast/transpose/reshape) and
``ViewFuser`` end to end. `test_transpose_of_add_fuses_with_correct_index` in
particular is the Python-level regression test for the `ElementwiseFuser`
fix: a same-shape, non-identity view must be fused by `ViewFuser` (composing
index transforms correctly), never by `ElementwiseFuser` (which would splice
it in at the wrong coordinate). Whether the graph actually fused into one
kernel is covered by that MLIR test suite, not here -- these are e2e
correctness tests, checked against numpy as the oracle.
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_slice_static(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A standalone `mo.slice` with static bounds compiles and runs correctly."""
    with Graph(
        "slice_static",
        input_types=[TensorType(DType.float32, [8, 8], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(x[2:6, 2:6])

    a = np.random.randn(8, 8).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(out, a[2:6, 2:6], rtol=1e-5, atol=1e-5)


def test_slice_dynamic(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A `mo.slice` over a symbolic input shape compiles and runs correctly."""
    with Graph(
        "slice_dynamic",
        input_types=[
            TensorType(DType.float32, ["n", 8], device=DeviceRef.CPU())
        ],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(x[2:6, 2:6])

    a = np.random.randn(11, 8).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(out, a[2:6, 2:6], rtol=1e-5, atol=1e-5)


def test_broadcast_rank_preserving(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A rank-preserving `mo.static.broadcast_to` compiles and runs correctly."""
    with Graph(
        "broadcast_rank_preserving",
        input_types=[TensorType(DType.float32, [1, 4], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(x.broadcast_to([3, 4]))

    a = np.random.randn(1, 4).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(
        out, np.broadcast_to(a, (3, 4)), rtol=1e-5, atol=1e-5
    )


def test_broadcast_rank_expanding(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A rank-expanding `mo.static.broadcast_to` compiles and runs correctly."""
    with Graph(
        "broadcast_rank_expanding",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(x.broadcast_to([3, 4]))

    a = np.random.randn(4).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(
        out, np.broadcast_to(a, (3, 4)), rtol=1e-5, atol=1e-5
    )


def test_transpose_static(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A standalone `mo.transpose` (2D swap) compiles and runs correctly."""
    with Graph(
        "transpose_static",
        input_types=[TensorType(DType.float32, [2, 3], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(ops.transpose(x, 0, 1))

    a = np.random.randn(2, 3).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(out, a.T, rtol=1e-5, atol=1e-5)


def test_reshape_static(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A standalone `mo.static.reshape` (flatten) compiles and runs correctly."""
    with Graph(
        "reshape_static",
        input_types=[TensorType(DType.float32, [4, 6], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(x.reshape([24]))

    a = np.random.randn(4, 6).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(out, a.reshape([24]), rtol=1e-5, atol=1e-5)


def test_transpose_of_add_fuses_with_correct_index(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`transpose(add(x, y))` on a square matrix fuses via ViewFuser, correctly.

    Regression test for the `ElementwiseFuser` fix (commit "Reject a view as
    ElementwiseFuser's destination"): `x`/`y` are square, so `transpose`'s
    output shape matches `add`'s shape exactly. `ElementwiseFuser`'s old,
    shape-only compatibility check would have spliced `add` into `transpose`
    at the wrong (untransformed) coordinate, silently computing the wrong
    answer while leaving the transpose's own index math dead. If that bug
    ever came back, this would fail on the numeric check, not on a compile
    error.
    """
    with Graph(
        "transpose_of_add",
        input_types=[
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(ops.transpose(x + y, 0, 1))

    a = np.random.randn(3, 3).astype(np.float32)
    b = np.random.randn(3, 3).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a, b, fused=r"mo\.add.*mo\.transpose"
    )
    np.testing.assert_allclose(out, (a + b).T, rtol=1e-5, atol=1e-5)


def test_chained_slice_fuses(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A slice of a slice fuses into one kernel, composing against the first
    slice's view rather than the original tensor.
    """
    with Graph(
        "chained_slice",
        input_types=[
            TensorType(DType.float32, [10, 10], device=DeviceRef.CPU())
        ],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(x[1:9, 1:9][1:7, 1:7])

    a = np.random.randn(10, 10).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a, fused=r"mo\.slice.*mo\.slice"
    )
    np.testing.assert_allclose(out, a[1:9, 1:9][1:7, 1:7], rtol=1e-5, atol=1e-5)


def test_view_fuses_into_existing_prologue(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A slice fuses into `mo.gather`'s existing prologue capture.

    Regression test for the `LoopInvariantViewMotion` fix that let it hoist
    an index-transform chain out of a `map.iter.prologue`, not just a
    `map.iter.foreach.body` (see
    GraphCompiler/test/mo-opt/MAPDialect/Transforms/LoopInvariantViewMotion/
    loop-invariant-view-motion-prologue.mlir), and for the `EmitMojo` fix
    letting a fused prologue/epilogue capture a view (not just a plain
    materialized tensor) -- `collectFusedCaptureInfos`
    (GraphCompiler/lib/MOGGDialect/MOGGOps.cpp) now reads a capture's layout
    off its own already-emitted `ManagedTensorSlice.to_tile_tensor()` instead
    of assuming it's always a plain `mogg._tensor.create` result.
    """
    with Graph(
        "view_into_prologue",
        input_types=[
            TensorType(DType.float32, [10], device=DeviceRef.CPU()),
            TensorType(DType.int64, [3], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, indices = (v.tensor for v in graph.inputs)
        graph.output(ops.gather(x[2:8], indices, axis=0))

    a = np.random.randn(10).astype(np.float32)
    idx = np.array([0, 3, 5], dtype=np.int64)
    (out,) = run_and_verify_fusion(
        session, graph, a, idx, fused=r"mo\.slice.*mo\.gather"
    )
    np.testing.assert_allclose(out, a[2:8][idx], rtol=1e-5, atol=1e-5)


def test_broadcast_fuses_into_existing_epilogue(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A broadcast bias fuses into `mo.reduce.max`'s existing epilogue capture.

    Regression test for `ViewFuser`'s epilogue generalization (mirrors
    `broadcastIntoExistingEpilogue` in
    GraphCompiler/test/mo-opt/MAPDialect/Transforms/ViewFusion/
    view_fusion_into_epilogue.mlir) and for the `EmitMojo` fused-capture fix
    (`collectFusedCaptureInfos` in GraphCompiler/lib/MOGGDialect/MOGGOps.cpp)
    reading a capture's layout off its own `ManagedTensorSlice.to_tile_tensor()`
    instead of assuming every capture is a plain `mogg._tensor.create` result:
    `EpilogueFuser` first fuses the bias-add into the reduction's epilogue,
    capturing the broadcast's own result; `ViewFuser` then splices the
    broadcast itself into that same capture, since a transform op's
    `ParentOneOf` now permits living inside an epilogue.

    Uses `ops.max` rather than `ops.matmul` deliberately: a matmul-fused
    epilogue binds through `_bind_to_fused_compute_output` (a "compute
    lambda"), a separate, unrelated code path with its own currently-open gap
    in `EmitMojo`'s io_spec handling -- `ops.max`'s epilogue binds through the
    plain `_bind_to_fused_output` ("store lambda") path this fix actually
    covers, keeping this test isolated to what `ViewFuser`'s generalization is
    responsible for.
    """
    with Graph(
        "broadcast_into_epilogue",
        input_types=[
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, bias = (v.tensor for v in graph.inputs)
        reduced = ops.max(x, axis=0)
        graph.output(reduced + bias.broadcast_to([1, 4]))

    x_np = np.random.randn(4, 4).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, x_np, bias_np, fused=r"mo\.reduce\.max.*mo\.add"
    )
    ref = np.max(x_np, axis=0, keepdims=True) + bias_np
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_view_fanout_duplicates(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A view feeding two different elementwise consumers fuses into both.

    Unlike `ElementwiseFuser`'s cost-gated duplication, `ViewFuser` always
    considers a view cheap to duplicate.
    """
    with Graph(
        "view_fanout",
        input_types=[TensorType(DType.float32, [8], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        y = x[1:5]
        graph.output(ops.relu(y) + ops.negate(y))

    a = np.random.randn(8).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a, fused=r"mo\.slice.*mo\.relu"
    )
    y_ref = a[1:5]
    ref = np.maximum(y_ref, 0) + (-y_ref)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_reshape_producer_fuses_downstream(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A reshape feeding an elementwise consumer fuses in as the producer.

    Reshape is always the terminal (most-upstream) link in a fusion chain --
    nothing fuses into a reshape kernel -- but it may itself fuse into
    whatever consumes its result.
    """
    with Graph(
        "reshape_producer",
        input_types=[TensorType(DType.float32, [4, 6], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(ops.relu(x.reshape([24])))

    a = np.random.randn(4, 6).astype(np.float32)
    # An empty pattern here: a reshape is a pure index transform, so a fused
    # reshape leaves no leaf of its own in the summary -- there is nothing
    # distinguishing this from an unfused reshape to match on.
    (out,) = run_and_verify_fusion(session, graph, a)
    np.testing.assert_allclose(
        out, np.maximum(a.reshape([24]), 0), rtol=1e-5, atol=1e-5
    )
