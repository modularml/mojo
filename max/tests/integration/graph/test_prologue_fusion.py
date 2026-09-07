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

"""End-to-end tests for the new MAP-dialect fusion system's `PrologueFuser`.

Every graph here is compiled with ``MAX_GC_USE_ADV_FUSION`` set, exercising
``MOToMAP`` + ``PrologueFuser`` end to end. These mirror
`GraphCompiler/test/mo-opt/MAPDialect/Transforms/PrologueFusion/prologue_fusion.mlir`'s
cases with real ops rather than the MLIR suite's signature-only test kernels,
so each case here also proves numeric correctness, not just IR shape. Whether
a given graph actually fuses is covered by that MLIR suite, not here.

A few of that suite's cases have no real-op counterpart and are intentionally
not ported (see the trailing comment): they exist purely to probe fusion
*eligibility* through a signature-only (no computation) test kernel, so there
is nothing for a numeric e2e test to check beyond what the MLIR suite already
proves.
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_add_fuses_into_gather_data(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`gather(add(a, b), indices)` fuses the add into `gather`'s data capture.

    Mirrors `add_gather` in prologue_fusion.mlir.
    """
    with Graph(
        "add_fuses_into_gather_data",
        input_types=[
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, indices = (v.tensor for v in graph.inputs)
        data = a + b
        graph.output(ops.gather(data, indices, axis=0))

    a_np = np.random.randn(3, 3).astype(np.float32)
    b_np = np.random.randn(3, 3).astype(np.float32)
    idx_np = np.array([[0, 2], [1, 0]], dtype=np.int32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, idx_np, fused=r"mo\.add.*mo\.gather"
    )
    ref = (a_np + b_np)[idx_np]
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_add_fuses_into_gather_indices(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`gather(data, add(i0, i1))` fuses the add into `gather`'s indices capture.

    Mirrors `add_gather_indices` in prologue_fusion.mlir -- a *second*,
    independent prologue capture on the same opaque kernel.
    """
    with Graph(
        "add_fuses_into_gather_indices",
        input_types=[
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        data, i0, i1 = (v.tensor for v in graph.inputs)
        indices = i0 + i1
        graph.output(ops.gather(data, indices, axis=0))

    data_np = np.random.randn(3, 3).astype(np.float32)
    i0_np = np.array([[0, 1], [0, 1]], dtype=np.int32)
    i1_np = np.array([[0, 1], [1, 0]], dtype=np.int32)
    (out,) = run_and_verify_fusion(
        session, graph, data_np, i0_np, i1_np, fused=r"mo\.add.*mo\.gather"
    )
    ref = data_np[i0_np + i1_np]
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_chain_fuses_into_prologue(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`softmax(cast(add(a, a)))` fuses the whole elementwise chain into one
    prologue.

    Mirrors `chain` in prologue_fusion.mlir: a dtype-changing `mo.cast`
    mid-chain, with `PrologueFuser` absorbing an already-`ElementwiseFuser`-
    fused chain as a single unit.
    """
    with Graph(
        "chain_fuses_into_prologue",
        input_types=[TensorType(DType.float32, [2, 3], device=DeviceRef.CPU())],
    ) as graph:
        (a,) = (v.tensor for v in graph.inputs)
        chained = (a + a).cast(DType.float64)
        graph.output(ops.softmax(chained, axis=1))

    a_np = np.random.randn(2, 3).astype(np.float32)
    # `a + a` canonicalizes to `a * 2` before fusion, so the fused sequence
    # has a mul (not an add) ahead of the cast.
    (out,) = run_and_verify_fusion(
        session, graph, a_np, fused=r"mo\.mul.*mo\.cast.*mo\.reduce\.softmax"
    )

    doubled = (a_np + a_np).astype(np.float64)
    exp = np.exp(doubled - np.max(doubled, axis=1, keepdims=True))
    ref = exp / np.sum(exp, axis=1, keepdims=True)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_fuses_independently_per_conditional_branch(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """Each `mo.if` branch gets its own, independently-fused prologue.

    Mirrors `fuse_in_if`: the then-branch computes `gather(add(a, b),
    indices)`, the else-branch `gather(mul(a, b), indices)` -- same shapes,
    different producer op per branch.
    """
    with Graph(
        "fuses_independently_per_conditional_branch",
        input_types=[
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.bool, [], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, indices, cond = (v.tensor for v in graph.inputs)
        out_type = TensorType(DType.float32, [2, 2, 3], device=DeviceRef.CPU())

        def then_fn():  # noqa: ANN202
            return ops.gather(a + b, indices, axis=0)

        def else_fn():  # noqa: ANN202
            return ops.gather(a * b, indices, axis=0)

        (result,) = ops.cond(cond, [out_type], then_fn, else_fn)
        graph.output(result)

    a_np = np.random.randn(3, 3).astype(np.float32)
    b_np = np.random.randn(3, 3).astype(np.float32)
    idx_np = np.array([[0, 2], [1, 0]], dtype=np.int32)

    # Both branches compile regardless of which one a given call executes, so
    # the fusion check only needs to run once, against either branch's
    # pattern; the loop below is purely for the numeric check per branch.
    (out,) = run_and_verify_fusion(
        session,
        graph,
        a_np,
        b_np,
        idx_np,
        np.array(True),
        fused=r"mo\.add.*mo\.gather",
    )
    np.testing.assert_allclose(out, (a_np + b_np)[idx_np], rtol=1e-5, atol=1e-5)

    for cond_np, ref_fn in (
        (np.array(True), lambda: (a_np + b_np)[idx_np]),
        (np.array(False), lambda: (a_np * b_np)[idx_np]),
    ):
        (out,) = run_and_verify_fusion(
            session, graph, a_np, b_np, idx_np, cond_np
        )
        np.testing.assert_allclose(out, ref_fn(), rtol=1e-5, atol=1e-5)


def test_no_fuse_across_conditional_blocks(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A producer defined *outside* an `mo.if` cannot fuse into a gather
    inside one of its branches -- `PrologueFuser` never crosses block
    boundaries. Mirrors `no_fuse_across_blocks`.
    """
    with Graph(
        "no_fuse_across_conditional_blocks",
        input_types=[
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.bool, [], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, indices, cond = (v.tensor for v in graph.inputs)
        data = a + b
        out_type = TensorType(DType.float32, [2, 2, 3], device=DeviceRef.CPU())

        def then_fn():  # noqa: ANN202
            return ops.gather(data, indices, axis=0)

        def else_fn():  # noqa: ANN202
            return ops.gather(a, indices, axis=0)

        (result,) = ops.cond(cond, [out_type], then_fn, else_fn)
        graph.output(result)

    a_np = np.random.randn(3, 3).astype(np.float32)
    b_np = np.random.randn(3, 3).astype(np.float32)
    idx_np = np.array([[0, 2], [1, 0]], dtype=np.int32)

    for cond_np, ref_fn in (
        (np.array(True), lambda: (a_np + b_np)[idx_np]),
        (np.array(False), lambda: a_np[idx_np]),
    ):
        (out,) = run_and_verify_fusion(
            session, graph, a_np, b_np, idx_np, cond_np
        )
        np.testing.assert_allclose(out, ref_fn(), rtol=1e-5, atol=1e-5)


def test_no_fuse_multi_use(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A producer with a second use outside the consumer it would fuse into
    keeps computing standalone -- `PrologueFuser`'s `hasOneUse()` guard.
    Mirrors `no_fuse_multi_use`: `data` feeds both `gather` and the graph's
    own second output.
    """
    with Graph(
        "no_fuse_multi_use",
        input_types=[
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [3, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [2, 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, indices = (v.tensor for v in graph.inputs)
        data = a + b
        graph.output(ops.gather(data, indices, axis=0), data)

    a_np = np.random.randn(3, 3).astype(np.float32)
    b_np = np.random.randn(3, 3).astype(np.float32)
    idx_np = np.array([[0, 2], [1, 0]], dtype=np.int32)
    out, data_out = run_and_verify_fusion(session, graph, a_np, b_np, idx_np)
    data_ref = a_np + b_np
    np.testing.assert_allclose(out, data_ref[idx_np], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(data_out, data_ref, rtol=1e-5, atol=1e-5)


# NOTE: `prologue_fusion.mlir`'s `no_fuse_plain_input` (matmul never gains a
# prologue capture) is deliberately not ported here either: ANY `mo.matmul`
# -- regardless of whether anything fuses into it at all -- currently fails
# to compile under `MAX_GC_USE_ADV_FUSION=1` (a separate, already-tracked
# compute-lambda gap; see `test_view_fusion.py`'s
# `test_broadcast_fuses_into_existing_epilogue` docstring), so there is no
# real-op graph involving `ops.matmul` this suite can exercise today.
#
# `prologue_fusion.mlir`'s `two_fused_inputs`, `partial_fused_inputs`,
# and `no_fuse_same_producer_twice` cases are deliberately not ported here.
# Each exercises a signature-only, no-computation test kernel
# (`map_lower_two_fused_inputs` / `map_lower_partial_fused_inputs`) that
# exists purely to probe which of a multi-input custom kernel's operands are
# prologue-eligible -- there is no real MO op with two independently
# prologue-fusible tensor operands of matching type, and a `pass`-bodied
# kernel computes nothing an e2e numeric test could check. That structural
# question is already answered exhaustively by the MLIR suite.
