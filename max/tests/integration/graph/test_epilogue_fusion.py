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

"""End-to-end tests for the new MAP-dialect fusion system's `EpilogueFuser`.

Every graph here is compiled with ``MAX_GC_USE_ADV_FUSION`` set, exercising
``MOToMAP`` + ``EpilogueFuser`` end to end. These mirror
`GraphCompiler/test/mo-opt/MAPDialect/Transforms/EpilogueFusion/epilogue_fusion.mlir`'s
cases with real ops rather than the MLIR suite's signature-only test kernels,
so each case here also proves numeric correctness, not just IR shape. Whether
a given graph actually fuses is covered by that MLIR suite, not here.

`ops.max` (`mo.reduce.max`) stands in for the MLIR suite's
`map_lower_one_fused_output` wherever a plain fused-output producer is
needed: its epilogue binds through the store-lambda path (`FusedOutputTensor`
+ `_bind_to_fused_output`).

Several of that suite's cases are NOT ported here (see the trailing
comment): `matmul_add`/`chain` because `mo.matmul`'s same-dtype (compute
lambda) epilogue currently fails to compile under `MAX_GC_USE_ADV_FUSION=1`
(a separate, already-tracked compute-lambda gap -- see
`test_view_fusion.py`'s `test_broadcast_fuses_into_existing_epilogue`
docstring). A dtype-CHANGING matmul epilogue does compile now -- the
mixed-precision accumulating path (`AllocateAccumulatingBuffers`) gives it a
pre-cast scratch buffer -- so `cast(matmul(...))` is exercised below; `fuse_in_if`/`no_fuse_across_blocks` because of a newly-found,
currently-unreported gap: any epilogue-fusible op inside an `ops.cond`
(`mo.if`) branch fails the same way, with `'map.index.reshape' op MAPToMOGG:
unsupported op` -- reproduces even with both branches already the same
shape, so it isn't a shape-mismatch artifact of a particular test; ruled out
against the prologue side, which compiles and runs correctly inside `mo.if`
branches (see `test_prologue_fusion.py`'s
`test_fuses_independently_per_conditional_branch`). A couple of the
remaining cases have no real-op counterpart at all and are intentionally not
ported either (see the trailing comment).
"""

from __future__ import annotations

import numpy as np
from fusion_utils import run_and_verify_fusion
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def test_reduce_max_epilogue_fuses_cast(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`cast(reduce.max(x, axis))` fuses the cast into the reduction's
    epilogue, retyping the store path. Mirrors `cast_fused`.
    """
    with Graph(
        "reduce_max_epilogue_fuses_cast",
        input_types=[TensorType(DType.float32, [4, 4], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(ops.max(x, axis=0).cast(DType.int32))

    x_np = np.random.randn(4, 4).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, x_np, fused=r"mo\.reduce\.max.*mo\.cast"
    )
    assert out.dtype == np.int32
    ref = np.max(x_np, axis=0, keepdims=True).astype(np.int32)
    np.testing.assert_allclose(out, ref)


def test_matmul_epilogue_fuses_cast(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`cast(matmul(a, b))` fuses the cast into the matmul's epilogue. A
    dtype-changing (mixed-precision) accumulating kernel: the matmul must
    accumulate in f32, so `AllocateAccumulatingBuffers` gives the epilogue a
    pre-cast f32 scratch buffer to accumulate through while the store lands in
    the si32 output. Without it the kernel would accumulate f32 through the si32
    buffer. Mirrors `matmul-cast-epilogue-fusion.mlir`.
    """
    with Graph(
        "matmul_epilogue_fuses_cast",
        input_types=[
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(ops.matmul(a, b).cast(DType.int32))

    a_np = np.random.randn(2, 2).astype(np.float32)
    b_np = np.random.randn(2, 2).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.matmul.*mo\.cast"
    )
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, (a_np @ b_np).astype(np.int32))


def test_no_fuse_multi_use(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """A producer consumed by two separate downstream ops can only fuse into
    one of them at most -- and today fuses into neither, per `EpilogueFuser`'s
    single-claim-per-round `hasOneUse` gate. Mirrors `no_fuse_multi_use`:
    `reduce.max`'s result feeds both an `add` and a `mul`.
    """
    with Graph(
        "no_fuse_multi_use",
        input_types=[
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        reduced = ops.max(a, axis=0)
        graph.output(reduced + b, reduced * b)

    a_np = np.random.randn(4, 4).astype(np.float32)
    b_np = np.random.randn(4).astype(np.float32)
    r0, r1 = run_and_verify_fusion(session, graph, a_np, b_np)
    reduced_np = np.max(a_np, axis=0, keepdims=True)
    np.testing.assert_allclose(r0, reduced_np + b_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(r1, reduced_np * b_np, rtol=1e-5, atol=1e-5)


def test_no_fuse_same_result_twice(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`add(reduce.max(x, axis), reduce.max(x, axis))` -- the SAME producer
    feeding both operands of its own consumer -- has no single fusible
    operand to redirect, so it stays unfused. Mirrors `no_fuse_same_result_twice`.
    """
    with Graph(
        "no_fuse_same_result_twice",
        input_types=[TensorType(DType.float32, [4, 4], device=DeviceRef.CPU())],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        reduced = ops.max(x, axis=0)
        graph.output(reduced + reduced)

    x_np = np.random.randn(4, 4).astype(np.float32)
    (out,) = run_and_verify_fusion(session, graph, x_np)
    reduced_np = np.max(x_np, axis=0, keepdims=True)
    np.testing.assert_allclose(
        out, reduced_np + reduced_np, rtol=1e-5, atol=1e-5
    )


def test_two_producers_one_fuses(
    session: InferenceSession, adv_fusion_enabled: None
) -> None:
    """`add(reduce.max(a, axis), reduce.max(b, axis))` -- two independent
    producers competing for the same consumer -- fuses exactly one of them;
    the other stays its own kernel. Mirrors `two_producers_one_fuses`.
    """
    with Graph(
        "two_producers_one_fuses",
        input_types=[
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(ops.max(a, axis=0) + ops.max(b, axis=0))

    a_np = np.random.randn(4, 4).astype(np.float32)
    b_np = np.random.randn(4, 4).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.reduce\.max.*mo\.add"
    )
    ref = np.max(a_np, axis=0, keepdims=True) + np.max(
        b_np, axis=0, keepdims=True
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# NOTE: `epilogue_fusion.mlir`'s `no_fuse_plain_output` and
# `cycle_fuses_other_producer` cases are deliberately not ported here. Both
# exercise signature-only, no-computation test kernels
# (`map_lower_one_plain_output`, `map_lower_two_fused_outputs`) built to probe
# eligibility metadata a real op doesn't let us select independently of its
# actual behavior (a kernel with a non-fusable output; a two-output kernel
# whose second output could complete a cycle) -- there is nothing for an e2e
# numeric test to check beyond what the MLIR suite already proves.
