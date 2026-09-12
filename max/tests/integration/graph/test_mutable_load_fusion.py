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

"""End-to-end tests for `FuseMutableLoads`-shaped graphs.

These mirror `GraphCompiler/test/mo-opt/MAPDialect/Transforms/FuseMutableLoads/`'s
cases with real ops rather than the MLIR suite's `sampler.apply_penalties`
opaque kernel, so each case here also proves numeric correctness on the
mutated buffer, not just IR shape. The two load-only shapes fuse under both
pipelines; the four mutable-store write-back shapes don't yet fuse the store
into the producing kernel under the new MAP-dialect system (a Mojo-codegen gap
for a load fused via a new chain for the elementwise cases; `GEX-3964` for the
reduce case, where `EpilogueFuser` declines the store because
`map.iter.opaque` carries neither an out chain nor memory-effect fields), so
those four are marked `@xfail_under_adv_fusion` individually.

Every case in this file uses ``ops.buffer_load``/``ops.buffer_store`` around a
plain elementwise/view op -- `mo.add`, `mo.negative`, `mo.abs`, a static
slice -- since a load/store fuses into whatever kernel already exists
downstream, regardless of which op it is.

Not ported here (see the trailing comment): every case built around
`sampler.apply_penalties`, the MLIR suite's stand-in for an in-place opaque
kernel (buffer read *and* written directly, not through a load/store pair) --
no real op or existing test-only kernel has that exact shape.
"""

from __future__ import annotations

import re

import numpy as np
from fusion_utils import xfail_under_adv_fusion
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def _fused(model_summaries: list[str], pattern: str) -> bool:
    return any(re.search(pattern, s) for s in model_summaries)


@xfail_under_adv_fusion(
    "the new fusion system does not fuse the mutable store into the add kernel"
)
def test_add_buffer_lhs_fuses(session: InferenceSession) -> None:
    """`buffer_store(buf, buffer_load(buf) + rhs)` fuses the load into the
    add kernel, which gains a chain in/out pair it didn't have before -- the
    kernel's own tensor operand slot 0 becomes a buffer slot instead. Mirrors
    `single_add_buffer_lhs`.
    """
    with Graph(
        "add_buffer_lhs_fuses",
        input_types=[
            BufferType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        buf = graph.inputs[0].buffer
        rhs = graph.inputs[1].tensor
        ops.buffer_store(buf, ops.buffer_load(buf) + rhs)
        graph.output()

    model = session.load(graph)
    assert _fused(
        model.kernel_summaries,
        r"mo\.mutable\.load.*mo\.add.*mo\.mutable\.store",
    ), model.kernel_summaries

    buf_np = np.random.randn(4, 4).astype(np.float32)
    rhs_np = np.random.randn(4, 4).astype(np.float32)
    expected = buf_np + rhs_np
    buf_input = Buffer.from_numpy(buf_np)
    model.execute(buf_input, Buffer.from_numpy(rhs_np))
    np.testing.assert_allclose(
        buf_input.to_numpy(), expected, rtol=1e-5, atol=1e-5
    )


@xfail_under_adv_fusion(
    "the new fusion system does not fuse the mutable store into the add kernel"
)
def test_add_buffer_rhs_fuses(session: InferenceSession) -> None:
    """Same as `test_add_buffer_lhs_fuses`, but the buffer sits in the add's
    second operand slot -- the fused kernel's replacement index is computed
    from the load's own use, not assumed to be 0. Mirrors
    `single_add_buffer_rhs`.
    """
    with Graph(
        "add_buffer_rhs_fuses",
        input_types=[
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            BufferType(DType.float32, [4, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        lhs = graph.inputs[0].tensor
        buf = graph.inputs[1].buffer
        ops.buffer_store(buf, lhs + ops.buffer_load(buf))
        graph.output()

    model = session.load(graph)
    assert _fused(
        model.kernel_summaries,
        r"mo\.mutable\.load.*mo\.add.*mo\.mutable\.store",
    ), model.kernel_summaries

    lhs_np = np.random.randn(4, 4).astype(np.float32)
    buf_np = np.random.randn(4, 4).astype(np.float32)
    expected = lhs_np + buf_np
    buf_input = Buffer.from_numpy(buf_np)
    model.execute(Buffer.from_numpy(lhs_np), buf_input)
    np.testing.assert_allclose(
        buf_input.to_numpy(), expected, rtol=1e-5, atol=1e-5
    )


@xfail_under_adv_fusion(
    "the new fusion system does not fuse the mutable store into the add kernel"
)
def test_two_buffers_fuse_into_same_kernel(session: InferenceSession) -> None:
    """`buffer_store(buf2, buffer_load(buf1) + buffer_load(buf2))`: two
    independent loads fuse into the same add+store kernel. Mirrors
    `same_kernel_multiple_fusions`.
    """
    with Graph(
        "two_buffers_fuse_into_same_kernel",
        input_types=[
            BufferType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            BufferType(DType.float32, [4, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        buf1 = graph.inputs[0].buffer
        buf2 = graph.inputs[1].buffer
        ops.buffer_store(buf2, ops.buffer_load(buf1) + ops.buffer_load(buf2))
        graph.output()

    model = session.load(graph)
    assert _fused(
        model.kernel_summaries,
        r"mo\.mutable\.load.*mo\.mutable\.load.*mo\.add.*mo\.mutable\.store",
    ), model.kernel_summaries

    buf1_np = np.random.randn(4, 4).astype(np.float32)
    buf2_np = np.random.randn(4, 4).astype(np.float32)
    expected = buf1_np + buf2_np
    buf1_input = Buffer.from_numpy(buf1_np)
    buf2_input = Buffer.from_numpy(buf2_np)
    model.execute(buf1_input, buf2_input)
    np.testing.assert_allclose(
        buf2_input.to_numpy(), expected, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        buf1_input.to_numpy(), buf1_np, rtol=1e-5, atol=1e-5
    )


def test_multiple_pure_reads_parallelize(session: InferenceSession) -> None:
    """A buffer read by three independent, pure (non-writing) consumers gets
    an independent, parallelized load fused into each -- a read-after-read
    has no ordering hazard, so `ParallelizeLoadsWithMultipleUsers` splits the
    one shared load into three. Mirrors `multiple_load_fusion`.
    """
    with Graph(
        "multiple_pure_reads_parallelize",
        input_types=[BufferType(DType.float32, [2, 2], device=DeviceRef.CPU())],
    ) as graph:
        buf = graph.inputs[0].buffer
        a = ops.buffer_load(buf)
        b = ops.buffer_load(buf)
        c = ops.buffer_load(buf)
        graph.output(-a, ops.abs(b), c + c)

    buf_np = np.random.randn(2, 2).astype(np.float32)
    model = session.load(graph)
    assert _fused(model.kernel_summaries, r"mo\.mutable\.load.*mo\.negative"), (
        model.kernel_summaries
    )
    assert _fused(model.kernel_summaries, r"mo\.mutable\.load.*mo\.abs"), (
        model.kernel_summaries
    )
    # Legacy's own graph-build canonicalizer rewrites `c + c` into `c * 2`
    # (a broadcast + mul) before fusion runs, unlike the MLIR test this
    # mirrors, which feeds `mo.add` straight to `FuseMutableLoads` without
    # going through that pass -- the load still fuses into whatever
    # downstream kernel exists, which is all this case actually checks.
    assert _fused(model.kernel_summaries, r"mo\.mutable\.load.*mo\.mul"), (
        model.kernel_summaries
    )

    out1, out2, out3 = model.execute(Buffer.from_numpy(buf_np))
    np.testing.assert_allclose(out1.to_numpy(), -buf_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out2.to_numpy(), np.abs(buf_np), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        out3.to_numpy(), buf_np + buf_np, rtol=1e-5, atol=1e-5
    )


def test_slice_producer_fuses(session: InferenceSession) -> None:
    """A static slice of a loaded buffer fuses the load in, reading straight
    off the buffer through the slice's own index math. Mirrors
    `fusion_with_slice`.
    """
    with Graph(
        "slice_producer_fuses",
        input_types=[
            BufferType(DType.float32, [10, 10], device=DeviceRef.CPU())
        ],
    ) as graph:
        buf = graph.inputs[0].buffer
        x = ops.buffer_load(buf)
        graph.output(-x[2:8:2, 2:8:2])

    buf_np = np.random.randn(10, 10).astype(np.float32)
    model = session.load(graph)
    assert _fused(
        model.kernel_summaries, r"mo\.mutable\.load.*mo\.slice.*mo\.negative"
    ), model.kernel_summaries

    (out,) = model.execute(Buffer.from_numpy(buf_np))
    np.testing.assert_allclose(
        out.to_numpy(), -buf_np[2:8:2, 2:8:2], rtol=1e-5, atol=1e-5
    )


@xfail_under_adv_fusion(
    "GEX-3964: EpilogueFuser declines the mutable store because "
    "map.iter.opaque carries no out chain or memory-effect fields"
)
def test_mutable_store_consumer_fuses(session: InferenceSession) -> None:
    """`buffer_store(buf, reduce.max(x, axis))` fuses the store into the
    reduction's epilogue, writing the buffer directly rather than
    materializing an intermediate tensor. Mirrors
    `mutable_store_consumer_not_fused`.

    `EpilogueFuser` (the new system) declines this case outright (GEX-3964, a
    MOGG parity gap -- `map.iter.opaque` carries neither an out chain nor
    memory-effect fields to fuse a store soundly), so it fuses only under the
    legacy pipeline and is xfail'd under the new one.
    """
    with Graph(
        "mutable_store_consumer_fuses",
        input_types=[
            TensorType(DType.float32, [4, 4], device=DeviceRef.CPU()),
            BufferType(DType.float32, [1, 4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x = graph.inputs[0].tensor
        buf = graph.inputs[1].buffer
        ops.buffer_store(buf, ops.max(x, axis=0))
        graph.output()

    model = session.load(graph)
    assert _fused(
        model.kernel_summaries, r"mo\.reduce\.max.*mo\.mutable\.store"
    ), model.kernel_summaries

    x_np = np.random.randn(4, 4).astype(np.float32)
    buf_input = Buffer.from_numpy(np.zeros((1, 4), dtype=np.float32))
    model.execute(Buffer.from_numpy(x_np), buf_input)
    np.testing.assert_allclose(
        buf_input.to_numpy(),
        np.max(x_np, axis=0, keepdims=True),
        rtol=1e-5,
        atol=1e-5,
    )
