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

from layout import Layout, LayoutTensor, RuntimeLayout
from layout._fillers import random
from state_space.ssd_chunk import (
    ssd_intra_chunk_fwd_cpu,
    ssd_intra_chunk_fwd_cpu_naive,
)
from std.math import exp
from std.testing import TestSuite, assert_almost_equal

from std.utils.index import Index


def run_ssd_intra_chunk[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    rtol: Float64 = 0.01,
) raises:
    """Test the optimized SSD intra-chunk CPU kernel against an independent reference.
    """
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    # C: [batch, n_chunks, n_heads, chunk_len, state_dim]
    var C_heap = List(length=cb_count, fill=Scalar[dtype](0))
    var C_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )

    # B: [batch, n_chunks, n_heads, chunk_len, state_dim]
    var B_heap = List(length=cb_count, fill=Scalar[dtype](0))
    var B_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )

    # X: [batch, n_chunks, n_heads, chunk_len, head_dim]
    var X_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var X_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # A: [batch, n_chunks, n_heads, chunk_len]
    var A_heap = List(length=a_count, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )

    # Y: [batch, n_chunks, n_heads, chunk_len, head_dim]
    var Y_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var Y_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # Initialize inputs.
    random(C_h)
    random(B_h)
    random(X_h)
    random(A_h)

    # Make A negative-ish (a real SSM decays), so exp(cumsum diffs) stays
    # bounded. `random` fills in roughly [-1, 1]; map it into a small negative
    # range like real Mamba2 dt*A values.
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        # Map [-1, 1] -> [-0.6, -0.1].
        var scaled = Float32(-0.35) + Float32(0.25) * v
        A_h.ptr[i] = Scalar[dtype](scaled)

    # Call the optimized kernel.
    ssd_intra_chunk_fwd_cpu[
        dtype,
        C_h.layout,
        B_h.layout,
        X_h.layout,
        A_h.layout,
        Y_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        state_dim,
        head_dim,
        C_h,
        B_h,
        X_h,
        A_h,
        Y_h,
        ctx=None,
    )

    # Independent naive reference. Recompute strides from scratch and use a
    # straightforward triple-loop of the formula:
    #   A_cumsum[l] = sum_{k<=l} A[k]
    #   Ldecay[l,s] = exp(A_cumsum[l] - A_cumsum[s])  for s <= l
    #   scores[l,s] = (sum_n C[l,n]*B[s,n]) * Ldecay[l,s]
    #   Y[l,p]      = sum_{s<=l} scores[l,s] * X[s,p]
    var ref_heap = List(length=xy_count, fill=Scalar[dtype](0))

    var cb_h_stride = chunk_len * state_dim
    var cb_c_stride = n_heads * cb_h_stride
    var cb_b_stride = n_chunks * cb_c_stride

    var xy_h_stride = chunk_len * head_dim
    var xy_c_stride = n_heads * xy_h_stride
    var xy_b_stride = n_chunks * xy_c_stride

    var a_h_stride = chunk_len
    var a_c_stride = n_heads * a_h_stride
    var a_b_stride = n_chunks * a_c_stride

    for b in range(batch):
        for c in range(n_chunks):
            for h in range(n_heads):
                var cb_base = (
                    b * cb_b_stride + c * cb_c_stride + h * cb_h_stride
                )
                var xy_base = (
                    b * xy_b_stride + c * xy_c_stride + h * xy_h_stride
                )
                var a_base = b * a_b_stride + c * a_c_stride + h * a_h_stride

                # Prefix sum of A over the chunk.
                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc

                for l in range(chunk_len):
                    for p in range(head_dim):
                        var y_acc = Float32(0.0)
                        for s in range(l + 1):
                            # dot(C[l,:], B[s,:]).
                            var dot = Float32(0.0)
                            for n in range(state_dim):
                                var cv = C_h.ptr[
                                    cb_base + l * state_dim + n
                                ].cast[DType.float32]()
                                var bv = B_h.ptr[
                                    cb_base + s * state_dim + n
                                ].cast[DType.float32]()
                                dot += cv * bv
                            var decay = exp(cumsum[l] - cumsum[s])
                            var xv = X_h.ptr[xy_base + s * head_dim + p].cast[
                                DType.float32
                            ]()
                            y_acc += dot * decay * xv
                        ref_heap[xy_base + l * head_dim + p] = Scalar[dtype](
                            y_acc
                        )

    # Compare kernel output against the reference.
    for i in range(xy_count):
        assert_almost_equal(
            Y_h.ptr[i],
            ref_heap[i],
            rtol=rtol,
        )


def test_ssd_intra_chunk_golden() raises:
    """Golden-parity test against the canonical PyTorch SSD reference.

    Pins ``ssd_intra_chunk_fwd_cpu_naive`` to a FIXED tiny input whose expected
    output was computed independently by ``intra_chunk_diag`` in
    ``.planning/parity/ssd_minimal_ref.py`` (the vendored pure-PyTorch
    ``ssd_minimal_discrete`` stage-1 diagonal block).

    Shapes: batch=1, n_chunks=1, n_heads=1, chunk_len L=4, state_dim N=3,
    head_dim P=2; all tensors row-major.
    """
    comptime dtype = DType.float32
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    comptime batch = 1
    comptime n_chunks = 1
    comptime n_heads = 1
    comptime chunk_len = 4
    comptime state_dim = 3
    comptime head_dim = 2

    # Fixed literals (row-major). These mirror the values in the parity
    # reference; Y_expected is the reference's output for these inputs.
    var C_vals = [
        Float32(-0.15),
        0.79,
        0.95,
        -1.11,
        1.69,
        -0.89,
        -0.36,
        1.23,
        0.14,
        -1.68,
        0.32,
        0.13,
    ]
    var B_vals = [
        Float32(0.14),
        0.24,
        1.40,
        1.35,
        2.44,
        0.20,
        2.45,
        2.03,
        1.78,
        -0.92,
        -0.46,
        -0.72,
    ]
    var X_vals = [
        Float32(1.28),
        -0.99,
        1.81,
        -0.60,
        1.61,
        1.93,
        -0.42,
        -0.08,
    ]
    var A_vals = [Float32(-0.43), -0.34, -0.49, -0.46]
    var Y_expected = [
        Float32(1.918208),
        -1.483614,
        3.522012,
        -0.766567,
        6.067267,
        2.472605,
        -4.850489,
        -3.713203,
    ]

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    # Build LayoutTensors from the fixed literals (not random).
    var C_heap = List(length=cb_count, fill=Scalar[dtype](0))
    var C_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    for i in range(cb_count):
        C_h.ptr[i] = Scalar[dtype](C_vals[i])

    var B_heap = List(length=cb_count, fill=Scalar[dtype](0))
    var B_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    for i in range(cb_count):
        B_h.ptr[i] = Scalar[dtype](B_vals[i])

    var X_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var X_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    for i in range(xy_count):
        X_h.ptr[i] = Scalar[dtype](X_vals[i])

    var A_heap = List(length=a_count, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    for i in range(a_count):
        A_h.ptr[i] = Scalar[dtype](A_vals[i])

    var Y_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var Y_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    ssd_intra_chunk_fwd_cpu_naive[
        dtype,
        C_h.layout,
        B_h.layout,
        X_h.layout,
        A_h.layout,
        Y_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        state_dim,
        head_dim,
        C_h,
        B_h,
        X_h,
        A_h,
        Y_h,
    )

    # Pin to the canonical PyTorch reference numerics.
    for i in range(xy_count):
        assert_almost_equal(
            Y_h.ptr[i],
            Scalar[dtype](Y_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_intra_chunk_basic() raises:
    """Basic single-batch, single-chunk case."""
    run_ssd_intra_chunk[DType.float32](
        batch=1,
        n_chunks=1,
        n_heads=2,
        chunk_len=8,
        state_dim=16,
        head_dim=8,
    )


def test_ssd_intra_chunk_multi_chunk() raises:
    """Multiple chunks and heads."""
    run_ssd_intra_chunk[DType.float32](
        batch=1,
        n_chunks=3,
        n_heads=2,
        chunk_len=16,
        state_dim=16,
        head_dim=8,
    )


def test_ssd_intra_chunk_multi_batch() raises:
    """Multiple batch elements."""
    run_ssd_intra_chunk[DType.float32](
        batch=2,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        state_dim=8,
        head_dim=4,
    )


def test_ssd_intra_chunk_chunk_len_one() raises:
    """Degenerate chunk length of one (only the diagonal term)."""
    run_ssd_intra_chunk[DType.float32](
        batch=1,
        n_chunks=2,
        n_heads=1,
        chunk_len=1,
        state_dim=4,
        head_dim=4,
    )


def test_ssd_intra_chunk_optimized_matches_naive() raises:
    """Optimized CPU kernel matches the naive scalar reference."""
    comptime dtype = DType.float32
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    comptime batch = 1
    comptime n_chunks = 2
    comptime n_heads = 2
    comptime chunk_len = 16
    comptime state_dim = 16
    comptime head_dim = 8

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    var C_heap = List(length=cb_count, fill=Scalar[dtype](0))
    var C_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        C_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    var B_heap = List(length=cb_count, fill=Scalar[dtype](0))
    var B_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    var X_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var X_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    var A_heap = List(length=a_count, fill=Scalar[dtype](0))
    var A_h = LayoutTensor[dtype, layout_4d, MutAnyOrigin](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    var Y_fast_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var Y_fast_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_fast_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    var Y_naive_heap = List(length=xy_count, fill=Scalar[dtype](0))
    var Y_naive_h = LayoutTensor[dtype, layout_5d, MutAnyOrigin](
        Y_naive_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    random(C_h)
    random(B_h)
    random(X_h)
    random(A_h)
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        A_h.ptr[i] = Scalar[dtype](Float32(-0.35) + Float32(0.25) * v)

    ssd_intra_chunk_fwd_cpu[
        dtype,
        C_h.layout,
        B_h.layout,
        X_h.layout,
        A_h.layout,
        Y_fast_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        state_dim,
        head_dim,
        C_h,
        B_h,
        X_h,
        A_h,
        Y_fast_h,
    )
    ssd_intra_chunk_fwd_cpu_naive[
        dtype,
        C_h.layout,
        B_h.layout,
        X_h.layout,
        A_h.layout,
        Y_naive_h.layout,
    ](
        batch,
        n_chunks,
        n_heads,
        chunk_len,
        state_dim,
        head_dim,
        C_h,
        B_h,
        X_h,
        A_h,
        Y_naive_h,
    )

    for i in range(xy_count):
        assert_almost_equal(Y_fast_h.ptr[i], Y_naive_h.ptr[i], rtol=1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
