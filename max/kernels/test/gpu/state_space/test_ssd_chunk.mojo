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

from std.math import exp

from max.gpu.host import DeviceContext
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from std.random import rand
from state_space.ssd_chunk import (
    ssd_intra_chunk_fwd_gpu,
    ssd_intra_chunk_fwd_gpu_naive,
    ssd_intra_chunk_fwd_gpu_fused,
    ssd_intra_chunk_fwd_gpu_static,
    ssd_intra_chunk_fwd_gpu_static_mma,
)
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index


def run_ssd_intra_chunk_gpu[
    dtype: DType,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the optimized SSD intra-chunk GPU path and check against a reference.

    Uses batched ``C @ B^T`` (Modular tensor-core matmul) plus a Triton-style
    tiled chunk-scan kernel. Validates against the same independent scalar
    reference used by the CPU test.
    """
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    # ── Host tensors ─────────────────────────────────────────────────────────
    var C_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    var C_h = LayoutTensor[dtype, layout_5d, _](
        C_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](C_h.ptr, C_h.size())

    var B_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    var B_h = LayoutTensor[dtype, layout_5d, _](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](B_h.ptr, B_h.size())

    var X_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    var X_h = LayoutTensor[dtype, layout_5d, _](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](X_h.ptr, X_h.size())

    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var A_h = LayoutTensor[dtype, layout_4d, _](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](A_h.ptr, A_h.size())
    # Make A negative-ish (a real SSM decays) so exp(cumsum diffs) stays
    # bounded. `rand` fills in [0, 1); map into a small negative range.
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        # Map [0, 1) -> [-0.6, -0.1].
        var scaled = Float32(-0.6) + Float32(0.5) * v
        A_h.ptr[i] = Scalar[dtype](scaled)

    var Y_gpu_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    var Y_gpu_h = LayoutTensor[dtype, layout_5d, _](
        Y_gpu_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )

    # ── Device buffers ───────────────────────────────────────────────────────
    var C_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var B_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var X_device = ctx.enqueue_create_buffer[dtype](xy_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var Y_device = ctx.enqueue_create_buffer[dtype](xy_count)

    with ctx.push_context():
        ctx.enqueue_copy(C_device, C_h.ptr)
        ctx.enqueue_copy(B_device, B_h.ptr)
        ctx.enqueue_copy(X_device, X_h.ptr)
        ctx.enqueue_copy(A_device, A_h.ptr)

    var C_tt = TileTensor(
        C_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var B_tt = TileTensor(
        B_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var X_tt = TileTensor(
        X_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var A_tt = TileTensor(
        A_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var Y_tt = TileTensor(
        Y_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )

    with ctx.push_context():
        ssd_intra_chunk_fwd_gpu[
            dtype,
            C_tt.LayoutType,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            Y_tt.LayoutType,
        ](
            batch,
            n_chunks,
            n_heads,
            chunk_len,
            state_dim,
            head_dim,
            C_tt,
            B_tt,
            X_tt,
            A_tt,
            Y_tt,
            ctx,
        )

    with ctx.push_context():
        ctx.enqueue_copy(Y_gpu_h.ptr, Y_device)
    ctx.synchronize()

    # ── CPU reference: naive triple-loop of the SSD intra-chunk formula ──────
    var ref_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)

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

                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc

                for l in range(chunk_len):
                    for p in range(head_dim):
                        var y_acc = Float32(0.0)
                        for s in range(l + 1):
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

    # ── Compare GPU vs CPU reference ─────────────────────────────────────────
    for i in range(xy_count):
        assert_almost_equal(Y_gpu_h.ptr[i], ref_heap[i], rtol=rtol)


def run_ssd_intra_chunk_gpu_static[
    dtype: DType,
    chunk_len: Int,
    state_dim: Int,
    head_dim: Int,
    use_mma: Bool = False,
    use_fused: Bool = False,
](
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    a_min: Float32 = -0.6,
    a_max: Float32 = -0.1,
) raises:
    """Exercise a static-shape path against the scalar CPU reference.

    ``use_mma=False`` drives the default scalar-scan static path
    (``ssd_intra_chunk_fwd_gpu_static``); ``use_mma=True`` drives the
    tensor-core path that materialises ``M = causal_decay(CB)`` and computes
    ``Y = M @ X`` via two batched matmuls
    (``ssd_intra_chunk_fwd_gpu_static_mma``). ``chunk_len`` / ``state_dim`` /
    ``head_dim`` are comptime so ``batched_matmul`` sees static N/K.

    ``a_min`` / ``a_max`` bound the per-token decay ``A``. The scan factors
    ``exp(cum_l−cum_s) = exp(cum_l)·exp(−cum_s)``; the intermediate
    ``exp(−cum_s)`` can overflow FP32 once ``|cum|`` exceeds ~88, so long
    ``chunk_len`` tests use a milder range to stay in the regime real Mamba2
    decay rates occupy.
    """
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_5d = Layout.row_major[5]()

    var cb_count = batch * n_chunks * n_heads * chunk_len * state_dim
    var xy_count = batch * n_chunks * n_heads * chunk_len * head_dim
    var a_count = batch * n_chunks * n_heads * chunk_len

    var C_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    var C_h = LayoutTensor[dtype, layout_5d, _](
        C_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](C_h.ptr, C_h.size())

    var B_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    var B_h = LayoutTensor[dtype, layout_5d, _](
        B_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, state_dim)
        ),
    )
    rand[dtype](B_h.ptr, B_h.size())

    var X_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    var X_h = LayoutTensor[dtype, layout_5d, _](
        X_heap,
        RuntimeLayout[layout_5d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len, head_dim)
        ),
    )
    rand[dtype](X_h.ptr, X_h.size())

    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    var A_h = LayoutTensor[dtype, layout_4d, _](
        A_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch, n_chunks, n_heads, chunk_len)
        ),
    )
    rand[dtype](A_h.ptr, A_h.size())
    for i in range(a_count):
        var v = A_h.ptr[i].cast[DType.float32]()
        var scaled = a_min + (a_max - a_min) * v
        A_h.ptr[i] = Scalar[dtype](scaled)

    var Y_gpu_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)

    var C_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var B_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var X_device = ctx.enqueue_create_buffer[dtype](xy_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var Y_device = ctx.enqueue_create_buffer[dtype](xy_count)

    with ctx.push_context():
        ctx.enqueue_copy(C_device, C_h.ptr)
        ctx.enqueue_copy(B_device, B_h.ptr)
        ctx.enqueue_copy(X_device, X_h.ptr)
        ctx.enqueue_copy(A_device, A_h.ptr)

    var C_tt = TileTensor(
        C_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var B_tt = TileTensor(
        B_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(
        A_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var Y_tt = TileTensor(
        Y_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    with ctx.push_context():
        comptime if use_fused:
            ssd_intra_chunk_fwd_gpu_fused[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)
        elif use_mma:
            ssd_intra_chunk_fwd_gpu_static_mma[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)
        else:
            ssd_intra_chunk_fwd_gpu_static[
                dtype,
                chunk_len,
                state_dim,
                head_dim,
                C_tt.LayoutType,
                B_tt.LayoutType,
                X_tt.LayoutType,
                A_tt.LayoutType,
                Y_tt.LayoutType,
            ](batch, n_chunks, n_heads, C_tt, B_tt, X_tt, A_tt, Y_tt, ctx)

    with ctx.push_context():
        ctx.enqueue_copy(Y_gpu_heap, Y_device)
    ctx.synchronize()

    var ref_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)

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

                var cumsum = List[Float32](length=chunk_len, fill=0.0)
                var acc = Float32(0.0)
                for l in range(chunk_len):
                    acc += A_h.ptr[a_base + l].cast[DType.float32]()
                    cumsum[l] = acc

                for l in range(chunk_len):
                    for p in range(head_dim):
                        var y_acc = Float32(0.0)
                        for s in range(l + 1):
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

    for i in range(xy_count):
        assert_almost_equal(Y_gpu_heap[i], ref_heap[i], rtol=rtol)


def test_ssd_intra_chunk_gpu_static_scalar() raises:
    """Static-shape scalar-scan path against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32, chunk_len=16, state_dim=16, head_dim=8
    ](batch=1, n_chunks=3, n_heads=2, ctx=ctx)


def test_ssd_intra_chunk_gpu_static_mma() raises:
    """Static-shape tensor-core MMA path against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32, chunk_len=16, state_dim=16, head_dim=8, use_mma=True
    ](batch=1, n_chunks=3, n_heads=2, ctx=ctx)


def test_ssd_intra_chunk_gpu_fused_small() raises:
    """Fused single-pass MMA path (RFC 0009), small gate-hitting dims."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32,
        chunk_len=64,
        state_dim=128,
        head_dim=64,
        use_fused=True,
    ](batch=1, n_chunks=2, n_heads=2, ctx=ctx)


def test_ssd_intra_chunk_gpu_fused_mamba2() raises:
    """Fused single-pass MMA path at the Mamba2-130m profile (L=256,P=64,N=128).
    """
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32,
        chunk_len=256,
        state_dim=128,
        head_dim=64,
        use_fused=True,
    ](batch=1, n_chunks=2, n_heads=4, ctx=ctx)


def test_ssd_intra_chunk_gpu_static_mma_multi_batch() raises:
    """Static-shape MMA path with multiple batch elements."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32, chunk_len=8, state_dim=8, head_dim=4, use_mma=True
    ](batch=2, n_chunks=2, n_heads=2, ctx=ctx)


def test_ssd_intra_chunk_gpu_static_a100_mma_shape() raises:
    """Production-shape static scalar path exercising the changed GPU paths.

    Covers the code paths the GPU optimizations actually changed, which the
    tiny shapes above do NOT:

    - ``state_dim=128`` + ``chunk_len=128`` (multiple of 128) makes the
      ``CB = C @ B^T`` stage take ``batched_matmul``'s **A100 batched
      tensor-core path** (``multistage_gemm_cond``: N%128==0, K%32==0, K>=128).
    - ``chunk_len=128`` > ``SCAN_BLOCK_M`` (32 on GB10) → **4 M-tiles**, so the
      causal early-exit / tail-balancing across M-tiles is exercised.
    - ``head_dim=64`` > ``SCAN_BLOCK_N`` (32) → **2 N-tiles**.
    - The parallel cumsum runs one block per slice with ``block_dim=128``.
    """
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32, chunk_len=128, state_dim=128, head_dim=64
    ](batch=1, n_chunks=1, n_heads=2, ctx=ctx)


def test_ssd_intra_chunk_gpu_static_mma_a100_shape() raises:
    """Production-shape MMA path: decay kernel + ``Y = M @ X`` at the shape
    where the ``CB`` matmul takes the A100 batched tensor-core path
    (``state_dim=128``, ``chunk_len=128``)."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32,
        chunk_len=128,
        state_dim=128,
        head_dim=64,
        use_mma=True,
    ](batch=1, n_chunks=1, n_heads=2, ctx=ctx)


def test_ssd_intra_chunk_gpu_static_long_chunk() raises:
    """Full Mamba2 ``chunk_len=256`` with realistic (mild) decay.

    Covers the production chunk length: the Hillis-Steele cumsum at
    ``block_dim=256``, 8 scan M-tiles, and the A100 batched MMA. Uses a milder
    decay range so the factored ``exp(-cum_s)`` stays well within FP32 — the
    regime real Mamba2 dt*A rates occupy (aggressive decay over 256 tokens can
    push ``|cum|`` past the ~88 FP32-exp overflow point; see helper docstring).
    """
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu_static[
        DType.float32, chunk_len=256, state_dim=128, head_dim=16
    ](batch=1, n_chunks=1, n_heads=2, ctx=ctx, a_min=-0.08, a_max=-0.01)


def test_ssd_intra_chunk_gpu_multi_m_tile() raises:
    """Dynamic-shape path with ``chunk_len=64`` > ``SCAN_BLOCK_M`` so the scan
    spans multiple M-tiles (the tiny dynamic tests above are all single-tile).
    """
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=1,
        n_chunks=2,
        n_heads=2,
        chunk_len=64,
        state_dim=32,
        head_dim=16,
        ctx=ctx,
    )


def test_ssd_intra_chunk_gpu_golden() raises:
    """Golden-parity test on GPU against the canonical PyTorch SSD reference.

    Same FIXED tiny input as the CPU golden test (batch=1, n_chunks=1,
    n_heads=1, L=4, N=3, P=2). Pins the GPU kernel to ``intra_chunk_diag`` in
    ``.planning/parity/ssd_minimal_ref.py``.
    """
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 1
    comptime n_heads = 1
    comptime chunk_len = 4
    comptime state_dim = 3
    comptime head_dim = 2

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

    # Host buffers from fixed literals.
    var C_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    for i in range(cb_count):
        C_heap[i] = Scalar[dtype](C_vals[i])
    var B_heap = ctx.enqueue_create_host_buffer[dtype](cb_count)
    for i in range(cb_count):
        B_heap[i] = Scalar[dtype](B_vals[i])
    var X_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)
    for i in range(xy_count):
        X_heap[i] = Scalar[dtype](X_vals[i])
    var A_heap = ctx.enqueue_create_host_buffer[dtype](a_count)
    for i in range(a_count):
        A_heap[i] = Scalar[dtype](A_vals[i])
    var Y_heap = ctx.enqueue_create_host_buffer[dtype](xy_count)

    var C_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var B_device = ctx.enqueue_create_buffer[dtype](cb_count)
    var X_device = ctx.enqueue_create_buffer[dtype](xy_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var Y_device = ctx.enqueue_create_buffer[dtype](xy_count)

    with ctx.push_context():
        ctx.enqueue_copy(C_device, C_heap)
        ctx.enqueue_copy(B_device, B_heap)
        ctx.enqueue_copy(X_device, X_heap)
        ctx.enqueue_copy(A_device, A_heap)

    var C_tt = TileTensor(
        C_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var B_tt = TileTensor(
        B_device,
        row_major(batch, n_chunks, n_heads, chunk_len, state_dim),
    )
    var X_tt = TileTensor(
        X_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )
    var A_tt = TileTensor(
        A_device,
        row_major(batch, n_chunks, n_heads, chunk_len),
    )
    var Y_tt = TileTensor(
        Y_device,
        row_major(batch, n_chunks, n_heads, chunk_len, head_dim),
    )

    var total_slices = batch * n_chunks * n_heads

    var compiled_func = ctx.compile_function[
        ssd_intra_chunk_fwd_gpu_naive[
            dtype,
            C_tt.LayoutType,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            Y_tt.LayoutType,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(state_dim),
            Int32(head_dim),
            C_tt,
            B_tt,
            X_tt,
            A_tt,
            Y_tt,
            grid_dim=(1,),
            block_dim=(total_slices,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(Y_heap, Y_device)
    ctx.synchronize()

    for i in range(xy_count):
        assert_almost_equal(
            Y_heap[i],
            Scalar[dtype](Y_expected[i]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_ssd_intra_chunk_gpu_basic() raises:
    """Basic single-batch, single-chunk case against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=1,
        n_chunks=1,
        n_heads=2,
        chunk_len=8,
        state_dim=16,
        head_dim=8,
        ctx=ctx,
    )


def test_ssd_intra_chunk_gpu_multi_chunk() raises:
    """Multiple chunks and heads against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=1,
        n_chunks=3,
        n_heads=2,
        chunk_len=16,
        state_dim=16,
        head_dim=8,
        ctx=ctx,
    )


def test_ssd_intra_chunk_gpu_multi_batch() raises:
    """Multiple batch elements against the naive reference."""
    var ctx = DeviceContext()
    run_ssd_intra_chunk_gpu[DType.float32](
        batch=2,
        n_chunks=2,
        n_heads=2,
        chunk_len=8,
        state_dim=8,
        head_dim=4,
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
