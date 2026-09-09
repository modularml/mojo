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
#
# Benchmark driver for the fused GEMV + partial RMS norm kernel.
#
# Built on `std.benchmark` (same pattern as bench_matmul.mojo) so kbench
# orchestration and `dump_report()` just work.
#
# Two variants selected by the `fused` comptime flag:
#   -D fused=False  -> unfused 2-launch baseline (matmul + rms_norm_gpu;
#                       unnormed tail is a view into the matmul output)
#   -D fused=True   -> fused single-kernel path with flat per-block
#                       fetch_add grid sync + intra-block single-pass
#                       RMS reduce in the global-last arriver
#
# Cache-busting is on by default: a 512 MiB CacheBustingBuffer (>2x the
# B200 L2) rotates every user-visible tensor across iterations so each
# iter reads fresh HBM. Without this, gamma (3 KB) stays L1-hot and
# biases numbers down.
#
# Input validation (M=1, divisibility) runs pre-flight via plain
# `if ... raise`, and a post-bench verification re-runs the kernel once
# against a vendor-BLAS + host-side reference to catch numerical drift.
# ===----------------------------------------------------------------------=== #

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.memory import alloc
from std.sys import get_defined_bool, get_defined_int, size_of

from internal_utils import assert_almost_equal
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from std.math import sqrt
from layout import Coord, Idx, TileTensor, row_major

from internal_utils._cache_busting import CacheBustingBuffer
from internal_utils._utils import InitializationType
from nn.gemv_partial_norm import (
    gemv_and_partial_norm_unfused_with_scratch,
    gemv_and_partial_norm_with_scratch,
)


def _host_reference[
    c_type: DType, a_type: DType
](
    y_ref_ptr: MutPointer[Scalar[c_type], MutAnyOrigin],
    gamma_ptr: MutPointer[Scalar[a_type], MutAnyOrigin],
    normed_ref: MutPointer[Scalar[c_type], MutAnyOrigin],
    unnormed_ref: MutPointer[Scalar[c_type], MutAnyOrigin],
    n: Int,
    n_normed: Int,
    eps: Float32,
):
    """Reference partial RMS norm on a [1, n] row, computed in f64."""
    var n_unnormed = n - n_normed
    var sumsq: Float64 = 0.0
    for i in range(n_normed):
        var v = y_ref_ptr[i].cast[.float64]()
        sumsq += v * v
    var mean_sq = sumsq / Float64(n_normed)
    var norm_factor = Float64(1) / sqrt(mean_sq + eps.cast[.float64]())

    for i in range(n_normed):
        var v = y_ref_ptr[i].cast[.float64]()
        var g = gamma_ptr[i].cast[.float64]()
        normed_ref[i] = (v * norm_factor * g).cast[c_type]()

    for i in range(n_unnormed):
        unnormed_ref[i] = y_ref_ptr[n_normed + i]


def main() raises:
    comptime fused = get_defined_bool["fused", False]()
    comptime cache_bust = get_defined_bool["cache_bust", True]()
    comptime tile_n = get_defined_int["tile_n", 4]()
    comptime num_threads = get_defined_int["num_threads", 256]()
    comptime a_type = DType.bfloat16
    comptime c_type = DType.bfloat16

    # Primary shape (Kimi M=1 path). Only M=1 is supported by the fused path.
    var M = 1
    var N = 2112
    var K = 7168
    var N_NORMED = 1536
    var N_UNNORMED = N - N_NORMED

    comptime if fused:
        if M != 1:
            raise Error(
                "fused path is GEMV-only; expected M=1 but got M=",
                M,
                ". Use matmul + rms_norm_gpu for M>1.",
            )
    if N_NORMED <= 0 or N_NORMED > N or K <= 0:
        raise Error(
            "invalid primary shape: M=",
            M,
            " N=",
            N,
            " K=",
            K,
            " N_normed=",
            N_NORMED,
        )

    comptime a_shape = row_major(Coord(Idx[1], Idx[7168]))
    comptime b_shape = row_major(Coord(Idx[2112], Idx[7168]))
    comptime c_shape = row_major(Coord(Idx[1], Idx[2112]))
    comptime normed_shape = row_major(Coord(Idx[1], Idx[1536]))
    var unnormed_shape = row_major(Coord(Idx[1], N_UNNORMED))
    comptime gamma_shape = row_major(Idx[1536])

    comptime variant: String = "fused" if fused else "unfused"
    comptime run_name: String = "gemv_partial_norm/" + variant

    # Actual HBM traffic for both the unfused and fused paths under
    # cache-bust. The weight matrix dominates; everything else is KB-scale.
    #   Reads (all HBM-cold under cache-bust):
    #     weight      N * K       bytes / elem = a_type
    #     activation  M * K                      a_type
    #     gamma       N_normed                   a_type
    #   Writes (go through L2 to HBM):
    #     initial output  M * N                  c_type   (matmul y in
    #                                                     unfused; split
    #                                                     normed+unnormed
    #                                                     writes in fused)
    #     normed rewrite  M * N_normed           c_type   (rms_norm writes
    #                                                     normed_output
    #                                                     in unfused;
    #                                                     apply-norm
    #                                                     rewrites it in
    #                                                     fused)
    # The y re-read by rms_norm in unfused (and the normed_output re-read
    # in the fused apply phase) stay L2-hot, so they do not add to HBM
    # traffic.
    var read_bytes = (
        Int64(N) * Int64(K) * Int64(size_of[a_type]())
        + Int64(M) * Int64(K) * Int64(size_of[a_type]())
        + Int64(N_NORMED) * Int64(size_of[a_type]())
    )
    var write_bytes = Int64(M) * Int64(N) * Int64(size_of[c_type]()) + Int64(
        M
    ) * Int64(N_NORMED) * Int64(size_of[c_type]())
    var total_bytes = Int(read_bytes + write_bytes)

    with DeviceContext() as ctx:
        comptime simd_size = 4
        var cb_a = CacheBustingBuffer[a_type](
            M * K, simd_size, ctx, enabled=cache_bust
        )
        var cb_b = CacheBustingBuffer[a_type](
            N * K, simd_size, ctx, enabled=cache_bust
        )
        var cb_gamma = CacheBustingBuffer[a_type](
            N_NORMED, simd_size, ctx, enabled=cache_bust
        )
        var cb_y = CacheBustingBuffer[c_type](
            M * N, simd_size, ctx, enabled=cache_bust
        )
        var cb_normed = CacheBustingBuffer[c_type](
            M * N_NORMED, simd_size, ctx, enabled=cache_bust
        )
        var cb_unnormed = CacheBustingBuffer[c_type](
            M * N_UNNORMED, simd_size, ctx, enabled=cache_bust
        )
        cb_a.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_b.init_on_device(InitializationType.uniform_distribution, ctx)
        cb_gamma.init_on_device(InitializationType.uniform_distribution, ctx)

        # y_ref is the vendor-BLAS reference for post-bench verification.
        var y_ref_dev = ctx.enqueue_create_buffer[c_type](M * N)
        var y_ref_tensor = TileTensor(y_ref_dev, c_shape)

        var a_iter0 = TileTensor(cb_a.offset_ptr(0), a_shape)
        var b_iter0 = TileTensor(cb_b.offset_ptr(0), b_shape)

        var eps = Float32(0.001)

        # Kernel-internal scratch: reused across iters by design.
        var counter_buf = ctx.enqueue_create_buffer[.int32](1)
        ctx.enqueue_memset(counter_buf, Int32(0))

        ctx.synchronize()

        vendor_blas.matmul(
            ctx,
            y_ref_tensor.to_layout_tensor(),
            a_iter0.to_layout_tensor(),
            b_iter0.to_layout_tensor(),
            c_row_major=True,
            transpose_b=True,
        )

        @always_inline
        def kernel_launch(
            ctx: DeviceContext, iteration: Int
        ) raises {
            mut cb_a,
            mut cb_b,
            mut cb_gamma,
            mut cb_y,
            mut cb_normed,
            mut cb_unnormed,
            mut counter_buf,
            imm,
        }:
            var a_tensor = TileTensor(cb_a.offset_ptr(iteration), a_shape)
            var b_tensor = TileTensor(cb_b.offset_ptr(iteration), b_shape)
            var gamma_tensor = TileTensor(
                cb_gamma.offset_ptr(iteration), gamma_shape
            )
            var normed_tensor = TileTensor(
                cb_normed.offset_ptr(iteration), normed_shape
            )
            var unnormed_tensor = TileTensor(
                cb_unnormed.offset_ptr(iteration), unnormed_shape
            )

            comptime if fused:
                gemv_and_partial_norm_with_scratch[
                    transpose_b=True,
                    tile_n=tile_n,
                    num_threads=num_threads,
                ](
                    normed_tensor,
                    unnormed_tensor,
                    a_tensor,
                    b_tensor,
                    gamma_tensor,
                    eps,
                    counter_buf.unsafe_ptr()
                    .unsafe_mut_cast[True]()
                    .as_unsafe_any_origin(),
                    ctx,
                )
            else:
                gemv_and_partial_norm_unfused_with_scratch[transpose_b=True](
                    normed_tensor,
                    a_tensor,
                    b_tensor,
                    gamma_tensor,
                    eps,
                    cb_y.offset_ptr(iteration).as_unsafe_any_origin(),
                    ctx,
                )

        @always_inline
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, kernel_launch, ctx)

        var bw = ThroughputMeasure(BenchMetric.bytes, total_bytes)

        var m = Bench()
        m.bench_function(bench_func, BenchId(run_name), [bw])

        # Post-bench correctness verification: re-run the kernel into
        # iter=0's output slots and compare to the vendor-BLAS + host
        # reference.
        kernel_launch(ctx, 0)

        var y_ref_host_ptr = List(length=M * N, fill=Scalar[c_type](0))
        var gamma_host_ptr = List(length=N_NORMED, fill=Scalar[a_type](0))
        var normed_ref_ptr = List(length=M * N_NORMED, fill=Scalar[c_type](0))
        var unnormed_ref_ptr = List(
            length=M * N_UNNORMED, fill=Scalar[c_type](0)
        )
        var normed_ours_ptr = List(length=M * N_NORMED, fill=Scalar[c_type](0))
        var unnormed_ours_ptr = List(
            length=M * N_UNNORMED, fill=Scalar[c_type](0)
        )

        ctx.enqueue_copy(
            gamma_host_ptr.unsafe_ptr(), cb_gamma.offset_ptr(0), N_NORMED
        )
        ctx.enqueue_copy(y_ref_host_ptr, y_ref_dev)
        ctx.enqueue_copy(
            normed_ours_ptr.unsafe_ptr(),
            cb_normed.offset_ptr(0),
            M * N_NORMED,
        )
        ctx.enqueue_copy(
            unnormed_ours_ptr.unsafe_ptr(),
            cb_unnormed.offset_ptr(0),
            M * N_UNNORMED,
        )
        ctx.synchronize()

        _host_reference[c_type, a_type](
            y_ref_host_ptr.unsafe_ptr().as_unsafe_any_origin(),
            gamma_host_ptr.unsafe_ptr().as_unsafe_any_origin(),
            normed_ref_ptr.unsafe_ptr().as_unsafe_any_origin(),
            unnormed_ref_ptr.unsafe_ptr().as_unsafe_any_origin(),
            N,
            N_NORMED,
            eps,
        )

        assert_almost_equal(
            normed_ours_ptr.unsafe_ptr(),
            normed_ref_ptr.unsafe_ptr(),
            M * N_NORMED,
            atol=5e-2,
            rtol=5e-2,
        )
        # Unfused path doesn't populate `unnormed_output` (it's a view
        # into the matmul scratch). Only check for fused.
        comptime if fused:
            assert_almost_equal(
                unnormed_ours_ptr.unsafe_ptr(),
                unnormed_ref_ptr.unsafe_ptr(),
                M * N_UNNORMED,
                atol=1e-2,
                rtol=1e-2,
            )

        m.dump_report()
        _ = unnormed_ours_ptr^
        _ = normed_ours_ptr^
        _ = unnormed_ref_ptr^
        _ = normed_ref_ptr^
        _ = y_ref_host_ptr^
        _ = gamma_host_ptr^
