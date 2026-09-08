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
# Benchmarks three SM100 (Blackwell) BF16 matmul+1D-bias epilogue variants:
#
#   1. plain              — bare _matmul_gpu (no epilogue tensor)
#   2. compute_lambda_bias — 1D bias via elementwise_compute_lambda_fn (broadcast)
#   3. tma_bias           — 1D bias loaded via TMA epilogue (cp.async.bulk)
#
# Only runs on SM100+ (B200) GPUs; prints a warning and exits on other hardware.

from std.sys import (
    get_defined_dtype,
    get_defined_int,
    align_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import global_idx, grid_dim, block_dim, thread_idx, block_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.primitives import block
from std.memory import alloc, dealloc
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
)
from internal_utils._utils import InitializationType
from layout import (
    TileTensor,
    Idx,
    Coord,
    RowMajorLayout,
    row_major,
)
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.sm100_structured.default.dispatch_fused_bias_residual import (
    fused_bias_residual_matmul_dispatch_sm100,
)
from std.utils import IndexList


# ---------------------------------------------------------------------------
# GPU verification kernel (identical pattern to bench_matmul_tma_epilogue.mojo)
# ---------------------------------------------------------------------------


def _verify_buffers_gpu[
    c_type: DType, BLOCK_SIZE: Int
](
    output: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    reference: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    length: Int32,
    atol: Float32,
    rtol: Float32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    var abs_diff_sum: Float32 = 0
    var abs_ref_sum: Float32 = 0
    var max_violation = Float32.MIN_FINITE
    var out_nz: Float32 = 0
    var ref_nz: Float32 = 0

    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(length):
        var x = output[i].cast[.float32]()
        var y = reference[i].cast[.float32]()
        abs_diff_sum += abs(x - y)
        abs_ref_sum += abs(y)
        max_violation = max(max_violation, abs(x - y) - (atol + rtol * abs(y)))
        if x != 0:
            out_nz = 1.0
        if y != 0:
            ref_nz = 1.0
        i += stride

    abs_diff_sum = block.sum[block_size=BLOCK_SIZE](abs_diff_sum)
    abs_ref_sum = block.sum[block_size=BLOCK_SIZE](abs_ref_sum)
    max_violation = block.max[block_size=BLOCK_SIZE](max_violation)
    out_nz = block.max[block_size=BLOCK_SIZE](out_nz)
    ref_nz = block.max[block_size=BLOCK_SIZE](ref_nz)

    if thread_idx.x == 0:
        var base = block_idx.x * 5
        result[base + 0] = abs_diff_sum
        result[base + 1] = abs_ref_sum
        result[base + 2] = max_violation
        result[base + 3] = out_nz
        result[base + 4] = ref_nz


def _check_verification_result[
    c_type: DType,
    BLOCK_SIZE: Int,
    NUM_BLOCKS: Int,
](
    ctx: DeviceContext,
    c_device: DeviceBuffer[c_type],
    c_device_ref: DeviceBuffer[c_type],
    c_size: Int,
    label: String,
) raises:
    var rtol64: Float64
    var atol64: Float64
    rtol64, atol64 = pytorch_like_tolerances_for[.bfloat16]()
    var rtol = Float32(rtol64)
    var atol = Float32(atol64)

    var result_device = ctx.enqueue_create_buffer[.float32](NUM_BLOCKS * 5)

    comptime kernel = _verify_buffers_gpu[c_type, BLOCK_SIZE]
    ctx.enqueue_function[kernel](
        c_device,
        c_device_ref,
        Int32(c_size),
        atol,
        rtol,
        result_device,
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
    )

    var result_host_alloc = alloc[Float32](
        {count = NUM_BLOCKS * 5}
    ).into_managed()
    var result_host = result_host_alloc.unsafe_ptr()
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    var total_abs_diff: Float32 = 0
    var total_abs_ref: Float32 = 0
    var worst_violation = Float32.MIN_FINITE
    var any_out_nz: Float32 = 0
    var any_ref_nz: Float32 = 0

    for b_idx in range(NUM_BLOCKS):
        var base = b_idx * 5
        total_abs_diff += result_host[base + 0]
        total_abs_ref += result_host[base + 1]
        worst_violation = max(worst_violation, result_host[base + 2])
        any_out_nz = max(any_out_nz, result_host[base + 3])
        any_ref_nz = max(any_ref_nz, result_host[base + 4])

    dealloc(result_host_alloc^)

    if any_out_nz == 0:
        raise String(label, ": kernel output is all zeros")
    if any_ref_nz == 0:
        raise String(label, ": reference output is all zeros")

    if total_abs_ref > 0:
        var rel_diff = total_abs_diff / total_abs_ref
        if rel_diff > 0.001:
            raise String(
                label,
                " verification failed (relative_difference): ",
                rel_diff,
                " > 0.001",
            )

    if worst_violation > 0:
        raise String(
            label,
            " verification failed (element-wise tolerance): worst violation = ",
            worst_violation,
        )

    print(String("\n=== ", label, " PASSED ===\n"))


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def bench_matmul_1d_tma_epilogue[
    dtype: DType,
    N: Int,
    K: Int,
    *,
    variant: StringLiteral,
](
    ctx: DeviceContext,
    mut b: Bench,
    M: Int,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool = True,
) raises:
    comptime simd_size = 4
    comptime transpose_b = True

    var shape_c = Coord(M, Idx[N])
    var shape_a = Coord(M, Idx[K])
    var shape_b = Coord(Idx[N], Idx[K])

    var c_size = M * N
    var a_size = M * K
    var b_size = N * K

    # 1D bias layout: shape [N]
    comptime bias_layout = row_major(Coord(Idx[N]))

    var cb_a = CacheBustingBuffer[dtype](a_size, simd_size, ctx)
    var cb_b = CacheBustingBuffer[dtype](b_size, simd_size, ctx)
    var cb_c = CacheBustingBuffer[dtype](c_size, simd_size, ctx)
    var cb_bias = CacheBustingBuffer[dtype](N, simd_size, ctx)

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    cb_bias.init_on_device(init_type, ctx)

    @__copy_capture(cb_a, cb_b, cb_c, cb_bias)
    @always_inline
    def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
        var tensor_a = TileTensor(
            cb_a.offset_ptr(iteration), row_major(shape_a)
        )
        var tensor_b = TileTensor(
            cb_b.offset_ptr(iteration), row_major(shape_b)
        )
        var tensor_c = TileTensor(
            cb_c.offset_ptr(iteration), row_major(shape_c)
        )
        var bias_tile = TileTensor(cb_bias.offset_ptr(iteration), bias_layout)

        comptime if variant == "plain":
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
            ](tensor_c, tensor_a, tensor_b, ctx)

        elif variant == "compute_lambda_bias":

            @__parameter
            @always_inline
            @__copy_capture(tensor_c, bias_tile)
            def epilogue_lambda[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                # 1D bias: broadcast bias[j] across all M rows.
                var epi_val = bias_tile.load[width=width](Coord(idx[1])).cast[
                    _dtype
                ]()
                return val + epi_val

            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_compute_lambda_fn=epilogue_lambda,
            ](tensor_c, tensor_a, tensor_b, ctx)

        else:  # "tma_bias"
            # Wrap 1D bias as a (1, N) TileTensor to match the 2D epilogue
            # type the fused dispatcher expects. The kernel only uses the raw
            # pointer for 1D bias.
            var epi_1 = Int64(1)
            var epi_n = Int64(N)
            var epilogue_for_gpu = TileTensor(
                bias_tile.ptr, row_major(Coord(epi_1, epi_n))
            ).as_immut()
            fused_bias_residual_matmul_dispatch_sm100[
                transpose_b=transpose_b,
                epilogue_is_1d=True,
            ](
                tensor_c,
                tensor_a,
                tensor_b,
                epilogue_for_gpu.as_unsafe_any_origin(),
                ctx,
            )

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        2 * M * N * K,
    )

    var bench_name = String(
        variant,
        " (dtype=",
        String(dtype),
        ") : ",
        M,
        "_dynamic x ",
        N,
        " x ",
        K,
    )

    if run_benchmark:
        b.bench_function(
            bench_func,
            BenchId(bench_name),
            [flops],
        )
    else:
        kernel_launch(ctx, 0)

    if verify:
        # Allocate fresh device buffers for verification.
        var a_ver_dev = ctx.enqueue_create_buffer[dtype](a_size)
        var b_ver_dev = ctx.enqueue_create_buffer[dtype](b_size)
        var c_kernel_dev = ctx.enqueue_create_buffer[dtype](c_size)
        var c_ref_dev = ctx.enqueue_create_buffer[dtype](c_size)
        var bias_ver_dev = ctx.enqueue_create_buffer[dtype](N)

        var a_ver_nd = TileTensor(a_ver_dev, row_major(shape_a))
        var b_ver_nd = TileTensor(b_ver_dev, row_major(shape_b))
        var c_kernel_nd = TileTensor(c_kernel_dev, row_major(shape_c))
        var c_ref_nd = TileTensor(c_ref_dev, row_major(shape_c))
        var bias_ver_nd = TileTensor(bias_ver_dev, bias_layout)

        # Seed verification inputs from iteration-0 of the cache-busting
        # buffers.
        ctx.enqueue_copy(a_ver_dev, cb_a.offset_ptr(0))
        ctx.enqueue_copy(b_ver_dev, cb_b.offset_ptr(0))
        ctx.enqueue_copy(bias_ver_dev, cb_bias.offset_ptr(0))

        # cuBLAS reference (no epilogue tensor).
        vendor_blas.matmul[use_tf32=True](
            ctx,
            c_ref_nd,
            a_ver_nd,
            b_ver_nd,
            c_row_major=True,
            transpose_b=True,
        )

        # Run our kernel variant into the fresh c_kernel_dev buffer.
        comptime if variant == "plain":
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
            ](c_kernel_nd, a_ver_nd, b_ver_nd, ctx)

        elif variant == "compute_lambda_bias":

            @__parameter
            @always_inline
            @__copy_capture(bias_ver_nd)
            def ver_epilogue_lambda[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var epi_val = bias_ver_nd.load[width=width](Coord(idx[1])).cast[
                    _dtype
                ]()
                return val + epi_val

            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_compute_lambda_fn=ver_epilogue_lambda,
            ](c_kernel_nd, a_ver_nd, b_ver_nd, ctx)

        else:
            var ver_epi_1 = Int64(1)
            var ver_epi_n = Int64(N)
            var ver_epilogue = TileTensor(
                bias_ver_dev.unsafe_ptr(),
                row_major(Coord(ver_epi_1, ver_epi_n)),
            ).as_immut()
            fused_bias_residual_matmul_dispatch_sm100[
                transpose_b=transpose_b,
                epilogue_is_1d=True,
            ](
                c_kernel_nd,
                a_ver_nd,
                b_ver_nd,
                ver_epilogue.as_unsafe_any_origin(),
                ctx,
            )

        # Add 1D bias to reference output (broadcast across M rows).
        comptime if variant != "plain":
            var bias_host_alloc = alloc[Scalar[dtype]](
                {count = N}
            ).into_managed()
            var bias_host = bias_host_alloc.unsafe_ptr()
            var c_ref_host_alloc = alloc[Scalar[dtype]](
                {count = c_size}
            ).into_managed()
            var c_ref_host = c_ref_host_alloc.unsafe_ptr()
            ctx.enqueue_copy(bias_host, bias_ver_dev)
            ctx.enqueue_copy(c_ref_host, c_ref_dev)
            ctx.synchronize()

            for i in range(M):
                for j in range(N):
                    var ref_idx = i * N + j
                    c_ref_host[ref_idx] = (
                        c_ref_host[ref_idx].cast[.float32]()
                        + bias_host[j].cast[.float32]()
                    ).cast[dtype]()

            ctx.enqueue_copy(c_ref_dev, c_ref_host)
            ctx.synchronize()
            dealloc(bias_host_alloc^)
            dealloc(c_ref_host_alloc^)

        comptime NUM_BLOCKS = 32
        comptime BLOCK_SIZE = 256
        _check_verification_result[dtype, BLOCK_SIZE, NUM_BLOCKS](
            ctx,
            c_kernel_dev,
            c_ref_dev,
            c_size,
            String(variant),
        )

        _ = a_ver_dev^
        _ = b_ver_dev^
        _ = c_kernel_dev^
        _ = c_ref_dev^
        _ = bias_ver_dev^


# ---------------------------------------------------------------------------
# Convenience wrapper that runs all 3 variants
# ---------------------------------------------------------------------------


def create_1d_tma_epilogue_benches[
    dtype: DType,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    mut b: Bench,
    M: Int,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool,
) raises:
    bench_matmul_1d_tma_epilogue[dtype, N, K, variant="plain"](
        ctx,
        b,
        M,
        init_type,
        verify,
        run_benchmark=run_benchmark,
    )
    bench_matmul_1d_tma_epilogue[dtype, N, K, variant="compute_lambda_bias"](
        ctx,
        b,
        M,
        init_type,
        verify,
        run_benchmark=run_benchmark,
    )
    bench_matmul_1d_tma_epilogue[dtype, N, K, variant="tma_bias"](
        ctx,
        b,
        M,
        init_type,
        verify,
        run_benchmark=run_benchmark,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()

    var M = Int(arg_parse("M", 2048))
    comptime N = get_defined_int["N", 1536]()
    comptime K = get_defined_int["K", 4096]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", True)
    var run_benchmark = arg_parse("run_benchmark", True)

    var m = Bench()
    with DeviceContext() as ctx:
        comptime if not _is_sm10x_gpu(ctx.default_device_info):
            print(
                "bench_matmul_1d_tma_epilogue: this benchmark requires an"
                " SM100+ (Blackwell/B200) GPU — skipping."
            )
            return

        create_1d_tma_epilogue_benches[dtype, N, K](
            ctx,
            m,
            M,
            init_type,
            verify,
            run_benchmark=run_benchmark,
        )

    if run_benchmark:
        m.dump_report()
