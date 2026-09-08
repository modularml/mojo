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
# Benchmarks three SM100 (Blackwell) BF16 matmul+epilogue variants head-to-head:
#
#   1. plain         — bare _matmul_gpu (no epilogue tensor)
#   2. compute_lambda_bias — epilogue tensor applied via elementwise_compute_lambda_fn
#   3. tma_bias      — epilogue tensor passed as a TMA epilogue parameter to _matmul_gpu
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
    CoordLike,
    RowMajorLayout,
    row_major,
)
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.sm100_structured.default.dispatch_fused_bias_residual import (
    fused_bias_residual_matmul_dispatch_sm100,
)
from std.utils import IndexList


# ---------------------------------------------------------------------------
# GPU verification kernel (identical pattern to bench_matmul.mojo)
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
    """GPU kernel that computes verification metrics in one pass.

    Each block computes partial reductions and writes 5 Float32 values:
      [0] abs_diff_sum — for relative difference metric
      [1] abs_ref_sum  — for relative difference metric
      [2] max_violation — max(|x-y| - (atol + rtol*|y|)), <=0 means pass
      [3] out_nz — 1.0 if any output element is nonzero
      [4] ref_nz — 1.0 if any reference element is nonzero
    """
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
    """Run the GPU verification kernel and raise if tolerances are exceeded."""
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


def bench_matmul_tma_epilogue[
    dtype: DType,
    *,
    variant: StringLiteral,
](
    ctx: DeviceContext,
    mut b: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
    epilogue_shape: Coord,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool = True,
) raises:
    """Benchmark one of the three epilogue-matmul variants.

    Parameters:
        dtype: Element dtype (bfloat16).
        variant: One of "plain", "compute_lambda_bias", "tma_bias".

    Args:
        ctx: GPU device context.
        b: Bench harness.
        shape_c: Output shape (M x N).
        shape_a: A-matrix shape (M x K).
        shape_b: B-matrix shape (N x K) when transpose_b=True.
        epilogue_shape: Epilogue tensor shape (M x N).
        init_type: Initialization type for input tensors.
        verify: Whether to run correctness check after benchmarking.
        run_benchmark: If False, run a single iteration (for verify-only mode).
    """

    comptime transpose_b = True

    var cb_a = CacheBustingBuffer[dtype](shape_a.product(), ctx)
    var cb_b = CacheBustingBuffer[dtype](shape_b.product(), ctx)
    var cb_c = CacheBustingBuffer[dtype](shape_c.product(), ctx)
    var cb_epilogue = CacheBustingBuffer[dtype](epilogue_shape.product(), ctx)

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    # Initialize epilogue tensor with values in a narrow range (same as test files use rand).
    cb_epilogue.init_on_device(init_type, ctx)

    @__copy_capture(cb_a, cb_b, cb_c, cb_epilogue)
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
        var tensor_epilogue = TileTensor(
            cb_epilogue.offset_ptr(iteration), row_major(epilogue_shape)
        )

        comptime if variant == "plain":
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
            ](tensor_c, tensor_a, tensor_b, ctx)

        elif variant == "compute_lambda_bias":

            @__parameter
            @always_inline
            @__copy_capture(tensor_c, tensor_epilogue)
            def epilogue_lambda[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var epi_val = tensor_epilogue.load[width=width](
                    Coord(idx[0], idx[1])
                ).cast[_dtype]()
                return val + epi_val

            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_compute_lambda_fn=epilogue_lambda,
            ](tensor_c, tensor_a, tensor_b, ctx)

        else:  # "tma_bias"
            # Build epilogue TileTensor with RowMajorLayout[Int64, Int64] to
            # match the fused dispatcher's epilogue_tensor parameter type. Int
            # returns Int which mismatches; use Int64 directly.
            var epi_m = Int64(epilogue_shape[0].value())
            var epi_n = Int64(epilogue_shape[1].value())
            var epilogue_for_gpu = TileTensor(
                tensor_epilogue.ptr, row_major(Coord(epi_m, epi_n))
            ).as_immut()
            fused_bias_residual_matmul_dispatch_sm100[transpose_b=transpose_b,](
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
        2
        * Int(shape_c[0].value())
        * Int(shape_c[1].value())
        * Int(shape_a[1].value()),
    )

    var bench_name = String(
        variant,
        " (dtype=",
        String(dtype),
        ") : ",
        Int(shape_c[0].value()),
        "_dynamic x ",
        Int(shape_c[1].value()),
        " x ",
        Int(shape_a[1].value()),
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
        var M = Int(shape_c[0].value())
        var N = Int(shape_c[1].value())
        var K = Int(shape_a[1].value())
        var c_size = M * N
        var a_size = M * K
        var b_size = N * K  # transpose_b=True → B is N×K

        # Allocate fresh device buffers — independent of the cache-busting
        # buffers used during benchmarking.
        var a_ver_dev = ctx.enqueue_create_buffer[dtype](a_size)
        var b_ver_dev = ctx.enqueue_create_buffer[dtype](b_size)
        var c_kernel_dev = ctx.enqueue_create_buffer[dtype](c_size)
        var c_ref_dev = ctx.enqueue_create_buffer[dtype](c_size)
        var epilogue_ver_dev = ctx.enqueue_create_buffer[dtype](c_size)

        var a_ver_nd = TileTensor(a_ver_dev, row_major(shape_a))
        var b_ver_nd = TileTensor(b_ver_dev, row_major(shape_b))
        var c_kernel_nd = TileTensor(c_kernel_dev, row_major(shape_c))
        var c_ref_nd = TileTensor(c_ref_dev, row_major(shape_c))
        var epilogue_ver_nd = TileTensor(
            epilogue_ver_dev, row_major(epilogue_shape)
        )

        # Seed verification inputs from iteration-0 of the cache-busting buffers.
        ctx.enqueue_copy(a_ver_dev, cb_a.offset_ptr(0))
        ctx.enqueue_copy(b_ver_dev, cb_b.offset_ptr(0))
        ctx.enqueue_copy(epilogue_ver_dev, cb_epilogue.offset_ptr(0))

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
            @__copy_capture(epilogue_ver_nd)
            def ver_epilogue_lambda[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var epi_val = epilogue_ver_nd.load[width=width](
                    Coord(idx[0], idx[1])
                ).cast[_dtype]()
                return val + epi_val

            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_compute_lambda_fn=ver_epilogue_lambda,
            ](c_kernel_nd, a_ver_nd, b_ver_nd, ctx)

        else:
            var epi_m = Int64(epilogue_shape[0].value())
            var epi_n = Int64(epilogue_shape[1].value())
            var epilogue_for_ver = TileTensor(
                epilogue_ver_dev.unsafe_ptr(), row_major(Coord(epi_m, epi_n))
            ).as_immut()
            fused_bias_residual_matmul_dispatch_sm100[transpose_b=transpose_b,](
                c_kernel_nd,
                a_ver_nd,
                b_ver_nd,
                epilogue_for_ver.as_unsafe_any_origin(),
                ctx,
            )

        comptime if variant != "plain":
            # Add epilogue tensor to reference output on the host.
            var epilogue_host_alloc = alloc[Scalar[dtype]](
                {count = c_size}
            ).into_managed()
            var epilogue_host = epilogue_host_alloc.unsafe_ptr()
            var c_ref_host_alloc = alloc[Scalar[dtype]](
                {count = c_size}
            ).into_managed()
            var c_ref_host = c_ref_host_alloc.unsafe_ptr()
            ctx.enqueue_copy(epilogue_host, epilogue_ver_dev)
            ctx.enqueue_copy(c_ref_host, c_ref_dev)
            ctx.synchronize()

            for i in range(c_size):
                c_ref_host[i] = (
                    c_ref_host[i].cast[.float32]()
                    + epilogue_host[i].cast[.float32]()
                ).cast[dtype]()

            ctx.enqueue_copy(c_ref_dev, c_ref_host)
            ctx.synchronize()
            dealloc(epilogue_host_alloc^)
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
        _ = epilogue_ver_dev^


# ---------------------------------------------------------------------------
# Convenience wrapper that sets up shape Coord values and runs all 3 variants
# ---------------------------------------------------------------------------


def create_tma_epilogue_benches[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
    *,
    transpose_b: Bool = True,
](
    ctx: DeviceContext,
    mut b: Bench,
    m: MType,
    n: NType,
    k: KType,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool,
) raises:
    var shape_c = Coord(m, n)
    var shape_a = Coord(m, k)
    # transpose_b=True → B is stored as N×K
    var shape_b = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value],
        Idx[KType.static_value if transpose_b else NType.static_value],
    )
    var epilogue_shape = Coord(m, n)

    bench_matmul_tma_epilogue[dtype, variant="plain"](
        ctx,
        b,
        shape_c,
        shape_a,
        shape_b,
        epilogue_shape,
        init_type,
        verify,
        run_benchmark=run_benchmark,
    )
    bench_matmul_tma_epilogue[dtype, variant="compute_lambda_bias"](
        ctx,
        b,
        shape_c,
        shape_a,
        shape_b,
        epilogue_shape,
        init_type,
        verify,
        run_benchmark=run_benchmark,
    )
    bench_matmul_tma_epilogue[dtype, variant="tma_bias"](
        ctx,
        b,
        shape_c,
        shape_a,
        shape_b,
        epilogue_shape,
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
                "bench_matmul_tma_epilogue: this benchmark requires an SM100+"
                " (Blackwell/B200) GPU — skipping."
            )
            return

        create_tma_epilogue_benches[
            dtype,
            transpose_b=True,
        ](
            ctx,
            m,
            M,
            Idx[N],
            Idx[K],
            init_type,
            verify,
            run_benchmark=run_benchmark,
        )

    if run_benchmark:
        m.dump_report()
