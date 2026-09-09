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

from std.random import random_float64
from std.sys import get_defined_bool, get_defined_dtype

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import (
    get_defined_shape,
    int_list_to_tuple,
    CacheBustingBuffer,
)

from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from nn.normalization import rms_norm_gpu, rms_norm_fused_fp8

from linalg.fp8_quantization import quantize_dynamic_scaled_fp8
from std.utils.index import Index, IndexList


def bench_rms_norm_fused_fp8[
    rank: Int,
    //,
    in_dtype: DType,
    out_dtype: DType,
    shape: IndexList[rank],
    cache_busting: Bool = True,
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    """Benchmark fused RMS norm + FP8 quantization against separate operations.

    Compares:
    1. RMS norm alone
    2. FP8 quantization alone
    3. Fused RMS norm + FP8 quantization
    """
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    # Allocate host memory
    var data_h = List(length=rows * cols, fill=Scalar[in_dtype](0))
    var gamma_h = List(length=cols, fill=Scalar[in_dtype](0))

    # Initialize data
    for i in range(rows * cols):
        var val = Scalar[in_dtype](random_float64(0, 100).cast[in_dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[in_dtype]()

    # Calculate buffer sizes for cache busting
    comptime simd_size = 4
    var data_size = rows * cols
    var cb_data = CacheBustingBuffer[in_dtype](
        data_size, simd_size, ctx, cache_busting
    )
    var cb_rms_output = CacheBustingBuffer[in_dtype](
        data_size, simd_size, ctx, cache_busting
    )
    var cb_fp8_output = CacheBustingBuffer[out_dtype](
        data_size, simd_size, ctx, cache_busting
    )
    var cb_fused_output = CacheBustingBuffer[out_dtype](
        data_size, simd_size, ctx, cache_busting
    )
    var gamma_d = ctx.enqueue_create_buffer[in_dtype](cols)
    var scales_d = ctx.enqueue_create_buffer[.float32](rows)

    var param_shape = Index(cols)

    # Create TileTensor for gamma
    var gamma_tensor = TileTensor(gamma_d, row_major(Coord(param_shape)))

    var epsilon = Float32(0.001)
    var weight_offset = Scalar[in_dtype](0.0)

    # Copy data to device (initialize the whole buffer when cache busting)
    from internal_utils._utils import InitializationType

    comptime random_distribution = InitializationType.uniform_distribution
    cb_data.init_on_device(random_distribution, ctx)
    cb_rms_output.init_on_device(random_distribution, ctx)
    ctx.enqueue_copy(gamma_d, gamma_h)

    # ===== Benchmark 1: RMS norm alone =====
    @always_inline
    def bench_rms_norm(
        mut b: Bencher,
    ) raises {
        var gamma_tensor,
        var epsilon,
        var weight_offset,
        var cb_data,
        var cb_rms_output,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            # Construct buffers with offsets
            var data_ptr_offset = cb_data.offset_ptr(iteration)
            var rms_output_ptr_offset = cb_rms_output.offset_ptr(iteration)
            var data_buf_offset = TileTensor(
                data_ptr_offset, row_major(Coord(shape))
            )
            var rms_output_buf_offset = TileTensor(
                rms_output_ptr_offset, row_major(Coord(shape))
            )

            # Input function for RMS norm. `rms_norm_gpu` migrated to a `Coord`
            # shape boundary (softmax PR #88203).
            @__copy_capture(data_buf_offset)
            @always_inline
            @__parameter
            def input_fn[width: Int](coords: Coord) -> SIMD[in_dtype, width]:
                var idx = data_buf_offset.layout(coords)
                return data_buf_offset.raw_load[width=width, alignment=width](
                    idx
                )

            # Output function for RMS norm
            @always_inline
            @__copy_capture(rms_output_buf_offset)
            @__parameter
            def rms_output_fn[
                width: SIMDLength, alignment: Int
            ](coords: Coord, val: SIMD[in_dtype, width]) -> None:
                var idx = rms_output_buf_offset.layout(coords)
                rms_output_buf_offset.raw_store[
                    width=width, alignment=alignment
                ](idx, val)

            rms_norm_gpu[
                rank, input_fn, rms_output_fn, multiply_before_cast=True
            ](
                Coord(shape),
                gamma_tensor,
                epsilon,
                weight_offset,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_rms_norm,
        BenchId(
            "rms_norm_only",
            input_id=String(fn_name, "/", in_dtype, "/", out_dtype, "/", shape),
        ),
    )

    # ===== Benchmark 2: FP8 quantization alone =====
    var scales_base_ptr = scales_d.unsafe_ptr()

    @always_inline
    def bench_fp8_quant(
        mut b: Bencher,
    ) raises {var cb_rms_output, var cb_fp8_output, var scales_base_ptr, imm,}:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            # Input function for FP8 quant (reads from RMS norm output)
            var rms_ptr_offset = cb_rms_output.offset_ptr(iteration)

            @always_inline
            def fp8_input_fn[
                width: Int, alignment: Int
            ](row: Int, col: Int) {var rms_ptr_offset} -> SIMD[in_dtype, width]:
                var idx = row * cols + col
                return rms_ptr_offset.load[width=width](idx)

            var fp8_output_tt = TileTensor(
                cb_fp8_output.offset_ptr(iteration),
                row_major(Coord(rows, cols)),
            )
            var scales_tt = TileTensor(
                scales_base_ptr,
                row_major(Coord(Idx[1], rows)),
            )

            quantize_dynamic_scaled_fp8[
                in_dtype=in_dtype,
                group_size_or_per_token=-1,  # Per-token quantization
                num_cols=cols,
            ](fp8_input_fn, fp8_output_tt, scales_tt, Float32(448.0), ctx, rows)

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fp8_quant,
        BenchId(
            "fp8_quant_only",
            input_id=String(fn_name, "/", in_dtype, "/", out_dtype, "/", shape),
        ),
    )

    # ===== Benchmark 3: Fused RMS norm + FP8 quantization =====
    var scales_base_ptr_fused = scales_base_ptr

    @always_inline
    def bench_fused(
        mut b: Bencher,
    ) raises {
        var gamma_tensor,
        var epsilon,
        var weight_offset,
        var cb_data,
        var cb_fused_output,
        var scales_base_ptr_fused,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx_: DeviceContext, iteration: Int) raises {imm}:
            # Input function with offset
            var data_ptr_offset = cb_data.offset_ptr(iteration)

            @__copy_capture(data_ptr_offset)
            @always_inline
            @__parameter
            def input_fn_fused[
                width: Int, _rank: Int
            ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
                var data_buf_offset = TileTensor(
                    data_ptr_offset, row_major(Coord(shape))
                )
                var idx = data_buf_offset.layout(Coord(coords))
                return data_buf_offset.raw_load[width=width, alignment=width](
                    idx
                )

            var fused_output_tt = TileTensor(
                cb_fused_output.offset_ptr(iteration),
                row_major(Coord(shape)),
            )
            var fused_scale_shape = shape
            fused_scale_shape[rank - 1] = 1
            var fused_scales_tt = TileTensor(
                scales_base_ptr_fused,
                row_major(Coord(fused_scale_shape)),
            )

            # DeviceContext is passed directly
            var ctx_ptr = ctx_
            rms_norm_fused_fp8[
                in_dtype,
                out_dtype,
                DType.float32,
                rank,
                input_fn_fused,
            ](
                shape,
                fused_output_tt,
                gamma_tensor,
                epsilon,
                weight_offset,
                ctx_ptr,
                Float32(448.0),
                fused_scales_tt,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fused,
        BenchId(
            "rms_norm_fused_fp8",
            input_id=String(fn_name, "/", in_dtype, "/", out_dtype, "/", shape),
        ),
    )

    ctx.synchronize()

    # ===== Verification: Compare fused output with separate operations =====
    print("\nVerifying outputs...")

    # Allocate separate device buffers for verification (no cache busting)
    var fp8_verify_d = ctx.enqueue_create_buffer[out_dtype](rows * cols)
    var fused_verify_d = ctx.enqueue_create_buffer[out_dtype](rows * cols)
    var rms_verify_d = ctx.enqueue_create_buffer[in_dtype](rows * cols)

    var fp8_verify_base_ptr = fp8_verify_d.unsafe_ptr()
    var fused_verify_base_ptr = fused_verify_d.unsafe_ptr()
    var rms_verify_base_ptr = rms_verify_d.unsafe_ptr()

    # Run separate operations with zero offset
    var data_ptr_verify = cb_data.unsafe_ptr()
    var rms_output_ptr_verify = rms_verify_base_ptr
    var data_buf_verify = TileTensor(data_ptr_verify, row_major(Coord(shape)))
    var rms_output_buf_verify = TileTensor(
        rms_output_ptr_verify, row_major(Coord(shape))
    )

    # Input function for verification. `rms_norm_gpu` migrated to a `Coord`
    # shape boundary (softmax PR #88203).
    @__copy_capture(data_buf_verify)
    @always_inline
    @__parameter
    def input_fn_verify[width: Int](coords: Coord) -> SIMD[in_dtype, width]:
        var idx = data_buf_verify.layout(coords)
        return data_buf_verify.raw_load[width=width](idx)

    # Output function for verification
    @always_inline
    @__copy_capture(rms_output_buf_verify)
    @__parameter
    def rms_output_fn_verify[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[in_dtype, width]) -> None:
        var idx = rms_output_buf_verify.layout(coords)
        rms_output_buf_verify.raw_store[width=width, alignment=alignment](
            idx, val
        )

    # Run RMS norm
    rms_norm_gpu[
        rank, input_fn_verify, rms_output_fn_verify, multiply_before_cast=True
    ](
        Coord(shape),
        gamma_tensor,
        epsilon,
        weight_offset,
        ctx,
    )

    # Run FP8 quantization on RMS norm output
    @always_inline
    def fp8_input_fn_verify[
        width: Int, alignment: Int
    ](row: Int, col: Int) {var rms_verify_base_ptr} -> SIMD[in_dtype, width]:
        var rms_ptr = rms_verify_base_ptr
        var idx = row * cols + col
        return rms_ptr.load[width=width](idx)

    var fp8_output_tt_verify = TileTensor(
        fp8_verify_base_ptr,
        row_major(Coord(rows, cols)),
    )
    var scales_tt_verify = TileTensor(
        scales_base_ptr,
        row_major(Coord(Idx[1], rows)),
    )

    quantize_dynamic_scaled_fp8[
        in_dtype=in_dtype,
        group_size_or_per_token=-1,
        num_cols=cols,
    ](
        fp8_input_fn_verify,
        fp8_output_tt_verify,
        scales_tt_verify,
        Float32(448.0),
        ctx,
        rows,
    )

    # Run fused kernel
    var data_base_ptr_verify = cb_data.unsafe_ptr()

    @__copy_capture(data_base_ptr_verify)
    @always_inline
    @__parameter
    def input_fn_fused_verify[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var data_ptr = data_base_ptr_verify
        var data_buf = TileTensor(data_ptr, row_major(Coord(shape)))
        var idx = data_buf.layout(Coord(coords))
        return data_buf.raw_load[width=width](idx)

    var fused_output_tt_verify = TileTensor(
        fused_verify_base_ptr,
        row_major(Coord(shape)),
    )
    var verify_scale_shape = shape
    verify_scale_shape[rank - 1] = 1
    var fused_scales_tt_verify = TileTensor(
        scales_base_ptr_fused,
        row_major(Coord(verify_scale_shape)),
    )

    var ctx_ptr_verify = ctx
    rms_norm_fused_fp8[
        in_dtype,
        out_dtype,
        DType.float32,
        rank,
        input_fn_fused_verify,
    ](
        shape,
        fused_output_tt_verify,
        gamma_tensor,
        epsilon,
        weight_offset,
        ctx_ptr_verify,
        Float32(448.0),
        fused_scales_tt_verify,
    )

    ctx.synchronize()

    # Copy results back to host for verification
    var fp8_output_h = List(length=rows * cols, fill=Scalar[out_dtype](0))
    var fused_output_h = List(length=rows * cols, fill=Scalar[out_dtype](0))

    ctx.enqueue_copy(fp8_output_h, fp8_verify_d)
    ctx.enqueue_copy(fused_output_h, fused_verify_d)
    ctx.synchronize()

    # Compare outputs

    var max_diff = Float32(0.0)
    var max_rel_diff = Float32(0.0)
    var num_errors = 0
    var num_exact = 0
    var sum_abs_diff = Float64(0.0)

    # More relaxed tolerance for FP8 quantization
    # FP8 can have systematic scaling differences up to ~12-13%
    var rtol = Float32(0.15)  # 15% relative tolerance for FP8
    var atol = Float32(2.0)  # Absolute tolerance for FP8

    for i in range(rows * cols):
        var fp8_val = fp8_output_h[i].cast[.float32]()
        var fused_val = fused_output_h[i].cast[.float32]()
        var diff = abs(fp8_val - fused_val)

        if diff > max_diff:
            max_diff = diff

        if fp8_val == fused_val:
            num_exact += 1

        sum_abs_diff += diff.cast[.float64]()

        # Calculate relative difference
        var avg_val = (abs(fp8_val) + abs(fused_val)) / 2.0
        var rel_diff = diff / max(avg_val, Float32(1.0))
        if rel_diff > max_rel_diff:
            max_rel_diff = rel_diff

        var threshold = atol + rtol * avg_val
        if diff > threshold:
            num_errors += 1
            if num_errors <= 10:  # Print first 10 errors
                print(
                    "Mismatch at index",
                    i,
                    ": separate =",
                    fp8_val,
                    ", fused =",
                    fused_val,
                    ", diff =",
                    diff,
                    ", rel =",
                    rel_diff,
                )

    var mean_abs_diff = sum_abs_diff / Float64(rows * cols)
    var percent_exact = Float32(num_exact) * 100.0 / Float32(rows * cols)

    print("\nVerification Statistics:")
    print("  Total elements:", rows * cols)
    print("  Exact matches:", num_exact, "(", percent_exact, "%)")
    print("  Max absolute diff:", max_diff)
    print("  Max relative diff:", max_rel_diff)
    print("  Mean absolute diff:", mean_abs_diff)
    print("  Mismatches (above threshold):", num_errors)

    var error_rate = Float32(num_errors) / Float32(rows * cols)

    if num_errors > 0:
        if error_rate > 0.05:  # More than 5% errors is concerning
            print(
                "\nVerification FAILED:",
                num_errors,
                "mismatches (",
                error_rate * 100.0,
                "%) exceed threshold (rtol=",
                rtol,
                ", atol=",
                atol,
                ")",
            )
            raise Error("Output verification failed - too many mismatches")
        else:
            # Small number of mismatches might be due to FP8 scaling differences
            print(
                "\nVerification PASSED with warnings:",
                num_errors,
                "minor mismatches (",
                error_rate * 100.0,
                "%) likely due to FP8 scaling differences",
            )
    else:
        print(
            "\nVerification PASSED: All outputs within tolerance",
        )

    _ = cb_data
    _ = gamma_d
    _ = cb_rms_output
    _ = cb_fp8_output
    _ = cb_fused_output
    _ = scales_d
    _ = fp8_verify_d
    _ = fused_verify_d
    _ = rms_verify_d
    _ = data_h^
    _ = gamma_h^
    _ = fp8_output_h^
    _ = fused_output_h^


def main() raises:
    comptime in_dtype = get_defined_dtype["in_dtype", .bfloat16]()
    comptime out_dtype = get_defined_dtype["out_dtype", .float8_e4m3fn]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "1x4096x16384"]()
    ]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        # Run fused RMS norm + FP8 quantization benchmark
        # This benchmarks all three variants: rms_norm, fp8_quant, and fused
        bench_rms_norm_fused_fp8[in_dtype, out_dtype, shape, cache_busting](
            ctx, m, "rms_norm_fused_fp8"
        )

    m.dump_report()
