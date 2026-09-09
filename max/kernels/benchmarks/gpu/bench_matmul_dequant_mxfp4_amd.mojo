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
"""Benchmark for MXFP4 dequant-then-FP8 matmul on AMD CDNA GPUs.

Benchmarks the full mxfp4_dequant_matmul_amd pipeline:
  1. Dequant MXFP4 packed uint8 weights + E8M0 scales to FP8
  2. Cast BF16 activations to FP8
  3. FP8 GEMM via _matmul_gpu

Usage:
  br //max/kernels/benchmarks:bench_matmul_mxfp4.mojo.test -- --M 256
  br //max/kernels/benchmarks:bench_matmul_mxfp4.mojo.test -- --M 256 --run_benchmark false --verify true

Compile-time parameters (set via --define):
  N: Weight rows / output columns (default: 2880)
  K: Inner dimension, unpacked (default: 2880)

Runtime arguments:
  --M: Batch size / activation rows (default: 256)
  --verify: Run verification against vendor BLAS (default: false)
  --run_benchmark: Run benchmark iterations (default: true)
"""

from std.math import ceildiv
from std.sys import get_defined_int, size_of
from std.random import rand, randint
from std.memory import bitcast
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import arg_parse
from internal_utils._utils import InitializationType, init_vector_launch
from layout import Idx, Layout, LayoutTensor, TileTensor, row_major

import linalg.matmul.vendor.blas as vendor_blas
from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import (
    mxfp4_dequant_matmul_amd,
)
from linalg.mxfp4_dequant import dequant_mxfp4


def _fill_random_mxfp4_data[
    N: Int, K: Int
](
    ctx: DeviceContext,
    a_device: DeviceBuffer[.bfloat16],
    b_packed_device: DeviceBuffer[.uint8],
    b_scales_device: DeviceBuffer[.float8_e8m0fnu],
    M: Int,
) raises:
    """Fill input buffers with random data."""
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    # BF16 activations: random on device
    init_vector_launch[.bfloat16](
        a_device,
        M * K,
        InitializationType.uniform_distribution,
        ctx,
    )

    # Packed FP4 weights: random on device
    init_vector_launch[.uint8](
        b_packed_device,
        N * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )

    # E8M0 scales: fill on host then copy.
    # float8_e8m0fnu has no zero representation so init_vector_launch crashes.
    # Use exponent 127 (scale=1.0) for all scales.
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(127))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()


def verify_mxfp4_matmul[
    N: Int, K: Int
](
    ctx: DeviceContext,
    c_device: DeviceBuffer[.bfloat16],
    a_device: DeviceBuffer[.bfloat16],
    b_packed_device: DeviceBuffer[.uint8],
    b_scales_device: DeviceBuffer[.float8_e8m0fnu],
    M: Int,
) raises:
    """Verify mxfp4_dequant_matmul_amd output against vendor BLAS on dequanted FP8 data.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    var b_fp8 = ctx.enqueue_create_buffer[fp8_type](N * K)
    var a_fp8 = ctx.enqueue_create_buffer[fp8_type](M * K)

    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())
    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_fp8_tt = TileTensor(b_fp8, row_major[N, K]())
    var a_fp8_tt = TileTensor(a_fp8, row_major((M, Idx[K])))

    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import _cast_bf16_to_fp8

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed_tt,
        b_scales_tt,
        num_rows=N,
        num_cols=K,
    )
    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)
    ctx.synchronize()

    var c_ref = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_ref_lt = LayoutTensor[.bfloat16, Layout.row_major(N, N)](c_ref)

    vendor_blas.matmul(
        ctx,
        c_ref_lt,
        a_fp8_tt.to_layout_tensor(),
        b_fp8_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )
    ctx.synchronize()

    var c_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_host, c_device)
    ctx.enqueue_copy(c_ref_host, c_ref)
    ctx.synchronize()

    var sum_abs_diff = Float64(0.0)
    var sum_abs_ref = Float64(0.0)
    for i in range(M * N):
        var got = c_host[i].cast[.float64]()
        var exp = c_ref_host[i].cast[.float64]()
        sum_abs_diff += abs(got - exp)
        sum_abs_ref += abs(exp)

    var rel_diff = sum_abs_diff / max(sum_abs_ref, Float64(1e-12))
    print("  Verification: relative_difference =", rel_diff)
    if rel_diff > 0.01:
        raise String("MXFP4 matmul verification failed: rel_diff=", rel_diff)
    print("  PASSED")


def bench_dequant_mxfp4[
    N: Int, K: Int
](ctx: DeviceContext, mut b: Bench,) raises:
    """Benchmark dequant_mxfp4 (packed uint8 + E8M0 scales -> FP8) independently.
    """
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)
    comptime fp8_type = DType.float8_e4m3fn

    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )
    var b_fp8_device = ctx.enqueue_create_buffer[fp8_type](N * K)

    init_vector_launch[.uint8](
        b_packed_device,
        N * packed_K,
        InitializationType.uniform_distribution,
        ctx,
    )
    var bs_hbuf = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](N * scale_K)
    for i in range(N * scale_K):
        bs_hbuf[i] = bitcast[.float8_e8m0fnu](UInt8(127))
    ctx.enqueue_copy(b_scales_device, bs_hbuf)
    ctx.synchronize()

    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())
    var b_fp8_tt = TileTensor(b_fp8_device, row_major((Idx[N], Idx[K])))

    @always_inline
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut b_fp8_tt, imm}:
        dequant_mxfp4(
            ctx,
            b_fp8_tt,
            b_packed_tt,
            b_scales_tt,
            num_rows=N,
            num_cols=K,
        )

    @always_inline
    def bench_func(mut bencher: Bencher) raises {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    # Memory traffic: read packed (N*K/2) + scales (N*K/32), write FP8 (N*K)
    comptime total_bytes = N * packed_K + N * scale_K + N * K * size_of[
        fp8_type
    ]()
    var bandwidth = ThroughputMeasure(BenchMetric.bytes, total_bytes)

    b.bench_function(
        bench_func,
        BenchId(String("dequant_mxfp4(N=", N, ",K=", K, ")")),
        [bandwidth],
    )


def bench_cast_bf16_to_fp8[
    K: Int
](ctx: DeviceContext, mut b: Bench, M: Int,) raises:
    """Benchmark BF16 to FP8 activation cast independently."""
    comptime fp8_type = DType.float8_e4m3fn

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var a_fp8_device = ctx.enqueue_create_buffer[fp8_type](M * K)

    init_vector_launch[.bfloat16](
        a_device,
        M * K,
        InitializationType.uniform_distribution,
        ctx,
    )

    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8_device, row_major((M, Idx[K])))

    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import _cast_bf16_to_fp8

    @always_inline
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut a_fp8_tt, imm}:
        _cast_bf16_to_fp8(ctx, a_fp8_tt, a_tt, M, K)

    @always_inline
    def bench_func(mut bencher: Bencher) raises {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    # Memory traffic: read BF16 (M*K*2), write FP8 (M*K*1)
    var total_bytes = M * K * (size_of[DType.bfloat16]() + size_of[fp8_type]())
    var bandwidth = ThroughputMeasure(BenchMetric.bytes, total_bytes)

    b.bench_function(
        bench_func,
        BenchId(String("cast_bf16_to_fp8(M=", M, ",K=", K, ")")),
        [bandwidth],
    )


def bench_fp8_matmul[
    N: Int, K: Int
](ctx: DeviceContext, mut b: Bench, M: Int,) raises:
    """Benchmark _matmul_gpu on FP8 data (pure GEMM, no dequant)."""
    comptime fp8_type = DType.float8_e4m3fn

    # Init as BF16 then cast to FP8 to avoid init_vector_launch FP8 issues
    var a_bf16 = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_bf16 = ctx.enqueue_create_buffer[.bfloat16](N * K)
    init_vector_launch[.bfloat16](
        a_bf16,
        M * K,
        InitializationType.uniform_distribution,
        ctx,
    )
    init_vector_launch[.bfloat16](
        b_bf16,
        N * K,
        InitializationType.uniform_distribution,
        ctx,
    )

    var a_fp8 = ctx.enqueue_create_buffer[fp8_type](M * K)
    var b_fp8 = ctx.enqueue_create_buffer[fp8_type](N * K)
    var a_bf16_tt = TileTensor(a_bf16, row_major((M, Idx[K])))
    var b_bf16_tt = TileTensor(b_bf16, row_major((Idx[N], Idx[K])))
    var a_fp8_tt = TileTensor(a_fp8, row_major((M, Idx[K])))
    var b_fp8_tt = TileTensor(b_fp8, row_major((Idx[N], Idx[K])))

    from linalg.matmul.gpu.amd.mxfp4_dequant_matmul_amd import _cast_bf16_to_fp8

    _cast_bf16_to_fp8(ctx, a_fp8_tt, a_bf16_tt, M, K)
    _cast_bf16_to_fp8(ctx, b_fp8_tt, b_bf16_tt, N, K)
    ctx.synchronize()

    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))

    from linalg.matmul.gpu import _matmul_gpu

    @always_inline
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut c_tt, imm}:
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            c_tt, a_fp8_tt, b_fp8_tt, ctx
        )

    @always_inline
    def bench_func(mut bencher: Bencher) raises {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    var flops = ThroughputMeasure(BenchMetric.flops, 2 * M * N * K)

    b.bench_function(
        bench_func,
        BenchId(String("fp8_matmul(M=", M, ",N=", N, ",K=", K, ")")),
        [flops],
    )


def bench_mxfp4_matmul[
    N: Int, K: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    M: Int,
    verify: Bool,
    run_benchmark: Bool,
) raises:
    comptime packed_K = K // 2
    comptime scale_K = ceildiv(K, 32)

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_packed_device = ctx.enqueue_create_buffer[.uint8](N * packed_K)
    var b_scales_device = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        N * scale_K
    )
    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)

    _fill_random_mxfp4_data[N, K](
        ctx, a_device, b_packed_device, b_scales_device, M
    )

    var a_tt = TileTensor(a_device, row_major((M, Idx[K])))
    var b_packed_tt = TileTensor(b_packed_device, row_major[N, packed_K]())
    var b_scales_tt = TileTensor(b_scales_device, row_major[N, scale_K]())
    var c_tt = TileTensor(c_device, row_major((M, Idx[N])))

    @always_inline
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut c_tt, imm}:
        mxfp4_dequant_matmul_amd(c_tt, a_tt, b_packed_tt, b_scales_tt, ctx)
        ctx.synchronize()

    if run_benchmark:

        @always_inline
        def bench_func(mut bencher: Bencher) raises {imm}:
            bencher_iter_custom(bencher, kernel_launch, ctx)

        var flops = ThroughputMeasure(BenchMetric.flops, 2 * M * N * K)

        b.bench_function(
            bench_func,
            BenchId(
                String(
                    "mxfp4_matmul(M=",
                    M,
                    ",N=",
                    N,
                    ",K=",
                    K,
                    ")",
                )
            ),
            [flops],
        )
    else:
        kernel_launch(ctx, 0)

    if verify:
        verify_mxfp4_matmul[N, K](
            ctx,
            c_device,
            a_device,
            b_packed_device,
            b_scales_device,
            M,
        )


def main() raises:
    var M = Int(arg_parse("M", 256))
    comptime N = get_defined_int["N", 2880]()
    comptime K = get_defined_int["K", 2880]()
    var verify = arg_parse("verify", False)
    var run_benchmark = arg_parse("run_benchmark", True)
    var bench_all = arg_parse("bench_all", True)

    var b = Bench()
    with DeviceContext() as ctx:
        if run_benchmark:
            if bench_all:
                bench_dequant_mxfp4[N, K](ctx, b)
                bench_cast_bf16_to_fp8[K](ctx, b, M)
                bench_fp8_matmul[N, K](ctx, b, M)
            bench_mxfp4_matmul[N, K](ctx, b, M, verify, run_benchmark)

    if run_benchmark:
        b.dump_report()
