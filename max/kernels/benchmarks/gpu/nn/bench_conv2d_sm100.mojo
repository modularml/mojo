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
"""Benchmark SM100 Conv2D kernel against cuDNN.

Compares the SM100 structured conv2d kernel performance against cuDNN
for various problem sizes typical of VAE decoder workloads.

Both SM100 and cuDNN timings include the full conv2d operation with im2col.
SM100 uses hardware TMA im2col (CUTLASS pattern) for memory-efficient
coordinate transformation.

Usage:
    mojo max/kernels/benchmarks/gpu/nn/bench_conv2d_sm100.mojo

The benchmark reports:
- Time per iteration (ms)
- Throughput (TFLOPS)
- Comparison ratio (SM100 / cuDNN)
"""

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.conv.gpu.nvidia.sm100.conv2d import (
    conv2d_fprop,
    conv2d_fprop_with_residual,
)
from nn.conv.gpu.nvidia.sm100.conv_config import (
    Conv2dConfig,
    Conv2dProblemShape,
)
from nn.conv.conv import conv_cudnn
from std.random import rand

from std.utils.index import IndexList


def compute_conv_flops(
    batch: Int,
    out_height: Int,
    out_width: Int,
    out_channels: Int,
    in_channels: Int,
    filter_h: Int,
    filter_w: Int,
) -> Int:
    """Compute FLOPs for conv2d: 2 * N * H_out * W_out * C_out * C_in * R * S.
    """
    return (
        2
        * batch
        * out_height
        * out_width
        * out_channels
        * in_channels
        * filter_h
        * filter_w
    )


def bench_conv2d[
    dtype: DType,
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
](ctx: DeviceContext, num_iters: Int = 100, warmup_iters: Int = 10) raises:
    """Benchmark SM100 conv2d vs cuDNN using proper GPU timing.

    Both SM100 and cuDNN timings include the full conv2d operation.
    SM100 uses hardware TMA im2col (CUTLASS pattern) for memory-efficient
    coordinate transformation.
    """

    # Derived dimensions
    comptime out_height = in_height + 2 * pad_h - filter_h + 1
    comptime out_width = in_width + 2 * pad_w - filter_w + 1

    # Sizes
    comptime input_size = batch * in_height * in_width * in_channels
    comptime filter_size = out_channels * filter_h * filter_w * in_channels
    comptime output_size = batch * out_height * out_width * out_channels

    # FLOPs
    var flops = compute_conv_flops(
        batch,
        out_height,
        out_width,
        out_channels,
        in_channels,
        filter_h,
        filter_w,
    )
    print(
        "Conv2D: batch=",
        batch,
        " in=(",
        in_height,
        "x",
        in_width,
        "x",
        in_channels,
        ") filter=(",
        filter_h,
        "x",
        filter_w,
        ") out=(",
        out_height,
        "x",
        out_width,
        "x",
        out_channels,
        ") pad=(",
        pad_h,
        ",",
        pad_w,
        ") GFLOPS=",
        Float64(flops) / 1e9,
        sep="",
    )

    # Problem shape for implicit im2col
    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_height,
        in_width=in_width,
        in_channels=in_channels,
        out_channels=out_channels,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    # Allocate host memory
    var input_host_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var filter_host_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    var filter_nchw_host_ptr = List(length=filter_size, fill=Scalar[dtype](0))

    # Initialize with random data
    rand(input_host_ptr)
    rand(filter_host_ptr)

    # Convert filter KRSC -> NCHW for cuDNN
    for k in range(out_channels):
        for r in range(filter_h):
            for s in range(filter_w):
                for c in range(in_channels):
                    var krsc_idx = (
                        k * filter_h * filter_w * in_channels
                        + r * filter_w * in_channels
                        + s * in_channels
                        + c
                    )
                    var nchw_idx = (
                        k * in_channels * filter_h * filter_w
                        + c * filter_h * filter_w
                        + r * filter_w
                        + s
                    )
                    filter_nchw_host_ptr[nchw_idx] = filter_host_ptr[krsc_idx]

    # Allocate device memory
    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_nchw_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_sm100_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_cudnn_dev = ctx.enqueue_create_buffer[dtype](output_size)

    # Copy to device
    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_dev, filter_host_ptr)
    ctx.enqueue_copy(filter_nchw_dev, filter_nchw_host_ptr)
    ctx.synchronize()

    # Create TileTensor views for conv2d_fprop
    var input_tt = TileTensor(
        input_dev,
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_tt = TileTensor(
        filter_dev,
        row_major(
            Coord(IndexList[4](out_channels, filter_h, filter_w, in_channels))
        ),
    )
    var output_sm100_tt = TileTensor(
        output_sm100_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )

    # Create TileTensor views for cuDNN
    var input_dev_tensor = TileTensor(
        input_dev,
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_nchw_dev_tensor = TileTensor(
        filter_nchw_dev,
        row_major(
            Coord(IndexList[4](out_channels, in_channels, filter_h, filter_w))
        ),
    )
    var output_cudnn_dev_tensor = TileTensor(
        output_cudnn_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )

    # ==================== Warmup ====================
    for _ in range(warmup_iters):
        conv2d_fprop(output_sm100_tt, input_tt, filter_tt, problem, ctx)
        conv_cudnn[dtype, dtype, dtype](
            input_dev_tensor,
            filter_nchw_dev_tensor,
            output_cudnn_dev_tensor,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](pad_h, pad_w),
            1,
            ctx,
        )
    ctx.synchronize()

    # ==================== Benchmark SM100 implicit im2col ====================
    def sm100_implicit_kernel() raises {
        var input_tt, var filter_tt, var output_sm100_tt, imm
    }:
        conv2d_fprop(output_sm100_tt, input_tt, filter_tt, problem, ctx)

    var sm100_time_ns = ctx.execution_time(sm100_implicit_kernel, num_iters)
    var sm100_time_ms = Float64(sm100_time_ns) / 1e6 / Float64(num_iters)
    var sm100_tflops = Float64(flops) / (sm100_time_ms / 1000) / 1e12

    # ==================== Benchmark cuDNN ====================
    def cudnn_kernel() raises {
        var input_dev_tensor,
        var filter_nchw_dev_tensor,
        var output_cudnn_dev_tensor,
        imm,
    }:
        conv_cudnn[dtype, dtype, dtype](
            input_dev_tensor,
            filter_nchw_dev_tensor,
            output_cudnn_dev_tensor,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](pad_h, pad_w),
            1,
            ctx,
        )

    var cudnn_time_ns = ctx.execution_time(cudnn_kernel, num_iters)
    var cudnn_time_ms = Float64(cudnn_time_ns) / 1e6 / Float64(num_iters)
    var cudnn_tflops = Float64(flops) / (cudnn_time_ms / 1000) / 1e12

    # Verify outputs match
    var output_sm100_host_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var output_cudnn_host_ptr = List(length=output_size, fill=Scalar[dtype](0))
    ctx.enqueue_copy(output_sm100_host_ptr, output_sm100_dev)
    ctx.enqueue_copy(output_cudnn_host_ptr, output_cudnn_dev)
    ctx.synchronize()

    var max_diff: Float32 = 0.0
    for i in range(output_size):
        var diff = abs(
            output_sm100_host_ptr[i].cast[.float32]()
            - output_cudnn_host_ptr[i].cast[.float32]()
        )
        if diff > max_diff:
            max_diff = diff

    # Report
    var ratio = cudnn_time_ms / sm100_time_ms
    print(
        "  SM100 Im2col: ",
        sm100_time_ms,
        " ms  (",
        sm100_tflops,
        " TFLOPS)",
        sep="",
    )
    print(
        "  cuDNN conv2d: ",
        cudnn_time_ms,
        " ms  (",
        cudnn_tflops,
        " TFLOPS)",
        sep="",
    )
    print(
        "  Ratio: ",
        ratio,
        "x",
        "  (Both include full conv with im2col)",
        sep="",
    )
    print("  Correctness: max_diff=", max_diff, sep="")
    print()

    _ = input_dev^
    _ = filter_dev^
    _ = filter_nchw_dev^
    _ = output_sm100_dev^
    _ = output_cudnn_dev^
    _ = filter_nchw_host_ptr^
    _ = filter_host_ptr^
    _ = input_host_ptr^
    _ = output_cudnn_host_ptr^
    _ = output_sm100_host_ptr^


def bench_all_configs[
    dtype: DType,
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
](ctx: DeviceContext, num_iters: Int = 100, warmup_iters: Int = 10) raises:
    """Benchmark 1-SM vs 2-SM vs cuDNN."""

    comptime out_height = in_height + 2 * pad_h - filter_h + 1
    comptime out_width = in_width + 2 * pad_w - filter_w + 1
    comptime input_size = batch * in_height * in_width * in_channels
    comptime filter_size = out_channels * filter_h * filter_w * in_channels
    comptime output_size = batch * out_height * out_width * out_channels

    var flops = compute_conv_flops(
        batch,
        out_height,
        out_width,
        out_channels,
        in_channels,
        filter_h,
        filter_w,
    )

    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_height,
        in_width=in_width,
        in_channels=in_channels,
        out_channels=out_channels,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    print(
        "Conv2D: ",
        in_height,
        "x",
        in_width,
        "x",
        in_channels,
        " -> ",
        out_channels,
        "ch (",
        filter_h,
        "x",
        filter_w,
        " filter)  GFLOPS=",
        Float64(flops) / 1e9,
        sep="",
    )

    # Allocate host memory
    var input_host_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var filter_host_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    var filter_nchw_host_ptr = List(length=filter_size, fill=Scalar[dtype](0))

    rand(input_host_ptr)
    rand(filter_host_ptr)

    # Convert filter KRSC -> NCHW for cuDNN
    for k in range(out_channels):
        for r in range(filter_h):
            for s in range(filter_w):
                for c in range(in_channels):
                    var krsc_idx = (
                        k * filter_h * filter_w * in_channels
                        + r * filter_w * in_channels
                        + s * in_channels
                        + c
                    )
                    var nchw_idx = (
                        k * in_channels * filter_h * filter_w
                        + c * filter_h * filter_w
                        + r * filter_w
                        + s
                    )
                    filter_nchw_host_ptr[nchw_idx] = filter_host_ptr[krsc_idx]

    # Allocate device memory
    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_nchw_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_1sm_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_2sm_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_cudnn_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_dev, filter_host_ptr)
    ctx.enqueue_copy(filter_nchw_dev, filter_nchw_host_ptr)
    ctx.synchronize()

    var input_tt = TileTensor(
        input_dev,
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_tt = TileTensor(
        filter_dev,
        row_major(
            Coord(IndexList[4](out_channels, filter_h, filter_w, in_channels))
        ),
    )
    var output_1sm_tt = TileTensor(
        output_1sm_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )
    var output_2sm_tt = TileTensor(
        output_2sm_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )

    # Create TileTensor views for cuDNN
    var input_dev_tensor = TileTensor(
        input_dev,
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_nchw_dev_tensor = TileTensor(
        filter_nchw_dev,
        row_major(
            Coord(IndexList[4](out_channels, in_channels, filter_h, filter_w))
        ),
    )
    var output_cudnn_dev_tensor = TileTensor(
        output_cudnn_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )

    # Get configs
    comptime config_1sm = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm()
    comptime config_2sm = Conv2dConfig[dtype, dtype, dtype].default_bf16()

    # Warmup
    for _ in range(warmup_iters):
        conv2d_fprop[config=config_1sm](
            output_1sm_tt, input_tt, filter_tt, problem, ctx
        )
        conv2d_fprop[config=config_2sm](
            output_2sm_tt, input_tt, filter_tt, problem, ctx
        )
        conv_cudnn[dtype, dtype, dtype](
            input_dev_tensor,
            filter_nchw_dev_tensor,
            output_cudnn_dev_tensor,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](pad_h, pad_w),
            1,
            ctx,
        )
    ctx.synchronize()

    # Benchmark 1-SM
    def kernel_1sm() raises {
        var input_tt, var filter_tt, var output_1sm_tt, imm
    }:
        conv2d_fprop[config=config_1sm](
            output_1sm_tt, input_tt, filter_tt, problem, ctx
        )

    var time_1sm_ns = ctx.execution_time(kernel_1sm, num_iters)
    var time_1sm_ms = Float64(time_1sm_ns) / 1e6 / Float64(num_iters)
    var tflops_1sm = Float64(flops) / (time_1sm_ms / 1000) / 1e12

    # Benchmark 2-SM
    def kernel_2sm() raises {
        var input_tt, var filter_tt, var output_2sm_tt, imm
    }:
        conv2d_fprop[config=config_2sm](
            output_2sm_tt, input_tt, filter_tt, problem, ctx
        )

    var time_2sm_ns = ctx.execution_time(kernel_2sm, num_iters)
    var time_2sm_ms = Float64(time_2sm_ns) / 1e6 / Float64(num_iters)
    var tflops_2sm = Float64(flops) / (time_2sm_ms / 1000) / 1e12

    # Benchmark cuDNN
    def cudnn_kernel() raises {
        var input_dev_tensor,
        var filter_nchw_dev_tensor,
        var output_cudnn_dev_tensor,
        imm,
    }:
        conv_cudnn[dtype, dtype, dtype](
            input_dev_tensor,
            filter_nchw_dev_tensor,
            output_cudnn_dev_tensor,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            IndexList[2](pad_h, pad_w),
            1,
            ctx,
        )

    var cudnn_time_ns = ctx.execution_time(cudnn_kernel, num_iters)
    var cudnn_time_ms = Float64(cudnn_time_ns) / 1e6 / Float64(num_iters)
    var cudnn_tflops = Float64(flops) / (cudnn_time_ms / 1000) / 1e12

    # Report
    print(
        "  1-SM (cta_group=1): ",
        time_1sm_ms,
        " ms  (",
        tflops_1sm,
        " TFLOPS)  ",
        Int(tflops_1sm / cudnn_tflops * 100),
        "% of cuDNN",
        sep="",
    )
    print(
        "  2-SM (cta_group=2): ",
        time_2sm_ms,
        " ms  (",
        tflops_2sm,
        " TFLOPS)  ",
        Int(tflops_2sm / cudnn_tflops * 100),
        "% of cuDNN",
        sep="",
    )
    print(
        "  cuDNN:              ",
        cudnn_time_ms,
        " ms  (",
        cudnn_tflops,
        " TFLOPS)",
        sep="",
    )
    print()

    _ = input_dev^
    _ = filter_dev^
    _ = filter_nchw_dev^
    _ = output_1sm_dev^
    _ = output_2sm_dev^
    _ = output_cudnn_dev^
    _ = filter_nchw_host_ptr^
    _ = filter_host_ptr^
    _ = input_host_ptr^


def bench_residual[
    dtype: DType,
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_h: Int,
    filter_w: Int,
    pad_h: Int,
    pad_w: Int,
](ctx: DeviceContext, num_iters: Int = 100, warmup_iters: Int = 10) raises:
    """Benchmark conv2d vs conv2d+residual (fused) to measure fusion overhead.
    """

    comptime out_height = in_height + 2 * pad_h - filter_h + 1
    comptime out_width = in_width + 2 * pad_w - filter_w + 1
    comptime input_size = batch * in_height * in_width * in_channels
    comptime filter_size = out_channels * filter_h * filter_w * in_channels
    comptime output_size = batch * out_height * out_width * out_channels

    var flops = compute_conv_flops(
        batch,
        out_height,
        out_width,
        out_channels,
        in_channels,
        filter_h,
        filter_w,
    )

    var problem = Conv2dProblemShape(
        batch=batch,
        in_height=in_height,
        in_width=in_width,
        in_channels=in_channels,
        out_channels=out_channels,
        filter_h=filter_h,
        filter_w=filter_w,
        pad_h=pad_h,
        pad_w=pad_w,
    )

    print(
        "Conv2D+Residual: ",
        in_height,
        "x",
        in_width,
        "x",
        in_channels,
        " -> ",
        out_channels,
        "ch (",
        filter_h,
        "x",
        filter_w,
        " filter)  GFLOPS=",
        Float64(flops) / 1e9,
        sep="",
    )

    # Allocate
    var input_host_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var filter_host_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    var source_host_ptr = List(length=output_size, fill=Scalar[dtype](0))
    rand(input_host_ptr)
    rand(filter_host_ptr)
    rand(source_host_ptr)

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_res_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var source_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host_ptr)
    ctx.enqueue_copy(filter_dev, filter_host_ptr)
    ctx.enqueue_copy(source_dev, source_host_ptr)
    ctx.synchronize()

    var input_tt = TileTensor(
        input_dev,
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_tt = TileTensor(
        filter_dev,
        row_major(
            Coord(IndexList[4](out_channels, filter_h, filter_w, in_channels))
        ),
    )
    var output_tt = TileTensor(
        output_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )
    var output_res_tt = TileTensor(
        output_res_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )
    var source_tt = TileTensor(
        source_dev,
        row_major(
            Coord(IndexList[4](batch, out_height, out_width, out_channels))
        ),
    )

    comptime config_1sm = Conv2dConfig[dtype, dtype, dtype].default_bf16_1sm()

    # Warmup
    for _ in range(warmup_iters):
        conv2d_fprop[config=config_1sm](
            output_tt, input_tt, filter_tt, problem, ctx
        )
        conv2d_fprop_with_residual[config=config_1sm, has_residual=True](
            output_res_tt,
            input_tt,
            filter_tt,
            source_tt,
            Float32(1.0),
            problem,
            ctx,
        )
    ctx.synchronize()

    # Benchmark conv2d only
    def kernel_conv() raises {var input_tt, var filter_tt, var output_tt, imm}:
        conv2d_fprop[config=config_1sm](
            output_tt, input_tt, filter_tt, problem, ctx
        )

    var time_conv_ns = ctx.execution_time(kernel_conv, num_iters)
    var time_conv_ms = Float64(time_conv_ns) / 1e6 / Float64(num_iters)
    var tflops_conv = Float64(flops) / (time_conv_ms / 1000) / 1e12

    # Benchmark conv2d + fused residual
    def kernel_residual() raises {
        var input_tt, var filter_tt, var output_res_tt, var source_tt, imm
    }:
        conv2d_fprop_with_residual[config=config_1sm, has_residual=True](
            output_res_tt,
            input_tt,
            filter_tt,
            source_tt,
            Float32(1.0),
            problem,
            ctx,
        )

    var time_res_ns = ctx.execution_time(kernel_residual, num_iters)
    var time_res_ms = Float64(time_res_ns) / 1e6 / Float64(num_iters)
    var tflops_res = Float64(flops) / (time_res_ms / 1000) / 1e12

    # Report
    var overhead_pct = (time_res_ms - time_conv_ms) / time_conv_ms * 100
    print(
        "  Conv only:          ",
        time_conv_ms,
        " ms  (",
        tflops_conv,
        " TFLOPS)",
        sep="",
    )
    print(
        "  Conv+Residual fused:",
        " ",
        time_res_ms,
        " ms  (",
        tflops_res,
        " TFLOPS)",
        sep="",
    )
    print(
        "  Fusion overhead:     ",
        overhead_pct,
        "%",
        sep="",
    )
    print()

    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^
    _ = output_res_dev^
    _ = source_dev^
    _ = source_host_ptr^
    _ = filter_host_ptr^
    _ = input_host_ptr^


def main() raises:
    print("=" * 70)
    print("SM100 CONV2D BENCHMARK: 1-SM vs 2-SM vs cuDNN")
    print("=" * 70)
    print()
    print("Comparing both SM100 configurations against cuDNN.")
    print("SM100 uses hardware TMA im2col (CUTLASS pattern).")
    print("CUTLASS uses 1-SM mode (ClusterShape=1×1×1, TileShape=128×128×64).")
    print()

    with DeviceContext() as ctx:
        print("=== FLUX VAE DECODER LAYERS ===\n")

        # L1: 16x16, 512→512
        print("--- L1: 16x16, 512→512 ch, 3x3 ---")
        bench_all_configs[
            DType.bfloat16,
            batch=1,
            in_height=16,
            in_width=16,
            in_channels=512,
            out_channels=512,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=100)

        # L2: 32x32, 512→256
        print("--- L2: 32x32, 512→256 ch, 3x3 ---")
        bench_all_configs[
            DType.bfloat16,
            batch=1,
            in_height=32,
            in_width=32,
            in_channels=512,
            out_channels=256,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=100)

        # L3: 64x64, 256→128
        print("--- L3: 64x64, 256→128 ch, 3x3 ---")
        bench_all_configs[
            DType.bfloat16,
            batch=1,
            in_height=64,
            in_width=64,
            in_channels=256,
            out_channels=128,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=100)

        # L4: 128x128, 128→128
        print("--- L4: 128x128, 128→128 ch, 3x3 ---")
        bench_all_configs[
            DType.bfloat16,
            batch=1,
            in_height=128,
            in_width=128,
            in_channels=128,
            out_channels=128,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=50)

        print("=" * 70)
        print("RESIDUAL FUSION BENCHMARK")
        print("=" * 70)
        print()
        print("Comparing conv2d vs conv2d+residual (fused).")
        print("Measures the overhead of the fused epilogue load warp.")
        print()

        # Residual benchmarks (1-SM config)
        print("--- 16x16, 128→128 ch, 3x3 ---")
        bench_residual[
            DType.bfloat16,
            batch=1,
            in_height=16,
            in_width=16,
            in_channels=128,
            out_channels=128,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=100)

        print("--- 32x32, 256→256 ch, 3x3 ---")
        bench_residual[
            DType.bfloat16,
            batch=1,
            in_height=32,
            in_width=32,
            in_channels=256,
            out_channels=256,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=100)

        print("--- 64x64, 256→128 ch, 3x3 ---")
        bench_residual[
            DType.bfloat16,
            batch=1,
            in_height=64,
            in_width=64,
            in_channels=256,
            out_channels=128,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=100)

        print("--- 128x128, 128→128 ch, 3x3 ---")
        bench_residual[
            DType.bfloat16,
            batch=1,
            in_height=128,
            in_width=128,
            in_channels=128,
            out_channels=128,
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
        ](ctx, num_iters=50)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
