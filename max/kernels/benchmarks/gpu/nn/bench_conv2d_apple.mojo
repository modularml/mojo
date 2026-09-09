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
"""Apple M5: conv2d performance, THREE paths on identical shapes.

Times bf16 NHWC conv2d for:
  (a) FUSED online-im2col  -- `dispatch_fused_im2col_conv2d_apple`
      (A operand gathered from NHWC per MMA fragment, no `[M, K]` scratch).
  (b) MATERIALISE im2col   -- `dispatch_im2col_matmul_conv2d`
      (builds an `[M, K]` im2col matrix in global memory, then the GEMM).
  (c) NAIVE thread/pixel   -- `conv2d_gpu_naive_nhwc_rscf`.

Standalone `perf_counter` harness (warmup + hot loop), matching
`bench_apple_gpu_matmul.mojo` -- Apple is not on the shared `std.benchmark`
infra. Reports per-call kernel time (ms) and TFLOP/s for each path, plus the
two decision ratios:
  - fused/naive       (MUST be >= ~1.0x on the memory-bound C32->32 @512^2
                       shape -- that is what justified deleting the old
                       materialise->naive guard).
  - fused/materialise (SHOULD be within ~15% on compute-bound shapes; the
                       online gather pays per-fragment im2col address math
                       that the clean dense GEMM does not).

Run with: `mojo max/kernels/benchmarks/gpu/nn/bench_conv2d_apple.mojo`.
"""

from std.math import ceildiv
from std.time import perf_counter

from layout import (
    Idx,
    LTToTTLayout,
    Layout,
    LayoutTensor,
    TileTensor,
)
from max.gpu.host import DeviceContext
from nn.conv.conv import conv2d_gpu_naive_nhwc_rscf
from nn.conv.gpu.im2col_matmul_2d import (
    dispatch_fused_im2col_conv2d_apple,
    dispatch_im2col_matmul_conv2d,
)

from std.utils.index import IndexList


def bench_shape[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[2],
    pad: IndexList[2],
    warmup: Int = 20,
    hot: Int = 30,
](ctx: DeviceContext, name: StaticString) raises:
    comptime N = Int(input_layout.shape[0])
    comptime H = Int(input_layout.shape[1])
    comptime W = Int(input_layout.shape[2])
    comptime C = Int(input_layout.shape[3])

    comptime R = Int(filter_layout.shape[0])
    comptime S = Int(filter_layout.shape[1])
    comptime F = Int(filter_layout.shape[3])

    comptime H_out = (H + 2 * pad[0] - (R - 1) - 1) // stride[0] + 1
    comptime W_out = (W + 2 * pad[1] - (S - 1) - 1) // stride[1] + 1
    comptime output_layout = Layout.row_major(N, H_out, W_out, F)

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (output_layout.size())

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    var input_tt = TileTensor(
        input_dev.unsafe_ptr(), LTToTTLayout[input_layout]()
    )
    var filter_tt = TileTensor(
        filter_dev.unsafe_ptr(), LTToTTLayout[filter_layout]()
    )
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(), LTToTTLayout[output_layout]()
    )

    # 2 * M * N * K flops (MAC = 2).
    var flops = Float64(2 * N * H_out * W_out * F * R * S * C)

    # ---- (a) FUSED online-im2col ----
    for _ in range(warmup):
        _ = dispatch_fused_im2col_conv2d_apple(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )
    ctx.synchronize()
    var t_fused = perf_counter()
    for _ in range(hot):
        _ = dispatch_fused_im2col_conv2d_apple(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )
    ctx.synchronize()
    var fused_ms = (perf_counter() - t_fused) / Float64(hot) * 1e3

    # ---- (b) MATERIALISE im2col + AppleM5MatMul ----
    for _ in range(warmup):
        _ = dispatch_im2col_matmul_conv2d(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )
    ctx.synchronize()
    var t_mat = perf_counter()
    for _ in range(hot):
        _ = dispatch_im2col_matmul_conv2d(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )
    ctx.synchronize()
    var mat_ms = (perf_counter() - t_mat) / Float64(hot) * 1e3

    # ---- (c) NAIVE thread-per-pixel kernel ----
    var input_lt = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_lt = LayoutTensor[dtype, filter_layout](filter_dev.unsafe_ptr())
    var output_lt = LayoutTensor[dtype, output_layout](output_dev.unsafe_ptr())

    comptime block_size = 16
    comptime naive = conv2d_gpu_naive_nhwc_rscf[
        input_layout,
        filter_layout,
        output_layout,
        dtype,
        dtype,
        dtype,
        block_size,
        None,
    ]
    var gx = ceildiv(W_out, block_size)
    var gy = ceildiv(H_out, block_size)
    var gz = N

    @__parameter
    @always_inline
    def run_naive() raises:
        ctx.enqueue_function[naive](
            input_lt,
            filter_lt,
            output_lt,
            stride,
            IndexList[2](1, 1),
            pad,
            Int32(1),
            grid_dim=(gx, gy, gz),
            block_dim=(block_size, block_size),
        )

    for _ in range(warmup):
        run_naive()
    ctx.synchronize()
    var t_naive = perf_counter()
    for _ in range(hot):
        run_naive()
    ctx.synchronize()
    var naive_ms = (perf_counter() - t_naive) / Float64(hot) * 1e3

    var fused_tflops = flops / (fused_ms * 1e-3) / 1e12
    var mat_tflops = flops / (mat_ms * 1e-3) / 1e12
    var naive_tflops = flops / (naive_ms * 1e-3) / 1e12

    print(name)
    print(
        "  (a) FUSED       : ",
        fused_ms,
        " ms   (",
        fused_tflops,
        " TFLOP/s)",
    )
    print(
        "  (b) MATERIALISE : ",
        mat_ms,
        " ms   (",
        mat_tflops,
        " TFLOP/s)",
    )
    print(
        "  (c) NAIVE       : ",
        naive_ms,
        " ms   (",
        naive_tflops,
        " TFLOP/s)",
    )
    # Ratios > 1.0 mean fused is FASTER than the comparison path.
    print(
        "  ratio fused/naive       (naive_ms / fused_ms): ",
        naive_ms / fused_ms,
        "x",
    )
    print(
        "  ratio fused/materialise (mat_ms   / fused_ms): ",
        mat_ms / fused_ms,
        "x",
    )

    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        print("== bench_conv2d_apple (M5, bf16, warmup=20 hot=30) ==")
        print("== three paths: (a) FUSED  (b) MATERIALISE  (c) NAIVE ==")

        # ---- KB regime points (compute-bound spectrum) ----
        # C128->256 @128^2: large GEMM, compute-bound. Materialise should be
        # near-peak; fused must stay within ~15%.
        bench_shape[
            Layout.row_major(1, 128, 128, 128),
            Layout.row_major(3, 3, 128, 256),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,128,128,128] 3x3 s1 C128->256 (compute-bound)")

        # C128->128 @64^2.
        bench_shape[
            Layout.row_major(1, 64, 64, 128),
            Layout.row_major(3, 3, 128, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,64,64,128] 3x3 s1 C128->128 (compute-bound)")

        # C64->64 @256^2: mid-K, high spatial. Crossover region.
        bench_shape[
            Layout.row_major(1, 256, 256, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,256,256,64] 3x3 s1 C64->64 (crossover)")

        # ---- Memory-bound regime (the verdict-bar shape) ----
        # C32->32 @512^2: low C, very high spatial resolution. Materialise's
        # [M,K] round-trip lost to naive here (~0.28x). Fused MUST be >= ~1.0x
        # of naive or the guard deletion is unsafe.
        bench_shape[
            Layout.row_major(1, 512, 512, 32),
            Layout.row_major(3, 3, 32, 32),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,512,512,32] 3x3 s1 C32->32 @512^2 (MEM-BOUND verdict bar)")

        # ---- FLUX.2-VAE decoder shapes (production) ----
        # Decoder upsamples spatial while reducing channels; 3x3 s1 same-pad.
        # 128ch block at 256^2.
        bench_shape[
            Layout.row_major(1, 256, 256, 128),
            Layout.row_major(3, 3, 128, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,256,256,128] 3x3 s1 C128->128 (FLUX-VAE)")

        # 256->128 at 256^2 (channel reduction stage).
        bench_shape[
            Layout.row_major(1, 256, 256, 256),
            Layout.row_major(3, 3, 256, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,256,256,256] 3x3 s1 C256->128 (FLUX-VAE)")

        # 128->64 at 512^2 (late decoder, low-C high-res).
        bench_shape[
            Layout.row_major(1, 512, 512, 128),
            Layout.row_major(3, 3, 128, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "[1,512,512,128] 3x3 s1 C128->64 (FLUX-VAE late)")

        print("DONE")
