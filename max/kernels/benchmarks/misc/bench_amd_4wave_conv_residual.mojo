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
"""BF16 production-shape perf for amd_4wave_conv_fprop_with_residual.

One shape per invocation, driven by kbench via compile-time `-D`
defines and the companion YAML
`bench_amd_4wave_conv_residual.yaml`. Each invocation runs three
variants on the SAME (N, H, W, C_in, C_out, R, S, stride, pad) shape:

  1. **Direct** — plain `amd_4wave_conv` (aligned-K shapes only).
  2. **wrapper-no-res** — residual launcher with `has_residual=False`
     (early-out path). Should match direct within ±2%; the wrapper
     is supposed to be transparent when no residual is requested.
  3. **wrapper-fused** — residual launcher with `has_residual=True,
     beta=1.0` (in-kernel residual prefetch + FMA).

Standalone invocation (default values reproduce the `vae-256ch-3x3`
shape) is also supported:

  ```
  mojo max/kernels/benchmarks/misc/bench_amd_4wave_conv_residual.mojo
  ```

To iterate the full FLUX VAE shape suite from the YAML:

  ```
  python benchmarks/autotune/kbench.py \\
      max/kernels/benchmarks/misc/bench_amd_4wave_conv_residual.yaml
  ```

The K-pad path (e.g. FLUX VAE 128ch where `K_real=1152` rounds up to
`K_padded=1280`) is auto-detected from `R*S*C_in % 256`; the direct
baseline is comptime-gated off for K-padded shapes (no aligned-K
reference to compare against).
"""

from max.gpu.host import DeviceContext
from std.random import rand
from std.sys import get_defined_int, get_defined_string
from std.time import perf_counter_ns

from layout import TileTensor, row_major

from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv
from nn.conv.gpu.amd.amd_4wave_conv_residual import (
    amd_4wave_conv_fprop_with_residual,
)
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape


def bench_conv_residual[
    *,
    N_batch: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int = 3,
    S: Int = 3,
    stride: Int = 1,
    pad: Int = 1,
    warmup_iters: Int = 5,
    bench_iters: Int = 30,
    label: StaticString = "",
](ctx: DeviceContext) raises:
    """Bench fused residual vs non-regression (no-residual) vs direct
    `amd_4wave_conv` for one BF16 shape."""
    comptime in_dtype = DType.bfloat16
    comptime H_out = (H + 2 * pad - R) // stride + 1
    comptime W_out = (W + 2 * pad - S) // stride + 1
    comptime M_total = N_batch * H_out * W_out
    comptime K_real = R * S * C_in
    comptime flops_conv = 2 * M_total * K_real * C_out
    # +residual cost: M_total*C_out fp16/bf16 FMAs (negligible vs the
    # conv FLOPs but real for memory-bound shapes).

    # K-pad to multiple of 2*BK = 256 (handled internally by the
    # launcher when K_real isn't already aligned).
    comptime K_padded = ((K_real + 255) // 256) * 256

    print(
        "==",
        label,
        " N=",
        N_batch,
        " H=",
        H,
        " W=",
        W,
        " C_in=",
        C_in,
        " C_out=",
        C_out,
        " RxS=",
        R,
        "x",
        S,
        " s=",
        stride,
        " p=",
        pad,
        " M_total=",
        M_total,
        " K=",
        K_real,
    )

    # Buffers.
    var input_buf = ctx.enqueue_create_buffer[in_dtype](N_batch * H * W * C_in)
    var filter_buf = ctx.enqueue_create_buffer[in_dtype](C_out * K_real)
    var output_buf = ctx.enqueue_create_buffer[in_dtype](
        N_batch * H_out * W_out * C_out
    )
    var residual_buf = ctx.enqueue_create_buffer[in_dtype](
        N_batch * H_out * W_out * C_out
    )

    var input_host = ctx.enqueue_create_host_buffer[in_dtype](
        N_batch * H * W * C_in
    )
    var filter_host = ctx.enqueue_create_host_buffer[in_dtype](C_out * K_real)
    var residual_host = ctx.enqueue_create_host_buffer[in_dtype](
        N_batch * H_out * W_out * C_out
    )
    rand(input_host.unsafe_ptr(), N_batch * H * W * C_in)
    rand(filter_host.unsafe_ptr(), C_out * K_real)
    rand(residual_host.unsafe_ptr(), N_batch * H_out * W_out * C_out)
    ctx.enqueue_copy(input_buf, input_host)
    ctx.enqueue_copy(filter_buf, filter_host)
    ctx.enqueue_copy(residual_buf, residual_host)

    # Layouts. The 4-wave matmul kernel asserts on `K % 256 == 0`,
    # so the direct-baseline path is only valid for naturally-aligned
    # shapes. K-padded shapes skip the direct baseline.
    comptime _aligned = (K_real % 256) == 0
    comptime input_layout = row_major[N_batch, H, W, C_in]()
    comptime filter_4d_layout = row_major[C_out, R, S, C_in]()
    comptime output_4d_layout = row_major[N_batch, H_out, W_out, C_out]()
    comptime output_2d_layout = row_major[M_total, C_out]()

    var input_tt = TileTensor(input_buf, input_layout)
    var filter_4d_tt = TileTensor(filter_buf, filter_4d_layout)
    var output_4d_tt = TileTensor(output_buf, output_4d_layout)
    var output_2d_tt = TileTensor(output_buf, output_2d_layout)
    var residual_4d_tt = TileTensor(residual_buf, output_4d_layout)

    var problem = Conv2dProblemShape(
        batch=N_batch,
        in_height=H,
        in_width=W,
        in_channels=C_in,
        out_channels=C_out,
        filter_h=R,
        filter_w=S,
        pad_h=pad,
        pad_w=pad,
        stride_h=stride,
        stride_w=stride,
    )

    # ---- Warmup all paths (DVFS, allocator, kernel cache) ----
    comptime if _aligned:
        comptime filter_2d_layout = row_major[C_out, K_real]()
        var filter_2d_tt = TileTensor(filter_buf, filter_2d_layout)
        for _ in range(warmup_iters):
            amd_4wave_conv[
                H=H,
                W=W,
                H_out=H_out,
                W_out=W_out,
                R=R,
                S=S,
                stride_h=stride,
                stride_w=stride,
                pad_h=pad,
                pad_w=pad,
                C_in=C_in,
            ](input_tt, filter_2d_tt, output_2d_tt, ctx)
    for _ in range(warmup_iters):
        amd_4wave_conv_fprop_with_residual[
            in_dtype,
            in_dtype,
            in_dtype,
            has_residual=False,
        ](
            output_4d_tt,
            input_tt,
            filter_4d_tt,
            residual_4d_tt,
            beta=Float32(0.0),
            problem=problem,
            ctx=ctx,
        )
        amd_4wave_conv_fprop_with_residual[
            in_dtype,
            in_dtype,
            in_dtype,
            has_residual=True,
        ](
            output_4d_tt,
            input_tt,
            filter_4d_tt,
            residual_4d_tt,
            beta=Float32(1.0),
            problem=problem,
            ctx=ctx,
        )
    ctx.synchronize()

    # ---- Bench 1: direct amd_4wave_conv (baseline; aligned-K only) ----
    var ms_direct = Float64(0.0)
    var tflops_direct = Float64(0.0)
    comptime if _aligned:
        comptime filter_2d_layout = row_major[C_out, K_real]()
        var filter_2d_tt = TileTensor(filter_buf, filter_2d_layout)
        var t0 = perf_counter_ns()
        for _ in range(bench_iters):
            amd_4wave_conv[
                H=H,
                W=W,
                H_out=H_out,
                W_out=W_out,
                R=R,
                S=S,
                stride_h=stride,
                stride_w=stride,
                pad_h=pad,
                pad_w=pad,
                C_in=C_in,
            ](input_tt, filter_2d_tt, output_2d_tt, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ms_direct = Float64(t1 - t0) / Float64(bench_iters) / 1_000_000.0
        tflops_direct = Float64(flops_conv) / (ms_direct * 1e-3) / 1e12

    # ---- Bench 2: wrapper, has_residual=False (early-out path) ----
    var t0 = perf_counter_ns()
    for _ in range(bench_iters):
        amd_4wave_conv_fprop_with_residual[
            in_dtype,
            in_dtype,
            in_dtype,
            has_residual=False,
        ](
            output_4d_tt,
            input_tt,
            filter_4d_tt,
            residual_4d_tt,
            beta=Float32(0.0),
            problem=problem,
            ctx=ctx,
        )
    ctx.synchronize()
    var t2 = perf_counter_ns()
    var ms_wrapper_off = Float64(t2 - t0) / Float64(bench_iters) / 1_000_000.0
    var tflops_wrapper_off = (
        Float64(flops_conv) / (ms_wrapper_off * 1e-3) / 1e12
    )

    # ---- Bench 3: wrapper, has_residual=True (fused residual) ----
    var t3 = perf_counter_ns()
    for _ in range(bench_iters):
        amd_4wave_conv_fprop_with_residual[
            in_dtype,
            in_dtype,
            in_dtype,
            has_residual=True,
        ](
            output_4d_tt,
            input_tt,
            filter_4d_tt,
            residual_4d_tt,
            beta=Float32(1.0),
            problem=problem,
            ctx=ctx,
        )
    ctx.synchronize()
    var t4 = perf_counter_ns()
    var ms_fused = Float64(t4 - t3) / Float64(bench_iters) / 1_000_000.0
    var tflops_fused = Float64(flops_conv) / (ms_fused * 1e-3) / 1e12

    # ---- Report ----
    comptime if _aligned:
        var nr_ratio = tflops_wrapper_off / tflops_direct
        var fused_ratio = tflops_fused / tflops_direct
        print("  direct          ms=", ms_direct, " TFLOPS=", tflops_direct)
        print(
            "  wrapper-no-res  ms=",
            ms_wrapper_off,
            " TFLOPS=",
            tflops_wrapper_off,
            " ratio_vs_direct=",
            nr_ratio,
        )
        print(
            "  wrapper-fused   ms=",
            ms_fused,
            " TFLOPS=",
            tflops_fused,
            " ratio_vs_direct=",
            fused_ratio,
        )
        # Flag a regression if the wrapper-off path is >2% slower than
        # direct (it should be byte-identical at asm level).
        if nr_ratio < 0.98:
            print(
                "  WARNING: wrapper-no-res REGRESSION vs direct"
                " (ratio < 0.98×; expected ≈ 1.0×)"
            )
    else:
        # K-padded shape: no direct baseline (it'd require pre-K-padding
        # the filter outside the timing loop). Report the fused/no-res
        # ratio instead, which captures the residual fusion cost.
        var fused_vs_no_res = tflops_fused / tflops_wrapper_off
        print(
            "  (K-padded: K_real=",
            K_real,
            " K_padded=",
            K_padded,
            "; no direct baseline)",
        )
        print(
            "  wrapper-no-res  ms=",
            ms_wrapper_off,
            " TFLOPS=",
            tflops_wrapper_off,
        )
        print(
            "  wrapper-fused   ms=",
            ms_fused,
            " TFLOPS=",
            tflops_fused,
            " ratio_vs_no_res=",
            fused_vs_no_res,
        )


def main() raises:
    # One shape per invocation. Defaults reproduce the `vae-256ch-3x3`
    # row of the companion YAML so the bench can also be run directly
    # (`mojo bench_amd_4wave_conv_residual.mojo`) without kbench.
    comptime _N_BATCH = get_defined_int["N_BATCH", 1]()
    comptime _H = get_defined_int["H", 32]()
    comptime _W = get_defined_int["W", 32]()
    comptime _C_IN = get_defined_int["C_IN", 256]()
    comptime _C_OUT = get_defined_int["C_OUT", 256]()
    comptime _R = get_defined_int["R", 3]()
    comptime _S = get_defined_int["S", 3]()
    comptime _STRIDE = get_defined_int["STRIDE", 1]()
    comptime _PAD = get_defined_int["PAD", 1]()
    comptime _LABEL = get_defined_string["LABEL", "shape"]()

    with DeviceContext() as ctx:
        bench_conv_residual[
            N_batch=_N_BATCH,
            H=_H,
            W=_W,
            C_in=_C_IN,
            C_out=_C_OUT,
            R=_R,
            S=_S,
            stride=_STRIDE,
            pad=_PAD,
            label=_LABEL,
        ](ctx)
