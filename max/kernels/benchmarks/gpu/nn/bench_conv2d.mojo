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
"""Benchmark the native Mojo 2D conv paths against the platform's vendor impl.

Driven by `kbench` via a YAML shape sweep; one CSV row per `(impl, shape)`.
`impl` selects among:

  - `naive`     — naive NHWC-RSCF Mojo kernel (CPU/NVIDIA/AMD).
  - `im2col`    — `dispatch_im2col_matmul_conv2d` (NVIDIA-only currently).
                  If the dispatcher declines a shape (R=S=1, K<16, N<16,
                  non-bf16, grouped, etc.) the bench raises an `Error` so
                  kbench records the (impl, shape) as failed rather than
                  timing a no-op that misleadingly looks fastest.
  - `cudnn`     — NVIDIA cuDNN. Comptime-gated to NVIDIA targets.
  - `amd_4wave` — AMD 4-wave implicit-GEMM conv. Comptime-gated to AMD
                  targets. Supports FP8 (BF16 output), BF16 (BF16 output),
                  and FP16 (FP16 output). Requires the caller to pre-pad
                  the filter to a multiple of 2*BK = 256 along K; this
                  bench handles that internally.
  - `auto`      — platform-default. Resolves at comptime to `cudnn`
                  on NVIDIA, `amd_4wave` on AMD (with `naive` fallback
                  for dtypes the AMD path doesn't support).

Verification (`--verify=true`) uses cuDNN as the reference and is therefore
only valid on NVIDIA. AMD `--verify` is a no-op (use the standalone
correctness tests under max/kernels/test/gpu/nn/test_amd_4wave_conv*.mojo).

Usage (standalone NVIDIA):
    bazel test --config=remote-b200 \\
        //max/kernels/benchmarks:gpu/nn/bench_conv2d.mojo.run

Usage (kbench, NVIDIA):
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \\
        max/kernels/benchmarks/gpu/nn/bench_conv2d.yaml \\
        --target-accelerator cuda:b200 -c

Usage (standalone AMD, FP8):
    mojo max/kernels/benchmarks/gpu/nn/bench_conv2d.mojo \\
        -D dtype=float8_e4m3fn -D impl=amd_4wave \\
        -D C_in=128 -D C_out=128 -D R=3 -D S=3

Usage (standalone AMD, BF16):
    mojo max/kernels/benchmarks/gpu/nn/bench_conv2d.mojo \\
        -D dtype=bfloat16 -D impl=amd_4wave \\
        -D C_in=128 -D C_out=128 -D R=3 -D S=3
"""

from std.math import ceildiv
from std.random import rand
from std.sys import get_defined_dtype, get_defined_int, get_defined_string
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import (
    UNKNOWN_VALUE,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from nn.conv.conv import conv2d_gpu_naive_nhwc_rscf, conv_cudnn
from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv
from nn.conv.gpu.im2col_matmul_2d import dispatch_im2col_matmul_conv2d

from std.utils.index import IndexList


@always_inline
def _resolve_impl[impl: StaticString, dtype: DType]() -> StaticString:
    """Comptime-resolve `auto` to the platform-default impl.

    Uses `has_amd_gpu_accelerator()` / `has_nvidia_gpu_accelerator()`
    which detect the *build target's* GPU at comptime (working in
    host code, unlike `is_amd_gpu()` / `is_nvidia_gpu()` which only
    return true inside kernel codegen).

    Resolution:
      - NVIDIA → `cudnn`.
      - AMD + FP8 / bf16 / fp16 → `amd_4wave`.
      - AMD + other dtype → `naive`.
      - CPU-only → `naive`.
    """
    comptime if impl != "auto":
        return impl
    else:
        comptime if has_nvidia_gpu_accelerator():
            return "cudnn"
        else:
            comptime if has_amd_gpu_accelerator() and (
                dtype == .float8_e4m3fn
                or dtype == .bfloat16
                or dtype == .float16
            ):
                return "amd_4wave"
            else:
                return "naive"


def compute_conv2d_flops(
    batch: Int,
    h_out: Int,
    w_out: Int,
    c_out: Int,
    c_in: Int,
    r: Int,
    s: Int,
) -> Int:
    return 2 * batch * h_out * w_out * c_out * c_in * r * s


def bench_conv2d[
    dtype: DType,
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_r: Int,
    filter_s: Int,
    impl: StaticString,
    # Comptime pad — used by the `amd_4wave` arm (which needs pad as a
    # template parameter on the conv kernel). The other arms use the
    # runtime `pad_h` / `pad_w` args below, which should be kept in
    # sync with these comptime values when sweeping in kbench.
    pad_h_static: Int = 1,
    pad_w_static: Int = 1,
](
    ctx: DeviceContext,
    mut b: Bench,
    label: String,
    verify: Bool,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
) raises:
    var h_out = (in_height + 2 * pad_h - filter_r) // stride_h + 1
    var w_out = (in_width + 2 * pad_w - filter_s) // stride_w + 1

    comptime input_layout = Layout.row_major(
        batch, in_height, in_width, in_channels
    )
    comptime filter_rscf_layout = Layout.row_major(
        filter_r, filter_s, in_channels, out_channels
    )
    # Output spatial dims depend on runtime pad / stride, so leave them as
    # UNKNOWN_VALUE in the static layout and supply concrete sizes via a
    # RuntimeLayout below.
    comptime output_layout = Layout.row_major(
        batch, UNKNOWN_VALUE, UNKNOWN_VALUE, out_channels
    )

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_rscf_layout.size())
    var output_size = batch * h_out * w_out * out_channels

    var flops = compute_conv2d_flops(
        batch,
        h_out,
        w_out,
        out_channels,
        in_channels,
        filter_r,
        filter_s,
    )

    var bench_input_id = String(
        label,
        "/impl=",
        impl,
        "/dtype=",
        dtype,
        "/NHWC=(",
        batch,
        "x",
        in_height,
        "x",
        in_width,
        "x",
        in_channels,
        ")/filter=(",
        filter_r,
        "x",
        filter_s,
        ")/Cout=",
        out_channels,
        "/pad=(",
        pad_h,
        "x",
        pad_w,
        ")/stride=(",
        stride_h,
        "x",
        stride_w,
        ")",
    )

    var input_host = List(length=input_size, fill=Scalar[dtype](0))
    var filter_rscf_host = List(length=filter_size, fill=Scalar[dtype](0))
    var filter_fcrs_host = List(length=filter_size, fill=Scalar[dtype](0))
    rand[dtype](input_host)
    rand[dtype](filter_rscf_host)

    # RSCF [R,S,C,F] -> FCRS [F,C,R,S] for cuDNN.
    for f in range(out_channels):
        for c in range(in_channels):
            for r in range(filter_r):
                for s in range(filter_s):
                    var rscf_idx = (
                        r * filter_s * in_channels * out_channels
                        + s * in_channels * out_channels
                        + c * out_channels
                        + f
                    )
                    var fcrs_idx = (
                        f * in_channels * filter_r * filter_s
                        + c * filter_r * filter_s
                        + r * filter_s
                        + s
                    )
                    filter_fcrs_host[fcrs_idx] = filter_rscf_host[rscf_idx]

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_rscf_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_fcrs_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_ref_dev = ctx.enqueue_create_buffer[dtype](
        output_size if verify else 1
    )

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_rscf_dev, filter_rscf_host)
    ctx.enqueue_copy(filter_fcrs_dev, filter_fcrs_host)
    ctx.synchronize()

    var input_buf = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_rscf_buf = LayoutTensor[dtype, filter_rscf_layout](
        filter_rscf_dev.unsafe_ptr()
    )
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        IndexList[4](batch, h_out, w_out, out_channels)
    )
    var output_buf = LayoutTensor[dtype, output_layout](
        output_dev.unsafe_ptr(), output_runtime_layout
    )

    var input_tt = TileTensor(
        input_dev.unsafe_ptr(),
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_rscf_tt = TileTensor(
        filter_rscf_dev.unsafe_ptr(),
        row_major(
            Coord(IndexList[4](filter_r, filter_s, in_channels, out_channels))
        ),
    )
    var filter_fcrs_tt = TileTensor(
        filter_fcrs_dev.unsafe_ptr(),
        row_major(
            Coord(IndexList[4](out_channels, in_channels, filter_r, filter_s))
        ),
    )
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(),
        row_major(Coord(IndexList[4](batch, h_out, w_out, out_channels))),
    )

    var stride_idx = IndexList[2](stride_h, stride_w)
    var dilation_idx = IndexList[2](1, 1)
    var pad_idx = IndexList[2](pad_h, pad_w)

    comptime block_size = 16

    # Resolve `auto` to the platform-default impl at comptime via the
    # build target's accelerator (`has_{amd,nvidia}_gpu_accelerator()`).
    comptime resolved = _resolve_impl[impl, dtype]()

    # Probe the im2col dispatcher once. On decline, raise so kbench logs
    # this (impl, shape) as failed instead of timing a no-op (which would
    # otherwise look fastest in the CSV).
    comptime if resolved == "im2col":
        var accepted = dispatch_im2col_matmul_conv2d(
            input_tt,
            filter_rscf_tt,
            output_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        ctx.synchronize()
        if not accepted:
            raise Error(
                "dispatch_im2col_matmul_conv2d declined: " + bench_input_id
            )

    comptime if resolved == "im2col":

        @always_inline
        def im2col_kernel(ctx: DeviceContext) raises {imm}:
            _ = dispatch_im2col_matmul_conv2d(
                input_tt,
                filter_rscf_tt,
                output_tt,
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )

        @always_inline
        def im2col_bench(mut bencher: Bencher) raises {imm}:
            bencher_iter_custom(bencher, im2col_kernel, ctx)

        b.bench_function(
            im2col_bench,
            BenchId("conv2d_im2col", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )
    comptime if resolved == "cudnn":
        comptime assert has_nvidia_gpu_accelerator(), (
            "impl=cudnn requires an NVIDIA target. Build for an NVIDIA"
            " accelerator (e.g. cuda:b200) or pick a different impl."
        )

        @always_inline
        def cudnn_kernel(ctx: DeviceContext) raises {imm}:
            conv_cudnn[dtype, dtype, dtype](
                input_tt,
                filter_fcrs_tt,
                output_tt,
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )

        @always_inline
        def cudnn_bench(mut bencher: Bencher) raises {imm}:
            bencher_iter_custom(bencher, cudnn_kernel, ctx)

        b.bench_function(
            cudnn_bench,
            BenchId("conv2d_cudnn", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )
    comptime if resolved == "amd_4wave":
        comptime assert has_amd_gpu_accelerator(), (
            "impl=amd_4wave requires an AMD target. Build for an AMD"
            " accelerator (e.g. amdgpu:mi355) or pick a different impl."
        )
        comptime assert (
            dtype == .float8_e4m3fn or dtype == .bfloat16 or dtype == .float16
        ), (
            "impl=amd_4wave requires dtype in"
            " {float8_e4m3fn, bfloat16, float16}."
        )
        # FP8 accumulates into BF16; bf16/fp16 keep their input dtype
        # for the output (matches `structured_4wave_matmul`).
        comptime out_dtype = (
            DType.bfloat16 if dtype == DType.float8_e4m3fn else dtype
        )

        # Comptime H_out / W_out for the kernel's template params.
        # stride and dilation are hardcoded to 1 for the amd_4wave arm
        # (matches the test_amd_4wave_conv_strided.mojo paths that
        # exercise stride>1 / dilation>1 directly). The bench's runtime
        # `stride_h`/`stride_w` args must equal 1 for this arm.
        comptime H_OUT_STATIC = in_height + 2 * pad_h_static - filter_r + 1
        comptime W_OUT_STATIC = in_width + 2 * pad_w_static - filter_s + 1

        if (
            stride_h != 1
            or stride_w != 1
            or pad_h != pad_h_static
            or pad_w != pad_w_static
        ):
            raise Error(
                String(
                    (
                        "amd_4wave bench arm requires stride=1 and"
                        " pad_h/pad_w == pad_h_static/pad_w_static; got"
                        " runtime pad=("
                    ),
                    pad_h,
                    "x",
                    pad_w,
                    "), stride=(",
                    stride_h,
                    "x",
                    stride_w,
                    "), comptime pad=(",
                    pad_h_static,
                    "x",
                    pad_w_static,
                    ").",
                )
            )

        # K-padding: the 4-wave matmul schedule needs K_per_split % (2*BK)
        # == 0; the caller (this bench) zero-pads the trailing K rows of
        # the filter when R*S*C_in isn't already aligned. The conv
        # kernel takes the real C_in via its `C_in` comptime kwarg.
        comptime K_real = filter_r * filter_s * in_channels
        comptime K_padded = ((K_real + 255) // 256) * 256

        # Filter for the AMD path is laid out as [Cout, K_padded] in
        # F-R-S-C order (4-wave's expected layout), with zeros in the
        # padded trailing columns.
        var filter_frsc_host = List(
            length=out_channels * K_padded, fill=Scalar[dtype](0)
        )
        for f in range(out_channels):
            for r in range(filter_r):
                for s in range(filter_s):
                    for c in range(in_channels):
                        var rscf_idx = (
                            r * filter_s * in_channels * out_channels
                            + s * in_channels * out_channels
                            + c * out_channels
                            + f
                        )
                        var frsc_idx = (
                            f * K_padded
                            + r * filter_s * in_channels
                            + s * in_channels
                            + c
                        )
                        filter_frsc_host[frsc_idx] = filter_rscf_host[rscf_idx]

        var filter_frsc_dev = ctx.enqueue_create_buffer[dtype](
            out_channels * K_padded
        )
        var output_amd_dev = ctx.enqueue_create_buffer[out_dtype](output_size)
        ctx.enqueue_copy(filter_frsc_dev, filter_frsc_host)

        # Use comptime row_major layouts so the kernel can read its
        # comptime shape parameters (K, Cout) via `static_shape[i]`.
        # `Coord(IndexList(...))` would produce a dynamic-shape layout
        # and break the kernel's K-divisibility assert.
        comptime M_2D = batch * H_OUT_STATIC * W_OUT_STATIC
        comptime input_nhwc_layout = row_major[
            batch, in_height, in_width, in_channels
        ]()
        comptime filter_amd_layout = row_major[out_channels, K_padded]()
        comptime output_2d_layout = row_major[M_2D, out_channels]()

        var input_nhwc_amd = TileTensor(
            input_dev.unsafe_ptr(), input_nhwc_layout
        )
        var filter_frsc_tt = TileTensor(
            filter_frsc_dev.unsafe_ptr(), filter_amd_layout
        )
        # The kernel writes the output as a 2D [N*H_out*W_out, Cout]
        # view of the NHWC output buffer (aliases the same bytes for
        # packed layouts).
        var output_2d_tt = TileTensor(
            output_amd_dev.unsafe_ptr(), output_2d_layout
        )

        @always_inline
        def amd_4wave_kernel(ctx: DeviceContext) raises {imm}:
            amd_4wave_conv[
                H=in_height,
                W=in_width,
                H_out=H_OUT_STATIC,
                W_out=W_OUT_STATIC,
                R=filter_r,
                S=filter_s,
                pad_h=pad_h_static,
                pad_w=pad_w_static,
                C_in=in_channels,
            ](input_nhwc_amd, filter_frsc_tt, output_2d_tt, ctx)

        @always_inline
        def amd_4wave_bench(mut bencher: Bencher) raises {imm}:
            bencher_iter_custom(bencher, amd_4wave_kernel, ctx)

        b.bench_function(
            amd_4wave_bench,
            BenchId("conv2d_amd_4wave", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )

        _ = filter_frsc_dev^
        _ = output_amd_dev^
        _ = filter_frsc_host^
    comptime if resolved == "naive":
        # Naive Mojo NHWC-RSCF kernel.
        comptime naive_kernel = conv2d_gpu_naive_nhwc_rscf[
            input_layout,
            filter_rscf_layout,
            output_layout,
            dtype,
            dtype,
            dtype,
            block_size,
            None,
        ]
        var grid_dim_x = ceildiv(w_out * h_out, block_size)
        var grid_dim_y = batch

        @always_inline
        def naive_conv_kernel(ctx: DeviceContext) raises {imm}:
            ctx.enqueue_function[naive_kernel](
                input_buf,
                filter_rscf_buf,
                output_buf,
                stride_idx,
                dilation_idx,
                pad_idx,
                Int32(1),
                grid_dim=(grid_dim_x, grid_dim_y, 1),
                block_dim=(block_size, block_size, 1),
            )

        @always_inline
        def naive_bench(mut bencher: Bencher) raises {imm}:
            bencher_iter_custom(bencher, naive_conv_kernel, ctx)

        b.bench_function(
            naive_bench,
            BenchId("conv2d_naive", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )

    # Optional correctness cross-check against cuDNN. NVIDIA-only — cuDNN
    # is not available on AMD. The AMD `amd_4wave` arm has its own
    # dedicated correctness tests under max/kernels/test/gpu/nn/.
    var do_verify = verify
    comptime if not has_nvidia_gpu_accelerator():
        if verify:
            print("verify: skipped (requires NVIDIA cuDNN as the reference)")
        do_verify = False

    if do_verify:
        var output_ref_tt = TileTensor(
            output_ref_dev.unsafe_ptr(),
            row_major(Coord(IndexList[4](batch, h_out, w_out, out_channels))),
        )
        conv_cudnn[dtype, dtype, dtype](
            input_tt,
            filter_fcrs_tt,
            output_ref_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        ctx.synchronize()
        var output_host = List(length=output_size, fill=Scalar[dtype](0))
        var output_ref_host = List(length=output_size, fill=Scalar[dtype](0))
        ctx.enqueue_copy(output_host, output_dev)
        ctx.enqueue_copy(output_ref_host, output_ref_dev)
        ctx.synchronize()
        var max_diff: Float32 = 0.0
        for i in range(output_size):
            var a = output_host[i].cast[.float32]()
            var c = output_ref_host[i].cast[.float32]()
            var d = abs(a - c)
            if d > max_diff:
                max_diff = d
        print("verify max |", impl, " - cuDNN| = ", max_diff, sep="")
        _ = output_host^
        _ = output_ref_host^

    _ = input_dev^
    _ = filter_rscf_dev^
    _ = filter_fcrs_dev^
    _ = output_dev^
    _ = output_ref_dev^
    _ = filter_fcrs_host^
    _ = filter_rscf_host^
    _ = input_host^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime N = get_defined_int["N", 1]()
    comptime H = get_defined_int["H", 240]()
    comptime W = get_defined_int["W", 416]()
    comptime C_in = get_defined_int["C_in", 96]()
    comptime C_out = get_defined_int["C_out", 96]()
    comptime R = get_defined_int["R", 3]()
    comptime S = get_defined_int["S", 3]()
    comptime impl = get_defined_string["impl", "naive"]()
    # Comptime pad — used by the `amd_4wave` arm (which needs pad as a
    # kernel template parameter). Other arms continue to use the runtime
    # `pad_h` / `pad_w` args below. When sweeping in kbench, keep both
    # in sync (e.g. set both `pad_h_static` and `$pad_h` to the same
    # value in the YAML).
    comptime PAD_H = get_defined_int["pad_h_static", 1]()
    comptime PAD_W = get_defined_int["pad_w_static", 1]()

    var label = arg_parse("label", String("conv2d"))
    var verify = arg_parse("verify", False)
    # pad / stride flow into the cudnn/naive/im2col kernels as runtime
    # IndexLists; reading them via arg_parse avoids a fresh binary per
    # (pad, stride) combination for those arms. The `amd_4wave` arm
    # uses PAD_H / PAD_W (comptime) above.
    var pad_h = arg_parse("pad_h", PAD_H)
    var pad_w = arg_parse("pad_w", PAD_W)
    var stride_h = arg_parse("stride_h", 1)
    var stride_w = arg_parse("stride_w", 1)
    var warmup_iters = arg_parse("warmup_iters", 2)
    var max_iters = arg_parse("max_iters", 20)
    var max_runtime_secs = arg_parse("max_runtime_secs", 3.0)

    var m = Bench(
        BenchConfig(
            num_repetitions=1,
            num_warmup_iters=warmup_iters,
            max_iters=max_iters,
            max_runtime_secs=max_runtime_secs,
        )
    )
    with DeviceContext() as ctx:
        # `bench_conv2d` resolves `impl=auto` at comptime to the
        # platform-default arm via `_resolve_impl`.
        bench_conv2d[
            dtype,
            N,
            H,
            W,
            C_in,
            C_out,
            R,
            S,
            impl,
            pad_h_static=PAD_H,
            pad_w_static=PAD_W,
        ](ctx, m, label, verify, pad_h, pad_w, stride_h, stride_w)
    m.dump_report()
