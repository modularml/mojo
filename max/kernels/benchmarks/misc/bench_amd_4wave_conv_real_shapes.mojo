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
"""Production-shape perf comparison: amd_4wave_conv vs structured_4wave_matmul (framework body)
on host-im2col for FLUX VAE / ResNet / pointwise-projection layers.

Each row runs both kernels on a representative conv layer shape and
prints ms / TFLOPS / ratio. The matmul-on-host-im2col is the upper
bound (no in-line im2col address math); the gap measures the cost of
the conv loader's runtime decomposition.

Shape sources:
  - "FLUX VAE": Black Forest Labs FLUX.1-dev VAE encoder/decoder
    conv stages (3×3 same-pad on increasing channel counts).
  - "ResNet stem": ImageNet-classification first 3×3 conv at 64 ch.
  - "Pointwise": 1×1 projection layers (depthwise-separable, attention
    projections in vision encoders).
"""

from std.benchmark import keep
from max.gpu.host import DeviceContext
from std.random import rand
from std.sys import get_defined_dtype
from std.time import perf_counter_ns

from layout import TileTensor, row_major

from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul
from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


def _host_im2col_general[
    a_type: DType,
    *,
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    R: Int,
    S: Int,
    H_out: Int,
    W_out: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
    K_padded: Int,
](
    input_host_ptr: ImmPointer[Scalar[a_type], ImmutAnyOrigin],
    im2col_host_ptr: MutPointer[Scalar[a_type], MutAnyOrigin],
):
    for n in range(N):
        for ho in range(H_out):
            for wo in range(W_out):
                var m = (n * H_out + ho) * W_out + wo
                for k in range(K_padded):
                    im2col_host_ptr[m * K_padded + k] = 0
                for r in range(R):
                    var hi = ho * stride_h + r - pad_h
                    for s in range(S):
                        var wi = wo * stride_w + s - pad_w
                        if 0 <= hi and hi < H and 0 <= wi and wi < W:
                            for c in range(C_in):
                                var k = (r * S + s) * C_in + c
                                var in_idx = ((n * H + hi) * W + wi) * C_in + c
                                im2col_host_ptr[
                                    m * K_padded + k
                                ] = input_host_ptr[in_idx]


def _round_up(x: Int, mod: Int) -> Int:
    return ((x + mod - 1) // mod) * mod


def _bench_one[
    a_type: DType,
    c_type: DType,
    N_batch: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int = 1,
    stride_w: Int = 1,
    *,
    warmup_iters: Int = 5,
    bench_iters: Int = 30,
    label: StaticString = "",
](ctx: DeviceContext) raises:
    comptime H_out = (H + 2 * pad_h - R) // stride_h + 1
    comptime W_out = (W + 2 * pad_w - S) // stride_w + 1
    comptime M_total = N_batch * H_out * W_out
    comptime K_real = R * S * C_in
    comptime K_padded = _round_up(K_real, 256)
    comptime flops = 2 * M_total * K_real * C_out

    # Path classification (matches the comptime branches inside
    # TileLoaderLDSIm2col).
    var path: StaticString
    if R == 1 and S == 1 and pad_h == 0 and pad_w == 0:
        path = "pointwise"
    elif C_in % 128 == 0 and C_in >= 128:
        path = "uniform "
    else:
        path = "per-lane"

    # Allocate buffers.
    var input_dev = ctx.enqueue_create_buffer[a_type](N_batch * H * W * C_in)
    var filter_dev = ctx.enqueue_create_buffer[a_type](C_out * K_padded)
    var im2col_dev = ctx.enqueue_create_buffer[a_type](M_total * K_padded)
    var output_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)

    var input_host = ctx.enqueue_create_host_buffer[a_type](
        N_batch * H * W * C_in
    )
    var filter_host = ctx.enqueue_create_host_buffer[a_type](C_out * K_padded)
    rand(input_host.unsafe_ptr(), N_batch * H * W * C_in)
    rand(filter_host.unsafe_ptr(), C_out * K_padded)
    for f in range(C_out):
        for k in range(K_real, K_padded):
            filter_host[f * K_padded + k] = 0
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    var im2col_host = ctx.enqueue_create_host_buffer[a_type](M_total * K_padded)
    _host_im2col_general[
        a_type,
        N=N_batch,
        H=H,
        W=W,
        C_in=C_in,
        R=R,
        S=S,
        H_out=H_out,
        W_out=W_out,
        pad_h=pad_h,
        pad_w=pad_w,
        stride_h=stride_h,
        stride_w=stride_w,
        K_padded=K_padded,
    ](input_host.unsafe_ptr(), im2col_host.unsafe_ptr())
    ctx.enqueue_copy(im2col_dev, im2col_host)

    comptime nhwc_layout = row_major[N_batch, H, W, C_in]()
    comptime filter_layout = row_major[C_out, K_padded]()
    comptime im2col_layout = row_major[M_total, K_padded]()
    comptime output_layout = row_major[M_total, C_out]()

    var input_nhwc = TileTensor(input_dev, nhwc_layout)
    var im2col_2d = TileTensor(im2col_dev, im2col_layout)
    var filter = TileTensor(filter_dev, filter_layout)
    var output = TileTensor(output_dev, output_layout)

    # Reference matmul: always the framework-scheduled body so the
    # conv-vs-matmul ratio compares apples-to-apples (the conv path
    # also routes through the framework body for all dtypes). Use
    # `structured_4wave_matmul` for all dtypes.
    @__parameter
    @always_inline
    def _ref_matmul() raises:
        structured_4wave_matmul(im2col_2d, filter, output, ctx)

    # Warmup (DVFS, allocator).
    for _ in range(warmup_iters):
        _ref_matmul()
        amd_4wave_conv[
            H=H,
            W=W,
            H_out=H_out,
            W_out=W_out,
            R=R,
            S=S,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            C_in=C_in,
        ](input_nhwc, filter, output, ctx)
    ctx.synchronize()

    # Bench matmul (reference: no on-the-fly im2col cost).
    var t0_mm = perf_counter_ns()
    for _ in range(bench_iters):
        _ref_matmul()
    ctx.synchronize()
    var t1_mm = perf_counter_ns()
    var ms_mm = Float64(t1_mm - t0_mm) / Float64(bench_iters) / 1_000_000.0
    var tflops_mm = Float64(flops) / (ms_mm * 1e-3) / 1e12

    # Bench conv on NHWC (with in-line im2col).
    var t0_cv = perf_counter_ns()
    for _ in range(bench_iters):
        amd_4wave_conv[
            H=H,
            W=W,
            H_out=H_out,
            W_out=W_out,
            R=R,
            S=S,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            C_in=C_in,
        ](input_nhwc, filter, output, ctx)
    ctx.synchronize()
    var t1_cv = perf_counter_ns()
    var ms_cv = Float64(t1_cv - t0_cv) / Float64(bench_iters) / 1_000_000.0
    var tflops_cv = Float64(flops) / (ms_cv * 1e-3) / 1e12

    var ratio = tflops_cv / tflops_mm

    print(
        "| ",
        label,
        " | NHWC=",
        N_batch,
        "x",
        H,
        "x",
        W,
        "x",
        C_in,
        " | RxS=",
        R,
        "x",
        S,
        " stride=",
        stride_h,
        " pad=",
        pad_h,
        " | Cout=",
        C_out,
        " | M=",
        M_total,
        " K=",
        K_real,
        " | ",
        path,
        " | conv ",
        tflops_cv,
        " | mm ",
        tflops_mm,
        " | ",
        ratio,
        " |",
    )

    keep(output_dev.unsafe_ptr())


def _run_suite[a_type: DType, c_type: DType](ctx: DeviceContext) raises:
    """Runs the full production-shape sweep for one dtype."""
    # ----- Pointwise (1x1 conv) -----
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=64,
        W=64,
        C_in=512,
        C_out=128,
        R=1,
        S=1,
        pad_h=0,
        pad_w=0,
        label="pointwise-1",
    ](ctx)
    _bench_one[
        a_type,
        c_type,
        N_batch=2,
        H=32,
        W=32,
        C_in=1024,
        C_out=256,
        R=1,
        S=1,
        pad_h=0,
        pad_w=0,
        label="pointwise-2",
    ](ctx)
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=16,
        W=16,
        C_in=2048,
        C_out=512,
        R=1,
        S=1,
        pad_h=0,
        pad_w=0,
        label="pointwise-3",
    ](ctx)

    # ----- FLUX VAE 3x3 same-pad blocks (uniform substrip path) -----
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=64,
        W=64,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        pad_h=1,
        pad_w=1,
        label="FLUX VAE-256ch",
    ](ctx)
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=32,
        W=32,
        C_in=512,
        C_out=512,
        R=3,
        S=3,
        pad_h=1,
        pad_w=1,
        label="FLUX VAE-512ch",
    ](ctx)
    _bench_one[
        a_type,
        c_type,
        N_batch=2,
        H=16,
        W=16,
        C_in=512,
        C_out=512,
        R=3,
        S=3,
        pad_h=1,
        pad_w=1,
        label="FLUX VAE-512ch-b2",
    ](ctx)

    # ----- FLUX VAE downsampling (3x3 stride=2) -----
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=64,
        W=64,
        C_in=256,
        C_out=512,
        R=3,
        S=3,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        label="FLUX VAE-down",
    ](ctx)

    # ----- ResNet stem (3x3 C_in=64, per-lane substrip slow path) -----
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=64,
        W=64,
        C_in=64,
        C_out=128,
        R=3,
        S=3,
        pad_h=1,
        pad_w=1,
        label="ResNet-stem",
    ](ctx)
    _bench_one[
        a_type,
        c_type,
        N_batch=2,
        H=32,
        W=32,
        C_in=64,
        C_out=64,
        R=3,
        S=3,
        pad_h=1,
        pad_w=1,
        label="ResNet-stem-b2",
    ](ctx)

    # ----- K-padded 3x3 (C_in=128, K_real=1152, K_padded=1280) -----
    _bench_one[
        a_type,
        c_type,
        N_batch=1,
        H=32,
        W=32,
        C_in=128,
        C_out=128,
        R=3,
        S=3,
        pad_h=1,
        pad_w=1,
        label="K-pad-C128",
    ](ctx)


def main() raises:
    # Pick the input dtype via `-D DTYPE=<dtype>`. Output dtype mirrors
    # the matmul: FP8 → BF16, BF16 → BF16, FP16 → FP16. Default = FP8.
    comptime a_type = get_defined_dtype["DTYPE", .float8_e4m3fn]()
    comptime c_type = (
        DType.bfloat16 if a_type == DType.float8_e4m3fn else a_type
    )

    with DeviceContext() as ctx:
        print(
            "== dtype:",
            String(a_type),
            "(matmul ref: structured_4wave_matmul)",
        )
        print(
            "| Shape | NHWC | RxS,stride,pad | Cout | M, K | path | conv"
            " TFLOPS | matmul TFLOPS | conv/matmul |"
        )
        print(
            "|-------|------|----------------|------|------|------|-------------|---------------|-------------|"
        )
        _run_suite[a_type, c_type](ctx)
