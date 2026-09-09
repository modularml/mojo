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
"""BF16 production-shape perf comparison: amd_4wave_conv vs MIOpen.

MIOpen is the vendor reference on AMD (mirrors cuDNN on NVIDIA). This
bench times `amd_4wave_conv` (BF16, routes through the framework-
scheduled body) against `conv_miopen` (BF16) on the FLUX VAE / ResNet /
pointwise production shape suite. FP8 is not covered here because the
MIOpen Mojo wrappers don't support FP8 yet (see
`bench_amd_4wave_conv_real_shapes.mojo` for FP8 perf vs matmul).

Numerical equivalence between the two paths is validated by
`test_amd_4wave_conv_bf16.mojo` (vs `structured_4wave_matmul`,
byte-identical) and `test_amd_4wave_conv_miopen.mojo` (vs MIOpen,
L1-rel < 1%).
"""

from std.benchmark import keep
from max.gpu.host import DeviceContext
from std.random import rand
from std.time import perf_counter_ns
from std.utils import IndexList

from layout import Coord, TileTensor, row_major

from nn.conv.conv import conv_miopen
from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


def _permute_filter_frsc_to_rscf_host[
    dtype: DType
](
    src_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    *,
    F: Int,
    R: Int,
    S: Int,
    C: Int,
):
    """Permutes `[F, R, S, C]` (4wave's filter layout) to `[R, S, C, F]`
    (MIOpen's NHWC-default RSCF layout).
    """
    for f in range(F):
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    var src_idx = ((f * R + r) * S + s) * C + c
                    var dst_idx = ((r * S + s) * C + c) * F + f
                    dst_ptr[dst_idx] = src_ptr[src_idx]


def _round_up(x: Int, mod: Int) -> Int:
    return ((x + mod - 1) // mod) * mod


def _bench_one[
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
    comptime a_type = DType.bfloat16
    comptime c_type = DType.bfloat16
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
    # Note: `C_in >= 128` would trigger an `Int.__ge__` comptime
    # parse-order cycle in this file (same as bench_pingpong_matmul);
    # using `> 127` sidesteps the cycle while preserving the predicate.
    elif C_in % 128 == 0 and C_in > 127:
        path = "uniform "
    else:
        path = "per-lane"

    # Allocate buffers (separate buffers per path; both consume the
    # same random input but the 4wave path needs an F-R-S-C filter
    # while MIOpen needs R-S-C-F).
    var input_dev = ctx.enqueue_create_buffer[a_type](N_batch * H * W * C_in)
    var filter_frsc_dev = ctx.enqueue_create_buffer[a_type](C_out * K_padded)
    var filter_rscf_dev = ctx.enqueue_create_buffer[a_type](
        R * S * C_in * C_out
    )
    var output_4wave_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)
    var output_miopen_dev = ctx.enqueue_create_buffer[c_type](
        N_batch * H_out * W_out * C_out
    )

    var input_host = ctx.enqueue_create_host_buffer[a_type](
        N_batch * H * W * C_in
    )
    var filter_frsc_host = ctx.enqueue_create_host_buffer[a_type](
        C_out * K_padded
    )
    rand(input_host.unsafe_ptr(), N_batch * H * W * C_in)
    rand(filter_frsc_host.unsafe_ptr(), C_out * K_padded)
    # Zero the K-padding region of the FRSC filter.
    for f in range(C_out):
        for k in range(K_real, K_padded):
            filter_frsc_host[f * K_padded + k] = 0
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_frsc_dev, filter_frsc_host)

    # Build the RSCF view of the same filter weights (drop the K-pad).
    var filter_rscf_host = ctx.enqueue_create_host_buffer[a_type](
        R * S * C_in * C_out
    )
    # 4wave's FRSC has K=K_padded stride; MIOpen wants compact RSCF
    # over R*S*C_in. Pack out of the FRSC[:, :K_real] window.
    for f in range(C_out):
        for r in range(R):
            for s in range(S):
                for c in range(C_in):
                    var frsc_idx = f * K_padded + (r * S + s) * C_in + c
                    var rscf_idx = ((r * S + s) * C_in + c) * C_out + f
                    filter_rscf_host[rscf_idx] = filter_frsc_host[frsc_idx]
    ctx.enqueue_copy(filter_rscf_dev, filter_rscf_host)

    comptime nhwc_in_layout = row_major[N_batch, H, W, C_in]()
    comptime filter_frsc_layout = row_major[C_out, K_padded]()
    comptime output_4wave_layout = row_major[M_total, C_out]()
    comptime nhwc_in_dims = IndexList[4](N_batch, H, W, C_in)
    comptime rscf_dims = IndexList[4](R, S, C_in, C_out)
    comptime nhwc_out_dims = IndexList[4](N_batch, H_out, W_out, C_out)
    comptime miopen_in_layout = row_major(Coord(nhwc_in_dims))
    comptime miopen_filter_layout = row_major(Coord(rscf_dims))
    comptime miopen_out_layout = row_major(Coord(nhwc_out_dims))

    var input_nhwc = TileTensor(input_dev, nhwc_in_layout)
    var filter_4wave = TileTensor(filter_frsc_dev, filter_frsc_layout)
    var output_4wave = TileTensor(output_4wave_dev, output_4wave_layout)
    var input_miopen = TileTensor(input_dev, miopen_in_layout)
    var filter_miopen = TileTensor(filter_rscf_dev, miopen_filter_layout)
    var output_miopen = TileTensor(output_miopen_dev, miopen_out_layout)

    # Warmup (DVFS, MIOpen algo selection cache, allocator).
    for _ in range(warmup_iters):
        conv_miopen(
            input_miopen,
            filter_miopen,
            output_miopen,
            IndexList[2](stride_h, stride_w),
            IndexList[2](1, 1),
            IndexList[2](pad_h, pad_w),
            1,
            ctx,
        )
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
        ](input_nhwc, filter_4wave, output_4wave, ctx)
    ctx.synchronize()

    # Bench MIOpen.
    var t0_mio = perf_counter_ns()
    for _ in range(bench_iters):
        conv_miopen(
            input_miopen,
            filter_miopen,
            output_miopen,
            IndexList[2](stride_h, stride_w),
            IndexList[2](1, 1),
            IndexList[2](pad_h, pad_w),
            1,
            ctx,
        )
    ctx.synchronize()
    var t1_mio = perf_counter_ns()
    var ms_mio = Float64(t1_mio - t0_mio) / Float64(bench_iters) / 1_000_000.0
    var tflops_mio = Float64(flops) / (ms_mio * 1e-3) / 1e12

    # Bench amd_4wave_conv.
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
        ](input_nhwc, filter_4wave, output_4wave, ctx)
    ctx.synchronize()
    var t1_cv = perf_counter_ns()
    var ms_cv = Float64(t1_cv - t0_cv) / Float64(bench_iters) / 1_000_000.0
    var tflops_cv = Float64(flops) / (ms_cv * 1e-3) / 1e12

    var ratio = tflops_cv / tflops_mio

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
        " | miopen ",
        tflops_mio,
        " | ",
        ratio,
        " |",
    )

    keep(output_4wave_dev.unsafe_ptr())
    keep(output_miopen_dev.unsafe_ptr())


def main() raises:
    with DeviceContext() as ctx:
        print("== amd_4wave_conv (BF16) vs MIOpen (BF16) on MI355X ==")
        print(
            "| Shape | NHWC | RxS,stride,pad | Cout | M, K | path | conv"
            " TFLOPS | miopen TFLOPS | conv/miopen |"
        )
        print(
            "|-------|------|----------------|------|------|------|-------------|---------------|-------------|"
        )

        # ----- Pointwise (1x1 conv) -----
        _bench_one[
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

        # ----- FLUX VAE 3x3 same-pad blocks -----
        _bench_one[
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

        # ----- ResNet stem (3x3 C_in=64) -----
        _bench_one[
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
