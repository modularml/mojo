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
"""Cross-check amd_4wave_conv (FP8 in, BF16 out) against MIOpen (BF16 in, BF16 out).

The MIOpen Mojo wrappers do not currently support FP8, so we cast the
FP8 input + filter up to BF16 (FP8 -> BF16 is exact widening) and feed
the BF16 view through `conv_miopen` as an independent reference.

The numerical inputs are bit-identical between the two paths after the
FP8->BF16 widening. What differs is the MMA precision:

  - amd_4wave_conv  multiplies FP8(x) * FP8(y) at FP8 mantissa precision
  - conv_miopen      multiplies BF16(x) * BF16(y), where BF16 carries the
                     exact widening of the same FP8 values

The expected gap is the accumulated FP8-vs-BF16 per-multiply rounding
noise summed over K = R*S*C; tolerance below is calibrated for that.
"""

from max.gpu.host import DeviceContext
from std.random import rand

from layout import Coord, TileTensor, row_major
from std.utils import IndexList

from nn.conv.conv import conv_miopen
from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


def _cast_fp8_to_bf16_host(
    fp8_ptr: ImmPointer[Float8_e4m3fn, _],
    bf16_ptr: MutPointer[BFloat16, _],
    count: Int,
):
    for i in range(count):
        bf16_ptr[i] = BFloat16(Float32(fp8_ptr[i]))


def _permute_filter_frsc_to_rscf_host[
    dtype: DType
](
    src_ptr: ImmPointer[Scalar[dtype], _],
    dst_ptr: MutPointer[Scalar[dtype], _],
    *,
    F: Int,
    R: Int,
    S: Int,
    C: Int,
):
    """Permutes [F, R, S, C] (the 4wave kernel's filter layout) to
    [R, S, C, F] (MIOpen's RSCF default).
    """
    for f in range(F):
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    var src_idx = ((f * R + r) * S + s) * C + c
                    var dst_idx = ((r * S + s) * C + c) * F + f
                    dst_ptr[dst_idx] = src_ptr[src_idx]


def test_4wave_conv_vs_miopen[
    N_batch: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int = 3,
    S: Int = 3,
    pad_h: Int = 1,
    pad_w: Int = 1,
    *,
    tolerance: Float32 = 1.0,
](ctx: DeviceContext) raises:
    comptime H_out = (H + 2 * pad_h - R) + 1  # stride=1, dilation=1
    comptime W_out = (W + 2 * pad_w - S) + 1
    comptime M_total = N_batch * H_out * W_out
    comptime K = R * S * C_in

    print(
        "== test_4wave_conv_vs_miopen:",
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
        " R=",
        R,
        " S=",
        S,
        " pad=",
        pad_h,
        "x",
        pad_w,
    )

    # -------- Allocate buffers --------
    var input_fp8 = ctx.enqueue_create_buffer[.float8_e4m3fn](
        N_batch * H * W * C_in
    )
    var input_bf16 = ctx.enqueue_create_buffer[.bfloat16](
        N_batch * H * W * C_in
    )
    var filter_fp8_frsc = ctx.enqueue_create_buffer[.float8_e4m3fn](C_out * K)
    var filter_bf16_rscf = ctx.enqueue_create_buffer[.bfloat16](
        R * S * C_in * C_out
    )
    var output_conv = ctx.enqueue_create_buffer[.bfloat16](M_total * C_out)
    var output_miopen = ctx.enqueue_create_buffer[.bfloat16](
        N_batch * H_out * W_out * C_out
    )

    # -------- Init random FP8 + materialize BF16 views --------
    var input_fp8_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](
        N_batch * H * W * C_in
    )
    var filter_fp8_frsc_host = ctx.enqueue_create_host_buffer[
        DType.float8_e4m3fn
    ](C_out * K)
    rand(input_fp8_host.unsafe_ptr(), N_batch * H * W * C_in)
    rand(filter_fp8_frsc_host.unsafe_ptr(), C_out * K)

    var input_bf16_host = ctx.enqueue_create_host_buffer[.bfloat16](
        N_batch * H * W * C_in
    )
    _cast_fp8_to_bf16_host(
        input_fp8_host.unsafe_ptr(),
        input_bf16_host.unsafe_ptr(),
        N_batch * H * W * C_in,
    )

    # FP8 filter is in [F, R, S, C] (4wave's layout); we need [R, S, C, F]
    # for MIOpen, in BF16.
    var filter_bf16_frsc_host = ctx.enqueue_create_host_buffer[.bfloat16](
        C_out * K
    )
    _cast_fp8_to_bf16_host(
        filter_fp8_frsc_host.unsafe_ptr(),
        filter_bf16_frsc_host.unsafe_ptr(),
        C_out * K,
    )
    var filter_bf16_rscf_host = ctx.enqueue_create_host_buffer[.bfloat16](
        R * S * C_in * C_out
    )
    _permute_filter_frsc_to_rscf_host[.bfloat16](
        filter_bf16_frsc_host.unsafe_ptr(),
        filter_bf16_rscf_host.unsafe_ptr(),
        F=C_out,
        R=R,
        S=S,
        C=C_in,
    )

    # Push to device.
    ctx.enqueue_copy(input_fp8, input_fp8_host)
    ctx.enqueue_copy(input_bf16, input_bf16_host)
    ctx.enqueue_copy(filter_fp8_frsc, filter_fp8_frsc_host)
    ctx.enqueue_copy(filter_bf16_rscf, filter_bf16_rscf_host)

    # -------- amd_4wave_conv path (FP8) --------
    var input_nhwc = TileTensor(input_fp8, row_major[N_batch, H, W, C_in]())
    var filter_4wave = TileTensor(filter_fp8_frsc, row_major[C_out, K]())
    var output_4wave = TileTensor(output_conv, row_major[M_total, C_out]())
    amd_4wave_conv[
        H=H,
        W=W,
        H_out=H_out,
        W_out=W_out,
        R=R,
        S=S,
        pad_h=pad_h,
        pad_w=pad_w,
    ](input_nhwc, filter_4wave, output_4wave, ctx)

    # -------- conv_miopen path (BF16, RSCF filter) --------
    comptime nhwc_in_dims = IndexList[4](N_batch, H, W, C_in)
    comptime rscf_filter_dims = IndexList[4](R, S, C_in, C_out)
    comptime nhwc_out_dims = IndexList[4](N_batch, H_out, W_out, C_out)
    comptime nhwc_in_layout = row_major(Coord(nhwc_in_dims))
    comptime rscf_filter_layout = row_major(Coord(rscf_filter_dims))
    comptime nhwc_out_layout = row_major(Coord(nhwc_out_dims))
    var input_miopen = TileTensor(input_bf16, nhwc_in_layout)
    var filter_miopen = TileTensor(filter_bf16_rscf, rscf_filter_layout)
    var output_miopen_tt = TileTensor(output_miopen, nhwc_out_layout)
    conv_miopen(
        input_miopen,
        filter_miopen,
        output_miopen_tt,
        IndexList[2](1, 1),  # stride
        IndexList[2](1, 1),  # dilation
        IndexList[2](pad_h, pad_w),  # padding
        1,  # num_groups
        ctx,
    )

    # -------- Compare --------
    var conv_host = ctx.enqueue_create_host_buffer[.bfloat16](M_total * C_out)
    var miopen_host = ctx.enqueue_create_host_buffer[.bfloat16](
        N_batch * H_out * W_out * C_out
    )
    ctx.enqueue_copy(conv_host, output_conv)
    ctx.enqueue_copy(miopen_host, output_miopen)
    ctx.synchronize()

    var max_diff: Float32 = 0
    var sum_abs_diff: Float32 = 0
    var sum_abs_ref: Float32 = 0
    for i in range(M_total * C_out):
        var cv = Float32(conv_host[i])
        var mv = Float32(miopen_host[i])
        var d = abs(cv - mv)
        if d > max_diff:
            max_diff = d
        sum_abs_diff += d
        sum_abs_ref += abs(mv)
    var rel_diff = sum_abs_diff / sum_abs_ref if sum_abs_ref > 0 else Float32(0)
    # Two-tier PASS criterion. amd_4wave_conv multiplies FP8 operands at
    # FP8 mantissa precision (~6% relative per multiply); conv_miopen
    # uses BF16 (~0.4% per multiply) on the widened-from-FP8 inputs.
    # The per-multiply gap accumulates over K = R*S*C, so individual
    # max-element diff can be several absolute units even when the
    # aggregate L1-relative diff is tiny. Pass if EITHER:
    #   - max_abs_diff < `tolerance` absolute, OR
    #   - L1-relative < 1e-2 (the kernels agree on average to within 1%).
    var rel_pass = rel_diff < 0.01
    var abs_pass = max_diff < tolerance
    print(
        "  max_abs_diff=",
        max_diff,
        "  L1_rel=",
        rel_diff,
        "  abs_tol=",
        tolerance,
        "  pass(abs)=",
        abs_pass,
        "  pass(rel)=",
        rel_pass,
    )
    if not (abs_pass or rel_pass):
        raise Error(
            "amd_4wave_conv (FP8) vs conv_miopen (BF16) exceeds both abs"
            " and L1-rel tolerances"
        )
    print("  PASS")


def main() raises:
    with DeviceContext() as ctx:
        # 3x3 pad=1 same-size convs.
        test_4wave_conv_vs_miopen[
            N_batch=1,
            H=16,
            W=16,
            C_in=256,
            C_out=128,
            pad_h=1,
            pad_w=1,
        ](ctx)
        test_4wave_conv_vs_miopen[
            N_batch=2,
            H=14,
            W=14,
            C_in=512,
            C_out=128,
            pad_h=1,
            pad_w=1,
        ](ctx)
        # 3x3 no pad interior.
        test_4wave_conv_vs_miopen[
            N_batch=1,
            H=16,
            W=16,
            C_in=256,
            C_out=128,
            pad_h=0,
            pad_w=0,
        ](ctx)
