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
"""Numerical correctness test for amd_4wave_conv_fprop_with_residual.

Validates `D = Conv(A,B) + beta*C` against a host-computed reference
across all branches of the launcher:
  - `has_residual=False`: plain conv (early-out).
  - `beta=0.0` with `has_residual=True`: plain conv (early-out).
  - `beta=1.0`: skip connection.
  - `beta=0.5`: scaled residual.

Uses HK MHA in-main parametrization — single BUILD target, dtype
iteration in `main()`. PyTorch-like tolerance per dtype.
"""

from max.gpu.host import DeviceContext
from std.random import rand
from std.testing import assert_equal
from std.utils import IndexList

from layout import TileTensor, row_major

from nn.conv.gpu.amd.amd_4wave_conv_residual import (
    amd_4wave_conv_fprop_with_residual,
)
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape


# Pure pre-residual compute lambda used by the "compute_lambda" test
# variant — adds a comptime-fixed bias to the MMA output. Mirrors the
# canonical UNet/ResNet "conv + bias + residual" fusion (bias is
# trivially in-register, real bias-per-channel would index by
# coords[1] = C_out). Used to validate the SM100-style ordering
# `D = lambda(Conv(A,B)) + beta * C`.
@__parameter
@always_inline
def _bias_compute_lambda[
    _dtype: DType,
    _width: SIMDLength,
    *,
    alignment: Int = 1,
](coords: IndexList[2], val: SIMD[_dtype, _width]) capturing -> SIMD[
    _dtype, _width
]:
    return val + SIMD[_dtype, _width](0.5)


def _permute_filter_frsc_to_kpadded_host[
    dtype: DType
](
    src_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    *,
    F: Int,
    R: Int,
    S: Int,
    C: Int,
    K_padded: Int,
):
    """Pack [F, R*S*C] filter into [F, K_padded] with trailing zeros.

    The kernel expects `[Cout, K_padded]` row-major where each row is
    `(r, s, c)` flattened (= R*S*C real elements) followed by zeros
    up to `K_padded`.
    """
    var K_real = R * S * C
    for f in range(F):
        for k in range(K_real):
            dst_ptr[f * K_padded + k] = src_ptr[f * K_real + k]
        for k in range(K_real, K_padded):
            dst_ptr[f * K_padded + k] = Scalar[dtype](0)


def _conv2d_residual_host_ref[
    in_dtype: DType,
    out_dtype: DType,
](
    input_host: ImmPointer[Scalar[in_dtype], _],
    filter_host: ImmPointer[Scalar[in_dtype], _],
    residual_host: ImmPointer[Scalar[out_dtype], _],
    ref_host: MutPointer[Scalar[out_dtype], _],
    *,
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    stride: Int,
    pad: Int,
    H_out: Int,
    W_out: Int,
    beta: Float32,
    apply_residual: Bool,
    # When non-zero, applied to the conv output BEFORE the residual add
    # (mirrors the device-side compute-lambda's `+0.5` bias). The
    # ordering matches `D = lambda(Conv(A,B)) + beta * C` from SM100.
    compute_lambda_bias: Float32 = Float32(0.0),
):
    """Host conv2d + residual reference.

    Filter is [C_out, R, S, C_in] row-major (no K-padding here — host
    does the real R*S*C_in dot).
    """
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                for c_out in range(C_out):
                    var acc = Float32(0.0)
                    for r in range(R):
                        for s in range(S):
                            var h_in = h_out * stride + r - pad
                            var w_in = w_out * stride + s - pad
                            if 0 <= h_in < H and 0 <= w_in < W:
                                for c in range(C_in):
                                    var a = Float32(
                                        input_host[
                                            ((n * H + h_in) * W + w_in) * C_in
                                            + c
                                        ]
                                    )
                                    var b = Float32(
                                        filter_host[
                                            ((c_out * R + r) * S + s) * C_in + c
                                        ]
                                    )
                                    acc = acc + a * b
                    # Cast through dtype to match kernel's post-cast
                    # compute_lambda input precision.
                    var v = acc.cast[out_dtype]()
                    if compute_lambda_bias != Float32(0.0):
                        v = (Float32(v) + compute_lambda_bias).cast[out_dtype]()
                    if apply_residual:
                        var sk = Float32(
                            residual_host[
                                ((n * H_out + h_out) * W_out + w_out) * C_out
                                + c_out
                            ]
                        )
                        v = (Float32(v) + beta * sk).cast[out_dtype]()
                    ref_host[
                        ((n * H_out + h_out) * W_out + w_out) * C_out + c_out
                    ] = v


def _compare[
    dtype: DType
](
    actual: ImmPointer[Scalar[dtype], _],
    expected: ImmPointer[Scalar[dtype], _],
    *,
    n_elems: Int,
    label: StaticString,
    rel_tol: Float32,
    abs_tol: Float32,
) raises:
    var max_diff = Float32(0.0)
    var max_rel = Float32(0.0)
    var n_exceed = 0
    var n_printed = 0
    for i in range(n_elems):
        var a = Float32(actual[i])
        var e = Float32(expected[i])
        var d = abs(a - e)
        var threshold = abs_tol + rel_tol * abs(e)
        var rel = d / max(abs(e), Float32(1e-5))
        if d > max_diff:
            max_diff = d
        if rel > max_rel:
            max_rel = rel
        if d > threshold:
            n_exceed += 1
            if n_printed < 3:
                print("  ", label, " mismatch at i=", i, ": a=", a, " e=", e)
                n_printed += 1
    print(
        "  ",
        label,
        ": max_abs=",
        max_diff,
        " max_rel=",
        max_rel,
        " n_exceed=",
        n_exceed,
        "/",
        n_elems,
    )
    assert_equal(n_exceed, 0)


def test_conv_residual[
    in_dtype: DType,
    *,
    # Output dtype. Defaults to in_dtype for BF16/FP16; FP8 callers
    # pass `out_dtype=DType.bfloat16` (FP8 output isn't supported by
    # the 4-wave kernel — accumulator/store path is BF16).
    out_dtype: DType = in_dtype,
    N: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int,
    S: Int,
    stride: Int,
    pad: Int,
    beta: Float32,
    has_residual: Bool,
    # When True, wire `_bias_compute_lambda` (adds 0.5 to MMA output
    # BEFORE the residual FMA). The host reference adds 0.5 in the
    # same place. Validates the SM100-style
    # `D = lambda(Conv(A,B)) + beta * C` ordering.
    use_compute_lambda: Bool = False,
    # When > 1.0, scale the random residual to this magnitude (post-
    # rand multiply). Use a large `residual_scale` to stress the
    # residual addressing path: a bug that drops or mis-addresses the
    # residual produces a per-cell diff proportional to the residual
    # magnitude, which must exceed the BF16 relative tolerance
    # (`rel_tol * |expected|`) to be caught. With the default in
    # [0, 1) range the diff is BF16-noise-comparable at the small
    # shapes the existing sweep uses; multi-workgroup shapes need
    # residual_scale >= ~10 to surface positional addressing bugs.
    residual_scale: Float32 = 1.0,
    label: StaticString,
](ctx: DeviceContext) raises:
    """Run amd_4wave_conv_fprop_with_residual on the given shape and
    validate against host reference."""

    comptime H_out = (H + 2 * pad - R) // stride + 1
    comptime W_out = (W + 2 * pad - S) // stride + 1
    comptime K_real = R * S * C_in
    # K-pad to multiple of 2*BK = 256. K_real may already be aligned
    # (e.g. C_in=256, R=S=3 → K=2304).
    comptime K_padded = ((K_real + 255) // 256) * 256

    print(
        "==",
        label,
        " in_dtype=",
        in_dtype,
        " out_dtype=",
        out_dtype,
        " N=",
        N,
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
        " beta=",
        beta,
        " has_residual=",
        has_residual,
        " K_real=",
        K_real,
        " K_padded=",
        K_padded,
    )

    var n_in = N * H * W * C_in
    var n_filter_real = C_out * K_real
    var n_out = N * H_out * W_out * C_out

    # Device buffers. Filter is in caller-facing un-padded layout
    # `[C_out, R, S, C_in]` (= [C_out, K_real] flattened). The launcher
    # handles K-padding internally when K_real % 256 != 0.
    var input_buf = ctx.enqueue_create_buffer[in_dtype](n_in)
    var filter_buf = ctx.enqueue_create_buffer[in_dtype](n_filter_real)
    var output_buf = ctx.enqueue_create_buffer[out_dtype](n_out)
    var residual_buf = ctx.enqueue_create_buffer[out_dtype](n_out)

    # Host-side: random input + filter + residual.
    var input_host = ctx.enqueue_create_host_buffer[in_dtype](n_in)
    var filter_host = ctx.enqueue_create_host_buffer[in_dtype](n_filter_real)
    var residual_host = ctx.enqueue_create_host_buffer[out_dtype](n_out)
    rand(input_host.unsafe_ptr(), n_in)
    rand(filter_host.unsafe_ptr(), n_filter_real)
    rand(residual_host.unsafe_ptr(), n_out)
    # Scale residual to surface positional addressing bugs at large M.
    comptime if residual_scale != Float32(1.0):
        for i in range(n_out):
            residual_host[i] = (
                Float32(residual_host[i]) * residual_scale
            ).cast[out_dtype]()

    ctx.enqueue_copy(input_buf, input_host)
    ctx.enqueue_copy(filter_buf, filter_host)
    ctx.enqueue_copy(residual_buf, residual_host)

    # Shape definitions for the launcher.
    var problem = Conv2dProblemShape(
        batch=N,
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

    # NHWC for input + output + residual, KRSC for filter — un-padded
    # `[C_out, R, S, C_in]`. The launcher handles K-padding internally.
    comptime input_layout = row_major[N, H, W, C_in]()
    comptime filter_layout = row_major[C_out, R, S, C_in]()
    comptime output_layout = row_major[N, H_out, W_out, C_out]()

    var input_tt = TileTensor(input_buf, input_layout)
    var filter_tt = TileTensor(filter_buf, filter_layout)
    var output_tt = TileTensor(output_buf, output_layout)
    var residual_tt = TileTensor(residual_buf, output_layout)

    comptime if use_compute_lambda:
        amd_4wave_conv_fprop_with_residual[
            in_dtype,
            in_dtype,
            out_dtype,
            has_residual=has_residual,
            elementwise_compute_lambda_fn=_bias_compute_lambda,
        ](
            output_tt,
            input_tt,
            filter_tt,
            residual_tt,
            beta=beta,
            problem=problem,
            ctx=ctx,
        )
    else:
        amd_4wave_conv_fprop_with_residual[
            in_dtype,
            in_dtype,
            out_dtype,
            has_residual=has_residual,
        ](
            output_tt,
            input_tt,
            filter_tt,
            residual_tt,
            beta=beta,
            problem=problem,
            ctx=ctx,
        )

    # Compute host reference.
    var ref_host = ctx.enqueue_create_host_buffer[out_dtype](n_out)
    _conv2d_residual_host_ref[in_dtype, out_dtype](
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        residual_host.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        N=N,
        H=H,
        W=W,
        C_in=C_in,
        C_out=C_out,
        R=R,
        S=S,
        stride=stride,
        pad=pad,
        H_out=H_out,
        W_out=W_out,
        beta=beta,
        apply_residual=has_residual and beta != 0.0,
        compute_lambda_bias=Float32(0.5) if use_compute_lambda else Float32(
            0.0
        ),
    )

    var output_host = ctx.enqueue_create_host_buffer[out_dtype](n_out)
    ctx.enqueue_copy(output_host, output_buf)
    ctx.synchronize()

    # PyTorch-like tolerance. FP8-in (BF16-out) carries the FP8 input
    # rounding error through the accumulator; treat as FP8 tolerance.
    comptime _ref_dtype = in_dtype
    comptime rel_tol = Float32(0.05) if _ref_dtype.is_float8() else (
        Float32(1.6e-2) if _ref_dtype == .bfloat16 else Float32(1e-3)
    )
    comptime abs_tol = Float32(0.01) if _ref_dtype.is_float8() else Float32(
        1e-5
    )
    _compare[out_dtype](
        output_host.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        n_elems=n_out,
        label=label,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def run_dtype_sweep[
    in_dtype: DType, *, out_dtype: DType = in_dtype
](ctx: DeviceContext) raises:
    """Run the four residual variants for one dtype on a naturally
    K-aligned shape (C_in=C_out=256, R=S=3 → K=2304=9×256)."""

    print("-- in_dtype=", in_dtype, "  out_dtype=", out_dtype, "  --")

    # (a) has_residual=False — plain conv early-out.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(0.0),
        has_residual=False,
        label="plain-no-residual",
    ](ctx)

    # (b) beta=0.0 with has_residual=True — early-out to plain conv.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(0.0),
        has_residual=True,
        label="beta-0-early-out",
    ](ctx)

    # (c) beta=1.0 — pure skip connection (the canonical UNet residual).
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        label="beta-1-skip",
    ](ctx)

    # (d) beta=0.5 — scaled residual.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(0.5),
        has_residual=True,
        label="beta-0.5-scaled",
    ](ctx)

    # (e) K-padded shape (FLUX VAE 128ch): K_real = 9*128 = 1152,
    # K_padded = 1280. Exercises the new K-pad path.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=128,
        C_out=128,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        label="vae-128ch-kpad",
    ](ctx)

    # (f) compute_lambda + residual — `D = (Conv + 0.5) + 1.0 * C`.
    # Validates the SM100-style ordering: the compute lambda fires
    # BEFORE the residual add. If the order were swapped on AMD the
    # max_abs diff would be ~1.0 (since each cell would gain/lose the
    # 0.5 bias depending on the residual's sign distribution).
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        use_compute_lambda=True,
        label="compute-lambda+residual",
    ](ctx)

    # (g) compute_lambda alone (no residual) — `D = Conv + 0.5`.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=8,
        W=8,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(0.0),
        has_residual=False,
        use_compute_lambda=True,
        label="compute-lambda-only",
    ](ctx)

    # (h) Multi-workgroup M with large-magnitude residual. The
    # smallest shape (H=W=8) above gives M=64 < BM=128, so the kernel
    # launches a single workgroup in the M dim and only ever runs
    # `pid_m=0`. Bugs in residual addressing that fire at
    # `pid_m >= 1` are invisible to the small-shape sweep. Going to
    # H=W=16 doubles M to 256 and forces two workgroups along M.
    # `residual_scale=20` puts the per-cell expected residual
    # contribution well above the BF16 `rel_tol * |expected|`
    # tolerance, so a missing or mis-addressed residual produces a
    # caught diff rather than hiding in BF16 noise.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=16,
        W=16,
        C_in=128,
        C_out=128,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        residual_scale=Float32(20.0),
        label="multi-wg-M2-kpad",
    ](ctx)

    # (i) Multi-workgroup M with K-aligned shape (C_in=256 so
    # K_real=9*256=2304 is naturally 256-aligned, no K-pad). M=256,
    # num_pid_m=2.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=16,
        W=16,
        C_in=256,
        C_out=256,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        residual_scale=Float32(20.0),
        label="multi-wg-M2-knopad",
    ](ctx)

    # (j) Larger multi-workgroup along M (num_pid_m=8). Exposes any
    # bug where the residual address-arithmetic only works for the
    # first few `pid_m` values (e.g. an overflow boundary that fires
    # past pid_m=4).
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=32,
        W=32,
        C_in=128,
        C_out=128,
        R=3,
        S=3,
        stride=1,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        residual_scale=Float32(20.0),
        label="multi-wg-M8",
    ](ctx)

    # (k) Multi-workgroup M with stride=2 (downsample-style). The
    # stride-2 path produces H_out=H/2, so for H=W=16, stride=2,
    # pad=1: H_out=W_out=8, M=64 → num_pid_m=1. Bump to H=W=32 →
    # H_out=W_out=16, M=256 → num_pid_m=2.
    test_conv_residual[
        in_dtype,
        out_dtype=out_dtype,
        N=1,
        H=32,
        W=32,
        C_in=128,
        C_out=128,
        R=3,
        S=3,
        stride=2,
        pad=1,
        beta=Float32(1.0),
        has_residual=True,
        residual_scale=Float32(20.0),
        label="multi-wg-M2-stride2",
    ](ctx)


def main() raises:
    with DeviceContext() as ctx:
        # In-main dtype iteration (HK MHA pattern).
        run_dtype_sweep[.bfloat16](ctx)
        # FP8 input → BF16 output (the 4-wave kernel doesn't support
        # FP8 output today, accumulation/store goes through BF16).
        # The residual is BF16; residual FMA happens in F32 — identical
        # to the BF16-in path numerically once the FP8 quant noise is
        # accounted for via the wider FP8 tolerance.
        run_dtype_sweep[.float8_e4m3fn, out_dtype=DType.bfloat16](ctx)
    print("==== amd_4wave_conv_fprop_with_residual tests PASSED ====")
