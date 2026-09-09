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
"""Cross-check `dispatch_amd_4wave_conv3d` against MIOpen on MI355X.

End-to-end correctness sweep for the native 3D implicit-GEMM conv
path. Each shape runs both:

  - `dispatch_amd_4wave_conv3d` (the new path being added by this PR),
  - `conv_miopen[conv_rank=3, ...]` (the reference, autotuned per
    shape via MIOpen's Find mode).

and asserts the two outputs match within the BF16 reduction-order
noise floor (either max-abs diff < absolute tolerance OR
L1-relative diff < 1%). Both paths consume the same random NDHWC
input + QRSCF/FCQRS filter, so the only difference is the kernel
math + reduction order.

Coverage axes:

  - pad_d ∈ {0, 1}            -- CausalConv3dCached pre-pad vs symmetric
  - stride_d ∈ {1, 2}         -- temporal downsample on/off
  - C_in ∈ {16, 192}          -- simd_width boundary (bf16 simd_w=8) vs typical
  - filter_is_fcqrs ∈ {F, T}  -- QRSCF (WAN's layout) vs FCQRS

MI355X-only; the dispatcher gates on `amdgpu:gfx950` and returns
False on any other accelerator (which would skip the test silently).
"""

from max.gpu.host import DeviceContext
from std.random import rand

from layout import Coord, Idx, TileTensor, row_major
from std.utils import IndexList

from nn.conv.conv import conv_miopen
from nn.conv.gpu.amd.dispatch_3d import dispatch_amd_4wave_conv3d


def _permute_qrscf_to_fcqrs_host[
    dtype: DType
](
    src_ptr: ImmPointer[Scalar[dtype], _],
    dst_ptr: MutPointer[Scalar[dtype], _],
    *,
    Q: Int,
    R: Int,
    S: Int,
    C: Int,
    F: Int,
):
    """QRSCF [Q,R,S,C,F] -> FCQRS [F,C,Q,R,S]. Used to build the FCQRS
    reference input from the same QRSCF source used by the QRSCF path,
    so both legs of the comparison see identical data."""
    for q in range(Q):
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    for f in range(F):
                        var src_idx = (((q * R + r) * S + s) * C + c) * F + f
                        var dst_idx = (((f * C + c) * Q + q) * R + r) * S + s
                        dst_ptr[dst_idx] = src_ptr[src_idx]


def test_4wave_conv3d_vs_miopen[
    N: Int,
    D: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    # Output spatial dims — caller computes (Mojo's comptime evaluator
    # doesn't bind locally-derived `comptime D_out: DType = ...` cleanly into
    # `Idx[D_out]` template positions in 5D row_major calls; passing
    # them as explicit function template params sidesteps the issue.
    D_out: Int,
    H_out: Int,
    W_out: Int,
    Q: Int = 3,
    R: Int = 3,
    S: Int = 3,
    stride_d: Int = 1,
    stride_h: Int = 1,
    stride_w: Int = 1,
    pad_d: Int = 0,
    pad_h: Int = 1,
    pad_w: Int = 1,
    *,
    abs_tolerance: Float32 = 16.0,
](ctx: DeviceContext) raises:
    """Runs both dispatcher legs (QRSCF + FCQRS) for one shape and
    asserts each matches MIOpen within BF16 noise. Tolerance defaults
    are calibrated for K = Q*R*S*C_in around 5000 (typical WAN VAE).
    """
    comptime assert (
        D_out == (D + 2 * pad_d - Q) // stride_d + 1
    ), "D_out doesn't match (D + 2*pad_d - Q) // stride_d + 1"
    comptime assert (
        H_out == (H + 2 * pad_h - R) // stride_h + 1
    ), "H_out doesn't match"
    comptime assert (
        W_out == (W + 2 * pad_w - S) // stride_w + 1
    ), "W_out doesn't match"
    comptime M_total = N * D_out * H_out * W_out
    comptime K = Q * R * S * C_in

    print(
        "== test_4wave_conv3d_vs_miopen:",
        " N=",
        N,
        " D=",
        D,
        " H=",
        H,
        " W=",
        W,
        " C_in=",
        C_in,
        " C_out=",
        C_out,
        " Q=",
        Q,
        " R=",
        R,
        " S=",
        S,
        " stride=",
        stride_d,
        "x",
        stride_h,
        "x",
        stride_w,
        " pad=",
        pad_d,
        "x",
        pad_h,
        "x",
        pad_w,
        " D_out=",
        D_out,
        " H_out=",
        H_out,
        " W_out=",
        W_out,
    )

    # -------- Allocate buffers --------
    var input_dev = ctx.enqueue_create_buffer[.bfloat16](N * D * H * W * C_in)
    var filter_qrscf_dev = ctx.enqueue_create_buffer[.bfloat16](
        Q * R * S * C_in * C_out
    )
    var filter_fcqrs_dev = ctx.enqueue_create_buffer[.bfloat16](
        C_out * C_in * Q * R * S
    )
    var output_qrscf_dev = ctx.enqueue_create_buffer[.bfloat16](
        N * D_out * H_out * W_out * C_out
    )
    var output_fcqrs_dev = ctx.enqueue_create_buffer[.bfloat16](
        N * D_out * H_out * W_out * C_out
    )
    var output_miopen_dev = ctx.enqueue_create_buffer[.bfloat16](
        N * D_out * H_out * W_out * C_out
    )

    # -------- Init random input + filters --------
    var input_host = ctx.enqueue_create_host_buffer[.bfloat16](
        N * D * H * W * C_in
    )
    var filter_qrscf_host = ctx.enqueue_create_host_buffer[.bfloat16](
        Q * R * S * C_in * C_out
    )
    rand(input_host.unsafe_ptr(), N * D * H * W * C_in)
    rand(filter_qrscf_host.unsafe_ptr(), Q * R * S * C_in * C_out)
    var filter_fcqrs_host = ctx.enqueue_create_host_buffer[.bfloat16](
        C_out * C_in * Q * R * S
    )
    _permute_qrscf_to_fcqrs_host[.bfloat16](
        filter_qrscf_host.unsafe_ptr(),
        filter_fcqrs_host.unsafe_ptr(),
        Q=Q,
        R=R,
        S=S,
        C=C_in,
        F=C_out,
    )
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_qrscf_dev, filter_qrscf_host)
    ctx.enqueue_copy(filter_fcqrs_dev, filter_fcqrs_host)

    # Static 5D layouts. The dispatcher gates on `input.static_shape[i]`
    # so we need every dim baked into the layout type at comptime —
    # the Coord/IndexList path produces RuntimeInt dims, which the
    # dispatcher reads as -1 and declines. `row_major[*idxs]()` returns
    # a layout with ComptimeInt dims, which the dispatcher reads as the
    # baked-in values. (Note: TileTensor(buf, row_major[a,b,c,d,e]())
    # at call-site rejected mixing function-template Ints and locally-
    # derived comptime Ints in earlier attempts; binding everything to
    # `Idx[...]` (ComptimeInt) via the Coord-typed-pack overload works
    # because that overload's element types are explicit.)
    var input_tt = TileTensor(
        input_dev,
        row_major(Idx[N], Idx[D], Idx[H], Idx[W], Idx[C_in]),
    )
    var filter_qrscf_tt = TileTensor(
        filter_qrscf_dev,
        row_major(Idx[Q], Idx[R], Idx[S], Idx[C_in], Idx[C_out]),
    )
    var filter_fcqrs_tt = TileTensor(
        filter_fcqrs_dev,
        row_major(Idx[C_out], Idx[C_in], Idx[Q], Idx[R], Idx[S]),
    )
    var output_qrscf_tt = TileTensor(
        output_qrscf_dev,
        row_major(
            Idx[N],
            Idx[D_out],
            Idx[H_out],
            Idx[W_out],
            Idx[C_out],
        ),
    )
    var output_fcqrs_tt = TileTensor(
        output_fcqrs_dev,
        row_major(
            Idx[N],
            Idx[D_out],
            Idx[H_out],
            Idx[W_out],
            Idx[C_out],
        ),
    )
    var output_miopen_tt = TileTensor(
        output_miopen_dev,
        row_major(
            Idx[N],
            Idx[D_out],
            Idx[H_out],
            Idx[W_out],
            Idx[C_out],
        ),
    )

    var stride_idx = IndexList[3](stride_d, stride_h, stride_w)
    var dilation_idx = IndexList[3](1, 1, 1)
    var pad_idx = IndexList[3](pad_d, pad_h, pad_w)

    # -------- MIOpen reference (3D conv, QRSCF filter) --------
    conv_miopen[conv_rank=3, filter_is_fcrs=False](
        input_tt,
        filter_qrscf_tt,
        output_miopen_tt,
        stride_idx,
        dilation_idx,
        pad_idx,
        1,  # num_groups
        ctx,
    )

    # -------- Native 4-wave conv3d, QRSCF filter --------
    var qrscf_accepted = dispatch_amd_4wave_conv3d[
        input_type=DType.bfloat16,
        filter_type=DType.bfloat16,
        output_type=DType.bfloat16,
        filter_is_fcqrs=False,
    ](
        input_tt,
        filter_qrscf_tt,
        output_qrscf_tt,
        stride_idx,
        dilation_idx,
        pad_idx,
        1,
        ctx,
    )
    if not qrscf_accepted:
        raise Error(
            "QRSCF leg: dispatch_amd_4wave_conv3d declined a shape it"
            " should have handled — gate logic regression"
        )

    # -------- Native 4-wave conv3d, FCQRS filter --------
    var fcqrs_accepted = dispatch_amd_4wave_conv3d[
        input_type=DType.bfloat16,
        filter_type=DType.bfloat16,
        output_type=DType.bfloat16,
        filter_is_fcqrs=True,
    ](
        input_tt,
        filter_fcqrs_tt,
        output_fcqrs_tt,
        stride_idx,
        dilation_idx,
        pad_idx,
        1,
        ctx,
    )
    if not fcqrs_accepted:
        raise Error(
            "FCQRS leg: dispatch_amd_4wave_conv3d declined a shape it"
            " should have handled — gate logic regression"
        )

    # -------- Compare both legs to MIOpen --------
    comptime output_elems = M_total * C_out
    var output_qrscf_host_cmp = ctx.enqueue_create_host_buffer[.bfloat16](
        output_elems
    )
    var output_fcqrs_host_cmp = ctx.enqueue_create_host_buffer[.bfloat16](
        output_elems
    )
    var output_miopen_host_cmp = ctx.enqueue_create_host_buffer[.bfloat16](
        output_elems
    )
    ctx.enqueue_copy(output_qrscf_host_cmp, output_qrscf_dev)
    ctx.enqueue_copy(output_fcqrs_host_cmp, output_fcqrs_dev)
    ctx.enqueue_copy(output_miopen_host_cmp, output_miopen_dev)
    ctx.synchronize()

    var max_diff_qrscf: Float32 = 0
    var sum_abs_diff_qrscf: Float32 = 0
    var sum_abs_ref: Float32 = 0
    var max_diff_fcqrs: Float32 = 0
    var sum_abs_diff_fcqrs: Float32 = 0
    for i in range(output_elems):
        var q_v = Float32(output_qrscf_host_cmp[i])
        var f_v = Float32(output_fcqrs_host_cmp[i])
        var m_v = Float32(output_miopen_host_cmp[i])
        var dq = abs(q_v - m_v)
        var df = abs(f_v - m_v)
        if dq > max_diff_qrscf:
            max_diff_qrscf = dq
        if df > max_diff_fcqrs:
            max_diff_fcqrs = df
        sum_abs_diff_qrscf += dq
        sum_abs_diff_fcqrs += df
        sum_abs_ref += abs(m_v)
    var rel_q = (
        sum_abs_diff_qrscf / sum_abs_ref if sum_abs_ref > 0 else Float32(0)
    )
    var rel_f = (
        sum_abs_diff_fcqrs / sum_abs_ref if sum_abs_ref > 0 else Float32(0)
    )
    var qrscf_pass = (max_diff_qrscf < abs_tolerance) or (rel_q < 0.01)
    var fcqrs_pass = (max_diff_fcqrs < abs_tolerance) or (rel_f < 0.01)
    print(
        "  QRSCF: max_abs_diff=",
        max_diff_qrscf,
        " L1_rel=",
        rel_q,
        " pass=",
        qrscf_pass,
    )
    print(
        "  FCQRS: max_abs_diff=",
        max_diff_fcqrs,
        " L1_rel=",
        rel_f,
        " pass=",
        fcqrs_pass,
    )
    if not qrscf_pass:
        raise Error("QRSCF leg failed correctness vs MIOpen")
    if not fcqrs_pass:
        raise Error("FCQRS leg failed correctness vs MIOpen")
    print("  PASS")


def main() raises:
    with DeviceContext() as ctx:
        # All test shapes require M_total = N*D_out*H_out*W_out to be
        # a multiple of 64 (the dispatcher's static-launch block-tiling
        # requirement). 32×32 spatial keeps the shapes small + divisible.

        # Typical WAN VAE-style 3×3×3, C_in=C_out=192. pad_d=0 mirrors
        # the CausalConv3dCached pattern (temporal pre-pad happens
        # outside the conv). D_out = D - Q + 1 = 3. M = 1·3·32·32 = 3072.
        test_4wave_conv3d_vs_miopen[
            N=1,
            D=5,
            H=32,
            W=32,
            C_in=192,
            C_out=192,
            D_out=3,
            H_out=32,
            W_out=32,
            Q=3,
            R=3,
            S=3,
            pad_d=0,
            pad_h=1,
            pad_w=1,
        ](ctx)

        # C_in=16 — at the bf16 simd_width=8 boundary; the WAN VAE
        # `decoder.conv_in` shape. Exercises the per-lane substrip
        # slow path inside the loader.
        test_4wave_conv3d_vs_miopen[
            N=1,
            D=5,
            H=32,
            W=32,
            C_in=16,
            C_out=384,
            D_out=3,
            H_out=32,
            W_out=32,
            Q=3,
            R=3,
            S=3,
            pad_d=0,
            pad_h=1,
            pad_w=1,
        ](ctx)

        # pad_d=1 (symmetric temporal). Halo lanes on the D axis must
        # route to the SRD-OOB sentinel; broader halo than the cached
        # CausalConv3d shape. D_out = D + 2*pad_d - Q + 1 = 7. M=7168.
        test_4wave_conv3d_vs_miopen[
            N=1,
            D=7,
            H=32,
            W=32,
            C_in=192,
            C_out=192,
            D_out=7,
            H_out=32,
            W_out=32,
            Q=3,
            R=3,
            S=3,
            pad_d=1,
            pad_h=1,
            pad_w=1,
        ](ctx)

        # Temporal downsample: stride_d=stride_h=stride_w=2. Tests the
        # (stride, pad_d, pad_hw) static-launch enumeration.
        # D_out = (7 - 3) // 2 + 1 = 3, H/W_out = (32+2-3)//2+1 = 16.
        # M = 1·3·16·16 = 768.
        test_4wave_conv3d_vs_miopen[
            N=1,
            D=7,
            H=32,
            W=32,
            C_in=192,
            C_out=192,
            D_out=3,
            H_out=16,
            W_out=16,
            Q=3,
            R=3,
            S=3,
            stride_d=2,
            stride_h=2,
            stride_w=2,
            pad_d=0,
            pad_h=1,
            pad_w=1,
        ](ctx)
