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
"""Cross-dtype correctness sweep for amd_4wave_conv on MI355X.

Black-box correctness check covering all three supported dtypes
(FP8, BF16, FP16) through `AMD4WaveMatmul.run_conv2d`. Per-dtype CI
targets are generated from the BUILD file via `-D DTYPE=<dtype>`
(same pattern as `test_4wave_matmul.mojo`). The reference is
`structured_4wave_matmul` fed a host-materialized im2col matrix,
so equality is byte-identical — the test fails on any mismatch.

Shape regimes (each dtype runs all eight):
  - 1×1 pointwise (address math collapses to `m*C + k`).
  - 3×3 interior (uniform-substrip path).
  - 3×3 pad=1 (halo lanes route to SRD-OOB sentinel).
  - 3×3 stride=2 pad=1 (downsampling).
  - 3×3 C_in<BK (per-lane substrip slow path).
  - 3×3 dilation=2 pad=2 (dilation > 1).
  - 3×3 N=2 (batched M = N*H*W decomposition).
  - 3×3 C_in=384 K-padded (large K-pad alignment).

The FP8 1×1 case (K_padded == 2*BK == 256 → single outer K-iter)
historically hit a kernel-scheduling corner: the framework-scheduled
body's per-block emit issues frag-loads BEFORE the per-block
`wait_vm[0]`, so when the main loop runs zero times the first
epilogue block's `ds_read` raced with the prologue's still-in-flight
`buffer_load_lds → LDS` writes. The kernel now emits a
`wait_vm(0) + wait_lgkm(0) + s_barrier` drain between prologue and
main loop when `num_K_iters_static == 1`, mirroring the handwritten
body's top-of-iter sync — both BM=64 and BM=128 are correct at this
corner.
"""

from max.gpu.host import DeviceContext
from std.random import rand
from std.sys import get_defined_dtype

from layout import TileTensor, row_major

from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul
from nn.conv.gpu.amd.amd_4wave_conv import amd_4wave_conv


# Dtype to test, set via `-D DTYPE=<dtype>` in BUILD.bazel. Defaults to
# BF16 so `mojo test_amd_4wave_conv_dtypes.mojo` (no -D) still runs
# locally with sensible behavior.
comptime IN_DTYPE = get_defined_dtype["DTYPE", .bfloat16]()

# Output dtype: BF16 when input is FP8 (the 4-wave kernel doesn't
# support FP8 output today); otherwise matches the input.
comptime OUT_DTYPE = DType.bfloat16 if IN_DTYPE.is_float8() else IN_DTYPE


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
    input_host_ptr: ImmPointer[Scalar[a_type], _],
    im2col_host_ptr: MutPointer[Scalar[a_type], _],
):
    """Materializes the im2col matrix with halo zeros and K-padding zeros."""
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


def test_conv[
    a_type: DType,
    c_type: DType,
    N_batch: Int,
    H: Int,
    W: Int,
    C_in: Int,
    C_out: Int,
    R: Int = 1,
    S: Int = 1,
    pad_h: Int = 0,
    pad_w: Int = 0,
    stride_h: Int = 1,
    stride_w: Int = 1,
    *,
    label: StaticString = "",
](ctx: DeviceContext) raises:
    comptime H_out = (H + 2 * pad_h - R) // stride_h + 1
    comptime W_out = (W + 2 * pad_w - S) // stride_w + 1
    comptime M_total = N_batch * H_out * W_out
    comptime K_real = R * S * C_in
    comptime K_padded = _round_up(K_real, 256)

    print(
        "== ",
        String(a_type),
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
        " stride=",
        stride_h,
        "x",
        stride_w,
        " pad=",
        pad_h,
        "x",
        pad_w,
        " H_out=",
        H_out,
        " W_out=",
        W_out,
        " M=",
        M_total,
        " K_padded=",
        K_padded,
    )

    var input_dev = ctx.enqueue_create_buffer[a_type](N_batch * H * W * C_in)
    var filter_dev = ctx.enqueue_create_buffer[a_type](C_out * K_padded)
    var im2col_dev = ctx.enqueue_create_buffer[a_type](M_total * K_padded)
    var output_conv_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)
    var output_mm_dev = ctx.enqueue_create_buffer[c_type](M_total * C_out)

    var input_host = ctx.enqueue_create_host_buffer[a_type](
        N_batch * H * W * C_in
    )
    var filter_host = ctx.enqueue_create_host_buffer[a_type](C_out * K_padded)
    rand(input_host.unsafe_ptr(), N_batch * H * W * C_in)
    rand(filter_host.unsafe_ptr(), C_out * K_padded)
    # Zero the K-padding region of the filter (caller's responsibility).
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
    var filter_conv = TileTensor(filter_dev, filter_layout)
    var output_conv = TileTensor(output_conv_dev, output_layout)
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
    ](input_nhwc, filter_conv, output_conv, ctx)

    var im2col_2d = TileTensor(im2col_dev, im2col_layout)
    var filter_mm = TileTensor(filter_dev, filter_layout)
    var output_mm = TileTensor(output_mm_dev, output_layout)
    structured_4wave_matmul(im2col_2d, filter_mm, output_mm, ctx)

    var conv_host = ctx.enqueue_create_host_buffer[c_type](M_total * C_out)
    var mm_host = ctx.enqueue_create_host_buffer[c_type](M_total * C_out)
    ctx.enqueue_copy(conv_host, output_conv_dev)
    ctx.enqueue_copy(mm_host, output_mm_dev)
    ctx.synchronize()

    var mismatches = 0
    for i in range(M_total * C_out):
        if conv_host[i] != mm_host[i]:
            mismatches += 1
    if mismatches > 0:
        print("  FAIL:", mismatches, "/", M_total * C_out, "differ")
        raise Error(
            String(a_type) + " amd_4wave_conv != structured_4wave_matmul"
        )
    print("  PASS:", M_total * C_out, "elements byte-identical")


def main() raises:
    with DeviceContext() as ctx:
        # ----- 1×1 pointwise (address math collapses to m*C + k) -----
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=16,
            W=16,
            C_in=256,
            C_out=128,
            label="1x1-pointwise",
        ](ctx)

        # ----- 3×3 interior, no halo (uniform-substrip path) -----
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=0,
            pad_w=0,
            label="3x3-interior",
        ](ctx)

        # ----- 3×3 pad=1 (halo via SRD-OOB sentinel) -----
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
            label="3x3-halo",
        ](ctx)

        # ----- 3×3 stride=2 pad=1 (downsampling) -----
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            stride_h=2,
            stride_w=2,
            pad_h=1,
            pad_w=1,
            label="3x3-stride2",
        ](ctx)

        # ----- 3×3 C_in=64 (K_real=576 ≪ 2*BK=256, per-lane substrip) -----
        # C_in=64 < BK=128 forces the per-lane substrip slow path.
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=16,
            W=16,
            C_in=64,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
            label="3x3-Cin64-perlane",
        ](ctx)

        # ----- 3×3 dilation=2 pad=2 (dilation > 1) -----
        # Tests h_in = h_out*stride_h + kh*dilation_h - pad_h.
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=16,
            W=16,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            stride_h=1,
            stride_w=1,
            pad_h=2,
            pad_w=2,
            label="3x3-dilation2",
        ](ctx)

        # ----- 3×3 N=2 (batched M = N*H*W) -----
        # Exercises the M-decomposition `(n, h_out, w_out)` divmod.
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=2,
            H=14,
            W=14,
            C_in=128,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
            label="3x3-batch2",
        ](ctx)

        # ----- 3×3 K-padded C_in=384 (uniform substrip + large K-pad) -----
        # K_real = 3456, K_padded = 3584 (3.6% padding). Stresses the
        # K-pad zero-fill behavior at a different alignment than C_in=64.
        test_conv[
            IN_DTYPE,
            OUT_DTYPE,
            N_batch=1,
            H=14,
            W=14,
            C_in=384,
            C_out=128,
            R=3,
            S=3,
            pad_h=1,
            pad_w=1,
            label="3x3-Cin384-kpad",
        ](ctx)
