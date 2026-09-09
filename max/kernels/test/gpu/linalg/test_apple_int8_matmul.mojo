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
"""Correctness tests for the Apple M5 int8 W8A8 matmul (`int8_matmul.mojo`).

Three stages, all gated on Apple M5 (`compute_capability == 5`):

- Stage 1 (`_run_gemm_vs_quant_ref`): the W8A8 GEMM+dequant kernel vs an fp32
  reference that applies the IDENTICAL symmetric-absmax int8 quant to A and B,
  does the fp32 matmul of the quantized *ints*, and dequantizes
  (`int32 * a_scale[row] * b_scale[col]`). This isolates kernel / layout /
  accumulate / epilogue bugs from the quantization error itself; it must match to
  bf16-store rounding. Covers clean-interior, ragged M/N, and K-tail shapes plus
  the FLUX.2-Klein Linear shapes, and the optional-bias path.

- Stage 2 (`_run_act_quant`): the online per-row activation-quant kernel
  (bf16 -> int8 + per-row fp32 scale) vs the host symmetric-absmax/127 quant.

- Stage 3 (`_run_end_to_end`): the full pipeline (online-quant A, then the W8A8
  GEMM with pre-quantized B) vs the fp32 reference, so the two device kernels are
  validated together.

The int8 quant error *vs a true bf16 matmul* is deliberately NOT asserted here
(that is a model-quality question, gated at the render level when the path is
wired into FLUX). These tests pin the kernel numerics, not the quant scheme's
accuracy.
"""

from std.collections import Optional
from std.random import random_float64, random_si64, seed
from max.gpu.host import DeviceContext, HostBuffer
from std.math import round

from layout import TileTensor
from layout.tile_layout import row_major

from linalg.matmul.gpu.apple.int8_matmul import (
    enqueue_apple_int8_matmul,
    enqueue_apple_int8_quantize_activation,
)


# ===----------------------------------------------------------------------=== #
# Host helpers
# ===----------------------------------------------------------------------=== #


def _host_quant_rows(
    fp: HostBuffer[.float32],
    q: HostBuffer[.int8],
    scale: HostBuffer[.float32],
    rows: Int,
    cols: Int,
):
    """Per-row symmetric-absmax/127 int8 quant (the reference scheme)."""
    for i in range(rows):
        var amax = Float32(0)
        for j in range(cols):
            amax = max(amax, abs(fp[i * cols + j]))
        scale[i] = amax / 127.0 if amax != 0.0 else Float32(0)
        var mult = 127.0 / amax if amax != 0.0 else Float32(0)
        for j in range(cols):
            var qi = Int(round(fp[i * cols + j] * mult))
            q[i * cols + j] = Int8(qi)


def _fill_fp32(buf: HostBuffer[.float32], count: Int, lo: Float64, hi: Float64):
    for i in range(count):
        buf[i] = random_float64(lo, hi).cast[.float32]()


# ===----------------------------------------------------------------------=== #
# Stage 1: GEMM+dequant vs the fp32-of-quantized-ints reference
# ===----------------------------------------------------------------------=== #


def _run_gemm_vs_quant_ref[
    with_bias: Bool
](
    ctx: DeviceContext,
    M: Int,
    N: Int,
    K: Int,
    name: String,
    i32_override: Optional[Bool] = None,
) raises:
    print("== stage1", name, M, "x", N, "x", K, "bias", with_bias)

    var af = ctx.enqueue_create_host_buffer[.float32](M * K)
    var bf = ctx.enqueue_create_host_buffer[.float32](N * K)
    _fill_fp32(af, M * K, -1.0, 1.0)
    _fill_fp32(bf, N * K, -1.0, 1.0)

    var aq = ctx.enqueue_create_host_buffer[.int8](M * K)
    var bq = ctx.enqueue_create_host_buffer[.int8](N * K)
    var asc = ctx.enqueue_create_host_buffer[.float32](M)
    var bsc = ctx.enqueue_create_host_buffer[.float32](N)
    _host_quant_rows(af, aq, asc, M, K)  # A per-row
    _host_quant_rows(bf, bq, bsc, N, K)  # B per-column == per-row of B(N,K)

    var bias_h = ctx.enqueue_create_host_buffer[.bfloat16](max(N, 1))
    for j in range(N):
        bias_h[j] = random_si64(Int64(-2), Int64(2)).cast[.bfloat16]()

    var ad = ctx.enqueue_create_buffer[.int8](M * K)
    var bd = ctx.enqueue_create_buffer[.int8](N * K)
    var asd = ctx.enqueue_create_buffer[.float32](M)
    var bsd = ctx.enqueue_create_buffer[.float32](N)
    var biasd = ctx.enqueue_create_buffer[.bfloat16](max(N, 1))
    var cd = ctx.enqueue_create_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(ad, aq)
    ctx.enqueue_copy(bd, bq)
    ctx.enqueue_copy(asd, asc)
    ctx.enqueue_copy(bsd, bsc)
    ctx.enqueue_copy(biasd, bias_h)

    var a_tt = TileTensor(ad.unsafe_ptr(), row_major(M, K)).as_immut()
    var b_tt = TileTensor(bd.unsafe_ptr(), row_major(N, K)).as_immut()
    var as_tt = TileTensor(asd.unsafe_ptr(), row_major(M)).as_immut()
    var bs_tt = TileTensor(bsd.unsafe_ptr(), row_major(N)).as_immut()
    var bias_tt = TileTensor(
        biasd.unsafe_ptr(), row_major(max(N, 1))
    ).as_immut()
    var c_tt = TileTensor(cd.unsafe_ptr(), row_major(M, N))

    enqueue_apple_int8_matmul[c_type=.bfloat16, has_bias=with_bias](
        c_tt, a_tt, b_tt, as_tt, bs_tt, bias_tt, ctx, i32_override
    )

    var c_h = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(c_h, cd)
    ctx.synchronize()

    _ = ad^
    _ = bd^
    _ = asd^
    _ = bsd^
    _ = biasd^
    _ = cd^

    var pass_ = True
    var maxrel = Float32(0)
    for i in range(M):
        for j in range(N):
            var acc = 0
            for kk in range(K):
                acc += Int(aq[i * K + kk]) * Int(bq[j * K + kk])
            var expected = Float32(acc) * asc[i] * bsc[j]
            comptime if with_bias:
                expected += Float32(bias_h[j])
            var got = Float32(c_h[i * N + j])
            var rel = abs(got - expected) / (abs(expected) + Float32(1e-3))
            maxrel = max(maxrel, rel)
            # bf16 store: ~2^-8 relative + a small absolute floor.
            if abs(got - expected) > Float32(2e-2) + Float32(1.0e-2) * abs(
                expected
            ):
                if pass_:
                    print("FAIL:", i, j, "got", got, "exp", expected)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS maxrel=", maxrel)


# ===----------------------------------------------------------------------=== #
# Stage 2: online per-row activation quant vs host quant
# ===----------------------------------------------------------------------=== #


def _run_act_quant(ctx: DeviceContext, M: Int, K: Int, name: String) raises:
    print("== stage2 act-quant", name, M, "x", K)

    var af = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    for i in range(M * K):
        af[i] = (random_float64(-1.0, 1.0)).cast[.bfloat16]()

    var ad = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var qd = ctx.enqueue_create_buffer[.int8](M * K)
    var sd = ctx.enqueue_create_buffer[.float32](M)
    ctx.enqueue_copy(ad, af)

    var a_tt = TileTensor(ad.unsafe_ptr(), row_major(M, K)).as_immut()
    var q_tt = TileTensor(qd.unsafe_ptr(), row_major(M, K))
    var s_tt = TileTensor(sd.unsafe_ptr(), row_major(M))
    enqueue_apple_int8_quantize_activation[.bfloat16](q_tt, a_tt, s_tt, ctx)

    var q_h = ctx.enqueue_create_host_buffer[.int8](M * K)
    var s_h = ctx.enqueue_create_host_buffer[.float32](M)
    ctx.enqueue_copy(q_h, qd)
    ctx.enqueue_copy(s_h, sd)
    ctx.synchronize()
    _ = ad^
    _ = qd^
    _ = sd^

    # Host reference (same absmax/127 over the bf16-rounded values).
    var pass_ = True
    for i in range(M):
        var amax = Float32(0)
        for j in range(K):
            amax = max(amax, abs(Float32(af[i * K + j])))
        var exp_scale = amax / 127.0 if amax != 0.0 else Float32(0)
        if abs(s_h[i] - exp_scale) > Float32(1e-6) * (exp_scale + 1.0):
            if pass_:
                print("FAIL scale row", i, "got", s_h[i], "exp", exp_scale)
            pass_ = False
        var mult = 127.0 / amax if amax != 0.0 else Float32(0)
        for j in range(K):
            var qi = Int(round(Float32(af[i * K + j]) * mult))
            if Int(q_h[i * K + j]) != qi:
                if pass_:
                    print("FAIL q", i, j, "got", Int(q_h[i * K + j]), "exp", qi)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Stage 3: end-to-end (device quant A -> device W8A8 GEMM) vs fp32 ref
# ===----------------------------------------------------------------------=== #


def _run_end_to_end(
    ctx: DeviceContext, M: Int, N: Int, K: Int, name: String
) raises:
    print("== stage3 e2e", name, M, "x", N, "x", K)

    var af = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var bf = ctx.enqueue_create_host_buffer[.float32](N * K)
    for i in range(M * K):
        af[i] = (random_float64(-1.0, 1.0)).cast[.bfloat16]()
    _fill_fp32(bf, N * K, -1.0, 1.0)

    # B pre-quantized on host (weight-load path); A quantized on device.
    var bq = ctx.enqueue_create_host_buffer[.int8](N * K)
    var bsc = ctx.enqueue_create_host_buffer[.float32](N)
    _host_quant_rows(bf, bq, bsc, N, K)

    var ad = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var aqd = ctx.enqueue_create_buffer[.int8](M * K)
    var asd = ctx.enqueue_create_buffer[.float32](M)
    var bd = ctx.enqueue_create_buffer[.int8](N * K)
    var bsd = ctx.enqueue_create_buffer[.float32](N)
    var biasd = ctx.enqueue_create_buffer[.bfloat16](max(N, 1))
    var cd = ctx.enqueue_create_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(ad, af)
    ctx.enqueue_copy(bd, bq)
    ctx.enqueue_copy(bsd, bsc)

    var a_tt = TileTensor(ad.unsafe_ptr(), row_major(M, K)).as_immut()
    var aq_tt = TileTensor(aqd.unsafe_ptr(), row_major(M, K))
    var as_tt = TileTensor(asd.unsafe_ptr(), row_major(M))
    enqueue_apple_int8_quantize_activation[.bfloat16](aq_tt, a_tt, as_tt, ctx)

    var b_tt = TileTensor(bd.unsafe_ptr(), row_major(N, K)).as_immut()
    var bs_tt = TileTensor(bsd.unsafe_ptr(), row_major(N)).as_immut()
    var bias_tt = TileTensor(
        biasd.unsafe_ptr(), row_major(max(N, 1))
    ).as_immut()
    var c_tt = TileTensor(cd.unsafe_ptr(), row_major(M, N))
    enqueue_apple_int8_matmul[c_type=.bfloat16, has_bias=False](
        c_tt, aq_tt.as_immut(), b_tt, as_tt.as_immut(), bs_tt, bias_tt, ctx
    )

    # Read back the device-quantized A + scales for the reference.
    var aq_h = ctx.enqueue_create_host_buffer[.int8](M * K)
    var as_h = ctx.enqueue_create_host_buffer[.float32](M)
    var c_h = ctx.enqueue_create_host_buffer[.bfloat16](M * N)
    ctx.enqueue_copy(aq_h, aqd)
    ctx.enqueue_copy(as_h, asd)
    ctx.enqueue_copy(c_h, cd)
    ctx.synchronize()
    _ = ad^
    _ = aqd^
    _ = asd^
    _ = bd^
    _ = bsd^
    _ = biasd^
    _ = cd^

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = 0
            for kk in range(K):
                acc += Int(aq_h[i * K + kk]) * Int(bq[j * K + kk])
            var expected = Float32(acc) * as_h[i] * bsc[j]
            var got = Float32(c_h[i * N + j])
            if abs(got - expected) > Float32(2e-2) + Float32(1.0e-2) * abs(
                expected
            ):
                if pass_:
                    print("FAIL:", i, j, "got", got, "exp", expected)
                pass_ = False
    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


# ===----------------------------------------------------------------------=== #
# Test entry points
# ===----------------------------------------------------------------------=== #


def test_gemm(ctx: DeviceContext) raises:
    seed(0)
    # Clean interior.
    _run_gemm_vs_quant_ref[False](ctx, 64, 64, 64, "clean-single-tile")
    _run_gemm_vs_quant_ref[False](ctx, 128, 128, 256, "clean-multi-tile")
    # Ragged M/N (edge tiles).
    _run_gemm_vs_quant_ref[False](ctx, 100, 200, 64, "ragged-mn")
    _run_gemm_vs_quant_ref[False](ctx, 96, 96, 96, "ragged-96")
    # K tail (K not a multiple of BK=64), M/N-aligned. `k-tail-80`/`k-tail-48`
    # are K % 16 == 0 (width-16 full strips + masked tail); `k-tail-144` has two
    # full width-16 strips before the masked tail; `k-tail-130` is K % 16 != 0,
    # exercising the masked width-4 full-strip path (the case the pre-#91003
    # bounded fallback made slow).
    _run_gemm_vs_quant_ref[False](ctx, 64, 64, 80, "k-tail-80")
    _run_gemm_vs_quant_ref[False](ctx, 128, 128, 48, "k-tail-48")
    _run_gemm_vs_quant_ref[False](ctx, 64, 64, 144, "k-tail-144")
    _run_gemm_vs_quant_ref[False](ctx, 128, 256, 130, "k-tail-130")
    # Bias path.
    _run_gemm_vs_quant_ref[True](ctx, 128, 128, 64, "bias-clean")
    _run_gemm_vs_quant_ref[True](ctx, 100, 200, 64, "bias-ragged")
    # FLUX.2-Klein Linear shapes at a small M (host ref is O(M*N*K)).
    _run_gemm_vs_quant_ref[False](ctx, 64, 3072, 3072, "klein-attn")
    _run_gemm_vs_quant_ref[False](ctx, 64, 3072, 9216, "klein-ff-out")
    # Force both interior legs on one clean K%16==0 shape: the i64 `_mma_width16`
    # is otherwise dead in CI (clean shapes auto-route to the i32 path), so a
    # regression to it would pass unnoticed.
    _run_gemm_vs_quant_ref[False](
        ctx, 128, 128, 256, "interior-i64-forced", i32_override=False
    )
    _run_gemm_vs_quant_ref[False](
        ctx, 128, 128, 256, "interior-i32-forced", i32_override=True
    )


def test_act_quant(ctx: DeviceContext) raises:
    seed(1)
    _run_act_quant(ctx, 64, 3072, "klein-K3072")
    _run_act_quant(ctx, 100, 512, "ragged-m")
    _run_act_quant(ctx, 8, 12288, "klein-K12288")


def test_end_to_end(ctx: DeviceContext) raises:
    seed(2)
    _run_end_to_end(ctx, 128, 256, 128, "clean")
    _run_end_to_end(ctx, 100, 200, 64, "ragged")
    _run_end_to_end(ctx, 64, 3072, 3072, "klein-attn")


def main() raises:
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 (compute_capability == 5) required")
        return
    test_gemm(ctx)
    test_act_quant(ctx)
    test_end_to_end(ctx)
    print("ALL TESTS PASSED")
