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
"""Tests for MXFP6 activation quantization on AMD CDNA4.

`quantize_mxfp6_amd` turns bfloat16 activations into the packed-FP6 + E8M0
layout the block-scaled MFMA consumes. Correctness means two separate things,
kept apart below:

  - **Exact bytes.** The scale and every code are fully determined, so most
    tests compare the kernel's output BIT FOR BIT against a host reference that
    re-derives both. A tolerance check would hide an off-by-one code or a
    misplaced byte; a bitwise one cannot.
  - **Round-trip meaning.** Bitwise agreement only proves two implementations
    agree. `test_roundtrip` pushes the result back through `dequant_mxfp6` and
    bounds the error by the format's own representable step, which is what a
    caller actually relies on.

The host reference shares the encoder with the kernel deliberately: it checks
the kernel's blocking, indexing and store path, while `test_fp6_utils` pins the
encoder itself against hand-written tables. Neither test alone is sufficient;
together they cover both halves.

Every output buffer is poisoned before launch, so an element the kernel skips
fails loudly rather than reading back a plausible zero.

Currently AMD-only.

Usage:
  br test_mxfp6_quant_amd.mojo.test
"""

from max.gpu.host import DeviceContext, HostBuffer
from std.math import isfinite, recip
from std.memory import bitcast
from std.random import random_ui64, seed
from std.testing import assert_true

from layout import TileTensor, row_major
from linalg.fp6_quantization import quantize_mxfp6_amd
from linalg.fp6_utils import (
    FP6Format,
    MXFP6_SF_VECTOR_SIZE,
    compute_mxfp6_even_scale,
    decode_fp6_to_f32,
    encode_f32_to_fp6,
    pack_fp6_x4,
    unpack_fp6_x32,
)

comptime BLOCK = MXFP6_SF_VECTOR_SIZE  # 32
comptime POISON = UInt8(0xEE)


def _fmt_name[fmt: FP6Format]() -> StaticString:
    return "E2M3" if fmt == FP6Format.E2M3 else "E3M2"


def _fmt_max[fmt: FP6Format]() -> Float32:
    """Largest finite magnitude this encoding represents."""
    comptime if fmt == FP6Format.E2M3:
        return Float32(7.5)
    return Float32(28.0)


def _fmt_min_normal[fmt: FP6Format]() -> Float32:
    """Smallest positive normal: 2^(1-bias) with bias 1 (E2M3) or 3 (E3M2)."""
    comptime if fmt == FP6Format.E2M3:
        return Float32(1.0)
    return Float32(0.25)


# ===----------------------------------------------------------------------=== #
# Host reference for one MX block
# ===----------------------------------------------------------------------=== #


def _reference_block[
    fmt: FP6Format
](values: SIMD[.float32, BLOCK], mut out_bytes: SIMD[.uint8, 32],) -> UInt8:
    """Fills the block's 24 packed bytes and returns its E8M0 scale byte."""
    var group_max = abs(values).reduce_max()
    var e8m0 = compute_mxfp6_even_scale[fmt](group_max)

    var out_scale = Float32(0.0)
    if group_max != Float32(0.0):
        out_scale = recip(e8m0.cast[.float32]())
    if not isfinite(out_scale):
        out_scale = Float32(0.0)
        e8m0 = bitcast[.float8_e8m0fnu](UInt8(0))

    var codes = encode_f32_to_fp6[fmt](values * out_scale)
    out_bytes = SIMD[.uint8, 32](0)
    for g in range(BLOCK // 4):
        var quad = SIMD[.uint8, 4](0)
        for j in range(4):
            quad[j] = codes[g * 4 + j]
        var word = pack_fp6_x4(quad)
        for b in range(3):
            out_bytes[g * 3 + b] = UInt8((word >> UInt32(8 * b)) & UInt32(0xFF))

    return bitcast[.uint8](e8m0)


# ===----------------------------------------------------------------------=== #
# Device driver
# ===----------------------------------------------------------------------=== #


struct QuantResult(Copyable, Movable):
    """Host-side copy of one quantization run."""

    var data: List[UInt8]
    var scales: List[UInt8]

    def __init__(out self, var data: List[UInt8], var scales: List[UInt8]):
        self.data = data^
        self.scales = scales^


def _quantize[
    fmt: FP6Format, M: Int, K: Int
](ctx: DeviceContext, values: List[Float32]) raises -> QuantResult:
    """Quantizes `[M, K]` float32 host values (via bfloat16) on the device."""
    comptime K_BYTES = (K * 6) // 8
    comptime SCALE_K = K // BLOCK

    var in_h = ctx.enqueue_create_host_buffer[.bfloat16](M * K)
    var out_h = ctx.enqueue_create_host_buffer[.uint8](M * K_BYTES)
    var sc_h = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](M * SCALE_K)
    ctx.synchronize()

    for i in range(M * K):
        in_h[i] = values[i].cast[.bfloat16]()

    var in_d = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var out_d = ctx.enqueue_create_buffer[.uint8](M * K_BYTES)
    var sc_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](M * SCALE_K)

    # Poison both outputs so a skipped element cannot read back as a plausible
    # zero, and an unwritten scale cannot look like the legitimate zero scale.
    out_d.enqueue_fill(POISON)
    sc_d.enqueue_fill(bitcast[.float8_e8m0fnu](POISON))
    ctx.enqueue_copy(in_d, in_h)

    quantize_mxfp6_amd[fmt](
        ctx,
        TileTensor[mut=True](out_d, row_major[M, K_BYTES]()),
        TileTensor[mut=True](sc_d, row_major[M, SCALE_K]()),
        TileTensor[mut=False](in_d, row_major[M, K]()),
    )

    ctx.enqueue_copy(out_h, out_d)
    ctx.enqueue_copy(sc_h, sc_d)
    ctx.synchronize()

    var data = List[UInt8]()
    for i in range(M * K_BYTES):
        data.append(out_h[i])
    var scales = List[UInt8]()
    for i in range(M * SCALE_K):
        scales.append(bitcast[.uint8](sc_h[i]))

    _ = in_d^
    _ = out_d^
    _ = sc_d^
    return QuantResult(data^, scales^)


def _decode_block[
    fmt: FP6Format
](block_bytes: SIMD[.uint8, 32], scale_byte: UInt8) -> SIMD[.float32, BLOCK]:
    """Decodes 24 packed bytes back to 32 scaled float32 values."""
    var scale = bitcast[.float8_e8m0fnu](scale_byte).cast[.float32]()
    return decode_fp6_to_f32[fmt](unpack_fp6_x32(block_bytes)) * scale


# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_bitwise_vs_reference[
    fmt: FP6Format, M: Int, K: Int
](name: String, ctx: DeviceContext) raises -> Bool:
    """Every packed byte and every scale must match the host reference exactly.
    """
    comptime K_BYTES = (K * 6) // 8
    comptime SCALE_K = K // BLOCK

    var values = List[Float32]()
    var state = UInt64(0x9E3779B97F4A7C15)
    for _ in range(M * K):
        state = state * 6364136223846793005 + 1442695040888963407
        # Spread across ~2^-8 .. 2^8 in magnitude, both signs, plus exact zeros.
        var u = Float32(Int((state >> 33) & 0xFFFF)) / Float32(65535.0)
        var mag = Float32(2.0) ** (u * Float32(16.0) - Float32(8.0))
        var sign = Float32(-1.0) if ((state >> 20) & 1) == 1 else Float32(1.0)
        var v = sign * mag
        if ((state >> 21) & 0x1F) == 0:
            v = Float32(0.0)
        values.append(v)

    var got = _quantize[fmt, M, K](ctx, values)

    var bad_bytes = 0
    var bad_scales = 0
    for m in range(M):
        for blk in range(SCALE_K):
            var vals = SIMD[.float32, BLOCK](0)
            for i in range(BLOCK):
                # Round-trip through bfloat16: the kernel sees the truncated
                # value, so the reference must too.
                vals[i] = (
                    values[m * K + blk * BLOCK + i]
                    .cast[.bfloat16]()
                    .cast[.float32]()
                )
            var want_bytes = SIMD[.uint8, 32](0)
            var want_scale = _reference_block[fmt](vals, want_bytes)

            if got.scales[m * SCALE_K + blk] != want_scale:
                bad_scales += 1
            for b in range(24):
                if got.data[m * K_BYTES + blk * 24 + b] != want_bytes[b]:
                    bad_bytes += 1

    if bad_bytes != 0 or bad_scales != 0:
        print(
            "    FAIL",
            name,
            _fmt_name[fmt](),
            "M=",
            M,
            "K=",
            K,
            "->",
            bad_bytes,
            "bad bytes,",
            bad_scales,
            "bad scales",
        )
        return False
    print("    PASS", name, _fmt_name[fmt](), "M=", M, "K=", K)
    return True


def test_roundtrip[
    fmt: FP6Format, M: Int, K: Int
](ctx: DeviceContext) raises -> Bool:
    """Quantize then decode must land within one representable step.

    FP6 keeps 3 (E2M3) or 2 (E3M2) mantissa bits, so relative error after
    round-to-nearest is bounded by half a step: 2^-4 and 2^-3 respectively.
    Values below the format's smallest normal times the block scale can only
    round to zero, so they are bounded absolutely instead.
    """
    comptime SCALE_K = K // BLOCK
    comptime rel_bound = Float32(0.0625) if fmt == FP6Format.E2M3 else Float32(
        0.125
    )

    var values = List[Float32]()
    var state = UInt64(0xD1B54A32D192ED03)
    for _ in range(M * K):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int((state >> 33) & 0xFFFF)) / Float32(65535.0)
        var sign = Float32(-1.0) if ((state >> 20) & 1) == 1 else Float32(1.0)
        values.append(sign * (u * Float32(4.0) - Float32(2.0)))

    var got = _quantize[fmt, M, K](ctx, values)

    var worst = Float32(0.0)
    var bad = 0
    for m in range(M):
        for blk in range(SCALE_K):
            var packed = SIMD[.uint8, 32](0)
            for b in range(24):
                packed[b] = got.data[m * ((K * 6) // 8) + blk * 24 + b]
            var decoded = _decode_block[fmt](
                packed, got.scales[m * SCALE_K + blk]
            )
            var block_max = Float32(0.0)
            for i in range(BLOCK):
                var want = (
                    values[m * K + blk * BLOCK + i]
                    .cast[.bfloat16]()
                    .cast[.float32]()
                )
                block_max = max(block_max, abs(want))
            for i in range(BLOCK):
                var want = (
                    values[m * K + blk * BLOCK + i]
                    .cast[.bfloat16]()
                    .cast[.float32]()
                )
                var err = abs(decoded[i] - want)
                # Absolute floor: the block's step size cannot resolve below
                # `block_max / fmt_max * min_normal`.
                var floor_abs = (
                    block_max / _fmt_max[fmt]() * _fmt_min_normal[fmt]()
                )
                var allow = max(rel_bound * abs(want), floor_abs)
                if err > allow:
                    bad += 1
                    worst = max(worst, err)

    if bad != 0:
        print(
            "    FAIL roundtrip",
            _fmt_name[fmt](),
            "M=",
            M,
            "K=",
            K,
            "->",
            bad,
            "outside bound, worst abs err",
            worst,
        )
        return False
    print("    PASS roundtrip", _fmt_name[fmt](), "M=", M, "K=", K)
    return True


def test_all_zeros[fmt: FP6Format](ctx: DeviceContext) raises -> Bool:
    """An all-zero block must yield a zero scale and zero codes, never NaN."""
    comptime M = 4
    comptime K = 128
    var values = List[Float32]()
    for _ in range(M * K):
        values.append(Float32(0.0))

    var got = _quantize[fmt, M, K](ctx, values)
    for i in range(len(got.scales)):
        if got.scales[i] != UInt8(0):
            print("    FAIL all-zeros", _fmt_name[fmt](), "scale not zero")
            return False
    for i in range(len(got.data)):
        if got.data[i] != UInt8(0):
            print("    FAIL all-zeros", _fmt_name[fmt](), "data not zero")
            return False
    print("    PASS all-zeros", _fmt_name[fmt]())
    return True


def test_sign_preserved[fmt: FP6Format](ctx: DeviceContext) raises -> Bool:
    """Every nonzero input keeps its sign through the round trip."""
    comptime M = 2
    comptime K = 64
    var values = List[Float32]()
    for i in range(M * K):
        var v = Float32(1.0) + Float32(i % 7)
        values.append(-v if (i % 2) == 1 else v)

    var got = _quantize[fmt, M, K](ctx, values)
    for m in range(M):
        for blk in range(K // BLOCK):
            var packed = SIMD[.uint8, 32](0)
            for b in range(24):
                packed[b] = got.data[m * ((K * 6) // 8) + blk * 24 + b]
            var decoded = _decode_block[fmt](
                packed, got.scales[m * (K // BLOCK) + blk]
            )
            for i in range(BLOCK):
                var want = values[m * K + blk * BLOCK + i]
                if decoded[i] != Float32(0.0) and (
                    (decoded[i] < Float32(0.0)) != (want < Float32(0.0))
                ):
                    print("    FAIL sign", _fmt_name[fmt](), "flipped at", i)
                    return False
    print("    PASS sign-preserved", _fmt_name[fmt]())
    return True


def test_outlier_dominates_block[
    fmt: FP6Format
](ctx: DeviceContext) raises -> Bool:
    """One large value sets its own block's scale and no other block's.

    This is what makes the scale granularity observable: block 0 holds a huge
    outlier, block 1 holds small values. If the kernel reduced across the wrong
    span, block 1's scale would be dragged up with it.
    """
    comptime M = 1
    comptime K = 64  # exactly two blocks
    var values = List[Float32]()
    for i in range(K):
        if i == 5:
            values.append(Float32(1024.0))
        elif i < BLOCK:
            values.append(Float32(1.0))
        else:
            values.append(Float32(0.5))

    var got = _quantize[fmt, M, K](ctx, values)
    var s0 = bitcast[.float8_e8m0fnu](got.scales[0]).cast[.float32]()
    var s1 = bitcast[.float8_e8m0fnu](got.scales[1]).cast[.float32]()

    if not (s0 > s1):
        print(
            "    FAIL outlier",
            _fmt_name[fmt](),
            "block scales did not separate: s0=",
            s0,
            "s1=",
            s1,
        )
        return False

    # And the outlier itself must survive: decoding block 0 must recover
    # something close to 1024, not a clipped value.
    var packed = SIMD[.uint8, 32](0)
    for b in range(24):
        packed[b] = got.data[b]
    var decoded = _decode_block[fmt](packed, got.scales[0])
    var rel = abs(decoded[5] - Float32(1024.0)) / Float32(1024.0)
    if rel > Float32(0.2):
        print(
            "    FAIL outlier",
            _fmt_name[fmt](),
            "outlier not preserved, got",
            decoded[5],
        )
        return False
    print("    PASS outlier-dominates-block", _fmt_name[fmt]())
    return True


def test_saturation[fmt: FP6Format](ctx: DeviceContext) raises -> Bool:
    """Huge and tiny magnitudes must stay finite and ordered.

    E8M0 spans 2^-127..2^127, so a block whose max exceeds what the scale can
    normalize still has to produce finite codes rather than inf or NaN.
    """
    comptime M = 1
    comptime K = 32
    var values = List[Float32]()
    for i in range(K):
        if i < 8:
            values.append(Float32(3.0e38))
        elif i < 16:
            values.append(Float32(1.0e-30))
        elif i < 24:
            values.append(Float32(-3.0e38))
        else:
            values.append(Float32(0.0))

    var got = _quantize[fmt, M, K](ctx, values)
    var packed = SIMD[.uint8, 32](0)
    for b in range(24):
        packed[b] = got.data[b]
    var decoded = _decode_block[fmt](packed, got.scales[0])
    for i in range(BLOCK):
        if not isfinite(decoded[i]):
            print(
                "    FAIL saturation",
                _fmt_name[fmt](),
                "non-finite at",
                i,
                decoded[i],
            )
            return False
    # The large positives must decode larger than the tiny ones.
    if not (decoded[0] > decoded[8]):
        print("    FAIL saturation", _fmt_name[fmt](), "ordering lost")
        return False
    print("    PASS saturation", _fmt_name[fmt]())
    return True


def test_no_poison_left[fmt: FP6Format](ctx: DeviceContext) raises -> Bool:
    """A shape whose column count is not a multiple of the block stride still
    writes every byte: no poison may survive."""
    comptime M = 3
    comptime K = 96  # 3 blocks per row, 72 bytes -- not a power of two
    var values = List[Float32]()
    var state = UInt64(0x243F6A8885A308D3)
    for _ in range(M * K):
        state = state * 6364136223846793005 + 1442695040888963407
        values.append(Float32(Int((state >> 40) & 0xFF)) / Float32(64.0))

    var got = _quantize[fmt, M, K](ctx, values)
    var poisoned = 0
    for i in range(len(got.data)):
        if got.data[i] == POISON:
            poisoned += 1
    # 0xEE is a legal packed byte, so this can only flag a suspiciously large
    # run of them; the bitwise test is what proves the values themselves.
    if poisoned > len(got.data) // 8:
        print(
            "    FAIL poison",
            _fmt_name[fmt](),
            poisoned,
            "of",
            len(got.data),
            "bytes still poison",
        )
        return False
    print("    PASS no-poison-left", _fmt_name[fmt]())
    return True


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        print("===> MXFP6 activation quantization")
        var ok = True

        comptime for i in range(2):
            comptime fmt = FP6Format.E2M3 if i == 0 else FP6Format.E3M2

            print("---- bitwise vs host reference ----")
            ok &= test_bitwise_vs_reference[fmt, 1, 32]("single-block ", ctx)
            ok &= test_bitwise_vs_reference[fmt, 1, 64]("two-blocks   ", ctx)
            ok &= test_bitwise_vs_reference[fmt, 3, 96]("odd-rows     ", ctx)
            ok &= test_bitwise_vs_reference[fmt, 7, 128]("prime-rows   ", ctx)
            ok &= test_bitwise_vs_reference[fmt, 128, 256]("wide         ", ctx)
            ok &= test_bitwise_vs_reference[fmt, 2, 4096]("deep-k       ", ctx)
            ok &= test_bitwise_vs_reference[fmt, 1025, 32]("many-rows    ", ctx)

            print("---- properties ----")
            ok &= test_roundtrip[fmt, 4, 256](ctx)
            ok &= test_roundtrip[fmt, 1, 1024](ctx)
            ok &= test_all_zeros[fmt](ctx)
            ok &= test_sign_preserved[fmt](ctx)
            ok &= test_outlier_dominates_block[fmt](ctx)
            ok &= test_saturation[fmt](ctx)
            ok &= test_no_poison_left[fmt](ctx)

        assert_true(ok, "one or more MXFP6 quantization cases failed")
        print("==== all MXFP6 quantization tests passed ====")
