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
"""Host-side tests for the FP6 encode/decode helpers.

These exercise pure numerics, so they run anywhere -- no AMD GPU required. The
one thing they deliberately do NOT establish is whether `pack_fp6_x4`'s bit
order matches what the CDNA4 `f8f6f4` MFMA expects in its 24-byte fragment;
that needs MI355 hardware and belongs with the kernel work.
"""

from std.memory import bitcast
from std.testing import assert_equal, assert_true, TestSuite

from linalg.fp6_utils import (
    FP6Format,
    compute_mxfp6_even_scale,
    decode_fp6_to_bf16,
    decode_fp6_to_f32,
    encode_f32_to_fp6,
    fp6_reference_table,
    pack_fp6_x4,
    unpack_fp6_x4,
    unpack_fp6_x32,
)


def _assert_bit_identical[
    fmt: FP6Format
](code: Int, got: Float32, expected: Float32) raises:
    """Compares as raw bits so `+0.0` and `-0.0` do not compare equal."""
    assert_equal(
        Int(bitcast[.uint32](got)),
        Int(bitcast[.uint32](expected)),
        String("code ") + String(code) + " decoded to the wrong bit pattern",
    )


def _check_decode_matches_table[fmt: FP6Format]() raises:
    comptime table = fp6_reference_table[fmt]()
    for code in range(64):
        var got = decode_fp6_to_f32[fmt](UInt8(UInt8(code)))
        _assert_bit_identical[fmt](code, got[0], table[code])


def test_decode_e2m3_matches_table() raises:
    _check_decode_matches_table[FP6Format.E2M3]()


def test_decode_e3m2_matches_table() raises:
    _check_decode_matches_table[FP6Format.E3M2]()


def _check_decode_vectorized[fmt: FP6Format]() raises:
    """The width-16 path must agree with the width-1 path lane for lane."""
    comptime table = fp6_reference_table[fmt]()
    for base in range(0, 64, 16):
        var codes = SIMD[.uint8, 16](0)
        for lane in range(16):
            codes[lane] = UInt8(base + lane)
        var got = decode_fp6_to_f32[fmt](codes)
        for lane in range(16):
            _assert_bit_identical[fmt](
                base + lane, got[lane], table[base + lane]
            )


def test_decode_e2m3_vectorized() raises:
    _check_decode_vectorized[FP6Format.E2M3]()


def test_decode_e3m2_vectorized() raises:
    _check_decode_vectorized[FP6Format.E3M2]()


def _check_decode_bf16_exact[fmt: FP6Format]() raises:
    """Every FP6 value fits bfloat16 exactly, so the cast must not round."""
    for code in range(64):
        var as_f32 = decode_fp6_to_f32[fmt](UInt8(UInt8(code)))
        var as_bf16 = decode_fp6_to_bf16[fmt](UInt8(UInt8(code)))
        assert_equal(
            Int(bitcast[.uint32](as_bf16[0].cast[.float32]())),
            Int(bitcast[.uint32](as_f32[0])),
            String("code ") + String(code) + " lost precision in bfloat16",
        )


def test_decode_e2m3_bf16_exact() raises:
    _check_decode_bf16_exact[FP6Format.E2M3]()


def test_decode_e3m2_bf16_exact() raises:
    _check_decode_bf16_exact[FP6Format.E3M2]()


def _check_encode_round_trip[fmt: FP6Format]() raises:
    """Decode then encode must return the original code for all 64 codes."""
    for code in range(64):
        var value = decode_fp6_to_f32[fmt](UInt8(UInt8(code)))
        var back = encode_f32_to_fp6[fmt](value)
        assert_equal(
            Int(back[0]),
            code,
            String("code ") + String(code) + " did not survive a round trip",
        )


def test_encode_e2m3_round_trip() raises:
    _check_encode_round_trip[FP6Format.E2M3]()


def test_encode_e3m2_round_trip() raises:
    _check_encode_round_trip[FP6Format.E3M2]()


def test_encode_e2m3_ties_to_even() raises:
    """Midpoints must resolve toward the even mantissa, in both regimes."""
    comptime fmt = FP6Format.E2M3

    # Subnormal grid (step 0.125): 0.0625 sits between codes 0 and 1 -> 0.
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(0.0625))[0]), 0)
    # 0.1875 sits between codes 1 and 2 -> 2 (even mantissa).
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(0.1875))[0]), 2)

    # Normal regime: 3.125 lies between 3.0 (mantissa 4) and 3.25 (mantissa 5).
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(3.125))[0]), 20)
    # 3.375 lies between 3.25 (mantissa 5) and 3.5 (mantissa 6) -> 6.
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(3.375))[0]), 22)


def test_encode_e3m2_ties_to_even() raises:
    comptime fmt = FP6Format.E3M2

    # Subnormal grid (step 0.0625): 0.03125 -> code 0, 0.09375 -> code 2.
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(0.03125))[0]), 0)
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(0.09375))[0]), 2)

    # Normal regime: 1.125 lies between 1.0 (mantissa 0) and 1.25 (mantissa 1).
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(1.125))[0]), 12)


def _check_encode_saturates[fmt: FP6Format]() raises:
    """FP6 has no Inf, so out-of-range magnitudes clamp to the extreme code."""
    comptime max_magnitude_code = 31
    var big = encode_f32_to_fp6[fmt](Float32(1.0e30))
    assert_equal(Int(big[0]), max_magnitude_code)

    var negative_big = encode_f32_to_fp6[fmt](Float32(-1.0e30))
    assert_equal(Int(negative_big[0]), max_magnitude_code | 0x20)

    # Exactly the format maximum must encode to the same code, not overflow.
    comptime table = fp6_reference_table[fmt]()
    var at_max = encode_f32_to_fp6[fmt](Float32(table[max_magnitude_code]))
    assert_equal(Int(at_max[0]), max_magnitude_code)


def test_encode_e2m3_saturates() raises:
    _check_encode_saturates[FP6Format.E2M3]()


def test_encode_e3m2_saturates() raises:
    _check_encode_saturates[FP6Format.E3M2]()


def _check_encode_preserves_signed_zero[fmt: FP6Format]() raises:
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(0.0))[0]), 0)
    assert_equal(Int(encode_f32_to_fp6[fmt](Float32(-0.0))[0]), 0x20)


def test_encode_e2m3_signed_zero() raises:
    _check_encode_preserves_signed_zero[FP6Format.E2M3]()


def test_encode_e3m2_signed_zero() raises:
    _check_encode_preserves_signed_zero[FP6Format.E3M2]()


def test_pack_unpack_round_trip() raises:
    """Every 4-code group must survive pack -> unpack unchanged."""
    # Sweeping all 64^4 groups is wasteful; stride the space instead so each
    # element position sees every code and the byte-straddling positions (1 and
    # 2) see many neighbour combinations.
    for a in range(64):
        for b in range(0, 64, 7):
            for c in range(0, 64, 5):
                for d in range(0, 64, 11):
                    var original = SIMD[.uint8, 4](
                        UInt8(a), UInt8(b), UInt8(c), UInt8(d)
                    )
                    var restored = unpack_fp6_x4(pack_fp6_x4(original))
                    for i in range(4):
                        assert_equal(
                            Int(restored[i]),
                            Int(original[i]),
                            "pack/unpack round trip changed an element",
                        )


def test_pack_uses_all_24_bits() raises:
    """No packed bit may be dropped, and none may spill past bit 23."""
    assert_equal(
        Int(pack_fp6_x4(SIMD[.uint8, 4](0, 0, 0, 0))),
        0,
        "all-zero group must pack to zero",
    )
    assert_equal(
        Int(pack_fp6_x4(SIMD[.uint8, 4](63, 63, 63, 63))),
        0xFFFFFF,
        "all-ones group must fill exactly bits 23:0",
    )

    # Each element must land in its own 6-bit slot.
    for i in range(4):
        var group = SIMD[.uint8, 4](0)
        group[i] = 63
        assert_equal(
            Int(pack_fp6_x4(group)),
            0x3F << (6 * i),
            "element landed in the wrong bit slot",
        )


def test_unpack_x32_matches_scalar_path() raises:
    """The 32-element bulk unpack must agree with eight `unpack_fp6_x4` calls.
    """
    var fragment = SIMD[.uint8, 32](0)
    var expected = SIMD[.uint8, 32](0)

    for group in range(8):
        var quad = SIMD[.uint8, 4](0)
        for j in range(4):
            # Spread codes so no two groups share a pattern and every bit
            # position in the 24-bit group varies across the fragment.
            quad[j] = UInt8((group * 7 + j * 13 + 5) % 64)
        var packed = pack_fp6_x4(quad)
        for b in range(3):
            fragment[group * 3 + b] = UInt8(
                (packed >> UInt32(b * 8)) & UInt32(0xFF)
            )
        for j in range(4):
            expected[group * 4 + j] = quad[j]

    # Poison the padding bytes: the unpack must ignore lanes 24..31.
    for i in range(24, 32):
        fragment[i] = UInt8(0xFF)

    var got = unpack_fp6_x32(fragment)
    for i in range(32):
        assert_equal(
            Int(got[i]),
            Int(expected[i]),
            String("element ") + String(i) + " unpacked incorrectly",
        )


def _check_even_scale_round_trip[fmt: FP6Format]() raises:
    """The scale must let the block maximum survive a quantize/dequantize.

    Note the contract deliberately being tested. Even-mode rounding does NOT
    guarantee `block_max / scale <= format_max`: a block max of 7.6 in E2M3
    gets scale 1.0 and overshoots 7.5, then rounds back down to it, which is
    the right trade because it keeps a finer grid for the other 31 elements.
    So the assertion is on round-trip error, not on the normalized magnitude.
    """
    comptime M = fmt.mantissa_width()
    # Worst-case relative error of round-to-nearest with M mantissa bits is
    # half an ulp, i.e. 2^-(M+1). The block max always lands in the normal
    # range, where that bound holds.
    comptime tolerance = (1.0 / Float32(1 << (M + 1))) * 1.001

    var probes = SIMD[.float32, 8](
        1.0e-6, 0.01, 0.5, 1.0, 7.6, 100.0, 4096.0, 1.0e12
    )
    for i in range(8):
        var block_max = probes[i]
        var scale = compute_mxfp6_even_scale[fmt](block_max)
        var scale_f32 = scale.cast[.float32]()
        assert_true(scale_f32 > 0.0, "scale must be positive and finite")

        var code = encode_f32_to_fp6[fmt](Float32(block_max / scale_f32))
        var recovered = decode_fp6_to_f32[fmt](code)[0] * scale_f32
        var relative_error = abs(recovered - block_max) / block_max
        assert_true(
            relative_error <= tolerance,
            String("block max ")
            + String(block_max)
            + " round-tripped to "
            + String(recovered),
        )


def test_compute_even_scale_e2m3() raises:
    _check_even_scale_round_trip[FP6Format.E2M3]()


def test_compute_even_scale_e3m2() raises:
    _check_even_scale_round_trip[FP6Format.E3M2]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
