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

from std.sys.info import CompilationTarget, is_64bit

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from std.utils.numerics import (
    FPUtils,
    get_accum_type,
    inf,
    isfinite,
    isinf,
    isnan,
    max_finite,
    max_or_inf,
    min_finite,
    min_or_neg_inf,
    nan,
    neg_inf,
    nextafter,
)


# TODO: improve coverage and organization of these tests
def test_FPUtils() raises:
    assert_equal(FPUtils[.float32].mantissa_width(), 23)
    assert_equal(FPUtils[.float32].exponent_bias(), 127)

    comptime FPU64 = FPUtils[.float64]

    assert_equal(FPU64.mantissa_width(), 52)
    assert_equal(FPU64.exponent_bias(), 1023)

    assert_equal(FPU64.get_exponent(FPU64.set_exponent(1, 2)), 2)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(1, 3)), 3)
    assert_equal(FPU64.get_exponent(FPU64.set_exponent(-1, 4)), 4)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(-1, 5)), 5)
    assert_true(FPU64.get_sign(FPU64.set_sign(0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(0, False)))
    assert_true(FPU64.get_sign(FPU64.set_sign(-0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(-0, False)))
    assert_false(FPU64.get_sign(1))
    assert_true(FPU64.get_sign(-1))
    assert_false(FPU64.get_sign(FPU64.pack(False, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(False, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(False, 6, 12)), 12)
    assert_true(FPU64.get_sign(FPU64.pack(True, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(True, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(True, 6, 12)), 12)


def test_get_accum_type() raises:
    assert_equal(get_accum_type[.float32](), DType.float32)
    assert_equal(get_accum_type[.float64](), DType.float64)
    assert_equal(get_accum_type[.bfloat16](), DType.float32)
    assert_equal(get_accum_type[.int8](), DType.int8)
    assert_equal(get_accum_type[.int16](), DType.int16)
    assert_equal(get_accum_type[.int32](), DType.int32)
    assert_equal(get_accum_type[.int64](), DType.int64)
    assert_equal(get_accum_type[.uint8](), DType.uint8)
    assert_equal(get_accum_type[.uint16](), DType.uint16)
    assert_equal(get_accum_type[.uint32](), DType.uint32)
    assert_equal(get_accum_type[.uint64](), DType.uint64)


def test_isfinite() raises:
    assert_true(isfinite(Float32(33)))

    assert_false(isfinite(inf[.bfloat16]()))
    assert_false(isfinite(neg_inf[.bfloat16]()))
    assert_false(isfinite(nan[.bfloat16]()))

    assert_false(isfinite(inf[.float16]()))
    assert_false(isfinite(inf[.float32]()))
    assert_false(isfinite(inf[.float64]()))
    assert_false(isfinite(neg_inf[.float16]()))
    assert_false(isfinite(neg_inf[.float32]()))
    assert_false(isfinite(neg_inf[.float64]()))
    assert_false(isfinite(nan[.float16]()))
    assert_false(isfinite(nan[.float32]()))
    assert_false(isfinite(nan[.float64]()))


def test_isinf() raises:
    assert_false(isinf(Float32(33)))

    assert_true(isinf(inf[.bfloat16]()))
    assert_true(isinf(neg_inf[.bfloat16]()))
    assert_false(isinf(nan[.bfloat16]()))

    assert_true(isinf(inf[.float16]()))
    assert_true(isinf(inf[.float32]()))
    assert_true(isinf(inf[.float64]()))
    assert_true(isinf(neg_inf[.float16]()))
    assert_true(isinf(neg_inf[.float32]()))
    assert_true(isinf(neg_inf[.float64]()))
    assert_false(isinf(nan[.float16]()))
    assert_false(isinf(nan[.float32]()))
    assert_false(isinf(nan[.float64]()))


def test_isnan() raises:
    assert_false(isnan(Float32(33)))

    assert_false(isnan(inf[.bfloat16]()))
    assert_false(isnan(neg_inf[.bfloat16]()))
    assert_true(isnan(nan[.bfloat16]()))

    assert_false(isnan(inf[.float16]()))
    assert_false(isnan(inf[.float32]()))
    assert_false(isnan(inf[.float64]()))
    assert_false(isnan(neg_inf[.float16]()))
    assert_false(isnan(neg_inf[.float32]()))
    assert_false(isnan(neg_inf[.float64]()))
    assert_true(isnan(nan[.float16]()))
    assert_true(isnan(nan[.float32]()))
    assert_true(isnan(nan[.float64]()))


# ===-------------------------------------------------------------------=== #
# Float8 / Float4 isnan / isinf / isfinite coverage
#
# These are regression tests for some floating point dtypes, where compiling `isnan` /
# `isinf` / `isfinite` on `float8_e3m4` and `float4_e2m1fn` failed because
# `llvm.is.fpclass` has no overload for those types. Constructing values
# via `from_bits` lets us cover NaN / Inf / finite encodings without
# relying on `nan[T]()` (which is undefined for several of these dtypes).
# ===-------------------------------------------------------------------=== #


def test_isnan_float8_e4m3fn() raises:
    # E4M3FN: the only NaN encodings are 0x7F (+NaN) and 0xFF (-NaN); no Inf.
    assert_true(isnan(Float8_e4m3fn(from_bits=UInt8(0x7F))))
    assert_true(isnan(Float8_e4m3fn(from_bits=UInt8(0xFF))))
    assert_false(isnan(Float8_e4m3fn(from_bits=UInt8(0x00))))
    assert_false(isnan(Float8_e4m3fn(from_bits=UInt8(0x7E))))
    assert_false(isnan(Float8_e4m3fn(from_bits=UInt8(0x80))))


def test_isnan_float8_e5m2() raises:
    # E5M2: NaN at exp=11111 + mantissa nonzero, i.e. 0x7D / 0x7E / 0x7F
    # and their negative counterparts; 0x7C / 0xFC are +/-Inf.
    assert_true(isnan(Float8_e5m2(from_bits=UInt8(0x7D))))
    assert_true(isnan(Float8_e5m2(from_bits=UInt8(0x7E))))
    assert_true(isnan(Float8_e5m2(from_bits=UInt8(0x7F))))
    assert_true(isnan(Float8_e5m2(from_bits=UInt8(0xFD))))
    assert_true(isnan(Float8_e5m2(from_bits=UInt8(0xFF))))
    assert_false(isnan(Float8_e5m2(from_bits=UInt8(0x7C))))
    assert_false(isnan(Float8_e5m2(from_bits=UInt8(0xFC))))
    assert_false(isnan(Float8_e5m2(from_bits=UInt8(0x00))))
    assert_false(isnan(Float8_e5m2(from_bits=UInt8(0x3C))))


def test_isnan_float8_e8m0fnu() raises:
    # E8M0FNU: only NaN encoding is 0xFF.
    assert_true(isnan(Float8_e8m0fnu(from_bits=UInt8(0xFF))))
    assert_false(isnan(Float8_e8m0fnu(from_bits=UInt8(0x00))))
    assert_false(isnan(Float8_e8m0fnu(from_bits=UInt8(0x7F))))
    assert_false(isnan(Float8_e8m0fnu(from_bits=UInt8(0xFE))))


def test_isnan_float8_e3m4() raises:
    # E3M4 follows IEEE-style semantics: NaN is exp=111 with mantissa nonzero,
    # Inf is exp=111 with mantissa zero. Multiple bit patterns are NaN.
    # 0x70 = 0_111_0000 = +Inf, 0xF0 = -Inf — must NOT be NaN.
    assert_false(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x70))))
    assert_false(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0xF0))))
    # Several distinct NaN bit patterns must all be detected.
    assert_true(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x71))))
    assert_true(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x78))))
    assert_true(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x7F))))
    assert_true(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0xF1))))
    assert_true(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0xF8))))
    assert_true(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0xFF))))
    # Zero and finite values are not NaN.
    assert_false(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x00))))
    assert_false(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x80))))
    assert_false(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0x6F))))
    assert_false(isnan(Scalar[.float8_e3m4](from_bits=UInt8(0xEF))))


def test_isnan_float4_e2m1fn() raises:
    # E2M1FN has no NaN encoding; cover all 16 representable magnitudes
    # (+/-{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}) via Float32 casts.
    assert_false(isnan(Float4_e2m1fn(Float32(0.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(0.5))))
    assert_false(isnan(Float4_e2m1fn(Float32(1.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(1.5))))
    assert_false(isnan(Float4_e2m1fn(Float32(2.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(3.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(4.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(6.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(-0.5))))
    assert_false(isnan(Float4_e2m1fn(Float32(-1.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(-1.5))))
    assert_false(isnan(Float4_e2m1fn(Float32(-2.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(-3.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(-4.0))))
    assert_false(isnan(Float4_e2m1fn(Float32(-6.0))))


def test_isinf_float8_e5m2() raises:
    # E5M2 has both +Inf (0x7C) and -Inf (0xFC).
    assert_true(isinf(Float8_e5m2(from_bits=UInt8(0x7C))))
    assert_true(isinf(Float8_e5m2(from_bits=UInt8(0xFC))))
    # NaN bit patterns are not Inf.
    assert_false(isinf(Float8_e5m2(from_bits=UInt8(0x7D))))
    assert_false(isinf(Float8_e5m2(from_bits=UInt8(0xFF))))
    # Finite values are not Inf.
    assert_false(isinf(Float8_e5m2(from_bits=UInt8(0x00))))
    assert_false(isinf(Float8_e5m2(from_bits=UInt8(0x3C))))


def test_isinf_float8_e3m4() raises:
    # E3M4 has +Inf at 0x70 and -Inf at 0xF0.
    assert_true(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0x70))))
    assert_true(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0xF0))))
    # NaN bit patterns are not Inf.
    assert_false(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0x71))))
    assert_false(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0x7F))))
    assert_false(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0xFF))))
    # Finite values are not Inf.
    assert_false(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0x00))))
    assert_false(isinf(Scalar[.float8_e3m4](from_bits=UInt8(0x6F))))


def test_isinf_float4_e2m1fn() raises:
    # E2M1FN has no Inf encoding; spot-check several representable values.
    assert_false(isinf(Float4_e2m1fn(Float32(0.0))))
    assert_false(isinf(Float4_e2m1fn(Float32(1.0))))
    assert_false(isinf(Float4_e2m1fn(Float32(6.0))))
    assert_false(isinf(Float4_e2m1fn(Float32(-1.0))))
    assert_false(isinf(Float4_e2m1fn(Float32(-6.0))))


def test_isinf_float8_e4m3fn() raises:
    # E4M3FN has no Inf encoding (0x7F / 0xFF are NaN, not Inf).
    assert_false(isinf(Float8_e4m3fn(from_bits=UInt8(0x00))))
    assert_false(isinf(Float8_e4m3fn(from_bits=UInt8(0x7E))))
    assert_false(isinf(Float8_e4m3fn(from_bits=UInt8(0x7F))))
    assert_false(isinf(Float8_e4m3fn(from_bits=UInt8(0x80))))
    assert_false(isinf(Float8_e4m3fn(from_bits=UInt8(0xFF))))


def test_isinf_float8_e8m0fnu() raises:
    # E8M0FNU has no Inf encoding (0xFF is NaN, all others are finite).
    assert_false(isinf(Float8_e8m0fnu(from_bits=UInt8(0x00))))
    assert_false(isinf(Float8_e8m0fnu(from_bits=UInt8(0x7F))))
    assert_false(isinf(Float8_e8m0fnu(from_bits=UInt8(0xFE))))
    assert_false(isinf(Float8_e8m0fnu(from_bits=UInt8(0xFF))))


def test_isfinite_float8_e3m4() raises:
    # Zero and other finite encodings.
    assert_true(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0x00))))
    assert_true(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0x80))))
    assert_true(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0x6F))))
    assert_true(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0xEF))))
    # Inf encodings.
    assert_false(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0x70))))
    assert_false(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0xF0))))
    # NaN encodings.
    assert_false(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0x71))))
    assert_false(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0x7F))))
    assert_false(isfinite(Scalar[.float8_e3m4](from_bits=UInt8(0xFF))))


def test_isfinite_float4_e2m1fn() raises:
    # E2M1FN: every representable value is finite.
    assert_true(isfinite(Float4_e2m1fn(Float32(0.0))))
    assert_true(isfinite(Float4_e2m1fn(Float32(1.0))))
    assert_true(isfinite(Float4_e2m1fn(Float32(6.0))))
    assert_true(isfinite(Float4_e2m1fn(Float32(-1.0))))
    assert_true(isfinite(Float4_e2m1fn(Float32(-6.0))))


def overflow_int[dtype: DType]() -> Bool:
    comptime assert (
        dtype.is_integral()
    ), "comparison only valid on integral types"
    return max_finite[dtype]() + 1 < max_finite[dtype]()


def overflow_fp[dtype: DType]() -> Bool:
    comptime assert (
        dtype.is_floating_point()
    ), "comparison only valid on floating point types"
    return max_finite[dtype]() + 1 == max_finite[dtype]()


def test_max_finite() raises:
    assert_almost_equal(max_finite[.float32](), 3.4028235e38)
    assert_almost_equal(max_finite[.float64](), 1.7976931348623157e308)

    assert_true(max_finite[.bool]())

    assert_true(overflow_int[.int8]())
    assert_true(overflow_int[.uint8]())
    assert_true(overflow_int[.int16]())
    assert_true(overflow_int[.uint16]())
    assert_true(overflow_int[.int32]())
    assert_true(overflow_int[.uint32]())
    assert_true(overflow_int[.int64]())
    assert_true(overflow_int[.uint64]())
    assert_true(overflow_int[.int]())
    assert_true(overflow_int[.uint]())

    assert_true(overflow_fp[.float32]())
    assert_true(overflow_fp[.float64]())

    assert_equal(max_finite[.int8](), 127)
    assert_equal(max_finite[.uint8](), 255)
    assert_equal(max_finite[.int16](), 32767)
    assert_equal(max_finite[.uint16](), 65535)
    assert_equal(max_finite[.int32](), 2147483647)
    assert_equal(max_finite[.uint32](), 4294967295)
    assert_equal(max_finite[.int64](), 9223372036854775807)
    assert_equal(max_finite[.uint64](), 18446744073709551615)
    # FIXME(#5214): uncomment once it is closed
    # assert_equal(
    #     max_finite[DType.int128](), 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF
    # )
    # assert_equal(
    #     max_finite[DType.uint128](), 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF
    # )
    # assert_equal(
    #     max_finite[DType.int256](),
    #     0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
    # )
    # assert_equal(
    #     max_finite[DType.uint256](),
    #     0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
    # )

    comptime if is_64bit():
        assert_equal(max_finite[.int](), 9223372036854775807)
        assert_equal(max_finite[.uint](), 18446744073709551615)
    else:
        assert_equal(max_finite[.int](), 2147483647)
        assert_equal(max_finite[.uint](), 4294967295)


def underflow_int[dtype: DType]() -> Bool:
    comptime assert (
        dtype.is_integral()
    ), "comparison only valid on integral types"
    return min_finite[dtype]() - 1 > min_finite[dtype]()


def underflow_fp[dtype: DType]() -> Bool:
    comptime assert (
        dtype.is_floating_point()
    ), "comparison only valid on floating point types"
    return min_finite[dtype]() - 1 == min_finite[dtype]()


def test_min_finite() raises:
    assert_almost_equal(min_finite[.float32](), -3.4028235e38)
    assert_almost_equal(min_finite[.float64](), -1.7976931348623157e308)

    assert_false(min_finite[.bool]())

    assert_true(underflow_int[.int8]())
    assert_true(underflow_int[.uint8]())
    assert_true(underflow_int[.int16]())
    assert_true(underflow_int[.uint16]())
    assert_true(underflow_int[.int32]())
    assert_true(underflow_int[.uint32]())
    assert_true(underflow_int[.int64]())
    assert_true(underflow_int[.uint64]())
    assert_true(underflow_int[.int]())
    assert_true(underflow_int[.uint]())

    assert_true(underflow_fp[.float32]())
    assert_true(underflow_fp[.float64]())

    assert_equal(min_finite[.int8](), -128)
    assert_equal(min_finite[.int16](), -32768)
    assert_equal(min_finite[.int32](), -2147483648)
    assert_equal(min_finite[.int64](), -9223372036854775808)
    # FIXME(#5214): uncomment once it is closed
    # assert_equal(
    #     min_finite[DType.int128](), -0x8000_0000_0000_0000_0000_0000_0000_0000
    # )
    # assert_equal(
    #     min_finite[DType.int256](),
    #     -0x8000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000,
    # )

    comptime if is_64bit():
        assert_equal(min_finite[.int](), -9223372036854775808)
        assert_equal(min_finite[.uint](), 0)
    else:
        assert_equal(min_finite[.int](), -2147483648)
        assert_equal(min_finite[.uint](), 0)


def test_max_or_inf() raises:
    assert_almost_equal(max_or_inf[.float32](), inf[.float32]())
    assert_almost_equal(max_or_inf[.float64](), inf[.float64]())
    assert_true(max_or_inf[.bool]())


def test_min_or_neg_inf() raises:
    assert_almost_equal(min_or_neg_inf[.float32](), neg_inf[.float32]())
    assert_almost_equal(min_or_neg_inf[.float64](), neg_inf[.float64]())
    assert_false(min_or_neg_inf[.bool]())


def test_neg_inf() raises:
    assert_false(isfinite(neg_inf[.float32]()))
    assert_false(isfinite(neg_inf[.float64]()))
    assert_true(isinf(neg_inf[.float32]()))
    assert_true(isinf(neg_inf[.float64]()))
    assert_false(isnan(neg_inf[.float32]()))
    assert_false(isnan(neg_inf[.float64]()))
    assert_equal(-inf[.float32](), neg_inf[.float32]())
    assert_equal(-inf[.float64](), neg_inf[.float64]())


def test_nextafter() raises:
    assert_true(isnan(nextafter(nan[.float32](), nan[.float32]())))
    assert_true(isinf(nextafter(inf[.float32](), inf[.float32]())))
    assert_true(isinf(nextafter(-inf[.float32](), -inf[.float32]())))
    assert_almost_equal(nextafter(Float64(0), Float64(0)), 0)
    assert_almost_equal(nextafter(Float64(0), Float64(1)), 5e-324)
    assert_almost_equal(nextafter(Float64(0), Float64(-1)), -5e-324)
    assert_almost_equal(nextafter(Float64(1), Float64(0)), 0.99999999999999988)
    assert_almost_equal(
        nextafter(Float64(-1), Float64(0)), -0.99999999999999988
    )
    assert_almost_equal(
        nextafter(SIMD[.float64, 2](0, 1), SIMD[.float64, 2](0, 1)),
        SIMD[.float64, 2](0, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[.float64, 2](0, 1), SIMD[.float64, 2](1, 1)),
        SIMD[.float64, 2](5e-324, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[.float64, 2](0, 1), SIMD[.float64, 2](-1, 1)),
        SIMD[.float64, 2](-5e-324, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[.float64, 2](1, 1), SIMD[.float64, 2](0, 0)),
        SIMD[.float64, 2](0.99999999999999988, 0.99999999999999988),
    )
    assert_almost_equal(
        nextafter(SIMD[.float64, 2](-1, -1), SIMD[.float64, 2](0, 0)),
        SIMD[.float64, 2](-0.99999999999999988, -0.99999999999999988),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
