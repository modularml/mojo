# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from collections import InlineArray, List
from math import (
    align_down,
    align_up,
    ceil,
    ceildiv,
    clamp,
    copysign,
    cos,
    exp2,
    factorial,
    floor,
    frexp,
    gcd,
    iota,
    isclose,
    isqrt,
    lcm,
    log,
    log2,
    sin,
    sqrt,
    trunc,
    ulp,
)
from sys.info import has_neon

from testing import assert_almost_equal, assert_equal, assert_false, assert_true

from utils import Span
from utils.numerics import inf, isinf, isnan, nan, neg_inf


fn test_sin() raises:
    assert_almost_equal(sin(Float32(1.0)), 0.841470956802)


fn test_cos() raises:
    assert_almost_equal(cos(Float32(1.0)), 0.540302276611)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        assert_equal(cos(BFloat16(2.0)), -0.416015625)


fn test_factorial() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(1), 1)
    assert_equal(factorial(15), 1307674368000)
    assert_equal(factorial(20), 2432902008176640000)


def test_copysign():
    var x = Int32(2)
    assert_equal(2, copysign(x, x))
    assert_equal(-2, copysign(x, -x))
    assert_equal(2, copysign(-x, x))
    assert_equal(-2, copysign(-x, -x))

    assert_equal(1, copysign(Float32(1), Float32(2)))
    assert_equal(-1, copysign(Float32(1), Float32(-2)))
    assert_equal(neg_inf[DType.float32](), copysign(inf[DType.float32](), -2.0))
    assert_equal(-nan[DType.float32](), copysign(nan[DType.float32](), -2.0))

    # Test some cases with 0 and signed zero
    assert_equal(1, copysign(Float32(1.0), Float32(0.0)))
    assert_equal(0, copysign(Float32(0.0), Float32(1.0)))
    assert_equal(0, copysign(Float32(-0.0), Float32(1.0)))
    assert_equal(-0, copysign(Float32(0.0), Float32(-1.0)))

    # TODO: Add some test cases for SIMD vector with width > 1


def test_isclose():
    assert_true(isclose(Int64(2), Int64(2), atol=0, rtol=0))
    assert_false(isclose(Int64(2), Int64(3), atol=0, rtol=0))

    assert_true(isclose(Float32(2), Float32(2)))
    assert_true(isclose(Float32(2), Float32(2), rtol=1e-9))
    assert_true(isclose(Float32(2), Float32(2.00001), rtol=1e-3))
    assert_true(
        isclose(nan[DType.float32](), nan[DType.float32](), equal_nan=True)
    )

    assert_true(
        isclose(
            SIMD[DType.float32, 4](1, 2, 3, nan[DType.float32]()),
            SIMD[DType.float32, 4](1, 2, 3, nan[DType.float32]()),
            equal_nan=True,
        ).reduce_and()
    )

    assert_false(
        isclose(
            SIMD[DType.float32, 4](1, 2, nan[DType.float32](), 3),
            SIMD[DType.float32, 4](1, 2, nan[DType.float32](), 4),
            equal_nan=True,
        ).reduce_and()
    )


def test_ceil():
    # We just test that the `ceil` function resolves correctly for a few common
    # types. Types should test their own `__ceil__` implementation explicitly.
    assert_equal(ceil(0), 0)
    assert_equal(ceil(Int(5)), 5)
    assert_equal(ceil(1.5), 2.0)
    assert_equal(ceil(Float32(1.4)), 2.0)
    assert_equal(ceil(Float64(-3.6)), -3.0)


def test_floor():
    # We just test that the `floor` function resolves correctly for a few common
    # types. Types should test their own `__floor__` implementation explicitly.
    assert_equal(floor(0), 0)
    assert_equal(floor(Int(5)), 5)
    assert_equal(floor(1.5), 1.0)
    assert_equal(floor(Float32(1.6)), 1.0)
    assert_equal(floor(Float64(-3.4)), -4.0)


def test_trunc():
    # We just test that the `trunc` function resolves correctly for a few common
    # types. Types should test their own `__trunc__` implementation explicitly.
    assert_equal(trunc(0), 0)
    assert_equal(trunc(Int(5)), 5)
    assert_equal(trunc(1.5), 1.0)
    assert_equal(trunc(Float32(1.6)), 1.0)
    assert_equal(trunc(Float64(-3.4)), -3.0)


def test_exp2():
    assert_equal(exp2(Float32(1)), 2.0)
    assert_almost_equal(exp2(Float32(0.2)), 1.148696)
    assert_equal(exp2(Float32(0)), 1.0)
    assert_equal(exp2(Float32(-1)), 0.5)
    assert_equal(exp2(Float32(2)), 4.0)


def test_iota():
    alias length = 103
    var offset = 2

    var vector = List[Int32]()
    vector.resize(length, 0)

    var buff = vector.data
    iota(buff, length, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    iota(vector, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    var vector2 = List[Int]()
    vector2.resize(length, 0)
    iota(vector2, offset)

    for i in range(length):
        assert_equal(vector2[i], offset + i)


alias F32x4 = SIMD[DType.float32, 4]
alias F64x4 = SIMD[DType.float64, 4]


def test_sqrt():
    var i = SIMD[DType.index, 4](0, 1, 2, 3)
    assert_equal(sqrt(i**2), i)
    assert_equal(sqrt(64), 8)
    assert_equal(sqrt(63), 7)

    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0)

    var s1_f32 = sqrt(f32x4)
    assert_equal(s1_f32[0], 0.0)
    assert_almost_equal(s1_f32[1], 0.70710)
    assert_equal(s1_f32[2], 1.0)
    assert_almost_equal(s1_f32[3], 1.22474)

    var s2_f32 = sqrt(0.5 * f32x4)
    assert_equal(s2_f32[0], 0.0)
    assert_equal(s2_f32[1], 0.5)
    assert_almost_equal(s2_f32[2], 0.70710)
    assert_almost_equal(s2_f32[3], 0.86602)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0)

    var s1_f64 = sqrt(f64x4)
    assert_equal(s1_f64[0], 0.0)
    assert_almost_equal(s1_f64[1], 0.70710)
    assert_equal(s1_f64[2], 1.0)
    assert_almost_equal(s1_f64[3], 1.22474)

    var s2_f64 = sqrt(0.5 * f64x4)
    assert_equal(s2_f64[0], 0.0)
    assert_equal(s2_f64[1], 0.5)
    assert_almost_equal(s2_f64[2], 0.70710)
    assert_almost_equal(s2_f64[3], 0.86602)


def test_isqrt():
    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f32 = isqrt(f32x4)
    assert_equal(s1_f32[0], 1.0)
    assert_almost_equal(s1_f32[1], 0.81649)
    assert_almost_equal(s1_f32[2], 0.70710)
    assert_almost_equal(s1_f32[3], 0.63245)

    var s2_f32 = isqrt(0.5 * f32x4)
    assert_almost_equal(s2_f32[0], 1.41421)
    assert_almost_equal(s2_f32[1], 1.15470)
    assert_equal(s2_f32[2], 1.0)
    assert_almost_equal(s2_f32[3], 0.89442)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f64 = isqrt(f64x4)
    assert_equal(s1_f64[0], 1.0)
    assert_almost_equal(s1_f64[1], 0.81649)
    assert_almost_equal(s1_f64[2], 0.70710)
    assert_almost_equal(s1_f64[3], 0.63245)

    var s2_f64 = isqrt(0.5 * f64x4)
    assert_almost_equal(s2_f64[0], 1.41421)
    assert_almost_equal(s2_f64[1], 1.15470)
    assert_equal(s2_f64[2], 1.0)
    assert_almost_equal(s2_f64[3], 0.89442)


def _test_frexp_impl[type: DType](*, atol: Float32, rtol: Float32):
    var res0 = frexp(Scalar[type](123.45))
    assert_almost_equal(
        res0[0].cast[DType.float32](), 0.964453, atol=atol, rtol=rtol
    )
    assert_almost_equal(
        res0[1].cast[DType.float32](), 7.0, atol=atol, rtol=rtol
    )

    var res1 = frexp(Scalar[type](0.1))
    assert_almost_equal(
        res1[0].cast[DType.float32](), 0.8, atol=atol, rtol=rtol
    )
    assert_almost_equal(
        res1[1].cast[DType.float32](), -3.0, atol=atol, rtol=rtol
    )

    var res2 = frexp(Scalar[type](-0.1))
    assert_almost_equal(
        res2[0].cast[DType.float32](), -0.8, atol=atol, rtol=rtol
    )
    assert_almost_equal(
        res2[1].cast[DType.float32](), -3.0, atol=atol, rtol=rtol
    )

    var res3 = frexp(SIMD[type, 4](0, 2, 4, 5))
    assert_almost_equal(
        res3[0].cast[DType.float32](),
        SIMD[DType.float32, 4](0.0, 0.5, 0.5, 0.625),
        atol=atol,
        rtol=rtol,
    )
    assert_almost_equal(
        res3[1].cast[DType.float32](),
        SIMD[DType.float32, 4](-0.0, 2.0, 3.0, 3.0),
        atol=atol,
        rtol=rtol,
    )


def _test_log_impl[type: DType](*, atol: Float32, rtol: Float32):
    var res0 = log(Scalar[type](123.45))
    assert_almost_equal(
        res0.cast[DType.float32](), 4.8158, atol=atol, rtol=rtol
    )

    var res1 = log(Scalar[type](0.1))
    assert_almost_equal(
        res1.cast[DType.float32](), -2.3025, atol=atol, rtol=rtol
    )

    var res2 = log(SIMD[type, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[DType.float32](),
        SIMD[DType.float32, 4](0.0, 0.693147, 1.38629, 1.6094),
        atol=atol,
        rtol=rtol,
    )

    var res3 = log(Scalar[type](2.7182818284590452353602874713526624977572))
    assert_almost_equal(res3.cast[DType.float32](), 1.0, atol=atol, rtol=rtol)

    var res4 = isinf(log(SIMD[type, 4](0, 1, 0, 0)))
    assert_equal(res4, SIMD[DType.bool, 4](True, False, True, True))


def _test_log2_impl[type: DType](*, atol: Float32, rtol: Float32):
    var res0 = log2(Scalar[type](123.45))
    assert_almost_equal(
        res0.cast[DType.float32](), 6.9477, atol=atol, rtol=rtol
    )

    var res1 = log2(Scalar[type](0.1))
    assert_almost_equal(
        res1.cast[DType.float32](), -3.3219, atol=atol, rtol=rtol
    )

    var res2 = log2(SIMD[type, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[DType.float32](),
        SIMD[DType.float32, 4](0.0, 1.0, 2.0, 2.3219),
        atol=atol,
        rtol=rtol,
    )


def test_frexp():
    _test_frexp_impl[DType.float32](atol=1e-4, rtol=1e-5)
    _test_frexp_impl[DType.float16](atol=1e-2, rtol=1e-5)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        _test_frexp_impl[DType.bfloat16](atol=1e-1, rtol=1e-5)


def test_log():
    _test_log_impl[DType.float32](atol=1e-4, rtol=1e-5)
    _test_log_impl[DType.float16](atol=1e-2, rtol=1e-5)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        _test_log_impl[DType.bfloat16](atol=1e-1, rtol=1e-5)


def test_log2():
    _test_log2_impl[DType.float32](atol=1e-4, rtol=1e-5)
    _test_log2_impl[DType.float16](atol=1e-2, rtol=1e-5)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        _test_log2_impl[DType.bfloat16](atol=1e-1, rtol=1e-5)


def test_gcd():
    var l = List(2, 4, 6, 8, 16)
    var il = InlineArray[Int, 5](4, 16, 2, 8, 6)
    assert_equal(gcd(Span[Int](il)), 2)
    assert_equal(gcd(2, 4, 6, 8, 16), 2)
    assert_equal(gcd(l), 2)
    assert_equal(gcd(88, 24), 8)
    assert_equal(gcd(0, 0), 0)
    assert_equal(gcd(1, 0), 1)
    assert_equal(gcd(-2, 4), 2)
    assert_equal(gcd(-2, -4), 2)
    assert_equal(gcd(-2, 0), 2)
    assert_equal(gcd(2, -4), 2)
    assert_equal(gcd(24826148, 45296490), 526)
    assert_equal(gcd(0, 9), 9)
    assert_equal(gcd(4, 4), 4)
    assert_equal(gcd(8), 8)
    assert_equal(gcd(), 0)
    assert_equal(gcd(List[Int]()), 0)
    assert_equal(gcd(List(16)), 16)


def test_lcm():
    assert_equal(lcm(-2, 4), 4)
    assert_equal(lcm(2345, 23452), 54994940)
    var l = List(4, 6, 7, 3)
    assert_equal(lcm(Span(l)), 84)
    assert_equal(lcm(l), 84)
    assert_equal(lcm(4, 6, 7, 3), 84)
    assert_equal(lcm(), 1)
    assert_equal(lcm(List(3)), 3)
    assert_equal(lcm(List[Int]()), 1)
    assert_equal(lcm(0, 4), 0)
    assert_equal(lcm(5, 33), 165)
    assert_equal(lcm(-34, -56, -32), 3808)
    var il = InlineArray[Int, 5](4, 16, 2, 8, 6)
    assert_equal(lcm(Span[Int](il)), 48)
    assert_equal(lcm(345, 623, 364, 84, 93), 346475220)
    assert_equal(lcm(0, 0), 0)


def test_ulp():
    assert_true(isnan(ulp(nan[DType.float32]())))
    assert_true(isinf(ulp(inf[DType.float32]())))
    assert_true(isinf(ulp(-inf[DType.float32]())))
    assert_almost_equal(ulp(Float64(0)), 5e-324)
    assert_equal(ulp(Float64.MAX_FINITE), 1.99584030953472e292)
    assert_equal(ulp(Float64(5)), 8.881784197001252e-16)
    assert_equal(ulp(Float64(-5)), 8.881784197001252e-16)


def test_ceildiv():
    # NOTE: these tests are here mostly to ensure the ceildiv method exists.
    # Types that opt in to CeilDivable, should test their own dunder methods for
    # correctness.
    assert_equal(ceildiv(53.6, 1.35), 40.0)

    # Test the IntLiteral overload.
    alias a: IntLiteral = ceildiv(1, 7)
    assert_equal(a, 1)
    alias b: IntLiteral = ceildiv(548, -7)
    assert_equal(b, -78)

    # Test the Int overload.
    assert_equal(ceildiv(Int(1), Int(7)), 1)
    assert_equal(ceildiv(Int(548), Int(-7)), -78)
    assert_equal(ceildiv(Int(-55), Int(8)), -6)
    assert_equal(ceildiv(Int(-55), Int(-8)), 7)

    # Test the UInt overload.
    assert_equal(ceildiv(UInt(1), UInt(7)), UInt(1))
    assert_equal(ceildiv(UInt(546), UInt(7)), UInt(78))


def test_align_down():
    assert_equal(align_down(1, 7), 0)
    assert_equal(align_down(548, -7), 553)
    assert_equal(align_down(-548, -7), -546)
    assert_equal(align_down(-548, 7), -553)

    # Test the UInt overload.
    assert_equal(align_down(UInt(1), UInt(7)), UInt(0))
    assert_equal(align_down(UInt(546), UInt(7)), UInt(546))


def test_align_up():
    assert_equal(align_up(1, 7), 7)
    assert_equal(align_up(548, -7), 546)
    assert_equal(align_up(-548, -7), -553)
    assert_equal(align_up(-548, 7), -546)

    # Test the UInt overload.
    assert_equal(align_up(UInt(1), UInt(7)), UInt(7))
    assert_equal(align_up(UInt(546), UInt(7)), UInt(546))


def test_clamp():
    assert_equal(clamp(Int(1), 0, 1), 1)
    assert_equal(clamp(Int(2), 0, 1), 1)
    assert_equal(clamp(Int(-2), 0, 1), 0)

    assert_equal(clamp(UInt(1), UInt(0), UInt(1)), UInt(1))
    assert_equal(clamp(UInt(2), UInt(0), UInt(1)), UInt(1))
    assert_equal(clamp(UInt(1), UInt(2), UInt(4)), UInt(2))

    assert_equal(
        clamp(SIMD[DType.float32, 4](0, 1, 3, 4), 0, 1),
        SIMD[DType.float32, 4](0, 1, 1, 1),
    )


def main():
    test_sin()
    test_cos()
    test_factorial()
    test_copysign()
    test_isclose()
    test_ceil()
    test_floor()
    test_trunc()
    test_exp2()
    test_iota()
    test_sqrt()
    test_isqrt()
    test_frexp()
    test_log()
    test_log2()
    test_gcd()
    test_lcm()
    test_ulp()
    test_ceildiv()
    test_align_down()
    test_align_up()
    test_clamp()
