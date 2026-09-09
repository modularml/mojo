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

from std.math import (
    acos,
    align_down,
    align_up,
    asin,
    atanh,
    cbrt,
    ceil,
    ceildiv,
    clamp,
    comb,
    copysign,
    cos,
    cosh,
    erfc,
    expm1,
    exp2,
    factorial,
    floor,
    fma,
    frexp,
    gcd,
    iota,
    isclose,
    lcm,
    log,
    log1p,
    log2,
    perm,
    pi,
    rsqrt,
    sin,
    sinh,
    sqrt,
    trunc,
    ulp,
)
from std.math.math import _call_libm
from std.sys import CompilationTarget

from std.testing import TestSuite
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_true,
)

from std.utils.numerics import inf, isinf, isnan, nan, neg_inf


def test_sin() raises:
    assert_almost_equal(sin(Float32(1.0)), 0.841470956802)

    comptime s_45 = sin(pi / 4)
    assert_almost_equal(s_45, 0.7071067811865475)

    comptime s_30 = sin(pi / 6)
    assert_almost_equal(s_30, 0.5)

    comptime s_60 = sin(pi / 3)
    assert_almost_equal(s_60, 0.8660254037844387)

    # Compare the compile time values against the runtime values to make sure
    # they align.
    assert_almost_equal(s_45, sin(pi / 4))
    assert_almost_equal(s_30, sin(pi / 6))
    assert_almost_equal(s_60, sin(pi / 3))


def test_cos() raises:
    assert_almost_equal(cos(Float32(1.0)), 0.540302276611)

    assert_equal(cos(BFloat16(2.0)), -0.416015625)

    comptime c_45 = cos(pi / 4)
    assert_almost_equal(c_45, 0.7071067811865476)

    comptime c_30 = cos(pi / 6)
    assert_almost_equal(c_30, 0.8660254037844386)

    comptime c_60 = cos(pi / 3)
    assert_almost_equal(c_60, 0.4999999999999999)

    # Compare the compile time values against the runtime values to make sure
    # they align.
    assert_almost_equal(c_45, cos(pi / 4))
    assert_almost_equal(c_30, cos(pi / 6))
    assert_almost_equal(c_60, cos(pi / 3))


def test_factorial() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(1), 1)
    assert_equal(factorial(15), 1307674368000)
    assert_equal(factorial(20), 2432902008176640000)


def test_comb() raises:
    assert_equal(comb(0, 0), 1)
    assert_equal(comb(5, 0), 1)
    assert_equal(comb(5, 5), 1)
    assert_equal(comb(5, 1), 5)
    assert_equal(comb(5, 2), 10)
    assert_equal(comb(10, 3), 120)
    assert_equal(comb(3, 5), 0)  # k > n returns 0
    # Symmetry: C(n, k) == C(n, n-k)
    assert_equal(comb(10, 4), comb(10, 6))


def test_perm() raises:
    assert_equal(perm(5, 0), 1)
    assert_equal(perm(5, 1), 5)
    assert_equal(perm(5, 2), 20)
    assert_equal(perm(5, 5), 120)
    # perm(n) with default k=-1 delegates to factorial(n)
    assert_equal(perm(5), factorial(5))
    assert_equal(perm(0), 1)
    assert_equal(perm(10, 3), 720)


def test_copysign() raises:
    var x = Int32(2)
    assert_equal(x, copysign(x, x))
    assert_equal(-x, copysign(x, -x))
    assert_equal(x, copysign(-x, x))
    assert_equal(-x, copysign(-x, -x))

    assert_equal(Float32(1.0), copysign(Float32(1.0), Float32(2.0)))
    assert_equal(Float32(-1.0), copysign(Float32(1.0), Float32(-2.0)))
    assert_equal(neg_inf[.float32](), copysign(inf[.float32](), -2.0))
    assert_equal(
        String(-nan[.float32]()),
        String(copysign(nan[.float32](), -2.0)),
    )

    # Test some cases with 0 and signed zero
    assert_equal(Float32(1.0), copysign(Float32(1.0), Float32(0.0)))
    assert_equal(Float32(0.0), copysign(Float32(0.0), Float32(1.0)))
    assert_equal(Float32(0.0), copysign(Float32(-0.0), Float32(1.0)))
    assert_equal(Float32(-0.0), copysign(Float32(0.0), Float32(-1.0)))

    # TODO: Add some test cases for SIMD vector with width > 1


def _test_isclose_numerics[*, symm: Bool]() raises:
    comptime dtype = DType.float64
    comptime T = SIMD[dtype, 2]

    comptime atol = 1e-8
    comptime rtol = 1e-5

    comptime inf_ = inf[dtype]()
    comptime nan_ = nan[dtype]()
    comptime v = T(0.1, 0.2)

    def edge_val[symm: Bool](a: T, atol: T, rtol: T) -> T:
        """Creates a value at the tolerance boundary that should be considered close to `a`.
        """
        assert all(a.ge(0))

        comptime if symm:
            # |a - b| ≤ max(atol, rtol * max(|a|, |b|))
            return a - max(atol, rtol * a)
        else:
            # |a - b| ≤ atol + rtol * |b|
            return a + atol + rtol * a

    var all_close: List[Tuple[T, T]] = [
        (T(1, 0), T(1, 0)),
        (T(atol), T(0)),
        (T(inf_, -inf_), T(inf_, -inf_)),
        (T(1), edge_val[symm](1, atol, 0)),
        (edge_val[symm](v, atol, 0), v),
    ]

    comptime if not symm:
        all_close += [
            (edge_val[symm](v, 0, rtol), v),
            (edge_val[symm](v, atol, rtol), v),
        ]

    for item in all_close:
        var a, b = item
        var res = isclose[symmetrical=symm](a, b, atol=atol, rtol=rtol)
        assert_true(all(res))

    var none_close: List[Tuple[T, T]] = [
        (T(inf_, 0), T(1, inf_)),
        (T(inf_, -inf_), T(1, 0)),
        (T(inf_, inf_), T(1, -inf_)),
        (T(inf_, inf_), T(1, 0)),
        (T(nan_, 0), T(nan_, -inf_)),
    ]

    comptime if symm:
        none_close += [(v, v + atol + rtol)]
    else:
        none_close += [
            (T(0), edge_val[symm](0, 2.0 * atol, 0)),
            (T(1), edge_val[symm](1, 2.0 * atol, rtol)),
            (T(1), edge_val[symm](1, atol, 2.0 * rtol)),
            (v, edge_val[symm](v, atol, 2.0 * rtol)),
            (v, edge_val[symm](v, 1.1 * atol, 1.1 * rtol)),
        ]

    for item in none_close:
        var a, b = item
        var res = isclose[symmetrical=symm](a, b, atol=atol, rtol=rtol)
        assert_false(any(res))


def test_isclose() raises:
    # floating-point
    comptime dtype = DType.float32
    comptime S = Scalar[dtype]
    comptime T = SIMD[dtype, 4]
    comptime nan_ = nan[dtype]()

    assert_true(isclose(S(2), S(2)))
    assert_true(isclose(S(2), S(2), rtol=1e-9))
    assert_true(isclose(S(2), S(2.00001), rtol=1e-3))
    assert_true(isclose(nan_, nan_, equal_nan=True))

    assert_true(
        all(isclose(T(1, 2, 3, nan_), T(1, 2, 3, nan_), equal_nan=True))
    )

    assert_false(
        all(isclose(T(1, 2, nan_, 3), T(1, 2, nan_, 4), equal_nan=True))
    )

    _test_isclose_numerics[symm=False]()
    _test_isclose_numerics[symm=True]()


def test_ceil() raises:
    # We just test that the `ceil` function resolves correctly for a few common
    # types. Types should test their own `__ceil__` implementation explicitly.
    assert_equal(ceil(0), 0)
    assert_equal(ceil(Int(5)), 5)
    assert_equal(ceil(1.5), 2.0)
    assert_equal(ceil(Float32(1.4)), 2.0)
    assert_equal(ceil(Float64(-3.6)), -3.0)


def test_floor() raises:
    # We just test that the `floor` function resolves correctly for a few common
    # types. Types should test their own `__floor__` implementation explicitly.
    assert_equal(floor(0), 0)
    assert_equal(floor(Int(5)), 5)
    assert_equal(floor(1.5), 1.0)
    assert_equal(floor(Float32(1.6)), 1.0)
    assert_equal(floor(Float64(-3.4)), -4.0)


def test_trunc() raises:
    # We just test that the `trunc` function resolves correctly for a few common
    # types. Types should test their own `__trunc__` implementation explicitly.
    assert_equal(trunc(0), 0)
    assert_equal(trunc(Int(5)), 5)
    assert_equal(trunc(1.5), 1.0)
    assert_equal(trunc(Float32(1.6)), 1.0)
    assert_equal(trunc(Float64(-3.4)), -3.0)


def test_exp2() raises:
    assert_equal(exp2(Float32(1)), 2.0)
    assert_almost_equal(exp2(Float32(0.2)), 1.148696)
    assert_equal(exp2(Float32(0)), 1.0)
    assert_equal(exp2(Float32(-1)), 0.5)
    assert_equal(exp2(Float32(2)), 4.0)
    assert_almost_equal(exp2(Float32(-125)), 2.3509887e-38)
    assert_almost_equal(exp2(Float32(125)), 4.2535296e37)

    assert_equal(exp2(Float64(1)), 2.0)
    assert_almost_equal(exp2(Float64(0.2)), 1.148696)
    assert_equal(exp2(Float64(0)), 1.0)
    assert_equal(exp2(Float64(-1)), 0.5)
    assert_equal(exp2(Float64(2)), 4.0)
    assert_almost_equal(exp2(Float64(-127)), 5.877471754111438e-39)
    assert_almost_equal(exp2(Float64(127)), 1.7014118346046923e38)
    assert_almost_equal(exp2(Float64(-1023)), 1.1125369292536007e-308)
    assert_almost_equal(exp2(Float64(1023)), 8.98846567431158e307)


def test_iota() raises:
    comptime length = 103
    var offset = 2

    var vector = List[Int32](unsafe_uninit_length=length)

    var buff = vector.unsafe_ptr()
    iota(buff, length, offset)
    for i in range(length):
        assert_equal(vector[i], Int32(offset + i))

    iota(vector, offset)
    for i in range(length):
        assert_equal(vector[i], Int32(offset + i))

    var vector2 = List[Int](unsafe_uninit_length=length)

    iota(vector2, offset)
    for i in range(length):
        assert_equal(vector2[i], offset + i)


comptime F32x4 = SIMD[.float32, 4]
comptime F64x4 = SIMD[.float64, 4]


def test_sqrt() raises:
    assert_equal(sqrt(-1), 0)
    assert_equal(sqrt(0), 0)
    assert_equal(sqrt(1), 1)
    assert_equal(sqrt(63), 7)
    assert_equal(sqrt(64), 8)
    assert_equal(sqrt(2**34 - 1), 2**17 - 1)
    assert_equal(sqrt(2**34), 2**17)
    assert_equal(sqrt(10**16), 10**8)
    assert_equal(sqrt(Int.MAX), 3037000499)

    var i = SIMD[.int, 4](0, 1, 2, 3)
    assert_equal(sqrt(i**2), i)

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


def test_rsqrt() raises:
    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f32 = rsqrt(f32x4)
    assert_equal(s1_f32[0], 1.0)
    assert_almost_equal(s1_f32[1], 0.81649)
    assert_almost_equal(s1_f32[2], 0.70710)
    assert_almost_equal(s1_f32[3], 0.63245)

    var s2_f32 = rsqrt(0.5 * f32x4)
    assert_almost_equal(s2_f32[0], 1.41421)
    assert_almost_equal(s2_f32[1], 1.15470)
    assert_equal(s2_f32[2], 1.0)
    assert_almost_equal(s2_f32[3], 0.89442)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f64 = rsqrt(f64x4)
    assert_equal(s1_f64[0], 1.0)
    assert_almost_equal(s1_f64[1], 0.81649)
    assert_almost_equal(s1_f64[2], 0.70710)
    assert_almost_equal(s1_f64[3], 0.63245)

    var s2_f64 = rsqrt(0.5 * f64x4)
    assert_almost_equal(s2_f64[0], 1.41421)
    assert_almost_equal(s2_f64[1], 1.15470)
    assert_equal(s2_f64[2], 1.0)
    assert_almost_equal(s2_f64[3], 0.89442)


def _test_frexp_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = frexp(Scalar[dtype](123.45))
    assert_almost_equal(
        res0[0].cast[.float32](), 0.964453, atol=atol, rtol=rtol
    )
    assert_almost_equal(res0[1].cast[.float32](), 7.0, atol=atol, rtol=rtol)

    var res1 = frexp(Scalar[dtype](0.1))
    assert_almost_equal(res1[0].cast[.float32](), 0.8, atol=atol, rtol=rtol)
    assert_almost_equal(res1[1].cast[.float32](), -3.0, atol=atol, rtol=rtol)

    var res2 = frexp(Scalar[dtype](-0.1))
    assert_almost_equal(res2[0].cast[.float32](), -0.8, atol=atol, rtol=rtol)
    assert_almost_equal(res2[1].cast[.float32](), -3.0, atol=atol, rtol=rtol)

    var res3 = frexp(SIMD[dtype, 4](0, 2, 4, 5))
    assert_almost_equal(
        res3[0].cast[.float32](),
        SIMD[.float32, 4](0.0, 0.5, 0.5, 0.625),
        atol=atol,
        rtol=rtol,
    )
    assert_almost_equal(
        res3[1].cast[.float32](),
        SIMD[.float32, 4](-0.0, 2.0, 3.0, 3.0),
        atol=atol,
        rtol=rtol,
    )


def _test_log_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = log(Scalar[dtype](123.45))
    assert_almost_equal(res0.cast[.float32](), 4.8158, atol=atol, rtol=rtol)

    var res1 = log(Scalar[dtype](0.1))
    assert_almost_equal(res1.cast[.float32](), -2.3025, atol=atol, rtol=rtol)

    var res2 = log(SIMD[dtype, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[.float32](),
        SIMD[.float32, 4](0.0, 0.693147, 1.38629, 1.6094),
        atol=atol,
        rtol=rtol,
    )

    var res3 = log(Scalar[dtype](2.7182818284590452353602874713526624977572))
    assert_almost_equal(res3.cast[.float32](), 1.0, atol=atol, rtol=rtol)

    var res4 = isinf(log(SIMD[dtype, 4](0, 1, 0, 0)))
    assert_equal(res4, SIMD[.bool, 4](True, False, True, True))


def _test_log2_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = log2(Scalar[dtype](123.45))
    assert_almost_equal(res0.cast[.float32](), 6.9477, atol=atol, rtol=rtol)

    var res1 = log2(Scalar[dtype](0.1))
    assert_almost_equal(res1.cast[.float32](), -3.3219, atol=atol, rtol=rtol)

    var res2 = log2(SIMD[dtype, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[.float32](),
        SIMD[.float32, 4](0.0, 1.0, 2.0, 2.3219),
        atol=atol,
        rtol=rtol,
    )


def _test_log1p_impl[
    dtype: DType
](*, atol: Float64, rtol: Float64) raises where dtype.is_floating_point():
    var res0 = log1p(Scalar[dtype](123.45))
    assert_almost_equal(res0.cast[.float32](), 4.8239, atol=atol, rtol=rtol)

    var res1 = log1p(Scalar[dtype](0.1))
    assert_almost_equal(res1.cast[.float32](), 0.0953102, atol=atol, rtol=rtol)

    var res2 = log1p(SIMD[dtype, 4](1, 2, 4, 5))
    assert_almost_equal(
        res2.cast[.float32](),
        SIMD[.float32, 4](0.693147, 1.09861, 1.60944, 1.79176),
        atol=atol,
        rtol=rtol,
    )

    var res3 = log1p(SIMD[dtype, 4](0.00001, 0.000002, 0.000004, 0.00005))
    assert_almost_equal(
        res3.cast[.float32](),
        SIMD[.float32, 4](9.99995e-6, 2.0e-6, 3.99999e-6, 0.0000499988),
        atol=atol,
        rtol=rtol,
    )

    var res4 = log1p(SIMD[dtype, 4](0.707107, 0.807107, 0.9, 1))
    assert_almost_equal(
        res4.cast[.float32](),
        SIMD[.float32, 4](0.5348, 0.591727, 0.641854, 0.693147),
        atol=atol,
        rtol=rtol,
    )


def test_frexp() raises:
    _test_frexp_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_frexp_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_frexp_impl[.bfloat16](atol=1e-1, rtol=1e-5)


def test_log() raises:
    _test_log_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_log_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_log_impl[.bfloat16](atol=1e-1, rtol=1e-5)


def test_log2() raises:
    _test_log2_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_log2_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_log2_impl[.bfloat16](atol=1e-1, rtol=1e-5)


def test_log1p() raises:
    _test_log1p_impl[.float64](atol=1e-4, rtol=1e-5)
    _test_log1p_impl[.float32](atol=1e-4, rtol=1e-5)
    _test_log1p_impl[.float16](atol=1e-2, rtol=1e-5)

    _test_log1p_impl[.bfloat16](atol=1e-1, rtol=1e-5)


def test_log1p_accuracy() raises:
    # Compare log1p against libm across several regimes.

    # Small values near zero (critical regime where log1p matters).
    var small_vals: Array[Float64, 14] = [
        1e-15,
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
        1e-2,
        -1e-15,
        -1e-12,
        -1e-10,
        -1e-8,
        -1e-6,
        -1e-4,
        -1e-2,
    ]
    for val in small_vals:
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f64 small mismatch for the value = ", val),
        )

    # Float32 small values.
    var small_vals_f32: Array[Float32, 6] = [
        1e-7,
        1e-5,
        1e-3,
        -1e-7,
        -1e-5,
        -1e-3,
    ]
    for val in small_vals_f32:
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f32 small mismatch for the value = ", val),
        )

    # Moderate values.
    comptime n = 1_000.0
    for i in range(Float64(0.0), n, Float64(1.0)):
        var val = Float64(i) / (n / 10.5) - 0.5
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f64 moderate mismatch for the value = ", val),
        )

    # Values near the singularity at x = -1.
    var near_neg1: Array[Float64, 5] = [
        -0.9,
        -0.99,
        -0.999,
        -0.9999,
        -0.99999,
    ]
    for val in near_neg1:
        assert_almost_equal(
            log1p(val),
            _call_libm["log1p"](val),
            msg=String("f64 near-singularity mismatch for the value = ", val),
        )

    # Edge cases.
    assert_true(isnan(log1p(nan[.float64]())))
    assert_equal(log1p(Float64(0)), Float64(0))
    assert_true(isinf(log1p(Float64(-1))))
    assert_true(isnan(log1p(Float64(-2))))


def test_gcd() raises:
    var l = [2, 4, 6, 8, 16]
    var il: Array[Int, 5] = [4, 16, 2, 8, 6]
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


def test_lcm() raises:
    assert_equal(lcm(-2, 4), 4)
    assert_equal(lcm(2345, 23452), 54994940)
    var l = [4, 6, 7, 3]
    assert_equal(lcm(Span(l)), 84)
    assert_equal(lcm(l), 84)
    assert_equal(lcm(4, 6, 7, 3), 84)
    assert_equal(lcm(), 1)
    assert_equal(lcm(List[Int]()), 1)
    assert_equal(lcm(0, 4), 0)
    assert_equal(lcm(5, 33), 165)
    assert_equal(lcm(-34, -56, -32), 3808)
    var il: Array[Int, 5] = [4, 16, 2, 8, 6]
    assert_equal(lcm(Span[Int](il)), 48)
    assert_equal(lcm(345, 623, 364, 84, 93), 346475220)
    assert_equal(lcm(0, 0), 0)


def test_ulp() raises:
    assert_true(isnan(ulp(nan[.float32]())))
    assert_true(isinf(ulp(inf[.float32]())))
    assert_true(isinf(ulp(-inf[.float32]())))
    assert_almost_equal(ulp(Float64(0)), 5e-324)
    assert_equal(ulp(Float64.MAX_FINITE), 1.99584030953472e292)
    assert_equal(ulp(Float64(5)), 8.881784197001252e-16)
    assert_equal(ulp(Float64(-5)), 8.881784197001252e-16)


def test_ceildiv() raises:
    # NOTE: these tests are here mostly to ensure the ceildiv method exists.
    # Types that opt in to CeilDivable, should test their own dunder methods for
    # correctness.
    assert_equal(ceildiv(53.6, 1.35), 40.0)

    # Test the IntLiteral overload.
    comptime a: type_of(1) = ceildiv(1, 7)
    assert_equal(a, 1)
    comptime b: type_of(-78) = ceildiv(548, -7)
    assert_equal(b, -78)

    # Test the Int overload.
    assert_equal(ceildiv(Int(1), Int(7)), 1)
    assert_equal(ceildiv(Int(548), Int(-7)), -78)
    assert_equal(ceildiv(Int(-55), Int(8)), -6)
    assert_equal(ceildiv(Int(-55), Int(-8)), 7)

    # Test the UInt overload.
    assert_equal(
        ceildiv(UInt(1), UInt(7)),
        UInt(1),
    )
    assert_equal(
        ceildiv(UInt(546), UInt(7)),
        UInt(78),
    )

    # Test the SIMD overload.
    assert_equal(ceildiv(Float32(5), 2), ceildiv(5, 2))
    assert_equal((UInt32(5) + 1) // 2, ceildiv(5, 2))
    assert_equal(ceildiv(UInt32(5), UInt32(2)), ceildiv(5, 2))


def test_ceildiv_unsigned_overflow() raises:
    # Regression test for https://github.com/modular/modular/issues/6836:
    # `self + denominator - 1` overflows near the unsigned type's max, which
    # used to wrap the result to a small value instead of the true ceiling.
    assert_equal(ceildiv(UInt.MAX, UInt(2)), UInt(9223372036854775808))
    assert_equal(ceildiv(UInt.MAX, UInt(3)), UInt(6148914691236517205))
    assert_equal(ceildiv(UInt.MAX, UInt(1)), UInt.MAX)

    # Exact multiples must not round up.
    assert_equal(ceildiv(UInt(8), UInt(4)), UInt(2))
    assert_equal(ceildiv(UInt(0), UInt(4)), UInt(0))

    # Small values, unaffected by the overflow, stay correct.
    assert_equal(ceildiv(UInt(7), UInt(2)), UInt(4))
    assert_equal(ceildiv(UInt(1), UInt(7)), UInt(1))

    # Signed types never hit the overflowing branch; confirm no regression.
    assert_equal(ceildiv(Int(-55), Int(8)), -6)
    assert_equal(ceildiv(Int(548), Int(-7)), -78)

    # SIMD vector width > 1, near the type max in every lane.
    comptime max8 = UInt8.MAX
    var lanes = SIMD[.uint8, 4](max8, max8 - 1, 8, 7)
    var divisors = SIMD[.uint8, 4](2, 2, 4, 2)
    assert_equal(
        ceildiv(lanes, divisors),
        SIMD[.uint8, 4](128, 127, 2, 4),
    )


def test_align_down() raises:
    assert_equal(align_down(1, 7), 0)
    assert_equal(align_down(548, -7), 553)
    assert_equal(align_down(-548, -7), -546)
    assert_equal(align_down(-548, 7), -553)

    # Test the UInt overload.
    assert_equal(
        align_down(UInt(1), UInt(7)),
        UInt(0),
    )
    assert_equal(
        align_down(UInt(546), UInt(7)),
        UInt(546),
    )

    # Test the SIMD overload with various integer types.
    assert_equal(align_down(UInt32(385), UInt32(64)), UInt32(384))
    assert_equal(align_down(UInt32(512), UInt32(64)), UInt32(512))
    assert_equal(align_down(UInt32(0), UInt32(64)), UInt32(0))
    assert_equal(align_down(UInt16(100), UInt16(32)), UInt16(96))
    assert_equal(align_down(Int32(385), Int32(64)), Int32(384))
    assert_equal(align_down(Int32(-1), Int32(64)), Int32(-64))

    # Test SIMD vector (width > 1).
    assert_equal(
        align_down(
            SIMD[.uint32, 4](385, 512, 63, 0),
            SIMD[.uint32, 4](64, 64, 64, 64),
        ),
        SIMD[.uint32, 4](384, 512, 0, 0),
    )


def test_align_up() raises:
    assert_equal(align_up(1, 7), 7)
    assert_equal(align_up(548, -7), 546)
    assert_equal(align_up(-548, -7), -553)
    assert_equal(align_up(-548, 7), -546)

    # Test the UInt overload.
    assert_equal(
        align_up(UInt(1), UInt(7)),
        UInt(7),
    )
    assert_equal(
        align_up(UInt(546), UInt(7)),
        UInt(546),
    )

    # Test the SIMD overload with various integer types.
    assert_equal(align_up(UInt32(385), UInt32(64)), UInt32(448))
    assert_equal(align_up(UInt32(512), UInt32(64)), UInt32(512))
    assert_equal(align_up(UInt32(0), UInt32(64)), UInt32(0))
    assert_equal(align_up(UInt16(100), UInt16(32)), UInt16(128))
    assert_equal(align_up(Int32(385), Int32(64)), Int32(448))
    assert_equal(align_up(Int32(-1), Int32(64)), Int32(0))

    # Test SIMD vector (width > 1).
    assert_equal(
        align_up(
            SIMD[.uint32, 4](385, 512, 63, 0),
            SIMD[.uint32, 4](64, 64, 64, 64),
        ),
        SIMD[.uint32, 4](448, 512, 64, 0),
    )


def test_clamp() raises:
    assert_equal(clamp(Int(1), 0, 1), 1)
    assert_equal(clamp(Int(2), 0, 1), 1)
    assert_equal(clamp(Int(-2), 0, 1), 0)

    assert_equal(
        clamp(UInt(1), UInt(0), UInt(1)),
        UInt(1),
    )
    assert_equal(
        clamp(UInt(2), UInt(0), UInt(1)),
        UInt(1),
    )
    assert_equal(
        clamp(UInt(1), UInt(2), UInt(4)),
        UInt(2),
    )

    assert_equal(
        clamp(SIMD[.float32, 4](0, 1, 3, 4), 0, 1),
        SIMD[.float32, 4](0, 1, 1, 1),
    )


def test_fma() raises:
    # Test Int overload
    assert_equal(fma(5, 3, 2), 17)  # 5*3 + 2 = 17
    assert_equal(fma(-2, 3, 4), -2)  # -2*3 + 4 = -2
    assert_equal(fma(0, 100, 5), 5)  # 0*100 + 5 = 5

    # Test UInt (uses SIMD overload since UInt = UInt)
    assert_equal(
        fma(UInt(5), UInt(3), UInt(2)),
        UInt(17),
    )
    assert_equal(
        fma(
            UInt(1000000),
            UInt(1000),
            UInt(500),
        ),
        UInt(1000000500),
    )
    assert_equal(
        fma(
            UInt(0),
            UInt(100),
            UInt(5),
        ),
        UInt(5),
    )

    # Test SIMD overload with float
    assert_almost_equal(fma(Float32(2.5), Float32(4.0), Float32(1.5)), 11.5)
    assert_almost_equal(
        fma(
            SIMD[.float32, 4](1, 2, 3, 4),
            SIMD[.float32, 4](2, 2, 2, 2),
            SIMD[.float32, 4](1, 1, 1, 1),
        ),
        SIMD[.float32, 4](3, 5, 7, 9),
    )


def test_atanh() raises:
    assert_equal(atanh(Float32(1)), inf[.float32]())
    assert_equal(atanh(Float32(-1)), -inf[.float32]())
    assert_true(isnan(atanh(Float32(2))))
    assert_true(isnan(atanh(Float32(-2))))
    assert_almost_equal(
        atanh(SIMD[.float32, 4](0.5, 0.15, 0.9, 0.0)),
        atanh(SIMD[.float64, 4](0.5, 0.15, 0.9, 0.0)).cast[.float32](),
    )

    assert_equal(atanh(Float32(0)), Float32(0), msg="atanh(0)")
    assert_almost_equal(
        atanh(Float32(0.1)), Float32(0.1003353477310756), msg="atanh(0.1)"
    )
    assert_almost_equal(
        atanh(Float32(-0.1)), Float32(-0.1003353477310756), msg="atanh(-0.1)"
    )
    assert_almost_equal(
        atanh(Float32(0.2)), Float32(0.202732554054082), msg="atanh(0.2)"
    )
    assert_almost_equal(
        atanh(Float32(0.3)), Float32(0.3095196042031118), msg="atanh(0.3)"
    )
    assert_almost_equal(
        atanh(Float32(0.4)), Float32(0.4236489301936017), msg="atanh(0.4)"
    )
    assert_almost_equal(
        atanh(Float32(0.5)), Float32(0.54930614433405489), msg="atanh(0.5)"
    )
    assert_almost_equal(
        atanh(Float32(0.6)), Float32(0.6931471805599453), msg="atanh(0.6)"
    )
    assert_almost_equal(
        atanh(Float32(0.7)), Float32(0.8673005276940542), msg="atanh(0.7)"
    )
    assert_almost_equal(
        atanh(Float32(0.8)), Float32(1.0986122886681098), msg="atanh(0.8)"
    )
    assert_almost_equal(
        atanh(Float32(0.9)), Float32(1.4722194895832204), msg="atanh(0.9)"
    )

    assert_almost_equal(
        atanh(Float32(-0.5)), Float32(-0.54930614433405489), msg="atanh(-0.5)"
    )

    assert_almost_equal(
        atanh(Float32(-0.9297103072)),
        Float32(-1.65625),
        msg="atanh(-0.9297103072)",
    )


def test_sinh() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            sinh(val),
            _call_libm["sinh"](val),
            msg=String("mismatch for the value = ", val),
        )


def test_cosh() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            cosh(val),
            _call_libm["cosh"](val),
            msg=String("mismatch for the value = ", val),
        )


def test_expm1() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            expm1(val),
            _call_libm["expm1"](val),
            msg=String("mismatch for the value = ", val),
        )


def test_asin() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            asin(val),
            _call_libm["asin"](val),
            msg=String("mismatch for the value = ", val),
        )

    # Float64 accuracy across [-1, 1].
    for i in range(n):
        var val = Float64(i) / (n * 2) - 1
        assert_almost_equal(
            asin(val),
            _call_libm["asin"](val),
            msg=String("f64 mismatch for the value = ", val),
        )

    # Edge cases.
    assert_equal(asin(Float64(0)), Float64(0))
    assert_almost_equal(asin(Float64(1)), pi / 2.0)
    assert_almost_equal(asin(Float64(-1)), -(pi / 2.0))
    assert_true(isnan(asin(nan[.float64]())))


def test_erfc() raises:
    comptime n = 10_000
    for i in range(n):
        var val = Float32(i) / (n * Float32(2) / 10) - 10
        assert_almost_equal(
            erfc(val),
            _call_libm["erfc"](val),
        )


def test_cbrt() raises:
    comptime n = 1_0000
    for i in range(n):
        var val = Float32(i) / (n * Float32(2) / 10) - 10
        assert_almost_equal(
            cbrt(val),
            _call_libm["cbrt"](val),
            msg=String("mismatch for the value = ", val, " at index = ", i),
        )


def test_acos() raises:
    comptime n = 1_000
    for i in range(n):
        var val = Float32(i) / (n * 2) - 1
        assert_almost_equal(
            acos(val),
            _call_libm["acos"](val),
            msg=String("mismatch for the value = ", val),
        )

    # Float64 accuracy across [-1, 1].
    for i in range(n):
        var val = Float64(i) / (n * 2) - 1
        assert_almost_equal(
            acos(val),
            _call_libm["acos"](val),
            msg=String("f64 mismatch for the value = ", val),
        )

    # Edge cases.
    assert_almost_equal(acos(Float64(1)), Float64(0))
    assert_almost_equal(acos(Float64(-1)), pi)
    assert_almost_equal(acos(Float64(0)), pi / 2.0)
    assert_true(isnan(acos(nan[.float64]())))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
