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

from collections import InlineArray
from sys import has_neon

from builtin.simd import _modf
from memory import UnsafePointer
from testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from utils import IndexList, StaticTuple, unroll
from utils.numerics import isfinite, isinf, isnan, nan


def test_cast():
    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True).cast[DType.bool](),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True).cast[DType.int32](),
        SIMD[DType.int32, 4](0, 1, 0, 1),
    )

    assert_equal(
        SIMD[DType.float32, 4](0, 1, 0, -12).cast[DType.int32](),
        SIMD[DType.int32, 4](0, 1, 0, -12),
    )

    assert_equal(
        SIMD[DType.float32, 4](0, 1, 0, -12).cast[DType.bool](),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    var b: UInt16 = 128
    assert_equal(int(b.cast[DType.uint8]()), 128)
    assert_equal(int(b.cast[DType.uint16]()), 128)
    assert_equal(int(b.cast[DType.int8]()), -128)
    assert_equal(int(b.cast[DType.int16]()), 128)

    @parameter
    if not has_neon():
        assert_equal(
            BFloat16(33.0).cast[DType.float32]().cast[DType.bfloat16](), 33
        )
        assert_equal(
            Float16(33.0).cast[DType.float32]().cast[DType.float16](), 33
        )
        assert_equal(
            Float64(33.0).cast[DType.float32]().cast[DType.float16](), 33
        )


def test_simd_variadic():
    assert_equal(str(SIMD[DType.index, 4](52, 12, 43, 5)), "[52, 12, 43, 5]")


def test_convert_simd_to_string():
    var a: SIMD[DType.float32, 2] = 5
    assert_equal(str(a), "[5.0, 5.0]")

    var b: SIMD[DType.float64, 4] = 6
    assert_equal(str(b), "[6.0, 6.0, 6.0, 6.0]")

    var c: SIMD[DType.index, 8] = 7
    assert_equal(str(c), "[7, 7, 7, 7, 7, 7, 7, 7]")

    # TODO: uncomment when https://github.com/modularml/mojo/issues/2353 is fixed
    # assert_equal(str(UInt32(-1)), "4294967295")
    assert_equal(str(UInt64(-1)), "18446744073709551615")

    assert_equal(str((UInt16(32768))), "32768")
    assert_equal(str((UInt16(65535))), "65535")
    assert_equal(str((Int16(-2))), "-2")

    assert_equal(str(UInt64(16646288086500911323)), "16646288086500911323")

    # https://github.com/modularml/mojo/issues/556
    assert_equal(
        str(
            SIMD[DType.uint64, 4](
                0xA0761D6478BD642F,
                0xE7037ED1A0B428DB,
                0x8EBC6AF09C88C6E3,
                0x589965CC75374CC3,
            )
        ),
        (
            "[11562461410679940143, 16646288086500911323, 10285213230658275043,"
            " 6384245875588680899]"
        ),
    )

    assert_equal(
        str(
            SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448)
        ),
        "[-943274556, -875902520, -808530484, -741158448]",
    )


def test_simd_repr():
    assert_equal(
        SIMD[DType.int32, 4](1, 2, 3, 4).__repr__(),
        "SIMD[DType.int32, 4](1, 2, 3, 4)",
    )
    assert_equal(
        SIMD[DType.int32, 4](-1, 2, -3, 4).__repr__(),
        "SIMD[DType.int32, 4](-1, 2, -3, 4)",
    )
    assert_equal(
        SIMD[DType.bool, 2](True, False).__repr__(),
        "SIMD[DType.bool, 2](True, False)",
    )
    assert_equal(Int32(4).__repr__(), "SIMD[DType.int32, 1](4)")
    assert_equal(
        Float64(235234523.3452).__repr__(),
        "SIMD[DType.float64, 1](235234523.3452)",
    )
    assert_equal(
        Float32(2897239).__repr__(), "SIMD[DType.float32, 1](2897239.0)"
    )
    assert_equal(Float16(324).__repr__(), "SIMD[DType.float16, 1](324.0)")
    assert_equal(
        SIMD[DType.float32, 4](
            Float32.MAX, Float32.MIN, -0.0, nan[DType.float32]()
        ).__repr__(),
        "SIMD[DType.float32, 4](inf, -inf, -0.0, nan)",
    )


def test_issue_1625():
    var size = 16
    alias simd_width = 8
    var ptr = UnsafePointer[Int64].alloc(size)
    for i in range(size):
        ptr[i] = i

    var x = ptr.load[width = 2 * simd_width](0)
    var evens_and_odds = x.deinterleave()

    # FIXME (40568) should directly use the SIMD assert_equal
    assert_equal(
        str(evens_and_odds[0]),
        str(SIMD[DType.int64, 8](0, 2, 4, 6, 8, 10, 12, 14)),
    )
    assert_equal(
        str(evens_and_odds[1]),
        str(SIMD[DType.int64, 8](1, 3, 5, 7, 9, 11, 13, 15)),
    )
    ptr.free()


def test_issue_20421():
    var a = UnsafePointer[UInt8, alignment=64].alloc(count=16 * 64)
    for i in range(16 * 64):
        a[i] = i & 255
    var av16 = a.offset(128 + 64 + 4).bitcast[Int32]().load[
        width=4, alignment=1
    ]()
    assert_equal(
        av16,
        SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448),
    )
    a.free()


def test_issue_30237():
    alias dtype = DType.float32
    alias simd_width = 1
    alias coefficients_len = 7
    alias coefficients = InlineArray[SIMD[dtype, simd_width], coefficients_len](
        4.89352455891786e-03,
        6.37261928875436e-04,
        1.48572235717979e-05,
        5.12229709037114e-08,
        -8.60467152213735e-11,
        2.00018790482477e-13,
        -2.76076847742355e-16,
    )

    @parameter
    @always_inline
    fn eval1(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        var c_last = coefficients[coefficients_len - 1]
        var c_second_from_last = coefficients[coefficients_len - 2]

        var result = x.fma(c_last, c_second_from_last)

        @parameter
        for idx in range(coefficients_len - 2):
            var c = coefficients[coefficients_len - 3 - idx]
            result = x.fma(result, c)

        return result

    @parameter
    @always_inline
    fn eval2(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        var c_last = coefficients[coefficients_len - 1]
        var c_second_from_last = coefficients[coefficients_len - 2]

        var result = x.fma(c_last, c_second_from_last)

        for idx in range(coefficients_len - 2):
            var coefs = coefficients
            var c = coefs[coefficients_len - 3 - idx]
            result = x.fma(result, c)

        return result

    alias x = 6.0
    alias x2 = x * x
    var result1 = eval1(x2)
    var result2 = eval2(x2)

    assert_equal(result1, result2)


def test_bool():
    assert_true(Scalar[DType.bool](True).__bool__())
    assert_false(Scalar[DType.bool](False).__bool__())
    assert_true(Scalar[DType.int32](5).__bool__())
    assert_false(Scalar[DType.int32](0).__bool__())
    assert_true(Scalar[DType.float32](5.0).__bool__())
    assert_false(Scalar[DType.float32](0.0).__bool__())


def test_truthy():
    alias dtypes = (
        DType.bool,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float16,
        DType.float32,
        DType.float64,
        DType.index,
    )

    @parameter
    fn test_dtype[type: DType]() raises:
        # Scalars of 0-values are false-y, 1-values are truth-y
        assert_false(Scalar[type](False))
        assert_true(Scalar[type](True))

    @parameter
    fn test_dtype_unrolled[i: Int]() raises:
        alias type = dtypes.get[i, DType]()
        test_dtype[type]()

    unroll[test_dtype_unrolled, dtypes.__len__()]()

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        test_dtype[DType.bfloat16]()


def test_len():
    var i1 = Int32(0)
    assert_equal(i1.__len__(), 1)

    alias I32 = SIMD[DType.int32, 4]
    var i2 = I32(-1)
    assert_equal(4, i2.__len__())
    var i3 = I32(-1, 0, 1, 3)
    assert_equal(4, i3.__len__())

    alias I8 = SIMD[DType.int8, 1]
    var i4 = I8(1)
    assert_equal(1, i4.__len__())

    alias UI64 = SIMD[DType.uint64, 16]
    var i5 = UI64(10)
    assert_equal(16, i5.__len__())
    var i6 = UI64(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert_equal(16, i6.__len__())

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        alias BF16 = SIMD[DType.bfloat16, 2]
        var f1 = BF16(0.0)
        assert_equal(2, f1.__len__())
        var f2 = BF16(0.1, 0.2)
        assert_equal(2, f2.__len__())

    alias F = SIMD[DType.float64, 8]
    var f3 = F(1.0)
    assert_equal(8, f3.__len__())
    var f4 = F(0, -1.0, 1.0, -1.111, 1.111, -2.2222, 2.2222, 3.1415)
    assert_equal(8, f4.__len__())


def test_add():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__add__(0), I(-2, -4, 0, 1))
    assert_equal(i.__add__(Int32(0)), I(-2, -4, 0, 1))
    assert_equal(i.__add__(2), I(0, -2, 2, 3))
    assert_equal(i.__add__(Int32(2)), I(0, -2, 2, 3))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__add__(i2), I(3, 1, 0, 3))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__add__(f2), F(0, 0, 0, 0, 0, 0, 0, 0))


def test_radd():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__radd__(0), I(-2, -4, 0, 1))
    assert_equal(i.__radd__(Int32(0)), I(-2, -4, 0, 1))
    assert_equal(i.__radd__(2), I(0, -2, 2, 3))
    assert_equal(i.__radd__(Int32(2)), I(0, -2, 2, 3))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__radd__(i2), I(3, 1, 0, 3))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__radd__(f2), F(0, 0, 0, 0, 0, 0, 0, 0))


def test_iadd():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    i.__iadd__(0)
    assert_equal(i, I(-2, -4, 0, 1))
    i.__iadd__(Int32(0))
    assert_equal(i, I(-2, -4, 0, 1))
    i.__iadd__(2)
    assert_equal(i, I(0, -2, 2, 3))
    i.__iadd__(I(0, -2, 2, 3))
    assert_equal(i, I(0, -4, 4, 6))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    i1.__iadd__(i2)
    assert_equal(i1, I(3, 1, 0, 3))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    f1.__iadd__(f2)
    assert_equal(f1, F(0, 0, 0, 0, 0, 0, 0, 0))


def test_sub():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__sub__(0), I(-2, -4, 0, 1))
    assert_equal(i.__sub__(Int32(0)), I(-2, -4, 0, 1))
    assert_equal(i.__sub__(2), I(-4, -6, -2, -1))
    assert_equal(i.__sub__(Int32(2)), I(-4, -6, -2, -1))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__sub__(i2), I(-1, -9, -6, 1))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__sub__(f2), F(2, -2, 2, -2, 2, -2, 2, -2))


def test_rsub():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    assert_equal(i.__rsub__(0), I(2, 4, 0, -1))
    assert_equal(i.__rsub__(Int32(0)), I(2, 4, 0, -1))
    assert_equal(i.__rsub__(2), I(4, 6, 2, 1))
    assert_equal(i.__rsub__(Int32(2)), I(4, 6, 2, 1))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    assert_equal(i1.__rsub__(i2), I(1, 9, 6, -1))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    assert_equal(f1.__rsub__(f2), F(-2, 2, -2, 2, -2, 2, -2, 2))


def test_isub():
    alias I = SIMD[DType.int32, 4]
    var i = I(-2, -4, 0, 1)
    i.__isub__(0)
    assert_equal(i, I(-2, -4, 0, 1))
    i.__isub__(Int32(0))
    assert_equal(i, I(-2, -4, 0, 1))
    i.__isub__(2)
    assert_equal(i, I(-4, -6, -2, -1))
    i.__isub__(I(0, -2, 2, 3))
    assert_equal(i, I(-4, -4, -4, -4))

    var i1 = I(1, -4, -3, 2)
    var i2 = I(2, 5, 3, 1)
    i1.__isub__(i2)
    assert_equal(i1, I(-1, -9, -6, 1))

    alias F = SIMD[DType.float32, 8]
    var f1 = F(1, -1, 1, -1, 1, -1, 1, -1)
    var f2 = F(-1, 1, -1, 1, -1, 1, -1, 1)
    f1.__isub__(f2)
    assert_equal(f1, F(2, -2, 2, -2, 2, -2, 2, -2))


def test_ceil():
    assert_equal(Float32.__ceil__(Float32(1.5)), 2.0)
    assert_equal(Float32.__ceil__(Float32(-1.5)), -1.0)
    assert_equal(Float32.__ceil__(Float32(3.0)), 3.0)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        assert_equal(BFloat16.__ceil__(BFloat16(2.5)), 3.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(
        F.__ceil__(F(0.0, 1.4, -42.5, -12.6)), F(0.0, 2.0, -42.0, -12.0)
    )

    alias I = SIMD[DType.int32, 4]
    var i = I(0, 2, -42, -12)
    assert_equal(I.__ceil__(i), i)

    alias U = SIMD[DType.uint32, 4]
    var u = U(0, 2, 42, 12)
    assert_equal(U.__ceil__(u), u)

    alias B = SIMD[DType.bool, 4]
    var b = B(True, False, True, False)
    assert_equal(B.__ceil__(b), b)


def test_floor():
    assert_equal(Float32.__floor__(Float32(1.5)), 1.0)
    assert_equal(Float32.__floor__(Float32(-1.5)), -2.0)
    assert_equal(Float32.__floor__(Float32(3.0)), 3.0)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        assert_equal(BFloat16.__floor__(BFloat16(2.5)), 2.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(
        F.__floor__(F(0.0, 1.6, -42.5, -12.4)), F(0.0, 1.0, -43.0, -13.0)
    )

    alias I = SIMD[DType.int32, 4]
    var i = I(0, 2, -42, -12)
    assert_equal(I.__floor__(i), i)

    alias U = SIMD[DType.uint32, 4]
    var u = U(0, 2, 42, 12)
    assert_equal(U.__floor__(u), u)

    alias B = SIMD[DType.bool, 4]
    var b = B(True, False, True, False)
    assert_equal(B.__floor__(b), b)


def test_trunc():
    assert_equal(Float32.__trunc__(Float32(1.5)), 1.0)
    assert_equal(Float32.__trunc__(Float32(-1.5)), -1.0)
    assert_equal(Float32.__trunc__(Float32(3.0)), 3.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(
        F.__trunc__(F(0.0, 1.6, -42.5, -12.4)), F(0.0, 1.0, -42.0, -12.0)
    )

    alias I = SIMD[DType.int32, 4]
    var i = I(0, 2, -42, -12)
    assert_equal(I.__trunc__(i), i)

    alias U = SIMD[DType.uint32, 4]
    var u = U(0, 2, 42, 12)
    assert_equal(U.__trunc__(u), u)

    alias B = SIMD[DType.bool, 4]
    var b = B(True, False, True, False)
    assert_equal(B.__trunc__(b), b)


def test_round():
    assert_equal(Float32.__round__(Float32(2.5)), 3.0)
    assert_equal(Float32.__round__(Float32(-3.5)), -4.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(F.__round__(F(1.5, 2.5, -2.5, -3.5)), F(2.0, 3.0, -3.0, -4.0))


def test_roundeven():
    assert_equal(Float32(2.5).roundeven(), 2.0)
    assert_equal(Float32(-3.5).roundeven(), -4.0)

    alias F = SIMD[DType.float32, 4]
    assert_equal(F(1.5, 2.5, -2.5, -3.5).roundeven(), F(2.0, 2.0, -2.0, -4.0))


def test_div():
    assert_false(isfinite(Float32(33).__truediv__(0)))
    assert_false(isfinite(Float32(0).__truediv__(0)))

    assert_true(isinf(Float32(33).__truediv__(0)))
    assert_false(isinf(Float32(0).__truediv__(0)))

    assert_false(isnan(Float32(33).__truediv__(0)))
    assert_true(isnan(Float32(0).__truediv__(0)))

    alias F32 = SIMD[DType.float32, 4]
    var res = F32.__truediv__(F32(1, 0, 3, -1), F32(0, 0, 1, 0))
    alias B = SIMD[DType.bool, 4]
    assert_equal(isfinite(res), B(False, False, True, False))
    assert_equal(isinf(res), B(True, False, False, True))
    assert_equal(isnan(res), B(False, True, False, False))


def test_floordiv():
    assert_equal(Int32(2).__floordiv__(2), 1)
    assert_equal(Int32(2).__floordiv__(Int32(2)), 1)
    assert_equal(Int32(2).__floordiv__(Int32(3)), 0)

    assert_equal(Int32(2).__floordiv__(-2), -1)
    assert_equal(Int32(2).__floordiv__(Int32(-2)), -1)
    assert_equal(Int32(99).__floordiv__(Int32(-2)), -50)

    assert_equal(UInt32(2).__floordiv__(2), 1)
    assert_equal(UInt32(2).__floordiv__(UInt32(2)), 1)
    assert_equal(UInt32(2).__floordiv__(UInt32(3)), 0)

    assert_equal(Float32(2).__floordiv__(2), 1)
    assert_equal(Float32(2).__floordiv__(Float32(2)), 1)
    assert_equal(Float32(2).__floordiv__(Float32(3)), 0)

    assert_equal(Float32(2).__floordiv__(-2), -1)
    assert_equal(Float32(2).__floordiv__(Float32(-2)), -1)
    assert_equal(Float32(99).__floordiv__(Float32(-2)), -50)

    alias I = SIMD[DType.int32, 4]
    var i = I(2, 4, -2, -4)
    assert_equal(i.__floordiv__(2), I(1, 2, -1, -2))
    assert_equal(i.__floordiv__(Int32(2)), I(1, 2, -1, -2))

    alias F = SIMD[DType.float32, 4]
    var f = F(3, -4, 1, 5)
    assert_equal(f.__floordiv__(3), F(1, -2, 0, 1))
    assert_equal(f.__floordiv__(Float32(3)), F(1, -2, 0, 1))


def test_rfloordiv():
    alias I = SIMD[DType.int32, 4]
    var i = I(2, 4, -2, -4)
    assert_equal(i.__rfloordiv__(2), I(1, 0, -1, -1))
    assert_equal(i.__rfloordiv__(Int32(2)), I(1, 0, -1, -1))

    alias F = SIMD[DType.float32, 4]
    var f = F(3, -4, 1, 5)
    assert_equal(f.__rfloordiv__(3), F(1, -1, 3, 0))
    assert_equal(f.__rfloordiv__(Float32(3)), F(1, -1, 3, 0))


def test_mod():
    assert_equal(Int32(99) % Int32(1), 0)
    assert_equal(Int32(99) % Int32(3), 0)
    assert_equal(Int32(99) % Int32(-2), -1)
    assert_equal(Int32(99) % Int32(8), 3)
    assert_equal(Int32(99) % Int32(-8), -5)
    assert_equal(Int32(2) % Int32(-1), 0)
    assert_equal(Int32(2) % Int32(-2), 0)

    assert_equal(UInt32(99) % UInt32(1), 0)
    assert_equal(UInt32(99) % UInt32(3), 0)

    assert_equal(
        SIMD[DType.int32, 2](7, 7) % Int(4), SIMD[DType.int32, 2](3, 3)
    )

    var a = SIMD[DType.float32, 16](
        3.1,
        3.1,
        3.1,
        3.1,
        3.1,
        3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        -3.1,
        3.1,
        3.1,
        -3.1,
        -3.1,
    )
    var b = SIMD[DType.float32, 16](
        3.2,
        2.2,
        1.2,
        -3.2,
        -2.2,
        -1.2,
        3.2,
        2.2,
        1.2,
        -3.2,
        -2.2,
        -1.2,
        3.1,
        -3.1,
        3.1,
        -3.1,
    )
    assert_equal(
        a % b,
        SIMD[DType.float32, 16](
            3.0999999046325684,
            0.89999985694885254,
            0.69999980926513672,
            -0.10000014305114746,
            -1.3000001907348633,
            -0.5000002384185791,
            0.10000014305114746,
            1.3000001907348633,
            0.5000002384185791,
            -3.0999999046325684,
            -0.89999985694885254,
            -0.69999980926513672,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )


def test_rmod():
    assert_equal(Int32(3).__rmod__(Int(4)), 1)

    alias I = SIMD[DType.int32, 2]
    var i = I(78, 78)
    assert_equal(i.__rmod__(Int(78)), I(0, 0))

    alias F = SIMD[DType.float32, 4]
    var f = F(3, -4, 1, 5)
    assert_equal(f.__rmod__(3), F(0, -1, 0, 3))
    assert_equal(f.__rmod__(Float32(3)), F(0, -1, 0, 3))


def test_rotate():
    alias simd_width = 4
    alias type = DType.uint32

    assert_equal(
        SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[1](),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        SIMD[DType.uint32, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[5](),
        SIMD[DType.uint32, 8](1, 0, 1, 0, 0, 1, 0, 1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[1](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[3](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-1](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-3](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-4](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[1](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[3](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[4](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-1](),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-2](),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-3](),
        SIMD[type, simd_width](1, 1, 0, 1),
    )


def test_shift():
    alias simd_width = 4
    alias type = DType.uint32

    assert_equal(
        SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).shift_right[1](),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        SIMD[DType.uint32, 8](11, 0, 13, 12, 0, 100, 0, 0).shift_right[5](),
        SIMD[DType.uint32, 8](0, 0, 0, 0, 0, 11, 0, 13),
    )

    assert_equal(
        SIMD[DType.float64, 8](11.1, 0, 13.1, 12.2, 0, 100.4, 0, 0).shift_right[
            5
        ](),
        SIMD[DType.float64, 8](0, 0, 0, 0, 0, 11.1, 0, 13.1),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[1](),
        SIMD[type, simd_width](0, 1, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[2](),
        SIMD[type, simd_width](1, 1, 0, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[3](),
        SIMD[type, simd_width](1, 0, 0, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_left[4](),
        SIMD[type, simd_width](0, 0, 0, 0),
    )

    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[0](),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[1](),
        SIMD[type, simd_width](0, 1, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[2](),
        SIMD[type, simd_width](0, 0, 1, 0),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[3](),
        SIMD[type, simd_width](0, 0, 0, 1),
    )
    assert_equal(
        SIMD[type, simd_width](1, 0, 1, 1).shift_right[4](),
        SIMD[type, simd_width](0, 0, 0, 0),
    )


def test_shuffle():
    alias dtype = DType.int32
    alias width = 4

    vec = SIMD[dtype, width](100, 101, 102, 103)

    assert_equal(
        vec.shuffle[3, 2, 1, 0](), SIMD[dtype, width](103, 102, 101, 100)
    )
    assert_equal(
        vec.shuffle[0, 2, 4, 6](vec), SIMD[dtype, width](100, 102, 100, 102)
    )

    assert_equal(
        vec._shuffle_variadic[7, 6, 5, 4, 3, 2, 1, 0, output_size = 2 * width](
            vec
        ),
        SIMD[dtype, 2 * width](103, 102, 101, 100, 103, 102, 101, 100),
    )

    assert_equal(
        vec._shuffle_list[width, StaticTuple[Int, width](3, 2, 1, 0)](vec),
        SIMD[dtype, width](103, 102, 101, 100),
    )
    assert_equal(
        vec._shuffle_list[width, StaticTuple[Int, width](0, 2, 4, 6)](vec),
        SIMD[dtype, width](100, 102, 100, 102),
    )

    assert_equal(
        vec._shuffle_list[
            2 * width, StaticTuple[Int, 2 * width](7, 6, 5, 4, 3, 2, 1, 0)
        ](vec),
        SIMD[dtype, 2 * width](103, 102, 101, 100, 103, 102, 101, 100),
    )


def test_shuffle_dynamic_size_4_uint8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )

    indices = SIMD[DType.uint8, 4](3, 3, 5, 5)

    result = lookup_table._dynamic_shuffle(indices)
    expected_result = SIMD[DType.uint8, 4](30, 30, 50, 50)
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_8_uint8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )

    # Let's use size 8
    indices = SIMD[DType.uint8, 8](3, 3, 5, 5, 7, 7, 9, 0)

    result = lookup_table._dynamic_shuffle(indices)
    expected_result = SIMD[DType.uint8, 8](30, 30, 50, 50, 70, 70, 90, 0)
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_16_uint8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    var indices = SIMD[DType.uint8, 16](
        3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 0, 1
    )
    result = lookup_table._dynamic_shuffle(indices)
    expected_result = SIMD[DType.uint8, 16](
        30, 30, 50, 50, 70, 70, 90, 90, 110, 110, 130, 130, 150, 150, 0, 10
    )
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_32_uint8():
    var table_lookup = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    # fmt: off
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 ,
        11, 11, 13, 13, 15, 15, 0 , 1 ,
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ,
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = table_lookup._dynamic_shuffle(indices)

    expected_result = SIMD[DType.uint8, 32](
        30 , 30 , 50 , 50 , 70 , 70 , 90 , 90 ,
        110, 110, 130, 130, 150, 150, 0  , 10 ,
        0  , 10 , 20 , 30 , 40 , 50 , 60 , 70 ,
        80 , 90 , 100, 110, 120, 130, 140, 150,
    )
    # fmt: on
    assert_equal(result, expected_result)


def test_shuffle_dynamic_size_64_uint8():
    var table_lookup = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    # fmt: off
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 ,
        11, 11, 13, 13, 15, 15, 0 , 1 ,
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ,
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = table_lookup._dynamic_shuffle(indices.join(indices))

    expected_result = SIMD[DType.uint8, 32](
        30 , 30 , 50 , 50 , 70 , 70 , 90 , 90 ,
        110, 110, 130, 130, 150, 150, 0  , 10 ,
        0  , 10 , 20 , 30 , 40 , 50 , 60 , 70 ,
        80 , 90 , 100, 110, 120, 130, 140, 150,
    )
    # fmt: on
    assert_equal(result, expected_result.join(expected_result))


def test_shuffle_dynamic_size_32_float():
    # fmt: off
    var table_lookup = SIMD[DType.float64, 16](
        0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
        80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
    )
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 ,
        11, 11, 13, 13, 15, 15, 0 , 1 ,
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 ,
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = table_lookup._dynamic_shuffle(indices)

    expected_result = SIMD[DType.float64, 32](
        30. , 30. , 50. , 50. , 70. , 70. , 90. , 90. ,
        110., 110., 130., 130., 150., 150., 0.  , 10. ,
        0.  , 10. , 20. , 30. , 40. , 50. , 60. , 70. ,
        80. , 90. , 100., 110., 120., 130., 140., 150.,
    )
    # fmt: on
    assert_equal(result, expected_result)


def test_insert():
    assert_equal(Int32(3).insert(Int32(4)), 4)

    assert_equal(
        SIMD[DType.index, 4](0, 1, 2, 3).insert(SIMD[DType.index, 2](9, 6)),
        SIMD[DType.index, 4](9, 6, 2, 3),
    )

    assert_equal(
        SIMD[DType.index, 4](0, 1, 2, 3).insert[offset=1](
            SIMD[DType.index, 2](9, 6)
        ),
        SIMD[DType.index, 4](0, 9, 6, 3),
    )

    assert_equal(
        SIMD[DType.index, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=4](
            SIMD[DType.index, 4](9, 6, 3, 7)
        ),
        SIMD[DType.index, 8](0, 1, 2, 3, 9, 6, 3, 7),
    )

    assert_equal(
        SIMD[DType.index, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=3](
            SIMD[DType.index, 4](9, 6, 3, 7)
        ),
        SIMD[DType.index, 8](0, 1, 2, 9, 6, 3, 7, 8),
    )


def test_join():
    alias I2 = SIMD[DType.int32, 2]
    assert_equal(Int32(3).join(Int32(4)), I2(3, 4))

    alias I4 = SIMD[DType.int32, 4]
    assert_equal(I2(5, 6).join(I2(9, 10)), I4(5, 6, 9, 10))

    vec = I4(100, 101, 102, 103)
    assert_equal(
        vec.join(vec),
        SIMD[DType.int32, 8](100, 101, 102, 103, 100, 101, 102, 103),
    )


def test_interleave():
    assert_equal(
        str(Int32(0).interleave(Int32(1))), str(SIMD[DType.index, 2](0, 1))
    )

    assert_equal(
        SIMD[DType.index, 2](0, 2).interleave(SIMD[DType.index, 2](1, 3)),
        SIMD[DType.index, 4](0, 1, 2, 3),
    )


def test_deinterleave():
    var tup2 = SIMD[DType.float32, 2](1, 2).deinterleave()
    assert_equal(tup2[0], Float32(1))
    assert_equal(tup2[1], Float32(2))

    var tup4 = SIMD[DType.index, 4](0, 1, 2, 3).deinterleave()

    assert_equal(tup4[0], __type_of(tup4[0])(0, 2))
    assert_equal(tup4[1], __type_of(tup4[0])(1, 3))


def test_extract():
    assert_equal(Int64(99).slice[1](), 99)
    assert_equal(Int64(99).slice[1, offset=0](), 99)

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[4](),
        SIMD[DType.index, 4](99, 1, 2, 4),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2, offset=0](),
        SIMD[DType.index, 2](99, 1),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2, offset=2](),
        SIMD[DType.index, 2](2, 4),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2, offset=1](),
        SIMD[DType.index, 2](1, 2),
    )


def test_limits():
    @parameter
    fn test_integral_overflow[type: DType]() raises:
        var max_value = Scalar[type].MAX
        var min_value = Scalar[type].MIN
        assert_equal(max_value + 1, min_value)

    test_integral_overflow[DType.index]()
    test_integral_overflow[DType.int8]()
    test_integral_overflow[DType.uint8]()
    test_integral_overflow[DType.int16]()
    test_integral_overflow[DType.uint16]()
    test_integral_overflow[DType.int32]()
    test_integral_overflow[DType.uint32]()
    test_integral_overflow[DType.int64]()
    test_integral_overflow[DType.uint64]()


def test_abs():
    assert_equal(abs(Float32(1.0)), 1)
    assert_equal(abs(Float32(-1.0)), 1)
    assert_equal(abs(Float32(0.0)), 0)
    assert_equal(
        abs(SIMD[DType.float32, 4](0.0, 1.5, -42.5, -12.7)),
        SIMD[DType.float32, 4](0.0, 1.5, 42.5, 12.7),
    )
    assert_equal(
        abs(SIMD[DType.int32, 4](0, 2, -42, -12)),
        SIMD[DType.int32, 4](0, 2, 42, 12),
    )
    assert_equal(
        abs(SIMD[DType.uint32, 4](0, 2, 42, 12)),
        SIMD[DType.uint32, 4](0, 2, 42, 12),
    )
    assert_equal(
        abs(SIMD[DType.bool, 4](True, False, True, False)),
        SIMD[DType.bool, 4](True, False, True, False),
    )


def test_clamp():
    alias F = SIMD[DType.float32, 4]
    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(f.clamp(-6.0, 5.5), F(-6.0, -5.0, 5.0, 5.5))

    alias I = SIMD[DType.int32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(i.clamp(-7, 4), I(-7, -5, 4, 4))


def test_indexing():
    var s = SIMD[DType.int32, 4](1, 2, 3, 4)
    assert_equal(s[False], 1)
    assert_equal(s[int(2)], 3)
    assert_equal(s[3], 4)


def test_reduce():
    @parameter
    def test_dtype[type: DType]():
        alias X8 = SIMD[type, 8]
        alias X4 = SIMD[type, 4]
        alias X2 = SIMD[type, 2]
        alias X1 = SIMD[type, 1]
        var x8: X8
        var x4: X4
        var x2: X2
        var x1: X1

        @parameter
        if type.is_numeric():
            # reduce_add
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(4, 6, 8, 10)
            x2 = X2(12, 16)
            x1 = X1(int(28))  # TODO: fix MOCO-697 and use X1(28) instead
            assert_equal(x8.reduce_add(), x1)
            assert_equal(x4.reduce_add(), x1)
            assert_equal(x2.reduce_add(), x1)
            assert_equal(x1.reduce_add(), x1)
            assert_equal(x8.reduce_add[2](), x2)
            assert_equal(x4.reduce_add[2](), x2)
            assert_equal(x2.reduce_add[2](), x2)
            assert_equal(x8.reduce_add[4](), x4)
            assert_equal(x4.reduce_add[4](), x4)
            assert_equal(x8.reduce_add[8](), x8)
            assert_equal(X2(6, 3).reduce_add(), 9)

            # reduce_mul
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(0, 5, 12, 21)
            x2 = X2(0, 105)
            x1 = X1(int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_mul(), x1)
            assert_equal(x4.reduce_mul(), x1)
            assert_equal(x2.reduce_mul(), x1)
            assert_equal(x1.reduce_mul(), x1)
            assert_equal(x8.reduce_mul[2](), x2)
            assert_equal(x4.reduce_mul[2](), x2)
            assert_equal(x2.reduce_mul[2](), x2)
            assert_equal(x8.reduce_mul[4](), x4)
            assert_equal(x4.reduce_mul[4](), x4)
            assert_equal(x8.reduce_mul[8](), x8)
            assert_equal(X2(6, 3).reduce_mul(), 18)

            # reduce_min
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(0, 1, 2, 3)
            x2 = X2(0, 1)
            x1 = X1(int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_min(), x1)
            assert_equal(x4.reduce_min(), x1)
            assert_equal(x2.reduce_min(), x1)
            assert_equal(x1.reduce_min(), x1)
            assert_equal(x8.reduce_min[2](), x2)
            assert_equal(x4.reduce_min[2](), x2)
            assert_equal(x2.reduce_min[2](), x2)
            assert_equal(x8.reduce_min[4](), x4)
            assert_equal(x4.reduce_min[4](), x4)
            assert_equal(x8.reduce_min[8](), x8)
            assert_equal(X2(6, 3).reduce_min(), 3)

            # reduce_max
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(4, 5, 6, 7)
            x2 = X2(6, 7)
            x1 = X1(int(7))  # TODO: fix MOCO-697 and use X1(7) instead
            assert_equal(x8.reduce_max(), x1)
            assert_equal(x4.reduce_max(), x1)
            assert_equal(x2.reduce_max(), x1)
            assert_equal(x1.reduce_max(), x1)
            assert_equal(x8.reduce_max[2](), x2)
            assert_equal(x4.reduce_max[2](), x2)
            assert_equal(x2.reduce_max[2](), x2)
            assert_equal(x8.reduce_max[4](), x4)
            assert_equal(x4.reduce_max[4](), x4)
            assert_equal(x8.reduce_max[8](), x8)
            assert_equal(X2(6, 3).reduce_max(), 6)

        @parameter
        if type.is_signed():
            # reduce_add
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(4, -6, 8, -10)
            x2 = X2(12, -16)
            x1 = X1(int(-4))  # TODO: fix MOCO-697 and use X1(-4) instead
            assert_equal(x8.reduce_add(), x1)
            assert_equal(x4.reduce_add(), x1)
            assert_equal(x2.reduce_add(), x1)
            assert_equal(x1.reduce_add(), x1)
            assert_equal(x8.reduce_add[2](), x2)
            assert_equal(x4.reduce_add[2](), x2)
            assert_equal(x2.reduce_add[2](), x2)
            assert_equal(x8.reduce_add[4](), x4)
            assert_equal(x4.reduce_add[4](), x4)
            assert_equal(x8.reduce_add[8](), x8)
            assert_equal(X2(6, -3).reduce_add(), 3)

            # reduce_mul
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(0, 5, 12, 21)
            x2 = X2(0, 105)
            x1 = X1(int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_mul(), x1)
            assert_equal(x4.reduce_mul(), x1)
            assert_equal(x2.reduce_mul(), x1)
            assert_equal(x1.reduce_mul(), x1)
            assert_equal(x8.reduce_mul[2](), x2)
            assert_equal(x4.reduce_mul[2](), x2)
            assert_equal(x2.reduce_mul[2](), x2)
            assert_equal(x8.reduce_mul[4](), x4)
            assert_equal(x4.reduce_mul[4](), x4)
            assert_equal(x8.reduce_mul[8](), x8)
            assert_equal(X2(6, -3).reduce_mul(), -18)

            # reduce_min
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(0, -5, 2, -7)
            x2 = X2(0, -7)
            x1 = X1(int(-7))  # TODO: fix MOCO-697 and use X1(-7) instead
            assert_equal(x8.reduce_min(), x1)
            assert_equal(x4.reduce_min(), x1)
            assert_equal(x2.reduce_min(), x1)
            assert_equal(x1.reduce_min(), x1)
            assert_equal(x8.reduce_min[2](), x2)
            assert_equal(x4.reduce_min[2](), x2)
            assert_equal(x2.reduce_min[2](), x2)
            assert_equal(x8.reduce_min[4](), x4)
            assert_equal(x4.reduce_min[4](), x4)
            assert_equal(x8.reduce_min[8](), x8)
            assert_equal(X2(6, -3).reduce_min(), -3)

            # reduce_max
            x8 = X8(0, -1, 2, -3, 4, -5, 6, -7)
            x4 = X4(4, -1, 6, -3)
            x2 = X2(6, -1)
            x1 = X1(int(6))  # TODO: fix MOCO-697 and use X1(6) instead
            assert_equal(x8.reduce_max(), x1)
            assert_equal(x4.reduce_max(), x1)
            assert_equal(x2.reduce_max(), x1)
            assert_equal(x1.reduce_max(), x1)
            assert_equal(x8.reduce_max[2](), x2)
            assert_equal(x4.reduce_max[2](), x2)
            assert_equal(x2.reduce_max[2](), x2)
            assert_equal(x8.reduce_max[4](), x4)
            assert_equal(x4.reduce_max[4](), x4)
            assert_equal(x8.reduce_max[8](), x8)
            assert_equal(X2(6, -3).reduce_max(), 6)

        @parameter
        if type is DType.bool:
            # reduce_and
            var x8b = SIMD[DType.bool, 8](
                False, False, True, True, False, True, False, True
            )
            var x4b = SIMD[DType.bool, 4](False, False, False, True)
            var x2b = SIMD[DType.bool, 2](False, False)
            var x1b = SIMD[DType.bool, 1](False)
            assert_equal(x8b.reduce_and(), x1b)
            assert_equal(x4b.reduce_and(), x1b)
            assert_equal(x2b.reduce_and(), x1b)
            assert_equal(x1b.reduce_and(), x1b)
            assert_equal(x8b.reduce_and[2](), x2b)
            assert_equal(x4b.reduce_and[2](), x2b)
            assert_equal(x2b.reduce_and[2](), x2b)
            assert_equal(x8b.reduce_and[4](), x4b)
            assert_equal(x4b.reduce_and[4](), x4b)
            assert_equal(x8b.reduce_and[8](), x8b)
            assert_equal(SIMD[DType.bool, 2](True, True).reduce_and(), True)

            # reduce_or
            x8b = SIMD[DType.bool, 8](
                False, False, True, True, False, True, False, True
            )
            x4b = SIMD[DType.bool, 4](False, True, True, True)
            x2b = SIMD[DType.bool, 2](True, True)
            x1b = SIMD[DType.bool, 1](True)
            assert_equal(x8b.reduce_or(), x1b)
            assert_equal(x4b.reduce_or(), x1b)
            assert_equal(x2b.reduce_or(), x1b)
            assert_equal(x1b.reduce_or(), x1b)
            assert_equal(x8b.reduce_or[2](), x2b)
            assert_equal(x4b.reduce_or[2](), x2b)
            assert_equal(x2b.reduce_or[2](), x2b)
            assert_equal(x8b.reduce_or[4](), x4b)
            assert_equal(x4b.reduce_or[4](), x4b)
            assert_equal(x8b.reduce_or[8](), x8b)
            assert_equal(SIMD[DType.bool, 2](False, False).reduce_or(), False)

        @parameter
        if type.is_integral():
            # reduce_and
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(0, 1, 2, 3)
            x2 = X2(0, 1)
            x1 = X1(int(0))  # TODO: fix MOCO-697 and use X1(0) instead
            assert_equal(x8.reduce_and(), x1)
            assert_equal(x4.reduce_and(), x1)
            assert_equal(x2.reduce_and(), x1)
            assert_equal(x1.reduce_and(), x1)
            assert_equal(x8.reduce_and[2](), x2)
            assert_equal(x4.reduce_and[2](), x2)
            assert_equal(x2.reduce_and[2](), x2)
            assert_equal(x8.reduce_and[4](), x4)
            assert_equal(x4.reduce_and[4](), x4)
            assert_equal(x8.reduce_and[8](), x8)
            assert_equal(X2(6, 3).reduce_and(), 2)

            # reduce_or
            x8 = X8(0, 1, 2, 3, 4, 5, 6, 7)
            x4 = X4(4, 5, 6, 7)
            x2 = X2(6, 7)
            x1 = X1(int(7))  # TODO: fix MOCO-697 and use X1(7) instead
            assert_equal(x8.reduce_or(), x1)
            assert_equal(x4.reduce_or(), x1)
            assert_equal(x2.reduce_or(), x1)
            assert_equal(x1.reduce_or(), x1)
            assert_equal(x8.reduce_or[2](), x2)
            assert_equal(x4.reduce_or[2](), x2)
            assert_equal(x2.reduce_or[2](), x2)
            assert_equal(x8.reduce_or[4](), x4)
            assert_equal(x4.reduce_or[4](), x4)
            assert_equal(x8.reduce_or[8](), x8)
            assert_equal(X2(6, 3).reduce_or(), 7)

    test_dtype[DType.bool]()
    test_dtype[DType.int8]()
    test_dtype[DType.int16]()
    test_dtype[DType.int32]()
    test_dtype[DType.int64]()
    test_dtype[DType.uint8]()
    test_dtype[DType.uint16]()
    test_dtype[DType.uint32]()
    test_dtype[DType.uint64]()
    test_dtype[DType.float16]()
    test_dtype[DType.float32]()
    test_dtype[DType.float64]()
    test_dtype[DType.index]()

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        test_dtype[DType.bfloat16]()


def test_reduce_bit_count():
    var int_0xFFFF = Int32(0xFFFF)
    assert_equal(int_0xFFFF.reduce_bit_count(), 16)

    var int_iota8 = SIMD[DType.int32, 8](0, 1, 2, 3, 4, 5, 6, 7)
    assert_equal(int_iota8.reduce_bit_count(), 12)

    var bool_true = Scalar[DType.bool](True)
    assert_equal(bool_true.reduce_bit_count(), 1)

    var bool_false = Scalar[DType.bool](False)
    assert_equal(bool_false.reduce_bit_count(), 0)

    var bool_true16 = SIMD[DType.bool, 16](True)
    assert_equal(bool_true16.reduce_bit_count(), 16)


def test_pow():
    alias nan = FloatLiteral.nan
    alias neg_zero = FloatLiteral.negative_zero
    alias inf = FloatLiteral.infinity
    alias neg_inf = FloatLiteral.negative_infinity

    # Float32 tests
    alias F32x4 = SIMD[DType.float32, 4]
    alias F32x8 = SIMD[DType.float32, 8]

    var f32x4_val = F32x4(0, 1, 2, 3)
    var f32x8_val = F32x8(0, 1, 2, 3, 4, 5, 6, 7)
    assert_equal(f32x4_val.__pow__(10.0), F32x4(0.0, 1.0, 1024.0, 59049.0))
    assert_almost_equal(
        f32x8_val.__pow__(15.0),
        F32x8(
            0.0,
            1.0,
            32768.0,
            14348907.0,
            1073741824.0,
            30517578125.0,
            470184984576.0,
            4747561509943.0,
        ),
    )
    assert_almost_equal(
        f32x4_val.__pow__(-1.0), F32x4(inf, 1.0, 0.5, 0.333333333)
    )
    assert_equal(f32x4_val.__pow__(0.0), F32x4(1.0, 1.0, 1.0, 1.0))
    assert_equal(F32x4(1, 1, 1, 1).__pow__(100.0), F32x4(1.0, 1.0, 1.0, 1.0))
    assert_equal(
        F32x4(inf, -inf, nan, 1).__pow__(3.0), F32x4(inf, -inf, nan, 1.0)
    )
    assert_almost_equal(
        f32x4_val.__pow__(0.5), F32x4(0.0, 1.0, 1.414213562, 1.732050808)
    )

    assert_almost_equal(
        F32x4(1, 2, 3, 4).__pow__(F32x4(2, 3, 2, 1)), F32x4(1.0, 8.0, 9.0, 4.0)
    )

    var f32x4_neg_zero = F32x4(neg_zero, neg_zero, neg_zero, neg_zero)
    assert_equal(
        f32x4_neg_zero.__pow__(F32x4(2.0, 3.0, 1.0, 4.0)),
        F32x4(0.0, 0.0, 0.0, 0.0),
    )
    assert_equal(
        f32x4_neg_zero.__pow__(3.0),
        F32x4(neg_zero, neg_zero, neg_zero, neg_zero),
    )

    assert_almost_equal(
        F32x4(neg_zero, 1.0, 2.0, 3.0).__pow__(F32x4(2.0, 4.0, 8.0, 16.0)),
        F32x4(0.0, 1.0, 256.0, 43046721.0),
    )

    assert_equal(
        F32x4(2.0, 3.0, 4.0, 5.0).__pow__(neg_zero), F32x4(1.0, 1.0, 1.0, 1.0)
    )

    assert_equal(
        F32x4(inf, neg_inf, nan, 1.0).__pow__(F32x4(2.0, 3.0, 2.0, 0.0)),
        F32x4(inf, neg_inf, nan, 1.0),
    )

    assert_equal(
        F32x4(neg_inf, neg_inf, neg_inf, neg_inf).__pow__(
            F32x4(2.0, 3.0, 4.0, 5.0)
        ),
        F32x4(inf, neg_inf, inf, neg_inf),
    )

    # Float64 tests
    alias F64x4 = SIMD[DType.float64, 4]

    assert_equal(
        F64x4(0, 1, 2, 3).__pow__(20.0),
        F64x4(0.0, 1.0, 1048576.0, 3486784401.0),
    )

    assert_almost_equal(
        F64x4(1.0, 2.0, 3.0, 4.0).__pow__(F64x4(2.0, 3.0, 2.0, 1.0)),
        F64x4(1.0, 8.0, 9.0, 4.0),
    )

    # Int32 tests
    alias I32x4 = SIMD[DType.int32, 4]

    var i32x4_val = I32x4(0, 1, 2, 3)

    assert_equal(i32x4_val.__pow__(20), I32x4(0, 1, 1048576, 3486784401))
    assert_equal(i32x4_val.__pow__(0), I32x4(1, 1, 1, 1))
    assert_equal(I32x4(-2, -1, 0, 1).__pow__(3), I32x4(-8, -1, 0, 1))
    assert_equal(
        I32x4(2, 2, 2, 2).__pow__(30),
        I32x4(1073741824, 1073741824, 1073741824, 1073741824),
    )

    assert_equal(
        I32x4(2, 3, 4, 5).__pow__(I32x4(3, 2, 1, 0)), I32x4(8, 9, 4, 1)
    )

    var i32x4_edge_base = I32x4(-2147483648, -1, 0, 2147483647)
    var i32x4_edge_exp = I32x4(31, 31, 31, 31)
    assert_equal(
        i32x4_edge_base.__pow__(i32x4_edge_exp), I32x4(0, -1, 0, 2147483647)
    )
    assert_equal(i32x4_edge_base.__pow__(32), I32x4(0, 1, 0, 1))

    # UInt32 tests
    alias U32x4 = SIMD[DType.uint32, 4]

    var u32x4_val = U32x4(0, 1, 2, 3)

    assert_equal(u32x4_val.__pow__(20), U32x4(0, 1, 1048576, 3486784401))

    assert_equal(
        U32x4(1, 2, 3, 4).__pow__(U32x4(0, 1, 2, 3)), U32x4(1, 2, 9, 64)
    )

    var u32x4_edge_base = U32x4(0, 1, 2147483647, 4294967295)
    assert_equal(
        u32x4_edge_base.__pow__(U32x4(31, 31, 31, 31)),
        U32x4(0, 1, 2147483647, 4294967295),
    )
    assert_equal(u32x4_edge_base.__pow__(32), U32x4(0, 1, 1, 1))

    # Int8 tests
    alias I8x4 = SIMD[DType.int8, 4]

    var i8x4_val = I8x4(0, 1, 2, 3)

    assert_equal(i8x4_val.__pow__(2), I8x4(0, 1, 4, 9))
    assert_equal(i8x4_val.__pow__(7), I8x4(0, 1, 128, -117))
    assert_equal(I8x4(-128, -1, 0, 127).__pow__(3), I8x4(0, -1, 0, 127))

    # UInt8 tests
    alias U8x4 = SIMD[DType.uint8, 4]

    var u8x4_val = U8x4(0, 1, 2, 3)
    assert_equal(u8x4_val.__pow__(2), U8x4(0, 1, 4, 9))
    assert_equal(u8x4_val.__pow__(8), U8x4(0, 1, 0, 161))
    assert_equal(u8x4_val.__pow__(U8x4(3, 5, 7, 9)), U8x4(0, 1, 128, 227))


def test_powf():
    assert_almost_equal(Float32(2.0) ** Float32(0.5), 1.4142135)
    assert_almost_equal(Float32(2.0) ** Float32(-0.5), 0.707107)
    assert_almost_equal(Float32(50.0) ** Float32(2.5), 17677.6695297)
    assert_almost_equal(Float32(12.0) ** Float32(0.4), 2.70192)
    assert_almost_equal(Float32(-1.0) ** Float32(-1), -1)
    assert_almost_equal(Float32(0.001) ** Float32(0.001), 0.99311605)

    assert_almost_equal(Float64(0.001) ** Float64(0.001), 0.99311605)

    assert_almost_equal(Float32(-4) ** Float32(-3), -0.015625)

    assert_almost_equal(
        SIMD[DType.float64, 8](1.0, -1.0, 2.0, -2.0, 4.0, -4.0, -2.0, 3.0)
        ** SIMD[DType.float64, 8](1, 2, 3, 4, 5, 6, 2, 1),
        SIMD[DType.float64, 8](1, 1, 8, 16, 1024, 4096, 4, 3),
    )


def test_rpow():
    alias F32x4 = SIMD[DType.float32, 4]
    alias I32x4 = SIMD[DType.int32, 4]

    var f32x4_val = F32x4(0, 1, 2, 3)
    var i32x4_val = I32x4(0, 1, 2, 3)

    assert_equal(0**i32x4_val, I32x4(1, 0, 0, 0))
    assert_equal(2**i32x4_val, I32x4(1, 2, 4, 8))
    assert_equal((-1) ** i32x4_val, I32x4(1, -1, 1, -1))

    assert_equal(Int(0) ** i32x4_val, I32x4(1, 0, 0, 0))
    assert_equal(Int(2) ** i32x4_val, I32x4(1, 2, 4, 8))
    assert_equal(Int(-1) ** i32x4_val, I32x4(1, -1, 1, -1))

    assert_equal(UInt(2) ** i32x4_val, I32x4(1, 2, 4, 8))
    assert_equal(UInt(0) ** i32x4_val, I32x4(1, 0, 0, 0))

    assert_almost_equal(1.0**f32x4_val, F32x4(1.0, 1.0, 1.0, 1.0))
    assert_almost_equal(2.5**f32x4_val, F32x4(1.0, 2.5, 6.25, 15.625))
    assert_almost_equal(3.0**f32x4_val, F32x4(1.0, 3.0, 9.0, 27.0))


def test_modf():
    var f32 = _modf(Float32(123.5))
    assert_almost_equal(f32[0], 123)
    assert_almost_equal(f32[1], 0.5)

    var f64 = _modf(Float64(123.5))
    assert_almost_equal(f64[0], 123)
    assert_almost_equal(f64[1], 0.5)

    f64 = _modf(Float64(0))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0)

    f64 = _modf(Float64(0.5))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0.5)

    f64 = _modf(Float64(-0.5))
    assert_almost_equal(f64[0], -0)
    assert_almost_equal(f64[1], -0.5)

    f64 = _modf(Float64(-1.5))
    assert_almost_equal(f64[0], -1)
    assert_almost_equal(f64[1], -0.5)


def test_split():
    var tup = SIMD[DType.index, 8](1, 2, 3, 4, 5, 6, 7, 8).split()
    assert_equal(tup[0], __type_of(tup[0])(1, 2, 3, 4))
    assert_equal(tup[1], __type_of(tup[1])(5, 6, 7, 8))


def test_contains():
    var x = SIMD[DType.int8, 4](1, 2, 3, 4)
    assert_true(1 in x and 2 in x and 3 in x and 4 in x)
    assert_false(0 in x or 5 in x)
    var y = SIMD[DType.float16, 4](1, 2, 3, 4)
    assert_true(1 in y and 2 in y and 3 in y and 4 in y)
    assert_false(0 in y or 5 in y)


def test_comparison():
    alias dtypes = (
        DType.bool,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.float16,
        DType.float32,
        DType.float64,
        DType.index,
    )

    @parameter
    fn test_dtype[type: DType]() raises:
        alias X4 = SIMD[type, 4]

        @parameter
        if type.is_signed():
            var simd_val = X4(-10, -8, -6, -4)

            assert_true(simd_val.__lt__(X4(-1)).reduce_and())
            assert_false(simd_val.__lt__(X4(-12)).reduce_or())
            var mixed_lt = simd_val.__lt__(X4(-6))
            assert_true(mixed_lt[0])
            assert_true(mixed_lt[1])
            assert_false(mixed_lt[2])
            assert_false(mixed_lt[3])

            assert_true(simd_val.__le__(X4(-4)).reduce_and())
            assert_false(simd_val.__le__(X4(-11)).reduce_or())
            var mixed_le = simd_val.__le__(X4(-8))
            assert_true(mixed_le[0])
            assert_true(mixed_le[1])
            assert_false(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(simd_val.__eq__(X4(-10, -8, -6, -4)).reduce_and())
            assert_false(simd_val.__eq__(X4(0)).reduce_or())
            var mixed_eq = simd_val.__eq__(X4(-10))
            assert_true(mixed_eq[0])
            assert_false(mixed_eq[1])
            assert_false(mixed_eq[2])
            assert_false(mixed_eq[3])

            assert_true(simd_val.__ne__(X4(0)).reduce_and())
            assert_false(simd_val.__ne__(X4(-10, -8, -6, -4)).reduce_or())
            var mixed_ne = simd_val.__ne__(X4(-8))
            assert_true(mixed_ne[0])
            assert_false(mixed_ne[1])
            assert_true(mixed_ne[2])
            assert_true(mixed_ne[3])

            assert_true(simd_val.__gt__(X4(-11)).reduce_and())
            assert_false(simd_val.__gt__(X4(-1)).reduce_or())
            var mixed_gt = simd_val.__gt__(X4(-6))
            assert_false(mixed_gt[0])
            assert_false(mixed_gt[1])
            assert_false(mixed_gt[2])
            assert_true(mixed_gt[3])

            assert_true(simd_val.__ge__(X4(-10)).reduce_and())
            assert_false(simd_val.__ge__(X4(-1)).reduce_or())
            var mixed_ge = simd_val.__ge__(X4(-6))
            assert_false(mixed_ge[0])
            assert_false(mixed_ge[1])
            assert_true(mixed_ge[2])
            assert_true(mixed_ge[3])

        @parameter
        if type.is_numeric():
            var simd_val = X4(1, 2, 3, 4)

            assert_true(simd_val.__lt__(X4(5)).reduce_and())
            assert_false(simd_val.__lt__(X4(0)).reduce_or())
            var mixed_lt = simd_val.__lt__(X4(3))
            assert_true(mixed_lt[0])
            assert_true(mixed_lt[1])
            assert_false(mixed_lt[2])
            assert_false(mixed_lt[3])

            assert_true(simd_val.__le__(X4(4)).reduce_and())
            assert_false(simd_val.__le__(X4(0)).reduce_or())
            var mixed_le = simd_val.__le__(X4(3))
            assert_true(mixed_le[0])
            assert_true(mixed_le[1])
            assert_true(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(simd_val.__eq__(X4(1, 2, 3, 4)).reduce_and())
            assert_false(simd_val.__eq__(X4(5)).reduce_or())
            var mixed_eq = simd_val.__eq__(X4(1))
            assert_true(mixed_eq[0])
            assert_false(mixed_eq[1])
            assert_false(mixed_eq[2])
            assert_false(mixed_eq[3])

            assert_true(simd_val.__ne__(X4(5)).reduce_and())
            assert_false(simd_val.__ne__(X4(1, 2, 3, 4)).reduce_or())
            var mixed_ne = simd_val.__ne__(X4(4))
            assert_true(mixed_ne[0])
            assert_true(mixed_ne[1])
            assert_true(mixed_ne[2])
            assert_false(mixed_ne[3])

            assert_true(simd_val.__gt__(X4(0)).reduce_and())
            assert_false(simd_val.__gt__(X4(4)).reduce_or())
            var mixed_gt = simd_val.__gt__(X4(2))
            assert_false(mixed_gt[0])
            assert_false(mixed_gt[1])
            assert_true(mixed_gt[2])
            assert_true(mixed_gt[3])

            assert_true(simd_val.__ge__(X4(1)).reduce_and())
            assert_false(simd_val.__ge__(X4(5)).reduce_or())
            var mixed_ge = simd_val.__ge__(X4(2))
            assert_false(mixed_ge[0])
            assert_true(mixed_ge[1])
            assert_true(mixed_ge[2])
            assert_true(mixed_ge[3])

        @parameter
        if type is DType.bool:
            var all_true = SIMD[DType.bool, 4](True)
            var all_false = SIMD[DType.bool, 4](False)
            var mixed = SIMD[DType.bool, 4](True, True, False, False)

            assert_true(all_false.__lt__(all_true).reduce_and())
            assert_false(all_true.__lt__(all_false).reduce_or())
            var mixed_lt = all_false.__lt__(mixed)
            assert_true(mixed_lt[0])
            assert_true(mixed_lt[1])
            assert_false(mixed_lt[2])
            assert_false(mixed_lt[3])

            assert_true(all_false.__le__(all_true).reduce_and())
            assert_false(all_true.__le__(all_false).reduce_or())
            var mixed_le = all_true.__le__(mixed)
            assert_true(mixed_le[0])
            assert_true(mixed_le[1])
            assert_false(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(
                all_true.__eq__(
                    SIMD[DType.bool, 4](True, True, True, True)
                ).reduce_and()
            )
            assert_false(all_true.__eq__(all_false).reduce_or())
            var mixed_eq = all_true.__eq__(mixed)
            assert_true(mixed_eq[0])
            assert_true(mixed_eq[1])
            assert_false(mixed_le[2])
            assert_false(mixed_le[3])

            assert_true(all_true.__ne__(all_false).reduce_and())
            assert_false(
                all_true.__ne__(
                    SIMD[DType.bool, 4](True, True, True, True)
                ).reduce_or()
            )
            var mixed_ne = all_true.__ne__(mixed)
            assert_false(mixed_ne[0])
            assert_false(mixed_ne[1])
            assert_true(mixed_ne[2])
            assert_true(mixed_ne[3])

            assert_true(all_true.__gt__(all_false).reduce_and())
            assert_false(all_false.__gt__(all_true).reduce_or())
            var mixed_gt = all_true.__gt__(mixed)
            assert_false(mixed_gt[0])
            assert_false(mixed_gt[1])
            assert_true(mixed_gt[2])
            assert_true(mixed_gt[3])

            assert_true(all_true.__ge__(all_false).reduce_and())
            assert_false(all_false.__ge__(all_true).reduce_or())
            var mixed_ge = all_true.__ge__(mixed)
            assert_true(mixed_ge[0])
            assert_true(mixed_ge[1])
            assert_true(mixed_ge[2])
            assert_true(mixed_ge[3])

    @parameter
    fn test_dtype_unrolled[i: Int]() raises:
        alias type = dtypes.get[i, DType]()
        test_dtype[type]()

    unroll[test_dtype_unrolled, dtypes.__len__()]()

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        test_dtype[DType.bfloat16]()


def test_float_conversion():
    assert_almost_equal(float(Int32(45)), 45.0)
    assert_almost_equal(float(Float32(34.32)), 34.32)
    assert_almost_equal(float(UInt64(36)), 36.0)


def test_reversed():
    fn test[D: DType]() raises:
        assert_equal(SIMD[D, 4](1, 2, 3, 4).reversed(), SIMD[D, 4](4, 3, 2, 1))

    test[DType.uint8]()
    test[DType.uint16]()
    test[DType.uint32]()
    test[DType.uint64]()
    test[DType.int8]()
    test[DType.int16]()
    test[DType.int32]()
    test[DType.int64]()
    test[DType.float16]()
    test[DType.float32]()
    test[DType.float64]()


def main():
    test_abs()
    test_add()
    test_cast()
    test_ceil()
    test_convert_simd_to_string()
    test_simd_repr()
    test_deinterleave()
    test_div()
    test_extract()
    test_floor()
    test_floordiv()
    test_iadd()
    test_indexing()
    test_insert()
    test_interleave()
    test_issue_1625()
    test_issue_20421()
    test_issue_30237()
    test_isub()
    test_join()
    test_len()
    test_limits()
    test_clamp()
    test_mod()
    test_pow()
    test_powf()
    test_rpow()
    test_radd()
    test_reduce()
    test_reduce_bit_count()
    test_rfloordiv()
    test_rmod()
    test_rotate()
    test_round()
    test_roundeven()
    test_rsub()
    test_shift()
    test_shuffle()
    test_shuffle_dynamic_size_4_uint8()
    test_shuffle_dynamic_size_8_uint8()
    test_shuffle_dynamic_size_16_uint8()
    test_shuffle_dynamic_size_32_uint8()
    test_shuffle_dynamic_size_64_uint8()
    test_shuffle_dynamic_size_32_float()
    test_simd_variadic()
    test_sub()
    test_trunc()
    test_bool()
    test_truthy()
    test_modf()
    test_split()
    test_contains()
    test_comparison()
    test_float_conversion()
    test_reversed()
    # TODO: add tests for __and__, __or__, anc comparison operators
