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

from sys import has_neon
from utils.numerics import isfinite, isinf, isnan

from testing import assert_equal, assert_not_equal, assert_true, assert_false


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
    assert_equal(str(Scalar[DType.address](22)), "0x16")
    assert_equal(str(Scalar[DType.address](0xDEADBEAF)), "0xdeadbeaf")

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


def test_issue_20421():
    var a = DTypePointer[DType.uint8]().alloc(16 * 64, alignment=64)
    for i in range(16 * 64):
        a[i] = i & 255
    var av16 = a.offset(128 + 64 + 4).bitcast[DType.int32]().load[width=4]()
    assert_equal(
        av16,
        SIMD[DType.int32, 4](-943274556, -875902520, -808530484, -741158448),
    )
    a.free()


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
        # DType.address  # TODO(29920)
    )

    @parameter
    fn test_dtype[type: DType]() raises:
        # # Scalars of 0-values are false-y, 1-values are truth-y
        assert_false(Scalar[type](False).__bool__())
        assert_true(Scalar[type](True).__bool__())

        # # SIMD vectors are truth-y if _all_ values are truth-y
        assert_true(SIMD[type, 2](True, True).__bool__())

        # # SIMD vectors are false-y if _any_ values are false-y
        assert_false(SIMD[type, 2](False, True).__bool__())
        assert_false(SIMD[type, 2](True, False).__bool__())
        assert_false(SIMD[type, 2](False, False).__bool__())

    @parameter
    fn test_dtype_unrolled[i: Int]() raises:
        alias type = dtypes.get[i, DType]()
        test_dtype[type]()

    unroll[test_dtype_unrolled, dtypes.__len__()]()

    @parameter
    if not has_neon():
        # TODO bfloat16 is not supported on neon #30525
        test_dtype[DType.bfloat16]()


def test_len():
    var i1 = Int32(0)
    assert_equal(i1.__len__(), 1)

    alias I32 = SIMD[DType.int32, 4]
    var i2 = I32(-1, 0, 1)
    assert_equal(4, i2.__len__())
    var i3 = I32(-1, 0, 1, 3)
    assert_equal(4, i3.__len__())

    alias I8 = SIMD[DType.int8, 1]
    var i4 = I8(1)
    assert_equal(1, i4.__len__())

    alias UI64 = SIMD[DType.uint64, 16]
    var i5 = UI64(10, 20, 30, 40)
    assert_equal(16, i5.__len__())
    var i6 = UI64(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    assert_equal(16, i6.__len__())

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


def test_ceil():
    assert_equal(Float32.__ceil__(Float32(1.5)), 2.0)
    assert_equal(Float32.__ceil__(Float32(-1.5)), -1.0)
    assert_equal(Float32.__ceil__(Float32(3.0)), 3.0)

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
        vec._shuffle_list[7, 6, 5, 4, 3, 2, 1, 0, output_size = 2 * width](vec),
        SIMD[dtype, 2 * width](103, 102, 101, 100, 103, 102, 101, 100),
    )

    assert_equal(
        vec.shuffle[StaticIntTuple[width](3, 2, 1, 0)](),
        SIMD[dtype, width](103, 102, 101, 100),
    )
    assert_equal(
        vec.shuffle[StaticIntTuple[width](0, 2, 4, 6)](vec),
        SIMD[dtype, width](100, 102, 100, 102),
    )

    assert_equal(
        vec._shuffle_list[
            2 * width, StaticIntTuple[2 * width](7, 6, 5, 4, 3, 2, 1, 0)
        ](vec),
        SIMD[dtype, 2 * width](103, 102, 101, 100, 103, 102, 101, 100),
    )


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
    vec = SIMD[DType.int32, 4](100, 101, 102, 103)

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
    assert_equal(tup4[0], SIMD[DType.index, 2](0, 2))
    assert_equal(tup4[1], SIMD[DType.index, 2](1, 3))


def test_address():
    assert_equal(Scalar[DType.address](1), 1)
    assert_not_equal(Scalar[DType.address](1), 0)

    assert_true(Bool(Scalar[DType.address](12) > 1))
    assert_true(Bool(Scalar[DType.address](1) < 12))


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


def test_add_with_overflow():
    var value_u8: UInt8
    var overflowed_u8: Scalar[DType.bool]
    value_u8, overflowed_u8 = UInt8(UInt8.MAX).add_with_overflow(1)
    assert_equal(value_u8, UInt8.MIN)
    assert_equal(overflowed_u8, True)

    var value_u8x4: SIMD[DType.uint8, 4]
    var overflowed_u8x4: SIMD[DType.bool, 4]
    value_u8x4, overflowed_u8x4 = SIMD[DType.uint8, 4](
        1, UInt8.MAX, 1, UInt8.MAX
    ).add_with_overflow(SIMD[DType.uint8, 4](0, 1, 0, 1))
    assert_equal(value_u8x4, SIMD[DType.uint8, 4](1, UInt8.MIN, 1, UInt8.MIN))
    assert_equal(overflowed_u8x4, SIMD[DType.bool, 4](False, True, False, True))

    var value_i8: Int8
    var overflowed_i8: Scalar[DType.bool]
    value_i8, overflowed_i8 = Int8(Int8.MAX).add_with_overflow(1)
    assert_equal(value_i8, Int8.MIN)
    assert_equal(overflowed_i8, True)

    var value_i8x4: SIMD[DType.int8, 4]
    var overflowed_i8x4: SIMD[DType.bool, 4]
    value_i8x4, overflowed_i8x4 = SIMD[DType.int8, 4](
        1, Int8.MAX, 1, Int8.MAX
    ).add_with_overflow(SIMD[DType.int8, 4](0, 1, 0, 1))
    assert_equal(value_i8x4, SIMD[DType.int8, 4](1, Int8.MIN, 1, Int8.MIN))
    assert_equal(overflowed_i8x4, SIMD[DType.bool, 4](False, True, False, True))

    var value_u32: UInt32
    var overflowed_u32: Scalar[DType.bool]
    value_u32, overflowed_u32 = UInt32(UInt32.MAX).add_with_overflow(1)
    assert_equal(value_u32, UInt32.MIN)
    assert_equal(overflowed_u32, True)

    var value_u32x4: SIMD[DType.uint32, 4]
    var overflowed_u32x4: SIMD[DType.bool, 4]
    value_u32x4, overflowed_u32x4 = SIMD[DType.uint32, 4](
        1, UInt32.MAX, 1, UInt32.MAX
    ).add_with_overflow(SIMD[DType.uint32, 4](0, 1, 0, 1))
    assert_equal(
        value_u32x4, SIMD[DType.uint32, 4](1, UInt32.MIN, 1, UInt32.MIN)
    )
    assert_equal(
        overflowed_u32x4, SIMD[DType.bool, 4](False, True, False, True)
    )

    var value_i32: Int32
    var overflowed_i32: Scalar[DType.bool]
    value_i32, overflowed_i32 = Int32(Int32.MAX).add_with_overflow(1)
    assert_equal(value_i32, Int32.MIN)
    assert_equal(overflowed_i32, True)

    var value_i32x4: SIMD[DType.int32, 4]
    var overflowed_i32x4: SIMD[DType.bool, 4]
    value_i32x4, overflowed_i32x4 = SIMD[DType.int32, 4](
        1, Int32.MAX, 1, Int32.MAX
    ).add_with_overflow(SIMD[DType.int32, 4](0, 1, 0, 1))
    assert_equal(value_i32x4, SIMD[DType.int32, 4](1, Int32.MIN, 1, Int32.MIN))
    assert_equal(
        overflowed_i32x4, SIMD[DType.bool, 4](False, True, False, True)
    )


def test_sub_with_overflow():
    var value_u8: UInt8
    var overflowed_u8: Scalar[DType.bool]
    value_u8, overflowed_u8 = UInt8(UInt8.MIN).sub_with_overflow(1)
    assert_equal(value_u8, UInt8.MAX)
    assert_equal(overflowed_u8, True)

    var value_u8x4: SIMD[DType.uint8, 4]
    var overflowed_u8x4: SIMD[DType.bool, 4]
    value_u8x4, overflowed_u8x4 = SIMD[DType.uint8, 4](
        1, UInt8.MIN, 1, UInt8.MIN
    ).sub_with_overflow(SIMD[DType.uint8, 4](0, 1, 0, 1))
    assert_equal(value_u8x4, SIMD[DType.uint8, 4](1, UInt8.MAX, 1, UInt8.MAX))
    assert_equal(overflowed_u8x4, SIMD[DType.bool, 4](False, True, False, True))

    var value_i8: Int8
    var overflowed_i8: Scalar[DType.bool]
    value_i8, overflowed_i8 = Int8(Int8.MIN).sub_with_overflow(1)
    assert_equal(value_i8, Int8.MAX)
    assert_equal(overflowed_i8, True)

    var value_i8x4: SIMD[DType.int8, 4]
    var overflowed_i8x4: SIMD[DType.bool, 4]
    value_i8x4, overflowed_i8x4 = SIMD[DType.int8, 4](
        1, Int8.MIN, 1, Int8.MIN
    ).sub_with_overflow(SIMD[DType.int8, 4](0, 1, 0, 1))
    assert_equal(value_i8x4, SIMD[DType.int8, 4](1, Int8.MAX, 1, Int8.MAX))
    assert_equal(overflowed_i8x4, SIMD[DType.bool, 4](False, True, False, True))

    var value_u32: UInt32
    var overflowed_u32: Scalar[DType.bool]
    value_u32, overflowed_u32 = UInt32(UInt32.MIN).sub_with_overflow(1)
    assert_equal(value_u32, UInt32.MAX)
    assert_equal(overflowed_u32, True)

    var value_u32x4: SIMD[DType.uint32, 4]
    var overflowed_u32x4: SIMD[DType.bool, 4]
    value_u32x4, overflowed_u32x4 = SIMD[DType.uint32, 4](
        1, UInt32.MIN, 1, UInt32.MIN
    ).sub_with_overflow(SIMD[DType.uint32, 4](0, 1, 0, 1))
    assert_equal(
        value_u32x4, SIMD[DType.uint32, 4](1, UInt32.MAX, 1, UInt32.MAX)
    )
    assert_equal(
        overflowed_u32x4, SIMD[DType.bool, 4](False, True, False, True)
    )

    var value_i32: Int32
    var overflowed_i32: Scalar[DType.bool]
    value_i32, overflowed_i32 = Int32(Int32.MIN).sub_with_overflow(1)
    assert_equal(value_i32, Int32.MAX)
    assert_equal(overflowed_i32, True)

    var value_i32x4: SIMD[DType.int32, 4]
    var overflowed_i32x4: SIMD[DType.bool, 4]
    value_i32x4, overflowed_i32x4 = SIMD[DType.int32, 4](
        1, Int32.MIN, 1, Int32.MIN
    ).sub_with_overflow(SIMD[DType.int32, 4](0, 1, 0, 1))
    assert_equal(value_i32x4, SIMD[DType.int32, 4](1, Int32.MAX, 1, Int32.MAX))
    assert_equal(
        overflowed_i32x4, SIMD[DType.bool, 4](False, True, False, True)
    )


def test_mul_with_overflow():
    alias uint8_max_x2 = 254
    var value_u8: UInt8
    var overflowed_u8: Scalar[DType.bool]
    value_u8, overflowed_u8 = UInt8(UInt8.MAX).mul_with_overflow(2)
    assert_equal(value_u8, uint8_max_x2)
    assert_equal(overflowed_u8, True)

    var value_u8x4: SIMD[DType.uint8, 4]
    var overflowed_u8x4: SIMD[DType.bool, 4]
    value_u8x4, overflowed_u8x4 = SIMD[DType.uint8, 4](
        1, UInt8.MAX, 1, UInt8.MAX
    ).mul_with_overflow(SIMD[DType.uint8, 4](0, 2, 0, 2))
    assert_equal(
        value_u8x4, SIMD[DType.uint8, 4](0, uint8_max_x2, 0, uint8_max_x2)
    )
    assert_equal(overflowed_u8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias int8_max_x2 = -2
    var value_i8: Int8
    var overflowed_i8: Scalar[DType.bool]
    value_i8, overflowed_i8 = Int8(Int8.MAX).mul_with_overflow(2)
    assert_equal(value_i8, int8_max_x2)
    assert_equal(overflowed_i8, True)

    var value_i8x4: SIMD[DType.int8, 4]
    var overflowed_i8x4: SIMD[DType.bool, 4]
    value_i8x4, overflowed_i8x4 = SIMD[DType.int8, 4](
        1, Int8.MAX, 1, Int8.MAX
    ).mul_with_overflow(SIMD[DType.int8, 4](0, 2, 0, 2))
    assert_equal(
        value_i8x4, SIMD[DType.int8, 4](0, int8_max_x2, 0, int8_max_x2)
    )
    assert_equal(overflowed_i8x4, SIMD[DType.bool, 4](False, True, False, True))

    alias uint32_max_x2 = 4294967294
    var value_u32: UInt32
    var overflowed_u32: Scalar[DType.bool]
    value_u32, overflowed_u32 = UInt32(UInt32.MAX).mul_with_overflow(2)
    assert_equal(value_u32, uint32_max_x2)
    assert_equal(overflowed_u32, True)

    var value_u32x4: SIMD[DType.uint32, 4]
    var overflowed_u32x4: SIMD[DType.bool, 4]
    value_u32x4, overflowed_u32x4 = SIMD[DType.uint32, 4](
        1, UInt32.MAX, 1, UInt32.MAX
    ).mul_with_overflow(SIMD[DType.uint32, 4](0, 2, 0, 2))
    assert_equal(
        value_u32x4, SIMD[DType.uint32, 4](0, uint32_max_x2, 0, uint32_max_x2)
    )
    assert_equal(
        overflowed_u32x4, SIMD[DType.bool, 4](False, True, False, True)
    )

    alias int32_max_x2 = -2
    var value_i32: Int32
    var overflowed_i32: Scalar[DType.bool]
    value_i32, overflowed_i32 = Int32(Int32.MAX).mul_with_overflow(2)
    assert_equal(value_i32, int32_max_x2)
    assert_equal(overflowed_i32, True)

    var value_i32x4: SIMD[DType.int32, 4]
    var overflowed_i32x4: SIMD[DType.bool, 4]
    value_i32x4, overflowed_i32x4 = SIMD[DType.int32, 4](
        1, Int32.MAX, 1, Int32.MAX
    ).mul_with_overflow(SIMD[DType.int32, 4](0, 2, 0, 2))
    assert_equal(
        value_i32x4, SIMD[DType.int32, 4](0, int32_max_x2, 0, int32_max_x2)
    )
    assert_equal(
        overflowed_i32x4, SIMD[DType.bool, 4](False, True, False, True)
    )


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


def test_min_max_clamp():
    alias F = SIMD[DType.float32, 4]

    var f = F(-10.5, -5.0, 5.0, 10.0)
    assert_equal(f.min(F(-9.0, -6.0, -4.0, 10.5)), F(-10.5, -6.0, -4.0, 10.0))
    assert_equal(f.min(-4.0), F(-10.5, -5.0, -4.0, -4.0))
    assert_equal(f.max(F(-9.0, -6.0, -4.0, 10.5)), F(-9.0, -5.0, 5.0, 10.5))
    assert_equal(f.max(-4.0), F(-4.0, -4.0, 5.0, 10.0))
    assert_equal(f.clamp(-6.0, 5.5), F(-6.0, -5.0, 5.0, 5.5))

    alias I = SIMD[DType.float32, 4]
    var i = I(-10, -5, 5, 10)
    assert_equal(i.min(I(-9, -6, -4, 11)), I(-10, -6, -4, 10))
    assert_equal(i.min(-4), I(-10, -5, -4, -4))
    assert_equal(i.max(I(-9, -6, -4, 11)), I(-9, -5, 5, 11))
    assert_equal(i.max(-4), I(-4, -4, 5, 10))
    assert_equal(i.clamp(-7, 4), I(-7, -5, 4, 4))


def test_indexer():
    assert_equal(5, Int8(5).__index__())
    assert_equal(56, UInt32(56).__index__())
    assert_equal(1, Scalar[DType.bool](True).__index__())
    assert_equal(0, Scalar[DType.bool](False).__index__())


def main():
    test_cast()
    test_simd_variadic()
    test_convert_simd_to_string()
    test_issue_20421()
    test_truthy()
    test_len()
    test_add()
    test_radd()
    test_iadd()
    test_ceil()
    test_floor()
    test_trunc()
    test_round()
    test_roundeven()
    test_div()
    test_floordiv()
    test_rfloordiv()
    test_mod()
    test_rmod()
    test_rotate()
    test_shift()
    test_shuffle()
    test_insert()
    test_join()
    test_interleave()
    test_deinterleave()
    test_address()
    test_extract()
    test_limits()
    test_add_with_overflow()
    test_sub_with_overflow()
    test_mul_with_overflow()
    test_abs()
    test_min_max_clamp()
    test_indexer()
