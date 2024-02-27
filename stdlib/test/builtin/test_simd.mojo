# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota, pow
from sys.info import has_neon, simdwidthof

from testing import *


# CHECK-LABEL: test_simd
def test_simd():
    print("== test_simd")

    # CHECK: True
    print(SIMD[DType.index]().__len__() == simdwidthof[DType.index]())

    # CHECK: 4
    print(SIMD[DType.index, 4]().__len__())

    # CHECK: [0, 0, 0, 0]
    print(SIMD[DType.index, 4]())

    # CHECK: [1, 1, 1, 1]
    print(SIMD[DType.index, 4](1))

    # CHECK: [1.{{0+}}, 1.{{0+}}]
    print(SIMD[DType.float16, 2](True))

    var simd_val = iota[DType.index, 4]()

    # CHECK: [0, 1, 2, 3]
    print(simd_val)

    # CHECK: [1, 2, 3, 4]
    print(simd_val + 1)

    # CHECK: [0, 2, 4, 6]
    print(simd_val * 2)

    # CHECK: [1, 2, 3, 4]
    print(1 + simd_val)

    # CHECK: [0, 2, 4, 6]
    print(2 * simd_val)

    # CHECK: [0, 4, 8, 12]
    print(simd_val << 2)

    # CHECK: [0, 1, 4, 9]
    print(simd_val**2)

    # CHECK: [0, 1, 8, 27]
    print(simd_val**3)

    # CHECK: 3
    print(simd_val.reduce_max())

    # CHECK: 6
    print(simd_val.reduce_add())

    # Check: True
    print((simd_val > 2).reduce_or())

    # Check: False
    print((simd_val > 3).reduce_or())

    # Check: 3
    print(int(Float32(3.0)))

    # Check: -4
    print(int(Float32(-3.5)))

    # CHECK: [16, 20]
    print(iota[DType.index, 8](1).reduce_add[2]())

    # CHECK: [105, 384]
    print(iota[DType.index, 8](1).reduce_mul[2]())

    # CHECK: [1, 2]
    print(iota[DType.index, 8](1).reduce_min[2]())

    # CHECK: [7, 8]
    print(iota[DType.index, 8](1).reduce_max[2]())

    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True)
        * SIMD[DType.bool, 4](False, True, True, False),
        SIMD[DType.bool, 4](False, True, False, True)
        & SIMD[DType.bool, 4](False, True, True, False),
    )

    assert_equal(int(Float64(0.25)), 0)
    assert_equal(int(Float64(-0.25)), 0)
    assert_equal(int(Float64(1.25)), 1)
    assert_equal(int(Float64(-1.25)), -1)
    assert_equal(int(Float64(-390.8)), -390)


# CHECK-LABEL: test_cast
def test_cast():
    print("== test_cast")

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


# CHECK-LABEL: test_iota
fn test_iota():
    print("== test_iota")

    # CHECK: [0, 1, 2, 3]
    print(iota[DType.index, 4]())

    # CHECK: 0
    print(iota[DType.index, 1]())


# CHECK-LABEL: test_slice
fn test_slice():
    print("== test_slice")

    var val = iota[DType.index, 4]()

    # CHECK: [0, 1]
    print(val.slice[2]())

    # CHECK: [2, 3]
    print(val.slice[2](2))

    var s2 = iota[DType.int32, 2](0)

    # CHECK: 0
    print(s2.slice[1](0))


# CHECK-LABEL: test_pow
fn test_pow():
    print("== test_pow")

    alias simd_width = 4

    var simd_val = iota[DType.float32, simd_width]()

    # CHECK: [0.0, 1.0, 4.0, 9.0]
    print(pow[DType.float32, DType.float32, simd_width](simd_val, 2.0))

    # CHECK: [inf, 1.0, 0.5, 0.3333333432674408]
    print(pow(simd_val, -1))

    # CHECK: [0.0, 1.0, 1.41421{{[0-9]+}}, 1.73205{{[0-9]+}}]
    print(pow[DType.float32, DType.float32, simd_width](simd_val, 0.5))

    # CHECK: [0.70710{{[0-9]+}}, 0.57735{{[0-9]+}}, 0.5, 0.44721{{[0-9]+}}]
    print(pow[DType.float32, DType.float32, simd_width](simd_val + 2, -0.5))

    # CHECK: [0.0, 1.0, 4.0, 9.0]
    print(pow(simd_val, SIMD[DType.int32, simd_width](2)))

    # CHECK: [0.0, 1.0, 8.0, 27.0]
    print(pow(simd_val, SIMD[DType.int32, simd_width](3)))

    var simd_val_int = iota[DType.int32, simd_width]()

    # CHECK: [0, 1, 4, 9]
    print(pow(simd_val_int, 2))


# CHECK-LABEL: test_simd_variadic
fn test_simd_variadic():
    print("== test_simd_variadic")

    # CHECK: [52, 12, 43, 5]
    print(SIMD[DType.index, 4](52, 12, 43, 5))


# CHECK-LABEL: test_simd_bool
fn test_simd_bool():
    print("== test_simd_bool")

    var v0 = iota[DType.index, 4]()

    # CHECK: [False, True, False, False]
    print((v0 > 0) & (v0 < 2))

    # CHECK: [True, False, False, True]
    print((v0 > 2) | (v0 < 1))


# CHECK-LABEL: test_truthy
def test_truthy():
    print("== test_truthy")

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
        assert_equal(False, Scalar[type](False).__bool__())
        assert_equal(True, Scalar[type](True).__bool__())

        # # SIMD vectors are truth-y if _all_ values are truth-y
        assert_equal(True, SIMD[type, 2](True, True).__bool__())

        # # SIMD vectors are false-y if _any_ values are false-y
        assert_equal(False, SIMD[type, 2](False, True).__bool__())
        assert_equal(False, SIMD[type, 2](True, False).__bool__())
        assert_equal(False, SIMD[type, 2](False, False).__bool__())

    @parameter
    fn test_dtype_unrolled[i: Int]() raises:
        alias type = dtypes.get[i, DType]()
        test_dtype[type]()

    unroll[test_dtype_unrolled, dtypes.__len__()]()

    @parameter
    if not has_neon():
        # TODO bfloat16 is not supported on neon #30525
        test_dtype[DType.bfloat16]()


# CHECK-LABEL: test_floordiv
def test_floordiv():
    print("== test_floordiv")

    assert_equal(Int32(2) // Int32(2), 1)
    assert_equal(Int32(2) // Int32(3), 0)
    assert_equal(Int32(2) // Int32(-2), -1)
    assert_equal(Int32(99) // Int32(-2), -50)

    assert_equal(UInt32(2) // UInt32(2), 1)
    assert_equal(UInt32(2) // UInt32(3), 0)

    assert_equal(Float32(2) // Float32(2), 1)
    assert_equal(Float32(2) // Float32(3), 0)
    assert_equal(Float32(2) // Float32(-2), -1)
    assert_equal(Float32(99) // Float32(-2), -50)


# CHECK-LABEL: test_mod
def test_mod():
    print("== test_mod")

    assert_equal(Int32(99) % Int32(1), 0)
    assert_equal(Int32(99) % Int32(3), 0)
    assert_equal(Int32(99) % Int32(-2), -1)
    assert_equal(Int32(99) % Int32(8), 3)
    assert_equal(Int32(99) % Int32(-8), -5)
    assert_equal(Int32(2) % Int32(-1), 0)
    assert_equal(Int32(2) % Int32(-2), 0)

    assert_equal(UInt32(99) % UInt32(1), 0)
    assert_equal(UInt32(99) % UInt32(3), 0)

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


# CHECK-LABEL: test_rotate
fn test_rotate():
    print("== test_rotate")

    alias simd_width = 4
    alias type = DType.uint32

    # CHECK: [0, 1, 0, 1, 1, 0, 1, 0]
    print(SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[1]())
    # CHECK: [1, 0, 1, 0, 0, 1, 0, 1]
    print(SIMD[DType.uint32, 8](1, 0, 1, 1, 0, 1, 0, 0).rotate_right[5]())

    # CHECK: [1, 0, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[0]())
    # CHECK: [0, 1, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[1]())
    # CHECK: [1, 1, 1, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[2]())
    # CHECK: [1, 1, 0, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[3]())
    # CHECK: [1, 1, 0, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-1]())
    # CHECK: [1, 1, 1, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-2]())
    # CHECK: [0, 1, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-3]())
    # CHECK: [1, 0, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_left[-4]())

    # CHECK: [1, 0, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[0]())
    # CHECK: [1, 1, 0, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[1]())
    # CHECK: [1, 1, 1, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[2]())
    # CHECK: [0, 1, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[3]())
    # CHECK: [1, 0, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[4]())
    # CHECK: [0, 1, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-1]())
    # CHECK: [1, 1, 1, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-2]())
    # CHECK: [1, 1, 0, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).rotate_right[-3]())


# CHECK-LABEL: test_shift
fn test_shift():
    print("== test_shift")

    alias simd_width = 4
    alias type = DType.uint32

    # CHECK: [0, 1, 0, 1, 1, 0, 1, 0]
    print(SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0).shift_right[1]())
    # CHECK: [0, 0, 0, 0, 0, 11, 0, 13]
    print(SIMD[DType.uint32, 8](11, 0, 13, 12, 0, 100, 0, 0).shift_right[5]())

    # CHECK: [0.0, 0.0, 0.0, 0.0, 0.0, 11.1, 0.0, 13.1]
    print(
        SIMD[DType.float64, 8](11.1, 0, 13.1, 12.2, 0, 100.4, 0, 0).shift_right[
            5
        ]()
    )

    # CHECK: [1, 0, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_left[0]())
    # CHECK: [0, 1, 1, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_left[1]())
    # CHECK: [1, 1, 0, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_left[2]())
    # CHECK: [1, 0, 0, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_left[3]())
    # CHECK: [0, 0, 0, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_left[4]())

    # CHECK: [1, 0, 1, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_right[0]())
    # CHECK: [0, 1, 0, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_right[1]())
    # CHECK: [0, 0, 1, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_right[2]())
    # CHECK: [0, 0, 0, 1]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_right[3]())
    # CHECK: [0, 0, 0, 0]
    print(SIMD[type, simd_width](1, 0, 1, 1).shift_right[4]())


# CHECK-LABEL: test_join
fn test_join():
    print("== test_join")

    # CHECK: [3, 4]
    print(Int32(3).join(Int32(4)))

    var s0 = SIMD[DType.index, 4](0, 1, 2, 3)
    var s1 = SIMD[DType.index, 4](5, 6, 7, 8)

    # CHECK: [0, 1, 2, 3, 5, 6, 7, 8]
    print(s0.join(s1))

    var s2 = SIMD[DType.index, 2](5, 6)
    var s3 = SIMD[DType.index, 2](9, 10)

    # CHECK: [5, 6, 9, 10]
    print(s2.join(s3))

    var s4 = iota[DType.index, 32](1)
    var s5 = iota[DType.index, 32](33)
    # CHECK: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    # CHECK: 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    # CHECK: 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    # CHECK: 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    # CHECK: 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    # CHECK: 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    print(s4.join(s5))


def test_interleave():
    assert_equal(Int32(0).interleave(Int32(1)), SIMD[DType.index, 2](0, 1))

    assert_equal(
        SIMD[DType.index, 2](0, 2).interleave(SIMD[DType.index, 2](1, 3)),
        SIMD[DType.index, 4](0, 1, 2, 3),
    )


def test_deinterleave():
    var ts = SIMD[DType.index, 4](0, 1, 2, 3).deinterleave()
    assert_equal(ts[0], SIMD[DType.index, 2](0, 2))
    assert_equal(ts[1], SIMD[DType.index, 2](1, 3))


def test_address():
    assert_equal(Scalar[DType.address](1), 1)
    assert_not_equal(Scalar[DType.address](1), 0)

    assert_true(Bool(Scalar[DType.address](12) > 1))
    assert_true(Bool(Scalar[DType.address](1) < 12))


def test_extract():
    assert_equal(Int64(99).slice[1](0), 99)

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[4](),
        SIMD[DType.index, 4](99, 1, 2, 4),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2](0),
        SIMD[DType.index, 2](99, 1),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2](2),
        SIMD[DType.index, 2](2, 4),
    )

    assert_equal(
        SIMD[DType.index, 4](99, 1, 2, 4).slice[2](1),
        SIMD[DType.index, 2](1, 2),
    )


def main():
    test_simd()
    test_cast()
    test_iota()
    test_slice()
    test_pow()
    test_simd_variadic()
    test_simd_bool()
    test_truthy()
    test_floordiv()
    test_mod()
    test_rotate()
    test_shift()
    test_join()
    test_interleave()
    test_deinterleave()
    test_address()
    test_extract()
