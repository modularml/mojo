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
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import has_neon, simdwidthof

from testing import assert_equal, assert_not_equal, assert_true


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


# CHECK-LABEL: test_simd_variadic
fn test_simd_variadic():
    print("== test_simd_variadic")

    # CHECK: [52, 12, 43, 5]
    print(SIMD[DType.index, 4](52, 12, 43, 5))


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


# CHECK-LABEL: test_insert
fn test_insert():
    print("== test_insert")

    # CHECK: 4
    print(Int32(3).insert(Int32(4)))

    # CHECK: [9, 6, 2, 3]
    print(SIMD[DType.index, 4](0, 1, 2, 3).insert(SIMD[DType.index, 2](9, 6)))

    # CHECK: [0, 9, 6, 3]
    print(
        SIMD[DType.index, 4](0, 1, 2, 3).insert[offset=1](
            SIMD[DType.index, 2](9, 6)
        )
    )

    # CHECK: [0, 1, 2, 3, 9, 6, 3, 7]
    print(
        SIMD[DType.index, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=4](
            SIMD[DType.index, 4](9, 6, 3, 7)
        )
    )

    # CHECK: [0, 1, 2, 9, 6, 3, 7, 8]
    print(
        SIMD[DType.index, 8](0, 1, 2, 3, 5, 6, 7, 8).insert[offset=3](
            SIMD[DType.index, 4](9, 6, 3, 7)
        )
    )


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


def main():
    test_cast()
    test_simd_variadic()
    test_truthy()
    test_floordiv()
    test_mod()
    test_rotate()
    test_shift()
    test_insert()
    test_interleave()
    test_deinterleave()
    test_address()
    test_extract()
