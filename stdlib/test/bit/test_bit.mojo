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

from bit import (
    rotate_bits_left,
    rotate_bits_right,
    bit_width,
    bit_ceil,
    bit_floor,
    is_power_of_two,
    countl_zero,
    countr_zero,
    bit_reverse,
    byte_swap,
    pop_count,
    bit_not,
)

from testing import assert_equal




def test_countl_zero():
    assert_equal(countl_zero(-1), 0)
    assert_equal(countl_zero(0), 64)
    assert_equal(countl_zero(1), 63)
    assert_equal(countl_zero(2), 62)
    assert_equal(countl_zero(3), 62)
    assert_equal(countl_zero(4), 61)


def test_countl_zero_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-1, 0, 1, 2)
    assert_equal(countl_zero(var1), SIMD[int8_t, simd_width](0, 8, 7, 6))

    alias var2 = SIMD[int8_t, simd_width](3, 4, 5, 8)
    assert_equal(countl_zero(var2), SIMD[int8_t, simd_width](6, 5, 5, 4))

    alias var3 = SIMD[int16_t, simd_width](-1, 0, 1, 2)
    assert_equal(countl_zero(var3), SIMD[int16_t, simd_width](0, 16, 15, 14))

    alias var4 = SIMD[int16_t, simd_width](3, 4, 5, 8)
    assert_equal(countl_zero(var4), SIMD[int16_t, simd_width](14, 13, 13, 12))

    alias var5 = SIMD[int32_t, simd_width](-1, 0, 1, 2)
    assert_equal(countl_zero(var5), SIMD[int32_t, simd_width](0, 32, 31, 30))

    alias var6 = SIMD[int32_t, simd_width](3, 4, 5, 8)
    assert_equal(countl_zero(var6), SIMD[int32_t, simd_width](30, 29, 29, 28))

    alias var7 = SIMD[int64_t, simd_width](-1, 0, 1, 2)
    assert_equal(countl_zero(var7), SIMD[int64_t, simd_width](0, 64, 63, 62))

    alias var8 = SIMD[int64_t, simd_width](3, 4, 5, 8)
    assert_equal(countl_zero(var8), SIMD[int64_t, simd_width](62, 61, 61, 60))


def test_countr_zero():
    assert_equal(countr_zero(-1), 0)
    assert_equal(countr_zero(0), 64)
    assert_equal(countr_zero(1), 0)
    assert_equal(countr_zero(2), 1)
    assert_equal(countr_zero(3), 0)
    assert_equal(countr_zero(4), 2)


def test_countr_zero_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-1, 0, 1, 2)
    assert_equal(countr_zero(var1), SIMD[int8_t, simd_width](0, 8, 0, 1))

    alias var2 = SIMD[int8_t, simd_width](3, 4, 5, 8)
    assert_equal(countr_zero(var2), SIMD[int8_t, simd_width](0, 2, 0, 3))

    alias var3 = SIMD[int16_t, simd_width](-1, 0, 1, 2)
    assert_equal(countr_zero(var3), SIMD[int16_t, simd_width](0, 16, 0, 1))

    alias var4 = SIMD[int16_t, simd_width](3, 4, 5, 8)
    assert_equal(countr_zero(var4), SIMD[int16_t, simd_width](0, 2, 0, 3))

    alias var5 = SIMD[int32_t, simd_width](-1, 0, 1, 2)
    assert_equal(countr_zero(var5), SIMD[int32_t, simd_width](0, 32, 0, 1))

    alias var6 = SIMD[int32_t, simd_width](3, 4, 5, 8)
    assert_equal(countr_zero(var6), SIMD[int32_t, simd_width](0, 2, 0, 3))

    alias var7 = SIMD[int64_t, simd_width](-1, 0, 1, 2)
    assert_equal(countr_zero(var7), SIMD[int64_t, simd_width](0, 64, 0, 1))

    alias var8 = SIMD[int64_t, simd_width](3, 4, 5, 8)
    assert_equal(countr_zero(var8), SIMD[int64_t, simd_width](0, 2, 0, 3))


def test_bit_reverse_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-1, 0, 1, 2)
    assert_equal(bit_reverse(var1), SIMD[int8_t, simd_width](-1, 0, -128, 64))

    alias var2 = SIMD[int16_t, simd_width](-1, 0, 1, 2)
    assert_equal(
        bit_reverse(var2), SIMD[int16_t, simd_width](-1, 0, -32768, 16384)
    )

    alias var3 = SIMD[int32_t, simd_width](-1, 0, 1, 2)
    assert_equal(
        bit_reverse(var3),
        SIMD[int32_t, simd_width](-1, 0, -2147483648, 1073741824),
    )

    alias var4 = SIMD[int64_t, simd_width](-1, 0, 1, 2)
    assert_equal(
        bit_reverse(var4),
        SIMD[int64_t, simd_width](
            -1, 0, -9223372036854775808, 4611686018427387904
        ),
    )


def test_byte_reverse_simd():
    alias simd_width = 4
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var2 = SIMD[int16_t, simd_width](-1, 0, 1, 2)
    assert_equal(byte_swap(var2), SIMD[int16_t, simd_width](-1, 0, 256, 512))

    alias var3 = SIMD[int32_t, simd_width](-1, 0, 1, 2)
    assert_equal(
        byte_swap(var3), SIMD[int32_t, simd_width](-1, 0, 16777216, 33554432)
    )

    alias var4 = SIMD[int64_t, simd_width](-1, 0, 1, 2)
    assert_equal(
        byte_swap(var4),
        SIMD[int64_t, simd_width](-1, 0, 72057594037927936, 144115188075855872),
    )


def test_pop_count_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-1, 0, 27, 8)
    assert_equal(pop_count(var1), SIMD[int8_t, simd_width](8, 0, 4, 1))

    alias var2 = SIMD[int16_t, simd_width](-1, 0, 27, 8)
    assert_equal(pop_count(var2), SIMD[int16_t, simd_width](16, 0, 4, 1))

    alias var3 = SIMD[int32_t, simd_width](-1, 0, 27, 8)
    assert_equal(pop_count(var3), SIMD[int32_t, simd_width](32, 0, 4, 1))

    alias var4 = SIMD[int64_t, simd_width](-1, 0, 27, 8)
    assert_equal(pop_count(var4), SIMD[int64_t, simd_width](64, 0, 4, 1))


def test_bit_not_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-1, 0, 27, 8)
    assert_equal(bit_not(var1), SIMD[int8_t, simd_width](0, -1, -28, -9))

    alias var2 = SIMD[int16_t, simd_width](-1, 0, 27, 8)
    assert_equal(bit_not(var2), SIMD[int16_t, simd_width](0, -1, -28, -9))

    alias var3 = SIMD[int32_t, simd_width](-1, 0, 27, 8)
    assert_equal(bit_not(var3), SIMD[int32_t, simd_width](0, -1, -28, -9))

    alias var4 = SIMD[int64_t, simd_width](-1, 0, 27, 8)
    assert_equal(bit_not(var4), SIMD[int64_t, simd_width](0, -1, -28, -9))


def test_is_power_of_two():
    assert_equal(is_power_of_two(-1), False)
    assert_equal(is_power_of_two(0), False)
    assert_equal(is_power_of_two(1), True)
    assert_equal(is_power_of_two(2), True)
    assert_equal(is_power_of_two(3), False)
    assert_equal(is_power_of_two(4), True)
    assert_equal(is_power_of_two(5), False)


def test_is_power_of_two_simd():
    alias simd_width = 4
    alias type = DType.int8
    alias return_type = DType.bool

    alias var1 = SIMD[type, simd_width](-1, 0, 1, 2)
    assert_equal(
        is_power_of_two(var1),
        SIMD[DType.bool, simd_width](False, False, True, True),
    )

    alias var2 = SIMD[type, simd_width](3, 4, 5, 8)
    assert_equal(
        is_power_of_two(var2),
        SIMD[DType.bool, simd_width](False, True, False, True),
    )


def test_bit_width():
    assert_equal(bit_width(-2), 1)
    assert_equal(bit_width(-1), 0)
    assert_equal(bit_width(1), 1)
    assert_equal(bit_width(2), 2)
    assert_equal(bit_width(4), 3)
    assert_equal(bit_width(5), 3)


def test_bit_width_simd():
    alias simd_width = 4
    alias type = DType.int8

    alias var1 = SIMD[type, simd_width](-2, -1, 3, 4)
    assert_equal(bit_width(var1), SIMD[type, simd_width](1, 0, 2, 3))

    alias var2 = SIMD[type, simd_width](1, 2, 3, 4)
    assert_equal(bit_width(var2), SIMD[type, simd_width](1, 2, 2, 3))


def test_bit_ceil():
    assert_equal(bit_ceil(-2), 1)
    assert_equal(bit_ceil(1), 1)
    assert_equal(bit_ceil(2), 2)
    assert_equal(bit_ceil(4), 4)
    assert_equal(bit_ceil(5), 8)


def test_bit_ceil_simd():
    alias simd_width = 4
    alias type = DType.int8

    alias var1 = SIMD[type, simd_width](-2, -1, 3, 4)
    assert_equal(bit_ceil(var1), SIMD[type, simd_width](1, 1, 4, 4))

    alias var2 = SIMD[type, simd_width](1, 2, 3, 4)
    assert_equal(bit_ceil(var2), SIMD[type, simd_width](1, 2, 4, 4))


def test_bit_floor():
    assert_equal(bit_floor(-2), 0)
    assert_equal(bit_floor(1), 1)
    assert_equal(bit_floor(2), 2)
    assert_equal(bit_floor(4), 4)
    assert_equal(bit_floor(5), 4)


def test_bit_floor_simd():
    alias simd_width = 4
    alias type = DType.int8

    alias var1 = SIMD[type, simd_width](-1, -2, 3, 4)
    assert_equal(bit_floor(var1), SIMD[type, simd_width](0, 0, 2, 4))

    alias var2 = SIMD[type, simd_width](4, 5, 6, 7)
    assert_equal(bit_floor(var2), SIMD[type, simd_width](4, 4, 4, 4))


def test_rotate_bits_int():
    assert_equal(rotate_bits_left[0](104), 104)
    assert_equal(rotate_bits_left[2](104), 416)
    assert_equal(rotate_bits_left[-2](104), 26)

    assert_equal(rotate_bits_right[0](104), 104)
    assert_equal(rotate_bits_right[2](104), 26)
    assert_equal(rotate_bits_right[-2](104), 416)


def test_rotate_bits_simd():
    alias simd_width = 1
    alias type = DType.uint8

    assert_equal(rotate_bits_left[0](UInt64(104)), 104)
    assert_equal(rotate_bits_left[0](SIMD[type, simd_width](104)), 104)
    assert_equal(
        rotate_bits_left[2](SIMD[type, 2](104)), SIMD[type, 2](161, 161)
    )

    assert_equal(rotate_bits_left[2](Scalar[type](104)), 161)
    assert_equal(rotate_bits_left[11](Scalar[type](15)), 120)
    assert_equal(rotate_bits_left[0](Scalar[type](96)), 96)
    assert_equal(rotate_bits_left[1](Scalar[type](96)), 192)
    assert_equal(rotate_bits_left[2](Scalar[type](96)), 129)
    assert_equal(rotate_bits_left[3](Scalar[type](96)), 3)
    assert_equal(rotate_bits_left[4](Scalar[type](96)), 6)
    assert_equal(rotate_bits_left[5](Scalar[type](96)), 12)

    assert_equal(rotate_bits_right[0](UInt64(104)), 104)
    assert_equal(rotate_bits_right[0](SIMD[type, simd_width](104)), 104)
    assert_equal(
        rotate_bits_right[2](SIMD[type, 2](104)), SIMD[type, 2](26, 26)
    )

    assert_equal(rotate_bits_right[2](Scalar[type](104)), 26)
    assert_equal(rotate_bits_right[11](Scalar[type](15)), 225)
    assert_equal(rotate_bits_right[0](Scalar[type](96)), 96)
    assert_equal(rotate_bits_right[1](Scalar[type](96)), 48)
    assert_equal(rotate_bits_right[2](Scalar[type](96)), 24)
    assert_equal(rotate_bits_right[3](Scalar[type](96)), 12)
    assert_equal(rotate_bits_right[4](Scalar[type](96)), 6)
    assert_equal(rotate_bits_right[5](Scalar[type](96)), 3)
    assert_equal(rotate_bits_right[6](Scalar[type](96)), 129)


def main():
    test_rotate_bits_int()
    test_rotate_bits_simd()
    test_bit_ceil()
    test_bit_ceil_simd()
    test_bit_floor()
    test_bit_floor_simd()
    test_bit_width()
    test_bit_width_simd()
    test_is_power_of_two()
    test_is_power_of_two_simd()
    test_countl_zero()
    test_countl_zero_simd()
    test_countr_zero()
    test_countr_zero_simd()
    test_bit_reverse_simd()
    test_byte_reverse_simd()
    test_pop_count_simd()
    test_bit_not_simd()
