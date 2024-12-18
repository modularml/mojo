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
# RUN: %bare-mojo %s

from bit import (
    next_power_of_two,
    prev_power_of_two,
    bit_not,
    bit_reverse,
    bit_width,
    byte_swap,
    count_leading_zeros,
    count_trailing_zeros,
    is_power_of_two,
    pop_count,
    rotate_bits_left,
    rotate_bits_right,
    log2_floor,
)
from math import log2, floor
from testing import assert_equal


def test_count_leading_zeros():
    assert_equal(count_leading_zeros(-(2**59)), 0)
    assert_equal(count_leading_zeros(-(2**20)), 0)
    assert_equal(count_leading_zeros(-1), 0)
    assert_equal(count_leading_zeros(-1), 0)
    assert_equal(count_leading_zeros(0), 64)
    assert_equal(count_leading_zeros(1), 63)
    assert_equal(count_leading_zeros(2), 62)
    assert_equal(count_leading_zeros(3), 62)
    assert_equal(count_leading_zeros(4), 61)
    assert_equal(count_leading_zeros(2**20), 43)
    assert_equal(count_leading_zeros(2**59), 4)


def test_count_leading_zeros_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-(2**6), 0, -1, 2**6)
    assert_equal(
        count_leading_zeros(var1), SIMD[int8_t, simd_width](0, 8, 0, 1)
    )

    alias var3 = SIMD[int16_t, simd_width](-(2**14), 0, -1, 2**14)
    assert_equal(
        count_leading_zeros(var3), SIMD[int16_t, simd_width](0, 16, 0, 1)
    )

    alias var5 = SIMD[int32_t, simd_width](-(2**30), 0, -1, 2**30)
    assert_equal(
        count_leading_zeros(var5), SIMD[int32_t, simd_width](0, 32, 0, 1)
    )

    # TODO: use this line after #2882 is fixed
    # alias var7 = SIMD[int64_t, simd_width](-(2**62), 0, -1, 2**62)
    alias var7 = SIMD[int64_t, simd_width](
        -4611686018427387904, 0, -1, 4611686018427387904
    )
    assert_equal(
        count_leading_zeros(var7), SIMD[int64_t, simd_width](0, 64, 0, 1)
    )


def test_count_trailing_zeros():
    assert_equal(count_trailing_zeros(-(2**59)), 59)
    assert_equal(count_trailing_zeros(-(2**20)), 20)
    assert_equal(count_trailing_zeros(-1), 0)
    assert_equal(count_trailing_zeros(0), 64)
    assert_equal(count_trailing_zeros(1), 0)
    assert_equal(count_trailing_zeros(2), 1)
    assert_equal(count_trailing_zeros(3), 0)
    assert_equal(count_trailing_zeros(4), 2)
    assert_equal(count_trailing_zeros(2**20), 20)
    assert_equal(count_trailing_zeros(2**59), 59)


def test_count_trailing_zeros_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-(2**6), 0, -1, 2**6)
    assert_equal(
        count_trailing_zeros(var1), SIMD[int8_t, simd_width](6, 8, 0, 6)
    )

    alias var3 = SIMD[int16_t, simd_width](-(2**14), 0, -1, 2**14)
    assert_equal(
        count_trailing_zeros(var3), SIMD[int16_t, simd_width](14, 16, 0, 14)
    )

    alias var5 = SIMD[int32_t, simd_width](-(2**30), 0, -1, 2**30)
    assert_equal(
        count_trailing_zeros(var5), SIMD[int32_t, simd_width](30, 32, 0, 30)
    )

    # TODO: use this line after #2882 is fixed
    # alias var7 = SIMD[int64_t, simd_width](-(2**62), 0, -1, 2**62)
    alias var7 = SIMD[int64_t, simd_width](
        -4611686018427387904, 0, -1, 4611686018427387904
    )
    assert_equal(
        count_trailing_zeros(var7), SIMD[int64_t, simd_width](62, 64, 0, 62)
    )


def test_bit_reverse():
    assert_equal(bit_reverse(-(2**32)), 4294967295)
    assert_equal(bit_reverse(-1), -1)
    assert_equal(bit_reverse(0), 0)
    assert_equal(bit_reverse(1), -(2**63))
    assert_equal(bit_reverse(2), 2**62)
    assert_equal(bit_reverse(8), 2**60)
    assert_equal(bit_reverse(2**63), 1)


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


def test_byte_swap():
    assert_equal(byte_swap(0x0000), 0x0000000000000000)
    assert_equal(byte_swap(0x0102), 0x0201000000000000)
    assert_equal(byte_swap(0x0201), 0x0102000000000000)
    assert_equal(byte_swap(-0x0123456789ABCDEF), 0x1132547698BADCFE)
    assert_equal(byte_swap(0x0000000001234567), 0x6745230100000000)
    assert_equal(byte_swap(0x56789ABCDEF01234), 0x3412F0DEBC9A7856)
    assert_equal(byte_swap(0x23456789ABCDEF01), 0x01EFCDAB89674523)


def test_byte_swap_simd():
    alias simd_width = 4
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var2 = SIMD[int16_t, simd_width](-0x0123, 0x0000, 0x0102, 0x0201)
    assert_equal(
        byte_swap(var2),
        SIMD[int16_t, simd_width](0xDDFE, 0x0000, 0x0201, 0x0102),
    )

    alias var3 = SIMD[int32_t, simd_width](
        -0x01234567, 0x01234567, 0x56789ABC, 0x89ABCDEF
    )
    assert_equal(
        byte_swap(var3),
        SIMD[int32_t, simd_width](
            0x99BADCFE, 0x67452301, 0xBC9A7856, 0xEFCDAB89
        ),
    )

    alias var4 = SIMD[int64_t, simd_width](
        -0x0123456789ABCDEF,
        0x0123456789ABCDEF,
        0x56789ABCDEF01234,
        0x23456789ABCDEF01,
    )
    assert_equal(
        byte_swap(var4),
        SIMD[int64_t, simd_width](
            0x1132547698BADCFE,
            0xEFCDAB8967452301,
            0x3412F0DEBC9A7856,
            0x01EFCDAB89674523,
        ),
    )


def test_pop_count():
    assert_equal(pop_count(-111444444), 51)
    assert_equal(pop_count(0), 0)
    assert_equal(pop_count(1), 1)
    assert_equal(pop_count(2), 1)
    assert_equal(pop_count(3), 2)
    assert_equal(pop_count(4), 1)
    assert_equal(pop_count(5), 2)
    assert_equal(pop_count(3000000), 10)


def test_pop_count_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-114, 0, 100, 2**6)
    assert_equal(pop_count(var1), SIMD[int8_t, simd_width](4, 0, 3, 1))

    alias var2 = SIMD[int16_t, simd_width](-11444, 0, 3000, 2**13)
    assert_equal(pop_count(var2), SIMD[int16_t, simd_width](8, 0, 7, 1))

    alias var3 = SIMD[int32_t, simd_width](-111444, 0, 30000, 2**29)
    assert_equal(pop_count(var3), SIMD[int32_t, simd_width](22, 0, 7, 1))

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[int64_t, simd_width](-111444444, 0, 3000000, 2**59)
    alias var4 = SIMD[int64_t, simd_width](
        -111444444, 0, 3000000, 576460752303423488
    )
    assert_equal(pop_count(var4), SIMD[int64_t, simd_width](51, 0, 10, 1))


def test_bit_not_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-114, 0, 100, 2**6)
    assert_equal(bit_not(var1), SIMD[int8_t, simd_width](113, -1, -101, -65))

    alias var2 = SIMD[int16_t, simd_width](-11444, 0, 3000, 2**13)
    assert_equal(
        bit_not(var2), SIMD[int16_t, simd_width](11443, -1, -3001, -8193)
    )

    alias var3 = SIMD[int32_t, simd_width](-111444, 0, 30000, 2**29)
    assert_equal(
        bit_not(var3), SIMD[int32_t, simd_width](111443, -1, -30001, -536870913)
    )

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[int64_t, simd_width](-111444444, 0, 3000000, 2**59)
    alias var4 = SIMD[int64_t, simd_width](
        -111444444, 0, 3000000, 576460752303423488
    )
    assert_equal(
        bit_not(var4),
        SIMD[int64_t, simd_width](111444443, -1, -3000001, -(2**59) - 1),
    )


def test_is_power_of_two():
    assert_equal(is_power_of_two(Int.MIN), False)
    assert_equal(is_power_of_two(-(2**59)), False)
    assert_equal(is_power_of_two(-1), False)
    assert_equal(is_power_of_two(0), False)
    assert_equal(is_power_of_two(1), True)
    assert_equal(is_power_of_two(2), True)
    assert_equal(is_power_of_two(3), False)
    assert_equal(is_power_of_two(4), True)
    assert_equal(is_power_of_two(5), False)
    assert_equal(is_power_of_two(2**59), True)
    assert_equal(is_power_of_two(Int.MAX), False)


def test_is_power_of_two_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-114, 0, 100, 2**6)
    assert_equal(
        is_power_of_two(var1),
        SIMD[DType.bool, simd_width](False, False, False, True),
    )

    alias var2 = SIMD[int16_t, simd_width](-11444, 0, 3000, 2**13)
    assert_equal(
        is_power_of_two(var2),
        SIMD[DType.bool, simd_width](False, False, False, True),
    )

    alias var3 = SIMD[int32_t, simd_width](-111444, 0, 30000, 2**29)
    assert_equal(
        is_power_of_two(var3),
        SIMD[DType.bool, simd_width](False, False, False, True),
    )

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[int64_t, simd_width](-111444444, 0, 3000000, 2**59)
    alias var4 = SIMD[int64_t, simd_width](
        -111444444, 0, 3000000, 576460752303423488
    )
    assert_equal(
        is_power_of_two(var4),
        SIMD[DType.bool, simd_width](False, False, False, True),
    )

    assert_equal(is_power_of_two(Int64.MIN), False)


def test_bit_width():
    assert_equal(bit_width(-(2**59)), 59)
    assert_equal(bit_width(-2), 1)
    assert_equal(bit_width(-1), 0)
    assert_equal(bit_width(0), 0)
    assert_equal(bit_width(1), 1)
    assert_equal(bit_width(2), 2)
    assert_equal(bit_width(4), 3)
    assert_equal(bit_width(5), 3)
    assert_equal(bit_width(2**59), 60)


def test_bit_width_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-114, 0, 100, 2**6)
    assert_equal(bit_width(var1), SIMD[int8_t, simd_width](7, 0, 7, 7))

    alias var2 = SIMD[int16_t, simd_width](-11444, 0, 3000, 2**13)
    assert_equal(bit_width(var2), SIMD[int16_t, simd_width](14, 0, 12, 14))

    alias var3 = SIMD[int32_t, simd_width](-111444, 0, 30000, 2**29)
    assert_equal(bit_width(var3), SIMD[int32_t, simd_width](17, 0, 15, 30))

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[int64_t, simd_width](-111444444, 0, 3000000, 2**59)
    alias var4 = SIMD[int64_t, simd_width](
        -111444444, 0, 3000000, 576460752303423488
    )
    assert_equal(bit_width(var4), SIMD[int64_t, simd_width](27, 0, 22, 60))


def test_next_power_of_two():
    assert_equal(next_power_of_two(-(2**59)), 1)
    assert_equal(next_power_of_two(-2), 1)
    assert_equal(next_power_of_two(1), 1)
    assert_equal(next_power_of_two(2), 2)
    assert_equal(next_power_of_two(4), 4)
    assert_equal(next_power_of_two(5), 8)
    assert_equal(next_power_of_two(2**59 - 3), 2**59)


def test_next_power_of_two_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-114, 0, 2**7 - 3, 2**6)
    assert_equal(
        next_power_of_two(var1), SIMD[int8_t, simd_width](1, 1, 2**7, 2**6)
    )

    alias var2 = SIMD[int16_t, simd_width](-11444, 0, 2**12 - 3, 2**13)
    assert_equal(
        next_power_of_two(var2),
        SIMD[int16_t, simd_width](1, 1, 2**12, 2**13),
    )

    alias var3 = SIMD[int32_t, simd_width](-111444, 0, 2**14 - 3, 2**29)
    assert_equal(
        next_power_of_two(var3),
        SIMD[int32_t, simd_width](1, 1, 2**14, 2**29),
    )

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[int64_t, simd_width](-111444444, 1, 2**22-3, 2**59)
    alias var4 = SIMD[int64_t, simd_width](
        -111444444, 1, 2**22 - 3, 576460752303423488
    )
    assert_equal(
        next_power_of_two(var4),
        SIMD[int64_t, simd_width](1, 1, 2**22, 2**59),
    )


def test_prev_power_of_two():
    assert_equal(prev_power_of_two(-(2**59)), 0)
    assert_equal(prev_power_of_two(-2), 0)
    assert_equal(prev_power_of_two(1), 1)
    assert_equal(prev_power_of_two(2), 2)
    assert_equal(prev_power_of_two(4), 4)
    assert_equal(prev_power_of_two(5), 4)
    assert_equal(prev_power_of_two(2**59), 2**59)


def test_prev_power_of_two_simd():
    alias simd_width = 4
    alias int8_t = DType.int8
    alias int16_t = DType.int16
    alias int32_t = DType.int32
    alias int64_t = DType.int64

    alias var1 = SIMD[int8_t, simd_width](-114, 0, 2**5 + 3, 2**6)
    assert_equal(
        prev_power_of_two(var1), SIMD[int8_t, simd_width](0, 0, 2**5, 2**6)
    )

    alias var2 = SIMD[int16_t, simd_width](-11444, 0, 2**12 + 3, 2**13)
    assert_equal(
        prev_power_of_two(var2),
        SIMD[int16_t, simd_width](0, 0, 2**12, 2**13),
    )

    alias var3 = SIMD[int32_t, simd_width](-111444, 0, 2**14 + 3, 2**29)
    assert_equal(
        prev_power_of_two(var3),
        SIMD[int32_t, simd_width](0, 0, 2**14, 2**29),
    )

    # TODO: use this line after #2882 is fixed
    # alias var4 = SIMD[int64_t, simd_width](-111444444, 1, 2**22+3, 2**59)
    alias var4 = SIMD[int64_t, simd_width](
        -111444444, 1, 2**22 + 3, 576460752303423488
    )
    assert_equal(
        prev_power_of_two(var4),
        SIMD[int64_t, simd_width](0, 1, 2**22, 2**59),
    )


def test_rotate_bits_int():
    assert_equal(rotate_bits_left[0](104), 104)
    assert_equal(rotate_bits_left[2](104), 416)
    assert_equal(rotate_bits_left[-2](104), 26)
    assert_equal(rotate_bits_left[8](104), 26624)
    assert_equal(rotate_bits_left[-8](104), 7493989779944505344)

    assert_equal(rotate_bits_right[0](104), 104)
    assert_equal(rotate_bits_right[2](104), 26)
    assert_equal(rotate_bits_right[-2](104), 416)
    assert_equal(rotate_bits_right[8](104), 7493989779944505344)
    assert_equal(rotate_bits_right[-8](104), 26624)


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


fn _log2_floor(n: Int) -> Int:
    return int(floor(log2(float(n))))


def test_log2_floor():
    assert_equal(log2_floor(0), 0)
    for i in range(1, 100):
        assert_equal(
            log2_floor(i),
            _log2_floor(i),
            msg="mismatching value for the input value of " + str(i),
        )

    fn _check_alias[n: Int](expected: Int) raises:
        alias res = log2_floor(n)
        assert_equal(res, expected)

    _check_alias[0](0)
    _check_alias[1](0)
    _check_alias[2](1)
    _check_alias[15](3)
    _check_alias[32](5)


def main():
    test_rotate_bits_int()
    test_rotate_bits_simd()
    test_next_power_of_two()
    test_next_power_of_two_simd()
    test_prev_power_of_two()
    test_prev_power_of_two_simd()
    test_bit_width()
    test_bit_width_simd()
    test_is_power_of_two()
    test_is_power_of_two_simd()
    test_count_leading_zeros()
    test_count_leading_zeros_simd()
    test_count_trailing_zeros()
    test_count_trailing_zeros_simd()
    test_bit_reverse()
    test_bit_reverse_simd()
    test_byte_swap()
    test_byte_swap_simd()
    test_pop_count()
    test_pop_count_simd()
    test_bit_not_simd()
    test_log2_floor()
