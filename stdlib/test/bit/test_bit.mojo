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
    has_single_bit,
)

from testing import assert_equal


def test_has_single_bit():
    assert_equal(has_single_bit(-1), False)
    assert_equal(has_single_bit(0), False)
    assert_equal(has_single_bit(1), True)
    assert_equal(has_single_bit(2), True)
    assert_equal(has_single_bit(3), False)
    assert_equal(has_single_bit(4), True)
    assert_equal(has_single_bit(5), False)


def test_has_single_bit_simd():
    alias simd_width = 4
    alias type = DType.int8
    alias return_type = DType.bool

    alias var1 = SIMD[type, simd_width](-1, 0, 1, 2)
    assert_equal(
        has_single_bit(var1),
        SIMD[return_type, simd_width](False, False, True, True),
    )

    alias var2 = SIMD[type, simd_width](3, 4, 5, 8)
    assert_equal(
        has_single_bit(var2),
        SIMD[return_type, simd_width](False, True, False, True),
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
    test_has_single_bit()
    test_has_single_bit_simd()
