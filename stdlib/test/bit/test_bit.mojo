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

from bit import rotate_bits_left, rotate_bits_right

from testing import assert_equal


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
