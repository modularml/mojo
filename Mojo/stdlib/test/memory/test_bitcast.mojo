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

from std.memory import bitcast, pack_bits
from std.testing import TestSuite
from std.testing import assert_equal


def test_bitcast() raises:
    assert_equal(
        bitcast[.int8, 8](SIMD[.int16, 4](1, 2, 3, 4)),
        SIMD[.int8, 8](1, 0, 2, 0, 3, 0, 4, 0),
    )

    assert_equal(
        bitcast[.int32, 1](SIMD[.int8, 4](0xFF, 0x00, 0xFF, 0x55)),
        Int32(1442775295),
    )


def test_pack_bits() raises:
    comptime b1 = Scalar[.bool](True)
    assert_equal(pack_bits(b1).cast[.bool](), b1)
    assert_equal(pack_bits(b1).cast[.uint8](), UInt8(0b0000_0001))

    comptime b2 = SIMD[.bool, 2](1, 0)
    assert_equal(pack_bits(b2).cast[.uint8](), UInt8(0b0000_0001))

    comptime b4 = SIMD[.bool, 4](1, 1, 0, 1)
    assert_equal(pack_bits(b4).cast[.uint8](), UInt8(0b0000_1011))

    comptime b8 = SIMD[.bool, 8](1, 1, 1, 0, 1, 0, 1, 0)
    assert_equal(pack_bits(b8), UInt8(0b0101_0111))

    comptime b16 = SIMD[.bool, 16](
        1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1
    )
    assert_equal(pack_bits(b16), UInt16(0b1000_1010_0101_0111))
    assert_equal(
        pack_bits[.uint8, 2](b16),
        SIMD[.uint8, 2](0b0101_0111, 0b1000_1010),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
