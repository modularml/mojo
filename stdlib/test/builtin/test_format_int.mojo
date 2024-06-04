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

from builtin.format_int import _format_int
from testing import assert_equal


fn test_format_int() raises:
    assert_equal(_format_int[DType.index](123), "123")
    assert_equal(_format_int[DType.index](4, 2), "100")
    assert_equal(_format_int[DType.index](255, 2), "11111111")
    assert_equal(_format_int[DType.index](254, 2), "11111110")
    assert_equal(_format_int[DType.index](255, 36), "73")

    assert_equal(_format_int[DType.index](-123, 10), "-123")
    assert_equal(_format_int[DType.index](-999_999_999, 10), "-999999999")

    #
    # Max and min i64 values in base 10
    #

    assert_equal(_format_int(Int64.MAX_FINITE, 10), "9223372036854775807")

    assert_equal(_format_int(Int64.MIN_FINITE, 10), "-9223372036854775808")

    #
    # Max and min i64 values in base 2
    #

    assert_equal(
        _format_int(Int64.MAX_FINITE, 2),
        "111111111111111111111111111111111111111111111111111111111111111",
    )

    assert_equal(
        _format_int(Int64.MIN_FINITE, 2),
        "-1000000000000000000000000000000000000000000000000000000000000000",
    )


fn test_hex() raises:
    assert_equal(hex(0), "0x0")
    assert_equal(hex(1), "0x1")
    assert_equal(hex(5), "0x5")
    assert_equal(hex(10), "0xa")
    assert_equal(hex(255), "0xff")
    assert_equal(hex(128), "0x80")
    assert_equal(hex(1 << 16), "0x10000")

    # Max and min i64 values in base 16
    assert_equal(hex(Int64.MAX_FINITE), "0x7fffffffffffffff")

    # Negative values
    assert_equal(hex(-0), "0x0")
    assert_equal(hex(-1), "-0x1")
    assert_equal(hex(-10), "-0xa")
    assert_equal(hex(-255), "-0xff")

    assert_equal(hex(Int64.MIN_FINITE), "-0x8000000000000000")

    # SIMD values
    assert_equal(hex(Int32(45)), "0x2d")
    assert_equal(hex(Int8(2)), "0x2")
    assert_equal(hex(Int8(-2)), "-0x2")
    assert_equal(hex(Scalar[DType.bool](True)), "0x1")
    assert_equal(hex(False), "0x0")


@value
struct Ind(Indexer):
    fn __index__(self) -> Int:
        return 1


def test_bin_scalar():
    assert_equal(bin(Int8(2)), "0b10")
    assert_equal(bin(Int32(123)), "0b1111011")
    assert_equal(bin(Int32(-123)), "-0b1111011")
    assert_equal(bin(Scalar[DType.bool](True)), "0b1")
    assert_equal(bin(Scalar[DType.bool](False)), "0b0")


def test_bin_int():
    assert_equal(bin(0), "0b0")
    assert_equal(bin(1), "0b1")
    assert_equal(bin(-1), "-0b1")
    assert_equal(bin(4), "0b100")
    assert_equal(bin(Int(-4)), "-0b100")
    assert_equal(bin(389703), "0b1011111001001000111")
    assert_equal(bin(-10), "-0b1010")


def test_bin_bool():
    assert_equal(bin(True), "0b1")
    assert_equal(bin(False), "0b0")


def test_indexer():
    assert_equal(bin(Ind()), "0b1")
    assert_equal(hex(Ind()), "0x1")


def main():
    test_format_int()
    test_hex()
    test_bin_scalar()
    test_bin_int()
    test_bin_bool()
    test_indexer()
