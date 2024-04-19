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

from builtin.hex import _format_int
from testing import assert_equal


fn test_format_int() raises:
    assert_equal(_format_int(123), "123")
    assert_equal(_format_int(4, 2), "100")
    assert_equal(_format_int(255, 2), "11111111")
    assert_equal(_format_int(254, 2), "11111110")
    assert_equal(_format_int(255, 36), "73")

    assert_equal(_format_int(-123, 10), "-123")
    assert_equal(_format_int(-999_999_999, 10), "-999999999")

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

    #
    # Negative values
    #

    assert_equal(hex(-0), "0x0")
    assert_equal(hex(-1), "-0x1")
    assert_equal(hex(-10), "-0xa")
    assert_equal(hex(-255), "-0xff")

    assert_equal(hex(Int64.MIN_FINITE), "-0x8000000000000000")


def main():
    test_format_int()
    test_hex()
