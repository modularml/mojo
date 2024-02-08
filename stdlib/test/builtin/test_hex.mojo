# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from testing import assert_equal
from math.limit import max_finite, min_finite

from builtin.hex import _format_int


fn test_format_int() raises:
    assert_equal(_format_int(123), "123")
    assert_equal(_format_int(4, 2), "100")
    assert_equal(_format_int(255, 2), "11111111")
    assert_equal(_format_int(254, 2), "11111110")
    assert_equal(_format_int(255, 36), "73")

    assert_equal(_format_int(-123, 10), "-123")
    assert_equal(_format_int(-999_999_999, 10), "-999999999")

    # Max and min i64 values in base 10
    assert_equal(
        _format_int(max_finite[DType.int64](), 10), "9223372036854775807"
    )

    assert_equal(
        _format_int(min_finite[DType.int64](), 10), "-9223372036854775808"
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
    assert_equal(hex(max_finite[DType.int64]()), "0x7fffffffffffffff")

    #
    # Negative values
    #

    assert_equal(hex(-0), "0x0")
    assert_equal(hex(-1), "-0x1")
    assert_equal(hex(-10), "-0xa")
    assert_equal(hex(-255), "-0xff")

    assert_equal(hex(min_finite[DType.int64]()), "-0x8000000000000000")


def main():
    test_format_int()
    test_hex()
