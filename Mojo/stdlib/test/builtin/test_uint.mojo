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


from std.bit import count_trailing_zeros
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)
from std.sys import bit_width_of


def test_simple_uint() raises:
    assert_equal(String(UInt(32)), "32")

    assert_equal(String(UInt(0)), "0")
    assert_equal(String(UInt()), "0")

    assert_equal(String(UInt(18446744073709551615)), "18446744073709551615")


def test_equality() raises:
    assert_equal(UInt(32), UInt(32))
    assert_equal(UInt(0), UInt(0))
    assert_equal(UInt(), UInt(0))
    assert_equal(UInt(18446744073709551615), UInt(18446744073709551615))
    assert_equal(
        UInt(18446744073709551615 - 10), UInt(18446744073709551615 - 10)
    )

    assert_true(UInt(32).__eq__(UInt(32)))
    assert_true(UInt(0).__eq__(UInt(0)))
    assert_true(UInt().__eq__(UInt(0)))
    assert_true(UInt(18446744073709551615).__eq__(UInt(18446744073709551615)))
    assert_true(
        UInt(18446744073709551615 - 10).__eq__(UInt(18446744073709551615 - 10))
    )

    assert_false(UInt(32).__eq__(UInt(0)))
    assert_false(UInt(0).__eq__(UInt(32)))
    assert_false(UInt(0).__eq__(UInt(18446744073709551615)))
    assert_false(UInt(18446744073709551615).__eq__(UInt(0)))
    assert_false(
        UInt(18446744073709551615).__eq__(UInt(18446744073709551615 - 10))
    )


def test_inequality() raises:
    assert_not_equal(UInt(32), UInt(0))
    assert_not_equal(UInt(0), UInt(32))
    assert_not_equal(UInt(0), UInt(18446744073709551615))
    assert_not_equal(UInt(18446744073709551615), UInt(0))
    assert_not_equal(
        UInt(18446744073709551615), UInt(18446744073709551615 - 10)
    )

    assert_false(UInt(32).__ne__(UInt(32)))
    assert_false(UInt(0).__ne__(UInt(0)))
    assert_false(UInt().__ne__(UInt(0)))
    assert_false(UInt(18446744073709551615).__ne__(UInt(18446744073709551615)))
    assert_false(
        UInt(18446744073709551615 - 10).__ne__(UInt(18446744073709551615 - 10))
    )

    assert_true(UInt(32).__ne__(UInt(0)))
    assert_true(UInt(0).__ne__(UInt(32)))
    assert_true(UInt(0).__ne__(UInt(18446744073709551615)))
    assert_true(UInt(18446744073709551615).__ne__(UInt(0)))
    assert_true(
        UInt(18446744073709551615).__ne__(UInt(18446744073709551615 - 10))
    )


def test_properties() raises:
    assert_equal(UInt.MIN, UInt(0))
    if bit_width_of[DType.int]() == 32:
        assert_equal(UInt.MAX, (1 << 32) - 1)
    else:
        assert_equal(UInt.MAX, (1 << 64) - 1)


def test_add() raises:
    assert_equal(UInt.__add__(UInt(3), UInt(3)), UInt(6))
    assert_equal(UInt.__add__(UInt(Int(-2)), UInt(3)), UInt(1))
    assert_equal(UInt.__add__(UInt(2), UInt(Int(-3))), UInt(Int(-1)))
    assert_equal(UInt.__add__(UInt(5), UInt(Int(-5))), UInt(0))
    assert_equal(UInt.__add__(UInt(Int(-5)), UInt(Int(-4))), UInt(Int(-9)))


def test_sub() raises:
    assert_equal(UInt.__sub__(UInt(3), UInt(3)), UInt(0))
    assert_equal(UInt.__sub__(UInt(Int(-2)), UInt(3)), UInt(Int(-5)))
    assert_equal(UInt.__sub__(UInt(2), UInt(Int(-3))), UInt(5))
    assert_equal(UInt.__sub__(UInt(5), UInt(4)), UInt(1))
    assert_equal(UInt.__sub__(UInt(4), UInt(5)), UInt(Int(-1)))


# TODO(MSTDL-2195): decide about what __truediv__ should return.
# def test_div():
#     var n = UInt(5)
#     var d = UInt(2)
#     assert_equal(2.5, UInt.__truediv__(n, d))
#     UInt.__itruediv__(n, d)
#     assert_equal(UInt(2), n)


def test_pow() raises:
    assert_equal(UInt(1), UInt.__pow__(UInt(3), UInt(0)))
    assert_equal(UInt(27), UInt.__pow__(UInt(3), UInt(3)))
    assert_equal(UInt(81), UInt.__pow__(UInt(3), UInt(4)))


def test_ceil() raises:
    assert_equal(UInt.__ceil__(UInt(5)), UInt(5))
    assert_equal(UInt.__ceil__(UInt(0)), UInt(0))
    assert_equal(UInt.__ceil__(UInt(Int(-5))), UInt(Int(-5)))


def test_floor() raises:
    assert_equal(UInt.__floor__(UInt(5)), UInt(5))
    assert_equal(UInt.__floor__(UInt(0)), UInt(0))
    assert_equal(UInt.__floor__(UInt(Int(-5))), UInt(Int(-5)))


def test_round() raises:
    assert_equal(UInt.__round__(UInt(5)), UInt(5))
    assert_equal(UInt.__round__(UInt(0)), UInt(0))
    assert_equal(UInt.__round__(UInt(Int(-5))), UInt(Int(-5)))
    assert_equal(UInt.__round__(UInt(5), 1), UInt(5))
    assert_equal(UInt.__round__(UInt(0), 1), UInt(0))
    assert_equal(UInt.__round__(UInt(Int(-5)), 1), UInt(Int(-5)))
    assert_equal(UInt.__round__(UInt(100), -2), UInt(100))


def test_trunc() raises:
    assert_equal(UInt.__trunc__(UInt(5)), UInt(5))
    assert_equal(UInt.__trunc__(UInt(0)), UInt(0))


def test_floordiv() raises:
    assert_equal(UInt(1), UInt.__floordiv__(UInt(2), UInt(2)))
    assert_equal(UInt(0), UInt.__floordiv__(UInt(2), UInt(3)))
    assert_equal(UInt(2), UInt.__floordiv__(UInt(100), UInt(50)))
    assert_equal(UInt(0), UInt.__floordiv__(UInt(2), UInt(Int(-2))))
    assert_equal(UInt(0), UInt.__floordiv__(UInt(99), UInt(Int(-2))))


def test_mod() raises:
    assert_equal(UInt(0), UInt.__mod__(UInt(99), UInt(1)))
    assert_equal(UInt(0), UInt.__mod__(UInt(99), UInt(3)))
    assert_equal(UInt(99), UInt.__mod__(UInt(99), UInt(Int(-2))))
    assert_equal(UInt(3), UInt.__mod__(UInt(99), UInt(8)))
    assert_equal(UInt(99), UInt.__mod__(UInt(99), UInt(Int(-8))))
    assert_equal(UInt(2), UInt.__mod__(UInt(2), UInt(Int(-1))))
    assert_equal(UInt(2), UInt.__mod__(UInt(2), UInt(Int(-2))))
    assert_equal(UInt(3), UInt.__mod__(UInt(3), UInt(Int(-2))))
    assert_equal(UInt(1), UInt.__mod__(UInt(Int(-3)), UInt(2)))


def test_divmod() raises:
    var a, b = divmod(UInt(7), UInt(3))
    assert_equal(a, UInt(2))
    assert_equal(b, UInt(1))

    a, b = divmod(UInt(0), UInt(5))
    assert_equal(a, UInt(0))
    assert_equal(b, UInt(0))

    a, b = divmod(UInt(5), UInt(0))
    assert_equal(a, UInt(0))
    assert_equal(b, UInt(0))


def test_abs() raises:
    assert_equal(UInt(Int(-5)).__abs__(), UInt(18446744073709551611))
    assert_equal(UInt(2).__abs__(), UInt(2))
    assert_equal(UInt(0).__abs__(), UInt(0))


def test_indexer() raises:
    assert_true(5 == index(UInt(5)))
    assert_true(987 == index(UInt(987)))


def test_simd_conversion() raises:
    assert_equal(UInt(Int32(1)), UInt(1))
    assert_equal(UInt(UInt32(32)), UInt(32))


def test_comparison() raises:
    assert_true(UInt.__lt__(UInt(1), UInt(7)))
    assert_false(UInt.__lt__(UInt(7), UInt(7)))
    assert_false(UInt.__lt__(UInt(7), UInt(2)))

    assert_true(UInt.__le__(UInt(1), UInt(7)))
    assert_true(UInt.__le__(UInt(7), UInt(7)))
    assert_false(UInt.__le__(UInt(7), UInt(2)))

    assert_false(UInt.__gt__(UInt(1), UInt(7)))
    assert_false(UInt.__gt__(UInt(7), UInt(7)))
    assert_true(UInt.__gt__(UInt(7), UInt(2)))

    assert_false(UInt.__ge__(UInt(1), UInt(7)))
    assert_true(UInt.__ge__(UInt(7), UInt(7)))
    assert_true(UInt.__ge__(UInt(7), UInt(2)))


def test_pos() raises:
    assert_equal(UInt(2).__pos__(), UInt(2))
    assert_equal(UInt(0).__pos__(), UInt(0))


def test_hash() raises:
    assert_not_equal(hash(UInt(123)), hash(UInt(456)))
    assert_equal(hash(UInt(123)), hash(UInt(123)))
    assert_equal(hash(UInt(456)), hash(UInt(456)))


def test_comptime() raises:
    comptime a: UInt = 32
    # Verify that count_trailing_zeros works at comptime.
    comptime n = count_trailing_zeros(Int(a))
    assert_equal(n, 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
