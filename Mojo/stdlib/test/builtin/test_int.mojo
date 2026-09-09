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

from std.sys import bit_width_of
from std.python import PythonObject
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_properties() raises:
    assert_equal(Int.MAX, (1 << bit_width_of[DType.int]() - 1) - 1)
    assert_equal(Int.MIN, -(1 << bit_width_of[DType.int]() - 1))


def test_add() raises:
    assert_equal(Int.__add__(Int(3), Int(3)), 6)
    assert_equal(Int.__add__(Int(-2), Int(3)), 1)
    assert_equal(Int.__add__(Int(2), Int(-3)), -1)
    assert_equal(Int.__add__(Int(5), Int(-5)), 0)
    assert_equal(Int.__add__(Int(-5), Int(-4)), -9)


def test_sub() raises:
    assert_equal(Int.__sub__(Int(3), Int(3)), 0)
    assert_equal(Int.__sub__(Int(-2), Int(3)), -5)
    assert_equal(Int.__sub__(Int(2), Int(-3)), 5)
    assert_equal(Int.__sub__(Int(5), Int(4)), 1)
    assert_equal(Int.__sub__(Int(4), Int(5)), -1)


def test_div() raises:
    var n = Int(5)
    var d = Int(2)
    # TODO: re-enable when Int.__truediv__ is re-enabled
    # assert_equal(2.5, Int.__truediv__(n, d))
    Int.__itruediv__(n, d)
    assert_equal(2, n)


def test_pow() raises:
    assert_equal(1, Int.__pow__(Int(3), Int(0)))
    assert_equal(27, Int.__pow__(Int(3), Int(3)))
    assert_equal(81, Int.__pow__(Int(3), Int(4)))

    # Negative exponents: 1 ** n == 1 for all n.
    assert_equal(1, Int.__pow__(Int(1), Int(-1)))
    assert_equal(1, Int.__pow__(Int(1), Int(-5)))

    # Negative exponents: (-1) ** n depends on parity.
    assert_equal(-1, Int.__pow__(Int(-1), Int(-1)))
    assert_equal(1, Int.__pow__(Int(-1), Int(-2)))
    assert_equal(-1, Int.__pow__(Int(-1), Int(-3)))

    # Negative exponents: |base| > 1 truncates to 0.
    assert_equal(0, Int.__pow__(Int(2), Int(-1)))
    assert_equal(0, Int.__pow__(Int(3), Int(-2)))


def test_ceil() raises:
    assert_equal(Int.__ceil__(Int(5)), 5)
    assert_equal(Int.__ceil__(Int(0)), 0)
    assert_equal(Int.__ceil__(Int(-5)), -5)


def test_floor() raises:
    assert_equal(Int.__floor__(Int(5)), 5)
    assert_equal(Int.__floor__(Int(0)), 0)
    assert_equal(Int.__floor__(Int(-5)), -5)


def test_round() raises:
    assert_equal(Int.__round__(Int(5)), 5)
    assert_equal(Int.__round__(Int(0)), 0)
    assert_equal(Int.__round__(Int(-5)), -5)
    assert_equal(Int.__round__(5, 1), 5)
    assert_equal(Int.__round__(0, 1), 0)
    assert_equal(Int.__round__(-5, 1), -5)
    assert_equal(Int.__round__(100, -2), 100)
    assert_equal(Int.__round__(1342, 0), 1342)
    assert_equal(Int.__round__(1342, -1), 1340)
    assert_equal(Int.__round__(1342, -2), 1300)
    assert_equal(Int.__round__(1342, -3), 1000)
    assert_equal(Int.__round__(1342, -4), 0)
    assert_equal(Int.__round__(1342, -5), 0)


def test_trunc() raises:
    assert_equal(Int.__trunc__(Int(5)), 5)
    assert_equal(Int.__trunc__(Int(0)), 0)
    assert_equal(Int.__trunc__(Int(-5)), -5)


def test_floordiv() raises:
    assert_equal(1, Int.__floordiv__(Int(2), Int(2)))
    assert_equal(0, Int.__floordiv__(Int(2), Int(3)))
    assert_equal(-1, Int.__floordiv__(Int(2), Int(-2)))
    assert_equal(-50, Int.__floordiv__(Int(99), Int(-2)))
    assert_equal(-1, Int.__floordiv__(Int(-1), Int(10)))


def test_mod() raises:
    assert_equal(0, Int.__mod__(Int(99), Int(1)))
    assert_equal(0, Int.__mod__(Int(99), Int(3)))
    assert_equal(-1, Int.__mod__(Int(99), Int(-2)))
    assert_equal(3, Int.__mod__(Int(99), Int(8)))
    assert_equal(-5, Int.__mod__(Int(99), Int(-8)))
    assert_equal(0, Int.__mod__(Int(2), Int(-1)))
    assert_equal(0, Int.__mod__(Int(2), Int(-2)))
    assert_equal(-1, Int.__mod__(Int(3), Int(-2)))
    assert_equal(1, Int.__mod__(Int(-3), Int(2)))


def test_divmod() raises:
    var a, b = divmod(7, 3)
    assert_equal(a, 2)
    assert_equal(b, 1)

    a, b = divmod(-7, 3)
    assert_equal(a, -3)
    assert_equal(b, 2)

    a, b = divmod(-7, -3)
    assert_equal(a, 2)
    assert_equal(b, -1)

    a, b = divmod(7, -3)
    assert_equal(a, -3)
    assert_equal(b, -2)

    a, b = divmod(0, 5)
    assert_equal(a, 0)
    assert_equal(b, 0)

    a, b = divmod(5, 0)
    assert_equal(a, 0)
    assert_equal(b, 0)


def test_abs() raises:
    assert_equal(Int(-5).__abs__(), 5)
    assert_equal(Int(2).__abs__(), 2)
    assert_equal(Int(0).__abs__(), 0)
    assert_equal(Int.MIN.__abs__(), Int.MIN)


def test_int_write_repr_to() raises:
    def check(i: Int, expected: String) raises:
        var string = String()
        i.write_repr_to(string)
        assert_equal(string, expected)

    check(Int(3), "Int(3)")
    check(Int(-3), "Int(-3)")
    check(Int(0), "Int(0)")
    check(Int(100), "Int(100)")
    check(Int(-100), "Int(-100)")


def test_indexer() raises:
    assert_true(5 == index(Int(5)))
    assert_true(987 == index(Int(987)))


def test_bool() raises:
    assert_true(Int(5).__bool__())
    assert_false(Int(0).__bool__())


def test_decimal_digit_count() raises:
    assert_equal(Int(0)._decimal_digit_count(), 1)
    assert_equal(Int(1)._decimal_digit_count(), 1)
    assert_equal(Int(2)._decimal_digit_count(), 1)
    assert_equal(Int(3)._decimal_digit_count(), 1)
    assert_equal(Int(9)._decimal_digit_count(), 1)

    assert_equal(Int(10)._decimal_digit_count(), 2)
    assert_equal(Int(11)._decimal_digit_count(), 2)
    assert_equal(Int(99)._decimal_digit_count(), 2)

    assert_equal(Int(100)._decimal_digit_count(), 3)
    assert_equal(Int(101)._decimal_digit_count(), 3)
    assert_equal(Int(999)._decimal_digit_count(), 3)

    assert_equal(Int(1000)._decimal_digit_count(), 4)

    assert_equal(Int(-1000)._decimal_digit_count(), 4)
    assert_equal(Int(-999)._decimal_digit_count(), 3)
    assert_equal(Int(-1)._decimal_digit_count(), 1)

    assert_equal(Int.MAX._decimal_digit_count(), 19)
    assert_equal(Int.MIN._decimal_digit_count(), 19)


def test_comparison() raises:
    assert_true(Int(5).__lt__(Int(10)))
    assert_true(Int(-10).__lt__(Int(-5)))
    assert_false(Int(0).__lt__(Int(0)))
    assert_false(Int(10).__lt__(Int(5)))

    assert_true(Int(5).__le__(Int(10)))
    assert_true(Int(-10).__le__(Int(-5)))
    assert_true(Int(0).__le__(Int(0)))
    assert_false(Int(10).__le__(Int(5)))

    assert_true(Int(5).__eq__(Int(5)))
    assert_true(Int(0).__eq__(Int(0)))
    assert_false(Int(0).__eq__(Int(1)))
    assert_false(Int(5).__eq__(Int(10)))

    assert_true(Int(5).__ne__(Int(10)))
    assert_true(Int(0).__ne__(Int(1)))
    assert_false(Int(5).__ne__(Int(5)))
    assert_false(Int(0).__ne__(Int(0)))

    assert_true(Int(10).__gt__(Int(5)))
    assert_true(Int(-5).__gt__(Int(-10)))
    assert_false(Int(0).__gt__(Int(0)))
    assert_false(Int(5).__gt__(Int(10)))

    assert_true(Int(10).__ge__(Int(5)))
    assert_true(Int(5).__ge__(Int(5)))
    assert_true(Int(-5).__ge__(Int(-10)))
    assert_false(Int(5).__ge__(Int(10)))


def test_float_conversion() raises:
    assert_equal(Float64(Int(45)), Float64(45))


def test_is_power_of_two() raises:
    assert_equal(Int.MIN.is_power_of_two(), False)
    assert_equal(Int(-(2**59)).is_power_of_two(), False)
    assert_equal(Int(-1).is_power_of_two(), False)
    assert_equal(Int(0).is_power_of_two(), False)
    assert_equal(Int(1).is_power_of_two(), True)
    assert_equal(Int(2).is_power_of_two(), True)
    assert_equal(Int(3).is_power_of_two(), False)
    assert_equal(Int(4).is_power_of_two(), True)
    assert_equal(Int(5).is_power_of_two(), False)
    assert_equal(UInt64(Int.MAX).is_power_of_two(), False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
