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

from sys.info import bitwidthof

from memory import UnsafePointer
from python import PythonObject
from testing import assert_equal, assert_false, assert_raises, assert_true


def test_properties():
    assert_equal(Int.MAX, (1 << bitwidthof[DType.index]() - 1) - 1)
    assert_equal(Int.MIN, -(1 << bitwidthof[DType.index]() - 1))


def test_add():
    assert_equal(Int.__add__(Int(3), Int(3)), 6)
    assert_equal(Int.__add__(Int(-2), Int(3)), 1)
    assert_equal(Int.__add__(Int(2), Int(-3)), -1)
    assert_equal(Int.__add__(Int(5), Int(-5)), 0)
    assert_equal(Int.__add__(Int(-5), Int(-4)), -9)


def test_sub():
    assert_equal(Int.__sub__(Int(3), Int(3)), 0)
    assert_equal(Int.__sub__(Int(-2), Int(3)), -5)
    assert_equal(Int.__sub__(Int(2), Int(-3)), 5)
    assert_equal(Int.__sub__(Int(5), Int(4)), 1)
    assert_equal(Int.__sub__(Int(4), Int(5)), -1)


def test_div():
    var n = Int(5)
    var d = Int(2)
    assert_equal(2.5, Int.__truediv__(n, d))
    Int.__itruediv__(n, d)
    assert_equal(2, n)


def test_pow():
    assert_equal(1, Int.__pow__(Int(3), Int(0)))
    assert_equal(27, Int.__pow__(Int(3), Int(3)))
    assert_equal(81, Int.__pow__(Int(3), Int(4)))


def test_ceil():
    assert_equal(Int.__ceil__(Int(5)), 5)
    assert_equal(Int.__ceil__(Int(0)), 0)
    assert_equal(Int.__ceil__(Int(-5)), -5)


def test_floor():
    assert_equal(Int.__floor__(Int(5)), 5)
    assert_equal(Int.__floor__(Int(0)), 0)
    assert_equal(Int.__floor__(Int(-5)), -5)


def test_round():
    assert_equal(Int.__round__(Int(5)), 5)
    assert_equal(Int.__round__(Int(0)), 0)
    assert_equal(Int.__round__(Int(-5)), -5)
    assert_equal(Int.__round__(5, 1), 5)
    assert_equal(Int.__round__(0, 1), 0)
    assert_equal(Int.__round__(-5, 1), -5)
    assert_equal(Int.__round__(100, -2), 100)


def test_trunc():
    assert_equal(Int.__trunc__(Int(5)), 5)
    assert_equal(Int.__trunc__(Int(0)), 0)
    assert_equal(Int.__trunc__(Int(-5)), -5)


def test_floordiv():
    assert_equal(1, Int.__floordiv__(Int(2), Int(2)))
    assert_equal(0, Int.__floordiv__(Int(2), Int(3)))
    assert_equal(-1, Int.__floordiv__(Int(2), Int(-2)))
    assert_equal(-50, Int.__floordiv__(Int(99), Int(-2)))
    assert_equal(-1, Int.__floordiv__(Int(-1), Int(10)))


def test_mod():
    assert_equal(0, Int.__mod__(Int(99), Int(1)))
    assert_equal(0, Int.__mod__(Int(99), Int(3)))
    assert_equal(-1, Int.__mod__(Int(99), Int(-2)))
    assert_equal(3, Int.__mod__(Int(99), Int(8)))
    assert_equal(-5, Int.__mod__(Int(99), Int(-8)))
    assert_equal(0, Int.__mod__(Int(2), Int(-1)))
    assert_equal(0, Int.__mod__(Int(2), Int(-2)))
    assert_equal(-1, Int.__mod__(Int(3), Int(-2)))
    assert_equal(1, Int.__mod__(Int(-3), Int(2)))


def test_divmod():
    var a: Int
    var b: Int
    a, b = divmod(7, 3)
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


def test_abs():
    assert_equal(Int(-5).__abs__(), 5)
    assert_equal(Int(2).__abs__(), 2)
    assert_equal(Int(0).__abs__(), 0)


def test_string_conversion():
    assert_equal(Int(3).__str__(), "3")
    assert_equal(Int(-3).__str__(), "-3")
    assert_equal(Int(0).__str__(), "0")
    assert_equal(Int(100).__str__(), "100")
    assert_equal(Int(-100).__str__(), "-100")


def test_int_representation():
    assert_equal(Int(3).__repr__(), "3")
    assert_equal(Int(-3).__repr__(), "-3")
    assert_equal(Int(0).__repr__(), "0")
    assert_equal(Int(100).__repr__(), "100")
    assert_equal(Int(-100).__repr__(), "-100")


def test_indexer():
    assert_equal(5, Int(5).__index__())
    assert_equal(987, Int(987).__index__())


def test_bool():
    assert_true(Int(5).__bool__())
    assert_false(Int(0).__bool__())
    assert_true(Int(5).__as_bool__())
    assert_false(Int(0).__as_bool__())


def test_decimal_digit_count():
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


def test_int_uint():
    var u1 = UInt(42)
    assert_equal(42, int(u1))

    var u2 = UInt(0)
    assert_equal(0, int(u2))


def test_comparison():
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


def test_float_conversion():
    assert_equal(float(Int(45)), Float64(45))


def test_conversion_from_python():
    # Test conversion from Python '5'
    assert_equal(Int.try_from_python(PythonObject(5)), 5)

    # Test error trying conversion from Python '"str"'
    with assert_raises(contains="an integer is required"):
        _ = Int.try_from_python(PythonObject("str"))

    # Test conversion from Python '-1'
    assert_equal(Int.try_from_python(PythonObject(-1)), -1)


def main():
    test_properties()
    test_add()
    test_sub()
    test_div()
    test_pow()
    test_ceil()
    test_floor()
    test_round()
    test_trunc()
    test_floordiv()
    test_mod()
    test_divmod()
    test_abs()
    test_string_conversion()
    test_int_representation()
    test_indexer()
    test_bool()
    test_decimal_digit_count()
    test_comparison()
    test_int_uint()
    test_float_conversion()
    test_conversion_from_python()
