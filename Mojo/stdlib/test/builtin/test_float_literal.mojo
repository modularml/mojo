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

from std.testing import assert_equal, assert_false, assert_true, TestSuite

comptime nan = FloatLiteral.nan
comptime neg_zero = FloatLiteral.negative_zero
comptime inf = FloatLiteral.infinity
comptime neg_inf = FloatLiteral.negative_infinity


def test_init() raises:
    comptime n: IntLiteral[(4).value] = 4
    assert_equal(4.0, FloatLiteral(n))


def test_division() raises:
    assert_equal(FloatLiteral.__truediv__(4.4, 0.5), 8.8)

    assert_equal(FloatLiteral.__floordiv__(4.4, 0.5), 8.0)
    assert_equal(FloatLiteral.__floordiv__(-4.4, 0.5), -9.0)
    assert_equal(FloatLiteral.__floordiv__(4.4, -0.5), -9.0)
    assert_equal(FloatLiteral.__floordiv__(-4.4, -0.5), 8.0)


def test_mod() raises:
    assert_equal(FloatLiteral.__mod__(4.5, 2), 0.5)
    assert_equal(FloatLiteral.__mod__(-4.5, 2), 1.5)
    assert_equal(FloatLiteral.__mod__(6, 2.5), 1.0)


def test_int_conversion() raises:
    assert_equal(Int(Float64(-4.0)), -4)
    assert_equal(Int(Float64(-4.5)), -4)
    assert_equal(Int(Float64(-4.3)), -4)
    assert_equal(Int(Float64(4.5)), 4)
    assert_equal(Int(Float64(4.0)), 4)


def test_bool() raises:
    assert_false(FloatLiteral.__bool__(0.0))

    assert_true(FloatLiteral.__bool__(2.0))


def test_is_special_value() raises:
    assert_true(nan.is_nan())
    assert_false(neg_zero.is_nan())
    assert_true(neg_zero.is_neg_zero())
    assert_false(nan.is_neg_zero())


def test_abs() raises:
    assert_equal(abs(-4.4), 4.4)
    assert_equal(abs(4.4), 4.4)
    assert_equal(abs(0.0), 0.0)

    assert_equal(abs(neg_zero), 0.0)
    assert_equal(abs(inf), inf)
    assert_equal(abs(neg_inf), inf)


def test_comparison() raises:
    assert_true(FloatLiteral.__lt__(4.4, 10.4))
    assert_true(FloatLiteral.__lt__(-10.4, -4.4))
    assert_false(FloatLiteral.__lt__(0.0, 0.0))
    assert_false(FloatLiteral.__lt__(10.4, 4.4))
    assert_false(FloatLiteral.__lt__(neg_inf, neg_inf))
    assert_false(FloatLiteral.__lt__(neg_zero, neg_zero))
    assert_false(FloatLiteral.__lt__(neg_zero, 0.0))
    assert_true(FloatLiteral.__lt__(neg_inf, inf))
    assert_false(FloatLiteral.__lt__(inf, inf))
    assert_false(FloatLiteral.__lt__(nan, 10.0))
    assert_false(FloatLiteral.__lt__(10.0, nan))
    assert_false(FloatLiteral.__lt__(nan, inf))
    assert_false(FloatLiteral.__lt__(inf, nan))
    assert_false(FloatLiteral.__lt__(neg_inf, nan))
    assert_false(FloatLiteral.__lt__(nan, neg_zero))

    assert_true(FloatLiteral.__le__(4.4, 10.4))
    assert_true(FloatLiteral.__le__(-10.4, 4.4))
    assert_true(FloatLiteral.__le__(0.0, 0.0))
    assert_false(FloatLiteral.__le__(10.4, 4.4))
    assert_true(FloatLiteral.__le__(neg_inf, neg_inf))
    assert_true(FloatLiteral.__le__(neg_zero, neg_zero))
    assert_true(FloatLiteral.__le__(neg_zero, 0.0))
    assert_true(FloatLiteral.__le__(neg_inf, inf))
    assert_true(FloatLiteral.__le__(inf, inf))
    assert_false(FloatLiteral.__le__(nan, 10.0))
    assert_false(FloatLiteral.__le__(10.0, nan))
    assert_false(FloatLiteral.__le__(nan, inf))
    assert_false(FloatLiteral.__le__(inf, nan))
    assert_false(FloatLiteral.__le__(neg_inf, nan))
    assert_false(FloatLiteral.__le__(nan, neg_zero))

    assert_true(FloatLiteral.__eq__(4.4, 4.4))
    assert_false(FloatLiteral.__eq__(4.4, 42.0))
    assert_true(FloatLiteral.__eq__(neg_inf, neg_inf))
    assert_true(FloatLiteral.__eq__(neg_zero, neg_zero))
    assert_true(FloatLiteral.__eq__(neg_zero, 0.0))
    assert_false(FloatLiteral.__eq__(neg_inf, inf))
    assert_true(FloatLiteral.__eq__(inf, inf))
    assert_false(FloatLiteral.__eq__(nan, 10.0))
    assert_false(FloatLiteral.__eq__(10.0, nan))
    assert_false(FloatLiteral.__eq__(nan, inf))
    assert_false(FloatLiteral.__eq__(inf, nan))
    assert_false(FloatLiteral.__eq__(neg_inf, nan))
    assert_false(FloatLiteral.__eq__(nan, neg_zero))

    assert_false(FloatLiteral.__ne__(4.4, 4.4))
    assert_true(FloatLiteral.__ne__(4.4, 42.0))
    assert_false(FloatLiteral.__ne__(neg_inf, neg_inf))
    assert_false(FloatLiteral.__ne__(neg_zero, neg_zero))
    assert_false(FloatLiteral.__ne__(neg_zero, 0.0))
    assert_true(FloatLiteral.__ne__(neg_inf, inf))
    assert_false(FloatLiteral.__ne__(inf, inf))
    assert_true(FloatLiteral.__ne__(nan, 10.0))
    assert_true(FloatLiteral.__ne__(10.0, nan))
    assert_true(FloatLiteral.__ne__(nan, inf))
    assert_true(FloatLiteral.__ne__(inf, nan))
    assert_true(FloatLiteral.__ne__(neg_inf, nan))
    assert_true(FloatLiteral.__ne__(nan, neg_zero))

    assert_true(FloatLiteral.__gt__(10.4, 4.4))
    assert_true(FloatLiteral.__gt__(-4.4, -10.4))
    assert_false(FloatLiteral.__gt__(0.0, 0.0))
    assert_false(FloatLiteral.__gt__(4.4, 10.4))
    assert_false(FloatLiteral.__gt__(neg_inf, neg_inf))
    assert_false(FloatLiteral.__gt__(neg_zero, neg_zero))
    assert_false(FloatLiteral.__gt__(neg_zero, 0.0))
    assert_true(FloatLiteral.__gt__(inf, neg_inf))
    assert_false(FloatLiteral.__gt__(inf, inf))
    assert_false(FloatLiteral.__gt__(nan, 10.0))
    assert_false(FloatLiteral.__gt__(10.0, nan))
    assert_false(FloatLiteral.__gt__(nan, inf))
    assert_false(FloatLiteral.__gt__(inf, nan))
    assert_false(FloatLiteral.__gt__(neg_inf, nan))
    assert_false(FloatLiteral.__gt__(nan, neg_zero))

    assert_true(FloatLiteral.__ge__(10.4, 4.4))
    assert_true(FloatLiteral.__ge__(-4.4, -10.4))
    assert_true(FloatLiteral.__ge__(4.4, 4.4))
    assert_false(FloatLiteral.__ge__(4.4, 10.4))
    assert_true(FloatLiteral.__ge__(neg_inf, neg_inf))
    assert_true(FloatLiteral.__ge__(neg_zero, neg_zero))
    assert_true(FloatLiteral.__ge__(neg_zero, 0.0))
    assert_true(FloatLiteral.__ge__(inf, neg_inf))
    assert_true(FloatLiteral.__ge__(inf, inf))
    assert_false(FloatLiteral.__ge__(nan, 10.0))
    assert_false(FloatLiteral.__ge__(10.0, nan))
    assert_false(FloatLiteral.__ge__(nan, inf))
    assert_false(FloatLiteral.__ge__(inf, nan))
    assert_false(FloatLiteral.__ge__(neg_inf, nan))
    assert_false(FloatLiteral.__ge__(nan, neg_zero))


def test_float_conversion() raises:
    assert_equal((4.5).__float__(), 4.5)
    assert_equal((0.0).__float__(), 0.0)


def test_float_literal_writable() raises:
    assert_equal(String(4.5), String(Float64(4.5)))
    assert_equal(repr(4.5), repr(Float64(4.5)))

    assert_equal(String(0.0), String(Float64(0.0)))
    assert_equal(repr(0.0), repr(Float64(0.0)))

    assert_equal(String(FloatLiteral.infinity), "inf")
    assert_equal(String(FloatLiteral.negative_infinity), "-inf")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
