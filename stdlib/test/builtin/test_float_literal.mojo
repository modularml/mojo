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

from testing import (
    assert_almost_equal,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

alias nan = FloatLiteral.nan
alias neg_zero = FloatLiteral.negative_zero
alias inf = FloatLiteral.infinity
alias neg_inf = FloatLiteral.negative_infinity


def test_ceil():
    assert_equal(FloatLiteral.__ceil__(1.5), 2.0)
    assert_equal(FloatLiteral.__ceil__(1.4), 2.0)
    assert_equal(FloatLiteral.__ceil__(-1.5), -1.0)
    assert_equal(FloatLiteral.__ceil__(-3.6), -3.0)
    assert_equal(FloatLiteral.__ceil__(3.0), 3.0)
    assert_equal(FloatLiteral.__ceil__(0.0), 0.0)

    assert_true(FloatLiteral.__ceil__(nan).is_nan())
    assert_true(FloatLiteral.__ceil__(neg_zero).is_neg_zero())
    assert_equal(FloatLiteral.__ceil__(inf), inf)
    assert_equal(FloatLiteral.__ceil__(neg_inf), neg_inf)


def test_floor():
    assert_equal(FloatLiteral.__floor__(1.5), 1.0)
    assert_equal(FloatLiteral.__floor__(1.6), 1.0)
    assert_equal(FloatLiteral.__floor__(-1.5), -2.0)
    assert_equal(FloatLiteral.__floor__(-3.4), -4.0)
    assert_equal(FloatLiteral.__floor__(3.0), 3.0)
    assert_equal(FloatLiteral.__floor__(0.0), 0.0)

    assert_true(FloatLiteral.__floor__(nan).is_nan())
    assert_true(FloatLiteral.__floor__(neg_zero).is_neg_zero())
    assert_equal(FloatLiteral.__floor__(inf), inf)
    assert_equal(FloatLiteral.__floor__(neg_inf), neg_inf)


def test_trunc():
    assert_equal(FloatLiteral.__trunc__(1.5), 1.0)
    assert_equal(FloatLiteral.__trunc__(1.6), 1.0)
    assert_equal(FloatLiteral.__trunc__(-1.5), -1.0)
    assert_equal(FloatLiteral.__trunc__(-3.6), -3.0)
    assert_equal(FloatLiteral.__trunc__(3.0), 3.0)
    assert_equal(FloatLiteral.__trunc__(0.0), 0.0)

    assert_true(FloatLiteral.__trunc__(nan).is_nan())
    assert_true(FloatLiteral.__trunc__(neg_zero).is_neg_zero())
    assert_equal(FloatLiteral.__trunc__(inf), inf)
    assert_equal(FloatLiteral.__trunc__(neg_inf), neg_inf)


def test_round():
    assert_equal(FloatLiteral.__round__(1.5), 2.0)
    assert_equal(FloatLiteral.__round__(1.6), 2.0)
    assert_equal(FloatLiteral.__round__(-1.5), -2.0)
    assert_equal(FloatLiteral.__round__(-3.6), -4.0)
    assert_equal(FloatLiteral.__round__(3.0), 3.0)
    assert_equal(FloatLiteral.__round__(0.0), 0.0)

    assert_true(FloatLiteral.__round__(nan).is_nan())
    assert_true(FloatLiteral.__round__(neg_zero).is_neg_zero())
    assert_equal(FloatLiteral.__round__(inf), inf)
    assert_equal(FloatLiteral.__round__(neg_inf), neg_inf)

    assert_equal(FloatLiteral.__round__(1.6, 0), 2.0)

    assert_equal(FloatLiteral.__round__(1.5, 1), 1.5)
    assert_equal(FloatLiteral.__round__(1.123, 1), 1.1)
    assert_equal(FloatLiteral.__round__(1.198, 2), 1.2)
    assert_equal(FloatLiteral.__round__(1.123, 2), 1.12)
    assert_equal(FloatLiteral.__round__(-1.5, 1), -1.5)
    assert_equal(FloatLiteral.__round__(-1.123, 1), -1.1)
    assert_equal(FloatLiteral.__round__(-1.198, 2), -1.2)
    assert_equal(FloatLiteral.__round__(-1.123, 2), -1.12)

    # Test rounding to nearest even number
    assert_equal(FloatLiteral.__round__(1.5, 0), 2.0)
    assert_equal(FloatLiteral.__round__(2.5, 0), 2.0)
    assert_equal(FloatLiteral.__round__(-2.5, 0), -2.0)
    assert_equal(FloatLiteral.__round__(-1.5, 0), -2.0)

    # Negative ndigits
    assert_equal(FloatLiteral.__round__(123.456, -1), 120.0)
    assert_equal(FloatLiteral.__round__(123.456, -2), 100.0)
    assert_equal(FloatLiteral.__round__(123.456, -3), 0.0)


fn round10(x: FloatLiteral) -> FloatLiteral:
    return round(x * 10.0) / 10.0


def test_round10():
    assert_equal(round10(FloatLiteral.__mod__(4.4, 0.5)), 0.4)
    assert_equal(round10(FloatLiteral.__mod__(-4.4, 0.5)), 0.1)
    assert_equal(round10(FloatLiteral.__mod__(4.4, -0.5)), -0.1)
    assert_equal(round10(FloatLiteral.__mod__(-4.4, -0.5)), -0.4)
    assert_equal(round10(FloatLiteral.__mod__(3.1, 1.0)), 0.1)


def test_division():
    assert_equal(FloatLiteral.__truediv__(4.4, 0.5), 8.8)

    assert_equal(FloatLiteral.__floordiv__(4.4, 0.5), 8.0)
    assert_equal(FloatLiteral.__floordiv__(-4.4, 0.5), -9.0)
    assert_equal(FloatLiteral.__floordiv__(4.4, -0.5), -9.0)
    assert_equal(FloatLiteral.__floordiv__(-4.4, -0.5), 8.0)


def test_exp():
    assert_almost_equal(FloatLiteral.__exp__(0.0), 1.0)
    assert_almost_equal(FloatLiteral.__exp__(0.5), 1.6487212707_0012814684)
    assert_almost_equal(FloatLiteral.__exp__(-0.6), 0.5488116360_9402643262)
    assert_almost_equal(FloatLiteral.__exp__(2.2), 9.0250134994_3412092647)
    assert_almost_equal(FloatLiteral.__exp__(-3.7), 0.0247235264_7033939120)
    assert_almost_equal(FloatLiteral.__exp__(1.0), 2.7182818284_5904523536)
    assert_almost_equal(FloatLiteral.__exp__(10.0), 22026.4657948067_1651695790)


def test_log():
    assert_almost_equal(
        FloatLiteral.__log__(0.0), FloatLiteral.nan, equal_nan=True
    )
    assert_almost_equal(FloatLiteral.__log__(0.5), -0.6931471805_5994530941)
    assert_almost_equal(
        FloatLiteral.__log__(-0.6), FloatLiteral.nan, equal_nan=True
    )
    assert_almost_equal(FloatLiteral.__log__(2.2), 0.7884573603_6427016946)
    assert_almost_equal(
        FloatLiteral.__log__(-3.7), FloatLiteral.nan, equal_nan=True
    )
    assert_almost_equal(FloatLiteral.__log__(1.0), 0.0)
    assert_almost_equal(FloatLiteral.__log__(10.0), 2.3025850929_9404568401)


def test_pow():
    assert_almost_equal(FloatLiteral.__pow__(0.0, 0.5), 0.0)
    assert_almost_equal(FloatLiteral.__pow__(0.0, -2.5), 0.0)
    assert_almost_equal(FloatLiteral.__pow__(0.0, 1), 0.0)
    assert_almost_equal(FloatLiteral.__pow__(0.0, -2), 0.0)

    assert_almost_equal(FloatLiteral.__pow__(1.0, 0.5), 1.0)
    assert_almost_equal(FloatLiteral.__pow__(1.0, -2.5), 1.0)
    assert_almost_equal(FloatLiteral.__pow__(1.0, 1), 1.0)
    assert_almost_equal(FloatLiteral.__pow__(1.0, -2), 1.0)

    assert_almost_equal(
        FloatLiteral.__pow__(-1.0, 0.5), FloatLiteral.nan, equal_nan=True
    )
    assert_almost_equal(
        FloatLiteral.__pow__(-1.0, -2.5), FloatLiteral.nan, equal_nan=True
    )
    assert_almost_equal(FloatLiteral.__pow__(-1.0, 1), -1.0)
    assert_almost_equal(FloatLiteral.__pow__(-1.0, 2), 1.0)
    assert_almost_equal(FloatLiteral.__pow__(-1.0, -3), -1.0)
    assert_almost_equal(FloatLiteral.__pow__(-1.0, 100), 1.0)
    assert_almost_equal(FloatLiteral.__pow__(-1.0, 101), -1.0)

    assert_almost_equal(FloatLiteral.__pow__(2.3, 0.4), 1.3953756318_5563997952)
    assert_almost_equal(FloatLiteral.__pow__(0.8, 0.2), 0.9563524997_9003698571)
    assert_almost_equal(FloatLiteral.__pow__(10.0, -4), 0.0001)
    assert_almost_equal(FloatLiteral.__pow__(10.0, -4.0), 0.0001)
    assert_almost_equal(FloatLiteral.__pow__(-4.0, -2), 0.0625)
    assert_almost_equal(FloatLiteral.__pow__(-4.0, -2.0), 0.0625)
    assert_almost_equal(FloatLiteral.__pow__(-4.0, 2), 16)
    assert_almost_equal(FloatLiteral.__pow__(-4.0, 2.0), 16)


def test_mod():
    assert_equal(FloatLiteral.__mod__(4.5, 2), 0.5)
    assert_equal(FloatLiteral.__mod__(-4.5, 2), 1.5)
    assert_equal(FloatLiteral.__mod__(6, 2.5), 1.0)


def test_div_mod():
    var t: Tuple[FloatLiteral, FloatLiteral] = FloatLiteral.__divmod__(4.5, 2.0)
    assert_equal(t[0], 2.0)
    assert_equal(t[1], 0.5)

    t = FloatLiteral.__divmod__(-4.5, 2.0)
    assert_equal(t[0], -3.0)
    assert_equal(t[1], 1.5)

    t = FloatLiteral.__divmod__(4.5, -2.0)
    assert_equal(t[0], -3.0)
    assert_equal(t[1], -1.5)

    t = FloatLiteral.__divmod__(6.0, 2.5)
    assert_equal(t[0], 2.0)
    assert_equal(t[1], 1.0)


def test_int_conversion():
    assert_equal(int(-4.0), -4)
    assert_equal(int(-4.5), -4)
    assert_equal(int(-4.3), -4)
    assert_equal(int(4.5), 4)
    assert_equal(int(4.0), 4)


def test_bool():
    assert_false(FloatLiteral.__bool__(0.0))
    assert_false(FloatLiteral.__as_bool__(0.0))

    assert_true(FloatLiteral.__bool__(2.0))
    assert_true(FloatLiteral.__as_bool__(2.0))


def test_is_special_value():
    assert_true(nan.is_nan())
    assert_false(neg_zero.is_nan())
    assert_true(neg_zero.is_neg_zero())
    assert_false(nan.is_neg_zero())


def test_abs():
    assert_equal((-4.4).__abs__(), 4.4)
    assert_equal((4.4).__abs__(), 4.4)
    assert_equal((0.0).__abs__(), 0.0)

    assert_true(FloatLiteral.__abs__(nan).is_nan())
    assert_false(FloatLiteral.__abs__(neg_zero).is_neg_zero())
    assert_equal(FloatLiteral.__abs__(neg_zero), 0.0)
    assert_equal(FloatLiteral.__abs__(inf), inf)
    assert_equal(FloatLiteral.__abs__(neg_inf), inf)


def test_comparison():
    assert_true(FloatLiteral.__lt__(4.4, 10.4))
    assert_true(FloatLiteral.__lt__(-10.4, -4.4))
    assert_false(FloatLiteral.__lt__(0.0, 0.0))
    assert_false(FloatLiteral.__lt__(10.4, 4.4))

    assert_true(FloatLiteral.__le__(4.4, 10.4))
    assert_true(FloatLiteral.__le__(-10.4, 4.4))
    assert_true(FloatLiteral.__le__(0.0, 0.0))
    assert_false(FloatLiteral.__le__(10.4, 4.4))

    # TODO: add tests for special values
    assert_true(FloatLiteral.__eq__(4.4, 4.4))
    assert_false(FloatLiteral.__eq__(4.4, 42.0))
    assert_false(FloatLiteral.__ne__(4.4, 4.4))
    assert_true(FloatLiteral.__ne__(4.4, 42.0))

    assert_true(FloatLiteral.__gt__(10.4, 4.4))
    assert_true(FloatLiteral.__gt__(-4.4, -10.4))
    assert_false(FloatLiteral.__gt__(0.0, 0.0))
    assert_false(FloatLiteral.__gt__(4.4, 10.4))

    assert_true(FloatLiteral.__ge__(10.4, 4.4))
    assert_true(FloatLiteral.__ge__(-4.4, -10.4))
    assert_true(FloatLiteral.__ge__(4.4, 4.4))
    assert_false(FloatLiteral.__ge__(4.4, 10.4))


def main():
    test_ceil()
    test_floor()
    test_trunc()
    test_round()
    test_round10()
    test_division()
    test_exp()
    test_log()
    test_pow()
    test_mod()
    test_div_mod()
    test_int_conversion()
    test_bool()
    test_is_special_value()
    test_abs()
    test_comparison()
