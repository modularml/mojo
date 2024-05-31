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
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
    assert_true,
    assert_false,
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


fn round10(x: Float64) -> Float64:
    # TODO: implement __div__ on FloatLiteral?
    return (round(Float64(x * 10)) / 10).value


def test_round10():
    assert_equal(round10(4.4 % 0.5), 0.4)
    assert_equal(round10(-4.4 % 0.5), 0.1)
    assert_equal(round10(4.4 % -0.5), -0.1)
    assert_equal(round10(-4.4 % -0.5), -0.4)
    assert_equal(round10(3.1 % 1.0), 0.1)


def test_division():
    assert_equal(4.4 / 0.5, 8.8)

    alias f1 = 4.4 // 0.5
    assert_equal(f1, 8.0)
    alias f2 = -4.4 // 0.5
    assert_equal(f2, -9.0)
    alias f3 = 4.4 // -0.5
    assert_equal(f3, -9.0)
    alias f4 = -4.4 // -0.5
    assert_equal(f4, 8.0)


def test_power():
    assert_almost_equal(4.5**2.5, 42.95673695)
    assert_almost_equal(4.5**-2.5, 0.023279235)
    # TODO (https://github.com/modularml/modular/issues/33045): Float64/SIMD has
    # issues with negative numbers raised to fractional powers.
    # assert_almost_equal((-4.5) ** 2.5, -42.95673695)
    # assert_almost_equal((-4.5) ** -2.5, -0.023279235)


def test_mod():
    assert_equal(4.5 % 2, 0.5)
    assert_equal(-4.5 % 2, 1.5)
    assert_equal(6 % 2.5, 1.0)


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


def test_boolean_comparable():
    var f1 = 0.0
    assert_false(f1)

    var f2 = 2.0
    assert_true(f2)

    var f3 = 1.0
    assert_true(f3)


def test_equality():
    var f1 = 4.4
    var f2 = 4.4
    var f3 = 42.0
    assert_equal(f1, f2)
    assert_not_equal(f1, f3)


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


def main():
    test_ceil()
    test_floor()
    test_trunc()
    test_round()
    test_round10()
    test_division()
    test_power()
    test_mod()
    test_div_mod()
    test_int_conversion()
    test_boolean_comparable()
    test_equality()
    test_is_special_value()
    test_abs()
