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


def test_division():
    # TODO: https://github.com/modularml/mojo/issues/1787
    # allow this at compile time
    assert_equal(FloatLiteral(4.4) / 0.5, 8.8)
    assert_equal(FloatLiteral(4.4) // 0.5, 8.0)
    assert_equal(FloatLiteral(-4.4) // 0.5, -9.0)
    assert_equal(FloatLiteral(4.4) // -0.5, -9.0)
    assert_equal(FloatLiteral(-4.4) // -0.5, 8.0)


def test_power():
    assert_almost_equal(FloatLiteral(4.5) ** 2.5, 42.95673695)
    assert_almost_equal(FloatLiteral(4.5) ** -2.5, 0.023279235)
    # TODO (https://github.com/modularml/modular/issues/33045): Float64/SIMD has
    # issues with negative numbers raised to fractional powers.
    # assert_almost_equal(FloatLiteral(-4.5) ** 2.5, -42.95673695)
    # assert_almost_equal(FloatLiteral(-4.5) ** -2.5, -0.023279235)


def test_int_conversion():
    assert_equal(int(FloatLiteral(-4.0)), -4)
    assert_equal(int(FloatLiteral(-4.5)), -4)
    assert_equal(int(FloatLiteral(-4.3)), -4)
    assert_equal(int(FloatLiteral(4.5)), 4)
    assert_equal(int(FloatLiteral(4.0)), 4)


def test_boolean_comparable():
    var f1 = FloatLiteral(0.0)
    assert_false(f1)

    var f2 = FloatLiteral(2.0)
    assert_true(f2)

    var f3 = FloatLiteral(1.0)
    assert_true(f2)


def test_equality():
    var f1 = FloatLiteral(4.4)
    var f2 = FloatLiteral(4.4)
    var f3 = FloatLiteral(42.0)
    assert_equal(f1, f2)
    assert_not_equal(f1, f3)


def test_is_special_value():
    assert_true(nan.is_nan())
    assert_false(neg_zero.is_nan())
    assert_true(neg_zero.is_neg_zero())
    assert_false(nan.is_neg_zero())


def test_abs():
    assert_equal(FloatLiteral(-4.4).__abs__(), 4.4)
    assert_equal(FloatLiteral(4.4).__abs__(), 4.4)
    assert_equal(FloatLiteral(0.0).__abs__(), 0.0)

    assert_true(FloatLiteral.__abs__(nan).is_nan())
    assert_false(FloatLiteral.__abs__(neg_zero).is_neg_zero())
    assert_equal(FloatLiteral.__abs__(neg_zero), 0.0)
    assert_equal(FloatLiteral.__abs__(inf), inf)
    assert_equal(FloatLiteral.__abs__(neg_inf), inf)


def main():
    test_floor()
    test_division()
    test_power()
    test_int_conversion()
    test_boolean_comparable()
    test_equality()
    test_is_special_value()
    test_abs()
