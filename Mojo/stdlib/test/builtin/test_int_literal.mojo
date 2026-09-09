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


def test_add() raises:
    assert_equal(IntLiteral.__add__(3, 3), 6)
    assert_equal(IntLiteral.__add__(-2, 3), 1)
    assert_equal(IntLiteral.__add__(2, -3), -1)
    assert_equal(IntLiteral.__add__(5, -5), 0)
    assert_equal(IntLiteral.__add__(-5, -4), -9)


def test_sub() raises:
    assert_equal(IntLiteral.__sub__(3, 3), 0)
    assert_equal(IntLiteral.__sub__(-2, 3), -5)
    assert_equal(IntLiteral.__sub__(2, -3), 5)
    assert_equal(IntLiteral.__sub__(5, 4), 1)
    assert_equal(IntLiteral.__sub__(4, 5), -1)


def test_ceil() raises:
    assert_equal(IntLiteral.__ceil__(5), 5)
    assert_equal(IntLiteral.__ceil__(0), 0)
    assert_equal(IntLiteral.__ceil__(-5), -5)


def test_floor() raises:
    assert_equal(IntLiteral.__floor__(5), 5)
    assert_equal(IntLiteral.__floor__(0), 0)
    assert_equal(IntLiteral.__floor__(-5), -5)


def test_trunc() raises:
    assert_equal(IntLiteral.__trunc__(5), 5)
    assert_equal(IntLiteral.__trunc__(0), 0)
    assert_equal(IntLiteral.__trunc__(-5), -5)


def test_floordiv() raises:
    assert_equal(IntLiteral.__floordiv__(2, 2), 1)
    assert_equal(IntLiteral.__floordiv__(2, 3), 0)
    assert_equal(IntLiteral.__floordiv__(2, -2), -1)
    assert_equal(IntLiteral.__floordiv__(99, -2), -50)


def test_mod() raises:
    assert_equal(IntLiteral.__mod__(99, 1), 0)
    assert_equal(IntLiteral.__mod__(99, 3), 0)
    assert_equal(IntLiteral.__mod__(99, -2), -1)
    assert_equal(IntLiteral.__mod__(99, 8), 3)
    assert_equal(IntLiteral.__mod__(99, -8), -5)
    assert_equal(IntLiteral.__mod__(2, -1), 0)
    assert_equal(IntLiteral.__mod__(2, -2), 0)
    assert_equal(IntLiteral.__mod__(3, -2), -1)
    assert_equal(IntLiteral.__mod__(-3, 2), 1)


def test_abs() raises:
    assert_equal(abs(-5), 5)
    assert_equal(abs(2), 2)
    assert_equal(abs(0), 0)


def test_indexer() raises:
    assert_true(1 == index(1))
    assert_true(88 == index(88))


def test_bool() raises:
    assert_true(IntLiteral.__bool__(5))
    assert_false(IntLiteral.__bool__(0))


def test_comparison() raises:
    assert_true((5).__lt__(10))
    assert_true((-10).__lt__(-5))
    assert_false((0).__lt__(0))
    assert_false((10).__lt__(5))

    assert_true((5).__le__(10))
    assert_true((-10).__le__(-5))
    assert_true((0).__le__(0))
    assert_false((10).__le__(5))

    assert_true((5).__eq__(5))
    assert_true((0).__eq__(0))
    assert_false((0).__eq__(1))
    assert_false((5).__eq__(10))

    assert_true((5).__ne__(10))
    assert_true((0).__ne__(1))
    assert_false((5).__ne__(5))
    assert_false((0).__ne__(0))

    assert_true((10).__gt__(5))
    assert_true((-5).__gt__(-10))
    assert_false((0).__gt__(0))
    assert_false((5).__gt__(10))

    assert_true((10).__ge__(5))
    assert_true((5).__ge__(5))
    assert_true((-5).__ge__(-10))
    assert_false((5).__ge__(10))


def test_shift() raises:
    assert_equal(IntLiteral.__lshift__(1, -42), 0)  # Dubious.
    assert_equal(IntLiteral.__lshift__(1, 0), 1)
    assert_equal(IntLiteral.__lshift__(1, 1), 2)
    assert_equal(IntLiteral.__lshift__(1, 6), 64)


def test_pow() raises:
    assert_equal(IntLiteral.__pow__(2, 64), 18446744073709551616)
    assert_equal(
        IntLiteral.__pow__(2, 255),
        57896044618658097711785492504343953926634992332820282019728792003956564819968,
    )
    assert_equal(
        IntLiteral.__pow__(2, 256),
        115792089237316195423570985008687907853269984665640564039457584007913129639936,
    )
    assert_equal(IntLiteral.__pow__(64, 0), 1)
    # Cannot exponentiate by a negative amount...
    assert_equal(IntLiteral.__pow__(64, -1), 0)
    # ... except for 1, which is always 1...
    assert_equal(IntLiteral.__pow__(1, -9), 1)
    # ... and -1, which is either 1 or -1
    assert_equal(IntLiteral.__pow__(-1, -3), -1)
    assert_equal(IntLiteral.__pow__(-1, -2), 1)


def test_add_float_literal() raises:
    assert_equal(2 + 0.5, 2.5)
    assert_equal(IntLiteral.__add__(3, 2.0), 5.0)
    assert_equal(IntLiteral.__add__(-2, 0.5), -1.5)


def test_sub_float_literal() raises:
    assert_equal(2 - 0.5, 1.5)
    assert_equal(IntLiteral.__sub__(3, 2.0), 1.0)
    assert_equal(IntLiteral.__sub__(-2, 0.5), -2.5)


def test_mul_float_literal() raises:
    assert_equal(2 * 0.5, 1.0)
    assert_equal(IntLiteral.__mul__(3, 2.0), 6.0)
    assert_equal(IntLiteral.__mul__(-2, 0.5), -1.0)


def test_truediv_float_literal() raises:
    assert_equal(2 / 0.5, 4.0)
    assert_equal(IntLiteral.__truediv__(3, 2.0), 1.5)
    assert_equal(IntLiteral.__truediv__(-2, 0.5), -4.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
