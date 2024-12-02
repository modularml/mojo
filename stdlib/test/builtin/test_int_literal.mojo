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

from testing import assert_equal, assert_false, assert_true


def test_add():
    assert_equal(IntLiteral.__add__(3, 3), 6)
    assert_equal(IntLiteral.__add__(-2, 3), 1)
    assert_equal(IntLiteral.__add__(2, -3), -1)
    assert_equal(IntLiteral.__add__(5, -5), 0)
    assert_equal(IntLiteral.__add__(-5, -4), -9)


def test_sub():
    assert_equal(IntLiteral.__sub__(3, 3), 0)
    assert_equal(IntLiteral.__sub__(-2, 3), -5)
    assert_equal(IntLiteral.__sub__(2, -3), 5)
    assert_equal(IntLiteral.__sub__(5, 4), 1)
    assert_equal(IntLiteral.__sub__(4, 5), -1)


def test_ceil():
    assert_equal(IntLiteral.__ceil__(5), 5)
    assert_equal(IntLiteral.__ceil__(0), 0)
    assert_equal(IntLiteral.__ceil__(-5), -5)


def test_floor():
    assert_equal(IntLiteral.__floor__(5), 5)
    assert_equal(IntLiteral.__floor__(0), 0)
    assert_equal(IntLiteral.__floor__(-5), -5)


def test_round():
    assert_equal(IntLiteral.__round__(5), 5)
    assert_equal(IntLiteral.__round__(0), 0)
    assert_equal(IntLiteral.__round__(-5), -5)
    assert_equal(IntLiteral.__round__(5, 1), 5)
    assert_equal(IntLiteral.__round__(0, 1), 0)
    assert_equal(IntLiteral.__round__(-5, 1), -5)
    assert_equal(IntLiteral.__round__(100, -2), 100)


def test_trunc():
    assert_equal(IntLiteral.__trunc__(5), 5)
    assert_equal(IntLiteral.__trunc__(0), 0)
    assert_equal(IntLiteral.__trunc__(-5), -5)


def test_floordiv():
    assert_equal(IntLiteral.__floordiv__(2, 2), 1)
    assert_equal(IntLiteral.__floordiv__(2, 3), 0)
    assert_equal(IntLiteral.__floordiv__(2, -2), -1)
    assert_equal(IntLiteral.__floordiv__(99, -2), -50)


def test_mod():
    assert_equal(IntLiteral.__mod__(99, 1), 0)
    assert_equal(IntLiteral.__mod__(99, 3), 0)
    assert_equal(IntLiteral.__mod__(99, -2), -1)
    assert_equal(IntLiteral.__mod__(99, 8), 3)
    assert_equal(IntLiteral.__mod__(99, -8), -5)
    assert_equal(IntLiteral.__mod__(2, -1), 0)
    assert_equal(IntLiteral.__mod__(2, -2), 0)
    assert_equal(IntLiteral.__mod__(3, -2), -1)
    assert_equal(IntLiteral.__mod__(-3, 2), 1)


def test_bit_width():
    assert_equal((0)._bit_width(), 1)
    assert_equal((-1)._bit_width(), 1)
    assert_equal((255)._bit_width(), 9)
    assert_equal((-256)._bit_width(), 9)


def test_abs():
    assert_equal(IntLiteral.__abs__(-5), 5)
    assert_equal(IntLiteral.__abs__(2), 2)
    assert_equal(IntLiteral.__abs__(0), 0)


def test_indexer():
    assert_equal(1, IntLiteral.__index__(1))
    assert_equal(88, IntLiteral.__index__(88))


def test_divmod():
    alias t0 = IntLiteral.__divmod__(2, 2)
    alias q0 = t0[0]
    alias r0 = t0[1]
    assert_equal(q0, 1)
    assert_equal(r0, 0)

    alias t1 = IntLiteral.__divmod__(2, 3)
    alias q1 = t1[0]
    alias r1 = t1[1]
    assert_equal(q1, 0)
    assert_equal(r1, 2)

    alias t2 = IntLiteral.__divmod__(99, -2)
    alias q2 = t2[0]
    alias r2 = t2[1]
    assert_equal(q2, -50)
    assert_equal(r2, -1)


def test_bool():
    assert_true(IntLiteral.__bool__(5))
    assert_false(IntLiteral.__bool__(0))
    assert_true(IntLiteral.__as_bool__(5))
    assert_false(IntLiteral.__as_bool__(0))


def test_comparison():
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


def main():
    test_add()
    test_sub()
    test_ceil()
    test_floor()
    test_round()
    test_trunc()
    test_floordiv()
    test_mod()
    test_divmod()
    test_bit_width()
    test_abs()
    test_indexer()
    test_bool()
    test_comparison()
