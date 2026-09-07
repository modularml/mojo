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

from std.testing import TestSuite, assert_equal, assert_true, assert_false

from std.utils import StaticTuple


def test_getitem() raises:
    # Should be constructible from a single element
    # as well as a variadic list of elements.
    var tup1 = StaticTuple[Int, 1](1)
    assert_equal(tup1[0], 1)

    var tup2 = StaticTuple[Int, 2](1, 1)
    assert_equal(tup2[0], 1)
    assert_equal(tup2[1], 1)

    var tup3 = StaticTuple[Int, 3](1, 2, 3)
    assert_equal(tup3[0], 1)
    assert_equal(tup3[1], 2)
    assert_equal(tup3[2], 3)

    assert_equal(tup1[Int(0)], 1)


def test_setitem() raises:
    var t = StaticTuple[Int, 3](1, 2, 3)

    t[0] = 100
    assert_equal(t[0], 100)

    t[1] = 200
    assert_equal(t[1], 200)

    t[2] = 300
    assert_equal(t[2], 300)

    comptime idx: Int = 0
    t[idx] = 400
    assert_equal(t[0], 400)


def test_equality() raises:
    var a = StaticTuple[Int, 3](1, 2, 3)
    var b = StaticTuple[Int, 3](1, 2, 3)
    var c = StaticTuple[Int, 3](1, 2, 4)

    assert_true(a == b)
    assert_false(a == c)
    assert_false(a != b)
    assert_true(a != c)


def test_comparison() raises:
    var a = StaticTuple[Int, 3](1, 2, 3)
    var b = StaticTuple[Int, 3](1, 2, 4)
    var c = StaticTuple[Int, 3](1, 2, 3)
    var d = StaticTuple[Int, 3](2, 0, 0)

    # Less than
    assert_true(a < b)
    assert_false(b < a)
    assert_false(a < c)

    # First element dominates
    assert_true(a < d)
    assert_false(d < a)

    # Less than or equal
    assert_true(a <= c)
    assert_true(a <= b)
    assert_false(b <= a)

    # Greater than
    assert_true(b > a)
    assert_false(a > b)
    assert_false(a > c)

    # Greater than or equal
    assert_true(a >= c)
    assert_true(b >= a)
    assert_false(a >= b)

    # Empty tuple
    var e1 = StaticTuple[Int, 0]()
    var e2 = StaticTuple[Int, 0]()
    assert_true(e1 == e2)
    assert_false(e1 < e2)
    assert_true(e1 <= e2)
    assert_true(e1 >= e2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
