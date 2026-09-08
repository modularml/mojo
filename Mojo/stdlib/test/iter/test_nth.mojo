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

from std.itertools import count
from std.testing import TestSuite, assert_equal, assert_false, assert_true


def test_nth_borrowed_basic() raises:
    var l = [10, 20, 30, 40]
    assert_equal(iter(l).nth(0).value(), 10)
    assert_equal(iter(l).nth(1).value(), 20)
    assert_equal(iter(l).nth(3).value(), 40)


def test_nth_borrowed_does_not_consume() raises:
    var l = [1, 2, 3]
    _ = iter(l).nth(0)
    _ = iter(l).nth(2)
    # The original list is untouched; we can still index normally.
    assert_equal(len(l), 3)
    assert_equal(l[0], 1)
    assert_equal(l[2], 3)


def test_nth_past_end_returns_none() raises:
    var l = [1, 2, 3]
    var result = iter(l).nth(3)
    assert_false(result)


def test_nth_far_past_end_returns_none() raises:
    var l = [1, 2, 3]
    var result = iter(l).nth(1000)
    assert_false(result)


def test_nth_empty_returns_none() raises:
    var l = List[Int]()
    var result = iter(l).nth(0)
    assert_false(result)
    var result2 = iter(l).nth(5)
    assert_false(result2)


def test_nth_owned() raises:
    var l: List[Int] = [100, 200, 300]
    var result = iter(l^).nth(1)
    assert_true(result)
    assert_equal(result.value(), 200)


def test_nth_owned_past_end() raises:
    var l: List[Int] = [1, 2]
    var result = iter(l^).nth(5)
    assert_false(result)


def test_nth_string_elements() raises:
    var l = [String("a"), String("b"), String("c")]
    assert_equal(iter(l).nth(1).value(), "b")


def test_nth_on_infinite_iterator() raises:
    # `count()` is infinite; `nth` over it should still terminate.
    # count(start=10, step=2) yields 10, 12, 14, 16, 18, 20, ...; nth(5) is 20.
    var result = count(start=10, step=2).nth(5)
    assert_equal(result.value(), 20)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
