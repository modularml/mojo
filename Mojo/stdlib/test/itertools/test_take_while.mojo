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

from std.itertools import take_while
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
)


def is_positive(x: Int) -> Bool:
    return x > 0


def less_than_5(x: Int) -> Bool:
    return x < 5


def always_true(x: Int) -> Bool:
    return True


def always_false(x: Int) -> Bool:
    return False


def is_even(x: Int) -> Bool:
    return x % 2 == 0


def test_take_while_basic() raises:
    """Tests basic take_while behavior."""
    var nums = [1, 2, 3, 4, 5, 6, 7]
    var it = take_while[less_than_5](nums)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)  # 5 fails predicate


def test_take_while_stops_at_first_failure() raises:
    """Tests that take_while stops at the first element failing the predicate.
    """
    var nums = [2, 4, 6, 7, 8, 10]  # 7 is odd
    var it = take_while[is_even](nums)

    assert_equal(next(it), 2)
    assert_equal(next(it), 4)
    assert_equal(next(it), 6)
    with assert_raises(contains="StopIteration"):
        _ = next(it)  # 7 is odd, stops


def test_take_while_all_pass() raises:
    """Tests take_while when all elements pass the predicate."""
    var nums = [1, 2, 3, 4]
    var it = take_while[less_than_5](nums)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)  # exhausted


def test_take_while_none_pass() raises:
    """Tests take_while when no elements pass the predicate."""
    var nums = [5, 6, 7, 8]
    var it = take_while[less_than_5](nums)

    with assert_raises(contains="StopIteration"):
        _ = next(it)  # first element fails


def test_take_while_empty() raises:
    """Tests take_while on an empty list."""
    var empty = List[Int]()
    var it = take_while[less_than_5](empty)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_single_pass() raises:
    """Tests take_while with single element that passes."""
    var nums = [3]
    var it = take_while[less_than_5](nums)

    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_single_fail() raises:
    """Tests take_while with single element that fails."""
    var nums = [10]
    var it = take_while[less_than_5](nums)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_in_for_loop() raises:
    """Tests take_while iterator in a for loop."""
    var nums = [1, 2, 3, 4, 5, 6, 7]
    var results = List[Int]()

    for num in take_while[less_than_5](nums):
        results.append(num)

    assert_equal(len(results), 4)
    assert_equal(results[0], 1)
    assert_equal(results[1], 2)
    assert_equal(results[2], 3)
    assert_equal(results[3], 4)


def test_take_while_always_true() raises:
    """Tests take_while with a predicate that always returns True."""
    var nums = [1, 2, 3]
    var it = take_while[always_true](nums)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_always_false() raises:
    """Tests take_while with a predicate that always returns False."""
    var nums = [1, 2, 3]
    var it = take_while[always_false](nums)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_is_positive() raises:
    """Tests take_while with is_positive predicate."""
    var nums = [5, 3, 1, 0, -1, 2]
    var it = take_while[is_positive](nums)

    assert_equal(next(it), 5)
    assert_equal(next(it), 3)
    assert_equal(next(it), 1)
    with assert_raises(contains="StopIteration"):
        _ = next(it)  # 0 is not positive


def test_take_while_from_range() raises:
    """Tests take_while on a range."""
    var it = take_while[less_than_5](range(10))

    assert_equal(next(it), 0)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_exhausted_stays_exhausted() raises:
    """Tests that once exhausted, take_while stays exhausted."""
    var nums = [1, 2, 10, 3, 4]
    var it = take_while[less_than_5](nums)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    with assert_raises(contains="StopIteration"):
        _ = next(it)  # 10 fails

    # Should still be exhausted
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_while_owned() raises:
    var l: List[Int] = [1, 2, 3, 4, 5, 6]
    var it = take_while[less_than_5](l^)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
