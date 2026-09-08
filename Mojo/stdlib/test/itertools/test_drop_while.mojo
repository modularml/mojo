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

from std.itertools import drop_while
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
)


def less_than_5(x: Int) -> Bool:
    return x < 5


def always_true(x: Int) -> Bool:
    return True


def always_false(x: Int) -> Bool:
    return False


def is_even(x: Int) -> Bool:
    return x % 2 == 0


def is_negative(x: Int) -> Bool:
    return x < 0


def test_drop_while_basic() raises:
    """Tests basic drop_while behavior."""
    var nums = [1, 2, 3, 4, 5, 6, 1, 2]
    var it = drop_while[less_than_5](nums)

    # Should drop 1, 2, 3, 4 and yield starting from 5
    assert_equal(next(it), 5)
    assert_equal(next(it), 6)
    assert_equal(next(it), 1)  # Yields even after predicate would be true
    assert_equal(next(it), 2)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_yields_after_first_failure() raises:
    """Tests that drop_while yields all elements after first predicate failure.
    """
    var nums = [2, 4, 6, 7, 8, 10, 3]
    var it = drop_while[is_even](nums)

    # Drops 2, 4, 6 (even), yields starting from 7 (odd)
    assert_equal(next(it), 7)
    assert_equal(next(it), 8)  # Still yields even numbers after
    assert_equal(next(it), 10)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_none_dropped() raises:
    """Tests drop_while when no elements are dropped (first fails predicate)."""
    var nums = [5, 1, 2, 3, 4]
    var it = drop_while[less_than_5](nums)

    # 5 fails predicate immediately, so nothing dropped
    assert_equal(next(it), 5)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_all_dropped() raises:
    """Tests drop_while when all elements are dropped."""
    var nums = [1, 2, 3, 4]
    var it = drop_while[less_than_5](nums)

    # All elements satisfy less_than_5
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_empty() raises:
    """Tests drop_while on an empty list."""
    var empty = List[Int]()
    var it = drop_while[less_than_5](empty)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_single_dropped() raises:
    """Tests drop_while with single element that is dropped."""
    var nums = [3]
    var it = drop_while[less_than_5](nums)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_single_kept() raises:
    """Tests drop_while with single element that is kept."""
    var nums = [10]
    var it = drop_while[less_than_5](nums)

    assert_equal(next(it), 10)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_in_for_loop() raises:
    """Tests drop_while iterator in a for loop."""
    var nums = [1, 2, 3, 4, 5, 6, 7]
    var results = List[Int]()

    for num in drop_while[less_than_5](nums):
        results.append(num)

    assert_equal(len(results), 3)
    assert_equal(results[0], 5)
    assert_equal(results[1], 6)
    assert_equal(results[2], 7)


def test_drop_while_always_true() raises:
    """Tests drop_while with a predicate that always returns True."""
    var nums = [1, 2, 3]
    var it = drop_while[always_true](nums)

    # All elements dropped
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_always_false() raises:
    """Tests drop_while with a predicate that always returns False."""
    var nums = [1, 2, 3]
    var it = drop_while[always_false](nums)

    # No elements dropped
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_from_range() raises:
    """Tests drop_while on a range."""
    var it = drop_while[less_than_5](range(10))

    # Drops 0, 1, 2, 3, 4
    assert_equal(next(it), 5)
    assert_equal(next(it), 6)
    assert_equal(next(it), 7)
    assert_equal(next(it), 8)
    assert_equal(next(it), 9)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_negative_numbers() raises:
    """Tests drop_while with negative numbers."""
    var nums = [-3, -2, -1, 0, 1, 2, -5]
    var it = drop_while[is_negative](nums)

    # Drops -3, -2, -1, yields from 0
    assert_equal(next(it), 0)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), -5)  # Yields negative after dropping phase
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_only_yields_once_triggered() raises:
    """Tests that drop_while yields all elements after first failure."""
    var nums = [0, 0, 0, 1, 0, 0, 0]

    def is_zero(x: Int) -> Bool:
        return x == 0

    var it = drop_while[is_zero](nums)

    # Drops the first three 0s, yields from 1 onwards
    assert_equal(next(it), 1)
    assert_equal(next(it), 0)  # Still yields zeros after
    assert_equal(next(it), 0)
    assert_equal(next(it), 0)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_while_owned() raises:
    var l: List[Int] = [1, 2, 3, 4, 5, 6, 1, 2]
    var it = drop_while[less_than_5](l^)
    assert_equal(next(it), 5)
    assert_equal(next(it), 6)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
