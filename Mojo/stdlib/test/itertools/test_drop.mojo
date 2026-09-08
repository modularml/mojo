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

from std.itertools import drop
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
)


def test_drop_basic() raises:
    """Tests basic drop behavior."""
    var nums = [1, 2, 3, 4, 5]
    var it = drop(nums, 2)

    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    assert_equal(next(it), 5)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_more_than_available() raises:
    """Tests drop when count exceeds iterable length."""
    var nums = [1, 2, 3]
    var it = drop(nums, 10)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_zero() raises:
    """Tests drop with count=0 yields all elements."""
    var nums = [1, 2, 3]
    var it = drop(nums, 0)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_empty() raises:
    """Tests drop on an empty iterable."""
    var empty = List[Int]()
    var it = drop(empty, 5)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_all() raises:
    """Tests drop with count equal to iterable length."""
    var nums = [10, 20, 30]
    var it = drop(nums, 3)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_single() raises:
    """Tests drop with count=1."""
    var nums = [42, 99]
    var it = drop(nums, 1)

    assert_equal(next(it), 99)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_in_for_loop() raises:
    """Tests drop iterator in a for loop."""
    var nums = [1, 2, 3, 4, 5]
    var results = List[Int]()

    for num in drop(nums, 2):
        results.append(num)

    assert_equal(len(results), 3)
    assert_equal(results[0], 3)
    assert_equal(results[1], 4)
    assert_equal(results[2], 5)


def test_drop_from_range() raises:
    """Tests drop on a range."""
    var it = drop(range(5), 3)

    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_then_take_composition() raises:
    """Tests composing drop and take to select a middle slice."""
    from std.itertools import take

    var nums = [1, 2, 3, 4, 5, 6, 7]
    var results = List[Int]()

    # drop(1).take(3) equivalent: elements at indices 1, 2, 3
    for num in take(drop(nums, 1), 3):
        results.append(num)

    assert_equal(len(results), 3)
    assert_equal(results[0], 2)
    assert_equal(results[1], 3)
    assert_equal(results[2], 4)


def test_drop_bounds() raises:
    """Tests bounds() on a drop iterator."""
    var nums = [1, 2, 3, 4, 5]

    # count less than iterable length
    var it1 = drop(nums, 2)
    var lower1, upper1 = it1.bounds()
    assert_equal(lower1, 3)
    assert_equal(upper1.value(), 3)

    # count greater than iterable length
    var it2 = drop(nums, 10)
    var lower2, upper2 = it2.bounds()
    assert_equal(lower2, 0)
    assert_equal(upper2.value(), 0)

    # count == 0
    var it3 = drop(nums, 0)
    var lower3, upper3 = it3.bounds()
    assert_equal(lower3, 5)
    assert_equal(upper3.value(), 5)

    # empty iterable
    var empty = List[Int]()
    var it4 = drop(empty, 5)
    var lower4, upper4 = it4.bounds()
    assert_equal(lower4, 0)
    assert_equal(upper4.value(), 0)


def test_drop_owned() raises:
    """Tests the consuming (`var`) overload of `drop`."""
    var nums: List[Int] = [1, 2, 3, 4, 5]
    var it = drop(nums^, 2)

    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    assert_equal(next(it), 5)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_copy_independent_advance() raises:
    """Tests that `.copy()` on a drop iterator produces an independent cursor.
    """
    var nums = [1, 2, 3, 4, 5]
    var it1 = drop(nums, 1)

    # Pull one value; this forces the inner iterator past the dropped prefix.
    assert_equal(next(it1), 2)

    var it2 = it1.copy()

    # Both iterators agree on the next yielded value.
    assert_equal(next(it1), 3)
    assert_equal(next(it2), 3)

    # The two iterators advance independently.
    assert_equal(next(it1), 4)
    assert_equal(next(it2), 4)


def test_drop_exhausted_stays_exhausted() raises:
    """Tests that once exhausted, drop stays exhausted across repeated calls."""
    var nums = [1, 2, 3]
    var it = drop(nums, 5)

    with assert_raises(contains="StopIteration"):
        _ = next(it)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_drop_nested() raises:
    """Tests composing `drop` on top of another `drop`."""
    var nums = [1, 2, 3, 4, 5]
    var results = List[Int]()

    for num in drop(drop(nums, 1), 2):
        results.append(num)

    assert_equal(len(results), 2)
    assert_equal(results[0], 4)
    assert_equal(results[1], 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
