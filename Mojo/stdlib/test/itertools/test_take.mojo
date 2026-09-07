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

from std.itertools import take
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
)


def test_take_basic() raises:
    """Tests basic take behavior."""
    var nums = [1, 2, 3, 4, 5]
    var it = take(nums, 3)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_more_than_available() raises:
    """Tests take when count exceeds iterable length."""
    var nums = [1, 2, 3]
    var it = take(nums, 10)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_zero() raises:
    """Tests take with count=0 yields nothing."""
    var nums = [1, 2, 3]
    var it = take(nums, 0)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_empty() raises:
    """Tests take on an empty iterable."""
    var empty = List[Int]()
    var it = take(empty, 5)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_all() raises:
    """Tests take with count equal to iterable length."""
    var nums = [10, 20, 30]
    var it = take(nums, 3)

    assert_equal(next(it), 10)
    assert_equal(next(it), 20)
    assert_equal(next(it), 30)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_single() raises:
    """Tests take with count=1."""
    var nums = [42, 99]
    var it = take(nums, 1)

    assert_equal(next(it), 42)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_in_for_loop() raises:
    """Tests take iterator in a for loop."""
    var nums = [1, 2, 3, 4, 5]
    var results = List[Int]()

    for num in take(nums, 3):
        results.append(num)

    assert_equal(len(results), 3)
    assert_equal(results[0], 1)
    assert_equal(results[1], 2)
    assert_equal(results[2], 3)


def test_take_from_range() raises:
    """Tests take on a range."""
    var it = take(range(100), 4)

    assert_equal(next(it), 0)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_exhausted_stays_exhausted() raises:
    """Tests that once exhausted, take stays exhausted."""
    var nums = [1, 2, 3]
    var it = take(nums, 2)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    with assert_raises(contains="StopIteration"):
        _ = next(it)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_bounds() raises:
    """Tests bounds() on a take iterator."""
    var nums = [1, 2, 3, 4, 5]

    # count less than iterable length
    var it1 = take(nums, 3)
    var lower1, upper1 = it1.bounds()
    assert_equal(lower1, 3)
    assert_equal(upper1.value(), 3)

    # count greater than iterable length
    var it2 = take(nums, 10)
    var lower2, upper2 = it2.bounds()
    assert_equal(lower2, 5)
    assert_equal(upper2.value(), 5)

    # count == 0
    var it3 = take(nums, 0)
    var lower3, upper3 = it3.bounds()
    assert_equal(lower3, 0)
    assert_equal(upper3.value(), 0)

    # empty iterable
    var empty = List[Int]()
    var it4 = take(empty, 5)
    var lower4, upper4 = it4.bounds()
    assert_equal(lower4, 0)
    assert_equal(upper4.value(), 0)


def test_take_owned() raises:
    """Tests the consuming (`var`) overload of `take`."""
    var nums: List[Int] = [1, 2, 3, 4, 5]
    var it = take(nums^, 3)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_take_copy_independent_advance() raises:
    """Tests that `.copy()` on a take iterator produces an independent cursor.
    """
    var nums = [1, 2, 3, 4, 5]
    var it1 = take(nums, 4)

    assert_equal(next(it1), 1)
    assert_equal(next(it1), 2)

    var it2 = it1.copy()

    # The copy picks up at the same position.
    assert_equal(next(it1), 3)
    assert_equal(next(it2), 3)

    # The two iterators advance independently.
    assert_equal(next(it1), 4)
    assert_equal(next(it2), 4)

    with assert_raises(contains="StopIteration"):
        _ = next(it1)
    with assert_raises(contains="StopIteration"):
        _ = next(it2)


def test_take_nested() raises:
    """Tests composing `take` on top of another `take`."""
    var nums = [1, 2, 3, 4, 5]
    var results = List[Int]()

    for num in take(take(nums, 4), 2):
        results.append(num)

    assert_equal(len(results), 2)
    assert_equal(results[0], 1)
    assert_equal(results[1], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
