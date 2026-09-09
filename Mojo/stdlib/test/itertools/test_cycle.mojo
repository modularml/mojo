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

from std.itertools import cycle
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)


def test_cycle_basic() raises:
    """Tests basic cycling through a list."""
    var nums = [1, 2, 3]
    var it = cycle(nums)

    # First cycle
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)

    # Second cycle
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)

    # Third cycle
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)


def test_cycle_strings() raises:
    """Tests cycling through a list of strings."""
    var colors = ["red", "green", "blue"]
    var it = cycle(colors)

    # First cycle
    assert_equal(next(it), "red")
    assert_equal(next(it), "green")
    assert_equal(next(it), "blue")

    # Second cycle
    assert_equal(next(it), "red")
    assert_equal(next(it), "green")
    assert_equal(next(it), "blue")


def test_cycle_single_element() raises:
    """Tests cycling through a single element."""
    var single = [42]
    var it = cycle(single)

    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)


def test_cycle_two_elements() raises:
    """Tests cycling through two elements."""
    var pair = [True, False]
    var it = cycle(pair)

    assert_true(next(it))
    assert_false(next(it))
    assert_true(next(it))
    assert_false(next(it))
    assert_true(next(it))
    assert_false(next(it))


def test_cycle_empty() raises:
    """Tests cycling through an empty list raises StopIteration."""
    var empty = List[Int]()
    var it = cycle(empty)

    with assert_raises():
        _ = next(it)


def test_cycle_empty_stays_exhausted() raises:
    """Tests that empty cycle consistently raises StopIteration."""
    var empty = List[Int]()
    var it = cycle(empty)

    # Multiple calls should all raise StopIteration
    for _ in range(5):
        with assert_raises():
            _ = next(it)


def test_cycle_is_lazy() raises:
    """Tests that cycle doesn't consume the source iterator at construction."""
    # Create a cycle but don't iterate - this should not consume the range
    var it = cycle(range(1000000))

    # If cycle were eager, it would have allocated a huge list
    # The lazy implementation just stores the iterator
    # We verify by just getting a few elements
    assert_equal(next(it), 0)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)


def test_cycle_multiple_resets() raises:
    """Tests that cycle correctly resets multiple times."""
    var nums = [1, 2]
    var it = cycle(nums)

    # Go through 5 complete cycles (10 elements)
    for _ in range(5):
        assert_equal(next(it), 1)
        assert_equal(next(it), 2)


def test_cycle_in_for_loop() raises:
    """Tests cycle iterator in a for loop with early termination."""
    var items = [10, 20, 30]
    var it = cycle(items)

    var results = List[Int]()
    var count = 0
    for val in it:
        results.append(val)
        count += 1
        if count >= 7:
            break

    assert_equal(len(results), 7)
    assert_equal(results[0], 10)
    assert_equal(results[1], 20)
    assert_equal(results[2], 30)
    assert_equal(results[3], 10)
    assert_equal(results[4], 20)
    assert_equal(results[5], 30)
    assert_equal(results[6], 10)


def test_cycle_large_count() raises:
    """Tests cycling many times."""
    var small = [1, 2]
    var it = cycle(small)

    var sum = 0
    for _ in range(1000):
        sum += next(it)

    # 1000 iterations: 500 ones and 500 twos
    assert_equal(sum, 1500)


def test_cycle_preserves_order() raises:
    """Tests that cycle preserves the original order of elements."""
    var ordered = [5, 4, 3, 2, 1]
    var it = cycle(ordered)

    # Check order is preserved across multiple cycles
    for _ in range(3):
        assert_equal(next(it), 5)
        assert_equal(next(it), 4)
        assert_equal(next(it), 3)
        assert_equal(next(it), 2)
        assert_equal(next(it), 1)


def test_cycle_from_range() raises:
    """Tests cycling through a range."""
    var it = cycle(range(3))

    assert_equal(next(it), 0)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 0)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)


def test_cycle_copyable() raises:
    """Tests that cycle iterator is copyable."""
    var nums = [1, 2, 3]
    var it1 = cycle(nums)

    # Advance it1
    assert_equal(next(it1), 1)
    assert_equal(next(it1), 2)

    # Copy the iterator
    var it2 = it1.copy()

    # Both should be at the same position
    assert_equal(next(it1), 3)
    assert_equal(next(it2), 3)

    # They should advance independently
    assert_equal(next(it1), 1)
    assert_equal(next(it2), 1)


def test_cycle_iterator_protocol() raises:
    """Tests that cycle conforms to iterator protocol."""
    var nums = [1, 2, 3]
    var it = cycle(nums)

    # __iter__ should return itself (copy)
    var it2 = iter(it)

    # Both should work
    assert_equal(next(it), 1)
    assert_equal(next(it2), 1)


def test_cycle_modulo_behavior() raises:
    """Tests that cycle correctly wraps around."""
    var items = [100, 200, 300, 400]
    var it = cycle(items)

    # Advance to various positions and verify correct element
    for i in range(20):
        var expected = items[i % 4]
        assert_equal(next(it), expected)


@fieldwise_init
struct _CopyableOwnedIter[size: Int](Copyable, IterableOwned, Iterator):
    """A `Copyable` owned iterator backed by an `Array` for testing."""

    comptime Element = Int
    comptime IteratorOwnedType: Iterator = Self
    var _data: Array[Int, Self.size]
    var _index: Int

    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Int:
        if self._index >= Self.size:
            raise StopIteration()
        var val = self._data[self._index]
        self._index += 1
        return val

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = Self.size - self._index
        return (remaining, remaining)


@fieldwise_init
struct _CopyableOwned[size: Int](IterableOwned):
    """An `IterableOwned` whose owned iterator is `Copyable` for testing."""

    comptime IteratorOwnedType: Iterator = _CopyableOwnedIter[Self.size]
    var _data: Array[Int, Self.size]

    def __iter__(var self) -> Self.IteratorOwnedType:
        return _CopyableOwnedIter(self._data.copy(), 0)


def _owned1(a: Int) -> _CopyableOwned[1]:
    var data: Array[Int, 1] = [a]
    return _CopyableOwned(data^)


def _owned3(a: Int, b: Int, c: Int) -> _CopyableOwned[3]:
    var data: Array[Int, 3] = [a, b, c]
    return _CopyableOwned(data^)


def test_cycle_owned() raises:
    var it = cycle(_owned3(1, 2, 3))

    # First cycle
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)

    # Second cycle
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)

    # Third cycle
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)


def test_cycle_owned_single() raises:
    var it = cycle(_owned1(42))

    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
