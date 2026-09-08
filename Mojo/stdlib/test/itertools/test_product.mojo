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

from std.itertools import product
from std.itertools.itertools import _Product2, _Product3, _Product4
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)
from test_utils import Observable, CopyCounter


def test_product2() raises:
    var l1 = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]

    var it = product(l1, l2)

    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)

    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_product2_param() raises:
    var trip_count = 0

    comptime for i, j in product(range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 4)


def test_product2_unequal() raises:
    """Checks the product if the two input iterators are unequal."""
    var l1 = ["hey", "hi", "hello", "holla"]
    var l2 = [10, 20, 30]

    var it = product(l1, l2)

    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)

    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "holla")
    assert_equal(elem[1], 30)

    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_product3() raises:
    """Tests the product of three iterables."""
    var l1 = [1, 2]
    var l2 = [10, 20]
    var l3 = [100, 200]

    var it = product(l1, l2, l3)

    # Expected order: (1,10,100), (1,10,200), (1,20,100), (1,20,200),
    #                 (2,10,100), (2,10,200), (2,20,100), (2,20,200)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)

    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_product3_param() raises:
    """Tests the product of three iterables with parameter for loop."""
    var trip_count = 0

    comptime for i, j, k in product(range(2), range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        assert_true(k in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 8)


def test_product4() raises:
    """Tests the product of four iterables."""
    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    var l4 = [7, 8]

    var count = 0
    for a, b, c, d in product(l1, l2, l3, l4):
        assert_true(a in (1, 2))
        assert_true(b in (3, 4))
        assert_true(c in (5, 6))
        assert_true(d in (7, 8))
        count += 1

    # Should have 2 * 2 * 2 * 2 = 16 elements
    assert_equal(count, 16)


def test_product4_param() raises:
    """Tests the product of four iterables with parameter for loop."""
    var trip_count = 0

    comptime for i, j, k, l in product(range(2), range(2), range(2), range(2)):
        assert_true(i in (0, 1))
        assert_true(j in (0, 1))
        assert_true(k in (0, 1))
        assert_true(l in (0, 1))
        trip_count += 1

    assert_equal(trip_count, 16)


def test_product4_order() raises:
    """Tests that product(a, b, c, d) produces elements in the correct order."""
    var l1 = [0, 1]
    var l2 = [0, 1]
    var l3 = [0, 1]
    var l4 = [0, 1]

    var it = product(l1, l2, l3, l4)

    # Rightmost iterator varies fastest
    var elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 0)
    assert_equal(elem[2], 0)
    assert_equal(elem[3], 0)

    elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 0)
    assert_equal(elem[2], 0)
    assert_equal(elem[3], 1)

    elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 0)
    assert_equal(elem[2], 1)
    assert_equal(elem[3], 0)


def test_product_bounds() raises:
    var it = product(range(3), range(2))
    for i in range(6, -1, -1):
        assert_equal(it.bounds()[0], i)
        assert_equal(it.bounds()[1].value(), i)
        try:
            _ = next(it)
        except:
            pass


struct TestCopyIterator[
    CopyOrigin: MutOrigin,
](Copyable, Iterator):
    comptime Element = NoneType

    var counter: Observable[CopyOrigin=Self.CopyOrigin]

    def __init__(out self, ref[Self.CopyOrigin] copies: Int):
        self.counter = Observable[CopyOrigin=Self.CopyOrigin](
            copies=Pointer(to=copies)
        )

    def __next__(mut self) raises StopIteration -> Self.Element:
        return None


def test_product2_copies() raises:
    var copy_a = 0
    var copy_b = 0
    var _product_iter = _Product2(
        TestCopyIterator(copy_a), TestCopyIterator(copy_b)
    )
    assert_equal(copy_a, 0)
    assert_equal(copy_b, 1)


def test_product3_copies() raises:
    var copy_a = 0
    var copy_b = 0
    var copy_c = 0
    var _product_iter = _Product3(
        TestCopyIterator(copy_a),
        TestCopyIterator(copy_b),
        TestCopyIterator(copy_c),
    )
    assert_equal(copy_a, 0)
    assert_equal(copy_b, 1)
    assert_equal(copy_c, 3)


def test_product4_copies() raises:
    var copy_a = 0
    var copy_b = 0
    var copy_c = 0
    var copy_d = 0
    var _product_iter = _Product4(
        TestCopyIterator(copy_a),
        TestCopyIterator(copy_b),
        TestCopyIterator(copy_c),
        TestCopyIterator(copy_d),
    )
    assert_equal(copy_a, 0)
    assert_equal(copy_b, 1)
    assert_equal(copy_c, 3)
    assert_equal(copy_d, 7)


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


def _owned2(a: Int, b: Int) -> _CopyableOwned[2]:
    var data: Array[Int, 2] = [a, b]
    return _CopyableOwned(data^)


def _owned3(a: Int, b: Int, c: Int) -> _CopyableOwned[3]:
    var data: Array[Int, 3] = [a, b, c]
    return _CopyableOwned(data^)


def test_product2_owned() raises:
    var it = product(_owned3(1, 2, 3), _owned2(10, 20))

    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)

    elem = next(it)
    assert_equal(elem[0], 3)
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], 3)
    assert_equal(elem[1], 20)

    with assert_raises():
        _ = it.__next__()


def test_product3_owned() raises:
    var it = product(_owned2(1, 2), _owned2(10, 20), _owned2(100, 200))

    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 200)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 100)

    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)

    with assert_raises():
        _ = it.__next__()


def test_product4_owned() raises:
    var count = 0
    for a, b, c, d in product(
        _owned2(1, 2), _owned2(3, 4), _owned2(5, 6), _owned2(7, 8)
    ):
        assert_true(a in (1, 2))
        assert_true(b in (3, 4))
        assert_true(c in (5, 6))
        assert_true(d in (7, 8))
        count += 1

    assert_equal(count, 16)


# The `_Product{3,4}` iterators flatten a right-nested tuple on each `next()`.
# That flatten moves elements out rather than copying them, so a yielded element
# carries only the copies the product machinery makes while replaying the inner
# iterators, plus none from the flatten. These tests pin those counts so a
# regression that reintroduces a per-element copy in the flatten path is caught.


def test_product3_element_copies() raises:
    var l1 = [CopyCounter(1), CopyCounter(2)]
    var l2 = [CopyCounter(10), CopyCounter(20)]
    var l3 = [CopyCounter(100), CopyCounter(200)]

    for a, b, c in product(l1, l2, l3):
        assert_equal(a.copy_count, 2)
        assert_equal(b.copy_count, 2)
        # The fastest-varying (rightmost) element is copied once.
        assert_equal(c.copy_count, 1)


def test_product4_element_copies() raises:
    var l1 = [CopyCounter(1), CopyCounter(2)]
    var l2 = [CopyCounter(3), CopyCounter(4)]
    var l3 = [CopyCounter(5), CopyCounter(6)]
    var l4 = [CopyCounter(7), CopyCounter(8)]

    for a, b, c, d in product(l1, l2, l3, l4):
        assert_equal(a.copy_count, 2)
        assert_equal(b.copy_count, 2)
        assert_equal(c.copy_count, 2)
        # The fastest-varying (rightmost) element is copied once.
        assert_equal(d.copy_count, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
