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

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)


def test_zip2() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var it = zip(l, l2)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_zip_destructure() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var count = 0
    for a, b in zip(l, l2):
        assert_equal(a, l[count])
        assert_equal(b, l2[count])
        count += 1


def test_zip3() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var l3 = [100, 200, 300]
    var it = zip(l, l2, l3)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)
    assert_equal(elem[2], 300)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_zip4() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20, 30]
    var l3 = [100, 200, 300]
    var l4 = [1000, 2000, 3000]
    var it = zip(l, l2, l3, l4)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    assert_equal(elem[2], 100)
    assert_equal(elem[3], 1000)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    assert_equal(elem[2], 200)
    assert_equal(elem[3], 2000)
    elem = next(it)
    assert_equal(elem[0], "hello")
    assert_equal(elem[1], 30)
    assert_equal(elem[2], 300)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_zip_unequal_lengths() raises:
    var l = ["hey", "hi", "hello"]
    var l2 = [10, 20]
    var it = zip(l, l2)
    var elem = next(it)
    assert_equal(elem[0], "hey")
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], "hi")
    assert_equal(elem[1], 20)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_zip5() raises:
    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    var l4 = [7, 8]
    var l5 = [9, 10]
    var it = zip(l1, l2, l3, l4, l5)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 3)
    assert_equal(elem[2], 5)
    assert_equal(elem[3], 7)
    assert_equal(elem[4], 9)
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 4)
    assert_equal(elem[2], 6)
    assert_equal(elem[3], 8)
    assert_equal(elem[4], 10)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_zip6_destructure() raises:
    var l1 = [1, 2]
    var l2 = [3, 4]
    var l3 = [5, 6]
    var l4 = [7, 8]
    var l5 = [9, 10]
    var l6 = [11, 12]
    var count = 0
    for a, b, c, d, e, f in zip(l1, l2, l3, l4, l5, l6):
        assert_equal(a, l1[count])
        assert_equal(b, l2[count])
        assert_equal(c, l3[count])
        assert_equal(d, l4[count])
        assert_equal(e, l5[count])
        assert_equal(f, l6[count])
        count += 1
    assert_equal(count, 2)


def test_zip_stop_iteration_destroys_partial_tuple() raises:
    # Uses non-trivially-destructible `String` elements (heap-allocated) so
    # that failing to run destructors for partially-built tuples would leak
    # under ASAN. The shorter list stops first, so the first iterator
    # successfully yields a `String` that must be destroyed before
    # `StopIteration` propagates out of `__next__`.
    var long = [String("aaa"), String("bbb"), String("ccc")]
    var short = [String("x"), String("y")]
    var count = 0
    for a, b in zip(long, short):
        assert_equal(a, long[count])
        assert_equal(b, short[count])
        count += 1
    assert_equal(count, 2)


@fieldwise_init
struct TestIter(ImplicitlyCopyable, Iterable, Iterator):
    comptime Element = Int
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var lower: Int
    var upper: Optional[Int]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(mut self) raises StopIteration -> Self.Element:
        return 42

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return (self.lower, self.upper)


def test_zip_bounds() raises:
    # same size bounds
    var zipA = zip(TestIter(2, {2}), TestIter(2, {2}))
    assert_equal(zipA.bounds()[0], 2)
    assert_equal(zipA.bounds()[1].value(), 2)

    # different size bounds
    var zipB = zip(TestIter(3, {3}), TestIter(2, {2}), TestIter(1, {1}))
    assert_equal(zipB.bounds()[0], 1)
    assert_equal(zipB.bounds()[1].value(), 1)

    # `None` upper get replaced with discrete upper bound
    var zipC = zip(TestIter(0, None), TestIter(2, {3}))
    assert_equal(zipC.bounds()[0], 0)
    assert_equal(zipC.bounds()[1].value(), 3)

    # Preserves `None` upper if all are none
    var zipD = zip(
        TestIter(1, {None}),
        TestIter(2, {None}),
        TestIter(3, {None}),
        TestIter(4, {None}),
    )
    assert_equal(zipD.bounds()[0], 1)
    assert_false(Bool(zipD.bounds()[1]))


def test_zip2_owned() raises:
    var la: List[String] = ["a", "b"]
    var lb: List[Int] = [1, 2]
    var it = zip(la^, lb^)
    var elem = next(it)
    assert_equal(elem[0], "a")
    assert_equal(elem[1], 1)
    elem = next(it)
    assert_equal(elem[0], "b")
    assert_equal(elem[1], 2)
    with assert_raises():
        _ = next(it)


def test_zip3_owned() raises:
    var la: List[Int] = [1, 2]
    var lb: List[Int] = [3, 4]
    var lc: List[Int] = [5, 6]
    var it = zip(la^, lb^, lc^)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 3)
    assert_equal(elem[2], 5)
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 4)
    assert_equal(elem[2], 6)
    with assert_raises():
        _ = next(it)


def test_zip4_owned() raises:
    var la: List[Int] = [1, 2]
    var lb: List[Int] = [3, 4]
    var lc: List[Int] = [5, 6]
    var ld: List[Int] = [7, 8]
    var it = zip(la^, lb^, lc^, ld^)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 3)
    assert_equal(elem[2], 5)
    assert_equal(elem[3], 7)
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 4)
    assert_equal(elem[2], 6)
    assert_equal(elem[3], 8)
    with assert_raises():
        _ = next(it)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
