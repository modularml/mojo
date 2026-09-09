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

from std.collections.interval import Interval, IntervalElement, IntervalTree

from test_utils import check_write_to
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)
from std.testing import TestSuite


def test_interval() raises:
    # Create an interval from 1 to 10 (exclusive)
    var interval = Interval(1, 10)

    # Test basic properties
    assert_equal(interval.start, 1)
    assert_equal(interval.end, 10)
    assert_equal(len(interval), 9)
    assert_equal(len(Interval(-10, -1)), 9)

    # Test string representations
    assert_equal(String(interval), "(1, 10)")
    assert_equal(
        repr(interval),
        "Interval[SIMD[DType.int, 1]](start=Int(1), end=Int(10))",
    )

    # Test equality comparisons
    assert_equal(interval, Interval(1, 10))
    assert_not_equal(interval, Interval(1, 11))

    # Test less than comparisons
    assert_true(
        interval < Interval(2, 11), msg=String(interval, " < Interval(2, 11)")
    )
    assert_false(
        interval < Interval(1, 11), msg=String(interval, " < Interval(1, 11)")
    )

    # Test greater than comparisons
    assert_true(
        interval > Interval(0, 9), msg=String(interval, " > Interval(0, 9)")
    )
    assert_false(
        interval > Interval(1, 11), msg=String(interval, " > Interval(1, 11)")
    )

    # Test less than or equal comparisons
    assert_true(
        interval <= Interval(1, 10), msg=String(interval, " <= Interval(1, 10)")
    )
    assert_true(
        interval <= Interval(1, 11), msg=String(interval, " <= Interval(1, 11)")
    )
    assert_false(
        interval <= Interval(0, 9), msg=String(interval, " <= Interval(0, 9)")
    )

    # Test greater than or equal comparisons
    assert_true(
        interval >= Interval(1, 10), msg=String(interval, " >= Interval(1, 10)")
    )
    assert_true(
        interval >= Interval(2, 9), msg=String(interval, " >= Interval(2, 9)")
    )
    assert_false(
        interval >= Interval(1, 11), msg=String(interval, " >= Interval(1, 11)")
    )

    # Test interval containment
    assert_true(
        interval in Interval(1, 11), msg=String(interval, " in Interval(1, 11)")
    )
    assert_false(
        interval in Interval(1, 9), msg=String(interval, " in Interval(1, 9)")
    )
    assert_true(
        interval in Interval(1, 10), msg=String(interval, " in Interval(1, 10)")
    )
    assert_true(
        interval in Interval(1, 11), msg=String(interval, " in Interval(1, 11)")
    )
    assert_false(
        interval in Interval(1, 9), msg=String(interval, " in Interval(1, 9)")
    )

    # Test point containment
    assert_true(1 in interval, msg="1 in interval")
    assert_false(0 in interval)

    # Test interval overlap
    assert_true(interval.overlaps(Interval(1, 10)))
    assert_true(interval.overlaps(Interval(1, 9)))
    assert_false(interval.overlaps(Interval(-10, -1)))

    # Test interval union
    assert_equal(interval.union(Interval(1, 10)), Interval(1, 10))
    assert_equal(interval.union(Interval(1, 9)), Interval(1, 10))

    # Test interval intersection
    assert_equal(interval.intersection(Interval(1, 10)), Interval(1, 10))
    assert_equal(interval.intersection(Interval(1, 9)), Interval(1, 9))
    assert_equal(interval.intersection(Interval(3, 5)), Interval(3, 5))

    # Test empty interval checks
    assert_true(Bool(interval))
    assert_false(Bool(Interval(0, 0)))


struct MyType(
    Comparable,
    Floatable,
    ImplicitlyCopyable,
    IntervalElement,
    Writable,
):
    var value: Float64

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float64, /):
        self.value = value

    def __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __sub__(self, other: Self) -> Self:
        return Self(self.value - other.value)

    def __int__(self) -> Int:
        return Int(self.value)

    def __float__(self) -> Float64:
        return self.value

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.value)


def test_interval_floating() raises:
    # Create an interval with floating point values using MyType wrapper.
    var interval = Interval(MyType(2.4), MyType(3.5))

    # Verify the interval start and end values are correctly set.
    assert_equal(interval.start.value, 2.4)
    assert_equal(interval.end.value, 3.5)

    # Test union operation with overlapping interval.
    var union = interval.union(Interval(MyType(3.0), MyType(4.5)))

    # Verify union produces expected interval bounds.
    assert_equal(union, Interval(MyType(2.4), MyType(4.5)))

    # Verify length of union interval is correct.
    assert_equal(len(union), 2)


def test_interval_tree() raises:
    var tree = IntervalTree[Int, MyType]()
    tree.insert((15, 20), MyType(33.0))
    tree.insert((10, 30), MyType(34.0))
    tree.insert((17, 19), MyType(35.0))
    tree.insert((5, 20), MyType(36.0))
    tree.insert((12, 15), MyType(37.0))
    tree.insert((30, 40), MyType(38.0))
    print(tree)

    var elems = tree.search((10, 15))
    assert_equal(len(elems), 3)
    assert_equal(Float64(elems[0]), 34.0)
    assert_equal(Float64(elems[1]), 37.0)
    assert_equal(Float64(elems[2]), 36.0)


def test_interval_write_to() raises:
    check_write_to(Interval(1, 10), expected="(1, 10)", is_repr=False)
    check_write_to(Interval(0, 0), expected="(0, 0)", is_repr=False)


def test_interval_write_repr_to() raises:
    check_write_to(
        Interval(1, 10),
        expected="Interval[SIMD[DType.int, 1]](start=Int(1), end=Int(10))",
        is_repr=True,
    )


def test_interval_tree_write_to() raises:
    var tree = IntervalTree[Int, MyType]()
    tree.insert((1, 5), MyType(1.0))
    # write_to produces the ASCII tree drawing
    check_write_to(tree, contains="(1, 5)", is_repr=False)


def test_interval_tree_write_repr_to() raises:
    var tree = IntervalTree[Int, MyType]()
    tree.insert((1, 5), MyType(1.0))
    check_write_to(
        tree, contains="IntervalTree[SIMD[DType.int, 1], MyType](", is_repr=True
    )

    var output = String()
    tree.write_repr_to(output)
    assert_true(output.endswith(")"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
