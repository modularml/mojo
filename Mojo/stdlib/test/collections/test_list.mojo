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

from std.sys.info import size_of
from std.memory import forget_deinit

from test_utils import (
    CopyCountedStruct,
    CopyCounter,
    DelCounter,
    ExplicitDestroy,
    MoveCounter,
    MoveOnly,
    NonMovable,
    Observable,
    TriviallyCopyableMoveCounter,
    check_write_to,
)
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
    TestSuite,
)
from std.testing.prop import PropTest

# TODO(MOCO-522): Figure out desired behavior for importing files with only
# extensions in them.
from std.testing.prop.strategy import SIMD, List


def test_mojo_issue_698() raises:
    var list = List[Float64]()
    for i in range(5):
        list.append(Float64(i))

    assert_equal(0.0, list[0])
    assert_equal(1.0, list[1])
    assert_equal(2.0, list[2])
    assert_equal(3.0, list[3])
    assert_equal(4.0, list[4])


def test_list() raises:
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(5 * size_of[Int](), list.byte_length())
    assert_equal(0, list[0])
    assert_equal(1, list[1])
    assert_equal(2, list[2])
    assert_equal(3, list[3])
    assert_equal(4, list[4])

    assert_equal(0, list[len(list) - 5])
    assert_equal(3, list[len(list) - 2])
    assert_equal(4, list[len(list) - 1])

    list[2] = -2
    assert_equal(-2, list[2])

    list[len(list) - 5] = 5
    assert_equal(5, list[len(list) - 5])
    list[len(list) - 2] = 3
    assert_equal(3, list[len(list) - 2])
    list[len(list) - 1] = 7
    assert_equal(7, list[len(list) - 1])


struct WeirdList[T: AnyType]:
    def __init__(out self, var *values: Self.T, __list_literal__: NoneType):
        pass


def take_generic_weird_list(list: WeirdList[_]) raises:
    pass


def test_list_literal() raises:
    var list: List[Int] = [1, 2, 3]
    assert_equal(3, len(list))
    assert_equal(1, list[0])
    assert_equal(2, list[1])
    assert_equal(3, list[2])

    var list2 = [1.0, 2.5]
    assert_equal(2, len(list2))
    assert_equal(1.0, list2[0])
    assert_equal(2.5, list2[1])

    # Test parameter inference of the T element type.
    take_generic_weird_list([1.0, 2.0])

    # Heterogenous lists
    # take_generic_weird_list([1.0, 2])
    # take_generic_weird_list([1, 2.0])


def test_list_unsafe_get() raises:
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(0, list.unsafe_get(0))
    assert_equal(1, list.unsafe_get(1))
    assert_equal(2, list.unsafe_get(2))
    assert_equal(3, list.unsafe_get(3))
    assert_equal(4, list.unsafe_get(4))

    list[2] = -2
    assert_equal(-2, list.unsafe_get(2))

    list.clear()
    list.append(2)
    assert_equal(2, list.unsafe_get(0))


def test_list_unsafe_set() raises:
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    list.unsafe_set(0, 0)
    list.unsafe_set(1, 10)
    list.unsafe_set(2, 20)
    list.unsafe_set(3, 30)
    list.unsafe_set(4, 40)

    assert_equal(list[0], 0)
    assert_equal(list[1], 10)
    assert_equal(list[2], 20)
    assert_equal(list[3], 30)
    assert_equal(list[4], 40)


def test_list_clear() raises:
    var list: List = [1, 2, 3]
    assert_equal(len(list), 3)
    assert_equal(list.capacity(), 3)
    list.clear()

    assert_equal(len(list), 0)
    assert_equal(list.capacity(), 3)


def test_list_to_bool_conversion() raises:
    assert_false(List[String]())
    assert_true(List[String](["a"]))
    assert_true(List[String](["", "a"]))
    assert_true(List[String]([""]))


def test_list_pop() raises:
    var list = List[Int]()
    # Test pop with index
    for i in range(6):
        list.append(i)

    # try popping from index 3 for 3 times
    for i in range(3, 6):
        assert_equal(i, list.pop(3))

    # list should have 3 elements now
    assert_equal(3, len(list))
    assert_equal(0, list[0])
    assert_equal(1, list[1])
    assert_equal(2, list[2])

    # Test pop with index 0 (first element)
    for i in range(0, 2):
        assert_equal(i, list.pop(0))

    # test default index as well
    assert_equal(2, list.pop())
    list.append(2)
    assert_equal(2, list.pop())

    # list should be empty now
    assert_equal(0, len(list))


def test_list_variadic_constructor() raises:
    var l: List = [2, 4, 6]
    assert_equal(3, len(l))
    assert_equal(2, l[0])
    assert_equal(4, l[1])
    assert_equal(6, l[2])

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l[3])

    #
    # Test variadic construct copying behavior
    #

    var l2 = [CopyCounter(), CopyCounter(), CopyCounter()]

    assert_equal(len(l2), 3)
    assert_equal(l2[0].copy_count, 0)
    assert_equal(l2[1].copy_count, 0)
    assert_equal(l2[2].copy_count, 0)


def test_list_resize() raises:
    var l: List[Int] = [1]
    assert_equal(1, len(l))
    l.resize(2, 0)
    assert_equal(2, len(l))
    assert_equal(l[1], 0)
    l.shrink(0)
    assert_equal(len(l), 0)


# TODO: Rework to use property testing framework.
def test_list_reverse_property_test() raises:
    def properties(forward: List[Int]) raises:
        var rev = forward.copy()
        rev.reverse()

        assert_equal(len(forward), len(rev))
        for a, b in zip(forward, reversed(rev)):
            assert_equal(a, b)

    PropTest().test[properties](List[Int].strategy(Int.strategy()))


def test_list_reverse() raises:
    #
    # Test reversing the list []
    #

    var vec = List[Int]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the list [123]
    #

    vec = []

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    var vec2: List = ["one", "two", "three"]

    assert_equal(len(vec2), 3)
    assert_equal(vec2[0], "one")
    assert_equal(vec2[1], "two")
    assert_equal(vec2[2], "three")

    vec2.reverse()

    assert_equal(len(vec2), 3)
    assert_equal(vec2[0], "three")
    assert_equal(vec2[1], "two")
    assert_equal(vec2[2], "one")

    #
    # Test reversing the list [5, 10]
    #

    vec = []
    vec.append(5)
    vec.append(10)

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 5)
    assert_equal(vec[1], 10)

    vec.reverse()

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 10)
    assert_equal(vec[1], 5)


def test_list_reverse_move_count() raises:
    # Create this vec with enough capacity to avoid moves due to resizing.
    var vec = List[MoveCounter[Int]](capacity=5)
    vec.append(MoveCounter(1))
    vec.append(MoveCounter(2))
    vec.append(MoveCounter(3))
    vec.append(MoveCounter(4))
    vec.append(MoveCounter(5))

    assert_equal(len(vec), 5)
    assert_equal(vec[0].value, 1)
    assert_equal(vec[1].value, 2)
    assert_equal(vec[2].value, 3)
    assert_equal(vec[3].value, 4)
    assert_equal(vec[4].value, 5)

    assert_equal(vec[0].move_count, 1)
    assert_equal(vec[1].move_count, 1)
    assert_equal(vec[2].move_count, 1)
    assert_equal(vec[3].move_count, 1)
    assert_equal(vec[4].move_count, 1)

    vec.reverse()

    assert_equal(len(vec), 5)
    assert_equal(vec[0].value, 5)
    assert_equal(vec[1].value, 4)
    assert_equal(vec[2].value, 3)
    assert_equal(vec[3].value, 2)
    assert_equal(vec[4].value, 1)

    # NOTE:
    # Earlier elements went through 2 moves and later elements went through 3
    # moves because the implementation of List.reverse arbitrarily
    # chooses to perform the swap of earlier and later elements by moving the
    # earlier element to a temporary (+1 move), directly move the later element
    # into the position the earlier element was in, and then move from the
    # temporary into the later position (+1 move).
    assert_equal(vec[0].move_count, 2)
    assert_equal(vec[1].move_count, 2)
    assert_equal(vec[2].move_count, 1)
    assert_equal(vec[3].move_count, 3)
    assert_equal(vec[4].move_count, 3)


def test_list_insert() raises:
    #
    # Test the list [1, 2, 3] created with insert
    #

    var v1 = List[Int]()
    v1.insert(len(v1), 1)
    v1.insert(len(v1), 3)
    v1.insert(1, 2)

    assert_equal(len(v1), 3)
    assert_equal(v1[0], 1)
    assert_equal(v1[1], 2)
    assert_equal(v1[2], 3)

    #
    # Test the list [1, 2, 3, 4, 5] created with interior and boundary indices
    #

    var v2 = List[Int]()
    v2.insert(0, 2)
    v2.insert(len(v2), 3)
    v2.insert(len(v2), 5)
    v2.insert(2, 4)
    v2.insert(0, 1)

    assert_equal(len(v2), 5)
    assert_equal(v2[0], 1)
    assert_equal(v2[1], 2)
    assert_equal(v2[2], 3)
    assert_equal(v2[3], 4)
    assert_equal(v2[4], 5)

    #
    # Test the list [1, 2, 3, 4] created by inserting at the front
    #

    var v3 = List[Int]()
    v3.insert(0, 4)
    v3.insert(0, 3)
    v3.insert(0, 2)
    v3.insert(0, 1)

    assert_equal(len(v3), 4)
    assert_equal(v3[0], 1)
    assert_equal(v3[1], 2)
    assert_equal(v3[2], 3)
    assert_equal(v3[3], 4)

    #
    # Test the list [1, 2, 3, 4, 5, 6, 7, 8] created with insert
    #

    var v4 = List[Int]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i, value in enumerate(v4):
        assert_equal(value, i + 1)


def test_list_index() raises:
    var test_list_a: List = [10, 20, 30, 40, 50]

    # Basic Functionality Tests
    assert_equal(test_list_a.index(10), 0)
    assert_equal(test_list_a.index(30), 2)
    assert_equal(test_list_a.index(50), 4)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(60)

    # Tests With Start Parameter
    assert_equal(test_list_a.index(30, start=1), 2)
    assert_equal(test_list_a.index(30, start=-4), 2)
    assert_equal(test_list_a.index(30, start=-1000), 2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=3)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=5)

    # Tests With Start and End Parameters
    assert_equal(test_list_a.index(30, start=1, stop=3), 2)
    assert_equal(test_list_a.index(30, start=-4, stop=-2), 2)
    assert_equal(test_list_a.index(30, start=-1000, stop=1000), 2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=1, stop=2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=3, stop=1)

    # Tests With End Parameter Only
    assert_equal(test_list_a.index(30, stop=3), 2)
    assert_equal(test_list_a.index(30, stop=-2), 2)
    assert_equal(test_list_a.index(30, stop=1000), 2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, stop=1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, stop=2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(60, stop=50)

    # Edge Cases and Special Conditions
    assert_equal(test_list_a.index(10, start=-5, stop=-1), 0)
    assert_equal(test_list_a.index(10, start=0, stop=50), 0)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(50, start=-5, stop=-1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(50, start=0, stop=-1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=-4, stop=-1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=5, stop=50)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = List[Int]().index(10)

    # Test empty slice
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=1, stop=1)
    # Test empty slice with 0 start and end
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=0, stop=0)

    var test_list_b: List = [10, 20, 30, 20, 10]

    # Test finding the first occurrence of an item
    assert_equal(test_list_b.index(10), 0)
    assert_equal(test_list_b.index(20), 1)

    # Test skipping the first occurrence with a start parameter
    assert_equal(test_list_b.index(20, start=2), 3)

    # Test constraining search with start and end, excluding last occurrence
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_b.index(10, start=1, stop=4)

    # Test search within a range that includes multiple occurrences
    assert_equal(test_list_b.index(20, start=1, stop=4), 1)

    # Verify error when constrained range excludes occurrences
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_b.index(20, start=4, stop=5)


def test_list_append() raises:
    var items = List[UInt32]()
    items.append(1)
    items.append(2)
    items.append(3)
    assert_equal(items, [UInt32(1), 2, 3])


def test_list_insert_move_count() raises:
    # Preallocate so reallocation does not contribute moves.
    var vec = List[MoveCounter[Int]](capacity=4)
    vec.append(MoveCounter(1))
    vec.append(MoveCounter(2))
    vec.append(MoveCounter(3))

    vec.insert(0, MoveCounter(0))

    assert_equal(len(vec), 4)
    assert_equal(vec[0].value, 0)
    assert_equal(vec[1].value, 1)
    assert_equal(vec[2].value, 2)
    assert_equal(vec[3].value, 3)

    # Inserting shifts the tail right by one, so each displaced element takes
    # exactly one additional move on top of the one that appended it.
    assert_equal(vec[0].move_count, 1)
    assert_equal(vec[1].move_count, 2)
    assert_equal(vec[2].move_count, 2)
    assert_equal(vec[3].move_count, 2)


def test_list_insert_middle_move_count() raises:
    # Inserting past the front must leave everything before `i` alone. Index 0
    # drops `i` out of the shift arithmetic, so an off-by-one in it only shows
    # up on a middle insert.
    var vec = List[MoveCounter[Int]](capacity=5)
    vec.append(MoveCounter(0))
    vec.append(MoveCounter(1))
    vec.append(MoveCounter(3))
    vec.append(MoveCounter(4))

    vec.insert(2, MoveCounter(2))

    assert_equal(len(vec), 5)
    for i in range(5):
        assert_equal(vec[i].value, i)

    assert_equal(vec[0].move_count, 1)
    assert_equal(vec[1].move_count, 1)
    assert_equal(vec[2].move_count, 1)
    assert_equal(vec[3].move_count, 2)
    assert_equal(vec[4].move_count, 2)


def test_list_extend() raises:
    var items: List[UInt32] = [1, 2, 3]
    var copy = items.copy()
    items.extend(copy^)
    assert_equal(items, [UInt32(1), 2, 3, 1, 2, 3])

    items = [1, 2, 3]
    copy = [1, 2, 3]

    # Extend with span
    items.extend(Span(copy))
    assert_equal(items, [UInt32(1), 2, 3, 1, 2, 3])

    # Extend with whole SIMD
    items: List[UInt32] = [1, 2, 3]
    items.extend(SIMD[.uint32, 4](1, 2, 3, 4))
    assert_equal(items, [UInt32(1), 2, 3, 1, 2, 3, 4])
    # Extend with part of SIMD
    items: List[UInt32] = [1, 2, 3]
    items.extend(SIMD[.uint32, 4](1, 2, 3, 4), count=3)
    assert_equal(items, [UInt32(1), 2, 3, 1, 2, 3])


def test_list_extend_non_trivial() raises:
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other lists
    #   - extend() using optimal number of move constructor calls

    # Preallocate with enough capacity to avoid reallocation making the
    # move count checks below flaky.
    var v1 = List[MoveCounter[String]](capacity=5)
    v1.append(MoveCounter[String]("Hello"))
    v1.append(MoveCounter[String]("World"))

    var v2 = List[MoveCounter[String]](capacity=3)
    v2.append(MoveCounter[String]("Foo"))
    v2.append(MoveCounter[String]("Bar"))
    v2.append(MoveCounter[String]("Baz"))

    v1.extend(v2^)

    assert_equal(len(v1), 5)
    assert_equal(v1[0].value, "Hello")
    assert_equal(v1[1].value, "World")
    assert_equal(v1[2].value, "Foo")
    assert_equal(v1[3].value, "Bar")
    assert_equal(v1[4].value, "Baz")

    assert_equal(v1[0].move_count, 1)
    assert_equal(v1[1].move_count, 1)
    assert_equal(v1[2].move_count, 2)
    assert_equal(v1[3].move_count, 2)
    assert_equal(v1[4].move_count, 2)


def test_list_extend_trivial_copy_nontrivial_move() raises:
    var v1 = List[TriviallyCopyableMoveCounter](capacity=1)
    var v2: List = [TriviallyCopyableMoveCounter(0)]

    assert_equal(v2[0].move_count, 1)

    v1.extend(v2^)

    # `extend()` should call move constructor, not perform any copies.
    assert_equal(v1[0].move_count, 2)


def test_list_extend_growth_is_amortized() raises:
    # Extending in a loop must at least double the capacity each time it
    # reallocates, otherwise repeated extends are quadratic. Seven chunks of
    # four separate the two policies: growing to exactly what each call asks
    # for ends at a capacity of 28, while at least doubling clears 32. The
    # bound is a lower one so that a future policy which overshoots more
    # stays passing.
    var owned = List[Int]()
    var spanned = List[Int]()
    for _ in range(7):
        var chunk: List[Int] = [1, 2, 3, 4]
        spanned.extend(Span(chunk))
        owned.extend(chunk^)

    assert_equal(len(owned), 28)
    assert_equal(len(spanned), 28)
    assert_true(owned.capacity() >= 32)
    assert_true(spanned.capacity() >= 32)

    # `reserve` keeps honoring the requested capacity exactly.
    var exact = List[Int](capacity=4)
    exact.reserve(5)
    assert_equal(exact.capacity(), 5)


def test_list_resize_growth_is_amortized() raises:
    # `resize` grows through the same path as `extend`, so growing by a fixed
    # increment must double as well.
    var filled = List[Int]()
    var uninit = List[Int]()
    for length in range(4, 29, 4):
        filled.resize(length, 0)
        uninit.resize(unsafe_uninit_length=length)

    assert_equal(len(filled), 28)
    assert_equal(len(uninit), 28)
    assert_true(filled.capacity() >= 32)
    assert_true(uninit.capacity() >= 32)


def test_2d_dynamic_list() raises:
    var list = List[List[Int]]()

    for i in range(2):
        var v = List[Int]()
        for j in range(3):
            v.append(i + j)
        list.append(v^)

    assert_equal(0, list[0][0])
    assert_equal(1, list[0][1])
    assert_equal(2, list[0][2])
    assert_equal(1, list[1][0])
    assert_equal(2, list[1][1])
    assert_equal(3, list[1][2])

    assert_equal(2, len(list))
    assert_equal(2, list.capacity())

    assert_equal(3, len(list[0]))

    list[0].clear()
    assert_equal(0, len(list[0]))
    assert_equal(4, list[0].capacity())

    list.clear()
    assert_equal(0, len(list))
    assert_equal(2, list.capacity())


def test_list_explicit_copy() raises:
    var list = List[CopyCounter[]]()
    list.append(CopyCounter())
    var list_copy = list.copy()
    assert_equal(0, list[0].copy_count)
    assert_equal(1, list_copy[0].copy_count)

    var l2 = List[Int]()
    for i in range(10):
        l2.append(i)

    var l2_copy = l2.copy()
    assert_equal(len(l2), len(l2_copy))
    for val1, val2 in zip(l2, l2_copy):
        assert_equal(val1, val2)


def test_no_extra_copies_with_sugared_set_by_field() raises:
    var list = List[List[CopyCountedStruct]](capacity=1)
    var child_list = List[CopyCountedStruct](capacity=2)
    child_list.append(CopyCountedStruct("Hello"))
    child_list.append(CopyCountedStruct("World"))

    # No copies here.  Constructing with List[CopyCountedStruct](CopyCountedStruct("Hello")) is a copy.
    assert_equal(0, child_list[0].counter.copy_count)
    assert_equal(0, child_list[1].counter.copy_count)
    list.append(child_list^)

    list[0][1].value = "Mojo"
    assert_equal("Mojo", list[0][1].value)

    assert_equal(0, list[0][0].counter.copy_count)
    assert_equal(0, list[0][1].counter.copy_count)


# Ensure correct behavior of copy ctor
# as reported in GH issue 27875 internally and
# https://github.com/modular/modular/issues/1493
def test_list_copy_constructor() raises:
    var vec = List[Int](capacity=1)
    var vec_copy = vec.copy()
    vec_copy.append(1)  # Ensure copy constructor doesn't crash
    _ = vec^  # To ensure previous one doesn't invoke move constructor


def test_list_iter() raises:
    var vs = List[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    def sum(vs: List[Int]) -> Int:
        var sum = 0
        for v in vs:
            sum += v
        return sum

    assert_equal(6, sum(vs))


def test_list_iter_mutable() raises:
    var vs = [1, 2, 3]

    for ref v in vs:
        v += 1

    var sum = 0
    for v in vs:
        sum += v

    assert_equal(9, sum)


# We use `MutAnyOrigin` to bypass exclusivity checking
# otherwise we cannot construct a list of Observables where
# all point to the same copy/move/deinit counter.
comptime ObservableElement = Observable[
    CopyOrigin=MutAnyOrigin,
    MoveOrigin=MutAnyOrigin,
    DelOrigin=MutAnyOrigin,
]


def make_observable_list(
    *, mut copies: Int, mut moves: Int, mut dels: Int, length: Int
) -> List[ObservableElement]:
    var list = List[ObservableElement](capacity=length)
    for _i in range(length):
        list.append(
            ObservableElement(
                copies=Pointer[Int, MutAnyOrigin](to=copies),
                moves=Pointer[Int, MutAnyOrigin](to=moves),
                dels=Pointer[Int, MutAnyOrigin](to=dels),
            )
        )
    return list^


def test_list_iter_owned() raises:
    var copies = 0
    var moves = 0
    var dels = 0

    var list = make_observable_list(
        copies=copies, moves=moves, dels=dels, length=2
    )
    assert_equal(copies, 0)
    assert_equal(moves, 2)
    assert_equal(dels, 0)

    for _elem in list^:
        pass

    assert_equal(copies, 0)
    assert_equal(moves, 4)
    assert_equal(dels, 2)


def test_list_iter_owned_destroys_elements_if_not_consumed() raises:
    var copies = 0
    var moves = 0
    var dels = 0

    var list = make_observable_list(
        copies=copies, moves=moves, dels=dels, length=2
    )
    var _ = list^.__iter__()
    assert_equal(copies, 0)
    assert_equal(moves, 2)
    assert_equal(dels, 2)


def test_list_iter_owned_destroys_elements_if_partially_consumed() raises:
    var copies = 0
    var moves = 0
    var dels = 0

    var list = make_observable_list(
        copies=copies, moves=moves, dels=dels, length=2
    )

    var iter = list^.__iter__()
    assert_equal(copies, 0)
    assert_equal(moves, 2)
    assert_equal(dels, 0)

    var _ = iter.__next__()
    assert_equal(copies, 0)
    assert_equal(moves, 3)
    assert_equal(dels, 1)

    _ = iter^
    assert_equal(copies, 0)
    assert_equal(moves, 3)
    assert_equal(dels, 2)


def test_list_iter_owned_move_only() raises:
    # Consuming iteration only requires `Movable & Deinitable`, not
    # `Copyable`: each element is moved out of the list, not copied.
    var list = [MoveOnly[Int](0), MoveOnly[Int](1), MoveOnly[Int](2)]

    var total = 0
    for elem in list^:
        total += elem.data

    assert_equal(total, 3)


def test_list_iter_owned_bounds() raises:
    var iter = iter([1, 2, 3])
    for i in range(3, 0, -1):
        assert_equal((i, Optional(i)), iter.bounds())
        _ = iter.__next__()


def _test_list_iter_bounds[
    I: Iterator
](var list_iter: I, list_len: Int) raises where conforms_to(
    I.Element, Deinitable
):
    var iter = list_iter^

    for i in range(list_len):
        var lower, upper = iter.bounds()
        assert_equal(list_len - i, lower)
        assert_equal(list_len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_list_iter_bounds() raises:
    var list = [1, 2, 3]
    _test_list_iter_bounds(iter(list), len(list))
    _test_list_iter_bounds(reversed(list), len(list))


def test_list_span() raises:
    var vs: List = [1, 2, 3]

    var es = List(vs[1:])
    assert_equal(es[0], 2)
    assert_equal(es[1], 3)
    assert_equal(len(es), 2)

    es = List(vs[0 : len(vs) - 1])
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(len(es), 2)

    es = vs[1:-1:1]
    assert_equal(es[0], 2)
    assert_equal(len(es), 1)

    es = vs[::-1]
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(es[2], 1)
    assert_equal(len(es), 3)

    es = List(vs[:])
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
    assert_equal(len(es), 3)

    assert_equal(vs[1:0:-1][0], 2)
    assert_equal(vs[2:1:-1][0], 3)
    es = vs[:0:-1]
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(vs[2::-1][0], 3)

    assert_equal(len(vs[1:2:-1]), 0)

    assert_equal(0, len(vs[:-1:-2]))
    assert_equal(0, len(vs[-50::-1]))
    es = vs[:-50:-1]
    assert_equal(3, len(es))
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(es[2], 1)
    es = vs[::50]
    assert_equal(1, len(es))
    assert_equal(es[0], 1)
    es = vs[::-50]
    assert_equal(1, len(es))
    assert_equal(es[0], 3)
    es = vs[50::-50]
    assert_equal(1, len(es))
    assert_equal(es[0], 3)
    es = vs[-50::50]
    assert_equal(1, len(es))
    assert_equal(es[0], 1)


def test_list_realloc_trivial_types() raises:
    var a = List[Int]()
    for i in range(100):
        a.append(i)

    assert_equal(len(a), 100)
    for i in range(100):
        assert_equal(a[i], i)

    var b = List[Int8]()
    for i in range(100):
        b.append(Int8(i))

    assert_equal(len(b), 100)
    for i in range(100):
        assert_equal(b[i], Int8(i))


def test_list_realloc_trivial_copy_nontrivial_move() raises:
    var lst = List[TriviallyCopyableMoveCounter](capacity=1)

    lst.append(TriviallyCopyableMoveCounter(0))
    assert_equal(lst[0].move_count, 1)

    lst.reserve(10)

    # Reallocating the list should call move constructor, not perform any copies.
    assert_equal(lst[0].move_count, 2)


def test_list_boolable() raises:
    assert_true(List[Int]([1]))
    assert_false(List[Int]())


def test_list_write_to() raises:
    check_write_to([1, 2, 3], expected="[1, 2, 3]", is_repr=False)
    check_write_to(
        ["a", "b", "c", "foo"], expected="[a, b, c, foo]", is_repr=False
    )
    check_write_to(List[Int](), expected="[]", is_repr=False)
    check_write_to([42], expected="[42]", is_repr=False)


def test_list_count() raises:
    var list: List = [1, 2, 3, 2, 5, 6, 7, 8, 9, 10]
    assert_equal(1, list.count(1))
    assert_equal(2, list.count(2))
    assert_equal(0, list.count(4))

    var list2 = List[Int]()
    assert_equal(0, list2.count(1))


def test_list_add() raises:
    var a: List = [1, 2, 3]
    var b: List = [4, 5, 6]
    var c = a + b.copy()
    assert_equal(len(c), 6)
    # check that original values aren't modified
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    assert_equal(String(c), "[1, 2, 3, 4, 5, 6]")

    a += b.copy()
    assert_equal(len(a), 6)
    assert_equal(String(a), "[1, 2, 3, 4, 5, 6]")
    assert_equal(len(b), 3)

    a = [1, 2, 3]
    a += b^
    assert_equal(len(a), 6)
    assert_equal(String(a), "[1, 2, 3, 4, 5, 6]")

    var d: List = [1, 2, 3]
    var e: List = [4, 5, 6]
    var f = d + e^
    assert_equal(len(f), 6)
    assert_equal(String(f), "[1, 2, 3, 4, 5, 6]")

    var l: List = [1, 2, 3]
    l += []
    assert_equal(len(l), 3)


def test_list_mult() raises:
    var a: List = [1, 2, 3]
    var b = a * 2
    assert_equal(len(b), 6)
    assert_equal(String(b), "[1, 2, 3, 1, 2, 3]")
    b = a * 3
    assert_equal(len(b), 9)
    assert_equal(String(b), "[1, 2, 3, 1, 2, 3, 1, 2, 3]")
    a *= 2
    assert_equal(len(a), 6)
    assert_equal(String(a), "[1, 2, 3, 1, 2, 3]")

    var l: List = [1, 2]
    l *= 1
    assert_equal(len(l), 2)

    l *= 0
    assert_equal(len(l), 0)
    assert_equal(len(List[Int]([1, 2, 3]) * 0), 0)


def test_list_contains() raises:
    var x = [1, 2, 3]
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)

    # TODO: implement List.__eq__ for Self[Copyable & Comparable]
    # var y = List[List[Int]]()
    # y.append([1, 2])
    # assert_equal([1, 2] in y,True)
    # assert_equal([0, 1] in y,False)


def test_list_eq_ne() raises:
    var l1: List = [1, 2, 3]
    var l2: List = [1, 2, 3]
    assert_true(l1 == l2)
    assert_false(l1 != l2)

    var l3: List = [1, 2, 3, 4]
    assert_false(l1 == l3)
    assert_true(l1 != l3)

    var l4 = List[Int]()
    var l5 = List[Int]()
    assert_true(l4 == l5)
    assert_true(l1 != l4)

    var l6: List = ["a", "b", "c"]
    var l7: List = ["a", "b", "c"]
    var l8: List = ["a", "b"]
    assert_true(l6 == l7)
    assert_false(l6 != l7)
    assert_false(l6 == l8)


struct NonEquatable(Copyable):
    pass


def test_list_conditional_conformances() raises:
    assert_true(conforms_to(List[Int], Equatable))
    assert_false(conforms_to(List[NonEquatable], Equatable))

    assert_true(conforms_to(List[Int], Writable))
    assert_false(conforms_to(List[NonEquatable], Writable))

    # Owned iteration requires `Movable & Deinitable` elements, but
    # not `Copyable`: a consuming iterator moves elements out rather than
    # copying them.
    assert_true(conforms_to(List[Int], IterableOwned))
    # `MoveOnly[Int]` is movable and implicitly deletable but not copyable.
    assert_true(conforms_to(List[MoveOnly[Int]], IterableOwned))
    # `ExplicitDestroy` is not implicitly deletable, so the consuming iterator
    # cannot destroy any unconsumed elements.
    assert_false(conforms_to(List[ExplicitDestroy], IterableOwned))

    # List is still Movable despite its element type not being Movable
    assert_true(conforms_to(List[Pinned], Movable))


def test_list_init_span() raises:
    var l = [String("a"), "bb", "cc", "def"]
    var sp = Span(l)
    var l2 = List[String](sp)
    for val1, val2 in zip(l, l2):
        assert_equal(val1, val2)


def test_list_init_iter() raises:
    var l = [String("a"), "bb", "cc", "def"]
    var it = iter(l)
    var l2 = List[String](it)
    assert_equal(len(l), len(l2))
    assert_equal(l[0], l2[0])
    assert_equal(l[1], l2[1])
    assert_equal(l[2], l2[2])
    assert_equal(l[3], l2[3])


def test_indexing() raises:
    var l = [1, 2, 3]
    assert_equal(l[Int(1)], 2)
    assert_equal(l[2], 3)


# ===-------------------------------------------------------------------===#
# List dtor tests
# ===-------------------------------------------------------------------===#


def test_list_dtor() raises:
    var dtor_count = 0

    var ptr = Pointer(to=dtor_count).as_imm()
    var l = List[DelCounter[ptr.origin]]()
    assert_equal(dtor_count, 0)

    l.append(DelCounter(ptr))
    assert_equal(dtor_count, 0)

    l^.__deinit__()
    assert_equal(dtor_count, 1)


def test_destructor_trivial_elements() raises:
    var dtor_count = 0

    var ptr = Pointer(to=dtor_count).as_imm()
    var l = List[DelCounter[ptr.origin, trivial_del=True]]()
    l.append(DelCounter[ptr.origin, trivial_del=True](ptr))

    l^.__deinit__()

    assert_equal(dtor_count, 0)


def test_list_write_repr_to() raises:
    check_write_to(
        List([1, 2, 3]),
        expected="List[SIMD[DType.int, 1]]([Int(1), Int(2), Int(3)])",
        is_repr=True,
    )
    check_write_to(
        List([1]), expected="List[SIMD[DType.int, 1]]([Int(1)])", is_repr=True
    )
    check_write_to(
        List[Int](), expected="List[SIMD[DType.int, 1]]([])", is_repr=True
    )
    # Non-`Int` scalar elements also print using their type alias.
    var uints: List[UInt] = [1, 2]
    check_write_to(
        uints,
        expected="List[SIMD[DType.uint, 1]]([UInt(1), UInt(2)])",
        is_repr=True,
    )
    var floats: List[Float32] = [1.0, 2.0]
    check_write_to(
        floats,
        expected="List[SIMD[DType.float32, 1]]([Float32(1.0), Float32(2.0)])",
        is_repr=True,
    )


def test_list_fill_constructor() raises:
    var l = List[Int32](length=10, fill=3)
    assert_equal(len(l), 10)

    for i in range(10):
        assert_equal(l[i], 3)

    var l2 = List[String](length=20, fill="hi")
    assert_equal(len(l2), 20)

    for i in range(20):
        assert_equal(l2[i], "hi")


def test_list_fill_with_constructor() raises:
    var squares = List(length=5, fill_with=lambda (i: Int) -> Int: i * i)
    assert_equal(len(squares), 5)
    for i in range(5):
        assert_equal(squares[i], i * i)

    var strs = List(length=3, fill_with=lambda (i: Int) -> String: String(i))
    assert_equal(len(strs), 3)
    for i in range(3):
        assert_equal(strs[i], String(i))


def test_list_fill_with_named_function() raises:
    def cube(i: Int) {imm} -> Int:
        return i * i * i

    var cubes = List(length=4, fill_with=cube)
    assert_equal(len(cubes), 4)
    for i in range(4):
        assert_equal(cubes[i], i * i * i)


def test_list_fill_with_non_movable() raises:
    # `fill_with=` constructs each element in place, so it works even for a
    # non-`Movable` element type -- no constraint on `T` at all.
    var l = List(
        length=3, fill_with=lambda (i: Int) -> NonMovable: NonMovable(i * 10)
    )
    assert_equal(l[0].value, 0)
    assert_equal(l[1].value, 10)
    assert_equal(l[2].value, 20)


def test_uninit_ctor() raises:
    var list = List[String](unsafe_uninit_length=2)

    Pointer(to=list[0]).unsafe_write("hello ")
    Pointer(to=list[1]).unsafe_write("world")
    assert_equal(list[0], "hello ")
    assert_equal(list[1], "world")

    # Resize with uninitialized memory.
    var list2 = List[String]()
    list2.resize(unsafe_uninit_length=2)
    list2.unsafe_ptr().unsafe_offset(0).unsafe_write("hello ")
    list2.unsafe_ptr().unsafe_offset(1).unsafe_write("world")
    assert_equal(list2[0], "hello ")
    assert_equal(list2[1], "world")


def _test_copyinit_trivial_types[dt: DType]() raises:
    comptime sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert_equal(len(sizes), 10)
    var test_current_size = 1

    comptime for sizes_index in range(len(sizes)):
        comptime current_size = rebind[Int](sizes[sizes_index])
        var x = List[Scalar[dt]]()
        for i in range(current_size):
            x.append(Scalar[dt](i))
        var y = x.copy()
        assert_equal(test_current_size, current_size)
        assert_equal(len(y), current_size)
        assert_not_equal(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()))
        for i in range(current_size):
            assert_equal(Scalar[dt](i), x[i])
            assert_equal(y[i], x[i])
        test_current_size *= 2
    assert_equal(test_current_size, 1024)


def test_copyinit_trivial_types_dtypes() raises:
    comptime dtypes = (
        DType.int64,
        DType.int32,
        DType.float64,
        DType.float32,
        DType.uint8,
        DType.int8,
        DType.bool,
    )

    comptime for index_dtype in range(len(dtypes)):
        _test_copyinit_trivial_types[rebind[DType](dtypes[index_dtype])]()


def test_list_comprehension() raises:
    var l1 = [x * x for x in range(10) if x & 1]
    assert_equal(l1, [1, 9, 25, 49, 81])

    var l2 = [x * y for x in range(3) for y in l1]
    assert_equal(l2, [0, 0, 0, 0, 0, 1, 9, 25, 49, 81, 2, 18, 50, 98, 162])


def test_list_can_infer_iterable_element_type() raises:
    var string = "Mojo🔥"
    var l = List(string.codepoints())
    assert_true(type_of(l) == List[Codepoint])
    assert_equal(
        l,
        [
            Codepoint.ord("M"),
            Codepoint.ord("o"),
            Codepoint.ord("j"),
            Codepoint.ord("o"),
            Codepoint.ord("🔥"),
        ],
    )


def test_list_hash() raises:
    # Equal lists should produce the same hash.
    assert_equal(hash([1, 2, 3]), hash([1, 2, 3]))

    # Empty list hashing.
    assert_equal(hash(List[Int]()), hash(List[Int]()))

    # Different lists should (likely) produce different hashes.
    assert_not_equal(hash([1, 2, 3]), hash([1, 2, 4]))
    assert_not_equal(hash([1, 2]), hash([2, 1]))

    # Hashable conformance is conditional.
    assert_true(conforms_to(List[Int], Hashable))
    assert_true(conforms_to(List[String], Hashable))


def test_list_move_only() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `List[T: Movable]`.
    assert_false(conforms_to(List[MoveOnly[Int]], Copyable))

    var l: List = [MoveOnly[Int](0), MoveOnly[Int](1)]
    assert_equal(l[0], MoveOnly[Int](0))
    assert_equal(l[1], MoveOnly[Int](1))

    l.append(MoveOnly[Int](2))
    assert_equal(l[2], MoveOnly[Int](2))

    # Methods that don't require `Copyable` still work on a move-only element.
    var popped = l.pop()
    assert_equal(popped, MoveOnly[Int](2))
    assert_equal(len(l), 2)

    l.insert(0, MoveOnly[Int](-1))
    assert_equal(l[0], MoveOnly[Int](-1))
    assert_equal(len(l), 3)

    l.clear()
    assert_equal(len(l), 0)


def test_list_with_explicit_destroy_type() raises:
    var list = [ExplicitDestroy(0), ExplicitDestroy(1)]

    var destroyed = List[Int]()

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    list^.deinit_with(destroy_closure)

    assert_equal(destroyed, [0, 1])


def test_empty_list_with_explicit_destroy_type() raises:
    var list = List[ExplicitDestroy]()

    var destroyed = 0

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed += 1
        e^.destroy()

    list^.deinit_with(destroy_closure)

    assert_equal(destroyed, 0)


def test_extend_list_with_explicit_destroy_type() raises:
    var list1: List = [ExplicitDestroy(0)]
    var list2: List = [ExplicitDestroy(1), ExplicitDestroy(2)]
    list1.extend(list2^)

    var destroyed = List[Int]()

    def destroy_closure(var e: ExplicitDestroy) {mut}:
        destroyed.append(e.value)
        e^.destroy()

    list1^.deinit_with(destroy_closure)
    assert_equal(destroyed, [0, 1, 2])


@align(32)
@fieldwise_init
struct Overaligned(Copyable, Movable):
    var x: Float32


def test_overaligned_struct_realloc() raises:
    # `@align` pads the struct's stride, and size_of must match it.
    assert_equal(size_of[Overaligned](), 32)
    var l = List[Overaligned]()
    for i in range(17):
        l.append(Overaligned(Float32(i)))
    for i in range(17):
        assert_equal(l[i].x, Float32(i))


struct Pinned(Movable where False):
    pass


def test_list_non_movable() raises:
    var empty = List[Pinned]()
    assert_equal(len(empty), 0)


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
