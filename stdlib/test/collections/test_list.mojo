# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from collections import List
from memory import UnsafePointer
from sys.info import sizeof
from test_utils import CopyCounter, MoveCounter
from testing import assert_equal, assert_false, assert_raises, assert_true

from utils import Span


def test_mojo_issue_698[sbo_size: Int]():
    var list = List[Float64, sbo_size]()
    for i in range(5):
        list.append(i)

    assert_equal(0.0, list[0])
    assert_equal(1.0, list[1])
    assert_equal(2.0, list[2])
    assert_equal(3.0, list[3])
    assert_equal(4.0, list[4])


def test_list[sbo_size: Int]():
    var list = List[Int, sbo_size]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(5 * sizeof[Int](), list.bytecount())
    assert_equal(0, list[0])
    assert_equal(1, list[1])
    assert_equal(2, list[2])
    assert_equal(3, list[3])
    assert_equal(4, list[4])

    assert_equal(0, list[-5])
    assert_equal(3, list[-2])
    assert_equal(4, list[-1])

    list[2] = -2
    assert_equal(-2, list[2])

    list[-5] = 5
    assert_equal(5, list[-5])
    list[-2] = 3
    assert_equal(3, list[-2])
    list[-1] = 7
    assert_equal(7, list[-1])


def test_list_unsafe_get[sbo_size: Int]():
    var list = List[Int, sbo_size]()

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


def test_list_unsafe_set[sbo_size: Int]():
    var list = List[Int, sbo_size]()

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


def test_list_clear[sbo_size: Int]():
    var list = List[Int, sbo_size](1, 2, 3)
    assert_equal(len(list), 3)
    # When we have a small buffer, the small buffer size
    # is the minimum capacity of the list
    assert_equal(list.capacity, max(3, sbo_size))
    list.clear()

    assert_equal(len(list), 0)
    assert_equal(list.capacity, max(3, sbo_size))


def test_list_to_bool_conversion[sbo_size: Int]():
    assert_false(List[String, sbo_size]())
    assert_true(List[String, sbo_size]("a"))
    assert_true(List[String, sbo_size]("", "a"))
    assert_true(List[String, sbo_size](""))


def test_list_pop[sbo_size: Int]():
    var list = List[Int, sbo_size]()
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

    # Test pop with negative index
    for i in range(0, 2):
        assert_equal(i, list.pop(-len(list)))

    # test default index as well
    assert_equal(2, list.pop())
    list.append(2)
    assert_equal(2, list.pop())

    # list should be empty now
    assert_equal(0, len(list))
    # capacity should be 1 according to shrink_to_fit behavior
    # but if there is a small buffer, the capacity can't go lower
    # than small buffer size
    assert_equal(max(1, sbo_size), list.capacity)


def test_list_variadic_constructor[sbo_size: Int]():
    var l = List[Int, sbo_size](2, 4, 6)
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

    var l2 = List[CopyCounter](CopyCounter(), CopyCounter(), CopyCounter())

    assert_equal(len(l2), 3)
    assert_equal(l2[0].copy_count, 0)
    assert_equal(l2[1].copy_count, 0)
    assert_equal(l2[2].copy_count, 0)


def test_list_resize[sbo_size: Int]():
    var l = List[Int, sbo_size](1)
    assert_equal(1, len(l))
    l.resize(2, 0)
    assert_equal(2, len(l))
    assert_equal(l[1], 0)
    l.resize(0)
    assert_equal(len(l), 0)


def test_list_reverse[sbo_size: Int]():
    #
    # Test reversing the list []
    #

    var vec = List[Int, sbo_size]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the list [123]
    #

    vec = List[Int, sbo_size]()

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    vec2 = List[String, sbo_size]("one", "two", "three")

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

    vec = List[Int, sbo_size]()
    vec.append(5)
    vec.append(10)

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 5)
    assert_equal(vec[1], 10)

    vec.reverse()

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 10)
    assert_equal(vec[1], 5)


def test_list_reverse_move_count[sbo_size: Int]():
    # Create this vec with enough capacity to avoid moves due to resizing.
    var vec = List[MoveCounter[Int], sbo_size](capacity=5)
    vec.append(MoveCounter(1))
    vec.append(MoveCounter(2))
    vec.append(MoveCounter(3))
    vec.append(MoveCounter(4))
    vec.append(MoveCounter(5))

    assert_equal(len(vec), 5)
    assert_equal(vec.data[0].value, 1)
    assert_equal(vec.data[1].value, 2)
    assert_equal(vec.data[2].value, 3)
    assert_equal(vec.data[3].value, 4)
    assert_equal(vec.data[4].value, 5)

    assert_equal(vec.data[0].move_count, 1)
    assert_equal(vec.data[1].move_count, 1)
    assert_equal(vec.data[2].move_count, 1)
    assert_equal(vec.data[3].move_count, 1)
    assert_equal(vec.data[4].move_count, 1)

    vec.reverse()

    assert_equal(len(vec), 5)
    assert_equal(vec.data[0].value, 5)
    assert_equal(vec.data[1].value, 4)
    assert_equal(vec.data[2].value, 3)
    assert_equal(vec.data[3].value, 2)
    assert_equal(vec.data[4].value, 1)

    # NOTE:
    # Earlier elements went through 2 moves and later elements went through 3
    # moves because the implementation of List.reverse arbitrarily
    # chooses to perform the swap of earlier and later elements by moving the
    # earlier element to a temporary (+1 move), directly move the later element
    # into the position the earlier element was in, and then move from the
    # temporary into the later position (+1 move).
    assert_equal(vec.data[0].move_count, 2)
    assert_equal(vec.data[1].move_count, 2)
    assert_equal(vec.data[2].move_count, 1)
    assert_equal(vec.data[3].move_count, 3)
    assert_equal(vec.data[4].move_count, 3)


def test_list_insert[sbo_size: Int]():
    #
    # Test the list [1, 2, 3] created with insert
    #

    v1 = List[Int, sbo_size]()
    v1.insert(len(v1), 1)
    v1.insert(len(v1), 3)
    v1.insert(1, 2)

    assert_equal(len(v1), 3)
    assert_equal(v1[0], 1)
    assert_equal(v1[1], 2)
    assert_equal(v1[2], 3)

    #
    # Test the list [1, 2, 3, 4, 5] created with negative and positive index
    #

    v2 = List[Int, sbo_size]()
    v2.insert(-1729, 2)
    v2.insert(len(v2), 3)
    v2.insert(len(v2), 5)
    v2.insert(-1, 4)
    v2.insert(-len(v2), 1)

    assert_equal(len(v2), 5)
    assert_equal(v2[0], 1)
    assert_equal(v2[1], 2)
    assert_equal(v2[2], 3)
    assert_equal(v2[3], 4)
    assert_equal(v2[4], 5)

    #
    # Test the list [1, 2, 3, 4] created with negative index
    #

    v3 = List[Int, sbo_size]()
    v3.insert(-11, 4)
    v3.insert(-13, 3)
    v3.insert(-17, 2)
    v3.insert(-19, 1)

    assert_equal(len(v3), 4)
    assert_equal(v3[0], 1)
    assert_equal(v3[1], 2)
    assert_equal(v3[2], 3)
    assert_equal(v3[3], 4)

    #
    # Test the list [1, 2, 3, 4, 5, 6, 7, 8] created with insert
    #

    v4 = List[Int, sbo_size]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i in range(len(v4)):
        assert_equal(v4[i], i + 1)


def test_list_index[sbo_size: Int]():
    var test_list_a = List[Int, sbo_size](10, 20, 30, 40, 50)

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
        _ = List[Int, sbo_size]().index(10)

    # Test empty slice
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=1, stop=1)
    # Test empty slice with 0 start and end
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=0, stop=0)

    var test_list_b = List[Int, sbo_size](10, 20, 30, 20, 10)

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


def test_list_extend[sbo_size: Int]():
    #
    # Test extending the list [1, 2, 3] with itself
    #

    vec = List[Int, sbo_size]()
    vec.append(1)
    vec.append(2)
    vec.append(3)

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)

    var copy = vec
    vec.extend(copy)

    # vec == [1, 2, 3, 1, 2, 3]
    assert_equal(len(vec), 6)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)
    assert_equal(vec[3], 1)
    assert_equal(vec[4], 2)
    assert_equal(vec[5], 3)


def test_list_extend_non_trivial[sbo_size: Int]():
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other lists
    #   - extend() using optimal number of __moveinit__() calls

    # Preallocate with enough capacity to avoid reallocation making the
    # move count checks below flaky.
    var v1 = List[MoveCounter[String], sbo_size](capacity=5)
    v1.append(MoveCounter[String]("Hello"))
    v1.append(MoveCounter[String]("World"))

    # different sbo sizes should work with extend()
    var v2 = List[MoveCounter[String], sbo_size + 1](capacity=3)
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

    assert_equal(v1.data[0].move_count, 1)
    assert_equal(v1.data[1].move_count, 1)
    assert_equal(v1.data[2].move_count, 2)
    assert_equal(v1.data[3].move_count, 2)
    assert_equal(v1.data[4].move_count, 2)


def test_2d_dynamic_list[sbo_size: Int]():
    # different small buffer sizes should work
    alias outer_size_sbo_size = sbo_size + 1
    var list = List[List[Int, sbo_size], outer_size_sbo_size]()

    for i in range(2):
        var v = List[Int, sbo_size]()
        for j in range(3):
            v.append(i + j)
        list.append(v)

    assert_equal(0, list[0][0])
    assert_equal(1, list[0][1])
    assert_equal(2, list[0][2])
    assert_equal(1, list[1][0])
    assert_equal(2, list[1][1])
    assert_equal(3, list[1][2])

    assert_equal(2, len(list))
    assert_equal(max(2, outer_size_sbo_size), list.capacity)

    assert_equal(3, len(list[0]))

    list[0].clear()
    assert_equal(0, len(list[0]))
    # we verify that the capacity didn't decrease
    assert_true(list[0].capacity >= max(3, sbo_size))

    list.clear()
    assert_equal(0, len(list))
    assert_equal(max(2, outer_size_sbo_size), list.capacity)


def test_list_explicit_copy[sbo_size: Int]():
    var list = List[CopyCounter, sbo_size]()
    list.append(CopyCounter())
    var list_copy = List(other=list)
    assert_equal(0, list[0].copy_count)
    assert_equal(1, list_copy[0].copy_count)

    var l2 = List[Int]()
    for i in range(10):
        l2.append(i)

    var l2_copy = List(other=l2)
    assert_equal(len(l2), len(l2_copy))
    for i in range(len(l2)):
        assert_equal(l2[i], l2_copy[i])


@value
struct CopyCountedStruct(CollectionElement):
    var counter: CopyCounter
    var value: String

    fn __init__(inout self, *, other: Self):
        self.counter = CopyCounter(other=other.counter)
        self.value = String(other=other.value)

    fn __init__(inout self, value: String):
        self.counter = CopyCounter()
        self.value = value


def test_no_extra_copies_with_sugared_set_by_field[sbo_size: Int]():
    var list = List[List[CopyCountedStruct, sbo_size], sbo_size](capacity=1)
    var child_list = List[CopyCountedStruct, sbo_size](capacity=2)
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


# Ensure correct behavior of __copyinit__
# as reported in GH issue 27875 internally and
# https://github.com/modularml/mojo/issues/1493
def test_list_copy_constructor[sbo_size: Int]():
    var vec = List[Int, sbo_size](capacity=1)
    var vec_copy = vec
    vec_copy.append(1)  # Ensure copy constructor doesn't crash
    _ = vec^  # To ensure previous one doesn't invoke move constructor


def test_list_iter[sbo_size: Int]():
    var vs = List[Int, sbo_size]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: List[Int, _]) -> Int:
        var sum = 0
        for v in vs:
            sum += v[]
        return sum

    assert_equal(6, sum(vs))


def test_list_iter_mutable[sbo_size: Int]():
    var vs = List[Int, sbo_size](1, 2, 3)

    for v in vs:
        v[] += 1

    var sum = 0
    for v in vs:
        sum += v[]

    assert_equal(9, sum)


def test_list_span[sbo_size: Int]():
    var vs = List[Int, sbo_size](1, 2, 3)

    var es = vs[1:]
    assert_equal(es[0], 2)
    assert_equal(es[1], 3)
    assert_equal(len(es), 2)

    es = vs[:-1]
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

    es = vs[:]
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
    es = vs[-50::]
    assert_equal(3, len(es))
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
    es = vs[:-50:-1]
    assert_equal(3, len(es))
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(es[2], 1)
    es = vs[:50:]
    assert_equal(3, len(es))
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
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


def test_list_realloc_trivial_types():
    a = List[Int, hint_trivial_type=True]()
    for i in range(100):
        a.append(i)

    assert_equal(len(a), 100)
    for i in range(100):
        assert_equal(a[i], i)

    b = List[Int8, hint_trivial_type=True]()
    for i in range(100):
        b.append(Int8(i))

    assert_equal(len(b), 100)
    for i in range(100):
        assert_equal(b[i], Int8(i))


def test_list_realloc_trivial_types[sbo_size: Int]():
    a = List[Int, sbo_size, hint_trivial_type=True]()
    for i in range(100):
        a.append(i)

    b = List[Int8, sbo_size, hint_trivial_type=True]()
    for i in range(100):
        b.append(Int8(i))


def test_list_boolable[sbo_size: Int]():
    assert_true(List[Int, sbo_size](1))
    assert_false(List[Int, sbo_size]())


def test_constructor_from_pointer[sbo_size: Int]():
    new_pointer = UnsafePointer[Int8].alloc(5)
    new_pointer[0] = 0
    new_pointer[1] = 1
    new_pointer[2] = 2
    # rest is not initialized

    var some_list = List[Int8, sbo_size](
        unsafe_pointer=new_pointer, size=3, capacity=5
    )
    assert_equal(some_list[0], 0)
    assert_equal(some_list[1], 1)
    assert_equal(some_list[2], 2)
    assert_equal(len(some_list), 3)
    # Here the small buffer is not used because a pointer was
    # passed to the constructor and we don't need to do a copy,
    # so the capacity is the one given and has nothing to do
    # with the small buffer size.
    assert_equal(some_list.capacity, 5)


def test_constructor_from_other_list_through_pointer[sbo_size: Int]():
    initial_list = List[Int8, sbo_size](0, 1, 2)
    # we do a backup of the size and capacity because
    # the list attributes will be invalid after the steal_data call
    var size = len(initial_list)
    var capacity = initial_list.capacity
    # We check that it's possible to use different small buffer sizes
    alias new_list_sbo_size = sbo_size + 1
    var some_list = List[Int8, new_list_sbo_size](
        unsafe_pointer=initial_list.steal_data(), size=size, capacity=capacity
    )
    assert_equal(some_list[0], 0)
    assert_equal(some_list[1], 1)
    assert_equal(some_list[2], 2)
    assert_equal(len(some_list), size)
    assert_true(some_list.capacity >= capacity)


def test_converting_list_to_string[sbo_size: Int]():
    # This is also testing the method `to_format` because
    # essentially, `List.__str__()` just creates a String and applies `to_format` to it.
    # If we were to write unit tests for `to_format`, we would essentially copy-paste the code
    # of `List.__str__()`
    var my_list = List[Int, sbo_size](1, 2, 3)
    assert_equal(my_list.__str__(), "[1, 2, 3]")

    var my_list4 = List[String, sbo_size]("a", "b", "c", "foo")
    assert_equal(my_list4.__str__(), "['a', 'b', 'c', 'foo']")


def test_list_count[sbo_size: Int]():
    var list = List[Int, sbo_size](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, list.count(1))
    assert_equal(2, list.count(2))
    assert_equal(0, list.count(4))

    var list2 = List[Int, sbo_size]()
    assert_equal(0, list2.count(1))


def test_list_add[sbo_size: Int]():
    # We make sure that it works with different small buffer sizes
    alias a_sbo_size = sbo_size
    alias b_sbo_size = sbo_size + 1
    alias c_sbo_size = sbo_size + 2
    alias d_sbo_size = sbo_size + 3
    alias e_sbo_size = sbo_size + 4
    alias l_sbo_size = sbo_size + 5

    var a = List[Int, a_sbo_size](1, 2, 3)
    var b = List[Int, b_sbo_size](4, 5, 6)
    var c = a + b
    assert_equal(len(c), 6)
    # check that original values aren't modified
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    assert_equal(c.__str__(), "[1, 2, 3, 4, 5, 6]")

    a += b
    assert_equal(len(a), 6)
    assert_equal(a.__str__(), "[1, 2, 3, 4, 5, 6]")
    assert_equal(len(b), 3)

    a = List[Int, a_sbo_size](1, 2, 3)
    a += b^
    assert_equal(len(a), 6)
    assert_equal(a.__str__(), "[1, 2, 3, 4, 5, 6]")

    var d = List[Int, d_sbo_size](1, 2, 3)
    var e = List[Int, e_sbo_size](4, 5, 6)
    var f = d + e^
    assert_equal(len(f), 6)
    assert_equal(f.__str__(), "[1, 2, 3, 4, 5, 6]")

    var l = List[Int, l_sbo_size](1, 2, 3)
    l += List[Int, sbo_size]()
    assert_equal(len(l), 3)


def test_list_mult[sbo_size: Int]():
    var a = List[Int, sbo_size](1, 2, 3)
    var b = a * 2
    assert_equal(len(b), 6)
    assert_equal(b.__str__(), "[1, 2, 3, 1, 2, 3]")
    b = a * 3
    assert_equal(len(b), 9)
    assert_equal(b.__str__(), "[1, 2, 3, 1, 2, 3, 1, 2, 3]")
    a *= 2
    assert_equal(len(a), 6)
    assert_equal(a.__str__(), "[1, 2, 3, 1, 2, 3]")

    var l = List[Int, sbo_size](1, 2)
    l *= 1
    assert_equal(len(l), 2)

    l *= 0
    assert_equal(len(l), 0)
    assert_equal(len(List[Int, sbo_size](1, 2, 3) * 0), 0)


def test_list_contains[sbo_size: Int]():
    var x = List[Int, sbo_size](1, 2, 3)
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)

    # TODO: implement List.__eq__ for Self[ComparableCollectionElement]
    # var y = List[List[Int]]()
    # y.append(List(1,2))
    # assert_equal(List(1,2) in y,True)
    # assert_equal(List(0,1) in y,False)


def test_list_eq_ne():
    var l1 = List[Int](1, 2, 3)
    var l2 = List[Int](1, 2, 3)
    assert_true(l1 == l2)
    assert_false(l1 != l2)

    var l3 = List[Int](1, 2, 3, 4)
    assert_false(l1 == l3)
    assert_true(l1 != l3)

    var l4 = List[Int]()
    var l5 = List[Int]()
    assert_true(l4 == l5)
    assert_true(l1 != l4)

    var l6 = List[String]("a", "b", "c")
    var l7 = List[String]("a", "b", "c")
    var l8 = List[String]("a", "b")
    assert_true(l6 == l7)
    assert_false(l6 != l7)
    assert_false(l6 == l8)


def test_list_init_span[sbo_size: Int]():
    var l = List[String, sbo_size]("a", "bb", "cc", "def")
    var sp = Span(l)
    var l2 = List[String, sbo_size](sp)
    for i in range(len(l)):
        assert_equal(l[i], l2[i])


def test_indexing[sbo_size: Int]():
    var l = List[Int, sbo_size](1, 2, 3)
    assert_equal(l[int(1)], 2)
    assert_equal(l[False], 1)
    assert_equal(l[True], 2)
    assert_equal(l[2], 3)


def test_materialization[sbo_size: Int]():
    # TODO: Fix materialization when sbo is used
    alias l = List[Int](10, 20, 30)
    var l2 = l
    assert_equal(l[0], l2[0])
    assert_equal(l[1], l2[1])
    assert_equal(l[2], l2[2])
    assert_equal(l2[0], 10)
    assert_equal(l2[1], 20)
    assert_equal(l2[2], 30)


# ===-------------------------------------------------------------------===#
# List dtor tests
# ===-------------------------------------------------------------------===#
var g_dtor_count: Int = 0


struct DtorCounter(CollectionElement):
    # NOTE: payload is required because List does not support zero sized structs.
    var payload: Int

    fn __init__(inout self):
        self.payload = 0

    fn __init__(inout self, *, other: Self):
        self.payload = other.payload

    fn __copyinit__(inout self, existing: Self, /):
        self.payload = existing.payload

    fn __moveinit__(inout self, owned existing: Self, /):
        self.payload = existing.payload
        existing.payload = 0

    fn __del__(owned self):
        g_dtor_count += 1


def inner_test_list_dtor[sbo_size: Int]():
    # explicitly reset global counter
    g_dtor_count = 0

    var l = List[DtorCounter]()
    assert_equal(g_dtor_count, 0)

    l.append(DtorCounter())
    assert_equal(g_dtor_count, 0)

    l.__del__()
    assert_equal(g_dtor_count, 1)


def test_list_dtor[sbo_size: Int]():
    # call another function to force the destruction of the list
    inner_test_list_dtor[sbo_size]()

    # verify we still only ran the destructor once
    assert_equal(g_dtor_count, 1)


def test_list_repr():
    var l = List(1, 2, 3)
    assert_equal(l.__repr__(), "[1, 2, 3]")
    var empty = List[Int]()
    assert_equal(empty.__repr__(), "[]")


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#
def main():
    @parameter
    for small_buffer_size in range(8):
        test_mojo_issue_698[small_buffer_size]()
        test_list[small_buffer_size]()
        test_list_unsafe_get[small_buffer_size]()
        test_list_unsafe_set[small_buffer_size]()
        test_list_clear[small_buffer_size]()
        test_list_to_bool_conversion[small_buffer_size]()
        test_list_pop[small_buffer_size]()
        test_list_variadic_constructor[small_buffer_size]()
        test_list_resize[small_buffer_size]()
        test_list_reverse[small_buffer_size]()
        test_list_reverse_move_count[small_buffer_size]()
        test_list_insert[small_buffer_size]()
        test_list_index[small_buffer_size]()
        test_list_extend[small_buffer_size]()
        test_list_extend_non_trivial[small_buffer_size]()
        test_list_explicit_copy[small_buffer_size]()
        test_no_extra_copies_with_sugared_set_by_field[small_buffer_size]()
        test_list_copy_constructor[small_buffer_size]()
        test_2d_dynamic_list[small_buffer_size]()
        test_list_iter[small_buffer_size]()
        test_list_iter_mutable[small_buffer_size]()
        test_list_span[small_buffer_size]()
        test_list_realloc_trivial_types[small_buffer_size]()
        test_list_realloc_trivial_types()
        test_list_boolable[small_buffer_size]()
        test_constructor_from_pointer[small_buffer_size]()
        test_constructor_from_other_list_through_pointer[small_buffer_size]()
        test_converting_list_to_string[small_buffer_size]()
        test_list_count[small_buffer_size]()
        test_list_add[small_buffer_size]()
        test_list_mult[small_buffer_size]()
        test_list_contains[small_buffer_size]()
        test_indexing[small_buffer_size]()
        test_materialization[small_buffer_size]()
        test_list_dtor[small_buffer_size]()
        test_list_repr()
