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

from test_utils import CopyCounter, MoveCounter
from testing import (
    assert_equal,
    assert_not_equal,
    assert_false,
    assert_true,
    assert_raises,
)


def test_mojo_issue_698():
    var list = List[Float64]()
    for i in range(5):
        list.append(i)

    assert_equal(0.0, list[0])
    assert_equal(1.0, list[1])
    assert_equal(2.0, list[2])
    assert_equal(3.0, list[3])
    assert_equal(4.0, list[4])


def test_list():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var list = List[Int, sbo_size]()

        for i in range(5):
            list.append(i)

        assert_equal(5, len(list))
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

    unroll[test_function, 10]()


def test_list_clear():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var list = List[Int, sbo_size](1, 2, 3)
        assert_equal(len(list), 3)
        assert_equal(list.capacity, max(3, sbo_size))
        list.clear()

        assert_equal(len(list), 0)
        assert_equal(list.capacity, max(3, sbo_size))

    unroll[test_function, 10]()


def test_list_to_bool_conversion():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        assert_false(List[String, sbo_size]())
        assert_true(List[String, sbo_size]("a"))
        assert_true(List[String, sbo_size]("", "a"))
        assert_true(List[String, sbo_size](""))

    unroll[test_function, 10]()


def test_list_pop():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var list = List[Int, sbo_size]()
        # Test pop with index
        for i in range(6):
            list.append(i)

        # try poping from index 3 for 3 times
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
        assert_equal(list.capacity, max(1, sbo_size))

    unroll[test_function, 10]()


def test_list_variadic_constructor():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var l = List[Int, sbo_size](2, 4, 6)
        assert_equal(3, len(l))
        assert_equal(2, l[0])
        assert_equal(4, l[1])
        assert_equal(6, l[2])

        l.append(8)
        assert_equal(4, len(l))
        assert_equal(8, l[3])

    unroll[test_function, 10]()


def test_list_resize():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var l = List[Int, sbo_size](1)
        assert_equal(1, len(l))
        l.resize(2, 0)
        assert_equal(2, len(l))
        assert_equal(l[1], 0)
        l.resize(0)
        assert_equal(len(l), 0)

    unroll[test_function, 10]()


def test_list_reverse():
    @parameter
    fn test_function[sbo_size: Int]() raises:
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

        var vec2 = List[String, sbo_size]("one", "two", "three")

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

        #
        # Test reversing the list [1, 2, 3, 4, 5] starting at the 3rd position
        # to produce [1, 2, 5, 4, 3]
        #

        vec = List[Int, sbo_size]()
        vec.append(1)
        vec.append(2)
        vec.append(3)
        vec.append(4)
        vec.append(5)

        assert_equal(len(vec), 5)
        assert_equal(vec[0], 1)
        assert_equal(vec[1], 2)
        assert_equal(vec[2], 3)
        assert_equal(vec[3], 4)
        assert_equal(vec[4], 5)

        vec._reverse(start=2)

        assert_equal(len(vec), 5)
        assert_equal(vec[0], 1)
        assert_equal(vec[1], 2)
        assert_equal(vec[2], 5)
        assert_equal(vec[3], 4)
        assert_equal(vec[4], 3)

        #
        # Test edge case of reversing the list [1, 2, 3] but starting after the
        # last element.
        #

        vec = List[Int, sbo_size]()
        vec.append(1)
        vec.append(2)
        vec.append(3)

        vec._reverse(start=len(vec))

        assert_equal(len(vec), 3)
        assert_equal(vec[0], 1)
        assert_equal(vec[1], 2)
        assert_equal(vec[2], 3)

    unroll[test_function, 10]()


def test_list_reverse_move_count():
    @parameter
    fn test_function[sbo_size: Int]() raises:
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

        # Keep vec alive until after we've done the last `vec.data + N` read.
        _ = vec^

    unroll[test_function, 10]()


def test_list_insert():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        #
        # Test the list [1, 2, 3] created with insert
        #

        var v1 = List[Int, sbo_size]()
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

        var v2 = List[Int, sbo_size]()
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

        var v3 = List[Int, sbo_size]()
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

        var v4 = List[Int, sbo_size]()
        for i in range(4):
            v4.insert(0, 4 - i)
            v4.insert(len(v4), 4 + i + 1)

        for i in range(len(v4)):
            assert_equal(v4[i], i + 1)

    unroll[test_function, 10]()


def test_list_index():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var test_list_a = List[Int, sbo_size](10, 20, 30, 40, 50)

        # Basic Functionality Tests
        assert_equal(__type_of(test_list_a).index(test_list_a, 10), 0)
        assert_equal(__type_of(test_list_a).index(test_list_a, 30), 2)
        assert_equal(__type_of(test_list_a).index(test_list_a, 50), 4)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 60)

        # Tests With Start Parameter
        assert_equal(__type_of(test_list_a).index(test_list_a, 30, start=1), 2)
        assert_equal(__type_of(test_list_a).index(test_list_a, 30, start=-4), 2)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 30, start=3)
        with assert_raises(
            contains=(
                "Given 'start' parameter (5) is out of range. List only has 5"
                " elements."
            )
        ):
            __type_of(test_list_a).index(test_list_a, 30, start=5)

        # Tests With Start and End Parameters
        assert_equal(
            __type_of(test_list_a).index(test_list_a, 30, start=1, end=3), 2
        )
        assert_equal(
            __type_of(test_list_a).index(test_list_a, 30, start=-4, end=-2), 2
        )
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 30, start=1, end=2)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 30, start=3, end=1)

        # Tests With End Parameter Only
        assert_equal(__type_of(test_list_a).index(test_list_a, 30, end=3), 2)
        assert_equal(__type_of(test_list_a).index(test_list_a, 30, end=-2), 2)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 30, end=1)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 30, end=2)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 60, end=50)

        # Edge Cases and Special Conditions
        assert_equal(
            __type_of(test_list_a).index(test_list_a, 10, start=-5, end=-1), 0
        )
        assert_equal(
            __type_of(test_list_a).index(test_list_a, 10, start=0, end=50), 0
        )
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 50, start=-5, end=-1)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 50, start=0, end=-1)
        with assert_raises(contains="ValueError: Given element is not in list"):
            __type_of(test_list_a).index(test_list_a, 10, start=-4, end=-1)
        with assert_raises(
            contains=(
                "Given 'start' parameter (5) is out of range. List only has 5"
                " elements."
            )
        ):
            __type_of(test_list_a).index(test_list_a, 10, start=5, end=50)
        with assert_raises(
            contains="Cannot find index of a value in an empty list."
        ):
            __type_of(List[Int, sbo_size]()).index(List[Int](), 10)

        var test_list_b = List[Int](10, 20, 30, 20, 10)

        # Test finding the first occurrence of an item
        assert_equal(__type_of(test_list_b).index(test_list_b, 10), 0)
        assert_equal(__type_of(test_list_b).index(test_list_b, 20), 1)

        # Test skipping the first occurrence with a start parameter
        assert_equal(__type_of(test_list_b).index(test_list_b, 20, start=2), 3)

        # Test constraining search with start and end, excluding last occurrence
        with assert_raises(contains="ValueError: Given element is not in list"):
            _ = __type_of(test_list_b).index(test_list_b, 10, start=1, end=4)

        # Test search within a range that includes multiple occurrences
        assert_equal(
            __type_of(test_list_b).index(test_list_b, 20, start=1, end=4), 1
        )

        # Verify error when constrained range excludes occurrences
        with assert_raises(contains="ValueError: Given element is not in list"):
            _ = __type_of(test_list_b).index(test_list_b, 20, start=4, end=5)

    unroll[test_function, 10]()


def test_list_extend():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        #
        # Test extending the list [1, 2, 3] with itself
        #

        var vec = List[Int, sbo_size]()
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

        vec._reverse(start=3)

        # vec == [1, 2, 3, 3, 2, 1]
        assert_equal(len(vec), 6)
        assert_equal(vec[0], 1)
        assert_equal(vec[1], 2)
        assert_equal(vec[2], 3)
        assert_equal(vec[3], 3)
        assert_equal(vec[4], 2)
        assert_equal(vec[5], 1)

    unroll[test_function, 10]()


def test_list_extend_non_trivial():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        # Tests three things:
        #   - extend() for non-plain-old-data types
        #   - extend() with mixed-length self and other lists
        #   - extend() using optimal number of __moveinit__() calls

        # Preallocate with enough capacity to avoid reallocation making the
        # move count checks below flaky.
        var v1 = List[MoveCounter[String], sbo_size](capacity=5)
        v1.append(MoveCounter[String]("Hello"))
        v1.append(MoveCounter[String]("World"))

        # We check that it's possible with different buffer sizes
        var v2 = List[MoveCounter[String], sbo_size + 1](capacity=3)
        v2.append(MoveCounter[String]("Foo"))
        v2.append(MoveCounter[String]("Bar"))
        v2.append(MoveCounter[String]("Baz"))

        v1.extend(v2)

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

        # Keep v1 alive until after we've done the last `vec.data + N` read.
        _ = v1^

    unroll[test_function, 10]()


def test_2d_dynamic_list():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        alias sbo_size_inner_list = sbo_size + 1
        var list = List[List[Int, sbo_size_inner_list], sbo_size]()

        for i in range(2):
            var v = List[Int]()
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
        assert_equal(max(2, sbo_size), list.capacity)

        assert_equal(3, len(list[0]))

        list[0].clear()
        assert_equal(0, len(list[0]))
        assert_equal(max(4, sbo_size_inner_list), list[0].capacity)

        list.clear()
        assert_equal(0, len(list))
        assert_equal(max(2, sbo_size), list.capacity)

    unroll[test_function, 10]()


def test_list_explicit_copy():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var list = List[CopyCounter, sbo_size]()
        list.append(CopyCounter()^)
        var list_copy = List(list)
        assert_equal(0, list.__get_ref(0)[].copy_count)
        assert_equal(1, list_copy.__get_ref(0)[].copy_count)

        var l2 = List[Int, sbo_size + 1]()
        for i in range(10):
            l2.append(i)

        var l2_copy = List[Int, sbo_size + 2](l2)
        assert_equal(len(l2), len(l2_copy))
        for i in range(len(l2)):
            assert_equal(l2[i], l2_copy[i])

    unroll[test_function, 10]()


# Ensure correct behavior of __copyinit__
# as reported in GH issue 27875 internally and
# https://github.com/modularml/mojo/issues/1493
def test_list_copy_constructor_issue_1493():
    var vec = List[Int](capacity=1)
    var vec_copy = vec
    vec_copy.append(1)  # Ensure copy constructor doesn't crash
    _ = vec^  # To ensure previous one doesn't invoke move constuctor


def test_list_iter():
    var vs = List[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: List[Int]) -> Int:
        var sum = 0
        for v in vs:
            sum += v[]
        return sum

    assert_equal(6, sum(vs))


def test_list_iter_mutable():
    var vs = List[Int](1, 2, 3)

    for v in vs:
        v[] += 1

    var sum = 0
    for v in vs:
        sum += v[]

    assert_equal(9, sum)


def test_list_span():
    var vs = List[Int](1, 2, 3)

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


def test_list_boolable():
    assert_true(List[Int](1))
    assert_false(List[Int]())


def test_constructor_from_pointer():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var new_pointer = UnsafePointer[Int8].alloc(5)
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
        assert_equal(some_list.capacity, 5)

    unroll[test_function, 10]()


def test_constructor_from_other_list_through_pointer():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var initial_list = List[Int8, sbo_size](0, 1, 2)
        # we do a backup of the size and capacity because
        # the list attributes will be invalid after the steal_data call
        var size = len(initial_list)
        var capacity = initial_list.capacity
        var some_list = List[Int8, sbo_size + 1](
            unsafe_pointer=initial_list.steal_data(),
            size=size,
            capacity=capacity,
        )
        assert_equal(some_list[0], 0)
        assert_equal(some_list[1], 1)
        assert_equal(some_list[2], 2)
        assert_equal(len(some_list), size)
        assert_equal(some_list.capacity, capacity)

    unroll[test_function, 10]()


def test_converting_list_to_string():
    var my_list = List[Int](1, 2, 3)
    assert_equal(__type_of(my_list).__str__(my_list), "[1, 2, 3]")

    var my_list4 = List[String]("a", "b", "c", "foo")
    assert_equal(
        __type_of(my_list4).__str__(my_list4), "['a', 'b', 'c', 'foo']"
    )


def test_list_count():
    var list = List[Int](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, __type_of(list).count(list, 1))
    assert_equal(2, __type_of(list).count(list, 2))
    assert_equal(0, __type_of(list).count(list, 4))

    var list2 = List[Int]()
    assert_equal(0, __type_of(list2).count(list2, 1))


def test_list_add():
    var a = List[Int](1, 2, 3)
    var b = List[Int](4, 5, 6)
    var c = a + b
    assert_equal(len(c), 6)
    # check that original values aren't modified
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    assert_equal(__type_of(c).__str__(c), "[1, 2, 3, 4, 5, 6]")

    a += b
    assert_equal(len(a), 6)
    assert_equal(__type_of(a).__str__(a), "[1, 2, 3, 4, 5, 6]")
    assert_equal(len(b), 3)

    a = List[Int](1, 2, 3)
    a += b^
    assert_equal(len(a), 6)
    assert_equal(__type_of(a).__str__(a), "[1, 2, 3, 4, 5, 6]")

    var d = List[Int](1, 2, 3)
    var e = List[Int](4, 5, 6)
    var f = d + e^
    assert_equal(len(f), 6)
    assert_equal(__type_of(f).__str__(f), "[1, 2, 3, 4, 5, 6]")

    var l = List[Int](1, 2, 3)
    l += List[Int]()
    assert_equal(len(l), 3)


def test_list_mult():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var a = List[Int, sbo_size](1, 2, 3)
        var b = a * 2
        assert_equal(len(b), 6)
        assert_equal(__type_of(b).__str__(b), "[1, 2, 3, 1, 2, 3]")
        b = a * 3
        assert_equal(len(b), 9)
        assert_equal(__type_of(b).__str__(b), "[1, 2, 3, 1, 2, 3, 1, 2, 3]")
        a *= 2
        assert_equal(len(a), 6)
        assert_equal(__type_of(a).__str__(a), "[1, 2, 3, 1, 2, 3]")

        var l = List[Int, sbo_size](1, 2)
        l *= 1
        assert_equal(len(l), 2)

        l *= 0
        assert_equal(len(l), 0)
        assert_equal(len(List[Int, sbo_size](1, 2, 3) * 0), 0)

    unroll[test_function, 10]()


def test_bytes_with_small_size_optimization():
    var small_list = List[Int8, _small_buffer_size=3]()
    assert_equal(len(small_list), 0)

    small_list.append(0)
    small_list.append(10)
    small_list.append(20)

    assert_equal(len(small_list), 3)
    assert_equal(small_list.capacity, 3)

    # We should still be on the stack
    assert_true(small_list._sbo_is_in_use())

    assert_equal(small_list[0], 0)
    assert_equal(small_list[1], 10)
    assert_equal(small_list[2], 20)

    # We check that the pointer makes sense
    var pointer = small_list.unsafe_ptr()
    assert_equal(pointer[0], 0)
    assert_equal(pointer[1], 10)
    assert_equal(pointer[2], 20)

    small_list.append(30)
    assert_equal(small_list[0], 0)
    assert_equal(small_list[1], 10)
    assert_equal(small_list[2], 20)
    assert_equal(small_list[3], 30)
    assert_false(small_list._sbo_is_in_use())
    assert_equal(len(small_list), 4)

    small_list[1] = 100
    assert_equal(small_list[1], 100)

    small_list[3] = 13
    assert_equal(small_list[3], 13)

    small_list.append(42)
    assert_equal(small_list[4], 42)
    assert_equal(len(small_list), 5)

    # We check that the pointer makes sense
    pointer = small_list.unsafe_ptr()
    assert_equal(pointer[0], 0)
    assert_equal(pointer[1], 100)
    assert_equal(pointer[2], 20)
    assert_equal(pointer[3], 13)
    assert_equal(pointer[4], 42)

    # Otherwise, everything is destroyed before we had the chance to check the pointer
    _ = small_list


def test_list_copy_constructor():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var vec = List[Int, sbo_size](3, 2, 1)
        var vec_copy = vec
        assert_not_equal(vec.data, vec_copy.data)
        vec_copy.append(1)  # Ensure copy constructor doesn't crash
        vec_copy[0] = 0
        assert_equal(len(vec), 3)
        assert_equal(len(vec_copy), 4)
        assert_equal(vec[0], 3)
        assert_equal(vec_copy[0], 0)
        assert_not_equal(vec.data, vec_copy.data)

        _ = vec^  # To ensure previous one doesn't invoke move constuctor

    unroll[test_function, 10]()


def test_list_move_constructor():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var vec = List[UInt8, sbo_size](0, 10, 20, 30)

        var vec_moved = vec^
        assert_equal(len(vec_moved), 4)
        assert_equal(vec_moved[0], 0)
        assert_equal(vec_moved[1], 10)
        assert_equal(vec_moved[2], 20)
        assert_equal(vec_moved[3], 30)

        vec_moved.append(40)
        assert_equal(len(vec_moved), 5)
        assert_equal(vec_moved[4], 40)

        vec_moved[0] = 1
        assert_equal(vec_moved[0], 1)

        # Verify the pointer
        var pointer = vec_moved.unsafe_ptr()
        assert_equal(pointer[0], 1)
        assert_equal(pointer[1], 10)
        assert_equal(pointer[2], 20)
        assert_equal(pointer[3], 30)
        assert_equal(pointer[4], 40)

        _ = vec_moved  # Keep the list alive for pointer tests

    unroll[test_function, 10]()


def test_constructor_from_another_list():
    @parameter
    fn test_function[sbo_size: Int]() raises:
        var vec = List[UInt8, sbo_size](0, 10, 20, 30)

        var vec_copy = List[UInt8, sbo_size + 1](vec)
        assert_equal(len(vec_copy), 4)
        assert_equal(vec_copy[0], 0)
        assert_equal(vec_copy[1], 10)
        assert_equal(vec_copy[2], 20)
        assert_equal(vec_copy[3], 30)

    unroll[test_function, 10]()


def main():
    test_mojo_issue_698()
    test_list()
    test_list_clear()
    test_list_to_bool_conversion()
    test_list_pop()
    test_list_variadic_constructor()
    test_list_resize()
    test_list_reverse()
    test_list_reverse_move_count()
    test_list_insert()
    test_list_index()
    test_list_extend()
    test_list_extend_non_trivial()
    test_list_explicit_copy()
    test_list_copy_constructor_issue_1493()
    test_2d_dynamic_list()
    test_list_iter()
    test_list_iter_mutable()
    test_list_span()
    test_list_boolable()
    test_constructor_from_pointer()
    test_constructor_from_other_list_through_pointer()
    test_converting_list_to_string()
    test_list_count()
    test_list_add()
    test_list_mult()
    test_bytes_with_small_size_optimization()
    test_list_copy_constructor()
    test_list_move_constructor()
    test_constructor_from_another_list()
