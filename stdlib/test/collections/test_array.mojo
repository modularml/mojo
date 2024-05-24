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

from collections.array import Array

from test_utils import CopyCounter, MoveCounter
from testing import assert_equal, assert_false, assert_true, assert_raises
from utils import Span


def test_inlined_fixed_array():
    var array = Array[Int, 5](10)

    for i in range(5):
        array.append(i)

    # Verify it's iterable
    var index = 0
    for element in array:
        assert_equal(array[index], element[])
        index += 1

    assert_equal(5, len(array))

    # Can assign a specified index in static data range via `setitem`
    array[2] = -2
    assert_equal(0, array[0])
    assert_equal(1, array[1])
    assert_equal(-2, array[2])
    assert_equal(3, array[3])
    assert_equal(4, array[4])

    assert_equal(0, array[-5])
    assert_equal(3, array[-2])
    assert_equal(4, array[-1])

    array[-5] = 5
    assert_equal(5, array[-5])
    array[-2] = 3
    assert_equal(3, array[-2])
    array[-1] = 7
    assert_equal(7, array[-1])

    # Can assign past the static size into the regrowable dynamic data portion
    for j in range(5, 10):
        array.append(j)

    assert_equal(10, len(array))

    # Verify the dynamic data got properly assigned to from above
    assert_equal(5, array[5])
    assert_equal(6, array[6])
    assert_equal(7, array[7])
    assert_equal(8, array[8])
    assert_equal(9, array[9])

    assert_equal(9, array[-1])

    # Assign a specified index in the dynamic_data portion
    array[5] = -2
    assert_equal(-2, array[5])

    array.clear()
    assert_equal(0, len(array))


def test_inlined_fixed_array_with_default():
    var array = Array[Int](10)

    for i in range(5):
        array.append(i)

    assert_equal(5, len(array))

    array[2] = -2

    assert_equal(0, array[0])
    assert_equal(1, array[1])
    assert_equal(-2, array[2])
    assert_equal(3, array[3])
    assert_equal(4, array[4])

    for j in range(5, 10):
        array.append(j)

    assert_equal(10, len(array))

    assert_equal(5, array[5])

    array[5] = -2
    assert_equal(-2, array[5])

    array.clear()
    assert_equal(0, len(array))


def test_indexing_vec():
    var array = Array[Int](10)
    for i in range(5):
        array.append(i)
    assert_equal(0, array[int(0)])
    assert_equal(1, array[True])
    assert_equal(2, array[2])


def test_mojo_issue_698():
    var arr = Array[Float64]()
    for i in range(5):
        arr.append(i)

    assert_equal(0.0, arr[0])
    assert_equal(1.0, arr[1])
    assert_equal(2.0, arr[2])
    assert_equal(3.0, arr[3])
    assert_equal(4.0, arr[4])


def test_list():
    var arr = Array[Int]()

    for i in range(5):
        arr.append(i)

    assert_equal(5, len(arr))
    assert_equal(0, arr[0])
    assert_equal(1, arr[1])
    assert_equal(2, arr[2])
    assert_equal(3, arr[3])
    assert_equal(4, arr[4])

    assert_equal(0, arr[-5])
    assert_equal(3, arr[-2])
    assert_equal(4, arr[-1])

    arr[2] = -2
    assert_equal(-2, arr[2])

    arr[-5] = 5
    assert_equal(5, arr[-5])
    arr[-2] = 3
    assert_equal(3, arr[-2])
    arr[-1] = 7
    assert_equal(7, arr[-1])


def test_list_clear():
    var arr = Array[Int](1, 2, 3)
    assert_equal(len(arr), 3)
    assert_equal(arr.capacity, 3)
    arr.clear()

    assert_equal(len(arr), 0)
    assert_equal(arr.capacity, 3)


def test_list_to_bool_conversion():
    assert_false(Array[StringLiteral]())
    assert_true(Array[StringLiteral]("a"))
    assert_true(Array[StringLiteral]("", "a"))
    assert_true(Array[StringLiteral](""))


def test_list_pop():
    var arr = Array[Int]()
    # Test pop with index
    for i in range(6):
        arr.append(i)

    # try poping from index 3 for 3 times
    for i in range(3, 6):
        assert_equal(i, arr.pop(3))

    # list should have 3 elements now
    assert_equal(3, len(arr))
    assert_equal(0, arr[0])
    assert_equal(1, arr[1])
    assert_equal(2, arr[2])

    # Test pop with negative index
    for i in range(0, 2):
        assert_equal(i, arr.pop(-len(arr)))

    # test default index as well
    assert_equal(2, arr.pop())
    arr.append(2)
    assert_equal(2, arr.pop())

    # arr should be empty now
    assert_equal(0, len(arr))
    # capacity should be 1 according to shrink_to_fit behavior
    assert_equal(1, arr.capacity)


def test_list_variadic_constructor():
    var l = Array[Int](2, 4, 6)
    assert_equal(3, len(l))
    assert_equal(2, l[0])
    assert_equal(4, l[1])
    assert_equal(6, l[2])

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l[3])


def test_list_resize():
    var l = Array[Int](1)
    assert_equal(1, len(l))
    l.resize(2, 0)
    assert_equal(2, len(l))
    assert_equal(l[1], 0)
    l.resize(0)
    assert_equal(len(l), 0)


def test_list_reverse():
    #
    # Test reversing the list []
    #

    var vec = Array[Int]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the list [123]
    #

    vec = Array[Int]()

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    var vec2 = Array[StringLiteral]("one", "two", "three")

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

    vec = Array[Int]()
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

    vec = Array[Int]()
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
    # Test reversing the list [1, 2, 3] with negative indexes
    #

    vec = Array[Int]()
    vec.append(1)
    vec.append(2)
    vec.append(3)

    vec._reverse(start=-2)

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 3)
    assert_equal(vec[2], 2)

    #
    # Test reversing the list [1, 2] with out of bounds indexes
    #
    vec = Array[Int]()
    vec.append(1)
    vec.append(2)

    with assert_raises(contains="IndexError"):
        vec._reverse(start=-3)

    with assert_raises(contains="IndexError"):
        vec._reverse(start=3)

    #
    # Test edge case of reversing the list [1, 2, 3] but starting after the
    # last element.
    #

    vec = Array[Int]()
    vec.append(1)
    vec.append(2)
    vec.append(3)

    vec._reverse(start=len(vec))

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)


def test_list_insert():
    #
    # Test the list [1, 2, 3] created with insert
    #

    var v1 = Array[Int]()
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

    var v2 = Array[Int]()
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

    var v3 = Array[Int]()
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

    var v4 = Array[Int]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i in range(len(v4)):
        assert_equal(v4[i], i + 1)


def test_list_index():
    var test_list_a = Array[Int](10, 20, 30, 40, 50)

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
        _ = Array[Int]().index(10)

    # Test empty slice
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=1, stop=1)
    # Test empty slice with 0 start and end
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=0, stop=0)

    var test_list_b = Array[Int](10, 20, 30, 20, 10)

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


def test_list_extend():
    #
    # Test extending the list [1, 2, 3] with itself
    #

    var vec = Array[Int]()
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


def test_list_extend_non_trivial():
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other lists
    #   - extend() using optimal number of __moveinit__() calls

    # Preallocate with enough capacity to avoid reallocation making the
    # move count checks below flaky.
    var v1 = Array[MoveCounter[StringLiteral]](current_capacity=5)
    v1.append(MoveCounter[StringLiteral]("Hello"))
    v1.append(MoveCounter[StringLiteral]("World"))

    var v2 = Array[MoveCounter[StringLiteral]](current_capacity=3)
    v2.append(MoveCounter[StringLiteral]("Foo"))
    v2.append(MoveCounter[StringLiteral]("Bar"))
    v2.append(MoveCounter[StringLiteral]("Baz"))

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


def test_2d_dynamic_list():
    var arr = Array[Array[Int]]()

    for i in range(2):
        var v = Array[Int]()
        for j in range(3):
            v.append(i + j)
        arr.append(v)

    assert_equal(0, Array[0][0])
    assert_equal(1, Array[0][1])
    assert_equal(2, Array[0][2])
    assert_equal(1, Array[1][0])
    assert_equal(2, Array[1][1])
    assert_equal(3, Array[1][2])

    assert_equal(2, len(arr))
    assert_equal(2, arr.capacity)

    assert_equal(3, len(Array[0]))

    Array[0].clear()
    assert_equal(0, len(Array[0]))
    assert_equal(4, Array[0].capacity)

    arr.clear()
    assert_equal(0, len(arr))
    assert_equal(2, arr.capacity)


def test_list_explicit_copy():
    var arr = Array[CopyCounter]()
    arr.append(CopyCounter())
    var arr_copy = Array(arr)
    assert_equal(0, arr.__get_ref(0)[].copy_count)
    assert_equal(1, arr_copy.__get_ref(0)[].copy_count)

    var l2 = Array[Int]()
    for i in range(10):
        l2.append(i)

    var l2_copy = List(l2)
    assert_equal(len(l2), len(l2_copy))
    for i in range(len(l2)):
        assert_equal(l2[i], l2_copy[i])


# Ensure correct behavior of __copyinit__
# as reported in GH issue 27875 internally and
# https://github.com/modularml/mojo/issues/1493
def test_list_copy_constructor():
    var vec = Array[Int](current_capacity=1)
    var vec_copy = vec
    vec_copy.append(1)  # Ensure copy constructor doesn't crash
    _ = vec^  # To ensure previous one doesn't invoke move constuctor


def test_list_iter():
    var vs = Array[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: Array[Int]) -> Int:
        var sum = 0
        for v in vs:
            sum += v[]
        return sum

    assert_equal(6, sum(vs))


def test_list_iter_mutable():
    var vs = Array[Int](1, 2, 3)

    for v in vs:
        v[] += 1

    var sum = 0
    for v in vs:
        sum += v[]

    assert_equal(9, sum)


def test_list_span():
    var vs = Array[Int](1, 2, 3)

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
    assert_true(Array[Int](1))
    assert_false(Array[Int]())


def test_constructor_from_pointer():
    new_pointer = UnsafePointer[Int8].alloc(5)
    new_pointer[0] = 0
    new_pointer[1] = 1
    new_pointer[2] = 2
    # rest is not initialized

    var some_array = Array[Int8](
        unsafe_pointer=new_pointer, size=3, current_capacity=5
    )
    assert_equal(some_array[0], 0)
    assert_equal(some_array[1], 1)
    assert_equal(some_array[2], 2)
    assert_equal(len(some_array), 3)
    assert_equal(some_array.capacity, 5)


def test_constructor_from_other_list_through_pointer():
    # var initial_array = Array[Int8](0, 1, 2)
    # # we do a backup of the size and capacity because
    # # the list attributes will be invalid after the steal_data call
    # var size = len(initial_array)
    # var capacity = initial_array.capacity
    # var some_array = Array[Int8](
    #     unsafe_pointer=initial_array.steal_data(), size=size
    # )
    # assert_equal(some_array[0], 0)
    # assert_equal(some_array[1], 1)
    # assert_equal(some_array[2], 2)
    # assert_equal(len(some_list), size)
    # assert_equal(some_list.capacity, capacity)
    pass


def test_converting_list_to_string():
    var my_list = Array[Int](1, 2, 3)
    assert_equal(str(my_list), "[1, 2, 3]")

    var my_list4 = Array[StringLiteral]("a", "b", "c", "foo")
    assert_equal(str(my_list4), "['a', 'b', 'c', 'foo']")


def test_list_count():
    var list = Array[Int](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, list.count(1))
    assert_equal(2, list.count(2))
    assert_equal(0, list.count(4))

    var list2 = Array[Int]()
    assert_equal(0, list2.count(1))


def test_list_add():
    var a = Array[Int](1, 2, 3)
    var b = Array[Int](4, 5, 6)
    var c = a + b
    assert_equal(len(c), 6)
    # check that original values aren't modified
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    assert_equal(str(c), "[1, 2, 3, 4, 5, 6]")

    a += b
    assert_equal(len(a), 6)
    assert_equal(str(a), "[1, 2, 3, 4, 5, 6]")
    assert_equal(len(b), 3)

    a = Array[Int](1, 2, 3)
    a += b^
    assert_equal(len(a), 6)
    assert_equal(str(a), "[1, 2, 3, 4, 5, 6]")

    var d = Array[Int](1, 2, 3)
    var e = Array[Int](4, 5, 6)
    var f = d + e^
    assert_equal(len(f), 6)
    assert_equal(str(f), "[1, 2, 3, 4, 5, 6]")

    var l = Array[Int](1, 2, 3)
    l += Array[Int]()
    assert_equal(len(l), 3)


def test_list_mult():
    var a = Array[Int](1, 2, 3)
    var b = a * 2
    assert_equal(len(b), 6)
    assert_equal(str(b), "[1, 2, 3, 1, 2, 3]")
    b = a * 3
    assert_equal(len(b), 9)
    assert_equal(str(b), "[1, 2, 3, 1, 2, 3, 1, 2, 3]")
    a *= 2
    assert_equal(len(a), 6)
    assert_equal(str(a), "[1, 2, 3, 1, 2, 3]")

    var l = Array[Int](1, 2)
    l *= 1
    assert_equal(len(l), 2)

    l *= 0
    assert_equal(len(l), 0)
    assert_equal(len(Array[Int](1, 2, 3) * 0), 0)


def test_list_contains():
    var x = Array[Int](1, 2, 3)
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)


def test_list_init_span():
    var l = Array[StringLiteral]("a", "bb", "cc", "def")
    var sp = Span(l)
    var l2 = Array[StringLiteral](sp)
    for i in range(len(l)):
        assert_equal(l[i], l2[i])


def test_indexing_list():
    var l = Array[Int](1, 2, 3)
    assert_equal(l[int(1)], 2)
    assert_equal(l[False], 1)
    assert_equal(l[True], 2)
    assert_equal(l[2], 3)


def test_inline_list():
    var arr = Array[Int]()

    for i in range(5):
        arr.append(i)

    assert_equal(5, len(arr))
    assert_equal(0, arr[0])
    assert_equal(1, arr[1])
    assert_equal(2, arr[2])
    assert_equal(3, arr[3])
    assert_equal(4, arr[4])

    assert_equal(0, arr[-5])
    assert_equal(3, arr[-2])
    assert_equal(4, arr[-1])

    arr[2] = -2
    assert_equal(-2, arr[2])

    arr[-5] = 5
    assert_equal(5, arr[-5])
    arr[-2] = 3
    assert_equal(3, arr[-2])
    arr[-1] = 7
    assert_equal(7, arr[-1])


def test_append_triggers_a_move():
    var inline_array = Array[MoveCounter[Int], current_capacity=32]()

    var nb_elements_to_add = 8
    for i in range(nb_elements_to_add):
        inline_array.append(MoveCounter(i))

    # Using .append() should trigger a move and not a copy+delete.
    for i in range(nb_elements_to_add):
        assert_equal(inline_array[i].move_count, 1)


@value
struct ValueToCountDestructor(CollectionElement):
    var value: Int
    var destructor_counter: UnsafePointer[Array[Int]]

    fn __del__(owned self):
        self.destructor_counter[].append(self.value)


def test_destructor():
    """Ensure we delete the right number of elements."""
    var destructor_counter = Array[Int]()
    alias capacity = 32
    var inline_list = Array[
        ValueToCountDestructor,
        current_capacity=capacity,
        max_stack_size=capacity,
    ]()

    for index in range(capacity):
        inline_list.append(
            ValueToCountDestructor(index, UnsafePointer(destructor_counter))
        )

    # Private api use here:
    inline_list._size = 8

    # This is the last use of the inline list, so it should be destroyed here, along with each element.
    # It's important that we only destroy the first 8 elements, and not the 32 elements.
    # This is because we assume that the last 24 elements are not initialized (not true in this case,
    # but if we ever run the destructor on the fake 24 uninitialized elements,
    # it will be accounted for in destructor_counter).
    assert_equal(len(destructor_counter), 8)
    for i in range(8):
        assert_equal(destructor_counter[i], i)


def test_list_unsafe_set_and_get():
    var arr = Array[Int]()

    for i in range(5):
        arr.unsafe_set(i, i)

    assert_equal(5, len(arr))
    assert_equal(0, arr.unsafe_get(0)[])
    assert_equal(1, arr.unsafe_get(1)[])
    assert_equal(2, arr.unsafe_get(2)[])
    assert_equal(3, arr.unsafe_get(3)[])
    assert_equal(4, arr.unsafe_get(4)[])

    arr[2] = -2
    assert_equal(-2, arr.unsafe_get(2)[])

    arr.clear()
    arr.unsafe_set(0, 2)
    assert_equal(2, arr.unsafe_get(0)[])


def main():
    # from InlinedFixedVector
    test_inlined_fixed_array()
    test_inlined_fixed_array_with_default()
    test_indexing_vec()
    # from inline_list
    test_append_triggers_a_move()
    test_destructor()
    # from List
    test_mojo_issue_698()
    test_list()
    test_list_clear()
    test_list_to_bool_conversion()
    test_list_pop()
    test_list_variadic_constructor()
    test_list_resize()
    test_list_reverse()
    test_list_insert()
    test_list_index()
    test_list_extend()
    test_list_extend_non_trivial()
    test_list_explicit_copy()
    test_list_copy_constructor()
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
    test_list_contains()
    test_indexing_list()
    # from array
    test_list_unsafe_set_and_get()
