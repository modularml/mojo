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
# RUN: %mojo -debug-level full %s

from collections import List

from test_utils import CopyCounter, MoveCounter
from testing import *


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
    var list = List[Int]()

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

    # pop_back shall return the last element
    # and adjust the size
    assert_equal(7, list.pop_back())
    assert_equal(4, len(list))

    # Verify that capacity shrinks as the list goes smaller
    while list.size > 1:
        _ = list.pop_back()

    assert_equal(1, len(list))
    assert_equal(
        1, list.size
    )  # pedantically ensure len and size refer to the same thing
    assert_equal(4, list.capacity)

    # Verify that capacity doesn't become 0 when the list gets empty.
    _ = list.pop_back()
    assert_equal(0, len(list))

    # FIXME: revisit that pop_back is actually doing shrink_to_fit behavior
    # under the hood which will be surprising to users
    assert_equal(2, list.capacity)

    list.clear()
    assert_equal(0, len(list))
    assert_equal(2, list.capacity)


def test_list_variadic_constructor():
    var l = List[Int](2, 4, 6)
    assert_equal(3, len(l))
    assert_equal(2, l[0])
    assert_equal(4, l[1])
    assert_equal(6, l[2])

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l[3])


def test_list_reverse():
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

    vec = List[Int]()

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    vec2 = List[String]("one", "two", "three")

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

    vec = List[Int]()
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

    vec = List[Int]()
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

    vec = List[Int]()
    vec.append(1)
    vec.append(2)
    vec.append(3)

    vec._reverse(start=len(vec))

    assert_equal(len(vec), 3)
    assert_equal(vec[0], 1)
    assert_equal(vec[1], 2)
    assert_equal(vec[2], 3)


def test_list_reverse_move_count():
    # Create this vec with enough capacity to avoid moves due to resizing.
    var vec = List[MoveCounter[Int]](capacity=5)
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


def test_list_extend():
    #
    # Test extending the list [1, 2, 3] with itself
    #

    vec = List[Int]()
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
    var v1 = List[MoveCounter[String]](capacity=5)
    v1.append(MoveCounter[String]("Hello"))
    v1.append(MoveCounter[String]("World"))

    var v2 = List[MoveCounter[String]](capacity=3)
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


def test_2d_dynamic_list():
    var list = List[List[Int]]()

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
    assert_equal(2, list.capacity)

    assert_equal(3, len(list[0]))

    list[0].clear()
    assert_equal(0, len(list[0]))
    assert_equal(4, list[0].capacity)

    list.clear()
    assert_equal(0, len(list))
    assert_equal(2, list.capacity)


def test_list_explicit_copy():
    var list = List[CopyCounter]()
    list.append(CopyCounter()^)
    var list_copy = List(list)
    assert_equal(0, list.__get_ref(0)[].copy_count)
    assert_equal(1, list_copy.__get_ref(0)[].copy_count)

    var l2 = List[Int]()
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


def main():
    test_mojo_issue_698()
    test_list()
    test_list_variadic_constructor()
    test_list_reverse()
    test_list_reverse_move_count()
    test_list_extend()
    test_list_extend_non_trivial()
    test_list_explicit_copy()
    test_list_copy_constructor()
    test_2d_dynamic_list()
    test_list_iter()
    test_list_iter_mutable()
    test_list_span()
