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

from collections import InlineList, Set

from test_utils import MoveCounter
from testing import assert_equal, assert_false, assert_raises, assert_true


def test_list():
    var list = InlineList[Int]()

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


def test_append_triggers_a_move():
    var inline_list = InlineList[MoveCounter[Int], capacity=32]()

    var nb_elements_to_add = 8
    for index in range(nb_elements_to_add):
        inline_list.append(MoveCounter(index))

    # Using .append() should trigger a move and not a copy+delete.
    for i in range(nb_elements_to_add):
        assert_equal(inline_list[i].move_count, 1)


@value
struct ValueToCountDestructor(CollectionElementNew):
    var value: Int
    var destructor_counter: UnsafePointer[List[Int]]

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.value = other.value
        self.destructor_counter = other.destructor_counter

    fn __del__(owned self):
        self.destructor_counter[].append(self.value)


def test_destructor():
    """Ensure we delete the right number of elements."""
    var destructor_counter = List[Int]()
    alias capacity = 32
    var inline_list = InlineList[ValueToCountDestructor, capacity=capacity]()

    for index in range(capacity):
        inline_list.append(
            ValueToCountDestructor(
                index, UnsafePointer.address_of(destructor_counter)
            )
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


def test_list_iter():
    var vs = InlineList[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: InlineList[Int]) -> Int:
        var sum = 0
        for v in vs:
            sum += v[]
        return sum

    assert_equal(6, sum(vs))


def test_list_iter_mutable():
    var vs = InlineList[Int, 3](1, 2, 3)

    for v in vs:
        v[] += 1

    var sum = 0
    for v in vs:
        sum += v[]

    assert_equal(9, sum)


def test_list_contains():
    var x = InlineList[Int](1, 2, 3)
    assert_false(0 in x)
    assert_true(x.__contains__(1))
    assert_false(x.__contains__(4))


def test_list_variadic_constructor():
    var l = InlineList[Int](2, 4, 6)
    assert_equal(3, len(l))
    assert_equal(2, l[0])
    assert_equal(4, l[1])
    assert_equal(6, l[2])

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l[3])


def test_list_count():
    var list = InlineList[Int](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, list.count(1))
    assert_equal(2, list.count(2))
    assert_equal(0, list.count(4))

    var list2 = InlineList[Int]()
    assert_equal(0, list2.count(1))


def test_list_boolable():
    assert_true(InlineList[Int](1))
    assert_false(InlineList[Int]())


def test_indexing():
    var list = InlineList[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(list[True], 1)
    assert_equal(list[int(4)], 4)
    assert_equal(list[0], 0)


def main():
    test_list()
    test_append_triggers_a_move()
    test_destructor()
    test_list_iter()
    test_list_iter_mutable()
    test_list_contains()
    test_list_variadic_constructor()
    test_list_count()
    test_list_boolable()
    test_indexing()
