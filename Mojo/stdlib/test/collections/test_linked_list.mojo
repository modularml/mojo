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

from std.collections import LinkedList
from std.hashlib import hash

from test_utils import (
    CopyCountedStruct,
    CopyCounter,
    DelCounter,
    ExplicitDestroy,
    MoveCounter,
    MoveOnly,
    check_write_to,
)
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_construction() raises:
    var l1 = LinkedList[Int]()
    assert_equal(len(l1), 0)

    var l2 = LinkedList[Int](1, 2, 3)
    assert_equal(len(l2), 3)
    assert_equal(l2.get_nth(0), 1)
    assert_equal(l2.get_nth(1), 2)
    assert_equal(l2.get_nth(2), 3)


def test_linkedlist_literal() raises:
    var l: LinkedList[Int] = [1, 2, 3]
    assert_equal(3, len(l))
    assert_equal(1, l.get_nth(0))
    assert_equal(2, l.get_nth(1))
    assert_equal(3, l.get_nth(2))

    var l2: LinkedList[Float64] = [1, 2.5]
    assert_equal(2, len(l2))
    assert_equal(1.0, l2.get_nth(0))
    assert_equal(2.5, l2.get_nth(1))

    var l3: LinkedList[Int] = []
    assert_equal(0, len(l3))


def test_append() raises:
    var l1 = LinkedList[Int]()
    l1.append(1)
    l1.append(2)
    l1.append(3)
    assert_equal(len(l1), 3)
    assert_equal(l1.get_nth(0), 1)
    assert_equal(l1.get_nth(1), 2)
    assert_equal(l1.get_nth(2), 3)


def test_prepend() raises:
    var l1 = LinkedList[Int]()
    l1.prepend(1)
    l1.prepend(2)
    l1.prepend(3)
    assert_equal(len(l1), 3)
    assert_equal(l1.get_nth(0), 3)
    assert_equal(l1.get_nth(1), 2)
    assert_equal(l1.get_nth(2), 1)


def test_copy() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    var l2 = l1.copy()
    assert_equal(len(l2), 3)
    assert_equal(l2.get_nth(0), 1)
    assert_equal(l2.get_nth(1), 2)
    assert_equal(l2.get_nth(2), 3)


def test_reverse() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    l1.reverse()
    assert_equal(len(l1), 3)
    assert_equal(l1.get_nth(0), 3)
    assert_equal(l1.get_nth(1), 2)
    assert_equal(l1.get_nth(2), 1)


def test_reverse_prev_pointers() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    l1.reverse()

    # After reverse, forward order is [3, 2, 1].
    # Backward iteration via __reversed__ should yield [1, 2, 3].
    var riter = l1.__reversed__()
    assert_equal(riter.__next__(), 1)
    assert_equal(riter.__next__(), 2)
    assert_equal(riter.__next__(), 3)


def test_pop() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(l1.pop(), 3)
    assert_equal(len(l1), 2)
    assert_equal(l1.get_nth(0), 1)
    assert_equal(l1.get_nth(1), 2)


def test_pop_copies() raises:
    var l1 = LinkedList[CopyCounter[]](
        CopyCounter(),
        CopyCounter(),
        CopyCounter(),
        CopyCounter(),
        CopyCounter(),
    )
    assert_equal(l1.pop().copy_count, 0)
    assert_equal(len(l1), 4)
    assert_equal(l1.pop().copy_count, 0)
    assert_equal(len(l1), 3)
    assert_equal(l1.pop(1).copy_count, 0)
    assert_equal(len(l1), 2)
    assert_equal(l1.maybe_pop(1).value().copy_count, 0)
    assert_equal(len(l1), 1)
    assert_equal(l1.maybe_pop().value().copy_count, 0)
    assert_equal(len(l1), 0)


def test_getitem() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(l1.get_nth(0), 1)
    assert_equal(l1.get_nth(1), 2)
    assert_equal(l1.get_nth(2), 3)

    assert_equal(l1.get_nth(len(l1) - 1), 3)
    assert_equal(l1.get_nth(len(l1) - 2), 2)
    assert_equal(l1.get_nth(len(l1) - 3), 1)


def test_setitem() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    l1.get_nth(0) = 4
    assert_equal(l1.get_nth(0), 4)
    assert_equal(l1.get_nth(1), 2)
    assert_equal(l1.get_nth(2), 3)

    l1.get_nth(len(l1) - 1) = 5
    assert_equal(l1.get_nth(0), 4)
    assert_equal(l1.get_nth(1), 2)
    assert_equal(l1.get_nth(2), 5)


def test_pop_on_empty_list() raises:
    with assert_raises():
        var ll = LinkedList[Int]()
        _ = ll.pop()


def test_optional_pop_on_empty_linked_list() raises:
    var ll = LinkedList[Int]()
    var result = ll.maybe_pop()
    assert_false(Bool(result))


def test_list() raises:
    var list = LinkedList[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(0, list.get_nth(0))
    assert_equal(1, list.get_nth(1))
    assert_equal(2, list.get_nth(2))
    assert_equal(3, list.get_nth(3))
    assert_equal(4, list.get_nth(4))

    assert_equal(0, list.get_nth(len(list) - 5))
    assert_equal(3, list.get_nth(len(list) - 2))
    assert_equal(4, list.get_nth(len(list) - 1))

    list.get_nth(2) = -2
    assert_equal(-2, list.get_nth(2))

    list.get_nth(len(list) - 5) = 5
    assert_equal(5, list.get_nth(len(list) - 5))
    list.get_nth(len(list) - 2) = 3
    assert_equal(3, list.get_nth(len(list) - 2))
    list.get_nth(len(list) - 1) = 7
    assert_equal(7, list.get_nth(len(list) - 1))


def test_list_clear() raises:
    var list = LinkedList[Int](1, 2, 3)
    assert_equal(len(list), 3)
    list.clear()

    assert_equal(len(list), 0)


def test_list_to_bool_conversion() raises:
    assert_false(LinkedList[String]())
    assert_true(LinkedList[String]("a"))
    assert_true(LinkedList[String]("", "a"))
    assert_true(LinkedList[String](""))


def test_list_pop() raises:
    var list = LinkedList[Int]()
    # Test pop with index
    for i in range(6):
        list.append(i)

    assert_equal(6, len(list))

    # try popping from index 3 for 3 times
    for i in range(3, 6):
        assert_equal(i, list.pop(3))

    # list should have 3 elements now
    assert_equal(3, len(list))
    assert_equal(0, list.get_nth(0))
    assert_equal(1, list.get_nth(1))
    assert_equal(2, list.get_nth(2))

    # Test pop with index 0 (first element)
    for i in range(0, 2):
        var popped: Int = list.pop(0)
        assert_equal(i, popped)

    # test default index as well
    assert_equal(2, list.pop())
    list.append(2)
    assert_equal(2, list.pop())

    # list should be empty now
    assert_equal(0, len(list))


def test_list_variadic_constructor() raises:
    var l = LinkedList[Int](2, 4, 6)
    assert_equal(3, len(l))
    assert_equal(2, l.get_nth(0))
    assert_equal(4, l.get_nth(1))
    assert_equal(6, l.get_nth(2))

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l.get_nth(3))

    #
    # Test variadic construct copying behavior
    #

    var l2 = LinkedList[CopyCounter[]](
        CopyCounter(), CopyCounter(), CopyCounter()
    )

    assert_equal(len(l2), 3)
    assert_equal(l2.get_nth(0).copy_count, 0)
    assert_equal(l2.get_nth(1).copy_count, 0)
    assert_equal(l2.get_nth(2).copy_count, 0)


def test_list_reverse() raises:
    #
    # Test reversing the list []
    #

    var vec = LinkedList[Int]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the list [123]
    #

    vec = LinkedList[Int]()

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec.get_nth(0), 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec.get_nth(0), 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    var vec2 = LinkedList[String]("one", "two", "three")

    assert_equal(len(vec2), 3)
    assert_equal(vec2.get_nth(0), "one")
    assert_equal(vec2.get_nth(1), "two")
    assert_equal(vec2.get_nth(2), "three")

    vec2.reverse()

    assert_equal(len(vec2), 3)
    assert_equal(vec2.get_nth(0), "three")
    assert_equal(vec2.get_nth(1), "two")
    assert_equal(vec2.get_nth(2), "one")

    #
    # Test reversing the list [5, 10]
    #

    vec = LinkedList[Int]()
    vec.append(5)
    vec.append(10)

    assert_equal(len(vec), 2)
    assert_equal(vec.get_nth(0), 5)
    assert_equal(vec.get_nth(1), 10)

    vec.reverse()

    assert_equal(len(vec), 2)
    assert_equal(vec.get_nth(0), 10)
    assert_equal(vec.get_nth(1), 5)


def test_list_insert() raises:
    #
    # Test the list [1, 2, 3] created with insert
    #

    var v1 = LinkedList[Int]()
    v1.insert(len(v1), 1)
    v1.insert(len(v1), 3)
    v1.insert(1, 2)

    assert_equal(len(v1), 3)
    assert_equal(v1.get_nth(0), 1)
    assert_equal(v1.get_nth(1), 2)
    assert_equal(v1.get_nth(2), 3)

    #
    # Test the list [1, 2, 3, 4, 5] created with interior and boundary indices
    #

    var v2 = LinkedList[Int]()
    v2.insert(0, 2)
    v2.insert(len(v2), 3)
    v2.insert(len(v2), 5)
    v2.insert(2, 4)
    v2.insert(0, 1)

    assert_equal(len(v2), 5)
    assert_equal(v2.get_nth(0), 1)
    assert_equal(v2.get_nth(1), 2)
    assert_equal(v2.get_nth(2), 3)
    assert_equal(v2.get_nth(3), 4)
    assert_equal(v2.get_nth(4), 5)

    #
    # Test the list [1, 2, 3, 4] created by inserting at the front
    #

    var v3 = LinkedList[Int]()
    v3.insert(0, 4)
    v3.insert(0, 3)
    v3.insert(0, 2)
    v3.insert(0, 1)

    assert_equal(len(v3), 4)
    assert_equal(v3.get_nth(0), 1)
    assert_equal(v3.get_nth(1), 2)
    assert_equal(v3.get_nth(2), 3)
    assert_equal(v3.get_nth(3), 4)

    #
    # Test the list [1, 2, 3, 4, 5, 6, 7, 8] created with insert
    #

    var v4 = LinkedList[Int]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i, value in enumerate(v4):
        assert_equal(value, i + 1)


def test_list_extend_non_trivial() raises:
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other lists
    #   - extend() using optimal number of move constructor calls
    var v1 = LinkedList[MoveCounter[String]]()
    v1.append(MoveCounter[String]("Hello"))
    v1.append(MoveCounter[String]("World"))

    var v2 = LinkedList[MoveCounter[String]]()
    v2.append(MoveCounter[String]("Foo"))
    v2.append(MoveCounter[String]("Bar"))
    v2.append(MoveCounter[String]("Baz"))

    v1.extend(v2^)

    assert_equal(len(v1), 5)
    assert_equal(v1.get_nth(0).value, "Hello")
    assert_equal(v1.get_nth(1).value, "World")
    assert_equal(v1.get_nth(2).value, "Foo")
    assert_equal(v1.get_nth(3).value, "Bar")
    assert_equal(v1.get_nth(4).value, "Baz")

    assert_equal(v1.get_nth(0).move_count, 1)
    assert_equal(v1.get_nth(1).move_count, 1)
    assert_equal(v1.get_nth(2).move_count, 1)
    assert_equal(v1.get_nth(3).move_count, 1)
    assert_equal(v1.get_nth(4).move_count, 1)


def test_2d_dynamic_list() raises:
    var list = LinkedList[LinkedList[Int]]()

    for i in range(2):
        var v = LinkedList[Int]()
        for j in range(3):
            v.append(i + j)
        list.append(v^)

    assert_equal(0, list.get_nth(0).get_nth(0))
    assert_equal(1, list.get_nth(0).get_nth(1))
    assert_equal(2, list.get_nth(0).get_nth(2))
    assert_equal(1, list.get_nth(1).get_nth(0))
    assert_equal(2, list.get_nth(1).get_nth(1))
    assert_equal(3, list.get_nth(1).get_nth(2))

    assert_equal(2, len(list))

    assert_equal(3, len(list.get_nth(0)))

    list.get_nth(0).clear()

    assert_equal(0, len(list.get_nth(0)))

    list.clear()
    assert_equal(0, len(list))


def test_list_explicit_copy() raises:
    var list = LinkedList[CopyCounter[]]()
    list.append(CopyCounter())
    var list_copy = list.copy()
    assert_equal(0, list.get_nth(0).copy_count)
    assert_equal(1, list_copy.get_nth(0).copy_count)

    var l2 = LinkedList[Int]()
    for i in range(10):
        l2.append(i)

    var l2_copy = l2.copy()
    assert_equal(len(l2), len(l2_copy))
    for i, value in enumerate(l2):
        assert_equal(value, l2_copy.get_nth(i))


def test_no_extra_copies_with_sugared_set_by_field() raises:
    var list = LinkedList[LinkedList[CopyCountedStruct]]()
    var child_list = LinkedList[CopyCountedStruct]()
    child_list.append(CopyCountedStruct("Hello"))
    child_list.append(CopyCountedStruct("World"))

    # No copies here.  Constructing with LinkedList[CopyCountedStruct](CopyCountedStruct("Hello")) is a copy.
    assert_equal(0, child_list.get_nth(0).counter.copy_count)
    assert_equal(0, child_list.get_nth(1).counter.copy_count)

    list.append(child_list^)

    assert_equal(0, list.get_nth(0).get_nth(0).counter.copy_count)
    assert_equal(0, list.get_nth(0).get_nth(1).counter.copy_count)

    list.get_nth(0).get_nth(1).value = "Mojo"

    assert_equal(0, list.get_nth(0).get_nth(0).counter.copy_count)
    assert_equal(0, list.get_nth(0).get_nth(1).counter.copy_count)

    assert_equal("Mojo", list.get_nth(0).get_nth(1).value)

    assert_equal(0, list.get_nth(0).get_nth(0).counter.copy_count)
    assert_equal(0, list.get_nth(0).get_nth(1).counter.copy_count)


def test_list_boolable() raises:
    assert_true(LinkedList[Int](1))
    assert_false(LinkedList[Int]())


def test_list_count() raises:
    var list = LinkedList[Int](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, Int(list.count(1)))
    assert_equal(2, Int(list.count(2)))
    assert_equal(0, Int(list.count(4)))

    var list2 = LinkedList[Int]()
    assert_equal(0, Int(list2.count(1)))


def test_index() raises:
    var l = LinkedList[Int](1, 2, 3, 2, 5)
    assert_equal(l.index(1), 0)
    assert_equal(l.index(2), 1)
    assert_equal(l.index(5), 4)

    # Returns first occurrence
    assert_equal(l.index(2), 1)

    # Not found raises
    with assert_raises():
        _ = l.index(99)

    # Empty list raises
    var empty = LinkedList[Int]()
    with assert_raises():
        _ = empty.index(1)


def test_list_contains() raises:
    var x = LinkedList[Int](1, 2, 3)
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)

    # TODO: implement LinkedList.__eq__ for Self[Copyable & Comparable]
    # var y = LinkedList[LinkedList[Int]]()
    # y.append(LinkedList(1,2))
    # assert_equal(LinkedList(1,2) in y,True)
    # assert_equal(LinkedList(0,1) in y,False)


def test_list_eq_ne() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    var l2 = LinkedList[Int](1, 2, 3)
    assert_true(l1 == l2)
    assert_false(l1 != l2)

    var l3 = LinkedList[Int](1, 2, 3, 4)
    assert_false(l1 == l3)
    assert_true(l1 != l3)

    var l4 = LinkedList[Int]()
    var l5 = LinkedList[Int]()
    assert_true(l4 == l5)
    assert_true(l1 != l4)

    var l6 = LinkedList[String]("a", "b", "c")
    var l7 = LinkedList[String]("a", "b", "c")
    var l8 = LinkedList[String]("a", "b")
    assert_true(l6 == l7)
    assert_false(l6 != l7)
    assert_false(l6 == l8)


def test_indexing() raises:
    var l = LinkedList[Int](1, 2, 3)
    assert_equal(l.get_nth(Int(1)), 2)
    assert_equal(l.get_nth(2), 3)


# ===-------------------------------------------------------------------===#
# LinkedList dtor tests
# ===-------------------------------------------------------------------===#


def test_list_dtor() raises:
    var dtor_count = 0

    var ptr = Pointer(to=dtor_count).as_imm().as_unsafe_any_origin()
    var l = LinkedList[DelCounter[ptr.origin]]()
    assert_equal(dtor_count, 0)

    l.append(DelCounter(ptr))
    assert_equal(dtor_count, 0)

    l^.__deinit__()
    assert_equal(dtor_count, 1)


def test_iter() raises:
    var l = LinkedList[Int](1, 2, 3)
    var it = l.__iter__()
    assert_equal(it.__next__(), 1)
    assert_equal(it.__next__(), 2)
    assert_equal(it.__next__(), 3)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration

    var riter = l.__reversed__()
    assert_equal(riter.__next__(), 3)
    assert_equal(riter.__next__(), 2)
    assert_equal(riter.__next__(), 1)
    with assert_raises():
        _ = riter.__next__()  # raises StopIteration

    var i = 0
    for el in l:
        assert_equal(el, l.get_nth(i))
        i += 1

    i = 2
    for el in l.__reversed__():
        assert_equal(el, l.get_nth(i))
        i -= 1

    var ll = LinkedList[Int]()
    with assert_raises():
        var it = iter(ll)
        _ = it.__next__()  # raises StopIteration


def test_repr_wrap() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(
        repr(l1), "LinkedList[SIMD[DType.int, 1]]([Int(1), Int(2), Int(3)])"
    )


def test_write_to() raises:
    """Test Writable trait implementation."""
    check_write_to(
        LinkedList[Int](10, 20, 30), expected="[10, 20, 30]", is_repr=False
    )
    check_write_to(LinkedList[Int](), expected="[]", is_repr=False)
    check_write_to(LinkedList[Int](42), expected="[42]", is_repr=False)


def test_write_repr_to() raises:
    """Test write_repr_to implementation."""
    check_write_to(
        LinkedList[Int](1, 2, 3),
        expected="LinkedList[SIMD[DType.int, 1]]([Int(1), Int(2), Int(3)])",
        is_repr=True,
    )
    check_write_to(
        LinkedList[Int](1),
        expected="LinkedList[SIMD[DType.int, 1]]([Int(1)])",
        is_repr=True,
    )
    check_write_to(
        LinkedList[Int](),
        expected="LinkedList[SIMD[DType.int, 1]]([])",
        is_repr=True,
    )


def test_hash() raises:
    var l1 = LinkedList[Int](1, 2, 3)
    var l2 = LinkedList[Int](1, 2, 3)
    var l3 = LinkedList[Int](3, 2, 1)
    assert_equal(hash(l1), hash(l2))
    # Different order should (very likely) produce different hash
    assert_true(hash(l1) != hash(l3))


struct NonEquatable(Copyable):
    pass


def test_linked_list_conditional_conformances() raises:
    assert_true(conforms_to(LinkedList[Int], Equatable))
    assert_false(conforms_to(LinkedList[NonEquatable], Equatable))

    assert_true(conforms_to(LinkedList[Int], Writable))
    assert_false(conforms_to(LinkedList[NonEquatable], Writable))

    assert_true(conforms_to(LinkedList[Int], Copyable))
    assert_true(conforms_to(LinkedList[Int], Hashable))


def test_linked_list_iter_owned() raises:
    # Test that owned iteration works, for non-Copyable types
    var ll = LinkedList[MoveOnly[Int]](MoveOnly(1), MoveOnly(2), MoveOnly(3))
    var result = List[MoveOnly[Int]]()
    for var elem in ll^:
        result.append(elem^)

    assert_equal(len(result), 3)
    assert_equal(result[0], MoveOnly(1))
    assert_equal(result[1], MoveOnly(2))
    assert_equal(result[2], MoveOnly(3))


def test_linked_list_iter_owned_destroys_elements_if_not_consumed() raises:
    var dtor_count = 0
    var ptr = Pointer(to=dtor_count).as_imm().as_unsafe_any_origin()
    var ll = LinkedList[DelCounter[ptr.origin]]()
    ll.append(DelCounter(ptr))
    ll.append(DelCounter(ptr))
    ll.append(DelCounter(ptr))
    assert_equal(dtor_count, 0)

    # Create the owned iterator but never consume it; all elements should
    # still be destroyed when the iterator is dropped.
    var _ = ll^.__iter__()
    assert_equal(dtor_count, 3)


def test_linked_list_iter_owned_destroys_elements_if_partially_consumed() raises:
    var dtor_count = 0
    var ptr = Pointer(to=dtor_count).as_imm().as_unsafe_any_origin()
    var ll = LinkedList[DelCounter[ptr.origin]]()
    ll.append(DelCounter(ptr))
    ll.append(DelCounter(ptr))
    ll.append(DelCounter(ptr))
    assert_equal(dtor_count, 0)

    var it = ll^.__iter__()
    # Consume one element; it should be destroyed when dropped.
    var _ = it.__next__()
    assert_equal(dtor_count, 1)

    # Drop the iterator with two unconsumed elements remaining.
    _ = it^
    assert_equal(dtor_count, 3)


def test_linked_list_iter_owned_bounds() raises:
    var ll = LinkedList[Int](1, 2, 3)
    var it = ll^.__iter__()
    for i in range(3, 0, -1):
        assert_equal((i, Optional(i)), it.bounds())
        _ = it.__next__()

    assert_equal((0, Optional(0)), it.bounds())


def test_linked_list_move_only() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `LinkedList[T: Movable & Deinitable]`.
    assert_false(conforms_to(LinkedList[MoveOnly[Int]], Copyable))

    var l = LinkedList[MoveOnly[Int]]()
    l.append(MoveOnly[Int](0))
    l.append(MoveOnly[Int](1))
    l.prepend(MoveOnly[Int](-1))
    assert_equal(len(l), 3)
    assert_equal(l.get_nth(0), MoveOnly[Int](-1))
    assert_equal(l.get_nth(1), MoveOnly[Int](0))
    assert_equal(l.get_nth(2), MoveOnly[Int](1))

    # Methods that take/move don't require `Copyable`.
    var tail = l.pop()
    assert_equal(tail, MoveOnly[Int](1))
    assert_equal(len(l), 2)

    l.insert(0, MoveOnly[Int](42))
    assert_equal(l.get_nth(0), MoveOnly[Int](42))

    l.clear()
    assert_equal(len(l), 0)


# ===-------------------------------------------------------------------===#
# Conditional `Deinitable` (MSTDL-2775)
# ===-------------------------------------------------------------------===#


def test_linked_list_conditional_implicitly_deletable() raises:
    assert_true(conforms_to(LinkedList[Int], Deinitable))
    assert_false(conforms_to(LinkedList[ExplicitDestroy], Deinitable))
    assert_true(conforms_to(LinkedList[Int], IterableOwned))
    assert_true(conforms_to(LinkedList[MoveOnly[Int]], IterableOwned))
    assert_false(conforms_to(LinkedList[ExplicitDestroy], IterableOwned))


def test_linked_list_deinit_with() raises:
    var ll = LinkedList[ExplicitDestroy]()
    ll.append(ExplicitDestroy(1))
    ll.append(ExplicitDestroy(2))
    ll.append(ExplicitDestroy(3))
    ll.append(ExplicitDestroy(4))
    ll.append(ExplicitDestroy(5))
    var destroy_order = List[Int]()

    def dispose(var data: ExplicitDestroy) {mut}:
        destroy_order.append(data.value)
        data^.destroy()

    ll^.deinit_with(dispose)
    assert_equal(len(destroy_order), 5)
    for i in range(len(destroy_order)):
        assert_true(destroy_order[i] == i + 1)


def test_empty_linked_list_deinit_with() raises:
    # `deinit_with` on an empty (linear-valued) linked list must run and free the
    # backing without invoking the closure — there are no entries.
    var ll = LinkedList[ExplicitDestroy]()
    var calls = 0

    def dispose(var data: ExplicitDestroy) {mut}:
        calls += 1
        data^.destroy()

    ll^.deinit_with(dispose)
    assert_equal(calls, 0)


def test_linked_list_extend_explicit_destroy() raises:
    var a = LinkedList[ExplicitDestroy]()
    a.append(ExplicitDestroy(1))
    a.append(ExplicitDestroy(2))
    a.append(ExplicitDestroy(3))

    var b = LinkedList[ExplicitDestroy]()
    b.append(ExplicitDestroy(4))
    b.append(ExplicitDestroy(5))

    a.extend(
        b^
    )  # consumes `b`; its emptied husk must not leak (checked by LSAN)

    var order = List[Int]()

    def dispose(var data: ExplicitDestroy) {mut}:
        order.append(data.value)
        data^.destroy()

    a^.deinit_with(dispose)
    assert_equal(len(order), 5)
    for i in range(len(order)):
        assert_true(order[i] == i + 1)


def test_linked_list_extend_into_empty_explicit_destroy() raises:
    # Covers the empty-`self` branch of `extend` for a linear element type.
    var a = LinkedList[ExplicitDestroy]()
    var b = LinkedList[ExplicitDestroy]()
    b.append(ExplicitDestroy(1))
    b.append(ExplicitDestroy(2))

    a.extend(b^)

    var order = List[Int]()

    def dispose(var data: ExplicitDestroy) {mut}:
        order.append(data.value)
        data^.destroy()

    a^.deinit_with(dispose)
    assert_equal(len(order), 2)
    assert_true(order[0] == 1)
    assert_true(order[1] == 2)


def test_linked_list_insert_explicit_destroy() raises:
    # This only compiles because `insert` dropped its
    # `Deinitable` requirement; re-adding it breaks this. Covers the
    # head, tail, and middle branches.
    var l = LinkedList[ExplicitDestroy]()
    l.insert(0, ExplicitDestroy(2))  # [2]        (head into empty)
    l.insert(0, ExplicitDestroy(1))  # [1, 2]     (head)
    l.insert(len(l), ExplicitDestroy(4))  # [1, 2, 4]  (tail)
    l.insert(2, ExplicitDestroy(3))  # [1, 2, 3, 4]  (middle)

    var order = List[Int]()

    def dispose(var data: ExplicitDestroy) {mut}:
        order.append(data.value)
        data^.destroy()

    l^.deinit_with(dispose)
    assert_equal(len(order), 4)
    for i in range(len(order)):
        assert_true(order[i] == i + 1)


def test_linked_list_maybe_pop_explicit_destroy() raises:
    # `maybe_pop` returns `Optional[ExplicitDestroy]`, which is itself a linear
    # type: it can't be implicitly dropped and must be drained through
    # `Optional.deinit_with`.
    var ll = LinkedList[ExplicitDestroy]()
    ll.append(ExplicitDestroy(1))
    ll.append(ExplicitDestroy(2))

    var popped = ll.maybe_pop()
    var popped_val = popped.value().value  # tail == 2
    var len_after_pop = len(ll)  # 2 -> 1
    var survivor = ll.get_nth(0).value  # head == 1 remains
    popped^.deinit_with(ExplicitDestroy.destroy)

    # `maybe_pop` on an empty list yields an empty Optional.
    var empty = LinkedList[ExplicitDestroy]()
    var none = empty.maybe_pop()
    var was_empty = not Bool(none)
    none^.deinit_with(ExplicitDestroy.destroy)

    empty^.deinit_with(ExplicitDestroy.destroy)
    ll^.deinit_with(ExplicitDestroy.destroy)

    assert_equal(popped_val, 2)
    assert_equal(len_after_pop, 1)
    assert_equal(survivor, 1)
    assert_true(was_empty)


def test_linked_list_prepend_explicit_destroy() raises:
    var ll = LinkedList[ExplicitDestroy]()
    var empty_before = Bool(ll)

    ll.prepend(ExplicitDestroy(2))  # prepend into an empty list
    var nonempty_after = Bool(ll)
    ll.prepend(ExplicitDestroy(1))  # prepend onto a non-empty head

    var order = List[Int]()

    def dispose(var data: ExplicitDestroy) {mut}:
        order.append(data.value)
        data^.destroy()

    ll^.deinit_with(dispose)

    assert_false(empty_before)
    assert_true(nonempty_after)
    assert_equal(len(order), 2)
    assert_equal(order[0], 1)
    assert_equal(order[1], 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
