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

from std.collections import Set
from std.hashlib import hash

from test_utils import (
    CopyableExplicitDestroyKey,
    ExplicitDestroyKey,
    MoveOnly,
    check_write_to,
)
from std.testing import (
    assert_false,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_set_construction() raises:
    # Constructors.
    _ = Set[Int]()
    _ = Set[String]()
    _ = Set[Int](1, 2, 3)
    _ = Set(Set[Int](1, 2, 3))

    # Literals.
    var s1 = {1, 2, 3}
    assert_equal(s1, {1, 2, 3})

    var s2 = {"1", "2"}
    assert_equal(s2, {"1", "2"})


def test_set_move() raises:
    var s1 = {1, 2, 3}
    var s2 = s1^
    assert_equal(s2, {1, 2, 3})


def test_set_copy() raises:
    var s1 = {1, 2, 3}
    var s2 = s1.copy()
    assert_equal(s1, s2)


def test_len() raises:
    var s1 = Set[Int]()
    assert_equal(0, len(s1))

    var s2 = {1, 2, 3}
    assert_equal(3, len(s2))


def test_in() raises:
    var s1 = Set[Int]()
    assert_false(0 in s1)
    assert_false(1 in s1)

    var s2 = {1, 2, 3}
    assert_false(0 in s2)
    assert_true(1 in s2)
    assert_true(2 in s2)
    assert_true(3 in s2)
    assert_false(4 in s2)


def test_equal() raises:
    var s1 = Set[Int]()
    var s2 = {1, 2, 3}

    # TODO(#33178): Not using `assert_equal` and friends
    # since Set is not Stringable

    assert_true(s1 == s1)
    assert_true(s2 == s2)
    assert_true(s1 == {})
    assert_true(s2 == {3, 2, 1})
    assert_true(s1 != s2)
    assert_true(s2 != s1)
    assert_true(s2 != {1, 2, 2})
    assert_true(s2 != {1, 2, 4})


def test_bool() raises:
    assert_false(Set[Int]())
    assert_false(Set[Int](List[Int]()))
    assert_true(Set[Int](1))
    assert_true(Set[Int](1, 2, 3))


def test_intersection() raises:
    assert_equal(Set[Int]() & {}, {})
    assert_equal(Set[Int]() & {1, 2, 3}, {})
    assert_equal({1, 2, 3} & {1, 2, 3}, {1, 2, 3})
    assert_equal({1, 2, 3} & {}, {})
    assert_equal({1, 2, 3} & {3, 4}, {3})

    assert_equal(Set[Int]().intersection({}), {})
    assert_equal(Set[Int]().intersection({1, 2, 3}), {})
    assert_equal({1, 2, 3}.intersection({1, 2, 3}), {1, 2, 3})
    assert_equal({1, 2, 3}.intersection({}), {})
    assert_equal({1, 2, 3}.intersection({3, 4}), {3})

    var x = Set[Int]()
    x &= {1, 2, 3}
    assert_equal(x, {})

    x = Set[Int]()
    x &= {}
    assert_equal(x, {})

    x = {1, 2, 3}
    x &= {}
    assert_equal(x, {})

    x = {1, 2, 3}
    x &= {1, 2, 3}
    assert_equal(x, {1, 2, 3})

    x = {1, 2}
    x &= {2, 3}
    assert_equal(x, {2})


def test_union() raises:
    assert_equal(Set[Int]() | {}, {})
    assert_equal(Set[Int]() | {1, 2, 3}, {1, 2, 3})
    assert_equal({1, 2, 3} | {1, 2, 3}, {1, 2, 3})
    assert_equal({1, 2, 3} | {}, {1, 2, 3})
    assert_equal({1, 2, 3} | {3, 4}, {1, 2, 3, 4})

    assert_equal(Set[Int]().union({}), {})
    assert_equal(Set[Int]().union({1, 2, 3}), {1, 2, 3})
    assert_equal({1, 2, 3}.union({1, 2, 3}), {1, 2, 3})
    assert_equal({1, 2, 3}.union({}), {1, 2, 3})
    assert_equal({1, 2, 3}.union({3, 4}), {1, 2, 3, 4})

    var x = Set[Int]()
    x |= {1, 2, 3}
    assert_equal(x, {1, 2, 3})

    x = Set[Int]()
    x |= {}
    assert_equal(x, {})

    x = {1, 2, 3}
    x |= {}
    assert_equal(x, {1, 2, 3})

    x = {1, 2, 3}
    x |= {1, 2, 3}
    assert_equal(x, {1, 2, 3})

    x = {1, 2}
    x |= {2, 3}
    assert_equal(x, {1, 2, 3})


def test_subtract() raises:
    var s1 = Set[Int]()
    var s2 = {1, 2, 3}

    assert_equal(s1 - s1, s1)
    assert_equal(s1 - s2, s1)
    assert_equal(s2 - s2, s1)
    assert_equal(s2 - s1, s2)
    assert_equal(s2 - {3, 4}, {1, 2})


def test_difference_update() raises:
    var x = Set[Int]()
    x.difference_update({})
    assert_equal(x, {})

    x = {1, 2, 3}
    x.difference_update({1, 2, 3})
    assert_equal(x, {})

    x = {1, 2, 3}
    x.difference_update({})
    assert_equal(x, {1, 2, 3})

    x = {1, 2, 3}
    x.difference_update({3, 4})
    assert_equal(x, {1, 2})

    x = Set[Int]()
    x -= {}
    assert_equal(x, {})

    x = {1, 2, 3}
    x -= {1, 2, 3}
    assert_equal(x, {})

    x = {1, 2, 3}
    x -= {}
    assert_equal(x, {1, 2, 3})

    x = {1, 2, 3}
    x -= {3, 4}
    assert_equal(x, {1, 2})


def test_iter() raises:
    var sum = 0
    for e in Set[Int]():
        sum += e

    assert_equal(sum, 0)

    sum = 0
    for e in {1, 2, 3}:
        sum += e
    assert_equal(sum, 6)

    var my_set = {4, 5, 6}
    var it = enumerate(my_set)
    var elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 4)
    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 5)
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 6)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_add() raises:
    var s = Set[Int]()
    s.add(1)
    assert_equal(s, {1})

    s.add(2)
    assert_equal(s, {1, 2})

    s.add(3)
    assert_equal(s, {1, 2, 3})

    # 1 is already in the set
    s.add(1)
    assert_equal(s, {1, 2, 3})


def test_remove() raises:
    var s = {1, 2, 3}
    s.remove(1)
    assert_equal(s, {2, 3})

    s.remove(2)
    assert_equal(s, {3})

    s.remove(3)
    assert_equal(s, {})

    with assert_raises():
        # 1 not in the set, should raise
        s.remove(1)


def test_pop_insertion_order() raises:
    var s = {1, 2, 3}
    assert_equal(s.pop(), 3)
    assert_equal(s, {1, 2})

    s.add(4)

    assert_equal(s.pop(), 4)
    assert_equal(s, {1, 2})

    assert_equal(s.pop(), 2)
    assert_equal(s, {1})

    assert_equal(s.pop(), 1)
    assert_equal(s, {})

    with assert_raises():
        _ = s.pop()  # pop from empty set raises


def test_issubset() raises:
    assert_true(Set[Int]().issubset({1, 2, 3}))
    assert_true(Set[Int]() <= {1, 2, 3})

    assert_true({1, 2, 3}.issubset({1, 2, 3}))
    assert_true({1, 2, 3} <= {1, 2, 3})

    assert_true({2, 3}.issubset({1, 2, 3, 4}))
    assert_true({2, 3} <= {1, 2, 3, 4})

    assert_false({1, 2, 3, 4}.issubset({2, 3}))
    assert_false({1, 2, 3, 4} <= {2, 3})

    assert_false({1, 2, 3, 4, 5}.issubset({2, 3}))
    assert_false({1, 2, 3, 4, 5} <= {2, 3})

    assert_true(Set[Int]().issubset({}))
    assert_true(Set[Int]() <= {})

    assert_false({1, 2, 3}.issubset({4, 5, 6}))
    assert_false({1, 2, 3} <= {4, 5, 6})


def test_disjoint() raises:
    assert_true(Set[Int]().isdisjoint({}))
    assert_false({1, 2, 3}.isdisjoint({1, 2, 3}))
    assert_true({1, 2, 3}.isdisjoint({4, 5, 6}))
    assert_false({1, 2, 3}.isdisjoint({3, 4, 5}))
    assert_true(Set[Int]().isdisjoint({1, 2, 3}))
    assert_true({1, 2, 3}.isdisjoint({}))
    assert_false({1, 2, 3}.isdisjoint({3}))
    assert_true({1, 2, 3}.isdisjoint({4}))


def test_issuperset() raises:
    assert_true({1, 2, 3}.issuperset({}))
    assert_true({1, 2, 3} >= {})

    assert_true({1, 2, 3}.issuperset({1, 2, 3}))
    assert_true({1, 2, 3} >= {1, 2, 3})

    assert_true({1, 2, 3, 4}.issuperset({2, 3}))
    assert_true({1, 2, 3, 4} >= {2, 3})

    assert_false({2, 3}.issuperset({1, 2, 3, 4}))
    assert_false({2, 3} >= {1, 2, 3, 4})

    assert_false({1, 2, 3}.issuperset({4, 5, 6}))
    assert_false({1, 2, 3} >= {4, 5, 6})

    assert_false(Set[Int]().issuperset({1, 2, 3}))
    assert_false(Set[Int]() >= {1, 2, 3})

    assert_false({1, 2, 3}.issuperset({1, 2, 3, 4}))
    assert_false({1, 2, 3} >= {1, 2, 3, 4})

    assert_true(Set[Int]().issuperset({}))
    assert_true(Set[Int]() >= {})


def test_greaterthan() raises:
    assert_true({1, 2, 3, 4} > {2, 3})
    assert_false({2, 3} > {1, 2, 3, 4})
    assert_false({1, 2, 3} > {1, 2, 3})
    assert_false(Set[Int]() > {})
    assert_true({1, 2, 3} > {})


def test_lessthan() raises:
    assert_true({2, 3} < {1, 2, 3, 4})
    assert_false({1, 2, 3, 4} < {2, 3})
    assert_false({1, 2, 3} < {1, 2, 3})
    assert_false(Set[Int]() < {})
    assert_true(Set[Int]() < {1, 2, 3})


def test_symmetric_difference() raises:
    assert_true({1, 4} == {1, 2, 3}.symmetric_difference({2, 3, 4}))
    assert_true({1, 4} == {1, 2, 3} ^ {2, 3, 4})

    assert_true({1, 2, 3, 4, 5, 6} == {1, 2, 3}.symmetric_difference({4, 5, 6}))
    assert_true({1, 2, 3, 4, 5, 6} == {1, 2, 3} ^ {4, 5, 6})

    assert_true({1, 2, 3} == {1, 2, 3}.symmetric_difference({}))
    assert_true({1, 2, 3} == {1, 2, 3} ^ {})

    assert_true({1, 2, 3} == Set[Int]() ^ {1, 2, 3})
    assert_true({1, 2, 3} == Set[Int]() ^ {1, 2, 3})

    assert_true(Set[Int]() == Set[Int]().symmetric_difference({}))
    assert_true(Set[Int]() == Set[Int]() ^ {})

    assert_true(Set[Int]() == {1, 2, 3}.symmetric_difference({1, 2, 3}))
    assert_true(Set[Int]() == {1, 2, 3} ^ {1, 2, 3})


def test_symmetric_difference_update() raises:
    # Test case 1
    var set1 = {1, 2, 3}
    var set2 = {2, 3, 4}
    set1.symmetric_difference_update(set2)
    assert_true({1, 4} == set1)

    set1 = {1, 2, 3}
    set2 = {2, 3, 4}
    set1 ^= set2
    assert_true({1, 4} == set1)

    # Test case 2
    var set3 = {1, 2, 3}
    var set4 = {4, 5, 6}
    set3.symmetric_difference_update(set4)
    assert_true({1, 2, 3, 4, 5, 6} == set3)

    set3 = {1, 2, 3}
    set4 = {4, 5, 6}
    set3 ^= set4
    assert_true({1, 2, 3, 4, 5, 6} == set3)

    # Test case 3
    var set5 = {1, 2, 3}
    var set6 = Set[Int]()
    set5.symmetric_difference_update(set6)
    assert_true({1, 2, 3} == set5)

    set5 = {1, 2, 3}
    set6 = Set[Int]()
    set5 ^= set6
    assert_true({1, 2, 3} == set5)

    # Test case 4
    var set7 = Set[Int]()
    var set8 = {1, 2, 3}
    set7.symmetric_difference_update(set8)
    assert_true({1, 2, 3} == set7)

    set7 = Set[Int]()
    set8 = {1, 2, 3}
    set7 ^= set8
    assert_true({1, 2, 3} == set7)

    # Test case 5
    var set9 = Set[Int]()
    var set10 = Set[Int]()
    set9.symmetric_difference_update(set10)
    assert_true(set9 == {})

    set9 = Set[Int]()
    set10 = Set[Int]()
    set9 ^= set10
    assert_true(set9 == {})

    # Test case 6
    var set11 = {1, 2, 3}
    var set12 = {1, 2, 3}
    set11.symmetric_difference_update(set12)
    assert_true(set11 == {})

    set11 = {1, 2, 3}
    set12 = {1, 2, 3}
    set11 ^= set12
    assert_true(set11 == {})


def test_discard() raises:
    var set1 = {1, 2, 3}
    set1.discard(2)
    assert_true(set1 == {1, 3})

    var set2 = {1, 2, 3}
    set2.discard(4)
    assert_true(set2 == {1, 2, 3})

    var set3 = Set[Int]()
    set3.discard(1)
    assert_true(set3 == {})

    var set4 = {1, 2, 3, 4, 5}
    set4.discard(2)
    set4.discard(4)
    assert_true(set4 == {1, 3, 5})

    var set5 = {1, 2, 3}
    set5.discard(1)
    set5.discard(2)
    set5.discard(3)
    assert_true(set5 == {})


def test_clear() raises:
    # Shouldn't fail when clearing a 0 length set
    var set0 = Set[Int]()
    set0.clear()
    assert_equal(0, len(set0))

    var set1 = {1, 2, 3}
    set1.clear()
    assert_true(set1 == {})

    var set2 = Set[Int]()
    set2.clear()
    assert_true(set2 == {})

    var set3 = {1, 2, 3}
    set3.clear()
    set3.add(4)
    set3.add(5)
    assert_true(set3 == {4, 5})

    var set4 = {1, 2, 3}
    set4.clear()
    set4.clear()
    set4.clear()
    assert_true(set4 == {})

    var set5 = {1, 2, 3}
    set5.clear()
    assert_true(len(set5) == 0)


def test_set_comprehension() raises:
    var s1 = {x * x for x in range(10) if x & 1}
    assert_equal(s1, {1, 9, 25, 49, 81})

    var s2 = {x * y for x in range(3) for y in s1}
    assert_equal(s2, {0, 0, 0, 0, 0, 1, 9, 25, 49, 81, 2, 18, 50, 98, 162})


def test_set_write_to() raises:
    check_write_to(Set[Int](10, 20, 30), expected="{10, 20, 30}", is_repr=False)
    check_write_to(
        Set[String]("hello", "world"), expected="{hello, world}", is_repr=False
    )
    check_write_to(Set[Int](), expected="{}", is_repr=False)
    check_write_to(Set[Int](1), expected="{1}", is_repr=False)
    check_write_to({1, 2, 3}, expected="{1, 2, 3}", is_repr=False)


def test_set_write_repr_to() raises:
    # Test set with elements
    var s = {1, 2, 3}
    var output = String()
    s.write_repr_to(output)
    assert_true(output.startswith("Set[SIMD[DType.int, 1], Hasher="))
    assert_true(output.endswith("]({Int(1), Int(2), Int(3)})"))

    # Test empty set
    var empty = Set[Int]()
    var empty_output = String()
    empty.write_repr_to(empty_output)
    assert_true(empty_output.startswith("Set[SIMD[DType.int, 1], Hasher="))
    assert_true(empty_output.endswith("]({})"), empty_output)


def test_set_hash() raises:
    var s1 = {1, 2, 3}
    var s2 = {1, 2, 3}
    var s3 = {3, 2, 1}
    assert_equal(hash(s1), hash(s2))
    # Set hash is order-independent, so different insertion order gives same hash
    assert_equal(hash(s1), hash(s3))
    # Different sets should (very likely) produce different hash
    assert_true(hash(s1) != hash({4, 5, 6}))


def test_set_conditional_conformances() raises:
    assert_true(conforms_to(Set[Int], Copyable))
    assert_true(conforms_to(Set[Int], Equatable))
    assert_true(conforms_to(Set[Int], Comparable))
    assert_true(conforms_to(Set[Int], Hashable))
    assert_true(conforms_to(Set[Int], Writable))
    assert_true(conforms_to(Set[MoveOnly[Int]], IterableOwned))

    # Move-only element type drops the copy-requiring conformances.
    assert_false(conforms_to(Set[MoveOnly[Int]], Copyable))
    assert_false(conforms_to(Set[MoveOnly[Int]], Equatable))
    assert_false(conforms_to(Set[MoveOnly[Int]], Comparable))
    assert_false(conforms_to(Set[MoveOnly[Int]], Hashable))
    assert_false(conforms_to(Set[MoveOnly[Int]], Writable))

    # Deinitable conformance is conditional on the element type.
    assert_true(conforms_to(Set[Int], Deinitable))
    assert_true(conforms_to(Set[String], Deinitable))
    assert_true(conforms_to(Set[MoveOnly[Int]], Deinitable))
    # A linear (explicitly-destroyed) element makes the set itself linear.
    assert_false(conforms_to(Set[ExplicitDestroyKey], Deinitable))


def test_set_iter_owned() raises:
    # Test that owned iteration works, for non-Copyable types
    var s = Set[MoveOnly[Int]]()
    s.add(MoveOnly(1))
    s.add(MoveOnly(2))
    s.add(MoveOnly(3))

    var elems = List[Int]()
    for var elem in s^:
        elems.append(elem.data)

    assert_equal(len(elems), 3)
    assert_true(1 in elems)
    assert_true(2 in elems)
    assert_true(3 in elems)


def test_set_iter_copyable_linear_element() raises:
    # Borrowing iteration only reads elements, so it requires `Copyable` but
    # not `Deinitable` -- a linear element type still iterates.
    var s = Set[CopyableExplicitDestroyKey]()
    var disposed = List[Int]()

    def dispose(var key: CopyableExplicitDestroyKey) {mut}:
        disposed.append(key.value)
        key^.destroy()

    s.insert(CopyableExplicitDestroyKey(1)).deinit_with(dispose)
    s.insert(CopyableExplicitDestroyKey(2)).deinit_with(dispose)

    var seen = List[Int]()
    for element in s:
        seen.append(element.value)

    s^.deinit_with(dispose)

    assert_equal(len(seen), 2)
    assert_true(1 in seen)
    assert_true(2 in seen)


def test_set_iter_owned_bounds() raises:
    var s = Set[Int](1, 2, 3)
    var it = s^.__iter__()
    assert_equal(it.bounds()[0], 3)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 2)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 1)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 0)


def test_set_move_only_element() raises:
    # `MoveOnly[Int]` is not `Copyable`; this exercises the conditional
    # conformance path of `Set[T: KeyElement & Deinitable, H]`
    # where the element type is move-only. Copy-requiring ops (`union`,
    # `intersection`, iteration, ...) are unavailable for this `T`, but the
    # add / remove / contains / pop / discard / clear core remains usable.
    assert_false(conforms_to(Set[MoveOnly[Int]], Copyable))

    var s = Set[MoveOnly[Int]]()
    s.add(MoveOnly[Int](1))
    s.add(MoveOnly[Int](2))
    s.add(MoveOnly[Int](3))
    assert_equal(len(s), 3)
    assert_true(MoveOnly[Int](1) in s)
    assert_true(MoveOnly[Int](2) in s)
    assert_true(MoveOnly[Int](3) in s)
    assert_false(MoveOnly[Int](99) in s)

    # Adding a duplicate is a no-op on membership.
    s.add(MoveOnly[Int](1))
    assert_equal(len(s), 3)

    # `remove` removes by key without copying the key.
    s.remove(MoveOnly[Int](2))
    assert_equal(len(s), 2)
    assert_false(MoveOnly[Int](2) in s)

    # `discard` on a missing element is a no-op.
    s.discard(MoveOnly[Int](99))
    assert_equal(len(s), 2)

    # `pop` moves an element out.
    var popped = s.pop()
    assert_equal(len(s), 1)
    assert_false(popped in s)

    s.clear()
    assert_equal(len(s), 0)
    assert_false(s.__bool__())


def test_set_insert_linear() raises:
    # `insert` on a linear (non-`Deinitable`) element type moves any
    # displaced equal element out and returns it instead of destroying it in
    # place. The returned `Optional[T]` is itself linear and is consumed via
    # `deinit_with`. The final `deinit_with` covers `Set`'s per-element
    # teardown path end-to-end.
    var s = Set[ExplicitDestroyKey]()
    var disposed = List[Int]()

    def dispose(var key: ExplicitDestroyKey) {mut}:
        disposed.append(key.value)
        key^.destroy()

    # New elements: nothing displaced, so each returned `Optional` is empty.
    s.insert(ExplicitDestroyKey(1)).deinit_with(dispose)
    s.insert(ExplicitDestroyKey(2)).deinit_with(dispose)
    var len_after_new = len(s)
    var disposed_after_new = len(disposed)

    # Inserting an equal element displaces the previously-present one, which
    # comes back and is disposed by the caller (not the just-inserted element).
    s.insert(ExplicitDestroyKey(1)).deinit_with(dispose)
    var len_after_dup = len(s)
    var disposed_after_dup = len(disposed)

    s^.deinit_with(dispose)

    assert_equal(len_after_new, 2)
    assert_equal(disposed_after_new, 0)
    assert_equal(len_after_dup, 2)
    assert_equal(disposed_after_dup, 1)
    # Final teardown disposed the two survivors (1 and 2).
    assert_equal(len(disposed), 3)
    assert_true(1 in disposed)
    assert_true(2 in disposed)


def test_set_clear_with_linear() raises:
    # `clear_with` on a populated linear `Set`: every element reaches the
    # closure once, the set empties, and its capacity is reused.
    var s = Set[ExplicitDestroyKey]()
    var disposed = List[Int]()

    def dispose(var key: ExplicitDestroyKey) {mut}:
        disposed.append(key.value)
        key^.destroy()

    s.insert(ExplicitDestroyKey(1)).deinit_with(dispose)
    s.insert(ExplicitDestroyKey(2)).deinit_with(dispose)
    s.insert(ExplicitDestroyKey(3)).deinit_with(dispose)
    var len_before_clear = len(s)

    s.clear_with(dispose)
    var cleared = disposed.copy()
    var len_after_clear = len(s)

    # Capacity is retained, so the emptied set is reusable.
    s.insert(ExplicitDestroyKey(4)).deinit_with(dispose)
    var len_after_reuse = len(s)

    s^.deinit_with(dispose)

    assert_equal(len_before_clear, 3)
    assert_equal(len_after_clear, 0)
    assert_equal(len_after_reuse, 1)

    assert_equal(len(cleared), 3)
    assert_true(1 in cleared)
    assert_true(2 in cleared)
    assert_true(3 in cleared)

    # Final teardown disposed the reused survivor (4).
    assert_equal(len(disposed), 4)
    assert_true(4 in disposed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
