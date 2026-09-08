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

from std.memory import ArcPointer
from std.memory.arc_pointer import WeakPointer
from test_utils import ObservableDel, check_write_to
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def test_basic() raises:
    var p = ArcPointer(4)
    var p2 = p
    p2[] = 3
    assert_equal(3, p[])


def test_is() raises:
    var p = ArcPointer(3)
    var p2 = p
    var p3 = ArcPointer(3)
    assert_true(p is p2)
    assert_false(p is not p2)
    assert_false(p is p3)
    assert_true(p is not p3)


def test_deleter_not_called_until_no_references() raises:
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var p2 = p
    _ = p^
    assert_false(deleted)

    var vec = List[type_of(p)]()
    vec.append(p2)
    _ = p2^
    assert_false(deleted)
    _ = vec^
    assert_true(deleted)


def test_deleter_not_called_until_no_references_explicit_copy() raises:
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var p2 = p.copy()
    _ = p^
    assert_false(deleted)

    var vec = List[type_of(p)]()
    vec.append(p2.copy())
    _ = p2^
    assert_false(deleted)
    _ = vec^
    assert_true(deleted)


def test_count() raises:
    var a = ArcPointer(10)
    var b = a.copy()
    var c = a
    assert_equal(UInt64(3), a.count())
    _ = b^
    assert_equal(UInt64(2), a.count())
    _ = c
    assert_equal(UInt64(1), a.count())


def test_ptr() raises:
    var p = ArcPointer(42)
    assert_equal(42, p.ptr()[])

    # `ptr()` observes mutations made through `[]`.
    p[] = 99
    assert_equal(99, p.ptr()[])

    # Copies share the same allocation, so `ptr()` returns the same address.
    var p2 = p
    assert_equal(Int(p.ptr()), Int(p2.ptr()))


def test_unsafe_take_allocation_and_construct_from_raw_ptr() raises:
    var deleted = False
    var leaked = ArcPointer(ObservableDel(Pointer(to=deleted)))

    var raw = leaked^.unsafe_take_allocation().unsafe_leak()
    assert_false(deleted)

    var p = ArcPointer(unsafe_from_raw_pointer=raw)
    _ = p^
    assert_true(deleted)


def test_unsafe_take_allocation_does_not_decrement_refcount() raises:
    var leaked = ArcPointer(42)
    var copy = leaked.copy()

    assert_equal(UInt64(2), copy.count())
    var raw = leaked^.unsafe_take_allocation().unsafe_leak()

    # since we leaked the data, the refcount was not decremented
    assert_equal(UInt64(2), copy.count())

    _ = copy^

    var p = ArcPointer(unsafe_from_raw_pointer=raw)
    assert_equal(UInt64(1), p.count())


def test_write_to() raises:
    check_write_to(ArcPointer(42), expected="42", is_repr=False)
    check_write_to(ArcPointer("hello"), expected="hello", is_repr=False)


def test_write_repr_to() raises:
    check_write_to(
        ArcPointer(42),
        expected="ArcPointer[SIMD[DType.int, 1]](Int(42))",
        is_repr=True,
    )
    check_write_to(
        ArcPointer("hello"),
        expected="ArcPointer[String]('hello')",
        is_repr=True,
    )


def test_hash() raises:
    var p = ArcPointer(42)
    var q = p  # same allocation

    # Two pointers to the same object hash identically.
    assert_equal(hash(p), hash(q))

    # Distinct allocations with the same value hash identically (value-based).
    var r = ArcPointer(42)
    assert_equal(hash(p), hash(r))

    # Different values produce different hashes.
    var s = ArcPointer(99)
    assert_true(hash(p) != hash(s))


def test_eq() raises:
    var p = ArcPointer(42)
    var q = p  # same allocation

    # Same allocation: equal.
    assert_true(p == q)
    assert_false(p != q)

    # Different allocations, same value: equal (value-based).
    var r = ArcPointer(42)
    assert_true(p == r)
    assert_false(p != r)

    # Different values: not equal.
    var s = ArcPointer(99)
    assert_false(p == s)
    assert_true(p != s)


# ---------------------------------------------------------------------------
# WeakPointer tests
# ---------------------------------------------------------------------------


def test_weak_basic_upgrade() raises:
    var p = ArcPointer(7)
    var w = WeakPointer(downgrade=p)
    assert_equal(UInt64(1), p.count())
    assert_equal(UInt64(1), w.strong_count())

    var up = w.try_upgrade()
    assert_equal(UInt64(2), p.count())
    assert_true(up)
    assert_equal(7, up.value()[])
    _ = p^
    _ = up^


def test_downgrade_does_not_change_strong_count() raises:
    var p = ArcPointer(1)
    assert_equal(UInt64(1), p.count())

    var w = WeakPointer(downgrade=p)
    assert_equal(UInt64(1), p.count())
    assert_equal(UInt64(1), p.weak_count())

    _ = w^
    assert_equal(UInt64(0), p.weak_count())


def test_weak_count_via_arc() raises:
    var p = ArcPointer(0)
    assert_equal(UInt64(0), p.weak_count())

    var w1 = WeakPointer(downgrade=p)
    assert_equal(UInt64(1), p.weak_count())

    var w2 = w1
    assert_equal(UInt64(2), p.weak_count())

    var w3 = WeakPointer(downgrade=p)
    assert_equal(UInt64(3), p.weak_count())

    _ = w1^
    assert_equal(UInt64(2), p.weak_count())
    _ = w2^
    assert_equal(UInt64(1), p.weak_count())
    _ = w3^
    assert_equal(UInt64(0), p.weak_count())


def test_upgrade_increments_strong() raises:
    var p = ArcPointer(99)
    var w = WeakPointer(downgrade=p)

    var up = w.try_upgrade()
    assert_true(up)
    assert_equal(UInt64(2), p.count())

    _ = up^
    assert_equal(UInt64(1), p.count())


def test_upgrade_fails_after_last_strong_dropped() raises:
    var p = ArcPointer(123)
    var w = WeakPointer(downgrade=p)
    _ = p^

    var up = w.try_upgrade()
    assert_false(up)
    assert_equal(UInt64(0), w.strong_count())


def test_upgrade_returns_none_after_strong_drop() raises:
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var w = WeakPointer(downgrade=p)
    _ = p^
    var observed = deleted  # snapshot before any further use of `w`
    assert_true(observed)
    var up = w.try_upgrade()
    assert_false(up)
    _ = up^
    _ = w^


def test_payload_destroyed_when_strong_zero_even_with_weak() raises:
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var w = WeakPointer(downgrade=p)
    _ = p^
    _ = w^
    assert_true(deleted)


def test_allocation_kept_alive_by_weak() raises:
    # Strong drops while weak is live: weak.strong_count and weak.try_upgrade
    # must still be safe to call.
    var p = ArcPointer(5)
    var w = WeakPointer(downgrade=p)
    _ = p^

    assert_equal(UInt64(0), w.strong_count())
    var up = w.try_upgrade()
    assert_false(up)
    _ = up^
    _ = w^


def test_weak_clone_increments_weak_count() raises:
    var p = ArcPointer(11)
    var w1 = WeakPointer(downgrade=p)
    assert_equal(UInt64(1), p.weak_count())

    var w2 = w1.copy()
    assert_equal(UInt64(2), p.weak_count())

    var w3 = w2  # implicit copy
    assert_equal(UInt64(3), p.weak_count())

    _ = w1^
    _ = w2^
    _ = w3^
    assert_equal(UInt64(0), p.weak_count())


def test_weak_upgraded_arc_keeps_payload_alive() raises:
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var w = WeakPointer(downgrade=p)

    var up = w.try_upgrade()
    assert_true(up)

    # Drop original strong; an upgraded one keeps payload alive.
    _ = p^
    assert_false(deleted)

    # Drop the upgraded strong; now payload destroyed.
    _ = up^
    assert_true(deleted)

    _ = w^


def test_multiple_upgrades_then_drops() raises:
    var p = ArcPointer(42)
    var w = WeakPointer(downgrade=p)

    var u1 = w.try_upgrade().value()
    var u2 = w.try_upgrade().value()
    var u3 = w.try_upgrade().value()
    assert_equal(UInt64(4), p.count())

    _ = u1^
    _ = u2^
    _ = u3^
    assert_equal(UInt64(1), p.count())
    _ = w^


def test_weak_count() raises:
    var p = ArcPointer(0)
    var w = WeakPointer(downgrade=p)
    assert_equal(UInt64(1), w.strong_count())
    assert_equal(UInt64(1), w.weak_count())
    _ = p^
    assert_equal(UInt64(0), w.strong_count())
    # weak still live, even once strong drops to zero
    assert_equal(UInt64(1), w.weak_count())
    _ = w^


def test_only_weak_no_strong_then_free() raises:
    # Multiple WeakPointers outliving the last strong. Payload is destroyed
    # promptly when the last strong drops; weaks remain valid handles whose
    # try_upgrade() returns None.
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var w1 = WeakPointer(downgrade=p)
    var w2 = WeakPointer(downgrade=p)
    var w3 = WeakPointer(downgrade=p)
    assert_false(deleted)
    _ = p^
    assert_true(deleted)
    _ = w1^
    _ = w2^
    _ = w3^


def test_lifetime_probe_arc_drop_with_weak_in_scope() raises:
    var deleted = False
    var p = ArcPointer(ObservableDel(Pointer(to=deleted)))
    var w = WeakPointer(downgrade=p)
    _ = p^
    var observed = deleted
    var up = w.try_upgrade()
    _ = up^
    _ = w^

    # If `_ = p^` is honored as a real drop point: observed is True.
    # If lifetime extension defers p's drop to last-use of w: observed is False.
    assert_true(observed)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
