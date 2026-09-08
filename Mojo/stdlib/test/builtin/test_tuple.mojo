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

from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    assert_raises,
    TestSuite,
)
from test_utils import (
    CopyCounter,
    ExplicitDestroy,
    MoveOnly,
    ObservableMoveOnly,
)


def test_tuple_contains() raises:
    var a = (123, True, StaticString("Mojo is awesome"))

    assert_true(StaticString("Mojo is awesome") in a)
    assert_true(a.__contains__(StaticString("Mojo is awesome")))

    assert_false(StaticString("Hello world") in a)
    assert_false(a.__contains__(StaticString("Hello world")))

    assert_true(123 in a)
    assert_true(a.__contains__(123))

    assert_true(True in a)
    assert_true(a.__contains__(True))

    assert_false(False in a)
    assert_false(a.__contains__(False))

    assert_false(a.__contains__(1))
    assert_false(a.__contains__(0))
    assert_false(1 in a)
    assert_false(0 in a)

    var b = (False, True)
    assert_true(True in b)
    assert_true(b.__contains__(True))
    assert_true(False in b)
    assert_true(b.__contains__(False))
    assert_false(b.__contains__(1))
    assert_false(b.__contains__(0))

    var c = (1, 0)
    assert_false(c.__contains__(True))
    assert_false(c.__contains__(False))
    assert_false(True in c)
    assert_false(False in c)

    var d = (123, True, "Mojo is awesome")

    assert_true("Mojo is awesome" in d)
    assert_false(StaticString("Mojo is awesome") in d)
    assert_true(d.__contains__("Mojo is awesome"))

    assert_false("Hello world" in d)
    assert_false(d.__contains__("Hello world"))

    comptime a_alias = (123, True, StaticString("Mojo is awesome"))

    assert_true(StaticString("Mojo is awesome") in a_alias)
    assert_true(a_alias.__contains__(StaticString("Mojo is awesome")))

    assert_false(StaticString("Hello world") in a_alias)
    assert_false(a_alias.__contains__(StaticString("Hello world")))

    assert_true(123 in a_alias)
    assert_true(a_alias.__contains__(123))

    assert_true(True in a_alias)
    assert_true(a_alias.__contains__(True))

    assert_false(False in a_alias)
    assert_false(a_alias.__contains__(False))

    assert_false(a_alias.__contains__(1))
    assert_false(a_alias.__contains__(0))
    assert_false(1 in a_alias)
    assert_false(0 in a_alias)

    comptime b_alias = (False, True)
    assert_true(True in b_alias)
    assert_true(b_alias.__contains__(True))
    assert_true(False in b_alias)
    assert_true(b_alias.__contains__(False))
    assert_false(b_alias.__contains__(1))
    assert_false(b_alias.__contains__(0))

    comptime c_alias = (1, 0)
    assert_false(c_alias.__contains__(True))
    assert_false(c_alias.__contains__(False))
    assert_false(True in c_alias)
    assert_false(False in c_alias)

    comptime d_alias = (123, True, "Mojo is awesome")
    # Ensure `contains` itself works in comp-time domain
    comptime ok = 123 in d_alias
    assert_true(ok)

    assert_true("Mojo is awesome" in d_alias)
    assert_true(d_alias.__contains__("Mojo is awesome"))

    assert_false("Hello world" in d_alias)
    assert_false(d_alias.__contains__("Hello world"))


def test_tuple_unpack() raises:
    (var list) = [a + b for a, b in [(1, 2), (3, 4)]]
    assert_equal(list, [3, 7])

    var list2 = [a + b for a, b in [(1, 2), (3, 4)]]
    assert_equal(list2, [3, 7])


def test_tuple_default() raises:
    var t: Tuple[Int, String, Float32] = {}
    assert_equal(t[0], 0)
    assert_equal(t[1], "")
    assert_equal(t[2], 0.0)


def _default_construct[T: Defaultable]() -> T:
    return T()


def test_tuple_defaultable_generic_constraint() raises:
    var direct = Tuple[Int, String]()
    assert_equal(direct[0], 0)
    assert_equal(direct[1], "")

    var pair = _default_construct[Tuple[Int, String]]()
    assert_equal(pair[0], 0)
    assert_equal(pair[1], "")

    var nested = _default_construct[Tuple[Int, Tuple[String, Float32]]]()
    assert_equal(nested[0], 0)
    assert_equal(nested[1][0], "")
    assert_equal(nested[1][1], 0.0)


def test_tuple_comparison() raises:
    assert_equal((1, 2, 3), (1, 2, 3))
    assert_false((1, 2, 3) != (1, 2, 3))
    assert_not_equal((1, 2, 3), (1, 2, 4))
    assert_false((1, 2, 3) < (1, 2, 3))
    assert_false((1, 2, 3) > (1, 2, 3))
    assert_true((1, 2, 3) <= (1, 2, 3))
    assert_true((1, 2, 3) >= (1, 2, 3))
    assert_true((1, 2, 3) < (1, 2, 4))
    assert_true((1, 2, 3) > (1, 2, 2))
    assert_true((1, 2, 3) <= (1, 2, 4))
    assert_true((1, 2, 3) >= (1, 2, 2))
    assert_false((1, 2, 3) < (1, 2, 2))
    assert_false((1, 2, 3) > (1, 2, 4))
    assert_true((1, 2, 3) <= (1, 2, 4))
    assert_true((1, 2, 3) >= (1, 2, 2))
    assert_true(Tuple() <= Tuple())


def test_tuple_comparison_different_types() raises:
    assert_false((1, "foo") == (1, "bar"))
    assert_true((1, "foo") != (1, "bar"))
    assert_false((1, "foo") < (1, "bar"))
    assert_true((1, "foo") > (1, "bar"))


def test_tuple_reverse_odd() raises:
    var t = ("hi", 1, 4.5)
    var reversed_t = t^.reverse()
    assert_equal(reversed_t, (4.5, 1, "hi"))


def test_tuple_reverse_empty() raises:
    var t = Tuple[]()
    assert_equal(t^.reverse(), ())


def test_tuple_reverse_even() raises:
    var t = (Bool(True), Int(42))
    var t_reversed = t^.reverse()
    assert_equal(t_reversed, (Int(42), Bool(True)))


def test_tuple_reverse_copy_count() raises:
    var t = (CopyCounter(),)
    var t2 = t^.reverse()
    assert_equal(t2[0].copy_count, 0)


def test_tuple_concat() raises:
    var t = ("hi", "hey", 1)
    var t2 = (4.5, "hello")
    var concatted = t^.concat(t2^)
    assert_equal(concatted, ("hi", "hey", 1, 4.5, "hello"))


def test_tuple_empty_concat() raises:
    var t = ()
    var t2 = ()
    var concatted = t^.concat(t2^)
    assert_equal(concatted, ())


def test_tuple_identity_concat() raises:
    var t = (Bool(True),)
    var t2 = ()
    var concatted = t^.concat(t2^)
    assert_equal(concatted, (Bool(True),))


def test_tuple_concat_copy_count() raises:
    var t = (CopyCounter(),)
    var t2 = (String(""),)
    var t3 = t^.concat(t2^)
    assert_equal(t3[0].copy_count, 0)


# This test doesn't need to run, it just needs to compile
def test_tuple_size_parse_time() raises:
    def func_with_where_clause(t: Tuple) where type_of(t).__len__() < 4:
        pass

    func_with_where_clause((1, 3, 2))


def test_tuple_works_with_non_copyable_types() raises:
    var tuple = (MoveOnly[Int](42), 55)
    var moved = tuple^
    assert_equal(moved[0].data, 42)
    assert_equal(moved[1], 55)


def test_tuple_write_to() raises:
    var s = String()
    (1, 2, 3).write_to(s)
    assert_equal(s, "(1, 2, 3)")

    s = String()
    (1,).write_to(s)
    assert_equal(s, "(1,)")

    s = String()
    ().write_to(s)
    assert_equal(s, "()")

    # write_to uses write_to on elements, so strings are unquoted.
    s = String()
    (1, "hello").write_to(s)
    assert_equal(s, "(1, hello)")

    s = String()
    (True, 42, "hi").write_to(s)
    assert_equal(s, "(True, 42, hi)")


def test_tuple_write_repr_to() raises:
    var s = String()
    (1, 2, 3).write_repr_to(s)
    assert_equal(
        s,
        (
            "Tuple[SIMD[DType.int, 1], SIMD[DType.int, 1], SIMD[DType.int,"
            " 1]](Int(1), Int(2), Int(3))"
        ),
    )

    s = String()
    (1,).write_repr_to(s)
    assert_equal(s, "Tuple[SIMD[DType.int, 1]](Int(1),)")

    s = String()
    ().write_repr_to(s)
    assert_equal(s, "Tuple[]()")

    # write_repr_to uses write_repr_to on elements, so strings are quoted.
    s = String()
    (1, "hello").write_repr_to(s)
    assert_equal(s, "Tuple[SIMD[DType.int, 1], String](Int(1), 'hello')")

    s = String()
    (True, 42, "hi").write_repr_to(s)
    assert_equal(
        s, "Tuple[Bool, SIMD[DType.int, 1], String](True, Int(42), 'hi')"
    )


def test_tuple_assert_equal() raises:
    # Direct tuple-to-tuple comparisons via assert_equal.
    assert_equal((), ())
    assert_equal((1,), (1,))
    assert_equal((1, 2, 3), (1, 2, 3))
    assert_equal((1, "hello"), (1, "hello"))
    assert_equal((True, 42, "hi"), (True, 42, "hi"))


def test_tuple_assert_not_equal() raises:
    assert_not_equal((1, 2), (1, 3))
    assert_not_equal((1, "foo"), (1, "bar"))


def test_tuple_conditional_conformances() raises:
    # Copyable conformance is conditional on all element types being Copyable.
    assert_true(conforms_to(Tuple[], Comparable))
    assert_true(conforms_to(Tuple[Int], Comparable))
    assert_true(conforms_to(Tuple[], Copyable))
    assert_true(conforms_to(Tuple[Int], Copyable))
    assert_true(conforms_to(Tuple[Int, String], Copyable))
    assert_true(conforms_to(Tuple[Int, Tuple[Int, Float32]], Copyable))

    # Defaultable conformance is conditional on all element types being
    # Defaultable.
    assert_true(conforms_to(Tuple[], Defaultable))
    assert_true(conforms_to(Tuple[Int], Defaultable))
    assert_true(conforms_to(Tuple[Int, String], Defaultable))
    assert_true(conforms_to(Tuple[Int, Tuple[Int, Float32]], Defaultable))

    # ImplicitlyCopyable conformance is conditional on all element types being
    # ImplicitlyCopyable (and Copyable).
    assert_true(conforms_to(Tuple[], ImplicitlyCopyable))
    assert_true(conforms_to(Tuple[Int], ImplicitlyCopyable))
    assert_true(conforms_to(Tuple[Int, String], ImplicitlyCopyable))

    # Writable conformance is conditional on all element types being Writable.
    assert_true(conforms_to(Tuple[Int], Writable))
    assert_true(conforms_to(Tuple[Int, String], Writable))
    assert_true(conforms_to(Tuple[], Writable))

    # Equatable conformance is conditional on all element types being Equatable.
    assert_true(conforms_to(Tuple[], Equatable))
    assert_true(conforms_to(Tuple[Int], Equatable))
    assert_true(conforms_to(Tuple[Int, String], Equatable))

    # Hashable conformance is conditional on all element types being Hashable.
    assert_true(conforms_to(Tuple[], Hashable))
    assert_true(conforms_to(Tuple[Int], Hashable))
    assert_true(conforms_to(Tuple[Int, String], Hashable))

    # Deinitable conformance is conditional on all element types being
    # Deinitable.
    assert_true(conforms_to(Tuple[], Deinitable))
    assert_true(conforms_to(Tuple[Int], Deinitable))
    assert_true(conforms_to(Tuple[Int, String], Deinitable))
    assert_true(conforms_to(Tuple[Int, Tuple[Int, Float32]], Deinitable))
    assert_false(conforms_to(Tuple[ExplicitDestroy], Deinitable))
    assert_false(conforms_to(Tuple[Int, ExplicitDestroy], Deinitable))
    # A tuple nesting a linear tuple is itself linear.
    assert_false(conforms_to(Tuple[Tuple[ExplicitDestroy]], Deinitable))

    # conforms_to correctly returns False for non-conforming element types.
    assert_false(conforms_to(Tuple[MoveOnly[Int]], Copyable))
    assert_false(conforms_to(Tuple[MoveOnly[Int]], Defaultable))
    assert_false(conforms_to(Tuple[MoveOnly[Int]], ImplicitlyCopyable))


def test_tuple_hash() raises:
    # Equal tuples should produce the same hash.
    assert_equal(hash((1, 2, 3)), hash((1, 2, 3)))
    assert_equal(hash(("a", "b")), hash(("a", "b")))

    # Empty tuple hashing.
    assert_equal(hash(Tuple[]()), hash(Tuple[]()))

    # Different tuples should (likely) produce different hashes.
    assert_not_equal(hash((1, 2, 3)), hash((1, 2, 4)))
    assert_not_equal(hash((1, 2)), hash((2, 1)))


def test_tuple_assert_equal_failure_message() raises:
    with assert_raises(contains="left: (1, 2)"):
        assert_equal((1, 2), (1, 3))


def test_tuple_conditional_register_passable() raises:
    # All RP types
    assert_true(conforms_to(Tuple[Int, Bool], RegisterPassable))
    assert_true(conforms_to(Tuple[Int], RegisterPassable))

    # All non-RP types
    assert_false(conforms_to(Tuple[String, List[Int]], RegisterPassable))

    # Mixture of RP and non-RP
    assert_false(conforms_to(Tuple[Int, String], RegisterPassable))
    assert_false(conforms_to(Tuple[Bool, List[Int], Int], RegisterPassable))


# ===-------------------------------------------------------------------===#
# consume_elements
# ===-------------------------------------------------------------------===#


def test_tuple_consume_elements_move_only() raises:
    var t = (MoveOnly[Int](10), MoveOnly[Int](20), MoveOnly[Int](30))
    var collected = [0, 0, 0]

    @__parameter
    def handler[idx: Int](var elt: t.Ts[idx]):
        var e = rebind_var[MoveOnly[Int]](elt^)
        collected[idx] = e.data

    t^.consume_elements[handler]()
    assert_equal(collected, [10, 20, 30])


def test_tuple_consume_elements_destroys_once() raises:
    var actions = List[String]()
    var actions_ptr = Pointer(to=actions).as_imm()
    comptime Observed = ObservableMoveOnly[actions_ptr.origin]

    var t = (
        Observed(1, actions_ptr),
        Observed(2, actions_ptr),
        Observed(3, actions_ptr),
    )
    assert_equal(actions_ptr[unsafe_offset=0].count("__deinit__"), 0)

    @__parameter
    def handler[idx: Int](var elt: t.Ts[idx]):
        # Discarding the owned `elt` runs its destructor exactly once.
        _ = rebind_var[Observed](elt^)

    t^.consume_elements[handler]()
    # Each element is destroyed once and `deinit self` disables the tuple's own
    # destructor, so there is no double-free.
    assert_equal(actions_ptr[unsafe_offset=0].count("__deinit__"), 3)


def test_tuple_consume_elements_heterogeneous() raises:
    var t = (String("hello"), 42, List([1, 2, 3]))
    var got_str = String()
    var got_int = 0
    var got_sum = 0

    @__parameter
    def handler[idx: Int](var elt: t.Ts[idx]):
        comptime if idx == 0:
            got_str = rebind_var[String](elt^)
        elif idx == 1:
            got_int = rebind_var[Int](elt^)
        else:
            var lst = rebind_var[List[Int]](elt^)
            for x in lst:
                got_sum += x

    t^.consume_elements[handler]()
    assert_equal(got_str, "hello")
    assert_equal(got_int, 42)
    assert_equal(got_sum, 6)


def test_tuple_consume_elements_single() raises:
    var t = (MoveOnly[Int](7),)
    var collected = [0]

    @__parameter
    def handler[idx: Int](var elt: t.Ts[idx]):
        var e = rebind_var[MoveOnly[Int]](elt^)
        collected[idx] = e.data

    t^.consume_elements[handler]()
    assert_equal(collected, [7])


# Drives `consume_elements` in a generic context, where the element bound stays
# `Movable & Deinitable`, so an empty tuple compiles (a concrete
# `Tuple[]()` degrades its element type to a non-deletable `AnyType`).
def _count_consumed[*Ts: Movable & Deinitable](var t: Tuple[*Ts]) -> Int:
    var count = 0

    @__parameter
    def handler[idx: Int](var elt: t.Ts[idx]):
        _ = elt^
        count += 1

    t^.consume_elements[handler]()
    return count


def test_tuple_consume_elements_empty() raises:
    assert_equal(_count_consumed(Tuple[]()), 0)
    assert_equal(_count_consumed((1, 2, 3)), 3)


def test_tuple_deinit_with() raises:
    # A `Tuple` with a linear (non-`Deinitable`) element must be torn
    # down explicitly with `deinit_with()`.
    var t = (ExplicitDestroy(0), ExplicitDestroy(1), ExplicitDestroy(2))
    var destroyed = List[Int]()

    @__parameter
    def dispose[idx: Int](var elt: t.Ts[idx]):
        var e = rebind_var[ExplicitDestroy](elt^)
        destroyed.append(e.value)
        e^.destroy()

    t^.deinit_with[dispose]()
    assert_equal(destroyed, [0, 1, 2])


def test_tuple_deinit_with_heterogeneous() raises:
    # `deinit_with` also works when only some elements are linear; the closure
    # is handed each element by value and destroys it however is appropriate.
    var t = (String("hello"), ExplicitDestroy(42))
    var got_str = String()
    var got_val = 0

    @__parameter
    def dispose[idx: Int](var elt: t.Ts[idx]):
        comptime if idx == 0:
            got_str = rebind_var[String](elt^)
        else:
            var e = rebind_var[ExplicitDestroy](elt^)
            got_val = e.value
            e^.destroy()

    t^.deinit_with[dispose]()
    assert_equal(got_str, "hello")
    assert_equal(got_val, 42)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
