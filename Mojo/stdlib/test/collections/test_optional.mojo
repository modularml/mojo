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

from std.builtin.device_passable import DevicePassable
from std.collections import OptionalReg
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from std.sys import size_of

from std.testing import *
from std.testing import TestSuite
from test_utils import (
    DelCounter,
    ExplicitDelOnly,
    MoveOnly,
    PinnedExplicitDelOnly,
    check_write_to,
)


def test_basic() raises:
    # Assign to vars to remove compiler warnings.
    var false = False
    var true = True

    var a = Optional(1)
    var b = Optional[Int](None)

    assert_true(a)
    assert_false(b)

    assert_true(a and true)
    assert_true(true and a)
    assert_false(a and false)

    assert_false(b and true)
    assert_false(b and false)

    assert_true(a or true)
    assert_true(a or false)

    assert_true(b or true)
    assert_false(b or false)

    assert_equal(1, a.value())

    # Test invert operator
    assert_false(~a)
    assert_true(~b)

    # TODO(27776): can't inline these, they need to be mutable lvalues
    var a1 = a.or_else(2)
    var b1 = b.or_else(2)

    assert_equal(1, a1)
    assert_equal(2, b1)

    assert_equal(1, a.unsafe_take())

    # TODO: this currently only checks for mutable references.
    # We may want to come back and add an immutable test once
    # there are the language features to do so.
    var a2 = Optional(1)
    a2.value() = 2
    assert_equal(a2.value(), 2)


def test_optional_reg_basic() raises:
    # Assign to vars to remove compiler warnings
    var false = False
    var true = True

    var val: OptionalReg[Int] = None
    assert_false(val.__bool__())

    val = 15
    assert_true(val.__bool__())

    assert_equal(val.value(), 15)

    assert_true(val or False)
    assert_true(val and True)

    assert_true(false or val)
    assert_true(true and val)

    assert_equal(OptionalReg[Int]().or_else(33), 33)
    assert_equal(OptionalReg[Int](42).or_else(33), 42)


def test_optional_is() raises:
    var a = Optional(1)
    assert_false(a is None)

    a = Optional[Int](None)
    assert_true(a is None)


def test_optional_isnot() raises:
    var a = Optional(1)
    assert_true(a is not None)

    a = Optional[Int](None)
    assert_false(a is not None)


def test_optional_reg_is() raises:
    var a = OptionalReg(1)
    assert_false(a is None)

    a = OptionalReg[Int](None)
    assert_true(a is None)


def test_optional_reg_isnot() raises:
    var a = OptionalReg(1)
    assert_true(a is not None)

    a = OptionalReg[Int](None)
    assert_false(a is not None)


def test_optional_take_mutates() raises:
    var opt1 = Optional[Int](5)

    assert_true(opt1)

    var value: Int = opt1.take()

    assert_equal(value, 5)
    # The optional should now be empty
    assert_false(opt1)


def test_optional_explicit_copy() raises:
    var v1 = Optional[String]("test")

    var v2 = v1.copy()

    assert_equal(v1.value(), "test")
    assert_equal(v2.value(), "test")

    v2.value() += "ing"

    assert_equal(v1.value(), "test")
    assert_equal(v2.value(), "testing")


def test_optional_conformance() raises:
    assert_true(conforms_to(Optional[Int], Writable))


@fieldwise_init
struct _NonWritable(Movable):
    """A `Movable & Deinitable` type that is not `Writable`,
    `Copyable`, or `Hashable` — used to exercise the negative case of
    `Optional`'s conditional conformances."""

    var n: Int


def test_optional_conditional_conformances() raises:
    assert_true(conforms_to(Optional[Int], Writable))
    assert_true(conforms_to(Optional[String], Writable))
    assert_false(conforms_to(Optional[_NonWritable], Writable))

    assert_true(conforms_to(Optional[Int], Copyable))
    assert_true(conforms_to(Optional[String], Copyable))
    assert_false(conforms_to(Optional[MoveOnly[Int]], Copyable))

    assert_true(conforms_to(Optional[Int], Hashable))
    assert_true(conforms_to(Optional[String], Hashable))
    # `MoveOnly[T]` is conditionally `Hashable` when `T` is `Hashable`.
    assert_true(conforms_to(Optional[MoveOnly[Int]], Hashable))
    assert_false(conforms_to(Optional[_NonWritable], Hashable))


struct _NonTrivial(Copyable):
    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    def __deinit__(deinit self):
        pass


def test_optional_triviality() raises:
    comptime trivial = Optional[Int]
    assert_true(IsTriviallyCopyable[trivial])
    assert_true(IsTriviallyMovable[trivial])
    assert_true(IsTriviallyDeinitable[trivial])

    comptime not_trivial = Optional[_NonTrivial]
    assert_false(IsTriviallyCopyable[not_trivial])
    assert_false(IsTriviallyMovable[not_trivial])
    assert_false(IsTriviallyDeinitable[not_trivial])


def test_optional_write_to() raises:
    check_write_to(Optional[Int](None), expected="None", is_repr=False)
    check_write_to(Optional[Int](42), expected="42", is_repr=False)


def test_optional_write_repr_to() raises:
    check_write_to(
        Optional[Int](None),
        expected="Optional[SIMD[DType.int, 1]](None)",
        is_repr=True,
    )
    check_write_to(
        Optional[Int](42),
        expected="Optional[SIMD[DType.int, 1]](Int(42))",
        is_repr=True,
    )


def test_optional_hash() raises:
    # Equal optionals should produce the same hash.
    assert_equal(hash(Optional(1)), hash(Optional(1)))
    assert_equal(hash(Optional("hello")), hash(Optional("hello")))

    # None optionals should hash equally.
    assert_equal(hash(Optional[Int](None)), hash(Optional[Int](None)))

    # Different values should (likely) produce different hashes.
    assert_not_equal(hash(Optional(1)), hash(Optional(2)))

    # Present vs None should produce different hashes.
    assert_not_equal(hash(Optional(0)), hash(Optional[Int](None)))

    # A value stored in an Optional should have a different hash than that
    # value on its own.
    assert_not_equal(hash(1), hash(Optional(1)))


def test_optional_equality() raises:
    var o = Optional(10)
    var n = Optional[Int]()
    assert_true(o == 10)
    assert_true(o != 11)
    assert_true(o != n)
    assert_true(o != None)
    assert_true(n != 11)
    assert_true(n == n)
    assert_true(n == None)


def test_optional_copied() raises:
    var data = "foo"

    var opt_ref: Optional[Pointer[String, origin_of(data)]] = Optional(
        Pointer(to=data)
    )

    # Copy the optional Pointer value.
    var opt_owned: Optional[String] = opt_ref.copied()

    assert_equal(opt_owned.value(), "foo")


def test_optional_unwrap() raises:
    var a = Optional(123)
    assert_true(a)
    assert_equal(123, a[])
    a = Optional[Int](None)
    with assert_raises(contains="EmptyOptionalError"):
        _ = a[]


def test_optional_repr_wrap() raises:
    var o = Optional(10)
    assert_equal(repr(o), "Optional[SIMD[DType.int, 1]](Int(10))")
    o = None
    assert_equal(repr(o), "Optional[SIMD[DType.int, 1]](None)")


def test_optional_iter() raises:
    var o = Optional(10)
    var it = o.__iter__()
    assert_equal(it.bounds()[0], 1)
    assert_equal(it.bounds()[1].value(), 1)
    assert_equal(it.__next__(), 10)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration

    var called = False
    var o2 = Optional(20)
    for _value in o2:
        called = True
    assert_true(called)


def test_optional_iter_empty() raises:
    var o = Optional[Int](None)
    var it = o.__iter__()
    assert_equal(it.bounds()[0], 0)
    assert_equal(it.bounds()[1].value(), 0)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration

    var called = False
    var o2 = Optional[Int](None)
    for _value in o2:
        called = True
    assert_false(called)


def test_optional_of_move_only_type() raises:
    var opt1 = Optional(MoveOnly[Int](5))
    # Test moving the optional
    var opt2 = opt1^
    # Test moving out of the optional
    var val: MoveOnly[Int] = opt2.take()

    assert_equal(val.data, 5)
    assert_false(opt2)

    # Test move-only default value
    val = opt2^.or_else(MoveOnly(10))
    assert_equal(val.data, 10)


def test_nicheable_size() raises:
    comptime PointerType = Pointer[Int, AnyOrigin[mut=True]]

    assert_equal(size_of[Optional[PointerType]](), size_of[PointerType]())
    assert_true(size_of[Optional[Int]]() > size_of[Int]())


def test_optional_reg_nicheable_size() raises:
    comptime PointerType = Pointer[Int, AnyOrigin[mut=True]]

    assert_equal(size_of[OptionalReg[PointerType]](), size_of[PointerType]())
    assert_true(size_of[OptionalReg[Int]]() > size_of[Int]())


def test_optional_to_optional_reg_implicit_conversion() raises:
    var optional = Optional[Int](42)
    var optional_reg: OptionalReg[Int] = optional
    assert_equal(optional_reg.value(), 42)


struct NotEquatable(Movable):
    pass


def test_optional_conforms_to_equatable() raises:
    assert_true(conforms_to(Optional[Int], Equatable))
    assert_false(conforms_to(Optional[NotEquatable], Equatable))


def test_optional_conditional_register_passable() raises:
    assert_true(conforms_to(Optional[Int], RegisterPassable))
    assert_true(conforms_to(Optional[Bool], RegisterPassable))
    assert_false(conforms_to(Optional[List[Int]], RegisterPassable))
    assert_false(conforms_to(Optional[String], RegisterPassable))


def test_optional_conditional_device_passable() raises:
    assert_true(conforms_to(Optional[Int], DevicePassable))
    assert_true(conforms_to(Optional[Float32], DevicePassable))
    assert_false(conforms_to(Optional[Bool], DevicePassable))
    assert_false(conforms_to(Optional[String], DevicePassable))
    assert_false(conforms_to(Optional[MoveOnly[Int]], DevicePassable))


def test_optional_iter_owned() raises:
    var count = 0
    var opt = Optional(42)
    for elem in opt^:
        assert_equal(elem, 42)
        count += 1
    assert_equal(count, 1)


def test_optional_iter_owned_none() raises:
    var count = 0
    var opt = Optional[Int](None)
    for elem in opt^:
        count += 1
        _ = elem
    assert_equal(count, 0)


def test_optional_iter_owned_bounds() raises:
    var opt = Optional(42)
    var it = opt^.__iter__()
    assert_equal(it.bounds()[0], 1)
    _ = it.__next__()
    assert_equal(it.bounds()[0], 0)


def double(var x: Int) -> Float64:
    return Float64(x) * 2.0


def test_map_with_value() raises:
    var opt = Optional(21)
    var result = opt^.map(double)
    assert_true(result)
    assert_equal(result.value(), 42.0)


def test_map_with_none() raises:
    var opt = Optional[Int](None)
    var result = opt^.map(double)
    assert_false(result)


def test_map_with_closure_that_takes_by_read() raises:
    var opt = Optional[String]("hello")

    def closure_by_read(s: String) -> String:
        return s + "42"

    var result1 = opt.map(closure_by_read)
    assert_equal(result1[], "hello42")


def try_parse_int(var s: String) -> Optional[Int]:
    try:
        return Int(s)
    except:
        return None


def test_and_then_with_value() raises:
    var opt = Optional("42")
    var result = opt^.and_then(try_parse_int)
    assert_true(result)
    assert_equal(result.value(), 42)


def test_and_then_with_value_returns_none() raises:
    var opt = Optional("not_a_number")
    var result = opt^.and_then(try_parse_int)
    assert_false(result)


def test_and_then_with_none() raises:
    var opt = Optional[String](None)
    var result = opt^.and_then(try_parse_int)
    assert_false(result)


def _unwrap_linear(var x: ExplicitDelOnly) -> Int:
    var data = x.data
    x^.destroy()
    return data


def _unwrap_linear_opt(var x: ExplicitDelOnly) -> Optional[Int]:
    var data = x.data
    x^.destroy()
    return data


def test_map_linear_type() raises:
    # `map` moves the linear value out and consumes the `Optional`; the
    # emptied `Optional` is destroyed without instantiating `Variant.__deinit__`.
    var opt = Optional(ExplicitDelOnly(5))
    var result = opt^.map(_unwrap_linear)
    assert_equal(result.value(), 5)

    # The empty branch is destroyed without calling `_unwrap_linear`.
    var empty = Optional[ExplicitDelOnly](None)
    assert_false(empty^.map(_unwrap_linear))


def test_and_then_linear_type() raises:
    var opt = Optional(ExplicitDelOnly(7))
    var result = opt^.and_then(_unwrap_linear_opt)
    assert_equal(result.value(), 7)

    var empty = Optional[ExplicitDelOnly](None)
    assert_false(empty^.and_then(_unwrap_linear_opt))


def test_optional_linear_type_deinit_with() raises:
    # `Optional` holding a linear value is destroyed via `deinit_with`.
    var v1 = Optional(ExplicitDelOnly(5))
    v1^.deinit_with(ExplicitDelOnly.destroy)

    # `Optional[T]` holding `None` is destroyed without invoking
    # `deinit_func`.
    var v2 = Optional[ExplicitDelOnly](None)
    v2^.deinit_with(ExplicitDelOnly.destroy)


def test_optional_linear_type_deinit_assert_empty() raises:
    # An empty `Optional` holding a linear type is destroyed without a
    # caller-provided deinitializer via `deinit_assert_empty`.
    var v = Optional[ExplicitDelOnly](None)
    v^.deinit_assert_empty()


def test_optional_deinit_with_runs_exactly_once() raises:
    # Verify `deinit_func` runs exactly once on the contained value.
    var counter = 0

    def increment_counter(var _value: Int) {mut counter}:
        counter += 1

    var opt = Optional[Int](42)
    opt^.deinit_with(increment_counter)
    assert_equal(counter, 1)


def test_optional_deinit_with_none_does_not_call_destroy() raises:
    # Verify `deinit_func` is not called when the `Optional` is empty.
    var counter = 0

    def increment_counter(var _value: Int) {mut counter}:
        counter += 1

    var opt = Optional[Int](None)
    opt^.deinit_with(increment_counter)
    assert_equal(counter, 0)


# `Deinitable` but explicitly not `Movable`: rejected at the parameter
# bound under the old `Optional[T: Movable]`, admitted under the `AnyType` floor.
struct _NotMovable(Deinitable, Movable where False):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


def test_optional_non_movable_element_type() raises:
    # `Optional` of a non-`Movable` element type is now a valid type.
    var empty = Optional[_NotMovable]()
    assert_false(empty)
    assert_equal(empty.bounds()[0], 0)


def test_optional_non_movable_construct_in_place() raises:
    # `init_with=` populates an `Optional` with a non-`Movable` value in place.
    def make() -> _NotMovable:
        return _NotMovable(7)

    var opt = Optional[_NotMovable](init_with=make)
    assert_true(opt)
    assert_equal(opt.bounds()[0], 1)
    assert_equal(opt.value().x, 7)


def test_optional_pinned_linear_type_deinit_with() raises:
    var counter = 0

    def destroy(var value: PinnedExplicitDelOnly) {mut counter}:
        counter += 1
        value^.destroy()

    def make() -> PinnedExplicitDelOnly:
        return PinnedExplicitDelOnly(7)

    var opt = Optional[PinnedExplicitDelOnly](init_with=make)
    var is_some = opt.__bool__()
    var data = opt.value().data
    # Destroy before potentially raising after assert
    opt^.deinit_with(destroy)
    assert_true(is_some)
    assert_equal(data, 7)
    assert_equal(counter, 1)

    # The empty case is destroyed without invoking `deinit_func`.
    var empty = Optional[PinnedExplicitDelOnly](None)
    empty^.deinit_with(destroy)
    assert_equal(counter, 1)


def test_optional_pinned_linear_type_deinit_assert_empty() raises:
    var empty = Optional[PinnedExplicitDelOnly](None)
    empty^.deinit_assert_empty()


def test_optional_niched_storage_deinit_with() raises:
    comptime PointerType = Pointer[Int, AnyOrigin[mut=True]]

    var x = 42
    var some = Optional[PointerType](
        Pointer(to=x).unsafe_origin_cast[AnyOrigin[mut=True]]()
    )
    some^.deinit_with(PointerType.__deinit__)

    var empty = Optional[PointerType](None)
    empty^.deinit_with(PointerType.__deinit__)

    var empty2 = Optional[PointerType](None)
    empty2^.deinit_assert_empty()


def test_optional_construct_in_place_calls_closure_once() raises:
    var calls = 0

    def make() {mut calls} -> Int:
        calls += 1
        return 7

    var opt = Optional[Int](init_with=make)
    assert_equal(opt.value(), 7)
    assert_equal(calls, 1)


def test_optional_construct_in_place_destroys_value_once() raises:
    # `init_with=` marks the storage's active type by hand before emplacing, so check
    # that teardown runs the element's destructor exactly once.
    var del_count = 0
    var counter_ptr = Pointer(to=del_count)
    comptime Counter = DelCounter[origin_of(del_count)]

    def make() {counter_ptr} -> Counter:
        return Counter(counter_ptr)

    var opt = Optional[Counter](init_with=make)
    assert_true(opt)
    _ = opt^
    assert_equal(del_count, 1)


def _assert_simd_bool_round_trips[width: SIMDLength]() raises:
    var all_true = Optional(SIMD[DType.bool, width](fill=True))
    assert_true(all_true.value().reduce_and())

    var all_false = Optional(SIMD[DType.bool, width](fill=False))
    assert_false(all_false.value().reduce_or())

    # A mixed pattern catches a payload that is wide enough but misplaced.
    var mixed = SIMD[DType.bool, width](fill=False)
    mixed[0] = True
    mixed[width - 1] = True
    assert_equal(Optional(mixed).value(), mixed)


def test_optional_simd_bool() raises:
    # A `SIMD[bool, N]` payload is N bytes to the compiler's own layout but
    # lowers to `vector<N x i1>`, which LLVM allocates in ceil(N/8). Sizing the
    # union from the lowered type undersized it and crashed the compiler
    # (MOCO-4689).
    _assert_simd_bool_round_trips[2]()
    _assert_simd_bool_round_trips[4]()
    _assert_simd_bool_round_trips[8]()
    _assert_simd_bool_round_trips[16]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
