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

from std.ffi import _Global
from std.memory import MaybeUninit
from std.os import abort
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from std.sys import size_of

from test_utils import (
    MoveCopyCounter,
    ObservableDel,
    ConfigureTrivial,
    MoveOnly,
    ExplicitDelOnly,
    NonMovable,
    Observable,
    PinnedExplicitDelOnly,
    check_write_to,
)
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from std.benchmark import keep

from std.utils import Variant
from std.utils._nicheable import UnsafeNicheable

comptime TEST_VARIANT_POISON = _Global[
    "TEST_VARIANT_POISON", _initialize_poison
]


def _initialize_poison() -> Bool:
    return False


def _poison_ptr() -> Pointer[Bool, MutUntrackedOrigin]:
    try:
        return TEST_VARIANT_POISON.get_or_create_ptr()
    except:
        abort("Failed to get or create TEST_VARIANT_POISON")


def assert_no_poison() raises:
    assert_false(_poison_ptr().unsafe_take_pointee())


struct Poison(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        _poison_ptr().write(True)

    def __init__(out self, *, deinit move: Self):
        _poison_ptr().write(True)

    def __deinit__(deinit self):
        _poison_ptr().write(True)


comptime TestVariant = Variant[MoveCopyCounter, Poison]


def test_basic() raises:
    comptime IntOrString = Variant[Int, String]
    var i = IntOrString(4)
    var s = IntOrString("4")

    # isa
    assert_true(i.isa[Int]())
    assert_false(i.isa[String]())
    assert_true(s.isa[String]())
    assert_false(s.isa[Int]())

    # get
    assert_equal(4, i[Int])
    assert_equal("4", s[String])
    # we don't test what happens when you `get` the wrong type.
    # have fun!

    # set
    i.set[String]("i")
    assert_false(i.isa[Int]())
    assert_true(i.isa[String]())
    assert_equal("i", i[String])


def test_copy() raises:
    var v1 = TestVariant(MoveCopyCounter())
    var v2 = v1
    # didn't call copyinit
    assert_equal(v1[MoveCopyCounter].copied, 0)
    assert_equal(v2[MoveCopyCounter].copied, 1)
    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_explicit_copy() raises:
    var v1 = TestVariant(MoveCopyCounter())

    # Perform explicit copy
    var v2 = v1.copy()

    # Test copy counts
    assert_equal(v1[MoveCopyCounter].copied, 0)
    assert_equal(v2[MoveCopyCounter].copied, 1)

    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_move() raises:
    var v1 = TestVariant(MoveCopyCounter())
    var v2 = v1
    # didn't call moveinit
    assert_equal(v1[MoveCopyCounter].moved, 1)
    assert_equal(v2[MoveCopyCounter].moved, 1)
    # test that we didn't call the other moveinit too!
    assert_no_poison()


def test_del() raises:
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(Pointer(to=deleted)))
    _ = v1^  # call __deinit__
    assert_true(deleted)
    # test that we didn't call the other deleter too!
    assert_no_poison()


def test_set_calls_deleter() raises:
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var deleted2: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(Pointer(to=deleted)))
    v1.set(ObservableDel(Pointer(to=deleted2)))
    assert_true(deleted)
    assert_false(deleted2)
    _ = v1^
    assert_true(deleted2)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_replace() raises:
    var v1: Variant[Int, String] = 998
    var x = v1.replace[String, Int]("hello")

    assert_equal(x, 998)


def test_unwrap_doesnt_call_deleter() raises:
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(Pointer(to=deleted)))
    assert_false(deleted)
    var v2 = v1^.unsafe_unwrap[ObservableDel[]]()
    assert_false(deleted)
    _ = v2
    assert_true(deleted)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_get_returns_mutable_reference() raises:
    var v1: Variant[Int, String] = 42
    var x = v1[Int]
    assert_equal(42, x)
    x = 100
    assert_equal(100, x)
    v1.set[String]("hello")
    assert_equal(100, x)  # the x reference is still valid

    var v2: Variant[Int, String] = "something"
    v2[String] = "something else"
    assert_equal(v2[String], "something else")


def test_is_type_supported() raises:
    var _x: Variant[Float64, Int32]
    assert_equal(_x.is_type_supported[Float64](), True)
    assert_equal(_x.is_type_supported[Int32](), True)
    assert_equal(_x.is_type_supported[Float32](), False)
    assert_equal(_x.is_type_supported[UInt32](), False)
    var _y: Variant[SIMD[.uint8, 2], SIMD[.uint8, 4]]
    assert_equal(_y.is_type_supported[SIMD[.uint8, 2]](), True)
    assert_equal(_y.is_type_supported[SIMD[.uint8, 4]](), True)
    assert_equal(_y.is_type_supported[SIMD[.uint8, 8]](), False)


def test_variant_works_with_move_only_types() raises:
    var v1 = Variant[MoveOnly[Int], MoveOnly[String]](MoveOnly[Int](42))
    var v2 = v1^
    assert_equal(v2[MoveOnly[Int]].data, 42)


def test_variant_linear_type_unwrap() raises:
    var v = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))

    var x = v^.unwrap[ExplicitDelOnly]()

    var data = x.data
    # Destroy before potentially raising after assert
    x^.destroy()
    assert_equal(data, 5)


def test_variant_linear_type_deinit_with() raises:
    # Test destroying a linear variant element in-place
    var v1 = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))
    v1^.deinit_with[ExplicitDelOnly](ExplicitDelOnly.destroy)

    # Test destroying a non-linear variant element in-place
    var v2 = Variant[ExplicitDelOnly, String]("notlinear")
    v2^.deinit_with[String](String.__deinit__)


def test_variant_pinned_linear_type_deinit_with() raises:
    def make() -> PinnedExplicitDelOnly:
        return PinnedExplicitDelOnly(5)

    var v = Variant[PinnedExplicitDelOnly, Int](init_with=make)
    var data = v[PinnedExplicitDelOnly].data
    # Destroy before potentially raising after assert
    v^.deinit_with[PinnedExplicitDelOnly](PinnedExplicitDelOnly.destroy)
    assert_equal(data, 5)


def test_variant_linear_type_move() raises:
    var v1 = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))
    var v2 = v1^

    v2^.deinit_with[ExplicitDelOnly](ExplicitDelOnly.destroy)


def test_variant_deinit_with_stateful_closure() raises:
    # Verify a closure capturing local state can be passed as
    # `deinit_func` (not just a thin function reference).
    var counter = 0

    def increment_counter(var _value: Int) {mut counter}:
        counter += 1

    var v = Variant[Int, String](42)
    v^.deinit_with[Int](increment_counter)
    assert_equal(counter, 1)


def test_variant_trivial_del() raises:
    comptime yes = ConfigureTrivial[del_is_trivial=True]
    comptime no = ConfigureTrivial[del_is_trivial=False]

    assert_true(IsTriviallyDeinitable[Variant[yes]])
    assert_false(IsTriviallyDeinitable[Variant[no]])
    assert_false(IsTriviallyDeinitable[Variant[yes, no]])

    # TODO (MOCO-3016):
    # check variant of linear type
    # assert_false(IsTriviallyDeinitable[Variant[LinearType]])


def test_variant_trivial_copyinit() raises:
    comptime yes = ConfigureTrivial[copyinit_is_trivial=True]
    comptime no = ConfigureTrivial[copyinit_is_trivial=False]

    assert_true(IsTriviallyCopyable[Variant[yes]])
    assert_false(IsTriviallyCopyable[Variant[no]])
    assert_false(IsTriviallyCopyable[Variant[yes, no]])

    # check variant of move-only type. `Variant[MoveOnly[Int]]` does not
    # conform to `Copyable`, so we read the trivial-flag field directly
    # rather than calling `IsTriviallyCopyable`, which constrains its
    # type parameter to `Copyable`.
    assert_false(Variant[MoveOnly[Int]].__copy_ctor_is_trivial)


def test_variant_trivial_moveinit() raises:
    comptime yes = ConfigureTrivial[moveinit_is_trivial=True]
    comptime no = ConfigureTrivial[moveinit_is_trivial=False]

    assert_true(IsTriviallyMovable[Variant[yes]])
    assert_false(IsTriviallyMovable[Variant[no]])
    assert_false(IsTriviallyMovable[Variant[yes, no]])

    # check variant of non-movable type
    # # TODO(MOCO-3383): Compiler issue with folding non-struct types
    # assert_false(IsTriviallyMovable[Variant[NonMovable]])


def test_variant_write_to() raises:
    var v = Variant[Int, String](42)
    check_write_to(v, expected="42", is_repr=False)
    v = "hello"
    check_write_to(v, expected="hello", is_repr=False)


def test_variant_write_repr_to() raises:
    var v = Variant[Int, String](42)
    check_write_to(
        v, expected="Variant[SIMD[DType.int, 1], String](Int(42))", is_repr=True
    )
    v = "hello"
    check_write_to(
        v, expected="Variant[SIMD[DType.int, 1], String]('hello')", is_repr=True
    )


@fieldwise_init
struct EmptyAndTrivial[Tag: Int = 0](TrivialRegisterPassable):
    pass


def test_variant_niche_optimization_size() raises:
    comptime NicheableType = Observable[opt_into_unsafe_niche=True]

    # Fits the optional-niche criteria
    assert_equal(
        size_of[Variant[NicheableType, EmptyAndTrivial[]]](),
        size_of[NicheableType](),
    )
    assert_equal(
        size_of[Variant[EmptyAndTrivial[], NicheableType]](),
        size_of[NicheableType](),
    )

    # Int does _not_ implement `UnsafeNicheable`
    assert_true(size_of[Variant[Int, EmptyAndTrivial[]]]() > size_of[Int]())
    assert_true(size_of[Variant[EmptyAndTrivial[], Int]]() > size_of[Int]())

    # More than 1 "empty type" does not opt into the niche optimization
    assert_true(
        size_of[
            Variant[NicheableType, EmptyAndTrivial[0], EmptyAndTrivial[1]]
        ]()
        > size_of[NicheableType]()
    )


def test_niched_variant_correctly_handles_lifecycle() raises:
    var copies = 0
    var moves = 0
    var dels = 0
    comptime NicheableType = Observable[
        CopyOrigin=origin_of(copies),
        MoveOrigin=origin_of(moves),
        DelOrigin=origin_of(dels),
        opt_into_unsafe_niche=True,
    ]
    comptime VariantType = Variant[NicheableType, EmptyAndTrivial[]]

    var empty = VariantType(EmptyAndTrivial())
    _ = empty^
    assert_equal(copies, 0)
    assert_equal(moves, 0)
    assert_equal(dels, 0)

    var observe = VariantType(
        NicheableType(
            copies=Pointer(to=copies),
            moves=Pointer(to=moves),
            dels=Pointer(to=dels),
        )
    )
    assert_equal(copies, 0)
    assert_equal(moves, 1)
    assert_equal(dels, 0)

    var copy_observe = observe.copy()
    assert_equal(copies, 1)
    assert_equal(moves, 2)
    assert_equal(dels, 0)

    _ = copy_observe^
    _ = observe^
    assert_equal(copies, 1)
    assert_equal(moves, 2)
    assert_equal(dels, 2)


def test_variant_eq() raises:
    comptime IntOrStr = Variant[Int, String]
    assert_true(IntOrStr(42) == IntOrStr(42))
    assert_true(IntOrStr("hello") == IntOrStr("hello"))
    assert_false(IntOrStr(42) == IntOrStr(99))
    assert_false(IntOrStr("hello") == IntOrStr("world"))
    assert_false(IntOrStr(42) == IntOrStr("42"))

    assert_true(IntOrStr(1) != IntOrStr(2))
    assert_true(IntOrStr(42) != IntOrStr("42"))
    assert_false(IntOrStr(1) != IntOrStr(1))

    comptime V3 = Variant[Int, String, Float64]
    assert_true(V3(42) == V3(42))
    assert_false(V3(42) == V3(3.14))
    assert_false(V3(42) == V3("42"))

    comptime V1 = Variant[Int]
    assert_true(V1(42) == V1(42))
    assert_false(V1(42) != V1(42))
    assert_true(V1(42) != V1(99))


def test_variant_hash() raises:
    comptime IntOrStr = Variant[Int, String]
    assert_equal(hash(IntOrStr(42)), hash(IntOrStr(42)))
    assert_equal(hash(IntOrStr("hello")), hash(IntOrStr("hello")))
    assert_true(hash(IntOrStr(42)) != hash(IntOrStr(99)))
    assert_true(hash(IntOrStr(42)) != hash(IntOrStr("42")))

    comptime V1 = Variant[Int]
    assert_equal(hash(V1(42)), hash(V1(42)))
    assert_true(hash(V1(42)) != hash(V1(99)))


@fieldwise_init
struct _Bare(Movable):
    """A `Movable & Deinitable` type that conforms to nothing
    else — used to exercise the negative case of `Variant`'s conditional
    conformances."""

    var n: Int


@fieldwise_init
struct _Pinned(Movable where False):
    """A non-`Movable` (pinned) type; still implicitly deletable by default."""

    var value: Int


def test_variant_conditional_conformances() raises:
    assert_true(conforms_to(Variant[Int, String], Equatable))
    assert_true(conforms_to(Variant[Int], Equatable))
    assert_true(conforms_to(Variant[Int, String], Hashable))
    assert_true(conforms_to(Variant[Int], Hashable))
    assert_true(conforms_to(Variant[Int, String], Writable))
    assert_true(conforms_to(Variant[Int], Writable))

    assert_false(conforms_to(Variant[_Bare], Equatable))
    assert_false(conforms_to(Variant[_Bare], Hashable))
    assert_false(conforms_to(Variant[_Bare], Writable))

    # Copyable: all types Copyable
    assert_true(conforms_to(Variant[Int, String], Copyable))
    assert_true(conforms_to(Variant[Int], Copyable))

    # Copyable: move-only type
    assert_false(conforms_to(Variant[MoveOnly[Int]], Copyable))
    assert_false(conforms_to(Variant[Int, MoveOnly[Int]], Copyable))

    # ImplicitlyCopyable: all types ImplicitlyCopyable
    assert_true(conforms_to(Variant[Int, Bool], ImplicitlyCopyable))

    # ImplicitlyCopyable: not all types ImplicitlyCopyable
    assert_false(conforms_to(Variant[MoveOnly[Int]], ImplicitlyCopyable))

    # RegisterPassable: all types RP
    assert_true(conforms_to(Variant[Int, Bool], RegisterPassable))
    assert_true(conforms_to(Variant[Int], RegisterPassable))

    # RegisterPassable: all types non-RP
    assert_false(conforms_to(Variant[String, List[Int]], RegisterPassable))

    # RegisterPassable: mixture of RP and non-RP
    assert_false(conforms_to(Variant[Int, String], RegisterPassable))
    assert_false(conforms_to(Variant[Bool, List[Int], Int], RegisterPassable))

    # Movable: all types Movable
    assert_true(conforms_to(Variant[Int, String], Movable))

    # Movable: non-Movable (pinned) alternative
    assert_false(conforms_to(Variant[_Pinned, Int], Movable))

    # Movable: linear alternative (Movable, not Deinitable)
    assert_true(conforms_to(Variant[ExplicitDelOnly, Int], Movable))


def test_variant_admits_non_movable_type() raises:
    # The `AnyType` floor admits a non-`Movable` type in the type list. The
    # value constructor still requires `Movable`, so it stores the movable
    # type; the non-`Movable` type is populated via `init_with=` (see
    # `test_variant_closure_construction`).
    var v = Variant[_Pinned, Int](42)
    assert_true(v.isa[Int]())
    assert_false(v.isa[_Pinned]())
    assert_equal(v[Int], 42)


def test_variant_closure_construction() raises:
    # Populate a non-`Movable` (pinned) type in place via a closure.
    def make_pinned() -> _Pinned:
        return _Pinned(7)

    var v = Variant[_Pinned, Int](init_with=make_pinned)
    assert_true(v.isa[_Pinned]())
    assert_false(v.isa[Int]())
    assert_equal(v[_Pinned].value, 7)

    # The closure's return type selects the type; here a `Movable` one.
    def make_int() -> Int:
        return 42

    var v2 = Variant[_Pinned, Int](init_with=make_int)
    assert_true(v2.isa[Int]())
    assert_equal(v2[Int], 42)


def test_variant_closure_replacement() raises:
    # Replace the movable value with a closure-constructed non-`Movable` value.
    def make_pinned() -> _Pinned:
        return _Pinned(9)

    var v = Variant[_Pinned, Int](0)
    assert_true(v.isa[Int]())
    v.set(init_with=make_pinned)
    assert_true(v.isa[_Pinned]())
    assert_equal(v[_Pinned].value, 9)


def test_variant_closure_called_once() raises:
    # The initializer closure is invoked exactly once per construction/set.
    var calls = 0

    def make() {mut calls} -> Int:
        calls += 1
        return 5

    var v = Variant[Int, String](init_with=make)
    assert_equal(calls, 1)
    assert_equal(v[Int], 5)

    v.set(init_with=make)
    assert_equal(calls, 2)
    assert_equal(v[Int], 5)


def test_variant_closure_set_calls_deleter() raises:
    # Closure-based `set` destroys the outgoing value before emplacing the new
    # one.
    comptime TestDeleterVariant = Variant[ObservableDel[], Int]
    var deleted = False
    var v = TestDeleterVariant(ObservableDel(Pointer(to=deleted)))
    assert_false(deleted)

    def make() -> Int:
        return 5

    v.set(init_with=make)
    assert_true(deleted)
    assert_true(v.isa[Int]())
    assert_equal(v[Int], 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
