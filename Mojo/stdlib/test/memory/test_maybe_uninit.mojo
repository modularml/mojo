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

from std.memory import (
    MaybeUninit,
    unsafe_memcmp,
)
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from std.sys import size_of
from test_utils import (
    AbortOnDel,
    ConfigureTrivial,
    DelRecorder,
    ExplicitDelOnly,
    DelCounter,
    MoveCounter,
    PinnedExplicitDelOnly,
    TriviallyCopyableMoveCounter,
)
from std.testing import *


def test_maybe_uninitialized() raises:
    # Every time an Int is destroyed, it's going to be recorded here.
    var destructor_recorder = List[Int]()

    var ptr = Pointer(to=destructor_recorder).as_imm()
    var a = MaybeUninit[DelRecorder[ptr.origin]]()
    a.unsafe_write(DelRecorder(42, ptr))

    var value = a.unsafe_assume_init().value
    var dels_after_write = len(destructor_recorder)
    var ptr_value = a.unsafe_ptr()[].value

    a.unsafe_ptr().unsafe_deinit_pointee()
    var dels_after_deinit = len(destructor_recorder)
    var recorded_value = destructor_recorder[0]

    # `DelRecorder` isn't trivially deinitable, so `a` is linear and must be
    # forgotten before any call (like `assert_equal`) that could raise.
    a^.unsafe_forget()

    assert_equal(value, 42)
    assert_equal(dels_after_write, 0)
    assert_equal(ptr_value, 42)
    assert_equal(dels_after_deinit, 1)
    assert_equal(recorded_value, 42)
    assert_equal(len(destructor_recorder), 1)


def test_unsafe_forget_skips_destructor() raises:
    var a = MaybeUninit[AbortOnDel]()
    a.unsafe_write(AbortOnDel(42))
    a^.unsafe_forget()

    var b = MaybeUninit[AbortOnDel](AbortOnDel(42))
    b^.unsafe_forget()


def test_unsafe_deinit_runs_destructor_once() raises:
    var count = 0
    var a = MaybeUninit(DelCounter(Pointer(to=count)))
    a^.unsafe_deinit()

    assert_equal(count, 1)


def test_write_over_initialized_leaks_previous_value() raises:
    var count = 0
    var a = MaybeUninit(DelCounter(Pointer(to=count)))

    # Does not invoke the destructor of the initial `DelCounter`
    a.unsafe_write(DelCounter(Pointer(to=count)))

    # Invoke the destructor once.
    a^.unsafe_deinit()

    assert_equal(count, 1)


def test_write() raises:
    var a = MaybeUninit[Int]()
    a.write(41)
    assert_equal(a.unsafe_assume_init(), 41)

    a.write(42)
    assert_equal(a.unsafe_assume_init(), 42)


def test_unsafe_assume_init_move() raises:
    var a = MoveCounter(0)
    var uninit = MaybeUninit[MoveCounter[Int]](a^)
    var moved = uninit^.unsafe_assume_init()
    assert_equal(moved.move_count, 2)


def test_zeroed() raises:
    # For Int, zeroed memory is valid and should be 0.
    var a = MaybeUninit[Int].zeroed()
    assert_equal(a.unsafe_assume_init(), 0)

    # For UInt64, zeroed memory should also be 0.
    var b = MaybeUninit[UInt64].zeroed()
    assert_equal(b.unsafe_assume_init(), 0)

    var c = MaybeUninit[String].zeroed()
    var arr = Array[Byte, size_of[String]()](fill=0)
    var cmp = unsafe_memcmp(
        c.unsafe_ptr().unsafe_bitcast[Byte](),
        arr.unsafe_ptr(),
        size_of[String](),
    )
    # All-zero bytes aren't a valid `String`, so forget rather than deinit.
    c^.unsafe_forget()
    assert_equal(cmp, 0)

    # A linear `T` keeps the wrapper linear even after `.zeroed()`.
    comptime LinearWrapper = MaybeUninit[ExplicitDelOnly]
    var d = LinearWrapper.zeroed()
    d^.unsafe_forget()
    assert_false(conforms_to(LinearWrapper, Deinitable))


def test_triviality() raises:
    comptime Trivial = MaybeUninit[Int]
    comptime NotTrivial = MaybeUninit[
        ConfigureTrivial[
            copyinit_is_trivial=False,
            moveinit_is_trivial=False,
            del_is_trivial=False,
        ]
    ]

    assert_true(IsTriviallyCopyable[Trivial])
    assert_true(IsTriviallyMovable[Trivial])
    assert_true(IsTriviallyDeinitable[Trivial])

    assert_false(IsTriviallyCopyable[NotTrivial])
    assert_false(IsTriviallyMovable[NotTrivial])
    assert_false(IsTriviallyDeinitable[NotTrivial])


def test_conditional_register_passable() raises:
    assert_true(conforms_to(MaybeUninit[Int], RegisterPassable))
    assert_true(conforms_to(MaybeUninit[Bool], RegisterPassable))
    assert_false(conforms_to(MaybeUninit[List[Int]], RegisterPassable))
    assert_false(conforms_to(MaybeUninit[String], RegisterPassable))


def test_conditional_conformance() raises:
    comptime DelOnly = MaybeUninit[ConfigureTrivial[del_is_trivial=True]]
    assert_true(conforms_to(DelOnly, Deinitable))
    assert_false(conforms_to(DelOnly, Movable))
    assert_false(conforms_to(DelOnly, Copyable))
    assert_false(conforms_to(DelOnly, ImplicitlyCopyable))

    comptime MoveOnly = MaybeUninit[ConfigureTrivial[moveinit_is_trivial=True]]
    assert_false(conforms_to(MoveOnly, Deinitable))
    assert_true(conforms_to(MoveOnly, Movable))
    assert_false(conforms_to(MoveOnly, Copyable))
    assert_false(conforms_to(MoveOnly, ImplicitlyCopyable))

    # Trivially copyable but not trivially movable: `ImplicitlyCopyable`
    # requires both.
    assert_true(IsTriviallyCopyable[TriviallyCopyableMoveCounter])
    assert_false(IsTriviallyMovable[TriviallyCopyableMoveCounter])
    comptime CopyOnly = MaybeUninit[TriviallyCopyableMoveCounter]
    assert_false(conforms_to(CopyOnly, Movable))
    assert_false(conforms_to(CopyOnly, Copyable))
    assert_false(conforms_to(CopyOnly, ImplicitlyCopyable))

    comptime CopyAndMove = MaybeUninit[
        ConfigureTrivial[copyinit_is_trivial=True, moveinit_is_trivial=True]
    ]
    assert_false(conforms_to(CopyAndMove, Deinitable))
    assert_true(conforms_to(CopyAndMove, Movable))
    assert_true(conforms_to(CopyAndMove, Copyable))
    assert_true(conforms_to(CopyAndMove, ImplicitlyCopyable))

    # `String`: trivially movable, but not trivially copyable or deinitable.
    assert_true(IsTriviallyMovable[String])
    comptime StringWrapper = MaybeUninit[String]
    assert_false(conforms_to(StringWrapper, Deinitable))
    assert_true(conforms_to(StringWrapper, Movable))
    assert_false(conforms_to(StringWrapper, Copyable))
    assert_false(conforms_to(StringWrapper, ImplicitlyCopyable))


# This test doesn't need to run, it just needs to compile
def _test_trivial_register_passable_types[T: TrivialRegisterPassable]():
    var uninit = MaybeUninit[T]()

    # OK: Check that the type is implicitly copyable
    var _uninit2 = uninit

    # OK: Check that the type is movable
    var _uninit3 = uninit^

    # OK: No need to expliclilty deinit


def test_linear_payload_round_trip() raises:
    assert_false(conforms_to(MaybeUninit[ExplicitDelOnly], Deinitable))
    assert_true(conforms_to(MaybeUninit[ExplicitDelOnly], Movable))

    var a = MaybeUninit[ExplicitDelOnly]()
    a.unsafe_write(ExplicitDelOnly(5))

    var value = a^.unsafe_assume_init()
    var data = value.data
    value^.destroy()

    assert_equal(data, 5)


def test_pinned_linear_payload_stays_pinned() raises:
    assert_false(conforms_to(MaybeUninit[PinnedExplicitDelOnly], Deinitable))
    assert_false(conforms_to(MaybeUninit[PinnedExplicitDelOnly], Movable))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
