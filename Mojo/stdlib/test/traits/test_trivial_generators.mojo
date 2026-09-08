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
"""Tests for `IsTriviallyMovable` / `IsTriviallyCopyable` / `IsTriviallyDeinitable`."""

from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from std.testing import TestSuite, assert_equal, assert_false, assert_true
from test_utils import ConfigureTrivial


@fieldwise_init
struct AllTrivial(Copyable):
    """A struct whose move/copy/deinit are all trivial."""

    var value: Int


struct NoneTrivial(Copyable):
    """A struct whose move/copy/deinit are all non-trivial because of user-defined
    lifecycle methods."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit move: Self):
        self.value = move.value

    def __deinit__(deinit self):
        pass


struct NonMovable(Movable where False):
    pass


struct NonCopyable(Movable):
    pass


struct NonDeinitable(Deinitable where False, Movable):
    pass


def test_builtin_scalar_types() raises:
    assert_true(IsTriviallyMovable[Int])
    assert_true(IsTriviallyCopyable[Int])
    assert_true(IsTriviallyDeinitable[Int])

    assert_true(IsTriviallyMovable[Bool])
    assert_true(IsTriviallyCopyable[Bool])
    assert_true(IsTriviallyDeinitable[Bool])

    assert_true(IsTriviallyMovable[Float64])
    assert_true(IsTriviallyCopyable[Float64])
    assert_true(IsTriviallyDeinitable[Float64])


def test_string_is_non_trivial_copy_and_del() raises:
    # `String` owns a heap buffer, so copy and destruction are not trivial.
    assert_false(IsTriviallyCopyable[String])
    assert_false(IsTriviallyDeinitable[String])
    # Moves remain a bit-copy though.
    assert_true(IsTriviallyMovable[String])


def test_struct_with_only_trivial_fields() raises:
    assert_true(IsTriviallyMovable[AllTrivial])
    assert_true(IsTriviallyCopyable[AllTrivial])
    assert_true(IsTriviallyDeinitable[AllTrivial])


def test_struct_with_user_defined_lifecycle() raises:
    assert_false(IsTriviallyMovable[NoneTrivial])
    assert_false(IsTriviallyCopyable[NoneTrivial])
    assert_false(IsTriviallyDeinitable[NoneTrivial])


def test_non_movable_type_is_never_trivial() raises:
    assert_false(IsTriviallyMovable[NonMovable])
    assert_false(IsTriviallyCopyable[NonMovable])


def test_non_copyable_type_is_trivially_movable_but_not_copyable() raises:
    assert_true(IsTriviallyMovable[NonCopyable])
    assert_false(IsTriviallyCopyable[NonCopyable])


def test_non_deinitable_type_is_never_trivially_deletable() raises:
    assert_false(IsTriviallyDeinitable[NonDeinitable])
    assert_true(IsTriviallyMovable[NonDeinitable])


def test_configure_trivial_flags() raises:
    # Each flag can be toggled independently.
    comptime AllOn = ConfigureTrivial[
        del_is_trivial=True,
        copyinit_is_trivial=True,
        moveinit_is_trivial=True,
    ]
    assert_true(IsTriviallyMovable[AllOn])
    assert_true(IsTriviallyCopyable[AllOn])
    assert_true(IsTriviallyDeinitable[AllOn])

    comptime OnlyMove = ConfigureTrivial[
        del_is_trivial=False,
        copyinit_is_trivial=False,
        moveinit_is_trivial=True,
    ]
    assert_true(IsTriviallyMovable[OnlyMove])
    assert_false(IsTriviallyCopyable[OnlyMove])
    assert_false(IsTriviallyDeinitable[OnlyMove])

    comptime OnlyCopy = ConfigureTrivial[
        del_is_trivial=False,
        copyinit_is_trivial=True,
        moveinit_is_trivial=False,
    ]
    assert_false(IsTriviallyMovable[OnlyCopy])
    assert_true(IsTriviallyCopyable[OnlyCopy])
    assert_false(IsTriviallyDeinitable[OnlyCopy])

    comptime OnlyDel = ConfigureTrivial[
        del_is_trivial=True,
        copyinit_is_trivial=False,
        moveinit_is_trivial=False,
    ]
    assert_false(IsTriviallyMovable[OnlyDel])
    assert_false(IsTriviallyCopyable[OnlyDel])
    assert_true(IsTriviallyDeinitable[OnlyDel])


def test_helpers_match_underlying_flags() raises:
    # The helpers must agree with the raw trait fields they wrap.
    assert_equal(IsTriviallyMovable[Int], Int.__move_ctor_is_trivial)
    assert_equal(IsTriviallyCopyable[Int], Int.__copy_ctor_is_trivial)
    assert_equal(IsTriviallyDeinitable[Int], Int.__del__is_trivial)
    assert_equal(IsTriviallyMovable[String], String.__move_ctor_is_trivial)
    assert_equal(IsTriviallyCopyable[String], String.__copy_ctor_is_trivial)
    assert_equal(IsTriviallyDeinitable[String], String.__del__is_trivial)


def has_trivial_where_clauses[
    T: AnyType
]() where (
    IsTriviallyDeinitable[T]
    and IsTriviallyMovable[T]
    and IsTriviallyCopyable[T]
):
    pass


# NOTE: We don't actually need to call this function to test.
# We only need to make sure the passing `T` to `has_trivial_where_clauses` compiles successfully.
# As this proves the `T` conforming to `TrivialRegisterPassable` is enough evidence to conclude
# that the deinit/move/copy are trivial.
def takes_TRP_type[T: TrivialRegisterPassable]():
    has_trivial_where_clauses[T]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
