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
# tests.mojo
# Tests for trivial.mdx code examples and prose claims.
#
# Page claims, and where each is verified:
#   1. The three predicates import from `std.traits`.
#      -> the import below; source: stdlib/std/traits/__init__.mojo:23-25
#   2. `IsTriviallyCopyable[T]` is true when `T` is `Copyable` and copying a
#      value's bits has no side effects.
#      -> test_trivially_copyable; source: stdlib/std/traits/copyable.mojo:93
#   3. `IsTriviallyMovable[T]` is true when `T` is `Movable` and moving a
#      value's bits has no side effects.
#      -> test_trivially_movable; source: stdlib/std/traits/movable.mojo:75
#   4. `IsTriviallyDeinitable[T]` is true when `T` is `Deinitable` and
#      deinitialization is a no-op.
#      -> test_trivially_deinitable; source: stdlib/std/traits/deinitable.mojo:97
#   5. Each predicate is a `comptime` value, so it is usable in `comptime if`
#      and `comptime assert`.
#      -> test_predicates_are_comptime
#
# Not tested:
#   - "the compiler can optimize or eliminate them" (a codegen claim, with no
#     source-level observation available from a test)
from std.testing import assert_false, assert_true
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)

# --- Types that stand on each side of the three predicates ---

# Bitwise copy, bitwise move, no-op deinit.
comptime Trivial = Int


@fieldwise_init
struct SideEffects(Copyable, Movable):
    """Custom lifecycle methods, so no operation is trivial."""

    var value: Int

    def __init__(out self, *, copy: Self):
        self.value = copy.value + 1  # a side effect of copying

    def __init__(out self, *, deinit move: Self):
        self.value = move.value

    def __deinit__(deinit self):
        _ = self.value  # a deinitializer body, so deinit is not a no-op


# --- IsTriviallyCopyable ---


def test_trivially_copyable() raises:
    assert_true(IsTriviallyCopyable[Trivial])
    assert_true(IsTriviallyCopyable[SIMD[DType.float32, 4]])
    # `String` is `Copyable`, but copying owns heap storage.
    assert_false(IsTriviallyCopyable[String])
    assert_false(IsTriviallyCopyable[SideEffects])


# --- IsTriviallyMovable ---


def test_trivially_movable() raises:
    assert_true(IsTriviallyMovable[Trivial])
    assert_true(IsTriviallyMovable[SIMD[DType.float32, 4]])
    assert_false(IsTriviallyMovable[SideEffects])


# --- IsTriviallyDeinitable ---


def test_trivially_deinitable() raises:
    assert_true(IsTriviallyDeinitable[Trivial])
    assert_true(IsTriviallyDeinitable[SIMD[DType.float32, 4]])
    # `String` releases heap storage, so its deinitializer is not a no-op.
    assert_false(IsTriviallyDeinitable[String])
    assert_false(IsTriviallyDeinitable[SideEffects])


# --- The predicates are compile-time values ---


def test_predicates_are_comptime() raises:
    comptime assert IsTriviallyCopyable[Trivial]
    comptime assert not IsTriviallyDeinitable[String]

    var branch_taken: Bool
    comptime if IsTriviallyMovable[Trivial]:
        branch_taken = True
    else:
        branch_taken = False
    assert_true(branch_taken)


def main() raises:
    test_trivially_copyable()
    test_trivially_movable()
    test_trivially_deinitable()
    test_predicates_are_comptime()
