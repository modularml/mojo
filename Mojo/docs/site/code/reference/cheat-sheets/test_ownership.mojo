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
# Tests for the Mojo Ownership cheat sheet.
#
# Each test asserts a behavioral claim the card makes, so the card fails CI if a
# claim drifts. The compile-error boundaries the card states (used-after-transfer,
# `mut` needs a mutable arg, `var x = plain_copyable` errors) cannot be asserted
# at runtime and are not covered here.
# ===----------------------------------------------------------------------=== #
from std.testing import assert_equal


struct Box(ImplicitlyCopyable, Movable):
    var v: Int

    def __init__(out self, v: Int):
        self.v = v


def take_imm(imm x: Int) -> Int:  # imm = immutable borrow
    return x


def take_mut(mut x: Int):  # mut = mutable borrow (writes through)
    x += 100


def take_var(var b: Box) -> Int:  # var = owns the value
    return b.v


def first[T: Movable](ref xs: List[T]) -> ref[xs[0]] T:
    return xs[0]  # len(xs) known to be > 0


# Literals / Trivial values: a fixed-width scalar is a one-lane SIMD.
def test_int_simd_aliases() raises:
    var i32: Int32 = 5
    var s: SIMD[
        DType.int32, 1
    ] = i32  # no conversion: Int32 IS SIMD[DType.int32, 1]
    assert_equal(Int(s), 5)


# var owns; ref refers and writes through.
def test_var_owns_ref_refers() raises:
    var data: List[Int] = [1, 2, 3]
    ref view = data[0]
    view = 9
    assert_equal(data[0], 9)


# Var assignment: construct / implicit copy / explicit copy / transfer / copy-out-of-ref.
def test_var_assignment() raises:
    var made = Box(1)  # construct
    assert_equal(made.v, 1)
    var icopy = made  # implicit copy (ImplicitlyCopyable)
    assert_equal(icopy.v, 1)
    assert_equal(made.v, 1)  # a copy leaves the source owning its value
    var ecopy = made.copy()  # explicit copy
    assert_equal(ecopy.v, 1)
    var lst: List[Box] = [Box(7)]
    ref r = lst[0]
    var copied = r  # copy out of a reference
    assert_equal(copied.v, 7)
    var moved = made^  # transfer (last use of made)
    assert_equal(moved.v, 1)


# Mutability: mut writes through; imm borrows.
def test_mut_and_read() raises:
    var m = 0
    take_mut(m)
    assert_equal(m, 100)
    assert_equal(take_imm(m), 100)


# Call sites into an owning var param follow the var assignment rules.
def test_call_sites() raises:
    assert_equal(take_var(Box(1)), 1)  # construct into the arg
    var b = Box(2)
    assert_equal(take_var(b), 2)  # f(x): implicit copy; b still usable
    assert_equal(b.v, 2)
    assert_equal(take_var(b.copy()), 2)  # f(x.copy()): explicit copy
    var lst: List[Box] = [Box(3)]
    ref r = lst[0]
    assert_equal(take_var(r), 3)  # a reference copies out
    assert_equal(take_var(b^), 2)  # f(x^): transfer (last use of b)


# Origins: a ref return writes through to its source.
def test_ref_return_origin() raises:
    var lst: List[Int] = [10, 20, 30]
    ref x = first(lst)
    x = 99
    assert_equal(lst[0], 99)


# Returns: implicit copy / explicit copy / transfer out.
def ret_copy() -> Box:
    var b = Box(5)
    return b


def ret_explicit() -> Box:
    var b = Box(6)
    return b.copy()


def ret_transfer() -> Box:
    var b = Box(7)
    return b^


def test_returns() raises:
    assert_equal(ret_copy().v, 5)
    assert_equal(ret_explicit().v, 6)
    assert_equal(ret_transfer().v, 7)


# Non-owning views: a contiguous slice is a Span view that writes through to the
# source (no copy); a codepoint slice is a StringSpan view.
def test_views() raises:
    var data: List[String] = ["a", "b", "c", "d", "e"]
    var s = data[1:3]  # contiguous slice returns a Span view, no copy
    s[0] = "X"  # writes through to the source
    assert_equal(data[1], "X")
    assert_equal(s[0], "X")
    var text = "Hello, World!"
    var hi = text[codepoint=0:5]  # StringSpan view
    assert_equal(String(hi), "Hello")


# Consuming iteration: `^` drains a collection, moving each element out.
def test_consuming_iteration() raises:
    var items: List[Box] = [Box(1), Box(2), Box(3)]
    var total = 0
    for var x in items^:  # items drained; each x is owned
        total += x.v
    assert_equal(total, 6)


def main() raises:
    test_int_simd_aliases()
    test_var_owns_ref_refers()
    test_var_assignment()
    test_mut_and_read()
    test_call_sites()
    test_ref_return_origin()
    test_returns()
    test_views()
    test_consuming_iteration()
