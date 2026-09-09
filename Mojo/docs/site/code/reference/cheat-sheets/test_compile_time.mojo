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
# test_compile_time.mojo
# Tests for the Mojo "compile-time" cheat-sheet card.
#
# Exercises the card's claims so one that drifts stops compiling or fails an
# assert: parameters, where clauses, trait bounds, running a function at
# compile time, comptime if/for, sys.info queries, comptime members, and the
# inlining decorators.
#
# Not tested (no portable assertable value, or a compile error can't be
# asserted at runtime):
#   - reflect[T] field surface beyond .name(): newly introduced and documented
#     as incomplete; the wider field API may still shift.
#   - numeric precision: float equality is touchy to assert, and 2**200
#     overflows a 64-bit Int so it can't be materialized to compare.
#   - comptime if on hardware facts (is_nvidia_gpu, ...): machine-dependent.
#   - the compile-time boundary (no file I/O, no raising, runs on CPU): these
#     are compile errors, which a runtime test can't assert.
from std.testing import assert_equal
from std.sys.info import size_of, align_of
from std.builtin.globals import global_constant


# --- parameters: a value parameter and a type parameter ---
def scaled[factor: Int](x: Int) -> Int:
    return x * factor  # factor is a compile-time value parameter


def pick[T: Copyable](a: T, b: T, take_first: Bool) -> T:
    return a.copy() if take_first else b.copy()  # T is a type parameter


def test_parameters() raises:
    assert_equal(scaled[3](10), 30)
    assert_equal(pick[Int](1, 2, True), 1)


# --- where: gate on a numeric truth ---
def block[w: Int]() -> Int where w.is_power_of_two():
    return w  # only callable when w is a power of two


def test_where() raises:
    assert_equal(block[8](), 8)


# --- conformances: a trait bound admits only proven operations ---
def largest[T: Comparable & Copyable](xs: List[T]) -> T:
    var best_i = 0
    for i in range(len(xs)):
        if xs[i] > xs[best_i]:
            best_i = i
    return xs[best_i].copy()


def test_conformances() raises:
    assert_equal(largest([3, 1, 4, 1, 5]), 5)


# --- run code at compile time: any function, no marker ---
def square(x: Int) -> Int:
    return x * x


def test_run_at_compile_time() raises:
    comptime nine = square(3)  # square runs while compiling
    assert_equal(nine, 9)


# --- query the target ---
def test_query_target() raises:
    assert_equal(size_of[Int32](), 4)
    assert_equal(size_of[Int64](), 8)
    assert_equal(align_of[Int32](), 4)


# --- comptime members live on types ---
struct Stack[T: Copyable]:
    comptime Element = Self.T  # associated type (reached through Self)
    comptime capacity = 1024  # a compile-time value member


def test_comptime_members() raises:
    assert_equal(Stack[Int].capacity, 1024)
    var e: Stack[Int].Element = 5  # Element resolves to Int for Stack[Int]
    assert_equal(e, 5)


# --- comptime for: fully unrolled, index is a constant ---
def test_comptime_for() raises:
    var total = 0
    comptime for i in range(4):
        total += i  # 0 + 1 + 2 + 3
    assert_equal(total, 6)


# --- comptime if: only the live branch compiles ---
def test_comptime_if() raises:
    var width: Int
    comptime if size_of[Int64]() == 8:
        width = 64  # the live branch (condition is known)
    else:
        width = 0
    assert_equal(width, 64)


# --- inlining: force or forbid ---
@always_inline
def add_inline(a: Int, b: Int) -> Int:
    return a + b


@no_inline
def add_separate(a: Int, b: Int) -> Int:
    return a + b


def test_inlining() raises:
    assert_equal(add_inline(2, 3), 5)
    assert_equal(add_separate(2, 3), 5)


# --- conditional availability: a method exists only when its condition holds ---
@fieldwise_init
struct Buf[n: Int]:
    var value: Int

    def first(self) -> Int where Self.n > 0:  # exists only when n > 0
        return self.value


def test_conditional_availability() raises:
    var b = Buf[3](42)
    assert_equal(b.first(), 42)


# --- type_of: capture the type of an expression ---
def test_type_of() raises:
    var x = 5
    var y: type_of(x) = x + 5  # type_of(x) is Int
    assert_equal(y, 10)


# --- conditional construction: default-construct only when proven ---
def test_conditional_construction() raises:
    var x = 0
    comptime if conforms_to(type_of(x), Defaultable):
        var y = type_of(x)()  # the default constructor (no make_default)
        assert_equal(y, x)  # both 0


# --- reflect: read a type's name ---
def test_reflect() raises:
    comptime name = reflect[Int].name()
    # Int unifies with SIMD, so its reflected name is the SIMD spelling.
    assert_equal(String(name), "SIMD[DType.int, 1]")


# --- materialization: comptime value -> runtime ---
comptime POWERS: Array[Int, 4] = [1, 2, 4, 8]


def test_materialization() raises:
    comptime table: List[Int] = [3, 5, 7, 11, 13]
    var t = materialize[table]()  # heap-backed comptime List -> runtime List
    assert_equal(t[1], 5)
    ref g = global_constant[POWERS]()  # static table, indexed without a copy
    assert_equal(g[2], 4)


def main() raises:
    test_parameters()
    test_where()
    test_conformances()
    test_run_at_compile_time()
    test_query_target()
    test_comptime_members()
    test_comptime_for()
    test_comptime_if()
    test_inlining()
    test_conditional_availability()
    test_type_of()
    test_conditional_construction()
    test_reflect()
    test_materialization()
