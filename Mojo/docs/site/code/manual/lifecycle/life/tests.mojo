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
# Tests for life.mdx code examples.
#
# Not tested (no runnable behavior to assert from this file):
#   - `struct Target` with an `@implicit` initializer whose body is `# ...`
#     (a signature sketch; `Source` is never defined on the page)
#   - the bare `def __init__(out self, *, copy: Self)` signature fragment
#   - the bare `def __init__(out self, *, deinit move: Self)` signature fragment
#   - "An initializer must set up every field in a value" (a compile error;
#     verified separately as `'self.b' is uninitialized at the implicit return
#     from this function`)
#   - "All initializers use the `out self` argument convention" (a compile
#     error; verified as `__init__ method must return Self type with 'out'
#     argument`)
#   - transferring a `Pinned` value (a compile error; the type's immovability
#     is asserted here through `conforms_to` instead)
#   - the claim that braced shorthand helps most for compiler-inferred
#     parameterized types (prose pointer, deliberately unillustrated)
from std.atomic import Atomic
from std.memory import OwnedPointer
from std.testing import assert_equal, assert_false, assert_true
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)

# --- Fieldwise initialization ---
# The page shows both of these as `MyStruct`, once generated and once written
# out, so they are renamed here to coexist in one file.


@fieldwise_init
struct MyStructFieldwise:
    var field1: Int
    var field2: String


struct MyStructHandWritten:
    var field1: Int
    var field2: String

    def __init__(out self, field1: Int, field2: String):
        self.field1 = field1
        self.field2 = field2


def test_fieldwise_and_hand_written_agree() raises:
    var generated = MyStructFieldwise(1, "Hello")
    var written = MyStructHandWritten(1, "Hello")
    assert_equal(generated.field1, written.field1)
    assert_equal(generated.field2, written.field2)
    assert_equal(generated.field2, "Hello")


# --- Overloading initializers, and delegating between them ---


struct RetryPolicy:
    var max_attempts: Int
    var delay_ms: Int

    def __init__(out self):
        self = self.__init__(3)

    def __init__(out self, max_attempts: Int, delay_ms: Int = 1000):
        self.max_attempts = max_attempts
        self.delay_ms = delay_ms


def test_initializer_overloads() raises:
    var standard = RetryPolicy()
    var persistent = RetryPolicy(10)
    var aggressive = RetryPolicy(10, 250)
    # The no-argument overload delegates, so it picks up the default delay.
    assert_equal(standard.max_attempts, 3)
    assert_equal(standard.delay_ms, 1000)
    assert_equal(persistent.max_attempts, 10)
    assert_equal(persistent.delay_ms, 1000)
    assert_equal(aggressive.max_attempts, 10)
    assert_equal(aggressive.delay_ms, 250)


# --- No-initializer types ---


struct HTTPStatus:
    comptime OK = 200
    comptime NOT_FOUND = 404
    comptime INTERNAL_SERVER_ERROR = 500

    @staticmethod
    def is_success(code: Int) -> Bool:
        # 2xx is the HTTP status success class
        return 200 <= code < 300


def handle(status_code: Int) -> String:
    if HTTPStatus.is_success(status_code):
        return "ok"
    return "failed"


def test_no_initializer_type() raises:
    assert_equal(HTTPStatus.OK, 200)
    assert_equal(HTTPStatus.NOT_FOUND, 404)
    assert_equal(HTTPStatus.INTERNAL_SERVER_ERROR, 500)
    # Both range bounds, to pin the chained comparison.
    assert_false(HTTPStatus.is_success(199))
    assert_true(HTTPStatus.is_success(200))
    assert_true(HTTPStatus.is_success(299))
    assert_false(HTTPStatus.is_success(300))
    assert_equal(handle(204), "ok")
    assert_equal(handle(404), "failed")


# --- Initializers and implicit conversion ---


def test_optional_implicit_conversion() raises:
    var greeting: Optional[String] = None
    assert_false(greeting)
    greeting = String("Salve!")
    assert_true(greeting)
    assert_equal(greeting.value(), "Salve!")


struct Complex:
    var real: Float64
    var imag: Float64

    def __init__(out self, real: Float64, imag: Float64):
        self.real = real
        self.imag = imag

    @implicit
    def __init__(out self, value: Float64):
        self = Complex(value, 0.0)


def magnitude_squared(value: Complex) -> Float64:
    return value.real * value.real + value.imag * value.imag


def test_implicit_conversion_on_argument() raises:
    assert_equal(magnitude_squared(3.0), 9.0)


def test_implicit_conversion_on_return() raises:
    assert_equal(make_complex().real, 4.0)
    assert_equal(make_complex().imag, 0.0)


def make_complex() -> Complex:
    # The page states that implicit conversion also applies when returning.
    return 4.0


# --- Braced shorthand ---


def test_braced_shorthand() raises:
    # Spelled out, as the page's "Instead of:" block shows.
    assert_equal(magnitude_squared(Complex(real=3.0, imag=0.0)), 9.0)
    assert_equal(magnitude_squared(Complex(3.0, 0.0)), 9.0)
    assert_equal(magnitude_squared(Complex(value=3.0)), 9.0)
    assert_equal(magnitude_squared(Complex(3.0)), 9.0)
    # Braced, as the page's "You write:" block shows.
    assert_equal(magnitude_squared({real = 3.0, imag = 0.0}), 9.0)
    assert_equal(magnitude_squared({3.0, 0.0}), 9.0)
    assert_equal(magnitude_squared({value = 3.0}), 9.0)
    assert_equal(magnitude_squared({3.0}), 9.0)


struct Pair:
    var a: Int
    var b: Int

    def __init__(out self, a: Int, b: Int = 0):
        self.a = a
        self.b = b


def total(p: Pair) -> Int:
    return p.a + p.b


def test_braced_shorthand_without_implicit_conversion() raises:
    # `Pair` has no `@implicit` initializer, which is the page's claim that
    # braced shorthand works whether a type provides implicit initialization
    # or not.
    assert_equal(total({a = 1, b = 2}), 3)
    assert_equal(total({1, 2}), 3)
    assert_equal(total({1}), 1)


# --- Copy and move initializers ---


@fieldwise_init
struct ValueType(Copyable):
    var n: Int


def test_copy_and_move_shorthands() raises:
    var value = ValueType(7)
    # `value.copy()` and `ValueType(copy=value)` are the same operation.
    assert_equal(value.copy().n, 7)
    assert_equal(ValueType(copy=value).n, 7)
    # `value^` and `ValueType(move=value^)` are the same operation. The move
    # initializer consumes its argument, so the long form needs the transfer.
    var moved = value^
    assert_equal(ValueType(move=moved^).n, 7)


def copy_return[T: Copyable](foo: T) -> T:
    var copy = foo.copy()
    return copy^


def test_copyable_constraint_in_generic_code() raises:
    # All `Copyable` types are also `Movable`, so the copy can be transferred
    # out on return.
    assert_equal(copy_return(ValueType(9)).n, 9)


def test_copy_is_a_deep_copy_of_the_binding() raises:
    var original = ValueType(1)
    var duplicate = original.copy()
    duplicate.n = 2
    assert_equal(original.n, 1)
    assert_equal(duplicate.n, 2)


# --- Transfer is the default ---


def test_transfer_needs_no_declared_conformance() raises:
    # `RetryPolicy` declares no conformances and still transfers.
    assert_true(conforms_to(RetryPolicy, Movable))
    var policy = RetryPolicy(3, 1000)
    var transferred = policy^
    assert_equal(transferred.max_attempts, 3)
    assert_equal(transferred.delay_ms, 1000)


# --- Move-only and immovable types ---


struct Pinned(Movable where False):
    var n: Int

    def __init__(out self, n: Int):
        self.n = n


def test_move_only_types() raises:
    assert_true(conforms_to(OwnedPointer[Int], Movable))
    assert_false(conforms_to(OwnedPointer[Int], Copyable))
    # `Atomic` is move-only, not immovable.
    assert_true(conforms_to(Atomic[Int64], Movable))
    assert_false(conforms_to(Atomic[Int64], Copyable))
    var counter = Atomic[Int64](0)
    _ = counter.fetch_add(5)
    var relocated = counter^
    assert_equal(relocated.load(), 5)


def test_immovable_type() raises:
    assert_false(conforms_to(Pinned, Movable))
    assert_false(conforms_to(Pinned, Copyable))
    var pinned = Pinned(5)
    assert_equal(pinned.n, 5)


# --- Trivial lifecycle methods ---


struct MoveOnlyTrivialBits:
    var n: Int

    def __init__(out self, n: Int):
        self.n = n


def test_trivial_lifecycle_predicates() raises:
    assert_true(IsTriviallyCopyable[Int])
    assert_true(IsTriviallyMovable[Int])
    assert_true(IsTriviallyDeinitable[Int])
    # `String` disagrees across all three, so one type exercises each.
    assert_false(IsTriviallyCopyable[String])
    assert_true(IsTriviallyMovable[String])
    assert_false(IsTriviallyDeinitable[String])


def test_trivial_predicates_require_conformance() raises:
    # Each predicate is false for a type that doesn't conform to the matching
    # trait, even when the bits themselves are trivial. This is what the page's
    # "is true when `T` is `Copyable` and ..." wording covers.
    assert_false(conforms_to(MoveOnlyTrivialBits, Copyable))
    assert_false(IsTriviallyCopyable[MoveOnlyTrivialBits])
    assert_true(IsTriviallyMovable[MoveOnlyTrivialBits])
    assert_false(IsTriviallyMovable[Pinned])


def main() raises:
    test_fieldwise_and_hand_written_agree()
    test_initializer_overloads()
    test_no_initializer_type()
    test_optional_implicit_conversion()
    test_implicit_conversion_on_argument()
    test_implicit_conversion_on_return()
    test_braced_shorthand()
    test_braced_shorthand_without_implicit_conversion()
    test_copy_and_move_shorthands()
    test_copyable_constraint_in_generic_code()
    test_copy_is_a_deep_copy_of_the_binding()
    test_transfer_needs_no_declared_conformance()
    test_move_only_types()
    test_immovable_type()
    test_trivial_lifecycle_predicates()
    test_trivial_predicates_require_conformance()
