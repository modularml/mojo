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
# Tests for simple-statements.mdx code examples.
#
# Not tested (intentional errors, warning-only forms, interactive
# constructs, or style counter-examples):
#   - `x + y` unused-result example (warning only, no runtime behavior)
#   - `print("hello")` / `trigger()` side-effect placeholders (no
#     observable state to assert)
#   - `import numpy as np` (Python interop setup out of scope)
#   - `from std.math import *` (wildcard import is an anti-pattern to
#     check into source)
#   - `var x` alone (intentional error: must have type or initializer)
#   - Module-scope bare `return 42` (intentional error)
#   - Module-scope `raise Error("oops")` outside try (intentional error)
#   - `var temperature, name = 98.6, "Bob"` counter-example (style
#     guidance only; both forms compile)
#   - `x @= m` matmul augmented op (requires a struct with
#     `__imatmul__`; covered in operators tests)
from std.testing import assert_equal


# --- Stubs referenced by code examples ---


def compute() -> Int:
    return 7


def update() -> Int:
    return 0


def returns_pair() -> Tuple[Int, Int]:
    return (1, 2)


def perform_some_test() -> Bool:
    return True


def log(e: Error):
    pass


def validate(value: Int) raises -> Bool:
    if value < 0:
        raise Error("value must be non-negative")
    return True


# --- Expression statements: discarding a result ---


def test_underscore_discard() raises:
    _ = update()


# --- Assignment statements: var with inference ---


def test_var_inference() raises:
    var x = 42
    var name = "Alice"
    var result = compute()
    assert_equal(x, 42)
    assert_equal(name, "Alice")
    assert_equal(result, 7)


# --- Assignment statements: ref binding ---


def test_ref_binding_read() raises:
    var my_list = [10, 20, 30, 40]
    ref y = my_list[3]
    assert_equal(y, 40)


def test_ref_binding_write_through() raises:
    var my_list = [10, 20, 30, 40]
    ref y = my_list[3]
    y = 99
    assert_equal(my_list[3], 99)


def test_ref_binding_observe_underlying() raises:
    var my_list = [10, 20, 30, 40]
    ref y = my_list[3]
    my_list[3] = 77
    assert_equal(y, 77)


# --- Assignment statements: annotated ---


def test_annotated_assignment() raises:
    var x: Int = 42
    var name: String = "Alice"
    var values: List[Float64] = []
    assert_equal(x, 42)
    assert_equal(name, "Alice")
    assert_equal(len(values), 0)


def test_var_type_only_then_assign() raises:
    var x: Int
    x = 42
    assert_equal(x, 42)


# --- Assignment statements: comptime type alias ---


def test_comptime_type_alias() raises:
    comptime Vec3 = List[Float64]
    var position: Vec3 = [0.0, 0.0, 0.0]
    assert_equal(len(position), 3)
    assert_equal(position[0], 0.0)


# --- Multiple assignment: chained, same value ---


def test_chained_same_value() raises:
    var x = var y = var z = "Hello"
    assert_equal(x, "Hello")
    assert_equal(y, "Hello")
    assert_equal(z, "Hello")


# --- Multiple assignment: chained ref/var mix ---


def test_chained_ref_var_mix() raises:
    ref a = var b = var c = "Hello"
    assert_equal(a, "Hello")
    assert_equal(b, "Hello")
    assert_equal(c, "Hello")
    a = "World"
    # `ref a` is bound to `b`'s storage; writing through `a` mutates `b`.
    assert_equal(b, "World")
    # `c` got its own copy of "Hello" and is independent.
    assert_equal(c, "Hello")


# --- Multiple assignment: destructuring ---


def test_destructuring_bare() raises:
    var a, b = 1, 2
    assert_equal(a, 1)
    assert_equal(b, 2)


def test_destructuring_parenthesized() raises:
    var (c, d) = (1, 2)
    assert_equal(c, 1)
    assert_equal(d, 2)


def test_tuple_return_unpack() raises:
    var e, f = returns_pair()
    assert_equal(e, 1)
    assert_equal(f, 2)


# --- Simple swaps ---


def test_simple_swap() raises:
    var a, b, c = 1, 2, 3
    a, b, c = c, a, b
    assert_equal(a, 3)
    assert_equal(b, 1)
    assert_equal(c, 2)


# --- Augmented assignment ---


def test_augmented_arithmetic_int() raises:
    var x = 10
    x += 5
    assert_equal(x, 15)
    x -= 2
    assert_equal(x, 13)
    x *= 3
    assert_equal(x, 39)
    x //= 2
    assert_equal(x, 19)
    x %= 7
    assert_equal(x, 5)
    x **= 2
    assert_equal(x, 25)


def test_augmented_float_div() raises:
    var x: Float64 = 20.0
    x /= 4.0
    assert_equal(x, 5.0)


def test_augmented_bitwise() raises:
    var bits = 0b1100
    bits &= 0b1010
    assert_equal(bits, 0b1000)
    bits |= 0b0011
    assert_equal(bits, 0b1011)
    bits ^= 0b1111
    assert_equal(bits, 0b0100)
    bits <<= 2
    assert_equal(bits, 0b10000)
    bits >>= 1
    assert_equal(bits, 0b1000)


# --- pass statement ---


def not_ready():
    pass


struct Empty:
    pass


def test_pass_in_function() raises:
    not_ready()


# --- return statement ---


def greet(name: String):
    if not name:
        return
    print(t"Hello, {name}!")


def get_value() -> Int:
    return 42


def early_exit(items: List[Int], target: Int) -> Bool:
    for item in items:
        if item == target:
            return True
    return False


def test_return_void_early() raises:
    greet("")


def test_return_value() raises:
    assert_equal(get_value(), 42)


def test_early_exit_found() raises:
    var items: List = [1, 2, 3, 4, 5]
    assert_equal(early_exit(items, 3), True)


def test_early_exit_not_found() raises:
    var items: List = [1, 2, 3, 4, 5]
    assert_equal(early_exit(items, 99), False)


# --- raise statement ---


def test_validate_happy_path() raises:
    assert_equal(validate(5), True)


def test_validate_raises() raises:
    var raised = False
    try:
        _ = validate(-1)
    except e:
        raised = True
    assert_equal(raised, True)


def mitigate_risk():
    try:
        if not perform_some_test():
            raise Error("test failed")
    except e:
        log(e)


def test_mitigate_risk_no_raise() raises:
    # `perform_some_test` stub returns True, so no error is raised
    # inside `mitigate_risk`. The function itself does not propagate.
    mitigate_risk()


def reraise_through(value: Int) raises:
    try:
        _ = validate(value)
    except e:
        raise  # Bare raise re-raises the current error.


def test_bare_reraise() raises:
    var raised = False
    try:
        reraise_through(-1)
    except e:
        raised = True
    assert_equal(raised, True)


# --- break / continue ---


def collect_loop_values() -> List[Int]:
    var out = List[Int]()
    for x in range(10):
        if x == 5:
            break
        if x % 2 == 0:
            continue
        out.append(x)
    return out^


def test_break_continue() raises:
    var result = collect_loop_values()
    assert_equal(len(result), 2)
    assert_equal(result[0], 1)
    assert_equal(result[1], 3)


# --- comptime declarations ---


comptime SIZE = 256
comptime MAX = SIZE * 2


def test_comptime_constants() raises:
    assert_equal(SIZE, 256)
    assert_equal(MAX, 512)


# Trait composition alias and a trait with an associated comptime type.
# Declaring these at module scope is the test: the file compiling is the
# verification that the syntax and the semantic check pass.
comptime Permissive = ImplicitlyCopyable & Deinitable


trait SimpleTrait(Writable):
    comptime Element = Permissive


def test_comptime_trait_declared() raises:
    # No runtime assertion needed; module-scope declarations above are
    # what's under test.
    pass


def main() raises:
    test_underscore_discard()
    test_var_inference()
    test_ref_binding_read()
    test_ref_binding_write_through()
    test_ref_binding_observe_underlying()
    test_annotated_assignment()
    test_var_type_only_then_assign()
    test_comptime_type_alias()
    test_chained_same_value()
    test_chained_ref_var_mix()
    test_destructuring_bare()
    test_destructuring_parenthesized()
    test_tuple_return_unpack()
    test_simple_swap()
    test_augmented_arithmetic_int()
    test_augmented_float_div()
    test_augmented_bitwise()
    test_pass_in_function()
    test_return_void_early()
    test_return_value()
    test_early_exit_found()
    test_early_exit_not_found()
    test_validate_happy_path()
    test_validate_raises()
    test_mitigate_risk_no_raise()
    test_bare_reraise()
    test_break_continue()
    test_comptime_constants()
    test_comptime_trait_declared()
