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
# Tests for lambda-expressions.mdx code examples.
#
# Not tested (rejected forms have no runnable behavior to assert). Each is
# documented in the page as a paraphrased error gist; the compiler tests that
# pin the exact diagnostics live in
# KGEN/test/mojo-parser/exprs/expressions_errors.mojo:
#   - `lambda x, y: x + y` (unparenthesized arguments)
#   - `lambda (x) {} -> Int: x` (untyped argument)
#   - `lambda: 5` and `lambda (x: Int) {}: x + 1` (non-`None` body under an
#     elided return type)
#   - `lambda (x: Int) {} -> Int: x + z` (free variable under explicit `{}`)
#   - `lambda (x: Int) {imm list} -> None: list.append(x)` (mutation through an
#     `imm` capture)
#   - a lambda closure in a `comptime` initializer, a type parameter, or a
#     default parameter
#   - a lambda closure, capturing or parametric, assigned where a `thin`
#     function pointer is required
from std.testing import assert_equal, assert_raises

# --- Helpers used across the thin-pointer and parameter tests ---


def apply_thin(f: def(x: Int) thin -> Int, arg: Int) -> Int:
    return f(arg)


def apply_raising(f: def(x: Int) raises thin -> Int, arg: Int) raises -> Int:
    return f(arg)


def call_with[f: def(x: Int) thin -> Int]() -> Int:
    return f(10)


def call_with_default[
    f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + 5
]() -> Int:
    return f(10)


def make_adder() -> def(x: Int) thin -> Int:
    return lambda (x: Int) -> Int: x + 10


def with_default_arg(
    x: Int, cb: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + 1
) -> Int:
    return cb(x)


def take_two(*fs: def(x: Int) thin -> Int) -> Int:
    return fs[0](1) + fs[1](1)


def total_as_ints[T: Intable & Copyable](args: List[T]) -> Int:
    var to_int: def(v: T) thin -> Int = lambda (v: T) -> Int: Int(v)
    var total = 0
    for ref a in args:
        total += to_int(a)
    return total


# `T` is inferred from the lambda passed for `f`, so both a thin lambda
# and a lambda closure bind here.
def call_inferred[T: def(x: Int) -> Int, //](f: T, arg: Int) -> Int:
    return f(arg)


# `R` appears only inside the function-trait bound on `F`, so it is solved
# from the lambda's return type.
def call_for_result[R: AnyType, F: def(x: Int) -> R, //](f: F, arg: Int) -> R:
    return f(arg)


# --- Syntax: the fully explicit form ---


def test_explicit_form() raises:
    var f = lambda (x: Int) {} -> Int: x + 1
    assert_equal(f(4), 5)


def test_conventions() raises:
    var read_arg = lambda (x: Int) {} -> Int: x + 1
    # AKA: var read_arg = lambda (imm x: Int) {} -> Int: x + 1
    var own_arg = lambda (var x: Int) {} -> Int: x + 1
    var mut_arg = lambda (mut x: Int) {}: x.__iadd__(1)

    var list: List[Int] = [1, 2, 3]
    mut_arg(list[0])
    assert_equal(list, [2, 2, 3])
    assert_equal(read_arg(1), 2)
    assert_equal(own_arg(1), 2)


def test_parameterized() raises:
    var inc = lambda [T: Intable](x: T) -> Int: Int(x) + 1
    assert_equal(inc[Float64](3.5), 4)
    assert_equal(inc[Int](3), 4)
    assert_equal(inc(3.5), 4)


def test_concrete() raises:
    var hello = lambda (x: String) {} -> String: "Hello, " + x
    assert_equal(hello("World"), "Hello, World")
    assert_equal(hello("Mojo"), "Hello, Mojo")


def test_no_arguments() raises:
    var no_args = lambda -> Int: 42
    assert_equal(no_args(), 42)


def test_base_capture() raises:
    var z = 10
    var f = lambda (x: Int) -> Int: x + z  # `z` is captured
    assert_equal(f(5), 15)


def test_mut_capture() raises:
    var list: List[Int] = [1, 2, 3]
    var f = lambda (x: Int) {mut list}: list.append(x)
    f(10)
    assert_equal(list, [1, 2, 3, 10])


def test_parameter_capture() raises:
    # N is lambda-owned, bound at each call
    var f = lambda [N: Int](x: Int) {} -> Int: x + N
    assert_equal(f[5](3), 8)


def test_thin_parameter_enclosing_scope() raises:
    assert_equal(total_as_ints([1.5, 2.5, 3.9]), 6)


# --- Elision: the capture list may be omitted ---


def test_elided_capture_list() raises:
    var f = lambda (x: Int) -> Int: x * 2
    assert_equal(f(4), 8)


# --- Elision: an omitted return type defaults to `None` ---


def test_elided_return_type() raises:
    var list: List[Int] = [1]
    var push = lambda (x: Int) {mut list}: list.append(x)
    push(2)
    push(3)
    assert_equal(len(list), 3)
    assert_equal(list[2], 3)


# --- Elision: the bare form takes no arguments ---


def test_bare_lambda() raises:
    var count = 0
    var bump = lambda {mut count}: count.__iadd__(1)
    bump()
    bump()
    assert_equal(count, 2)


# --- Captures: an omitted capture list captures free variables by `imm` ---


def test_omitted_capture_list_is_imm() raises:
    var z = 3
    var f = lambda (x: Int) -> Int: x + z
    assert_equal(f(1), 4)
    # An `imm` capture holds a reference, so the lambda observes the
    # current value at each call, not a snapshot.
    z = 30
    assert_equal(f(1), 31)


# --- Captures: a named capture ---


def test_named_capture() raises:
    var z = 3
    var f = lambda (x: Int) {imm z} -> Int: x + z
    assert_equal(f(1), 4)


# --- Captures: a `mut` capture-all mutates the outer value ---


def test_mut_capture_all() raises:
    var total = 0
    var add = lambda (x: Int) {mut}: total.__iadd__(x)
    add(5)
    add(7)
    assert_equal(total, 12)


# --- Captures: a `var` capture takes an owned copy (a snapshot) ---


def test_var_capture_snapshot() raises:
    var z = 3
    var f = lambda (x: Int) {var z} -> Int: x + z
    assert_equal(f(1), 4)
    z = 30
    assert_equal(z, 30)
    # The copy was taken when the lambda was created.
    assert_equal(f(1), 4)


# --- Captures: mixed conventions in one list ---


def test_mixed_conventions() raises:
    var counts: List = [0]
    var step = 2
    var bump = lambda (x: Int) {mut counts, imm step}: counts.append(x + step)
    bump(1)
    assert_equal(len(counts), 2)
    assert_equal(counts[1], 3)


# --- Compile-time: a thin lambda binds to a `comptime` ---

comptime INC = lambda (x: Int) -> Int: x + 1


struct Ops:
    comptime inc = lambda (x: Int) {} -> Int: x + 1


def test_comptime_bind() raises:
    comptime local_inc = lambda (x: Int) {} -> Int: x + 1
    assert_equal(local_inc(1), 2)
    assert_equal(INC(1), 2)
    assert_equal(Ops.inc(3), 4)


# --- Compile-time: applying a lambda at comptime folds the result ---


def test_comptime_apply() raises:
    comptime whole = (lambda (x: Int) {} -> Int: x * 2)(21)
    assert_equal(whole, 42)


# --- Compile-time: a comptime lambda is one promoted function ---


def test_comptime_reference() raises:
    comptime dbl = lambda (x: Int) {} -> Int: x * 2
    comptime compose = lambda (x: Int) {} -> Int: dbl(x) + 1
    assert_equal(compose(4), 9)


# --- Compile-time: an enclosing parameter folds in and keeps the lambda thin ---

comptime ADD_GEN[N: Int] = lambda (x: Int) {} -> Int: x + N
comptime ADD_LAM = lambda [N: Int](x: Int) {} -> Int: x + N


def add_local[N: Int](v: Int) -> Int:
    comptime f = lambda (x: Int) {} -> Int: x + N
    return f(v)


def test_comptime_parameters() raises:
    assert_equal(ADD_GEN[7](3), 10)
    assert_equal(ADD_LAM[7](3), 10)
    assert_equal(add_local[7](3), 10)


# --- Compile-time: a thin lambda binds to a `thin` function-typed parameter ---


def call_with_enclosing[N: Int]() -> Int:
    return call_with[lambda (x: Int) {} -> Int: x + N]()


def test_thin_fn_typed_parameter() raises:
    assert_equal(call_with[lambda (x: Int) {} -> Int: x * 2](), 20)
    # The capture list may be elided and the lambda is still thin.
    assert_equal(call_with[lambda (x: Int) -> Int: x * 4](), 40)
    assert_equal(call_with_default(), 15)
    assert_equal(call_with_enclosing[7](), 17)


# --- Compile-time: the parameter can be on a struct, including its default ---


struct Scaled[f: def(x: Int) thin -> Int = lambda (x: Int) {} -> Int: x * 3]:
    var v: Int

    def __init__(out self, arg: Int):
        self.v = Self.f(arg)


def test_struct_fn_parameter() raises:
    assert_equal(Scaled[lambda (x: Int) {} -> Int: x + 1](4).v, 5)
    assert_equal(Scaled(4).v, 12)


# --- Runtime: a thin lambda works as a `thin` function pointer ---


struct Holder:
    var cb: def(x: Int) thin -> Int

    def __init__(out self):
        self.cb = lambda (x: Int) -> Int: x * 2


def thin_with_enclosing_param[N: Int]() -> Int:
    var f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + N
    return f(1)


struct SelfParamDecay[P: Int]:
    var v: Int

    def __init__(out self):
        var f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + Self.P
        self.v = f(1)


def test_thin_pointer_typed_var() raises:
    var t: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + 1
    assert_equal(t(1), 2)
    # The variable holds a function value, so it is rebindable.
    t = lambda (x: Int) -> Int: x * 3
    assert_equal(t(2), 6)
    # An explicit `{}` is thin too, and works the same way.
    t = lambda (x: Int) {} -> Int: x + 7
    assert_equal(t(2), 9)


def test_thin_pointer_untyped_var() raises:
    var u = lambda (x: Int) -> Int: x + 100
    assert_equal(u(1), 101)
    u = lambda (x: Int) -> Int: x + 1000
    assert_equal(u(1), 1001)


def test_thin_pointer_return_value() raises:
    assert_equal(make_adder()(5), 15)


def test_thin_pointer_argument() raises:
    assert_equal(apply_thin(lambda (x: Int) -> Int: x - 1, 4), 3)


def test_thin_pointer_default_argument() raises:
    assert_equal(with_default_arg(4), 5)
    assert_equal(with_default_arg(4, lambda (x: Int) -> Int: x * 10), 40)


def test_thin_pointer_struct_field() raises:
    var h = Holder()
    assert_equal(h.cb(3), 6)
    h.cb = lambda (x: Int) -> Int: x + 30
    assert_equal(h.cb(3), 33)


def test_thin_pointer_enclosing_parameter() raises:
    assert_equal(thin_with_enclosing_param[5](), 6)
    assert_equal(SelfParamDecay[7]().v, 8)


def test_thin_pointer_variadic_pack() raises:
    assert_equal(
        take_two(lambda (x: Int) -> Int: x + 1, lambda (x: Int) -> Int: x * 2),
        4,
    )


# --- Effects: a raising lambda fills a `raises thin` parameter ---


def test_raising_lambda() raises:
    assert_equal(apply_raising(lambda (x: Int) raises -> Int: x + 1, 2), 3)


def checked(x: Int) raises -> Int:
    if x < 0:
        raise Error("negative")
    return x


def test_raising_lambda_actually_raises() raises:
    var f = lambda (x: Int) raises -> Int: checked(x) + 1
    assert_equal(f(1), 2)
    with assert_raises(contains="negative"):
        _ = f(-1)


# --- C ABI: a thin lambda fills a C-ABI function pointer ---


def test_c_abi_callback() raises:
    # `abi("C")` goes on the lambda; the variable's type is inferred.
    var fp = lambda (a: Int32, b: Int32) abi("C") -> Int32: a + b
    assert_equal(fp(1, 2), 3)

    # With explicit typing, the annotation must carry `abi("C")` as well.
    var typed: def(Int32, Int32) thin abi("C") -> Int32 = (
        lambda (a: Int32, b: Int32) abi("C") -> Int32: a * b
    )
    assert_equal(typed(3, 4), 12)


# --- Parameters: a lambda can declare its own compile-time parameters ---


def test_parametric_lambda() raises:
    var pf = lambda [N: Int](x: Int) {} -> Int: x + N
    assert_equal(pf[5](3), 8)
    assert_equal(pf[10](3), 13)


def test_parametric_lambda_nested() raises:
    var pn = lambda [N: Int](x: Int) {} -> Int: (
        lambda (y: Int) {} -> Int: y + N
    )(x)
    assert_equal(pn[10](3), 13)


# --- Arguments: variadic forms ---


def test_variadic_args() raises:
    var va = lambda (*args: Int) {} -> Int: len(args)
    assert_equal(va(10, 20, 30), 3)


def test_variadic_kwargs() raises:
    var kw = lambda (var **kwargs: Int) {} -> Int: len(kwargs)
    assert_equal(kw(a=1, b=2), 2)


def test_variadic_both() raises:
    var both = lambda (*args: Int, var **kwargs: Int) {} -> Int: len(
        args
    ) + len(kwargs)
    assert_equal(both(1, 2, a=3), 3)


def test_variadic_thin_pointer() raises:
    var vd: def(* args: Int) thin -> Int = lambda (*args: Int) -> Int: len(args)
    assert_equal(vd(1, 2, 3), 3)


# --- Arguments: argument conventions ---


def test_argument_conventions() raises:
    var imm_arg = lambda (imm x: Int) {} -> Int: x + 1
    assert_equal(imm_arg(1), 2)
    var var_arg = lambda (var x: Int) {} -> Int: x + 1
    assert_equal(var_arg(1), 2)
    var mut_arg = lambda (mut x: Int) {}: x.__iadd__(1)
    var v = 1
    mut_arg(v)
    assert_equal(v, 2)


# --- Nesting: a lambda body can contain another lambda ---


def test_nested_lambda() raises:
    var f = lambda (x: Int) {} -> Int: (lambda (y: Int) {imm x} -> Int: y + x)(
        3
    )
    assert_equal(f(6), 9)


# --- Inference: a lambda binds an inferred function-typed parameter ---


def test_inferred_parameter_type() raises:
    assert_equal(call_inferred(lambda (x: Int) {} -> Int: x + 1, 4), 5)
    var z = 10
    # Inference works for a lambda closure too.
    assert_equal(call_inferred(lambda (x: Int) {mut} -> Int: x + z, 5), 15)
    assert_equal(call_for_result(lambda (x: Int) {} -> Int: x * 2, 6), 12)


def main() raises:
    test_explicit_form()
    test_conventions()
    test_parameterized()
    test_no_arguments()
    test_concrete()
    test_base_capture()
    test_mut_capture()
    test_parameter_capture()
    test_thin_parameter_enclosing_scope()
    test_elided_capture_list()
    test_elided_return_type()
    test_bare_lambda()
    test_omitted_capture_list_is_imm()
    test_named_capture()
    test_mut_capture_all()
    test_var_capture_snapshot()
    test_mixed_conventions()
    test_comptime_bind()
    test_comptime_apply()
    test_comptime_reference()
    test_comptime_parameters()
    test_thin_fn_typed_parameter()
    test_struct_fn_parameter()
    test_thin_pointer_typed_var()
    test_thin_pointer_untyped_var()
    test_thin_pointer_return_value()
    test_thin_pointer_argument()
    test_thin_pointer_default_argument()
    test_thin_pointer_struct_field()
    test_thin_pointer_enclosing_parameter()
    test_thin_pointer_variadic_pack()
    test_c_abi_callback()
    test_raising_lambda()
    test_raising_lambda_actually_raises()
    test_parametric_lambda()
    test_parametric_lambda_nested()
    test_variadic_args()
    test_variadic_kwargs()
    test_variadic_both()
    test_variadic_thin_pointer()
    test_argument_conventions()
    test_nested_lambda()
    test_inferred_parameter_type()
