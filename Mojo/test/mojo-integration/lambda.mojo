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
# RUN: %mojo %s | FileCheck %s

# A lambda desugars to an anonymous def, constructed at emit time. A thin
# (capture-free, non-parametric) lambda is the promoted function's value,
# exactly as a `def` referenced by name is -- in a parameter context (a
# `comptime` initializer or a `thin` fn-typed parameter) and in runtime slots,
# where it decays to a `thin` fn-pointer. A capturing lambda constructs a
# closure instance.


def callInt[T: def(x: Int) -> Int, //](f: T, arg: Int):
    print(f(arg))


# The return type `R` appears only inside the function-trait bound on `F`, so it
# is inferred from the lambda passed for `f` (closure type-parameter inference).
def callRet[R: AnyType, F: def(x: Int) -> R, //](f: F, arg: Int) -> R:
    return f(arg)


def addBase(base: Int):
    # A lambda capturing the enclosing function's parameter `base`. A
    # register-passable `Int` argument can only be captured by `imm`.
    callInt(lambda (x: Int) {imm base} -> Int: x + base, 5)


def addEnclosingParam[N: Int]() -> Int:
    # A lambda referencing the enclosing parametric function's compile-time
    # parameter `N`. A parameter is a compile-time value, so it is referenced
    # directly rather than through the (empty) capture list.
    return (lambda (x: Int) {} -> Int: x + N)(3)


def test_construct_and_infer():
    # Non-capturing lambda passed to a parametric HOF.
    callInt(lambda (x: Int) {} -> Int: x + 1, 4)
    # CHECK: 5

    # Capturing lambda: the default '{mut}' convention captures `z`.
    var z = 10
    callInt(lambda (x: Int) {mut} -> Int: x + z, 5)
    # CHECK: 15

    # `R` is inferred from the lambda's return type: here `Int`.
    print(callRet(lambda (x: Int) {} -> Int: x * 2, 6))
    # CHECK: 12

    # Inference also works for a capturing lambda.
    print(callRet(lambda (x: Int) {mut} -> Int: x + z, 7))
    # CHECK: 17

    # Capturing an enclosing function's parameter (rather than a local).
    addBase(100)
    # CHECK: 105

    # Nested lambda: the outer lambda's body invokes an inner lambda, which
    # captures the outer lambda's argument `x`.
    callInt(
        lambda (x: Int) {} -> Int: (lambda (y: Int) {imm x} -> Int: y + x)(3), 6
    )
    # CHECK: 9

    # Parametric lambda: the compile-time parameter `N` is bound at the call
    # site with `f[K]`.
    var pf = lambda [N: Int](x: Int) {} -> Int: x + N
    print(pf[5](3))
    # CHECK: 8

    # Nested lambda where the outer is parametric: the inner lambda references
    # the outer's compile-time parameter `N` directly.
    var pn = lambda [N: Int](x: Int) {} -> Int: (
        lambda (y: Int) {} -> Int: y + N
    )(x)
    print(pn[10](3))
    # CHECK: 13

    # A lambda referencing the compile-time parameter of an enclosing
    # parametric function.
    print(addEnclosingParam[7]())
    # CHECK: 10

    # Variadic arguments: `*args` positionally, `**kwargs` packed into an
    # OwnedKwargsDict. All three lambdas are thin, so all three are promoted
    # plain functions called through their fn value.
    var va = lambda (*args: Int) {} -> Int: len(args)
    print(va(10, 20, 30))
    # CHECK: 3
    var kwl = lambda (var **kwargs: Int) {} -> Int: len(kwargs)
    print(kwl(a=1, b=2))
    # CHECK: 2
    var both = lambda (*args: Int, var **kwargs: Int) {} -> Int: len(
        args
    ) + len(kwargs)
    print(both(1, 2, a=3))
    # CHECK: 3


# ===----------------------------------------------------------------------=== #
# Binding to a `comptime`
# ===----------------------------------------------------------------------=== #


struct Ops:
    comptime inc = lambda (x: Int) {} -> Int: x + 1


def test_comptime_bind():
    # A thin lambda binds to a `comptime` just like `comptime f = some_def`.
    # Also works at struct scope (`Ops.inc`).
    comptime comptime_inc = lambda (x: Int) {} -> Int: x + 1
    print(comptime_inc(1))
    # CHECK: 2
    print(Ops.inc(3))
    # CHECK: 4

    # A lambda nested inside a comptime-bound lambda's body also works.
    comptime nested = lambda (x: Int) {} -> Int: (
        lambda (y: Int) {} -> Int: y * 2
    )(x) + 1
    print(nested(4))
    # CHECK: 9


def test_comptime_apply():
    # Applying a lambda *at* comptime folds its result to a constant.
    comptime whole = (lambda (x: Int) {} -> Int: x * 2)(21)
    print(whole)
    # CHECK: 42


def test_comptime_reference():
    # Using a comptime lambda from another lambda's body is a *reference* to its
    # one promoted function (like naming a def), not an inlined copy.
    comptime dbl = lambda (x: Int) {} -> Int: x * 2
    comptime compose = lambda (x: Int) {} -> Int: dbl(x) + 1
    print(compose(4))
    # CHECK: 9


# An enclosing parameter is a comptime value, not a runtime capture, so it folds
# in. It can be the alias's own parameter (`add_gen`), the lambda's own
# parameter (`add_lam`), or an enclosing function's (`add_def`).
comptime add_gen[N: Int] = lambda (x: Int) {} -> Int: x + N
comptime add_lam = lambda [N: Int](x: Int) {} -> Int: x + N


def add_def[N: Int](v: Int) -> Int:
    comptime f = lambda (x: Int) {} -> Int: x + N
    return f(v)


def test_comptime_parameters():
    # All three placements, same inputs `[7](3)` -> 10.
    print(add_gen[7](3))
    # CHECK: 10
    print(add_lam[7](3))
    # CHECK: 10
    print(add_def[7](3))
    # CHECK: 10


# ===----------------------------------------------------------------------=== #
# As a function-typed parameter
# ===----------------------------------------------------------------------=== #


# A thin lambda folds to a function literal, so it binds to a `thin` fn-typed
# parameter (a concrete type) at a call site or as a default.
def call_with[f: def(x: Int) thin -> Int]() -> Int:
    return f(10)


def call_with_default[
    f: def(x: Int) thin -> Int = lambda (x: Int) {} -> Int: x + 5
]() -> Int:
    return f(10)


# The inline lambda argument may itself reference an enclosing comptime
# parameter (`N`); the fold and the parameter binding compose.
def call_with_enclosing[N: Int]() -> Int:
    return call_with[lambda (x: Int) {} -> Int: x + N]()


def test_fn_typed_parameter():
    print(call_with[lambda (x: Int) {} -> Int: x * 2]())
    # CHECK: 20
    print(call_with_default())
    # CHECK: 15
    print(call_with_enclosing[7]())
    # CHECK: 17

    # The capture list may be elided: capturing nothing, the lambda is thin and
    # still binds to the thin fn-typed parameter.
    print(call_with[lambda (x: Int) -> Int: x * 4]())
    # CHECK: 40


# The parameter can be on a struct rather than a function -- including as the
# struct parameter's default value.
struct Scaled[f: def(x: Int) thin -> Int = lambda (x: Int) {} -> Int: x * 3]:
    var v: Int

    def __init__(out self, arg: Int):
        self.v = Self.f(arg)


def test_struct_fn_parameter():
    print(Scaled[lambda (x: Int) {} -> Int: x + 1](4).v)
    # CHECK: 5
    print(Scaled(4).v)
    # CHECK: 12


# ===----------------------------------------------------------------------=== #
# Decay to a thin fn-pointer in runtime slots
# ===----------------------------------------------------------------------=== #


def applyThin(f: def(x: Int) thin -> Int, arg: Int) -> Int:
    return f(arg)


def mkAdder() -> def(x: Int) thin -> Int:
    return lambda (x: Int) -> Int: x + 10


def applyRaises(f: def(x: Int) raises thin -> Int, arg: Int) raises -> Int:
    return f(arg)


def withDefault(
    x: Int, cb: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + 1
) -> Int:
    return cb(x)


struct Holder:
    var cb: def(x: Int) thin -> Int

    def __init__(out self):
        self.cb = lambda (x: Int) -> Int: x * 2


def paramDecay[N: Int]() -> Int:
    var f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + N
    return f(1)


# The enclosing parameter may be a struct's, reached via `Self`.
struct PSDecay[P: Int]:
    var v: Int

    def __init__(out self):
        var f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + Self.P
        self.v = f(1)


def take(*fs: def(x: Int) thin -> Int) -> Int:
    return fs[0](1) + fs[1](1)


# Overload set with both a function-trait-inference candidate and a runtime
# `thin` fn-pointer candidate: a lambda ranks exactly as a `def` name does and
# picks the thin runtime overload.
def pick[T: def(x: Int) -> Int, //](f: T, arg: Int) -> Int:
    return f(arg) + 100


def pick(f: def(x: Int) thin -> Int, arg: Int) -> Int:
    return f(arg) + 200


def test_thin_decay() raises:
    # A thin lambda is the promoted function's value, as a `def` name is, so it
    # decays to a `thin` fn-pointer in a typed var...
    var t: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + 1
    print(t(1))
    # CHECK: 2

    # ...which is rebindable, like `var t = some_def`.
    t = lambda (x: Int) -> Int: x * 3
    print(t(2))
    # CHECK: 6

    # A written `{}` is explicitly thin (unlike a written `{imm}`/`{mut}`,
    # which reifies a closure instance) and decays the same way.
    t = lambda (x: Int) {} -> Int: x + 7
    print(t(2))
    # CHECK: 9

    # In a return slot...
    print(mkAdder()(5))
    # CHECK: 15

    # ...in an argument slot...
    print(applyThin(lambda (x: Int) -> Int: x - 1, 4))
    # CHECK: 3

    # ...as a defaulted runtime-argument value...
    print(withDefault(4))
    # CHECK: 5
    print(withDefault(4, lambda (x: Int) -> Int: x * 10))
    # CHECK: 40

    # ...into a struct field, both at construction and by reassignment...
    var h = Holder()
    print(h.cb(3))
    # CHECK: 6
    h.cb = lambda (x: Int) -> Int: x + 30
    print(h.cb(3))
    # CHECK: 33

    # ...and for a raising thin lambda into a `raises thin` slot.
    print(applyRaises(lambda (x: Int) raises -> Int: x + 1, 2))
    # CHECK: 3

    # An untyped var binds the function value too: callable and rebindable.
    var u = lambda (x: Int) -> Int: x + 100
    print(u(1))
    # CHECK: 101
    u = lambda (x: Int) -> Int: x + 1000
    print(u(1))
    # CHECK: 1001

    # A thin lambda referencing an enclosing parameter still decays: the
    # reference is bound at promotion, like a stateless nested def using `N`.
    print(paramDecay[5]())
    # CHECK: 6

    # The same holds for an enclosing STRUCT parameter (`Self.P`).
    print(PSDecay[7]().v)
    # CHECK: 8

    # A variadic lambda decays as well: the pack's implicit origin parameters
    # bind at the reference, as they do for a named variadic `def`.
    var vd: def(* args: Int) thin -> Int = lambda (*args: Int) -> Int: len(args)
    print(vd(1, 2, 3))
    # CHECK: 3

    # Decay also feeds a variadic `thin` fn-pointer pack -- a distinct
    # emission path from a plain argument.
    print(take(lambda (x: Int) -> Int: x + 1, lambda (x: Int) -> Int: x * 2))
    # CHECK: 4

    # Overload ranking matches a def name: the thin runtime overload wins.
    print(pick(lambda (x: Int) -> Int: x, 1))
    # CHECK: 201


def main() raises:
    test_construct_and_infer()
    test_comptime_bind()
    test_comptime_apply()
    test_comptime_reference()
    test_comptime_parameters()
    test_fn_typed_parameter()
    test_struct_fn_parameter()
    test_thin_decay()
