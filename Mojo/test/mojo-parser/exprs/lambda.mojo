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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# A lambda desugars to a synthetic anonymous def constructed at emit time: a
# thin (capture-free, non-parametric) lambda promotes to a plain function whose
# value the use site binds like a `def` referenced by name, while a capturing
# or parametric lambda constructs a closure instance. These tests check the
# constructed IR; lambda.mojo (mojo-integration) checks that it also executes.

# COM: Non-capturing lambda: empty capture list, explicit return type. Thin, so
# COM: it promotes to a plain function (no storage struct, no closure instance)
# COM: and the var binds its function value -- as `var f = some_def` does.

# The var holds the promoted function's value, not a closure instance (the
# NOT precedes the matches: storage structs print before the enclosing fn).
# CHECK-NOT: @"{{.*}}`lambda_0::__storage"
# CHECK: kgen.create_closure[{{.*}}("x": !Int) -> !{{.*}}Int{{[0-9]*}}>: @{{.*}}::@"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"]()

# The synthetic def is named `lambda_<n> and promoted to module scope; its body
# (x + 1) is a plain thin function's -- no capture machinery.
# CHECK: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"(%x: !Int) -> !{{.*}}Int{{[0-9]*}} attributes {{{.*}}sourceName = "`lambda_0"{{.*}}synthetic}
# CHECK: kgen.param.constant{{.*}}<{{{.*}}1}>
# CHECK: lit.call tail @{{.*}}::@"__add__{{.*}}"{{.*}}(%x, %{{.*}})
# CHECK: lit.return


def withNoCapture():
    var f = lambda (x: Int) {} -> Int: x + 1


# // -----

# COM: Default-all capture by mut: `{mut}` captures every used outer var by mut.

# Nested storage methods print inside the struct (before the enclosing fn's
# init call), so body checks are CHECK-DAG.
# CHECK-DAG: lit.struct.decl @"withCapturingMut()::`lambda_0::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {{{.*}}synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK-DAG: lit.fn @"`lambda_0(::SIMD[DType.int, 1]){{.*}}"{{.*}} capturing -> {{.*}} attributes {{{.*}}sourceName = "`lambda_0"{{.*}}synthetic}
# CHECK-DAG: lit.ref.struct.ger %{{.*}}[z]
# CHECK-DAG: lit.call tail @{{.*}}::@"__add__{{.*}}"{{.*}}(%x, %{{.*}})
# CHECK: lit.call @{{.*}}::@"withCapturingMut()::`lambda_0::__storage"::@"__init__


def withCapturingMut():
    var z = 3
    var f = lambda (x: Int) {mut} -> Int: x + z


# // -----

# COM: Named capture by imm (a bare `{z}` is equivalent): captured by immutable ref.

# CHECK: lit.struct.decl @"withCapturingRead()::`lambda_0::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {{{.*}}synthetic}
# CHECK: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>


def withCapturingRead():
    var z = 3
    var f = lambda (x: Int) {imm z} -> Int: x + z


# // -----

# COM: Named capture by var: captured by value (owned), so the storage field is the
# COM: value itself, not a reference.

# CHECK: lit.struct.decl @"withCapturingVar()::`lambda_0::__storage"({{.*}}) register_passable_trivial attributes {{{.*}}synthetic}
# CHECK: lit.struct.field z : !Int{{[0-9]*}}


def withCapturingVar():
    var z = 3
    var f = lambda (x: Int) {var z} -> Int: x + z


# // -----

# COM: Multiple named captures with mixed conventions: z by imm, w by mut.

# CHECK: lit.struct.decl @"withCapturingMixed()::`lambda_0::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {{{.*}}synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK-DAG: lit.struct.field w : !lit.ref<{{.*}}, mut {{.*}}>
# The lambda instantiates the scope-qualified storage, capturing `z` and `w`.
# CHECK: lit.call @{{.*}}::@"withCapturingMixed()::`lambda_0::__storage"::@"__init__


def withCapturingMixed():
    var z = 3
    var w = 4
    var f = lambda (x: Int) {imm z, mut w} -> Int: x + z + w


# // -----

# CHECK: lit.struct.decl @"withCapturingOverride()::`lambda_0::__storage"<{{.*}}>({{.*}}) attributes {{{.*}}synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK-DAG: lit.struct.field w : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK: lit.call @{{.*}}List{{.*}}append


def withCapturingOverride():
    var z = List[Int]()
    var w = 4
    var f = lambda (x: Int) {imm, mut z} -> None: z.append(x + w)


# // -----

# COM: Parameter list: `[N: Int]` becomes a closure parameter (mangled into the symbol
# COM: as `[::SIMD[DType.int, 1]]` and declared as `<N: ...>`).

# CHECK: lit.struct.decl @"withParameter()::`lambda_0::__storage"
# CHECK: lit.fn @"`lambda_0[::SIMD[DType.int, 1]](::SIMD[DType.int, 1]){{.*}}"<N: !Int{{[0-9]*}}>{{.*}} capturing -> {{.*}}


def withParameter():
    var f = lambda [N: Int](x: Int) {} -> Int: x + N


# // -----

# COM: Effects: `raises` (after the argument list, before the capture list) makes the
# COM: promoted function throwing, with the throws ABI (byref error + bool return).

# CHECK: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"{{.*}}byref_error{{.*}} throws -> {{.*}}


def withEffect():
    var f = lambda (x: Int) raises {} -> Int: x + 1


# // -----

# CHECK: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"{{.*}}(%x: !Int{{[0-9]*}}) -> {{.*}}


def withReadArg():
    var f = lambda (imm x: Int) {} -> Int: x + 1


# // -----

# CHECK: lit.fn @"{{.*}}`lambda_0(
# CHECK-SAME: %x: !lit.ref<{{.*}}> mut) -> {{.*}}


def withMutArg():
    var f = lambda (mut x: Int) {} -> Int: x


# // -----

# CHECK: lit.fn @"{{.*}}`lambda_0(
# CHECK-SAME: %x: !lit.ref<{{.*}}> owned_in_mem) -> {{.*}}


def withVarArg():
    var f = lambda (var x: Int) {} -> Int: x + 1


# // -----

# CHECK: lit.struct.decl @"withRefArg()::`lambda_0::__storage"
# CHECK: , %x: !lit.ref<{{.*}}> ref) capturing -> {{.*}}


def withRefArg():
    var f = lambda (ref x: Int) {} -> Int: x + 1


# // -----

# COM: Variadic arguments: `*args` is a positional pack; `**kwargs` packs into
# COM: an `OwnedKwargsDict`. All three are thin -- a pack's implicit origin
# COM: parameters bind at the reference, as for a named variadic `def` -- so all
# COM: three promote to plain functions.

# No storage struct for any of them, and each promoted fn takes the pack
# directly (a closure instance would take its storage first and be `capturing`).
# CHECK-NOT: @"withVariadics()::`lambda_0::__storage"
# CHECK-NOT: @"withVariadics()::`lambda_1::__storage"
# CHECK-NOT: @"withVariadics()::`lambda_2::__storage"
# CHECK-DAG: lit.fn @"{{.*}}`lambda_0[{{.*}}](::SIMD[DType.int, 1]*){{.*}}"{{.*}}vararg) -> !{{.*}}Int{{[0-9]*}}
# CHECK-DAG: lit.fn @"{{.*}}`lambda_1(kwargs:::SIMD[DType.int, 1]**){{.*}}"{{.*}}vararg) -> !{{.*}}Int{{[0-9]*}}
# CHECK-DAG: lit.fn @"{{.*}}`lambda_2[{{.*}}](::SIMD[DType.int, 1]*,kwargs:::SIMD[DType.int, 1]**){{.*}}"{{.*}}vararg) -> !{{.*}}Int{{[0-9]*}}


def withVariadics():
    var v = lambda (*args: Int) {} -> Int: 0
    var kw = lambda (var **kwargs: Int) {} -> Int: 0
    var both = lambda (*args: Int, var **kwargs: Int) {} -> Int: 0


# // -----

# COM: Everything at once: parameter + argument convention (`var`) + effects + capture +
# COM: return type. The symbol mangles the parameter (`[::SIMD[DType.int, 1]]`) and owned arg (`::Int$`),
# COM: declares the parameter `<N: ...>`, captures `z`, and is throwing.

# CHECK: lit.struct.decl @"withEverything()::`lambda_0::__storage"
# CHECK: lit.struct.field z : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK: lit.fn @"`lambda_0[::SIMD[DType.int, 1]](::SIMD[DType.int, 1]${{.*}}"<{{.*}}N: !Int{{[0-9]*}}>{{.*}} throws|capturing -> {{.*}}


def withEverything():
    var z = 3
    var f = lambda [N: Int](var x: Int) raises {mut} -> Int: x + N + z


# // -----

# COM: A thin lambda bound to a `comptime` promotes to a free function (no
# COM: storage struct), and the alias folds to that function's literal -- as
# COM: `comptime f = some_def` does. Check the promoted fn's definition and the
# COM: alias's reference to it (`{{.*}}` absorbs the promotion mangling suffix).

# CHECK-DAG: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"{{.*}}-> !{{.*}}Int{{[0-9]*}}
# CHECK-DAG: lit.alias.decl {{.*}}func.literal{{.*}}func.symbol<@{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}>


comptime inc = lambda (x: Int) {} -> Int: x + 1


def withComptimeBound() -> Int:
    return inc(1)


# // -----

# COM: Elided return type defaults to `None`, like a `def` with no `->`. `f` is
# COM: thin (promotes, no storage); `g` captures `lst`, so it stays a closure.
# COM: Nested `lambda_1` methods print inside storage before thin `lambda_0`.

# CHECK-NOT: @"withElidedReturn()::`lambda_0::__storage"
# CHECK-DAG: lit.struct.decl @"withElidedReturn()::`lambda_1::__storage"
# CHECK-DAG: lit.struct.field lst : !lit.ref<{{.*}}, mut {{.*}}>
# CHECK-DAG: lit.fn @"{{.*}}`lambda_1(){{.*}}"{{.*}} capturing -> !kgen.none
# CHECK-DAG: lit.call @{{.*}}List{{.*}}append
# CHECK: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"(%x: !Int) -> !kgen.none


def withElidedReturn():
    var f = lambda (x: Int) {}: None
    var lst: List = [1]
    var g = lambda {mut}: lst.append(2)


# // -----

# COM: Omitted capture list defaults to `{imm}`: free variables are imm-captured (an
# COM: immutable ref); with no free variables the lambda is thin, like an explicit `{}`,
# COM: and promotes to a plain function (no storage struct). `multi` imm-captures
# COM: several free variables at once. (Structs emit in reverse order, hence CHECK-DAG.)

# CHECK-NOT: @"{{.*}}`lambda_0::__storage"
# CHECK-DAG: lit.struct.decl @"withOmittedCaptures()::`lambda_1::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {{{.*}}synthetic}
# CHECK-DAG: lit.struct.decl @"withOmittedCaptures()::`lambda_2::__storage"<{{.*}}>({{.*}}) register_passable_trivial attributes {{{.*}}synthetic}
# CHECK-DAG: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK-DAG: lit.struct.field w : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK: lit.call @{{.*}}::@"withOmittedCaptures()::`lambda_2::__storage"::@"__init__({{.*}},{{.*}})"{{.*}}("z": {{.*}}, "w": {{.*}} ref, |
# CHECK: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"(%x: !Int)


def withOmittedCaptures():
    var thin = lambda (x: Int) -> Int: x + 1
    var z = 3
    var w = 4
    var reads = lambda (x: Int) -> Int: x + z
    var multi = lambda (x: Int) -> Int: x + z + w


# // -----

# COM: Everything together, with elision: parameter + owned arg + effects, an omitted
# COM: capture list (`z` imm-captured) and an omitted return type (`None`). The body is a
# COM: `None`-returning call that reads the captured `z`.

# CHECK: lit.struct.decl @"withEverythingAndWithElision()::`lambda_0::__storage"
# CHECK: lit.struct.field z : !lit.ref<{{.*}}, imm {{.*}}>
# CHECK: lit.fn @"`lambda_0[::SIMD[DType.int, 1]](::SIMD[DType.int, 1]${{.*}}"<{{.*}}N: !Int{{[0-9]*}}>{{.*}}!lit.ref<none, {{.*}}> byref_result{{.*}} throws|capturing -> {{.*}}


def noop(v: Int):
    pass


def withEverythingAndWithElision():
    var z = 3
    var f = lambda [N: Int](var x: Int) raises: noop(z + N + x)


# // -----

# COM: The comptime fold composes with elision: with the capture list omitted and
# COM: nothing captured, the lambda is thin, so it promotes and the alias folds to
# COM: the promoted function's literal -- exactly like the explicit-`{}` fold
# COM: (cf. withComptimeBound). No `__storage` struct exists for it.

# CHECK-DAG: lit.fn @"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"{{.*}}-> !{{.*}}Int{{[0-9]*}}
# CHECK-DAG: lit.alias.decl {{.*}}func.literal{{.*}}func.symbol<@{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}>
# CHECK-NOT: __storage


comptime inc_elided = lambda (x: Int) -> Int: x + 1


def withComptimeBoundElided() -> Int:
    return inc_elided(1)
# // -----

# COM: A thin lambda decays into a typed `thin` fn-pointer slot, as a `def`
# COM: name does: the var's element type is the fn generator, both the
# COM: initializer and a REbinding assign the promoted function's value, and
# COM: the call through the var is indirect.

# CHECK: lit.var.decl "f" var : !lit.ref<!lit.generator<("x": !Int) -> !{{.*}}Int{{[0-9]*}}>
# CHECK: kgen.create_closure[{{.*}}: @{{.*}}::@"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"]()
# CHECK: kgen.create_closure[{{.*}}: @{{.*}}::@"{{.*}}`lambda_1(::SIMD[DType.int, 1]){{.*}}"]()
# CHECK: lit.call_indirect


def typedDecay() -> Int:
    var f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + 1
    f = lambda (x: Int) -> Int: x * 3
    return f(2)


# // -----

# COM: A thin lambda referencing an ENCLOSING parameter still decays: the
# COM: reference is bound, not free -- promotion prepends the parameter and the
# COM: use site binds it, exactly as a stateless nested `def` using `N` does.
# COM: (Only a lambda's OWN unbound parameters keep the closure-instance form.)

# CHECK: kgen.create_closure[{{.*}}@"{{.*}}`lambda_0(::SIMD[DType.int, 1]){{.*}}"<:!Int N>]()
# CHECK: lit.call_indirect


def paramBound[N: Int]() -> Int:
    var f: def(x: Int) thin -> Int = lambda (x: Int) -> Int: x + N
    return f(1)
