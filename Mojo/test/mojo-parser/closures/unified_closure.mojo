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
# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -o %t.mlir

# RUN: FileCheck %s --enable-var-scope --check-prefixes=S0 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S1 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S2 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S3 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S4 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S5 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S6 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S7 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S8 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S9 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S10 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S11 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S12 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S13 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S14 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S15 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S16 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S17 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S18 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S19 < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=KWARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=STAR_ARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=KWARGS_FN_PTR < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=STAR_ARGS_KWARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=MIXED_KWARGS < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=MIXED_KWARGS_FN_PTR < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=STAR_ARGS_KWARGS_FN_PTR < %t.mlir
# RUN: FileCheck %s --enable-var-scope --check-prefixes=S20 < %t.mlir
# COM: Verify generated trait and storage-struct structure (no parametric wrapper).
# S0-DAG: [[S0_PARENT:!Int_AnyType_Deinitable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>
# S0-DAG: [[S0_IMPL_PARENT:!Int_AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>
# S0-DAG: [[S0_INT:!.*]] = !lit.struct<#SIMD <{{.*}}>>
# S0-DAG: lit.trait.decl @"def(y: Int) -> Int"<?, *"_Self`{{.*}}": [[S0_PARENT]]>([[S0_PARENT]])
# S0-DAG: lit.fn @"__call__($0,::SIMD[DType.int, 1])"[mut *"self`"](%{{.*}}: !lit.ref<:{{.*}}, mut *"self`"> imm_mem, |, %y: {{.*}}) capturing -> {{.*}} attributes {sourceName = "__call__", specialFnKind = 0 : i8, synthetic} {

# S0: lit.struct.decl @"{{.*}}s0_make_closure{{.*}}::my_closure::__storage"([[S0_IMPL_PARENT]]) attributes {definesClosure,{{.*}}synthetic}
# S0-NEXT: move :{{.*}}@{{.*}}::@"{{.*}}::my_closure::__storage"::@"__init__(move:
# S0-NEXT: copy :{{.*}}@{{.*}}::@"{{.*}}::my_closure::__storage"::@"__init__(copy:
# S0: lit.fn @"my_closure{{.*}}"[mut {{.*}}](%{{.*}}: !lit.ref<!storage{{.*}}, mut {{.*}}> imm_mem, |, %y: {{.*}}) capturing -> {{.*}}
# S0: lit.fn @"__init__(move:{{.*}}::my_closure::__storage$)"
# S0: lit.fn @"__deinit__({{.*}}::my_closure::__storage$)"
# S0: kgen.witness "__call__{{.*}}" : {{.*}} = @{{.*}}::@"{{.*}}::my_closure::__storage"::@"__call__{{.*}}"
# S0: kgen.witness "__init__(move:$0$)" : {{.*}} = @{{.*}}::@"{{.*}}::my_closure::__storage"::@"__init__(move:
# S0: kgen.witness "__deinit__{{.*}}" : {{.*}} = @{{.*}}::@"{{.*}}::my_closure::__storage"::@"__deinit__(
# S0-NOT: lit.struct.decl @"def(y: Int) -> Int_{{[^"]*}}"<impl:


# With -split-input-file and --kgen-print-inline-type-values, the closure trait may be printed as _Self: !Int or *"_Self`0x": !Int.


def s0_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y


# COM: Verify Nested closures are supported
# S1-DAG: lit.trait.decl @"def[y: def(z: Int) -> Int]{{.*}}"
# S1-DAG: lit.trait.decl @"def(z: Int) -> Int"
# S1-DAG: lit.struct.decl @"{{.*}}s1_make_closure{{.*}}::my_closure::__storage"
# S1-DAG: lit.struct.decl @"{{.*}}my_nested_closure::__storage"


def s1_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        def my_nested_closure(z: Int) {var x, var mem} -> Int:
            return x

        return x + y


# COM: Ensure identical closure traits are reused; no parametric wrapper is emitted.
# S2-COUNT-1: lit.trait.decl @"def(y: Int) {{.*}} -> Int"
# S2-NOT: lit.struct.decl @"def(y: Int) {{.*}} -> Int_{{.*}}"<impl:


def s2_make_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


def make_identical_closure(x: Int):
    def my_closure(y: Int) {var} -> Int:
        return y


# COM: Test that parametric functions in traits are handled correctly
# S3: [[S3_TRAIT:!None_AnyType_Deinitable_Movable.*]] = !lit.trait<@"def[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) -> None", @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>
# S3: lit.trait.decl @"def[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) -> None"<?, *"_Self`{{.*}}": [[S3_TRAIT]]>(!{{.*}}) unspecified attributes {{{.*}}} {
# S3: lit.fn @"__call__{{.*}}"<T: !AnyType_Movable_MyInterface, b: !kgen.param<:!AnyType_Movable_MyInterface T>, c: {{.*}}Foo <:!AnyType_Movable {{.*}}, :!kgen.param<:!AnyType_Movable_MyInterface T> b>>
# S3-SAME: [mut *"self`", imm *"[[S3_L1:.*]]`"](%0[*""]: !lit.ref<:[[S3_TRAIT]] *"_Self`{{.*}}", mut *"self`"> imm_mem, |, %a: !lit.ref<:!AnyType_Movable_MyInterface T, imm *"[[S3_L1]]`"> imm_mem) capturing -> !kgen.none


trait s3_MyInterface(Movable):
    def thing(self):
        ...


struct Foo[T: Movable, b: T](Movable where False):
    pass


def s3_make_closure(x: Int, mem: String) -> Int:
    def parametric[T: s3_MyInterface, b: T, c: Foo[T, b]](a: T) {var}:
        _ = mem

    return x


# COM: Explicit origins are handled on the storage struct (no parametric wrapper).
# S4: [[S4_TRAIT:!None_AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable.*]] = !lit.trait<@"def[{{.*}}](a: ref[lt] String, b: String) -> None",
# S4: lit.struct.decl @"{{.*}}s4_make_closure{{.*}}::mutate::__storage"([[S4_TRAIT]]) attributes {definesClosure,{{.*}}synthetic}
# S4: kgen.conformance @"def[{{.*}}](a: ref[lt] String, b: String) -> None" {
# S4-DAG: kgen.witness "__call__{{.*}}" : {{.*}} = @{{.*}}::@"{{.*}}::mutate::__storage"::@"__call__{{.*}}"
# S4-NOT: lit.struct.decl @"def[{{.*}}](a: ref[lt] String, b: String) -> None_{{[^"]*}}"<impl:


def s4_make_closure(x: Int, mem: String) -> Int:
    def mutate[
        lt: Origin[mut=True]
    ](a: Pointer[String, lt]._mlir_lit_ref, b: String) {var}:
        _ = mem

    return x


# COM: Verify storage constructor takes the captured value (no wrapper impl arg).
# S5: [[S5_TRAIT:!None_AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable.*]] = !lit.trait<@"def[T: s5_MyInterface](a: T) -> None", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>
# S5: lit.struct.decl @"{{.*}}s5_make_closure{{.*}}::parametric::__storage"([[S5_TRAIT]]) attributes {definesClosure,{{.*}}synthetic}
# S5: lit.fn @"__init__(::String)"[imm *"mem`", mut *"self`"]
# S5-NOT: lit.fn @"__init__($0$)"[mut *"impl`", mut *"self`"]


trait s5_MyInterface:
    def thing(self):
        ...


def s5_make_closure(x: Int, mem: String) -> Int:
    def parametric[T: s5_MyInterface](a: T) {var}:
        _ = mem

    return x


# COM: Verify the closure instance is created as the storage struct alone
# COM: (no parametric wrapper around it).
# S6-DAG: lit.var.decl "my_closure" var
# S6-DAG: lit.call {{.*}}s6_make_closure{{.*}}::my_closure::__storage"::@"__init__
# S6-NOT: lit.var.decl "my_closure.storage" var
# S6-NOT: lit.call {{.*}}::@"def(y: Int) -> Int_{{.*}}::@"__init__($0$)"


def s6_make_closure(x: Int, mem: String):
    def my_closure(y: Int) {var x, var mem} -> Int:
        return x + y


# COM: Check that the argument is augmented at the definition site.
# S7-DAG: [[S7_TRAIT:!Int_AnyType_Deinitable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>

# S7: lit.fn @"s7_take_closure{{.*}}"<f: [[S7_TRAIT]]>[imm *"myFunc`"](%myFunc: !lit.ref<:[[S7_TRAIT]] f, imm *"myFunc`"> imm_mem, %x: !Int{{.*}}) capturing -> !kgen.none
# S7-NEXT: %0 = lit.call tail[!lit.generator<[1](!lit.ref<:[[S7_TRAIT]] f, mut *[0,0]> imm_mem, |, "y": !Int{{.*}}) capturing -> !Int{{.*}}>: #kgen.get_witness<:[[S7_TRAIT]] f, @"def(y: Int) -> Int", "__call__{{.*}}">][imm *"myFunc`"](%myFunc, %x)
# S7-NEXT: lit.ownership.use %0
# S7-NEXT: %none = kgen.param.constant: none = <#kgen.none>


def s7_take_closure[f: def(y: Int) -> Int](myFunc: f, x: Int):
    _ = myFunc(x)


# COM: Ensure the transformed parameters are propagated into the underlying closure trait.
# S8-DAG: [[S8_TRAIT:!Int_AnyType_Deinitable_Movable.*]] = !lit.trait<@"def(y: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>
# S8-DAG: [[S8_INT:!Int.*]] = !lit.struct<#SIMD <{{.*}}>>
# S8-DAG: lit.trait.decl @"def(y: Int) -> Int"
# S8-DAG: lit.fn *"nested[def(y: Int) -> Int & ::AnyType & ::Deinitable & ::Movable]($0,::SIMD[DType.int, 1])"<closure2: [[S8_TRAIT]]>


def s8_take_closure[closure1: def(y: Int) -> Int](x: Int):
    def nested[
        closure2: def(y: Int) -> Int
    ](impl: closure2, y: Int) {var x} -> Int:
        return x


# COM: ensure many closure parameters are handled.
# S9: lit.fn @"take_closures{{.*}})"
# S9-SAME: <closure1: !Int_AnyType_Deinitable_Movable{{[0-9]*}}, T: !Int{{[0-9]*}}, closure2: !Int_AnyType_Deinitable_Movable{{[0-9]*}}, U: !Int{{[0-9]*}}>
# S9-SAME: [imm *"[[S9_L0:.*]]`", imm *"[[S9_L1:.*]]`1"]
# S9-SAME: (%impl1: !lit.ref<:!Int_AnyType_Deinitable_Movable{{[0-9]*}} closure1, imm *"[[S9_L0]]`"> imm_mem
# S9-SAME: , %impl2: !lit.ref<:!Int_AnyType_Deinitable_Movable{{[0-9]*}} closure2, imm *"[[S9_L1]]`1"> imm_mem, %x: !Int{{[0-9]*}}) capturing -> !kgen.none


def take_closures[
    closure1: def(y: Int) -> Int,
    T: Int,
    closure2: def(y: Int, z: Int) -> Int,
    U: Int,
](impl1: closure1, impl2: closure2, x: Int):
    pass


# COM: Unified Closure Parameters compose
# S10-DAG: [[S10_INNER:!Int_AnyType_Deinitable_Movable.*]] = !lit.trait<@"def(z: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Deinitable, @{{.*}}::@Movable>
# S10-DAG: lit.fn @"__call__[def(z: Int) -> Int{{.*}}"<y: [[S10_INNER]]>
# S10-DAG: lit.fn @"nested[def[y: def(z: Int) -> Int](impl: y, u: Int) -> Int & ::AnyType & ::Deinitable & ::Movable]($0,::SIMD[DType.int, 1])"
# S10-DAG: %impl: !lit.ref<:!Int_AnyType_Deinitable_Movable{{.*}} x, imm *{{.*}} imm_mem
# S10-DAG: %do_not_dce_int: !Int{{.*}}) capturing -> !kgen.none attributes {{.*}}sourceName = "nested"


# TODO: remove the 'do_not_dce_int' argument (MOCO 2461)
def nested[
    x: def[y: def(z: Int) -> Int](impl: y, u: Int) -> Int, //
](impl: x, do_not_dce_int: Int):
    pass


# COM: Check that the closure storage struct is generated correctly.
# S11-DAG: [[S11_TRAIT:!Int_AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable.*]] = !lit.trait<@"def(z: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>
# S11-DAG: lit.struct.decl @"s11_bindIt(::SIMD[DType.int, 1],::SIMD[DType.int, 1],::String)::myclosure::__storage"
# S11-DAG: kgen.conformance @{{.*}}::@AnyType {
# S11-DAG: kgen.conformance @{{.*}}::@Deinitable {
# S11-DAG: kgen.witness "__deinit__{{.*}}" : {{.*}} = @{{.*}}::@"s11_bindIt{{.*}}::myclosure::__storage"::@"__deinit__
# S11-DAG: kgen.conformance @{{.*}}::@Movable {
# S11-DAG: kgen.witness "__init__(move:$0$)" : {{.*}} = @{{.*}}::@"s11_bindIt{{.*}}::myclosure::__storage"::@"__init__(move:
# S11-DAG: kgen.conformance @"def(z: Int) -> Int" {
# S11-DAG: kgen.witness "__call__{{.*}}" : {{.*}} = @{{.*}}::@"s11_bindIt{{.*}}::myclosure::__storage"::@"__call__


def s11_bindIt(x: Int, y: Int, mem: String) -> Int:
    def myclosure(z: Int) {var x, var y, var mem} -> Int:
        return x + y + z


# COM: Check that parameters are emitted correctly

# S12: lit.struct.decl @"s12_bindIt({{.*}})::myclosure::__storage"
# S12: kgen.witness "__call__{{.*}}" : !lit.generator<<"my_param": !AnyType>
# S12-SAME: [1](!lit.ref<{{.*}}, mut *[0,0]> imm_mem, |, "z": !Int{{.*}}) capturing -> !kgen.none>
# S12-SAME: = @{{.*}}::@"s12_bindIt({{.*}})::myclosure::__storage"::@"__call__{{.*}}"

# S12-DAG: lit.file_module


def s12_bindIt(mem: String) -> Int:
    def myclosure[my_param: AnyType](z: Int) {var}:
        _ = mem


# COM: Captured mutable reference contributes byRefMut's origin to storage.
# S13-LABEL: lit.fn @"nonemptyOriginSet(::String)"
# S13: lit.call {{.*}}::myclosure::__storage"::@"__init__
# S13-SAME: <:origin<true> *"byRefMut
# S13-NOT: lit.call @unified_closure::@"def() -> None_{{.*}}"::@"__init__


def nonemptyOriginSet(mut byRefMut: String):
    def myclosure() {mut byRefMut}:
        pass


# COM: Verify that closures can be rebound to compatible traits
# S14-DAG: lit.struct.decl @"s14_bindIt{{.*}}::myclosure::__storage"
# S14-DAG: kgen.witness "__call__($0,::SIMD[DType.int, 1])"
# S14-DAG: imm_mem, !Int{{.*}}, |) capturing -> !Int{{.*}}> = rebind(:!lit.generator<[1]({{.*}}imm_mem, |, "x": !Int{{.*}}) capturing -> !{{.*}}>
# S14-DAG: @{{.*}}::@"s14_bindIt{{.*}}::myclosure::__storage"::@"__call__


def s14_takeIt[C: def(Int) -> Int](closure: C):
    _ = closure(3)


def s14_bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    s14_takeIt[type_of(myclosure)](myclosure)


# COM: Verify that closures can be rebound even when traits are combined
# S15-DAG: lit.struct.decl @"s15_bindIt{{.*}}::myclosure::__storage"
# S15-DAG: kgen.witness "__call__($0,::SIMD[DType.int, 1])"
# S15-DAG: imm_mem, |, "y": !Int{{.*}}) capturing -> !Int{{.*}}> = rebind(:!lit.generator<[1]({{.*}}imm_mem, |, "x": !Int{{.*}}) capturing -> !{{.*}}>
# S15-DAG: @{{.*}}::@"s15_bindIt{{.*}}::myclosure::__storage"::@"__call__


def s15_takeIt[C: Copyable & def(y: Int) -> Int](closure: C):
    _ = closure(3)


def s15_bindIt(z: Int, mem: String):
    def myclosure(x: Int) {var} -> Int:
        _ = mem
        return z

    s15_takeIt[type_of(myclosure)](myclosure)


# COM: Verify that all closures are rebound when closure traits are combined or inherited

# S16-DAG: lit.struct.decl @MultipleClosure
# S16-DAG: kgen.conformance @"def(Bool) -> Int"
# S16-DAG: kgen.witness "__call__($0,::Bool)"
# S16-DAG: imm_mem, !Bool, |) capturing -> !Int{{.*}}> = rebind(:!lit.generator<[1]({{.*}}imm_mem, !Bool) capturing -> !Int>
# S16-DAG: @{{.*}}::@MultipleClosure::@"__call__(unified_closure::MultipleClosure,::Bool)"
# S16-DAG: kgen.conformance @"def(Int) -> Int"
# S16-DAG: kgen.witness "__call__($0,::SIMD[DType.int, 1])"
# S16-DAG: imm_mem, !Int{{.*}}, |) capturing -> !Int{{.*}}> = rebind(:!lit.generator<[1]({{.*}}imm_mem, !Int) capturing -> !Int>
# S16-DAG: @{{.*}}::@MultipleClosure::@"__call__(unified_closure::MultipleClosure,::SIMD[DType.int, 1])"


def s16_takeIt[C: (def(Bool) -> Int) & def(Int) -> Int](closure: C):
    _ = closure(3)


trait BoolWrapper(def(Bool) -> Int):
    pass


struct MultipleClosure(BoolWrapper, Movable, def(Int) -> Int):
    def __init__(out self):
        pass

    def __call__(self, x: Bool) -> Int:
        return 1

    def __call__(self, x: Int) -> Int:
        return 2


def s16_bindIt(z: Int):
    var fakeclosure = MultipleClosure()

    s16_takeIt[type_of(fakeclosure)](fakeclosure)


# COM: Verify that closures can be rebound with differing parameter names
# S17-DAG: lit.struct.decl @"s17_bindIt{{.*}}::myclosure::__storage"
# S17-DAG: kgen.conformance @"def[x: Int](y: Int) -> Int"
# S17-DAG: kgen.witness "__call__[::SIMD[DType.int, 1]]($0,::SIMD[DType.int, 1])"
# S17-DAG: imm_mem, |, "y": !Int{{.*}}) capturing -> !Int{{.*}}> = rebind(:!lit.generator<<"a": !Int{{.*}}>[1]({{.*}}imm_mem, |, "b": !Int{{.*}}) capturing -> !{{.*}}>
# S17-DAG: @{{.*}}::@"s17_bindIt{{.*}}::myclosure::__storage"::@"__call__


def s17_takeIt[C: def[x: Int](y: Int) -> Int](closure: C):
    # see MOCO-2606
    _ = closure.__call__[2](3)


def s17_bindIt(z: Int, mem: String):
    def myclosure[a: Int](b: Int) {var} -> Int:
        _ = mem
        return z

    s17_takeIt[type_of(myclosure)](myclosure)


# COM: Ensure that structs can conform to the closure trait

# S18-DAG: lit.struct.decl @custom(!Int_AnyType_Deinitable_Movable{{.*}})


struct custom(def(x: Int) -> Int):
    def __call__(self, x: Int) capturing -> Int:
        return x


# COM: Storage conforms to Copyable (no parametric wrapper).
# S19-DAG: !lit.trait<@"def(x: Int) -> Int", @{{.*}}::@AnyType, @{{.*}}::@Copyable, @{{.*}}::@Deinitable, @{{.*}}::@ImplicitlyCopyable, @{{.*}}::@Movable>
# S19-NOT: lit.struct.decl @"def(x: Int) -> Int_{{.*}}"<impl:


def takeItImplicit[T: ImplicitlyCopyable](impl: T):
    pass


def s19_takeIt[T: Copyable](impl: T):
    pass


@fieldwise_init
struct CopyMe(ImplicitlyCopyable):
    var x: Int
    var y: Int


@fieldwise_init
struct OneOfAKind(Movable):
    var x: Int
    var y: Int


def useIt(var x: OneOfAKind):
    pass


@no_inline
def giveIt(z: Int, cm: CopyMe, var one: OneOfAKind):
    def aThing(x: Int) {var z, var cm} -> Int:
        return z + x

    takeItImplicit(aThing)
    s19_takeIt(aThing)

    def anotherThing(x: Int) {var^} -> Int:
        useIt(one^)
        return x


# COM: KWARGS: a `**kwargs` argument is accepted directly on the storage
# COM: struct's `__call__` (no parametric wrapper hop). The packed dict is
# COM: passed as a single `**` operand at the call site.

# Storage promoted method / canonical __call__ takes the dict kwargs...
# KWARGS: lit.fn @"g(kwargs:::SIMD[DType.int, 1]**)`"
# KWARGS: lit.fn @"kwargs_throughWrapper()"
# ...and the call site invokes the canonical `__call__` with the packed dict.
# KWARGS: lit.call{{.*}}::g::__storage"::@"__call__{{.*}}({{.*}}kwargs:::SIMD{{.*}}(%{{.*}}, %__call_result_tmp__)


def kwargs_throughWrapper() -> Int:
    var z = 1

    def g(var **kwargs: Int) {imm z} -> Int:
        return z

    return g(a=1, b=2)


# COM: STAR_ARGS: `*args` is accepted directly on the storage struct's
# COM: `__call__` (no parametric wrapper hop).

# STAR_ARGS: lit.fn @"h[{{.*}}(::SIMD[DType.int, 1]*)`"
# STAR_ARGS: lit.fn @"star_args_throughWrapper()"
# STAR_ARGS: lit.call{{.*}}::h::__storage"::@"__call__{{.*}}({{.*}}(%{{.*}}, %{{.*}})


def star_args_throughWrapper() -> Int:
    var z = 1

    def h(*args: Int) {imm z} -> Int:
        return z

    return h(1, 2)


# COM: KWARGS_FN_PTR: binding a plain `**kwargs` function into a closure-typed
# COM: value mints its own (fn-pointer) wrapper; its forwarding is pinned
# COM: separately.

# KWARGS_FN_PTR: lit.fn @"__call__({{.*}}_PtrWrapper[$0],kwargs:::SIMD[DType.int, 1]**)"
# KWARGS_FN_PTR: lit.call{{.*}}: Impl]{{.*}}(%kwargs)
# KWARGS_FN_PTR: lit.fn @"kwargs_fn_ptr_useFnWrapper()"


def kwargs_fn_ptr_top(var **kwargs: Int) -> Int:
    return 1


def kwargs_fn_ptr_takeClosure(f: Some[def(var ** kwargs: Int) -> Int]) -> Int:
    return f(a=1)


def kwargs_fn_ptr_useFnWrapper() -> Int:
    return kwargs_fn_ptr_takeClosure(kwargs_fn_ptr_top)


# COM: STAR_ARGS_KWARGS: `*args` and `**kwargs` together are accepted
# COM: directly on the storage struct's `__call__`.

# STAR_ARGS_KWARGS: lit.fn @"b[{{.*}}(::SIMD[DType.int, 1]*,kwargs:::SIMD[DType.int, 1]**)`"
# STAR_ARGS_KWARGS: lit.fn @"star_args_kwargs_throughWrapper()"
# STAR_ARGS_KWARGS: lit.call{{.*}}::b::__storage"::@"__call__{{.*}}({{.*}}(%{{.*}}, %{{.*}}, %__call_result_tmp__)


def star_args_kwargs_throughWrapper() -> Int:
    var z = 1

    def b(*args: Int, var **kwargs: Int) {imm z} -> Int:
        return z

    return b(1, 2, a=3)


# COM: MIXED_KWARGS: a named keyword-only argument is accepted alongside
# COM: `**kwargs` directly on the storage struct's `__call__`.

# MIXED_KWARGS: lit.fn @"m(::SIMD[DType.int, 1],named:::SIMD[DType.int, 1],kwargs:::SIMD[DType.int, 1]**)`"
# MIXED_KWARGS: lit.fn @"mixed_kwargs_throughWrapper()"
# MIXED_KWARGS: lit.call{{.*}}::m::__storage"::@"__call__{{.*}}({{.*}}(%{{.*}}, %{{.*}}, %{{.*}}, %__call_result_tmp__)


def mixed_kwargs_throughWrapper() -> Int:
    var z = 1

    def m(x: Int, *, named: Int, var **kwargs: Int) {imm z} -> Int:
        return z + x + named

    return m(1, named=2, a=3, b=4)


# A defaulted keyword-only argument is accepted the same way on storage
# `__call__`. The extra `y` keeps this closure's trait name distinct from
# mixed_kwargs_throughWrapper's -- the trait name omits defaults, and a
# collision is an "invalid redefinition".
def mixed_kwargs_defaultedThroughWrapper() -> Int:
    var z = 1

    def m(x: Int, y: Int, *, named: Int = 7, var **kwargs: Int) {imm z} -> Int:
        return z + named

    return m(1, 2, a=3)


# A second closure with the same signature reuses the cached trait /
# storage shape (no parametric wrapper).
def mixed_kwargs_duplicateSignature() -> Int:
    var z = 2

    def m(x: Int, *, named: Int, var **kwargs: Int) {imm z} -> Int:
        return z

    return m(1, named=2, a=3)


# COM: MIXED_KWARGS_FN_PTR: the same mixed signature forwards through the
# COM: fn-pointer wrapper minted when a plain function is bound into a
# COM: closure-typed value.

# MIXED_KWARGS_FN_PTR: lit.fn @"__call__{{.*}}_PtrWrapper[$0],::SIMD[DType.int, 1],named:::SIMD[DType.int, 1],kwargs:::SIMD[DType.int, 1]**)"
# MIXED_KWARGS_FN_PTR: lit.call{{.*}}: Impl]{{.*}}(%x, %named, %kwargs)
# MIXED_KWARGS_FN_PTR: lit.fn @"mixed_kwargs_fn_ptr_useFnBinding()"


def mixed_kwargs_fn_ptr_top(x: Int, *, named: Int, var **kwargs: Int) -> Int:
    return x + named


def mixed_kwargs_fn_ptr_takeClosure(
    f: Some[def(x: Int, *, named: Int, var ** kwargs: Int) -> Int]
) -> Int:
    return f(1, named=2, a=3)


def mixed_kwargs_fn_ptr_useFnBinding() -> Int:
    return mixed_kwargs_fn_ptr_takeClosure(mixed_kwargs_fn_ptr_top)


# COM: STAR_ARGS_KWARGS_FN_PTR: the both-variadics signature forwards through
# COM: the fn-pointer wrapper as well.

# STAR_ARGS_KWARGS_FN_PTR: lit.fn @"__call__{{.*}}def(*args: Int, var **kwargs: Int) thin -> Int_PtrWrapper[$0],::SIMD[DType.int, 1]*,kwargs:::SIMD[DType.int, 1]**)"
# STAR_ARGS_KWARGS_FN_PTR: lit.call{{.*}}(%args, %kwargs)
# STAR_ARGS_KWARGS_FN_PTR: lit.fn @"star_args_kwargs_fn_ptr_useFnWrapper()"


def star_args_kwargs_fn_ptr_top(*args: Int, var **kwargs: Int) -> Int:
    return 1


def star_args_kwargs_fn_ptr_takeClosure(
    f: Some[def(* args: Int, var ** kwargs: Int) -> Int]
) -> Int:
    return f(1, 2, a=3)


def star_args_kwargs_fn_ptr_useFnWrapper() -> Int:
    return star_args_kwargs_fn_ptr_takeClosure(star_args_kwargs_fn_ptr_top)


# COM: Ensure index replacement is asserted on name not attribute identity since replacement operates over uncanonical form
# S20-DAG: lit.trait.decl @"def[X: Copyable & Deinitable, //](y: X) -> None{{.*}}"


# COM: S20_Bound has sugared type
comptime S20_Bound = Copyable & Deinitable


def s20_apply[X: S20_Bound, //, F: def(X)](f: F):
    pass


def s20_bindIt[X: S20_Bound]():
    # COM: Capture of sugared type
    def reap(y: X):
        pass

    # COM: To inflate reap into a closure type we need to bind the parameters of its wrapper type.
    #      That means we must match a parameter to a value. We asserted that there is a parameter
    #      to bind that value to but we assumed that the types were equal when in fact they only
    #      need to be canonically equal.
    s20_apply(reap)
