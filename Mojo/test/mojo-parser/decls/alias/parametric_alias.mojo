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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

##===----------------------------------------------------------------------===##
# declarations
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"noParam{{.*}}": !alias_Int1 = <rebind(:!Int {:scalar<index> 78})>
comptime noParam: Int = 78

# CHECK: lit.alias.decl *"emptyParams{{.*}}": !alias_Int1 = <rebind(:!Int {:scalar<index> 89})>
comptime emptyParams[]: Int = 89

# CHECK: lit.alias.decl *"idInt{{.*}}": !lit.generator<<"x": !Int>!alias_Int1> = <#kgen.gen<rebind(:!Int *(0,0))>>
comptime idInt[x: Int]: Int = x

# A parametric comptime may annotate a body type that depends on its parameters
# (e.g. `SIMD[DType.int, x]`). That annotation is a result-type schema for the
# generator, not a concrete type to construct — accept the initializer without
# "cannot construct a value with parametric type".
# CHECK: lit.alias.decl *"ComptimeWithParametricType{{.*}}": !lit.generator<<"x": !Int>{{.*}}!lit.struct<#SIMD
comptime ComptimeWithParametricType[x: Int]: SIMD[.int, x] = SIMD[.int, x]()

# CHECK: lit.alias.decl *"myIntAdd{{.*}}": !lit.generator<<"x": !Int, "y": !Int>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}add(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">){{.*}}>>
comptime myIntAdd[x: Int, y: Int] = x + y

# CHECK: lit.alias.decl *"myDefaultAdd{{.*}}": !lit.generator<<"x": !Int, "y": !Int = {:scalar<index> 1}>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}add(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">){{.*}}>>
comptime myDefaultAdd[x: Int, y: Int = 1] = x + y

# CHECK: lit.alias.decl *"myDependentDefaultAdd{{.*}}": !lit.generator<<"x": !Int, "y": !Int = *(0,0)>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}scalar<index> = add(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">){{.*}}>>
comptime myDependentDefaultAdd[x: Int, y: Int = x] = x + y

# CHECK: lit.alias.decl *"myIntFMA{{.*}}": !lit.generator<<"x": !Int, "y": !Int, "z": !Int>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}add(mul(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">), #lit.struct.extract<:!Int *(0,2), "_mlir_value">){{.*}})>>
comptime myIntFMA[x: Int, y: Int, z: Int] = x * y + z

# CHECK: lit.alias.decl *"myTypeSelector{{.*}}": !lit.generator<<"cond": !Bool, "t_type": !AnyType, "f_type": !AnyType>!AnyType> = <#kgen.gen<
# CHECK-SAME: cond(#lit.struct.extract<:!Bool *(0,0), "_mlir_value">, *(0,1), *(0,2))>>
comptime myTypeSelector[
    cond: Bool, t_type: AnyType, f_type: AnyType
] = t_type if cond else f_type

# CHECK: lit.alias.decl *"whereBool{{.*}}": !lit.generator<
# CHECK-SAME: "cond": !Bool
# CHECK-SAME: {<{{.*}}#lit.struct.extract<:!Bool *(0,0), "_mlir_value">{{.*}}>}
# CHECK-SAME: !AnyType> = <#kgen.gen<!Int>>
comptime whereBool[cond: Bool]: AnyType where cond = Int

# CHECK: lit.alias.decl *"nonParametricWhere{{.*}}": !lit.generator<
# CHECK-SAME: {<true{{.*}}>}
# CHECK-SAME: !AnyType> = <#kgen.gen<!Int>>
comptime nonParametricWhere: AnyType where True = Int

# CHECK: lit.alias.decl *"emptyParamWhere{{.*}}": !lit.generator<
# CHECK-SAME: {<true{{.*}}>}
# CHECK-SAME: !AnyType> = <#kgen.gen<!Int>>
comptime emptyParamWhere[]: AnyType where True = Int

# CHECK: lit.alias.decl *"explicitEmptyGeneratorAsType{{.*}}": !AnyType = <!Int>
comptime explicitEmptyGeneratorAsType: AnyType = __mlir_attr[
    `#kgen.gen<`, Int, `> : !lit.generator<<>`, +AnyType, `>`
]

# CHECK-LABEL: lit.fn @"implicit_empty_generator_param
# CHECK: lit.alias.decl *"bound{{.*}}": !AnyType
# CHECK-SAME: = <bind_params(:!lit.generator<<>!AnyType> g)>
def implicit_empty_generator_param[
    g: __generator_type AnyType
]():
    comptime bound: AnyType = g


# CHECK-LABEL: lit.fn @"implicit_generator_constraint_drop
# CHECK-NOT: rebind(:!lit.generator<<"x": !Int, {<
# CHECK: lit.alias.decl *"dropped{{.*}}": !lit.generator<<"x": !Int>!AnyType>
# CHECK-SAME: = <#kgen.gen<!Int>>
# CHECK: lit.alias.decl *"used{{.*}}": !lit.generator<<"a": !Int>!AnyType> = <#kgen.gen<!Int>>
def implicit_generator_constraint_drop[cond: Bool]() where cond:
    comptime constrained[x: Int]: AnyType where cond = Int
    comptime dropped: __generator_type[x: Int] AnyType = constrained
    comptime used[a: Int]: AnyType = constrained[a]


# CHECK-LABEL: lit.fn @"implicit_nonparametric_where
def implicit_nonparametric_where[cond: Bool]() where cond:
    # CHECK: lit.alias.decl *"constrained{{.*}}": !lit.generator<
    # CHECK-SAME: {<{{.*}}#lit.struct.extract<:!Bool cond, "_mlir_value">{{.*}}>}
    # CHECK-SAME: !AnyType> = <#kgen.gen<!Int>>
    comptime constrained: AnyType where cond = Int
    # CHECK: lit.alias.decl *"used{{.*}}": !AnyType = <!Int>
    comptime used: AnyType = constrained


@fieldwise_init
struct PS[a: Int, b: Int, c: Int](Movable where False):
    pass


# CHECK: lit.alias.decl *"PS_xy3{{.*}}": !lit.generator<<"x": !Int, "y": !Int>meta<!lit.struct<#PS <:!Int *(0,0), :!Int *(0,1), :!Int {:scalar<index> 3}>>>> = <#kgen.gen<@parametric_alias::@PS<:!Int *(0,0), :!Int *(0,1), :!Int {:scalar<index> 3}>>>
comptime PS_xy3[x: Int, y: Int] = PS[x, y, 3]

# CHECK: lit.alias.decl *"PS_21x{{.*}}": !lit.generator<<"x": !Int>meta<!lit.struct<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int *(0,0)>>>> = <#kgen.gen<@parametric_alias::@PS<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int *(0,0)>>>
comptime PS_21x[x: Int] = PS[2, 1, x]

# CHECK: lit.alias.decl *"PS_21xy{{.*}}": !lit.generator<<"x": !Int, "y": !Int>meta<!lit.struct<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int sugar_builtin(apply({{.*}}mul(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">){{.*}}>>>> = <#kgen.gen<@parametric_alias::@PS<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int sugar_builtin(apply({{.*}}mul(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">){{.*}})>>>
comptime PS_21xy[x: Int, y: Int] = PS[2, 1, x * y]


# CHECK: lit.trait.decl @MyTrait
trait MyTrait:
    # CHECK-NEXT: lit.alias.decl *"ParamType{{.*}}": !lit.generator<<"a": !Int>!AnyType>
    comptime ParamType[a: Int]: AnyType


# CHECK: lit.struct.decl @MyStruct
struct MyStruct[a: Int, b: Int](MyTrait, Movable where False):
    # CHECK: lit.alias.decl *"ParamType{{.*}}": !lit.generator<<"a1": !Int>meta<!Int>> = <#kgen.gen<#alias_Int>>
    comptime ParamType[a1: Int] = Int
    # CHECK: kgen.conformance @{{.*}}::@MyTrait {
    # CHECK-NEXT: kgen.witness "ParamType" : !lit.generator<<"a": !Int>!AnyType> = #kgen.gen<!Int>


##===----------------------------------------------------------------------===##
# trailing 'where' clauses
##===----------------------------------------------------------------------===##
# Trailing `where` clauses on a parameterized comptime alias attach body
# constraints to the alias's generator type, just like for structs and
# functions. `conforms_to(T, AnyType)` is provable from `T`'s declared bound,
# so each clause folds to `true` -- the slot still shows the clause attached.

# CHECK: lit.alias.decl *"trailingWhereId{{.*}}": !lit.generator<<"T": !AnyType, {<true,
comptime trailingWhereId[T: AnyType] where conforms_to(T, AnyType) = T

# CHECK: lit.alias.decl *"trailingWhereTyped{{.*}}": !lit.generator<<"T": !AnyType, {<true,
comptime trailingWhereTyped[T: AnyType]: AnyType where conforms_to(
    T, AnyType
) = T

# CHECK: lit.alias.decl *"trailingWhereMulti{{.*}}": !lit.generator<<"T": !AnyType, "U": !AnyType, {<true,
# CHECK-SAME: <true,
comptime trailingWhereMulti[T: AnyType, U: AnyType] where conforms_to(
    T, AnyType
) where conforms_to(U, AnyType) = T


# CHECK: lit.trait.decl @TraitWithTrailingWhereAlias
trait TraitWithTrailingWhereAlias:
    # CHECK-NEXT: lit.alias.decl *"AssocWhere{{.*}}": !lit.generator<<"T": !AnyType, {<true,
    comptime AssocWhere[T: AnyType]: AnyType where conforms_to(T, AnyType)


##===----------------------------------------------------------------------===##
# usages
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"__SomeImpl{{.*}}": !lit.generator<<"Trait": !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable, "T": !kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(0,0)>>!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(0,0)>> = <#kgen.gen<*(0,1)>>
comptime __SomeImpl[Trait: TrivialRegisterPassable, T: Trait] = T
# CHECK: lit.alias.decl *"Some{{.*}}": !lit.generator<<"Trait": !AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable>!lit.generator<<"T": !kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(1,0)>>!kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable *(1,0)>>> = <#kgen.gen<#kgen.gen<*(0,0)>>>
comptime Some[Trait: TrivialRegisterPassable] = __SomeImpl[Trait, ...]

# CHECK: lit.alias.decl *"myDouble{{.*}}": !lit.generator<<"x": !Int>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}mul(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, 2){{.*}})>>
comptime myDouble[x: Int] = myDependentDefaultAdd[x]


# CHECK-LABEL: fn @"expect_two_ints
# CHECK-SAME: <binop: !lit.generator<<"x": !Int, "y": !Int>!Int>>
def expect_two_ints[binop: type_of(myIntAdd)]():
    pass


# CHECK-LABEL: fn @"implicit_conversions()"
def implicit_conversions():
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> #alias_myIntAdd
    expect_two_ints[myIntAdd]()
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> {{.*}}#kgen.gen<
    expect_two_ints[myDefaultAdd]()
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> #kgen.gen<
    expect_two_ints[myIntFMA[z=2, ...]]()
    # CHECK-NEXT: :!lit.generator<<"x": !Int, "y": !Int>!Int> {{.*}}#kgen.gen<
    expect_two_ints[myIntFMA[x=2, ...]]()


# CHECK-LABEL: lit.fn @"test_type_equality()"
def test_type_equality():
    # CHECK-NEXT: %[[PS_345:.*]] = lit.var.decl "ps_345" {{.*}}<#PS <:!Int {:scalar<index> 3}, :!Int {:scalar<index> 4}, :!Int {:scalar<index> 5}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {:scalar<index> 3}, :!Int {:scalar<index> 4}, :!Int {:scalar<index> 5}>(%[[PS_345]])
    var ps_345: PS[3, 4, 5] = PS[idInt[3], myIntAdd[2, 2], myDefaultAdd[4]]()

    # CHECK-NEXT: %[[PS_215:.*]] = lit.var.decl "ps_215" {{.*}}<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 5}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 5}>(%[[PS_215]])
    var ps_215: PS_21x[5] = PS[2, 1, 5]()

    # CHECK-NEXT: %[[PS_216:.*]] = lit.var.decl "ps_216" {{.*}}<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 6}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 6}>(%[[PS_216]])
    var ps_216: PS_21x[6] = PS_21xy[2, 3]()

    # CHECK-NEXT: %[[PS_213:.*]] = lit.var.decl "ps_213" {{.*}}<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 3}>
    # CHECK-NEXT: @PS::@"__init__()"{{.*}}<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 3}>(%[[PS_213]])
    var ps_213: PS_21x[myIntFMA[1, 3, 0]] = PS_xy3[2, 1]()


def two_identical_inputs[T: AnyType](x: T, y: T):
    pass


# CHECK-LABEL: fn @"test_type_inference()"
def test_type_inference():
    # CHECK: lit.call @parametric_alias::@"two_identical_inputs
    # CHECK-SAME: <:!AnyType @parametric_alias::@PS<:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 5}>>
    # CHECK-SAME: "x": !lit.ref<!lit.struct<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 5}>>
    # CHECK-SAME: "y": !lit.ref<!lit.struct<#PS <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}, :!Int {:scalar<index> 5}>>
    two_identical_inputs(PS_21x[5](), PS[2, 1, 5]())


# CHECK-LABEL: fn @"partial_binding()"
def partial_binding():
    # CHECK: lit.alias.decl *"myIntMulPlus3{{.*}}": !lit.generator<<"x": !Int, "y": !Int>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}add(mul(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(0,1), "_mlir_value">), 3){{.*}})>>
    comptime myIntMulPlus3 = myIntFMA[z=3, ...]
    # CHECK: lit.alias.decl *"myIntMul2Plus3{{.*}}": !lit.generator<<"x": !Int>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}add(mul(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, 2), 3){{.*}})>>
    comptime myIntMul2Plus3 = myIntMulPlus3[y=2, ...]
    # CHECK: lit.alias.decl *"myEleven{{.*}}": !Int = <{:scalar<index> 11}>
    comptime myEleven = myIntMul2Plus3[x=4]


# CHECK-LABEL: fn @"nested_generators()"
def nested_generators():
    # CHECK-NEXT: lit.alias.decl *"myCurriedIntAdd{{.*}}": !lit.generator<<"x": !Int>!lit.generator<<"y": !Int>!Int>> = <#kgen.gen<#kgen.gen<sugar_builtin(apply({{.*}}add(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(1,0), "_mlir_value">){{.*}})>>>
    comptime myCurriedIntAdd[x: Int] = myIntAdd[x, _]

    # CHECK-NEXT: lit.alias.decl *"myRenamedCurriedIntAdd{{.*}}": !lit.generator<<"a": !Int>!lit.generator<<"y": !Int>!Int>> = <#kgen.gen<#kgen.gen<sugar_builtin(apply({{.*}}add(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, #lit.struct.extract<:!Int *(1,0), "_mlir_value">){{.*}}>>
    comptime myRenamedCurriedIntAdd[a: Int] = myCurriedIntAdd[a]

    # CHECK-NEXT: lit.alias.decl *"myAdd2{{.*}}": !lit.generator<<"y": !Int>!Int> = <#kgen.gen<sugar_builtin(apply({{.*}}add(#lit.struct.extract<:!Int *(0,0), "_mlir_value">, 2){{.*}}>>
    comptime myAdd2 = myRenamedCurriedIntAdd[2]

    # CHECK-NEXT: lit.alias.decl *"myFive{{.*}}": !Int = <{:scalar<index> 5}>
    comptime myFive = myAdd2[3]

    # CHECK-NEXT: lit.alias.decl *"mySix{{.*}}": !Int = <{:scalar<index> 6}>
    comptime mySix = myRenamedCurriedIntAdd[2][4]


# CHECK: lit.fn @"dependent_function_type[::Bool]()"<cond: !Bool>[mut *"__result__`"](?, %__result__: !lit.ref<:!AnyType cond(#lit.struct.extract<:!Bool cond, "_mlir_value">, !Int, !FloatDyn), mut *"__result__`"> byref_result)
def dependent_function_type[
    cond: Bool
]() -> myTypeSelector[cond, Int, FloatDyn]:
    pass


##===----------------------------------------------------------------------===##
# advanced usage in traits
##===----------------------------------------------------------------------===##
trait TraitWithParamAlias:
    comptime MyReturnType[m: Bool]: AnyType

    def getReturn[m: Bool](self) -> Self.MyReturnType[m]:
        ...


@fieldwise_init
struct MyElemType[m: Bool](Movable where False):
    pass


struct MyConformingStruct(TraitWithParamAlias, Movable where False):
    # The return type is a parametric type that references `m`. This tests that
    # such a parametric type can be instantiated by the trait method `getReturn`
    # to obtain the expected `getReturn` type from this struct.
    comptime MyReturnType[m: Bool]: AnyType = MyElemType[m]

    def getReturn[m: Bool](self) -> Self.MyReturnType[m]:
        return MyElemType[m]()


##===----------------------------------------------------------------------===##
# Infer struct method call with generator
##===----------------------------------------------------------------------===##


@fieldwise_init
struct ParamStructInferFrom[x: Int](Movable where False):
    pass


struct InferMeFromVariousStuff[x: Int](Movable where False):
    def __init__(out self, p: ParamStructInferFrom[Self.x]):
        pass


comptime InferMeFromGenerator[x: Int] = InferMeFromVariousStuff[x]
comptime InferMeFromPartiallyBoundStruct = InferMeFromVariousStuff[_]


def call___init___via_various_kinds_of_things():
    var p = ParamStructInferFrom[1]()
    # CHECK: lit.call @{{.*}}::@InferMeFromVariousStuff::@"__init__
    _ = InferMeFromVariousStuff(p)
    # CHECK: lit.call @{{.*}}::@InferMeFromVariousStuff::@"__init__
    _ = InferMeFromGenerator(p)
    # CHECK: lit.call @{{.*}}::@InferMeFromVariousStuff::@"__init__
    _ = InferMeFromPartiallyBoundStruct(p)


comptime MyStructGenerator[b: Int] = MyStruct[1, b]
# CHECK: lit.alias.decl *"MyStructGeneratorDotA{{.*}}": !Int = <{:scalar<index> 1}>
comptime MyStructGeneratorDotA = MyStructGenerator.a


##===----------------------------------------------------------------------===##
# auto-parameterization propagates body constraints onto the signature
##===----------------------------------------------------------------------===##
# When a parameterized comptime alias with a trailing `where` clause is
# auto-parameterized (its inferred parameters hoisted onto a function), the
# alias's body constraints are attached to the function's generator signature so
# they are checked once the inferred parameters are bound. The constraint shows
# up as a `sugar_preserved` body constraint in the function type. This applies
# to every auto-parameterization form: argument types, value-parameter types,
# and variadic element types.


@fieldwise_init
struct WhereTag[n: Int](Copyable, Movable):
    pass


comptime WherePositive[n: Int, //] where n > 0 = WhereTag[n]


# CHECK: lit.fn @"where_arg
# CHECK-SAME: {<sugar_preserved({{.*}}@std::@builtin::@stubs::@SIMD::@"__gt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}
def where_arg(p: WherePositive):
    pass


# CHECK: lit.fn @"where_param
# CHECK-SAME: {<sugar_preserved({{.*}}@std::@builtin::@stubs::@SIMD::@"__gt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}
def where_param[p: WherePositive]():
    pass


# CHECK: lit.fn @"where_variadic
# CHECK-SAME: {<sugar_preserved({{.*}}@std::@builtin::@stubs::@SIMD::@"__gt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}
def where_variadic(*p: WherePositive):
    pass


##===----------------------------------------------------------------------===##
# Handle type alias with extra constraints
##===----------------------------------------------------------------------===##

struct Iter[Cond: Bool]:
    def __init__(out self):
        pass


struct Collection[Cond: Bool]:
    comptime Alias: AnyType where Self.Cond = Iter[Self.Cond]

    # CHECK: lit.fn @"iter
    # CHECK: lit.var.decl "__call_result_tmp__" synth : !lit.ref<!lit.struct<#Iter <:!Bool Cond>>
    # CHECK: lit.call @parametric_alias::@Iter::@"__init__()"{{.*}}<:!Bool Cond>
    def iter(self) where Self.Cond:
        _ = Self.Alias()


##===----------------------------------------------------------------------===##
# Constrained alias generator in a mangled signature
##===----------------------------------------------------------------------===##
# Mangling prints types without a SharedState, so the constraint's reference to
# the generator's own parameter has no name to substitute. Printing it used to
# read past the (empty) parameter-index bindings and assert.

def constrained_alias_in_mangled_name():
    comptime Constrained[x: Int]: AnyType where x > 0 = Int

    # CHECK-LABEL: lit.fn @"inner
    # CHECK-SAME: __generator_type[{{.*}}] ::AnyType where (
    def inner[G: type_of(Constrained)]():
        pass

    inner[Constrained]()
