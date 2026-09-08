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

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct TakesIntParam[a: Int](Movable where False):
    pass


trait SomeTrait:
    pass


struct ParamType[x: Int](SomeTrait, TrivialRegisterPassable):
    pass


##===----------------------------------------------------------------------===##
# inferred Parameters
##===----------------------------------------------------------------------===##


struct DependentParam[x: Int, y: ParamType[x]](TrivialRegisterPassable):
    pass


def inferred_param_from_arg[x: Int, //](y: ParamType[x]):
    pass


def inferred_param_from_param[x: Int, //, y: ParamType[x]]():
    pass


def inferred_param_variadic[x: Int, //, *y: ParamType[x]]():
    pass


def inferred_with_default[x: Int, //, y: ParamType[x], z: Int = 1]():
    pass


def inferred_trait[T: SomeTrait, //, y: T]():
    pass


def inferred_dependent_param[
    x: Int, y: ParamType[x], //, z: DependentParam[x, y]
]():
    pass


def inferred_partial[x: Int, //, y: Int](z: ParamType[x]):
    pass


def inferred_partial_dependent[x: Int, //, y: Int, z: ParamType[x]]():
    pass


struct InferredStruct[x: Int, //, y: Int, z: ParamType[x]](Movable where False):
    pass


struct InferredStructConversion[
    x: Int, //, y: TrivialRegisterPassable, z: ParamType[x]
](Movable where False):
    pass


# CHECK-LABEL: lit.fn @"test_inferred_params
def test_inferred_params[x: Int, y: ParamType[x], z: DependentParam[x, y]]():
    # CHECK: inferred_param_from_arg{{.*}}<:!Int x>(%0)
    inferred_param_from_arg(y)
    # CHECK: inferred_param_from_param{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y>
    inferred_param_from_param[y]()
    # CHECK: inferred_param_variadic{{.*}}<:!Int x, :param_list<!lit.struct<#ParamType <:!Int x>>> [y, y],
    inferred_param_variadic[y, y]()
    # CHECK: inferred_trait{{.*}}<:!SomeTrait_AnyType @inferred::@ParamType<:!Int x>, :!lit.struct<#ParamType <:!Int x>> y>
    inferred_trait[y]()
    # CHECK: inferred_with_default{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!Int {:scalar<index> 1}>()
    inferred_with_default[y]()
    # CHECK: inferred_with_default{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!Int {:scalar<index> 2}>
    inferred_with_default[y, 2]()
    # CHECK: inferred_dependent_param{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!lit.struct<#DependentParam <:!Int x, :!lit.struct<#ParamType <:!Int x>> y>> z>()
    inferred_dependent_param[z]()
    # CHECK: inferred_dependent_param{{.*}}<:!Int x, :!lit.struct<#ParamType <:!Int x>> y, :!lit.struct<#DependentParam <:!Int x, :!lit.struct<#ParamType <:!Int x>> y>> z>()
    inferred_dependent_param[x=x, z]()

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.generator<<"x": !Int, +>!kgen.func.literal<{{.*}}("z": !lit.struct<#ParamType <:!Int *(0,0)>>)
    comptime partially_bound = inferred_partial[1]
    # CHECK: lit.call tail @inferred::@"inferred_partial{{.*}}"<:!Int x, :!Int {:scalar<index> 1}>(
    partially_bound(y)

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: !lit.generator<<"x": !Int, +, "z": !lit.struct<#ParamType <:!Int *(0,0)>>
    # CHECK-SAME: inferred_partial_dependent{{.*}}<:!Int *(0,0), :!Int {:scalar<index> 1}, :!lit.struct<#ParamType <:!Int *(0,0)>> *(0,1)>>
    comptime partially_bound_dependent = inferred_partial_dependent[1, ...]
    # CHECK-NEXT: !lit.generator<<>!kgen.func.literal<{{.*}}inferred_partial_dependent{{.*}}<:!Int x, :!Int {:scalar<index> 1}, :!lit.struct<#ParamType <:!Int x>> y>>
    comptime fully_bound = partially_bound_dependent[y]

    # CHECK: alias.decl [[PARTIALLY_BOUND:.*]]: meta<!lit.struct<#InferredStruct <:!Int ?, :!Int {:scalar<index> 1}, :!lit.struct<#ParamType <:!Int ?>> ?>,
    # CHECK-SAME: <"x": !Int, +, "z": !lit.struct<#ParamType <:!Int *(0,0)>>>>> = <{{.*}}@InferredStruct<:!Int ?, :!Int {:scalar<index> 1}, :!lit.struct<#ParamType <:!Int ?>> ?>>
    comptime partially_bound_type = InferredStruct[1, ...]
    # CHECK-NEXT: partially_bound_explicit_inferred{{.*}} = <@inferred::@InferredStruct<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}, :!lit.struct<#ParamType <:!Int {:scalar<index> 1}>> ?>>
    comptime partially_bound_explicit_inferred = InferredStruct[x=1, 2, ...]
    # CHECK-NEXT: fully_bound_type{{.*}}<@inferred::@InferredStruct<:!Int x, :!Int {:scalar<index> 1}, :!lit.struct<#ParamType <:!Int x>> y>>
    comptime fully_bound_type = partially_bound_type[y]

    # CHECK-NEXT: lit.var.decl "inferred_type" var : !lit.ref<!lit.struct<#InferredStructConversion <:!Int x, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable !Int, :!lit.struct<#ParamType <:!Int x>> y>>
    var inferred_type: InferredStructConversion[Int, y]


# Multiply should work even though it is @always_inline("builtin")
def mul2_caller[n: Int, t: TakesIntParam[n * 2]]():
    return mul2_callee[t]()


def mul2_callee[n: Int, //, some_t: TakesIntParam[n * 2]]():
    pass


##===----------------------------------------------------------------------===##
# Inferred Self parameters
##===----------------------------------------------------------------------===##


trait FancyTrait(ImplicitlyCopyable):
    def __eq__(self, other: Self) -> Bool:
        ...


struct MyFancyStruct(FancyTrait):
    def __eq__(self, other: Self) -> Bool:
        return False


@fieldwise_init
struct MyOptional[T: ImplicitlyCopyable](Movable where False):
    @__allow_legacy_custom_self_type
    def __eq__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        pass

    # CHECK-LABEL: lit.fn @"__ne__
    @__allow_legacy_custom_self_type
    def __ne__[U: FancyTrait](self: MyOptional[U], rhs: MyOptional[U]) -> Bool:
        # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%self, %rhs)
        return not (self == rhs)


# CHECK-LABEL: lit.fn @"testMyOptional
def testMyOptional(a: MyOptional[MyFancyStruct]):
    # CHECK-NEXT: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a.__eq__(a)
    # CHECK: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = MyOptional.__eq__(a, a)
    # CHECK: lit.call {{.*}}MyOptional::@"__eq__{{.*}}(%a, %a)
    _ = a == a


# CHECK-LABEL: lit.fn @"findall
# CHECK-NEXT: lit.call tail @std::@builtin::@stubs::@Pointer::@"__init__{{.*}}(%self)
struct DefBoxInference(Movable where False):
    def findall(self) raises -> DefBoxInferenceIter[origin_of(self)]:
        return DefBoxInferenceIter[origin_of(self)](Pointer(to=self))


@fieldwise_init
struct DefBoxInferenceIter[
    origin: Origin[],
](Movable where False):
    @implicit
    def __init__(out self, regex: Pointer[DefBoxInference, Self.origin]):
        pass


# MOCO-3326: Improve inference of Origin type from !lit.origin.
# This fails if we can't map "origin_of(list) in the ctor below to Self.origin
# the Origin value.
def test_origin_inference[
    xis_mutable: Bool,
    //,
    xorigin: Origin[mut=xis_mutable],
](ref[xorigin] list: String) -> MyOriginTaking[xorigin]:
    return MyOriginTaking(list)


struct MyOriginTaking[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
](Movable where False):
    def __init__(
        ref[Self.origin] list: String, out self: MyOriginTaking[origin_of(list)]
    ):
        pass
