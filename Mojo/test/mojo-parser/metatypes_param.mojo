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

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @Param
@fieldwise_init
struct Param[x: Int](TrivialRegisterPassable):
    comptime value = Self.x

    @staticmethod
    def foo():
        pass

    # COM: Test self type of parametric struct.
    # CHECK-LABEL: lit.fn @"self_type
    # CHECK-SAME: -> !lit.struct<[[SELF:.*]]>
    @staticmethod
    def self_type() -> Self:
        pass


# CHECK-LABEL: lit.struct.decl @TwoParam
@fieldwise_init
struct TwoParam[x: Int, y: Int](TrivialRegisterPassable):
    comptime first = Self.x
    comptime second = Self.y

    @staticmethod
    def foo():
        pass


# CHECK-LABEL: fully_bound_alias
def fully_bound_alias():
    # COM: Test alias to a fully bound parametric type.
    # CHECK: BoundType{{.*}}: meta<!lit.struct<{{.*}}Param <{{.*}}1{{.*}}>>> = <{{.*}}@Param<{{.*}}1{{.*}}>>
    comptime BoundType = Param[1]
    # CHECK: alias_value{{.*}} = <sugar_member_alias(!alias_BoundType1, "value", {:scalar<index> 1})>
    comptime alias_value = BoundType.value
    # CHECK: call {{.*}}@Param::@"foo()"<{{.*}}1{{.*}}>
    BoundType.foo()
    # CHECK: call {{.*}}@Param::@"self_type()"<{{.*}}1{{.*}}>{{.*}} -> !lit.struct<#Param <{{.*}}1{{.*}}>>
    _ = BoundType.self_type()


# CHECK-LABEL: unbound_alias
def unbound_alias():
    # COM: Test alias to a fully unbound parametric type.
    # CHECK: [[UNBOUND:\*"Unbound.*]]: meta<!lit.struct<#Param <:!Int ?>, <"x": !Int>>> = <{{.*}}@Param<:!Int ?>>
    comptime Unbound = Param
    # CHECK: unbound_value{{.*}} = <sugar_member_alias(!lit.struct<#Param <:!Int {:scalar<index> 2}>>, "value", {:scalar<index> 2})>
    comptime unbound_value = Unbound[2].value
    # CHECK: call {{.*}}@Param::@"foo()"<:!Int {:scalar<index> 2}>
    Unbound[2].foo()
    # CHECK: unbound_function{{.*}}: !lit.generator<<"x": !Int, +>!kgen.func.literal<{{.*}}@Param::@"foo()"<:!Int *(0,0)>>
    comptime unbound_function = Unbound.foo

    # COM: Test fully unbound alias can be fully bound.
    # CHECK: BoundFromUnbound{{.*}}: meta<!lit.struct<#Param <:!Int {:scalar<index> 1}>>> =
    # CHECK-SAME: <@metatypes_param::@Param<:!Int {:scalar<index> 1}>>
    comptime BoundFromUnbound = Unbound[1]


# CHECK-LABEL: partially_bound_alias
def partially_bound_alias():
    # COM: Test partially binding a type.
    # CHECK: [[PBOUND:\*"PartiallyBound.*]]: meta<!lit.struct<#TwoParam <:!Int {:scalar<index> 1}, :!Int ?>, <"y": !Int>>> = <{{.*}}@TwoParam<:!Int {:scalar<index> 1}, :!Int ?>>
    comptime PartiallyBound = TwoParam[1, _]

    # COM: Test taking a function from a partially bound type.
    # CHECK: [[PBOUND_FN:\*"PartiallyBounddef.*]]: !lit.generator<<"y": !Int, +>!kgen.func.literal<{{.*}}@TwoParam::@"foo()"<:!Int {:scalar<index> 1}, :!Int *(0,0)>>
    comptime PartiallyBounddef = PartiallyBound.foo
    # CHECK: FullyBounddef{{.*}}@metatypes_param::@TwoParam::@"foo()"<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>>
    comptime FullyBounddef = PartiallyBounddef[y=2]

    # COM: Test fully binding a partially bound type.
    # CHECK: *"BoundFromPartial`3": meta<!lit.struct<#TwoParam <:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>>> =
    # CHECK-SAME: <@metatypes_param::@TwoParam<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>>
    comptime BoundFromPartial = PartiallyBound[2]
    # CHECK: first{{.*}} = <sugar_member_alias(!alias_BoundFromPartial1, "first", {:scalar<index> 1})>
    comptime first = BoundFromPartial.first
    # CHECK: second{{.*}} = <sugar_member_alias(!alias_BoundFromPartial1, "second", {:scalar<index> 2})>
    comptime second = BoundFromPartial.second
    # CHECK: def_from_bound{{.*}}@metatypes_param::@TwoParam::@"foo()"<:!Int {:scalar<index> 1}, :!Int {:scalar<index> 2}>>
    comptime def_from_bound = BoundFromPartial.foo


# CHECK-LABEL: partially_bound_kw
def partially_bound_kw():
    # COM: Test partially binding the parameters out-of-order with keywords.
    # CHECK: TwoParam <:!Int ?, :!Int {:scalar<index> 1}>
    comptime PartiallyBound = TwoParam[y=1, ...]
    # CHECK: TwoParam <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}>
    comptime FullyBound = PartiallyBound[x=2]

    # COM: Test emission of fully bound type.
    # CHECK: expr_type{{.*}}#TwoParam <:!Int {:scalar<index> 2}, :!Int {:scalar<index> 1}>
    var expr_type: FullyBound


# CHECK-LABEL: lit.fn @"partial_autoparam
# CHECK-SAME: <?, [[X:.*]]: !Int>(%value: !lit.struct<#TwoParam <:!Int [[X]], :!Int {:scalar<index> 1}>>
def partial_autoparam(value: TwoParam[y=1, ...]):
    comptime first = value.x
    comptime second = value.y


# CHECK-LABEL: lit.struct.decl @ParamVarArg<{{.*}}, F: !Int, I: !lit.struct<#ParameterList{{.*}} pos_vararg>
@fieldwise_init
struct ParamVarArg[F: Int, *I: Int](TrivialRegisterPassable):
    # CHECK-LABEL: lit.fn @"self_type
    # CHECK-SAME: #ParamVarArg <:param_list<!Int> {{.*}}, :!Int F, :!lit.struct<#ParameterList {{.*}} I>
    @staticmethod
    def self_type() -> Self:
        # CHECK: lit.alias.decl {{.*}}Unbound{{.*}}: {{.*}}ParamVarArg <:param_list<!Int> ?, :!Int ?, {{.*}}>, <{{.*}}, "F": !Int, "I": !lit.struct<#ParameterList{{.*}} pos_vararg>>
        comptime Unbound = ParamVarArg
        # CHECK: lit.alias.decl {{.*}}BoundSome{{.*}}: {{.*}}ParamVarArg <:param_list<!Int> ?, :!Int {:scalar<index> 1}, 
        comptime BoundSome = Unbound[1, ...]
        # CHECK: lit.alias.decl {{.*}}BoundFinal{{.*}}: {{.*}}ParamVarArg <:param_list<!Int> [{:scalar<index> 3}, {:scalar<index> 4}], :!Int {:scalar<index> 1}, 
        comptime BoundFinal = BoundSome[3, 4]

        # CHECK: BoundMore{{.*}}: {{.*}}ParamVarArg <:param_list<!Int> [{:scalar<index> 2}, {:scalar<index> 1}], :!Int {:scalar<index> 1}, 
        comptime BoundMore = Unbound[1, 2, 1]


struct ParamType[x: __mlir_type.index](RegisterPassable):
    pass


struct DependentParam[
    a: __mlir_type.index, b: __mlir_type.index, c: ParamType[b]
](Movable where False):
    pass


# CHECK-LABEL: lit.fn @"direct_binding
def direct_binding():
    # Test direct bind of StructType
    # CHECK: alias.decl *"a{{.*}} meta<!lit.struct<#DependentParam <?, ?, :!lit.struct<#ParamType <?>> ?>, <"a": index, "b": index, "c": !lit.struct<#ParamType <*(0,1)>>
    comptime a = DependentParam
    # CHECK: alias.decl *"b{{.*}} meta<!lit.struct<#DependentParam <1, ?, :!lit.struct<#ParamType <?>> ?>, <"b": index, "c": !lit.struct<#ParamType <*(0,0)>>>>
    comptime b = DependentParam[__mlir_attr.`1:index`, ...]
    # CHECK: alias.decl *"c{{.*}} meta<!lit.struct<#DependentParam <1, 2, :!lit.struct<#ParamType <2>> ?>, <"c": !lit.struct<#ParamType <2>>>
    comptime c = DependentParam[__mlir_attr.`1:index`, __mlir_attr.`2:index`, _]

    # Test partial bind of StructType
    # CHECK: alias.decl *"d{{.*}} meta<!lit.struct<#DependentParam <1, 2, :!lit.struct<#ParamType <2>> ?>, <"c": !lit.struct<#ParamType <2>>>>
    comptime d = DependentParam[__mlir_attr.`1:index`, ...][
        __mlir_attr.`2:index`, _
    ]


# CHECK: lit.fn @"indirect_binding
def indirect_binding():
    # CHECK: lit.alias.decl [[a:\*"a.*"]]: meta
    comptime a = DependentParam
    # Test indirect binds.
    # CHECK: lit.alias.decl [[b:\*"b.*"]]: meta<!lit.struct<#DependentParam <1, ?, :!lit.struct<#ParamType <?>> ?>, <"b": index, "c": !lit.struct<#ParamType <*(0,0)>>{{.*}}>> = <@metatypes_param::@DependentParam<1, ?, :!lit.struct<#ParamType <?>> ?>>
    comptime b = a[__mlir_attr.`1:index`, ...]
    # CHECK: lit.alias.decl [[c:\*"c.*"]]: meta<!lit.struct<#DependentParam <1, 2, :!lit.struct<#ParamType <2>> ?>, <"c": !lit.struct<#ParamType <2>>{{.*}}>> = <@metatypes_param::@DependentParam<1, 2, :!lit.struct<#ParamType <2>> ?>>
    comptime c = b[__mlir_attr.`2:index`, _]
    # CHECK: lit.alias.decl [[d:\*"d.*"]]: meta<!lit.struct<#DependentParam <1, 2, :!lit.struct<#ParamType <2>> *?>>> = <@metatypes_param::@DependentParam<1, 2, :!lit.struct<#ParamType <2>> *?>>
    comptime d = c[
        __mlir_attr[`#kgen.unknown : `, ParamType[__mlir_attr.`2:index`]]
    ]
