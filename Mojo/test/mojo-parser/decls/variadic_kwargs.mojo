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
#
# Variadic keyword arguments are tested here, because some attributes and types
# need to be checked, and a separate file makes it easier.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"variadic_kwargs
# CHECK-SAME: %a: !Int, %b: !Int, %args: !lit.ref<!lit.struct<#VariadicList{{.*}}> imm_mem|pos_vararg, *, %c: !Int, %d: !Int,
# CHECK-SAME: %kwargs: !lit.ref<!lit.struct<#StringDict <:!AnyType_Copyable_ImplicitlyCopyable_Movable !Int>>, mut {{.*}}> owned_in_mem|kw_vararg)
def variadic_kwargs(a: Int, b: Int, *args: Int, c: Int, d: Int, var **kwargs: Int):
    pass


# CHECK-LABEL: lit.fn @"variadic_kwargs_def_with_type
def variadic_kwargs_def_with_type(var **kwargs: Int) raises:
    pass


# CHECK-SAME: (*, %kwargs: !lit.ref<!lit.struct<#StringDict <:!AnyType_Copyable_ImplicitlyCopyable_Movable !Int>>, mut {{.*}}> owned_in_mem|kw_vararg,
def takes_int_variadic_kwargs(var **kwargs: Int):
    pass


def takes_int_variadic_kwargs_multiline(
    var **kwargs: Int,
):
    pass


# CHECK-LABEL: lit.fn @"test_variadic_kwargs
def test_variadic_kwargs():
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl
    # CHECK-SAME: !lit.struct<#StringDict <:!AnyType_Copyable_ImplicitlyCopyable_Movable !Int>>
    # CHECK: lit.call {{.*}}@StringDict::@"__init__{{.*}}(%[[DICT_VAR]])

    # CHECK: %[[X_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "x">
    # CHECK: %[[X_VAL:.*]] = lit.var.decl {{.*}}!Int,
    # CHECK: %[[IDX9:.*]] = kgen.param.constant: !Int = <{:scalar<index> 9}>
    # CHECK: lit.ref.store %[[IDX9]], %[[X_VAL]]
    # CHECK: lit.call {{.*}}@StringDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[X_KEY]], %[[X_VAL]])

    # CHECK: %[[S_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "stuff">
    # CHECK: %[[S_VAL:.*]] = lit.var.decl {{.*}}!Int,
    # CHECK: %[[IDX8:.*]] = kgen.param.constant: !Int = <{:scalar<index> 8}>
    # CHECK: lit.ref.store %[[IDX8]], %[[S_VAL]]
    # CHECK: lit.call {{.*}}@StringDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[S_KEY]], %[[S_VAL]])

    # CHECK: lit.call {{.*}}@"takes_int_variadic_kwargs{{.*}}(%[[DICT_VAR]])
    takes_int_variadic_kwargs(x=9, stuff=8)

# MOCO-2199 - Forwarding of kwargs
# CHECK-LABEL: lit.fn @"pass_kwargs
def pass_kwargs(var **kwargs: Int):
    # CHECK-NEXT: lit.ownership.use %kwargs
    # CHECK-NEXT: lit.call {{.*}}@"takes_int_variadic_kwargs_multiline{{.*}}(%kwargs)
    takes_int_variadic_kwargs_multiline(**kwargs^)


# `*` and `**` unpacks combine with ordinary positional and keyword operands.
# CHECK-LABEL: lit.fn @"pass_both
def pass_both(*args: Int, var **kwargs: Int):
    # CHECK: lit.call {{.*}}@"variadic_kwargs{{.*}}, %args, %{{.*}}, %{{.*}}, %kwargs)
    variadic_kwargs(1, 2, *args, c=3, d=4, **kwargs^)


# A literal keyword binding its own named parameter combines with a `**`
# splat, with no `*` unpack involved.
# CHECK-LABEL: lit.fn @"pass_named_and_kwargs
def pass_named_and_kwargs(var **kwargs: Int):
    # CHECK: lit.call {{.*}}@"named_and_kwargs{{.*}}(%{{[0-9]+}}, %{{[0-9]+}}, %kwargs)
    named_and_kwargs(1, named=2, **kwargs^)


def named_and_kwargs(x: Int, *, named: Int, var **kwargs: Int):
    pass


# A literal keyword after the `**` splat still binds its own named parameter.
# CHECK-LABEL: lit.fn @"pass_kw_after_splat
def pass_kw_after_splat(var **kwargs: Int):
    # CHECK: lit.call {{.*}}@"named_and_kwargs{{.*}}(%{{[0-9]+}}, %{{[0-9]+}}, %kwargs)
    named_and_kwargs(1, **kwargs^, named=2)


# A `**` splat combines with a variadic pack, whether the pack is forwarded
# whole or built element-wise.
def takes_pack_and_kwargs[*Ts: Intable](*pack: *Ts, var **kwargs: Int):
    pass


# CHECK-LABEL: lit.fn @"pass_pack_and_kwargs
def pass_pack_and_kwargs[*Ts: Intable](*pack: *Ts, var **kwargs: Int):
    # CHECK: %[[PACK:.*]] = kgen.rebind %pack
    # CHECK: lit.call {{.*}}@"takes_pack_and_kwargs{{.*}}(%[[PACK]], %kwargs)
    takes_pack_and_kwargs(*pack, **kwargs^)


# CHECK-LABEL: lit.fn @"pass_elements_and_kwargs
def pass_elements_and_kwargs(var **kwargs: Int):
    # CHECK: lit.call {{.*}}@VariadicPack::@"__init__
    # CHECK: lit.call {{.*}}@"takes_pack_and_kwargs{{.*}}(%{{.*}}, %kwargs)
    takes_pack_and_kwargs(1, 2, **kwargs^)


struct Forwarder:
    def m(self, *args: Int, var **kwargs: Int):
        pass


# A method call forwards both packed variadics, with `self` prepended.
# CHECK-LABEL: lit.fn @"method_forward
def method_forward(s: Forwarder, *args: Int, var **kwargs: Int):
    # CHECK: lit.call {{.*}}@Forwarder::@"m{{.*}}(%s, %args, %kwargs)
    s.m(*args, **kwargs^)


def defaulted_named(*, named: Int = 5, var **kwargs: Int):
    pass


# A lone `**` splat against a defaulted keyword-only parameter: the default
# is materialized and the dict forwards whole.
# CHECK-LABEL: lit.fn @"lone_splat_defaulted
def lone_splat_defaulted(var **kwargs: Int):
    # CHECK: %[[DEF:.*]] = kgen.param.constant: !Int = <{:scalar<index> 5}>
    # CHECK: lit.call {{.*}}@"defaulted_named{{.*}}(%[[DEF]], %kwargs)
    defaulted_named(**kwargs^)


trait SomeTrait(ImplicitlyCopyable):
    pass


def infers_param_from_kwargs[T: SomeTrait](var **kwargs: T):
    pass


@fieldwise_init
struct MemOnly(SomeTrait):
    pass


# CHECK-LABEL: lit.fn @"test_variadic_kwargs_param_inference
def test_variadic_kwargs_param_inference():
    # CHECK: %s = lit.var.decl "s" var : !lit.ref<!MemOnly,
    # CHECK: lit.call {{.*}}MemOnly::@"__init__{{.*}}(%s)
    var s = MemOnly()

    # CHECK: %[[M:.*]] = lit.var.decl
    # CHECK: lit.call {{.*}}MemOnly::@"__init__{{.*}}(%[[M]])

    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl {{.*}}#StringDict <:!AnyType_Copyable_ImplicitlyCopyable_Movable !MemOnly>
    # CHECK: lit.call {{.*}}@StringDict::@"__init__{{.*}}(%[[DICT_VAR]])
    # CHECK: %[[Y_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "y">

    # CHECK: lit.call {{.*}}@StringDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[Y_KEY]], %[[M]])

    # CHECK: %[[S:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!MemOnly,
    # CHECK: lit.memcpy %s, %[[S]] : <!MemOnly,
    # CHECK: %[[Z_KEY:.*]] = kgen.param.constant: {{.*}}#StringLiteral <:string "z">
    # CHECK: lit.call {{.*}}@StringDict::@"_insert{{.*}}(%[[DICT_VAR]], %[[Z_KEY]], %[[S]])
    infers_param_from_kwargs(y=MemOnly(), z=s)


# COM: test that the inferred type of variables is correct when the initializer
# COM: expression has variadic keyword arguments.
# COM: Issue https://github.com/modularml/modular/issues/35215
def takes_kw(var **kwargs: MemOnly) -> Int:
    return 0


# CHECK-LABEL: lit.fn @"test_takes_kw_in_assignment
def test_takes_kw_in_assignment(x: MemOnly):
    # CHECK: %[[DICT_VAR:.*]] = lit.var.decl{{.*}}#StringDict <:!AnyType_Copyable_ImplicitlyCopyable_Movable !MemOnly>
    # CHECK: %[[RES:.*]] = lit.call {{.*}}@"takes_kw{{.*}}(%[[DICT_VAR]])
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<:meta<!Int> #alias_Int,
    # CHECK: lit.ref.store %[[RES]], %b
    var b = takes_kw(y=x, z=x)
