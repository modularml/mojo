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

# Type refinement inside the branches of a ternary `exp1 if cond else exp2`.
#
# A `conforms_to(T, Trait)` guard must refine `T` inside the corresponding
# branch, matching the behaviour of the `comptime if conforms_to(...)`
# statement form. Keep this test parser-only and focused on refinement.
#
# Related to MOCO-4209.
#
# RUN: %parse-mojo-isolated %s | FileCheck %s


trait StaticExtra:
    @staticmethod
    def static_value() -> Int:
        ...


trait GuardedParam:
    pass


trait OriginalParam:
    pass


trait RefinedParam:
    pass


def accepts_guarded_param[T: GuardedParam]() -> Int:
    return 0


def accepts_original_param[T: OriginalParam]() -> Int:
    return 0


def accepts_refined_param[T: RefinedParam]() -> Int:
    return 0


# The true branch must see `T: StaticExtra` from the ternary condition, so the
# `T.static_value()` call resolves to the trait witness.
# CHECK-LABEL: lit.fn @"ternary_refines_true_branch
# CHECK: kgen.param.if <{{.*}}conforms_to(:!AnyType T, {{.*}}StaticExtra{{.*}})> -> !alias_Int1 {
# CHECK: lit.call tail{{.*}}#kgen.get_witness<:!AnyType T, @ternary_type_refinement::@StaticExtra, "static_value()">
# CHECK: } else {
# CHECK: kgen.param.constant{{.*}}<rebind(:!Int {:scalar<index> 0})>
def ternary_refines_true_branch[T: AnyType]() -> Int:
    return T.static_value() if conforms_to(T, StaticExtra) else 0


# With a negated guard, the refinement attaches to the `else` branch instead.
# CHECK-LABEL: lit.fn @"ternary_negated_refines_else_branch
# CHECK: kgen.param.if <{{.*}}not(conforms_to(:!AnyType T, {{.*}}StaticExtra{{.*}}))> -> !alias_Int1 {
# CHECK: kgen.param.constant{{.*}}<rebind(:!Int {:scalar<index> 0})>
# CHECK: } else {
# CHECK: lit.call tail{{.*}}#kgen.get_witness<:!AnyType T, @ternary_type_refinement::@StaticExtra, "static_value()">
def ternary_negated_refines_else_branch[T: AnyType]() -> Int:
    return 0 if not conforms_to(T, StaticExtra) else T.static_value()


# The guarded type value is refined (downcast) when bound to a stricter
# parameter inside the true branch.
# CHECK-LABEL: lit.fn @"ternary_refines_type_value_binding
# CHECK: lit.call tail @{{.*}}@"accepts_guarded_param{{.*}}<:!AnyType_GuardedParam downcast(:!AnyType T)>
def ternary_refines_type_value_binding[T: AnyType]() -> Int:
    return accepts_guarded_param[T]() if conforms_to(T, GuardedParam) else 0


# Refinement preserves the original declared bound within the branch, and must
# not leak to the unrelated `T` use after the ternary.
# CHECK-LABEL: lit.fn @"ternary_refinement_preserves_and_scopes
# CHECK: lit.call tail @{{.*}}@"accepts_refined_param{{.*}}<:!AnyType_RefinedParam upcast(:!AnyType_OriginalParam_RefinedParam downcast(:!AnyType_OriginalParam T))>
# CHECK: } {elseIsolated}
# CHECK-NOT: downcast
# CHECK: lit.call tail @{{.*}}@"accepts_original_param{{.*}}<:!AnyType_OriginalParam T>
def ternary_refinement_preserves_and_scopes[T: OriginalParam]() -> Int:
    return (
        accepts_refined_param[T]() if conforms_to(T, RefinedParam) else 0
    ) + accepts_original_param[T]()


# The RHS of a `comptime` binding is a parameter-context expression; the exact
# form from the issue must refine there too (the true value resolves to the
# `static_value` witness under the `conforms_to` condition).
# CHECK-LABEL: lit.fn @"ternary_refines_in_comptime_binding
# CHECK: lit.alias.decl {{.*}}"value{{.*}} = <cond(
# CHECK-SAME: #kgen.get_witness<:!AnyType T, @ternary_type_refinement::@StaticExtra, "static_value()">
def ternary_refines_in_comptime_binding[T: AnyType]() -> Int:
    comptime value = T.static_value() if conforms_to(T, StaticExtra) else 0
    return value
