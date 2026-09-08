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

# Verify that body constraints proved at a binding site are stripped
# from the specialized generator's metadata. Covers:
#   - Full binding: every constraint discharged collapses the wrapper
#     to the rebound body type (eager instantiation).
#   - Partial binding via a `comptime` alias value (`GeneratorAttr`).
#   - Partial binding via a function symbol lifted to a function literal.
#   - Multi-parameter constraints, which may discharge fully, retain
#     fully (rebound), or retain with the bound parameters substituted.

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


##===----------------------------------------------------------------------===##
# Baselines: declared signatures still carry body constraints
##===----------------------------------------------------------------------===##


# `conforms_to(T, AnyType)` is provable from `T`'s declared bound, so the
# clause folds to `true` at the declaration rather than being retained.
# CHECK: lit.alias.decl *"trivial_passthrough{{.*}}": !lit.generator<<"T": !AnyType, {<true,
comptime trivial_passthrough[T: AnyType]
    where conforms_to(T, AnyType) = T


# CHECK: lit.alias.decl *"intable_passthrough{{.*}}": !lit.generator<<"T": !AnyType, {<sugar_preserved({{.*}}conforms_to(:!AnyType *(0,0)
comptime intable_passthrough[T: AnyType]
    where conforms_to(T, Intable) = T


# CHECK: lit.alias.decl *"two_param_passthrough{{.*}}": !lit.generator<<"T": !AnyType, "U": !AnyType, {<sugar_preserved({{.*}}conforms_to(:!AnyType *(0,0)
comptime two_param_passthrough[T: AnyType, U: AnyType]
    where conforms_to(T, Intable) = U


# CHECK-LABEL: lit.fn @"need_a_positive
# CHECK-SAME: <a: !Int, b: !Int, {<sugar_preserved({{.*}}ge(:scalar<index>{{.*}}1))
def need_a_positive[a: Int, b: Int]() where a > 0:
    pass


# CHECK: lit.alias.decl *"ordered_pair{{.*}}": !lit.generator<<"x": !Int, "y": !Int, {<sugar_preserved(
comptime ordered_pair[x: Int, y: Int] where x < y = x + y


struct PlainStruct(Movable where False):
    pass


struct IntableStruct(Intable, Movable where False):
    def __int__(self) -> Int:
        return 0


##===----------------------------------------------------------------------===##
# Full binding -> all constraints discharged -> eager instantiation
##===----------------------------------------------------------------------===##


def use_trivial_dischargeable():
    # CHECK-LABEL: lit.fn @"use_trivial_dischargeable
    # CHECK: lit.alias.decl *"z_trivial{{.*}}": !AnyType = <!PlainStruct>
    comptime z_trivial = trivial_passthrough[PlainStruct]


def use_intable_dischargeable():
    # CHECK-LABEL: lit.fn @"use_intable_dischargeable
    # CHECK: lit.alias.decl *"z_intable{{.*}}": !AnyType = <!IntableStruct>
    comptime z_intable = intable_passthrough[IntableStruct]


# Caller's trailing `where` provides the evidence to discharge the
# constraint even though the binding is parametric.
def use_caller_evidence_discharge[T: AnyType]() where conforms_to(T, Intable):
    # CHECK-LABEL: lit.fn @"use_caller_evidence_discharge[
    # CHECK: lit.alias.decl *"z_caller{{.*}}": !AnyType = <T>
    comptime z_caller = intable_passthrough[T]


# Multi-variable constraint with both parameters bound to literals that
# satisfy `x < y`: the constraint is concrete and provable, so eager
# instantiation collapses the alias to its body value.
def use_multivar_full_dischargeable():
    # CHECK-LABEL: lit.fn @"use_multivar_full_dischargeable
    # CHECK: lit.alias.decl *"sum8{{.*}}": !Int = <{:scalar<index> 8}>
    comptime sum8 = ordered_pair[3, 5]


##===----------------------------------------------------------------------===##
# Partial binding of a `comptime` alias value
##===----------------------------------------------------------------------===##


# Single-variable constraint discharged: the residual generator type
# drops the constraint clause entirely.
def use_alias_partial_binding():
    # CHECK-LABEL: lit.fn @"use_alias_partial_binding()"
    # CHECK: lit.alias.decl *"curried{{.*}}": !lit.generator<<"U": !AnyType>!AnyType>
    comptime curried = two_param_passthrough[IntableStruct, _]


# Multi-variable constraint with only the first parameter bound: the
# constraint still depends on `y`, so it is retained but rebound with
# the bound value substituted in for `x`. The sugared form holds onto
# the original `Int.__lt__({3}, y)` call and the folded scalar form
# canonicalizes `3 < y` to `y >= 4` for integers.
#
# Re-binding the residual `y` with a value that satisfies the rebound
# constraint then discharges it, and eager instantiation collapses the
# alias to the folded body value.
def use_multivar_partial_x_bound():
    # CHECK-LABEL: lit.fn @"use_multivar_partial_x_bound()"
    # CHECK: lit.alias.decl *"curried_x3{{.*}}": !lit.generator<<"y": !Int,
    # CHECK-SAME: <sugar_preserved({{.*}}@std::@builtin::@stubs::@SIMD::@"__lt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}{:scalar<index> 3}, *(0,0)){{.*}}ge(:scalar<index>{{.*}}*(0,0){{.*}}, 4))
    comptime curried_x3 = ordered_pair[3, _]
    # CHECK: lit.alias.decl *"fully_bound{{.*}}": !Int = <{:scalar<index> 8}>
    comptime fully_bound = curried_x3[5]


# Symmetric: bind only the second parameter; the residual constraint
# substitutes `y = 5` and still depends on `x`. The folded scalar form
# is `x < 5`, which doesn't simplify further.
def use_multivar_partial_y_bound():
    # CHECK-LABEL: lit.fn @"use_multivar_partial_y_bound()"
    # CHECK: lit.alias.decl *"curried_y5{{.*}}": !lit.generator<<"x": !Int,
    # CHECK-SAME: <sugar_preserved({{.*}}@std::@builtin::@stubs::@SIMD::@"__lt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}*(0,0), {:scalar<index> 5}){{.*}}lt(:scalar<index>{{.*}}*(0,0){{.*}}, 5))
    comptime curried_y5 = ordered_pair[_, 5]


##===----------------------------------------------------------------------===##
# Partial binding of a function symbol
##===----------------------------------------------------------------------===##


# Binding `a = 5` discharges `a > 0`; the residual function-literal
# generator type drops the constraint clause.
def use_fn_symbol_partial_binding():
    # CHECK-LABEL: lit.fn @"use_fn_symbol_partial_binding()"
    # CHECK: lit.alias.decl *"f5{{.*}}": !lit.generator<<"b": !Int>!kgen.func.literal<
    comptime f5 = need_a_positive[5, _]


##===----------------------------------------------------------------------===##
# `and` / `or` constraints under partial binding
##===----------------------------------------------------------------------===##


# CHECK: lit.alias.decl *"and_pair{{.*}}": !lit.generator<<"x": !Int, "y": !Int, {<sugar_preserved({{.*}}and(
comptime and_pair[x: Int, y: Int] where x > 0 and y > 0 = x + y


# CHECK: lit.alias.decl *"or_pair{{.*}}": !lit.generator<<"x": !Int, "y": !Int, {<sugar_preserved({{.*}}or(
comptime or_pair[x: Int, y: Int] where x > 0 or y > 0 = x + y


# `and` with one conjunct discharged at full binding, other parametric:
# the AND as a whole isn't proved (still depends on `y`), so it's
# retained -- but the discharged conjunct (`5 > 0`) folds to True and
# is dropped from the residual, leaving only the `y > 0` proposition.
def use_and_partial_retain_other():
    # CHECK-LABEL: lit.fn @"use_and_partial_retain_other()"
    # CHECK: lit.alias.decl *"and_partial{{.*}}": !lit.generator<<"y": !Int,
    # CHECK-SAME: <sugar_preserved({{.*}}__gt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}*(0,0), {:scalar<index> 0}){{.*}}ge(:scalar<index>{{.*}}*(0,0){{.*}}, 1))
    comptime and_partial = and_pair[5, _]


# Both `and` conjuncts concrete and provable: the constraint is
# discharged in full and eager instantiation collapses the alias to the
# folded body value (5 + 3 = 8).
def use_and_full_dischargeable():
    # CHECK-LABEL: lit.fn @"use_and_full_dischargeable()"
    # CHECK: lit.alias.decl *"and_full{{.*}}": !Int = <{:scalar<index> 8}>
    comptime and_full = and_pair[5, 3]


# Both `or` disjuncts concrete and at least one provable: the
# constraint is discharged and the alias collapses to the body value
# (-1 + 5 = 4).
def use_or_full_dischargeable():
    # CHECK-LABEL: lit.fn @"use_or_full_dischargeable()"
    # CHECK: lit.alias.decl *"or_full{{.*}}": !Int = <{:scalar<index> 4}>
    comptime or_full = or_pair[-1, 5]


# `or` with the bound disjunct false: short-circuits to the other
# disjunct, which is still parametric, so the constraint is retained
# but mentions only the unbound `y`.
def use_or_partial_one_false_keep_other():
    # CHECK-LABEL: lit.fn @"use_or_partial_one_false_keep_other()"
    # CHECK: lit.alias.decl *"or_partial_neg{{.*}}": !lit.generator<<"y": !Int,
    # CHECK-SAME: <sugar_preserved({{.*}}__gt__(::SIMD[$0, $1],::SIMD[$0, $1])"{{.*}}*(0,0), {:scalar<index> 0}){{.*}}ge(:scalar<index>{{.*}}*(0,0){{.*}}, 1))
    comptime or_partial_neg = or_pair[-1, _]
