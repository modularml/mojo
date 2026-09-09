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

# This test checks for diagnostic quality when source location information
# isn't available. It does so by precompiling a package and deleting the
# original source. In such cases the compiler should be able to pretty-print
# useful information for the user from the in-memory MLIR constructs.

# Copy the package to a temporary directory
# RUN: mkdir -p %t
# RUN: cp -r %S/inputs/diags_package %t

# Precompile the package
# RUN: mojo precompile %t/diags_package -o %t/diags_package.mojoc

# Remove the source
# RUN: rm -r %t/diags_package

# RUN: not %mojo-build -I %t %s 2>&1 | FileCheck %s

import diags_package


def my_unprovable_constraints[
    x: Int
]() where diags_package.unfoldable_predicate(x):
    pass


def main():
    # CHECK: error: invalid call to 'fn_missing_constraint': violated constraint
    # CHECK: note: constraint declared here evaluated to False, expected '(n > Int(0))'
    # FIXME: Improve the pretty-printing of this constraint
    # CHECK-NEXT: ($0 > Int(0))
    # CHECK: note: function declared here
    # CHECK-NEXT: def fn_missing_constraint[n: Int]() where (n > Int(0))
    diags_package.fn_missing_constraint[0]()

    # CHECK: error: no matching function in call to 'overloaded_function'
    # CHECK: candidate not viable: missing required argument: 'n'
    # CHECK-NEXT: def overloaded_function(n: Int)
    # CHECK: note: candidate not viable: missing required argument: 'n'
    # CHECK-NEXT: def overloaded_function(n: Float64)
    # CHECK: note: candidate not viable: missing required argument: 'n'
    # CHECK-NEXT: def overloaded_function(n: Int, m: Float64)
    diags_package.overloaded_function()

    # CHECK: error: no matching function in initialization
    # CHECK: note: candidate not viable: failed to infer parameter 'b' of parent struct 'PosOnlyStruct'
    # CHECK-NEXT: def __init__(out self) # note - generated function
    # CHECK: note: struct declared here
    # CHECK-NEXT: struct PosOnlyStruct[a: Int, b: Int, /, c: Int = Int(9)]    # note - synthetic signature
    _ = diags_package.PosOnlyStruct[1, c=9]()

    # CHECK: error: deprecated implicit conversion from 'IntLiteral[1]' to 'DeprecatedImplicitConversion'
    # CHECK: note: call 'DeprecatedImplicitConversion(...)' explicitly
    # CHECK: note: implicit constructor for 'DeprecatedImplicitConversion' declared here
    # CHECK: def __init__(out self, value: Int)    # note - synthetic signature
    _: diags_package.DeprecatedImplicitConversion = 1

    # CHECK: error: invalid call to 'my_unprovable_constraints': lacking evidence to prove correctness
    # CHECK-NEXT: _ = my_unprovable_constraints[0]()
    # CHECK: note: cannot prove constraint for candidate
    # CHECK: note: constraint declared here needs evidence for 'unfoldable_predicate(Int(0))'
    # CHECK-NEXT: ]() where diags_package.unfoldable_predicate(x):
    # CHECK: note: cannot evaluate call to non-builtin function declared here
    # CHECK-NEXT: def unfoldable_predicate(y: Int) -> Bool    # note - synthetic signature
    _ = my_unprovable_constraints[0]()

    # CHECK: error: invalid call to 'unprovable_constraints': lacking evidence to prove correctness
    # CHECK-NEXT: _ = diags_package.unprovable_constraints[0]()
    # CHECK: note: cannot prove constraint for candidate
    # CHECK: note: constraint declared here needs evidence for 'unfoldable_predicate(Int(0))'
    # FIXME: We're not printing anything here, unlike the above. It would be
    #        nice to print the constraint along with the function
    # CHECK: note: cannot evaluate call to non-builtin function declared here
    # CHECK-NEXT: def unfoldable_predicate(y: Int) -> Bool    # note - synthetic signature
    _ = diags_package.unprovable_constraints[0]()

    # CHECK: error: violated constraint
    # CHECK: note: constraint declared here evaluated to False, expected '(a > Int(0))'
    # FIXME: Improve the pretty-printing of this constraint
    # CHECK-NEXT: ($0 > Int(0))
    # CHECK: note: function declared here
    # CHECK-NEXT: def constraint_fn[a: Int, b: Int]() where (a > Int(0))    # note - synthetic signature
    comptime _ = diags_package.constraint_fn[0, 1]


# CHECK: error: name conflict between parameter 'a' in the default trait method and a parameter in the struct
# CHECK-NEXT: struct ConflictStruct[a: Int](diags_package.ConflictTraitName):
# CHECK: note: trait method declared here
# CHECK-NEXT: def test[a: Int](self)    # note - synthetic signature
struct ConflictStruct[a: Int](diags_package.ConflictTraitName):
    pass


# CHECK: error: trait method requirement 'test' has conflicting default implementations in 'ConflictTraitMethod' and 'OtherConflictTraitMethod'; you must implement it manually
# CHECK-NEXT: struct ConflictMethod(diags_package.OtherConflictTraitMethod):
# CHECK: note: original default implementation from trait 'ConflictTraitMethod' here
# CHECK-NEXT: def test(self) -> Bool    # note - synthetic signature
# CHECK: note: conflicting implementation from trait 'OtherConflictTraitMethod' here
# CHECK-NEXT: def test(self) -> Bool    # note - synthetic signature
struct ConflictMethod(diags_package.OtherConflictTraitMethod):
    pass


# CHECK: error: 'StructViolation' does not implement all requirements for 'NoDefaultFunc'
# CHECK-NEXT: struct StructViolation(diags_package.StillNoDefaultFunc):
# CHECK: note: required function 'doSomething' is not implemented
# CHECK-NEXT: def doSomething(self)    # note - synthetic signature
# CHECK: note: trait 'NoDefaultFunc' declared here
# CHECK-NEXT: trait NoDefaultFunc    # note - synthetic signature
struct StructViolation(diags_package.StillNoDefaultFunc):
    def doEverything(self):
        pass


# CHECK: note: ambiguous reference to 'handle': lacking evidence to select candidate
# CHECK: note: cannot prove constraint for candidate
# CHECK: note: constraint declared here needs evidence for 'conforms_to(T, Intable)'
# CHECK: note: candidate is valid but cannot be selected until other candidates are disproved
# CHECK: note: provide evidence for or against the constraints here to aid in candidate selection
# CHECK: note: required by trait method here
# CHECK-NEXT: def handle(self)    # note - synthetic signature
# CHECK: note: trait 'UnprovableCandidateTrait' declared here
# CHECK-NEXT: trait UnprovableCandidateTrait    # note - synthetic signature
struct UnprovableWithValidCandidate[T: Movable](
    Movable,
    diags_package.UnprovableCandidateTrait where conforms_to(T, Copyable),
):
    # Unprovable: Intable is unrelated to Copyable - can't prove or disprove.
    def handle(self) where conforms_to(Self.T, Intable):
        pass

    # Provable: Copyable matches the conformance constraint.
    # But we still error because we can't rule out the above candidate.
    def handle(self) where conforms_to(Self.T, Copyable):
        pass


# CHECK: error: 'StructConformingExplicitlyWithNoMatchingAlias' does not implement all requirements for 'TraitWithMember'
# CHECK: note: required member 'N' is not specified
# FIXME: Show the synthetic comptime member here too
# CHECK: note: trait 'TraitWithMember' declared here
# CHECK-NEXT: trait TraitWithMember    # note - synthetic signature
struct StructConformingExplicitlyWithNoMatchingAlias(
    diags_package.TraitWithMember
):
    pass


# CHECK: error: 'StructConformingExplicitlyWithMismatchedAlias' does not implement all requirements for 'TraitWithMember'
# CHECK: note: comptime member 'N' type 'Bool' does not conform to trait's required type 'Int'
# FIXME: Show the synthetic comptime member here too
# CHECK: note: trait 'TraitWithMember' declared here
# CHECK-NEXT: trait TraitWithMember    # note - synthetic signature
struct StructConformingExplicitlyWithMismatchedAlias(
    diags_package.TraitWithMember
):
    comptime N: Bool = Bool()


# CHECK: error: 'StructConformingExplicitlyWithMemberSameName' does not implement all requirements for 'TraitWithMember'
# CHECK: note: required member 'N' is not specified
# FIXME: Show the synthetic comptime member here too
# CHECK: note: trait 'TraitWithMember' declared here
# CHECK-NEXT: trait TraitWithMember    # note - synthetic signature
struct StructConformingExplicitlyWithMemberSameName(
    diags_package.TraitWithMember
):
    var N: Int
