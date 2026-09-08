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

# Test struct-level trailing `where` body constraints. These constraints are
# checked during parameter inference (`InferenceState::checkBodyConstraints`)
# at every site that binds the struct's parameters, so a violation surfaces
# at instantiation rather than at first use of an instance.
#
# Each section uses its own struct/function decl so per-section diagnostic
# notes don't bleed across test sites (mirrors the convention in
# `constraint_overload_errors.mojo`).

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Satisfied body constraint at instantiation - positive case
##===----------------------------------------------------------------------===##
# Verifies the new `checkBodyConstraints` path doesn't produce false
# positives on the happy path.


struct SatisfiedStruct[N: Int]
    (Movable where False) where N > 0:
    pass


def use_satisfied_struct():
    var x: SatisfiedStruct[5]


##===----------------------------------------------------------------------===##
# Violated body constraint at instantiation
##===----------------------------------------------------------------------===##
# Binding `N = -1` violates `N > 0`. The new check folds this into a
# binding-time error rather than letting it leak into downstream uses.


# expected-note @below {{'ViolatedStruct' declared here}}
struct ViolatedStruct[N: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    (Movable where False) where N > 0:
    pass


def use_violated_struct():
    # expected-error @below {{violated constraint}}
    var x: ViolatedStruct[-1]


##===----------------------------------------------------------------------===##
# Unprovable body constraint without evidence (Issue #1 regression)
##===----------------------------------------------------------------------===##
# When the caller is itself parametric and offers no assumption that
# discharges the body constraint, the binding is unprovable. The fix in
# `ParamInf::inferForStruct` ensures the strict (non-discarding) path
# returns a single "lacking evidence" error and does NOT silently return
# valid bindings while emitting a top-level error against the same site.


# expected-note @below {{cannot prove constraint}}
struct UnprovableStruct[N: Int]
    # expected-note @below {{constraint declared here needs evidence for}}
    (Movable where False) where N > 0:
    pass


def use_unprovable_struct[K: Int]():
    # expected-error @below {{lacking evidence to prove correctness}}
    var x: UnprovableStruct[K]


##===----------------------------------------------------------------------===##
# Unprovable body constraint dischargeable from caller's assumptions
##===----------------------------------------------------------------------===##
# The caller's `where K > 0` is threaded through as an additional assumption
# during constraint checking, so the body constraint is provable.


struct DischargeableStruct[N: Int]
    (Movable where False) where N > 0:
    pass


def use_dischargeable_struct[K: Int]() where K > 0:
    var x: DischargeableStruct[K]


##===----------------------------------------------------------------------===##
# Body constraint enforced at function-signature type formation
##===----------------------------------------------------------------------===##
# A function signature that mentions a parameterized struct re-enters
# `inferForStruct` to type-check the parameter type itself. Unprovable body
# constraints from that site are collected by the deferral feature and
# discharged later in `TypeCheckedParamList::emitBodyConstraints` using the
# trailing `where` clauses as additional assumptions (see
# `body_constraint_deferral.mojo` for positive-discharge coverage). When no
# trailing `where` clause can discharge the deferred constraint, the
# discharge step surfaces a hard error here. (The function-body analog is
# covered above by `use_dischargeable_struct`, where the trailing `where`
# is already in scope at the binding site.)


struct PositiveOnly[N: Int]
    # expected-note @below {{constraint declared here needs evidence for}}
    (Movable where False) where N > 0:
    pass


# expected-error @below {{lacking evidence to prove correctness}}
# expected-note @below {{add a trailing 'where' clause that requires '(K > Int(0))'}}
def bad_signature_use[K: Int](x: PositiveOnly[K]):
    pass


def call_bad_signature_use():
    var p: PositiveOnly[7]


##===----------------------------------------------------------------------===##
# Multi-parameter body constraint
##===----------------------------------------------------------------------===##
# A constraint that mentions two parameters must be checked once both are
# bound, and yield a hard binding-time error when violated.


struct OrderedPair[A: Int, B: Int]
    (Movable where False) where A < B:
    pass


def use_ordered_pair_satisfied():
    var p: OrderedPair[1, 5]


# expected-note @below {{'ViolatedOrderedPair' declared here}}
struct ViolatedOrderedPair[A: Int, B: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    (Movable where False) where A < B:
    pass


def use_ordered_pair_violated():
    # expected-error @below {{violated constraint}}
    var p: ViolatedOrderedPair[5, 1]


##===----------------------------------------------------------------------===##
# Body constraint references trait conformance
##===----------------------------------------------------------------------===##
# A struct gated on `conforms_to` should only accept type bindings whose
# inferred parameter satisfies the trait. This is the value-level analog of
# the existing trait-conformance verification flow but applied at struct
# instantiation rather than at conformance declaration.


struct OnlyIntableSatisfied[T: AnyType]
    (Movable where False) where conforms_to(T, Intable):
    pass


struct ConcreteIntable(Intable, Movable where False):
    def __int__(self) -> Int:
        return 0


def use_intable_satisfied():
    var x: OnlyIntableSatisfied[ConcreteIntable]


# expected-note @below {{'OnlyIntableViolated' declared here}}
struct OnlyIntableViolated[T: AnyType]
    # expected-note @below {{constraint declared here evaluated to False}}
    (Movable where False) where conforms_to(T, Intable):
    pass


struct NotIntable(Movable where False):
    pass


def use_intable_violated():
    # expected-error @below {{violated constraint}}
    var x: OnlyIntableViolated[NotIntable]


# expected-note @below {{cannot prove constraint}}
struct OnlyIntableUnprovable[T: AnyType]
    # expected-note @below {{constraint declared here needs evidence for}}
    (Movable where False) where conforms_to(T, Intable):
    pass


def use_intable_dischargeable[T: AnyType]() where conforms_to(T, Intable):
    var x: OnlyIntableUnprovable[T]


def use_intable_unprovable[T: AnyType]():
    # expected-error @below {{lacking evidence to prove correctness}}
    var x: OnlyIntableUnprovable[T]
