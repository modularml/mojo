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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Auto-Predication During Signature Emission
#
# When a parametric struct appears in a function or struct signature
# (parameter declaration, function argument, return type, or thrown type),
# binding the struct's parameters can produce unprovable body constraints:
# the trailing `where` clauses are parsed AFTER the signature, so their
# assumptions aren't yet in scope at the binding site.
#
# The auto-predication feature collects these unprovable body constraints into a
# deferral context during signature emission and ensures that each one is
# implied by any body constraints on the signature.
#
# Wiring covered by these tests:
#   - Function parameter declaration types (`TypeCheckedParamList::create`)
#   - Function argument types         (`typeCheckOneArgument`)
#   - Function return types           (`typeCheckResult`)
#   - Function `raises T` thrown types (`typeCheckResult` errorType branch)
#   - Struct parameter declaration types (`TypeCheckedParamList::create`)
#
# Per-parameter `where` constraints are intentionally NOT deferred (they
# would change candidate viability during overload resolution); the cases
# below only cover body (trailing) constraints.
##===----------------------------------------------------------------------===##


##===----------------------------------------------------------------------===##
# Discharged via trailing `where` on a function
##===----------------------------------------------------------------------===##
# A parametric binding whose body constraint is unprovable at the binding
# site is deferred and successfully discharged once the trailing `where`
# clause has been folded into the decl scope's known assumptions. The
# cases below should type-check without diagnostics.


struct PositiveOnly[N: Int] (Movable where False) where N > 0:
    pass


struct OnlyIntable[T: AnyType] (Movable where False) where conforms_to(T, Intable):
    pass


# Parameter declaration position.
def discharged_from_param_int[K: Int, X: PositiveOnly[K]]() where K > 0:
    pass


def discharged_from_param_trait[T: AnyType, X: OnlyIntable[T]]()
    where conforms_to(T, Intable):
    pass


# Argument type position.
def discharged_from_arg_int[K: Int](x: PositiveOnly[K]) where K > 0:
    pass


def discharged_from_arg_trait[T: AnyType](x: OnlyIntable[T])
    where conforms_to(T, Intable):
    pass


# Return type position.
def discharged_from_ret[K: Int]() -> PositiveOnly[K] where K > 0:
    pass


# Thrown type position (`raises T`).
def discharged_from_throws[K: Int]() raises PositiveOnly[K] where K > 0:
    pass


##===----------------------------------------------------------------------===##
# Discharged via trailing `where` on a struct
##===----------------------------------------------------------------------===##
# Structs go through the same flow as functions, so deferral and discharge apply
# identically to struct parameter declaration positions.


struct DischargedStruct[K: Int, X: PositiveOnly[K]] (Movable where False) where K > 0:
    pass


struct DischargedStructTrait[T: AnyType, X: OnlyIntable[T]]
    (Movable where False) where conforms_to(T, Intable):
    pass


##===----------------------------------------------------------------------===##
# Multiple deferrals — all discharged
##===----------------------------------------------------------------------===##
# When a signature introduces multiple deferred body constraints, the
# discharge loop iterates over each independently and discharges those
# implied by the trailing `where`.


def all_discharged[A: Int, B: Int,
                   X: PositiveOnly[A], Y: PositiveOnly[B]]()
    where A > 0 where B > 0:
    pass


struct AllDischargedStruct[A: Int, B: Int,
                            X: PositiveOnly[A], Y: PositiveOnly[B]]
    (Movable where False) where A > 0 where B > 0:
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function parameter declaration position
##===----------------------------------------------------------------------===##


struct PosForParam[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(K > Int(0))'}}
def undischarged_param[K: Int,
                       # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                       X: PosForParam[K]]():
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function argument position
##===----------------------------------------------------------------------===##


struct PosForArg[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(K > Int(0))'}}
def undischarged_arg[K: Int](
        # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
        x: PosForArg[K]):
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function return type position
##===----------------------------------------------------------------------===##


struct PosForRet[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
# expected-note @below {{add a trailing 'where' clause that requires '(K > Int(0))'}}
def undischarged_ret[K: Int]() -> PosForRet[K]:
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function `raises T` thrown type position
##===----------------------------------------------------------------------===##


struct PosForThrows[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
# expected-note @below {{add a trailing 'where' clause that requires '(K > Int(0))'}}
def undischarged_throws[K: Int]() raises PosForThrows[K]:
    pass


##===----------------------------------------------------------------------===##
# Unprovable in struct parameter declaration position
##===----------------------------------------------------------------------===##


struct PosForStruct[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(K > Int(0))'}}
struct UndischargedStruct[K: Int,
                          # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                          X: PosForStruct[K]](Movable where False):
    pass


##===----------------------------------------------------------------------===##
# Multiple deferrals — partial discharge
##===----------------------------------------------------------------------===##
# Each deferred constraint is re-checked independently; the discharge loop
# reports an error per still-unprovable constraint while silently
# discharging the rest. Here only `A > 0` is in the where clause, so the
# `B > 0` constraint from `PosForPartial[B]` remains unprovable.


struct PosForPartial[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(B > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(B > Int(0))'}}
def partial_discharge[A: Int, B: Int, X: PosForPartial[A],
                      # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                      Y: PosForPartial[B]]() where A > 0:
    pass


##===----------------------------------------------------------------------===##
# Unprovable trait conformance — function parameter declaration
##===----------------------------------------------------------------------===##


struct IntableForParam[T: AnyType]
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Intable)'}}
    (Movable where False) where conforms_to(T, Intable):
    pass


# expected-note @below {{add a trailing 'where' clause that requires 'conforms_to(T, Intable)'}}
def undischarged_trait_param[T: AnyType,
                             # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                             X: IntableForParam[T]]():
    pass


##===----------------------------------------------------------------------===##
# Unprovable trait conformance — function argument
##===----------------------------------------------------------------------===##


struct IntableForArg[T: AnyType]
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Intable)'}}
    (Movable where False) where conforms_to(T, Intable):
    pass


# expected-note @below {{add a trailing 'where' clause that requires 'conforms_to(T, Intable)'}}
def undischarged_trait_arg[T: AnyType](
        # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
        x: IntableForArg[T]):
    pass


##===----------------------------------------------------------------------===##
# Multiple deferrals — none discharged
##===----------------------------------------------------------------------===##
# When no trailing `where` clause is present, every deferred body constraint
# remains unprovable and the discharge loop emits one error per constraint.


struct PosForNoneA[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(A > Int(0))'}}
    (Movable where False) where N > 0:
    pass


struct PosForNoneB[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(B > Int(0))'}}
    (Movable where False) where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(A > Int(0))'}}
# expected-note @below {{add a trailing 'where' clause that requires '(B > Int(0))'}}
def all_undischarged[A: Int,
                     # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                     X: PosForNoneA[A],
                     B: Int,
                     # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                     Y: PosForNoneB[B]]():
    pass


##===----------------------------------------------------------------------===##
# Multi-candidate deferral rejection
##===----------------------------------------------------------------------===##
# When the deferral sink is installed but overload resolution leaves more
# than one body-constraint-inconclusive candidate, deferral is rejected
# (we cannot commit to a single candidate). The normal inconclusiveness
# error fires, plus an extra note explaining why deferral did not apply.


# expected-note @below {{cannot prove constraint for candidate}}
def mc_target[K: Int](dummy: Int) -> Int
    # expected-note @below {{constraint declared here}}
    where K > 0:
    return dummy


# expected-note @below {{cannot prove constraint for candidate}}
def mc_target[K: Int](dummy: Int) -> Int
    # expected-note @below {{constraint declared here}}
    where K < 100:
    return dummy


struct MCReceiver[i: Int](Movable where False):
    pass


def multi_cand_deferral_rejected[K: Int](
    # expected-error @below {{ambiguous call to 'mc_target': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    # expected-note @below {{body constraints cannot be deferred because more than one candidate is inconclusive}}
    x: MCReceiver[mc_target[K](0)]):
    pass


##===----------------------------------------------------------------------===##
# Trait-bounded parameter slot conformance — discharged via trailing `where`
##===----------------------------------------------------------------------===##
# Distinct from the cases above: here the *slot itself* is trait-bounded
# (`T: Copyable`), so binding a value whose current bound is weaker (e.g. a
# `Movable` type parameter) fails the metatype conversion at the binding site
# rather than producing a nested body constraint. The conformance obligation is
# reified and deferred, then discharged by the trailing
# `where conforms_to(..., Copyable)`. These cases should type-check without
# diagnostics.


struct NeedsCopyable[T: Copyable](Movable where False):
    pass


# Return type position.
def slot_discharged_ret[T: Movable]() -> NeedsCopyable[T]
    where conforms_to(T, Copyable):
    pass


# Argument type position.
def slot_discharged_arg[T: Movable](x: NeedsCopyable[T])
    where conforms_to(T, Copyable):
    pass


# Parameter declaration position.
def slot_discharged_param[T: Movable, X: NeedsCopyable[T]]()
    where conforms_to(T, Copyable):
    pass


# Thrown type position (`raises T`).
def slot_discharged_throws[T: Movable]() raises NeedsCopyable[T]
    where conforms_to(T, Copyable):
    pass


# Method form (the MOCO-4190 reproducer): `Self.T` from the parent struct.
struct SlotContainer[T: Movable](Movable where False):
    def get(self) -> NeedsCopyable[Self.T] where conforms_to(Self.T, Copyable):
        pass


# Struct parameter declaration position.
struct SlotDischargedStruct[T: Movable, X: NeedsCopyable[T]]
    (Movable where False) where conforms_to(T, Copyable):
    pass


##===----------------------------------------------------------------------===##
# Trait-bounded parameter slot conformance — undischarged
##===----------------------------------------------------------------------===##
# Without a trailing `where`, the deferred conformance obligation remains
# unprovable and the discharge loop reports it at the binding site, pointing the
# user at the `where` clause they should add.


# expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
# expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Copyable)'}}
# expected-note @below {{add a trailing 'where' clause that requires 'conforms_to(T, Copyable)'}}
def slot_undischarged_ret[T: Movable]() -> NeedsCopyable[T]:
    pass


##===----------------------------------------------------------------------===##
# Trait-bounded parameter slot conformance — concrete receiver stays a hard error
##===----------------------------------------------------------------------===##
# A concrete type that does not conform to the slot's trait can never be rescued
# by a trailing `where`, so deferral must NOT apply: the precise binding-site
# diagnostic is preserved. This guards the type-parameter gate on deferral.


struct MoveOnly(Movable):
    pass


# expected-note @below {{'NeedsCopyableConcrete' declared here}}
struct NeedsCopyableConcrete[T: Copyable](Movable where False):
    pass


# expected-error @below {{'NeedsCopyableConcrete' parameter 'T' has 'Copyable' type, but value has type 'AnyStruct[MoveOnly]'}}
def concrete_non_copyable() -> NeedsCopyableConcrete[MoveOnly]:
    pass


##===----------------------------------------------------------------------===##
# Trait-bounded slot conformance — compound (conditionally-conforming) receiver
##===----------------------------------------------------------------------===##
# The receiver need not be a bare type parameter. A compound parametric type
# with a *conditional* conformance (e.g. `CondCopyable[U]`, which is `Copyable`
# only when `U` is) reports `NeedsEvidence` rather than `No`, so it is also
# deferred. The obligation is folded before it is recorded, reducing the
# conditional conformance to its condition, so a trailing `where` discharges it
# whether it names the compound receiver or just the condition. Diagnostics
# report the reduced form, which is what a `where` clause has to state.


struct CondCopyable[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable), Movable
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


# Discharged: the trailing `where` names the compound receiver's conformance.
def cond_discharged[U: Movable & Deinitable]() -> NeedsCopyable[CondCopyable[U]]
    where conforms_to(CondCopyable[U], Copyable):
    pass


# Discharged: naming the condition works too, since the obligation reduces to it.
def cond_discharged_via_condition[U: Movable & Deinitable]() ->
    NeedsCopyable[CondCopyable[U]] where conforms_to(U, Copyable):
    pass


# Undischarged: no trailing `where`, so the deferred obligation stays unprovable
# and the discharge loop points at the `where` that would satisfy it.
# expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
# expected-note @below {{constraint declared here needs evidence for 'conforms_to(U, Copyable)'}}
# expected-note @below {{add a trailing 'where' clause that requires 'conforms_to(U, Copyable)'}}
def cond_undischarged[U: Movable & Deinitable]() -> NeedsCopyable[CondCopyable[U]]:
    pass
