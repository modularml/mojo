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

# Test errors for conditional trait conformance.
# These errors are detected during struct declaration parsing and conformance
# verification, before any instantiation occurs.
#
# NOTE: Declaration-time errors (diamond, ancestor) are tested first because
# they are emitted before method constraint errors during parsing.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# ===========================================================================
# Diamond with different constraints requires explicit Base
# ===========================================================================
# When a struct conditionally conforms to DerivedA and DerivedB with different
# constraints, and both inherit from Base, the user must explicitly list Base.


trait DiamondBase:
    pass


trait DiamondDerivedA(DiamondBase):
    pass


trait DiamondDerivedB(DiamondBase):
    pass


struct DiamondMissingExplicitBase[T: Movable & Deinitable](
    # expected-error @below {{ancestor trait 'DiamondBase' is reached via multiple inheritance paths with different constraints; it must be explicitly listed in the conformance list with the desired constraint}}
    DiamondDerivedA where conforms_to(T, Copyable),
    DiamondDerivedB where conforms_to(T, Intable),
    Movable,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


# ===========================================================================
# Derived constraint must imply ancestor constraint
# ===========================================================================
# When both a derived trait and its ancestor are explicitly listed with
# constraints, the derived constraint must logically imply the ancestor's.


trait AncestorImplicationBase:
    pass


trait AncestorImplicationDerived(AncestorImplicationBase):
    pass


struct DerivedDoesNotImplyAncestor[T: Movable & Deinitable](
    # Base requires Intable
    AncestorImplicationBase where conforms_to(T, Intable),
    # But Derived only requires Copyable - this doesn't imply Intable!
    # expected-error @below {{constraint for 'AncestorImplicationDerived' does not imply constraint for ancestor trait 'AncestorImplicationBase'; strengthen the derived constraint by adding the ancestor's constraint with 'and'}}
    AncestorImplicationDerived where conforms_to(T, Copyable),
    Movable,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


# ===========================================================================
# Unconditional derived with conditional ancestor
# ===========================================================================
# When a derived trait is listed unconditionally but its ancestor is explicitly
# listed conditionally, this is inconsistent: the derived trait always requires
# the ancestor, so the ancestor cannot be conditional.


trait UnconditionalDerivedBase:
    pass


trait UnconditionalDerivedChild(UnconditionalDerivedBase):
    pass


struct UnconditionalDerivedConditionalAncestor[T: Movable & Deinitable](
    # Derived is unconditional - always conforms
    Movable,
    # But ancestor is conditional - inconsistent!
    UnconditionalDerivedBase where conforms_to(T, Copyable),
    # expected-error @below {{constraint for 'UnconditionalDerivedChild' does not imply constraint for ancestor trait 'UnconditionalDerivedBase'; strengthen the derived constraint by adding the ancestor's constraint with 'and'}}
    UnconditionalDerivedChild,
):
    var data: Self.T

    def __init__(out self, var data: Self.T):
        self.data = data^


# ===========================================================================
# Conditional conformance to TrivialRegisterPassable is not supported
# ===========================================================================
# TrivialRegisterPassable depends on struct body (field triviality) which
# creates parser cycle risks and requires composing the user's where-clause
# with field-level triviality witnesses.


struct ConditionalTrivialRegPassable[T: Movable & Deinitable](
    Movable,
    # expected-error @below {{conditional conformance to 'TrivialRegisterPassable' is not supported}}
    TrivialRegisterPassable where conforms_to(T, TrivialRegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


# ===========================================================================
# Conditional conformance to RegisterPassable IS allowed
# ===========================================================================
# The struct stays pessimistically MemoryOnly at declaration time; the
# parametric isMemoryOnly bit resolves per-instantiation during lowering.


struct ConditionalRegPassable[T: Movable & Deinitable](
    Movable,
    RegisterPassable where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


# Origin rejection: returning an origin of a conditionally-RP argument is
# rejected because the type might expand to RegisterPassable (in which case
# the argument would be promoted to a register and the origin would dangle).
# expected-error @+3 {{cannot return 'x's origin, because it might expand to a RegisterPassable type}}
def bad_conditional_rp_origin[
    T: Movable & Deinitable
](x: ConditionalRegPassable[T]) -> ref[x] ConditionalRegPassable[T]:
    return x


# Workaround: using `ref` convention forces indirect passing, so the argument
# always has a stable memory address regardless of RP status. This must compile.
def ok_conditional_rp_ref_origin[
    T: Movable & Deinitable
](ref x: ConditionalRegPassable[T]) -> ref[x] ConditionalRegPassable[T]:
    return x


# ===========================================================================
# RP trait conformance with weaker constraint
# ===========================================================================
# When a struct has explicit RegisterPassable with constraint C_rp and also
# conforms to a derived RP trait with a weaker constraint C_conf, the
# ancestor implication check in DeclResolution.cpp rejects it (fires first).
# verifyAndBuildConformance in Traits.cpp has an independent implication
# check as defense-in-depth.


trait RPRequiringTrait(RegisterPassable):
    def rp_trait_method(self) -> Int:
        ...


struct RPTraitWeakerConstraint[T: Movable & Deinitable](
    Movable,
    # expected-error @below {{constraint for 'RPRequiringTrait' does not imply constraint for ancestor trait 'RegisterPassable'}}
    RPRequiringTrait where conforms_to(T, Movable),
    RegisterPassable where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def rp_trait_method(self) -> Int:
        return 42


# ===========================================================================
# Non-RP struct conditionally conforming to RP trait
# ===========================================================================
# A struct with NO RegisterPassable conformance that conditionally conforms
# to a trait inheriting from RegisterPassable. The canonical trait
# propagation adds RegisterPassable as an ancestor with the same constraint,
# so the struct ends up with conditional RP matching the conformance — the
# implication check in verifyAndBuildConformance passes trivially.
# This test documents that such usage is accepted (no error expected).


struct NoExplicitRPConformsToRPTrait[T: Deinitable & Movable](
    Deinitable,
    Movable,
    RPRequiringTrait where conforms_to(T, RegisterPassable),
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def rp_trait_method(self) -> Int:
        return 42


# ===========================================================================
# Unconditional conformance with conditional method
# ===========================================================================
# A struct that unconditionally claims to conform to a trait, but the method
# implementing the trait requirement has a constraint that can't be proven
# from nothing (the unconditional conformance provides no assumptions).


# expected-note @below {{trait 'UnconditionalConformanceTrait' declared here}}
trait UnconditionalConformanceTrait:
    # expected-note @+2 {{required by trait method here}}
    # expected-note @below {{invalid reference to 'do_something': lacking evidence to prove correctness}}
    def do_something(self):
        ...


# expected-error @+2 {{does not implement all requirements for 'UnconditionalConformanceTrait'}}
# expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
struct UnconditionalWithConditionalMethod[x: Int](
    Movable, UnconditionalConformanceTrait
):
    # expected-note @+2 {{cannot prove constraint for candidate}}
    # expected-note @below {{constraint declared here needs evidence for '(x > Int(10))'}}
    def do_something(self) where Self.x > 10:
        pass


# ===========================================================================
# Conditional conformance with non-implied method constraint
# ===========================================================================
# A struct that conditionally conforms to a trait (requires T: Intable),
# but the method has a different constraint (requires T: Copyable) that
# cannot be proven from the conformance constraint.


# expected-note @below {{trait 'MismatchedConstraintTrait' declared here}}
trait MismatchedConstraintTrait:
    # expected-note @+2 {{required by trait method here}}
    # expected-note @below {{invalid reference to 'process': lacking evidence to prove correctness}}
    def process(self):
        ...


# expected-error @+2 {{does not implement all requirements for 'MismatchedConstraintTrait'}}
# expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
struct MismatchedConstraints[T: Movable](
    MismatchedConstraintTrait where conforms_to(T, Intable), Movable
):
    # This method requires Copyable, but conformance only guarantees Intable
    # expected-note @+2 {{cannot prove constraint for candidate}}
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Copyable)'}}
    def process(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Weaker conformance constraint with stronger method constraint
# ===========================================================================
# A struct with a weaker conformance constraint (T: Copyable) but a method
# that requires a stronger constraint (T: Copyable AND Intable).


# expected-note @below {{trait 'WeakerConformanceTrait' declared here}}
trait WeakerConformanceTrait:
    # expected-note @+2 {{required by trait method here}}
    # expected-note @below {{invalid reference to 'execute': lacking evidence to prove correctness}}
    def execute(self):
        ...


# expected-error @+2 {{does not implement all requirements for 'WeakerConformanceTrait'}}
# expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
struct WeakerConformanceStrongerMethod[T: Movable](
    Movable,
    WeakerConformanceTrait where conforms_to(T, Copyable),
):
    # This method requires BOTH Copyable AND Intable, but conformance only guarantees Copyable
    # expected-note @below {{cannot prove constraint for candidate}}
    def execute(
        self,
        # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Intable) if conforms_to(T, Copyable) else conforms_to(T, Copyable)'}}
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Intable):
        pass


# ===========================================================================
# Conditional conformance with both unconditional and conditional methods
# ===========================================================================
# A struct with conditional conformance where both an unconditional method and
# a conditional method (whose constraint is implied) exist. Both are valid
# candidates, causing ambiguity.


# expected-note @below {{trait 'AmbiguousMethodTrait' declared here}}
trait AmbiguousMethodTrait:
    # expected-note @below {{ambiguous use of 'perform'}}
    def perform(self):
        ...


# expected-error @below {{does not implement all requirements for 'AmbiguousMethodTrait'}}
struct AmbiguousUnconditionalAndConditional[T: Movable](
    AmbiguousMethodTrait where conforms_to(T, Copyable),
    Movable,
):
    # Unconditional method - always valid
    # expected-note @below {{candidate declared here}}
    def perform(self):
        pass

    # Conditional method - also valid because conformance implies T: Copyable
    # expected-note @below {{candidate declared here}}
    def perform(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Multiple conditional methods both implied by stronger conformance
# ===========================================================================
# A struct with a strong conformance constraint (T: Copyable AND T: Intable)
# but two methods with weaker, different constraints. Both are implied by the
# conformance, causing ambiguity.


# expected-note @below {{trait 'MultipleConditionsTrait' declared here}}
trait MultipleConditionsTrait:
    # expected-note @below {{ambiguous use of 'run'}}
    def run(self):
        ...


# expected-error @below {{does not implement all requirements for 'MultipleConditionsTrait'}}
struct AmbiguousBothConditional[T: Movable](
    Movable,
    MultipleConditionsTrait where conforms_to(T, Copyable) and conforms_to(
        T, Intable
    ),
):
    # Method requiring Copyable - implied by conformance
    # expected-note @below {{candidate declared here}}
    def run(self) where conforms_to(Self.T, Copyable):
        pass

    # Method requiring Intable - also implied by conformance
    # expected-note @below {{candidate declared here}}
    def run(self) where conforms_to(Self.T, Intable):
        pass


# ===========================================================================
# Unprovable overload causes error even with a valid candidate
# ===========================================================================
# Following overload selection rules, ALL candidates' constraints must be
# definitively provable or disproved. If any candidate has unprovable
# constraints (constraints we can neither prove nor disprove from the
# conformance), we error - even if another candidate has provable constraints.
#
# This prevents ambiguity: the user might have intended the unprovable
# candidate to be selected, but our constraint system can't verify that.


# expected-note @below {{trait 'UnprovableCandidateTrait' declared here}}
trait UnprovableCandidateTrait:
    # expected-note @+2 {{required by trait method here}}
    # expected-note @below {{ambiguous reference to 'handle': lacking evidence to select candidate}}
    def handle(self):
        ...


# expected-error @+2 {{does not implement all requirements for 'UnprovableCandidateTrait'}}
# expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
struct UnprovableWithValidCandidate[T: Movable](
    Movable,
    UnprovableCandidateTrait where conforms_to(T, Copyable),
):
    # Unprovable: Intable is unrelated to Copyable - can't prove or disprove.
    # expected-note @+2 {{cannot prove constraint for candidate}}
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Intable)'}}
    def handle(self) where conforms_to(Self.T, Intable):
        pass

    # Provable: Copyable matches the conformance constraint.
    # But we still error because we can't rule out the above candidate.
    # expected-note @below {{candidate is valid but cannot be selected until other candidates are disproved}}
    def handle(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Method with `where not` is disproved (contradicts conformance)
# ===========================================================================
# When a method has `where not X` and the conformance requires X, the method
# is "disproved" - definitively not a valid candidate. This is different from
# "unprovable" because we CAN make a determination (it's definitely invalid).
#
# A disproved candidate is simply not a candidate, so the requirement goes
# unwitnessed. The candidate is still listed with its `where` clause, so the
# contradiction with the conformance is visible.


# expected-note @below {{trait 'ContradictingConstraintTrait' declared here}}
trait ContradictingConstraintTrait:
    # expected-note @below {{no 'apply' candidates have type 'def(self: DisprovedWithWhereNot[T]) thin -> None'}}
    def apply(self):
        ...


# expected-error @below {{does not implement all requirements for 'ContradictingConstraintTrait'}}
struct DisprovedWithWhereNot[T: Movable](
    ContradictingConstraintTrait where conforms_to(T, Copyable),
    Movable,
):
    # Disproved: `not conforms_to(T, Copyable)` contradicts conformance.
    # expected-note @below {{candidate declared here with type 'def[T: Movable, //](self: DisprovedWithWhereNot[T]) thin -> None where not conforms_to(T, Copyable).__bool__()'}}
    def apply(self) where not conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Synthesized default trait method gated by conformance constraint
# ===========================================================================
# When a struct conditionally conforms to a trait, synthesized default method
# wrappers must carry the conformance constraint. Calling the default method
# without evidence that the constraint is satisfied should be rejected.


trait DefaultMethodTrait:
    def custom_default(self) -> Int:
        return 42


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{def custom_default(self) -> Int where conforms_to(T, Copyable)    # note - generated function}}
struct ConditionalDefaultMethod[T: Movable](
    # expected-note @below {{constraint declared here}}
    DefaultMethodTrait where conforms_to(T, Copyable),
    Movable,
):
    def __init__(out self):
        pass


def call_default_externally[T: Movable](x: ConditionalDefaultMethod[T]):
    # expected-error @below {{lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint}}
    var a = x.custom_default()


# ===========================================================================
# Zero-field struct: synthesized copy/move init gated by constraint
# ===========================================================================
# A zero-field struct with conditional conformance. The compiler auto-
# synthesizes an empty copy/move init whose body trivially succeeds, so
# the conformance constraint on the defOp is the only thing preventing
# misuse.


trait ZeroFieldTrait:
    def zero_field_method(self) -> Int:
        return 0


# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{def zero_field_method(self) -> Int where conforms_to(T, Copyable)    # note - generated function}}
struct ZeroFieldConditional[T: Movable](
    Movable,
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, Copyable)'}}
    ZeroFieldTrait where conforms_to(T, Copyable),
):
    def __init__(out self):
        pass


def call_on_zero_field[T: Movable](x: ZeroFieldConditional[T]):
    # expected-error @below {{lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint}}
    var a = x.zero_field_method()


# ===========================================================================
# OR-gated constraint does not provide definite conformance for copy synthesis
# ===========================================================================
# When the conditional Copyable conformance uses an OR constraint, the
# compiler cannot determine which branch holds, so it cannot synthesize a
# copy constructor via trait downcast.  This tests that
# constraintImpliesConformance correctly rejects OR disjuncts.


struct CopySynthFailsWithOR[T: Deinitable & Movable](
    Copyable where conforms_to(T, Copyable) or conforms_to(T, Intable),
    Deinitable,
    Movable,
):
    # expected-error @below {{cannot synthesize copy constructor because field 'value' has non-copyable type}}
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


# ===========================================================================
# Multi-field struct where constraint only covers one parameter
# ===========================================================================
# The Copyable conformance constraint mentions T but not U.  The synthesized
# copy constructor can trait-downcast T (covered by the constraint) but has
# no evidence for U, so copy synthesis must fail on the second field.


struct CopySynthFailsMultiField[
    T: Deinitable & Movable,
    U: Deinitable & Movable,
](
    Copyable where conforms_to(T, Copyable),
    Deinitable,
    Movable,
):
    var first: Self.T
    # expected-error @below {{cannot synthesize copy constructor because field 'second' has non-copyable type}}
    var second: Self.U

    def __init__(out self, var first: Self.T, var second: Self.U):
        self.first = first^
        self.second = second^


# ===========================================================================
# comptime member constraint not implied by the conformance constraint
# ===========================================================================
# A conditional conformance discharges a comptime member's trailing `where`
# clause only when the conformance constraint implies it. Here the conformance
# is gated on `n >= 0` but the member requires `n >= 10`, which is not implied,
# so the constrained member type cannot satisfy the trait's plain `Int`
# requirement.


# expected-note @below {{trait 'CondAliasNotImplied' declared here}}
trait CondAliasNotImplied:
    # expected-note-re @below {{comptime member 'SIZE' type{{.*}}does not conform to trait's required type 'Int'}}
    comptime SIZE: Int


# expected-error @below {{'CondAliasNotImpliedStruct[n]' does not implement all requirements for 'CondAliasNotImplied'}}
struct CondAliasNotImpliedStruct[n: Int = -1](CondAliasNotImplied where n >= 0, Movable where False):
    comptime SIZE: Int where Self.n >= 10 = Self.n


# ===========================================================================
# constrained comptime member referenced as result type in an unprovable env
# ===========================================================================

trait Moco4214Op:
    comptime Output: AnyType

    def operate(self) -> Self.Output:
        ...


@fieldwise_init
struct Moco4214List[T: AnyType](
    Moco4214Op where conforms_to(T, Movable), Movable where False,
):
    # expected-note @below {{constraint declared here needs evidence}}
    comptime Output: AnyType where conforms_to(Self.T, Movable) = Int

    # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
    # expected-note @below {{add a trailing 'where' clause}}
    def operate(self) -> Self.Output:
        return Int(123)
