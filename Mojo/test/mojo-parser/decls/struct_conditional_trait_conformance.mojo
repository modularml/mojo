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
# RUN:  %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt --kgen-print-inline-type-values | FileCheck %s

# Test file for conditional trait conformance parsing.
# This tests that `where` clauses in struct trait inheritance lists
# are correctly parsed and placed in the canonicalTrait attribute.

# Type aliases with constraints are generated for constrained trait compositions.
# Check for constrained trait type aliases containing the expected constraints:
# CHECK-DAG: @Copyable where #kgen.constraint<{{.*}}conforms_to(:{{.*}} T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)
# CHECK-DAG: @Intable where #kgen.constraint<{{.*}}conforms_to(:{{.*}} T, :meta<!AnyType_Deinitable_Intable> !AnyType_Deinitable_Intable)


# ===========================================================================
# Unconditional conformance - struct is always Movable
# ===========================================================================
# The struct should NOT have any constraint in its trait type.
# CHECK: lit.struct.decl @UnconditionalMovable<T: !AnyType_Deinitable_Movable>
# CHECK-NOT: where #kgen.constraint
# CHECK-SAME: attributes
struct UnconditionalMovable[T: Movable & Deinitable](Movable):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^


# ===========================================================================
# Single conditional conformance - Copyable only when T is Copyable
# ===========================================================================
# Verify the ConformanceOp has the constraint attached:
# CHECK: lit.struct.decl @ConditionalCopyable<T: !AnyType_Deinitable_Movable>
# CHECK: kgen.conformance @std::@builtin::@stubs::@Copyable
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)
struct ConditionalCopyable[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable), Movable
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __init__(out self, *, deinit move: Self):
        self.value = move.value^

    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.value = copy.value.copy()


# ===========================================================================
# Multiple conditional conformances - Copyable and Intable
# ===========================================================================
# Verify ConformanceOps have constraints for both Copyable and Intable:
# CHECK: lit.struct.decl @MultipleConditionalConformances<T: !AnyType_Deinitable_Movable>
# CHECK: kgen.conformance @std::@builtin::@stubs::@Copyable
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)
# CHECK: kgen.conformance @std::@builtin::@stubs::@Intable
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Deinitable_Intable> !AnyType_Deinitable_Intable)
struct MultipleConditionalConformances[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable),
    Intable where conforms_to(T, Intable),
    Movable,
):
    var inner: Self.T

    def __init__(out self, var inner: Self.T):
        self.inner = inner^

    def __init__(out self, *, deinit move: Self):
        self.inner = move.inner^

    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        self.inner = copy.inner.copy()

    def __int__(self) -> Int where conforms_to(Self.T, Intable):
        return 0


# ===========================================================================
# Disproved candidate with provable alternative - selects provable
# ===========================================================================
# When a struct has both:
# - A method with `where not` that contradicts the conformance (disproved)
# - A method with matching constraints (provable)
# The provable candidate is correctly selected with no error.
#
# Just verify the struct declaration is generated (no compilation error):
# CHECK: lit.struct.decl @DisprovedWithProvableAlternative<T: !AnyType_Deinitable_Movable>


trait WhereNotTestTrait:
    def where_not_method(self):
        ...


struct DisprovedWithProvableAlternative[T: Movable & Deinitable](
    WhereNotTestTrait where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # This method is "disproved" - its constraint contradicts the conformance.
    # The conformance requires T: Copyable, but this method requires NOT that.
    def where_not_method(self) where not conforms_to(Self.T, Copyable):
        pass

    # This method is "provable" - its constraint matches the conformance.
    # It will be selected as the witness table entry.
    def where_not_method(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Witness selection with matching vs contradicting constraints
# ===========================================================================
# This test demonstrates that when selecting a witness for a trait method:
# - Overloads with provable constraints (matching conformance) are selected
# - Overloads with disproved constraints (contradicting conformance) are skipped
#
# The key is that the constraints must be on the SAME trait as the conformance.
# Using `where not conforms_to(T, Copyable)` works because it directly
# contradicts `conforms_to(T, Copyable)`.
#
# NOTE: Using UNRELATED traits like `where not conforms_to(T, Intable)` would
# NOT work - it would be "unprovable" and cause an error.
#
# CHECK: lit.struct.decl @WitnessSelectionWithWhereNot<T: !AnyType_Deinitable_Movable>

trait Greeter:
    def greet(self): ...

struct WitnessSelectionWithWhereNot[T: Movable & Deinitable](
    Greeter where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # Overload 1: Provable - constraint matches the conformance.
    # This is selected as the witness for the Greeter trait.
    def greet(self) where conforms_to(Self.T, Copyable):
        pass

    # Overload 2: Disproved - constraint directly contradicts the conformance.
    # The conformance guarantees T: Copyable, so `not conforms_to(T, Copyable)`
    # is definitively false. This overload is skipped.
    def greet(self) where not conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Multiple overloads with stronger conformance constraint
# ===========================================================================
# When the conformance has a compound constraint (A and B), we can use
# `where not B` to filter out an overload because the conformance implies B.
#
# Here: conformance is `conforms_to(T, Copyable) and conforms_to(T, Intable)`
# - Method with `where conforms_to(T, Intable)` → provable (conformance implies it)
# - Method with `where not conforms_to(T, Intable)` → disproved (conformance implies Intable)
#
# CHECK: lit.struct.decl @CompoundConformanceWithWhereNot<T: !AnyType_Deinitable_Movable>

trait Formatter:
    def format(self): ...

struct CompoundConformanceWithWhereNot[T: Movable & Deinitable](
    Formatter where conforms_to(T, Copyable) and conforms_to(T, Intable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # Overload 1: Provable - conformance implies both Copyable AND Intable,
    # so it certainly implies just Intable.
    def format(self) where conforms_to(Self.T, Intable):
        pass

    # Overload 2: Disproved - conformance implies Intable, so
    # `not conforms_to(T, Intable)` contradicts it.
    def format(self) where not conforms_to(Self.T, Intable):
        pass


# ===========================================================================
# Compound method constraint with contradicting part
# ===========================================================================
# When a method has `where not X and Y`, and the conformance implies X,
# the `not X` part contradicts the conformance, making the whole constraint
# disproved (since AND requires all parts to be true).
#
# This is the pattern the reviewer mentioned: users can add extra conditions
# with `and`, but as long as one part contradicts the conformance, the
# overload is filtered out.
#
# CHECK: lit.struct.decl @CompoundMethodConstraint<T: !AnyType_Deinitable_Movable>

trait Processor:
    def process(self): ...

struct CompoundMethodConstraint[T: Movable & Deinitable](
    Processor where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    # Overload 1: Provable - constraint matches the conformance.
    # This is selected as the witness.
    def process(self) where conforms_to(Self.T, Copyable):
        pass

    # Overload 2: Disproved - the `not conforms_to(T, Copyable)` part
    # contradicts the conformance. Even though there's an extra
    # `and conforms_to(T, Intable)` condition, the contradiction on the
    # first part makes the whole AND false.
    def process(self)
        where not conforms_to(Self.T, Copyable) and conforms_to(Self.T, Intable):
        pass


# ===========================================================================
# Trait composition with conditional conformance
# ===========================================================================
# When a trait composition `A & B` has a conditional conformance, both A and B
# should get the same constraint. Verify both conformance ops have constraints.
#
# CHECK: lit.struct.decl @CompositionConditional<T: !AnyType_Deinitable_Movable>
# CHECK: kgen.conformance{{.*}}@CompTraitA
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)
# CHECK: kgen.conformance{{.*}}@CompTraitB
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)

trait CompTraitA:
    def comp_a_method(self): ...

trait CompTraitB:
    def comp_b_method(self): ...

struct CompositionConditional[T: Movable & Deinitable](
    CompTraitA & CompTraitB where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def comp_a_method(self) where conforms_to(Self.T, Copyable):
        pass

    def comp_b_method(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Duplicate trait with same constraint (valid - no conflict)
# ===========================================================================
# Listing the same trait twice with the same constraint is redundant but valid.
#
# CHECK: lit.struct.decl @DuplicateSameConstraint<T: !AnyType_Deinitable_Movable>
# CHECK: kgen.conformance{{.*}}@DupSameTrait
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)

trait DupSameTrait:
    def dup_method(self): ...

struct DuplicateSameConstraint[T: Movable & Deinitable](
    DupSameTrait where conforms_to(T, Copyable),
    DupSameTrait where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def dup_method(self) where conforms_to(Self.T, Copyable):
        pass


# ===========================================================================
# Composition + standalone with same constraint (valid - no conflict)
# ===========================================================================
# A & B where cond, A where cond — A appears twice but with the same
# constraint, which is valid.
#
# CHECK: lit.struct.decl @CompositionStandaloneSameConstraint<T: !AnyType_Deinitable_Movable>
# CHECK: kgen.conformance{{.*}}@CSTraitA
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)
# CHECK: kgen.conformance{{.*}}@CSTraitB
# CHECK: } where #kgen.constraint<{{.*}}conforms_to(:!AnyType_Deinitable_Movable T, :meta<!AnyType_Copyable_Movable> !AnyType_Copyable_Movable)

trait CSTraitA:
    def cs_a_method(self): ...

trait CSTraitB:
    def cs_b_method(self): ...

struct CompositionStandaloneSameConstraint[T: Movable & Deinitable](
    CSTraitA & CSTraitB where conforms_to(T, Copyable),
    CSTraitA where conforms_to(T, Copyable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def cs_a_method(self) where conforms_to(Self.T, Copyable):
        pass

    def cs_b_method(self) where conforms_to(Self.T, Copyable):
        pass

# ===========================================================================
# MOCO-3347
# ===========================================================================


struct MyOptional[T: Movable](
    ImplicitlyCopyable where conforms_to(T, ImplicitlyCopyable) and conforms_to(
        T, Copyable
    ),
    Movable,
):
    pass

# ===========================================================================
# Split conforms_to constraints imply composite ancestor constraint
# ===========================================================================
# conforms_to(T, A) AND conforms_to(T, B) should imply conforms_to(T, A & B)
# when checking that a derived trait's constraint implies its ancestor's.
# This should parse without error (no "does not imply" diagnostic).

trait SplitAncestor:
    pass

trait SplitDerived(SplitAncestor):
    pass

# CHECK-LABEL: lit.struct.decl @SplitImpliesComposite
struct SplitImpliesComposite[T: Movable & Deinitable](
    SplitAncestor where conforms_to(T, Copyable & Intable),
    SplitDerived where conforms_to(T, Copyable) and conforms_to(T, Intable),
    Movable,
):
    pass

# Other direction: composite implies split (subsumption).
# CHECK-LABEL: lit.struct.decl @CompositeImpliesSplit
struct CompositeImpliesSplit[T: Movable & Deinitable](
    SplitAncestor where conforms_to(T, Copyable) and conforms_to(T, Intable),
    SplitDerived where conforms_to(T, Copyable & Intable),
    Movable,
):
    pass


#CHECK-LABEL: lit.struct.decl @Node
struct Node[ElementType: ImplicitlyCopyable](Movable):
    var value: MyOptional[Self.ElementType]
    # CHECK-LABEL: lit.fn @"__init__
    def __init__(out self, value: MyOptional[Self.ElementType] = None):
        # `MyOptional[Self.ElementType]` is implicitly copyable.

        # CHECK: lit.memcpy %value, %0
        self.value = value


# ===========================================================================
# Conditional `comptime` member referenced by a conditional method (MOCO-4214).
# ===========================================================================
trait Moco4214Op:
    comptime Output: AnyType

    def operate(self) -> Self.Output:
        ...


# CHECK-LABEL: lit.struct.decl @Moco4214List
@fieldwise_init
struct Moco4214List[T: AnyType](
    Moco4214Op where conforms_to(T, Movable), Movable where False,
):
    comptime Output: AnyType where conforms_to(Self.T, Movable) = Int

    # CHECK-LABEL: lit.fn @"operate(struct_conditional_trait_conformance::Moco4214List
    # CHECK-SAME: -> !Int
    def operate(self) -> Self.Output where conforms_to(Self.T, Movable):
        return Int(123)


struct Moco4214Payload(Movable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value


# --- Conditional `comptime` member referenced by a method ARGUMENT type. ----
trait Moco4214ArgOp:
    comptime Output: AnyType

    def consume(self, x: Self.Output) -> Int:
        ...


# CHECK-LABEL: lit.struct.decl @Moco4214ArgList
@fieldwise_init
struct Moco4214ArgList[T: AnyType](
    Moco4214ArgOp where conforms_to(T, Movable), Movable where False,
):
    comptime Output: AnyType where conforms_to(Self.T, Movable) = Int

    # `x`'s block argument must be retyped to the witness `!Int`, so the body
    # can return it where a concrete `Int` is expected.
    # CHECK-LABEL: lit.fn @"consume(struct_conditional_trait_conformance::Moco4214ArgList
    # CHECK-SAME: %x: !Int) -> !alias_Int1
    def consume(self, x: Self.Output) -> Int where conforms_to(Self.T, Movable):
        return x


# --- Conditional `comptime` member as a `raises` method's RESULT type. ------
# A `raises` method synthesizes both an `__error__` and a `__result__` block
# argument; normalization must keep `__result__` in sync with `fullArgTypes`
# even with `__error__` sitting in between it and the regular arguments.
trait Moco4214RaiseOp:
    comptime Output: AnyType

    def make(self) raises -> Self.Output:
        ...


# CHECK-LABEL: lit.struct.decl @Moco4214RaiseList
@fieldwise_init
struct Moco4214RaiseList[T: AnyType](
    Moco4214RaiseOp where conforms_to(T, Movable), Movable where False,
):
    comptime Output: AnyType where conforms_to(Self.T, Movable) = Moco4214Payload

    # CHECK-LABEL: lit.fn @"make(struct_conditional_trait_conformance::Moco4214RaiseList
    # CHECK-SAME: %__error__: !lit.ref<!Error,{{.*}}> byref_error
    # CHECK-SAME: %__result__: !lit.ref<!Moco4214Payload,{{.*}}> byref_result
    def make(self) raises -> Self.Output where conforms_to(Self.T, Movable):
        return Moco4214Payload(456)


# ===========================================================================
# `where` clause type mismatch between return type and returned expr
# (MOCO-4296).
# ===========================================================================
# A method's declared return type can be a compound instantiation (e.g.
# `Moco4296Iter[Self.T]`) where a NESTED type argument's trait bound is only
# satisfied via the method's own trailing `where` clause, not via `T`'s own
# declaration. The declared return type then carries a `downcast` on that
# argument, while the constructor-inferred type of the returned expression
# does not carry one; the two must still be recognized as zero-cost
# convertible.
struct Moco4296Collection[T: AnyType](Movable):
    # CHECK-LABEL: lit.fn @"foo(struct_conditional_trait_conformance::Moco4296Collection
    # Ensure the declared return type keeps its `downcast` on `T`.
    # CHECK-SAME: %__result__: !lit.ref<!lit.struct<#Moco4296Iter <{{.*}}downcast(:!AnyType T)>>,{{.*}}> byref_result
    def foo(
        var self,
    ) -> Moco4296Iter[Self.T] where conforms_to(Self.T, Movable & Deinitable):
        # CHECK: lit.call {{.*}}@struct_conditional_trait_conformance::@Moco4296Iter::@"__init__
        return Moco4296Iter(self^)


@fieldwise_init
struct Moco4296Iter[T: Movable & Deinitable](Movable where False):
    var _collection: Moco4296Collection[Self.T]


# ===========================================================================
# Nested `downcast` on a type argument
# ===========================================================================

# Like MOCO-4296, but the `downcast` is not on a direct parameter binding of
# the declared return type.
@fieldwise_init
struct Array[ElementType: Movable, size: Int](Movable where False):
    pass


@fieldwise_init
struct MyPtr[type: AnyType](Copyable, Movable):
    pass


@fieldwise_init
struct MyIter[T: Copyable, size: Int](Movable where False):
    var ptr: MyPtr[Array[Self.T, Self.size]]


# CHECK-LABEL: lit.fn @"make_iter
# CHECK-SAME: %__result__: !lit.ref<!lit.struct<#MyIter <{{.*}}downcast(:!AnyType_Movable ElementType){{.*}}>>,{{.*}}> byref_result
# CHECK: kgen.rebind {{.*}}Array<:!AnyType_Movable upcast(:!AnyType_Copyable_Movable downcast(:!AnyType_Movable ElementType))
# CHECK: lit.call {{.*}}@struct_conditional_trait_conformance::@MyIter::@"__init__
def make_iter[
    ElementType: Movable, size: Int
]() -> MyIter[downcast[ElementType, Copyable], size]:
    return {MyPtr[Array[ElementType, size]]()}
