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

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @below {{trait 'MyMovable' declared here}}
trait MyMovable:
    # expected-note @below {{required function '__init__' is not implemented}}
    def __init__(out self, *, deinit move: Self):
        ...


trait ErroneousTrait:
    # expected-error @+2 {{'self' argument must have type 'Self' in trait method declaration, but actually has type 'Int'}}
    @__allow_legacy_custom_self_type
    def add(self: Int):
        ...


trait FooTrait:
    def foo(self):
        ...


# expected-error @below {{'where' clauses in conformance lists are only supported on structs}}
trait WhereClauseInTraitConformanceList(FooTrait where True):
    pass


struct ParamType[x: Int](FooTrait, Movable where False):
    def foo(self):
        pass


def invalid_trait_bind():
    # expected-error @below {{'ParamType[_]' is not concrete, use '[]' to bind missing parameters}}
    comptime Bound: FooTrait = ParamType


def different_trait_types[
    T: ImplicitlyCopyable, U: ImplicitlyCopyable
](x: T) -> U:
    # expected-error @below {{cannot implicitly convert 'T' value to 'U'}}
    return x


# expected-note @below {{trait 'SimpleTrait' declared here}}
trait SimpleTrait:
    # expected-note @below {{required function 'some_method' is not implemented}}
    # expected-note @below {{no 'some_method' candidates have type 'def(self: ParamDoesNotConform[x]) thin -> None'}}
    def some_method(self):
        ...


# expected-error @below {{'TraitStruct' does not implement all requirements for 'MyMovable'}}
struct TraitStruct(MyMovable, SimpleTrait, Movable where False):
    def some_method(self):
        pass


# expected-note @+1 {{function declared here}}
def test_many_things_of_specified_trait[
    element_type: type_of(AnyType), //, *element_types: element_type
]():
    pass


# expected-error @below {{'DoesNotConform' does not implement all requirements for 'SimpleTrait'}}
struct DoesNotConform(SimpleTrait, Movable where False):
    pass


# expected-error @below {{'ParamDoesNotConform[x]' does not implement all requirements for 'SimpleTrait'}}
struct ParamDoesNotConform[x: Int](SimpleTrait, Movable where False):
    # expected-note @below {{candidate declared here with type 'def(self: ParamDoesNotConform[x], y: Int) thin -> None' (specialized from 'def[x: Int, //](self: ParamDoesNotConform[x], y: Int) thin -> None')}}
    def some_method(self, y: Int):
        pass


def call_many_things_of_specified_trait(a: TraitStruct):
    # This is ok!
    test_many_things_of_specified_trait[
        element_type=AnyType, TraitStruct, Int
    ]()

    # expected-error @+2 {{'test_many_things_of_specified_trait' parameter 'element_types' has 'Movable' type, but value has type 'AnyStruct[TraitStruct]'}}
    test_many_things_of_specified_trait[
        element_type=Movable, TraitStruct, TraitStruct
    ]()

    test_many_things_of_specified_trait[
        element_type=SimpleTrait,
        TraitStruct,
        # This will succeed, the error will be raised when resolving `DoesNotConform`.
        DoesNotConform,
    ]()


# expected-note@+1 {{trait 'TrivialTrait' declared here}}
trait TrivialTrait(TrivialRegisterPassable):
    # expected-note@+1 {{required function 'doSomething' is not implemented}}
    def doSomething(self):
        ...


# expected-note@+1 {{inherited through 'MemTraitViolation' here}}
trait MemTraitViolation(TrivialTrait):
    def bar(self):
        ...


trait NonTrivialRGTrait(RegisterPassable):
    def bar(self):
        ...


# expected-error @+1 {{does not implement all requirements for}}
struct StructViolation2(TrivialTrait):
    pass


# expected-error @+1 {{does not implement all requirements for}}
struct StructViolation3(MemTraitViolation):
    def bar(self):
        pass


trait NotDeinitable:
    def foo(self):
        ...


@fieldwise_init
struct Bar[T: NotDeinitable](Movable where False):  # expected-note {{'Bar' declared here}}
    pass


def bindAnyTraitToTrait():
    # expected-error @+1 {{'Bar' parameter 'T' has 'NotDeinitable' type, but value has type 'AnyTrait[NotDeinitable]'}}
    var _list = Bar[NotDeinitable]()


def anytrait_assignment():
    # expected-error @below {{cannot implicitly convert 'AnyTrait[ImplicitlyCopyable]' value to 'AnyTrait[FooTrait]' in comptime initializer}}
    comptime t: type_of(FooTrait) = ImplicitlyCopyable


trait SomeTrait:
    comptime A: Int


@fieldwise_init
struct TakeInt[A: Int](Movable where False):
    pass


# expected-note @below {{function declared here}}
def take_two_inferred_params[Size: Int](x: TakeInt[Size], y: TakeInt[Size]):
    pass


def call_take_two_inferred_params[T: SomeTrait](x: T):
    # expected-error @below {{invalid call to 'take_two_inferred_params': value passed to 'y' cannot be converted from 'TakeInt[Int(1)]' to 'TakeInt[T.A]'}}
    take_two_inferred_params(TakeInt[T.A](), TakeInt[1]())


# Check that a trait method with a default implementation returning a non-None
# type may not use 'pass'.
trait TBar:
    # expected-error @+4 {{trait method with a return type must not use 'pass'; use '...' to declare the method as required}}
    # expected-note @below {{in 'bar', declared here}}
    # expected-note @below {{original default implementation from trait 'TBar' here}}
    def bar(self) -> Int:
        pass


trait TBarSub(TBar):
    # expected-note @below {{conflicting implementation from trait 'TBarSub' here}}
    def bar(self) -> Int:
        return 0


# expected-error @+1 {{trait method requirement 'bar' has conflicting default implementations in 'TBar' and 'TBarSub'; you must implement it manually}}
struct TBarActual(TBarSub, Movable where False):
    pass


trait WhereClauseOnTraitMethod:
    # expected-error @+1 {{'where' clauses on trait methods are not supported}}
    def guarded_method(self) where Self.x > 10:
        pass


trait ConflictTraitName:
    # expected-note @+1 {{trait method declared here}}
    def test[a: Int](self):
        pass


# expected-error @+1 {{name conflict between parameter 'a' in the default trait method and a parameter in the struct}}
struct ConflictStruct[a: Int](ConflictTraitName, Movable where False):
    pass


# ===----------------------------------------------------------------------=== #
# Synthesized __deinit__
# ===----------------------------------------------------------------------=== #


# MOCO-4252
struct CondDeletableField1[T: AnyType](
    Deinitable where conforms_to(T, Deinitable), Movable where False,
):
    var value: Self.T  # Ok


struct CondDeletableField2[T: AnyType, X: Int](
    Deinitable where conforms_to(T, Deinitable) and X > 10, Movable where False,
):
    var value: Self.T  # Ok


struct CondDeletableField3[T: AnyType, X: Int](
    Deinitable where X > 10, Movable where False,
):
    var value: Self.T  # expected-error {{field 'value' has non-'Deinitable' type 'T'}}


# MOCO-4262
struct NeverDeletableInner(Deinitable where False, Movable where False):
    pass


struct NeverDeletableOuter(Deinitable where False, Movable where False):
    var m: NeverDeletableInner


# MOCO-4332: same opt-out idiom as MOCO-4262 above, but for Movable/Copyable.
# A struct whose own move/copy-ctor synthesis is doomed by `where False` must
# not require its fields to conform to Movable/Copyable either, even when the
# field's own conformance is *also* a doomed `where False`.
struct NeverMovableInner(Movable where False):
    pass


struct NeverMovableOuter(Movable where False):
    var m: NeverMovableInner


struct NeverCopyableInner(Copyable where False):
    pass


struct NeverCopyableOuter(Copyable where False):
    var m: NeverCopyableInner


# Field-movability is irrelevant once the outer struct opts out: because a
# `Movable where False` struct synthesizes no move constructor, its fields are
# never checked for movability. Both a movable field and a wholly non-movable
# one must therefore compile without any field-focused diagnostic.
struct UnconditionallyMovableField(Movable):
    pass


struct NeverMovableOuterWithMovableField(Movable where False):
    var m: UnconditionallyMovableField


struct NotMovableAtAll:
    pass


struct NeverMovableOuterWithNonMovableField(Movable where False):
    var m: NotMovableAtAll


# ===----------------------------------------------------------------------=== #
# Synthesized functions with unsatisfiable where clauses
# ===----------------------------------------------------------------------=== #


# MOCO-4303: a `Movable where False` opt-out synthesizes no move constructor,
# so a call that could only resolve to that synthesized candidate finds no
# candidates at all rather than being "explained" against a never-callable one.
struct NeverMovable[T: AnyType](Movable where False):
    pass


def never_movable_call_is_no_candidates_found():
    # expected-error @+1 {{invalid call to '__init__': no candidates found}}
    var z = NeverMovable[Int]()


struct NeverCopyable(Movable, Copyable where False):
    var v: Int

    def __init__(out self, v: Int):
        self.v = v


def never_copyable_call_is_no_candidates_found():
    var a = NeverCopyable(1)
    # A `Copyable where False` opt-out synthesizes no copy constructor, so
    # `copy` is not a member at all.
    # expected-error @+1 {{'NeverCopyable' value has no attribute 'copy'}}
    var b = a.copy()


# Same as above, but exercised through a user-defined trait with a defaulted
# method (synthesizeDefaultTraitMethodWrapper) instead of a builtin special
# member (Movable/Copyable/Deinitable). Confirms the fix isn't
# specific to compiler-synthesized special members.
trait Greeter:
    def greet(self, name: String) -> String:
        return "Hello, " + name


struct SilentGreeter(Greeter where False, Movable where False):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


def never_greeter_call_is_no_candidates_found():
    var s = SilentGreeter(1)
    # A `Greeter where False` opt-out synthesizes no default-method wrapper, so
    # `greet` is not a member at all.
    # expected-error @+1 {{'SilentGreeter' value has no attribute 'greet'}}
    var msg = s.greet()
