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

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt --kgen-print-inline-type-values | FileCheck %s


# CHECK-LABEL: lit.trait.decl @Trait
# CHECK-SAME: <?, [[T:.*]]: !AnyType_Trait>
trait Trait:
    # CHECK: lit.fn @"f0{{.*}}(%self: !lit.ref<:!AnyType_Trait [[T]], imm {{.*}}> imm_mem) -> !kgen.none
    # CHECK-NEXT: kgen.unreachable
    def f0(self):
        ...

    # CHECK: lit.fn @"f1{{.*}}(%self: !lit.ref<{{.*}}> mut) -> !kgen.none
    # CHECK-NEXT: kgen.unreachable
    def f1(mut self):
        ...

    # CHECK: lit.fn @"f2{{.*}}(%self: !lit.ref<{{.*}}> mut) -> !kgen.none attributes {defaultedTraitFn,
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.return %none : !kgen.none
    # CHECK-NEXT: lit.end_fn
    def f2(mut self):
        pass

    # CHECK: lit.fn @"f3{{.*}}(%self: !lit.ref<{{.*}}> imm_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, mut *"__result__`2x2"> byref_result) throws -> !kgen.scalar<bool>
    # CHECK-NEXT: kgen.unreachable
    def f3(self) raises:
        ...

    # CHECK: lit.fn @"f4{{.*}}(%self: !lit.ref<{{.*}}> imm_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, mut *"__result__`2x2"> byref_result) throws -> !kgen.scalar<bool>
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.ref.store %none, %__result__ : <none, mut *"__result__`2x2">
    # CHECK-NEXT: %simd = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return %simd : !kgen.scalar<bool>
    # CHECK-NEXT: lit.end_fn
    def f4(self) raises:
        pass

    # CHECK: lit.fn @"f5{{.*}}(%self: !lit.ref<{{.*}}> mut, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, {{.*}}> byref_result) throws -> !kgen.scalar<bool>
    # CHECK-NEXT: kgen.unreachable
    def f5(mut self) raises:
        ...

    # CHECK: lit.fn @"f6{{.*}}(%self: !lit.ref<{{.*}}> mut, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<none, {{.*}}> byref_result) throws -> !kgen.scalar<bool>
    # CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
    # CHECK-NEXT: lit.ref.store %none, %__result__ : <none, mut *"__result__`2x2">
    # CHECK-NEXT: %simd = kgen.param.constant: scalar<bool> = <false>
    # CHECK-NEXT: lit.return %simd : !kgen.scalar<bool>
    # CHECK-NEXT: lit.end_fn
    def f6(mut self) raises:
        pass

    def overloaded(self):
        ...

    def overloaded(self, x: Int):
        ...

    def overloaded(self, x: __mlir_type.`!kgen.string`):
        ...

    # CHECK-LABEL: lit.fn @"parametric{{.*}}<x: !Int>
    def parametric[x: Int](self):
        ...


# CHECK-LABEL: lit.trait.decl @EmptyTrait
trait EmptyTrait:
    pass


# CHECK-LABEL: lit.trait.decl @Trait1
# CHECK-SAME: <?, [[T:.*]]: !AnyType_Trait1>
trait Trait1:
    # CHECK: lit.fn @"f{{.*}}(%self: !lit.ref<{{.*}}> imm_mem, ?, %__result__: !lit.ref<:!AnyType_Trait1 [[T]], mut {{.*}}> byref_result) -> !kgen.none
    def f(self) -> Self:
        ...


trait Trait2:
    def f(self) -> Self:
        ...


# CHECK-LABEL: lit.struct.decl @StructWithTraits({{.*}}Trait1_Trait2)
struct StructWithTraits(Trait1, Trait2, Movable where False):
    # CHECK: lit.fn @"f{{.*}}(%self: !lit.ref<!StructWithTraits, imm {{.*}}> imm_mem, ?, %{{.*}}: !lit.ref<!StructWithTraits, mut {{.*}}> byref_result) -> !kgen.none
    def f(self) -> Self:
        ...


# CHECK-LABEL: lit.trait.decl @CFMTrait
trait CFMTrait:
    # CHECK: lit.fn @"f1{{.*}}(%self: !lit.ref<{{.*}}> imm_mem) -> !kgen.none
    def f1(self):
        pass

    # CHECK: lit.fn @"f2()"() -> !kgen.none
    @staticmethod
    def f2():
        pass


# CHECK-LABEL: lit.struct.decl @CFMStruct({{.*}}CFMTrait)
struct CFMStruct(CFMTrait, Movable where False):
    # CHECK: lit.fn @"f1({{.*}})"[{{.*}}](%self: !lit.ref<!CFMStruct, imm {{.*}}> imm_mem) -> !kgen.none
    def f1(self):
        pass

    # CHECK: lit.fn @"f2()"() -> !kgen.none
    @staticmethod
    def f2():
        pass


# Test for struct with parameters and function with parameters.
# CHECK-LABEL: lit.trait.decl @CFMTraitParams
trait CFMTraitParams:
    # CHECK: lit.fn @"f1{{.*}}"<x: !AnyType_CFMTraitParams>[{{.*}}](
    def f1[x: CFMTraitParams](self):
        pass


# CHECK-LABEL: lit.struct.decl @CFMStructParams
struct CFMStructParams[t1: TrivialRegisterPassable, t2: TrivialRegisterPassable](
    CFMTraitParams, Movable where False
):
    # CHECK: lit.fn @"f1{{.*}}"<x: !AnyType_CFMTraitParams>[{{.*}}](%self: !lit.ref<!lit.struct<#CFMStructParams <:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable t1, :!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable t2>>{{.*}}> imm_mem)
    def f1[x: CFMTraitParams](self):
        pass


# ===----------------------------------------------------------------------=== #
# Call Emission
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.fn @"generic_trait_fn{{.*}}<T: !AnyType_Trait>
# CHECK-SAME: %x: !lit.ref<:!AnyType_Trait T, imm {{.*}}> imm_mem
def generic_trait_fn[T: Trait](x: T):
    # CHECK: lit.call tail[!lit.generator<[1]("self": {{.*}} imm_mem) -> !kgen.none>:
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Trait T, @traits::@Trait, "f0{{.*}}">]{{.*}}(%x)
    x.f0()

    # CHECK: lit.call tail[!lit.generator<[1]("self": {{[^)]*}}) -> !kgen.none>:
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Trait T, @traits::@Trait, "overloaded{{.*}}">]{{.*}}(%x)
    x.overloaded()
    # CHECK: lit.call tail[!lit.generator<[1]("self": {{.*}}, "x": !Int)
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Trait T, @traits::@Trait, "overloaded{{.*}}">]{{.*}}(%x, %{{.*}})
    x.overloaded(1)
    # CHECK: lit.call tail[!lit.generator<[1]("self": {{.*}}, "x": !kgen.string)
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Trait T, @traits::@Trait, "overloaded{{.*}}">]{{.*}}(%x, %{{.*}})
    x.overloaded(__mlir_attr.`"trait" : !kgen.string`)

    # CHECK: lit.call tail[!lit.generator<[1]("self": {{[^)]*}} imm_mem)
    # CHECK-SAME: bind_params(:!lit.generator<<"x": !Int>[1](
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Trait T, @traits::@Trait, "parametric{{.*}}">, {{.*}}1{{.*}})
    x.parametric[1]()


# CHECK-LABEL: lit.fn @"existential_arg
# CHECK-SAME: (%x: !lit.ref<!AnyType_Trait, imm {{.*}}>
def existential_arg(x: Trait):
    pass


trait SimpleTrait(ImplicitlyCopyable):
    def method(self, y: Int):
        ...

    def param_method[x: Int](self):
        ...


struct TraitStruct(SimpleTrait):
    def method(self, y: Int):
        pass

    def param_method[x: Int](self):
        pass


struct ParametricTraitStruct[z: Int](SimpleTrait):
    def method(self, y: Int):
        pass

    def param_method[x: Int](self):
        pass


def take_simple_trait[T: SimpleTrait]():
    pass


def infer_trait[T: SimpleTrait](value: T):
    pass


# CHECK-LABEL: lit.fn @"test_metatype_to_trait
def test_metatype_to_trait():
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait !TraitStruct
    take_simple_trait[TraitStruct]()
    # CHECK: call {{.*}}take_simple_trait{{.*}}<:!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait {{.*}}@ParametricTraitStruct<:!Int {:scalar<index> 2}>
    take_simple_trait[ParametricTraitStruct[2]]()


# CHECK-LABEL: lit.fn @"test_infer_trait
def test_infer_trait(
    a: TraitStruct, b: ParametricTraitStruct[2]
):
    # CHECK: call {{.*}}infer_trait{{.*}}<:!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait !TraitStruct
    infer_trait(a)
    # CHECK: call {{.*}}infer_trait{{.*}}<:!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait {{.*}}@ParametricTraitStruct<:!Int {:scalar<index> 2}>
    infer_trait(b)


trait StaticMethodTrait:
    @staticmethod
    def foobar():
        pass


struct StaticMethodStruct(StaticMethodTrait, ImplicitlyCopyable):
    @staticmethod
    def foobar():
        pass

    def __init__(out self, *, copy: Self):
        pass


# CHECK-LABEL: lit.fn @"trait_static_method{{.*}}<T: !AnyType_StaticMethodTrait
def trait_static_method[T: StaticMethodTrait]():
    # CHECK: lit.call tail[!lit.generator<() -> !kgen.none>: #kgen.get_witness<:!AnyType_StaticMethodTrait T, @traits::@StaticMethodTrait, "foobar{{.*}}">]()
    T.foobar()


# CHECK-LABEL: lit.fn @"copy_me
# CHECK-SAME: <T: !AnyType_Copyable_ImplicitlyCopyable_Movable
# CHECK-SAME: %value: !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable T, imm {{.*}}> imm_mem, ?,
# CHECK-SAME: %__result__: !lit.ref<:!AnyType_Copyable_ImplicitlyCopyable_Movable T, mut {{.*}}> byref_result
def copy_me[T: ImplicitlyCopyable](value: T) -> T:
    # CHECK-NEXT: lit.call tail[!lit.generator<[2](*, "copy": {{.*}}T, {{.*}}> imm_mem, ?, "self": {{.*}}T, {{.*}}> byref_result) -> !kgen.none>:
    # CHECK-SAME: #kgen.get_witness<:!AnyType_Copyable_ImplicitlyCopyable_Movable T, @std::@builtin::@stubs::@Copyable, "__init__(copy:$0)">]{{.*}}(%value, %__result__)
    return value


# CHECK-LABEL: lit.fn @"move_me
# CHECK-SAME: <T: !AnyType_Movable
# CHECK-SAME: :!AnyType_Movable T, {{.*}}> owned_in_mem
# CHECK-SAME: :!AnyType_Movable T, {{.*}}> byref_result
def move_me[T: Movable](var value: T) -> T:
    # CHECK-NEXT: lit.ownership.use %value
    # CHECK-NEXT: lit.call tail[{{.*}}#kgen.get_witness<:!AnyType_Movable T, @std::@builtin::@stubs::@Movable, "__init__(move:$0$)">]{{.*}}(%value, %__result__)
    return value^


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    @implicit
    def __init__(out self, x: Int):
        ...

    def __init__(out self, *, copy: Self):
        ...

    @staticmethod
    def may_throw() raises -> Self:
        ...

    def throwing_method(self) raises:
        ...


# ===----------------------------------------------------------------------=== #
# Calling Convention / Register Passable
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @RegTraitType
struct RegTraitType(RegisterPassable, TraitForReg):
    # CHECK-LABEL: lit.fn @"__init__
    # CHECK-SAME: (%x: !Int) -> !RegTraitType
    @implicit
    def __init__(out self, x: Int):
        pass

    def __init__(out self, *, copy: Self):
        pass

    @staticmethod
    def may_throw() raises -> Self:
        pass

    def throwing_method(self) raises:
        pass


# CHECK-LABEL: lit.fn @"raising_method
def raising_method[T: TraitForReg](x: T) raises:
    # CHECK: lit.call[{{.*}}: #kgen.get_witness<:!AnyType_TraitForReg T, @traits::@TraitForReg, "may_throw{{.*}}">][{{.*}}](%__error__, %__call_result_tmp__
    _ = T.may_throw()
    # CHECK: lit.call[{{.*}}: #kgen.get_witness<:!AnyType_TraitForReg T, @traits::@TraitForReg, "throwing_method{{.*}}">][{{.*}}](%{{.*}}, %__error__, %__call_result_tmp__
    x.throwing_method()


trait CrazyTrait:
    pass

    def foo[b: Int](self, c: Int) -> Self:
        ...


trait ChangedResultTypeTrait:
    @staticmethod
    def result_type() -> Self:
        ...


# COM: The calling convention rewrite results in a decl with two "overloads" that
# COM: differ only in result type. Ensure that the thunk gets selected.
# CHECK-LABEL: lit.struct.decl @ChangedResultTypeStruct
struct ChangedResultTypeStruct(ChangedResultTypeTrait, RegisterPassable):
    # CHECK-LABEL: lit.fn @"result_type()"() -> !ChangedResultTypeStruct
    @staticmethod
    def result_type() -> Self:
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}ChangedResultTypeTrait
    # CHECK-NEXT: kgen.witness "result_type{{.*}}" : !lit.generator<{{.*}}"__result__": !lit.ref<!ChangedResultTypeStruct, {{.*}}> byref_result) -> !kgen.none>{{.*}}def() thin -> traits::ChangedResultTypeStruct

# CHECK-LABEL: lit.fn @"convert_result_type
def convert_result_type():
    @__parameter
    def convert_result_type[T: ChangedResultTypeTrait]():
        pass

    # CHECK: call{{.*}}!ChangedResultTypeStruct
    convert_result_type[ChangedResultTypeStruct]()


trait SimpleTraitMethod:
    def foo(self):
        ...


struct VariadicTrait[*I: Int](RegisterPassable, SimpleTraitMethod):
    def foo(self):
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}SimpleTraitMethod
    # CHECK-NEXT: kgen.witness "foo{{.*}}" : !lit.generator<[1]("self": {{.*}} imm_mem) -> !kgen.none> = {{.*}}@"foo

# CHECK-LABEL: lit.fn @"test_bind_variadic
def test_bind_variadic():
    @__parameter
    def bind_trait[T: SimpleTraitMethod]():
        pass

    # CHECK: call{{.*}}@VariadicTrait<:param_list<!Int> []
    bind_trait[VariadicTrait[]]()


trait ThunkAmbiguity:
    def mismatched_arg(self):
        ...

    @staticmethod
    def mismatched_ret() -> Self:
        ...

    def __init__(out self):
        ...


struct ThunkAmbiguityRP(RegisterPassable, ThunkAmbiguity):
    def mismatched_arg(self):
        pass

    @staticmethod
    def mismatched_ret() -> Self:
        pass

    def __init__(out self):
        pass


# COM: Make sure that the generated thunks aren't select over the methods.


# CHECK-LABEL: lit.fn @"ambiguous_thunk
def ambiguous_thunk(x: ThunkAmbiguityRP):
    # CHECK-NOT: `thunk
    x.mismatched_arg()
    _ = ThunkAmbiguityRP.mismatched_ret()
    _ = ThunkAmbiguityRP()
    # CHECK-LABEL: lit.end_fn


trait OwnedArguments:
    def take(var self, var x: RegTraitType):
        ...


# CHECK-LABEL: lit.struct.decl @NoDtor
struct NoDtor(OwnedArguments, DefaultConstructible, RegisterPassable):
    def take(var self, var x: RegTraitType):
        pass

    def __init__(out self):
        pass

    def method(self):
        pass


trait DefaultConstructible:
    def __init__(out self):
        ...


def default_construct[T: DefaultConstructible]() -> T:
    return T()


# CHECK-LABEL: lit.fn @"generic_fn_return_type
def generic_fn_return_type():
    # CHECK: lit.var.decl "c" var : !lit.ref<!NoDtor,
    # CHECK-NEXT: call {{.*}}default_construct{{.*}}<:!AnyType_DefaultConstructible !NoDtor>(%c)
    var c = default_construct[NoDtor]()
    # CHECK: call {{.*}}@NoDtor::@"method
    c.method()


trait SimpleTraitA:
    def method(self):
        ...


trait SimpleTraitB:
    def method(self):
        ...


# CHECK-LABEL: lit.struct.decl @TwoThunks
# CHECK-SAME: (!AnyType_Deinitable_Movable_RegisterPassable_SimpleTraitA_SimpleTraitB)
struct TwoThunks(RegisterPassable, SimpleTraitA, SimpleTraitB):
    # CHECK: lit.fn @"method({{.*}}TwoThunks)"
    def method(self):
        pass


# https://linear.app/modularml/issue/MOCO-335/[bug]-register-passable-generates-phantom-trait-bound-overload
# CHECK-LABEL: lit.fn @"regpassable_reference
def regpassable_reference():
    # CHECK-NEXT: @TwoThunks::@"method
    comptime f = TwoThunks.method


trait RequiredType:
    comptime T: AnyType

    @staticmethod
    def use_it(arg: Self.T) -> Self.T:
        ...


struct RegPassableRequiredType(RequiredType, Movable where False):
    comptime T = Int

    @staticmethod
    def use_it(arg: Int) -> Int:
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}RequiredType
    # CHECK: kgen.witness "use_it{{.*}}" : {{.*}}def(arg: ::SIMD[DType.int, 1]) thin -> ::SIMD[DType.int, 1]


# CHECK-LABEL: lit.fn @"bind_regpassable_required_type
def bind_regpassable_required_type():
    # CHECK-NEXT: : !AnyType_RequiredType = <!RegPassableRequiredType>
    comptime T: RequiredType = RegPassableRequiredType


# ===----------------------------------------------------------------------=== #
# Special Functions
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: lit.struct.decl @RegTrivialSpecial
struct RegTrivialSpecial(TrivialRegisterPassable, AnyType, ImplicitlyCopyable):
    pass
    # CHECK: lit.fn @"__deinit__
    # CHECK: lit.fn @"__init__{{.*}}(*, %move:
    # CHECK: lit.fn @"__init__{{.*}}(*, %copy:

# COM: TrivialRegisterPassable should behave the same as
struct RegTrivialSpecialWithTrait(TrivialRegisterPassable):
    pass
    # CHECK: lit.fn @"__deinit__
    # CHECK: lit.fn @"__init__{{.*}}(*, %move:
    # CHECK: lit.fn @"__init__{{.*}}(*, %copy:

# CHECK-LABEL: lit.struct.decl @RegSpecial
struct RegSpecial(AnyType, ImplicitlyCopyable, RegisterPassable):
    def __init__(out self, *, copy: Self):
        pass

    # CHECK: lit.fn @"__deinit__
    # CHECK: lit.fn @"__init__{{.*}}(*, %move:


# CHECK-LABEL: lit.struct.decl @MemoryOnlySpecial
struct MemoryOnlySpecial(AnyType, ImplicitlyCopyable):
    pass
    # CHECK: lit.fn @"__deinit__
    # CHECK-SAME: [{{.*}} deinit_mem, |) -> !kgen.none
    # CHECK: return %none


def copy[T: ImplicitlyCopyable](x: T):
    pass


def move[T: Movable](x: T):
    pass


def destroy[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test_special_fn_traits
def test_special_fn_traits(
    mut x: RegTrivialSpecial, mut y: RegSpecial, mut z: MemoryOnlySpecial
):
    # COM: Just check that the implicit conversion succeeds.
    # CHECK-COUNT-9: lit.call
    copy(x)
    move(x)
    destroy(x)
    copy(y)
    move(y)
    destroy(y)
    copy(z)
    move(z)
    destroy(z)


# ===----------------------------------------------------------------------=== #
# Inheritance
# ===----------------------------------------------------------------------=== #


trait ParentTraitSameSig:
    def foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @ChildTraitSameSig
trait ChildTraitSameSig(ParentTraitSameSig):
    # CHECK-NEXT: lit.fn @"foo
    # CHECK-NEXT: kgen.unreachable
    def foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @GreatGrandFather
# CHECK-SAME: (!AnyType_GreatGrandFather)
trait GreatGrandFather:
    # CHECK: lit.fn @"foo
    def foo(self):
        ...


# CHECK-LABEL: lit.trait.decl @GrandFather
# CHECK-SAME: GreatGrandFather)
# CHECK-SAME: immediateParents = #kgen<trait_symbols[<@traits::@GreatGrandFather>]>
trait GrandFather(GreatGrandFather):
    # CHECK: lit.fn @"bar
    def bar(self):
        ...


# CHECK-LABEL: lit.trait.decl @Father
# CHECK-SAME: GrandFather_GreatGrandFather)
# CHECK-SAME: immediateParents = #kgen<trait_symbols[<@traits::@GrandFather>]>
trait Father(GrandFather):
    # CHECK: lit.fn @"baz
    def baz(self):
        ...


# CHECK-LABEL: lit.trait.decl @UnevenDiamond
# CHECK-SAME: Father_GrandFather_GreatGrandFather_UnevenDiamond)
# CHECK-SAME: immediateParents = #kgen<trait_symbols[<@traits::@Father>]>
trait UnevenDiamond(GreatGrandFather, Father):
    ...


# CHECK-LABEL: lit.struct.decl @TraitInheritance
# CHECK-SAME: Father_GrandFather_GreatGrandFather)
struct TraitInheritance(Father, Movable where False):
    def foo(self):
        pass

    def bar(self):
        pass

    def baz(self):
        pass

    # CHECK-LABEL: kgen.conformance @{{.*}}Father
    # CHECK-NEXT: kgen.witness "baz{{.*}}"

    # CHECK-LABEL: kgen.conformance @{{.*}}GrandFather
    # CHECK-NEXT: kgen.witness "bar{{.*}}"

    # CHECK-LABEL: kgen.conformance @{{.*}}GreatGrandFather
    # CHECK-NEXT: kgen.witness "foo{{.*}}"


# CHECK-LABEL: lit.fn @"test_trait_inheritance
def test_trait_inheritance():
    @__parameter
    def take_great_grand_father[T: GreatGrandFather]():
        pass

    @__parameter
    def take_grand_father[T: GrandFather]():
        pass

    @__parameter
    def take_father[T: Father]():
        pass

    # CHECK: call{{.*}}!TraitInheritance
    take_great_grand_father[TraitInheritance]()
    # CHECK: call{{.*}}!TraitInheritance
    take_grand_father[TraitInheritance]()
    # CHECK: call{{.*}}!TraitInheritance
    take_father[TraitInheritance]()


def infer_grand_father[T: GrandFather](x: T):
    pass


# CHECK-LABEL: lit.fn @"pass_up_trait
# CHECK-SAME: <T: !AnyType_Father_GrandFather_GreatGrandFather>
def pass_up_trait[T: Father](x: T):
    # CHECK-NEXT: call {{.*}}infer_grand_father{{.*}}<:!AnyType_GrandFather_GreatGrandFather upcast(:!AnyType_Father_GrandFather_GreatGrandFather T)>(%x)
    infer_grand_father(x)


# ===----------------------------------------------------------------------=== #
# Misc Bugs
# ===----------------------------------------------------------------------=== #


struct MovableType[T: Movable](TrivialRegisterPassable):
    pass


trait InCollection(Movable):
    pass


struct Collection[T: InCollection](Movable where False):
    var x: MovableType[Self.T]


# CHECK-LABEL: lit.struct.decl @Item
struct Item(TrivialRegisterPassable, InCollection):
    pass

    # CHECK-LABEL: kgen.conformance @{{.*}}Movable
    # CHECK-NEXT: kgen.witness "__init__(move:$0$)"


def take_movable(x: MovableType[Item]):
    pass


# CHECK-LABEL: lit.fn @"converted_metatype_struct_element
def converted_metatype_struct_element(x: Collection[Item]):
    # CHECK: call {{.*}}take_movable
    take_movable(x.x)


# CHECK-LABEL: lit.struct.decl @TraitMember
struct TraitMember[T: Movable & Deinitable](Movable where False):
    # CHECK: lit.fn @"__deinit__
    var value: Self.T


# COM: Misleading error about thunk functions when: (issue mojo-#1402)
#      the test has
#      - a struct conforms to a trait, e.g. Movable
#      - the struct has a field of another type with parameter as itself, e.g MyPointer[Self]
#      - the field struct type's parameter should conform to Movable


# CHECK-LABEL: lit.struct.decl @MyPointer
@fieldwise_init
struct MyPointer[T: AnyType](ImplicitlyCopyable):
    pass
    # CHECK: lit.fn @"__deinit__
    # CHECK: lit.fn @"__init__


# CHECK-LABEL: lit.struct.decl @HasMyPointerSelf
struct HasMyPointerSelf(AnyType, Movable where False):
    # CHECK: lit.struct.field x : !lit.struct<#MyPointer <:!AnyType
    var x: MyPointer[Self]
    # CHECK: lit.fn @"__deinit__

    def __init__(out self, *, deinit move: Self):
        pass


# Parser crash
# https://github.com/modularml/modular/issues/27897
# CHECK-LABEL: lit.fn @"check_trait_conversion_bymem_result_alias_crash
def retMemory[T: TraitForReg](value: T) -> MemoryOnlySpecial:
    pass


def check_trait_conversion_bymem_result_alias_crash(
    x: RegTraitType,
) -> MemoryOnlySpecial:
    return retMemory(x)


# Calling functions with implicit origins needs to cooperate.
def test[a: ABC]():
    _ = ABCOptionalParamInt[ABCDim(a)]()


trait SomeTrait:
    pass


struct ABC(SomeTrait, Movable where False):
    def __init__(out self):
        pass


struct ABCOptionalParamInt[dim_parametric: ABCDim](TrivialRegisterPassable):
    def __init__(out self):
        pass


struct ABCDim(Movable where False):
    def __init__[type: SomeTrait](out self, value: type):
        pass


trait TraitParameterized:
    def foo[T: SomeTrait](self):
        ...


struct ConcreteType(TraitParameterized, Movable where False):
    def foo[T: SomeTrait](self):
        pass


trait KeysBuilder:
    def add[x: Int](mut self):
        ...


struct KeysContainer[end: Int](KeysBuilder, Movable where False):
    def add[x: Int](mut self):
        pass


# CHECK-LABEL: lit.fn @"param_trait
def param_trait[T: SimpleTrait, value: T]():
    # CHECK-NEXT: apply({{.*}} #kgen.get_witness<:!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait T, @traits::@SimpleTrait, "method{{.*}}">{{.*}} store_to_mem(value), {{.*}}1{{.*}})
    comptime param = value.method(1)
    # CHECK-NEXT: [[VAR:%.*]] = lit.var.decl
    # CHECK-NEXT: [[VALUE:%.*]] = kgen.param.materialize
    # CHECK-NEXT: store [[VALUE]], [[VAR]]
    # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[VAR]]
    # CHECK: lit.call[{{.*}}#kgen.get_witness<:!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait T, @traits::@SimpleTrait, "method{{.*}}">{{.*}}([[IMM]],
    value.method(2)


trait Makeable:
    @staticmethod
    def make() -> Self:
        ...


struct MakeNamedResult(Makeable, RegisterPassable):
    @staticmethod
    def make(out out: Self):
        pass



# CHECK-LABEL: lit.fn @"check_named_result_regpassable
def check_named_result_regpassable():
    # CHECK-NEXET: @MakeNamedResult::@"make()"
    comptime T: Makeable = MakeNamedResult


# COM: Issue https://github.com/modularml/modular/issues/33939
# COM: Ensure parameter inference works between type value attributes.
trait OtherEmptyTrait(EmptyTrait):
    pass


struct Bar[T: EmptyTrait](Movable where False):
    pass


struct Foo[T: EmptyTrait](Movable where False):
    def infer_sub_trait[OT: OtherEmptyTrait](mut self, existing: Bar[OT]):
        pass


# CHECK-LABEL: lit.fn @"test_infer_sub_trait
def test_infer_sub_trait[T: OtherEmptyTrait](var foo: Foo[T], bar: Bar[T]):
    # CHECK: call {{.*}}@Foo::@"infer_sub_trait{{.*}}<:!AnyType_EmptyTrait upcast(:!AnyType_EmptyTrait_OtherEmptyTrait T), :!AnyType_EmptyTrait_OtherEmptyTrait T>(%foo, %bar)
    var copy = foo.infer_sub_trait(bar)


# ===----------------------------------------------------------------------=== #
# AnyTrait subtyping
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"anytrait_assignment
def anytrait_assignment():
    # CHECK-NEXT: meta<!AnyType_Movable> = <!AnyType_Movable>
    comptime t: type_of(AnyType & Movable) = AnyType&Movable


# CHECK-LABEL: lit.fn @"test_anytrait_subtyping
# CHECK-SAME: <ty: meta<!AnyType>>
def test_anytrait_subtyping[ty: type_of(AnyType)]():
    # Call trait metatype subtyping.
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:meta<!AnyType> !AnyType>()
    test_anytrait_subtyping[AnyType]()
    # CHECK-NEXT: lit.call {{.*}}test_anytrait_subtyping{{.*}}<:meta<!AnyType> !AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait>()
    test_anytrait_subtyping[SimpleTrait]()


# CHECK-LABEL: lit.fn @"take_many_things_of_specified_trait
# CHECK-SAME: <element_type: meta<!AnyType>,
# CHECK-SAME: element_types: !lit.struct<#TypeList {{.*}}:param_list<:meta<!AnyType> element_type>{{.*}} pos_vararg>()
def take_many_things_of_specified_trait[element_type: type_of(AnyType), //,
                                       *element_types: element_type]():
    pass


# CHECK-LABEL: lit.fn @"call_many_things_of_specified_trait
def call_many_things_of_specified_trait(a: TraitStruct):
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:meta<!AnyType> !AnyType, :param_list<!AnyType> [!TraitStruct]
    take_many_things_of_specified_trait[element_type=AnyType, TraitStruct]()

    # Int is movable.
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:meta<!AnyType> !AnyType_Movable, :param_list<!AnyType_Movable> [!Int]
    take_many_things_of_specified_trait[element_type=Movable, Int]()

    # TraitStruct conforms to SimpleTrait.
    # CHECK-NEXT: lit.call {{.*}}take_many_things_of_specified_trait
    # CHECK-SAME: <:meta<!AnyType> !AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait, :param_list<!AnyType_Copyable_ImplicitlyCopyable_Movable_SimpleTrait> [!TraitStruct, !TraitStruct]
    take_many_things_of_specified_trait[element_type=SimpleTrait, TraitStruct, TraitStruct]()


comptime _AnyTypeMetaType = type_of(AnyType)

# CHECK-LABEL: lit.struct.decl @TestAnyTrait
struct TestAnyTrait[element_trait: _AnyTypeMetaType](Movable where False):
    # CHECK: lit.fn @"take_any_type
    # CHECK-SAME: <b_type: !AnyType>[{{.*}}](%self:
    # CHECK-SAME: %b_value: !lit.ref<:!AnyType b_type, imm {{.*}} imm_mem)
    def take_any_type[b_type: AnyType](self, b_value: b_type):
        pass

    # CHECK: lit.fn @"test
    # CHECK-SAME: <a_type: !kgen.param<:meta<!AnyType> element_trait>>
    # CHECK-SAME: (%self: {{.*}}%a_value: !lit.ref<:{{.*}} element_trait> a_type, imm {{.*}}> imm_mem
    def test[a_type: Self.element_trait](self, a_value: a_type):
        self.take_any_type(a_value)


struct ParamType[x: Int](TrivialRegisterPassable):
    pass

# CHECK: lit.trait.decl @RGTrait{{.*}}
trait RGTrait(Deinitable, RegisterPassable):
    # CHECK-NEXT: lit.fn @"doSomething{{.*}}"[imm *"{{.*}}"](%self: !lit.ref<:!AnyType_Deinitable_Movable_RegisterPassable_RGTrait *"{{.*}}", imm *"{{.*}}"> imm_mem) -> !kgen.none
    def doSomething(self):
        ...

# CHECK-LABEL: lit.trait.decl @RGTrivialTrait{{.*}} register_passable_trivial
trait RGTrivialTrait(TrivialRegisterPassable):
    # CHECK-NEXT: lit.fn @"doSomething{{.*}}"(%self: !kgen.param<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable_RGTrivialTrait {{.*}}>) -> !kgen.none
    def doSomething(self):
        ...


# https://github.com/modular/mojo/issues/3540: Using the output slot breaks trait conformance
# CHECK-LABEL: lit.struct.decl @TestNamedResultConformance
struct TestNamedResultConformance(TrivialRegisterPassable, Trait1):

    # CHECK: lit.fn @"f
    # CHECK-SAME: (%self: !TestNamedResultConformance) -> !TestNamedResultConformance
    def f(self, out output: Self):
        pass

def test_pack_of_traits1[elt_trait: _AnyTypeMetaType, //, *elt_types: elt_trait]
                       (var *args: *elt_types):
     pass

def test_pack_of_traits2[elt_trait: _AnyTypeMetaType, //, *elt_types: elt_trait](
    var storage: VariadicPack[element_trait=elt_trait, _, *elt_types]):
     pass


comptime _MovableMetaType = type_of(Movable)

def take_anytype_ref[type: AnyType](ref value: type): pass

# CHECK-LABEL: lit.fn @"pass_movable_mt_ref
def pass_movable_mt_ref[elt_trait: _MovableMetaType, PassT: elt_trait](mut a: PassT):
    # CHECK-NEXT: lit.call {{.*}}@"take_anytype_ref
    # CHECK-SAME: <:!AnyType upcast(:!kgen.param<:meta<!AnyType_Movable> elt_trait> PassT),
    # CHECK-SAME: : !lit.generator<("value":{{.*}} elt_trait> PassT, mut *"a`"> ref) -> !kgen.none>
    take_anytype_ref(a)

comptime _CollectionElementMetaType = type_of(ImplicitlyCopyable)

struct FormVariadicPackWithCastedElementVariadic[
    element_trait: _CollectionElementMetaType, //,
    *element_types: element_trait](Movable where False):

    def __init__(out self, var *args: *Self.element_types):
        # This should work.
        self.foo(args^)
    def foo(self, var storage: VariadicPack[element_trait=Self.element_trait, _, *Self.element_types]):
        pass

# This tests that we can take UnsafePointer (which has an AnyType bound for T)
# and conditional conformance rebind the parametric type with AnyType bound down
# to Movable correctly.
def take_movable_pointer[T: Movable&AnyType](ptr: UnsafePointer[T, AnyOrigin[mut=True]]): pass
# CHECK-LABEL: test_parametric_anytype_movable
# CHECK-SAME: %ptr: !lit.struct<#Pointer <{{.*}}meta<!AnyType_Copyable_ImplicitlyCopyable_Movable> element_trait>
def test_parametric_anytype_movable[element_trait: _CollectionElementMetaType, //,
                                  *element_types: element_trait]
                                  (ptr: UnsafePointer[element_types[0], AnyOrigin[mut=True]]):

        # CHECK: lit.call {{.*}}take_movable_pointer
        # CHECK-SAME: <:!AnyType_Movable {{.*}}!AnyType_Copyable_ImplicitlyCopyable_Movable> element_trait>
        take_movable_pointer(ptr)


# Check that a trait method with a default implementation returning None may
# use 'pass'.
trait TBar:
    def bar(self) -> None:
        pass


# MOCO-2918: Default traits methods don't work if they have variadic packs
trait TraitWithVariadicPackDefault:
    def foo[*Ts: Movable](self, *args: *Ts):
        pass
struct StructInheritingVariadicPackDefault(TraitWithVariadicPackDefault, Movable where False):
    pass


comptime Composition = ImplicitlyCopyable

# CHECK-LABEL: lit.fn @"mlir_type_trait_conformance
def mlir_type_trait_conformance():
    # CHECK: !AnyType = <[{{.*}}::@__MLIRType<:non_struct_type index>, index]>
    comptime Any: AnyType = __mlir_type.index
    # CHECK: !AnyType_Copyable_ImplicitlyCopyable_Movable = <[{{.*}}::@__MLIRType<:non_struct_type index>, index]>
    comptime Copy: ImplicitlyCopyable = __mlir_type.index
    # CHECK: !AnyType_Movable = <[{{.*}}::@__MLIRType<:non_struct_type index>, index]>
    comptime Move: Movable = __mlir_type.index
    # CHECK: !alias_Composition1 = <#kgen.type<!lit.struct<#MLIRType <:non_struct_type index>>, index>>
    comptime Comp: Composition = __mlir_type.index

# Check to ensure default params don't bind too early.
trait TraitWDefault:
    def test[linear_idx_type: Int = 0](self) -> Int:
        ...

struct TestTraitWDefault(TraitWDefault, Movable where False):
    def test[linear_idx_type: Int = 0](self) -> Int:
        pass

# MOCO-3309: Ensure that keyword arguments work with trait conformance.
trait HasFooKw:
    def __init__(out self, *, foo: Self):
        ...

trait HasBarKw:
    def __init__(out self, *, bar: Self):
        ...

struct TestKWArgs(HasFooKw, HasBarKw, Movable where False):
    def __init__(out self, *, foo: Self):
        pass

    def __init__(out self, *, bar: Self):
        pass


# MOCO-4303: a struct conforming to a trait via `where False` can still
# define its own, differently-shaped overload of the same method name. The
# never-callable synthesized wrapper (from taking the trait's default under
# the unsatisfiable conformance) must not interfere with resolving a call
# that only matches the struct's own method -- it's excluded from
# consideration on its own merits (its own where clause always fails), not
# because it collides with, shadows, or otherwise gets confused with the
# real method.
trait Greeter:
    def greet(self, name: String) -> String:
        return "Hello, " + name

struct FriendlyGreeter(Greeter where False, Movable where False):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    def greet(self) -> String:
        return "Hi there"

def friendly_greeter_own_overload_is_unaffected():
    var s = FriendlyGreeter(1)
    var msg = s.greet()  # Ok -- resolves to FriendlyGreeter's own `greet`.


# MOCO-4429: two traits requiring the identical method shape, one is Movable
# with `where False` and the other a plain user trait also opted out. Neither
# needs any method implemented on the struct, so this must conform with no
# diagnostics.
trait CustomNeverMove:
    def __init__(out self, *, deinit move: Self):
        ...

struct BothOptOut(CustomNeverMove where False, Movable where False):
    pass
