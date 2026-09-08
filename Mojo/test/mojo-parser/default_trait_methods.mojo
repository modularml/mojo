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

# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
struct RP(TrivialRegisterPassable):
    pass


@fieldwise_init
struct NonRP(Movable where False):
    pass


trait Foo:
    def rp(self) -> RP:
        return RP()

    def non_rp(self) -> NonRP:
        return NonRP()

    @no_inline
    def no_inline(self, x: RP) -> RP:
        return RP()

    @always_inline
    @staticmethod
    def always_inline(x: RP) -> RP:
        return RP()


# CHECK-LABEL: lit.struct.decl @Bar
struct Bar(Foo, Movable where False):
    # CHECK: lit.fn @"rp(default_trait_methods::Bar)Foo"{{.*}}([[SELF:%[^:]+]]: {{.*}}) -> !RP
    # CHECK: lit.call @default_trait_methods::@Foo::@"rp($0)"{{.*}}([[SELF]])

    # CHECK: lit.fn @"non_rp(default_trait_methods::Bar)Foo"{{.*}}([[SELF:%[^:]+]]: {{.*}}, {{.*}}, [[RESULT:%[^:]+]]: {{.*}}) -> !kgen.none
    # CHECK: lit.call @default_trait_methods::@Foo::@"non_rp($0)"{{.*}}([[SELF]], [[RESULT]])

    # Make sure we preserve inline annotations on the wrapper methods
    # CHECK: lit.fn @"no_inline
    # CHECK-SAME: no_inline
    # CHECK: lit.fn @"always_inline
    # CHECK-SAME: always_inline

    # CHECK: kgen.conformance{{.*}}@Foo
    # CHECK-DAG: kgen.witness "rp{{.*}}"
    # CHECK-DAG: kgen.witness "non_rp{{.*}}"
    # CHECK-DAG: kgen.witness "no_inline{{.*}}"
    # CHECK-DAG: kgen.witness "always_inline{{.*}}"
    pass


@fieldwise_init
struct Zork(TrivialRegisterPassable):
    pass


trait AA1:
    comptime X: ImplicitlyCopyable

    def zork(self, x: Self.X) -> Self.X:
        return x


# Check that we handle traits with associated aliases properly
# CHECK-LABEL: lit.struct.decl @TAA
struct TAA(AA1, Movable where False):
    comptime X = Zork

    # CHECK: lit.fn @"zork(default_trait_methods::TAA,default_trait_methods::Zork)AA1"{{.*}}([[SELF:%[^:]+]]: {{.*}}, [[X:%[^:]+]]: {{.*}}, {{.*}}, [[RESULT:%[^:]+]]: {{.*}}) -> !kgen.none
    # CHECK: lit.call @default_trait_methods::@AA1::@"zork($0,$0.X)"{{.*}}([[SELF]], [[X]], [[RESULT]])

    # CHECK: kgen.conformance{{.*}}@AA1
    # CHECK-DAG: kgen.witness "zork{{.*}}"


# Test parameterized types in default trait methods
@fieldwise_init
struct ParamRPType[x: Int, y: Int](TrivialRegisterPassable):
    var value: Int


trait Barable:
    def bar(self):
        ...


trait ParamInputTrait:
    @staticmethod
    def process_parameterized[T: Barable](item: T) -> Int:
        item.bar()
        return 100

    def return_parameterized[x: Int, y: Int](self) -> ParamRPType[x, y]:
        return ParamRPType[x, y](x * y)


# CHECK-LABEL: lit.struct.decl @SimpleTestStruct
struct SimpleTestStruct(ParamInputTrait, Movable where False):
    # Check that we generate proper wrapper for parameterized input method
    # CHECK: lit.fn @"process_parameterized
    # CHECK: lit.call @default_trait_methods::@ParamInputTrait::@"process_parameterized

    # Check that we generate proper wrapper for parameterized return type
    # CHECK: lit.fn @"return_parameterized
    # CHECK: lit.call @default_trait_methods::@ParamInputTrait::@"return_parameterized

    # CHECK: kgen.conformance{{.*}}@ParamInputTrait
    # CHECK-DAG: kgen.witness "process_parameterized{{.*}}"
    # CHECK-DAG: kgen.witness "return_parameterized{{.*}}"
    pass


# COM: Test parameterized struct with parameters whose names are the same as a
# parameters used in the trait methods.
# CHECK-LABEL: lit.struct.decl @ParamTestStruct
# TODO(MOCO-3274): detect and raise a better error message when there is a name\
# conflict between a struct parameter and a defaulted trait method parameter.
struct ParamTestStruct[T1: Int, x1: Bool](ParamInputTrait, Movable where False):
    # CHECK: lit.fn @"process_parameterized
    # CHECK-SAME: <T: !Barable_AnyType>
    # CHECK-SAME: %item: !lit.ref<:!Barable_AnyType T,
    # CHECK: lit.call @default_trait_methods::@ParamInputTrait::@"process_parameterized
    # CHECK-SAME: <:!ParamInputTrait_AnyType @default_trait_methods::@ParamTestStruct<:!Int T1, :!Bool x1>, :!Barable_AnyType T>

    # CHECK: lit.fn @"return_parameterized
    # CHECK-SAME: <x: !Int, y: !Int>
    # CHECK-SAME: %self: !lit.ref<!lit.struct<#ParamTestStruct <:!Int T1, :!Bool x1>>,
    # CHECK-SAME: -> !lit.struct<#ParamRPType <:!Int x, :!Int y>>
    # CHECK: lit.call @default_trait_methods::@ParamInputTrait::@"return_parameterized
    # CHECK-SAME: <:!ParamInputTrait_AnyType @default_trait_methods::@ParamTestStruct<:!Int T1, :!Bool x1>, :!Int x, :!Int y>

    # CHECK: kgen.conformance{{.*}}@ParamInputTrait
    # CHECK-DAG: kgen.witness "process_parameterized{{.*}}"
    # CHECK-DAG: kgen.witness "return_parameterized{{.*}}"
    pass


@fieldwise_init
struct BarableStruct(Barable, Movable where False):
    def bar(self):
        pass


# COM: Test that visiting the same struct twice during default-trait-method
# synthesis doesn't crash.
trait FooA:
    def foo(self) -> RP:
        return RP()

    def bar(self) -> RP:
        return RP()


trait FooB(FooA):
    def foo(self) -> RP:
        return RP()

    def bar(self) -> RP:
        return RP()


# CHECK-LABEL: lit.struct.decl @FooA_FooB_Struct(
struct FooA_FooB_Struct(FooB, Movable where False):
    # CHECK-DAG: lit.fn @"foo(default_trait_methods::FooA_FooB_Struct)"[
    # CHECK-DAG: lit.fn @"bar(default_trait_methods::FooA_FooB_Struct)"[
    # CHECK-NOT: lit.fn @"foo(default_trait_methods::FooA_FooB_Struct)FooA"[
    # CHECK-NOT: lit.fn @"bar(default_trait_methods::FooA_FooB_Struct)FooB"[
    # CHECK-NOT: lit.fn @"foo(default_trait_methods::FooA_FooB_Struct)FooB"[
    # CHECK-NOT: lit.fn @"bar(default_trait_methods::FooA_FooB_Struct)FooA"[
    def foo(self) -> RP:
        return RP()

    def bar(self) -> RP:
        return RP()


# CHECK: kgen.conformance @{{.*}}::@FooA {
# CHECK-DAG:   kgen.witness "foo($0)"
# CHECK-DAG:   kgen.witness "bar($0)"
# CHECK: }

# CHECK: kgen.conformance @{{.*}}::@FooB {
# CHECK-DAG:   kgen.witness "foo($0)"
# CHECK-DAG:   kgen.witness "bar($0)"
# CHECK: }


# COM: Fix for MOCO-2540: Test variadic packs in default trait methods.
# COM: The wrapper forwards the pack argument directly (as a BlockArgument) rather
# COM: than constructing it via RefPackCreateOp, which origin tracking must handle.
trait DefaultWithVariadicPack:
    def variadic_method[*Ts: AnyType](self, *args: *Ts):
        pass


# CHECK-LABEL: lit.struct.decl @UsingDefaultWithVariadicPack
@fieldwise_init
struct UsingDefaultWithVariadicPack(DefaultWithVariadicPack, Movable where False):
    # CHECK: lit.fn @"variadic_method[{{.*}},*::AnyType{{.*}}](default_trait_methods::UsingDefaultWithVariadicPack
    # CHECK: lit.call @default_trait_methods::@DefaultWithVariadicPack::@"variadic_method
    # CHECK: kgen.conformance{{.*}}DefaultWithVariadicPack
    # CHECK-DAG: kgen.witness "variadic_method{{.*}}"
    pass
