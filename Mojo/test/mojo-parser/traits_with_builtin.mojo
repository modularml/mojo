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

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


# COM: Just check that conformance checking succeeds.
trait TraitForReg(Copyable):
    @implicit
    def __init__(out self, x: Int):
        ...

    @staticmethod
    def may_throw() raises -> Self:
        ...


struct RegTypeTrivial(TraitForReg, TrivialRegisterPassable):
    @implicit
    def __init__(out self, x: Int):
        pass

    @staticmethod
    def may_throw() raises -> Self:
        pass


trait AsyncTrait:
    async def foo(self) -> Int:
        ...

    async def bar(self) raises -> Int:
        ...


struct AsyncStruct(AsyncTrait, Movable where False):
    async def foo(self) -> Int:
        pass

    async def bar(self) raises -> Int:
        pass


# CHECK-LABEL: lit.struct.decl @AsyncStructReg
struct AsyncStructReg(AsyncTrait, TrivialRegisterPassable):
    async def foo(self) -> Int:
        pass

    async def bar(self) raises -> Int:
        pass


trait Explicit:
    def __int__(self) -> Int:
        ...


trait Implicit:
    def __as_int__(self) -> Int:
        ...


@fieldwise_init
struct Foo(Explicit, Implicit, Movable where False):
    def __int__(self) -> Int:
        return 42

    def __as_int__(self) -> Int:
        return 42


struct Bar(Movable where False):
    @implicit
    def __init__[T: Implicit](out self, value: T):
        pass

    def __init__[T: Explicit](out self, value: T):
        pass


# CHECK-LABEL: lit.fn @"construct_implicit_type_explicitly
def construct_implicit_type_explicitly():
    _ = Bar(Foo())


# CHECK-LABEL: lit.fn @"async_trait
def async_trait[T: AsyncTrait](value: T):
    # CHECK: lit.async.call[!lit.generator<[2]("self": {{.*}} imm_mem, ?, "__result__": !lit.ref<:meta<!Int> #alias_Int, mut *[0,1]> byref_result) async -> !kgen.none>: #kgen.get_witness
    _ = value.foo()


def take_intable[T: Intable](x: T):
    pass


# CHECK-LABEL: lit.fn @"nonmaterializable_trait
def nonmaterializable_trait():
    # CHECK-NEXT: [[SLOT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!Int,
    # CHECK-NEXT: [[VAL:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: store [[VAL]], [[SLOT]]
    # CHECK-NEXT:  = lit.ref.immut [[SLOT]]
    # CHECK-NEXT: call {{.*}}take_intable{{.*}}<:!AnyType_Deinitable_Intable !Int
    take_intable(1)
