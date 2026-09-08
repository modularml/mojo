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

# ===----------------------------------------------------------------------=== #
#
# This tests that the builtin ImplicitlyCopyable and Movable traits are pulled from the
# Mojo Parser Tests stubs and not from the Mojo Stdlib package.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait ToCastInto:
    def test(self):
        ...


# Keep the target trait generic so contextual evaluation cannot merge it with
# the source bound. The downcast must still preserve the source's stronger
# trivial convention.
# CHECK-LABEL: lit.fn @"raw_downcast_preserves_trivial_passability
def raw_downcast_preserves_trivial_passability[
    T: TrivialRegisterPassable, Trait: type_of(AnyType)
](value: downcast[T, Trait]):
    var copy = value
    _ = copy


# CHECK-LABEL: lit.fn @"trait_downcast_reg_type
def trait_downcast_reg_type[T: TrivialRegisterPassable](x: T):
    # CHECK: lit.var.decl "y" var : !lit.ref<:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable_ToCastInto downcast(:!AnyType_Copyable_Deinitable_ImplicitlyCopyable_Movable_RegisterPassable_TrivialRegisterPassable T), mut *"y`1">
    var y = trait_downcast[ToCastInto](x)
    y.test()


# CHECK-LABEL: lit.fn @"trait_downcast_anytype
def trait_downcast_anytype[T: AnyType](x: T):
    # CHECK: lit.var.decl "y" ref : !lit.ref<!lit.ref<:!AnyType_ToCastInto downcast(:!AnyType T), imm *"x`">, mut *"y`1">
    ref y = trait_downcast[ToCastInto](x)
    y.test()


struct ListIterator[T: Copyable & Deinitable](Movable where False):
    var t: Self.T


struct List[T: Movable & Deinitable](Movable where False):
    var t: Self.T

    # CHECK: lit.alias.decl *"Iterator`": meta<!lit.struct<#ListIterator <:{{.*}} downcast(:!AnyType_Deinitable_Movable T)>>>
    comptime Iterator = ListIterator[downcast[Self.T, Copyable]]

    def iter(self) -> Self.Iterator:
        pass


def sink[T: ToCastInto](x: T):
    pass


def foo[T: Movable & ToCastInto & Deinitable](l: List[T]):
    var iter = l.iter()
    # Make sure ToCastInfo conformance survives the downcast.
    # CHECK: lit.call @{{.*}}::@"sink{{.*}}
    sink(iter.t)
