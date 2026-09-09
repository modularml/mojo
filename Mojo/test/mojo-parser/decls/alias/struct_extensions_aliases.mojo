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

# RUN: %parse-mojo-isolated -split-input-file %s | FileCheck %s


struct Spaceship:
    var fuel: Int


__extension Spaceship:
    comptime MaxSpeed: Int = 42


# CHECK-LABEL: lit.fn @"test_function
def test_function():
    # CHECK: lit.alias.decl *"MySpeed`"
    comptime MySpeed = Spaceship.MaxSpeed
    # CHECK: lit.var.decl "speed"
    var speed: Int = MySpeed


# // -----

# Test accessing an alias whose type is another alias, and materializing it.


struct ZInt(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, other: ZInt):
        pass


struct Spaceship:
    var fuel: ZInt


__extension Spaceship:
    comptime InnerType: AnyType = ZInt
    comptime MaxSpeed: InnerType = ZInt()


# CHECK-LABEL: lit.fn @"test_function()"
def test_function():
    # Note how this is resolving to ZInt right here, it means the lookup worked.
    # CHECK: lit.alias.decl *"MySpeed`": !alias_InnerType1 = <sugar_member_alias(!Spaceship, "MaxSpeed",
    comptime MySpeed = Spaceship.MaxSpeed
    # Note how this is resolving to ZInt right here, it means the lookup worked.
    # CHECK: lit.var.decl "speed" var : !lit.ref<:!AnyType #alias_InnerType,
    var speed = MySpeed


# // -----

# Test an extension method accessing and materializing an extension alias.


struct ZInt(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, other: ZInt):
        pass


struct Spaceship:
    var fuel: ZInt


__extension Spaceship:
    comptime MaxSpeed = ZInt()

    # CHECK-LABEL: lit.fn @"get_max_speed
    def get_max_speed(self: Spaceship) -> ZInt:
        # Note how it's ZInt right here, it means the lookup worked.
        # CHECK: kgen.param.materialize: !ZInt
        return ZInt(MaxSpeed)


# // -----

# Test an extension method accessing a struct alias via self argument.


struct ZInt(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, other: ZInt):
        pass


struct Rocket:
    comptime DefaultFuel = ZInt()
    var fuel: ZInt


__extension Rocket:
    # CHECK-LABEL: lit.fn @"get_default_fuel
    def get_default_fuel(self) -> ZInt:
        # Note how it's ZInt right here, it means the lookup worked.
        # CHECK: kgen.param.materialize: !ZInt
        return self.DefaultFuel  # access via self


## // -----

# Tests an extension's alias accessing its struct's parameter declaration.


struct Rocket[T: AnyType]:
    pass


__extension Rocket:
    comptime FuelType = Self.T


# // -----

# Test accessing a generic struct's extension's alias
# Also tests calling an extension method on a generic container


struct ZInt(ImplicitlyCopyable):
    def __init__(out self):
        pass

    def __init__(out self, other: ZInt):
        pass


struct Container[T: ImplicitlyCopyable & Deinitable](Movable where False):
    var data: Self.T


__extension Container:
    comptime ElementType = Self.T
    comptime DefaultSize = ZInt()

    def get_element_via_self(self: Container[Self.T]) -> Self.T:
        return self.data


# CHECK-LABEL: lit.fn @"test_self_alias_with_generic_1
def test_self_alias_with_generic_1(container: Container[ZInt]):
    # Note how it's ZInt right here, it means the lookup worked.
    # CHECK: lit.alias.decl *"MyElementType
    # CHECK-SAME: <sugar_member_alias(!lit.struct<#Container <:{{.*}} !ZInt>>, "ElementType", !ZInt)>
    comptime MyElementType = Container[ZInt].ElementType


# CHECK-LABEL: lit.fn @"test_self_alias_with_generic_2
def test_self_alias_with_generic_2(container: Container[ZInt]):
    # Note how it's ZInt right here, it means the lookup worked.
    # CHECK: %element = lit.var.decl "element" var : !lit.ref<!ZInt,
    var element = container.get_element_via_self()
