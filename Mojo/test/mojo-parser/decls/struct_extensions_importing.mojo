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

# RUN: %parse-mojo-isolated -I %S/inputs %s -split-input-file | FileCheck %s


from struct_package import PlainStruct


# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: targetStruct = @struct_package::@plain_struct::@PlainStruct
__extension PlainStruct:
    # CHECK-LABEL: lit.fn @"sparklebark
    # CHECK-SAME: %self: !lit.ref<!PlainStruct, imm *"{{.*}}">
    def sparklebark(self: PlainStruct):
        pass


# // -----

# Make Sure We Gracefully Handle Redundant Imports (MSWGHRI)

from struct_package import PlainStruct
from struct_package import PlainStruct
from struct_package import PlainStruct


# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: targetStruct = @struct_package::@plain_struct::@PlainStruct
__extension PlainStruct:
    # CHECK-LABEL: lit.fn @"sparklebark
    # CHECK-SAME: %self: !lit.ref<!PlainStruct, imm *"{{.*}}">
    def sparklebark(self: PlainStruct):
        pass


# // -----

from struct_package import PlainStruct


trait Flying:
    def fly_to(mut self, new_location: Int):
        ...


# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: immediateParents = #kgen<trait_symbols[<@struct_extensions_importing::@Flying>]>
# CHECK-SAME: targetStruct = @struct_package::@plain_struct::@PlainStruct
__extension PlainStruct(Flying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!PlainStruct, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    def fly_to(mut self: PlainStruct, new_location: Int):
        self.set_location(new_location)

    # CHECK: kgen.conformance @struct_extensions_importing::@Flying {


def launch_flying[F: Flying](mut flying: F):
    flying.fly_to(2)


# CHECK-LABEL: lit.fn @"launch_ship
def launch_ship(mut ship: PlainStruct):
    # CHECK: lit.call tail @struct_extensions_importing::@"launch_flying[::AnyType & struct_extensions_importing::Flying]
    # CHECK-SAME: <:!AnyType_Flying !PlainStruct>
    launch_flying(ship)


# // -----


from trait_package import Flying as ImportedFlying


struct Spaceship(Movable where False):
    var location: Int

    def __init__(out self):
        self.location = 0

    def set_location(mut self, new_location: Int):
        self.location = new_location


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: immediateParents = #kgen<trait_symbols[<@trait_package::@plain_trait::@Flying>]>
# CHECK-SAME: targetStruct = @struct_extensions_importing::@Spaceship
__extension Spaceship(ImportedFlying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    def fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)

    # CHECK: kgen.conformance @trait_package::@plain_trait::@Flying {


def launch_flying2[F: ImportedFlying](mut flying: F):
    flying.fly_to(2)


# CHECK-LABEL: lit.fn @"launch_ship2
def launch_ship2(mut ship: Spaceship):
    # CHECK: lit.call {{.*}}@"launch_flying2[::AnyType & trait_package::plain_trait::Flying]
    # CHECK-SAME: <:!AnyType_Flying !Spaceship>
    launch_flying2(ship)


# // -----

# Tests a certain cycle that happened once.
# The cycle:
# - Extension signature resolution wanted to find its target struct,
#   so it looked up "MyStruct" with resolve=false.
#   - The lookup found the MyStruct struct and MyStruct extension.
#   - It then signature-resolved all things it found (because it wasn't really
#     respecting resolve=false that well), including the extension.
#     - Boom, cycle.
# Solution was to make it so the lookup doesn't necessarily resolve everything
# it finds when resolve=false.
# TODO(MOCO-522): Better solution would be to make it so the lookup can find
# only the struct and not its extensions.

from struct_and_extension_package import MyStruct


def use_struct():
    var s = MyStruct(2)
    s.extended_method()
