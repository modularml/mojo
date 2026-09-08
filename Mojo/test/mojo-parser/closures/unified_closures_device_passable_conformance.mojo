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

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

# COM: Verify DevicePassable storage conformance emits lit __device_type structs
# COM: with device field layouts, including nested captured closures.

# CHECK-LABEL: lit.struct.decl @"{{.*}}parametricCaptureClosure{{.*}}__storage::__device_type"
# CHECK-SAME: attributes {synthetic} {
# CHECK-NEXT: lit.struct.field capture3 : !lit.struct<#SIMD {{.*}}
# CHECK-NEXT: lit.struct.field capture2 : !kgen.param<:!AnyType #kgen.get_witness<:!{{.*}}Hosty{{.*}} Y, @std::@builtin::@device_passable::@DevicePassable, "device_type">>

# CHECK-LABEL: lit.struct.decl @"{{.*}}capturesClosure{{.*}}__storage"
# CHECK-SAME: attributes {definesClosure,{{.*}}synthetic}
# CHECK-NEXT: lit.struct.field capture2 : !HostInt
# CHECK-NEXT: lit.struct.field anotherClosure : !{{.*}}storage{{.*}}

# CHECK-LABEL: lit.struct.decl @"{{.*}}anotherClosure{{.*}}__storage::__device_type"
# CHECK-SAME: attributes {synthetic} {
# CHECK-NEXT: lit.struct.field capture1 : {{.*}}SIMD{{.*}}
# CHECK-NEXT: lit.struct.field capture3 : {{.*}}SIMD{{.*}}

# CHECK-LABEL: lit.struct.decl @"{{.*}}anotherClosure{{.*}}__storage"
# CHECK-SAME: !None_DevicePassable
# CHECK-SAME: attributes {definesClosure,{{.*}}synthetic}
# CHECK-NEXT: lit.struct.field capture1 : !HostString
# CHECK-NEXT: lit.struct.field capture3 : !HostString

# Host-side DevicePassable types with non-identity device representations.
# DevicePassable closure conformance requires register-passable captures
# (see MOCO-4045 guard in ClosureEmitter), so these are TrivialRegisterPassable.
@fieldwise_init
struct HostInt(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = Int
    var value: Int

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value * 2, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostInt"


struct HostString(DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = SIMD[.int32, 4]
    var value: SIMD[.int32, 4]
    # Deliberately absent from device_type so host and device layouts differ.
    var host_padding: SIMD[.int32, 4]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        var device_value = self.value
        device_value = device_value + 1
        encoder.encode(device_value, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostString"


def use[T: DevicePassable](b: T, c: Int):
    pass


def createDevicePassableClosures(
    capture1: HostString,
    capture2: HostInt,
    capture3: HostString,
) raises:
    def anotherClosure(argument: Int) {var}:
        use(capture1, argument)
        use(capture3, argument)

    def capturesClosure(argument: Int) {var}:
        use(capture2, argument)
        anotherClosure(argument)


trait Hosty(DevicePassable):
    pass


@fieldwise_init
struct HostIntViaHosty(Hosty, ImplicitlyCopyable, TrivialRegisterPassable):
    comptime device_type: AnyType = Int
    var value: Int

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self.value * 2, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostIntViaHosty"


struct HostStringParametric[W: Int](
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    comptime device_type: AnyType = SIMD[.int32, Self.W]
    var value: SIMD[.int32, Self.W]
    var host_padding: SIMD[.int32, Self.W]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        var device_value = self.value
        device_value = device_value + 1
        encoder.encode(device_value, target)

    @staticmethod
    def get_type_name() -> String:
        return "HostStringParametric"


def createParametricDevicePassableClosures[
    X: Int,
    Y: Hosty
    & ImplicitlyCopyable
    & Deinitable
    & TrivialRegisterPassable,
](
    capture1: HostStringParametric[X],
    capture2: Y,
    capture3: HostStringParametric[X],
) raises:
    def parametricCaptureClosure(argument: Int) {var}:
        use(capture3, argument)
        use(capture2, argument)
