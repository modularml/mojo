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

# Metal-only test: when a unified closure captures `DevicePointer`s (the
# register-passable, `DevicePassable` borrow of a `DeviceBuffer`), encoding the
# closure with a `MetalDeviceTypeEncoder` must register every referenced
# buffer's handle in `MetalDeviceTypeEncoder._buffers`. The Metal launch path
# relies on that list to bind the buffers a kernel touches.
#
# A closure cannot capture a `DeviceBuffer` directly. `DeviceBuffer` is itself
# `DevicePassable`, but the synthesized closure conformance also requires
# register-passable capture storage (the MOCO-4045 guard in `ClosureEmitter`)
# and `DeviceBuffer` is memory-only. `DevicePointer` is the register-passable
# borrow of one, and its `_to_device_type` dispatches to `encode_device_ptr`,
# which appends the owning buffer's handle.

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.testing import assert_equal, assert_true, TestSuite

from max.gpu.host import DeviceContext, DevicePointer
from max.gpu.host._device_context_metal import MetalDeviceTypeEncoder


# The device image of `PtrPair`: each member is the bare device pointer that
# `DevicePointer.device_type` maps to. A struct field cannot spell `AnyOrigin`,
# so the untracked origin stands in — a device pointer carries no tracked
# lifetime, and `Pointer._is_convertible_to_device_type` accepts the spelling.
@fieldwise_init
struct PtrPairDevice(ImplicitlyCopyable, TrivialRegisterPassable):
    var first: Pointer[Float32, MutUntrackedOrigin]
    var second: Pointer[Float32, MutUntrackedOrigin]


# A register-passable aggregate holding `DevicePassable` `DevicePointer`
# members. Its own encoding defers to `encode_fields`, which must recurse
# through the struct into each pointer (the behavior under test). A closure may
# only capture it because it conforms to `DevicePassable` itself: closure
# conformance is derived from the conformance of every capture, so an aggregate
# that merely contains `DevicePassable` members is not device-encodable.
@fieldwise_init
struct PtrPair[
    first_mut: Bool,
    second_mut: Bool,
    //,
    first_origin: Origin[mut=first_mut],
    second_origin: Origin[mut=second_mut],
](DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable):
    var first: DevicePointer[.float32, Self.first_origin]
    var second: DevicePointer[.float32, Self.second_origin]

    comptime device_type: AnyType = PtrPairDevice

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self.device_type](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "PtrPair"


def test_closure_registers_captured_buffers() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[.float32](16)
    var b = ctx.enqueue_create_buffer[.float32](32)
    var pa = a.device_ptr()
    var pb = b.device_ptr()

    # `pa`/`pb` are captured by value; the body uses them so they are really
    # captured. The closure is encoded, never executed.
    def k(z: Int) {var pa, var pb} -> Int:
        return pa.offset() + pb.offset()

    var storage = alloc[type_of(k)]({count = 1}).unsafe_leak()
    var encoder = MetalDeviceTypeEncoder()
    k._to_device_type(
        encoder, storage.unsafe_bitcast[NoneType]().as_unsafe_any_origin()
    )

    # Both captured pointers route through `encode_device_ptr`, registering
    # their owning buffers' handles — and nothing else.
    assert_equal(len(encoder._buffers), 2)

    var handle_a = a._handle.value()
    var handle_b = b._handle.value()
    var found_a = False
    var found_b = False
    for ref handle in encoder._buffers:
        if handle == handle_a:
            found_a = True
        if handle == handle_b:
            found_b = True
    assert_true(found_a, "buffer a was not registered in _buffers")
    assert_true(found_b, "buffer b was not registered in _buffers")

    storage.unsafe_free()


def test_closure_registers_buffers_via_nested_struct() raises:
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[.float32](16)
    var b = ctx.enqueue_create_buffer[.float32](32)
    var pair = PtrPair(a.device_ptr(), b.device_ptr())

    # The closure captures a struct whose `DevicePassable` members are one
    # level down; `encode_fields` must recurse into it.
    def k(z: Int) {var pair} -> Int:
        return pair.first.offset() + pair.second.offset()

    var storage = alloc[type_of(k)]({count = 1}).unsafe_leak()
    var encoder = MetalDeviceTypeEncoder()
    k._to_device_type(
        encoder, storage.unsafe_bitcast[NoneType]().as_unsafe_any_origin()
    )

    assert_equal(len(encoder._buffers), 2)

    var handle_a = a._handle.value()
    var handle_b = b._handle.value()
    var found_a = False
    var found_b = False
    for ref handle in encoder._buffers:
        if handle == handle_a:
            found_a = True
        if handle == handle_b:
            found_b = True
    assert_true(found_a, "nested buffer a was not registered in _buffers")
    assert_true(found_b, "nested buffer b was not registered in _buffers")

    storage.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
