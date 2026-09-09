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

"""Low-level probe for the mechanism `DevicePointerEngine` relies on.

`DevicePointerEngine` (in `layout/tensor_engine.mojo`) is a `TensorEngine`
whose handle is a `DevicePointer`. At the kernel boundary the `DevicePointer`
encodes to a bare device `Pointer` (`DevicePointer._to_device_type` ->
`encode_device_ptr`), and the device-side operations reinterpret the handle's
first `size_of[Pointer]` bytes as that pointer. This file validates that
underlying stdlib mechanism directly, without the `layout` dependency the
`TileTensor`-level probe (`test_device_pointer_tile_storage_kernel`) carries.

Two preconditions are probed:

A. A host-side struct containing a `DevicePointer` can be the host type of a
   kernel argument, with a `TrivialRegisterPassable` device-side type as its
   `device_type`. The kernel reads the transformed bytes and they line up with
   both fields the kernel touches (the encoded pointer AND a sibling field next
   to it).

B. On the device, code can reinterpret the struct's first
   `size_of[Pointer]` bytes as a `Pointer` and use it — the cast
   pattern `DevicePointerEngine.load`/`store` use on device.

This stays in the OneS/TwoS pattern (distinct host and device types, manual
conversion in `_to_device_type`) — the host struct keeps the full
`DevicePointer` while the device struct holds only the bare pointer.
"""

from asyncrt_test_utils import create_test_device_context
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DevicePointer
from std.testing import TestSuite, assert_equal


# Device-side view: bare `Pointer` followed by a sentinel. This is what
# the kernel actually receives. Both fields are `TrivialRegisterPassable`, so
# the struct as a whole is a valid kernel-argument type. The struct is
# parameterized on `origin` because a field may not expose an `Any` origin
# directly.
@fieldwise_init
struct DevicePtrAndSentinel[mut: Bool, //, origin: Origin[mut=mut]](
    DevicePassable, ImplicitlyCopyable, TrivialRegisterPassable
):
    var raw_ptr: Pointer[Float32, Self.origin]
    var sentinel: Int32

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DevicePtrAndSentinel"


# Host-side wrapper: holds a `DevicePointer` (host-only) plus a sentinel.
# `_to_device_type` manually peels off the `Pointer` view and writes the
# device-side struct. This is the OneS/TwoS pattern; it does NOT use
# `encode_fields` because the host and device structs differ in shape.
struct DevicePointerAndSentinel[mut: Bool, //, origin: Origin[mut=mut]](
    DevicePassable, ImplicitlyCopyable
):
    var dp: DevicePointer[.float32, Self.origin]
    var sentinel: Int32

    def __init__(
        out self,
        var dp: DevicePointer[.float32, Self.origin],
        sentinel: Int32,
    ):
        self.dp = dp
        self.sentinel = sentinel

    comptime device_type: AnyType = DevicePtrAndSentinel[MutAnyOrigin]

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(
            DevicePtrAndSentinel[MutAnyOrigin](
                self.dp.unsafe_ptr(), self.sentinel
            ),
            target,
        )

    @staticmethod
    def get_type_name() -> String:
        return "DevicePointerAndSentinel"


def write_sentinel_via_direct_field(arg: DevicePtrAndSentinel[MutAnyOrigin]):
    """Baseline path: access `raw_ptr` and `sentinel` as ordinary fields on
    the device-side struct.

    If this fails, the host->device translation itself is broken (the encoded
    `Pointer` didn't land where the device-side struct expects it, or
    `sentinel` is at the wrong offset).
    """
    if global_idx.x != 0:
        return
    arg.raw_ptr[unsafe_offset=0] = Float32(arg.sentinel)


def write_sentinel_via_reinterpret(arg: DevicePtrAndSentinel[MutAnyOrigin]):
    """Probe path: reinterpret the first `size_of[Pointer]` bytes of the
    struct as a `Pointer` and write through it, instead of using the
    named field.

    This is the cast pattern `DevicePointerEngine.load`/`store` use on device
    — the struct's static type wouldn't necessarily expose a `Pointer`
    field, but its first bytes ARE one.

    If this fails but `write_sentinel_via_direct_field` succeeds, the
    reinterpret pattern itself isn't viable and the design needs a different
    on-device access mechanism.
    """
    if global_idx.x != 0:
        return
    var reinterpreted = Pointer(to=arg).unsafe_bitcast[
        Pointer[Float32, MutAnyOrigin]
    ]()[]
    reinterpreted[unsafe_offset=0] = Float32(arg.sentinel)


def test_kernel_receives_encoded_pointer_and_sibling() raises:
    """Precondition A: host wrapper with `DevicePointer` and a sibling field
    encodes correctly and the kernel sees both."""
    var ctx = create_test_device_context()
    comptime expected: Int32 = 0x1234

    var buf = ctx.enqueue_create_buffer[.float32](1)
    buf.enqueue_fill(Float32(-1))

    var arg = DevicePointerAndSentinel(buf.device_ptr(), expected)
    var compiled = ctx.compile_function[write_sentinel_via_direct_field]()
    ctx.enqueue_function(compiled, arg, grid_dim=1, block_dim=1)

    with buf.map_to_host() as host:
        assert_equal(host[0], Float32(expected))


def test_kernel_reinterpret_first_bytes_as_unsafe_pointer() raises:
    """Precondition B: on the device side, the wrapper's first bytes can be
    reinterpreted as a `Pointer` and used to write through."""
    var ctx = create_test_device_context()
    comptime expected: Int32 = 0xC0DE

    var buf = ctx.enqueue_create_buffer[.float32](1)
    buf.enqueue_fill(Float32(-1))

    var arg = DevicePointerAndSentinel(buf.device_ptr(), expected)
    var compiled = ctx.compile_function[write_sentinel_via_reinterpret]()
    ctx.enqueue_function(compiled, arg, grid_dim=1, block_dim=1)

    with buf.map_to_host() as host:
        assert_equal(host[0], Float32(expected))


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    var suite = TestSuite()
    suite.test[test_kernel_receives_encoded_pointer_and_sibling]()
    suite.test[test_kernel_reinterpret_first_bytes_as_unsafe_pointer]()
    suite^.run()
