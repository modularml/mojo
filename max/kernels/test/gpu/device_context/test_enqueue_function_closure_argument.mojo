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

# Passing a capturing closure as a compiled kernel argument.

from max.gpu.host import DeviceContext
from std.builtin.device_passable import DevicePassable
from std.testing import assert_equal


def apply_closure[
    closure_type: (def(Int32, MutPointer[Int32, MutAnyOrigin]) -> None)
    & DevicePassable
    & ImplicitlyCopyable
](value: Int32, dst: MutPointer[Int32, MutAnyOrigin], closure: closure_type,):
    closure(value, dst)


def assert_closure_encodable[ClosureType: DevicePassable]():
    comptime assert not ClosureType._is_convertible_to_device_type[
        ClosureType
    ]()
    comptime assert ClosureType._is_implicitly_encodable_to[ClosureType]()


def test_enqueue_compiled_kernel_accepts_closure_argument(
    ctx: DeviceContext,
) raises:
    var buf = ctx.enqueue_create_buffer[.int32](1)
    buf.enqueue_fill(Int32(0))
    var captured = Int32(42)

    def closure(
        value: Int32, dst: MutPointer[Int32, MutAnyOrigin]
    ) {var captured}:
        dst[] = captured + value

    assert_closure_encodable[type_of(closure)]()

    comptime kernel = apply_closure[type_of(closure)]
    var compiled = ctx.compile_function[kernel]()
    ctx.enqueue_function(
        compiled,
        Int32(1),
        buf,
        closure,
        grid_dim=1,
        block_dim=1,
    )

    var host = ctx.enqueue_create_host_buffer[.int32](1)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    assert_equal(host[0], Int32(43))


def main() raises:
    with DeviceContext() as ctx:
        test_enqueue_compiled_kernel_accepts_closure_argument(ctx)
