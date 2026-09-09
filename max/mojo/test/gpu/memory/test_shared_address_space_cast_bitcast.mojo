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
#
# Erasing a SHARED pointer to GENERIC and reinterpreting it as a different
# type used to abort the Metal/AIR compiler instead of raising a normal
# compile error (MOCO-4417).
#
# ===----------------------------------------------------------------------=== #

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation


def _kernel(out_ptr: Pointer[UInt16, MutAnyOrigin]):
    # Keep the buffer as an `Pointer`: this test pins the
    # `Pointer` address-space-cast and bitcast codegen path.
    var smem = unsafe_stack_allocation[
        1,
        UInt32,
        address_space=.SHARED,
    ]()

    if thread_idx.x == 0:
        smem[unsafe_offset=0] = UInt32(0xCAFEBABE)
    barrier()

    # Erase-then-reinterpret is what triggers the crash.
    var generic = smem.unsafe_address_space_cast[.GENERIC]()
    var halves = generic.unsafe_bitcast[UInt16]()

    if thread_idx.x == 0:
        out_ptr[unsafe_offset=0] = halves[unsafe_offset=0]
        out_ptr[unsafe_offset=1] = halves[unsafe_offset=1]


# CHECK-LABEL: == test_shared_address_space_cast_bitcast
def main() raises:
    print("== test_shared_address_space_cast_bitcast")

    with DeviceContext() as ctx:
        var out_device = ctx.enqueue_create_buffer[.uint16](2)
        var compiled = ctx.compile_function[_kernel]()
        ctx.enqueue_function(
            compiled, out_device, grid_dim=(1,), block_dim=(32,)
        )
        ctx.synchronize()

        with out_device.map_to_host() as out_host:
            # Always reads zero on Metal regardless of what's stored; 0xCAFEBABE
            # rules out this being a coincidence of a zero-valued half.
            # CHECK: halves: 0x0 0x0
            print("halves:", hex(out_host[0]), hex(out_host[1]))
