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

from std.sys.info import align_of

from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu import thread_idx
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal


def test_external_shared_mem(ctx: DeviceContext) raises:
    def dynamic_smem_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        var sram = unsafe_stack_allocation[
            16,
            Float32,
            address_space=.SHARED,
        ]()
        var dynamic_sram = external_memory[
            Float32,
            address_space=.SHARED,
            alignment=align_of[Float32](),
        ]()
        dynamic_sram[thread_idx.x] = Float32(thread_idx.x)
        sram[thread_idx.x] = Float32(thread_idx.x)
        barrier()
        data[thread_idx.x] = dynamic_sram[thread_idx.x] + sram[thread_idx.x]

    var res_host_ptr = alloc[Float32](16)
    var res_device = ctx.enqueue_create_buffer[.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    comptime kernel_func = dynamic_smem_kernel
    ctx.enqueue_function[kernel_func, dump_llvm=True](
        res_device,
        grid_dim=1,
        block_dim=16,
        shared_mem_bytes=24960,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(24960),
    )

    ctx.enqueue_copy(res_host_ptr, res_device)
    ctx.synchronize()

    for i in range(16):
        assert_equal(res_host_ptr[i], Float32(2 * i))

    _ = res_device
    res_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        test_external_shared_mem(ctx)
