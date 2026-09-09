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

from max.gpu.host import DeviceContext
from max.gpu import thread_idx
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from std.testing import assert_equal


def test_external_shared_mem(ctx: DeviceContext) raises:
    print("== test_external_shared_mem")

    def dynamic_smem_kernel(data: MutPointer[Float32, MutAnyOrigin]):
        var dynamic_sram = external_memory[
            Float32, address_space=.SHARED, alignment=4
        ]()
        dynamic_sram[thread_idx.x] = Float32(thread_idx.x)
        barrier()
        data[thread_idx.x] = dynamic_sram[thread_idx.x]

    var res_host_ptr = ctx.enqueue_create_host_buffer[.float32](16)
    ctx.synchronize()
    for i in range(16):
        res_host_ptr[i] = Float32(0)
    var res_device = ctx.enqueue_create_buffer[.float32](16)

    ctx.enqueue_copy(res_device, res_host_ptr)

    comptime kernel = dynamic_smem_kernel

    # 16 KB allocation — valid on all platforms including Metal (32 KB limit).
    ctx.enqueue_function[kernel](
        res_device,
        grid_dim=1,
        block_dim=16,
        shared_mem_bytes=16 * 1024,
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    ctx.synchronize()

    var expected: List[Float32] = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ]
    for i in range(16):
        print(res_host_ptr[i])
        assert_equal(res_host_ptr[i], expected[i])

    _ = res_device


def main() raises:
    with DeviceContext() as ctx:
        test_external_shared_mem(ctx)
