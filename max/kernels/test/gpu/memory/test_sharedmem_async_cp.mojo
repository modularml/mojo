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

import std.time

from max.gpu import memory, sync
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation


def copy_via_shared(
    src: ImmPointer[Float32, ImmutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
):
    var thread_id = Int(thread_idx.x)
    var mem_buff: MutPointer[
        Float32, MutAnyOrigin, address_space=.SHARED
    ] = unsafe_stack_allocation[16, Float32, address_space=.SHARED]()
    var src_global: MutPointer[
        Float32, MutAnyOrigin, address_space=.GLOBAL
    ] = src.address_space_cast[.GLOBAL]()

    memory.async_copy[4](
        src_global + thread_id,
        mem_buff + thread_id,
    )

    var m_barrier = unsafe_stack_allocation[
        1, DType.int32, address_space=.SHARED
    ]()
    sync.mbarrier_init(m_barrier, 16)
    sync.mbarrier(m_barrier)
    var state = sync.mbarrier_arrive(m_barrier)
    var not_wait = False
    while not not_wait:
        std.time.sleep(100 * 1e-6)
        not_wait = sync.mbarrier_test_wait(m_barrier, state)

    dst[thread_id] = mem_buff[thread_id]


# CHECK-LABEL: run_copy_via_shared
def run_copy_via_shared(ctx: DeviceContext) raises:
    print("== run_copy_via_shared")
    var in_data = alloc[Float32](16)
    var out_data = alloc[Float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    var in_device = ctx.enqueue_create_buffer[.float32](16)
    var out_device = ctx.enqueue_create_buffer[.float32](16)

    ctx.enqueue_copy(in_device, in_data)
    ctx.enqueue_copy(out_device, out_data)

    comptime kernel = copy_via_shared
    ctx.enqueue_function[kernel](
        in_device,
        out_device,
        grid_dim=(1,),
        block_dim=(16,),
    )

    ctx.enqueue_copy(out_data, out_device)

    ctx.synchronize()

    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 10.0
    # CHECK: 11.0
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    # CHECK: 16.0
    for i in range(16):
        print(out_data.load(i))

    in_data.free()
    out_data.free()
    _ = in_device
    _ = out_device
    _ = copy_via_shared_gpu^


def main() raises:
    with DeviceContext() as ctx:
        run_copy_via_shared(ctx)
