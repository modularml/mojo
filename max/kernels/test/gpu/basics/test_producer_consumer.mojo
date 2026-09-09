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

from std.sys import size_of
from std.math.uutils import umod, udivmod

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu import warp_id as get_warp_id
from max.gpu.host import DeviceContext
from max.gpu.memory import async_copy
from max.gpu.sync import async_copy_arrive
from layout.tma_async import PipelineState, SharedMemBarrier
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._fillers import random
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from std.utils import IndexList


def producer_consumer_kernel[NUM_THREADS: Int]():
    var warp_id = get_warp_id()
    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init(Int32(NUM_THREADS))

    barrier()

    if warp_id == 0:
        if thread_idx.x == 0:
            print("Producer thread_idx: ", thread_idx.x, "warp_idx: ", warp_id)

        _ = mbar[0].arrive()
    else:
        mbar[0].wait(UInt32(mbar[0].arrive()))
        if umod(thread_idx.x, 8) == 0:
            print("Consumer thread_idx:", thread_idx.x, "warp_idx: ", warp_id)


def test_producer_consumer_kernel(ctx: DeviceContext) raises:
    comptime kernel = producer_consumer_kernel[64]
    ctx.enqueue_function[kernel](
        grid_dim=(1),
        block_dim=(64),
    )

    # CHECK-DAG: Producer thread_idx:  0 warp_idx:  0
    # CHECK-DAG: Consumer thread_idx: 32 warp_idx:  1
    # CHECK-DAG: Consumer thread_idx: 40 warp_idx:  1
    # CHECK-DAG: Consumer thread_idx: 48 warp_idx:  1
    # CHECK-DAG: Consumer thread_idx: 56 warp_idx:  1


def producer_consumer_pipeline_kernel[Q_SIZE: Int](num_iters: Int):
    var k_tile_iters = num_iters

    var producer_mbar = unsafe_stack_allocation[
        Q_SIZE,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    var consumer_mbar = unsafe_stack_allocation[
        Q_SIZE,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    comptime for i in range(Q_SIZE):
        if thread_idx.x == 0:
            producer_mbar[i].init(1)
            consumer_mbar[i].init(128)

    barrier()

    var k_tile = 0

    comptime for i in range(Q_SIZE):
        if thread_idx.x == 0:
            # pretend to load into smem tile
            print("prefetch: ", i)
            _ = producer_mbar[i].arrive()
        k_tile += 1
        k_tile_iters -= 1

    var write_state = PipelineState[Q_SIZE]()
    var read_state = PipelineState[Q_SIZE]()

    # producer-consumer loop
    while k_tile_iters > -Q_SIZE:
        var read_idx = read_state.index()
        producer_mbar[read_idx].wait(read_state.phase())

        if thread_idx.x == 0:
            print("consuming: ", read_idx)
        # pretend to do wgmma
        _ = consumer_mbar[read_idx].arrive()
        read_state.step()

        if thread_idx.x == 0:
            var write_idx = write_state.index()
            consumer_mbar[write_idx].wait(write_state.phase())
            print("producing: ", write_idx)
            # pretend to copy into smem tile
            _ = producer_mbar[write_idx].arrive()
            write_state.step()
        k_tile += 1
        k_tile_iters -= 1


def test_producer_consumer_pipeline_kernel(ctx: DeviceContext) raises:
    comptime kernel = producer_consumer_pipeline_kernel[4]
    ctx.enqueue_function[kernel](
        4,
        grid_dim=(1),
        block_dim=(128),
    )

    # CHECK: prefetch:  0
    # CHECK: prefetch:  1
    # CHECK: prefetch:  2
    # CHECK: prefetch:  3
    # CHECK: consuming:  0
    # CHECK: producing:  0
    # CHECK: consuming:  1
    # CHECK: producing:  1
    # CHECK: consuming:  2
    # CHECK: producing:  2
    # CHECK: consuming:  3
    # CHECK: producing:  3


def cpaysnc_producer_consumer_pipeline_kernel[
    num_stages: Int,
](src: Span[Float32, ImmutAnyOrigin], dst: Span[Float32, MutAnyOrigin]):
    comptime size_per_copy = 16 // size_of[DType.float32]()
    comptime size_per_stage = size_per_copy * 128

    var warpgroup_idx, warpgroup_tid = udivmod(thread_idx.x, 128)

    var smem = unsafe_stack_allocation[
        size_per_stage * num_stages,
        DType.float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    # Initialize smem buffer
    if warpgroup_idx == 0:
        for i in range(num_stages):
            var offset = i * size_per_stage + thread_idx.x * size_per_copy

            comptime for j in range(size_per_copy):
                smem[offset + j] = -1000.0

    barrier()

    var produced_mbar = unsafe_stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    var consumed_mbar = unsafe_stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    comptime for i in range(num_stages):
        if thread_idx.x == 0:
            produced_mbar[i].init(128)
            consumed_mbar[i].init(128)

    # Ensure all threads see initialized barriers.
    barrier()

    # producer group
    if warpgroup_idx == 0:
        for i in range(num_stages):
            var offset = i * size_per_stage + thread_idx.x * size_per_copy
            async_copy[16](
                (src.unsafe_ptr() + offset).address_space_cast[.GLOBAL](),
                smem + offset,
            )
            async_copy_arrive(produced_mbar[i].unsafe_ptr())
            _ = produced_mbar[i].arrive()

    # consumer group, add stage index to buffer and write back.
    else:
        var read_pipeline_states = PipelineState[num_stages]()

        for i in range(num_stages):
            produced_mbar[i].wait(read_pipeline_states.phase())

            var offset = i * size_per_stage + warpgroup_tid * size_per_copy

            comptime for j in range(size_per_copy):
                smem[offset + j] += Float32(i)

            read_pipeline_states.step()

        # write back to global memory.
        for i in range(num_stages):
            var offset = i * size_per_stage + warpgroup_tid * size_per_copy

            comptime for j in range(size_per_copy):
                dst[offset + j] = smem[offset + j]


def test_cpasync_producer_consumer_pipeline[
    num_stages: Int
](ctx: DeviceContext) raises:
    comptime size_per_stage = 128 * (16 // size_of[DType.float32]())
    comptime size = num_stages * size_per_stage
    comptime shape1d = IndexList[1](size)

    comptime layout_1d = Layout(UNKNOWN_VALUE)
    var src_device_buffer = ctx.enqueue_create_buffer[.float32](size)
    var src_device = LayoutTensor[.float32, layout_1d](
        src_device_buffer, RuntimeLayout[layout_1d].row_major(shape1d)
    )
    with src_device_buffer.map_to_host() as src_host_buffer:
        random(
            LayoutTensor[.float32, layout_1d](
                src_host_buffer, RuntimeLayout[layout_1d].row_major(shape1d)
            )
        )

    var dst_device_buffer = ctx.enqueue_create_buffer[.float32](size)
    var dst_device = LayoutTensor[.float32, layout_1d](
        dst_device_buffer, RuntimeLayout[layout_1d].row_major(shape1d)
    )

    comptime kernel = cpaysnc_producer_consumer_pipeline_kernel[num_stages]
    ctx.enqueue_function[kernel](
        Span[Float32](unsafe_ptr=src_device.ptr, length=size).as_imm(),
        Span[Float32](unsafe_ptr=dst_device.ptr, length=size),
        grid_dim=(1),
        block_dim=(256),
    )
    ctx.synchronize()

    # src = src_host.tensor
    # dst = dst_host.tensor
    with src_device_buffer.map_to_host() as src:
        with dst_device_buffer.map_to_host() as dst:
            for i in range(num_stages):
                for j in range(size_per_stage):
                    var idx = i * size_per_stage + j
                    if src[idx] + Float32(i) != dst[idx]:
                        print(idx, src[idx], dst[idx])
                        return
                    assert_equal(src[idx] + Float32(i), dst[idx])


def main() raises:
    with DeviceContext() as ctx:
        test_producer_consumer_kernel(ctx)
        test_producer_consumer_pipeline_kernel(ctx)
        test_cpasync_producer_consumer_pipeline[2](ctx)
