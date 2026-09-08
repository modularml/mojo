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
from std.math.uutils import ufloordiv

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    cluster_sync,
    cluster_sync_acquire,
    cluster_sync_release,
    clusterlaunchcontrol_query_cancel_get_first_ctaid,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_try_cancel,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceContext
from max.gpu import block_id_in_cluster, lane_id
from max.gpu.intrinsics import Scope
from max.gpu.memory import fence_mbarrier_init
from layout.tma_async import PipelineState, SharedMemBarrier
from std.memory import unsafe_stack_allocation
from std.testing import assert_almost_equal

from std.utils.static_tuple import StaticTuple

from layout import TileTensor, row_major


# Derived from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-example-vector-scalar-multiplication
def cluster_launch_control(
    data: MutPointer[Float32, MutAnyOrigin], n_dev: Int32
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var n = Int(n_dev)
    var result = unsafe_stack_allocation[
        1,
        UInt128,
        address_space=.SHARED,
        alignment=16,
    ]()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    var bx = UInt32(block_idx.x)
    var tidx = bx * UInt32(block_dim.x) + UInt32(thread_idx.x)
    if tidx < UInt32(n):
        data[tidx] = 1.0

    var phase: UInt32 = 0

    # Single thread barrier.
    if thread_idx.x == 0:
        mbar[0].init(1)

    var alpha = Int64(thread_idx.x)

    # Work-straling loop.
    while True:
        barrier()

        if thread_idx.x == 0:
            # Acquire write of result in the async proxy.
            # Matches `ptx::fence_proxy_async_generic_sync_restrict`.
            cluster_sync_acquire()

            # Matches `cg::invoke_one`. Selects the first thread in the block.
            if elect_one_sync_with_mask(mask=1):
                clusterlaunchcontrol_try_cancel(result, mbar.bitcast[Int64]())

            # Matches `ptx::mbarrier_arrive_expect_tx`.
            _ = mbar[0].expect_bytes_relaxed(Int32(2 * size_of[UInt64]()))

        if tidx < UInt32(n):
            data[tidx] *= Float32(alpha)

        # Cancellation request synchronization:
        mbar[0].wait_acquire[Scope.BLOCK](phase)
        phase ^= 1

        # Cancellation request decoding.
        if not clusterlaunchcontrol_query_cancel_is_canceled(result):
            break

        bx = clusterlaunchcontrol_query_cancel_get_first_ctaid["x"](result)

        # Releases read result to the async proxy.
        # Matches `ptx::fence_proxy_async_generic_sync_restrict`.
        cluster_sync_release()


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
def pipeline_test_kernel[
    num_stages: Int, cluster_shape: StaticTuple[Int32, 3]
]():
    var clc_response = unsafe_stack_allocation[
        num_stages,
        UInt128,
        address_space=.SHARED,
        alignment=16,
    ]()

    var full_mbar = unsafe_stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=16,
    ]()

    var empty_mbar = unsafe_stack_allocation[
        num_stages,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=16,
    ]()

    comptime CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]

    comptime NUM_PRODUCER = 32
    comptime NUM_CONSUMERS_PER_CTA = 32
    comptime consumer_arv_count = NUM_PRODUCER + (
        NUM_CONSUMERS_PER_CTA * CLUSTER_SIZE
    )
    comptime producer_arv_count = 1

    var is_first_block_in_cluster = (
        block_id_in_cluster.x == 0 and block_id_in_cluster.y == 0
    )
    var wid = ufloordiv(thread_idx.x, 32)

    var pipeline_state = PipelineState[num_stages]()
    var pipeline_state_write = PipelineState[num_stages](0, 1, 0)

    if elect_one_sync():
        comptime for i in range(num_stages):
            full_mbar[i].init(producer_arv_count)
            empty_mbar[i].init(consumer_arv_count)

    fence_mbarrier_init()
    cluster_sync()

    var is_producer = is_first_block_in_cluster and wid == 0
    var is_consumer = wid == 4
    var is_valid = False

    while True:
        if is_producer:
            var write_idx = pipeline_state_write.index()
            empty_mbar[write_idx].wait(pipeline_state_write.phase())
            var pred: UInt32 = UInt32(1 if lane_id() < Int(CLUSTER_SIZE) else 0)
            full_mbar[write_idx].arrive_and_expect_bytes(
                Int32(2 * size_of[UInt64]()), UInt32(lane_id()), pred
            )
            # The warp sync ensures expect_tx is completed.
            if elect_one_sync():
                clusterlaunchcontrol_try_cancel[multicast=True](
                    clc_response + write_idx,
                    (full_mbar + write_idx).bitcast[Int64](),
                )

            pipeline_state_write.step()

        if is_producer or is_consumer:
            var full_idx = pipeline_state.index()
            full_mbar[full_idx].wait(pipeline_state.phase())
            is_valid = Bool(
                clusterlaunchcontrol_query_cancel_is_canceled(
                    clc_response + full_idx
                )
            )
            empty_mbar[full_idx].arrive_cluster(0)

            pipeline_state.step()

        if thread_idx.x == 128:
            print("cancelled: ", is_valid, block_idx.x, block_idx.y)
        if not is_valid:
            break

    cluster_sync()


def test_cluster_launch_control(ctx: DeviceContext) raises:
    comptime n = 4000

    var data = ctx.enqueue_create_buffer[.float32](n)

    comptime kernel = cluster_launch_control
    ctx.enqueue_function[kernel](
        data,
        Int32(n),
        grid_dim=((n + 1023) // 1024),
        block_dim=(1024),
    )

    var data_host_ptr = alloc[Float32](n)
    var data_host = TileTensor(data_host_ptr, row_major[n]())

    ctx.enqueue_copy(data_host_ptr, data)
    ctx.synchronize()

    for i in range(n):
        assert_almost_equal(data_host[i], Float32(i % 1024))


def test_cluster_pipeline(ctx: DeviceContext) raises:
    comptime kernel = pipeline_test_kernel[1, StaticTuple[Int32, 3](2, 2, 1)]
    ctx.enqueue_function[kernel](
        # Use more blocks than SMs to ensure cancel happens.
        grid_dim=(4, 4),
        block_dim=(256),
    )

    # CHECK-DAG: cancelled:  False 3 3
    # CHECK-DAG: cancelled:  False 1 0
    # CHECK-DAG: cancelled:  False 1 1
    # CHECK-DAG: cancelled:  False 2 0
    # CHECK-DAG: cancelled:  False 2 1
    # CHECK-DAG: cancelled:  False 1 2
    # CHECK-DAG: cancelled:  False 0 2
    # CHECK-DAG: cancelled:  False 1 3
    # CHECK-DAG: cancelled:  False 0 3
    # CHECK-DAG: cancelled:  False 3 0
    # CHECK-DAG: cancelled:  False 0 0
    # CHECK-DAG: cancelled:  False 3 2
    # CHECK-DAG: cancelled:  False 3 1
    # CHECK-DAG: cancelled:  False 0 1
    # CHECK-DAG: cancelled:  False 2 2
    # CHECK-DAG: cancelled:  False 2 3


def main() raises:
    with DeviceContext() as ctx:
        test_cluster_launch_control(ctx)
        test_cluster_pipeline(ctx)
