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

from std.math import ceildiv
from std.random import rand
from std.sys import argv
from std.sys.info import simd_width_of, size_of

import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.nvidia.tma import TMADescriptor, create_tma_descriptor
from max.gpu import (
    block_dim,
    block_idx,
    thread_idx,
    WARP_SIZE,
    lane_id,
    warp_id,
)
from max.gpu.memory import (
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
)
from max.gpu.sync import (
    barrier,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type

from layout import TileTensor, Coord, Idx, row_major


@always_inline
def block_reduce[
    dtype: DType, max_warps_per_block: Int = 32
](val: Scalar[dtype]) -> Scalar[dtype]:
    var m2_shared = unsafe_stack_allocation[
        max_warps_per_block, dtype, address_space=.SHARED
    ]()
    var m2_broadcast = unsafe_stack_allocation[
        1, dtype, address_space=.SHARED
    ]()

    var tid = thread_idx.x
    for i in range(tid, max_warps_per_block, block_dim.x):
        m2_shared[i] = 0

    if tid == 0:
        m2_broadcast[0] = 0

    barrier()

    var warp_m2 = warp.sum(val)

    var warp_id = warp_id[broadcast=True]()
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0 and lane_idx < max_warps_per_block:
        var block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            m2_shared[lane_idx]
        )
        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


def global_reduction_kernel[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int, _rank: Int](
        idx: IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
](d_out: MutPointer[Scalar[accum_type], MutAnyOrigin], num_cols_dev: Int32,):
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_cols = Int(num_cols_dev)
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * simd_width
    var vec_data = SIMD[accum_type, simd_width](0)

    if idx < num_cols:
        vec_data = input_fn[simd_width, 2](IndexList[2](row, idx)).cast[
            accum_type
        ]()

    var thread_sum = vec_data.reduce_add()

    var block_sum = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_sum
    )

    if thread_idx.x == 0:
        d_out[row] = block_sum


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
def tma_reduction_kernel[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
](
    descriptor: TMADescriptor,
    rows_dev: Int32,
    cols_dev: Int32,
    d_data: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    d_out: MutPointer[Scalar[accum_type], MutAnyOrigin],
):
    # `Int` is not device-passable; widen the fixed-width args.
    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var shmem = external_memory[
        Scalar[dtype], address_space=.SHARED, alignment=128
    ]()
    # Calculate elements offset for this block (row).
    var block_offset = block_idx.x

    # Create barrier for TMA transfer from GMEM to SMEM.
    var mbar = unsafe_stack_allocation[1, Int64, address_space=.SHARED]()

    var descriptor_ptr = Pointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    if thread_idx.x == 0:
        # Add expected_bytes requirement to barrier.
        var expected_bytes = cols * size_of[dtype]()
        mbarrier_arrive_expect_tx_shared(mbar, Int32(expected_bytes))
        cp_async_bulk_tensor_shared_cluster_global(
            shmem,
            descriptor_ptr,
            mbar,
            Index(0, block_offset),
        )

    # Wait for TMA to complete (expected_bytes transferred).
    mbarrier_try_wait_parity_shared(mbar, 0, 10_000_000)

    # Local thread reduction of loaded data.
    var vec_data = SIMD[accum_type, simd_width](0)
    var idx = thread_idx.x * simd_width
    if idx < cols:
        vec_data = shmem.load[width=simd_width](idx).cast[accum_type]()
    var local_sum = vec_data.reduce_add()

    # Block reduction of local sums.
    local_sum = block_reduce(local_sum)

    # Write block result to output buffer for result checking.
    if thread_idx.x == 0:
        d_out[block_idx.x] = local_sum.cast[accum_type]()


def test_tma_block_reduce[
    dtype: DType, use_tma: Bool
](ctx: DeviceContext, rows: Int, cols: Int, benchmark: Bool = False,) raises:
    var n = rows * cols
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime accum_type = get_accum_type[dtype]()

    var h_data = alloc[Scalar[dtype]](n)
    var expected_sum = Scalar[accum_type](0)
    rand[dtype](h_data, n)
    for i in range(n):
        expected_sum += h_data[i].cast[accum_type]()

    var d_data = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(d_data, h_data)

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    var result_host = alloc[Scalar[accum_type]](grid_dim)
    var d_out = ctx.enqueue_create_buffer[accum_type](grid_dim)
    ctx.enqueue_memset(d_out, 0)

    # Define the kernel launch function for benchmarking
    @always_inline
    def kernel_launch(ctx: DeviceContext) raises {imm} -> None:
        comptime if use_tma:
            var tma_desc = create_tma_descriptor[dtype, 2](
                d_data,
                (rows, cols),
                (cols, 1),
                (1, cols),
            )
            # Calculate shared memory size needed per row.
            var shared_mem_bytes = cols * size_of[dtype]()
            comptime kernel = tma_reduction_kernel[
                dtype, accum_type, simd_width
            ]
            ctx.enqueue_function[kernel](
                tma_desc,
                Int32(rows),
                Int32(cols),
                d_data,
                d_out,
                grid_dim=grid_dim,
                block_dim=block_dim,
                shared_mem_bytes=shared_mem_bytes,
            )
        else:
            var data_buf = TileTensor(d_data, row_major(rows, cols))

            # Change the input function to match RMS norm pattern
            @__copy_capture(data_buf)
            @always_inline
            @__parameter
            def input_fn_2d[
                width: Int, _rank: Int
            ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
                var coord = Coord(idx)
                comptime assert data_buf.flat_rank >= coord.flat_rank
                return data_buf.load[width=width](coord)

            comptime kernel = global_reduction_kernel[
                dtype,
                accum_type,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
            ]

            ctx.enqueue_function[kernel](
                d_out,
                Int32(cols),  # num_cols
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

    if benchmark:
        # Run kernel multiple times for benchmarking.
        comptime num_warmup = 5
        comptime num_iters = 100

        # Warmup runs.
        for _ in range(num_warmup):
            kernel_launch(ctx)

        # Timed runs
        var total_time = ctx.execution_time(kernel_launch, num_iters)
        var avg_time_ms = Float64(total_time) / Float64(num_iters) * 1e6

        print(
            "  Average kernel time for:",
            rows,
            "x",
            cols,
            ":",
            avg_time_ms,
            "ms",
        )
    else:
        # Single run for correctness testing.
        kernel_launch(ctx)

    ctx.enqueue_copy(result_host, d_out)
    ctx.synchronize()

    var total_sum = Scalar[accum_type](0)
    for i in range(grid_dim):
        total_sum += result_host[i]

    assert_almost_equal(
        total_sum,
        expected_sum,
        rtol=1e-2,
        atol=1e-2,
    )

    h_data.free()
    result_host.free()
    _ = d_data
    _ = d_out


def main() raises:
    var test_sizes = [128, 256, 512, 1024]
    var depths = [64, 128, 256]
    comptime dtype = DType.bfloat16

    # Parse command line arguments.
    var benchmark = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--benchmark" or args[i] == "--benchmark=yes":
            benchmark = True

    comptime use_tma = True

    with DeviceContext() as ctx:
        for test_size in test_sizes:
            for depth in depths:
                test_tma_block_reduce[dtype, use_tma](
                    ctx, test_size, depth, benchmark=benchmark
                )
