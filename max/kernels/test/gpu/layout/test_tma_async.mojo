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

from std.math import align_up, ceildiv
from std.sys import size_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
from max.gpu.memory import ReduceOp, fence_async_view_proxy
from max.gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from layout import Layout, LayoutTensor
from layout._fillers import arange, random
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_dram_to_sram, copy_sram_to_dram
from layout.tma_async import (
    create_tensor_tile,
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tma_tile,
    RaggedTensorMap,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal

from std.utils.index import Index, IndexList
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout.swizzle import make_swizzle


@__llvm_arg_metadata(ragged_tensor_map, `nvvm.grid_constant`)
def tma_ragged_store_kernel[
    dtype: DType,
    rank: Int,
    descriptor_rank: Int,
    descriptor_shape: IndexList[descriptor_rank],
    swizzle_mode: TensorMapSwizzle,
    remaining_global_dim_rank: Int,
    shared_n: Int,
    sequence_store_length: Int,
    using_max_descriptor_size: Bool = False,
](
    ragged_tensor_map: RaggedTensorMap[
        dtype,
        descriptor_shape,
        remaining_global_dim_rank,
        swizzle_mode=swizzle_mode,
    ],
    sequence_lengths: IndexList[rank],
):
    comptime shared_m = 128

    var smem_tensor = LayoutTensor[
        dtype,
        Layout.row_major(shared_m, shared_n),
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var seq_idx = block_idx.x
    var sequence_length = sequence_lengths[seq_idx]

    if thread_idx.x == 0:
        for i in range(sequence_length * shared_n):
            smem_tensor.ptr[i] = Scalar[dtype](i)

    var smem_iterator = smem_tensor.tiled_iterator[
        sequence_store_length, shared_n, axis=0
    ](0, 0)
    var cum_offset = 0

    for i in range(seq_idx):
        cum_offset += sequence_lengths[i]

    fence_async_view_proxy()

    var coordinates = IndexList[3](fill=0)

    if thread_idx.x == 0:
        ragged_tensor_map.store_ragged_tile[
            using_max_descriptor_size=using_max_descriptor_size
        ](
            coordinates,
            cum_offset,
            sequence_length,
            smem_iterator,
        )

        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape],
):
    comptime tileM = tile_shape[0]
    comptime tileN = tile_shape[1]
    comptime expected_bytes = _idx_product[tile_rank, tile_shape]() * size_of[
        dtype
    ]()

    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()
        mbar[0].expect_bytes(Int32(expected_bytes))
        tma_tile.async_copy(
            tile,
            mbar[0],
            (block_idx.x * tileN, block_idx.y * tileM),
        )
    # Ensure all threads sees initialized mbarrier
    barrier()
    mbar[0].wait()

    var dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


# Test loading tiles along the last axis.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_multiple_loads_kernel[
    dtype: DType,
    layout: Layout,
    tile_rank: Int,
    tile_shape: IndexList[tile_rank],
    thread_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape],
):
    comptime tileM = tile_shape[0]
    comptime tileN = tile_shape[1]
    comptime expected_bytes = _idx_product[tile_rank, tile_shape]() * size_of[
        dtype
    ]()

    comptime N = layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes))
            tma_tile.async_copy(
                tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )
        # Ensure all threads sees initialized mbarrier
        barrier()
        mbar[0].wait(phase)
        phase ^= 1

        var dst_tile = dst.tile[tileM, tileN](block_idx.y, i)
        copy_sram_to_dram[thread_layout](dst_tile, tile)


def sum_index_list[
    rank: Int,
    //,
    index_list: IndexList[rank],
]() -> Int:
    var sum = 0

    comptime for i in range(len(index_list)):
        sum += index_list[i]
    return sum


def max_length[rank: Int, //](index_list: IndexList[rank]) -> Int:
    var mx = 0
    for i in range(len(index_list)):
        mx = max(mx, index_list[i])
    return mx


def test_tma_ragged_store[
    rank: Int,
    //,
    dtype: DType,
    sequence_lengths: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    max_descriptor_length: Int = 8,
    using_max_descriptor_size: Bool = False,
](ctx: DeviceContext) raises:
    comptime depth = swizzle_mode.bytes() // size_of[dtype]()
    comptime total_num_sequences = sum_index_list[sequence_lengths]()
    comptime global_layout = Layout.row_major(total_num_sequences, depth)
    var max_length = max_length(sequence_lengths)

    comptime global_layout_size = global_layout.size()
    var device_buffer = ctx.enqueue_create_buffer[dtype](global_layout_size)

    comptime GlobalTensorType[sequence_length: Int] = LayoutTensor[
        dtype,
        Layout.row_major(sequence_length, depth),
        MutAnyOrigin,
        alignment=128,
    ]

    var global_tensor = GlobalTensorType[total_num_sequences](device_buffer)

    var ragged_tensor_map = RaggedTensorMap[
        dtype,
        IndexList[2](max_descriptor_length, depth),
        0,
        swizzle_mode=swizzle_mode,
    ](
        ctx,
        global_tensor.ptr,
        max_length,
        depth,
        rank,
        depth,
        IndexList[0](),
        IndexList[0](),
    )

    comptime kernel = tma_ragged_store_kernel[
        dtype,
        rank,
        2,
        IndexList[2](max_descriptor_length, depth),
        swizzle_mode,
        0,
        depth,
        max_descriptor_length,
        using_max_descriptor_size,
    ]

    ctx.enqueue_function[kernel](
        ragged_tensor_map,
        sequence_lengths,
        grid_dim=(rank),
        block_dim=(32),
    )

    with device_buffer.map_to_host() as host_buffer:
        var running_sequence = 0

        comptime for i in range(rank):
            comptime sequence_length = sequence_lengths[i]

            var adjusted = host_buffer.as_span()[running_sequence * depth :]
            var global_host_tensor = GlobalTensorType[sequence_length](
                adjusted.unsafe_ptr()
            )

            comptime if swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE:
                for i in range(global_host_tensor.size()):
                    assert_equal(adjusted[i], Scalar[dtype](i))
            else:
                comptime swizzle = make_swizzle[dtype, swizzle_mode]()
                for i in range(global_host_tensor.size()):
                    var swz_offset = swizzle(i)
                    assert_equal(adjusted[swz_offset], Scalar[dtype](i))


def test_tma_load_row_major[
    dtype: DType,
    src_layout: Layout,
    tile_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext) raises:
    comptime M = src_layout.shape[0].value()
    comptime N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime M_roundup = align_up(M, tileM)
    comptime N_roundup = align_up(N, tileN)

    var src = ManagedLayoutTensor[dtype, src_layout](ctx)
    var dst = ManagedLayoutTensor[
        dtype, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    comptime if dtype == .float8_e4m3fn:
        random(src.tensor())
    else:
        arange(src.tensor(), 0)

    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src.device_tensor())
    ctx.synchronize()

    comptime __tileM = type_of(tma_tensor).tile_shape[0]
    comptime __tileN = type_of(tma_tensor).tile_shape[1]
    comptime __thread_layout = Layout.row_major(__tileM, __tileN)
    comptime if load_along_last_dim:
        comptime kernel = test_tma_multiple_loads_kernel[
            type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            type_of(tma_tensor).rank,  # tile rank
            type_of(tma_tensor).tile_shape,  # tile shape
            __thread_layout,  # thread layout
        ]
        ctx.enqueue_function[kernel](
            dst.device_tensor(),
            tma_tensor,
            grid_dim=(1, M_roundup // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        comptime kernel = test_tma_load_kernel[
            type_of(tma_tensor).dtype,
            Layout.row_major(M_roundup, N_roundup),  # dst layout
            type_of(tma_tensor).rank,  # tile rank
            type_of(tma_tensor).tile_shape,  # tile shape
            __thread_layout,  # thread layout
        ]
        ctx.enqueue_function[kernel](
            dst.device_tensor(),
            tma_tensor,
            grid_dim=(N_roundup // tileN, M_roundup // tileM),
            block_dim=(tileM * tileN),
        )

    var src_host = src.tensor()
    var dst_host = dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    src_host[m, n].cast[.float32](),
                    dst_host[m, n].cast[.float32](),
                )
            else:
                assert_equal(dst_host[m, n].cast[.float32](), 0.0)
    ctx.synchronize()
    _ = src^
    _ = dst^


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_async_store_kernel[
    dtype: DType,
    tile_rank: Int,
    tile_shape_param: IndexList[tile_rank],
    desc_shape_param: IndexList[tile_rank],
    thread_layout: Layout,
    layout: Layout,
](
    tma_tile: TMATensorTile[
        dtype, tile_rank, tile_shape_param, desc_shape_param
    ],
    src: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    comptime tileM = tile_shape_param[0]
    comptime tileN = tile_shape_param[1]
    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation[]()

    var src_tile = src.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_dram_to_sram[thread_layout](tile, src_tile)

    barrier()
    fence_async_view_proxy()

    if thread_idx.x == 0:
        tma_tile.async_store(tile, (block_idx.x * tileN, block_idx.y * tileM))
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_async_multiple_store_kernel[
    dtype: DType,
    tile_rank: Int,
    tile_shape_param: IndexList[tile_rank],
    thread_layout: Layout,
    layout: Layout,
](
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape_param],
    src: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    comptime tileM = tile_shape_param[0]
    comptime tileN = tile_shape_param[1]
    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation[]()

    comptime N = layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    for i in range(num_iters):
        var src_tile = src.tile[tileM, tileN](block_idx.y, i)
        copy_dram_to_sram[thread_layout](tile, src_tile)

        barrier()
        fence_async_view_proxy()

        if thread_idx.x == 0:
            tma_tile.async_store(tile, (i * tileN, block_idx.y * tileM))
            cp_async_bulk_commit_group()
            # Wait for the TMA store to finish reading from shared memory
            # before the next iteration overwrites it.
            cp_async_bulk_wait_group[0]()

        barrier()


def test_tma_async_store[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext) raises:
    comptime src_M = src_layout.shape[0].value()
    comptime src_N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime dst_M = dst_layout.shape[0].value()
    comptime dst_N = dst_layout.shape[1].value()

    var src = ManagedLayoutTensor[.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[.float32, dst_layout](ctx)
    arange(src.tensor(), 1)
    arange(dst.tensor(), 100001)
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, dst.device_tensor())

    ctx.synchronize()

    comptime __tileM = type_of(tma_tensor).tile_shape[0]
    comptime __tileN = type_of(tma_tensor).tile_shape[1]
    comptime __thread_layout = Layout.row_major(__tileM, __tileN)
    comptime if load_along_last_dim:
        comptime kernel = test_tma_async_multiple_store_kernel[
            type_of(tma_tensor).dtype,
            type_of(tma_tensor).rank,
            type_of(tma_tensor).tile_shape,
            __thread_layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(1, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        comptime kernel = test_tma_async_store_kernel[
            type_of(tma_tensor).dtype,
            type_of(tma_tensor).rank,
            type_of(tma_tensor).tile_shape,
            type_of(tma_tensor).desc_shape,
            __thread_layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(src_N // tileN, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    ctx.synchronize()

    var src_host = src.tensor()
    var dst_host = dst.tensor()

    # Check M x N keep the same value
    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                src_host[m, n].cast[.float32](),
                dst_host[m, n].cast[.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_async_reduce_kernel[
    dtype: DType,
    tile_rank: Int,
    tile_shape_param: IndexList[tile_rank],
    thread_layout: Layout,
    layout: Layout,
](
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape_param],
    src: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    comptime tileM = tile_shape_param[0]
    comptime tileN = tile_shape_param[1]
    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation[]()

    var src_tile = src.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_dram_to_sram[thread_layout](tile, src_tile)

    barrier()
    fence_async_view_proxy()

    if thread_idx.x == 0:
        tma_tile.async_reduce[reduction_kind=ReduceOp.ADD](
            tile, (block_idx.x * tileN, block_idx.y * tileM)
        )
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
def test_tma_async_multiple_reduce_kernel[
    dtype: DType,
    tile_rank: Int,
    tile_shape_param: IndexList[tile_rank],
    thread_layout: Layout,
    layout: Layout,
](
    tma_tile: TMATensorTile[dtype, tile_rank, tile_shape_param],
    src: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    comptime tileM = tile_shape_param[0]
    comptime tileN = tile_shape_param[1]
    comptime __tile_layout = Layout.row_major(tileM, tileN)
    var tile = LayoutTensor[
        dtype,
        __tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation[]()

    comptime N = layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    for i in range(num_iters):
        var src_tile = src.tile[tileM, tileN](block_idx.y, i)
        copy_dram_to_sram[thread_layout](tile, src_tile)

        barrier()
        fence_async_view_proxy()

        if thread_idx.x == 0:
            tma_tile.async_reduce[reduction_kind=ReduceOp.ADD](
                tile, (i * tileN, block_idx.y * tileM)
            )
            cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def test_tma_async_reduce[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    load_along_last_dim: Bool = False,
](ctx: DeviceContext) raises:
    comptime src_M = src_layout.shape[0].value()
    comptime src_N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime dst_M = dst_layout.shape[0].value()
    comptime dst_N = dst_layout.shape[1].value()

    var src = ManagedLayoutTensor[.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[.float32, dst_layout](ctx)
    arange(src.tensor(), 1)
    arange(dst.tensor(), 3546)
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, dst.device_tensor())

    ctx.synchronize()

    comptime __reduce_tileM = type_of(tma_tensor).tile_shape[0]
    comptime __reduce_tileN = type_of(tma_tensor).tile_shape[1]
    comptime __reduce_thread_layout = Layout.row_major(
        __reduce_tileM, __reduce_tileN
    )
    comptime if load_along_last_dim:
        comptime kernel = test_tma_async_multiple_reduce_kernel[
            type_of(tma_tensor).dtype,
            type_of(tma_tensor).rank,
            type_of(tma_tensor).tile_shape,
            __reduce_thread_layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(1, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    else:
        comptime kernel = test_tma_async_reduce_kernel[
            type_of(tma_tensor).dtype,
            type_of(tma_tensor).rank,
            type_of(tma_tensor).tile_shape,
            __reduce_thread_layout,
            src_layout,
        ]
        ctx.enqueue_function[kernel](
            tma_tensor,
            src.device_tensor(),
            grid_dim=(src_N // tileN, src_M // tileM),
            block_dim=(tileM * tileN),
        )
    ctx.synchronize()

    var src_host = src.tensor()
    var dst_host = dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                src_host[m, n].cast[.float32]()
                + 3546
                + Float32(m * dst_N)
                + Float32(n),
                dst_host[m, n].cast[.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


# Test loading tiles along the last axis.
@__llvm_arg_metadata(a_tma_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_tile, `nvvm.grid_constant`)
def test_tma_loads_two_buffers_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_rank: Int,
    b_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    b_tile_shape: IndexList[b_tile_rank],
    a_thread_layout: Layout,
    b_thread_layout: Layout,
](
    a_dst: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b_dst: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    a_tma_tile: TMATensorTile[dtype, a_tile_rank, a_tile_shape],
    b_tma_tile: TMATensorTile[dtype, b_tile_rank, b_tile_shape],
):
    comptime tileM = a_tile_shape[0]
    comptime tileN = a_tile_shape[1]
    comptime expected_bytes = _idx_product[
        a_tile_rank, a_tile_shape
    ]() * size_of[dtype]()

    comptime N = a_layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    comptime __a_tile_layout = Layout.row_major(tileM, tileN)
    var a_tile = LayoutTensor[
        dtype,
        __a_tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime __b_tile_layout = Layout.row_major(
        b_tile_shape[0], b_tile_shape[1]
    )
    var b_tile = LayoutTensor[
        dtype,
        __b_tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes * 2))
            a_tma_tile.async_copy(
                a_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )
            b_tma_tile.async_copy(
                b_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        var a_dst_tile = a_dst.tile[tileM, tileN](block_idx.y, i)
        var b_dst_tile = b_dst.tile[tileM, tileN](block_idx.y, i)
        copy_sram_to_dram[a_thread_layout](a_dst_tile, a_tile)
        copy_sram_to_dram[b_thread_layout](b_dst_tile, b_tile)


def test_tma_load_two_buffers_row_major[
    src_layout: Layout, tile_layout: Layout, load_along_last_dim: Bool = False
](ctx: DeviceContext) raises:
    comptime M = src_layout.shape[0].value()
    comptime N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime M_roundup = align_up(M, tileM)
    comptime N_roundup = align_up(N, tileN)

    var a_src = ManagedLayoutTensor[.float32, src_layout](ctx)
    var b_src = ManagedLayoutTensor[.float32, src_layout](ctx)

    var a_dst = ManagedLayoutTensor[
        .float32, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    var b_dst = ManagedLayoutTensor[
        .float32, Layout.row_major(M_roundup, N_roundup)
    ](ctx)

    arange(a_src.tensor(), 1)
    arange(b_src.tensor(), 1)
    var a_tma_tensor = create_tma_tile[tileM, tileN](ctx, a_src.device_tensor())
    var b_tma_tensor = create_tma_tile[tileM, tileN](ctx, b_src.device_tensor())
    ctx.synchronize()

    comptime __a_tileM = type_of(a_tma_tensor).tile_shape[0]
    comptime __a_tileN = type_of(a_tma_tensor).tile_shape[1]
    comptime __b_tileM = type_of(b_tma_tensor).tile_shape[0]
    comptime __b_tileN = type_of(b_tma_tensor).tile_shape[1]
    comptime kernel = test_tma_loads_two_buffers_kernel[
        type_of(a_tma_tensor).dtype,
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        Layout.row_major(M_roundup, N_roundup),  # dst layout
        type_of(a_tma_tensor).rank,
        type_of(b_tma_tensor).rank,
        type_of(a_tma_tensor).tile_shape,
        type_of(b_tma_tensor).tile_shape,
        Layout.row_major(__a_tileM, __a_tileN),  # thread layout
        Layout.row_major(__b_tileM, __b_tileN),  # thread layout
    ]
    ctx.enqueue_function[kernel](
        a_dst.device_tensor(),
        b_dst.device_tensor(),
        a_tma_tensor,
        b_tma_tensor,
        grid_dim=(1, M_roundup // tileM),
        block_dim=(tileM * tileN),
    )

    var a_src_host = a_src.tensor()
    var a_dst_host = a_dst.tensor()

    var b_src_host = b_src.tensor()
    var b_dst_host = b_dst.tensor()

    # Check M x N keep the same value and others in M_roundup x N_roundup
    # are set to zeros.
    for m in range(M_roundup):
        for n in range(N_roundup):
            if m < M and n < N:
                assert_equal(
                    a_src_host[m, n].cast[.float32](),
                    a_dst_host[m, n].cast[.float32](),
                )

                assert_equal(
                    b_src_host[m, n].cast[.float32](),
                    b_dst_host[m, n].cast[.float32](),
                )

            else:
                assert_equal(a_dst_host[m, n].cast[.float32](), 0.0)
                assert_equal(b_dst_host[m, n].cast[.float32](), 0.0)
    ctx.synchronize()
    _ = a_src^
    _ = a_dst^

    _ = b_src^
    _ = b_dst^


# Test loading tiles along the last axis.
@__llvm_arg_metadata(a_tma_dst_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_dst_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_tma_src_tile, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_src_tile, `nvvm.grid_constant`)
def test_tma_loads_and_store_two_buffers_kernel[
    dtype: DType,
    a_tile_rank: Int,
    b_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    b_tile_shape: IndexList[b_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    /,
    *,
    a_layout: Layout,
    b_layout: Layout,
](
    a_tma_dst_tile: TMATensorTile[
        dtype, a_tile_rank, a_tile_shape, a_desc_shape
    ],
    b_tma_dst_tile: TMATensorTile[
        dtype, b_tile_rank, b_tile_shape, b_desc_shape
    ],
    a_tma_src_tile: TMATensorTile[
        dtype, a_tile_rank, a_tile_shape, a_desc_shape
    ],
    b_tma_src_tile: TMATensorTile[
        dtype, b_tile_rank, b_tile_shape, b_desc_shape
    ],
):
    comptime tileM = a_tile_shape[0]
    comptime tileN = a_tile_shape[1]
    comptime expected_bytes = _idx_product[
        a_tile_rank, a_tile_shape
    ]() * size_of[dtype]()

    comptime N = a_layout.shape[1].value()
    comptime num_iters = ceildiv(N, tileN)

    comptime __a_tile_layout = Layout.row_major(tileM, tileN)
    var a_tile = LayoutTensor[
        dtype,
        __a_tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime __b_tile_layout = Layout.row_major(
        b_tile_shape[0], b_tile_shape[1]
    )
    var b_tile = LayoutTensor[
        dtype,
        __b_tile_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()

    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes * 2))
            a_tma_src_tile.async_copy(
                a_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )
            b_tma_src_tile.async_copy(
                b_tile,
                mbar[0],
                (i * tileN, block_idx.y * tileM),
            )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        fence_async_view_proxy()

        if thread_idx.x == 0:
            a_tma_dst_tile.async_store(a_tile, (i * tileN, block_idx.y * tileM))
            b_tma_dst_tile.async_store(b_tile, (i * tileN, block_idx.y * tileM))
            cp_async_bulk_commit_group()

        cp_async_bulk_wait_group[0]()


def test_tma_load_and_store_two_buffers_row_major[
    src_layout: Layout, tile_layout: Layout, dst_layout: Layout
](ctx: DeviceContext) raises:
    comptime M = src_layout.shape[0].value()
    comptime N = src_layout.shape[1].value()
    comptime tileM = tile_layout.shape[0].value()
    comptime tileN = tile_layout.shape[1].value()
    comptime dst_M = dst_layout.shape[0].value()
    comptime dst_N = dst_layout.shape[1].value()

    var a_src = ManagedLayoutTensor[.float32, src_layout](ctx)
    var b_src = ManagedLayoutTensor[.float32, src_layout](ctx)
    var a_dst = ManagedLayoutTensor[.float32, dst_layout](ctx)
    var b_dst = ManagedLayoutTensor[.float32, dst_layout](ctx)

    # Initialize destinations to known values.
    comptime a_dst_value = 1.5
    comptime b_dst_value = 1.25

    var a_dst_host = a_dst.tensor()
    var b_dst_host = b_dst.tensor()
    # Ensure that the buffers have been fully created before accessing their data.
    ctx.synchronize()
    for m in range(dst_M):
        for n in range(dst_N):
            a_dst_host[m, n] = a_dst_value
            b_dst_host[m, n] = b_dst_value

    arange(a_src.tensor(), 1)
    arange(b_src.tensor(), 1)
    var a_tma_src_tensor = create_tensor_tile[Index(tileM, tileN)](
        ctx, a_src.device_tensor()
    )
    var b_tma_src_tensor = create_tensor_tile[Index(tileM, tileN)](
        ctx, b_src.device_tensor()
    )
    var a_tma_dst_tensor = create_tensor_tile[Index(tileM, tileN)](
        ctx, a_dst.device_tensor()
    )
    var b_tma_dst_tensor = create_tensor_tile[Index(tileM, tileN)](
        ctx, b_dst.device_tensor()
    )
    ctx.synchronize()

    comptime kernel = test_tma_loads_and_store_two_buffers_kernel[
        type_of(a_tma_src_tensor).dtype,
        type_of(a_tma_src_tensor).rank,
        type_of(b_tma_src_tensor).rank,
        type_of(a_tma_src_tensor).tile_shape,
        type_of(b_tma_src_tensor).tile_shape,
        type_of(a_tma_src_tensor).desc_shape,
        type_of(b_tma_src_tensor).desc_shape,
        a_layout=dst_layout,  # dst layout
        b_layout=dst_layout,  # dst layout
    ]
    ctx.enqueue_function[kernel](
        a_tma_dst_tensor,
        b_tma_dst_tensor,
        a_tma_src_tensor,
        b_tma_src_tensor,
        grid_dim=(1, dst_M // tileM),
        block_dim=(tileM * tileN),
    )

    var a_src_host = a_src.tensor()
    a_dst_host = a_dst.tensor()

    var b_src_host = b_src.tensor()
    b_dst_host = b_dst.tensor()

    for m in range(dst_M):
        for n in range(dst_N):
            if m < M and n < N:
                assert_equal(
                    a_src_host[m, n].cast[.float32](),
                    a_dst_host[m, n].cast[.float32](),
                )

                assert_equal(
                    b_src_host[m, n].cast[.float32](),
                    b_dst_host[m, n].cast[.float32](),
                )

            else:
                assert_equal(a_dst_host[m, n].cast[.float32](), a_dst_value)
                assert_equal(b_dst_host[m, n].cast[.float32](), b_dst_value)

    ctx.synchronize()
    _ = a_src^
    _ = a_dst^

    _ = b_src^
    _ = b_dst^


def main() raises:
    with DeviceContext() as ctx:
        print("test_tma_load_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(9, 24),
            tile_layout=Layout.row_major(3, 8),
        ](ctx)
        print("test_tma_load_oob_fill_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(7, 8),
            tile_layout=Layout.row_major(4, 4),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(10, 12),
            tile_layout=Layout.row_major(4, 8),
        ](ctx)

        print("test_tma_multiple_loads_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(12, 16),
            tile_layout=Layout.row_major(4, 4),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(24, 80),
            tile_layout=Layout.row_major(3, 16),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_multiple_loads_oob_fill_f32")
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(6, 20),
            tile_layout=Layout.row_major(4, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float32,
            src_layout=Layout.row_major(9, 60),
            tile_layout=Layout.row_major(8, 16),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_load_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(8, 32),
            tile_layout=Layout.row_major(4, 16),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(9, 48),
            tile_layout=Layout.row_major(3, 16),
        ](ctx)
        print("test_tma_load_oob_fill_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(7, 32),
            tile_layout=Layout.row_major(4, 16),
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(10, 48),
            tile_layout=Layout.row_major(4, 32),
        ](ctx)

        print("test_tma_multiple_loads_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(12, 64),
            tile_layout=Layout.row_major(4, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(24, 160),
            tile_layout=Layout.row_major(3, 64),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_multiple_loads_oob_fill_f8e4m3fn")
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(6, 80),
            tile_layout=Layout.row_major(4, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_row_major[
            dtype=DType.float8_e4m3fn,
            src_layout=Layout.row_major(9, 240),
            tile_layout=Layout.row_major(8, 64),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_store")
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
            dst_layout=Layout.row_major(8, 8),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(32, 24),
            tile_layout=Layout.row_major(16, 8),
            dst_layout=Layout.row_major(32, 24),
        ](ctx)

        print("test_tma_multiple_async_store")
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
            dst_layout=Layout.row_major(8, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(9, 24),
            tile_layout=Layout.row_major(3, 8),
            dst_layout=Layout.row_major(9, 24),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_store_oob")
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(6, 8),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(32, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(26, 8),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(6, 4),
        ](ctx)
        test_tma_async_store[
            src_layout=Layout.row_major(32, 16),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(26, 12),
        ](ctx)

        print("test_tma_async_reduce")
        test_tma_async_reduce[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
            dst_layout=Layout.row_major(8, 8),
        ](ctx)
        test_tma_async_reduce[
            src_layout=Layout.row_major(9, 24),
            tile_layout=Layout.row_major(3, 8),
            dst_layout=Layout.row_major(9, 24),
        ](ctx)

        print("test_tma_multiple_async_reduce")
        test_tma_async_reduce[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(4, 4),
            dst_layout=Layout.row_major(8, 8),
            load_along_last_dim=True,
        ](ctx)
        test_tma_async_reduce[
            src_layout=Layout.row_major(9, 24),
            tile_layout=Layout.row_major(3, 8),
            dst_layout=Layout.row_major(9, 24),
            load_along_last_dim=True,
        ](ctx)

        print("test_tma_async_reduce_oob")
        test_tma_async_reduce[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(6, 8),
        ](ctx)
        test_tma_async_reduce[
            src_layout=Layout.row_major(32, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(26, 8),
        ](ctx)
        test_tma_async_reduce[
            src_layout=Layout.row_major(8, 8),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(6, 4),
        ](ctx)
        test_tma_async_reduce[
            src_layout=Layout.row_major(32, 16),
            tile_layout=Layout.row_major(8, 8),
            dst_layout=Layout.row_major(26, 12),
        ](ctx)
        print("test_tma_load_two_buffer_row_major")
        test_tma_load_two_buffers_row_major[
            src_layout=Layout.row_major(32, 64),
            tile_layout=Layout.row_major(8, 16),
            load_along_last_dim=True,
        ](ctx)
        test_tma_load_two_buffers_row_major[
            src_layout=Layout.row_major(9, 60),
            tile_layout=Layout.row_major(8, 16),
        ](ctx)
        print("test_tma_load_and_store_two_buffer")
        test_tma_load_and_store_two_buffers_row_major[
            src_layout=Layout.row_major(32, 64),
            tile_layout=Layout.row_major(8, 16),
            dst_layout=Layout.row_major(32, 64),
        ](ctx)
        test_tma_load_and_store_two_buffers_row_major[
            src_layout=Layout.row_major(32, 64),
            tile_layout=Layout.row_major(16, 16),
            dst_layout=Layout.row_major(40, 64),
        ](ctx)

        print("test_2D_TMA_ragged_store")

        test_tma_ragged_store[
            DType.bfloat16,
            IndexList[3](55, 11, 32),
        ](ctx)

        test_tma_ragged_store[
            DType.bfloat16,
            IndexList[3](55, 11, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_ragged_store[
            DType.bfloat16,
            IndexList[3](55, 11, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        test_tma_ragged_store[
            DType.bfloat16,
            IndexList[3](55, 11, 32),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_ragged_store[
            DType.bfloat16,
            IndexList[3](8, 4, 2),
            using_max_descriptor_size=True,
        ](ctx)
