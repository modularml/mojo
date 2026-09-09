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
from std.sys import simd_width_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)
from layout import *
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import (
    binary_op_type,
    copy_dram_to_local,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_local,
    copy_local_to_shared,
    copy_sram_to_dram,
)

from std.utils import IndexList


@always_inline
def add_op[
    dtype: DType, width: SIMDLength
](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return lhs + rhs


# ----------------------------------------------------------------------
# async copy tests
# ----------------------------------------------------------------------
def async_copy_kernel[
    input_layout: Layout,
    BM: Int,
    BN: Int,
](input: LayoutTensor[.float32, input_layout, MutAnyOrigin]):
    var input_tile = input.tile[BM, BN](block_idx.y, block_idx.x)

    var smem_tile = LayoutTensor[
        .float32,
        Layout(IntTuple(BM, BN)),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async(input_tile)
    async_copy_wait_all()

    var tx = thread_idx.x
    var ty = thread_idx.y
    smem_tile[tx, ty] += Float32(ty)

    input_tile.copy_from(smem_tile)


def test_async_copy[
    layout: Layout, M: Int, N: Int, BM: Int, BN: Int
](ctx: DeviceContext) raises:
    print("=== test_async_copy")

    comptime managed_layout_tensor_type = ManagedLayoutTensor[
        .float32,
        layout,
    ]

    comptime element_type = managed_layout_tensor_type.element_type
    comptime idx_type = managed_layout_tensor_type.index_type

    comptime runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, N))

    var input = ManagedLayoutTensor[.float32, layout](runtime_layout, ctx)

    arange(input.tensor())

    comptime kernel = async_copy_kernel[layout, BM, BN]
    ctx.enqueue_function[kernel](
        input.device_tensor(), grid_dim=(N // BN, M // BM), block_dim=(BM, BN)
    )

    ctx.synchronize()

    print(input.tensor())


def run_async_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_async_copy
    # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
    # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
    # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
    # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
    # CHECK: 24.0   26.0   28.0   27.0   29.0   31.0
    # CHECK: 30.0   32.0   34.0   33.0   35.0   37.0
    test_async_copy[
        Layout.row_major(6, 6),
        M=6,
        N=6,
        BM=2,
        BN=3,
    ](ctx)

    # CHECK: === test_async_copy
    # CHECK: 0.0   2.0   4.0   3.0   5.0   7.0
    # CHECK: 6.0   8.0   10.0   9.0   11.0   13.0
    # CHECK: 12.0   14.0   16.0   15.0   17.0   19.0
    # CHECK: 18.0   20.0   22.0   21.0   23.0   25.0
    # CHECK: 24.0   26.0   28.0   27.0   29.0   31.0
    # CHECK: 30.0   32.0   34.0   33.0   35.0   37.0
    test_async_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=6,
        N=6,
        BM=2,
        BN=3,
    ](ctx)


# ----------------------------------------------------------------------
# swizzle copy tests
# ----------------------------------------------------------------------


def swizzle_copy[
    dtype: DType,
    layout: Layout,
    BM: Int,
    BK: Int,
    num_threads: Int,
](
    a: LayoutTensor[dtype, layout, MutAnyOrigin],
    b: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    comptime simd_size = simd_width_of[dtype]()

    # Double buffer in shared memory.
    var a_smem_tile = (
        LayoutTensor[
            dtype,
            Layout.row_major(BM, BK),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .fill(0)
    )

    comptime thread_layout = Layout.row_major(
        num_threads * simd_size // BK, BK // simd_size
    )

    copy_dram_to_sram_async[thread_layout=thread_layout, swizzle=True](
        a_smem_tile.vectorize[1, simd_size](),
        a.tile[BM, BK](block_idx.x, 0).vectorize[1, simd_size](),
    )

    async_copy_wait_all()
    barrier()

    # Write current stage to global memory.
    var b_gmem_tile = b.tile[BM, BK](block_idx.x, 0)
    var b_gmem_frag = b_gmem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](thread_idx.x)
    var a_smem_frag = a_smem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](thread_idx.x)
    b_gmem_frag.copy_from(a_smem_frag)


def test_swizzle_copy[
    layout: Layout,
    M: Int,
    K: Int,
    BM: Int,
    BK: Int,
    num_threads: Int,
    skew_M: Int = 0,
](ctx: DeviceContext) raises:
    print("=== test_swizzle_copy")

    comptime managed_layout_tensor_type = ManagedLayoutTensor[
        .float32,
        layout,
    ]

    comptime element_type = managed_layout_tensor_type.element_type
    comptime idx_type = managed_layout_tensor_type.index_type

    comptime a_runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M - skew_M, K))

    comptime b_runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, K))

    var a_tensor = ManagedLayoutTensor[
        .float32,
        layout,
    ](a_runtime_layout, ctx)
    arange(a_tensor.tensor())

    var b_tensor = ManagedLayoutTensor[
        .float32,
        layout,
    ](b_runtime_layout, ctx)

    comptime copy = swizzle_copy[
        DType.float32,
        layout,
        BM,
        BK,
        num_threads,
    ]

    ctx.enqueue_function[copy](
        a_tensor.device_tensor(),
        b_tensor.device_tensor(),
        grid_dim=(ceildiv(M, BM), 1, 1),
        block_dim=(num_threads, 1, 1),
    )

    ctx.synchronize()
    print(b_tensor.tensor())


def run_swizzle_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
    # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
    test_swizzle_copy[
        Layout.row_major(8, 16),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
    ](ctx)

    # CHECK: == test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
    # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
    test_swizzle_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
    ](ctx)

    # CHECK: === test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 108.0 109.0 110.0 111.0 104.0 105.0 106.0 107.0 100.0 101.0 102.0 103.0 96.0 97.0 98.0 99.0
    # CHECK: 124.0 125.0 126.0 127.0 120.0 121.0 122.0 123.0 116.0 117.0 118.0 119.0 112.0 113.0 114.0 115.0
    test_swizzle_copy[
        Layout.row_major(UNKNOWN_VALUE, 16),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
    ](ctx)


# ----------------------------------------------------------------------
# partial copy_dram_to_sram_async tests
# ----------------------------------------------------------------------


@always_inline
def partial_copy_dram_to_sram_async_kernel[
    layout: Layout,
    thread_layout: Layout,
    num_threads: Int,
    block_dim_count: Int,
](
    input: LayoutTensor[.float32, layout, MutAnyOrigin],
    output: LayoutTensor[.float32, layout, MutAnyOrigin],
):
    var smem_tile = (
        LayoutTensor[
            .float32,
            layout,
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .fill(-1.0)
    )

    copy_dram_to_sram_async[
        thread_layout=thread_layout,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](smem_tile.vectorize[1, 4](), input.vectorize[1, 4]())

    async_copy_commit_group()
    async_copy_wait_all()

    copy_sram_to_dram[
        thread_layout=thread_layout,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](
        output.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


def test_partial_copy_dram_to_sram_async[
    layout: Layout,
    thread_layout: Layout,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    block_dim_count: Int = 1,
](ctx: DeviceContext) raises:
    print("=== test_partial_copy_dram_to_sram_async")

    var input = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    arange(input.tensor())

    var output = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    comptime num_threads = block_dim_x * block_dim_y * block_dim_z
    comptime kernel_type = partial_copy_dram_to_sram_async_kernel[
        layout, thread_layout, num_threads, block_dim_count
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        output.device_tensor(),
        grid_dim=(1,),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )

    ctx.synchronize()

    print(output.tensor())


def run_partial_copy_dram_to_sram_async_tests(ctx: DeviceContext) raises:
    # CHECK: === test_partial_copy_dram_to_sram_async
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    test_partial_copy_dram_to_sram_async[
        layout=Layout.row_major(2, 16),
        thread_layout=Layout.row_major(2, 4),
        block_dim_x=32,
    ](ctx)

    # CHECK: === test_partial_copy_dram_to_sram_async
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    test_partial_copy_dram_to_sram_async[
        layout=Layout.row_major(2, 16),
        thread_layout=Layout.row_major(2, 4),
        block_dim_x=2,
        block_dim_y=2,
        block_dim_z=32,
        block_dim_count=3,
    ](ctx)


# ----------------------------------------------------------------------
# copy_dram_to_sram tests
# ----------------------------------------------------------------------


@always_inline
def copy_dram_to_sram_kernel[
    layout: Layout,
    thread_layout: Layout,
    num_threads: Int,
    block_dim_count: Int,
](
    input: LayoutTensor[.float32, layout, MutAnyOrigin],
    output: LayoutTensor[.float32, layout, MutAnyOrigin],
):
    var smem_tile = (
        LayoutTensor[
            .float32,
            layout,
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .fill(-1.0)
    )

    copy_dram_to_sram[
        thread_layout=thread_layout,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](smem_tile.vectorize[1, 4](), input.vectorize[1, 4]())

    barrier()

    copy_sram_to_dram[
        thread_layout=thread_layout,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](
        output.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


def test_copy_dram_to_sram[
    layout: Layout,
    thread_layout: Layout,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    block_dim_count: Int = 1,
](ctx: DeviceContext) raises:
    print("=== test_copy_dram_to_sram")

    var input = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    arange(input.tensor())

    var output = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    comptime num_threads = block_dim_x * block_dim_y * block_dim_z
    comptime kernel_type = copy_dram_to_sram_kernel[
        layout, thread_layout, num_threads, block_dim_count
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        output.device_tensor(),
        grid_dim=(1,),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )

    ctx.synchronize()

    print(output.tensor())


def run_copy_dram_to_sram_tests(ctx: DeviceContext) raises:
    # CHECK: === test_copy_dram_to_sram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_dram_to_sram[
        layout=Layout.row_major(8, 8),
        thread_layout=Layout.row_major(4, 2),
        block_dim_x=2,
        block_dim_y=2,
        block_dim_z=2,
        block_dim_count=3,
    ](ctx)


# ----------------------------------------------------------------------
# copy_sram_to_dram tests
# ----------------------------------------------------------------------


@always_inline
def copy_sram_to_dram_kernel[
    dtype: DType,
    layout: Layout,
    M: Int,
    N: Int,
    skew_M: Int,
    num_threads: Int,
    block_dim_count: Int,
    binary_op: Optional[binary_op_type] = None,
](input: LayoutTensor[dtype, layout, MutAnyOrigin]):
    comptime simd_size = simd_width_of[dtype]()

    comptime thread_layout = Layout.row_major(
        num_threads // (M // simd_size), N // simd_size
    )

    var input_tile = input.tile[M - skew_M, N](0, 0)

    var smem_tile = LayoutTensor[
        .float32,
        Layout.row_major(M, N),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()
    arange(smem_tile)

    copy_sram_to_dram[
        thread_layout=thread_layout,
        block_dim_count=block_dim_count,
        binary_op=binary_op,
    ](
        input_tile.vectorize[1, simd_size](),
        smem_tile.vectorize[1, simd_size](),
    )


def test_copy_sram_to_dram[
    dtype: DType,
    layout: Layout,
    M: Int,
    N: Int,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    block_dim_count: Int = 1,
    skew_M: Int = 0,
    binary_op: Optional[binary_op_type] = None,
](ctx: DeviceContext) raises:
    print("=== test_copy_sram_to_dram")

    comptime managed_layout_tensor_type = ManagedLayoutTensor[
        dtype,
        layout,
    ]

    comptime element_type = managed_layout_tensor_type.element_type
    comptime idx_type = managed_layout_tensor_type.index_type

    var runtime_layout = RuntimeLayout[
        layout,
        element_type=element_type,
        linear_idx_type=idx_type,
    ].row_major(
        IndexList[
            2,
            element_type=element_type,
        ](M - skew_M, N)
    )

    var input = managed_layout_tensor_type(runtime_layout, ctx)
    _ = input.tensor().fill(-1.0)

    comptime num_threads = block_dim_x * block_dim_y * block_dim_z
    comptime kernel_type = copy_sram_to_dram_kernel[
        dtype,
        input.layout,
        M,
        N,
        skew_M,
        num_threads,
        block_dim_count,
        binary_op,
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        grid_dim=(1,),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )

    ctx.synchronize()

    print(input.tensor().tile[M - skew_M, N](0, 0))


def run_copy_sram_to_dram_tests(ctx: DeviceContext) raises:
    # CHECK: === test_copy_sram_to_dram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_sram_to_dram[
        DType.float32, Layout.row_major(8, 8), M=8, N=8, block_dim_x=8
    ](ctx)

    # CHECK: === test_copy_sram_to_dram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_sram_to_dram[
        DType.float32,
        Layout.row_major(8, 8),
        M=8,
        N=8,
        block_dim_x=2,
        block_dim_y=4,
        block_dim_z=2,
        block_dim_count=3,
    ](ctx)

    # CHECK: === test_copy_sram_to_dram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_sram_to_dram[
        DType.float32,
        Layout.row_major(8, 8),
        M=8,
        N=8,
        block_dim_x=2,
        block_dim_y=4,
        block_dim_count=2,
    ](ctx)

    # CHECK: === test_copy_sram_to_dram
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_sram_to_dram[
        DType.bfloat16,
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        N=8,
        block_dim_x=8,
    ](ctx)


# ----------------------------------------------------------------------
# copy_local_to_local tests
# ----------------------------------------------------------------------


@always_inline
def copy_local_to_local_kernel[
    dtype: DType,
    layout: Layout,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    num_threads: Int,
    block_dim_count: Int,
](output: LayoutTensor[dtype, layout, MutAnyOrigin]):
    comptime simd_size = 2

    var reg_tile0 = LayoutTensor[
        .float32,
        Layout.row_major(MMA_M, MMA_N * simd_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()
    arange(reg_tile0)

    var reg_tile1 = (
        LayoutTensor[
            .bfloat16,
            Layout.row_major(MMA_M, MMA_N * simd_size),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    copy_local_to_local(
        reg_tile1,
        reg_tile0,
    )

    copy_local_to_dram[
        dst_thread_layout=Layout.row_major(
            WM // MMA_M, WN // simd_size // MMA_N
        ),
        block_dim_count=block_dim_count,
    ](
        output.vectorize[1, simd_size](),
        reg_tile1.vectorize[1, simd_size](),
    )


def test_copy_local_to_local[
    dtype: DType,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    block_dim_count: Int = 1,
](ctx: DeviceContext) raises:
    print("=== test_copy_local_to_local")

    comptime layout = Layout.row_major(WM, WN)
    var output = ManagedLayoutTensor[
        dtype,
        layout,
    ](ctx)

    comptime num_threads = block_dim_x * block_dim_y * block_dim_z
    comptime kernel_type = copy_local_to_local_kernel[
        dtype, layout, WM, WN, MMA_M, MMA_N, num_threads, block_dim_count
    ]
    ctx.enqueue_function[kernel_type](
        output.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )

    ctx.synchronize()

    print(output.tensor())


def run_copy_local_to_local_tests(ctx: DeviceContext) raises:
    # CHECK: === test_copy_local_to_local
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_local[
        DType.bfloat16, WM=8, WN=16, MMA_M=4, MMA_N=4, block_dim_x=4
    ](ctx)

    # CHECK: === test_copy_local_to_local
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_local[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        block_dim_x=2,
        block_dim_y=2,
        block_dim_count=2,
    ](ctx)

    # CHECK: === test_copy_local_to_local
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_local[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        block_dim_x=1,
        block_dim_y=2,
        block_dim_z=2,
        block_dim_count=3,
    ](ctx)


# ----------------------------------------------------------------------
# copy_dram_to_local tests
# ----------------------------------------------------------------------


@always_inline
def copy_dram_to_local_kernel[
    layout: Layout, num_threads: Int, block_dim_count: Int
](
    input: LayoutTensor[.float32, layout, MutAnyOrigin],
    output: LayoutTensor[.float32, layout, MutAnyOrigin],
):
    comptime thread_layout = Layout.row_major(4, 2)
    comptime num_active_threads = thread_layout.size()
    comptime simd_width = 2

    var reg_tile = (
        LayoutTensor[
            .float32,
            Layout.row_major(
                layout.size() // num_active_threads // simd_width, simd_width
            ),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    copy_dram_to_local[
        src_thread_layout=thread_layout,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](reg_tile.vectorize[1, simd_width](), input.vectorize[1, simd_width]())

    barrier()

    copy_local_to_dram[
        dst_thread_layout=thread_layout,
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](
        output.vectorize[1, simd_width](),
        reg_tile.vectorize[1, simd_width](),
    )


def test_copy_dram_to_local[
    layout: Layout,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    block_dim_count: Int = 1,
](ctx: DeviceContext) raises:
    print("=== test_copy_dram_to_local")

    var input = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    arange(input.tensor())

    var output = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    comptime num_threads = block_dim_x * block_dim_y * block_dim_z
    comptime kernel_type = copy_dram_to_local_kernel[
        layout, num_threads, block_dim_count
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        output.device_tensor(),
        grid_dim=(1,),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )

    ctx.synchronize()

    print(output.tensor())


def run_copy_dram_to_local_tests(ctx: DeviceContext) raises:
    # CHECK: === test_copy_dram_to_local
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_dram_to_local[Layout.row_major(8, 8), block_dim_x=8](ctx)

    # CHECK: === test_copy_dram_to_local
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    test_copy_dram_to_local[
        Layout.row_major(8, 8),
        block_dim_x=2,
        block_dim_y=2,
        block_dim_z=2,
        block_dim_count=3,
    ](ctx)


# ----------------------------------------------------------------------
# copy_local_to_dram tests
# ----------------------------------------------------------------------


@always_inline
def copy_local_to_sram_kernel[
    dtype: DType,
    layout: Layout,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    simd_size_row: Int,
    simd_size_col: Int,
    num_threads: Int,
    block_dim_count: Int = 1,
](output: LayoutTensor[dtype, layout, MutAnyOrigin]):
    var reg_tile0 = LayoutTensor[
        .float32,
        Layout.row_major(MMA_M * simd_size_row, MMA_N * simd_size_col),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()
    arange(reg_tile0)

    var smem_warp_tile = (
        LayoutTensor[
            dtype,
            Layout.row_major(WM, WN),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .fill(0)
    )

    copy_local_to_shared[
        thread_layout=Layout.row_major(
            WM // simd_size_row // MMA_M, WN // simd_size_col // MMA_N
        ),
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](
        smem_warp_tile.vectorize[simd_size_row, simd_size_col](),
        reg_tile0.vectorize[simd_size_row, simd_size_col](),
    )

    copy_sram_to_dram[
        thread_layout=Layout.row_major(
            WM // simd_size_row // MMA_M, WN // simd_size_col // MMA_N
        ),
        num_threads=num_threads,
        block_dim_count=block_dim_count,
    ](
        output.vectorize[simd_size_row, simd_size_col](),
        smem_warp_tile.vectorize[simd_size_row, simd_size_col](),
    )


def test_copy_local_to_sram[
    dtype: DType,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    simd_size_row: Int,
    simd_size_col: Int,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    block_dim_count: Int = 1,
](ctx: DeviceContext) raises:
    print(
        "=== test_copy_local_to_sram_",
        dtype,
        "_simd_size_",
        simd_size_row,
        simd_size_col,
        sep="",
    )

    comptime layout = Layout.row_major(WM, WN)
    var output = ManagedLayoutTensor[
        dtype,
        layout,
    ](ctx)

    comptime num_threads = block_dim_x * block_dim_y * block_dim_z
    comptime kernel_type = copy_local_to_sram_kernel[
        dtype,
        layout,
        WM,
        WN,
        MMA_M,
        MMA_N,
        simd_size_row,
        simd_size_col,
        num_threads,
        block_dim_count,
    ]
    ctx.enqueue_function[kernel_type](
        output.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )

    ctx.synchronize()

    print(output.tensor())


def run_copy_local_to_sram_tests_float32_simd_size_12(
    ctx: DeviceContext,
) raises:
    # CHECK: === test_copy_local_to_sram_float32_simd_size_12
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_sram[
        DType.float32,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=1,
        simd_size_col=2,
        block_dim_x=4,
    ](ctx)

    # CHECK: === test_copy_local_to_sram_float32_simd_size_12
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_sram[
        DType.float32,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=1,
        simd_size_col=2,
        block_dim_x=2,
        block_dim_y=1,
        block_dim_z=2,
        block_dim_count=3,
    ](ctx)


def run_copy_local_to_sram_tests_float32_simd_size_21(
    ctx: DeviceContext,
) raises:
    # CHECK: === test_copy_local_to_sram_float32_simd_size_21
    # CHECK: 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 3.0 3.0 3.0 3.0
    # CHECK: 4.0 4.0 4.0 4.0 5.0 5.0 5.0 5.0 6.0 6.0 6.0 6.0 7.0 7.0 7.0 7.0
    # CHECK: 8.0 8.0 8.0 8.0 9.0 9.0 9.0 9.0 10.0 10.0 10.0 10.0 11.0 11.0 11.0 11.0
    # CHECK: 12.0 12.0 12.0 12.0 13.0 13.0 13.0 13.0 14.0 14.0 14.0 14.0 15.0 15.0 15.0 15.0
    # CHECK: 16.0 16.0 16.0 16.0 17.0 17.0 17.0 17.0 18.0 18.0 18.0 18.0 19.0 19.0 19.0 19.0
    # CHECK: 20.0 20.0 20.0 20.0 21.0 21.0 21.0 21.0 22.0 22.0 22.0 22.0 23.0 23.0 23.0 23.0
    # CHECK: 24.0 24.0 24.0 24.0 25.0 25.0 25.0 25.0 26.0 26.0 26.0 26.0 27.0 27.0 27.0 27.0
    # CHECK: 28.0 28.0 28.0 28.0 29.0 29.0 29.0 29.0 30.0 30.0 30.0 30.0 31.0 31.0 31.0 31.0
    test_copy_local_to_sram[
        DType.float32,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=2,
        simd_size_col=1,
        block_dim_x=4,
    ](ctx)


def run_copy_local_to_sram_tests_bfloat16_simd_size_12(
    ctx: DeviceContext,
) raises:
    # CHECK: === test_copy_local_to_sram_bfloat16_simd_size_12
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 0.0 1.0 0.0 1.0 2.0 3.0 2.0 3.0 4.0 5.0 4.0 5.0 6.0 7.0 6.0 7.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 8.0 9.0 8.0 9.0 10.0 11.0 10.0 11.0 12.0 13.0 12.0 13.0 14.0 15.0 14.0 15.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 16.0 17.0 16.0 17.0 18.0 19.0 18.0 19.0 20.0 21.0 20.0 21.0 22.0 23.0 22.0 23.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    # CHECK: 24.0 25.0 24.0 25.0 26.0 27.0 26.0 27.0 28.0 29.0 28.0 29.0 30.0 31.0 30.0 31.0
    test_copy_local_to_sram[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=1,
        simd_size_col=2,
        block_dim_x=4,
    ](ctx)


def run_copy_local_to_sram_tests_bfloat16_simd_size_21(
    ctx: DeviceContext,
) raises:
    # CHECK: === test_copy_local_to_sram_bfloat16_simd_size_21
    # CHECK: 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 2.0 2.0 2.0 2.0 3.0 3.0 3.0 3.0
    # CHECK: 4.0 4.0 4.0 4.0 5.0 5.0 5.0 5.0 6.0 6.0 6.0 6.0 7.0 7.0 7.0 7.0
    # CHECK: 8.0 8.0 8.0 8.0 9.0 9.0 9.0 9.0 10.0 10.0 10.0 10.0 11.0 11.0 11.0 11.0
    # CHECK: 12.0 12.0 12.0 12.0 13.0 13.0 13.0 13.0 14.0 14.0 14.0 14.0 15.0 15.0 15.0 15.0
    # CHECK: 16.0 16.0 16.0 16.0 17.0 17.0 17.0 17.0 18.0 18.0 18.0 18.0 19.0 19.0 19.0 19.0
    # CHECK: 20.0 20.0 20.0 20.0 21.0 21.0 21.0 21.0 22.0 22.0 22.0 22.0 23.0 23.0 23.0 23.0
    # CHECK: 24.0 24.0 24.0 24.0 25.0 25.0 25.0 25.0 26.0 26.0 26.0 26.0 27.0 27.0 27.0 27.0
    # CHECK: 28.0 28.0 28.0 28.0 29.0 29.0 29.0 29.0 30.0 30.0 30.0 30.0 31.0 31.0 31.0 31.0
    test_copy_local_to_sram[
        DType.bfloat16,
        WM=8,
        WN=16,
        MMA_M=4,
        MMA_N=4,
        simd_size_row=2,
        simd_size_col=1,
        block_dim_x=4,
    ](ctx)


def main() raises:
    with DeviceContext() as ctx:
        run_async_copy_tests(ctx)
        run_swizzle_copy_tests(ctx)
        run_partial_copy_dram_to_sram_async_tests(ctx)
        run_copy_dram_to_sram_tests(ctx)
        run_copy_sram_to_dram_tests(ctx)
        run_copy_local_to_local_tests(ctx)
        run_copy_dram_to_local_tests(ctx)
        run_copy_local_to_sram_tests_float32_simd_size_12(ctx)
        run_copy_local_to_sram_tests_float32_simd_size_21(ctx)
        run_copy_local_to_sram_tests_bfloat16_simd_size_12(ctx)
        run_copy_local_to_sram_tests_bfloat16_simd_size_21(ctx)
