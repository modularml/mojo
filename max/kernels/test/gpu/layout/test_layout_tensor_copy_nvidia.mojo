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
    copy_dram_to_local,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_dram,
)

from std.utils import IndexList

# ----------------------------------------------------------------------
# dynamic async copy tests
# ----------------------------------------------------------------------


def async_dynamic_copy_kernel[
    input_layout: Layout,
    output_layout: Layout,
    BM: Int,
    BN: Int,
    num_rows: Int,
](
    input: LayoutTensor[.float32, input_layout, MutAnyOrigin],
    output: LayoutTensor[.float32, output_layout, MutAnyOrigin],
):
    var masked_input = LayoutTensor[
        .float32,
        input_layout,
        MutAnyOrigin,
        masked=True,
    ](
        input.ptr,
        type_of(input.runtime_layout)(
            type_of(input.runtime_layout.shape)(num_rows, input.dim[1]()),
            input.runtime_layout.stride,
        ),
    )

    var input_tile = masked_input.tile[BM, BN](block_idx.x, block_idx.y)
    var output_tile = output.tile[BM, BN](block_idx.x, block_idx.y)

    var smem_tile = LayoutTensor[
        .float32,
        Layout(IntTuple(BM, BN)),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async[is_masked=True](input_tile)
    async_copy_wait_all()

    output_tile.copy_from(smem_tile)


def test_dynamic_async_copy[
    M: Int, N: Int, BM: Int, BN: Int, num_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_dynamic_async_copy")

    comptime unknown_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    comptime input_runtime_layout = RuntimeLayout[
        unknown_layout,
        element_type=.int64,
        linear_idx_type=.int64,
    ].row_major(IndexList[2, element_type=.int64](M, N))

    comptime output_runtime_layout = RuntimeLayout[
        unknown_layout,
        element_type=.int64,
        linear_idx_type=.int64,
    ].row_major(IndexList[2, element_type=.int64](num_rows, N))

    var input = ManagedLayoutTensor[
        .float32,
        unknown_layout,
    ](input_runtime_layout, ctx)
    arange(input.tensor())

    var output = ManagedLayoutTensor[
        .float32,
        unknown_layout,
    ](output_runtime_layout, ctx)

    comptime kernel_type = async_dynamic_copy_kernel[
        unknown_layout,
        unknown_layout,
        BM,
        BN,
        num_rows,
    ]

    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        output.device_tensor(),
        grid_dim=(ceildiv(M, BM), ceildiv(M, BN)),
        block_dim=(1, 1),
    )

    ctx.synchronize()

    print(output.tensor())


def run_dynamic_async_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_dynamic_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0 16.0 17.0
    # CHECK: 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0
    test_dynamic_async_copy[
        M=6,
        N=6,
        BM=2,
        BN=3,
        num_rows=5,
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
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_swizzle_copy[
        Layout.row_major(UNKNOWN_VALUE, 16),
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
        skew_M=2,
    ](ctx)


# ----------------------------------------------------------------------
# masked async copy tests
# ----------------------------------------------------------------------


@always_inline
def masked_async_copy_kernel[
    layout: Layout, num_rows: Int
](input: LayoutTensor[.float32, layout, MutAnyOrigin]):
    comptime thread_layout = Layout.row_major(4, 2)

    var masked_input = LayoutTensor[
        .float32,
        layout,
        MutAnyOrigin,
        masked=True,
    ](
        input.ptr,
        type_of(input.runtime_layout)(
            type_of(input.runtime_layout.shape)(num_rows, input.dim[1]()),
            input.runtime_layout.stride,
        ),
    )

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

    copy_dram_to_sram_async[thread_layout=thread_layout](
        smem_tile.vectorize[1, 4](), masked_input.vectorize[1, 4]()
    )

    async_copy_commit_group()
    async_copy_wait_all()

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


def test_masked_async_copy[
    layout: Layout, M: Int, N: Int, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_async_copy")

    comptime managed_layout_tensor_type = ManagedLayoutTensor[
        .float32,
        layout,
    ]

    comptime element_type = managed_layout_tensor_type.element_type
    comptime idx_type = managed_layout_tensor_type.index_type

    comptime runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, N))

    var input = ManagedLayoutTensor[
        .float32,
        layout,
    ](runtime_layout, ctx)

    arange(input.tensor())

    var input_tensor = LayoutTensor[
        .float32,
        Layout.row_major(M, N),
        MutAnyOrigin,
    ](input.device_tensor().ptr)

    comptime kernel_type = masked_async_copy_kernel[
        input_tensor.layout, M - skew_rows
    ]

    ctx.enqueue_function[kernel_type](
        input_tensor,
        grid_dim=(1,),
        block_dim=(8,),
    )

    ctx.synchronize()

    print(input.tensor())


def run_masked_async_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_masked_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_async_copy[
        Layout.row_major(8, 8),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)

    # CHECK: === test_masked_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_async_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)


# ----------------------------------------------------------------------
# masked copy tests
# ----------------------------------------------------------------------


@always_inline
def masked_copy_kernel[
    layout: Layout, num_rows: Int
](input: LayoutTensor[.float32, layout, MutAnyOrigin]):
    comptime thread_layout = Layout.row_major(4, 2)

    var masked_input = LayoutTensor[
        .float32,
        layout,
        MutAnyOrigin,
        masked=True,
    ](
        input.ptr,
        type_of(input.runtime_layout)(
            type_of(input.runtime_layout.shape)(num_rows, input.dim[1]()),
            input.runtime_layout.stride,
        ),
    )

    var smem_tile = (
        LayoutTensor[
            .float32,
            layout,
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .fill(0)
    )

    copy_dram_to_sram[thread_layout=thread_layout](
        smem_tile.vectorize[1, 4](), masked_input.vectorize[1, 4]()
    )

    barrier()

    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


def test_masked_copy[
    layout: Layout, M: Int, N: Int, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_copy")

    comptime managed_layout_tensor_type = ManagedLayoutTensor[
        .float32,
        layout,
    ]

    comptime element_type = managed_layout_tensor_type.element_type
    comptime idx_type = managed_layout_tensor_type.index_type

    comptime runtime_layout = RuntimeLayout[
        layout, element_type=element_type, linear_idx_type=idx_type
    ].row_major(IndexList[2, element_type=element_type](M, N))

    var input = ManagedLayoutTensor[
        .float32,
        layout,
    ](runtime_layout, ctx)

    arange(input.tensor())

    var input_tensor = LayoutTensor[
        .float32,
        Layout.row_major(M, N),
        MutAnyOrigin,
    ](input.device_tensor().ptr)

    comptime kernel_type = masked_copy_kernel[
        input_tensor.layout, M - skew_rows
    ]
    ctx.enqueue_function[kernel_type](
        input_tensor, grid_dim=(1,), block_dim=(8,)
    )

    ctx.synchronize()

    print(input.tensor())


def run_masked_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_masked_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_copy[
        Layout.row_major(8, 8),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)

    # CHECK: === test_masked_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_copy[
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)


@always_inline
def masked_copy_dram_to_local_kernel[
    layout: Layout, num_rows: Int
](
    input: LayoutTensor[.float32, layout, MutAnyOrigin],
    output: LayoutTensor[.float32, layout, MutAnyOrigin],
):
    comptime thread_layout = Layout.row_major(4, 2)
    comptime num_threads = thread_layout.size()
    comptime simd_width = 2

    var masked_input = LayoutTensor[
        .float32,
        layout,
        MutAnyOrigin,
        masked=True,
    ](
        input.ptr,
        type_of(input.runtime_layout)(
            type_of(input.runtime_layout.shape)(num_rows, input.dim[1]()),
            input.runtime_layout.stride,
        ),
    )

    var reg_tile = (
        LayoutTensor[
            .float32,
            Layout.row_major(
                layout.size() // num_threads // simd_width, simd_width
            ),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(-1.0)
    )

    copy_dram_to_local[src_thread_layout=thread_layout](
        reg_tile.vectorize[1, simd_width](),
        masked_input.vectorize[1, simd_width](),
    )

    barrier()

    copy_local_to_dram[dst_thread_layout=thread_layout](
        output.vectorize[1, simd_width](),
        reg_tile.vectorize[1, simd_width](),
    )


def test_masked_copy_dram_to_local[
    layout: Layout, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_copy_dram_to_local")

    comptime M = layout.shape[0].value()

    var input = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    arange(input.tensor())

    var output = ManagedLayoutTensor[
        .float32,
        layout,
    ](ctx)

    comptime kernel_type = masked_copy_dram_to_local_kernel[
        layout, M - skew_rows
    ]
    ctx.enqueue_function[kernel_type](
        input.device_tensor(),
        output.device_tensor(),
        grid_dim=(1,),
        block_dim=(8,),
    )

    ctx.synchronize()

    print(output.tensor())


def run_copy_dram_to_local_tests(ctx: DeviceContext) raises:
    # CHECK: === test_masked_copy_dram_to_local
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    test_masked_copy_dram_to_local[Layout.row_major(8, 8), skew_rows=1](ctx)


def main() raises:
    with DeviceContext() as ctx:
        run_dynamic_async_copy_tests(ctx)
        run_swizzle_copy_tests(ctx)
        run_masked_async_copy_tests(ctx)
        run_masked_copy_tests(ctx)
        run_copy_dram_to_local_tests(ctx)
