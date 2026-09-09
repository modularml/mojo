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
from std.sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TMADescriptor, create_tma_descriptor
from max.gpu.primitives import warp
from max.gpu.memory import (
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
)
from internal_utils import assert_almost_equal
from layout import (
    CoordLike,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout.layout_tensor import LayoutTensorIter
from layout.tma_async import PipelineState, SharedMemBarrier

from std.utils.index import Index
from std.utils.numerics import get_accum_type


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@__llvm_arg_metadata(descriptor_a, `nvvm.grid_constant`)
@__llvm_arg_metadata(descriptor_b, `nvvm.grid_constant`)
def gemv_tma_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BLOCK_SIZE_M: Int,
    BLOCK_SIZE_K: Int,
    ROWS_PER_WARP: Int,
    NUM_PIPELINE_STAGES: Int,
](
    descriptor_a: TMADescriptor,
    descriptor_b: TMADescriptor,
    c: LayoutTensor[dtype, c_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    M_dev: Int32,
    N_dev: Int32,
    K_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var M = Int(M_dev)
    var N = Int(N_dev)
    var K = Int(K_dev)
    var bidx = block_idx.x
    var block_row = bidx * BLOCK_SIZE_M

    var warp_row_offset = warp_id() * ROWS_PER_WARP
    var global_row_idx = block_row + warp_row_offset

    comptime accum_type = get_accum_type[dtype]()

    comptime a_smem_layout = Layout.row_major(BLOCK_SIZE_M, BLOCK_SIZE_K)

    comptime b_smem_layout = Layout.row_major(BLOCK_SIZE_K)

    var descriptor_a_ptr = Pointer(to=descriptor_a).bitcast[NoneType]()
    var descriptor_b_ptr = Pointer(to=descriptor_b).bitcast[NoneType]()

    var a_smem_base = rebind[
        MutPointer[Scalar[dtype], address_space=.SHARED, MutUntrackedOrigin]
    ](
        external_memory[
            Scalar[dtype],
            address_space=.SHARED,
            alignment=128,
            name="tmem_A_dynamic_shared_memory",
        ]()
    )

    comptime a_size = a_smem_layout.size()

    var b_smem_base = (a_smem_base + NUM_PIPELINE_STAGES * a_size).bitcast[
        Scalar[dtype]
    ]()

    comptime b_size = b_smem_layout.size()

    var a_smem = LayoutTensorIter[
        dtype,
        a_smem_layout,
        address_space=.SHARED,
        alignment=128,
        circular=False,
    ](
        a_smem_base.as_unsafe_any_origin(),
        a_size * NUM_PIPELINE_STAGES,
    )

    var b_smem = LayoutTensorIter[
        dtype,
        b_smem_layout,
        address_space=.SHARED,
        alignment=128,
        circular=False,
    ](
        b_smem_base.as_unsafe_any_origin(),
        b_size * NUM_PIPELINE_STAGES,
    )

    var tma_mbar = (b_smem_base + b_size * NUM_PIPELINE_STAGES).bitcast[
        SharedMemBarrier
    ]()

    # Initialize dot products for all rows before column processing.
    var dot_products = Array[Scalar[accum_type], ROWS_PER_WARP](fill=0)

    if thread_idx.x == 0:
        comptime for i in range(NUM_PIPELINE_STAGES):
            tma_mbar[i].init()

    barrier()

    # Double buffering.
    var consumer_phase = PipelineState[NUM_PIPELINE_STAGES]()
    var producer_phase = PipelineState[NUM_PIPELINE_STAGES](0, 1, 0)

    for col_offset in range(0, K, BLOCK_SIZE_K):
        var current_block_size = min(BLOCK_SIZE_K, K - col_offset)

        # Producer: Thread 0 loads data.
        if thread_idx.x == 0:
            var stage = producer_phase.index()
            tma_mbar[stage].expect_bytes(
                Int32(
                    BLOCK_SIZE_M * current_block_size * size_of[dtype]()
                    + current_block_size * size_of[dtype]()
                )
            )

            cp_async_bulk_tensor_shared_cluster_global[
                Scalar[dtype],
                SharedMemBarrier,
                2,
            ](
                a_smem.next(stage)[].ptr,
                descriptor_a_ptr,
                Pointer(to=tma_mbar[stage]),
                Index(col_offset, block_row),
            )
            cp_async_bulk_tensor_shared_cluster_global[
                Scalar[dtype],
                SharedMemBarrier,
                1,
            ](
                b_smem.next(stage)[].ptr,
                descriptor_b_ptr,
                Pointer(to=tma_mbar[stage]),
                Index(col_offset),
            )
            producer_phase.step()

        # Consumer: All threads wait and process.
        var stage = consumer_phase.index()
        var phase = consumer_phase.phase()

        tma_mbar[stage].wait(phase)

        # Process current buffer.
        var current_a_tile = a_smem.next_unsafe(
            a_smem.linear_uint_type(Int(stage))
        )[]
        var current_b_tile = b_smem.next_unsafe(
            b_smem.linear_uint_type(Int(stage))
        )[]

        for k_idx in range(0, current_block_size, WARP_SIZE):
            var col_idx = k_idx + lane_id()
            if col_idx < current_block_size:
                var b_val = current_b_tile[col_idx]

                comptime for i in range(ROWS_PER_WARP):
                    var row_idx = warp_row_offset + i
                    if global_row_idx + i < M:
                        var a_val = current_a_tile[row_idx, col_idx]
                        dot_products[i] += rebind[type_of(dot_products[i])](
                            a_val.cast[accum_type]() * b_val.cast[accum_type]()
                        )

        consumer_phase.step()

    comptime for i in range(ROWS_PER_WARP):
        var global_row = global_row_idx + i
        if global_row < M:
            var final_dot_product = warp.sum(dot_products[i])
            if lane_id() == 0:
                c[global_row, 0] = Scalar[dtype](final_dot_product)


def gemv_tma[
    dtype: DType,
](
    c_device: DeviceBuffer[dtype],
    c_tt: TileTensor[mut=True, dtype, ...],
    a_device: DeviceBuffer[dtype],
    a_tt: TileTensor[dtype, ...],
    b_device: DeviceBuffer[dtype],
    b_tt: TileTensor[dtype, ...],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    # TODO: Tune further.
    comptime THREAD_NUM = 1024
    comptime BLOCK_SIZE_M = 64
    comptime BLOCK_SIZE_K = 256
    # Number of warps per block for 128 threads.
    comptime WARPS_PER_BLOCK = THREAD_NUM // WARP_SIZE
    comptime ROWS_PER_WARP = BLOCK_SIZE_M // WARPS_PER_BLOCK
    comptime NUM_PIPELINE_STAGES = 1

    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    var c = c_tt.to_layout_tensor()

    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 1

    var tma_desc_a = create_tma_descriptor[dtype, 2](
        a_device,
        (M, K),
        (K, 1),
        Index(BLOCK_SIZE_M, BLOCK_SIZE_K),
    )
    var tma_desc_b = create_tma_descriptor[dtype, 1](
        b_device,
        Index(K),
        Index(1),
        Index(BLOCK_SIZE_K),
    )
    # Shared memory needed for NUM_PIPELINE_STAGES A and B working tiles.
    # +8 bytes for each of NUM_PIPELINE_STAGES barriers.
    comptime smem_use = (
        NUM_PIPELINE_STAGES * BLOCK_SIZE_M * BLOCK_SIZE_K * size_of[dtype]()
        + NUM_PIPELINE_STAGES * BLOCK_SIZE_K * size_of[dtype]()
        + 8 * NUM_PIPELINE_STAGES
    )

    comptime kernel = gemv_tma_kernel[
        dtype,
        a.layout,
        b.layout,
        c.layout,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        ROWS_PER_WARP,
        NUM_PIPELINE_STAGES,
    ]

    ctx.enqueue_function[kernel](
        tma_desc_a,
        tma_desc_b,
        c,
        a,
        b,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_SIZE_M)),
        block_dim=(THREAD_NUM),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


def test_gemv_tma[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    benchmark: Bool = False,
) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    var a_shape = Coord(m, k)
    var b_shape = Coord(k)
    var c_shape = Coord(m, n)

    var a_size = M * K
    var b_size = K
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    rand[dtype](a_host_ptr.unsafe_ptr(), M * K)
    rand[dtype](b_host_ptr.unsafe_ptr(), K * N)

    var a_device = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[dtype](c_size)

    var a_tt = TileTensor(a_device, row_major(a_shape)).as_unsafe_any_origin()
    var b_tt = TileTensor(b_device, row_major(b_shape)).as_unsafe_any_origin()
    var c_tt = TileTensor(c_device, row_major(c_shape)).as_unsafe_any_origin()

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    gemv_tma(
        c_device,
        c_tt,
        a_device,
        a_tt,
        b_device,
        b_tt,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    if benchmark:
        comptime num_runs = 50
        comptime num_warmup = 10

        @always_inline
        def run_func(ctx: DeviceContext) raises {imm}:
            gemv_tma(
                c_device,
                c_tt,
                a_device,
                a_tt,
                b_device,
                b_tt,
                M,
                N,
                K,
                ctx,
            )

        for _ in range(num_warmup):
            run_func(ctx)
        ctx.synchronize()

        var nstime = Float64(ctx.execution_time(run_func, num_runs)) / Float64(
            num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12
        # Round TFLOPS to two decimal places for cleaner output.
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 3)
        print(
            t"{M}x{N}x{K}: DTYPE={dtype}",
            sectime * 1000,
            tflops_rounded,
        )
    else:
        # Compare with vendor BLAS for correctness.
        var b_2d_shape = Coord(K, Idx[1])
        var b_2d = TileTensor(b_device, row_major(b_2d_shape))
        var c_ref_tt = TileTensor(c_device_ref, row_major(c_shape))
        vendor_blas.matmul(
            ctx,
            c_ref_tt,
            a_tt,
            b_2d,
            c_row_major=True,
            transpose_b=False,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host_ptr, c_device)
        ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
        ctx.synchronize()

        comptime rtol = 1e-2
        assert_almost_equal(
            c_host_ptr.unsafe_ptr(),
            c_host_ref_ptr.unsafe_ptr(),
            c_size,
            atol=0.0001,
            rtol=rtol,
        )


def main() raises:
    with DeviceContext() as ctx:
        var benchmark = is_benchmark()
        test_gemv_tma[.bfloat16](
            ctx, Idx[256], Idx[1], Idx[256], benchmark=benchmark
        )
        test_gemv_tma[.bfloat16](
            ctx, Idx[4096], Idx[1], Idx[4096], benchmark=benchmark
        )

        test_gemv_tma[.float32](
            ctx, Idx[256], Idx[1], Idx[256], benchmark=benchmark
        )
        test_gemv_tma[.float32](
            ctx, Idx[4096], Idx[1], Idx[4096], benchmark=benchmark
        )
