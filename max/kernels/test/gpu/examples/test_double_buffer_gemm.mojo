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

from std.math import ceildiv, isclose
from std.math.uutils import udivmod
from std.sys import argv, simd_width_of
from std.sys.info import has_nvidia_gpu_accelerator, is_nvidia_gpu

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import async_copy_wait_all
from layout import Coord, Idx, IntTuple, LayoutTensor, TileTensor, row_major
from layout.layout import *
from layout.layout_tensor import (
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_local,
)
from layout.math import outer_product_acc
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_almost_equal


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


def sgemm_double_buffer[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    itype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
):
    comptime _uint = Scalar[itype]

    comptime simd_size = simd_width_of[c_type]()

    var M = c.shape[0]()
    var N = c.shape[1]()
    var K = a.shape[1]()

    comptime num_warps_m = (BM // WM)
    comptime num_warps_n = (BN // WN)

    var tid = thread_idx.x
    var warp_id, lane_id = udivmod(tid, WARP_SIZE)

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # Warp shape in 2D.
    comptime warp_dim_x = WN // TN
    comptime warp_dim_y = WM // TM
    comptime assert (
        warp_dim_x * warp_dim_y == WARP_SIZE
    ), "Warp 2d shape doesn't match 32 threads"

    # Pad BM to avoid back conflict
    comptime pad_avoid_bank_conflict = 4
    comptime BM_padded = BM + pad_avoid_bank_conflict

    # Double buffer in shared memory.
    comptime a_smem_size = BK * BM_padded
    var a_smem_tile = (
        LayoutTensor[
            a_type,
            Layout.row_major(2 * BK, BM_padded),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .slice[:, :BM]()
        .split[2]()
    )

    # Align the address by the maximum async copy size (16 bytes).
    comptime b_smem_size = BK * BN
    var b_smem_tile = (
        LayoutTensor[
            b_type,
            Layout.row_major(2 * BK, BN),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        .stack_allocation()
        .split[2]()
    )

    # Global memory tile.
    var a_gmem_tile = a.tile[BM, BK](block_idx.y, 0)
    var b_gmem_tile = b.tile[BK, BN](0, block_idx.x)

    # Load A tile from global memory to shared.
    # Row major thread layout for coalesced access.
    comptime thread_loada_gmem_layout = Layout.row_major(NUM_THREADS // BK, BK)
    comptime thread_storea_smem_layout = Layout.col_major(BK, NUM_THREADS // BK)
    copy_dram_to_sram_async[
        src_thread_layout=thread_loada_gmem_layout,
        dst_thread_layout=thread_storea_smem_layout,
    ](a_smem_tile[0], a_gmem_tile)

    # Load B tile from global memory to shared.
    # Row major thread layout for coalesced access.
    comptime thread_layout_loadb = Layout.row_major(
        (NUM_THREADS // BN) * simd_size, BN // simd_size
    )
    copy_dram_to_sram_async[
        src_thread_layout=thread_layout_loadb,
        dst_thread_layout=thread_layout_loadb,
    ](
        b_smem_tile[0].vectorize[1, simd_size](),
        b_gmem_tile.vectorize[1, simd_size](),
    )

    async_copy_wait_all()
    barrier()

    # Advance A and B to next k tile.
    a_gmem_tile = a.tile[BM, BK](block_idx.y, 1)
    b_gmem_tile = b.tile[BK, BN](1, block_idx.x)

    # Double buffer in registers (fragments in nvidia terms).
    comptime layout_a = Layout.row_major(TM)
    var a_reg: Array[
        LayoutTensor[
            a_type,
            layout_a,
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
        2,
    ] = [
        LayoutTensor[
            a_type,
            layout_a,
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation(),
        LayoutTensor[
            a_type,
            layout_a,
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation(),
    ]
    comptime layout_b = Layout.row_major(TN)
    var b_reg: Array[
        LayoutTensor[
            b_type,
            layout_b,
            MutAnyOrigin,
            address_space=.LOCAL,
        ],
        2,
    ] = [
        LayoutTensor[
            b_type,
            layout_b,
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation(),
        LayoutTensor[
            b_type,
            layout_b,
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation(),
    ]
    comptime layout_c = Layout.row_major(TM, TN)
    var c_reg = (
        LayoutTensor[
            c_type,
            layout_c,
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    # Thread swizzling
    # Warp has 2D Layout [warp_dim_x, warp_dim_y]. Current thread is mapped to
    # (mma_x, mma_y) in this layout as follow (the number is thread id).
    # 0  2  4  6  8  10 12 14
    # 1  3  5  7  9  11 13 15
    # 16 18 20 22 24 26 28 30
    # 17 19 21 23 25 27 29 31
    comptime thread_layout = Layout(
        IntTuple(IntTuple(2, 2), 8), IntTuple(IntTuple(1, 16), 2)
    ) if is_nvidia_gpu() else Layout(
        IntTuple(IntTuple(2, 2), 16), IntTuple(IntTuple(1, 32), 2)
    )

    # Load A fragments to the first buffer.
    var a_smem_warp_tile = a_smem_tile[0].tile[BK, WM](0, warp_y)
    var a_smem_warp_row = a_smem_warp_tile.tile[1, WM](0, 0).coalesce()
    copy_sram_to_local[src_warp_layout=thread_layout, axis=0](
        a_reg[0].vectorize[simd_size](), a_smem_warp_row.vectorize[simd_size]()
    )

    # Load B fragments to the first buffer.
    var b_smem_warp_tile = b_smem_tile[0].tile[BK, WN](0, warp_x)
    var b_smem_warp_row = b_smem_warp_tile.tile[1, WN](0, 0).coalesce()
    copy_sram_to_local[src_warp_layout=thread_layout, axis=1](
        b_reg[0].vectorize[simd_size](), b_smem_warp_row.vectorize[simd_size]()
    )

    var num_k_tiles = ceildiv(K, BK)

    # Update (num_k_tile - 1) tiles while switching buffers.
    # for k_tile_id in range(num_k_tiles - 1):
    for k_tile_id in range(num_k_tiles):
        # The shared memory buffer to be prefetched
        var prefetch_id = 1 if k_tile_id % 2 == 0 else 0

        comptime for k in range(BK):
            var next_k = (k + 1) % BK

            # Buffer id for the double register buffers. They alternate.
            var buffer_id = k % 2
            var next_buffer_id = (k + 1) % 2

            if k == BK - 1:
                async_copy_wait_all()
                barrier()

                a_smem_warp_tile = a_smem_tile[prefetch_id].tile[BK, WM](
                    0, warp_y
                )
                b_smem_warp_tile = b_smem_tile[prefetch_id].tile[BK, WN](
                    0, warp_x
                )

            # Fill the other A fragments buffer using the next row in A.
            var a_smem_warp_row = a_smem_warp_tile.tile[1, WM](
                next_k, 0
            ).coalesce()
            copy_sram_to_local[src_warp_layout=thread_layout, axis=0](
                a_reg[next_buffer_id].vectorize[simd_size](),
                a_smem_warp_row.vectorize[simd_size](),
            )

            var b_smem_warp_row = b_smem_warp_tile.tile[1, WN](
                next_k, 0
            ).coalesce()
            copy_sram_to_local[src_warp_layout=thread_layout, axis=1](
                b_reg[next_buffer_id].vectorize[simd_size](),
                b_smem_warp_row.vectorize[simd_size](),
            )

            # Load next k tile from global memory to shared memory.
            if k == 0 and k_tile_id < num_k_tiles - 1:
                a_gmem_tile = a.tile[BM, BK](block_idx.y, k_tile_id + 1)
                copy_dram_to_sram_async[
                    src_thread_layout=thread_loada_gmem_layout,
                    dst_thread_layout=thread_storea_smem_layout,
                ](a_smem_tile[prefetch_id], a_gmem_tile)

                b_gmem_tile = b.tile[BK, BN](k_tile_id + 1, block_idx.x)
                copy_dram_to_sram_async[
                    src_thread_layout=thread_layout_loadb,
                    dst_thread_layout=thread_layout_loadb,
                ](
                    b_smem_tile[prefetch_id].vectorize[1, simd_size](),
                    b_gmem_tile.vectorize[1, simd_size](),
                )

            outer_product_acc(c_reg, a_reg[buffer_id], b_reg[buffer_id])

    # Map global memory tile down to thread.
    var c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    var c_gmem_warp_tile = c_gmem_tile.tile[WM, WN](warp_y, warp_x)
    # Copy results to global memory.
    # Vectorize by [simd_size, simd_size] because the outer product results are
    # implicitly organized by simd_size x simd_size tiles.
    copy_local_to_dram[dst_thread_layout=thread_layout](
        c_gmem_warp_tile.vectorize[simd_size, simd_size](),
        c_reg.vectorize[simd_size, simd_size](),
    )


def test(ctx: DeviceContext) raises:
    comptime NUM_THREADS = 256
    comptime M = 8192
    comptime N = 8192
    comptime K = 128
    comptime BM = 128
    comptime BN = 128
    comptime BK = 16
    comptime WM = 32
    comptime WN = 64 if has_nvidia_gpu_accelerator() else 128
    comptime TM = 8
    comptime TN = 8

    comptime a_layout = Layout(IntTuple(M, K), IntTuple(K, 1))
    comptime b_layout = Layout(IntTuple(K, N), IntTuple(N, 1))
    comptime c_layout = Layout(IntTuple(M, N), IntTuple(N, 1))

    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](K * N)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](M * N)
    ctx.synchronize()

    for i in range(M * K):
        a_host[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = Float32(i)

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var c_tensor = LayoutTensor[.float32, c_layout](c_device)
    var a_tensor = LayoutTensor[.float32, a_layout](a_device)
    var b_tensor = LayoutTensor[.float32, b_layout](b_device)

    comptime gemm = sgemm_double_buffer[
        DType.float32,
        c_layout,
        DType.float32,
        a_layout,
        DType.float32,
        b_layout,
        DType.uint32,
        BM,
        BN,
        BK,
        WM,
        WN,
        TM,
        TN,
        NUM_THREADS,
    ]
    if is_benchmark():
        comptime nrun = 200
        comptime nwarmup = 2

        @always_inline
        def run_func(ctx: DeviceContext) raises {imm}:
            ctx.enqueue_function[gemm](
                c_tensor,
                a_tensor,
                b_tensor,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
                block_dim=(NUM_THREADS, 1, 1),
            )

        # Warmup
        for _ in range(nwarmup):
            run_func(ctx)

        var nstime = Float64(ctx.execution_time(run_func, nrun)) / Float64(nrun)
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * M * N * K * 1e-12
        print(nrun, "runs avg(s)", sectime, "TFlops/s", TFlop / sectime)

    ctx.enqueue_function[gemm](
        c_tensor,
        a_tensor,
        b_tensor,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM), 1),
        block_dim=(NUM_THREADS, 1, 1),
    )

    ctx.enqueue_copy(c_host, c_device)

    # Naive gemm.
    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function
    # requires exact type matches).
    var c_ref_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    comptime gemm_naive = matmul_kernel_naive[
        DType.float32,
        DType.float32,
        DType.float32,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
    ]
    ctx.enqueue_function[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        if not isclose(c_host[i], c_host_ref[i]):
            print(i, c_host[i], c_host_ref[i])
        assert_almost_equal(c_host[i], c_host_ref[i])

    _ = c_device
    _ = c_device_ref
    _ = a_device
    _ = b_device


def main() raises:
    with DeviceContext() as ctx:
        test(ctx)
