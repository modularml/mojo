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

from std.collections import Optional
from std.math import ceildiv
from std.math.uutils import udivmod, umod
from std.sys import has_amd_gpu_accelerator

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from layout import (
    Coord,
    TileTensor,
    TensorLayout,
    DefaultEngine,
    Idx,
    row_major,
    stack_allocation,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.intrinsics import ldg
from linalg.utils import elementwise_epilogue_type

from std.utils import StaticTuple
from std.utils.index import Index
from std.utils.numerics import isnan

comptime BLOCK_DIM = 8


# BM: The threadblock size for M dimension SMEM caching.
# BN: The threadblock size for N dimension SMEM caching.
# BK: The threadblock size for K dimension SMEM caching.
# WM: M dim of continuous tile computed by each warp.
# WN: N dim of continuous tile computed by each warp.
# WMITER: The number of subwarp tiling steps in M dimension.
# WNITER: The number of subwarp tiling steps in N dimension.
# TM: The per-thread tile size for M dimension.
# TN: The per-thread tile size for N dimension.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(NUM_THREADS))
)
def sgemm_warp_tiling_kernel[
    c_type: DType,
    CLayoutType: TensorLayout,
    a_type: DType,
    ALayoutType: TensorLayout,
    b_type: DType,
    BLayoutType: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WMITER: Int,
    WNITER: Int,
    TM: Int,
    TN: Int,
    NUM_THREADS: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    mat_c: TileTensor[
        c_type,
        CLayoutType,
        MutAnyOrigin,
        Engine=DefaultEngine[element_width=1],
    ],
    mat_a: TileTensor[
        a_type,
        ALayoutType,
        MutAnyOrigin,
        Engine=DefaultEngine[element_width=1],
    ],
    mat_b: TileTensor[
        b_type,
        BLayoutType,
        MutAnyOrigin,
        Engine=DefaultEngine[element_width=1],
    ],
    alpha: Scalar[c_type],
    beta: Scalar[c_type],
) where (a_type.is_numeric() and b_type.is_numeric()):
    var K = Int(mat_a.dim[1]())
    var N = Int(mat_c.dim[1]())

    var c_row = block_idx.y
    var c_col = block_idx.x

    # Placement of the warp in the threadblock tile.
    var warp_idx = warp_id()  # the warp this thread is in
    var warp_row, warp_col = udivmod(warp_idx, BN // WN)

    # Size of the warp sub-tile.
    comptime w_sub_m = WM // WMITER  # 64/2=32
    comptime w_sub_n = WN // WNITER  # 32/2=16

    # Placement of the thread in the warp sub-tile.
    var thread_Idx_In_warp = umod(thread_idx.x, WARP_SIZE)  # [0, 31]
    var thread_row_in_warp, thread_col_in_warp = udivmod(
        thread_Idx_In_warp, w_sub_n // TN
    )

    # Allocate space for the current blocktile in SMEM.
    # Pad the A tile in share memory to avoid bank conflicts.
    # Use 4 to comply with f4 alignment used in accumulation.
    comptime sram_bank_padding_size = 4
    comptime BM_padded = BM + sram_bank_padding_size
    var a_sram = stack_allocation[a_type, address_space=.SHARED](
        row_major[BK * BM_padded]()
    )
    var b_sram = stack_allocation[b_type, address_space=.SHARED](
        row_major[BK * BN]()
    )

    # Move blocktile to beginning of A's row and B's column.
    var aa_ptr = mat_a._storage + c_row * BM * K
    var bb_ptr = mat_b._storage + c_col * BN
    # Move C_ptr to warp's output tile
    var M_offset_warp = c_row * BM + warp_row * WM
    var N_offset_warp = c_col * BN + warp_col * WN
    var cc_ptr = mat_c._storage + M_offset_warp * N + N_offset_warp

    # Calculate the indices that this thread will load into SMEM.
    # We load 128bit / 32bit = 4 elements per thread at each step.
    var inner_row_a, inner_col_a = udivmod(thread_idx.x, BK // 4)
    comptime row_stride_a = (NUM_THREADS * 4) // BK
    var inner_row_b, inner_co_ib = udivmod(thread_idx.x, BN // 4)
    comptime row_stride_b = NUM_THREADS // (BN // 4)

    # TODO: We want these to be register-allocated!
    # Allocate thread-local cache for results in register file.
    var thread_results = stack_allocation[c_type,](
        row_major[WMITER, WNITER, TM, TN]()
    )
    _ = thread_results.fill(0)

    # We cache into registers on the warptile level.
    var reg_m = stack_allocation[a_type](row_major[WMITER, TM]())
    _ = reg_m.fill(0)

    var reg_n = stack_allocation[b_type](row_major[WNITER, TN]())
    _ = reg_n.fill(0)

    # Outer-most loop over block tiles.
    for _ in range(0, K, BK):
        for offset in range(0, BM - row_stride_a + 1, row_stride_a):
            # Load 4 elements at a time and store to shared memory.
            var tmp = ldg[width=4](
                aa_ptr + ((inner_row_a + offset) * K) + inner_col_a * 4
            )

            comptime for i in range(4):
                a_sram[
                    (inner_col_a * 4 + i) * BM_padded + inner_row_a + offset
                ] = tmp[i]

        for offset in range(0, BK - row_stride_b + 1, row_stride_b):
            # Load 4 elements at a time and store to shared memory.
            var tmp = ldg[width=4](
                bb_ptr + (inner_row_b + offset) * N + inner_co_ib * 4
            )
            b_sram.store[alignment=16](
                ((inner_row_b + offset) * BN + inner_co_ib * 4,),
                tmp,
            )

        barrier()

        for dot_idx in range(BK):
            # Populate registers for whole warptile.
            comptime for w_sub_row_idx in range(WMITER):
                comptime for i in range(0, TM, 4):
                    var vec = a_sram.load[width=4, alignment=16](
                        Coord(
                            (dot_idx * BM_padded)
                            + warp_row * WM
                            + w_sub_row_idx * w_sub_m
                            + thread_row_in_warp * TM
                            + i
                        )
                    )
                    reg_m.store((w_sub_row_idx, i), vec)

            comptime for w_sub_col_idx in range(WNITER):
                comptime for i in range(0, TN, 4):
                    var vec = b_sram.load[width=4, alignment=16](
                        Coord(
                            (dot_idx * BN)
                            + warp_col * WN
                            + w_sub_col_idx * w_sub_n
                            + thread_col_in_warp * TN
                        )
                    )
                    reg_n.store((w_sub_col_idx, i), vec)

            # Execute warptile matmul.
            comptime for w_sub_row_idx in range(WMITER):
                comptime for w_sub_col_idx in range(WNITER):
                    # Calculate per-thread results.
                    comptime for res_idx_m in range(TM):
                        comptime for res_idx_n in range(TN):
                            thread_results[
                                w_sub_row_idx,
                                w_sub_col_idx,
                                res_idx_m,
                                res_idx_n,
                            ] += (
                                reg_m[w_sub_row_idx, res_idx_m].cast[c_type]()
                                * reg_n[w_sub_col_idx, res_idx_n].cast[c_type]()
                            )
        aa_ptr = aa_ptr + BK  # move BK columns to right
        bb_ptr = bb_ptr + BK * N  # move BK rows down
        barrier()

    # Write out the results.
    comptime for w_sub_row_idx in range(WMITER):
        comptime for w_sub_col_idx in range(WNITER):
            # Move C pointer to current warp sub-tile.
            var M_offset_subtile = w_sub_row_idx * w_sub_m
            var N_offset_subtile = w_sub_col_idx * w_sub_n
            var C_interim = cc_ptr + M_offset_subtile * N + N_offset_subtile

            comptime for res_idx_m in range(TM):
                comptime for res_idx_n in range(0, TN, 4):
                    var M_offset_val = thread_row_in_warp * TM + res_idx_m
                    var N_offset_val = thread_col_in_warp * TN + res_idx_n
                    var c_idx = M_offset_val * N + N_offset_val
                    var result_vec = thread_results.load[width=4](
                        (
                            w_sub_row_idx,
                            w_sub_col_idx,
                            res_idx_m,
                            res_idx_n,
                        )
                    )

                    var vec = alpha * result_vec + beta * C_interim.load[
                        width=4, alignment=16
                    ](c_idx)

                    comptime if elementwise_lambda_fn:
                        comptime elementwise_lambda = elementwise_lambda_fn.value()
                        elementwise_lambda[c_type, 4](
                            Index(
                                M_offset_warp + M_offset_subtile + M_offset_val,
                                N_offset_warp + N_offset_subtile + N_offset_val,
                            ),
                            vec,
                        )
                    else:
                        C_interim.store[alignment=16](c_idx, vec)


def matmul_naive(
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    var x = global_idx.x
    var y = global_idx.y

    if x >= m or y >= n:
        return

    var accum = Float32(0)
    for i in range(k):
        accum = a_ptr[x * k + i] * b_ptr[i * n + y] + accum
    c_ptr[x * n + y] = accum


def bench_matmuls(mut m: Bench, ctx: DeviceContext) raises:
    print("== run_matmul_kernel_10")

    comptime M = 4096
    comptime N = 4096
    comptime K = 4096

    # TODO: Find best for target GPU.
    #       For A100 see below (based on siboehm repo).
    #       For MI300X we need to further autotune (below is a working version).
    # alias K10_NUM_THREADS = 256 if has_amd_gpu_accelerator() else 128
    # alias K10_BN = 128
    # alias K10_BM = 64
    # alias K10_BK = 16
    # alias K10_WN = 64
    # alias K10_WM = 32
    # alias K10_WNITER = 1
    # alias K10_TN = 4
    # alias K10_TM = 4
    # Settings for A6000
    comptime K10_NUM_THREADS = 256 if has_amd_gpu_accelerator() else 128
    comptime K10_BN = 128
    comptime K10_BM = 256 if has_amd_gpu_accelerator() else 128
    comptime K10_BK = 16
    comptime K10_WN = 64
    comptime K10_WM = 128 if has_amd_gpu_accelerator() else 64
    comptime K10_WNITER = 4
    comptime K10_TN = 4
    comptime K10_TM = 8

    comptime NUM_WARPS = K10_NUM_THREADS // WARP_SIZE
    comptime K10_WMITER = (K10_WM * K10_WN) // (
        WARP_SIZE * K10_TM * K10_TN * K10_WNITER
    )

    # Warptile in threadblocktile.
    comptime assert (K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0)
    comptime assert (K10_BN // K10_WN) * (K10_BM // K10_WM) == NUM_WARPS

    # Threads in the warp sub-tile.
    comptime assert (K10_WM * K10_WN) % (
        WARP_SIZE * K10_TM * K10_TN * K10_WNITER
    ) == 0

    # Warp sub-tile in warp tile.
    comptime assert (K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0)

    comptime assert (K10_NUM_THREADS * 4) % K10_BK == 0, (
        "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
        "issues during GMEM->SMEM tiling (loading only parts of the "
        "final row of Bs during each iteration)"
    )
    comptime assert (K10_NUM_THREADS * 4) % K10_BN == 0, (
        "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
        "issues during GMEM->SMEM tiling (loading only parts of the "
        "final row of As during each iteration)"
    )

    comptime assert (
        K10_BN % (16 * K10_TN) == 0
    ), "BN must be a multiple of 16*TN to avoid quantization effects"
    comptime assert (
        K10_BM % (16 * K10_TM) == 0
    ), "BM must be a multiple of 16*TM to avoid quantization effects"

    comptime assert (K10_BM * K10_BK) % (
        4 * K10_NUM_THREADS
    ) == 0, "BM*BK must be a multiple of 4*256 to vectorize loads"
    comptime assert (K10_BN * K10_BK) % (
        4 * K10_NUM_THREADS
    ) == 0, "BN*BK must be a multiple of 4*256 to vectorize loads"

    comptime assert K10_TM % 4 == 0, "TM must be a multiple of 4"

    comptime assert K10_TN % 4 == 0, "TN must be a multiple of 4"

    var a_host = alloc[Float32](M * K)
    var b_host = alloc[Float32](K * N)
    var c_host = alloc[Float32](M * N)
    var c_host_naive = alloc[Float32](M * N)

    for i in range(M * K):
        a_host[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = Float32(i + 1)

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_naive[i] = 0

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)

    comptime c_layout = row_major[M, N]()
    comptime a_layout = row_major[M, K]()
    comptime b_layout = row_major[K, N]()

    var c_buffer = TileTensor(c_device, c_layout)
    var a_buffer = TileTensor(a_device, a_layout)
    var b_buffer = TileTensor(b_device, b_layout)

    comptime sgemm_type = sgemm_warp_tiling_kernel[
        DType.float32,
        type_of(c_layout),
        DType.float32,
        type_of(a_layout),
        DType.float32,
        type_of(b_layout),
        BM=K10_BM,
        BN=K10_BN,
        BK=K10_BK,
        WM=K10_WM,
        WN=K10_WN,
        WMITER=K10_WMITER,
        WNITER=K10_WNITER,
        TM=K10_TM,
        TN=K10_TN,
        NUM_THREADS=K10_NUM_THREADS,
    ]

    @always_inline
    def bench_matmul_10(mut b: Bencher) {imm}:
        @always_inline
        def run_func(ctx: DeviceContext) raises {imm}:
            ctx.enqueue_function[sgemm_type](
                c_buffer,
                a_buffer,
                b_buffer,
                Float32(1),
                Float32(0),
                grid_dim=(ceildiv(N, K10_BN), ceildiv(M, K10_BM)),
                block_dim=(K10_NUM_THREADS,),
            )

        bencher_iter_custom(b, run_func, ctx)

    m.bench_function(
        bench_matmul_10,
        BenchId("matmul_sgemm_10"),
        [ThroughputMeasure(BenchMetric.elements, 2 * M * N * K)],
    )

    ctx.enqueue_copy(c_host, c_device)

    # Perform naive matmul to compare results & performance.

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host_naive)

    @always_inline
    def bench_naive(mut b: Bencher) {imm}:
        @always_inline
        def run_func_naive(ctx: DeviceContext) raises {imm}:
            ctx.enqueue_function[matmul_naive](
                a_device,
                b_device,
                c_device,
                Int32(M),
                Int32(N),
                Int32(K),
                grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        bencher_iter_custom(b, run_func_naive, ctx)

    m.bench_function(
        bench_naive,
        BenchId("matmul_naive"),
        # TODO: Pick relevant benchmetric
        [ThroughputMeasure(BenchMetric.elements, 2 * M * N * K)],
    )

    ctx.enqueue_copy(c_host_naive, c_device)
    ctx.synchronize()

    for i in range(M * N):
        if (
            c_host[i] != c_host_naive[i]
            or isnan(c_host_naive[i])
            or isnan(c_host[i])
        ):
            print(c_host[i])
            print(c_host_naive[i])
            raise "Failed ❌: results mismatch"


def main() raises:
    with DeviceContext() as ctx:
        var m = Bench()
        bench_matmuls(m, ctx)
        m.dump_report()
