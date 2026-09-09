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

"""Partial-K (.ws) MMA correctness: `bulk_mma_ws_partial` (SS) and
`bulk_mma_ws_ts_partial` (TS) on SM100 (B200).

The partial-K primitive
=======================
The full-tile `bulk_mma_ws` (SS) and `bulk_mma_ws_ts` (TS) wrappers issue
exactly `num_k_mmas` MMA_K blocks, unconditionally.  When the last KV tile
of an attention sequence is only partially populated, the trailing K-blocks
of the SMEM/TMEM operands hold uninitialized (possibly NaN) data.  An
unconditional MMA over those blocks computes `0 * NaN = NaN` and poisons the
accumulator.

The `*_partial` wrappers add a runtime `valid_k_mmas: UInt32` (the count of
LOADED MMA_K blocks) and a comptime `k_start: Int = 0`.  Internally each
k-block uses the absolute index `jj = k_start + k` and a SEPARATE,
warp-uniform predicate `%pv = (valid_k_mmas <= jj)`; the block's MMA is
guarded `@!%pv`, so blocks `jj >= valid_k_mmas` issue NO MMA at all -- their
A/B inputs are never read, which is exactly what avoids the `0 * NaN = NaN`
hazard.  `%pv` is kept independent of the single-lane elect predicate `%pj`
so the elect codegen is byte-identical to the full-tile wrappers (no
`BSYNC.RECONVERGENT` in the SASS).

The fold-agnostic oracle trick
==============================
For a partial run `partial[num_k_mmas=N, valid_k_mmas=V]`, the exact oracle
is the FULL wrapper instantiated at COMPILE TIME with `num_k_mmas=V`:
`full[num_k_mmas=V]`.  Both contract exactly blocks `[0, V)` using the SAME
descriptors and per-block offsets, so their outputs must match -- with NO
truncation math, NO zeroing, NO fold reasoning.

Each kernel below is generic over `[NUM_K_MMAS: Int, USE_PARTIAL: Bool]` and
`comptime if USE_PARTIAL:` switches between the `*_partial` wrapper (passing
the runtime `valid_k_mmas` kernel arg) and the plain full wrapper (ignoring
`valid_k_mmas`).  EVERYTHING else (TMA loads, descriptors, tcgen05_cp,
readback) is byte-identical to the model smoke tests:

  * SS path clones `qk_smoke_kernel` from test_mla_layout_g_mma_smoke.mojo:
    A[32,576] x B[64,576]^T FP8 e4m3, K=576, MMA_K=32, num_k_mmas=18, both
    operands k-major SMEM descriptors, transpose_b=True, M=32 readback.
  * TS path clones `dense_mma_ws_ts_kernel` from test_mma_ws_ts.mojo:
    Q[64,512] BF16 -> SMEM (TMA) -> TMEM (tcgen05_cp, 16 strips), K[64,512]
    -> SMEM, dual-GEMM fold B descriptor [64,512]->[128,256], MMA_N=128,
    MMA_K=16, num_k_mmas=16, fold-sum readback.

Variants
========
SS (test_ss_partial):
  1. Ground truth: full[18] vs naive K=576 (== existing test_qk_smoke).
  2. Degeneracy:  partial[18, valid=18] vs naive K=576 (partial == full).
  3. Partial + NaN safety: partial[18, valid=10], A/B tail cols [320,576)
     filled with a genuine FP8 NaN; assert every output is finite AND
     matches naive K=320.
  4. Multi-stage k_start>0: two back-to-back partial calls into the same
     accumulator, stage0 k_start=0 num_k_mmas=9, stage1 k_start=9
     num_k_mmas=9, both valid=18, vs naive K=576.
  5. Struct dispatch: SM100TensorAccumulator[a_tmem=False].mma (auto-ws for
     cta_group=1, MMA_M=32) vs naive K=576.
  6. Struct multi-stage: SM100TensorAccumulator[num_stages=3].mma
     (swizzle-aligned stage offsets) vs naive K=576.
  7. Struct partial degeneracy: struct.mma_maybe_partial_k[valid=18] vs
     naive K=576.
  8. Struct partial + NaN: struct.mma_maybe_partial_k[valid=10] vs naive
     K=320, finite-checked.
  9. Struct partial multi-stage: struct[num_stages=3].mma_maybe_partial_k
     [valid=10] (valid boundary inside stage 1; stage 2 fully skipped).

SS non-ws (test_ss_nonws_partial), M=128 so the struct auto-selects the
NON-ws path (plain tcgen05.mma + the new `bulk_mma_ss_partial`), standard
TMEM readback (row = lane):
  1. Ground truth: raw full[18] (bulk_mma) vs naive K=576.
  2. Degeneracy:  raw bulk_mma_ss_partial[18, valid=18] vs naive K=576.
  3. Struct partial degeneracy: struct.mma_maybe_partial_k[valid=18].
  4. Partial + NaN safety: raw partial[18, valid=10], NaN tail, finite.
  5. Struct partial + NaN: struct.mma_maybe_partial_k[valid=10], finite.
  6. Struct partial multi-stage: struct[num_stages=3].mma_maybe_partial_k
     [valid=10] (k_start>0 path of bulk_mma_ss_partial).

TS (test_ts_partial):
  1. Ground truth: full[16] vs fold-sum naive (== existing
     test_dense_mma_ws_ts).
  2. Degeneracy:  partial[16, valid=16] vs fold-sum naive (partial == full).
  3. Skip correctness (fold-agnostic, GPU-vs-GPU): partial[16, valid=10] vs
     full[10] on the SAME random Q/K; assert RAW [64,128] outputs are
     element-wise equal.  No fold-sum, no fold mapping needed.
  4. Struct dispatch: SM100TensorAccumulator[a_tmem=True].mma (auto-ws for
     cta_group=1, MMA_M=64) vs fold-sum naive.
  5. Struct multi-stage: SM100TensorAccumulator[num_stages=2].mma
     (3+1 split) vs fold-sum naive.
  6. Struct partial degeneracy: struct.mma_maybe_partial_k[valid=16] vs
     fold-sum naive.
  7. Struct partial skip: struct.mma_maybe_partial_k[valid=10] vs the
     full[10] oracle, raw outputs element-wise equal.

The TS NaN-safety variant is intentionally SKIPPED (see comment in
test_ts_partial): the `@!%pv` skip guard is byte-identical PTX in both the SS
and TS builders, so the SS NaN test (variant 3) already proves the guard's
NaN-safety for the shared mechanism, and the TS GPU-vs-GPU skip test proves
the TS operand/offset wiring sums exactly [0,V).  Reasoning about the dual
fold's NaN mapping `(c % 256)//16` adds risk without adding coverage.

Run via:
    ./bazelw test --config=remote-b200 \\
        //max/kernels/test/gpu/linalg:test_mma_ws_partial_sm100.mojo.test \\
        --curses=no --noshow_progress
"""

from std.math import ceildiv, isfinite
from std.memory import alloc, bitcast
from std.random import randn, seed
from std.sys import size_of

from max.gpu import thread_idx, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import (
    external_memory,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma_arrive,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_cp,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from layout import (
    Coord,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import tile_layout_k_major, tile_to_descriptor
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from linalg.arch.sm100.mma import smem_descriptor
from linalg.matmul.gpu import matmul_kernel_naive
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    bulk_mma,
    bulk_mma_ss_partial,
    bulk_mma_ws,
    bulk_mma_ws_partial,
    bulk_mma_ws_ts,
    bulk_mma_ws_ts_partial,
    elect,
)
from std.testing import assert_almost_equal, assert_true
from std.utils.index import Index, IndexList

# ---------------------------------------------------------------------------
# Shared compile-time constants
# ---------------------------------------------------------------------------
comptime FP8_TYPE = DType.float8_e4m3fn
comptime ACC_TYPE = DType.float32
comptime REF_TYPE = DType.bfloat16  # for naive matmul reference

# Single warpgroup = 4 warps = 128 threads.
comptime NUM_THREADS = 128
comptime NAIVE_BLOCK_DIM = 16


# ===========================================================================
# SS partial-K test (clone of qk_smoke_kernel)
# ===========================================================================
# QK shape: M=32, N=64, K=576 (transpose_b=True), FP8 e4m3.
comptime QK_M = 32
comptime QK_N = 64
comptime QK_K = 576

comptime QK_MMA_K = 32  # FP8 hardware MMA_K
comptime QK_NUM_K_MMAS = QK_K // QK_MMA_K  # 18

comptime QK_SWIZZLE = TensorMapSwizzle.SWIZZLE_64B

comptime QK_A_BYTES = QK_M * QK_K * size_of[FP8_TYPE]()  # 18432
comptime QK_B_BYTES = QK_N * QK_K * size_of[FP8_TYPE]()  # 36864

comptime QK_A_OFFSET = 0
comptime QK_B_OFFSET = QK_A_BYTES
comptime QK_META_OFFSET = QK_A_BYTES + QK_B_BYTES
# tmem_addr (8B), a_mbar (8B), b_mbar (8B), mma_mbar (8B) = 32B.
comptime QK_TOTAL_SMEM = QK_META_OFFSET + 32

comptime MAX_TMEM_COLS: UInt32 = 512

# FP8 tolerances (~3 bits of mantissa, K up to 576 of N(0,1) elements).
comptime ATOL_QK: Float32 = 5.0
comptime RTOL_QK: Float32 = 0.10

# Partial-run parameters.
comptime QK_VALID = 10  # loaded MMA_K blocks (variant 3): K_valid = 320.
comptime QK_VALID_K = QK_VALID * QK_MMA_K  # 320
comptime QK_STAGE0 = 9  # multi-stage split point (variant 4).


# ---------------------------------------------------------------------------
# Kernel: SS QK partial -- generic over NUM_K_MMAS and USE_PARTIAL.
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def ss_qk_partial_kernel[
    NUM_K_MMAS: Int,
    USE_PARTIAL: Bool,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    USE_STRUCT: Bool = False,
    NUM_STAGES: Int = 1,
](
    a_tma_op: TMATensorTile[FP8_TYPE, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[FP8_TYPE, b_tile_rank, b_tile_shape, b_desc_shape],
    c_output: LayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N), MutAnyOrigin
    ],
    valid_k_mmas: UInt32,
):
    """SS (.ws) MMA: C [32,64] = A [32,576] x B [64,576]^T (FP8 e4m3).

    The MMA call is comptime-branched between the full-tile `bulk_mma_ws`
    (USE_PARTIAL=False, ignores `valid_k_mmas`) and `bulk_mma_ws_partial`
    (USE_PARTIAL=True, passes `valid_k_mmas`).  Everything else is identical
    to `qk_smoke_kernel`.
    """

    # ---- Dynamic SMEM ----
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var a_smem_ptr = (smem_base + QK_A_OFFSET).bitcast[Scalar[FP8_TYPE]]()
    var b_smem_ptr = (smem_base + QK_B_OFFSET).bitcast[Scalar[FP8_TYPE]]()

    var a_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(QK_M, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](a_smem_ptr.as_unsafe_any_origin())
    var b_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(QK_N, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](b_smem_ptr.as_unsafe_any_origin())

    # ---- Metadata ----
    var metadata_ptr = (smem_base + QK_META_OFFSET).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    var a_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var b_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    if elect_one_thread:
        a_mbar[0].init()
        b_mbar[0].init()
        mma_mbar[0].init()

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    if elect_one_thread:
        a_mbar[0].expect_bytes(Int32(QK_A_BYTES))
        a_tma_op.async_copy(a_smem_tile, a_mbar[0], (0, 0))

        b_mbar[0].expect_bytes(Int32(QK_B_BYTES))
        b_tma_op.async_copy(b_smem_tile, b_mbar[0], (0, 0))
    barrier()

    a_mbar[0].wait()
    b_mbar[0].wait()
    barrier()

    # ---- SMEM descriptors (k-major for both A and B) ----
    var a_desc = smem_descriptor[
        BMN=QK_M,
        BK=QK_K,
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](a_smem_ptr)
    var b_desc = smem_descriptor[
        BMN=QK_N,
        BK=QK_K,
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](b_smem_ptr)

    comptime mma_idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        ACC_TYPE,
        FP8_TYPE,
        FP8_TYPE,
        Index[dtype=.uint32](QK_M, QK_N),
        transpose_b=True,
    ]()

    var c_tmem = tmem_addr

    var e: Int32 = 0
    if wid == 0:
        e = elect()

    comptime if USE_STRUCT:
        # Struct-driven dispatch: SM100TensorAccumulator auto-selects the
        # ws path (cta_group=1, MMA_M=32 <= 64).  Always contracts the full
        # BK=576, so it is only valid at NUM_K_MMAS == 18; NUM_STAGES > 1
        # exercises the multi-stage ws arm (stage k-offsets must stay
        # swizzle-aligned for `mma`: 18 % NUM_STAGES == 0 and blocks/stage
        # even; `mma_maybe_partial_k` makes no such assumption).
        comptime assert NUM_K_MMAS == QK_NUM_K_MMAS
        comptime UMMA = SM100TensorAccumulator[
            FP8_TYPE,
            ACC_TYPE,
            MMA_M=QK_M,
            MMA_N=QK_N,
            BK=QK_K,
            a_tmem=False,
            mma_kind=UMMAKind.KIND_F8F6F4,
            swizzle_a=QK_SWIZZLE,
            swizzle_b=QK_SWIZZLE,
            transpose_b=True,
            cta_group=1,
            num_stages=NUM_STAGES,
        ]
        comptime assert UMMA.use_ws
        comptime for s in range(NUM_STAGES):
            comptime if USE_PARTIAL:
                UMMA.mma_maybe_partial_k[stage_idx=s](
                    a_desc,
                    b_desc,
                    c_tmem,
                    c_scale=UInt32(0),
                    elect=e,
                    valid_k_mmas=valid_k_mmas,
                )
            else:
                UMMA.mma[stage_idx=s](
                    a_desc, b_desc, c_tmem, c_scale=UInt32(0), elect=e
                )
    elif USE_PARTIAL:
        bulk_mma_ws_partial[
            UMMAKind.KIND_F8F6F4,
            FP8_TYPE,
            FP8_TYPE,
            a_BMN=QK_M,
            a_BK=QK_K,
            a_swizzle=QK_SWIZZLE,
            a_is_k_major=True,
            b_BMN=QK_N,
            b_BK=QK_K,
            b_swizzle=QK_SWIZZLE,
            b_is_k_major=True,
            num_k_mmas=NUM_K_MMAS,
            operand_size=size_of[FP8_TYPE](),
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=QK_MMA_K,
        ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e, valid_k_mmas)
    else:
        bulk_mma_ws[
            UMMAKind.KIND_F8F6F4,
            FP8_TYPE,
            FP8_TYPE,
            a_BMN=QK_M,
            a_BK=QK_K,
            a_swizzle=QK_SWIZZLE,
            a_is_k_major=True,
            b_BMN=QK_N,
            b_BK=QK_K,
            b_swizzle=QK_SWIZZLE,
            b_is_k_major=True,
            num_k_mmas=NUM_K_MMAS,
            operand_size=size_of[FP8_TYPE](),
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=QK_MMA_K,
        ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    # ---- Read C from TMEM (M=32 .ws cta_group::1: each warp owns N/4 cols
    # at TMEM offset 0). ----
    comptime QK_COLS_PER_WARP = QK_N // 4  # 16
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=QK_COLS_PER_WARP,
        dtype=ACC_TYPE,
        pack=False,
        width=QK_COLS_PER_WARP,
    ](c_tmem)

    tcgen05_load_wait()

    var lane_id = Int(tid % 32)
    var p_row = lane_id
    var p_col_base = Int(wid) * QK_COLS_PER_WARP
    for j in range(QK_COLS_PER_WARP):
        c_output[p_row, p_col_base + j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# Kernel: SS QK multi-stage (k_start>0) -- two back-to-back partial calls.
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def ss_qk_multistage_kernel[
    S: Int,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
](
    a_tma_op: TMATensorTile[FP8_TYPE, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[FP8_TYPE, b_tile_rank, b_tile_shape, b_desc_shape],
    c_output: LayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N), MutAnyOrigin
    ],
    valid_k_mmas: UInt32,
):
    """Two partial calls into the SAME accumulator, no barrier between.

    stage0: k_start=0,  num_k_mmas=S        (c_scale=0 -> %ps init at jj==0)
    stage1: k_start=S,  num_k_mmas=18-S     (first block k==0 sets %ps=1 ->
                                             accumulate onto stage0)
    Both pass valid_k_mmas=18, so no block is skipped; this exercises the
    `k == 0 or jj == 1` scale-predicate branch and the absolute-offset path
    with k_start>0.
    """
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var a_smem_ptr = (smem_base + QK_A_OFFSET).bitcast[Scalar[FP8_TYPE]]()
    var b_smem_ptr = (smem_base + QK_B_OFFSET).bitcast[Scalar[FP8_TYPE]]()

    var a_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(QK_M, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](a_smem_ptr.as_unsafe_any_origin())
    var b_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(QK_N, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](b_smem_ptr.as_unsafe_any_origin())

    var metadata_ptr = (smem_base + QK_META_OFFSET).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    var a_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var b_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    if elect_one_thread:
        a_mbar[0].init()
        b_mbar[0].init()
        mma_mbar[0].init()

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    if elect_one_thread:
        a_mbar[0].expect_bytes(Int32(QK_A_BYTES))
        a_tma_op.async_copy(a_smem_tile, a_mbar[0], (0, 0))

        b_mbar[0].expect_bytes(Int32(QK_B_BYTES))
        b_tma_op.async_copy(b_smem_tile, b_mbar[0], (0, 0))
    barrier()

    a_mbar[0].wait()
    b_mbar[0].wait()
    barrier()

    var a_desc = smem_descriptor[
        BMN=QK_M,
        BK=QK_K,
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](a_smem_ptr)
    var b_desc = smem_descriptor[
        BMN=QK_N,
        BK=QK_K,
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](b_smem_ptr)

    comptime mma_idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        ACC_TYPE,
        FP8_TYPE,
        FP8_TYPE,
        Index[dtype=.uint32](QK_M, QK_N),
        transpose_b=True,
    ]()

    var c_tmem = tmem_addr

    var e: Int32 = 0
    if wid == 0:
        e = elect()

    # Stage 0: blocks [0, S). c_scale=0 -> the absolute first block (jj==0)
    # initializes the accumulator (%ps = (c_scale != 0) = false).
    bulk_mma_ws_partial[
        UMMAKind.KIND_F8F6F4,
        FP8_TYPE,
        FP8_TYPE,
        a_BMN=QK_M,
        a_BK=QK_K,
        a_swizzle=QK_SWIZZLE,
        a_is_k_major=True,
        b_BMN=QK_N,
        b_BK=QK_K,
        b_swizzle=QK_SWIZZLE,
        b_is_k_major=True,
        num_k_mmas=S,
        operand_size=size_of[FP8_TYPE](),
        tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        mma_k=QK_MMA_K,
        k_start=0,
    ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e, valid_k_mmas)

    # Stage 1: blocks [S, 18). k_start=S so jj starts at S (> 1); the builder's
    # `k == 0` branch pins %ps=1 -> this stage accumulates onto stage 0.
    # NOTE: no barrier and a single mma_arrive after both stages -- the two
    # inline-asm sequences issue from the same elected lane into the same
    # accumulator and are ordered by the MMA pipeline.
    bulk_mma_ws_partial[
        UMMAKind.KIND_F8F6F4,
        FP8_TYPE,
        FP8_TYPE,
        a_BMN=QK_M,
        a_BK=QK_K,
        a_swizzle=QK_SWIZZLE,
        a_is_k_major=True,
        b_BMN=QK_N,
        b_BK=QK_K,
        b_swizzle=QK_SWIZZLE,
        b_is_k_major=True,
        num_k_mmas=QK_NUM_K_MMAS - S,
        operand_size=size_of[FP8_TYPE](),
        tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        mma_k=QK_MMA_K,
        k_start=S,
    ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e, valid_k_mmas)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    comptime QK_COLS_PER_WARP = QK_N // 4  # 16
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=QK_COLS_PER_WARP,
        dtype=ACC_TYPE,
        pack=False,
        width=QK_COLS_PER_WARP,
    ](c_tmem)

    tcgen05_load_wait()

    var lane_id = Int(tid % 32)
    var p_row = lane_id
    var p_col_base = Int(wid) * QK_COLS_PER_WARP
    for j in range(QK_COLS_PER_WARP):
        c_output[p_row, p_col_base + j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# SS helpers
# ---------------------------------------------------------------------------
def fill_random_fp8[
    dtype: DType
](ptr: MutPointer[Scalar[dtype], MutAnyOrigin], n: Int):
    """Generates random FP8 values via float32 RNG -> cast (matches the
    model smoke test)."""
    var f32_buf = alloc[Float32](n)
    randn[.float32](f32_buf, n)
    for i in range(n):
        var v = f32_buf[i] * 0.5
        if v > 2.0:
            v = 2.0
        if v < -2.0:
            v = -2.0
        ptr[i] = Scalar[dtype](v)
    f32_buf.free()


def dequant_fp8_to_bf16[
    src_dtype: DType, dst_dtype: DType
](
    src: ImmPointer[Scalar[src_dtype], _],
    dst: MutPointer[Scalar[dst_dtype], _],
    n: Int,
):
    for i in range(n):
        dst[i] = src[i].cast[dst_dtype]()


def fp8_nan() -> Scalar[FP8_TYPE]:
    """Returns a genuine FP8 e4m3fn NaN.

    e4m3fn ("fn" = finite) has NO infinities but DOES have a NaN encoding:
    sign=1, exp=1111, mant=111 -> 0xFF.  We build it via a byte bitcast
    rather than `Scalar[FP8_TYPE](Float32(nan))` so the NaN bit pattern is
    deterministic and independent of the f32->fp8 cast's NaN rounding.
    """
    return bitcast[FP8_TYPE](UInt8(0xFF))


def _ss_naive_ref[
    k_valid: Int, M: Int = QK_M
](
    ctx: DeviceContext,
    a_ref_dev: DeviceBuffer[REF_TYPE],
    b_ref_dev: DeviceBuffer[REF_TYPE],
) raises -> DeviceBuffer[ACC_TYPE]:
    """Computes C_ref[M,QK_N] = A_bf16[:, :k_valid] x B_bf16[:, :k_valid]^T
    via the naive matmul kernel (reads only the leading k_valid K-columns)."""
    var c_ref_dev = ctx.enqueue_create_buffer[ACC_TYPE](M * QK_N)

    var c_ref_tt = TileTensor(c_ref_dev, row_major(Coord(M, QK_N)))
    # A and B keep their physical row stride QK_K; only the K extent passed to
    # the kernel (k_valid) changes, so the naive reference reads cols
    # [0, k_valid) of each row.
    var a_tt = TileTensor(
        ImmPointer[Scalar[REF_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(a_ref_dev.unsafe_ptr())
        ),
        row_major(Coord(M, QK_K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[REF_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(b_ref_dev.unsafe_ptr())
        ),
        row_major(Coord(QK_N, QK_K)),
    )

    comptime gemm_naive = matmul_kernel_naive[
        ACC_TYPE,
        REF_TYPE,
        REF_TYPE,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        NAIVE_BLOCK_DIM,
        transpose_b=True,
    ]

    ctx.enqueue_function[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(QK_N),
        Int32(k_valid),
        grid_dim=(
            ceildiv(M, NAIVE_BLOCK_DIM),
            ceildiv(QK_N, NAIVE_BLOCK_DIM),
            1,
        ),
        block_dim=(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM, 1),
    )
    return c_ref_dev


def _ss_compare[
    M: Int = QK_M
](
    label: String,
    c_out_ptr: ImmPointer[Scalar[ACC_TYPE], _],
    c_ref_host: ImmPointer[Float32, _],
    require_finite: Bool,
) raises:
    """Compares the GPU output against the host reference within FP8 tol.

    Generic over the pointer origins so the caller can pass a LayoutTensor's
    `.ptr` (MutAnyOrigin) and an `alloc`'d host buffer (MutUntrackedOrigin)
    without an origin mismatch.

    When `require_finite` is True (the NaN-safety variant), additionally
    asserts every GPU output element is finite -- a leaked `0*NaN` would
    surface as a NaN here.
    """
    var max_abs_err: Float32 = 0.0
    var max_rel_err: Float32 = 0.0
    var num_failures: Int = 0
    var num_nonfinite: Int = 0
    for r in range(M):
        for c in range(QK_N):
            var idx = r * QK_N + c
            var ref_val = c_ref_host[idx]
            var gpu_val = c_out_ptr[idx]
            if not isfinite(gpu_val):
                num_nonfinite += 1
                continue
            var abs_err = abs(gpu_val - ref_val)
            var rel_err = abs_err / max(abs(ref_val), Float32(1.0))
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            if abs_err > ATOL_QK and rel_err > RTOL_QK:
                num_failures += 1

    print(
        "  ["
        + label
        + "] max abs err="
        + String(max_abs_err)
        + " max rel err="
        + String(max_rel_err)
        + " failures="
        + String(num_failures)
        + " nonfinite="
        + String(num_nonfinite)
        + " / "
        + String(M * QK_N)
    )

    if require_finite:
        assert_true(
            num_nonfinite == 0,
            msg=String(
                label,
                ": ",
                num_nonfinite,
                " non-finite GPU outputs (0*NaN leaked past the partial guard)",
            ),
        )
    assert_true(
        num_failures == 0,
        msg=String(
            label,
            " FAILED: ",
            num_failures,
            " elements exceed tolerance (max abs=",
            max_abs_err,
            ", max rel=",
            max_rel_err,
            ")",
        ),
    )


def test_ss_partial(ctx: DeviceContext) raises:
    print("=" * 70)
    print(
        "test_ss_partial: bulk_mma_ws_partial (SS .ws) M=32 N=64 K=576 FP8 e4m3"
    )
    print("=" * 70)

    # =====================================================================
    # Variant 1 (ground truth) + Variant 2 (degeneracy):
    # full[18] and partial[18, valid=18], both vs naive K=576.
    # =====================================================================
    seed(42)

    var a_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_M, QK_K)](ctx)
    var b_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_N, QK_K)](ctx)
    var a_host = a_inp.tensor[update=False]()
    var b_host = b_inp.tensor[update=False]()
    fill_random_fp8[FP8_TYPE](a_host.ptr, QK_M * QK_K)
    fill_random_fp8[FP8_TYPE](b_host.ptr, QK_N * QK_K)

    # BF16 dequantized copies for the naive reference (full K=576).
    var a_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_M * QK_K)
    var b_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_N * QK_K)
    var a_ref_host = alloc[Scalar[REF_TYPE]](QK_M * QK_K)
    var b_ref_host = alloc[Scalar[REF_TYPE]](QK_N * QK_K)
    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](a_host.ptr, a_ref_host, QK_M * QK_K)
    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](b_host.ptr, b_ref_host, QK_N * QK_K)
    ctx.enqueue_copy(a_ref_dev, a_ref_host)
    ctx.enqueue_copy(b_ref_dev, b_ref_host)

    var a_tma_op = create_tensor_tile[
        Index(QK_M, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, a_inp.device_tensor())
    var b_tma_op = create_tensor_tile[
        Index(QK_N, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, b_inp.device_tensor())

    # Reference for K=576 (variants 1, 2, 4).
    var c_ref_full_dev = _ss_naive_ref[QK_K](ctx, a_ref_dev, b_ref_dev)
    var c_ref_full = alloc[Float32](QK_M * QK_N)
    ctx.enqueue_copy(c_ref_full, c_ref_full_dev)
    ctx.synchronize()

    # ---- Variant 1: full[18] (== existing test_qk_smoke) ----
    var c_full_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N)
    ](ctx)
    comptime kern_full = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        False,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kern_full](
        a_tma_op,
        b_tma_op,
        c_full_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),  # ignored when USE_PARTIAL=False
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v1 full[18]",
        c_full_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 2: partial[18, valid=18] ----
    var c_deg_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(QK_M, QK_N)](
        ctx
    )
    comptime kern_partial = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kern_partial](
        a_tma_op,
        b_tma_op,
        c_deg_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),  # valid == num_k_mmas: no block skipped.
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v2 partial[18,valid=18]",
        c_deg_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 4: multi-stage k_start>0 (stage0=9, stage1=9, valid=18) ----
    var c_ms_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(QK_M, QK_N)](
        ctx
    )
    comptime kern_ms = ss_qk_multistage_kernel[
        QK_STAGE0,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kern_ms](
        a_tma_op,
        b_tma_op,
        c_ms_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v4 multistage(9+9,valid=18)",
        c_ms_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 5: SM100TensorAccumulator.mma (struct ws dispatch) ----
    # The struct auto-selects the ws path (cta_group=1, MMA_M=32); the
    # single-stage struct call must match the raw bulk_mma_ws ground truth.
    var c_st_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(QK_M, QK_N)](
        ctx
    )
    comptime kern_struct = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        False,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        USE_STRUCT=True,
    ]
    ctx.enqueue_function[kern_struct](
        a_tma_op,
        b_tma_op,
        c_st_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),  # ignored when USE_STRUCT (full contraction)
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v5 struct.mma",
        c_st_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 6: struct multi-stage (num_stages=3) ws arm ----
    # 18 k-blocks / 3 stages = 6 blocks (192 FP8 elements) per stage, a
    # multiple of the 64-element SWIZZLE_64B granularity, so the struct's
    # stage byte-offset decomposition is exact.  (num_stages=2 would put a
    # stage boundary at 288 elements -- NOT swizzle-aligned -- which the
    # offset math does not support; real kernels keep stages aligned.)
    var c_st_ms_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N)
    ](ctx)
    comptime kern_struct_ms = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        False,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        USE_STRUCT=True,
        NUM_STAGES=3,
    ]
    ctx.enqueue_function[kern_struct_ms](
        a_tma_op,
        b_tma_op,
        c_st_ms_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v6 struct.mma num_stages=3",
        c_st_ms_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 7: struct.mma_maybe_partial_k degeneracy (valid=18) ----
    # The struct's partial method on the ws path (-> bulk_mma_ws_partial)
    # with valid == num_k_mmas: every validity guard is never-true, so the
    # result must equal the full contraction.
    var c_st_pd_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N)
    ](ctx)
    comptime kern_struct_pd = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        USE_STRUCT=True,
    ]
    ctx.enqueue_function[kern_struct_pd](
        a_tma_op,
        b_tma_op,
        c_st_pd_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),  # valid == num_k_mmas: no block skipped.
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v7 struct.partial[valid=18]",
        c_st_pd_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    a_ref_host.free()
    b_ref_host.free()
    c_ref_full.free()
    _ = a_ref_dev
    _ = b_ref_dev
    _ = c_ref_full_dev
    _ = a_inp^
    _ = b_inp^
    _ = c_full_buf^
    _ = c_deg_buf^
    _ = c_ms_buf^
    _ = c_st_buf^
    _ = c_st_ms_buf^
    _ = c_st_pd_buf^

    # =====================================================================
    # Variant 3 (partial + NaN safety): partial[18, valid=10] with A/B tail
    # K-cols [320,576) filled with genuine FP8 NaN.  For SS (no fold),
    # descriptor-block k reads input K-cols [k*32,(k+1)*32), so blocks
    # [10,18) read exactly the NaN region [320,576) and the partial guard
    # MUST skip them.  Reference reads only the finite [0,320) region.
    # =====================================================================
    seed(123)

    var a_nan = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_M, QK_K)](ctx)
    var b_nan = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_N, QK_K)](ctx)
    var a_nan_host = a_nan.tensor[update=False]()
    var b_nan_host = b_nan.tensor[update=False]()
    fill_random_fp8[FP8_TYPE](a_nan_host.ptr, QK_M * QK_K)
    fill_random_fp8[FP8_TYPE](b_nan_host.ptr, QK_N * QK_K)

    # Poison the tail K-columns [QK_VALID_K, QK_K) of BOTH operands with NaN.
    var nan_byte = fp8_nan()
    for r in range(QK_M):
        for c in range(QK_VALID_K, QK_K):
            a_nan_host.ptr[r * QK_K + c] = nan_byte
    for r in range(QK_N):
        for c in range(QK_VALID_K, QK_K):
            b_nan_host.ptr[r * QK_K + c] = nan_byte

    # BF16 reference operands: dequant ONLY the finite [0, QK_VALID_K) region;
    # the naive ref is invoked with k=QK_VALID_K so it never reads the tail.
    # (We still allocate full-width buffers to keep the row stride = QK_K.)
    var a_nan_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_M * QK_K)
    var b_nan_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_N * QK_K)
    var a_nan_ref_host = alloc[Scalar[REF_TYPE]](QK_M * QK_K)
    var b_nan_ref_host = alloc[Scalar[REF_TYPE]](QK_N * QK_K)
    # Dequant the full buffer is unsafe (tail is NaN); zero the tail in the
    # reference instead so a stray read can't poison the host ref either.
    for r in range(QK_M):
        for c in range(QK_K):
            var v = a_nan_host.ptr[r * QK_K + c].cast[
                REF_TYPE
            ]() if c < QK_VALID_K else Scalar[REF_TYPE](0)
            a_nan_ref_host[r * QK_K + c] = v
    for r in range(QK_N):
        for c in range(QK_K):
            var v = b_nan_host.ptr[r * QK_K + c].cast[
                REF_TYPE
            ]() if c < QK_VALID_K else Scalar[REF_TYPE](0)
            b_nan_ref_host[r * QK_K + c] = v
    ctx.enqueue_copy(a_nan_ref_dev, a_nan_ref_host)
    ctx.enqueue_copy(b_nan_ref_dev, b_nan_ref_host)

    var a_nan_tma = create_tensor_tile[
        Index(QK_M, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, a_nan.device_tensor())
    var b_nan_tma = create_tensor_tile[
        Index(QK_N, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, b_nan.device_tensor())

    var c_ref_v_dev = _ss_naive_ref[QK_VALID_K](
        ctx, a_nan_ref_dev, b_nan_ref_dev
    )
    var c_ref_v = alloc[Float32](QK_M * QK_N)
    ctx.enqueue_copy(c_ref_v, c_ref_v_dev)
    ctx.synchronize()

    var c_nan_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(QK_M, QK_N)](
        ctx
    )
    comptime kern_nan = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_nan_tma).rank,
        type_of(a_nan_tma).tile_shape,
        type_of(a_nan_tma).desc_shape,
        type_of(b_nan_tma).rank,
        type_of(b_nan_tma).tile_shape,
        type_of(b_nan_tma).desc_shape,
    ]
    ctx.enqueue_function[kern_nan](
        a_nan_tma,
        b_nan_tma,
        c_nan_buf.device_tensor(),
        UInt32(QK_VALID),  # valid=10: blocks [10,18) must be skipped.
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v3 partial[18,valid=10]+NaN",
        c_nan_buf.tensor().ptr,
        c_ref_v,
        require_finite=True,  # any leaked 0*NaN -> non-finite -> fail.
    )

    # ---- Variant 8: struct.mma_maybe_partial_k[valid=10]+NaN ----
    # Same skip + NaN-safety check as v3, driven through the struct method
    # (ws dispatch -> bulk_mma_ws_partial).
    var c_st_nan_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N)
    ](ctx)
    comptime kern_st_nan = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_nan_tma).rank,
        type_of(a_nan_tma).tile_shape,
        type_of(a_nan_tma).desc_shape,
        type_of(b_nan_tma).rank,
        type_of(b_nan_tma).tile_shape,
        type_of(b_nan_tma).desc_shape,
        USE_STRUCT=True,
    ]
    ctx.enqueue_function[kern_st_nan](
        a_nan_tma,
        b_nan_tma,
        c_st_nan_buf.device_tensor(),
        UInt32(QK_VALID),  # valid=10: blocks [10,18) must be skipped.
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v8 struct.partial[valid=10]+NaN",
        c_st_nan_buf.tensor().ptr,
        c_ref_v,
        require_finite=True,
    )

    # ---- Variant 9: struct.mma_maybe_partial_k num_stages=3, valid=10 ----
    # Stage block ranges [0,6) / [6,12) / [12,18); valid=10 lands INSIDE
    # stage 1 (blocks 10,11 skipped) and stage 2 issues nothing at all --
    # exercising the per-stage k_start plumbing of the struct method.
    var c_st_nan_ms_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(QK_M, QK_N)
    ](ctx)
    comptime kern_st_nan_ms = ss_qk_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_nan_tma).rank,
        type_of(a_nan_tma).tile_shape,
        type_of(a_nan_tma).desc_shape,
        type_of(b_nan_tma).rank,
        type_of(b_nan_tma).tile_shape,
        type_of(b_nan_tma).desc_shape,
        USE_STRUCT=True,
        NUM_STAGES=3,
    ]
    ctx.enqueue_function[kern_st_nan_ms](
        a_nan_tma,
        b_nan_tma,
        c_st_nan_ms_buf.device_tensor(),
        UInt32(QK_VALID),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare(
        "v9 struct.partial 3-stage[valid=10]+NaN",
        c_st_nan_ms_buf.tensor().ptr,
        c_ref_v,
        require_finite=True,
    )

    print(
        "  SS partial PASSED"
        " (v1 ground / v2 degeneracy / v3 NaN / v4 multistage"
        " / v5 struct / v6 struct-multistage / v7 struct-partial-degeneracy"
        " / v8 struct-partial-NaN / v9 struct-partial-3stage-NaN)"
    )

    a_nan_ref_host.free()
    b_nan_ref_host.free()
    c_ref_v.free()
    _ = a_nan_ref_dev
    _ = b_nan_ref_dev
    _ = c_ref_v_dev
    _ = a_nan^
    _ = b_nan^
    _ = c_nan_buf^
    _ = c_st_nan_buf^
    _ = c_st_nan_ms_buf^


# ===========================================================================
# SS non-ws partial-K test (M=128: plain tcgen05.mma, standard TMEM layout)
# ===========================================================================
# Same N/K/dtype/swizzle as the ws SS test, but MMA_M=128 so the struct's
# auto-dispatch selects the NON-ws path: `bulk_mma` (full) and the new
# `bulk_mma_ss_partial` (partial).  The accumulator uses the standard
# (non-packed) TMEM layout: D row r lives in TMEM lane r, so warp w reads
# rows [32w, 32w+32) with NO column folding.
comptime NW_M = 128

comptime NW_A_BYTES = NW_M * QK_K * size_of[FP8_TYPE]()  # 73728
comptime NW_A_OFFSET = 0
comptime NW_B_OFFSET = NW_A_BYTES
comptime NW_META_OFFSET = NW_A_BYTES + QK_B_BYTES
comptime NW_TOTAL_SMEM = NW_META_OFFSET + 32

# SMEM operand layouts for the raw (non-struct) bulk_mma/bulk_mma_ss_partial
# calls; identical to SM100TensorAccumulator.a_layout/b_layout at M=128.
comptime NW_A_LAYOUT = tile_layout_k_major[FP8_TYPE, NW_M, QK_K, QK_SWIZZLE]()
comptime NW_B_LAYOUT = tile_layout_k_major[FP8_TYPE, QK_N, QK_K, QK_SWIZZLE]()


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def ss_nonws_partial_kernel[
    NUM_K_MMAS: Int,
    USE_PARTIAL: Bool,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    USE_STRUCT: Bool = False,
    NUM_STAGES: Int = 1,
](
    a_tma_op: TMATensorTile[FP8_TYPE, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[FP8_TYPE, b_tile_rank, b_tile_shape, b_desc_shape],
    c_output: LayoutTensor[
        ACC_TYPE, Layout.row_major(NW_M, QK_N), MutAnyOrigin
    ],
    valid_k_mmas: UInt32,
):
    """Non-ws SS MMA: C [128,64] = A [128,576] x B [64,576]^T (FP8 e4m3).

    Comptime-branched between the full-tile `bulk_mma` (USE_PARTIAL=False),
    the new `bulk_mma_ss_partial` (USE_PARTIAL=True), and the struct's
    non-ws dispatch (USE_STRUCT, with `mma` / `mma_maybe_partial_k` chosen
    by USE_PARTIAL).
    """

    # ---- Dynamic SMEM ----
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var a_smem_ptr = (smem_base + NW_A_OFFSET).bitcast[Scalar[FP8_TYPE]]()
    var b_smem_ptr = (smem_base + NW_B_OFFSET).bitcast[Scalar[FP8_TYPE]]()

    var a_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(NW_M, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](a_smem_ptr.as_unsafe_any_origin())
    var b_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(QK_N, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](b_smem_ptr.as_unsafe_any_origin())

    # ---- Metadata ----
    var metadata_ptr = (smem_base + NW_META_OFFSET).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    var a_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var b_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    if elect_one_thread:
        a_mbar[0].init()
        b_mbar[0].init()
        mma_mbar[0].init()

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    if elect_one_thread:
        a_mbar[0].expect_bytes(Int32(NW_A_BYTES))
        a_tma_op.async_copy(a_smem_tile, a_mbar[0], (0, 0))

        b_mbar[0].expect_bytes(Int32(QK_B_BYTES))
        b_tma_op.async_copy(b_smem_tile, b_mbar[0], (0, 0))
    barrier()

    a_mbar[0].wait()
    b_mbar[0].wait()
    barrier()

    # ---- SMEM descriptors (k-major for both A and B) ----
    var a_desc = smem_descriptor[
        BMN=NW_M,
        BK=QK_K,
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](a_smem_ptr)
    var b_desc = smem_descriptor[
        BMN=QK_N,
        BK=QK_K,
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](b_smem_ptr)

    comptime mma_idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        ACC_TYPE,
        FP8_TYPE,
        FP8_TYPE,
        Index[dtype=.uint32](NW_M, QK_N),
        transpose_b=True,
    ]()

    var c_tmem = tmem_addr

    var e: Int32 = 0
    if wid == 0:
        e = elect()

    comptime if USE_STRUCT:
        # Struct-driven dispatch: MMA_M=128 > 64, so the struct must select
        # the NON-ws path.
        comptime assert NUM_K_MMAS == QK_NUM_K_MMAS
        comptime UMMA = SM100TensorAccumulator[
            FP8_TYPE,
            ACC_TYPE,
            MMA_M=NW_M,
            MMA_N=QK_N,
            BK=QK_K,
            a_tmem=False,
            mma_kind=UMMAKind.KIND_F8F6F4,
            swizzle_a=QK_SWIZZLE,
            swizzle_b=QK_SWIZZLE,
            transpose_b=True,
            cta_group=1,
            num_stages=NUM_STAGES,
        ]
        comptime assert not UMMA.use_ws
        comptime for s in range(NUM_STAGES):
            comptime if USE_PARTIAL:
                UMMA.mma_maybe_partial_k[stage_idx=s](
                    a_desc,
                    b_desc,
                    c_tmem,
                    c_scale=UInt32(0),
                    elect=e,
                    valid_k_mmas=valid_k_mmas,
                )
            else:
                UMMA.mma[stage_idx=s](
                    a_desc, b_desc, c_tmem, c_scale=UInt32(0), elect=e
                )
    elif USE_PARTIAL:
        bulk_mma_ss_partial[
            NW_A_LAYOUT,
            NW_B_LAYOUT,
            num_k_mmas=NUM_K_MMAS,
            mma_k=QK_MMA_K,
            operand_size=size_of[FP8_TYPE](),
        ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e, valid_k_mmas)
    else:
        bulk_mma[
            NW_A_LAYOUT,
            NW_B_LAYOUT,
            num_k_mmas=NUM_K_MMAS,
            mma_k=QK_MMA_K,
            operand_size=size_of[FP8_TYPE](),
        ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    # ---- Read C from TMEM (M=128 non-ws cta_group::1: standard layout,
    # D row r in TMEM lane r; warp w implicitly reads its own subpartition
    # lanes [32w, 32w+32), all 64 columns). ----
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=QK_N,
        dtype=ACC_TYPE,
        pack=False,
        width=QK_N,
    ](c_tmem)

    tcgen05_load_wait()

    var lane_id = Int(tid % 32)
    var c_row = Int(wid) * 32 + lane_id
    for j in range(QK_N):
        c_output[c_row, j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


def test_ss_nonws_partial(ctx: DeviceContext) raises:
    print("=" * 70)
    print(
        "test_ss_nonws_partial: bulk_mma_ss_partial (SS non-ws)"
        " M=128 N=64 K=576 FP8 e4m3"
    )
    print("=" * 70)

    # =====================================================================
    # Clean data: v1 (raw full ground truth), v2 (raw partial degeneracy),
    # v3 (struct partial degeneracy), all vs naive K=576.
    # =====================================================================
    seed(42)

    var a_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(NW_M, QK_K)](ctx)
    var b_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_N, QK_K)](ctx)
    var a_host = a_inp.tensor[update=False]()
    var b_host = b_inp.tensor[update=False]()
    fill_random_fp8[FP8_TYPE](a_host.ptr, NW_M * QK_K)
    fill_random_fp8[FP8_TYPE](b_host.ptr, QK_N * QK_K)

    var a_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](NW_M * QK_K)
    var b_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_N * QK_K)
    var a_ref_host = alloc[Scalar[REF_TYPE]](NW_M * QK_K)
    var b_ref_host = alloc[Scalar[REF_TYPE]](QK_N * QK_K)
    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](a_host.ptr, a_ref_host, NW_M * QK_K)
    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](b_host.ptr, b_ref_host, QK_N * QK_K)
    ctx.enqueue_copy(a_ref_dev, a_ref_host)
    ctx.enqueue_copy(b_ref_dev, b_ref_host)

    var a_tma_op = create_tensor_tile[
        Index(NW_M, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, a_inp.device_tensor())
    var b_tma_op = create_tensor_tile[
        Index(QK_N, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, b_inp.device_tensor())

    var c_ref_full_dev = _ss_naive_ref[QK_K, M=NW_M](ctx, a_ref_dev, b_ref_dev)
    var c_ref_full = alloc[Float32](NW_M * QK_N)
    ctx.enqueue_copy(c_ref_full, c_ref_full_dev)
    ctx.synchronize()

    # ---- Variant 1: raw full[18] (bulk_mma, non-ws ground truth) ----
    var c_full_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NW_M, QK_N)
    ](ctx)
    comptime kern_full = ss_nonws_partial_kernel[
        QK_NUM_K_MMAS,
        False,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kern_full](
        a_tma_op,
        b_tma_op,
        c_full_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),  # ignored when USE_PARTIAL=False
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=NW_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(NW_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare[M=NW_M](
        "v1 full[18]",
        c_full_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 2: raw partial[18, valid=18] (degeneracy) ----
    var c_deg_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(NW_M, QK_N)](
        ctx
    )
    comptime kern_partial = ss_nonws_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kern_partial](
        a_tma_op,
        b_tma_op,
        c_deg_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),  # valid == num_k_mmas: no block skipped.
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=NW_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(NW_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare[M=NW_M](
        "v2 partial[18,valid=18]",
        c_deg_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    # ---- Variant 3: struct.mma_maybe_partial_k degeneracy (valid=18) ----
    var c_st_deg_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NW_M, QK_N)
    ](ctx)
    comptime kern_st_deg = ss_nonws_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        USE_STRUCT=True,
    ]
    ctx.enqueue_function[kern_st_deg](
        a_tma_op,
        b_tma_op,
        c_st_deg_buf.device_tensor(),
        UInt32(QK_NUM_K_MMAS),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=NW_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(NW_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare[M=NW_M](
        "v3 struct.partial[valid=18]",
        c_st_deg_buf.tensor().ptr,
        c_ref_full,
        require_finite=False,
    )

    a_ref_host.free()
    b_ref_host.free()
    c_ref_full.free()
    _ = a_ref_dev
    _ = b_ref_dev
    _ = c_ref_full_dev
    _ = a_inp^
    _ = b_inp^
    _ = c_full_buf^
    _ = c_deg_buf^
    _ = c_st_deg_buf^

    # =====================================================================
    # NaN-tail data: v4 (raw partial), v5 (struct partial), v6 (struct
    # partial, 3-stage), all valid=10 with A/B K-cols [320,576) = NaN,
    # vs naive K=320; every output must be finite.
    # =====================================================================
    seed(123)

    var a_nan = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(NW_M, QK_K)](ctx)
    var b_nan = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_N, QK_K)](ctx)
    var a_nan_host = a_nan.tensor[update=False]()
    var b_nan_host = b_nan.tensor[update=False]()
    fill_random_fp8[FP8_TYPE](a_nan_host.ptr, NW_M * QK_K)
    fill_random_fp8[FP8_TYPE](b_nan_host.ptr, QK_N * QK_K)

    var nan_byte = fp8_nan()
    for r in range(NW_M):
        for c in range(QK_VALID_K, QK_K):
            a_nan_host.ptr[r * QK_K + c] = nan_byte
    for r in range(QK_N):
        for c in range(QK_VALID_K, QK_K):
            b_nan_host.ptr[r * QK_K + c] = nan_byte

    var a_nan_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](NW_M * QK_K)
    var b_nan_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_N * QK_K)
    var a_nan_ref_host = alloc[Scalar[REF_TYPE]](NW_M * QK_K)
    var b_nan_ref_host = alloc[Scalar[REF_TYPE]](QK_N * QK_K)
    for r in range(NW_M):
        for c in range(QK_K):
            var v = a_nan_host.ptr[r * QK_K + c].cast[
                REF_TYPE
            ]() if c < QK_VALID_K else Scalar[REF_TYPE](0)
            a_nan_ref_host[r * QK_K + c] = v
    for r in range(QK_N):
        for c in range(QK_K):
            var v = b_nan_host.ptr[r * QK_K + c].cast[
                REF_TYPE
            ]() if c < QK_VALID_K else Scalar[REF_TYPE](0)
            b_nan_ref_host[r * QK_K + c] = v
    ctx.enqueue_copy(a_nan_ref_dev, a_nan_ref_host)
    ctx.enqueue_copy(b_nan_ref_dev, b_nan_ref_host)

    var a_nan_tma = create_tensor_tile[
        Index(NW_M, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, a_nan.device_tensor())
    var b_nan_tma = create_tensor_tile[
        Index(QK_N, QK_K), swizzle_mode=QK_SWIZZLE
    ](ctx, b_nan.device_tensor())

    var c_ref_v_dev = _ss_naive_ref[QK_VALID_K, M=NW_M](
        ctx, a_nan_ref_dev, b_nan_ref_dev
    )
    var c_ref_v = alloc[Float32](NW_M * QK_N)
    ctx.enqueue_copy(c_ref_v, c_ref_v_dev)
    ctx.synchronize()

    # ---- Variant 4: raw bulk_mma_ss_partial[valid=10]+NaN ----
    var c_nan_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(NW_M, QK_N)](
        ctx
    )
    comptime kern_nan = ss_nonws_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_nan_tma).rank,
        type_of(a_nan_tma).tile_shape,
        type_of(a_nan_tma).desc_shape,
        type_of(b_nan_tma).rank,
        type_of(b_nan_tma).tile_shape,
        type_of(b_nan_tma).desc_shape,
    ]
    ctx.enqueue_function[kern_nan](
        a_nan_tma,
        b_nan_tma,
        c_nan_buf.device_tensor(),
        UInt32(QK_VALID),  # valid=10: blocks [10,18) must be skipped.
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=NW_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(NW_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare[M=NW_M](
        "v4 partial[18,valid=10]+NaN",
        c_nan_buf.tensor().ptr,
        c_ref_v,
        require_finite=True,
    )

    # ---- Variant 5: struct.mma_maybe_partial_k[valid=10]+NaN ----
    var c_st_nan_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NW_M, QK_N)
    ](ctx)
    comptime kern_st_nan = ss_nonws_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_nan_tma).rank,
        type_of(a_nan_tma).tile_shape,
        type_of(a_nan_tma).desc_shape,
        type_of(b_nan_tma).rank,
        type_of(b_nan_tma).tile_shape,
        type_of(b_nan_tma).desc_shape,
        USE_STRUCT=True,
    ]
    ctx.enqueue_function[kern_st_nan](
        a_nan_tma,
        b_nan_tma,
        c_st_nan_buf.device_tensor(),
        UInt32(QK_VALID),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=NW_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(NW_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare[M=NW_M](
        "v5 struct.partial[valid=10]+NaN",
        c_st_nan_buf.tensor().ptr,
        c_ref_v,
        require_finite=True,
    )

    # ---- Variant 6: struct.mma_maybe_partial_k num_stages=3, valid=10 ----
    # Stage block ranges [0,6) / [6,12) / [12,18); valid=10 lands INSIDE
    # stage 1 and stage 2 issues nothing -- exercising bulk_mma_ss_partial's
    # k_start > 0 path through the struct's per-stage split.
    var c_st_nan_ms_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(NW_M, QK_N)
    ](ctx)
    comptime kern_st_nan_ms = ss_nonws_partial_kernel[
        QK_NUM_K_MMAS,
        True,
        type_of(a_nan_tma).rank,
        type_of(a_nan_tma).tile_shape,
        type_of(a_nan_tma).desc_shape,
        type_of(b_nan_tma).rank,
        type_of(b_nan_tma).tile_shape,
        type_of(b_nan_tma).desc_shape,
        USE_STRUCT=True,
        NUM_STAGES=3,
    ]
    ctx.enqueue_function[kern_st_nan_ms](
        a_nan_tma,
        b_nan_tma,
        c_st_nan_ms_buf.device_tensor(),
        UInt32(QK_VALID),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=NW_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(NW_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()
    _ss_compare[M=NW_M](
        "v6 struct.partial 3-stage[valid=10]+NaN",
        c_st_nan_ms_buf.tensor().ptr,
        c_ref_v,
        require_finite=True,
    )

    print(
        "  SS non-ws partial PASSED"
        " (v1 ground / v2 degeneracy / v3 struct-degeneracy / v4 NaN"
        " / v5 struct-NaN / v6 struct-3stage-NaN)"
    )

    a_nan_ref_host.free()
    b_nan_ref_host.free()
    c_ref_v.free()
    _ = a_nan_ref_dev
    _ = b_nan_ref_dev
    _ = c_ref_v_dev
    _ = a_nan^
    _ = b_nan^
    _ = c_nan_buf^
    _ = c_st_nan_buf^
    _ = c_st_nan_ms_buf^


# ===========================================================================
# TS partial-K test (clone of dense_mma_ws_ts_kernel)
# ===========================================================================
comptime TS_ROWS = 64
comptime TS_COLS = 512
comptime TS_OP_TYPE = DType.bfloat16

comptime TS_K_ROWS = 64
comptime TS_K_COLS = 512

comptime TS_SW_BYTES = 128
comptime TS_SW_K = TS_SW_BYTES // size_of[TS_OP_TYPE]()  # 64 BF16 per group

comptime TS_BF16_PER_STRIP = 16
comptime TS_U32_PER_STRIP = TS_BF16_PER_STRIP // 2  # 8
comptime TS_STRIPS_PER_PAIR = TS_SW_K // TS_BF16_PER_STRIP  # 4
comptime TS_NUM_GROUP_PAIRS = (TS_COLS // TS_SW_K) // 2  # 4
comptime TS_NUM_STRIPS = TS_NUM_GROUP_PAIRS * TS_STRIPS_PER_PAIR  # 16

comptime TS_Q_SMEM_LAYOUT = tile_layout_k_major[
    TS_OP_TYPE, TS_ROWS, TS_COLS, TensorMapSwizzle.SWIZZLE_128B
]()
comptime TS_Q_SMEM_BYTES = TS_Q_SMEM_LAYOUT.size() * size_of[TS_OP_TYPE]()

comptime TS_K_SMEM_LAYOUT = tile_layout_k_major[
    TS_OP_TYPE, TS_K_ROWS, TS_K_COLS, TensorMapSwizzle.SWIZZLE_128B
]()
comptime TS_K_SMEM_BYTES = TS_K_SMEM_LAYOUT.size() * size_of[TS_OP_TYPE]()

comptime TS_CANONICAL_LAYOUT = tile_to_descriptor[
    TS_OP_TYPE, TS_Q_SMEM_LAYOUT, is_k_major=True
]()
comptime TS_STRIDE_01 = TS_CANONICAL_LAYOUT[0].stride[1].value()
comptime TS_STRIDE_11 = TS_CANONICAL_LAYOUT[1].stride[1].value()
comptime TS_SBO = TS_STRIDE_01 * size_of[TS_OP_TYPE]()
comptime TS_LBO = TS_STRIDE_11 * size_of[TS_OP_TYPE]()

comptime TS_TOTAL_TMEM_COLS = TS_NUM_STRIPS * TS_U32_PER_STRIP  # 128
comptime TS_MAX_TMEM_COLS: UInt32 = 512

comptime TS_Q_TMA_EXPECTED_BYTES = TS_Q_SMEM_BYTES
comptime TS_K_TMA_EXPECTED_BYTES = TS_K_SMEM_BYTES

# MMA TS .ws (dual GEMM fold).
comptime TS_MMA_M = 64
comptime TS_MMA_N = 128
comptime TS_MMA_K = 16
comptime TS_ACCUM_TYPE = DType.float32

comptime TS_FOLDED_B_ROWS = 128  # N dimension of B (transpose_b)
comptime TS_FOLDED_B_COLS = 256  # K dimension of B
comptime TS_NUM_K_MMAS = TS_FOLDED_B_COLS // TS_MMA_K  # 16

comptime TS_A_TMEM_OFFSET: UInt32 = 0
comptime TS_C_TMEM_OFFSET: UInt32 = UInt32(TS_TOTAL_TMEM_COLS)  # 128
comptime TS_HALF_N = TS_MMA_N // 2  # 64

comptime TS_P_ROWS = TS_MMA_M  # 64
comptime TS_P_COLS = TS_MMA_N  # 128
comptime TS_P_REF_ROWS = 64
comptime TS_P_REF_COLS = 64

comptime TS_K_SMEM_OFFSET = TS_Q_SMEM_BYTES
comptime TS_METADATA_OFFSET = TS_K_SMEM_OFFSET + TS_K_SMEM_BYTES
comptime TS_TOTAL_SMEM_BYTES = TS_METADATA_OFFSET + 32

comptime TS_VALID = 10  # loaded folded MMA_K blocks (variant 3).


# ---------------------------------------------------------------------------
# Kernel: TS partial -- generic over NUM_K_MMAS_MMA and USE_PARTIAL.
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
def ts_partial_kernel[
    NUM_K_MMAS_MMA: Int,
    USE_PARTIAL: Bool,
    q_tile_rank: Int,
    q_tile_shape: IndexList[q_tile_rank],
    q_desc_shape: IndexList[q_tile_rank],
    k_tile_rank: Int,
    k_tile_shape: IndexList[k_tile_rank],
    k_desc_shape: IndexList[k_tile_rank],
    USE_STRUCT: Bool = False,
    NUM_STAGES: Int = 1,
](
    q_tma_op: TMATensorTile[
        TS_OP_TYPE, q_tile_rank, q_tile_shape, q_desc_shape
    ],
    k_tma_op: TMATensorTile[
        TS_OP_TYPE, k_tile_rank, k_tile_shape, k_desc_shape
    ],
    p_output: LayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS), MutAnyOrigin
    ],
    valid_k_mmas: UInt32,
):
    """TS (.ws) MMA: P = Q x K^T via dual-GEMM fold; raw output [64,128].

    The MMA call is comptime-branched between the full-tile `bulk_mma_ws_ts`
    (USE_PARTIAL=False, ignores `valid_k_mmas`) and `bulk_mma_ws_ts_partial`
    (USE_PARTIAL=True, passes `valid_k_mmas`).  All TMA loads + tcgen05_cp
    (16 strips) + fold B descriptor (BK=256) are identical to
    `dense_mma_ws_ts_kernel`; only the MMA call differs.
    """
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()

    var q_smem_ptr = smem_base.bitcast[Scalar[TS_OP_TYPE]]()
    var q_smem_tile = LayoutTensor[
        TS_OP_TYPE,
        TS_Q_SMEM_LAYOUT,
        address_space=.SHARED,
        alignment=128,
    ](q_smem_ptr.as_unsafe_any_origin())

    var k_smem_ptr = (smem_base + TS_K_SMEM_OFFSET).bitcast[
        Scalar[TS_OP_TYPE]
    ]()
    var k_smem_tile = LayoutTensor[
        TS_OP_TYPE,
        TS_K_SMEM_LAYOUT,
        address_space=.SHARED,
        alignment=128,
    ](k_smem_ptr.as_unsafe_any_origin())

    var metadata_ptr = (smem_base + TS_METADATA_OFFSET).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    var q_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var k_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    if elect_one_thread:
        q_mbar[0].init()
        k_mbar[0].init()
        mma_mbar[0].init()

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, TS_MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    if elect_one_thread:
        q_mbar[0].expect_bytes(Int32(TS_Q_TMA_EXPECTED_BYTES))
        q_tma_op.async_copy(q_smem_tile, q_mbar[0], (0, 0))

        k_mbar[0].expect_bytes(Int32(TS_K_TMA_EXPECTED_BYTES))
        k_tma_op.async_copy(k_smem_tile, k_mbar[0], (0, 0))
    barrier()

    q_mbar[0].wait()

    # ---- tcgen05_cp: Q swizzled SMEM -> TMEM (16 strips) ----
    var base_desc = MMASmemDescriptor.create[
        TS_SBO, TS_LBO, TensorMapSwizzle.SWIZZLE_128B
    ](q_smem_ptr)

    comptime for s in range(TS_NUM_STRIPS):
        comptime s_pair = s // TS_STRIPS_PER_PAIR
        comptime s_local = s % TS_STRIPS_PER_PAIR
        comptime strip_byte_offset = (
            s_pair * 2 * TS_ROWS * TS_SW_K + s_local * TS_BF16_PER_STRIP
        ) * size_of[TS_OP_TYPE]()

        var s_desc = base_desc + strip_byte_offset

        tcgen05_cp[cta_group=1, datapaths=128, bits=256](
            tmem_addr + UInt32(s * TS_U32_PER_STRIP), s_desc
        )

    barrier()

    k_mbar[0].wait()
    barrier()

    # ---- Folded B descriptor for K: [64,512] -> [128,256] ----
    var b_desc = smem_descriptor[
        BMN=TS_FOLDED_B_ROWS,
        BK=TS_FOLDED_B_COLS,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        is_k_major=True,
    ](k_smem_ptr)

    comptime MMA_IDESC = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        TS_ACCUM_TYPE,
        TS_OP_TYPE,
        TS_OP_TYPE,
        Index[dtype=.uint32](TS_MMA_M, TS_MMA_N),
        transpose_b=True,
    ]()

    var a_tmem = tmem_addr + TS_A_TMEM_OFFSET
    var c_tmem = tmem_addr + TS_C_TMEM_OFFSET

    var e: Int32 = 0
    if wid == 0:
        e = elect()

    comptime if USE_STRUCT:
        # Struct-driven dispatch: SM100TensorAccumulator auto-selects the
        # ws path (cta_group=1, MMA_M=64 <= 64).  Always contracts the full
        # folded BK=256, so it is only valid at NUM_K_MMAS_MMA == 16.
        # USE_PARTIAL routes through mma_maybe_partial_k (the ws-partial
        # arm); NUM_STAGES=2 exercises the multi-stage ws arm (3+1 split).
        comptime assert NUM_K_MMAS_MMA == TS_NUM_K_MMAS
        comptime UMMA = SM100TensorAccumulator[
            TS_OP_TYPE,
            TS_ACCUM_TYPE,
            MMA_M=TS_MMA_M,
            MMA_N=TS_MMA_N,
            BK=TS_FOLDED_B_COLS,
            a_tmem=True,
            swizzle_b=TensorMapSwizzle.SWIZZLE_128B,
            mma_kind=UMMAKind.KIND_F16,
            transpose_b=True,
            cta_group=1,
            num_stages=NUM_STAGES,
        ]
        comptime assert UMMA.use_ws
        comptime if USE_PARTIAL:
            comptime for s in range(NUM_STAGES):
                UMMA.mma_maybe_partial_k[stage_idx=s](
                    a_tmem,
                    b_desc,
                    c_tmem,
                    c_scale=UInt32(0),
                    elect=e,
                    valid_k_mmas=valid_k_mmas,
                )
        else:
            comptime for s in range(NUM_STAGES):
                UMMA.mma[stage_idx=s](
                    a_tmem, b_desc, c_tmem, c_scale=UInt32(0), elect=e
                )
    elif USE_PARTIAL:
        bulk_mma_ws_ts_partial[
            UMMAKind.KIND_F16,
            TS_OP_TYPE,
            b_BMN=TS_FOLDED_B_ROWS,
            b_BK=TS_FOLDED_B_COLS,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_is_k_major=True,
            num_k_mmas=NUM_K_MMAS_MMA,
            operand_size=size_of[TS_OP_TYPE](),
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](MMA_IDESC, a_tmem, b_desc, c_tmem, UInt32(0), e, valid_k_mmas)
    else:
        bulk_mma_ws_ts[
            UMMAKind.KIND_F16,
            TS_OP_TYPE,
            b_BMN=TS_FOLDED_B_ROWS,
            b_BK=TS_FOLDED_B_COLS,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_is_k_major=True,
            num_k_mmas=NUM_K_MMAS_MMA,
            operand_size=size_of[TS_OP_TYPE](),
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](MMA_IDESC, a_tmem, b_desc, c_tmem, UInt32(0), e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=TS_HALF_N,
        dtype=TS_ACCUM_TYPE,
        pack=False,
        width=TS_HALF_N,
    ](c_tmem)

    tcgen05_load_wait()

    var p_row = tid % TS_MMA_M  # 0-63
    var p_dp_half = tid // TS_MMA_M  # 0 or 1
    var p_col_base = p_dp_half * TS_HALF_N  # 0 or 64

    for j in range(TS_HALF_N):
        p_output[p_row, p_col_base + j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, TS_MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# TS helpers
# ---------------------------------------------------------------------------
def _ts_launch[
    NUM_K_MMAS_MMA: Int,
    USE_PARTIAL: Bool,
    USE_STRUCT: Bool = False,
    NUM_STAGES: Int = 1,
](
    ctx: DeviceContext,
    mut q_inp: ManagedLayoutTensor[
        TS_OP_TYPE, Layout.row_major(TS_ROWS, TS_COLS)
    ],
    mut k_inp: ManagedLayoutTensor[
        TS_OP_TYPE, Layout.row_major(TS_K_ROWS, TS_K_COLS)
    ],
    valid: UInt32,
    mut p_out_buf: ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ],
) raises:
    """Runs one TS kernel instantiation into `p_out_buf`."""
    var q_tma_op = create_tensor_tile[
        Index(TS_ROWS, TS_COLS),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](ctx, q_inp.device_tensor())
    var k_tma_op = create_tensor_tile[
        Index(TS_K_ROWS, TS_K_COLS),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](ctx, k_inp.device_tensor())

    comptime kernel = ts_partial_kernel[
        NUM_K_MMAS_MMA,
        USE_PARTIAL,
        type_of(q_tma_op).rank,
        type_of(q_tma_op).tile_shape,
        type_of(q_tma_op).desc_shape,
        type_of(k_tma_op).rank,
        type_of(k_tma_op).tile_shape,
        type_of(k_tma_op).desc_shape,
        USE_STRUCT=USE_STRUCT,
        NUM_STAGES=NUM_STAGES,
    ]
    ctx.enqueue_function[kernel](
        q_tma_op,
        k_tma_op,
        p_out_buf.device_tensor(),
        valid,
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=TS_TOTAL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(TS_TOTAL_SMEM_BYTES)
        ),
    )
    ctx.synchronize()


def _ts_foldsum_compare(
    label: String,
    p_ptr: ImmPointer[Scalar[TS_ACCUM_TYPE], _],
    ref_ptr: ImmPointer[Float32, _],
) raises:
    """Fold-sum comparison: P_gpu[:, c] + P_gpu[:, c+64] vs P_ref[64,64]."""
    var max_err: Float32 = 0.0
    for r in range(TS_P_REF_ROWS):
        for c in range(TS_P_REF_COLS):
            var ref_val = ref_ptr[r * TS_P_REF_COLS + c]
            var gpu_val = Float32(p_ptr[r * TS_P_COLS + c]) + Float32(
                p_ptr[r * TS_P_COLS + c + TS_P_ROWS]
            )
            var err = abs(gpu_val - ref_val) / max(abs(ref_val), Float32(1.0))
            if err > max_err:
                max_err = err
            assert_almost_equal(gpu_val, ref_val, atol=1.0, rtol=0.01)
    print("  [" + label + "] fold-sum max rel err=" + String(max_err))


def test_ts_partial(ctx: DeviceContext) raises:
    print("=" * 70)
    print(
        "test_ts_partial: bulk_mma_ws_ts_partial (TS .ws)"
        " Q[64,512] x K[64,512]^T BF16, dual fold MMA_N=128"
    )
    print("=" * 70)

    # ---- Random Q, K shared across all TS variants ----
    seed(42)
    var q_inp = ManagedLayoutTensor[
        TS_OP_TYPE, Layout.row_major(TS_ROWS, TS_COLS)
    ](ctx)
    var q_inp_host = q_inp.tensor[update=False]()
    randn[TS_OP_TYPE](q_inp_host.ptr, TS_ROWS * TS_COLS)

    var k_inp = ManagedLayoutTensor[
        TS_OP_TYPE, Layout.row_major(TS_K_ROWS, TS_K_COLS)
    ](ctx)
    var k_inp_host = k_inp.tensor[update=False]()
    randn[TS_OP_TYPE](k_inp_host.ptr, TS_K_ROWS * TS_K_COLS)

    # =====================================================================
    # Variant 1 (ground truth): full[16] vs fold-sum naive (== existing
    # test_dense_mma_ws_ts).
    # =====================================================================
    var p_full_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, False](
        ctx, q_inp, k_inp, UInt32(TS_NUM_K_MMAS), p_full_buf
    )

    # Fold-sum naive reference: P_ref[64,64] = Q x K^T (transpose_b=True).
    var p_ref_dev = ctx.enqueue_create_buffer[.float32](
        TS_P_REF_ROWS * TS_P_REF_COLS
    )
    var q_dev_ptr = q_inp.device_data.value().unsafe_ptr()
    var k_dev_ptr = k_inp.device_data.value().unsafe_ptr()
    var c_ref_tt = TileTensor(
        p_ref_dev, row_major(Coord(TS_P_REF_ROWS, TS_P_REF_COLS))
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[TS_OP_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(q_dev_ptr)
        ),
        row_major(Coord(TS_ROWS, TS_COLS)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[TS_OP_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(k_dev_ptr)
        ),
        row_major(Coord(TS_K_ROWS, TS_K_COLS)),
    )
    comptime gemm_naive = matmul_kernel_naive[
        .float32,
        TS_OP_TYPE,
        TS_OP_TYPE,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        NAIVE_BLOCK_DIM,
        transpose_b=True,
    ]
    ctx.enqueue_function[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(TS_P_REF_ROWS),
        Int32(TS_P_REF_COLS),
        Int32(TS_COLS),
        grid_dim=(
            ceildiv(TS_P_REF_ROWS, NAIVE_BLOCK_DIM),
            ceildiv(TS_P_REF_COLS, NAIVE_BLOCK_DIM),
            1,
        ),
        block_dim=(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM, 1),
    )
    var p_ref_host = ctx.enqueue_create_host_buffer[.float32](
        TS_P_REF_ROWS * TS_P_REF_COLS
    )
    ctx.enqueue_copy(p_ref_host, p_ref_dev)
    ctx.synchronize()

    var p_full_ptr = p_full_buf.tensor().ptr
    var max_err_full: Float32 = 0.0
    for r in range(TS_P_REF_ROWS):
        for c in range(TS_P_REF_COLS):
            var ref_val = p_ref_host[r * TS_P_REF_COLS + c]
            var gpu_val = Float32(p_full_ptr[r * TS_P_COLS + c]) + Float32(
                p_full_ptr[r * TS_P_COLS + c + TS_P_ROWS]
            )
            var err = abs(gpu_val - ref_val) / max(abs(ref_val), Float32(1.0))
            if err > max_err_full:
                max_err_full = err
            assert_almost_equal(gpu_val, ref_val, atol=1.0, rtol=0.01)
    print("  [v1 full[16]] fold-sum max rel err=" + String(max_err_full))

    # =====================================================================
    # Variant 2 (degeneracy): partial[16, valid=16] vs SAME fold-sum naive.
    # =====================================================================
    var p_deg_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, True](
        ctx, q_inp, k_inp, UInt32(TS_NUM_K_MMAS), p_deg_buf
    )
    var p_deg_ptr = p_deg_buf.tensor().ptr
    var max_err_deg: Float32 = 0.0
    for r in range(TS_P_REF_ROWS):
        for c in range(TS_P_REF_COLS):
            var ref_val = p_ref_host[r * TS_P_REF_COLS + c]
            var gpu_val = Float32(p_deg_ptr[r * TS_P_COLS + c]) + Float32(
                p_deg_ptr[r * TS_P_COLS + c + TS_P_ROWS]
            )
            var err = abs(gpu_val - ref_val) / max(abs(ref_val), Float32(1.0))
            if err > max_err_deg:
                max_err_deg = err
            assert_almost_equal(gpu_val, ref_val, atol=1.0, rtol=0.01)
    print(
        "  [v2 partial[16,valid=16]] fold-sum max rel err="
        + String(max_err_deg)
    )

    # =====================================================================
    # Variant 3 (skip correctness, fold-agnostic GPU-vs-GPU):
    # partial[16, valid=10] vs the oracle full[10] on the SAME Q/K.
    # Both issue the same 10 folded MMA blocks, so the RAW [64,128] outputs
    # must be element-wise identical -- no fold-sum, no fold mapping needed.
    # This proves the partial path sums exactly [0,10) and skips [10,16).
    #
    # NaN-safety for TS is intentionally NOT tested here: the `@!%pv` skip
    # guard is byte-identical PTX in build_mma_ss_ws_partial and
    # build_mma_ts_ws_partial (both emit `setp.le.u32 %pv,...` + `@!%pv`), so
    # the SS NaN variant already proves the guard never reads skipped blocks'
    # inputs, and this GPU-vs-GPU test proves the TS operand/offset wiring
    # contracts the correct block range.  Reasoning about the dual fold's
    # NaN col mapping (c%256)//16 would add risk without adding coverage.
    # =====================================================================
    var p_partial_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, True](
        ctx, q_inp, k_inp, UInt32(TS_VALID), p_partial_buf
    )

    var p_oracle_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    # Oracle: full wrapper compiled at num_k_mmas=TS_VALID (issues blocks
    # [0,TS_VALID) unconditionally). valid arg is ignored (USE_PARTIAL=False).
    _ts_launch[TS_VALID, False](
        ctx, q_inp, k_inp, UInt32(TS_VALID), p_oracle_buf
    )

    var p_partial_ptr = p_partial_buf.tensor().ptr
    var p_oracle_ptr = p_oracle_buf.tensor().ptr
    var max_diff: Float32 = 0.0
    var num_nonfinite: Int = 0
    for r in range(TS_P_ROWS):
        for c in range(TS_P_COLS):
            var idx = r * TS_P_COLS + c
            var pv = p_partial_ptr[idx]
            var ov = p_oracle_ptr[idx]
            if not isfinite(pv):
                num_nonfinite += 1
            var d = abs(pv - ov)
            if d > max_diff:
                max_diff = d
    print(
        "  [v3 partial[16,valid=10] vs full[10]] raw max |diff|="
        + String(max_diff)
        + " nonfinite="
        + String(num_nonfinite)
    )
    assert_true(
        num_nonfinite == 0,
        msg="TS v3: partial raw output has non-finite elements",
    )
    # Both paths issue the SAME 10 MMA blocks in the same order, so the raw
    # accumulators should be bitwise-identical; allow a hair for any reorder.
    for r in range(TS_P_ROWS):
        for c in range(TS_P_COLS):
            var idx = r * TS_P_COLS + c
            assert_almost_equal(
                p_partial_ptr[idx],
                p_oracle_ptr[idx],
                atol=1e-3,
                rtol=1e-3,
                msg="TS v3 raw mismatch at r=" + String(r) + " c=" + String(c),
            )

    # =====================================================================
    # Variant 4: SM100TensorAccumulator.mma (struct ws dispatch) full
    # contraction vs the SAME fold-sum naive reference.  The struct
    # auto-selects the ws path (cta_group=1, MMA_M=64).
    # =====================================================================
    var p_st_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, False, USE_STRUCT=True](
        ctx, q_inp, k_inp, UInt32(TS_NUM_K_MMAS), p_st_buf
    )
    _ts_foldsum_compare(
        "v4 struct.mma", p_st_buf.tensor().ptr, p_ref_host.unsafe_ptr()
    )

    # =====================================================================
    # Variant 5: struct multi-stage (num_stages=2 -> 3+1 split: stage0 owns
    # k-blocks [0,12), stage1 [12,16)) -- exercises the ws multi-stage arm
    # (offset A TMEM base + offset B descriptor).
    # =====================================================================
    var p_st_ms_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, False, USE_STRUCT=True, NUM_STAGES=2](
        ctx, q_inp, k_inp, UInt32(TS_NUM_K_MMAS), p_st_ms_buf
    )
    _ts_foldsum_compare(
        "v5 struct.mma num_stages=2",
        p_st_ms_buf.tensor().ptr,
        p_ref_host.unsafe_ptr(),
    )

    # =====================================================================
    # Variant 6: struct.mma_maybe_partial_k degeneracy (valid=16) -- the
    # ws-partial arm with no block skipped must match the naive reference.
    # =====================================================================
    var p_st_deg_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, True, USE_STRUCT=True](
        ctx, q_inp, k_inp, UInt32(TS_NUM_K_MMAS), p_st_deg_buf
    )
    _ts_foldsum_compare(
        "v6 struct.partial[valid=16]",
        p_st_deg_buf.tensor().ptr,
        p_ref_host.unsafe_ptr(),
    )

    # =====================================================================
    # Variant 7: struct.mma_maybe_partial_k skip-correctness (valid=10) vs
    # the SAME full[10] oracle as v3 -- raw [64,128] outputs element-wise
    # equal, proving the struct routes the absolute-k partial wiring.
    # =====================================================================
    var p_st_part_buf = ManagedLayoutTensor[
        TS_ACCUM_TYPE, Layout.row_major(TS_P_ROWS, TS_P_COLS)
    ](ctx)
    _ts_launch[TS_NUM_K_MMAS, True, USE_STRUCT=True](
        ctx, q_inp, k_inp, UInt32(TS_VALID), p_st_part_buf
    )
    var p_st_part_ptr = p_st_part_buf.tensor().ptr
    for r in range(TS_P_ROWS):
        for c in range(TS_P_COLS):
            var idx = r * TS_P_COLS + c
            assert_true(
                isfinite(p_st_part_ptr[idx]),
                msg="TS v7: struct partial output non-finite at r="
                + String(r)
                + " c="
                + String(c),
            )
            assert_almost_equal(
                p_st_part_ptr[idx],
                p_oracle_ptr[idx],
                atol=1e-3,
                rtol=1e-3,
                msg="TS v7 raw mismatch at r=" + String(r) + " c=" + String(c),
            )
    print("  [v7 struct.partial[valid=10] vs full[10]] raw outputs match")

    print(
        "  TS partial PASSED (v1 ground / v2 degeneracy / v3 skip-correctness"
        " / v4 struct / v5 struct-multistage / v6 struct-degeneracy"
        " / v7 struct-skip)"
    )

    _ = p_ref_dev
    _ = q_inp^
    _ = k_inp^
    _ = p_full_buf^
    _ = p_deg_buf^
    _ = p_partial_buf^
    _ = p_oracle_buf^
    _ = p_st_buf^
    _ = p_st_ms_buf^
    _ = p_st_deg_buf^
    _ = p_st_part_buf^


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() raises:
    with DeviceContext() as ctx:
        comptime if not _is_sm10x_gpu(ctx.default_device_info):
            print("Skipping: this test requires B200 (SM100)")
            return

        test_ss_partial(ctx)
        test_ss_nonws_partial(ctx)
        test_ts_partial(ctx)
        print("\nAll partial-K (.ws) MMA tests PASSED.")
