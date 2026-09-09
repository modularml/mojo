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

"""Test TMA -> SMEM -> TMEM -> MMA TS .ws pipeline for P = Q x K^T.

Loads a 64x512 BF16 Q matrix into SMEM via TMA (k-major, SWIZZLE_128B),
then copies it to TMEM via tcgen05_cp.  Loads a 64x512 BF16 K matrix
into a separate SMEM region via TMA (same layout).

The MMA computation uses tcgen05.mma.ws.cta_group::1.kind::f16 in TS
mode (A from TMEM, B from SMEM).  The K [64,512] SMEM data is
reinterpreted as [128,256] via the dual GEMM fold trick.  The MMA output
is [64,128] in TMEM: columns 0-63 = partial Q_even * K_even^T, columns
64-127 = partial Q_odd * K_odd^T.  Summing these two halves gives the
full 64x64 Q x K^T result, which is verified against a GPU naive matmul
reference (matmul_kernel_naive with transpose_b=True).

The fact that P = Q x K^T passes verification proves that both Q and K
were loaded correctly (Q through TMA -> SMEM -> TMEM, K through TMA ->
SMEM), so no separate readback verification is needed for either.
"""

from std.math import ceildiv
from std.memory import alloc
from std.random import rand, randn, seed
from std.sys import size_of

from max.gpu import thread_idx, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import (
    TensorMapSwizzle,
    prefetch_tma_descriptor,
)
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
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import tile_layout_k_major, tile_to_descriptor
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
    create_tma_tile_gather4,
)
from linalg.arch.sm100.mma import smem_descriptor
from linalg.matmul.gpu import matmul_kernel_naive
from nn.attention.gpu.nvidia.sm100.attention_utils import bulk_mma_ws_ts, elect
from std.testing import assert_almost_equal
from std.utils.index import Index, IndexList

# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------
comptime ROWS = 64
comptime COLS = 512
comptime OP_TYPE = DType.bfloat16

comptime NUM_THREADS = 128  # One warpgroup (4 warps)

# K matrix dimensions (same shape as Q for this test).
comptime K_ROWS = 64
comptime K_COLS = 512

# Swizzle group parameters.
# SWIZZLE_128B => 128 bytes per SMEM row => 64 BF16 per row.
comptime SW_BYTES = 128
comptime SW_K = SW_BYTES // size_of[OP_TYPE]()  # 64 BF16 per swizzle group

# Strip parameters: each tcgen05_cp[128x256b] handles 256 bits = 16 BF16
# = 8 uint32 per SMEM row.  4 strips exhaust one 128-byte row.
# With datapaths=128, each call reads 2 swizzle groups, so we process
# 4 group-pairs total => 4 pairs * 4 strips/pair = 16 strips.
comptime BF16_PER_STRIP = 16
comptime U32_PER_STRIP = BF16_PER_STRIP // 2  # 8
comptime STRIPS_PER_PAIR = SW_K // BF16_PER_STRIP  # 4
comptime NUM_GROUP_PAIRS = (COLS // SW_K) // 2  # 4
comptime NUM_STRIPS = NUM_GROUP_PAIRS * STRIPS_PER_PAIR  # 16

# Q SMEM layout: k-major 64x512 BF16 with SWIZZLE_128B.
comptime Q_SMEM_LAYOUT = tile_layout_k_major[
    OP_TYPE, ROWS, COLS, TensorMapSwizzle.SWIZZLE_128B
]()
comptime Q_SMEM_BYTES = Q_SMEM_LAYOUT.size() * size_of[OP_TYPE]()

# K SMEM layout: same k-major 64x512 BF16 with SWIZZLE_128B.
comptime K_SMEM_LAYOUT = tile_layout_k_major[
    OP_TYPE, K_ROWS, K_COLS, TensorMapSwizzle.SWIZZLE_128B
]()
comptime K_SMEM_BYTES = K_SMEM_LAYOUT.size() * size_of[OP_TYPE]()

# Derive SBO/LBO from the canonical layout (same pattern as
# test_tma_mma_sm100.mojo and production matmul code).
comptime CANONICAL_LAYOUT = tile_to_descriptor[
    OP_TYPE, Q_SMEM_LAYOUT, is_k_major=True
]()
comptime STRIDE_01 = CANONICAL_LAYOUT[0].stride[1].value()
comptime STRIDE_11 = CANONICAL_LAYOUT[1].stride[1].value()
# For k-major: SBO = stride01 * sizeof, LBO = stride11 * sizeof.
comptime SBO = STRIDE_01 * size_of[OP_TYPE]()
comptime LBO = STRIDE_11 * size_of[OP_TYPE]()

# Total TMEM columns: 16 strips * 8 cols/strip = 128.
comptime TOTAL_TMEM_COLS = NUM_STRIPS * U32_PER_STRIP  # 128
comptime MAX_TMEM_COLS: UInt32 = 512

# TMA expected bytes for Q = full tile = 65536 bytes.
comptime Q_TMA_EXPECTED_BYTES = Q_SMEM_BYTES
# TMA expected bytes for K = full tile = 65536 bytes.
comptime K_TMA_EXPECTED_BYTES = K_SMEM_BYTES

# ---------------------------------------------------------------------------
# MMA TS .ws constants (dual GEMM fold)
# ---------------------------------------------------------------------------
# MMA instruction shape: M=64 rows, N=128 cols, K=16 (bf16).
comptime MMA_M = 64
comptime MMA_N = 128
comptime MMA_K = 16
comptime ACCUM_TYPE = DType.float32

# Folded B descriptor shape: the K [64,512] SMEM data is reinterpreted as
# [128,256] — same physical bytes, different descriptor.
# 128 rows (BMN) x 256 cols (BK) with SWIZZLE_128B, k-major.
comptime FOLDED_B_ROWS = 128  # N dimension of B (transpose_b)
comptime FOLDED_B_COLS = 256  # K dimension of B

# Number of K-MMA iterations: 256 / 16 = 16.
comptime NUM_K_MMAS = FOLDED_B_COLS // MMA_K  # 16

# TMEM layout:
#   columns [0, 127]   : Q data (loaded by tcgen05_cp, 128 uint32 cols)
#   columns [128, 255] : C accumulator (MMA_N = 128 float32 cols)
# A operand = Q in TMEM at column 0 (tmem_addr + 0).
# C accumulator at column 128 (tmem_addr + 128).
comptime A_TMEM_OFFSET: UInt32 = 0
comptime C_TMEM_OFFSET: UInt32 = UInt32(TOTAL_TMEM_COLS)  # 128

# With .ws, each half of datapaths (dp 0-63 and dp 64-127) processes
# half the N columns.  Each thread reads MMA_N/2 = 64 float32 values.
comptime HALF_N = MMA_N // 2  # 64

# P output shape: [64, 128] float32 (raw MMA output).
comptime P_ROWS = MMA_M  # 64
comptime P_COLS = MMA_N  # 128

# P reference shape: [64, 64] float32 (Q x K^T after dual GEMM fold reduction).
comptime P_REF_ROWS = 64
comptime P_REF_COLS = 64

# Block dimension for naive matmul reference kernel.
comptime NAIVE_BLOCK_DIM = 16

# Dynamic SMEM layout:
#   [0, Q_SMEM_BYTES)              : Q tile (swizzled)
#   [Q_SMEM_BYTES, +K_SMEM_BYTES)  : K tile (swizzled)
#   then: tmem_addr (8B) + SharedMemBarrier for Q (8B)
#         + SharedMemBarrier for K (8B) + SharedMemBarrier for MMA (8B)
# Aligned to 128 for TMA.
comptime K_SMEM_OFFSET = Q_SMEM_BYTES  # K starts right after Q
comptime METADATA_OFFSET = K_SMEM_OFFSET + K_SMEM_BYTES
# tmem_addr: 8 bytes, q_mbar: 8 bytes, k_mbar: 8 bytes, mma_mbar: 8 bytes.
comptime TOTAL_SMEM_BYTES = METADATA_OFFSET + 32


# ---------------------------------------------------------------------------
# GPU Kernel
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
def dense_mma_ws_ts_kernel[
    q_tile_rank: Int,
    q_tile_shape: IndexList[q_tile_rank],
    q_desc_shape: IndexList[q_tile_rank],
    k_tile_rank: Int,
    k_tile_shape: IndexList[k_tile_rank],
    k_desc_shape: IndexList[k_tile_rank],
](
    q_tma_op: TMATensorTile[OP_TYPE, q_tile_rank, q_tile_shape, q_desc_shape],
    k_tma_op: TMATensorTile[OP_TYPE, k_tile_rank, k_tile_shape, k_desc_shape],
    p_output: LayoutTensor[
        ACCUM_TYPE, Layout.row_major(P_ROWS, P_COLS), MutAnyOrigin
    ],
):
    """Q: TMA -> SMEM -> TMEM (MMA reads from here).
    K: TMA -> SMEM (MMA reads from here).
    P = Q x K^T via MMA TS .ws -> output.
    """

    # ---- Dynamic shared memory base pointer ----
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()

    # ---- Q SMEM region ----
    var q_smem_ptr = smem_base.bitcast[Scalar[OP_TYPE]]()
    var q_smem_tile = LayoutTensor[
        OP_TYPE,
        Q_SMEM_LAYOUT,
        address_space=.SHARED,
        alignment=128,
    ](q_smem_ptr.as_unsafe_any_origin())

    # ---- K SMEM region (starts after Q) ----
    var k_smem_ptr = (smem_base + K_SMEM_OFFSET).bitcast[Scalar[OP_TYPE]]()
    var k_smem_tile = LayoutTensor[
        OP_TYPE,
        K_SMEM_LAYOUT,
        address_space=.SHARED,
        alignment=128,
    ](k_smem_ptr.as_unsafe_any_origin())

    # ---- Metadata region (after both SMEM tiles) ----
    var metadata_ptr = (smem_base + METADATA_OFFSET).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    # Q barrier at offset +8 bytes (2 x UInt32) from metadata start.
    var q_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    # K barrier at offset +16 bytes from metadata start.
    var k_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    # MMA barrier at offset +24 bytes from metadata start.
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    # ---- Initialize barriers ----
    if elect_one_thread:
        q_mbar[0].init()
        k_mbar[0].init()
        mma_mbar[0].init()

    # ---- Allocate TMEM (for Q only) ----
    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # ---- TMA loads: Q and K global -> SMEM (k-major, swizzle128) ----
    if elect_one_thread:
        # Q TMA load
        q_mbar[0].expect_bytes(Int32(Q_TMA_EXPECTED_BYTES))
        q_tma_op.async_copy(q_smem_tile, q_mbar[0], (0, 0))

        # K TMA load
        k_mbar[0].expect_bytes(Int32(K_TMA_EXPECTED_BYTES))
        k_tma_op.async_copy(k_smem_tile, k_mbar[0], (0, 0))

    barrier()

    # Wait for Q TMA to complete before TMEM copy.
    q_mbar[0].wait()

    # ---- tcgen05_cp: Q swizzled SMEM -> TMEM (16 strips) ----
    # Each strip copies 256 bits (16 BF16 = 8 uint32) per datapath.
    # The base descriptor encodes the swizzled layout via SBO/LBO.
    var base_desc = MMASmemDescriptor.create[
        SBO, LBO, TensorMapSwizzle.SWIZZLE_128B
    ](q_smem_ptr)

    comptime for s in range(NUM_STRIPS):
        # With datapaths=128, each call reads 2 consecutive swizzle groups.
        # s_pair selects which pair of groups (0-3); s_local selects which
        # 32-byte (16 BF16) chunk within the 128-byte SMEM row.
        #
        # The byte offset follows the production formula:
        #   k = s_pair * 2 * SW_K + s_local * BF16_PER_STRIP
        #   offset = ((k % SW_K) + (k // SW_K) * ROWS * SW_K) * sizeof
        # which simplifies to:
        #   s_pair * 2 * ROWS * SW_K * sizeof + s_local * BF16_PER_STRIP * sizeof
        comptime s_pair = s // STRIPS_PER_PAIR
        comptime s_local = s % STRIPS_PER_PAIR
        comptime strip_byte_offset = (
            s_pair * 2 * ROWS * SW_K + s_local * BF16_PER_STRIP
        ) * size_of[OP_TYPE]()

        var s_desc = base_desc + strip_byte_offset

        tcgen05_cp[cta_group=1, datapaths=128, bits=256](
            tmem_addr + UInt32(s * U32_PER_STRIP), s_desc
        )

    barrier()

    # ---- Wait for K TMA before MMA can read B from SMEM ----
    # We wait for the K barrier here (after Q TMEM work is done) so
    # the two TMA loads can overlap with the Q TMEM pipeline.
    k_mbar[0].wait()
    barrier()

    # ====================================================================
    # MMA TS .ws: P = Q x K^T using dual GEMM fold
    # ====================================================================
    # Q is still in TMEM at tmem_addr (columns 0-127).
    # K is still in SMEM at k_smem_ptr (64x512 BF16, swizzled).
    #
    # Dual GEMM fold: reinterpret K [64,512] as [128,256] for the B
    # descriptor.  Same physical SMEM bytes, different descriptor shape.

    # ---- Create folded B descriptor for K ----
    # The folded layout is [128, 256] k-major with SWIZZLE_128B.
    var b_desc = smem_descriptor[
        BMN=FOLDED_B_ROWS,
        BK=FOLDED_B_COLS,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        is_k_major=True,
    ](k_smem_ptr)

    # ---- Instruction descriptor for MMA_M=64, MMA_N=128, transpose_b=True ----
    comptime MMA_IDESC = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        ACCUM_TYPE,  # d_type
        OP_TYPE,  # a_type (bf16)
        OP_TYPE,  # b_type (bf16)
        Index[dtype=DType.uint32](MMA_M, MMA_N),
        transpose_b=True,
    ]()

    # ---- TMEM addresses ----
    var a_tmem = tmem_addr + A_TMEM_OFFSET  # Q data in TMEM
    var c_tmem = tmem_addr + C_TMEM_OFFSET  # Accumulator in TMEM

    # ---- Issue MMA (warp 0 only) ----
    # For TS (TMEM-A, SMEM-B) .ws MMA with cta_group::1, only one warp
    # must issue the tcgen05.mma instruction.  If all 4 warps issue it,
    # the hardware executes 4 independent MMAs that accumulate into the
    # same TMEM columns, producing 4x the correct result on accumulation
    # steps (c_scale=1).  The .ws suffix handles the 128-datapath split
    # (dp 0-63 vs dp 64-127) within a single warp's instruction.
    #
    # Note: this differs from SS (SMEM-SMEM) .ws MMA where all 4 warps
    # cooperatively issue the instruction and the hardware coordinates.
    var e: Int32 = 0
    if wid == 0:
        e = elect()
    bulk_mma_ws_ts[
        UMMAKind.KIND_F16,
        OP_TYPE,
        b_BMN=FOLDED_B_ROWS,
        b_BK=FOLDED_B_COLS,
        b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        b_is_k_major=True,
        num_k_mmas=NUM_K_MMAS,
        operand_size=size_of[OP_TYPE](),
        tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
    ](MMA_IDESC, a_tmem, b_desc, c_tmem, UInt32(0), e)

    # Signal MMA completion.  Only one thread commits and arrives.
    if elect_one_thread:
        mma_arrive(mma_mbar)

    # All threads wait for MMA to finish.
    mma_mbar[0].wait(0)

    # Fence to ensure TMEM accumulator writes are visible.
    tcgen05_fence_after()

    # ---- Read C accumulator from TMEM ----
    # With .ws and cta_group::1, 128 datapaths split:
    #   dp 0-63   (threads 0-63):   columns 0..63 of the output
    #   dp 64-127 (threads 64-127): columns 64..127 of the output
    # Each thread reads HALF_N = 64 float32 values.
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=HALF_N,
        dtype=ACCUM_TYPE,
        pack=False,
        width=HALF_N,
    ](c_tmem)

    tcgen05_load_wait()

    # ---- Write P accumulator -> global output ----
    var p_row = tid % MMA_M  # 0-63
    var p_dp_half = tid // MMA_M  # 0 or 1
    var p_col_base = p_dp_half * HALF_N  # 0 or 64

    for j in range(HALF_N):
        p_output[p_row, p_col_base + j] = c_frag[j]

    # ---- Deallocate TMEM ----
    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# Sparse GPU Kernel (parameterized)
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_gather4_tma, `nvvm.grid_constant`)
def sparse_mma_ws_ts_kernel[
    op_type: DType,
    rows: Int,
    cols: Int,
    q_tile_rank: Int,
    q_tile_shape: IndexList[q_tile_rank],
    q_desc_shape: IndexList[q_tile_rank],
    k_tile_rank: Int,
    k_tile_shape: IndexList[k_tile_rank],
    k_desc_shape: IndexList[k_tile_rank],
](
    q_tma_op: TMATensorTile[op_type, q_tile_rank, q_tile_shape, q_desc_shape],
    k_gather4_tma: TMATensorTile[
        op_type, k_tile_rank, k_tile_shape, k_desc_shape
    ],
    d_indices: MutPointer[Int32, MutAnyOrigin],
    p_output: LayoutTensor[
        .float32, Layout.row_major(rows, 2 * rows), MutAnyOrigin
    ],
):
    """Q [rows, cols]: TMA bulk -> SMEM (SWIZZLE_128B) -> TMEM.
    K [rows, cols]: TMA gather4 -> SMEM (SWIZZLE_128B).
    P = Q x K^T via MMA TS .ws -> output [rows, 2*rows].

    K is loaded from a global buffer using `rows` indices.
    Each gather4 call loads 4 rows x sw_k elements (one SWIZZLE_128B group).
    The SMEM destination for each call matches the bulk TMA sub-copy layout.

    All derived constants are computed from `op_type`, `rows`, and `cols`.
    """

    # ---- Derive all constants from parameters ----
    comptime accum_type = DType.float32
    comptime sw_bytes = 128
    comptime sw_k = sw_bytes // size_of[op_type]()
    comptime elems_per_strip = 16
    comptime u32_per_strip = elems_per_strip // 2
    comptime strips_per_pair = sw_k // elems_per_strip
    comptime num_group_pairs = (cols // sw_k) // 2
    comptime num_strips = num_group_pairs * strips_per_pair
    comptime total_tmem_cols = num_strips * u32_per_strip
    comptime max_tmem_cols: UInt32 = 512

    comptime q_smem_layout = tile_layout_k_major[
        op_type, rows, cols, TensorMapSwizzle.SWIZZLE_128B
    ]()
    comptime q_smem_bytes = q_smem_layout.size() * size_of[op_type]()
    comptime k_smem_layout = tile_layout_k_major[
        op_type, rows, cols, TensorMapSwizzle.SWIZZLE_128B
    ]()
    comptime k_smem_bytes = k_smem_layout.size() * size_of[op_type]()

    comptime canonical_layout = tile_to_descriptor[
        op_type, q_smem_layout, is_k_major=True
    ]()
    comptime stride_01 = canonical_layout[0].stride[1].value()
    comptime stride_11 = canonical_layout[1].stride[1].value()
    comptime sbo = stride_01 * size_of[op_type]()
    comptime lbo = stride_11 * size_of[op_type]()

    comptime k_smem_offset = q_smem_bytes
    comptime metadata_offset = k_smem_offset + k_smem_bytes
    comptime total_smem_bytes = metadata_offset + 32

    # MMA constants (dual GEMM fold).
    comptime mma_m = rows
    comptime mma_n = 2 * rows
    comptime mma_k = 16
    comptime folded_b_rows = 2 * rows
    comptime folded_b_cols = cols // 2
    comptime num_k_mmas = folded_b_cols // mma_k
    comptime a_tmem_offset: UInt32 = 0
    comptime c_tmem_offset: UInt32 = UInt32(total_tmem_cols)
    comptime half_n = mma_n // 2
    comptime p_cols = mma_n

    # ---- Dynamic shared memory base pointer ----
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()

    # ---- Q SMEM region ----
    var q_smem_ptr = smem_base.bitcast[Scalar[op_type]]()
    var q_smem_tile = LayoutTensor[
        op_type,
        q_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](q_smem_ptr.as_unsafe_any_origin())

    # ---- K SMEM region (starts after Q) ----
    var k_smem_ptr = (smem_base + k_smem_offset).bitcast[Scalar[op_type]]()

    # ---- Metadata region ----
    var metadata_ptr = (smem_base + metadata_offset).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    var q_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var k_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    # ---- Initialize barriers ----
    if elect_one_thread:
        q_mbar[0].init()
        k_mbar[0].init()
        mma_mbar[0].init()

    # ---- Allocate TMEM ----
    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # ---- Q TMA bulk load ----
    if elect_one_thread:
        q_mbar[0].expect_bytes(Int32(q_smem_bytes))
        q_tma_op.async_copy(q_smem_tile, q_mbar[0], (0, 0))

        # Prefetch K gather4 descriptor into constant cache.
        prefetch_tma_descriptor(
            Pointer(to=k_gather4_tma.descriptor).bitcast[NoneType]()
        )

    barrier()

    # Wait for Q TMA to complete before TMEM copy.
    q_mbar[0].wait()

    # ---- tcgen05_cp: Q swizzled SMEM -> TMEM ----
    var base_desc = MMASmemDescriptor.create[
        sbo, lbo, TensorMapSwizzle.SWIZZLE_128B
    ](q_smem_ptr)

    comptime for s in range(num_strips):
        comptime s_pair = s // strips_per_pair
        comptime s_local = s % strips_per_pair
        comptime strip_byte_offset = (
            s_pair * 2 * rows * sw_k + s_local * elems_per_strip
        ) * size_of[op_type]()

        var s_desc = base_desc + strip_byte_offset

        tcgen05_cp[cta_group=1, datapaths=128, bits=256](
            tmem_addr + UInt32(s * u32_per_strip), s_desc
        )

    barrier()

    # ---- K TMA gather4: all calls on a single barrier ----
    comptime box_width = k_tile_shape[1]
    comptime num_col_groups = ceildiv(cols, box_width)
    comptime num_4row_chunks = rows // 4
    comptime total_calls = num_col_groups * num_4row_chunks
    comptime bytes_per_call = 4 * box_width * size_of[op_type]()
    comptime total_expected_bytes = total_calls * bytes_per_call

    if elect_one_thread:
        k_mbar[0].expect_bytes(Int32(total_expected_bytes))

        comptime for cg in range(num_col_groups):
            var cg_elem_off = cg * rows * box_width

            comptime for c in range(num_4row_chunks):
                var idx_base = c * 4
                var chunk_elem_off = c * 4 * box_width
                var elem_off = cg_elem_off + chunk_elem_off
                var smem_dst_ptr = k_smem_ptr + elem_off
                var smem_dst_tile = LayoutTensor[
                    op_type,
                    Layout.row_major(4, box_width),
                    address_space=.SHARED,
                    alignment=128,
                ](smem_dst_ptr.as_unsafe_any_origin())

                k_gather4_tma.async_copy_gather4(
                    smem_dst_tile,
                    k_mbar[0],
                    col_idx=Int32(cg * box_width),
                    row0=d_indices[idx_base + 0],
                    row1=d_indices[idx_base + 1],
                    row2=d_indices[idx_base + 2],
                    row3=d_indices[idx_base + 3],
                )

    barrier()
    k_mbar[0].wait()
    barrier()

    # ====================================================================
    # MMA TS .ws: P = Q x K^T using dual GEMM fold
    # ====================================================================
    var b_desc = smem_descriptor[
        BMN=folded_b_rows,
        BK=folded_b_cols,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        is_k_major=True,
    ](k_smem_ptr)

    comptime mma_idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        op_type,
        op_type,
        Index[dtype=DType.uint32](mma_m, mma_n),
        transpose_b=True,
    ]()

    var a_tmem = tmem_addr + a_tmem_offset
    var c_tmem = tmem_addr + c_tmem_offset

    var e: Int32 = 0
    if wid == 0:
        e = elect()
    bulk_mma_ws_ts[
        UMMAKind.KIND_F16,
        op_type,
        b_BMN=folded_b_rows,
        b_BK=folded_b_cols,
        b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        b_is_k_major=True,
        num_k_mmas=num_k_mmas,
        operand_size=size_of[op_type](),
        tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
    ](mma_idesc, a_tmem, b_desc, c_tmem, UInt32(0), e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    # ---- Read C accumulator from TMEM ----
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=half_n,
        dtype=accum_type,
        pack=False,
        width=half_n,
    ](c_tmem)

    tcgen05_load_wait()

    # ---- Write P accumulator -> global output ----
    var p_row = tid % mma_m
    var p_dp_half = tid // mma_m
    var p_col_base = p_dp_half * half_n

    for j in range(half_n):
        p_output[p_row, p_col_base + j] = c_frag[j]

    # ---- Deallocate TMEM ----
    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_dense_mma_ws_ts(ctx: DeviceContext) raises:
    print(
        "test_dense_mma_ws_ts: Q 64x512 BF16 -> TMA -> SMEM -> TMEM (for MMA)"
    )
    print("+ K 64x512 BF16 -> TMA -> SMEM (for MMA)")
    print("+ P = Q x K^T via MMA TS .ws [64,128] -> verify")

    # ---- Allocate and fill Q input with random values ----
    seed(42)
    var q_inp = ManagedLayoutTensor[OP_TYPE, Layout.row_major(ROWS, COLS)](ctx)
    var q_inp_host = q_inp.tensor[update=False]()
    randn[OP_TYPE](q_inp_host.ptr, ROWS * COLS)

    # ---- Allocate and fill K input with random values ----
    var k_inp = ManagedLayoutTensor[OP_TYPE, Layout.row_major(K_ROWS, K_COLS)](
        ctx
    )
    var k_inp_host = k_inp.tensor[update=False]()
    randn[OP_TYPE](k_inp_host.ptr, K_ROWS * K_COLS)

    # ---- Allocate P output buffer (MMA result) ----
    var p_out_buf = ManagedLayoutTensor[
        ACCUM_TYPE, Layout.row_major(P_ROWS, P_COLS)
    ](ctx)

    # ---- Create TMA descriptors: k-major, SWIZZLE_128B ----
    var q_tma_op = create_tensor_tile[
        Index(ROWS, COLS),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](ctx, q_inp.device_tensor())

    var k_tma_op = create_tensor_tile[
        Index(K_ROWS, K_COLS),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](ctx, k_inp.device_tensor())

    # ---- Launch kernel ----
    comptime kernel = dense_mma_ws_ts_kernel[
        type_of(q_tma_op).rank,
        type_of(q_tma_op).tile_shape,
        type_of(q_tma_op).desc_shape,
        type_of(k_tma_op).rank,
        type_of(k_tma_op).tile_shape,
        type_of(k_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        q_tma_op,
        k_tma_op,
        p_out_buf.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=TOTAL_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(TOTAL_SMEM_BYTES)
        ),
    )
    ctx.synchronize()

    # ---- Verify P = Q x K^T ----
    # The raw MMA output is [64, 128]:
    #   P[:, 0:64]   = Q_even_k_blocks * K_even_k_blocks^T (partial)
    #   P[:, 64:128] = Q_odd_k_blocks * K_odd_k_blocks^T (partial)
    # Summing P[:, 0:64] + P[:, 64:128] gives the full 64x64 Q x K^T.
    print("  Verifying P = Q x K^T (MMA TS .ws, dual GEMM fold)...")
    var p_out_host = p_out_buf.tensor()

    # Print some raw P values for debugging.
    var p_out_ptr = p_out_host.ptr
    print(
        "  P raw[0,0]="
        + String(p_out_ptr[0])
        + " P raw[0,64]="
        + String(p_out_ptr[64])
    )
    print(
        "  P raw[63,63]="
        + String(p_out_ptr[63 * P_COLS + 63])
        + " P raw[63,127]="
        + String(p_out_ptr[63 * P_COLS + 127])
    )

    # Compute reference on GPU using the naive matmul kernel:
    #   P_ref[64,64] = Q[64,512] x K[64,512]^T (transpose_b=True).
    # Then compare against P[:, 0:64] + P[:, 64:128].
    var p_ref_device = ctx.enqueue_create_buffer[.float32](
        P_REF_ROWS * P_REF_COLS
    )

    # Build TileTensors for the naive kernel.
    # C (output) is mutable; A and B are immutable to match the
    # ImmutAnyOrigin parameters that matmul_kernel_naive expects.
    var q_device_ptr = q_inp.device_data.value().unsafe_ptr()
    var k_device_ptr = k_inp.device_data.value().unsafe_ptr()

    var c_ref_tt = TileTensor(
        p_ref_device,
        row_major(Coord(P_REF_ROWS, P_REF_COLS)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[OP_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(q_device_ptr)
        ),
        row_major(Coord(ROWS, COLS)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[OP_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(k_device_ptr)
        ),
        row_major(Coord(K_ROWS, K_COLS)),
    )

    comptime gemm_naive = matmul_kernel_naive[
        DType.float32,  # c_type
        OP_TYPE,  # a_type (bfloat16)
        OP_TYPE,  # b_type (bfloat16)
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
        Int32(P_REF_ROWS),  # m
        Int32(P_REF_COLS),  # n
        Int32(COLS),  # k
        grid_dim=(
            ceildiv(P_REF_ROWS, NAIVE_BLOCK_DIM),
            ceildiv(P_REF_COLS, NAIVE_BLOCK_DIM),
            1,
        ),
        block_dim=(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM, 1),
    )

    var p_ref_host = ctx.enqueue_create_host_buffer[.float32](
        P_REF_ROWS * P_REF_COLS
    )
    ctx.enqueue_copy(p_ref_host, p_ref_device)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    for r in range(P_REF_ROWS):
        for c in range(P_REF_COLS):
            var ref_val = p_ref_host[r * P_REF_COLS + c]

            # GPU result: sum of the two halves.
            var gpu_half0 = Float32(p_out_ptr[r * P_COLS + c])
            var gpu_half1 = Float32(p_out_ptr[r * P_COLS + c + P_ROWS])
            var gpu_val = gpu_half0 + gpu_half1

            var err = abs(gpu_val - ref_val) / max(abs(ref_val), Float32(1.0))
            if err > max_err:
                max_err = err

            assert_almost_equal(
                gpu_val,
                ref_val,
                atol=1.0,
                rtol=0.01,
                msg="P: r="
                + String(r)
                + " c="
                + String(c)
                + " gpu="
                + String(gpu_val)
                + " ref="
                + String(ref_val),
            )

    print("  P max relative error: " + String(max_err))
    print("  P PASSED")

    _ = p_ref_device
    _ = q_inp^
    _ = k_inp^
    _ = p_out_buf^


# ---------------------------------------------------------------------------
# Sparse Test (parameterized)
# ---------------------------------------------------------------------------
def test_sparse_mma_ws_ts[
    op_type: DType,
    rows: Int,
    cols: Int,
    total_tokens: Int,
](ctx: DeviceContext) raises:
    """Test sparse + MMA TS .ws pipeline with parameterized dimensions.

    Parameters:
        op_type: Data type for Q and K matrices (e.g. DType.bfloat16).
        rows: Number of rows in Q/K tile (must be 64 for MMA_M).
        cols: Number of columns (head dimension, e.g. 512).
        total_tokens: Number of tokens in the global K buffer.
    """
    # Derived constants.
    comptime sw_bytes = 128
    comptime sw_k = sw_bytes // size_of[op_type]()
    comptime p_rows = rows
    comptime p_cols = 2 * rows  # MMA_N = 2 * rows
    comptime p_ref_rows = rows
    comptime p_ref_cols = rows
    comptime num_threads = 128
    comptime naive_block_dim = 16

    # SMEM layout and sizes (derived from parameters).
    comptime q_smem_layout = tile_layout_k_major[
        op_type, rows, cols, TensorMapSwizzle.SWIZZLE_128B
    ]()
    comptime q_smem_bytes = q_smem_layout.size() * size_of[op_type]()
    comptime k_smem_bytes = q_smem_bytes  # Same shape
    comptime k_smem_offset = q_smem_bytes
    comptime metadata_offset = k_smem_offset + k_smem_bytes
    comptime total_smem_bytes = metadata_offset + 32

    print(
        "test_sparse_mma_ws_ts: Q "
        + String(rows)
        + "x"
        + String(cols)
        + " -> TMA bulk -> SMEM -> TMEM"
    )
    print(
        "                      + K sparse gathered from "
        + String(total_tokens)
        + "x"
        + String(cols)
        + " via "
        + String((cols // sw_k) * (rows // 4))
        + " TMA gather4 -> SMEM (SWIZZLE_128B)"
    )
    print(
        "                      + P = Q x K^T via MMA TS .ws ["
        + String(rows)
        + ","
        + String(p_cols)
        + "] -> verify"
    )

    # ---- Allocate and fill Q input [rows, cols] with random values ----
    seed(42)
    var q_inp = ManagedLayoutTensor[op_type, Layout.row_major(rows, cols)](ctx)
    var q_inp_host = q_inp.tensor[update=False]()
    randn[op_type](q_inp_host.ptr, rows * cols)

    # ---- Allocate the full K buffer [total_tokens, cols] ----
    var k_full = ManagedLayoutTensor[
        op_type, Layout.row_major(total_tokens, cols)
    ](ctx)
    var k_full_host = k_full.tensor[update=False]()
    randn[op_type](k_full_host.ptr, total_tokens * cols)

    # ---- Build non-contiguous indices into the full K buffer ----
    var h_indices = ctx.enqueue_create_host_buffer[.int32](rows)
    for i in range(rows):
        h_indices[i] = Int32((i * 37 + 13) % total_tokens)

    var d_indices = ctx.enqueue_create_buffer[.int32](rows)
    ctx.enqueue_copy(d_indices, h_indices)

    # ---- Build reference K [rows, cols] from selected rows ----
    var k_ref_host = ctx.enqueue_create_host_buffer[op_type](rows * cols)
    for i in range(rows):
        var src_row = Int(h_indices[i])
        for c in range(cols):
            k_ref_host[i * cols + c] = k_full_host.ptr[src_row * cols + c]

    var k_ref_device = ctx.enqueue_create_buffer[op_type](rows * cols)
    ctx.enqueue_copy(k_ref_device, k_ref_host)

    # ---- Allocate P output buffer [rows, 2*rows] ----
    var p_out_buf = ManagedLayoutTensor[
        .float32, Layout.row_major(p_rows, p_cols)
    ](ctx)

    # ---- Create Q TMA descriptor ----
    var q_tma_op = create_tensor_tile[
        Index(rows, cols),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](ctx, q_inp.device_tensor())

    # ---- Create K gather4 TMA tile ----
    _ = k_full.device_tensor()
    var k_gather4_tma = create_tma_tile_gather4[
        op_type, tile_width=cols, swizzle_mode=TensorMapSwizzle.SWIZZLE_128B
    ](ctx, k_full.device_data.value(), total_tokens)

    # ---- Launch sparse kernel ----
    comptime kernel = sparse_mma_ws_ts_kernel[
        op_type,
        rows,
        cols,
        type_of(q_tma_op).rank,
        type_of(q_tma_op).tile_shape,
        type_of(q_tma_op).desc_shape,
        type_of(k_gather4_tma).rank,
        type_of(k_gather4_tma).tile_shape,
        type_of(k_gather4_tma).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        q_tma_op,
        k_gather4_tma,
        d_indices.unsafe_ptr(),
        p_out_buf.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(num_threads),
        shared_mem_bytes=total_smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(total_smem_bytes)
        ),
    )
    ctx.synchronize()

    # ---- Verify P = Q x K_gathered^T ----
    print(
        "  Verifying P = Q x K_gathered^T"
        " (MMA TS .ws, sparse + dual GEMM fold)..."
    )
    var p_out_host = p_out_buf.tensor()
    var p_out_ptr = p_out_host.ptr
    print(
        "  P raw[0,0]="
        + String(p_out_ptr[0])
        + " P raw[0,"
        + String(rows)
        + "]="
        + String(p_out_ptr[rows])
    )

    # Compute reference: P_ref = Q x K_gathered^T.
    var p_ref_device = ctx.enqueue_create_buffer[.float32](
        p_ref_rows * p_ref_cols
    )

    var q_device_ptr = q_inp.device_data.value().unsafe_ptr()

    var c_ref_tt = TileTensor(
        p_ref_device,
        row_major(Coord(p_ref_rows, p_ref_cols)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[op_type], ImmutAnyOrigin](
            unsafe_from_address=Int(q_device_ptr)
        ),
        row_major(Coord(rows, cols)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[op_type], ImmutAnyOrigin](
            unsafe_from_address=Int(k_ref_device.unsafe_ptr())
        ),
        row_major(Coord(rows, cols)),
    )

    comptime gemm_naive = matmul_kernel_naive[
        DType.float32,
        op_type,
        op_type,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        naive_block_dim,
        transpose_b=True,
    ]

    ctx.enqueue_function[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(p_ref_rows),
        Int32(p_ref_cols),
        Int32(cols),
        grid_dim=(
            ceildiv(p_ref_rows, naive_block_dim),
            ceildiv(p_ref_cols, naive_block_dim),
            1,
        ),
        block_dim=(naive_block_dim, naive_block_dim, 1),
    )

    var p_ref_host = ctx.enqueue_create_host_buffer[.float32](
        p_ref_rows * p_ref_cols
    )
    ctx.enqueue_copy(p_ref_host, p_ref_device)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    for r in range(p_ref_rows):
        for c in range(p_ref_cols):
            var ref_val = p_ref_host[r * p_ref_cols + c]

            var gpu_half0 = Float32(p_out_ptr[r * p_cols + c])
            var gpu_half1 = Float32(p_out_ptr[r * p_cols + c + p_rows])
            var gpu_val = gpu_half0 + gpu_half1

            var err = abs(gpu_val - ref_val) / max(abs(ref_val), Float32(1.0))
            if err > max_err:
                max_err = err

            assert_almost_equal(
                gpu_val,
                ref_val,
                atol=1.0,
                rtol=0.01,
                msg="P sparse: r="
                + String(r)
                + " c="
                + String(c)
                + " gpu="
                + String(gpu_val)
                + " ref="
                + String(ref_val),
            )

    print("  P max relative error: " + String(max_err))
    print("  P sparse PASSED")

    _ = p_ref_device
    _ = k_ref_device
    _ = d_indices
    _ = q_inp^
    _ = k_full^
    _ = p_out_buf^


# ---------------------------------------------------------------------------
# Sparse Paged Test (parameterized)
# ---------------------------------------------------------------------------
def test_sparse_paged_mma_ws_ts[
    op_type: DType,
    num_heads: Int,
    head_size: Int,
    page_size: Int,
    num_blocks: Int,
    num_layers: Int,
    batch_size: Int,
    total_tokens: Int,
    topk: Int,
    kv_dim: Int,
](ctx: DeviceContext) raises:
    """Test sparse paged + MMA TS .ws pipeline with parameterized config.

    Parameters:
        op_type: Data type for Q and K matrices (e.g. DType.bfloat16).
        num_heads: Number of KV heads.
        head_size: Head dimension size.
        page_size: Number of tokens per page in the paged KV cache.
        num_blocks: Number of physical page blocks.
        num_layers: Number of layers in the KV cache.
        batch_size: Number of sequences in the batch.
        total_tokens: Total number of tokens per sequence.
        topk: Number of tokens to gather (must equal rows=64).
        kv_dim: KV dimension (1 for combined K/V, 2 for separate).
    """
    # Derived constants.
    comptime row_width = num_heads * head_size
    comptime rows = topk  # rows = topk = 64 for MMA_M
    comptime cols = row_width
    comptime paged_stride = kv_dim * num_layers * page_size
    comptime p_rows = rows
    comptime p_cols = 2 * rows  # MMA_N = 2 * rows
    comptime p_ref_rows = rows
    comptime p_ref_cols = rows
    comptime num_threads = 128
    comptime naive_block_dim = 16

    # SMEM layout and sizes (derived from parameters).
    comptime q_smem_layout = tile_layout_k_major[
        op_type, rows, cols, TensorMapSwizzle.SWIZZLE_128B
    ]()
    comptime q_smem_bytes = q_smem_layout.size() * size_of[op_type]()
    comptime k_smem_bytes = q_smem_bytes  # Same shape
    comptime k_smem_offset = q_smem_bytes
    comptime metadata_offset = k_smem_offset + k_smem_bytes
    comptime total_smem_bytes = metadata_offset + 32

    print(
        "test_sparse_paged_mma_ws_ts: Q "
        + String(rows)
        + "x"
        + String(cols)
        + " -> TMA bulk -> SMEM -> TMEM"
    )
    print(
        "+ K gathered from PAGED KV cache (page_size="
        + String(page_size)
        + ", num_blocks="
        + String(num_blocks)
        + ")"
    )
    print(
        "+ P = Q x K^T via MMA TS .ws ["
        + String(rows)
        + ","
        + String(p_cols)
        + "] -> verify"
    )

    # ---- Build the 6D blocks tensor ----
    # Shape: [num_blocks, kv_dim, num_layers, page_size, num_heads, head_size]
    comptime pg_shape_6d = IndexList[6](
        num_blocks,
        kv_dim,
        num_layers,
        page_size,
        num_heads,
        head_size,
    )
    comptime pg_layout_6d = Layout.row_major[6]()
    var blocks = ManagedLayoutTensor[op_type, pg_layout_6d](
        RuntimeLayout[pg_layout_6d].row_major(pg_shape_6d), ctx
    )
    var blocks_host = blocks.tensor[update=False]()

    # Fill entire blocks buffer with random data.
    seed(42)
    var block_elems = (
        num_blocks * kv_dim * num_layers * page_size * num_heads * head_size
    )
    rand[op_type](blocks_host.ptr, block_elems)

    # ---- Build cache_lengths ----
    comptime cache_len_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_managed = ManagedLayoutTensor[.uint32, cache_len_layout](
        RuntimeLayout[cache_len_layout].row_major(IndexList[1](batch_size)),
        ctx,
    )
    var cache_lengths_host = cache_lengths_managed.tensor[update=False]()
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(total_tokens)

    # ---- Build lookup_table with shuffled page assignments ----
    comptime lut_layout = Layout.row_major[2]()
    var max_pages_per_seq = (total_tokens + page_size - 1) // page_size
    var lut_managed = ManagedLayoutTensor[.uint32, lut_layout](
        RuntimeLayout[lut_layout].row_major(
            IndexList[2](batch_size, num_blocks)
        ),
        ctx,
    )
    var lut_host = lut_managed.tensor[update=False]()
    var lut_ptr = lut_host.ptr
    for s in range(batch_size):
        for p in range(max_pages_per_seq):
            var blk = ((s * max_pages_per_seq + p) * 37 + 13) % num_blocks
            lut_ptr[s * num_blocks + p] = UInt32(blk)

    # ---- Construct PagedKVCacheCollection and extract key cache ----
    comptime kv_params = KVCacheStaticParams(
        num_heads=num_heads,
        head_size=head_size,
    )
    var collection = PagedKVCacheCollection[op_type, kv_params, page_size](
        blocks.device_tensor(),
        cache_lengths_managed.device_tensor(),
        lut_managed.device_tensor(),
        UInt32(total_tokens),
        UInt32(total_tokens),
    )
    var kv_cache = collection.get_key_cache(0)

    # ---- Allocate and fill Q input [rows, cols] with random values ----
    seed(42)
    var q_inp = ManagedLayoutTensor[op_type, Layout.row_major(rows, cols)](ctx)
    var q_inp_host = q_inp.tensor[update=False]()
    randn[op_type](q_inp_host.ptr, rows * cols)

    # ---- Build gather indices from the paged cache ----
    var h_indices = ctx.enqueue_create_host_buffer[.int32](topk)
    for i in range(topk):
        var tok_idx = (i * 37 + 13) % total_tokens
        var page_within_seq = tok_idx // page_size
        var offset_in_page = tok_idx % page_size
        var phys_block = Int(lut_ptr[0 * num_blocks + page_within_seq])
        var phys_row = phys_block * paged_stride + offset_in_page
        h_indices[i] = Int32(phys_row)

    var d_indices = ctx.enqueue_create_buffer[.int32](topk)
    ctx.enqueue_copy(d_indices, h_indices)

    # ---- Build reference K_gathered on host ----
    var k_ref_host = ctx.enqueue_create_host_buffer[op_type](topk * row_width)
    for i in range(topk):
        var src_row = Int(h_indices[i])
        for c in range(row_width):
            k_ref_host[i * row_width + c] = blocks_host.ptr[
                src_row * row_width + c
            ]

    var k_ref_device = ctx.enqueue_create_buffer[op_type](topk * row_width)
    ctx.enqueue_copy(k_ref_device, k_ref_host)

    # ---- Allocate P output buffer [rows, 2*rows] ----
    var p_out_buf = ManagedLayoutTensor[
        .float32, Layout.row_major(p_rows, p_cols)
    ](ctx)

    # ---- Create Q TMA descriptor ----
    var q_tma_op = create_tensor_tile[
        Index(rows, cols),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
    ](ctx, q_inp.device_tensor())

    # ---- Create K gather4 TMA descriptor from the paged KV cache ----
    var k_gather4_tma = kv_cache.create_gather4_tma_tile[
        tile_width=row_width, swizzle_mode=TensorMapSwizzle.SWIZZLE_128B
    ](ctx)

    # ---- Launch sparse kernel ----
    comptime kernel = sparse_mma_ws_ts_kernel[
        op_type,
        rows,
        cols,
        type_of(q_tma_op).rank,
        type_of(q_tma_op).tile_shape,
        type_of(q_tma_op).desc_shape,
        type_of(k_gather4_tma).rank,
        type_of(k_gather4_tma).tile_shape,
        type_of(k_gather4_tma).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        q_tma_op,
        k_gather4_tma,
        d_indices.unsafe_ptr(),
        p_out_buf.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(num_threads),
        shared_mem_bytes=total_smem_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(total_smem_bytes)
        ),
    )
    ctx.synchronize()

    # ---- Verify P = Q x K_gathered^T ----
    print(
        "  Verifying P = Q x K_gathered^T"
        " (MMA TS .ws, sparse paged + dual GEMM fold)..."
    )
    var p_out_host = p_out_buf.tensor()
    var p_out_ptr = p_out_host.ptr
    print(
        "  P raw[0,0]="
        + String(p_out_ptr[0])
        + " P raw[0,"
        + String(rows)
        + "]="
        + String(p_out_ptr[rows])
    )

    # Compute reference: P_ref = Q x K_gathered^T.
    var p_ref_device = ctx.enqueue_create_buffer[.float32](
        p_ref_rows * p_ref_cols
    )

    var q_device_ptr = q_inp.device_data.value().unsafe_ptr()

    var c_ref_tt = TileTensor(
        p_ref_device,
        row_major(Coord(p_ref_rows, p_ref_cols)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[op_type], ImmutAnyOrigin](
            unsafe_from_address=Int(q_device_ptr)
        ),
        row_major(Coord(rows, cols)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[op_type], ImmutAnyOrigin](
            unsafe_from_address=Int(k_ref_device.unsafe_ptr())
        ),
        row_major(Coord(topk, row_width)),
    )

    comptime gemm_naive = matmul_kernel_naive[
        DType.float32,
        op_type,
        op_type,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        naive_block_dim,
        transpose_b=True,
    ]

    ctx.enqueue_function[gemm_naive](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(p_ref_rows),
        Int32(p_ref_cols),
        Int32(cols),
        grid_dim=(
            ceildiv(p_ref_rows, naive_block_dim),
            ceildiv(p_ref_cols, naive_block_dim),
            1,
        ),
        block_dim=(naive_block_dim, naive_block_dim, 1),
    )

    var p_ref_host = ctx.enqueue_create_host_buffer[.float32](
        p_ref_rows * p_ref_cols
    )
    ctx.enqueue_copy(p_ref_host, p_ref_device)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    for r in range(p_ref_rows):
        for c in range(p_ref_cols):
            var ref_val = p_ref_host[r * p_ref_cols + c]

            var gpu_half0 = Float32(p_out_ptr[r * p_cols + c])
            var gpu_half1 = Float32(p_out_ptr[r * p_cols + c + p_rows])
            var gpu_val = gpu_half0 + gpu_half1

            var err = abs(gpu_val - ref_val) / max(abs(ref_val), Float32(1.0))
            if err > max_err:
                max_err = err

            assert_almost_equal(
                gpu_val,
                ref_val,
                atol=1.0,
                rtol=0.01,
                msg="P sparse paged: r="
                + String(r)
                + " c="
                + String(c)
                + " gpu="
                + String(gpu_val)
                + " ref="
                + String(ref_val),
            )

    print("  P max relative error: " + String(max_err))
    print("  P sparse paged PASSED")
    _ = p_ref_device
    _ = k_ref_device
    _ = d_indices
    _ = q_inp^
    _ = blocks^
    _ = cache_lengths_managed^
    _ = lut_managed^
    _ = p_out_buf^


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() raises:
    with DeviceContext() as ctx:
        test_dense_mma_ws_ts(ctx)

        # Non-paged sparse: Q [64, 512] x K_gathered [64, 512]^T.
        test_sparse_mma_ws_ts[
            DType.bfloat16,  # op_type
            64,  # rows
            512,  # cols (head dimension)
            256,  # total_tokens
        ](ctx)

        # Sparse paged: Q [64, 512] x K_gathered [64, 512]^T from paged KV
        # cache with page_size=128, num_blocks=8.
        test_sparse_paged_mma_ws_ts[
            DType.bfloat16,  # op_type
            1,  # num_heads
            512,  # head_size (row_width = 1*512 = 512)
            128,  # page_size
            8,  # num_blocks
            1,  # num_layers
            1,  # batch_size
            256,  # total_tokens
            64,  # topk (= rows for MMA)
            1,  # kv_dim
        ](ctx)
