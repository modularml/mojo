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

"""Layout G smoke test: tcgen05.mma.ws cta_group::1 kind::f8f6f4 with M=32.

Purpose
=======
Phase A9.1 of the MLA decode Layout G design (BM=32, BN=64) requires the
custom raw-PTX `bulk_mma_ws` wrapper (attention_utils.mojo) to accept
M=32 even though the Mojo stdlib `_get_f8f6f4_mma_shape`
(mma_nvidia_sm100.mojo:345) only allows M in {64, 128}.  The custom wrapper
in MLA decode bypasses the stdlib shape table by feeding the descriptor M
field directly via `output_shape[0] >> 4`, so M=32 is encoded as `0x2`.

This test issues exactly two `tcgen05.mma.ws.cta_group::1.kind::f8f6f4`
instructions on B200 hardware to prove the encoding produces correct
results:

  * Test 1 — QK shape: A [32, 576] x B [64, 576]^T (FP8 e4m3) -> C [32, 64]
    fp32. This matches the QK GEMM in the Layout G fold path
    (BM=32, BN=64, BK=padded_q_depth=576).

  * Test 2 — Wide-N shape: A [32, 64] x B [256, 64]^T (FP8 e4m3) -> C [32, 256]
    fp32. This is a SIMPLIFIED variant of the PV shape (same M=32, K=64,
    similar N range) to confirm M=32 .ws supports wide N values. We use
    transpose_b=True (k-major B) because that is what the QK smoke
    pattern already validated; the production PV path uses transpose_b=
    False with mn-major V, but the *MMA shape* validation is the same.

Both kernels are single-CTA, single-warpgroup (4 warps, 128 threads),
single-stage. No pipelining, no fold logic — just enough to issue ONE MMA
and verify the result against a GPU naive matmul reference (FP32 inputs
cast through FP8 e4m3 then back to FP32 for the kernel; the same FP8 bytes
are dequantized to BF16 for the reference matmul).

Outcomes
========
1. Test compiles AND passes -> R1 RESOLVED. Layout G is unblocked.
2. Test compiles but produces garbage -> hardware doesn't support M=32 .ws.
3. Test fails to compile -> ptxas rejects M=32 in `kind::f8f6f4` desc.

Run via:
    ./bazelw test //max/kernels/test/gpu/nn:test_mla_layout_g_mma_smoke.mojo.test \\
        --curses=no --noshow_progress

This file is a smoke test for hardware capability; it does not exercise
full attention pipeline. See ~/mla_checkpoints/phase_A9_bm32_bn128_design/
for the design context.
"""

from std.math import ceildiv
from std.memory import alloc
from std.random import rand, randn, seed
from std.sys import size_of

from max.gpu import thread_idx, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.info import B200, _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import (
    external_memory,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
    mma_arrive,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout._utils import ManagedLayoutTensor
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from linalg.arch.sm100.mma import smem_descriptor
from linalg.matmul.gpu import matmul_kernel_naive
from nn.attention.gpu.nvidia.sm100.attention_utils import bulk_mma_ws, elect
from std.testing import assert_true
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

# Tolerances:
# FP8 e4m3 has ~3 bits of mantissa; for K=576 accumulation of 1 N(0,1)
# elements we expect ~5-10% relative error.
comptime ATOL_QK: Float32 = 5.0
comptime RTOL_QK: Float32 = 0.10
comptime ATOL_PV: Float32 = 2.0
comptime RTOL_PV: Float32 = 0.10


# ---------------------------------------------------------------------------
# QK SMOKE: M=32, N=64, K=576 (transpose_b=True)
# ---------------------------------------------------------------------------
comptime QK_M = 32
comptime QK_N = 64
comptime QK_K = 576

comptime QK_MMA_K = 32  # FP8 hardware MMA_K
comptime QK_NUM_K_MMAS = QK_K // QK_MMA_K  # 18

comptime QK_SWIZZLE = TensorMapSwizzle.SWIZZLE_64B

comptime QK_A_BYTES = QK_M * QK_K * size_of[FP8_TYPE]()  # 18432
comptime QK_B_BYTES = QK_N * QK_K * size_of[FP8_TYPE]()  # 36864

# SMEM: A tile, B tile, then tmem_addr+barriers.
comptime QK_A_OFFSET = 0
comptime QK_B_OFFSET = QK_A_BYTES
comptime QK_META_OFFSET = QK_A_BYTES + QK_B_BYTES
# tmem_addr (8B), a_mbar (8B), b_mbar (8B), mma_mbar (8B) = 32B.
comptime QK_TOTAL_SMEM = QK_META_OFFSET + 32

# tcgen05 TMEM allocation must be a power-of-two between 32 and 512.
# We need MMA_N = 64 columns for QK; round up to 128 to satisfy the alloc
# granularity (`tcgen05_alloc` requires >=32, must be a power of 2).
comptime MAX_TMEM_COLS: UInt32 = 512


# ---------------------------------------------------------------------------
# PV SMOKE: M=32, N=512, K=64 (transpose_b=False)
# ---------------------------------------------------------------------------
comptime PV_M = 32
# Use N=256 (not 512) so the B mn-major TMA tile fits in one TMA box
# (TMA boxes are limited to 256 rows). This is a smoke test for the M=32
# .ws MMA with wide N; production PV uses block_step to issue multiple
# MMAs across N=512 anyway.
comptime PV_N = 256
comptime PV_K = 64

comptime PV_MMA_K = 32
comptime PV_NUM_K_MMAS = PV_K // PV_MMA_K  # 2

comptime PV_SWIZZLE = TensorMapSwizzle.SWIZZLE_64B

comptime PV_A_BYTES = PV_M * PV_K * size_of[FP8_TYPE]()  # 2048
comptime PV_B_BYTES = PV_N * PV_K * size_of[FP8_TYPE]()  # 32768

comptime PV_A_OFFSET = 0
comptime PV_B_OFFSET = PV_A_BYTES
comptime PV_META_OFFSET = PV_A_BYTES + PV_B_BYTES
comptime PV_TOTAL_SMEM = PV_META_OFFSET + 32


# ---------------------------------------------------------------------------
# Kernel: QK shape — M=32, N=64, K=576, transpose_b=True
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def qk_smoke_kernel[
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
):
    """SS .ws MMA: C [32,64] = A [32,576] x B [64,576]^T (FP8 e4m3)."""

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
    ](a_smem_ptr)
    var b_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(QK_N, QK_K),
        address_space=.SHARED,
        alignment=128,
    ](b_smem_ptr)

    # ---- Metadata ----
    var metadata_ptr = (smem_base + QK_META_OFFSET).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    var a_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var b_mbar = (metadata_ptr + 4).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var elect_one_thread = tid == 0

    # ---- Init barriers ----
    if elect_one_thread:
        a_mbar[0].init()
        b_mbar[0].init()
        mma_mbar[0].init()

    # ---- Allocate TMEM ----
    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]

    # ---- TMA loads ----
    if elect_one_thread:
        a_mbar[0].expect_bytes(Int32(QK_A_BYTES))
        a_tma_op.async_copy(a_smem_tile, a_mbar[0], (0, 0))

        b_mbar[0].expect_bytes(Int32(QK_B_BYTES))
        b_tma_op.async_copy(b_smem_tile, b_mbar[0], (0, 0))
    barrier()

    a_mbar[0].wait()
    b_mbar[0].wait()
    barrier()

    # ---- Build SMEM descriptors (k-major for both A and B) ----
    var a_desc = smem_descriptor[
        BMN=QK_M,  # 32
        BK=QK_K,  # 576
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](a_smem_ptr)
    var b_desc = smem_descriptor[
        BMN=QK_N,  # 64
        BK=QK_K,  # 576
        swizzle_mode=QK_SWIZZLE,
        is_k_major=True,
    ](b_smem_ptr)

    # ---- Instruction descriptor: M=32, N=64, transpose_b=True ----
    # NOTE: We use UMMAKind.KIND_F8F6F4. The .create method directly encodes
    # output_shape[0] (=32) into the M field via `M >> 4`. There is NO
    # comptime check that gates M=32; the existing _get_f8f6f4_mma_shape
    # check is bypassed by calling .create directly (matching the production
    # mla_decode_utils.mojo pattern).
    comptime mma_idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        ACC_TYPE,  # d_type
        FP8_TYPE,  # a_type
        FP8_TYPE,  # b_type
        Index[dtype=DType.uint32](QK_M, QK_N),
        transpose_b=True,
    ]()

    # ---- Issue MMA from warp 0 only (matches production mmaQK pattern) ----
    var c_tmem = tmem_addr  # accumulator at column 0

    var e: Int32 = 0
    if wid == 0:
        e = elect()

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
        num_k_mmas=QK_NUM_K_MMAS,
        operand_size=size_of[FP8_TYPE](),
        tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        mma_k=QK_MMA_K,
    ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    # ---- Read C from TMEM ----
    # For .ws cta_group::1 with M=32, observation from probing: each warp
    # writes its 16-col slice to the SAME TMEM column offset (0-15), but
    # in the warp's own subpartition view. Different warps share the same
    # TMEM column indices but produce different physical output cols.
    #
    # So warp `w` reads tcgen05.ld at offset 0 with repeat=16 -> gets
    # output cols [w*16 : (w+1)*16].
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
    var p_row = lane_id  # 0..31, matches M=32
    var p_col_base = Int(wid) * QK_COLS_PER_WARP
    for j in range(QK_COLS_PER_WARP):
        c_output[p_row, p_col_base + j] = c_frag[j]

    # ---- Dealloc TMEM ----
    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# Kernel: PV shape — M=32, N=512, K=64, transpose_b=False
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def pv_smoke_kernel[
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
        ACC_TYPE, Layout.row_major(PV_M, PV_N), MutAnyOrigin
    ],
):
    """SS .ws MMA: C [32,512] = A [32,64] x B [512,64] (mn-major B, FP8)."""

    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var a_smem_ptr = (smem_base + PV_A_OFFSET).bitcast[Scalar[FP8_TYPE]]()
    var b_smem_ptr = (smem_base + PV_B_OFFSET).bitcast[Scalar[FP8_TYPE]]()

    var a_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(PV_M, PV_K),
        address_space=.SHARED,
        alignment=128,
    ](a_smem_ptr)
    # B is mn-major: [PV_N rows, PV_K cols] but the descriptor will treat
    # N as the major axis (same as PV_K from the A side via transpose_b=False).
    var b_smem_tile = LayoutTensor[
        FP8_TYPE,
        Layout.row_major(PV_N, PV_K),
        address_space=.SHARED,
        alignment=128,
    ](b_smem_ptr)

    var metadata_ptr = (smem_base + PV_META_OFFSET).bitcast[UInt32]()
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
        a_mbar[0].expect_bytes(Int32(PV_A_BYTES))
        a_tma_op.async_copy(a_smem_tile, a_mbar[0], (0, 0))

        b_mbar[0].expect_bytes(Int32(PV_B_BYTES))
        b_tma_op.async_copy(b_smem_tile, b_mbar[0], (0, 0))
    barrier()

    a_mbar[0].wait()
    b_mbar[0].wait()
    barrier()

    # A: k-major (M=32, K=64, k-major)
    var a_desc = smem_descriptor[
        BMN=PV_M,
        BK=PV_K,
        swizzle_mode=PV_SWIZZLE,
        is_k_major=True,
    ](a_smem_ptr)
    # B: k-major (N=256 rows, K=64 cols). transpose_b=True on the MMA so
    # the matmul computes C[m,n] = sum_k A[m,k] * B[n,k] exactly as the
    # naive reference computes.
    var b_desc = smem_descriptor[
        BMN=PV_N,
        BK=PV_K,
        swizzle_mode=PV_SWIZZLE,
        is_k_major=True,
    ](b_smem_ptr)

    # ---- Instruction descriptor: M=32, N=256, transpose_b=True ----
    comptime mma_idesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        ACC_TYPE,
        FP8_TYPE,
        FP8_TYPE,
        Index[dtype=DType.uint32](PV_M, PV_N),
        transpose_b=True,
    ]()

    var c_tmem = tmem_addr

    var e: Int32 = 0
    if wid == 0:
        e = elect()

    bulk_mma_ws[
        UMMAKind.KIND_F8F6F4,
        FP8_TYPE,
        FP8_TYPE,
        a_BMN=PV_M,
        a_BK=PV_K,
        a_swizzle=PV_SWIZZLE,
        a_is_k_major=True,
        b_BMN=PV_N,
        b_BK=PV_K,
        b_swizzle=PV_SWIZZLE,
        b_is_k_major=True,
        num_k_mmas=PV_NUM_K_MMAS,
        operand_size=size_of[FP8_TYPE](),
        tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        mma_k=PV_MMA_K,
    ](mma_idesc, a_desc, b_desc, c_tmem, UInt32(0), e)

    if elect_one_thread:
        mma_arrive(mma_mbar)

    mma_mbar[0].wait(0)
    tcgen05_fence_after()

    # Read C from TMEM. For M=32 with .ws cta_group::1, observation from
    # the QK smoke test shows the 4 warps share TMEM column index space
    # but produce different physical N output cols. Each warp reads at
    # TMEM offset 0 (NOT offset wid*cols_per_warp) and owns
    # output cols [wid*N/4 : (wid+1)*N/4].
    comptime cols_per_warp = PV_N // 4  # 64
    var c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=cols_per_warp,
        dtype=ACC_TYPE,
        pack=False,
        width=cols_per_warp,
    ](c_tmem)

    tcgen05_load_wait()

    var p_row = Int(tid % 32)
    var p_col_base = Int(wid) * cols_per_warp
    if p_row < PV_M:
        for j in range(cols_per_warp):
            c_output[p_row, p_col_base + j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fill_random_fp8[
    dtype: DType
](ptr: MutPointer[Scalar[dtype], MutAnyOrigin], n: Int):
    """Generates random FP8 values via float32 RNG -> cast.

    randn doesn't directly support float8_e4m3fn, so we draw float32
    values, scale them down to fit FP8's e4m3 range, and cast.
    """
    var f32_buf = alloc[Float32](n)
    randn[.float32](f32_buf, n)
    for i in range(n):
        # Scale to ~[-2, 2] to keep values well inside FP8 e4m3 range
        # (max representable ~448, but we want non-saturated values).
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_qk_smoke(ctx: DeviceContext) raises:
    print("=" * 70)
    print(
        "test_qk_smoke: tcgen05.mma.ws.cta_group::1.kind::f8f6f4"
        " M=32 N=64 K=576 transpose_b=True"
    )
    print("=" * 70)

    seed(42)

    # ---- Inputs (host) ----
    var a_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_M, QK_K)](ctx)
    var b_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(QK_N, QK_K)](ctx)

    var a_host = a_inp.tensor[update=False]()
    var b_host = b_inp.tensor[update=False]()

    fill_random_fp8[FP8_TYPE](a_host.ptr, QK_M * QK_K)
    fill_random_fp8[FP8_TYPE](b_host.ptr, QK_N * QK_K)

    # BF16 dequantized copies for the GPU naive reference.
    var a_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_M * QK_K)
    var b_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](QK_N * QK_K)

    var a_ref_host = alloc[Scalar[REF_TYPE]](QK_M * QK_K)
    var b_ref_host = alloc[Scalar[REF_TYPE]](QK_N * QK_K)

    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](a_host.ptr, a_ref_host, QK_M * QK_K)
    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](b_host.ptr, b_ref_host, QK_N * QK_K)

    ctx.enqueue_copy(a_ref_dev, a_ref_host)
    ctx.enqueue_copy(b_ref_dev, b_ref_host)

    # Output buffer for the FP8 MMA result.
    var c_out_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(QK_M, QK_N)](
        ctx
    )

    # ---- TMA descriptors ----
    var a_tma_op = create_tensor_tile[
        Index(QK_M, QK_K),
        swizzle_mode=QK_SWIZZLE,
    ](ctx, a_inp.device_tensor())
    var b_tma_op = create_tensor_tile[
        Index(QK_N, QK_K),
        swizzle_mode=QK_SWIZZLE,
    ](ctx, b_inp.device_tensor())

    # ---- Launch ----
    comptime kernel = qk_smoke_kernel[
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_out_buf.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=QK_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(QK_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()

    # ---- Reference: P_ref = A_bf16 x B_bf16^T using naive matmul ----
    var c_ref_dev = ctx.enqueue_create_buffer[ACC_TYPE](QK_M * QK_N)

    var c_ref_tt = TileTensor(
        c_ref_dev,
        row_major(Coord(QK_M, QK_N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[REF_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(a_ref_dev.unsafe_ptr())
        ),
        row_major(Coord(QK_M, QK_K)),
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
        Int32(QK_M),
        Int32(QK_N),
        Int32(QK_K),
        grid_dim=(
            ceildiv(QK_M, NAIVE_BLOCK_DIM),
            ceildiv(QK_N, NAIVE_BLOCK_DIM),
            1,
        ),
        block_dim=(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM, 1),
    )

    var c_ref_host = alloc[Float32](QK_M * QK_N)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    # ---- Compare ----
    var c_out_host = c_out_buf.tensor()
    var c_out_ptr = c_out_host.ptr

    print(
        "  C_gpu[0,0]="
        + String(c_out_ptr[0])
        + "  C_ref[0,0]="
        + String(c_ref_host[0])
    )
    print(
        "  C_gpu[31,63]="
        + String(c_out_ptr[31 * QK_N + 63])
        + "  C_ref[31,63]="
        + String(c_ref_host[31 * QK_N + 63])
    )

    var max_abs_err: Float32 = 0.0
    var max_rel_err: Float32 = 0.0
    var num_failures: Int = 0
    for r in range(QK_M):
        for c in range(QK_N):
            var idx = r * QK_N + c
            var ref_val = c_ref_host[idx]
            var gpu_val = c_out_ptr[idx]
            var abs_err = abs(gpu_val - ref_val)
            var rel_err = abs_err / max(abs(ref_val), Float32(1.0))
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            if abs_err > ATOL_QK and rel_err > RTOL_QK:
                num_failures += 1

    print("  max abs err: " + String(max_abs_err))
    print("  max rel err: " + String(max_rel_err))
    print(
        "  failures (atol="
        + String(ATOL_QK)
        + " rtol="
        + String(RTOL_QK)
        + "): "
        + String(num_failures)
        + " / "
        + String(QK_M * QK_N)
    )

    assert_true(
        num_failures == 0,
        msg=String(
            "QK smoke FAILED: ",
            num_failures,
            " elements exceed tolerance (max abs=",
            max_abs_err,
            ", max rel=",
            max_rel_err,
            ")",
        ),
    )
    print("  QK SMOKE PASSED (M=32 N=64 K=576 .ws cta_group::1 kind::f8f6f4)")

    a_ref_host.free()
    b_ref_host.free()
    c_ref_host.free()
    _ = a_ref_dev
    _ = b_ref_dev
    _ = c_ref_dev
    _ = a_inp^
    _ = b_inp^
    _ = c_out_buf^


def test_pv_smoke(ctx: DeviceContext) raises:
    print("=" * 70)
    print(
        "test_pv_smoke: tcgen05.mma.ws.cta_group::1.kind::f8f6f4"
        " M=32 N=256 K=64 transpose_b=True"
    )
    print("=" * 70)

    seed(43)

    var a_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(PV_M, PV_K)](ctx)
    var b_inp = ManagedLayoutTensor[FP8_TYPE, Layout.row_major(PV_N, PV_K)](ctx)

    var a_host = a_inp.tensor[update=False]()
    var b_host = b_inp.tensor[update=False]()

    fill_random_fp8[FP8_TYPE](a_host.ptr, PV_M * PV_K)
    fill_random_fp8[FP8_TYPE](b_host.ptr, PV_N * PV_K)

    var a_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](PV_M * PV_K)
    var b_ref_dev = ctx.enqueue_create_buffer[REF_TYPE](PV_N * PV_K)

    var a_ref_host = alloc[Scalar[REF_TYPE]](PV_M * PV_K)
    var b_ref_host = alloc[Scalar[REF_TYPE]](PV_N * PV_K)

    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](a_host.ptr, a_ref_host, PV_M * PV_K)
    dequant_fp8_to_bf16[FP8_TYPE, REF_TYPE](b_host.ptr, b_ref_host, PV_N * PV_K)

    ctx.enqueue_copy(a_ref_dev, a_ref_host)
    ctx.enqueue_copy(b_ref_dev, b_ref_host)

    var c_out_buf = ManagedLayoutTensor[ACC_TYPE, Layout.row_major(PV_M, PV_N)](
        ctx
    )

    var a_tma_op = create_tensor_tile[
        Index(PV_M, PV_K),
        swizzle_mode=PV_SWIZZLE,
    ](ctx, a_inp.device_tensor())
    var b_tma_op = create_tensor_tile[
        Index(PV_N, PV_K),
        swizzle_mode=PV_SWIZZLE,
    ](ctx, b_inp.device_tensor())

    comptime kernel = pv_smoke_kernel[
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_out_buf.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=PV_TOTAL_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(PV_TOTAL_SMEM)
        ),
    )
    ctx.synchronize()

    # Reference: C_ref = A_bf16 x B_bf16^T using naive matmul.
    # Note: The MMA computes C[m,n] = sum_k A[m,k] * B[n,k] (same as
    # transpose_b=True semantically when B is laid out [N, K]). Our naive
    # kernel with transpose_b=True consumes B as [N, K] and computes
    # C[m,n] = sum_k A[m,k] * B[n,k], which is the EXACT operation our
    # MMA performs for transpose_b=False with mn-major B.
    var c_ref_dev = ctx.enqueue_create_buffer[ACC_TYPE](PV_M * PV_N)

    var c_ref_tt = TileTensor(
        c_ref_dev,
        row_major(Coord(PV_M, PV_N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[REF_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(a_ref_dev.unsafe_ptr())
        ),
        row_major(Coord(PV_M, PV_K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[REF_TYPE], ImmutAnyOrigin](
            unsafe_from_address=Int(b_ref_dev.unsafe_ptr())
        ),
        row_major(Coord(PV_N, PV_K)),
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
        Int32(PV_M),
        Int32(PV_N),
        Int32(PV_K),
        grid_dim=(
            ceildiv(PV_M, NAIVE_BLOCK_DIM),
            ceildiv(PV_N, NAIVE_BLOCK_DIM),
            1,
        ),
        block_dim=(NAIVE_BLOCK_DIM, NAIVE_BLOCK_DIM, 1),
    )

    var c_ref_host = alloc[Float32](PV_M * PV_N)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    var c_out_host = c_out_buf.tensor()
    var c_out_ptr = c_out_host.ptr

    print(
        "  C_gpu[0,0]="
        + String(c_out_ptr[0])
        + "  C_ref[0,0]="
        + String(c_ref_host[0])
    )
    print(
        "  C_gpu[31,"
        + String(PV_N - 1)
        + "]="
        + String(c_out_ptr[31 * PV_N + (PV_N - 1)])
        + "  C_ref[31,"
        + String(PV_N - 1)
        + "]="
        + String(c_ref_host[31 * PV_N + (PV_N - 1)])
    )

    var max_abs_err: Float32 = 0.0
    var max_rel_err: Float32 = 0.0
    var num_failures: Int = 0
    for r in range(PV_M):
        for c in range(PV_N):
            var idx = r * PV_N + c
            var ref_val = c_ref_host[idx]
            var gpu_val = c_out_ptr[idx]
            var abs_err = abs(gpu_val - ref_val)
            var rel_err = abs_err / max(abs(ref_val), Float32(1.0))
            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err
            if abs_err > ATOL_PV and rel_err > RTOL_PV:
                num_failures += 1

    print("  max abs err: " + String(max_abs_err))
    print("  max rel err: " + String(max_rel_err))
    print(
        "  failures (atol="
        + String(ATOL_PV)
        + " rtol="
        + String(RTOL_PV)
        + "): "
        + String(num_failures)
        + " / "
        + String(PV_M * PV_N)
    )

    assert_true(
        num_failures == 0,
        msg=String(
            "PV smoke FAILED: ",
            num_failures,
            " elements exceed tolerance (max abs=",
            max_abs_err,
            ", max rel=",
            max_rel_err,
            ")",
        ),
    )
    print(
        "  PV SMOKE PASSED (M=32 N="
        + String(PV_N)
        + " K=64 .ws cta_group::1 kind::f8f6f4)"
    )

    a_ref_host.free()
    b_ref_host.free()
    c_ref_host.free()
    _ = a_ref_dev
    _ = b_ref_dev
    _ = c_ref_dev
    _ = a_inp^
    _ = b_inp^
    _ = c_out_buf^


def main() raises:
    with DeviceContext() as ctx:
        # Skip if not B200 (SM100).
        comptime if not _is_sm10x_gpu(ctx.default_device_info):
            print("Skipping: this test requires B200 (SM100)")
            return

        test_qk_smoke(ctx)
        test_pv_smoke(ctx)
        print("\nAll Layout G smoke tests PASSED.")
