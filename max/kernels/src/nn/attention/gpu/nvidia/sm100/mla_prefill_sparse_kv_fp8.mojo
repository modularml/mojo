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
"""SM100 (B200) sparse MLA prefill kernel with FP8 KV cache.

Thin FP8-KV variant mirroring `mla_prefill_sparse.mojo` (BF16 KV).  K/V latents
are read as INT64-packed FP8 via SWIZZLE_NONE gather4 TMA, dequantized to BF16
in SMEM (`convert_k/v_fp8_to_bf16`), then fed to the shared QK/SV MMA pipeline
(`MLAPrefillSparseCommon.mma[fp8_active=True]`).  Q is BF16 in both paths, so
the Q-load prologue and the MMA/softmax/correction machinery are shared through
`MLAPrefillSparseCommon` in `mla_prefill_sparse_utils.mojo`.
"""

from std.sys import size_of
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    warp_id,
    thread_idx,
    WARP_SIZE,
)
from max.gpu.sync import barrier
from std.math import ceildiv, exp2
from std.math.constants import log2e
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.primitives.cluster import cluster_sync
import max.gpu.primitives.warp as warp
from max.gpu.memory import (
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
    fence_mbarrier_init,
    fence_async_view_proxy,
)
from max.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceContext, FuncAttribute
from std.ffi import UnsafeUnion
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_cp,
    tcgen05_store_wait,
    tcgen05_fence_after,
    tcgen05_fence_before,
)

from nn.attention.mha_operand import MHAOperand, KVCacheMHAOperand
from nn.attention.mha_mask import MHAMask
from kv_cache.types import KVCacheT
from nn.attention.mha_utils import OptionallyStaticInt, MHAPartitionScheme
from nn.attention.gpu.nvidia.common import (
    elect,
    OptionalPointer,
    NullPointer,
    NonNullPointer,
)

from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.mha import q_num_matrix_view_rows

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    add_ftz,
    sub_ftz,
    mul_ftz,
    fma_ftz,
    exp2_emulation,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from linalg.arch.sm100.mma import smem_descriptor


from layout import (
    TileTensor,
    row_major,
    Idx,
    TensorLayout,
    TensorEngine,
    Coord,
    stack_allocation as tt_stack_allocation,
)
from layout.swizzle import make_swizzle
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    ld_shared_v4_u32,
    cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4,
    st_shared_v4_b32_at_bf16_elem_off,
    hmul2_bf16x8_by_scalar,
)
from std.memory import bitcast
from layout.tma_async import (
    create_tensor_tile,
    TMATensorTile,
    SharedMemBarrier,
    RaggedTMA3DTile,
    _gather4_box_width,
    _default_desc_shape,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.utils.numerics import min_or_neg_inf


from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLAPrefillSparseCommon,
    MLASparseConfig,
    MLASparseSharedMemory,
    QKMMAOp,
    SVMMAType,
)


# FP8 variant of MLASparseSharedMemory.  Embeds the BF16 struct as its
# first field (guaranteed offset 0) so all K/V/Q SMEM regions are at
# identical byte offsets — the bitcast in kernel_fp8() stays correct.
# FP8 scales and extra mbarriers are appended after.
#
# scale_block_size must be > 0 (the kernel assertion enforces this).
# K scales: TOPK_PER_CTA rows (K is CTA-split; 64 rows per CTA).
# V scales: B_TOPK rows (V is NOT CTA-split for scales; 128 rows per CTA).
#
# FP8 K staging: FP8 K data is loaded into the upper half of K SMEM
# (k_smem_ptr + K_SIZE bytes, i.e. bytes [36864..73727] of K SMEM).
# convert_k_fp8_to_bf16 reads ALL FP8 for both rows (0..31 and 32..63)
# into register staging arrays before issuing any BF16 writes, avoiding
# the SWIZZLE_128B aliasing that would otherwise corrupt rows 32..63.
# No dedicated buffer is needed — the total staging (144 u32) fits in
# WG1's 168-register budget.
struct MLASparseSharedMemoryFP8[config: MLASparseConfig, scale_block_size: Int]:
    comptime num_mbars = 2
    comptime TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    # K has qk_depth columns; V has only v_depth columns — use distinct counts.
    # scale_block_size == 0 => no-scale (unit-scale) mode: the k_scales/v_scales
    # arrays are unused; keep them size 1 (non-empty) to stay legal.
    comptime K_scales_per_token = ceildiv(
        Self.config.qk_depth, Self.scale_block_size
    ) if Self.scale_block_size > 0 else 1
    comptime V_scales_per_token = ceildiv(
        Self.config.v_depth, Self.scale_block_size
    ) if Self.scale_block_size > 0 else 1
    comptime K_SCALES_SIZE = Self.TOPK_PER_CTA * Self.K_scales_per_token
    comptime V_SCALES_SIZE = Self.config.B_TOPK * Self.V_scales_per_token

    var base: MLASparseSharedMemory[Self.config]
    var k_scales: Array[Float32, Self.K_SCALES_SIZE]
    var v_scales: Array[Float32, Self.V_SCALES_SIZE]
    var k_fp8_tma_done: Array[SharedMemBarrier, Self.num_mbars]
    var v_fp8_tma_done: Array[SharedMemBarrier, Self.num_mbars]


struct MLAPrefillSparseFP8[
    KVLUTType: MHAOperand,
    output_dtype: DType,
    config: MLASparseConfig,
](TrivialRegisterPassable):
    comptime qkv_dtype = Self.config.qkv_dtype
    comptime accum_dtype = DType.float32

    comptime q_smem_depth = Self.config.q_smem_depth
    comptime q_tmem_depth = Self.config.q_tmem_depth
    comptime qkv_dtype_size = size_of[Self.qkv_dtype]()

    comptime NUM_Q_HEADS_PER_CTA = Self.config.num_q_heads // Self.config.cta_group
    # Padded per-CTA head rows = the MMA M-tile per CTA (always 64: 64/1 at
    # head<=64, 128/2 at head128). Use this for geometry that must span the
    # full padded MMA tile (O TMEM read stride, S-scratch p1 offset, cp_q Q-row
    # stride); use NUM_Q_HEADS_PER_CTA for the real rows to load/store. Equal at
    # 64/128 (NFC); differ only when num_q_heads < 64.
    comptime PADDED_HEADS_PER_CTA = (
        Self.config.padded_num_q_heads // Self.config.cta_group
    )
    comptime B_TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    comptime NUM_SV_ATOMS = 2
    comptime SV_ATOM_MMA_N = Self.config.v_depth // Self.NUM_SV_ATOMS
    comptime V_DEPTH_PER_CTA = Self.SV_ATOM_MMA_N
    comptime V_BMN_PER_ATOM = Self.SV_ATOM_MMA_N // Self.config.cta_group
    comptime O_ATOM_PHYS_COLS = Self.SV_ATOM_MMA_N // 2
    comptime V_SMEM_COLS_PER_CTA = Self.V_BMN_PER_ATOM * Self.NUM_SV_ATOMS

    # The Q TMA box requests the REAL per-CTA head count so every sub-copy
    # reads only in-bounds GMEM heads (a padded box makes heads
    # [num_q_heads, 64) a middle-dimension OOB region whose TMA never
    # completes its mbarrier transaction -> deadlock). For num_q_heads < 64
    # the depth-tile sub-copies are issued manually so each real-head box
    # lands at the PADDED (64-row) depth-tile stride the BMN=64 QK-MMA reads
    # (see the Q-load in `kernel`). NFC at 64/128 (real == padded).
    comptime q_tile_shape = Index(
        1, Self.NUM_Q_HEADS_PER_CTA, Self.config.input_qk_depth
    )
    comptime q_desc_shape = _default_desc_shape[
        3, Self.qkv_dtype, Self.q_tile_shape, Self.config.q_swizzle_mode
    ]()

    comptime k_tile_width = Self.config.qk_depth
    comptime k_swizzle_mode = Self.config.k_swizzle_mode
    comptime k_tile_height = Self.B_TOPK_PER_CTA
    comptime k_gather_box = _gather4_box_width[
        Self.qkv_dtype, Self.k_tile_width, Self.k_swizzle_mode
    ]()
    comptime k_tile_shape = Index(Self.k_tile_height, Self.k_gather_box)
    comptime k_desc_shape = Index(1, Self.k_gather_box)

    comptime v_tile_width = Self.V_SMEM_COLS_PER_CTA
    # V uses SWIZZLE_128B; gather4 splits tile_width=128 into 2 col-groups
    # of 64 bf16 each (box_width = 128B / sizeof(bf16) = 64).
    comptime v_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    # tile_height is HALF of B_TOPK: each call to async_copy_gather4_tile
    # loads one K-half (64 rows × V_SMEM_COLS_PER_CTA cols), and load_v
    # invokes it twice with different smem bases. With tile_height=64 the
    # gather4 col-group stride (tile_height*box_w = 4096 elems) matches
    # the SW128 BMN=128 MN-major descriptor's mn_outer stride.
    comptime v_tile_height = Self.config.B_TOPK // 2
    comptime v_gather_box = _gather4_box_width[
        Self.qkv_dtype, Self.v_tile_width, Self.v_swizzle_mode
    ]()
    comptime v_tile_shape = Index(Self.v_tile_height, Self.v_gather_box)
    comptime v_desc_shape = Index(1, Self.v_gather_box)

    comptime o_tile_shape = Index(Self.NUM_Q_HEADS_PER_CTA, Self.config.v_depth)
    comptime o_desc_shape = Index(Self.NUM_Q_HEADS_PER_CTA, 64)

    # FP8 TMA swizzle modes. SWIZZLE_NONE paired with INT64 packing means one
    # gather4 descriptor covers the full row without inner col-tiling.
    comptime FP8_K_SWIZZLE = TensorMapSwizzle.SWIZZLE_NONE
    comptime FP8_V_SWIZZLE = TensorMapSwizzle.SWIZZLE_NONE

    # FP8 K TMA: INT64-packed, one descriptor covers 576 FP8 bytes/row.
    # BF16 path uses k_tile_shape/k_desc_shape; FP8 path uses these.
    # Kept separate because `DType.int64 if fp8 else qkv_dtype` trips a
    # Mojo parser bug in comptime positions.
    comptime k_tma_dtype_fp8 = DType.int64
    comptime k_tma_tile_width_fp8 = Self.config.qk_depth // 8
    comptime k_tma_swizzle_fp8 = Self.FP8_K_SWIZZLE
    comptime k_tma_gather_box_fp8 = _gather4_box_width[
        Self.k_tma_dtype_fp8,
        Self.k_tma_tile_width_fp8,
        Self.k_tma_swizzle_fp8,
    ]()
    comptime k_tma_tile_shape_fp8 = Index(
        Self.k_tile_height, Self.k_tma_gather_box_fp8
    )
    comptime k_tma_desc_shape_fp8 = Index(1, Self.k_tma_gather_box_fp8)

    # FP8 V TMA: INT64-packed, per-atom width (V_BMN_PER_ATOM / 8 INT64 elems).
    # Two gather4 calls per row group (one per SV atom) because the atoms'
    # gmem columns are not contiguous.
    comptime v_tma_dtype_fp8 = DType.int64
    comptime v_tma_tile_width_fp8 = Self.V_BMN_PER_ATOM // 8
    comptime v_tma_swizzle_fp8 = Self.FP8_V_SWIZZLE
    comptime v_tma_gather_box_fp8 = _gather4_box_width[
        Self.v_tma_dtype_fp8,
        Self.v_tma_tile_width_fp8,
        Self.v_tma_swizzle_fp8,
    ]()
    # FP8 V loads all B_TOPK rows at once (no key-half split like BF16).
    comptime v_tma_tile_height_fp8 = Self.config.B_TOPK
    comptime v_tma_tile_shape_fp8 = Index(
        Self.v_tma_tile_height_fp8, Self.v_tma_gather_box_fp8
    )
    comptime v_tma_desc_shape_fp8 = Index(1, Self.v_tma_gather_box_fp8)

    comptime SMemType = MLASparseSharedMemory[Self.config]
    comptime FULL_Q_TYPE = Self.SMemType.FULL_Q_TYPE
    comptime SHARED_QKV_TYPE = Self.SMemType.SHARED_QKV_TYPE
    comptime O_TYPE = Self.SMemType.O_TYPE

    comptime QKMMAOpType = QKMMAOp[
        Self.qkv_dtype, Self.accum_dtype, Self.config
    ]
    comptime SVMMAType = SVMMAType[
        Self.qkv_dtype, Self.accum_dtype, Self.config
    ]

    comptime O_TMEM_ADDR = 0
    # atom2's O sits right after atom1's, offset by the physical footprint
    # O_ATOM_PHYS_COLS (not the V-operand width V_BMN_PER_ATOM).
    comptime O_TMEM_ADDR_ATOM2 = Self.O_TMEM_ADDR + Self.O_ATOM_PHYS_COLS
    comptime P_TMEM_ADDR = 256
    comptime Q_TMEM_ADDR = 512 - Self.q_tmem_depth // 2

    # tcgen05.commit multicast mask for MMA-completion barriers: signal both
    # CTAs in the pair (0b11) at cta_group=2, only self (0b1) at cta_group=1.
    comptime CTA_MASK: UInt16 = 0b11 if Self.config.cta_group == 2 else 0b1

    comptime Common = MLAPrefillSparseCommon[
        Self.KVLUTType, Self.output_dtype, Self.config
    ]

    # ------------------------------------------------------------------
    # FP8 KV-cache helpers (used by kernel_fp8 only)
    # ------------------------------------------------------------------

    @always_inline
    @staticmethod
    def load_k_fp8_tma(
        k_tma_op_fp8: TMATensorTile[
            Self.k_tma_dtype_fp8,
            2,
            Self.k_tma_tile_shape_fp8,
            Self.k_tma_desc_shape_fp8,
        ],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        k_smem_fp8_ptr: UnsafePointer[
            mut=True, Float8_e4m3fn, address_space=.SHARED, ...
        ],
        k_fp8_tma_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k: UInt32,
        cta_id: UInt32,
        warp_idx: UInt32,
        indices_base: UInt32,
    ):
        # Load all TOPK_PER_CTA=64 rows of FP8 K into the upper half of K SMEM
        # (k_smem_fp8_ptr = k_smem_ptr + K_SIZE FP8 bytes).
        comptime num_warps = 4
        comptime num_rows_per_warp = 4

        var k_smem_i64 = k_smem_fp8_ptr.bitcast[Int64]()
        var k_smem_tensor_i64 = TileTensor(
            k_smem_i64,
            row_major[Self.B_TOPK_PER_CTA, Self.k_tma_tile_width_fp8](),
        )

        comptime for local_row in range(num_rows_per_warp):
            var indices_offset = (
                indices_base
                + UInt32(k) * UInt32(Self.config.B_TOPK)
                + cta_id * UInt32(Self.config.B_TOPK // Self.config.cta_group)
                + (UInt32(local_row) * UInt32(num_warps) + warp_idx) * UInt32(4)
            )
            var idx_v4 = Self.Common._raw_indices_to_tma_rows(
                kv_lut,
                indices.load[width=4](Coord(indices_offset)).cast[
                    DType.int32
                ](),
            )

            var warp_tile = k_smem_tensor_i64.tile[
                4, Self.k_tma_tile_width_fp8
            ](
                Coord(
                    UInt32(local_row) * UInt32(num_warps) + warp_idx,
                    Idx[0],
                )
            )
            k_tma_op_fp8.async_copy_gather4[cta_group=1](
                warp_tile,
                k_fp8_tma_done[],
                Int32(0),
                idx_v4[0],
                idx_v4[1],
                idx_v4[2],
                idx_v4[3],
            )

    @always_inline
    @staticmethod
    def load_k_scales_to_smem[
        scale_block_size: Int
    ](
        scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        k_scales_smem_ptr: UnsafePointer[
            mut=True, Float32, address_space=.SHARED, ...
        ],
        k: UInt32,
        cta_id: UInt32,
        indices_base: UInt32,
        num_kv_rows: Int32,
    ):
        comptime scales_per_token = ceildiv(
            Self.config.qk_depth, scale_block_size
        )
        comptime K_SCALES_SIZE = Self.B_TOPK_PER_CTA * scales_per_token
        var tid = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)
        var topk_base = (
            indices_base
            + UInt32(k) * UInt32(Self.config.B_TOPK)
            + cta_id * UInt32(Self.config.B_TOPK // Self.config.cta_group)
        )

        var i = tid
        while i < UInt32(K_SCALES_SIZE):
            var row = i // UInt32(scales_per_token)
            var block = i % UInt32(scales_per_token)
            var raw = indices.load[width=1](Coord(topk_base + row)).cast[
                DType.int32
            ]()
            var ok = raw >= Int32(0) and raw < num_kv_rows
            var scale_val: Float32 = 0.0
            if ok:
                var phys = Int(kv_lut.get_tma_row(raw))
                scale_val = scales_ptr[phys * scales_per_token + Int(block)]
            k_scales_smem_ptr[Int(i)] = scale_val
            i += UInt32(WARPGROUP_SIZE)

    @always_inline
    @staticmethod
    def convert_k_fp8_to_bf16[
        scale_block_size: Int
    ](
        k_smem_fp8_ptr: UnsafePointer[
            mut=True, Float8_e4m3fn, address_space=.SHARED, ...
        ],
        k_smem_bf16_ptr: UnsafePointer[
            mut=True, Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        k_scales_smem_ptr: UnsafePointer[
            mut=True, Float32, address_space=.SHARED, ...
        ],
    ):
        # k_smem_fp8_ptr points to the upper half of K SMEM (FP8 staging).
        # SWIZZLE_128B scatters row-0 BF16 writes into col-groups 4..8 which
        # overlap the FP8 staging region for row-1 ([36864..73727] bytes).
        # Fix: pre-read ALL FP8 for BOTH rows into register staging arrays
        # before issuing any BF16 writes.
        # 2 rows per thread (TOPK_PER_CTA=64, NUM_GROUPS=32 → ROWS_PER_GROUP=2).
        # Total staging: 4 arrays × 4×9 u32 = 144 u32 (requires WG1=168 regs).
        comptime fp8_type = DType.float8_e4m3fn
        comptime bf16_type = Self.qkv_dtype
        # scale_block_size == 0 => no-scale mode (unit scale); guard the divisor.
        comptime scales_per_token = ceildiv(
            Self.config.qk_depth, scale_block_size
        ) if scale_block_size > 0 else 1
        comptime BN_QK = 64
        comptime GROUP_SIZE = 4
        comptime NUM_GROUPS = WARPGROUP_SIZE // GROUP_SIZE  # 32
        comptime COLS_PER_GROUP = Self.config.qk_depth // (GROUP_SIZE * 16)  # 9
        comptime FP8_ROW_STRIDE = Self.config.qk_depth  # 576 bytes/row
        comptime BLOCK_ELEMS = Self.B_TOPK_PER_CTA * BN_QK

        comptime sw_bf16 = make_swizzle[
            bf16_type, TensorMapSwizzle.SWIZZLE_128B
        ]()

        var lane = UInt32(thread_idx.x) & UInt32(0x7F)
        var group_idx = lane // UInt32(GROUP_SIZE)
        var idx_in_group = lane % UInt32(GROUP_SIZE)

        # 2 rows per thread covering all TOPK_PER_CTA=64 rows:
        #   row_0 (0..31), row_1 (32..63)
        var row_0 = group_idx
        var row_1 = group_idx + UInt32(NUM_GROUPS)

        var fp8_base_0 = row_0 * UInt32(FP8_ROW_STRIDE) + idx_in_group * UInt32(
            16
        )
        var fp8_base_1 = row_1 * UInt32(FP8_ROW_STRIDE) + idx_in_group * UInt32(
            16
        )

        var col_bf16 = idx_in_group * UInt32(16)
        var sw0_a = Int(sw_bf16(row_0 * UInt32(BN_QK) + col_bf16))
        var sw0_b = Int(sw_bf16(row_0 * UInt32(BN_QK) + col_bf16 + UInt32(8)))
        var sw1_a = Int(sw_bf16(row_1 * UInt32(BN_QK) + col_bf16))
        var sw1_b = Int(sw_bf16(row_1 * UInt32(BN_QK) + col_bf16 + UInt32(8)))

        var src_u8 = k_smem_fp8_ptr.bitcast[UInt8]()

        var p0a_all = tt_stack_allocation[
            dtype=DType.uint32, address_space=.LOCAL
        ](row_major[4, COLS_PER_GROUP]())
        var p0b_all = tt_stack_allocation[
            dtype=DType.uint32, address_space=.LOCAL
        ](row_major[4, COLS_PER_GROUP]())
        var p1a_all = tt_stack_allocation[
            dtype=DType.uint32, address_space=.LOCAL
        ](row_major[4, COLS_PER_GROUP]())
        var p1b_all = tt_stack_allocation[
            dtype=DType.uint32, address_space=.LOCAL
        ](row_major[4, COLS_PER_GROUP]())

        comptime for c in range(COLS_PER_GROUP):
            comptime col_byte_off = c * GROUP_SIZE * 16
            var q0 = ld_shared_v4_u32(src_u8, Int(fp8_base_0) + col_byte_off)
            var q1 = ld_shared_v4_u32(src_u8, Int(fp8_base_1) + col_byte_off)
            var pa0 = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                fp8_dtype=fp8_type, out_dtype=bf16_type
            ](q0[0], q0[1])
            var pb0 = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                fp8_dtype=fp8_type, out_dtype=bf16_type
            ](q0[2], q0[3])
            var pa1 = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                fp8_dtype=fp8_type, out_dtype=bf16_type
            ](q1[0], q1[1])
            var pb1 = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                fp8_dtype=fp8_type, out_dtype=bf16_type
            ](q1[2], q1[3])
            # scale_block_size == 0 => no-scale mode: keep the FP8→BF16 cast
            # unscaled (unit scale), mirroring the scale-less decode read.
            comptime if scale_block_size > 0:
                var abs_col = UInt32(
                    c * GROUP_SIZE * 16
                ) + idx_in_group * UInt32(16)
                var block_idx = abs_col // UInt32(scale_block_size)
                var s0_fp32 = k_scales_smem_ptr[
                    Int(row_0) * scales_per_token + Int(block_idx)
                ]
                var s1_fp32 = k_scales_smem_ptr[
                    Int(row_1) * scales_per_token + Int(block_idx)
                ]
                var s0_bits = UInt32(
                    bitcast[.uint16, 1](s0_fp32.cast[bf16_type]())
                )
                var s1_bits = UInt32(
                    bitcast[.uint16, 1](s1_fp32.cast[bf16_type]())
                )
                var s0_u32 = s0_bits | (s0_bits << 16)
                var s1_u32 = s1_bits | (s1_bits << 16)
                pa0 = hmul2_bf16x8_by_scalar[bf16_type](pa0, s0_u32)
                pb0 = hmul2_bf16x8_by_scalar[bf16_type](pb0, s0_u32)
                pa1 = hmul2_bf16x8_by_scalar[bf16_type](pa1, s1_u32)
                pb1 = hmul2_bf16x8_by_scalar[bf16_type](pb1, s1_u32)
            p0a_all.ptr.store(c * 4, pa0)
            p0b_all.ptr.store(c * 4, pb0)
            p1a_all.ptr.store(c * 4, pa1)
            p1b_all.ptr.store(c * 4, pb1)

        named_barrier[Int32(WARPGROUP_SIZE)](3)

        comptime for c in range(COLS_PER_GROUP):
            var pa0 = p0a_all.ptr.load[width=4](c * 4)
            var pb0 = p0b_all.ptr.load[width=4](c * 4)
            var pa1 = p1a_all.ptr.load[width=4](c * 4)
            var pb1 = p1b_all.ptr.load[width=4](c * 4)
            var dst_block = k_smem_bf16_ptr + c * BLOCK_ELEMS
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=bf16_type](
                dst_block, sw0_a, pa0
            )
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=bf16_type](
                dst_block, sw0_b, pb0
            )
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=bf16_type](
                dst_block, sw1_a, pa1
            )
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=bf16_type](
                dst_block, sw1_b, pb1
            )

        fence_async_view_proxy()

    @always_inline
    @staticmethod
    def load_v_fp8_tma(
        v_tma_op_fp8: TMATensorTile[
            Self.v_tma_dtype_fp8,
            2,
            Self.v_tma_tile_shape_fp8,
            Self.v_tma_desc_shape_fp8,
        ],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        v_smem_fp8_ptr: UnsafePointer[
            mut=True, Float8_e4m3fn, address_space=.SHARED, ...
        ],
        v_fp8_tma_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k: UInt32,
        cta_id: UInt32,
        warp_idx: UInt32,
        indices_base: UInt32,
    ):
        comptime num_warps = 4
        comptime ROWS_PER_GATHER = 4
        comptime num_rows_per_warp = (
            Self.config.B_TOPK
        ) // ROWS_PER_GATHER // num_warps

        var v_smem_i64 = v_smem_fp8_ptr.bitcast[Int64]()
        # gather4 writes its 4 gathered rows PACKED at the descriptor box
        # width (v_tma_tile_width_fp8 int64s per row) starting at dst.ptr;
        # the dst tile's row stride is NOT consulted. Stage atom-major --
        # [2 atoms][B_TOPK rows][V_BMN_PER_ATOM bytes] -- so each call's
        # packed 4-row write lands disjoint at exactly the box pitch.
        # (A row-major [B_TOPK, V_SMEM_COLS_PER_CTA] view here made the
        # atom0/atom1 calls of every 4-row group double-write 3/4 of the
        # group's bytes -- nondeterministic winner -- and never write the
        # tail: the FP8 nondeterminism root cause.)
        comptime ATOM_W_I64 = Self.V_BMN_PER_ATOM // 8
        var v_smem_tensor_i64 = TileTensor(
            v_smem_i64,
            row_major[2 * Self.config.B_TOPK, ATOM_W_I64](),
        )

        comptime for local_row in range(num_rows_per_warp):
            var indices_offset = (
                indices_base
                + UInt32(k) * UInt32(Self.config.B_TOPK)
                + (UInt32(local_row) * num_warps + warp_idx)
                * UInt32(ROWS_PER_GATHER)
            )
            var idx_v4 = Self.Common._raw_indices_to_tma_rows(
                kv_lut,
                indices.load[width=4](Coord(indices_offset)).cast[
                    DType.int32
                ](),
            )

            comptime for atom_idx in range(2):
                var warp_tile = v_smem_tensor_i64.tile[
                    ROWS_PER_GATHER, ATOM_W_I64
                ](
                    Coord(
                        UInt32(
                            atom_idx * (Self.config.B_TOPK // ROWS_PER_GATHER)
                        )
                        + UInt32(local_row) * num_warps
                        + warp_idx,
                        Idx[0],
                    )
                )
                var col_idx = Int32(
                    atom_idx * (Self.V_DEPTH_PER_CTA // 8)
                ) + Int32(cta_id * UInt32(Self.V_BMN_PER_ATOM // 8))
                v_tma_op_fp8.async_copy_gather4[cta_group=1](
                    warp_tile,
                    v_fp8_tma_done[],
                    col_idx,
                    idx_v4[0],
                    idx_v4[1],
                    idx_v4[2],
                    idx_v4[3],
                )

    @always_inline
    @staticmethod
    def load_v_scales_to_smem[
        scale_block_size: Int
    ](
        scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        v_scales_smem_ptr: UnsafePointer[
            mut=True, Float32, address_space=.SHARED, ...
        ],
        k: UInt32,
        indices_base: UInt32,
        num_kv_rows: Int32,
    ):
        # V is NOT CTA-split: both CTAs load all B_TOPK V rows (they are
        # differentiated by which columns they write, not which rows).
        # No cta_id parameter — all B_TOPK indices are loaded by every CTA.
        #
        # The SMEM V-scale layout uses V's own block count
        # (ceildiv(v_depth, sbs)), but the HBM read stride is the CACHE's
        # per-token count ceildiv(qk_depth, sbs): the cache stores one scale
        # vector per token spanning the full qk_depth latent, and V is the
        # first v_depth (nope) columns of it. So V reads cache blocks
        # [0, ceildiv(v_depth, sbs)) at the cache stride. At tensorwise both
        # counts collapse to 1 (NFC vs the prior single-stride code).
        comptime scales_per_token = ceildiv(
            Self.config.v_depth, scale_block_size
        )
        comptime cache_scales_per_token = ceildiv(
            Self.config.qk_depth, scale_block_size
        )
        comptime V_SCALES_SIZE = Self.config.B_TOPK * scales_per_token
        var tid = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)
        var topk_base = indices_base + UInt32(k) * UInt32(Self.config.B_TOPK)

        var i = tid
        while i < UInt32(V_SCALES_SIZE):
            var row = i // UInt32(scales_per_token)
            var block = i % UInt32(scales_per_token)
            var raw = indices.load[width=1](Coord(topk_base + row)).cast[
                DType.int32
            ]()
            var ok = raw >= Int32(0) and raw < num_kv_rows
            var scale_val: Float32 = 0.0
            if ok:
                var phys = Int(kv_lut.get_tma_row(raw))
                scale_val = scales_ptr[
                    phys * cache_scales_per_token + Int(block)
                ]
            v_scales_smem_ptr[Int(i)] = scale_val
            i += UInt32(WARPGROUP_SIZE)

    @always_inline
    @staticmethod
    def convert_v_fp8_to_bf16[
        scale_block_size: Int
    ](
        v_smem_fp8_ptr: UnsafePointer[
            mut=True, Float8_e4m3fn, address_space=.SHARED, ...
        ],
        v_smem_bf16_ptr: UnsafePointer[
            mut=True, Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        v_scales_smem_ptr: UnsafePointer[
            mut=True, Float32, address_space=.SHARED, ...
        ],
        cta_id: UInt32,
    ):
        comptime fp8_type = DType.float8_e4m3fn
        comptime bf16_type = Self.qkv_dtype
        # V occupies v_depth columns per token, not qk_depth.  The SMEM scale
        # layout holds ceildiv(v_depth, sbs) blocks per row (cache blocks
        # 0..ceildiv(v_depth,sbs)-1, loaded by load_v_scales_to_smem).
        # scale_block_size == 0 => no-scale mode (unit scale); guard the divisor.
        comptime scales_per_token = ceildiv(
            Self.config.v_depth, scale_block_size
        ) if scale_block_size > 0 else 1

        comptime GROUP_SIZE = 4
        comptime NUM_GROUPS = WARPGROUP_SIZE // GROUP_SIZE  # 32
        # 256 cols / (4 threads × 16) = 4 col-iterations per group.
        comptime COLS_PER_GROUP = Self.V_SMEM_COLS_PER_CTA // (
            GROUP_SIZE * 16
        )  # 4
        comptime FP8_ROW_STRIDE = Self.V_SMEM_COLS_PER_CTA  # 256 bytes/row

        comptime BN_QK = 64  # cg col width in BF16
        comptime CG_ROWS = Self.config.B_TOPK // 2
        comptime CG_ELEMS = CG_ROWS * BN_QK  # BF16 elems per cg block
        comptime KH_ELEMS = CG_ROWS * Self.V_SMEM_COLS_PER_CTA
        # Rows of a key-half one 4-lane group owns: 2 at head128, 1 at head64.
        comptime ROWS_PER_KH = CG_ROWS // NUM_GROUPS

        comptime sw_bf16 = make_swizzle[
            bf16_type, TensorMapSwizzle.SWIZZLE_128B
        ]()

        var lane = UInt32(thread_idx.x) & UInt32(0x7F)
        var group_idx = lane // UInt32(GROUP_SIZE)
        var idx_in_group = lane % UInt32(GROUP_SIZE)
        var col_bf16 = idx_in_group * UInt32(16)

        var src_u8 = v_smem_fp8_ptr.bitcast[UInt8]()

        # The SWIZZLE_128B BF16 writes clobber the aliased FP8 staging region,
        # so within a key-half every FP8 read must complete (barrier) before
        # any BF16 write.
        var pa_all = tt_stack_allocation[
            dtype=DType.uint32, address_space=.LOCAL
        ](row_major[ROWS_PER_KH * COLS_PER_GROUP, 4]())
        var pb_all = tt_stack_allocation[
            dtype=DType.uint32, address_space=.LOCAL
        ](row_major[ROWS_PER_KH * COLS_PER_GROUP, 4]())

        # Convert the two key-halves in sequence to halve register footprint.
        comptime for kh in range(2):
            comptime for r in range(ROWS_PER_KH):
                # abs_row (absolute V smem row) indexes the FP8 read and
                # per-token scale lookup; rel_row (row within the key-half)
                # drives the swizzle in the write phase.
                var rel_row = group_idx + UInt32(r * NUM_GROUPS)
                var abs_row = rel_row + UInt32(kh * CG_ROWS)
                comptime for c in range(COLS_PER_GROUP):
                    # Atom-major packed staging (see load_v_fp8_tma): byte
                    # (row, abs_col) lives at atom * B_TOPK * V_BMN_PER_ATOM
                    # + row * V_BMN_PER_ATOM + abs_col % V_BMN_PER_ATOM.
                    # Every 64-col group c sits inside one atom, so the atom
                    # index is comptime.
                    comptime col_group_base = c * GROUP_SIZE * 16
                    comptime atom = col_group_base // Self.V_BMN_PER_ATOM
                    comptime atom_base = (
                        atom * Self.config.B_TOPK * Self.V_BMN_PER_ATOM
                        + col_group_base
                        - atom * Self.V_BMN_PER_ATOM
                    )
                    var q = ld_shared_v4_u32(
                        src_u8,
                        Int(
                            abs_row * UInt32(Self.V_BMN_PER_ATOM)
                            + idx_in_group * UInt32(16)
                        )
                        + atom_base,
                    )
                    var pa = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                        fp8_dtype=fp8_type, out_dtype=bf16_type
                    ](q[0], q[1])
                    var pb = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                        fp8_dtype=fp8_type, out_dtype=bf16_type
                    ](q[2], q[3])

                    # scale_block_size == 0 => no-scale mode: keep the FP8→BF16
                    # cast unscaled (unit scale), mirroring the scale-less
                    # decode read.
                    comptime if scale_block_size > 0:
                        var abs_col = UInt32(
                            c * GROUP_SIZE * 16
                        ) + idx_in_group * UInt32(16)
                        # `abs_col` is the SMEM column within THIS CTA's
                        # atom-major V slice. For blockwise scaling the scale
                        # index must be the block of the ABSOLUTE latent
                        # column: atom `atom` covers latent cols
                        # [atom*V_DEPTH_PER_CTA, ...) and CTA `cta_id` owns the
                        # [cta_id*V_BMN_PER_ATOM, +V_BMN_PER_ATOM) slice within
                        # that atom. (`atom*V_BMN_PER_ATOM` and
                        # `atom*V_DEPTH_PER_CTA` are comptime.) At cta_group==1
                        # V_DEPTH_PER_CTA==V_BMN_PER_ATOM and cta_id==0, so
                        # latent_col == abs_col — NFC vs the prior code; and at
                        # tensorwise block_idx is 0 regardless.
                        var latent_col = (
                            UInt32(atom * Self.V_DEPTH_PER_CTA)
                            + cta_id * UInt32(Self.V_BMN_PER_ATOM)
                            + (abs_col - UInt32(atom * Self.V_BMN_PER_ATOM))
                        )
                        var block_idx = latent_col // UInt32(scale_block_size)
                        var s_fp32 = v_scales_smem_ptr[
                            Int(abs_row) * scales_per_token + Int(block_idx)
                        ]
                        var s_bits = UInt32(
                            bitcast[.uint16, 1](s_fp32.cast[bf16_type]())
                        )
                        var s_u32 = s_bits | (s_bits << 16)
                        pa = hmul2_bf16x8_by_scalar[bf16_type](pa, s_u32)
                        pb = hmul2_bf16x8_by_scalar[bf16_type](pb, s_u32)

                    comptime slot = (r * COLS_PER_GROUP + c) * 4
                    pa_all.ptr.store(slot, pa)
                    pb_all.ptr.store(slot, pb)

            named_barrier[Int32(WARPGROUP_SIZE)](4)

            comptime for r in range(ROWS_PER_KH):
                var rel_row = group_idx + UInt32(r * NUM_GROUPS)
                var sw_a = Int(sw_bf16(rel_row * UInt32(BN_QK) + col_bf16))
                var sw_b = Int(
                    sw_bf16(rel_row * UInt32(BN_QK) + col_bf16 + UInt32(8))
                )
                comptime for c in range(COLS_PER_GROUP):
                    comptime slot = (r * COLS_PER_GROUP + c) * 4
                    var pa = pa_all.ptr.load[width=4](slot)
                    var pb = pb_all.ptr.load[width=4](slot)
                    var dst = v_smem_bf16_ptr + kh * KH_ELEMS + c * CG_ELEMS
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=bf16_type](
                        dst, sw_a, pa
                    )
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=bf16_type](
                        dst, sw_b, pb
                    )

        fence_async_view_proxy()

    # ------------------------------------------------------------------
    # kernel_fp8: FP8 KV-cache variant of kernel().
    # WG register budget (total 64,512 ≤ 65,536):
    #   WG0=dealloc[96], WG1=alloc[168], WG2=alloc[144], WG3=dealloc[96]
    # WG0/WG3 release regs (128→96) freeing the pool for WG1/WG2 to alloc.
    # WG1 needs 168 regs to stage 144 u32 of K FP8 before BF16 writes.
    # WG2 needs 144 regs to stage 128 u32 of V FP8 across 2 passes.
    # ------------------------------------------------------------------

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op_fp8, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op_fp8, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma_op, `nvvm.grid_constant`)
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](
            Int32(Self.config.cta_group), 1, 1
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"mla_prefill_sparse_fp8_{Self.qkv_dtype}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel_fp8[
        TopKLengthLayout: TensorLayout,
        IndicesLayout: TensorLayout,
        TopKLengthEngine: TensorEngine,
        IndicesEngine: TensorEngine,
        scale_block_size: Int,
    ](
        q_tma_op: TMATensorTile[
            Self.qkv_dtype,
            3,
            Self.q_tile_shape,
            Self.q_desc_shape,
        ],
        k_tma_op_fp8: TMATensorTile[
            Self.k_tma_dtype_fp8,
            2,
            Self.k_tma_tile_shape_fp8,
            Self.k_tma_desc_shape_fp8,
        ],
        v_tma_op_fp8: TMATensorTile[
            Self.v_tma_dtype_fp8,
            2,
            Self.v_tma_tile_shape_fp8,
            Self.v_tma_desc_shape_fp8,
        ],
        o_tma_op: TMATensorTile[
            Self.output_dtype,
            2,
            Self.o_tile_shape,
            Self.o_desc_shape,
        ],
        topk_lengths: TileTensor[
            .uint32, TopKLengthLayout, MutAnyOrigin, Engine=TopKLengthEngine
        ],
        indices: TileTensor[
            .uint32, IndicesLayout, MutAnyOrigin, Engine=IndicesEngine
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
        indices_stride: Int32,
        output_gmem_ptr: UnsafePointer[Scalar[Self.output_dtype], MutAnyOrigin],
        scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    ) where (topk_lengths.flat_rank == 1 and indices.flat_rank == 1):
        var cta_id = UInt32(block_idx.x % Self.config.cta_group)
        var seq_idx = UInt32(block_idx.x // Self.config.cta_group)
        var warp_idx = warp_id()
        var lane_idx = thread_idx.x % WARP_SIZE
        var warpgroup_idx = warp.broadcast(thread_idx.x // WARPGROUP_SIZE)
        var top_k_length = topk_lengths.load[width=1](Coord(seq_idx))
        var num_k_blocks = max(
            ceildiv(top_k_length, UInt32(Self.config.B_TOPK)), 1
        )
        var num_kv_rows = kv_lut.num_kv_rows()
        var indices_base = seq_idx * UInt32(indices_stride)

        if thread_idx.x == 0:
            q_tma_op.prefetch_descriptor()
            k_tma_op_fp8.prefetch_descriptor()
            v_tma_op_fp8.prefetch_descriptor()

        ref smem_fp8 = external_memory[
            UInt8, address_space=.SHARED, alignment=128
        ]().bitcast[MLASparseSharedMemoryFP8[Self.config, scale_block_size]]()[]
        ref smem = smem_fp8.base
        ref qkvo_union = smem.qkvo_union

        var full_q_ptr = qkvo_union.unsafe_get[Self.FULL_Q_TYPE]().unsafe_ptr()
        ref shared_qkv = qkvo_union.unsafe_get[Self.SHARED_QKV_TYPE]()
        var shared_qkv_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], origin_of(shared_qkv), address_space=.SHARED
        ] = shared_qkv.unsafe_ptr()
        var q_smem_ptr = shared_qkv_ptr
        var v_smem_ptr = shared_qkv_ptr + Self.SMemType.SHARED_Q_SIZE
        var k_smem_ptr = v_smem_ptr + Self.SMemType.V_SIZE
        ref o_union = qkvo_union.unsafe_get[Self.O_TYPE]()
        var o_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], origin_of(o_union), address_space=.SHARED
        ] = o_union.unsafe_ptr()
        var scores_ptr = smem.scores.unsafe_ptr()
        var p_ptr = smem.p.unsafe_ptr()
        var prologue_q_ptr = smem.prologue_q.unsafe_ptr()
        var prologue_q_cp_ptr = smem.prologue_q_cp.unsafe_ptr()
        var qk_ss_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.qk_ss_done), address_space=.SHARED
        ] = smem.qk_ss_done.unsafe_ptr()
        var qk_ts_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.qk_ts_done), address_space=.SHARED
        ] = smem.qk_ts_done.unsafe_ptr()
        var sv_p0_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.sv_p0_done), address_space=.SHARED
        ] = smem.sv_p0_done.unsafe_ptr()
        var sv_p1_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.sv_p1_done), address_space=.SHARED
        ] = smem.sv_p1_done.unsafe_ptr()
        var k_p0_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.k_p0_ready), address_space=.SHARED
        ] = smem.k_p0_ready.unsafe_ptr()
        var k_p1_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.k_p1_ready), address_space=.SHARED
        ] = smem.k_p1_ready.unsafe_ptr()
        var v_p0_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.v_p0_ready), address_space=.SHARED
        ] = smem.v_p0_ready.unsafe_ptr()
        var v_p1_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.v_p1_ready), address_space=.SHARED
        ] = smem.v_p1_ready.unsafe_ptr()
        var p_free_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.p_free), address_space=.SHARED
        ] = smem.p_free.unsafe_ptr()
        var so_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.so_ready), address_space=.SHARED
        ] = smem.so_ready.unsafe_ptr()
        var k_valid_ready_ptr: UnsafePointer[
            SharedMemBarrier,
            origin_of(smem.k_valid_ready),
            address_space=.SHARED,
        ] = smem.k_valid_ready.unsafe_ptr()
        var k_valid_free_ptr: UnsafePointer[
            SharedMemBarrier,
            origin_of(smem.k_valid_free),
            address_space=.SHARED,
        ] = smem.k_valid_free.unsafe_ptr()
        var is_k_valid_ptr: UnsafePointer[
            UInt8, origin_of(smem.is_k_valid), address_space=.SHARED
        ] = smem.is_k_valid.unsafe_ptr()
        var tmem_addr_ptr = smem.tmem_addr.unsafe_ptr()
        var rowwise_max_ptr: UnsafePointer[
            Float32, origin_of(smem.rowwise_max), address_space=.SHARED
        ] = smem.rowwise_max.unsafe_ptr()
        var rowwise_sum_ptr: UnsafePointer[
            Float32, origin_of(smem.rowwise_sum), address_space=.SHARED
        ] = smem.rowwise_sum.unsafe_ptr()

        # FP8-specific SMEM pointers.
        var k_scales_ptr = smem_fp8.k_scales.unsafe_ptr()
        var v_scales_ptr = smem_fp8.v_scales.unsafe_ptr()
        var k_fp8_tma_done_ptr: UnsafePointer[
            SharedMemBarrier,
            origin_of(smem_fp8.k_fp8_tma_done),
            address_space=.SHARED,
        ] = smem_fp8.k_fp8_tma_done.unsafe_ptr()
        var v_fp8_tma_done_ptr: UnsafePointer[
            SharedMemBarrier,
            origin_of(smem_fp8.v_fp8_tma_done),
            address_space=.SHARED,
        ] = smem_fp8.v_fp8_tma_done.unsafe_ptr()
        # FP8 K staging: upper half of K SMEM (k_smem_ptr + K_SIZE FP8 bytes).
        # FP8 V staging: upper half of V SMEM (v_smem_ptr + V_SIZE FP8 bytes).
        var k_smem_fp8_ptr = (
            k_smem_ptr.bitcast[Float8_e4m3fn]() + Self.SMemType.K_SIZE
        )
        var v_smem_fp8_ptr = (
            v_smem_ptr.bitcast[Float8_e4m3fn]() + Self.SMemType.V_SIZE
        )

        if warp_idx == 0:
            if elect_one_sync():
                prologue_q_ptr[].init(1)
                prologue_q_cp_ptr[].init(1)
                comptime for i in range(Self.SMemType.num_mbars):
                    qk_ss_done_ptr[i].init(1)
                    qk_ts_done_ptr[i].init(1)
                    sv_p0_done_ptr[i].init(1)
                    sv_p1_done_ptr[i].init(1)
                    # FP8: the converts arrive these manually (no TMA
                    # transaction). At cta_group=2 the MMA consumes BOTH
                    # CTAs' converted K/V, so CTA0's instance expects one
                    # arrival per CTA (arrive_cluster below).
                    k_p0_ready_ptr[i].init(Int32(Self.config.cta_group))
                    k_p1_ready_ptr[i].init(Int32(Self.config.cta_group))
                    v_p0_ready_ptr[i].init(Int32(Self.config.cta_group))
                    v_p1_ready_ptr[i].init(Int32(Self.config.cta_group))
                    p_free_ptr[i].init(
                        Int32(WARPGROUP_SIZE * Self.config.cta_group)
                    )
                    so_ready_ptr[i].init(
                        Int32(WARPGROUP_SIZE * Self.config.cta_group)
                    )
                    k_valid_ready_ptr[i].init(
                        Int32(Self.SMemType.NUM_KV_VALID_LANES)
                    )
                    k_valid_free_ptr[i].init(Int32(WARPGROUP_SIZE))
                # FP8 TMA arrival mbarriers — loop count from the FP8 struct
                # so a future change to num_mbars in MLASparseSharedMemoryFP8
                # doesn't silently under/over-initialize them.
                comptime fp8_num_mbars = MLASparseSharedMemoryFP8[
                    Self.config, scale_block_size
                ].num_mbars
                comptime for i in range(fp8_num_mbars):
                    k_fp8_tma_done_ptr[i].init(1)
                    v_fp8_tma_done_ptr[i].init(1)

                fence_mbarrier_init()

        cluster_sync()

        # Zero the padded Q SMEM (sub-64-head configs) and issue the Q TMA.
        # Shared with `kernel` (Q is BF16 in both KV-cache paths) so the
        # memset + manual depth-tile sub-copy fix lives in one place.
        Self.Common._load_q_prologue(
            full_q_ptr, q_tma_op, prologue_q_ptr, cta_id, seq_idx
        )

        if warp_idx == 0:
            tcgen05_alloc[Int32(Self.config.cta_group)](
                tmem_addr_ptr, Self.config.sm100_tmem_cols
            )
            tcgen05_release_allocation_lock[Int32(Self.config.cta_group)]()

        barrier()

        if warpgroup_idx == 0:
            warpgroup_reg_dealloc[96]()

            var idx_in_wg = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)

            comptime MAX_INIT_VAL = Float32(-1e30)
            var mi: Float32 = MAX_INIT_VAL
            var li: Float32 = 0.0
            var real_mi: Float32 = Float32(min_or_neg_inf[.float32]())

            var scale_log2e = scale * Float32(log2e)
            comptime P_PER_THREAD = Self.config.B_TOPK // 2  # 64
            comptime O_RESCALE_CHUNK = 32
            comptime NUM_O_RESCALE_CHUNKS = (
                Self.V_DEPTH_PER_CTA // O_RESCALE_CHUNK
            )

            # Per-thread base bf16 element offset into the scores smem,
            # matching the K-major SW128B layout the subsequent SV MMA
            # expects. Upper-half threads write the left key-half of S;
            # lower-half threads write the right key-half, which the s_p1
            # SV descriptor reads at +PADDED_HEADS_PER_CTA * B_TOPK/2 bf16
            # elements — the stride here MUST match that descriptor base.
            # phase1.cuh:166-167 hardcodes the equivalent uint128 offset as
            # 512 ((idx%64) + 64*((idx/64)*8)), which is only correct for
            # its B_TOPK=128 geometry; at B_TOPK=64 the key-half stride is
            # 256 uint128s, and the hardcoded 512 lands the second key-half
            # in dead smem while the MMA reads zeros (drops keys [32:64) of
            # every block from the numerator). ×8 converts uint128 units to
            # bf16 elements.
            comptime PAIR_KEYHALF_U128 = (
                Self.PADDED_HEADS_PER_CTA * P_PER_THREAD // 8
            )
            var s_smem_bf16_elem_base = Int(
                (
                    (idx_in_wg % UInt32(64))
                    + UInt32(PAIR_KEYHALF_U128) * (idx_in_wg / UInt32(64))
                )
                * UInt32(8)
            )

            for k in range(num_k_blocks):
                var cur_buf = k % UInt32(Self.SMemType.num_mbars)
                var cur_phase = (k / UInt32(Self.SMemType.num_mbars)) & 1

                qk_ts_done_ptr[cur_buf].wait(cur_phase)
                tcgen05_fence_after()

                var p = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=P_PER_THREAD,
                    dtype=DType.float32,
                    pack=False,
                    width=P_PER_THREAD,
                ](UInt32(Self.P_TMEM_ADDR))
                tcgen05_load_wait()
                tcgen05_fence_before()
                comptime if Self.config.cta_group == 2:
                    p_free_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))
                else:
                    _ = p_free_ptr[cur_buf].arrive()

                comptime MASK_BYTES_PER_BUF = Self.SMemType.MASK_BYTES_PER_BUF
                comptime MASK_BYTES_PER_THREAD = MASK_BYTES_PER_BUF // 2
                k_valid_ready_ptr[cur_buf].wait(cur_phase)
                var mask_byte_base = (
                    Int(cur_buf) * MASK_BYTES_PER_BUF
                    + Int(idx_in_wg // UInt32(64)) * MASK_BYTES_PER_THREAD
                )
                comptime for i in range(P_PER_THREAD):
                    comptime byte_offset = i // 8
                    comptime bit_idx = i % 8
                    var mask_byte = is_k_valid_ptr[mask_byte_base + byte_offset]
                    if ((mask_byte >> UInt8(bit_idx)) & UInt8(1)) == UInt8(0):
                        p[i] = Float32(min_or_neg_inf[.float32]())
                _ = k_valid_free_ptr[cur_buf].arrive()

                var cur_pi_max: Float32 = Float32(min_or_neg_inf[.float32]())
                comptime for i in range(P_PER_THREAD):
                    cur_pi_max = max(cur_pi_max, p[i])
                cur_pi_max = mul_ftz(cur_pi_max, scale_log2e)

                named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
                rowwise_max_ptr[idx_in_wg] = cur_pi_max
                named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
                cur_pi_max = max(
                    cur_pi_max,
                    rowwise_max_ptr[idx_in_wg ^ UInt32(64)],
                )
                real_mi = max(real_mi, cur_pi_max)

                # Per-lane decision for the STATE update (new_max/mi/li):
                # `idx_in_wg` packs one independent head-row's softmax
                # state per lane, so an OR here would leak a sibling
                # head's rescale trajectory into this one's.
                var should_scale_o = cur_pi_max - mi > Float32(6.0)

                var new_max: Float32
                var scale_for_old: Float32
                if not should_scale_o:
                    scale_for_old = 1.0
                    new_max = mi
                else:
                    new_max = max(cur_pi_max, mi)
                    scale_for_old = exp2(mi - new_max)
                mi = new_max
                li = mul_ftz(li, scale_for_old)

                # The O-rescale WALK below touches TMEM via tcgen05_ld/st,
                # which are warp-collective ops (datapaths=32) requiring
                # every lane to participate uniformly -- a per-lane branch
                # here would diverge the warp on those ops and hang. Vote
                # ANY (not the state above): any lane needing a rescale
                # pulls every lane through the walk, but each lane applies
                # its OWN `scale_for_old` (exactly 1.0 for a lane that
                # didn't need it), so a coerced lane's contribution is an
                # exact no-op multiply, not a value substitution.
                var any_rescale = warp.vote[.uint32](should_scale_o) != UInt32(
                    0
                )

                var s_bf16 = Array[Scalar[Self.qkv_dtype], P_PER_THREAD](
                    uninitialized=True
                )
                comptime for i in range(P_PER_THREAD):
                    var d: Float32 = p[i] * scale_log2e - new_max
                    var ed: Float32 = exp2(d)
                    li = li + ed
                    s_bf16[i] = ed.cast[Self.qkv_dtype]()

                if k > 0:
                    var prev_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
                    var prev_phase = (
                        (k - 1) / UInt32(Self.SMemType.num_mbars)
                    ) & 1
                    sv_p1_done_ptr[prev_buf].wait(prev_phase)

                var o_chunk_prefetch = Array[Float32, O_RESCALE_CHUNK](
                    uninitialized=True
                )
                if k > 0 and any_rescale:
                    tcgen05_fence_after()
                    o_chunk_prefetch = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=O_RESCALE_CHUNK,
                        dtype=DType.float32,
                        pack=False,
                        width=O_RESCALE_CHUNK,
                    ](UInt32(Self.O_TMEM_ADDR))

                # Write S to scores smem as 8 bf16 per uint128, stride 64
                # uint128 between writes--exactly the K-major SW128B layout
                # the SS-MMA reads. Keep the packed 128-bit store: a plain
                # SIMD[bf16, 8] store scalarizes into bank-conflicting
                # half-word stores.
                comptime for i in range(P_PER_THREAD // 8):
                    var s_vec = SIMD[Self.qkv_dtype, 8](
                        s_bf16[i * 8 + 0],
                        s_bf16[i * 8 + 1],
                        s_bf16[i * 8 + 2],
                        s_bf16[i * 8 + 3],
                        s_bf16[i * 8 + 4],
                        s_bf16[i * 8 + 5],
                        s_bf16[i * 8 + 6],
                        s_bf16[i * 8 + 7],
                    )
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.qkv_dtype](
                        scores_ptr,
                        s_smem_bf16_elem_base + i * 512,
                        bitcast[.uint32, 4](s_vec),
                    )

                # Rescale O (in TMEM) if mi changed materially; chunk 0
                # was prefetched above, chunks 1..N-1 load sequentially.
                if k > 0 and any_rescale:
                    tcgen05_load_wait()
                    var o_scaled_0 = Array[_, O_RESCALE_CHUNK](
                        fill_with_unrolled=lambda [j: Int]() -> Float32: (
                            mul_ftz(o_chunk_prefetch[j], scale_for_old)
                        )
                    )
                    tcgen05_st[
                        datapaths=32,
                        bits=32,
                        repeat=O_RESCALE_CHUNK,
                        pack=False,
                    ](UInt32(Self.O_TMEM_ADDR), o_scaled_0)
                    comptime for chunk_idx in range(1, NUM_O_RESCALE_CHUNKS):
                        var o_chunk = tcgen05_ld[
                            datapaths=32,
                            bits=32,
                            repeat=O_RESCALE_CHUNK,
                            dtype=DType.float32,
                            pack=False,
                            width=O_RESCALE_CHUNK,
                        ](
                            UInt32(Self.O_TMEM_ADDR)
                            + UInt32(chunk_idx * O_RESCALE_CHUNK)
                        )
                        tcgen05_load_wait()
                        var o_scaled = Array[_, O_RESCALE_CHUNK](
                            fill_with_unrolled=lambda [j: Int]() -> Float32: (
                                mul_ftz(o_chunk[j], scale_for_old)
                            )
                        )
                        tcgen05_st[
                            datapaths=32,
                            bits=32,
                            repeat=O_RESCALE_CHUNK,
                            pack=False,
                        ](
                            UInt32(Self.O_TMEM_ADDR)
                            + UInt32(chunk_idx * O_RESCALE_CHUNK),
                            o_scaled,
                        )
                    tcgen05_store_wait()
                    tcgen05_fence_before()

                fence_async_view_proxy()
                comptime if Self.config.cta_group == 2:
                    so_ready_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))
                else:
                    _ = so_ready_ptr[cur_buf].arrive()

            if real_mi == Float32(min_or_neg_inf[.float32]()):
                li = 0.0
                mi = Float32(min_or_neg_inf[.float32]())

            rowwise_sum_ptr[idx_in_wg] = li
            named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
            li = add_ftz(li, rowwise_sum_ptr[idx_in_wg ^ UInt32(64)])

            var last_buf = (num_k_blocks - 1) % UInt32(Self.SMemType.num_mbars)
            var last_phase = (
                (num_k_blocks - 1) / UInt32(Self.SMemType.num_mbars)
            ) & 1
            sv_p1_done_ptr[last_buf].wait(last_phase)
            tcgen05_fence_after()

            var output_scale: Float32
            # Only the real head rows [0, NUM_Q_HEADS_PER_CTA) may index the
            # `num_q_heads`-sized sink buffer; padded rows
            # [NUM_Q_HEADS_PER_CTA, 64) must NOT dereference it (that would read
            # past the buffer end when num_q_heads < 64). At head 64/128 the
            # bound is 64, so every row reads the sink exactly as before (NFC).
            # Mirrors the per-head sink guard in `kernel` (BF16).
            var sink_row = idx_in_wg % UInt32(64)
            if attn_sink_ptr and sink_row < UInt32(Self.NUM_Q_HEADS_PER_CTA):
                var sink_head_idx = (
                    cta_id * UInt32(Self.NUM_Q_HEADS_PER_CTA) + sink_row
                )
                var attn_sink_val = attn_sink_ptr.unsafe_value()[
                    Int(sink_head_idx)
                ] * Float32(log2e)
                output_scale = 1.0 / (li + exp2(attn_sink_val - mi))
            else:
                output_scale = 1.0 / li

            var have_valid_indices = warp.vote[.uint32](
                li != Float32(0.0)
            ) != UInt32(0)
            if not have_valid_indices:
                output_scale = 1.0

            var head_row_block = UInt32(warp_idx) % UInt32(2)
            var depth_col_block = UInt32(warp_idx) // UInt32(2)
            var local_lane = UInt32(lane_idx)
            var head_local = head_row_block * UInt32(32) + local_lane

            comptime GROUP_STRIDE = Self.NUM_Q_HEADS_PER_CTA * 64

            comptime o_sw = make_swizzle[
                Self.output_dtype, TensorMapSwizzle.SWIZZLE_128B
            ]()

            comptime for atom_idx in range(Self.NUM_SV_ATOMS):
                comptime atom_o_tmem_addr = (
                    Self.O_TMEM_ADDR + atom_idx * Self.O_ATOM_PHYS_COLS
                )

                comptime for chunk in range(2):
                    comptime CHUNK = 64
                    var col_group = (
                        Int(depth_col_block) * 2 + atom_idx * 4 + chunk
                    )
                    var c_chunk: Array[Float32, CHUNK]
                    c_chunk = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=CHUNK,
                        dtype=DType.float32,
                        pack=False,
                        width=CHUNK,
                    ](UInt32(atom_o_tmem_addr + chunk * CHUNK))
                    tcgen05_load_wait()

                    comptime for i in range(CHUNK // 2):
                        var v0_f32 = c_chunk[2 * i] * output_scale
                        var v1_f32 = c_chunk[2 * i + 1] * output_scale
                        var v = SIMD[Self.qkv_dtype, 2](
                            v0_f32.cast[Self.qkv_dtype](),
                            v1_f32.cast[Self.qkv_dtype](),
                        )
                        var smem_offset = col_group * GROUP_STRIDE + o_sw(
                            Int(head_local) * 64 + i * 2
                        )
                        # Store only the real head rows; padded rows
                        # [num_q_heads, 64) carry dropped MMA output and would
                        # clobber the next col-group at the real GROUP_STRIDE.
                        # Redirect them to a dead dump slot past the real O
                        # region using a BRANCH-FREE arithmetic mask: a
                        # divergent `if head_local < N` here splits warps 0/2
                        # at the head boundary and deadlocks the next
                        # warpgroup-collective `tcgen05_ld`. `async_store`
                        # reads only [0, NUM_Q_HEADS_PER_CTA * v_depth). NFC at
                        # 64/128 (head_local < 64 always -> mask 0 -> real off).
                        var pad_row = Int(
                            head_local >= UInt32(Self.NUM_Q_HEADS_PER_CTA)
                        )
                        var store_off = (
                            smem_offset * (1 - pad_row)
                            + (Self.SMemType.O_SIZE - 2) * pad_row
                        )
                        (o_ptr + store_off).store[width=2](v)

            named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
            if warp_idx == 0:
                if elect_one_sync():
                    fence_async_view_proxy()
                    var o_smem_tile = TileTensor(
                        o_ptr.bitcast[Scalar[Self.output_dtype]](),
                        row_major[
                            Self.NUM_Q_HEADS_PER_CTA, Self.config.v_depth
                        ](),
                    )
                    o_tma_op.async_store(
                        o_smem_tile,
                        (
                            0,
                            Int(seq_idx) * Self.config.num_q_heads
                            + Int(cta_id) * Self.NUM_Q_HEADS_PER_CTA,
                        ),
                    )
                    cp_async_bulk_commit_group()
            cp_async_bulk_wait_group[0]()

            if warp_idx == 0:
                tcgen05_dealloc[Int32(Self.config.cta_group)](
                    tmem_addr_ptr[], Self.config.sm100_tmem_cols
                )

        elif warpgroup_idx == 1:
            # K producer (FP8 path)
            warpgroup_reg_alloc[168]()
            var local_warp_idx = UInt32(warp_id() - 4)

            for k in range(num_k_blocks):
                var cur_buf = k % UInt32(Self.SMemType.num_mbars)
                var cur_phase = k / UInt32(Self.SMemType.num_mbars) & 1
                var prev_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
                var prev_phase = (k - 1) / UInt32(Self.SMemType.num_mbars) & 1

                # Load K scales cooperatively (all 128 WG1 threads).
                # Overlaps with the previous iteration's MMA while we wait.
                # scale_block_size == 0 => no-scale mode: skip the gather and
                # its flush; convert_k applies unit scale.
                comptime if scale_block_size > 0:
                    Self.load_k_scales_to_smem[scale_block_size](
                        scales_ptr,
                        indices,
                        kv_lut,
                        k_scales_ptr,
                        k,
                        cta_id,
                        indices_base,
                        Int32(num_kv_rows),
                    )
                    # Flush all 128 scale writes before any thread reads them
                    # in convert_k_fp8_to_bf16. The mbarrier.wait below only
                    # provides acquire ordering for TMA completion, not for
                    # this cooperative SMEM store loop.
                    named_barrier[Int32(WARPGROUP_SIZE)](5)

                # Wait for WG3 to finish with the previous K SMEM buffer
                # BEFORE issuing the new FP8 TMA.  The FP8 staging region
                # (k_smem[K_SIZE..]) overlaps with the p1 (TS MMA) columns
                # [24576..73727] of K SMEM, so we must wait for qk_ts_done
                # (not just qk_ss_done) before overwriting.
                if k > 0:
                    qk_ss_done_ptr[prev_buf].wait(prev_phase)
                    qk_ts_done_ptr[prev_buf].wait(prev_phase)

                # Arm k_fp8_tma_done mbarrier (one thread) and issue FP8 K TMA
                # (all four WG1 warps each issue a portion via elect_one_sync).
                # The barrier between expect_bytes and the TMA loop ensures
                # expect_bytes is globally visible before any TMA arrival can
                # satisfy the mbarrier (CUDA spec: expect_tx must happen-before
                # the cp.async.bulk.tensor that signals the same mbarrier).
                if local_warp_idx == 0 and elect_one_sync():
                    k_fp8_tma_done_ptr[cur_buf].expect_bytes(
                        Int32(Self.B_TOPK_PER_CTA * Self.config.qk_depth)
                    )
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                if elect_one_sync():
                    Self.load_k_fp8_tma(
                        k_tma_op_fp8,
                        indices,
                        kv_lut,
                        k_smem_fp8_ptr,
                        k_fp8_tma_done_ptr + cur_buf,
                        k,
                        cta_id,
                        local_warp_idx,
                        indices_base,
                    )

                # Wait for FP8 K TMA to land, then convert in-place.
                k_fp8_tma_done_ptr[cur_buf].wait(cur_phase)
                Self.convert_k_fp8_to_bf16[scale_block_size](
                    k_smem_fp8_ptr, k_smem_ptr, k_scales_ptr
                )

                # All 128 WG1 threads must finish their convert stores
                # (+ the per-thread async-proxy fence at the end of
                # convert_k) BEFORE the ready signal: the elected arrive
                # alone orders only its own thread's writes.
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                # Signal K ready (both p0 and p1 barriers). At cta_group=2
                # the MMA runs on CTA0 and reads BOTH CTAs' converted K
                # smem, so each CTA signals CTA0's barrier instance
                # (init count = cta_group).
                if local_warp_idx == 0 and elect_one_sync():
                    comptime if Self.config.cta_group == 2:
                        k_p0_ready_ptr[cur_buf].arrive_cluster(UInt32(0))
                        k_p1_ready_ptr[cur_buf].arrive_cluster(UInt32(0))
                    else:
                        _ = k_p0_ready_ptr[cur_buf].arrive()
                        _ = k_p1_ready_ptr[cur_buf].arrive()

        elif warpgroup_idx == 2:
            # V producer (FP8 path)
            warpgroup_reg_alloc[144]()
            var local_warp_idx = UInt32(warp_id() - 8)

            # Wait for Q copy to TMEM before starting V loads.
            prologue_q_cp_ptr[].wait()

            for k in range(num_k_blocks):
                var cur_buf = k % UInt32(Self.SMemType.num_mbars)
                var cur_phase = k / UInt32(Self.SMemType.num_mbars) & 1
                var prev_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
                var prev_phase = (k - 1) / UInt32(Self.SMemType.num_mbars) & 1

                # Wait for previous V SMEM to be free BEFORE issuing new TMA.
                # FP8 V staging [v_smem+V_SIZE..v_smem+2*V_SIZE] overlaps kh1 V BF16
                # data that WG3's SV MMA for k-1 is still reading. Must wait first.
                if k > 0:
                    sv_p0_done_ptr[prev_buf].wait(prev_phase)
                    sv_p1_done_ptr[prev_buf].wait(prev_phase)

                # Arm v_fp8_tma_done mbarrier (one thread per WG2), then
                # barrier so expect_bytes is visible before any TMA arrival
                # can satisfy the mbarrier (same CUDA-spec ordering as K).
                if local_warp_idx == 0 and elect_one_sync():
                    v_fp8_tma_done_ptr[cur_buf].expect_bytes(
                        Int32(Self.config.B_TOPK * Self.V_SMEM_COLS_PER_CTA)
                    )
                named_barrier[Int32(WARPGROUP_SIZE)](4)

                # Issue FP8 V TMA (one elected thread per warp in WG2).
                if elect_one_sync():
                    Self.load_v_fp8_tma(
                        v_tma_op_fp8,
                        indices,
                        kv_lut,
                        v_smem_fp8_ptr,
                        v_fp8_tma_done_ptr + cur_buf,
                        k,
                        cta_id,
                        local_warp_idx,
                        indices_base,
                    )

                # Load V scales cooperatively (all 128 WG2 threads).
                # Overlaps TMA latency while we wait for v_fp8_tma_done.
                # scale_block_size == 0 => no-scale mode: skip the gather and
                # its flush; convert_v applies unit scale.
                comptime if scale_block_size > 0:
                    Self.load_v_scales_to_smem[scale_block_size](
                        scales_ptr,
                        indices,
                        kv_lut,
                        v_scales_ptr,
                        k,
                        indices_base,
                        Int32(num_kv_rows),
                    )
                    # Flush all 128 scale writes before convert_v_fp8_to_bf16
                    # reads them. Use ID 6 (not 5) — named barriers are CTA-wide
                    # resources; WG1 uses ID 5 concurrently, so sharing it would
                    # mix arrivals from both warpgroups and defeat the flush.
                    named_barrier[Int32(WARPGROUP_SIZE)](6)

                # Wait for FP8 V TMA to complete (all threads spin).
                v_fp8_tma_done_ptr[cur_buf].wait(cur_phase)

                # Convert V FP8→BF16 in-place (all 128 WG2 threads).
                Self.convert_v_fp8_to_bf16[scale_block_size](
                    v_smem_fp8_ptr, v_smem_ptr, v_scales_ptr, cta_id
                )

                # All 128 WG2 threads must finish their convert stores
                # before the ready signal (same reasoning as K).
                named_barrier[Int32(WARPGROUP_SIZE)](6)
                # Signal both V halves ready. At cta_group=2 the MMA reads
                # BOTH CTAs' converted V smem; signal CTA0's instance.
                if local_warp_idx == 0 and elect_one_sync():
                    comptime if Self.config.cta_group == 2:
                        v_p0_ready_ptr[cur_buf].arrive_cluster(UInt32(0))
                        v_p1_ready_ptr[cur_buf].arrive_cluster(UInt32(0))
                    else:
                        _ = v_p0_ready_ptr[cur_buf].arrive()
                        _ = v_p1_ready_ptr[cur_buf].arrive()

        else:
            warpgroup_reg_dealloc[96]()

            if cta_id == 0 and warp_idx == 12 and elect_one_sync():
                var q_tmem_desc = Self.QKMMAOpType.tmem_descriptor_q(
                    q_smem_ptr + Self.SMemType.SHARED_Q_SIZE
                )
                # Expect exactly the REAL in-bounds Q bytes (num_q_heads):
                # both the 64/128 built-in copy and the sub-64 manual depth-
                # tile sub-copies read only real heads from GMEM, so the
                # transaction delivers num_q_heads * qk_depth * cta_group
                # elements. (The sub-64 padding rows are memset, not TMA'd, so
                # they contribute no bytes here.) NFC at 64/128:
                # NUM_Q_HEADS_PER_CTA == padded per-CTA == 64. Mirrors `kernel`.
                prologue_q_ptr[].expect_bytes(
                    Int32(
                        Self.NUM_Q_HEADS_PER_CTA
                        * Self.config.qk_depth
                        * Self.config.cta_group
                        * Self.qkv_dtype_size
                    )
                )
                prologue_q_ptr[].wait()
                tcgen05_fence_after()

                Self.Common.cp_q_from_smem_to_tmem(
                    q_tmem_desc, UInt32(Self.Q_TMEM_ADDR)
                )
                mma_arrive_multicast[cta_group=Self.config.cta_group](
                    prologue_q_cp_ptr,
                    Self.CTA_MASK,
                )

                for k in range(num_k_blocks + 1):
                    Self.Common.mma[fp8_active=True](
                        q_smem_ptr,
                        k_smem_ptr,
                        scores_ptr,
                        v_smem_ptr,
                        k_p0_ready_ptr,
                        k_p1_ready_ptr,
                        v_p0_ready_ptr,
                        v_p1_ready_ptr,
                        sv_p0_done_ptr,
                        sv_p1_done_ptr,
                        so_ready_ptr,
                        p_free_ptr,
                        qk_ss_done_ptr,
                        qk_ts_done_ptr,
                        k,
                        num_k_blocks,
                    )

            elif warp_idx == 13 and lane_idx < Self.SMemType.NUM_KV_VALID_LANES:
                Self.Common.kv_valid_producer(
                    indices,
                    is_k_valid_ptr,
                    k_valid_ready_ptr,
                    k_valid_free_ptr,
                    UInt32(lane_idx),
                    indices_base,
                    Int32(num_kv_rows),
                    Int32(top_k_length),
                    Int(num_k_blocks),
                )


@always_inline
def mla_prefill_sparse_fp8[
    output_dtype: DType,
    q_type: DType,
    cache_t: KVCacheT,
    config: MLASparseConfig,
    group: Int,
    q_depth: Int,
    scale_block_size: Int,
](
    output: TileTensor[output_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[.uint32, address_space=.GENERIC, ...],
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    scale: Float32,
    indices_stride: Int32,
    ctx: DeviceContext,
) raises:
    comptime assert q_depth == config.qk_depth
    comptime assert config.qk_depth == 576
    # This backend has not been given the narrow-row treatment its BF16-KV and
    # native-FP8 siblings have (no tail zero-fill, byte counts still derived
    # from the full width), so refuse the narrow row rather than transfer fewer
    # bytes than the barriers wait for.
    comptime assert config.input_qk_depth == config.qk_depth, (
        "the FP8-KV converter prefill takes only the full rotary-carrying row."
        " the narrow NoPE row is supported by the BF16-KV and native-FP8"
        " prefill kernels"
    )
    # Same head-count contract as the BF16 `mla_prefill_sparse` above: 128 (2-CTA
    # tile) or a multiple of 8 in (0, 64] (single-CTA tile padded to a legal
    # 64-row MMA M-tile). Q is BF16 in both paths, so the sub-64 padded Q load
    # (`_load_q_prologue`) is identical; the `% 8 == 0` requirement is the
    # SWIZZLE_128B 8-row core matrix. GLM 5.2 shards 64 heads to {8, 16, 32}.
    comptime assert config.num_q_heads == 128 or (
        0 < config.num_q_heads
        and config.num_q_heads <= 64
        and config.num_q_heads % 8 == 0
    ), (
        "sparse MLA prefill (fp8): num_q_heads must be 128 or a multiple of 8"
        " in (0, 64] (SWIZZLE_128B 8-row core matrix)"
    )
    comptime assert config.num_kv_heads == 1
    comptime assert size_of[output_dtype]() == size_of[q_type]()
    # Three FP8 scaling modes are supported:
    #  - No-scale (scale_block_size == 0): FP8 latents are read at unit scale
    #    (no dequant, scales_ptr unused / may be null). Mirrors the sparse
    #    DECODE kernel's read of today's scale-less MLA latent cache
    #    (quantization disabled). This is the current DSv3.2 production path.
    #  - Tensorwise: scale_block_size >= qk_depth => 1 scale per KV token.
    #  - Blockwise (cache-native, e.g. granularity 32): MLA stores ONE scale
    #    vector per token over the full qk_depth-wide (576) latent, so the
    #    cache holds ceildiv(qk_depth, scale_block_size) scales/token. K reads
    #    all of them; V (the v_depth-wide nope part) reads the first
    #    ceildiv(v_depth, scale_block_size) blocks of that SAME vector. Both
    #    index the HBM buffer with the cache stride ceildiv(qk_depth, sbs)
    #    (see load_k/v_scales_to_smem + convert_*), so one flat scales_ptr
    #    serves both. For SnapMLA (SERVOPT-1094) once the cache carries scales.
    comptime assert scale_block_size >= 0

    var num_q_rows = q_num_matrix_view_rows(q)
    var kv_operand = KVCacheMHAOperand(kv_cache)

    var q_tma_op = create_tensor_tile[
        Index(1, config.num_q_heads // config.cta_group, q_depth),
        swizzle_mode=config.q_swizzle_mode,
    ](ctx, q)

    # FP8 K TMA: INT64-packed, SWIZZLE_NONE, 72 INT64 elems per 576-byte row.
    var k_tma_op_fp8 = kv_operand.create_gather4_tma_tile[
        tile_width=config.qk_depth // 8,
        tile_stride=config.qk_depth // 8,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        tile_height=config.B_TOPK // config.cta_group,
        tma_dtype=DType.int64,
    ](ctx)

    # FP8 V TMA: INT64-packed, SWIZZLE_NONE.  tile_width = V_BMN_PER_ATOM / 8.
    var v_tma_op_fp8 = kv_operand.create_gather4_tma_tile[
        tile_width=config.v_depth // 2 // config.cta_group // 8,
        tile_stride=config.qk_depth // 8,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        tile_height=config.B_TOPK,
        tma_dtype=DType.int64,
    ](ctx)

    var output_2d_fp8 = TileTensor(
        output.ptr,
        row_major(num_q_rows * config.num_q_heads, config.v_depth),
    )
    var o_tma_op_fp8 = create_tensor_tile[
        Index(config.num_q_heads // config.cta_group, config.v_depth),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        __desc_shape=Index(config.num_q_heads // config.cta_group, 64),
    ](ctx, output_2d_fp8)

    comptime assert type_of(topk_lengths).flat_rank == 1
    comptime assert type_of(indices).flat_rank == 1
    comptime assert topk_lengths.element_size == 1
    comptime assert indices.element_size == 1
    comptime kernel = MLAPrefillSparseFP8[
        KVLUTType=type_of(kv_operand),
        output_dtype=output_dtype,
        config=config,
    ].kernel_fp8[
        type_of(topk_lengths).LayoutType,
        type_of(indices).LayoutType,
        type_of(topk_lengths).Engine,
        type_of(indices).Engine,
        scale_block_size,
    ]

    comptime smem_size = size_of[
        MLASparseSharedMemoryFP8[config, scale_block_size]
    ]()

    ctx.enqueue_function[kernel](
        q_tma_op,
        k_tma_op_fp8,
        v_tma_op_fp8,
        o_tma_op_fp8,
        topk_lengths,
        indices,
        kv_operand,
        scale,
        attn_sink_ptr,
        indices_stride,
        output.ptr,
        scales_ptr,
        grid_dim=(config.cta_group * num_q_rows, 1, 1),
        block_dim=(config.num_threads, 1, 1),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
