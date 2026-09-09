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
"""SM100 (B200) sparse MLA prefill kernel with BF16 KV cache.

BF16-KV variant.  K/V latents are read as BF16 via SWIZZLE_128B gather4 TMA and
fed directly to the shared QK/SV MMA pipeline.  The dtype-agnostic machinery
(Q-load prologue, `mma`, `cp_q_from_smem_to_tmem`, `_raw_indices_to_tma_rows`,
`kv_valid_producer`) is shared with the FP8-KV variant
(`mla_prefill_sparse_kv_fp8.mojo`) through `MLAPrefillSparseCommon` in
`mla_prefill_sparse_utils.mojo`.
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


struct MLAPrefillSparse[
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

    comptime k_tile_width = Self.config.input_qk_depth
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
    comptime k_tma_tile_width_fp8 = Self.config.input_qk_depth // 8
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

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma_op, `nvvm.grid_constant`)
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](
            Int32(Self.config.cta_group), 1, 1
        )
    )
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"mla_prefill_sparse_{Self.qkv_dtype}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel[
        TopKLengthLayout: TensorLayout,
        IndicesLayout: TensorLayout,
        TopKLengthEngine: TensorEngine,
        IndicesEngine: TensorEngine,
    ](
        q_tma_op: TMATensorTile[
            Self.qkv_dtype,
            3,
            Self.q_tile_shape,
            Self.q_desc_shape,
        ],
        k_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.k_tile_shape,
            Self.k_desc_shape,
        ],
        v_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.v_tile_shape,
            Self.v_desc_shape,
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
        # Per-query base offset into the indices buffer; each query row owns
        # `indices_stride` indices.
        var indices_base = seq_idx * UInt32(indices_stride)

        if thread_idx.x == 0:
            q_tma_op.prefetch_descriptor()
            k_tma_op.prefetch_descriptor()
            v_tma_op.prefetch_descriptor()

        ref smem = external_memory[
            UInt8, address_space=.SHARED, alignment=128
        ]().bitcast[Self.SMemType]()[]
        ref qkvo_union = smem.qkvo_union

        var full_q_ptr = qkvo_union.unsafe_get[Self.FULL_Q_TYPE]().unsafe_ptr()
        ref shared_qkv = qkvo_union.unsafe_get[Self.SHARED_QKV_TYPE]()
        var shared_qkv_ptr: UnsafePointer[
            mut=True,
            Scalar[Self.qkv_dtype],
            origin_of(shared_qkv),
            address_space=.SHARED,
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
        # Byte at offset `buf * MASK_BYTES_PER_BUF + j` holds bits for
        # keys `[j*8, j*8+8)` of buffer `buf`.
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

        if warp_idx == 0:
            if elect_one_sync():
                prologue_q_ptr[].init(1)
                prologue_q_cp_ptr[].init(1)
                comptime for i in range(Self.SMemType.num_mbars):
                    qk_ss_done_ptr[i].init(1)
                    qk_ts_done_ptr[i].init(1)
                    sv_p0_done_ptr[i].init(1)
                    sv_p1_done_ptr[i].init(1)
                    k_p0_ready_ptr[i].init(1)
                    k_p1_ready_ptr[i].init(1)
                    v_p0_ready_ptr[i].init(1)
                    v_p1_ready_ptr[i].init(1)
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

                fence_mbarrier_init()

        cluster_sync()

        # Zero the padded Q SMEM (sub-64-head configs) and issue the Q TMA.
        # Shared with kernel_fp8 (Q is BF16 in both KV-cache paths) so the
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
            warpgroup_reg_alloc[144]()

            var idx_in_wg = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)

            # FlashAttention online-softmax state. Mirrors phase1.cuh:154-164:
            # `mi` is the running max used to scale Pi; `li` is the running
            # sumexp; `real_mi` is the true running max (used only by the
            # all-invalid epilogue case). Both rows of one head are owned by
            # threads (t, t^64) and these three values stay identical between
            # paired threads after every update.
            comptime MAX_INIT_VAL = Float32(-1e30)
            var mi: Float32 = MAX_INIT_VAL
            var li: Float32 = 0.0
            var real_mi: Float32 = Float32(min_or_neg_inf[.float32]())

            var scale_log2e = scale * Float32(log2e)

            comptime USE_EXP2_EMULATION = Self.config.cta_group != 2
            comptime P_PER_THREAD = Self.config.B_TOPK // 2  # 64
            comptime O_RESCALE_CHUNK = 32
            # Same "per-CTA = D_V/2" convention as in phase1.cuh's
            # rescale loop (`(D_V/2)/CHUNK_SIZE`): we cover the full
            # per-CTA O accumulator (V_DEPTH_PER_CTA cols) in chunks of
            # CHUNK_SIZE. Previously this was double-halved.
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

                # Wait for P = QK^T (TS MMA done).
                qk_ts_done_ptr[cur_buf].wait(cur_phase)
                tcgen05_fence_after()

                # Load P from TMEM (64 fp32 per thread).
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
                # P is now in registers; release the P TMEM tile so MMA can
                # overwrite it on the next iteration. p_free is initialized
                # with count = WARPGROUP_SIZE*2 = 256 and the MMA leader
                # waits on CTA 0's instance, so every WG0 thread in *both*
                # cluster CTAs must arrive on CTA 0's barrier. Plain
                # `.arrive()` would only credit the local CTA; use
                # `arrive_cluster(0, 1)` to mirror phase1.cuh:180's
                # `bar_p_free[k%NUM_BUFS].arrive(0u)` (CUTLASS's
                # ClusterTransactionBarrier targeting cta 0).
                comptime if Self.config.cta_group == 2:
                    p_free_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))
                else:
                    _ = p_free_ptr[cur_buf].arrive()

                # Mask step (phase1.cuh:182-210): wait on warp-13's
                # validity bitmask for this k-block, then poison invalid
                # P entries with -inf so they drop out of the softmax.
                # Each thread owns P_PER_THREAD = B_TOPK/2 keys; thread
                # `t<64` reads the low half of the mask, thread `t>=64`
                # the high half.
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

                # Per-thread row max over local P (scaled to log2 domain).
                var cur_pi_max: Float32 = Float32(min_or_neg_inf[.float32]())
                comptime for i in range(P_PER_THREAD):
                    cur_pi_max = max(cur_pi_max, p[i])
                cur_pi_max = mul_ftz(cur_pi_max, scale_log2e)

                # Cross-thread max reduction: threads t and t^64 own the
                # same head row of P. Each writes its partial to a small
                # smem buffer, syncs, then reads its peer's value. Two
                # sync points avoid a WAR race.
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
                # head's rescale trajectory into this one's (>6 log2-units
                # means rescaling lifts mass by < 1/64 — phase1.cuh skips
                # the rescale below that threshold to reduce TMEM
                # traffic).
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
                var vscale = SIMD[.float32, 2](scale_log2e)
                var vneg_max = SIMD[.float32, 2](-new_max)
                comptime for j in range(P_PER_THREAD // 2):
                    var pj = SIMD[.float32, 2](
                        p[2 * j],
                        p[2 * j + 1],
                    )
                    var ed2 = exp2_emulation[
                        use_exp2_emulation=USE_EXP2_EMULATION
                    ](fma_ftz(pj, vscale, vneg_max))
                    li = li + ed2[0] + ed2[1]
                    s_bf16[2 * j] = ed2[0].cast[Self.qkv_dtype]()
                    s_bf16[2 * j + 1] = ed2[1].cast[Self.qkv_dtype]()

                # Wait until the previous SV MMA has drained the scores
                # smem before overwriting it. (sv_p1_done implies the
                # second half of the prev S@V completed, which is the last
                # use of the prev scores tile.)
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

                # Make scores smem writes (and any TMEM stores) visible to
                # the SV MMA, then release the so_ready slot for this k.
                # Same cluster-arrive reasoning as p_free above.
                fence_async_view_proxy()
                comptime if Self.config.cta_group == 2:
                    so_ready_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))
                else:
                    _ = so_ready_ptr[cur_buf].arrive()

            # ---------------- Epilogue (phase1.cuh:288-386) ----------------

            # All-invalid query case: real_mi stayed -inf, meaning no row
            # ever contributed. Reset li/mi to match the definition that
            # output_scale = 1/(li + exp2(sink - mi)) gives 0 when output
            # is unused.
            if real_mi == Float32(min_or_neg_inf[.float32]()):
                li = 0.0
                mi = Float32(min_or_neg_inf[.float32]())

            # Cross-thread li sum (paired threads share the row).
            rowwise_sum_ptr[idx_in_wg] = li
            named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
            li = add_ftz(li, rowwise_sum_ptr[idx_in_wg ^ UInt32(64)])

            # Wait for the final SV MMA to retire before reading O.
            var last_buf = (num_k_blocks - 1) % UInt32(Self.SMemType.num_mbars)
            var last_phase = (
                (num_k_blocks - 1) / UInt32(Self.SMemType.num_mbars)
            ) & 1
            sv_p1_done_ptr[last_buf].wait(last_phase)
            tcgen05_fence_after()

            # Per-head attention sink. A missing `attn_sink_ptr` (Optional
            # is None) is treated as -inf (same as phase1.cuh:315); when
            # set, the sink contributes one extra term `exp2(sink_h - mi)`
            # in log2 domain to the softmax normalizer.
            var output_scale: Float32
            # Only the real head rows [0, NUM_Q_HEADS_PER_CTA) may index the
            # `num_q_heads`-sized sink buffer; padded rows
            # [NUM_Q_HEADS_PER_CTA, 64) must NOT dereference it (that would read
            # past the buffer end when num_q_heads < 64). At head 64/128 the
            # bound is 64, so every row reads the sink exactly as before (NFC).
            # Mirrors the per-head sink guard in mla_decode_sparse.mojo.
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

            # Guard against deadlocks if some lanes' li==0 (entirely
            # invalid rows): tcgen05_ld below must run uniformly across
            # the warpgroup, so we vote and pick a uniform path.
            var have_valid_indices = warp.vote[.uint32](
                li != Float32(0.0)
            ) != UInt32(0)
            if not have_valid_indices:
                output_scale = 1.0

            # Single-load + warp-distributed write, matching the proven
            # pattern in `test_bulk_mma_pair_cta_sm100.mojo`.  Each SV atom
            # produced a per-CTA accumulator of shape (BM=64 heads ×
            # MMA_N=256 cols) stored in TMEM at addresses
            # O_TMEM_ADDR (atom1) and O_TMEM_ADDR_ATOM2 (atom2).  We load
            # 128 fp32 per thread per atom in two 64-cell chunks
            # (splitting reduces register pressure under WG0's 144-reg
            # budget).  Each warp's depth range:
            #   - Warp 0: heads 0..31 of CTA's head range, cluster cols 0..127
            #   - Warp 1: heads 32..63,                    cluster cols 0..127
            #   - Warp 2: heads 0..31,                     cluster cols 128..255
            #   - Warp 3: heads 32..63,                    cluster cols 128..255
            # Atom1 writes those cols to v_depth offsets 0..255; atom2
            # adds V_DEPTH_PER_CTA=256 to land in 256..511.
            var head_row_block = UInt32(warp_idx) % UInt32(2)
            var depth_col_block = UInt32(warp_idx) // UInt32(2)
            var local_lane = UInt32(lane_idx)
            var head_local = head_row_block * UInt32(32) + local_lane

            # O SMEM layout: [8 col_groups, NUM_Q_HEADS_PER_CTA real heads, 64
            # cols]. GROUP_STRIDE must equal the async_store copy_size
            # (o_desc_shape product = NUM_Q_HEADS_PER_CTA*64) so each col-group
            # block aligns with the TMA sub-copies. The padded MMA produces 64
            # output rows but only the real heads are stored (padded rows
            # [num_q_heads, 64) are skipped in the store below), so the padded
            # output is dropped here rather than written at a padded stride.
            # col_group = depth_col_block*2 + atom_idx*4 + chunk tiles v_depth=512.
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
                            0,  # col (depth, innermost in TMA coords)
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
            # K producer
            warpgroup_reg_dealloc[96]()
            var local_warp_idx = UInt32(warp_id() - 4)

            if elect_one_sync():
                for k in range(num_k_blocks):
                    Self.load_k(
                        k_tma_op,
                        indices,
                        kv_lut,
                        k_smem_ptr,
                        qk_ss_done_ptr,
                        qk_ts_done_ptr,
                        k_p0_ready_ptr,
                        k_p1_ready_ptr,
                        k,
                        cta_id,
                        local_warp_idx,
                        Int32(num_kv_rows),
                        indices_base,
                    )

        elif warpgroup_idx == 2:
            # V producer
            warpgroup_reg_dealloc[96]()
            var local_warp_idx = UInt32(warp_id() - 8)

            if elect_one_sync():
                prologue_q_cp_ptr[].wait()

                for k in range(num_k_blocks):
                    Self.load_v(
                        v_tma_op,
                        v_smem_ptr,
                        sv_p0_done_ptr,
                        sv_p1_done_ptr,
                        v_p0_ready_ptr,
                        v_p1_ready_ptr,
                        indices,
                        kv_lut,
                        k,
                        local_warp_idx,
                        cta_id,
                        indices_base,
                    )

        else:
            warpgroup_reg_alloc[168]()

            # leader CTA and MMA warp
            if cta_id == 0 and warp_idx == 12 and elect_one_sync():
                # use for copying q_tmem from smem to tmem
                var q_tmem_desc = Self.QKMMAOpType.tmem_descriptor_q(
                    q_smem_ptr + Self.SMemType.SHARED_Q_SIZE
                )
                # pair cta only signal leader cta for byte arrival.
                # Expect exactly the REAL in-bounds Q bytes (num_q_heads):
                # both the 64/128 built-in copy and the sub-64 manual depth-
                # tile sub-copies read only real heads from GMEM, so the
                # transaction delivers num_q_heads * qk_depth * cta_group
                # elements. (The sub-64 padding rows are memset, not TMA'd, so
                # they contribute no bytes here.) NFC at 64/128:
                # NUM_Q_HEADS_PER_CTA == padded per-CTA == 64.
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
                    Self.Common.mma(
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

            # KV-valid mask producer (mirrors phase1.cuh's warp 13).
            # `NUM_KV_VALID_LANES` active lanes, each owning
            # `INDICES_PER_LANE = B_TOPK / NUM_KV_VALID_LANES` indices
            # per k-block; the lane packs an 8-bit validity mask and
            # writes it to `is_k_valid[cur_buf][lane]`.  Valid means:
            #   (1) the index is in [0, num_kv_rows), and
            #   (2) its absolute position k*B_TOPK + lane*8 + i is
            #       below the per-query top_k_length (handles padding
            #       to indices_stride for short sequences).
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
    @staticmethod
    def k_tma_gather4_load[
        col_range: Tuple[UInt32, UInt32],
        num_rows: Int,
    ](
        tma_op: TMATensorTile[Self.qkv_dtype, 2, _, _],
        smem_barrier: UnsafePointer[
            SharedMemBarrier, address_space=.SHARED, ...
        ],
        smem_tensor: TileTensor[
            mut=True, Self.qkv_dtype, address_space=.SHARED, ...
        ],
        local_indices: Array[SIMD[.int32, 4], num_rows],
        warp_idx: UInt32,
    ):
        # layout for complying to 128B swizzle atom
        comptime col_dim = smem_tensor.static_shape[1]
        comptime num_col_tiles = col_dim // 64
        comptime row_dim = smem_tensor.static_shape[0]
        var tma_smem_tensor = TileTensor(
            smem_tensor.ptr,
            row_major[row_dim * num_col_tiles, 64](),
        )

        comptime outer_row_start = col_range[0] // 64
        comptime outer_row_end = col_range[1] // 64
        comptime for outer_row in range(outer_row_start, outer_row_end):
            var tma_tile = tma_smem_tensor.tile[row_dim, 64](
                Coord(Idx[Int(outer_row)], Idx[0])
            )
            comptime for inner_row in range(64 // 16):
                var inner_tma_tile = tma_tile.tile[16, 64](
                    Coord(Idx[Int(inner_row)], Idx[0])
                )
                var inner_warp_dist = inner_tma_tile.tile[4, 64](
                    Coord(warp_idx, Idx[0])
                )
                var indices = local_indices[inner_row]
                tma_op.async_copy_gather4[cta_group=Self.config.cta_group](
                    inner_warp_dist,
                    smem_barrier[],
                    Int32(outer_row * 64),
                    indices[0],
                    indices[1],
                    indices[2],
                    indices[3],
                )

    @always_inline
    @staticmethod
    def v_tma_gather4_load[
        local_row_range: Tuple[Int, Int],
    ](
        tma_op: TMATensorTile[Self.qkv_dtype, 2, _, _],
        smem_barrier: UnsafePointer[
            SharedMemBarrier, address_space=.SHARED, ...
        ],
        smem_tensor: TileTensor[
            mut=True, Self.qkv_dtype, address_space=.SHARED, ...
        ],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        warp_idx: UInt32,
        k: UInt32,
        cta_id: UInt32,
        indices_base: UInt32,
    ):
        # `local_row_range` is in (local_row) units (each local_row =
        # 4 token-rows × 4 warps = 16 rows of progress). With SWIZZLE_128B
        # and tile_width=V_SMEM_COLS_PER_CTA=128, gather4 box_width=64
        # forces 2 col-group calls per 4-row chunk. tile_height=B_TOPK/2
        # so each col-group occupies tile_height*64 = 4096 elems, matching
        # the BMN=128 SW128 descriptor's mn_outer stride. The smem base
        # passed by the caller selects which K-half this batch writes to.
        comptime row_start = local_row_range[0]
        comptime row_end = local_row_range[1]
        comptime num_warps = 4
        comptime gather_box = Self.v_gather_box  # 64 for SW128 bf16
        comptime num_col_groups = (
            Self.V_SMEM_COLS_PER_CTA // Self.v_gather_box
        )
        comptime num_cgs_per_atom = (Self.V_BMN_PER_ATOM // Self.v_gather_box)
        # Each warp owns 4 contiguous rows within every 16-row stripe.
        # Within a col-group: 4-row stride = 4 * gather_box = 256.
        var v_smem_base = smem_tensor.ptr + warp_idx * UInt32(
            4 * Self.v_gather_box
        )

        comptime for local_row in range(row_start, row_end):
            var token_idx_v4 = Self.Common._raw_indices_to_tma_rows(
                kv_lut,
                indices.load[width=4](
                    Coord(
                        indices_base
                        + k * UInt32(Self.config.B_TOPK)
                        + (UInt32(local_row) * UInt32(num_warps) + warp_idx)
                        * UInt32(4)
                    )
                ).cast[.int32](),
            )
            comptime for cg in range(num_col_groups):
                # local_row stride within col-group: 16 rows × 64 cols
                # = 1024 elems. Col-group stride: tile_height*gather_box.
                # smem_offset is relative to the batch base (`smem_tensor.ptr`)
                # so the second call (row_start=HALF_LOCAL_ROWS) maps
                # local_row=HALF_LOCAL_ROWS → smem_offset=0 within the
                # second-batch smem region.
                comptime smem_offset = (
                    (local_row - row_start)
                    * (4 * num_warps)
                    * Self.v_gather_box
                    + cg * Self.v_tile_height * Self.v_gather_box
                )
                # Interleaved gmem → smem mapping for 2 SV atoms:
                #   atom_idx = cg // num_cgs_per_atom, local_cg = cg % num_cgs_per_atom
                #   gmem_col = atom_idx * V_DEPTH_PER_CTA
                #             + cta_id  * V_BMN_PER_ATOM
                #             + local_cg * gather_box
                # The descriptor for atom1 (base = v_smem) sees smem cgs 0..1
                # which span cluster depths 0..255 (CTA0 cols 0..127 + CTA1
                # cols 128..255).  Atom2 (base shifted by 8192 elements) sees
                # smem cgs 2..3 = cluster depths 256..511.
                comptime atom_idx = cg // num_cgs_per_atom
                comptime local_cg = cg % num_cgs_per_atom
                tma_op.async_copy_gather4[cta_group=Self.config.cta_group](
                    TileTensor(v_smem_base + smem_offset, smem_tensor.layout),
                    smem_barrier[],
                    Int32(atom_idx * Self.V_DEPTH_PER_CTA)
                    + Int32(cta_id * UInt32(Self.V_BMN_PER_ATOM))
                    + Int32(local_cg * Self.v_gather_box),
                    token_idx_v4[0],
                    token_idx_v4[1],
                    token_idx_v4[2],
                    token_idx_v4[3],
                )

    @always_inline
    @staticmethod
    def load_k(
        k_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.k_tile_shape,
            Self.k_desc_shape,
        ],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        k_smem_ptr: UnsafePointer[
            mut=True, Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        qk_ss_done: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        qk_ts_done: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        k_p0_ready: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        k_p1_ready: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        k: UInt32,
        cta_id: UInt32,
        warp_idx: UInt32,
        num_kv_rows: Int32,
        indices_base: UInt32,
    ):
        # we break down B_TOPK // cta_group into groups of 16,
        # each group is loaded by 4 warps, each warp loads 4 rows
        comptime num_warps = 4
        comptime num_rows_per_warp = (
            Self.config.B_TOPK // Self.config.cta_group
        ) // 4 // num_warps

        var local_indices = Array[SIMD[.int32, 4], num_rows_per_warp](
            uninitialized=True
        )
        var max_idx: Int32 = -1
        var min_idx: Int32 = num_kv_rows

        var k_smem_tensor = TileTensor(
            k_smem_ptr,
            row_major[Self.B_TOPK_PER_CTA, Self.config.qk_depth](),
        )

        comptime for local_row in range(num_rows_per_warp):
            # Each (local_row, warp_idx) reads a disjoint 4-int chunk. The
            # ×4 multiplier matches phase1.cuh:401 — that line walks gIndices
            # as an `int4*` (stride 4 ints per pointer step), so successive
            # warps' chunks are 4 ints apart, not 1.
            var indices_offset = (
                indices_base
                + k * UInt32(Self.config.B_TOPK)
                + cta_id * UInt32(Self.config.B_TOPK // Self.config.cta_group)
                + (UInt32(local_row) * num_warps + warp_idx) * UInt32(4)
            )
            local_indices[local_row] = indices.load[width=4](
                Coord(indices_offset)
            ).cast[.int32]()
            max_idx = max(max_idx, local_indices[local_row].reduce_max())
            min_idx = min(min_idx, local_indices[local_row].reduce_min())

        var all_inval = min_idx == num_kv_rows or max_idx == -1
        # `>=` not `>`: skipping is only safe once both `num_mbars`
        # pipeline buffers have been filled at least once; otherwise an
        # initial all-invalid block could leave the K buffer in an
        # uninitialized (NaN-poisoned) state for a later valid block.
        var skip_tma = all_inval and k >= UInt32(Self.SMemType.num_mbars)

        var curr_buf = k % UInt32(Self.SMemType.num_mbars)
        var prev_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
        var prev_phase = (k - 1) / UInt32(Self.SMemType.num_mbars) & 1

        if k > 0:
            qk_ss_done[prev_buf].wait(prev_phase)

        # Convert the raw encoded indices to physical gather4 TMA rows
        # (folds in the num_layers block stride). The all-invalid / skip
        # decision above intentionally runs on the raw indices, whose `-1`
        # sentinel is preserved by the conversion.
        comptime for local_row in range(num_rows_per_warp):
            local_indices[local_row] = Self.Common._raw_indices_to_tma_rows(
                kv_lut, local_indices[local_row]
            )

        if not skip_tma:
            Self.k_tma_gather4_load[
                (UInt32(0), UInt32(Self.config.q_smem_depth)),
            ](
                k_tma_op,
                k_p0_ready + curr_buf,
                k_smem_tensor,
                local_indices,
                warp_idx,
            )
        else:
            # Skip the TMA and credit the K-load-ready barrier directly so
            # the MMA's outstanding `expect_bytes` is satisfied. (The original
            # branch credited `qk_ss_done` here, but that barrier is the MMA's
            # producer barrier and is signalled by `mma_arrive_multicast` —
            # crediting it from the loader corrupts the MMA handshake.)
            k_p0_ready[curr_buf].complete_transaction(
                0,
                Int32(
                    num_rows_per_warp
                    * 4
                    * Self.config.q_smem_depth
                    * Self.qkv_dtype_size
                ),
                1,
            )

        if k > 0:
            qk_ts_done[prev_buf].wait(prev_phase)

        if not skip_tma:
            Self.k_tma_gather4_load[
                (
                    UInt32(Self.config.q_smem_depth),
                    UInt32(Self.config.qk_depth),
                ),
            ](
                k_tma_op,
                k_p1_ready + curr_buf,
                k_smem_tensor,
                local_indices,
                warp_idx,
            )
        else:
            k_p1_ready[curr_buf].complete_transaction(
                0,
                Int32(
                    num_rows_per_warp
                    * 4
                    * Self.config.q_tmem_depth
                    * Self.qkv_dtype_size
                ),
                1,
            )

    @always_inline
    @staticmethod
    def load_v(
        v_tma_op: TMATensorTile[
            Self.qkv_dtype,
            2,
            Self.v_tile_shape,
            Self.v_desc_shape,
        ],
        v_smem_ptr: UnsafePointer[
            mut=True, Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        sv_p0_done: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        sv_p1_done: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        v_p0_ready: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        v_p1_ready: UnsafePointer[SharedMemBarrier, address_space=.SHARED, ...],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        k: UInt32,
        warp_idx: UInt32,
        cta_id: UInt32,
        indices_base: UInt32,
    ):
        var curr_buf = k % UInt32(Self.SMemType.num_mbars)
        var prev_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
        var prev_phase = (k - 1) / UInt32(Self.SMemType.num_mbars) & 1
        comptime num_warps = 4

        # Per-batch tensor descriptors: each K-half writes into its own
        # tile_height=B_TOPK/2 region of V smem, sized to match the
        # gather4 layout the SW128 MMA descriptor expects.
        var v_smem_tensor_p0 = TileTensor(
            v_smem_ptr,
            row_major[Self.v_tile_height, Self.V_SMEM_COLS_PER_CTA](),
        )
        # Batch 1 base = v_smem + v_tile_height * V_SMEM_COLS_PER_CTA
        # (= 16384 elems: 64 rows × 256 cols × 1 element, where 256 cols
        # covers both SV atoms' contributions per CTA).
        var v_smem_tensor_p1 = TileTensor(
            v_smem_ptr + Self.v_tile_height * Self.V_SMEM_COLS_PER_CTA,
            row_major[Self.v_tile_height, Self.V_SMEM_COLS_PER_CTA](),
        )

        # Per-warp work within one batch: (B_TOPK/2)/4 rows-per-gather4
        # / NUM_WARPS = HALF_LOCAL_ROWS local rows per warp per batch.
        comptime HALF_LOCAL_ROWS = (Self.config.B_TOPK // 2 // 4 // 4)

        # Split B_TOPK so the second load pipelines with the first S@V.
        if k > 0:
            sv_p0_done[prev_buf].wait(prev_phase)

        # K-half 0 (rows 0..63 in V) → writes to batch-0 smem region.
        Self.v_tma_gather4_load[(0, HALF_LOCAL_ROWS)](
            v_tma_op,
            v_p0_ready + curr_buf,
            v_smem_tensor_p0,
            indices,
            kv_lut,
            warp_idx,
            k,
            cta_id,
            indices_base,
        )

        if k > 0:
            sv_p1_done[prev_buf].wait(prev_phase)

        # K-half 1 (rows 64..127 in V) → writes to batch-1 smem region.
        # local_row offset relative to the batch so smem_offset computes
        # correctly: pass (HALF_LOCAL_ROWS, 2*HALF_LOCAL_ROWS) and let
        # the inner function compute indices_base offsets via the
        # k * B_TOPK + local_row * 16 formula (which already covers all
        # B_TOPK indices when local_row spans 0..2*HALF_LOCAL_ROWS-1).
        Self.v_tma_gather4_load[(HALF_LOCAL_ROWS, 2 * HALF_LOCAL_ROWS)](
            v_tma_op,
            v_p1_ready + curr_buf,
            v_smem_tensor_p1,
            indices,
            kv_lut,
            warp_idx,
            k,
            cta_id,
            indices_base,
        )


@always_inline
def mla_prefill_sparse[
    output_dtype: DType,
    q_type: DType,
    cache_t: KVCacheT,
    config: MLASparseConfig,
    group: Int,
    q_depth: Int,
](
    output: TileTensor[output_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[.uint32, address_space=.GENERIC, ...],
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    # Optional per-head attention sink. Pass `None` to skip the sink term
    # entirely; pass `Some(ptr)` with a buffer of `num_q_heads` fp32
    # values to add `exp2(sink_h - mi)` to the softmax normalizer per
    # head.
    attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    scale: Float32,
    indices_stride: Int32,
    ctx: DeviceContext,
) raises:
    comptime assert q_depth == config.qk_depth
    # DSv3.2 absorbed prefill is the only supported shape: qk_depth =
    # kv_lora_rank (512) + qk_rope_head_dim (64) = 576. The kernel hardcodes
    # q_smem_depth=192 / q_tmem_depth=384 summing to 576; other depths would
    # silently mis-stride the TMA copies.
    comptime assert config.qk_depth == 576
    # Head count: 128 runs the 2-CTA (cta_group=2) tile; a MULTIPLE OF 8 in
    # (0, 64] runs the single-CTA (cta_group=1) tile padded to a legal 64-row
    # MMA M-tile (`tcgen05.mma kind::f16` only permits M in {64, 128}). Real
    # heads are loaded/stored via num_q_heads; the padded rows [num_q_heads, 64)
    # are zeroed. The `% 8 == 0` requirement is the SWIZZLE_128B 8-row core
    # matrix: the manual sub-64 Q sub-copy in `_load_q_prologue` lands the real
    # box at the padded 64-row swizzle positions only when the box height is a
    # whole number of core matrices (cf. the `q_tile_rows % 8` assert in the
    # proven MSA decode Layout-G precedent, msa_1q_layout_g.mojo:110). GLM 5.2
    # shards its 64 heads to {8, 16, 32}/device (TP {8, 4, 2}); DSv3.2 uses 128
    # or 64. Counts like 4/2/20 are rejected fail-fast (untested/unsupported).
    comptime assert config.num_q_heads == 128 or (
        0 < config.num_q_heads
        and config.num_q_heads <= 64
        and config.num_q_heads % 8 == 0
    ), (
        "sparse MLA prefill: num_q_heads must be 128 or a multiple of 8 in"
        " (0, 64] (SWIZZLE_128B 8-row core matrix)"
    )
    comptime assert config.num_kv_heads == 1
    # The output smem buffer is allocated as `qkv_dtype` (it shares the
    # smem union with Q/K/V). We bitcast it to `output_dtype` at the TMA
    # store site, which is only sound if the two dtypes have the same
    # bit width.
    comptime assert size_of[output_dtype]() == size_of[q_type]()

    # num_q_rows == batch_size * q_seq_len
    var num_q_rows = q_num_matrix_view_rows(q)

    var kv_operand = KVCacheMHAOperand(kv_cache)

    # CTA pair splits heads at head128 (CTA0 upper, CTA1 lower); a single CTA
    # loads all heads at head<=64. The box requests the REAL per-CTA head
    # count so every sub-copy reads only in-bounds GMEM heads (a padded box
    # makes heads [num_q_heads, 64) a middle-dim OOB region whose TMA never
    # completes -> deadlock). For num_q_heads < 64 the kernel issues the
    # depth-tile sub-copies manually at the padded 64-row stride. NFC at
    # 64/128 (real == padded).
    var q_tma_op = create_tensor_tile[
        Index(1, config.num_q_heads // config.cta_group, q_depth),
        swizzle_mode=config.q_swizzle_mode,
    ](ctx, q)

    # for 2CTA MMA B_TOPK == 128
    # we load 64 tokens gathered from different kv blocks for each CTA
    # Both CTA in the pair load 64 into their peer
    # so each CTA will have 128 topk tokens
    var k_tma_op = kv_operand.create_gather4_tma_tile[
        tile_width=config.qk_depth,
        tile_stride=config.qk_depth,
        swizzle_mode=config.k_swizzle_mode,
        tile_height=config.B_TOPK // config.cta_group,
    ](ctx)

    # V is loaded with SWIZZLE_128B. With tile_width=V_SMEM_COLS_PER_CTA=
    # 128 and box_width=64, gather4 produces 2 col-groups per CTA, placed
    # at smem offsets 0 and B_TOPK*64 respectively. The per-CTA SV MMA
    # descriptor (descriptor_v) reads this layout via BMN =
    # V_SMEM_COLS_PER_CTA, BK = B_TOPK/2, MN-major, SWIZZLE_128B (matching
    # the decode kernel's `DecodeSM100PVSS.descriptor_v_block` pattern).
    #
    # tile_stride is the gmem row stride (qk_depth=576). tile_width is
    # V_SMEM_COLS_PER_CTA = MMA_N / cta_group; each CTA contributes its
    # slice of V to the cluster MMA, with `col_idx = cta_id *
    # V_SMEM_COLS_PER_CTA` selecting its V col range.
    var v_tma_op = kv_operand.create_gather4_tma_tile[
        tile_width=config.v_depth // 2 // config.cta_group,
        tile_stride=config.qk_depth,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        tile_height=config.B_TOPK // 2,
    ](ctx)

    # Output viewed 2D as [num_q_rows * num_q_heads, v_depth] so per-cluster
    # TMA stores address the right head range with a single [row, col] coord.
    var output_2d = TileTensor(
        output.ptr,
        row_major(num_q_rows * config.num_q_heads, config.v_depth),
    )

    var o_tma_op = create_tensor_tile[
        Index(config.num_q_heads // config.cta_group, config.v_depth),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        __desc_shape=Index(config.num_q_heads // config.cta_group, 64),
    ](ctx, output_2d)

    comptime assert type_of(topk_lengths).flat_rank == 1
    comptime assert type_of(indices).flat_rank == 1
    comptime assert topk_lengths.element_size == 1
    comptime assert indices.element_size == 1
    comptime kernel = MLAPrefillSparse[
        KVLUTType=type_of(kv_operand),
        output_dtype=output_dtype,
        config=config,
    ].kernel[
        type_of(topk_lengths).LayoutType,
        type_of(indices).LayoutType,
        type_of(topk_lengths).Engine,
        type_of(indices).Engine,
    ]

    comptime smem_size = size_of[MLASparseSharedMemory[config]]()

    ctx.enqueue_function[kernel](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        o_tma_op,
        topk_lengths,
        indices,
        kv_operand,
        scale,
        attn_sink_ptr,
        indices_stride,
        output.ptr,
        grid_dim=(config.cta_group * num_q_rows, 1, 1),
        block_dim=(config.num_threads, 1, 1),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
