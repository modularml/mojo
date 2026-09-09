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

from std.collections import OptionalReg
from std.math import ceildiv, clamp
from std.math.constants import log2e
from std.sys import size_of
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx,
    block_idx,
    warp_id,
    lane_id,
)
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import (
    CacheEviction,
    external_memory,
    fence_async_view_proxy,
)
from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.swizzle import make_swizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _gather4_box_width,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import bitcast
from layout import (
    ComptimeInt,
    DefaultEngine,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    elect_mma_arrive,
    expect_bytes_pred,
    SharedMemPointer,
    MBarType,
    ProducerPipeline,
    ConsumerPipeline,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    ORaggedTMATile,
    MLA_Decode_Pack,
    OffsetPosition,
    KVPipelineGeneric,
    KVLoad2CvtProducer,
    KVLoad2CvtConsumer,
    KVCvt2MmaProducer,
    KVCvt2MmaConsumer,
    DecodeSM100MiscMBars,
    DecodeSProducer,
    DecodePConsumer,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeSM100QKTSS,
    DecodeSM100PVSS,
    ld_shared_v4_u32,
    cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4,
    st_shared_v4_b32_at_bf16_elem_off,
    e8m0_to_bf16_broadcast,
    hmul2_bf16x8_by_scalar,
)


# ------------------------------------------------------------------------------
# MLA sparse decoding kernel struct for SM100 — all-FP8 KV variant
#
# Sibling of `MLA_SM100_Decode_Sparse`. The parent keeps rope in BF16 and
# loads nope (512 B) + rope (128 B) via two separate gather4 TMAs. This
# variant stores the full 576-byte KV row (nope+rope) as FP8 in HBM and
# loads it via a single INT64/SWIZZLE_NONE gather4 (tile_width=72 INT64 =
# 576 B). The convert WG then converts all 9 blocks (576 FP8 → 576 BF16)
# per row before UMMA consumes. Sparse machinery (idx_producer,
# _transform_indices_to_smem, idx_bars, mask handling via OffsetPosition,
# has_attn_sink / has_extra_kv / has_variable_topk) is preserved verbatim.
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_Sparse_KV_FP8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
    # Read-once shared-index q-fold (KERN-3141). When True, pack the
    # q_len_fold * num_q_heads query rows of one MTP decode step into the
    # BM=64 M tile so grid.y collapses to 1, gather the ONE shared top-k list
    # once, and let every folded row attend it in a single KV pass (no
    # per-position re-stream). Exact ONLY when every folded position refers to
    # the identical index list — the explicit `index_share` op contract,
    # enforced by the dispatch gate, never inferred here. Default False -> the
    # unfolded per-position baseline (grid.y = q_max_seq_len), byte-identical.
    # Unlike the dense native-FP8 kernels, this sparse kernel deliberately does
    # NOT implement a per-position phase-fold; the only fold is shared-index.
    #
    # Implementation approach based on myb/glm_52_mla_opt by Yingbo Ma
    # (reference commit 86d7d5760ec).
    fold_shared_index: Bool = False,
    # Number of q positions folded into the BM tile (the MTP step width);
    # meaningful only when fold_shared_index=True, else 1.
    q_len_fold: Int = 1,
    Engine: TensorEngine = DefaultEngine[element_width=1],
](TrivialRegisterPassable):
    comptime kv_type = Self.KVLUTType.dtype
    # KV type is FP8 for both nope and rope (all-FP8 KV variant).
    comptime fp8_type = DType.float8_e4m3fn
    comptime bf16_type = DType.bfloat16
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9  (full K row = nope+rope)
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    # 512 / 64 = 8  (nope-only; used for PV output dims)
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.q_type]()

    # FP8: 1 byte per element
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    # BF16: 2 bytes per element
    comptime bf16_bytes_per_element = size_of[Self.bf16_type]()

    # Full KV stage: 9 blocks * 4096 FP8 elems = 36864 FP8 bytes per stage.
    # After convert → 36864 BF16 elements = 73728 BF16 bytes per stage.
    comptime KVStageFP8Bytes = (
        Self.config.BN_QK
        * Self.config.padded_q_depth
        * Self.fp8_bytes_per_element
    )

    # Single gather4 TMA descriptor covering the full 576-byte row.
    #   INT64 dtype, SWIZZLE_NONE (linear SMEM layout).
    #   tile_width  = padded_q_depth / 8 = 72 INT64 elements (full 576 bytes)
    #   tile_stride = 72 (same; whole row)
    #   box_width   = tile_width = 72 (SWIZZLE_NONE => box == tile_width)
    #   num_col_groups = ceildiv(72, 72) = 1
    comptime kv_gather4_tile_width = Self.config.input_q_depth // 8
    comptime kv_gather4_box_w = _gather4_box_width[
        DType.int64,
        Self.kv_gather4_tile_width,
        TensorMapSwizzle.SWIZZLE_NONE,
    ]()
    comptime kv_gather4_num_col_groups = ceildiv(
        Self.config.input_q_depth // 8, Self.kv_gather4_box_w
    )
    # Number of 4-row chunks for BN_QK=64 rows: 64 / 4 = 16
    comptime gather4_num_4row_chunks = Self.config.BN_QK // 4
    # SMEM for gather4 indices: BN_QK Int32 values = 64 * 4 = 256 bytes
    comptime gather4_indices_bytes = Self.config.BN_QK * size_of[Int32]()

    # the stage element is the same for both K and V (BF16 after conversion)
    # This is 576/64 * 4096 = 9 * 4096 = 36864 BF16 elements per stage
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )
    comptime UMMAQKTSS = DecodeSM100QKTSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    # Number of producer arrivals for kv_load2cvt pipeline:
    # - Tensorwise (scale_block_size==0): 1 (just TMA via expect_bytes)
    # - Blockwise (scale_block_size>0):  33 (expect_bytes + 32 warp-8 threads
    #   arriving after scale stores, with release semantics covering each
    #   thread's SMEM writes, eliminating named barriers 4/5)
    comptime load2cvt_num_producer = 1 + (
        32 if Self.config.scale_block_size > 0 else 0
    )

    comptime Common_MLA_Op = MLA_SM100_Decode_Common[
        Self.q_type,
        Self.KVLUTType,
        Self.output_type,
        Self.SplitAccumType,
        Self.MaskType,
        Self.config,
        Self.ValidLengthType,
        Self._is_cache_length_accurate,
        Self.ragged,
    ]

    # --------------------------------------------------------------------------
    # MLA decoding main kernel function (Sparse variant — 4 warpgroups)
    # --------------------------------------------------------------------------
    #
    # This kernel has 4 warpgroups + 4 individual warps, unlike the BF16
    # kernel which has 3 warpgroups. The extra warpgroup (Convert WG)
    # handles the FP8→BF16 conversion in SMEM before MMA can consume the data.
    #
    # There are TWO KV SMEM regions:
    #   - kv_smem_fp8:  TMA target (FP8 data from HBM lands here)
    #   - kv_smem_bf16: Converted data (BF16, consumed by UMMA)
    #
    # There are TWO KV pipelines:
    #   - kv_load2cvt_pipe: Load Warp → Convert WG (and MMA warps for release)
    #   - kv_cvt2mma_pipe:  Convert WG → MMA Warps (QK and PV)
    # plus per-block handoff bars (cvt_blk_bars): Convert WG → QK MMA, one
    # per 64-col block per stage, so QK starts before the full stage converts.
    # PV stays whole-stage gated on kv_cvt2mma_pipe: P depends on softmax(S),
    # which needs all 9 QK blocks, so the stage is fully converted by then.
    #
    #  Warp assignments:
    #    Warps  0-3  : Softmax WG    (warpgroup 0)
    #    Warps  4-7  : Correction WG (warpgroup 1)
    #    Warp   8    : Load warp     (TMA loads FP8 KV from HBM → kv_smem_fp8)
    #    Warp   9    : MMA QK warp   (UMMA QK on kv_smem_bf16)
    #    Warp  10    : MMA PV warp   (UMMA PV on kv_smem_bf16)
    #    Warp  11    : Store warp    (TMA store output)
    #    Warps 12-15 : Convert WG    (warpgroup 3: FP8→BF16 in SMEM)
    #
    #                     Pipeline Diagram (Double-Buffered)
    #                     ====================================
    #
    #  Both kv_load2cvt_pipe and kv_cvt2mma_pipe have 2 stages (Slot 0/1).
    #  FP8 slots overlay the upper half of the BF16 slots in SMEM.
    #
    #    HBM (FP8 KV)                HBM (FP8 KV)
    #         |                           |
    #         | TMA Load (warp 8)         | TMA Load (warp 8)
    #         V                           V
    #    FP8 Slot 0 (SMEM)          FP8 Slot 1 (SMEM)
    #         |                           |
    #         |  kv_load2cvt_pipe (2 stages, 33 prod → 130 cons for blockwise)
    #         V                           V
    #    Convert WG (12-15)         Convert WG (12-15)
    #    FP8 → BF16                 FP8 → BF16
    #         |                           |
    #         |  kv_cvt2mma_pipe (2 stages, 128 prod → 2 cons)
    #         V                           V
    #    BF16 Slot 0 (SMEM)         BF16 Slot 1 (SMEM)
    #         |                           |
    #         V                           V
    #    UMMA QK → S0 (warp 9)     UMMA QK → S1 (warp 9)
    #         |                           |
    #   arrive mbar_s0             arrive mbar_s1
    #         |                           |
    #         |---- Softmax WG (warps 0-3) ----|
    #         |                                |
    #         V                                V
    #       wait_s0                          wait_s1
    #       S0 → P0                          S1 → P1
    #         |                                |
    #         |---- Correction WG (warps 4-7) ----|
    #         |   (scale O by correction factor    |
    #         |    before new P*V accumulation)    |
    #         V                                    V
    #    UMMA PV → O (warp 10)          UMMA PV → O (warp 10)
    #    (P0 * V → O accumulate)        (P1 * V → O accumulate)
    #         |                                    |
    #       arrive mbar_o                    arrive mbar_o
    #         |                                    |
    #       corr_done_bars signal -----------------|
    #         |                                    |
    #       wait_O_filled                    wait_O_filled
    #         |                                    |
    #       wait_out                           wait_out
    #         |                                    |
    #       Store warp (warp 11)             Store warp
    #

    # --------------------------------------------------------------------------
    # MLA decoding SMEMDescriptors for Q, K, V, P
    # --------------------------------------------------------------------------

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(extra_k_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    def kernel(
        q_tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        # Single K gather4 TMA covering full 576-byte row: INT64,
        # BN_QK(64) rows, SWIZZLE_NONE. tile_width=72 INT64 = 576 bytes.
        k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            # Per-warp output stripe (= BN_PV/4), not BN_QK.
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType=Self.ValidLengthType,
            MaskType=Self.MaskType,
            SplitAccumType=Self.SplitAccumType,
        ],
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        indices_stride_dev: Int32,
        topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
        attn_sink_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]],
        # Extra KV parameters: separate cache for always-attend tokens.
        # Single TMA covering full 576-byte row (all-FP8).
        extra_k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_indices_stride_dev: Int32,
        extra_scales_ptr: OptionalReg[
            UnsafePointer[Float32, origin=MutAnyOrigin]
        ],
        scalar_args: TileTensor[
            .int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
            Engine=Self.Engine,
        ],
    ):
        # SlidingWindowCausalMask is supported ONLY by the native FP8 backend
        # (MLA_SM100_Decode_QKV_FP8).  Reject it here at comptime.
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_Sparse_KV_FP8 only supports NullMask and"
            " CausalMask. Sliding window is supported only by"
            " MLA_SM100_Decode_QKV_FP8 (native FP8)."
        )
        # Shared-index fold contract (enforced here so a mis-wired dispatch
        # fails at comptime rather than silently corrupting output).
        comptime assert (
            not Self.fold_shared_index
            or Self.config.num_q_heads * Self.q_len_fold <= Self.config.BM
        ), "fold_shared_index requires num_q_heads * q_len_fold <= BM."
        comptime assert not (Self.fold_shared_index and Self.has_extra_kv), (
            "fold_shared_index does not support the extra always-attend KV"
            " cache; dispatch must not fold when extra_k is present."
        )
        # Softmax now includes the epilogue, so it needs more registers
        # Correction does less work now (no epilogue), so it needs fewer
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 72
        comptime num_reg_keep_mma_load_store = 72
        comptime num_reg_keep_fp8tofp16 = 184
        var indices_stride = Int(indices_stride_dev)
        var extra_indices_stride = Int(extra_indices_stride_dev)
        var batch_size = Int(scalar_args.ptr[0])
        var q_max_seq_len = Int(scalar_args.ptr[1])
        var num_partitions = mla_decode_pack.num_partitions
        var mask = mla_decode_pack.mask
        var valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr
        # OffsetPosition.__init__ handles sparse overrides (topk loading,
        # clamping to actual_tokens, extra_topk, and split-K recomputation)
        # when sparse_indices_stride > 0.
        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ](
            kv_lut,
            rebind[
                UnsafePointer[
                    Scalar[Self.ValidLengthType.dtype],
                    ImmutAnyOrigin,
                    address_space=.GENERIC,
                ]
            ](valid_length.value()),
            q_max_seq_len,
            num_partitions,
            batch_size,
            sparse_indices_stride=indices_stride,
            sparse_topk_lengths=topk_lengths,
            sparse_extra_indices_stride=extra_indices_stride,
            sparse_extra_topk_lengths=extra_topk_lengths,
        )

        # Re-derive topk and extra_topk for block-level control flow.
        # These are needed to determine which blocks belong to the original
        # cache vs the extra cache in the kernel loop.
        var topk: Int
        comptime if Self.has_variable_topk:
            topk = Int(
                topk_lengths.unsafe_value()[Int(offset_position.batch_idx)]
            )
        else:
            topk = indices_stride
        var extra_topk: Int = 0
        comptime if Self.has_extra_kv:
            comptime if Self.has_variable_topk:
                extra_topk = Int(
                    extra_topk_lengths.unsafe_value()[
                        Int(offset_position.batch_idx)
                    ]
                )
            else:
                extra_topk = extra_indices_stride
        # actual_tokens from OffsetPosition is now topk+extra_topk;
        # back-derive the clamped topk.
        topk = offset_position.num_keys - extra_topk

        # Compute num_orig_blocks and total blocks for the extended loop.
        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)
        var total_k_blocks = num_orig_blocks
        comptime if Self.has_extra_kv:
            total_k_blocks = num_orig_blocks + ceildiv(
                extra_topk, Self.config.BN_QK
            )

        # Early-exit PDL helper. The shared-index fold packs all q_len_fold
        # output/LSE slots into ONE CTA (grid.y=1), so every early-exit path
        # must write -inf LSE for EACH folded slot or the combine kernel sums
        # an uninitialised LSE. The unfolded baseline exits the single slot it
        # owns (block_idx.y). @always_inline + comptime pruning => when
        # fold_shared_index=False this is byte-identical to the prior inline
        # call (verified kernel-scoped in Phase 6).
        @__parameter
        @always_inline
        def _pdl_early_exit_all_q():
            comptime if Self.fold_shared_index:
                comptime for q_local in range(Self.q_len_fold):
                    Self.Common_MLA_Op.pdl_early_exit[fold_q=True](
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                        seq_idx_fold=UInt32(q_local),
                    )
            else:
                Self.Common_MLA_Op.pdl_early_exit(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                )

        # Early exit for split-K: CTAs with no work (num_keys_this_split == 0)
        # must still write -inf LSE and call launch_dependent_grids()
        # to fulfill the PDL contract with the combine kernel.  Skipping
        # launch_dependent_grids() causes the combine kernel to hang,
        # leading to CUDA_ERROR_ILLEGAL_ADDRESS.
        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                _pdl_early_exit_all_q()
                return

        # early exit: Skip blocks beyond actual sequence length for this batch
        # In ragged mode with split-K, q_max_seq_len can be > 1 (up to 8).
        # block_idx.y ranges from 0 to q_max_seq_len-1, but some sequences
        # may have fewer tokens. CTAs with block_idx.y >= seq_len must still
        # fulfill the PDL contract (write -inf LSE and call
        # launch_dependent_grids) or the combine kernel will hang.
        comptime if Self.ragged:
            # In ragged mode, block_idx.y is the query token index (0 to q_max_seq_len-1)
            # But this batch might have fewer tokens than q_max_seq_len.
            # Under fold_shared_index grid.y == 1, so this only fires for
            # seq_len == 0 (whole batch empty); _pdl_early_exit_all_q then
            # fulfils the PDL contract for every folded slot. Per-q_local
            # ragged fill for 0 < seq_len < q_len_fold is not needed (MTP
            # decode batches have uniform seq_len == q_len_fold).
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    _pdl_early_exit_all_q()

                return  # This query position doesn't exist for this batch
        var q_smem = external_memory[
            Scalar[Self.q_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        var kv_smem_bf16 = q_smem + Self.BlockElems * Self.NumQKBlocks

        # All-FP8 KV layout: TMA loads the full 576-byte row as FP8 into
        # the upper half of the first BF16 stage (dense-style overlay, see
        # mla_decode_kv_fp8.mojo:354-358). Total BF16 capacity is
        # 2 * NumQKBlocks * BlockElems = 73728 elements = 147456 bytes.
        # FP8 needs NumQKBlocks * BlockElems = 36864 bytes per stage, so
        # placing FP8 at the upper half of BF16 keeps two staged FP8 buffers
        # inside the BF16 allocation without collision.
        var kv_smem_fp8 = (
            kv_smem_bf16.bitcast[Scalar[Self.fp8_type]]()
            + Self.BlockElems * Self.NumQKBlocks
        )

        comptime kv_total_stages = Self.config.num_kv_stages
        # to reuse the K for V as well, we break KV as 9 stages of 64x64 to cover 64x576
        comptime kv_smem_total = Self.BlockElems * Self.NumQKBlocks * kv_total_stages

        # we need to use the KSmem for out pointer
        # We move P to the last slot of KV pipeline SO now we have tile of 64x
        # 32 of float or 64x64 of FP16 to save output into
        # tiles in SMEM and smooth the pipeline for the next batch if we use splitk
        var out_smem_start = kv_smem_bf16.bitcast[Scalar[Self.output_type]]()
        # there is potential to have two Tmem for S, because we have two K so we can
        # unblock the MMA while loading S to reg for softmax
        # if it was splitk we need to use the extra P slot. If not we need
        # to clear the KV slot before starting the max because KV slot is used by
        # MMA/load when max is valid.
        var out_smem_total = kv_smem_total

        var out_smem = out_smem_start

        # max_smem is double-buffered (2 x 128 elements) to avoid a race
        # condition in softmax; li_smem is a single 128-element buffer.
        var max_smem = (out_smem + out_smem_total).bitcast[
            Scalar[Self.AccumType]
        ]()

        var li_smem = (
            max_smem + 2 * WARPGROUP_SIZE
        )  # 128 x1 for SMEM correction for Softmax

        # Scale SMEM for blockwise FP8 scaling (e8m0, 1 byte per scale).
        # Double-buffered: stage 0 at scale_smem_base,
        # stage 1 at scale_smem_base + scale_smem_per_stage.
        # When scale_block_size == 0 (tensorwise), scale_smem_per_stage is 0
        # and this region is empty.
        var scale_smem_base = (li_smem + WARPGROUP_SIZE).bitcast[UInt8]()

        #  Now we have to define MBARS for the kernel
        var mbar_base: MBarType = (
            (
                scale_smem_base
                + Self.config.scale_smem_per_stage * Self.config.num_kv_stages
            )
            .bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base  # q uses 0
        var mbar_kv_base: MBarType = mbar_base + 1  # barrier total[1]

        var kv_cvt2mma_pipe = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ](mbar_kv_base)

        # Move mbar_base to the first free barrier *after* KV:
        mbar_base = mbar_kv_base + kv_cvt2mma_pipe.num_mbars()  # kv uses 1..4
        # Move mbar_base to the first free barrier *after* k done:
        var s_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](
            mbar_base
        )  # S uses 5..8
        mbar_base = s_bars.end()  # barrier total[9]
        var p_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ](
            mbar_base
        )  # P uses 9 .. 12
        mbar_base = p_bars.end()  # barrier total [13]
        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](
            mbar_base
        )  # O uses 13..16
        mbar_base = o_bars.end()  # barrier total [17]
        # C pipeline, Softmax -> Correction
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # C uses 17..18
        mbar_base = c_bars.end()  # barrier total [19]

        # Correction done barrier: Correction -> Softmax direction
        # Signals when Correction exits its while loop (all corrections done)
        # 2-stage pipeline to overlap correction with next softmax iteration
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # corr_done uses 19..22
        mbar_base = corr_done_bars.end()  # barrier total [23]
        # This is used for the pipeline between Load and convert fp8 to bf16
        var kv_load2cvt_pipe = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=Self.load2cvt_num_producer,
            num_consumer=WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ](
            mbar_base
        )  # kv_load2cvt_pipe uses 23..26
        mbar_base += kv_load2cvt_pipe.num_mbars()  # barrier total [27]
        # We need (num_out_stages * 2) more barriers for the out pipeline.
        # num_out_stages = (Depth/BN_QK) / blocks_per_stage = 8/2 = 4, so 4*2 = 8.
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](
            mbar_base
        )  # Write uses 27 + (num_out_stages)*2
        mbar_base += out_pipeline.num_mbars()

        # Index pipeline: warp 11 (producer, 32 threads) transforms
        # d_indices → TMA rows and loads scales, then signals readiness.
        # Warp 8 (consumer) waits for indices+scales before issuing TMA.
        # Double-buffered (2 stages) so warp 11 can run 1 tile ahead.
        #
        # Consumer count: always 32 (all threads of warp 8).
        # For blockwise scaling, the warp-8 threads' idx_cons.wait()
        # (acquire semantics) makes warp 11's scale writes visible;
        # their kv_load2cvt arrive() (release semantics) then propagates
        # that visibility to the convert WG.
        # For tensorwise, only the leader uses the indices for TMA, but
        # all 32 threads participate in the barrier (zero extra cost
        # since they're in the same warp and arrive simultaneously).
        var idx_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=32, num_consumer=32
        ](mbar_base)
        mbar_base = idx_bars.end()  # +4 barriers

        # Per-block cvt→QK handoff bars, one per 64-col BF16 block per KV
        # stage (9 x 2): all 128 convert threads arrive, the QK warp waits.
        # Reuse across tiles is safe: the convert WG re-arms a stage's bars
        # only after its cvt2mma acquire, i.e. after QK/PV released them.
        var cvt_blk_bars: MBarType = mbar_base
        mbar_base += Self.NumQKBlocks * Self.config.num_kv_stages

        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()

        # Double-buffered SMEM for transformed gather4 row indices.
        # d_indices stores physical_block * page_size + offset; we use
        # kv_lut.get_tma_row() to convert to actual TMA row indices.
        # Two BN_QK-sized buffers (2 * 64 * 4 = 512 bytes) for pipelining
        # between warp 11 (producer) and warp 8 (consumer).
        var idx_smem_base = (ptr_tmem_addr + 1).bitcast[Int32]()
        comptime idx_smem_stride = Self.config.BN_QK  # 64 Int32 values per stage

        var warp_idx = UInt32(warp_id[broadcast=True]())
        var is_leader = elect() != 0
        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                # only one thread will load the Q
                kv_cvt2mma_pipe.init()
                s_bars.init()
                p_bars.init()
                kv_load2cvt_pipe.init()
                idx_bars.init()
                o_bars.init()
                c_bars.init()
                out_pipeline.init()
                corr_done_bars.init()
                comptime for i in range(
                    Self.NumQKBlocks * Self.config.num_kv_stages
                ):
                    cvt_blk_bars[i].init(Int32(WARPGROUP_SIZE))
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
                comptime if Self.has_extra_kv:
                    extra_k_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        barrier()

        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()

            # Compute per-head attn_sink_log2 for this thread.
            # Each thread in the warpgroup handles one head.
            # FlashMLA reference: kernel.cuh:149
            # When attn_sink_ptr is null, attn_sink_log2 stays at -inf,
            # and exp2(-inf - mi) = 0, so the denominator is unchanged.
            var attn_sink_log2 = Float32(min_or_neg_inf[.float32]())
            comptime if Self.has_attn_sink:
                var lane_idx = Int(lane_id())
                var row = lane_idx & 0x3F
                var head_idx_local = Int(block_idx.x) * Self.config.BM + row
                if head_idx_local < Self.config.num_q_heads:
                    attn_sink_log2 = attn_sink_ptr.unsafe_value()[
                        head_idx_local
                    ] * Float32(log2e)

            # Shared-index fold: drive the shared utils fold-Q layout — BM=64
            # packs q_len_fold * num_q_heads rows and each row keeps its own
            # causal horizon (cache_len + score_row // num_q_heads + 1). There
            # is no phase-select mask: all rows attend the ONE shared gather.
            Self.Common_MLA_Op.Softmax[
                has_attn_sink=Self.has_attn_sink,
                fold_q=Self.fold_shared_index,
                q_len_fold=Self.q_len_fold,
            ](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                kv_smem_bf16.as_unsafe_any_origin(),
                max_smem.as_unsafe_any_origin(),
                li_smem.as_unsafe_any_origin(),
                out_smem.as_unsafe_any_origin(),
                c_bars,
                corr_done_bars,
                out_pipeline,
                offset_position,
                scale,
                mask,
                prompt_idx=UInt32(offset_position.batch_idx),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
                attn_sink_log2=attn_sink_log2,
            )
        elif warp_idx >= 4 and warp_idx < 8:  # correction warpgroup
            warpgroup_reg_dealloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        elif warp_idx >= 8 and warp_idx < 12:
            warpgroup_reg_dealloc[num_reg_keep_mma_load_store]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    q_smem.as_unsafe_any_origin(),
                    kv_smem_fp8.as_unsafe_any_origin(),
                    mbar_q,
                    kv_load2cvt_pipe,
                    offset_position,
                    idx_bars,
                    idx_smem_base,
                    num_orig_blocks,
                    topk,
                    extra_k_tma,
                    extra_topk,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_smem.as_unsafe_any_origin(),
                    kv_smem_bf16.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_cvt2mma_pipe,
                    kv_load2cvt_pipe,
                    cvt_blk_bars,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem_bf16.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_cvt2mma_pipe,
                    kv_load2cvt_pipe,
                    offset_position,
                )
            elif warp_idx == 11:
                # --- Index transform producer ---
                # Warp 11 is idle during the main loop (store only
                # activates in the epilogue), so we use it to
                # transform d_indices → TMA rows and load scales
                # into SMEM, one tile ahead of warp 8's TMA loads.
                var batch_d_indices_w11 = d_indices.unsafe_value() + (
                    offset_position.q_token_idx * indices_stride
                )
                var batch_extra_d_indices_w11 = extra_d_indices
                comptime if Self.has_extra_kv:
                    batch_extra_d_indices_w11 = (
                        extra_d_indices.unsafe_value()
                        + (offset_position.q_token_idx * extra_indices_stride)
                    )
                Self.idx_producer(
                    idx_bars,
                    idx_smem_base,
                    kv_lut,
                    batch_d_indices_w11,
                    topk,
                    scales_ptr,
                    scale_smem_base.as_unsafe_any_origin(),
                    offset_position,
                    num_orig_blocks,
                    extra_kv_lut,
                    batch_extra_d_indices_w11,
                    extra_topk,
                    extra_scales_ptr,
                )
                # --- Output store epilogue ---
                # Shared-index fold: scatter the BM=64 packed rows back to each
                # folded q position's output row (out_row_offset_at(q_local)).
                Self.Common_MLA_Op.store[
                    fold_q=Self.fold_shared_index,
                    q_len_fold=Self.q_len_fold,
                ](
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    offset_position,
                )
        else:
            warpgroup_reg_alloc[num_reg_keep_fp8tofp16]()
            # Use num_keys_this_split for loop bounds (each split processes its portion)
            var num_k_tiles = ceildiv(
                offset_position.num_keys_this_split, Self.config.BN_QK
            )
            Self.convertFP8ToBF16(
                kv_smem_fp8.as_unsafe_any_origin(),
                kv_smem_bf16.as_unsafe_any_origin(),
                kv_load2cvt_pipe,
                kv_cvt2mma_pipe,
                cvt_blk_bars,
                num_k_tiles,
                scale_smem_base.as_unsafe_any_origin(),
            )
        barrier()

        # PDL: Signal that this CTA is done so dependent grids (combine kernel) can start.
        # This must be called by all threads in the CTA after all work is complete.
        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    @staticmethod
    @always_inline
    def _transform_indices_to_smem(
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        idx_smem: SharedMemPointer[Int32],
        indices_base: Int,
        kv_lut: Self.KVLUTType,
        topk: UInt32,
    ):
        """Transform d_indices from physical_block*page_size+offset to TMA rows.

        All 32 threads of warp 8 cooperatively transform BN_QK=64 indices.
        Each thread handles 2 indices (lane and lane+32).

        d_indices values encode: physical_block * page_size + offset.
        The kernel calls kv_lut.get_tma_row() to convert each encoded
        index to the actual TMA row.  Invalid indices (-1) are preserved.
        """
        var lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1

        comptime for row_pass in range(2):
            var row_in_tile = lane + row_pass * 32
            var idx_pos = UInt32(indices_base + row_in_tile)
            var clamped_pos = min(idx_pos, max_idx)
            var raw_index = d_indices.unsafe_value()[Int(clamped_pos)]
            # Convert encoded index to TMA row via the KV cache.
            var tma_row = kv_lut.get_tma_row(raw_index)
            # Preserve -1 for invalid entries.
            if raw_index == -1:
                tma_row = -1
            idx_smem[row_in_tile] = tma_row

    # ------------------------------------------------------------------
    # Index producer: warp 11 transforms indices and loads scales
    # ------------------------------------------------------------------
    @staticmethod
    @always_inline
    def idx_producer(
        idx_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        kv_lut: Self.KVLUTType,
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        topk: Int,
        scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
        scale_smem_base: SharedMemPointer[UInt8],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
        num_orig_blocks: Int,
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_topk: Int,
        extra_scales_ptr: OptionalReg[
            UnsafePointer[Float32, origin=MutAnyOrigin]
        ],
    ):
        """Index transform producer running on warp 11 (32 threads).

        For each KV tile, transforms d_indices to TMA rows and (when
        blockwise) loads FP32 scales to scale SMEM. Signals idx_bars
        when each tile's data is ready. Runs 1 tile ahead of warp 8.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var idx_prod = idx_bars.producer()
        var orig_topk_u32 = UInt32(topk)

        # Compute original vs extra tile counts (mirrors warp 8 logic).
        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        var first_tile_from_orig = num_orig_tiles > 0
        var orig_indices_base = Int(offset_position.kv_start_row)

        # KV pipeline stage index tracker — mirrors warp 8's double-buffer
        # alternation (0, 1, 0, 1, ...) so scales land in the correct
        # scale_smem stage that the convert WG will read.
        var kv_stage_idx = UInt32(0)

        # --- Produce first tile (no acquire — idx_free starts ready) ---
        if first_tile_from_orig:
            var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                Self.config.BN_QK
            )
            Self._transform_indices_to_smem(
                d_indices,
                idx_smem,
                orig_indices_base,
                kv_lut,
                orig_topk_u32,
            )
            comptime if Self.config.scale_block_size > 0:
                Self._load_scales_for_tile_sparse(
                    scale_smem_base,
                    scales_ptr,
                    kv_stage_idx,
                    d_indices,
                    orig_indices_base,
                    kv_lut,
                    orig_topk_u32,
                )
            idx_prod.commit()
            orig_indices_base += Self.config.BN_QK
            kv_stage_idx ^= 1

        # --- Remaining original tiles (acquire before produce) ---
        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        var t: Int = 0
        while t < remaining_orig:
            idx_prod.acquire()
            var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                Self.config.BN_QK
            )
            Self._transform_indices_to_smem(
                d_indices,
                idx_smem,
                orig_indices_base,
                kv_lut,
                orig_topk_u32,
            )
            comptime if Self.config.scale_block_size > 0:
                Self._load_scales_for_tile_sparse(
                    scale_smem_base,
                    scales_ptr,
                    kv_stage_idx,
                    d_indices,
                    orig_indices_base,
                    kv_lut,
                    orig_topk_u32,
                )
            idx_prod.commit()
            orig_indices_base += Self.config.BN_QK
            kv_stage_idx ^= 1
            t += 1

        # --- Extra KV tiles ---
        comptime if Self.has_extra_kv:
            var extra_topk_u32 = UInt32(extra_topk)
            var extra_indices_base = max(
                0, Int(offset_position.kv_start_row) - topk
            )
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                # First tile from extra cache — no acquire.
                var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                    Self.config.BN_QK
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                )
                comptime if Self.config.scale_block_size > 0:
                    Self._load_scales_for_tile_sparse(
                        scale_smem_base,
                        extra_scales_ptr,
                        kv_stage_idx,
                        extra_d_indices,
                        extra_indices_base,
                        extra_kv_lut,
                        extra_topk_u32,
                    )
                idx_prod.commit()
                extra_indices_base += Self.config.BN_QK
                kv_stage_idx ^= 1
                num_extra_tiles -= 1

            var te: Int = 0
            while te < num_extra_tiles:
                idx_prod.acquire()
                var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                    Self.config.BN_QK
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                )
                comptime if Self.config.scale_block_size > 0:
                    Self._load_scales_for_tile_sparse(
                        scale_smem_base,
                        extra_scales_ptr,
                        kv_stage_idx,
                        extra_d_indices,
                        extra_indices_base,
                        extra_kv_lut,
                        extra_topk_u32,
                    )
                idx_prod.commit()
                extra_indices_base += Self.config.BN_QK
                kv_stage_idx ^= 1
                te += 1

    # ------------------------------------------------------------------
    # Load: warp 8 consumes indices from idx pipeline, issues TMA
    # ------------------------------------------------------------------
    @staticmethod
    @always_inline
    def load(
        q_tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        # Single K gather4 TMA: INT64, 64 rows, SWIZZLE_NONE, 576-byte row.
        k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem_fp8: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=Self.load2cvt_num_producer,
            num_consumer=WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
        idx_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        # Extra KV parameters for the extended loop.
        num_orig_blocks: Int,
        topk: Int,
        extra_k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        extra_topk: Int,
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Early exit if this split has no work (prevents producer/consumer deadlock)
        if num_k_tiles == 0:
            return

        var kv_load_prod = KVLoad2CvtProducer[Self.fp8_type, Self.config](
            kv_load2cvt_pipe,
            kv_smem_fp8,
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset

        # Number of original KV tiles to load in this split.
        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        expect_bytes_pred(
            mbar_q,
            Int32(
                Self.config.BM * Self.config.q_depth * size_of[Self.q_type]()
            ),
            elect_mask,
        )
        if is_leader:
            Self.Common_MLA_Op.load_q(q_tma, q_smem, mbar_q, 0, row)

        # Full 576-byte row: BN_QK * 72 INT64 = 64 * 576 = 36864 bytes.
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var first_tile_from_orig = num_orig_tiles > 0
        var idx_cons = idx_bars.consumer()

        # --- First tile (no kv_load acquire — pipeline starts ready) ---
        if first_tile_from_orig:
            Self._load_one_tile(
                kv_load_prod,
                is_leader,
                k_tma,
                idx_cons,
                idx_smem_base,
                kv_bytes,
            )

        # --- Remaining original KV tiles (always acquire) ---
        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        Self._load_tile_range(
            kv_load_prod,
            is_leader,
            k_tma,
            idx_cons,
            idx_smem_base,
            remaining_orig,
        )

        # --- Load extra KV tiles ---
        comptime if Self.has_extra_kv:
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                # First tile from extra cache — no kv_load acquire.
                Self._load_one_tile(
                    kv_load_prod,
                    is_leader,
                    extra_k_tma,
                    idx_cons,
                    idx_smem_base,
                    kv_bytes,
                )
                num_extra_tiles -= 1

            Self._load_tile_range(
                kv_load_prod,
                is_leader,
                extra_k_tma,
                idx_cons,
                idx_smem_base,
                num_extra_tiles,
            )

    @staticmethod
    @always_inline
    def _load_scales_for_tile(
        scale_smem_base: SharedMemPointer[UInt8],
        scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
        kv_lut: Self.KVLUTType,
        stage_idx: UInt32,
        tile_kv_row_start: UInt32,
        batch_idx: UInt32,
        num_keys: UInt32,
    ):
        """Load FP32 scales from HBM, convert to e8m0, store to scale SMEM.

        Called by all 32 threads of warp 8. Each thread handles 2 rows
        (32 threads * 2 rows = 64 = BN_QK). For each row: ONE page table
        lookup via row_idx, then load all scales_per_token FP32 values
        and convert to e8m0 (1 byte each) in SMEM.
        """
        comptime scales_per_token = Self.config.scales_per_token
        var scale_smem_stage = scale_smem_base + stage_idx * UInt32(
            Self.config.scale_smem_per_stage
        )
        var lane = thread_idx.x & 31
        var max_key = max(num_keys, UInt32(1)) - 1

        # Each of 32 threads handles 2 rows (rows lane and lane+32).
        comptime for row_pass in range(2):
            var row_in_tile = lane + row_pass * 32
            var tok_idx = tile_kv_row_start + UInt32(row_in_tile)
            var clamped_tok = min(tok_idx, max_key)
            # ONE page table lookup per row.
            var gmem_row = kv_lut.row_idx(batch_idx, clamped_tok)
            var row_base = scales_ptr + Int(gmem_row) * scales_per_token
            var smem_off = row_in_tile * scales_per_token

            # Cast each FP32 scale to e8m0 individually.
            # the scale per token is odd and manually doing the pair packing
            # did not improve the performance.
            # The compiler may still emit the 2x instruction
            # if it sees the opportunity.
            comptime for s in range(scales_per_token):
                var fp32_val = row_base[s]
                scale_smem_stage[smem_off + s] = bitcast[.uint8](
                    fp32_val.cast[.float8_e8m0fnu]()
                )

    @staticmethod
    @always_inline
    def _load_one_tile(
        mut kv_load_prod: KVLoad2CvtProducer[Self.fp8_type, Self.config],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[2],
        idx_smem_base: SharedMemPointer[Int32],
        kv_bytes: Int,
    ):
        """Load the first KV tile without kv_load pipeline acquire.

        Waits for warp 11 to produce indices (idx_cons.wait), then
        issues a single 576-byte gather4 TMA (covering full nope+rope
        as FP8) and signals idx_cons.release. The kv_load2cvt pipeline
        starts ready, so no acquire is needed.
        """
        var kv_stage_ptr = kv_load_prod.stage_base_ptr[qk_stage=0]()
        var k_mbar = kv_load_prod.producer_mbar[qk_stage=0]()

        # Wait for warp 11 to produce indices + scales for this tile.
        idx_cons.wait()
        var idx_smem = idx_smem_base + idx_cons.state.index() * UInt32(
            Self.config.BN_QK
        )

        expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))
        if is_leader:
            cur_k_tma.async_copy_gather4_tile[
                tile_width=Self.config.input_q_depth // 8,
                eviction_policy=CacheEviction.EVICT_LAST,
            ](
                kv_stage_ptr.bitcast[Int64](),
                k_mbar[],
                idx_smem,
                start_idx=0,
            )

        # Blockwise: all 32 threads arrive on kv_load2cvt producer mbar
        # to propagate scale visibility (written by warp 11) to the
        # convert WG via the acquire/release chain.
        comptime if Self.config.scale_block_size > 0:
            _ = k_mbar[].arrive()

        # Release idx buffer so warp 11 can overwrite for the next tile.
        idx_cons.release()
        kv_load_prod.commit_step()

    @staticmethod
    @always_inline
    def _load_tile_range(
        mut kv_load_prod: KVLoad2CvtProducer[Self.fp8_type, Self.config],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[2],
        idx_smem_base: SharedMemPointer[Int32],
        num_tiles: Int,
    ):
        """Load remaining KV tiles from a cache using a single gather4 TMA.

        Each iteration: acquire kv_load stage, wait for idx_cons,
        issue TMA (full 576-byte row), release idx_cons, commit kv_load.
        """
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var t: Int = 0
        while t < num_tiles:
            kv_load_prod.acquire[qk_stage=0]()

            var kv_stage_ptr = kv_load_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_load_prod.producer_mbar[qk_stage=0]()

            # Wait for warp 11 to produce indices + scales.
            idx_cons.wait()
            var idx_smem = idx_smem_base + idx_cons.state.index() * UInt32(
                Self.config.BN_QK
            )

            expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))
            if is_leader:
                cur_k_tma.async_copy_gather4_tile[
                    tile_width=Self.config.input_q_depth // 8,
                    eviction_policy=CacheEviction.EVICT_LAST,
                ](
                    kv_stage_ptr.bitcast[Int64](),
                    k_mbar[],
                    idx_smem,
                    start_idx=0,
                )

            comptime if Self.config.scale_block_size > 0:
                _ = k_mbar[].arrive()

            idx_cons.release()
            kv_load_prod.commit_step()
            t += 1

    @staticmethod
    @always_inline
    def _load_scales_for_tile_sparse(
        scale_smem_base: SharedMemPointer[UInt8],
        scales_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]],
        stage_idx: UInt32,
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        indices_base: Int,
        kv_lut: Self.KVLUTType,
        topk: UInt32,
    ):
        """Load FP32 scales for sparse tokens, convert to e8m0.

        Like _load_scales_for_tile but reads the encoded index from d_indices
        and uses kv_lut.get_tma_row() to convert each encoded index to the
        physical TMA row.

        Called by all 32 threads of warp 11. Each thread handles 2 rows
        (32 threads * 2 rows = 64 = BN_QK).
        """
        comptime scales_per_token = Self.config.scales_per_token
        var scale_smem_stage = scale_smem_base + stage_idx * UInt32(
            Self.config.scale_smem_per_stage
        )
        var lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1

        # Each of 32 threads handles 2 rows (rows lane and lane+32).
        comptime for row_pass in range(2):
            var row_in_tile = lane + row_pass * 32
            var idx_pos = UInt32(indices_base + row_in_tile)
            # Clamp to valid range within topk.
            var clamped_pos = min(idx_pos, max_idx)
            # Read encoded index and convert to TMA row via the KV cache.
            var raw_index = d_indices.unsafe_value()[Int(clamped_pos)]
            var gmem_row = Int(kv_lut.get_tma_row(raw_index))
            var row_base = scales_ptr.unsafe_value() + (
                gmem_row * scales_per_token
            )
            var smem_off = row_in_tile * scales_per_token

            comptime for s in range(scales_per_token):
                var fp32_val = row_base[s]
                scale_smem_stage[smem_off + s] = bitcast[.uint8](
                    fp32_val.cast[.float8_e8m0fnu]()
                )

    @staticmethod
    @always_inline
    def _cvt_block[
        c: Int
    ](
        src_u8: SharedMemPointer[UInt8],
        scale_smem_stage: SharedMemPointer[UInt8],
        fp8_base_0: Int,
        fp8_base_1: Int,
        row_0: Int,
        row_1: Int,
        idx_in_group: Int,
    ) -> StaticTuple[SIMD[.uint32, 4], 4]:
        """Loads and converts this thread's 2x16 FP8 bytes of col-block `c`.

        Returns four packed-BF16 u32x4 chunks (row 0 lo/hi, row 1 lo/hi),
        scaled when blockwise.
        """
        comptime GROUP_SIZE: Int = 4
        # Column byte offset: 4 threads * 16 bytes = 64 bytes per block.
        comptime col_byte_off: Int = c * GROUP_SIZE * 16

        var q0 = ld_shared_v4_u32(src_u8, fp8_base_0 + col_byte_off)
        var q1 = ld_shared_v4_u32(src_u8, fp8_base_1 + col_byte_off)

        var p0 = cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4[
            fp8_dtype=Self.fp8_type,
            out_dtype=Self.q_type,
        ](q0)
        var p1 = cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4[
            fp8_dtype=Self.fp8_type,
            out_dtype=Self.q_type,
        ](q1)

        # Blockwise scaling: absolute column for these 16 FP8 bytes is
        # c * GROUP_SIZE * 16 + idx_in_group * 16. The 16 bytes span at
        # most one scale block when scale_block_size >= 16 (minimum 32).
        comptime if Self.config.scale_block_size > 0:
            var abs_col = c * GROUP_SIZE * 16 + idx_in_group * 16
            var scale_idx = abs_col // Self.config.scale_block_size
            var s0 = e8m0_to_bf16_broadcast(
                scale_smem_stage[
                    row_0 * Self.config.scales_per_token + scale_idx
                ]
            )
            var s1 = e8m0_to_bf16_broadcast(
                scale_smem_stage[
                    row_1 * Self.config.scales_per_token + scale_idx
                ]
            )
            p0[0] = hmul2_bf16x8_by_scalar[Self.q_type](p0[0], s0)
            p0[1] = hmul2_bf16x8_by_scalar[Self.q_type](p0[1], s0)
            p1[0] = hmul2_bf16x8_by_scalar[Self.q_type](p1[0], s1)
            p1[1] = hmul2_bf16x8_by_scalar[Self.q_type](p1[1], s1)

        return StaticTuple[SIMD[.uint32, 4], 4](p0[0], p0[1], p1[0], p1[1])

    @staticmethod
    @always_inline
    def convertFP8ToBF16(
        kv_smem_fp8: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem_bf16: SharedMemPointer[Scalar[Self.q_type]],
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=Self.load2cvt_num_producer,
            num_consumer=WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ],
        kv_cvt2mma_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ],
        cvt_blk_bars: MBarType,
        num_k_tiles: Int,
        scale_smem_base: SharedMemPointer[UInt8],
    ):
        # Convert the full 576-byte row (nope+rope as FP8 → BF16).
        # FP8 data lands linearly in SMEM (INT64/SWIZZLE_NONE TMA):
        #   FP8 byte address = row * fp8_row_stride + col_byte
        # BF16 output uses SWIZZLE_128B layout.
        #
        # GROUP_SIZE=4 thread mapping to reduce bank conflicts:
        #   128 threads -> 32 groups of 4
        #   Each group handles 64/32 = 2 rows
        #   Each thread reads 16 FP8 bytes (v4.b32) per position
        #   4 threads * 16 bytes = 64 bytes per row per column iteration
        #   9 column iterations cover all 576 bytes per row
        # This reduces read bank conflicts from 32-way to 8-way while
        # keeping the same v4.b32 load width and instruction count.
        comptime sw_bf16 = make_swizzle[Self.q_type, Self.config.swizzle_mode]()

        comptime BN_QK: Int = Self.config.BN_QK
        # FP8 -> BF16 conversion slices the full row (padded_q_depth=576,
        # nope+rope) into 64-element chunks. Chunk size is independent
        # of the QK MMA's N tile width (BN_QK).
        comptime BK_QK_chunk: Int = 64
        comptime BlockElems: Int = Self.config.BM * BK_QK_chunk
        comptime fp8_row_stride: Int = Self.config.padded_q_depth  # 576

        # GROUP_SIZE=4 constants.
        comptime GROUP_SIZE: Int = 4
        comptime NUM_GROUPS: Int = WARPGROUP_SIZE // GROUP_SIZE  # 32
        comptime ROWS_PER_GROUP: Int = Self.config.BN_QK // NUM_GROUPS  # 2
        # Each thread reads 16 FP8 bytes per column position.
        # 576 / (4 threads * 16 bytes) = 9 column iterations.
        comptime COLS_PER_GROUP: Int = Self.config.padded_q_depth // (
            GROUP_SIZE * 16
        )  # 9

        var kv_load_cons_cvt = KVLoad2CvtConsumer[Self.fp8_type, Self.config](
            kv_load2cvt_pipe,
            kv_smem_fp8,
        )
        var kv_cvt_prod = KVCvt2MmaProducer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem_bf16
        )
        var lane: Int = thread_idx.x & 0x7F
        var group_idx: Int = lane // GROUP_SIZE  # 0..31
        var idx_in_group: Int = lane % GROUP_SIZE  # 0..3

        # Row assignment: interleaved so adjacent groups map to
        # adjacent rows for better SMEM locality.
        # Row r = local_row_idx * NUM_GROUPS + group_idx.
        var row_0: Int = 0 * NUM_GROUPS + group_idx
        var row_1: Int = 1 * NUM_GROUPS + group_idx

        # Precompute FP8 read base offsets (byte addresses).
        # Each thread reads 16 bytes starting at:
        #   row * fp8_row_stride + idx_in_group * 16
        var fp8_base_0: Int = row_0 * fp8_row_stride + idx_in_group * 16
        var fp8_base_1: Int = row_1 * fp8_row_stride + idx_in_group * 16

        # Precompute BF16 swizzled write offsets.
        # Each 16-byte v4.b32 load gives 16 FP8 values split into two
        # 8-BF16 chunks (p_a at col, p_b at col+8).
        # The BF16 column start for this thread: idx_in_group * 16.
        var col_bf16: Int = idx_in_group * 16
        var bf16_sw_0a: Int = sw_bf16(row_0 * BN_QK + col_bf16)
        var bf16_sw_0b: Int = sw_bf16(row_0 * BN_QK + col_bf16 + 8)
        var bf16_sw_1a: Int = sw_bf16(row_1 * BN_QK + col_bf16)
        var bf16_sw_1b: Int = sw_bf16(row_1 * BN_QK + col_bf16 + 8)

        # The FP8 raw slot overlays the upper half of its own BF16 stage
        # (kv_smem_fp8 = bf16 base + KVStageFP8Bytes, stride 2x that).
        # Blocks below the overlay never alias the FP8 source: store+publish
        # as they convert.  Blocks at/above it clobber this stage's FP8 rows,
        # so their stores wait for the all-reads named barrier.
        comptime num_free_blocks = Self.KVStageFP8Bytes // (
            Self.BlockElems * Self.bf16_bytes_per_element
        )  # 36864 / 8192 = 4
        comptime num_held_blocks = COLS_PER_GROUP - num_free_blocks  # 5

        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            kv_load_cons_cvt.wait()
            kv_cvt_prod.acquire()

            var src_u8 = kv_load_cons_cvt.stage_base_ptr().bitcast[UInt8]()
            var dst = kv_cvt_prod.stage_base_ptr()

            var cvt_stage_idx = kv_load_cons_cvt.pipe.state.index()
            var scale_smem_stage = scale_smem_base + cvt_stage_idx * UInt32(
                Self.config.scale_smem_per_stage
            )
            var blk_bar0 = cvt_blk_bars + kv_cvt_prod.stage_index() * UInt32(
                Self.NumQKBlocks
            )

            # Non-overlay blocks: convert, store, publish per block so the
            # QK MMA starts while the rest of the row converts.
            comptime for c in range(num_free_blocks):
                var p = Self._cvt_block[c](
                    src_u8,
                    scale_smem_stage,
                    fp8_base_0,
                    fp8_base_1,
                    row_0,
                    row_1,
                    idx_in_group,
                )
                var dst_block = dst + c * BlockElems
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block, bf16_sw_0a, p[0]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block, bf16_sw_0b, p[1]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block, bf16_sw_1a, p[2]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block, bf16_sw_1b, p[3]
                )
                fence_async_view_proxy()
                _ = blk_bar0[c].arrive()

            # Overlay blocks: convert to registers; stores deferred until
            # all FP8 reads finish.
            var p0a_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, num_held_blocks]())
            var p0b_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, num_held_blocks]())
            var p1a_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, num_held_blocks]())
            var p1b_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, num_held_blocks]())

            comptime for c in range(num_free_blocks, COLS_PER_GROUP):
                var p = Self._cvt_block[c](
                    src_u8,
                    scale_smem_stage,
                    fp8_base_0,
                    fp8_base_1,
                    row_0,
                    row_1,
                    idx_in_group,
                )
                p0a_all.ptr.store((c - num_free_blocks) * 4, p[0])
                p0b_all.ptr.store((c - num_free_blocks) * 4, p[1])
                p1a_all.ptr.store((c - num_free_blocks) * 4, p[2])
                p1b_all.ptr.store((c - num_free_blocks) * 4, p[3])

            # Single barrier: all 128 threads finish reads before the
            # overlay-aliased writes below.
            named_barrier[Int32(WARPGROUP_SIZE)](3)

            # FP8 reads done: free the raw slot so the next gather can
            # overlap our stores.  The QK/PV arrivals on the same mbar
            # still order the refill after the bf16 stores.
            kv_load_cons_cvt.release_all()

            # Store the overlay blocks, publishing each as it completes.
            comptime for c in range(num_free_blocks, COLS_PER_GROUP):
                var dst_block = dst + c * BlockElems
                comptime held = (c - num_free_blocks) * 4

                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    bf16_sw_0a,
                    p0a_all.ptr.load[width=4](held),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    bf16_sw_0b,
                    p0b_all.ptr.load[width=4](held),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    bf16_sw_1a,
                    p1a_all.ptr.load[width=4](held),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    bf16_sw_1b,
                    p1b_all.ptr.load[width=4](held),
                )
                fence_async_view_proxy()
                _ = blk_bar0[c].arrive()

            kv_cvt_prod.commit_all()
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MLA decoding MMA for Q, K, V, P blocks
    # --------------------------------------------------------------------------

    # -------------------------------------------------
    # PIPELINE LOOP:
    #   loop over tiles 1..num_k_tiles-1
    #   each iteration does:
    #     - PV(tile_idx-1) with prev_stage_idx  (then release its KV stage)
    #     - QK(tile_idx) with the next KV stage
    # -------------------------------------------------
    # QK process the Numkey vertically, meaning the C Scale for the first
    # block of all tiles is going to be zero the PV multiply the P horizontally
    # to V meaning only the C scale for prev tile for is going to be Zero for all
    # 9 block and after that it is going to be 1
    #                Q                                              KV0/1
    #   ___ ___ ___ ___ ___ ___ ___ ___ ___       ___ ___ ___ ___ ___ ___ ___ ___ ___
    #  |___|___|___|___|___|___|___|___|___|  T0 |___|___|___|___|___|___|___|___|___|
    #                                         T1 |___|___|___|___|___|___|___|___|___|
    #                                         T2 |___|___|___|___|___|___|___|___|___|
    #                                         T3 |___|___|___|___|___|___|___|___|___|
    #     S0     S1     S0    S1
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|
    #
    #     P0     P0     P0    P0
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|

    # We move it to It might be possible to create two P slot and put it at the
    # last slot of KV pipeline, Need to verify if that gives better performance.
    # QK process the Numkey vertically, meaning the C Scale for the first block
    # of all tiles is going to be zero the PV multiply the P horizontally to V
    # meaning only the C scale for prev tile for is going to be Zero for all 9 block
    # and after that it is going to be 1
    #                Q                                              KV0/1
    #   ___ ___ ___ ___ ___ ___ ___ ___ ___       ___ ___ ___ ___ ___ ___ ___ ___ _______
    #  |___|___|___|___|___|___|___|___|___|  T0 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T1 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T2 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T3 |___|___|___|___|___|___|___|___|__P0/1_|
    #     S0    S1    S0    S1
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|
    #
    #   P0     P1    P0    P1
    #  ______ ______ ______ ______
    # |__T0__|__T1__|__T2__|__T3__|

    @staticmethod
    @always_inline
    def mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_cvt2mma_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ],
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=Self.load2cvt_num_producer,
            num_consumer=WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ],
        cvt_blk_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
    ):
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()
        # Use num_keys_this_split for loop bounds (each split processes its portion)
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        var kv_cons = KVCvt2MmaConsumer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem
        )
        # ---  S producer wrapper (2-stage pipeline) ---
        var s_prod = DecodeSProducer(s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element

        mbar_q[].wait(0)
        var tile_idx: Int = 0

        while tile_idx < num_k_tiles:
            s_prod.acquire()

            var slot_idx: UInt32 = s_prod.slot_index()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            # Per-block gating: each 64-col K block issues as soon as the
            # convert WG publishes it on cvt_blk_bars, rather than waiting
            # for the whole stage on kv_cons.
            var k_slot_index = kv_cons.stage_index[qk_stage=0]()
            var b_desc = k_descriptor + k_slot_index * UInt32(
                stage_stride_in_bytes
            )
            var blk_bar0 = cvt_blk_bars + k_slot_index * UInt32(
                Self.NumQKBlocks
            )
            # Slot s is used at tiles s, s+2, ...; its n-th use waits
            # parity n & 1.
            var blk_parity = UInt32((tile_idx >> 1) & 1)

            comptime for blk in range(Self.NumQKBlocks):
                blk_bar0[blk].wait(blk_parity)
                Self.UMMAQKTSS.mma_block[
                    block_idx=blk, num_blocks=Self.NumQKBlocks
                ](
                    a=q_descriptor,
                    b=b_desc,
                    c=s_tmem_slot,
                    c_scale=UInt32(0),
                    elect=elect_mask,
                )
            tcgen05_fence_before()
            s_prod.commit_mma(elect_mask)
            kv_cons.release[qk_stage=0](elect_mask)
            elect_mma_arrive(
                kv_load2cvt_pipe.consumer_mbar[0](k_slot_index), elect_mask
            )
            tile_idx += 1

    @staticmethod
    @always_inline
    def mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_cvt2mma_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ],
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=Self.load2cvt_num_producer,
            num_consumer=WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        # ---  S producer wrapper (2-stage pipeline) ---
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = KVCvt2MmaConsumer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem
        )
        var p_cons = DecodePConsumer(p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())
        var p_smem_base = kv_smem + Self.NumVOBlocks * Self.BlockElems
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem_base)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.bytes_per_element

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            # PV does not have the k-rope so we don't need to do the last block
            # that block is used for P
            comptime for block in range(0, Self.NumVOBlocks, block_step):
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(stage_stride_in_bytes)
                    + UInt32(block * block_stride_in_bytes),
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN_QK // 2),
                    c_scale=c_scale,
                    elect=elect_mask,
                )
                o_prod.commit_mma(elect_mask)
            p_cons.release_mma(elect_mask)

            kv_cons.release[qk_stage=0](elect_mask)
            elect_mma_arrive(
                kv_load2cvt_pipe.consumer_mbar[0](v_slot_index), elect_mask
            )
            tcgen05_fence_before()
            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
