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

"""Implements the SM100 MLA decode kernel variant that loads KV cache in FP8 and converts to BF16 in shared memory before MMA."""

from std.math import ceildiv
from std.sys import size_of
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx,
    block_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import external_memory, fence_async_view_proxy
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
)
from std.memory import bitcast
from layout import (
    ComptimeInt,
    CoordLike,
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
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    elect_mma_arrive,
    expect_bytes_pred,
    SharedMemPointer,
    MBarType,
)
from nn.attention.gpu.nvidia.common import KVTMATile

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
    cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4,
    st_shared_v4_b32_at_bf16_elem_off,
    e8m0_to_bf16_broadcast,
    hmul2_bf16x8_by_scalar,
)


# ------------------------------------------------------------------------------
# MLA decoding kernel struct for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_KV_FP8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    Engine: TensorEngine,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](TrivialRegisterPassable):
    """FP8 KV decode kernel for MLA attention on SM100 GPUs.

    This kernel uses 4 warpgroups plus 4 individual warps, where a dedicated
    convert warpgroup performs FP8-to-BF16 conversion in shared memory before
    the MMA warps consume the data. Two KV pipelines coordinate the load,
    convert, and MMA stages in a double-buffered pipeline.

    Parameters:
        q_type: Element type of the query tensor; also the target type for
            FP8-to-BF16 KV conversion before MMA.
        KVLUTType: Operand describing the paged KV cache lookup table,
            including its element type and page table layout.
        output_type: Element type of the output tensor written by the
            store warp.
        SplitAccumType: `OptionalPointer` type for the split-K partial
            output accumulation buffer, or `NullPointer` when split-K is
            disabled.
        MaskType: Attention mask applied to the QK scores; only `NullMask`
            and `CausalMask` are supported.
        config: Tile sizes, thread counts, pipeline stages, swizzle modes,
            and tuning parameters for the decode kernel.
        ValidLengthType: `OptionalPointer` type for the per-batch valid key
            length tensor, or `NullPointer` when not provided.
        Engine: Engine policy of the `scalar_args` tile operand.
        _is_cache_length_accurate: Whether the reported cache length is
            exact (defaults to `False`).
        ragged: Whether variable-length sequences are enabled, allowing
            early exit for batches with fewer query tokens (defaults to
            `False`).
    """

    comptime kv_type = Self.KVLUTType.dtype
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.q_type]()
    # the stage element is the same for both K and V
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
    # MLA decoding main kernel function (FP8 variant — 4 warpgroups)
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
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"sm100_mla_decode_kv_fp8_{Self.q_type}_{Self.kv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        q_tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        k_tma: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BK_PV,  # tile_m =64
            BK=Self.config.input_q_depth,
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
        scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
        scalar_args: TileTensor[
            .int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
            Engine=Self.Engine,
        ],
    ):
        # SlidingWindowCausalMask is supported ONLY by the native FP8 backend
        # (MLA_SM100_Decode_QKV_FP8). Reject it here at comptime.
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_KV_FP8 only supports NullMask and CausalMask."
            " Sliding window is supported only by MLA_SM100_Decode_QKV_FP8"
            " (native FP8)."
        )
        # Softmax now includes the epilogue, so it needs more registers
        # Correction does less work now (no epilogue), so it needs fewer
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 72
        comptime num_reg_keep_mma_load_store = 72
        comptime num_reg_keep_fp8tofp16 = 184
        var batch_size = Int(scalar_args.raw_load(0))
        var q_max_seq_len = Int(scalar_args.raw_load(1))
        var num_partitions = mla_decode_pack.num_partitions
        var mask = mla_decode_pack.mask
        var valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr
        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
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
        )

        # Early exit for split-K: CTAs with no work (num_keys_this_split == 0)
        # must still write -inf LSE and call launch_dependent_grids()
        # to fulfill the PDL contract with the combine kernel. Skipping
        # launch_dependent_grids() causes the combine kernel to hang,
        # leading to CUDA_ERROR_ILLEGAL_ADDRESS.
        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                Self.Common_MLA_Op.pdl_early_exit(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                )
                return

        # early exit: Skip blocks beyond actual sequence length for this batch
        # In ragged mode with split-K, q_max_seq_len can be > 1 (up to 8).
        # block_idx.y ranges from 0 to q_max_seq_len-1, but some sequences
        # may have fewer tokens. CTAs with block_idx.y >= seq_len must still
        # fulfill the PDL contract (write -inf LSE and call
        # launch_dependent_grids) or the combine kernel will hang.
        comptime if Self.ragged:
            # In ragged mode, block_idx.y is the query token index (0 to q_max_seq_len-1)
            # But this batch might have fewer tokens than q_max_seq_len
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    Self.Common_MLA_Op.pdl_early_exit(
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                    )

                return  # This query position doesn't exist for this batch
        var q_smem = external_memory[
            Scalar[Self.q_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        var kv_smem_bf16 = q_smem + Self.BlockElems * Self.NumQKBlocks
        var kv_smem_fp8_upper0 = (
            kv_smem_bf16.bitcast[Scalar[Self.KVLUTType.dtype]]()
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

        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

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

        # barrier total [27 + (num_out_stages)*2]
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()

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
                o_bars.init()
                c_bars.init()
                out_pipeline.init()
                corr_done_bars.init()
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        barrier()

        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()
            Self.Common_MLA_Op.Softmax(
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
                    kv_lut,
                    q_smem.as_unsafe_any_origin(),
                    kv_smem_fp8_upper0.as_unsafe_any_origin(),
                    mbar_q,
                    kv_load2cvt_pipe,
                    offset_position,
                    scale_smem_base.as_unsafe_any_origin(),
                    scales_ptr,
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
                Self.Common_MLA_Op.store(
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
                kv_smem_fp8_upper0.as_unsafe_any_origin(),
                kv_smem_bf16.as_unsafe_any_origin(),
                kv_load2cvt_pipe,
                kv_cvt2mma_pipe,
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
    def load(
        q_tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        k_tma_fp8: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BK_PV,  # tile_m =64
            BK=Self.config.input_q_depth,
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem_fp8: SharedMemPointer[Scalar[Self.kv_type]],
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
        ],
        scale_smem_base: SharedMemPointer[UInt8],
        scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Early exit if this split has no work (prevents producer/consumer deadlock)
        if num_k_tiles == 0:
            return

        # Alignment of `kv_row` produced by mask-driven iteration.
        comptime base_alignment: Int = Self.MaskType.start_column_alignment[
            Self.config.BM, Self.config.BN_QK, Self.KVLUTType.page_size
        ]()

        var kv_load_prod = KVLoad2CvtProducer[Self.kv_type, Self.config](
            kv_load2cvt_pipe,
            kv_smem_fp8,
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset
        # Start KV from kv_start_row for split-K support
        var kv_row: UInt32 = UInt32(offset_position.kv_start_row)
        # Clamp kv_row to prevent OOB lookup_table access on the last tile.
        var num_keys_u32 = UInt32(offset_position.num_keys)
        kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
        var paged_rows = kv_lut.populate[Self.config.BN_QK, base_alignment](
            UInt32(offset_position.batch_idx), kv_row
        )

        expect_bytes_pred(
            mbar_q,
            Int32(
                Self.config.BM * Self.config.q_depth * size_of[Self.q_type]()
            ),
            elect_mask,
        )
        if is_leader:
            Self.Common_MLA_Op.load_q(q_tma, q_smem, mbar_q, 0, row)

        var k0_bar: MBarType = kv_load_prod.producer_mbar[qk_stage=0]()

        expect_bytes_pred(
            k0_bar,
            Int32(
                Self.config.BN_QK
                * Self.config.q_depth
                * size_of[Self.kv_type]()
            ),
            elect_mask,
        )
        var stage_ptr = kv_load_prod.stage_base_ptr[qk_stage=0]()
        paged_rows.tma_copy_k[needs_partial=False](
            k_tma_fp8,
            stage_ptr,
            k0_bar[],
            kv_head_idx=UInt32(0),
            elect=elect_mask,
        )

        # Load blockwise scales for tile 0 (all warp 8 threads load scales
        # into scale SMEM stage matching the KV pipeline stage).
        # Each thread's mbar.arrive() has release semantics, making its
        # prior SMEM writes visible to the converter's mbar.wait() (acquire).
        comptime if Self.config.scale_block_size > 0:
            var stage_idx = kv_load_prod.pipe.state.index()
            Self._load_scales_for_tile(
                scale_smem_base,
                scales_ptr,
                kv_lut,
                stage_idx,
                UInt32(offset_position.kv_start_row),
                UInt32(offset_position.batch_idx),
                num_keys_u32,
            )
            # Signal scale stores via the kv_load2cvt pipeline mbar.
            # Each thread's arrive() performs a release, covering its
            # prior SMEM writes. The converter's mbar.wait() (acquire)
            # will see all scale data once all 33 arrivals complete
            # (1 from expect_bytes + 32 from warp 8 threads).
            _ = k0_bar[].arrive()

        kv_load_prod.commit_step()

        kv_row += UInt32(Self.config.BN_QK)

        var tile_idx: Int = 1
        while tile_idx < num_k_tiles:
            kv_load_prod.acquire[qk_stage=0]()

            var stage_ptr = kv_load_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_load_prod.producer_mbar[qk_stage=0]()

            kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
            var paged_rows = kv_lut.populate[Self.config.BN_QK, base_alignment](
                UInt32(offset_position.batch_idx), kv_row
            )

            expect_bytes_pred(
                k_mbar,
                Int32(
                    Self.config.BN_QK
                    * Self.config.q_depth
                    * size_of[Self.kv_type]()
                ),
                elect_mask,
            )
            paged_rows.tma_copy_k[needs_partial=False](
                k_tma_fp8,
                stage_ptr,
                k_mbar[],
                kv_head_idx=UInt32(0),
                elect=elect_mask,
            )

            # Load blockwise scales for this tile (all warp 8 threads).
            comptime if Self.config.scale_block_size > 0:
                var stage_idx = kv_load_prod.pipe.state.index()
                Self._load_scales_for_tile(
                    scale_smem_base,
                    scales_ptr,
                    kv_lut,
                    stage_idx,
                    kv_row,
                    UInt32(offset_position.batch_idx),
                    num_keys_u32,
                )
                # Signal scale stores via the kv_load2cvt pipeline mbar.
                # Each thread's arrive() performs a release, covering its
                # prior SMEM writes. No separate named barrier needed.
                _ = k_mbar[].arrive()

            kv_row += UInt32(Self.config.BN_QK)
            kv_load_prod.commit_step()

            tile_idx += 1

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
    def convertFP8ToBF16(
        kv_smem_fp8: SharedMemPointer[Scalar[Self.kv_type]],
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
        num_k_tiles: Int,
        scale_smem_base: SharedMemPointer[UInt8],
    ):
        comptime sw_fp8 = make_swizzle[
            Self.kv_type, Self.config.kv_tma_swizzle_mode
        ]()
        comptime sw_bf16 = make_swizzle[Self.q_type, Self.config.swizzle_mode]()

        comptime BN_QK: Int = Self.config.BN_QK
        # FP8 -> BF16 conversion slices q_depth (576) into 64-element
        # chunks. This chunk size is independent of the QK MMA's N
        # tile width (BN_QK).
        comptime BK_QK_chunk: Int = 64
        comptime NumBlocks: Int = Self.config.q_depth // BK_QK_chunk
        comptime BlockElems: Int = Self.config.BM * BK_QK_chunk

        var kv_load_cons_cvt = KVLoad2CvtConsumer[Self.kv_type, Self.config](
            kv_load2cvt_pipe,
            kv_smem_fp8,
        )
        var kv_cvt_prod = KVCvt2MmaProducer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem_bf16
        )
        var lane: Int = thread_idx.x & 0x7F
        var row: Int = lane & 0x3F
        # XOR the half selection with row bits to spread
        # conflicting rows across different column halves.
        # Original pattern: rows 0,8,16,24 all access banks 0-3 (4-way conflict)
        # With this fix: rows 0,16 access col0, rows 8,24 access col32
        var half: Int = (lane >> 6) ^ ((row >> 3) & 1)
        var col0: Int = half * 32

        var direct0: Int = row * BN_QK + col0
        var direct1: Int = row * BN_QK + col0 + 16

        var phys_fp8_0: Int = sw_fp8(direct0)
        var phys_fp8_1: Int = sw_fp8(direct1)

        var phys_bf16_0a: Int = sw_bf16(direct0)
        var phys_bf16_0b: Int = sw_bf16(direct0 + 8)
        var phys_bf16_1a: Int = sw_bf16(direct1)
        var phys_bf16_1b: Int = sw_bf16(direct1 + 8)

        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            kv_load_cons_cvt.wait()
            kv_cvt_prod.acquire()

            var src_u8 = kv_load_cons_cvt.stage_base_ptr().bitcast[UInt8]()
            var dst = kv_cvt_prod.stage_base_ptr()

            # Compute the scale SMEM stage pointer for blockwise scaling.
            # When scale_block_size == 0, scale_smem_per_stage is 0 so
            # this pointer is never dereferenced (guarded by comptime if).
            # Scale visibility is guaranteed by the kv_load2cvt_pipe mbar:
            # warp 8's per-thread mbar.arrive() (release) after scale stores
            # ensures this mbar.wait() (acquire) sees all scale data.
            var cvt_stage_idx = kv_load_cons_cvt.pipe.state.index()
            var scale_smem_stage = scale_smem_base + cvt_stage_idx * UInt32(
                Self.config.scale_smem_per_stage
            )

            # First: Load all FP8 data and convert to BF16 in registers
            # This approach loads ALL blocks first, then uses ONE barrier,
            # then stores ALL blocks. This will significantly reduce the number of barriers.
            # and improve the performance (18 barriers vs 1 barrier here).
            var p0a_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, NumBlocks]())
            var p0b_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, NumBlocks]())
            var p1a_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, NumBlocks]())
            var p1b_all = tt_stack_allocation[
                dtype=.uint32, address_space=.LOCAL
            ](row_major[4, NumBlocks]())

            comptime for b in range(NumBlocks):
                var src_block_u8 = src_u8 + b * BlockElems

                var q0 = ld_shared_v4_u32(src_block_u8, phys_fp8_0)
                var q1 = ld_shared_v4_u32(src_block_u8, phys_fp8_1)

                var p0a = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype=Self.kv_type,
                    out_dtype=Self.q_type,
                ](q0[0], q0[1])

                var p0b = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype=Self.kv_type,
                    out_dtype=Self.q_type,
                ](q0[2], q0[3])

                var p1a = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype=Self.kv_type,
                    out_dtype=Self.q_type,
                ](q1[0], q1[1])

                var p1b = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype=Self.kv_type,
                    out_dtype=Self.q_type,
                ](q1[2], q1[3])

                # Blockwise scaling: multiply converted BF16 values by the
                # e8m0 scale for this (row, block) from scale SMEM.
                # scale_idx = (b * BN_QK + col0) // scale_block_size
                # All 32 columns this thread handles within a block share
                # the same scale when scale_block_size >= 32.
                comptime if Self.config.scale_block_size > 0:
                    var scale_idx = (
                        b * BN_QK + col0
                    ) // Self.config.scale_block_size
                    var scale_byte = scale_smem_stage[
                        row * Self.config.scales_per_token + scale_idx
                    ]
                    var scale_bf16 = e8m0_to_bf16_broadcast(scale_byte)
                    p0a = hmul2_bf16x8_by_scalar[Self.q_type](p0a, scale_bf16)
                    p0b = hmul2_bf16x8_by_scalar[Self.q_type](p0b, scale_bf16)
                    p1a = hmul2_bf16x8_by_scalar[Self.q_type](p1a, scale_bf16)
                    p1b = hmul2_bf16x8_by_scalar[Self.q_type](p1b, scale_bf16)

                p0a_all.raw_store(b * 4, p0a)
                p0b_all.raw_store(b * 4, p0b)
                p1a_all.raw_store(b * 4, p1a)
                p1b_all.raw_store(b * 4, p1b)

            # Single barrier. All 128 threads finish ALL reads before ANY writes
            named_barrier[Int32(WARPGROUP_SIZE)](3)

            # Second: Store all BF16 data from registers
            comptime for b in range(NumBlocks):
                var dst_block = dst + b * BlockElems

                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    phys_bf16_0a,
                    p0a_all.raw_load[width=4](b * 4),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    phys_bf16_0b,
                    p0b_all.raw_load[width=4](b * 4),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    phys_bf16_1a,
                    p1a_all.raw_load[width=4](b * 4),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.q_type](
                    dst_block,
                    phys_bf16_1b,
                    p1b_all.raw_load[width=4](b * 4),
                )

            fence_async_view_proxy()
            kv_cvt_prod.commit_all()
            kv_load_cons_cvt.release_all()
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
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
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

            kv_cons.wait[qk_stage=0]()
            var k_slot_index = kv_cons.stage_index[qk_stage=0]()

            Self.UMMAQKTSS.mma[stage_idx=0](
                a=q_descriptor,
                b=k_descriptor + k_slot_index * UInt32(stage_stride_in_bytes),
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
