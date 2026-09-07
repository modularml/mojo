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

"""Native FP8 MLA decode kernel for SM100 (B200): Layout G fold path.

qkv=fp8 / BM=32 / MMA_M=32 / 1x4 datapath
specialisations (softmax, correction, output store, TMA/MMA descriptor
sizing) can evolve without disturbing the Layout E kernel.

Activated by the dispatcher constructing
`MLA_SM100_Decode_Config(decode_layout_g=True)`. Key shapes:
  BM=32, BN_QK=64, BK_QKT=64, BK_PV=64, BN_PV=256, MMA_M=32,
  num_kv_stages=5, cta_group=1.

SMEM layout (BN_QK=64, 5 stages):
  Q FP8:      32 x 576 x 1     = 18432  B   (SWIZZLE_64B)
  KV stages:  5 x 64 x 576 x 1 = 184320 B   (SWIZZLE_64B)
  P stages:   5 x 32 x 64  x 1 = 10240  B   (SWIZZLE_64B)
  max/li:     128 x 4 x 3      = 1536   B
  barriers:   (6N+11) fixed + output barriers
"""

from std.collections import OptionalReg
from std.math import ceildiv, exp2, log2, recip
from std.math.constants import log2e
from std.sys import size_of
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.primitives.warp import _vote_nvidia_helper
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory, fence_async_view_proxy
from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout.tma_async import (
    SharedMemBarrier,
)
from layout import (
    ComptimeInt,
    CoordLike,
    Layout,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
    KVTMATile,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    SharedMemPointer,
    MBarType,
    sub_ftz,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    ORaggedTMATile,
    rows_owned,
    store_row_coords,
    MLA_Decode_Pack,
    OffsetPosition,
    KVPipelineGeneric,
    DecodeSM100MiscMBars,
    DecodeSProducerN,
    DecodeSConsumerN,
    DecodePProducerN,
    DecodePConsumerN,
    DecodeCConsumer,
    DecodeCProducer,
    DecodeOConsumer,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeOutConsumer,
    DecodeKVProducer,
    DecodeKVConsumer,
    DecodeSM100QKTSS_FP8,
    DecodeSM100PVSS_FP8,
    write_fp8_row_to_smem_chunked,
    write_bf16x2_row_to_smem_chunked,
)


# Native FP8 MLA decode kernel — Layout G (BM=32, MMA_M=32, 5 stages).
# All of Q, K, V, P are FP8 in SMEM.
struct MLA_SM100_Decode_QKV_FP8_Layout_G[
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
    # Layout G handles BOTH fold (fold_q==True, q_len_fold > 1) and non-fold
    # (fold_q==False, q_len_fold==1, num_heads <= 32) cases. The kernel body
    # has full `comptime if Self.fold_q ... else ...` branching for both.
    # The dispatcher is responsible for picking a (fold_q, q_len_fold) pair
    # that satisfies `num_heads * q_len_fold <= BM_G(32)`.
    fold_q: Bool = False,
    # Number of q_tokens folded into the BM=32 M tile under fold_q==True.
    q_len_fold: Int = 1,
](TrivialRegisterPassable):
    """Implements the native FP8 MLA decode kernel for SM100 (B200) using the Layout G fold path.

    Uses a BM=32, MMA_M=32, 1x4 datapath with five KV pipeline stages where all
    of Q, K, V, and P reside in shared memory as FP8. Handles both the
    query-fold (`fold_q==True`, `q_len_fold > 1`) and non-fold (`fold_q==False`,
    `q_len_fold==1`, `num_heads <= 32`) cases, with the dispatcher selecting a
    `(fold_q, q_len_fold)` pair satisfying `num_heads * q_len_fold <= 32`.

    Parameters:
        q_type: Element dtype of the query operand; also selects the MMA
            accumulator type via `get_accum_type`.
        KVLUTType: `MHAOperand` describing the KV cache lookup table; its
            `dtype` is `float8_e4m3fn` for this FP8 kernel.
        output_type: Element dtype of the output tensor; must be `bfloat16`.
        SplitAccumType: `OptionalPointer` type for the split-K LSE
            accumulation buffer written under `decoding_warp_split_k`.
        MaskType: `MHAMask` variant selecting the attention mask; supports
            `NullMask`, `CausalMask`, and `SlidingWindowCausalMask`.
        config: `MLA_SM100_Decode_Config` holding tile sizes, pipeline
            stage counts, head counts, and TMEM/SMEM layout.
        ValidLengthType: `OptionalPointer` type for the per-sequence valid
            length buffer consumed under ragged batching.
        Engine: Engine policy of the `scalar_args` tile operand.
        _is_cache_length_accurate: Whether the supplied cache length is
            exact rather than `cache_length + seq_len` (defaults to `False`).
        ragged: Whether the batch contains variable-length sequences,
            enabling early exit for blocks past the sequence length
            (defaults to `False`).
        fold_q: Whether to fold multiple query tokens into the BM=32 M
            tile (defaults to `False`).
        q_len_fold: Number of query tokens packed into the BM=32 M tile
            when `fold_q` is `True`; must satisfy
            `num_q_heads * q_len_fold <= 32` (defaults to 1).
    """

    comptime kv_type = Self.KVLUTType.dtype  # float8_e4m3fn
    comptime fp8_type = DType.float8_e4m3fn
    comptime AccumType = get_accum_type[Self.q_type]()
    # Dimension aliases:
    #   BM      Q tile rows / output rows
    #   BN_QK   N of QK MMA = KV cache tile length
    #   BK_QKT  SMEM staging block size (NumQKBlocks = 576 / BK_QKT)
    #   BK_PV   K of PV MMA
    #   BN_PV   N of PV MMA per invocation
    comptime BM = Self.config.BM
    comptime BN_QK = Self.config.BN_QK
    comptime BK_QKT = 64  # FP8 SWIZZLE_64B group width.
    comptime BK_PV = Self.config.BK_PV
    comptime BN_PV = Self.config.MMA_PV_N
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.BK_QKT  # 9
    comptime NumVOBlocks = Self.config.padded_depth // Self.BK_PV  # 8
    # Q/P block size: BM rows x BK_QKT cols. Used for Q-region sizing and
    # P-stage sizing (P is BM x BK_PV; today BK_QKT == BK_PV).
    comptime BlockElems = Self.BM * Self.BK_QKT
    # KV block size is BN_QK-rows x BK_QKT-cols (independent of BM). Layout E
    # has BM == BN_QK so BlockElems doubles as KVBlockElems there; Layout G
    # has BM=32 < BN_QK so we need the separate constant.
    comptime KVBlockElems = Self.BN_QK * Self.BK_QKT
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    comptime bf16_bytes_per_element = size_of[Self.q_type]()
    # KV stage element count must match what `DecodeKVConsumer` expects
    # (kv_stage_elems = BN_QK * q_depth); using BlockElems (= BM*BK_QKT)
    # under-sizes by BN_QK/BM at Layout G.
    comptime KVStageElems = Self.NumQKBlocks * Self.KVBlockElems
    comptime PStageElems = Self.BM * Self.BK_PV
    comptime output_tile_width = (Self.BK_PV // 2) * (
        4 // size_of[Self.output_type]()
    )

    # QK MMA: FP8 KIND_F8F6F4. M=32 / N=64 / K=576 (transpose_b=True).
    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    # PV MMA: FP8 KIND_F8F6F4. M=32 / N=256 / K=64 (transpose_b=False).
    # P-operand swizzle must match `write_fp8_row_to_smem_chunked`: BK_PV=64
    # uses SWIZZLE_64B, BK_PV=128 uses SWIZZLE_128B. Mismatch corrupts P.
    comptime _ummapv_p_swizzle = (
        TensorMapSwizzle.SWIZZLE_128B if Self.BK_PV
        == 128 else TensorMapSwizzle.SWIZZLE_64B
    )
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
        p_swizzle=Self._ummapv_p_swizzle,
    ]

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

    # KV / S / P pipeline depth (Layout G pins this to 5).
    comptime num_stages = Self.config.num_kv_stages

    # Number of leading k-tiles to skip under SlidingWindowCausalMask.
    # All warpgroups must call this with the same `offset_position` so
    # producer/consumer iteration counts agree.
    #   global_lo = max(cache_len + 1 - W, 0)
    #   local_lo  = max(global_lo - kv_start_row, 0)
    #   tile_skip = local_lo // BN_QK
    @staticmethod
    @always_inline
    def sliding_window_tile_skip(
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ) -> Int:
        comptime _W: Int = Self.MaskType.sliding_window_size()
        var global_lo = max(offset_position.cache_len() + 1 - _W, 0)
        var local_lo = max(global_lo - offset_position.kv_start_row, 0)
        return local_lo // Self.BN_QK

    # Layout G softmax (warps 0-3): 4-way SMEM reduction at MMA_M=32.
    #
    # Under the 1x4 tcgen05 datapath every warp sees all 32 M-rows; warps
    # differ in the column slice they own. Per-lane mapping:
    #   warp_in_wg   = lane_id >> 5                 # col quadrant
    #   lane_in_warp = lane_id & 0x1F               # row index
    #   col0         = warp_in_wg * (BN_QK >> 2)
    #   half_load    = BN_QK >> 2                   # fp32 elems / lane
    # Each lane owns one row x (BN_QK/4) cols register-resident; per-row
    # max/sum reduces in registers, then cross-warp via SMEM exchange.
    @staticmethod
    @always_inline
    def Softmax_Layout_G[
        num_sp_stages: Int,
    ](
        tmem_addr: UInt32,
        s_bars: DecodeSM100MiscMBars[
            num_stages=num_sp_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        p_bars: DecodeSM100MiscMBars[
            num_stages=num_sp_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        p_smem_ptr: SharedMemPointer[Scalar[Self.fp8_type]],
        max_smem: SharedMemPointer[
            Scalar[Self.AccumType]
        ],  # double-buffered (2 × 128 fp32)
        li_smem: SharedMemPointer[
            Scalar[Self.AccumType]
        ],  # single buffer (128 fp32)
        out_smem: SharedMemPointer[Scalar[Self.output_type]],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        out_pipeline: OutPipeline[
            num_out_stages=DecodeOutProducer[
                Self.output_type, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
        scale: Float32,
        mask: Self.MaskType,
        prompt_idx: UInt32,
        lse_accum_split_ptr: Self.SplitAccumType,
        batch_size: Int,
    ):
        comptime MaskName: String = Self.MaskType.name()
        comptime MaskTypeName: String = Self.MaskType.get_type_name()
        comptime assert Self.AccumType.is_floating_point()
        comptime assert (
            Self.BM == 32
        ), "Softmax_Layout_G requires BM=32 (1×4 datapath)."
        comptime assert (
            Self.BN_QK == 64
        ), "Softmax_Layout_G requires BN_QK == 64 (per-warp col quadrant)."

        comptime NoMask: Bool = (MaskName == "NullMask")
        comptime CausalMask: Bool = (MaskName == "CausalMask")
        comptime SlidingWindowMask: Bool = (
            MaskTypeName == "SlidingWindowCausalMask"
        )
        comptime _sliding_window_size: Int = Self.MaskType.sliding_window_size()

        # S TMEM base / stride (matches mma()).
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        # SMEM exchange buffers: 128 fp32 per buffer (one per WG thread).
        comptime smem_1d_layout = tt_row_major[WARPGROUP_SIZE]()
        var li_Smem_Tensor = TileTensor[
            Self.AccumType,
            type_of(smem_1d_layout),
            MutAnyOrigin,
            address_space=.SHARED,
        ](li_smem, smem_1d_layout)

        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        # Split-K loop bounds vs total keys for masking.
        var num_keys = offset_position.num_keys
        var num_keys_this_split = offset_position.num_keys_this_split
        var kv_start_row = offset_position.kv_start_row
        var cache_start_pos: UInt32 = 0
        var cache_len: Int = offset_position.cache_len()
        var start_pos: UInt32 = offset_position.start_pos(cache_start_pos)

        var s_cons = DecodeSConsumerN[num_sp_stages](s_bars.consumer())
        var p_prod = DecodePProducerN[num_sp_stages](p_bars.producer())
        var c_prod = DecodeCProducer(c_bars.producer())
        var warp_idx = warp_id[broadcast=True]()
        var lane_id = thread_idx.x
        var warp_in_wg: Int = Int(lane_id) >> 5  # 0..3 col-quadrant
        var lane_in_warp: Int = Int(lane_id) & 0x1F  # 0..31 row
        var row: Int = lane_in_warp  # all warps share row identity
        comptime quarter_load: Int = Self.BN_QK >> 2
        var col0: Int = warp_in_wg * quarter_load
        # All 4 warps issue tcgen05_ld at the SAME TMEM offset — per-warp
        # slice is implicit in the subpartition view. col0 is the global-key
        # column index used by apply_mask and the P SMEM writeback only.

        var q_head_idx: UInt32 = UInt32(block_idx.x) * UInt32(Self.BM) + UInt32(
            row
        )
        # apply_mask derives the causal limit via row // num_q_heads under fold_q.
        var score_row: UInt32
        comptime if Self.fold_q:
            score_row = UInt32(row)
        else:
            score_row = UInt32(block_idx.y)

        var mi: Scalar[Self.AccumType] = min_or_neg_inf[Self.AccumType]()
        var li: Scalar[Self.AccumType] = 0.0
        comptime log2e_f32 = Scalar[Self.AccumType](log2e)
        # Per-lane S width = BN_QK/4 (Layout E uses BN_QK/2).
        comptime half_load = quarter_load
        var scale_log2e = scale.cast[Self.AccumType]()

        var tiles_done: Int = 0
        var num_k_tiles = ceildiv(num_keys_this_split, Self.BN_QK)
        # Sliding-window leading-tile skip — must match the producer skip
        # exactly so barrier counts agree.
        comptime if SlidingWindowMask:
            var _W_sw: Int = _sliding_window_size
            var _global_lo_sw = max(cache_len + 1 - _W_sw, 0)
            var _local_lo_sw = max(_global_lo_sw - kv_start_row, 0)
            var _tile_skip_sw = _local_lo_sw // Self.BN_QK
            tiles_done = _tile_skip_sw
            if _tile_skip_sw >= num_k_tiles:
                num_k_tiles = _tile_skip_sw  # loop guard false
        var first_processed_tile_sw: Int = tiles_done

        while tiles_done < num_k_tiles:
            var slot_idx: UInt32 = s_cons.wait()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            tcgen05_fence_after()

            # Load S TMEM -> registers (16 fp32/lane, lane_id == row).
            var s_row = tt_stack_allocation[
                dtype=Self.AccumType, address_space=.LOCAL
            ](row_major[half_load]())
            var s_row_val = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=quarter_load,  # BN_QK/4 (16 at BN_QK=64, 32 at BN_QK=128)
                dtype=Self.AccumType,
                pack=False,
            ](s_tmem_slot)

            comptime for _i in range(type_of(s_row_val).length):
                s_row.raw_store(_i, s_row_val[_i])
            tcgen05_load_wait()

            s_cons.release()

            # Scale-fold + mask. apply_mask sees a half_load-wide tile
            # (BN_QK/4 instead of Layout E's BN_QK/2).
            var s_row_val_vectorized = s_row.vectorize[2]()
            comptime vs_count = ceildiv(half_load, 2)
            comptime for _vi in range(vs_count):
                s_row_val_vectorized[_vi] = (
                    s_row_val_vectorized[_vi] * scale_log2e
                )

            comptime _fold_q_num_heads: Int = (
                Self.config.num_q_heads if Self.fold_q else 0
            )
            comptime _causal_for_apply: Bool = CausalMask or SlidingWindowMask
            var current_max: Scalar[Self.AccumType]
            comptime if NoMask or CausalMask or SlidingWindowMask:
                current_max = Self.Common_MLA_Op.apply_mask[
                    half_load,
                    NonCausalMask=False,
                    CausalMask=_causal_for_apply,
                    fold_q_num_heads=_fold_q_num_heads,
                    SlidingWindowSize=_sliding_window_size,
                ](
                    tiles_done,
                    col0,
                    num_keys,
                    s_row,
                    mask,
                    prompt_idx,
                    q_head_idx,
                    score_row,
                    cache_len,
                    start_pos,
                    cache_start_pos,
                    kv_start_row,
                )
            else:
                current_max = Self.Common_MLA_Op.apply_mask[
                    half_load, NonCausalMask=True, CausalMask=False
                ](
                    tiles_done,
                    col0,
                    num_keys,
                    s_row,
                    mask,
                    prompt_idx,
                    q_head_idx,
                    score_row,
                    cache_len,
                    start_pos,
                    cache_start_pos,
                    kv_start_row,
                )
            current_max *= log2e_f32

            # Per-row max: register-only reduce + 4-way SMEM consolidation.
            # Double-buffered to avoid the W-W race between N+1's write and
            # iteration N's read.
            comptime rescale_threshold: Float32 = (
                Self.config.skip_correction_threshold
            )
            var buf_offset = (tiles_done & 1) * WARPGROUP_SIZE
            var max_buf = TileTensor[
                Self.AccumType,
                type_of(smem_1d_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](max_smem + buf_offset, smem_1d_layout)
            max_buf[lane_id] = current_max
            named_barrier[Int32(WARPGROUP_SIZE)](2)
            # Each lane reads the 4 col-quadrant partials for its row;
            # every warp ends up with the full BN_QK-col row max.
            var w0 = max_buf[lane_in_warp + 0][0]
            var w1 = max_buf[lane_in_warp + 32][0]
            var w2 = max_buf[lane_in_warp + 64][0]
            var w3 = max_buf[lane_in_warp + 96][0]
            current_max = max(max(w0, w1), max(w2, w3))
            var new_max: Scalar[Self.AccumType] = max(mi, current_max)
            var diff = sub_ftz(rebind[Float32](mi), rebind[Float32](new_max))
            var scale_for_old_max: Scalar[Self.AccumType]
            # Per-lane predicate, not a warp vote: under `fold_q`, `row =
            # lane_in_warp` packs `q_len_fold` distinct (never-committed
            # draft included) query tokens into one warp -- ALL of Layout
            # G's 32 rows share the same single warp (BM=32), so an OR
            # here leaks EVERY sibling token's rescale trajectory into
            # every other one's.
            if diff < rescale_threshold:
                scale_for_old_max = rebind[Scalar[Self.AccumType]](exp2(diff))
            else:
                scale_for_old_max = 1.0
                new_max = mi

            # exp2(s - new_max) + per-thread sum over the col-quadrant.
            var float2_register = s_row.vectorize[2]()
            var float2_current_sum: SIMD[Self.AccumType, 2] = 0.0
            comptime for i in range(0, half_load // 2):
                var element = float2_register[i]
                float2_register[i] = exp2(element.fma(log2e_f32, -new_max))
                float2_current_sum += float2_register[i]

            # Correction-scale write to TMEM (skip on the first processed
            # tile — no prior O accumulator to correct).
            if tiles_done > first_processed_tile_sw:
                c_prod.acquire()
                var _scale_tuple = Array[Scalar[Self.AccumType], 1](
                    fill=scale_for_old_max
                )
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](corr_scale_tmem, _scale_tuple)
                tcgen05_store_wait()
                c_prod.commit()

            # Wait for MMA to release P SMEM, then write the FP8 P-row
            # into the warp's col-quadrant slot.
            p_prod.acquire()
            var p_stage = p_prod.stage_index()
            var p_smem_stage = p_smem_ptr + p_stage * UInt32(Self.PStageElems)
            # P swizzle MUST match `Self._ummapv_p_swizzle` (the PV MMA P
            # descriptor) — mismatch corrupts the P read.
            comptime _p_swizzle = (
                TensorMapSwizzle.SWIZZLE_128B if Self.BK_PV
                == 128 else TensorMapSwizzle.SWIZZLE_64B
            )
            write_fp8_row_to_smem_chunked[
                half_load,
                out_dtype=Self.fp8_type,
                in_dtype=Self.AccumType,
                config=Self.config,
                row_size=Self.BK_PV,
                swizzle_kind=_p_swizzle,
            ](p_smem_stage, s_row, col0, row)

            fence_async_view_proxy()
            p_prod.commit()
            mi = new_max

            # Per-row sum: each warp keeps its quadrant-local li across the
            # kv-tile loop and we consolidate once after the loop. The shared
            # `scale_for_old_max` (already consolidated via `mi`) plus
            # distributivity makes per-warp partial accumulation exact.
            var local_sum: Scalar[Self.AccumType] = (
                float2_current_sum[0] + float2_current_sum[1]
            )
            li = li.fma(scale_for_old_max, local_sum)
            tiles_done += 1

        # Post-loop 4-way SMEM consolidation: sum the 4 quadrant-local
        # `li` partials into the full-row `li` consumed by the LSE write
        # and the output-store recip(li).
        li_Smem_Tensor[lane_id] = li
        named_barrier[Int32(WARPGROUP_SIZE)](2)
        var li_q0 = li_Smem_Tensor[lane_in_warp + 0][0]
        var li_q1 = li_Smem_Tensor[lane_in_warp + 32][0]
        var li_q2 = li_Smem_Tensor[lane_in_warp + 64][0]
        var li_q3 = li_Smem_Tensor[lane_in_warp + 96][0]
        li = li_q0 + li_q1 + li_q2 + li_q3

        # Split-K LSE write — all 4 warps hold identical (mi, li); pick
        # `warp_in_wg == 0` as the single writer.
        comptime if Self.config.decoding_warp_split_k:
            comptime if Self.fold_q:
                var q_local = row // Self.config.num_q_heads
                var head_local = row % Self.config.num_q_heads
                if warp_in_wg == 0 and row < (
                    Self.q_len_fold * Self.config.num_q_heads
                ):
                    var partial_lse: Scalar[Self.AccumType]
                    if q_local >= offset_position.seq_len:
                        partial_lse = min_or_neg_inf[Self.AccumType]()
                    else:
                        partial_lse = (
                            log2(max(li, Scalar[Self.AccumType](0))) + mi
                        )
                    var stride_batch = (
                        offset_position.max_seq_len * Self.config.num_q_heads
                    )
                    var stride_split = batch_size * stride_batch
                    var stride_seq = Self.config.num_q_heads
                    var lse_offset = (
                        offset_position.split_idx * stride_split
                        + offset_position.batch_idx * stride_batch
                        + q_local * stride_seq
                        + head_local
                    )
                    var lse_ptr = rebind[
                        UnsafePointer[
                            Scalar[Self.AccumType], origin=MutAnyOrigin
                        ]
                    ](lse_accum_split_ptr.value())
                    lse_ptr[lse_offset] = partial_lse
            else:
                var head_idx = block_idx.x * Self.BM + row
                if warp_in_wg == 0 and head_idx < Self.config.num_q_heads:
                    var partial_lse = (
                        log2(max(li, Scalar[Self.AccumType](0))) + mi
                    )
                    var seq_idx = block_idx.y
                    var stride_batch = (
                        offset_position.max_seq_len * Self.config.num_q_heads
                    )
                    var stride_split = batch_size * stride_batch
                    var stride_seq = Self.config.num_q_heads
                    var lse_offset = (
                        offset_position.split_idx * stride_split
                        + offset_position.batch_idx * stride_batch
                        + seq_idx * stride_seq
                        + head_idx
                    )
                    var lse_ptr = rebind[
                        UnsafePointer[
                            Scalar[Self.AccumType], origin=MutAnyOrigin
                        ]
                    ](lse_accum_split_ptr.value())
                    lse_ptr[lse_offset] = partial_lse

        # Output store epilogue (TMEM -> registers -> SMEM).
        #
        # Each warp `w` owns a 32 x BK_PV stripe per MMA PV round; SMEM
        # staging maps stripe w -> stage (w >> 1), half_idx (w & 1). All
        # 4 warps acquire/commit every slot to keep the 128-thread
        # producer barrier balanced; only the owning warp writes data.
        comptime assert (
            Self.AccumType == .float32
        ), "accumulator type should be float32"
        comptime assert (
            Self.output_type == .bfloat16
        ), "output type should be bfloat16"

        comptime DecodeOutProducerType_LG = DecodeOutProducer[
            Self.output_type, Self.config
        ]
        comptime blocks_per_stage_LG = DecodeOutProducerType_LG.blocks_per_stage
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)

        # Per-warp tile = 32 rows x BK_PV cols. The 1x4 datapath places
        # each warp's stripe in its own quadrant; tcgen05_ld returns the
        # owned slice directly.
        comptime per_warp_elems_LG: Int = Self.BN_PV >> 2
        comptime chunk_size_LG: Int = 16

        var out_prod = DecodeOutProducer[Self.output_type, Self.config](
            out_pipeline, out_smem
        )

        # Output scale = recip(li) with empty-split guard. Layout G is
        # always routed under split-K, so the no-split attn_sink branch
        # is unreachable here.
        var o_scale_li: Scalar[Self.AccumType]
        o_scale_li = recip(li) if li > 0 else 0

        # Per-warp ownership: 4 warps split the 4 BK_PV-stripes of one
        # round across 2 stages, half_idx 0/1 within each stage.
        var warp_in_wg_lg: UInt32 = (UInt32(thread_idx.x) >> 5) & UInt32(3)
        var owned_slot_LG: UInt32 = warp_in_wg_lg >> 1
        var owned_half_LG: UInt32 = warp_in_wg_lg & UInt32(1)

        comptime num_mma_pv_rounds_LG = (Self.config.depth // Self.BN_PV)
        comptime iters_per_mma_round_LG = 4 // blocks_per_stage_LG

        comptime for mma_round_lg in range(num_mma_pv_rounds_LG):
            corr_done_bars.mbar_base[mma_round_lg].wait(0)
            tcgen05_fence_after()

            # PV MMA writes round 1 at TMEM offset BN_PV/2 (= 128); reading
            # at mma_round * BN_PV would land in the S TMEM region.
            var o_tmem_base_LG: UInt32 = o_tmem + UInt32(mma_round_lg) * UInt32(
                Self.BN_PV // 2
            )
            var o_row_subtile_LG = tt_stack_allocation[
                dtype=Self.AccumType, address_space=.LOCAL
            ](row_major[per_warp_elems_LG]())
            var _o_ld_LG = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=per_warp_elems_LG,
                dtype=Self.AccumType,
                pack=False,
            ](o_tmem_base_LG)
            comptime for _i in range(per_warp_elems_LG):
                o_row_subtile_LG.raw_store(_i, _o_ld_LG[_i])
            tcgen05_load_wait()

            # All 4 warps acquire/commit every slot for the 128-thread
            # barrier; only the 2 owning warps write data.
            comptime for slot_LG in range(iters_per_mma_round_LG):
                out_prod.acquire()
                if owned_slot_LG == UInt32(slot_LG):
                    var stage_ptr_LG = out_prod.stage_base_ptr(
                        Int(owned_half_LG)
                    )
                    write_bf16x2_row_to_smem_chunked[
                        per_warp_elems_LG,
                        out_dtype=Self.output_type,
                        in_dtype=Self.AccumType,
                        config=Self.config,
                        chunk_size=chunk_size_LG,
                        scale_needed=True,
                    ](
                        stage_ptr_LG,
                        o_row_subtile_LG,
                        0,
                        Int(lane_in_warp),
                        o_scale_li,
                    )
                out_prod.commit_step()

    # Output store TMA path (warp 11): SMEM -> HBM.
    # SMEM staging maps one BK_PV stripe per half_idx with column layout
    #   col = mma_round * BN_PV + slot * (BN_PV // 2) + half_idx * BK_PV
    # Layout G's per-warp stripes are contiguous (vs. interleaved in E).
    @staticmethod
    @always_inline
    def Output_Store_Layout_G(
        out_pipeline: OutPipeline[
            num_out_stages=DecodeOutProducer[
                Self.output_type, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        out_smem: SharedMemPointer[Scalar[Self.output_type]],
        # TMA-tile BK is anchored to BN_PV/4 (the per-warp stripe width),
        # not BN_QK — Layout-G-128 has BN_QK=128 but still writes 64-col stripes.
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
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
        comptime assert (
            Self.BM == 32
        ), "Output_Store_Layout_G requires BM=32 (1×4 datapath)."

        comptime DecodeOutConsumerType_LG = DecodeOutConsumer[
            Self.output_type, Self.config
        ]
        comptime blocks_per_stage_LG = DecodeOutConsumerType_LG.blocks_per_stage
        comptime num_out_stages_LG = DecodeOutConsumerType_LG.num_out_stages
        comptime num_mma_pv_LG = (Self.config.padded_depth // Self.BN_PV)
        comptime num_out_stages_per_mma_LG = (
            num_out_stages_LG // num_mma_pv_LG
        )

        var out_cons = DecodeOutConsumer[Self.output_type, Self.config](
            out_pipeline, out_smem
        )
        var elect_mask = elect()
        var is_leader: Bool = elect_mask != 0
        var row: Int = offset_position.out_row_offset
        var rows_to_store = rows_owned[Self.config](Int(block_idx.x))

        # col = n * BN_PV + m * (BN_PV // 2) + k * BK_PV
        comptime slot_col_stride_LG: Int = Self.BN_PV // 2
        comptime half_col_stride_LG: Int = Self.BN_PV >> 2

        comptime for n_LG in range(0, num_mma_pv_LG):
            comptime for m_LG in range(0, num_out_stages_per_mma_LG):
                out_cons.wait()

                comptime for k_LG in range(0, blocks_per_stage_LG):
                    var stage_ptr_LG = out_cons.stage_base_ptr(k_LG)
                    var col_LG: Int = (
                        n_LG * Self.BN_PV
                        + m_LG * slot_col_stride_LG
                        + k_LG * half_col_stride_LG
                    )
                    comptime o_elements_LG = (
                        Self.config.out_rows * (Self.config.BN_PV // 4)
                    )
                    comptime o_tt_layout_LG = tt_row_major[o_elements_LG]()
                    comptime if Self.fold_q:
                        # Fold: BM=32 packs q_len_fold * num_q_heads rows;
                        # emit one TMA store per q_token.
                        comptime for q_local_LG in range(Self.q_len_fold):
                            var q_stage_ptr_LG = stage_ptr_LG + (
                                q_local_LG
                                * Self.config.num_q_heads
                                * (Self.config.BN_PV // 4)
                            )
                            var smem_tensor_LG = TileTensor[
                                Self.output_type,
                                type_of(o_tt_layout_LG),
                                MutAnyOrigin,
                                address_space=.SHARED,
                            ](q_stage_ptr_LG, o_tt_layout_LG)
                            if is_leader:
                                fence_async_view_proxy()
                                o_tma.async_store_3d(
                                    smem_tensor_LG,
                                    store_row_coords[Self.config.out_rows](
                                        col_LG,
                                        offset_position.out_row_offset_at(
                                            q_local_LG
                                        ),
                                        rows_to_store,
                                    ),
                                )
                    else:
                        var smem_tensor_LG = TileTensor[
                            Self.output_type,
                            type_of(o_tt_layout_LG),
                            MutAnyOrigin,
                            address_space=.SHARED,
                        ](stage_ptr_LG, o_tt_layout_LG)
                        if is_leader:
                            fence_async_view_proxy()
                            o_tma.async_store_3d(
                                smem_tensor_LG,
                                store_row_coords[Self.config.out_rows](
                                    col_LG, row, rows_to_store
                                ),
                            )
                out_cons.release(elect_mask)

        if is_leader:
            o_tma.commit_group()
        o_tma.wait_group[0]()

    # Main kernel — 3 warpgroups, 384 threads:
    #   warps  0-3  softmax
    #   warps  4-7  correction
    #   warp   8    TMA load
    #   warp   9    MMA QK
    #   warp  10    MMA PV
    #   warp  11    output store
    # The FP8 dequant scale is folded into the softmax scale:
    #   qk_scale = (1.0 / sqrt(d_qk)) * kv_scale.

    # Layout G correction warp (warps 4-7) — 1x4 datapath at M=32.
    # All 4 warps see all 32 rows; each owns one col-quadrant of BN_QK/4
    # cols and issues tcgen05_ld at the same TMEM offset (per-warp slice
    # is implicit in the subpartition view). The early-exit ballot agrees
    # across the WG because all 4 warps see the same scale_value after
    # the softmax 4-way SMEM exchange.
    @staticmethod
    @always_inline
    def Correction_Layout_G(
        tmem_addr: UInt32,
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
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
        comptime assert (
            Self.BM == 32
        ), "Correction_Layout_G requires BM=32 (1×4 datapath)."
        comptime assert Self.BN_PV == 256, (
            "Correction_Layout_G assumes BN_PV=256 per PV invocation (per-warp"
            " 64-col stripe, 2 PV rounds for head_dim=512)."
        )

        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        var o_cons = DecodeOConsumer(o_bars.consumer())
        var c_cons = DecodeCConsumer(c_bars.consumer())
        var tiles_done: Int = 1

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.BN_QK
        )

        # Sliding-window leading-tile skip; must align with the load /
        # softmax / MMA skips so producer/consumer iterations match.
        comptime _sliding_window_mask_corr: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask_corr:
            comptime _W_corr: Int = Self.MaskType.sliding_window_size()
            var _global_lo_corr = max(
                offset_position.cache_len() + 1 - _W_corr, 0
            )
            var _local_lo_corr = max(
                _global_lo_corr - offset_position.kv_start_row, 0
            )
            var _tile_skip_corr = _local_lo_corr // Self.BN_QK
            tiles_done = _tile_skip_corr + 1

        # Per-warp inner count: BN_PV/4 fp32 cols => BN_PV/8 float2 entries.
        comptime correction_inner_count: Int = Self.BN_PV >> 3

        while tiles_done < num_k_tiles:
            c_cons.wait()
            var scale_value_tuple = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=1,
                dtype=Self.AccumType,
                pack=False,
            ](corr_scale_tmem)
            tcgen05_load_wait()
            c_cons.release()
            var scale_value = scale_value_tuple[0]
            var change = _vote_nvidia_helper(scale_value < 1.0) != 0

            comptime o_range = Self.config.depth // Self.BN_PV
            comptime o_stride = Self.BN_PV // 2
            # All 4 warps issue tcgen05_ld at the same TMEM offset
            # (slot_idx * BN_PV/2 = 0 or 128). Each owns BN_PV/4 fp32 cols.
            comptime per_warp_corr_elems: Int = Self.BN_PV >> 2

            comptime for slot_idx in range(o_range):
                o_cons.wait()
                if change:
                    var o_tmem_subtile: UInt32 = o_tmem + UInt32(
                        slot_idx
                    ) * UInt32(o_stride)
                    var o_row_subtile = tt_stack_allocation[
                        dtype=Self.AccumType,
                        address_space=.LOCAL,
                    ](row_major[per_warp_corr_elems]())
                    var _o_ld_corr = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=per_warp_corr_elems,
                        dtype=Self.AccumType,
                        pack=False,
                    ](o_tmem_subtile)

                    comptime for _i in range(per_warp_corr_elems):
                        o_row_subtile.raw_store(_i, _o_ld_corr[_i])
                    tcgen05_load_wait()

                    var float2_register = o_row_subtile.vectorize[2]()

                    comptime for j in range(0, correction_inner_count):
                        var element = float2_register[j]
                        float2_register[j] = element * SIMD[Self.AccumType, 2](
                            scale_value
                        )
                    var _o_st_corr = Array[_, per_warp_corr_elems](
                        fill_with_unrolled=lambda [i: Int]() -> Scalar[
                            Self.AccumType
                        ]: o_row_subtile.raw_load(i)
                    )
                    tcgen05_st[
                        datapaths=32,
                        bits=32,
                        repeat=per_warp_corr_elems,
                        pack=False,
                    ](
                        o_tmem_subtile,
                        _o_st_corr,
                    )
                    tcgen05_store_wait()
                o_cons.release()
            tiles_done += 1

        # 2 corr_done slots match the 2 MMA PV rounds (depth/BN_PV = 2).
        o_cons.wait()
        _ = corr_done_bars.mbar_base[0].arrive()
        o_cons.release()
        o_cons.wait()
        _ = corr_done_bars.mbar_base[1].arrive()
        o_cons.release()

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__name(
        t"sm100_mla_decode_qkv_fp8_layout_g_{Self.q_type}_{Self.kv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        # Q/K/O TMA tile types must use the exact symbolic forms below —
        # Mojo type-checks comptime parameters by symbolic form, not value,
        # and the dispatcher constructs descriptors using these expressions.
        q_tma: QOTMATile[
            dtype=Self.kv_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,  # SWIZZLE_64B
        ],
        k_tma: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BN_QK,
            BK=Self.config.input_q_depth,
        ],
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            # Per-warp output stripe (BF16/SWIZZLE_128B clamps innermost to 64).
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
        comptime assert Self.config.decode_layout_g, (
            "MLA_SM100_Decode_QKV_FP8_Layout_G requires"
            " config.decode_layout_g==True (BM=32, MMA_M=32, num_kv_stages=5)."
            " Use MLA_SM100_Decode_QKV_FP8 for the Layout E (BM=64) path."
        )

        # Native FP8 backend supports NullMask, CausalMask, SlidingWindowCausalMask.
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask"
            or _mask_type_name == "CausalMask"
            or _mask_type_name == "SlidingWindowCausalMask"
        ), (
            "MLA_SM100_Decode_QKV_FP8_Layout_G supports NullMask, CausalMask,"
            " and SlidingWindowCausalMask only."
        )

        # Extract scalar launch args from the stable device buffer.
        var batch_size = Int(scalar_args.raw_load(0))
        var q_max_seq_len = Int(scalar_args.raw_load(1))
        var num_partitions = mla_decode_pack.num_partitions

        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 112
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

        # Early exit for split-K: CTAs with no work
        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                comptime if Self.fold_q:
                    # Fold owns all q_len_fold LSE slots; emit -inf per slot.
                    comptime for q_local in range(Self.q_len_fold):
                        Self.Common_MLA_Op.pdl_early_exit[fold_q=Self.fold_q](
                            offset_position.split_idx,
                            offset_position.batch_idx,
                            offset_position.max_seq_len,
                            batch_size,
                            lse_accum_split_ptr,
                            seq_idx_fold=UInt32(q_local),
                        )
                else:
                    Self.Common_MLA_Op.pdl_early_exit[fold_q=Self.fold_q](
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                    )
                return

        # Sliding-window split-K: a CTA whose entire split lies below
        # (causal_limit - W) has nothing to compute and must take the same
        # -inf-LSE early-exit path as num_keys_this_split == 0.
        comptime _sliding_window_mask: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask and Self.config.decoding_warp_split_k:
            var _num_k_tiles_total = ceildiv(
                offset_position.num_keys_this_split, Self.BN_QK
            )
            var _tile_skip = Self.sliding_window_tile_skip(offset_position)
            if _tile_skip >= _num_k_tiles_total:
                comptime if Self.fold_q:
                    comptime for q_local in range(Self.q_len_fold):
                        Self.Common_MLA_Op.pdl_early_exit[fold_q=Self.fold_q](
                            offset_position.split_idx,
                            offset_position.batch_idx,
                            offset_position.max_seq_len,
                            batch_size,
                            lse_accum_split_ptr,
                            seq_idx_fold=UInt32(q_local),
                        )
                else:
                    Self.Common_MLA_Op.pdl_early_exit[fold_q=Self.fold_q](
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                    )
                return

        # Early exit for ragged: skip blocks beyond actual sequence length
        comptime if Self.ragged:
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    comptime if Self.fold_q:
                        comptime for q_local in range(Self.q_len_fold):
                            Self.Common_MLA_Op.pdl_early_exit[
                                fold_q=Self.fold_q
                            ](
                                offset_position.split_idx,
                                offset_position.batch_idx,
                                offset_position.max_seq_len,
                                batch_size,
                                lse_accum_split_ptr,
                                seq_idx_fold=UInt32(q_local),
                            )
                    else:
                        Self.Common_MLA_Op.pdl_early_exit[fold_q=Self.fold_q](
                            offset_position.split_idx,
                            offset_position.batch_idx,
                            offset_position.max_seq_len,
                            batch_size,
                            lse_accum_split_ptr,
                        )

                return

        # SMEM allocation: Q FP8, KV stages, P stages, max/li, barriers.
        var q_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        var kv_smem = q_smem + Self.BlockElems * Self.NumQKBlocks
        comptime kv_total_stages = Self.config.num_kv_stages
        var p_smem = kv_smem + Self.KVStageElems * kv_total_stages
        comptime p_total_stages = Self.num_stages

        # Output SMEM aliases KV SMEM (same as Layout E).
        var out_smem_start = kv_smem
        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        var max_smem = (p_smem + Self.PStageElems * p_total_stages).bitcast[
            Scalar[Self.AccumType]
        ]()
        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        # Barrier layout (6N+11 fixed, N=num_kv_stages):
        # bar_q(1) + kv(2N) + s(2N) + p(2N) + o(4) + c(2) + corr_done(4)
        var mbar_base: MBarType = (
            (li_smem + WARPGROUP_SIZE)
            .bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        var mbar_kv_base: MBarType = mbar_base + 1

        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ](mbar_kv_base)

        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()
        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = s_bars.end()
        var p_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base = p_bars.end()
        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](mbar_base)
        mbar_base = o_bars.end()
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = c_bars.end()
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = corr_done_bars.end()
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base += out_pipeline.num_mbars()

        # TMEM column map (matches Layout E):
        #   TMEM_O      = 0     (PV accumulator)
        #   TMEM_S0..S5 = 256..447
        #   CORR_SCALE  = 448
        #   CORR_LI     = 449
        var warp_idx = UInt32(warp_id[broadcast=True]())
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        var is_leader = elect() != 0

        # Init: warp 8 inits barriers, warp 9 allocates TMEM.
        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                kv_pipeline.init()
                s_bars.init()
                p_bars.init()
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

        # Warpgroup dispatch (3-WG layout, same as Layout E):
        #   warps 0-3  softmax + per-warp output SMEM write
        #   warps 4-7  correction
        #   warp  8    TMA load
        #   warp  9    MMA QK
        #   warp 10    MMA PV
        #   warp 11    output TMA store
        if warp_idx < 4:
            warpgroup_reg_alloc[num_reg_softmax]()
            Self.Softmax_Layout_G[num_sp_stages=Self.num_stages,](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                p_smem.as_unsafe_any_origin(),
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
        elif warp_idx >= 4 and warp_idx < 8:
            warpgroup_reg_alloc[num_reg_correction]()
            Self.Correction_Layout_G(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_smem.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem.as_unsafe_any_origin(),
                    p_smem.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 11:
                Self.Output_Store_Layout_G(
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    offset_position,
                )
        barrier()

        # PDL: Signal that this CTA is done
        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    # Load: TMA Q (FP8) and TMA KV (FP8). Cloned from the Layout E sibling;
    # TMA descriptors are config-driven so the BM=32 shapes flow through.
    @staticmethod
    @always_inline
    def load(
        q_tma: QOTMATile[
            dtype=Self.kv_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,  # SWIZZLE_64B
        ],
        # `BN=Self.config.BN_QK` (not BK_PV) so this signature is type-
        # identical to `Common_MLA_Op.load_kv`. Mojo compares comptime
        # parameters by symbolic form, not value.
        k_tma: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BN_QK,
            BK=Self.config.input_q_depth,
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
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
        if offset_position.num_keys_this_split == 0:
            return

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.BN_QK
        )

        # Sliding-window early exit + leading-tile skip.
        comptime _sliding_window_mask: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        var _tile_skip: Int = 0
        comptime if _sliding_window_mask:
            _tile_skip = Self.sliding_window_tile_skip(offset_position)
            if _tile_skip >= num_k_tiles:
                return
            num_k_tiles -= _tile_skip

        var kv_prod = DecodeKVProducer[Self.kv_type, Self.config](
            kv_pipeline, kv_smem.bitcast[Scalar[Self.kv_type]]()
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset
        var kv_row: UInt32 = UInt32(offset_position.kv_start_row)
        comptime if _sliding_window_mask:
            kv_row += UInt32(_tile_skip * Self.BN_QK)
        var num_keys_u32 = UInt32(offset_position.num_keys)
        kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(
            UInt32(offset_position.batch_idx), kv_row
        )

        if is_leader:
            mbar_q[].expect_bytes(
                Int32(
                    Self.BM * Self.config.q_depth * Self.fp8_bytes_per_element
                )
            )
            comptime q_elems = type_of(q_tma).tile_shape[0] * type_of(
                q_tma
            ).tile_shape[1]
            comptime q_tt_layout = tt_row_major[q_elems]()
            var q_smem_tensor = TileTensor[
                Self.kv_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](q_smem.bitcast[Scalar[Self.kv_type]](), q_tt_layout)
            q_tma.async_copy(q_smem_tensor, mbar_q[], (0, row))

        # First KV tile.
        var k0_bar: MBarType = kv_prod.producer_mbar[qk_stage=0]()
        if is_leader:
            k0_bar[].expect_bytes(
                Int32(
                    Self.BN_QK
                    * Self.config.q_depth
                    * Self.fp8_bytes_per_element
                )
            )
            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            Self.Common_MLA_Op.load_kv(
                k_tma, stage_ptr, k0_bar, 0, Int(kv_gmem_row)
            )
        kv_prod.commit_step()
        kv_row += UInt32(Self.BN_QK)

        # Wait for Q TMA. Q arrives as FP8, ready for MMA.
        mbar_q[].wait(0)

        var tile_idx: Int = 1
        while tile_idx < num_k_tiles:
            kv_prod.acquire[qk_stage=0]()
            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()
            kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
            var kv_gmem_row: UInt32 = kv_lut.row_idx(
                UInt32(offset_position.batch_idx), kv_row
            )

            if is_leader:
                k_mbar[].expect_bytes(
                    Int32(
                        Self.BN_QK
                        * Self.config.q_depth
                        * Self.fp8_bytes_per_element
                    )
                )
                Self.Common_MLA_Op.load_kv(
                    k_tma, stage_ptr, k_mbar, 0, Int(kv_gmem_row)
                )

            kv_row += UInt32(Self.BN_QK)
            kv_prod.commit_step()
            tile_idx += 1

    # MMA QK: Q(FP8) x K(FP8) -> S(TMEM). M=32, N=64, K=576 (18 K-mmas).
    @staticmethod
    @always_inline
    def mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
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

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.BN_QK
        )

        if num_k_tiles == 0:
            return

        # Sliding-window leading-tile skip — must match `load` exactly.
        comptime _sliding_window_mask: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask:
            var _tile_skip = Self.sliding_window_tile_skip(offset_position)
            if _tile_skip >= num_k_tiles:
                return
            num_k_tiles -= _tile_skip

        var kv_cons = DecodeKVConsumer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.fp8_bytes_per_element

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
            tile_idx += 1

    # MMA PV: P(FP8) x V(FP8) -> O(TMEM). M=32, N=256, K=64 (2 K-mmas).
    @staticmethod
    @always_inline
    def mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
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
            offset_position.num_keys_this_split, Self.BN_QK
        )

        if num_k_tiles == 0:
            return

        # Sliding-window leading-tile skip — must match `load` exactly.
        comptime _sliding_window_mask: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask:
            var _tile_skip = Self.sliding_window_tile_skip(offset_position)
            if _tile_skip >= num_k_tiles:
                return
            num_k_tiles -= _tile_skip

        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        # V data lives in the first 512 cols of KV SMEM.
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime kv_stage_stride_in_bytes = Self.KVStageElems * Self.fp8_bytes_per_element
        comptime p_stage_stride_in_bytes = Self.PStageElems * Self.fp8_bytes_per_element
        # V advance per PV invocation = BN_PV * BK_PV bytes
        # (= (BN_PV/BK_PV) * KVBlockElems). Using BlockElems (BM*BK_QKT)
        # under-strides V by BN_QK/BM at Layout G.
        comptime pv_v_stride_in_bytes = (
            Self.BN_PV * Self.BK_PV * Self.fp8_bytes_per_element
        )
        # TMEM column packing: 2 fp32 accum cols per physical TMEM col.
        comptime pv_o_tmem_col_stride: UInt32 = UInt32(Self.BN_PV // 2)

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            # Walk [0, padded_depth) in BN_PV chunks (= 2 PV invocations).
            comptime for n_chunk_start in range(
                0, Self.config.padded_depth, Self.BN_PV
            ):
                comptime pv_invocation_idx = n_chunk_start // Self.BN_PV
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(p_stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(kv_stage_stride_in_bytes)
                    + UInt32(pv_invocation_idx * pv_v_stride_in_bytes),
                    c=o_tmem + UInt32(pv_invocation_idx) * pv_o_tmem_col_stride,
                    c_scale=c_scale,
                    elect=elect_mask,
                )
                o_prod.commit_mma(elect_mask)
            p_cons.release_mma(elect_mask)

            kv_cons.release[qk_stage=0](elect_mask)
            tcgen05_fence_before()

            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
