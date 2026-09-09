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

"""Native FP8 MLA decode kernel for SM100 (B200).

All-FP8 kernel: Q, K, V, and P are all FP8 e4m3 in SMEM.
Uses native FP8 WGMMA (tcgen05.mma.kind::f8f6f4) for both QK and PV.
Same 3-WG structure as the BF16 kernel (softmax WG, correction WG, MMA+Load+Store WG).

Q arrives as FP8 from TMA directly (like FlashInfer), no BF16 conversion.
KV arrives as FP8 from TMA directly (half the bytes of BF16).
P (softmax output) is written as FP8 e4m3 to a separate SMEM region.
The FP8 tensorwise dequant scale is folded into the softmax QK scale.

SMEM Layout (native FP8):
  Q FP8:      64 x 576 x 1 = 36864 bytes   (SWIZZLE_64B)
  KV stages:  N x 64 x 576 x 1 bytes  (SWIZZLE_64B, N=num_kv_stages, typically 4)
  P stages:   N x 64 x 64 x 1 bytes    (SWIZZLE_64B, separate from KV)
  max/li:     128 x 4 x 2 = 1024 bytes
  barriers:   (6N+11) fixed + output barriers
"""

from std.math import ceildiv
from std.sys import size_of
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import external_memory
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
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
)
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
    KVTMATile,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    expect_bytes_pred,
    SharedMemPointer,
    MBarType,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    ORaggedTMATile,
    MLA_Decode_Pack,
    OffsetPosition,
    KVPipelineGeneric,
    DecodeSM100MiscMBars,
    DecodeSProducerN,
    DecodePConsumerN,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeKVProducer,
    DecodeKVConsumer,
    DecodeSM100QKTSS_FP8,
    DecodeSM100PVSS_FP8,
)


# ------------------------------------------------------------------------------
# Native FP8 MLA decoding kernel struct for SM100
# All of Q, K, V, P are FP8 in SMEM.
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_QKV_FP8[
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
    # This is used when speculative decoding is enabled.
    fold_q: Bool = False,
    # number of q_tokens folded into
    # the BM=64 M tile under `fold_q=True`. Default 1.
    # Only used inside `comptime if Self.fold_q`
    q_len_fold: Int = 1,
](TrivialRegisterPassable):
    """Native FP8 MLA decode kernel struct for SM100 (B200).

    Holds all of Q, K, V, and P as FP8 e4m3 in shared memory and drives the
    three-warpgroup decode pipeline (softmax, correction, and MMA+load+store)
    using native FP8 WGMMA (`tcgen05.mma.kind::f8f6f4`) for both the QK and PV
    matmuls. The FP8 tensorwise dequant scale is folded into the softmax QK
    scale so no BF16 conversion of Q or KV is needed on the hot path.

    Parameters:
        q_type: The precision used for the softmax accumulator and
            output byte sizing (`bfloat16`), even though
            Q, K, V, and P are FP8 in shared memory.
        KVLUTType: The paged KV cache operand type, providing the
            KV dtype and page size used for TMA loads.
        output_type: The data type of the output tensor written via
            TMA store.
        SplitAccumType: The optional pointer type for the split-K
            LSE accumulation buffer used when decoding is
            partitioned across CTAs.
        MaskType: The attention mask type; one of `NullMask`,
            `CausalMask`, or `SlidingWindowCausalMask`.
        config: The decode configuration providing tile sizes,
            stage counts, head counts, and TMEM layout for the
            kernel.
        ValidLengthType: The optional pointer type for the
            per-request valid sequence length buffer.
        Engine: Engine policy of the `scalar_args` tile operand.
        _is_cache_length_accurate: Whether the cache length used
            for offset computation is accurate (defaults to
            `False`).
        ragged: Whether ragged (variable-length) sequences are
            used, skipping blocks beyond the actual sequence
            length (defaults to `False`).
        fold_q: Whether speculative decoding folds multiple Q
            tokens into the `BM=64` M tile (defaults to `False`).
        q_len_fold: Number of Q tokens folded into the `BM=64` M
            tile under `fold_q=True` (defaults to 1).
    """

    comptime kv_type = Self.KVLUTType.dtype  # float8_e4m3fn
    comptime fp8_type = DType.float8_e4m3fn
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    # FP8: 1 byte per element for all SMEM operands (Q, K, V, P)
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    # BF16: 2 bytes per element for output/accumulator
    comptime bf16_bytes_per_element = size_of[Self.q_type]()
    # KV stage element count (FP8): 9 blocks * 4096 elems = 36864
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    # P stage element count (FP8): 1 block * 4096 elems = 4096
    comptime PStageElems = Self.BlockElems
    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )

    # QK MMA: FP8 (KIND_F8F6F4) — both Q and K are FP8 in SMEM
    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    # PV MMA: FP8 (KIND_F8F6F4) — both P and V are FP8 in SMEM
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
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

    # Number of pipeline stages for KV, S, and P (dynamically computed, typically 4)
    comptime num_stages = Self.config.num_kv_stages

    # --------------------------------------------------------------------------
    # Sliding-window k-tile skip (only callable when MaskType is
    # SlidingWindowCausalMask). All warpgroups call this with the same
    # `offset_position` to avoid barriers deadlock.
    #
    # For SlidingWindowCausalMask with window_size W:
    #   global_lo = max(cache_len + 1 - W, 0)   # 0-based key index
    #   local_lo  = max(global_lo - kv_start_row, 0)
    #   tile_skip = local_lo // BN_QK
    # --------------------------------------------------------------------------
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
        return local_lo // Self.config.BN_QK

    # --------------------------------------------------------------------------
    # Main kernel function — 3 Warpgroups, 384 threads
    # --------------------------------------------------------------------------
    #    Softmax WG (warps 0-3), Correction WG (warps 4-7),
    #    MMA+Load+Store WG (warps 8-11)
    #    Warp assignments within WG2: warp 8 = Load, warp 9 = MMA QK,
    #                                 warp 10 = MMA PV, warp 11 = Store
    #
    # SMEM layout (all FP8, N=num_kv_stages, typically 4):
    #   Q FP8:     64 x 576 x 1 = 36864 bytes (SWIZZLE_64B)
    #   KV stages: N x 64 x 576 x 1 bytes (SWIZZLE_64B)
    #   P stages:  N x 64 x 64 x 1 bytes (SWIZZLE_64B, separate region)
    #   max/li:    128 x 4 x 2 = 1024 bytes
    #   barriers:  (6N+11) fixed + output barriers
    #
    # The FP8 dequant scale is folded into the softmax scale:
    #   qk_scale = (1.0 / sqrt(d_qk)) * kv_scale
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
    @__name(
        t"sm100_mla_decode_qkv_fp8_{Self.q_type}_{Self.kv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        # Q TMA is FP8 with SWIZZLE_64B (same as KV)
        q_tma: QOTMATile[
            dtype=Self.kv_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,  # SWIZZLE_64B
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
        # MaskType assertion: native FP8 backend supports NullMask, CausalMask,
        # and SlidingWindowCausalMask. Sliding window support is exclusive to
        # this backend.
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask"
            or _mask_type_name == "CausalMask"
            or _mask_type_name == "SlidingWindowCausalMask"
        ), (
            "MLA_SM100_Decode_QKV_FP8 supports NullMask, CausalMask, and"
            " SlidingWindowCausalMask only."
        )

        # Extract scalar launch args from the stable device buffer.
        var batch_size = Int(scalar_args.raw_load(0))
        var q_max_seq_len = Int(scalar_args.raw_load(1))
        var num_partitions = mla_decode_pack.num_partitions

        # Register allocation: same as BF16 kernel (3 WGs)
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
                    # pass explicit seq_idx=q_local so each
                    # slot's LSE is written; otherwise q_local>=1 slots stay
                    # uninitialized and poison the combine kernel.
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

        # Sliding-window split-K: a CTA whose entire split lies BELOW the
        # per-row lower bound (causal_limit - W) has nothing to compute and
        # must take the same -inf-LSE early-exit path as `num_keys_this_split
        # == 0`. Comptime-gated so non-sliding builds compile to byte-
        # identical PTX.
        comptime _sliding_window_mask: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask and Self.config.decoding_warp_split_k:
            var _num_k_tiles_total = ceildiv(
                offset_position.num_keys_this_split, Self.config.BN_QK
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
                        # Fold: under seq_len==0 emit -inf for every slot.
                        # Per-q_local ragged fill for 0 < seq_len < q_len_fold
                        # is deferred to A3 (LSE ragged fix).
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

        # ---- SMEM layout (all FP8, N stages) ----
        # Q FP8 region: 64 x 576 x 1 bytes = 36864 bytes
        var q_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        # KV FP8 SMEM: N stages, starts right after Q FP8
        # Each stage = 64 x 576 x 1 = 36864 bytes
        var kv_smem = q_smem + Self.BlockElems * Self.NumQKBlocks

        # P FP8 SMEM: separate region after KV stages
        # Each stage = 64 x 64 x 1 = 4096 bytes
        comptime kv_total_stages = Self.config.num_kv_stages
        var p_smem = kv_smem + Self.KVStageElems * kv_total_stages

        # P stages: num_stages * 4096 elems
        comptime p_total_stages = Self.num_stages

        # Output SMEM reuses KV SMEM (same as BF16 kernel)
        var out_smem_start = kv_smem
        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        # max_smem/li_smem: placed after P stages
        # max_smem is double-buffered (2 x 128 elements) to avoid a race
        # condition in softmax; li_smem is a single 128-element buffer.
        var max_smem = (p_smem + Self.PStageElems * p_total_stages).bitcast[
            Scalar[Self.AccumType]
        ]()
        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        # ---- Barrier layout (6N+11 fixed for N-stage pipelines) ----
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
        # S pipeline: N-stage (matching KV stages)
        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = s_bars.end()
        # P pipeline: N-stage (matching KV stages)
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

        var warp_idx = UInt32(warp_id[broadcast=True]())
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        var is_leader = elect() != 0

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

        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()
            Self.Common_MLA_Op.Softmax[
                native_fp8=True,
                num_sp_stages=Self.num_stages,
                fold_q=Self.fold_q,
                q_len_fold=Self.q_len_fold,
            ](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                p_smem.bitcast[
                    Scalar[Self.Common_MLA_Op.q_type]
                ]().as_unsafe_any_origin(),
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
            warpgroup_reg_alloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
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
                Self.Common_MLA_Op.store[
                    fold_q=Self.fold_q, q_len_fold=Self.q_len_fold
                ](
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

    # --------------------------------------------------------------------------
    # Load: TMA Q (FP8) directly, TMA KV (FP8) directly
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def load(
        q_tma: QOTMATile[
            dtype=Self.kv_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,  # SWIZZLE_64B
        ],
        k_tma: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BK_PV,
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
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Alignment of `kv_row` produced by mask-driven iteration.
        comptime base_alignment: Int = Self.MaskType.start_column_alignment[
            Self.config.BM, Self.config.BN_QK, Self.KVLUTType.page_size
        ]()

        # Sliding-window early exit + leading-tile skip (comptime-gated;
        # entire block compiles away for non-sliding masks).
        comptime _sliding_window_mask: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        # Lives only inside the comptime SW block.
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
        # Advance the starting KV row past skipped sliding-window tiles.
        comptime if _sliding_window_mask:
            kv_row += UInt32(_tile_skip * Self.config.BN_QK)
        var num_keys_u32 = UInt32(offset_position.num_keys)
        kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
        var paged_rows = kv_lut.populate[Self.config.BN_QK, base_alignment](
            UInt32(offset_position.batch_idx), kv_row
        )

        # Load Q via TMA as FP8 (no conversion needed)
        expect_bytes_pred(
            mbar_q,
            Int32(
                Self.config.BM
                * Self.config.q_depth
                * Self.fp8_bytes_per_element
            ),
            elect_mask,
        )
        if is_leader:
            # Q TMA: load FP8 Q directly into q_smem
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

        # Load first KV tile (FP8)
        var k0_bar: MBarType = kv_prod.producer_mbar[qk_stage=0]()
        expect_bytes_pred(
            k0_bar,
            Int32(
                Self.config.BN_QK
                * Self.config.q_depth
                * Self.fp8_bytes_per_element
            ),
            elect_mask,
        )
        var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
        paged_rows.tma_copy_k[needs_partial=False](
            k_tma,
            stage_ptr,
            k0_bar[],
            kv_head_idx=UInt32(0),
            elect=elect_mask,
        )
        kv_prod.commit_step()
        kv_row += UInt32(Self.config.BN_QK)

        # Wait for Q TMA to complete. Q arrives as FP8, ready for MMA.
        mbar_q[].wait(0)

        # Load remaining KV tiles
        var tile_idx: Int = 1
        while tile_idx < num_k_tiles:
            kv_prod.acquire[qk_stage=0]()
            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()
            kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
            var paged_rows = kv_lut.populate[Self.config.BN_QK, base_alignment](
                UInt32(offset_position.batch_idx), kv_row
            )

            expect_bytes_pred(
                k_mbar,
                Int32(
                    Self.config.BN_QK
                    * Self.config.q_depth
                    * Self.fp8_bytes_per_element
                ),
                elect_mask,
            )
            paged_rows.tma_copy_k[needs_partial=False](
                k_tma,
                stage_ptr,
                k_mbar[],
                kv_head_idx=UInt32(0),
                elect=elect_mask,
            )

            kv_row += UInt32(Self.config.BN_QK)
            kv_prod.commit_step()
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MMA QK: Q(FP8) x K(FP8) -> S(TMEM)
    # --------------------------------------------------------------------------
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
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        if num_k_tiles == 0:
            return

        # Sliding-window early exit + leading-tile skip (comptime-gated;
        # entire block compiles away for non-sliding masks). Must match
        # `load` exactly so consumer iterations equal producer iterations.
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
        # N-stage S producer
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        # FP8 descriptors for Q and K
        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        # Stage stride in bytes: FP8 elements (1 byte each)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.fp8_bytes_per_element

        # Q FP8 is ready (load warp waited on mbar_q and Q arrived as FP8).
        # The barrier() at the end of init ensures visibility.

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

    # --------------------------------------------------------------------------
    # MMA PV: P(FP8) x V(FP8) -> O(TMEM)
    # --------------------------------------------------------------------------
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
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        if num_k_tiles == 0:
            return

        # Sliding-window early exit + leading-tile skip (comptime-gated;
        # entire block compiles away for non-sliding masks). Must match
        # `load` exactly so consumer iterations equal producer iterations.
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
        # N-stage P consumer
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        # P descriptor: points to the separate P SMEM region
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        # V descriptor: points to the KV SMEM (V data is first 512 cols of KV)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        # FP8: 1 byte per element
        comptime kv_stage_stride_in_bytes = Self.KVStageElems * Self.fp8_bytes_per_element
        comptime p_stage_stride_in_bytes = Self.PStageElems * Self.fp8_bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.fp8_bytes_per_element

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            # PV does not have the k-rope so we don't need to do the last block
            comptime for block in range(0, Self.NumVOBlocks, block_step):
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(p_stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(kv_stage_stride_in_bytes)
                    + UInt32(block * block_stride_in_bytes),
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN_QK // 2),
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
