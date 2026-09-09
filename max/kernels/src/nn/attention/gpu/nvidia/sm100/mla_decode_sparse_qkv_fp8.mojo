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

"""Native FP8 sparse MLA decode kernel for SM100 (B200) -- unified gather.

Combines the OLD kernel's efficient contiguous KV gather
(mla_decode_sparse_kv_fp8.mojo: INT64/SWIZZLE_NONE gather4, 16 gather4
instructions/tile at 2304 B/txn) with the native-FP8 compute core
(tcgen05.mma.kind::f8f6f4, no dtype conversion). The two are decoupled by a
re-swizzle staging warpgroup: KV gathers contiguously into a LINEAR FP8
SMEM buffer, then WG3 (repurposing the old kernel's convert-WG role, minus
the dtype cast) permutes it FP8->FP8 into the SW64 byte layout the FP8
tcgen05 operand descriptor expects, into a SECOND SMEM buffer. QK^T and P*V
still run natively in FP8 reading the SW64 buffer -- only the KV
front-end changed; the compute core is byte-for-byte what
MLA_SM100_Decode_Sparse_QKV_FP8 already validated.

Root cause this fixes (see .claude/agent-memory/mojo-kernel-engineer/
mla-decode-sm100-native-fp8-warpgroup-count.md): the native kernel's own
K/Q operand layout mandates SWIZZLE_64B for tcgen05.mma.kind::f8f6f4, and a
gather4 TMA under SW64 has box_width=64B, so a 576-byte row needs
ceildiv(576,64)=9 column-groups x 16 four-row-chunks = 144 gather4
instructions/tile -- 9x the old kernel's 16 instructions/tile (SWIZZLE_NONE,
single 576-byte box) for identical bytes. That per-tile hardware-instruction
inflation, not pipeline depth or warp count, is what NCU's PC-sampled stall
attribution pinned to the KV-gather producer path.

4 warpgroups (back from 3 -- the re-swizzle stage, like the old kernel's
convert stage, needs its own resident warps; it is a pure FP8->FP8 layout
permute, not a dtype conversion):
  WG0 (warps  0-3):   Softmax  (native_fp8=True)
  WG1 (warps  4-7):   Correction
  WG2 (warps  8-11):  Load (warp 8, INT64/SWIZZLE_NONE gather4) / MMA QK
                      (warp 9) / MMA PV (warp 10) / idx_producer + Store
                      (warp 11) -- unchanged roles from the 3-WG kernel.
  WG3 (warps 12-15):  Re-swizzle: read the contiguously-gathered FP8 linearly,
                      write FP8 (unchanged bytes, no cast) into the SW64
                      layout via a manual swizzled SMEM store (all 128
                      threads cooperate, mirroring convertFP8ToBF16's
                      thread mapping but without the dtype cast or the
                      BF16-overlay aliasing -- this uses a separate,
                      non-overlaid destination buffer).

SMEM layout (FP8, N = num_kv_stages, M = num_kv_mma_stages; expect N ~= 2 at
the target shape, since the linear + SW64 buffer pair costs the same total
bytes/stage as the old kernel's own KV region). The two KV buffers do NOT
share a depth: `PV_REL(i)` frees the SW64 slot that `CVT` reclaims for tile
i + M, and that WAR edge is the back-edge of the loop that sets the kernel's
steady-state rate, so the rate is bounded by (cycle weight) / M. The gather
ring is not on that loop, so leftover SMEM buys depth on the SW64 side only
-- a symmetric extra stage does not fit:
  Q FP8:          64 x 576 x 1 = 36864 bytes         (SWIZZLE_64B, persistent)
  KV linear-in:   N x 64 x 576 x 1 bytes              (SWIZZLE_NONE, gather dest)
  KV SW64-out:    M x 64 x 576 x 1 bytes              (SWIZZLE_64B, MMA operand)
  P stages:       N x 64 x 64 x 1 bytes               (SWIZZLE_64B, separate region)
  max/li:         128 x 4 x 3 = 1536 bytes
  barriers:       (6N+2M+11) fixed + output + idx_bars (2N) barriers
  ptr_tmem_addr:  4 bytes
  idx_smem:       N x 64 x 4 bytes
"""

from std.collections import OptionalReg
from std.memory import UnsafePointer
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
from layout import (
    ComptimeInt,
    DefaultEngine,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
)
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import OptionalPointer
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
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
    ld_shared_v4_u32,
    st_shared_v4_b32_at_bf16_elem_off,
)


struct MLA_SM100_Decode_Sparse_QKV_FP8[
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
    fold_shared_index: Bool = False,
    q_len_fold: Int = 1,
    Engine: TensorEngine = DefaultEngine[element_width=1],
](TrivialRegisterPassable):
    """Sparse MLA decode with native FP8 WGMMA for SM100 (B200), unified gather.

    Gathers KV contiguously (INT64/SWIZZLE_NONE, like
    MLA_SM100_Decode_Sparse_KV_FP8) then re-swizzles FP8->FP8 (no dtype
    conversion) into the SW64 layout the native tcgen05.mma.kind::f8f6f4
    operand expects. QK^T and PV MMA are unchanged from the single-buffer
    native-FP8 kernel this supersedes.
    """

    comptime kv_type = Self.KVLUTType.dtype
    comptime fp8_type = DType.float8_e4m3fn
    comptime AccumType = get_accum_type[Self.q_type]()
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime PStageElems = Self.BlockElems
    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )

    # Contiguous gather4 TMA descriptor: INT64, SWIZZLE_NONE, single box
    # covering the full 576-byte row (matches
    # MLA_SM100_Decode_Sparse_KV_FP8 exactly -- 16 gather4 instructions/tile
    # instead of the SW64-fragmented 144/tile).
    comptime kv_gather4_tile_width = Self.config.input_q_depth // 8
    comptime kv_gather4_box_w = _gather4_box_width[
        DType.int64,
        Self.kv_gather4_tile_width,
        TensorMapSwizzle.SWIZZLE_NONE,
    ]()

    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    comptime num_stages = Self.config.num_kv_stages
    # Depth of the SW64 ring only. `PV_REL(i)` frees the slot that
    # `CVT_SW64_ACQ(i + num_mma_stages)` reclaims, and that WAR edge is the
    # back-edge of the loop that sets this kernel's steady-state rate, so the
    # bound is (cycle weight) / this depth. The linear gather ring is not on
    # that loop and stays at `num_stages`.
    comptime num_mma_stages = Self.config.num_kv_mma_stages

    # The ground-truth SW64 destination layout the MMA operand descriptor
    # already reads today -- the re-swizzle WG's job is to reproduce this
    # exact byte permutation from the contiguously-gathered linear source.
    comptime sw_fp8 = make_swizzle[
        Self.fp8_type, TensorMapSwizzle.SWIZZLE_64B
    ]()

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

    @staticmethod
    @always_inline
    def zero_input_tail(
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
    ):
        """Zero the SMEM columns past the input row, once per CTA.

        Once is enough without qualification: `p_smem` is its own region here,
        not an alias of the tail, so nothing rewrites these columns.
        """
        comptime tail_block0 = Self.config.input_q_depth // Self.config.BN_QK
        comptime tail_elems = (Self.NumQKBlocks - tail_block0) * Self.BlockElems
        comptime tail_off = tail_block0 * Self.BlockElems
        comptime zero = Scalar[Self.fp8_type](0)

        var tid = Int(thread_idx.x)
        for i in range(tid, tail_elems, Self.config.num_threads):
            q_smem[tail_off + i] = zero
        comptime for stage in range(Self.num_mma_stages):
            var base = kv_smem + stage * Self.KVStageElems + tail_off
            for i in range(tid, tail_elems, Self.config.num_threads):
                base[i] = zero

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
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
        ],
        # Single K gather4 TMA covering the row as stored: INT64,
        # SWIZZLE_NONE, tile_width = input_q_depth / 8 INT64 (matches
        # MLA_SM100_Decode_Sparse_KV_FP8's efficient contiguous gather).
        k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
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
        # Logical sparse indices (same layout as `d_indices`, -1 padding) for
        # position-based causal masking. See mla_decode_utils.mojo.
        logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        scales_ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
        attn_sink_ptr: OptionalReg[UnsafePointer[Float32, origin=MutAnyOrigin]],
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
        extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
        scalar_args: TileTensor[
            .int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
            Engine=Self.Engine,
        ],
    ):
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_Sparse_QKV_FP8 supports NullMask and CausalMask"
            " only. Sliding-window is not yet wired for sparse."
        )
        comptime assert (
            not Self.fold_shared_index
            or Self.config.num_q_heads * Self.q_len_fold <= Self.config.BM
        ), "fold_shared_index requires num_q_heads * q_len_fold <= BM."
        comptime assert not (
            Self.fold_shared_index and Self.has_extra_kv
        ), "fold_shared_index does not support extra always-attend KV cache."

        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 64
        comptime num_reg_reswizzle = 72

        var indices_stride = Int(indices_stride_dev)
        var extra_indices_stride = Int(extra_indices_stride_dev)
        var batch_size = Int(scalar_args.ptr[0])
        var q_max_seq_len = Int(scalar_args.ptr[1])
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
        topk = offset_position.num_keys - extra_topk

        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)

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

        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                _pdl_early_exit_all_q()
                return

        comptime if Self.ragged:
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    _pdl_early_exit_all_q()
                return

        var q_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        comptime kv_total_stages = Self.config.num_kv_stages

        # Linear (SWIZZLE_NONE) gather destination -- the re-swizzle WG's
        # source. Same per-stage size as the SW64 destination below.
        var kv_smem_linear = q_smem + Self.BlockElems * Self.NumQKBlocks
        # SW64 destination the native FP8 MMA reads -- byte-identical role
        # to the single-buffer kernel's `kv_smem`, one ring deeper.
        var kv_smem = kv_smem_linear + Self.KVStageElems * kv_total_stages
        var p_smem = kv_smem + Self.KVStageElems * Self.num_mma_stages

        # Output SMEM reuses the SW64 KV region (same as the single-buffer
        # native FP8 kernel and the BF16 kernel).
        var out_smem = kv_smem.bitcast[Scalar[Self.output_type]]()

        var max_smem = (p_smem + Self.PStageElems * Self.num_stages).bitcast[
            Scalar[Self.AccumType]
        ]()
        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        var mbar_base: MBarType = (
            (li_smem + WARPGROUP_SIZE)
            .bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        var mbar_kv_gather_base: MBarType = mbar_base + 1
        # Load (TMA, 1 leader thread) -> re-swizzle WG (128 threads, all
        # independently release).
        var kv_pipeline_gather = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_kv_gather_base)
        var mbar_kv_mma_base: MBarType = (
            mbar_kv_gather_base + kv_pipeline_gather.num_mbars()
        )
        # Re-swizzle WG (128 threads, all independently commit) -> mmaQK +
        # mmaPV (2 consumers) -- same consumer shape as the single-buffer
        # kernel's own KV pipeline, just producer side is now a warpgroup
        # instead of a single TMA leader thread.
        var kv_pipeline_mma = KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
        ](mbar_kv_mma_base)
        var mbar_base2: MBarType = (
            mbar_kv_mma_base + kv_pipeline_mma.num_mbars()
        )

        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base2)
        var mbar_base3: MBarType = s_bars.end()

        var p_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base3)
        var mbar_base4: MBarType = p_bars.end()

        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](mbar_base4)
        var mbar_base5: MBarType = o_bars.end()

        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base5)
        var mbar_base6: MBarType = c_bars.end()

        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base6)
        var mbar_base7: MBarType = corr_done_bars.end()

        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base7)
        var mbar_base8: MBarType = mbar_base7 + out_pipeline.num_mbars()

        # idx_bars: warp 11 (producer, 32 threads) -> warp 8 (consumer, 32
        # threads). Depth matches Self.num_stages (= config.num_kv_stages),
        # not a fixed 2 -- see the KB entry cited in the module docstring.
        var idx_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages, num_producer=32, num_consumer=32
        ](mbar_base8)
        var mbar_base9: MBarType = idx_bars.end()

        var ptr_tmem_addr = (mbar_base9).bitcast[UInt32]()
        var idx_smem_base = (ptr_tmem_addr + 1).bitcast[Int32]()
        comptime idx_smem_stride = Self.config.BN_QK

        var warp_idx = UInt32(warp_id[broadcast=True]())
        var is_leader = elect() != 0

        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                kv_pipeline_gather.init()
                kv_pipeline_mma.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                corr_done_bars.init()
                out_pipeline.init()
                idx_bars.init()
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
                comptime if Self.has_extra_kv:
                    extra_k_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        # Zero the NoPE tail before the warpgroups read it. The fence publishes
        # the generic-proxy stores to the async proxy for the tcgen05 read.
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            Self.zero_input_tail(
                q_smem.as_unsafe_any_origin(), kv_smem.as_unsafe_any_origin()
            )
        barrier()
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            fence_async_view_proxy()

        if warp_idx < 4:
            warpgroup_reg_alloc[num_reg_softmax]()

            var attn_sink_log2 = Float32(min_or_neg_inf[.float32]())
            comptime if Self.has_attn_sink:
                var lane_idx = Int(lane_id())
                var row = lane_idx & 0x3F
                var head_idx_local = Int(block_idx.x) * Self.config.BM + row
                if head_idx_local < Self.config.num_q_heads:
                    attn_sink_log2 = attn_sink_ptr.unsafe_value()[
                        head_idx_local
                    ] * Float32(log2e)

            Self.Common_MLA_Op.Softmax[
                native_fp8=True,
                num_sp_stages=Self.num_stages,
                has_attn_sink=Self.has_attn_sink,
                fold_q=Self.fold_shared_index,
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
                attn_sink_log2=attn_sink_log2,
                logical_indices=logical_indices,
                logical_indices_stride=indices_stride,
                logical_indices_len=topk,
            )
        elif warp_idx >= 4 and warp_idx < 8:
            warpgroup_reg_alloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        elif warp_idx >= 8 and warp_idx < 12:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem.as_unsafe_any_origin(),
                    kv_smem_linear.as_unsafe_any_origin(),
                    mbar_q,
                    kv_pipeline_gather,
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
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline_mma,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem.as_unsafe_any_origin(),
                    p_smem.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline_mma,
                    offset_position,
                )
            elif warp_idx == 11:
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
                    offset_position,
                    num_orig_blocks,
                    extra_kv_lut,
                    batch_extra_d_indices_w11,
                    extra_topk,
                )
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
            # WG3 (warps 12-15): re-swizzle staging warpgroup. Reads the
            # contiguously-gathered linear FP8, writes FP8 (no cast) into
            # the SW64 layout mmaQK/mmaPV already expect. Repurposes the
            # old kernel's convert-WG role/register budget without the
            # dtype conversion.
            warpgroup_reg_dealloc[num_reg_reswizzle]()
            Self.reswizzleFP8(
                kv_smem_linear.as_unsafe_any_origin(),
                kv_smem.as_unsafe_any_origin(),
                kv_pipeline_gather,
                kv_pipeline_mma,
                offset_position,
            )
        barrier()

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
        var lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1

        comptime for row_pass in range(2):
            var row_in_tile = lane + row_pass * 32
            var idx_pos = UInt32(indices_base + row_in_tile)
            var clamped_pos = min(idx_pos, max_idx)
            var raw_index = d_indices.unsafe_value()[Int(clamped_pos)]
            var tma_row = kv_lut.get_tma_row(raw_index)
            if raw_index == -1:
                tma_row = -1
            idx_smem[row_in_tile] = tma_row

    @staticmethod
    @always_inline
    def idx_producer(
        idx_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        kv_lut: Self.KVLUTType,
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        topk: Int,
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
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var idx_prod = idx_bars.producer()
        var orig_topk_u32 = UInt32(topk)

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
        # Stage index comes from idx_prod's own ring counter (PipelineState,
        # any depth), not a hand-rolled toggle.

        if first_tile_from_orig:
            var idx_smem = idx_smem_base + idx_prod.state.index() * UInt32(
                Self.config.BN_QK
            )
            Self._transform_indices_to_smem(
                d_indices,
                idx_smem,
                orig_indices_base,
                kv_lut,
                orig_topk_u32,
            )
            idx_prod.commit()
            orig_indices_base += Self.config.BN_QK

        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        var t: Int = 0
        while t < remaining_orig:
            idx_prod.acquire()
            var idx_smem = idx_smem_base + idx_prod.state.index() * UInt32(
                Self.config.BN_QK
            )
            Self._transform_indices_to_smem(
                d_indices,
                idx_smem,
                orig_indices_base,
                kv_lut,
                orig_topk_u32,
            )
            idx_prod.commit()
            orig_indices_base += Self.config.BN_QK
            t += 1

        comptime if Self.has_extra_kv:
            var extra_topk_u32 = UInt32(extra_topk)
            var extra_indices_base = max(
                0, Int(offset_position.kv_start_row) - topk
            )
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                var idx_smem = idx_smem_base + idx_prod.state.index() * UInt32(
                    Self.config.BN_QK
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                )
                idx_prod.commit()
                extra_indices_base += Self.config.BN_QK
                num_extra_tiles -= 1

            var te: Int = 0
            while te < num_extra_tiles:
                idx_prod.acquire()
                var idx_smem = idx_smem_base + idx_prod.state.index() * UInt32(
                    Self.config.BN_QK
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                )
                idx_prod.commit()
                extra_indices_base += Self.config.BN_QK
                te += 1

    @staticmethod
    @always_inline
    def load(
        q_tma: QOTMATile[
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
        ],
        k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem_linear: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        kv_pipeline_gather: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
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
            num_stages=Self.num_stages,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
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
        if num_k_tiles == 0:
            return

        var kv_prod = DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](kv_pipeline_gather, kv_smem_linear)
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        # Match the Q TMA's transfer size (the row as stored), not the SMEM width.
        expect_bytes_pred(
            mbar_q,
            Int32(
                Self.config.BM
                * Self.config.input_q_depth
                * Self.fp8_bytes_per_element
            ),
            elect_mask,
        )
        if is_leader:
            comptime q_elems = type_of(q_tma).tile_shape[0] * type_of(
                q_tma
            ).tile_shape[1]
            comptime q_tt_layout = tt_row_major[q_elems]()
            var q_smem_tensor = TileTensor[
                Self.fp8_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](q_smem.bitcast[Scalar[Self.fp8_type]](), q_tt_layout)
            q_tma.async_copy(q_smem_tensor, mbar_q[], (0, row))

        # The row as stored: BN_QK * (input_q_depth / 8) INT64 bytes.
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var first_tile_from_orig = num_orig_tiles > 0
        var idx_cons = idx_bars.consumer()

        if first_tile_from_orig:
            Self._load_one_tile(
                kv_prod,
                is_leader,
                k_tma,
                idx_cons,
                idx_smem_base,
                kv_bytes,
            )
            # Wait here so Q load overlaps with the first KV transfer.
            mbar_q[].wait(0)

        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        Self._load_tile_range(
            kv_prod,
            is_leader,
            k_tma,
            idx_cons,
            idx_smem_base,
            remaining_orig,
        )

        comptime if Self.has_extra_kv:
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                Self._load_one_tile(
                    kv_prod,
                    is_leader,
                    extra_k_tma,
                    idx_cons,
                    idx_smem_base,
                    kv_bytes,
                )
                # First tile comes from extra_kv since orig was empty.
                mbar_q[].wait(0)
                num_extra_tiles -= 1

            Self._load_tile_range(
                kv_prod,
                is_leader,
                extra_k_tma,
                idx_cons,
                idx_smem_base,
                num_extra_tiles,
            )

    @staticmethod
    @always_inline
    def _load_one_tile(
        mut kv_prod: DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[Self.num_stages],
        idx_smem_base: SharedMemPointer[Int32],
        kv_bytes: Int,
    ):
        var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
        var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

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

        idx_cons.release()
        kv_prod.commit_step()

    @staticmethod
    @always_inline
    def _load_tile_range(
        mut kv_prod: DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            DType.int64,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[Self.num_stages],
        idx_smem_base: SharedMemPointer[Int32],
        num_tiles: Int,
    ):
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var t: Int = 0
        while t < num_tiles:
            kv_prod.acquire[qk_stage=0]()

            var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

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

            idx_cons.release()
            kv_prod.commit_step()
            t += 1

    @staticmethod
    @always_inline
    def reswizzleFP8(
        kv_smem_linear: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem_sw64: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_pipeline_gather: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        """Re-swizzles one KV tile per iteration: FP8 linear -> FP8 SW64.

        Pure layout permute, no dtype cast (unlike the old kernel's
        convertFP8ToBF16). Uses a separate, non-overlaid destination
        buffer, so there is no free/held-block aliasing split to manage --
        every column block is read then written independently.

        Thread mapping mirrors convertFP8ToBF16 exactly (GROUP_SIZE=4,
        128 threads -> 32 groups of 2 rows each, 9 column iterations of
        16 bytes/thread cover the row as stored) so the same
        bank-conflict-reduction rationale applies.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime BN_QK: Int = Self.config.BN_QK
        comptime fp8_row_stride: Int = Self.config.input_q_depth
        comptime GROUP_SIZE: Int = 4
        comptime NUM_GROUPS: Int = WARPGROUP_SIZE // GROUP_SIZE  # 32
        # input_q_depth / (4 threads * 16 bytes) column iterations.
        comptime COLS_PER_GROUP: Int = fp8_row_stride // (GROUP_SIZE * 16)

        var kv_gather_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem_sw64)

        var lane: Int = thread_idx.x & 0x7F
        var group_idx: Int = lane // GROUP_SIZE  # 0..31
        var idx_in_group: Int = lane % GROUP_SIZE  # 0..3

        var row_0: Int = 0 * NUM_GROUPS + group_idx
        var row_1: Int = 1 * NUM_GROUPS + group_idx

        # Linear-source read base offsets (bytes == elements for FP8).
        var fp8_base_0: Int = row_0 * fp8_row_stride + idx_in_group * 16
        var fp8_base_1: Int = row_1 * fp8_row_stride + idx_in_group * 16

        # SW64-destination swizzled element offsets. One 16-byte read maps
        # to exactly one 16-byte write: the SW64 swizzle's `base` (untouched
        # low bits) covers 16 FP8 elements (log2_floor(16 // 1 byte) = 4),
        # so a 16-byte-aligned source chunk always lands at a 16-byte
        # -aligned (permuted) destination chunk -- no finer-grained scatter.
        var col: Int = idx_in_group * 16
        var sw_off_0: Int = Self.sw_fp8(row_0 * BN_QK + col)
        var sw_off_1: Int = Self.sw_fp8(row_1 * BN_QK + col)

        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            kv_mma_prod.acquire()

            var src_u8 = kv_gather_cons.stage_base_ptr().bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()

            comptime for c in range(COLS_PER_GROUP):
                var q0 = ld_shared_v4_u32(
                    src_u8, fp8_base_0 + c * GROUP_SIZE * 16
                )
                var q1 = ld_shared_v4_u32(
                    src_u8, fp8_base_1 + c * GROUP_SIZE * 16
                )
                var dst_block = dst + c * Self.BlockElems
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.fp8_type](
                    dst_block, sw_off_0, q0
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.fp8_type](
                    dst_block, sw_off_1, q1
                )

            fence_async_view_proxy()
            kv_gather_cons.release_all()
            kv_mma_prod.commit_all()
            tile_idx += 1

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
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = (
            Self.KVStageElems * Self.fp8_bytes_per_element
        )

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
            tile_idx += 1

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
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        if num_k_tiles == 0:
            return

        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        comptime kv_stage_stride_in_bytes = (
            Self.KVStageElems * Self.fp8_bytes_per_element
        )
        comptime p_stage_stride_in_bytes = (
            Self.PStageElems * Self.fp8_bytes_per_element
        )
        comptime block_stride_in_bytes = (
            Self.BlockElems * Self.fp8_bytes_per_element
        )

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

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
