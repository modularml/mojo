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

"""
Implements the SM100 (Blackwell) MLA prefill attention kernel with blockwise
FP8 scaling, converting block-scaled FP8 K_rope tiles to BF16 in shared
memory before the MMA stage.
"""

from std.sys import size_of
from std.math import ceildiv, align_up
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    StagedPipeline,
    VProducerPipeline,
    KConsumerPipeline,
    VConsumerPipeline,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    o_store_tma_blocks_per_op,
    PagedRowIndices,
    SharedMemPointer,
    elect,
    expect_bytes_pred,
)
from nn.attention.gpu.mha import q_num_matrix_view_rows
from nn.attention.gpu.nvidia.common import (
    get_seq_info,
    KVTMATile,
    kv_coord,
    NonNullPointer,
    NullPointer,
    Pack,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from layout.swizzle import make_swizzle
from linalg.arch.sm100.mma import smem_descriptor

from max.gpu.globals import WARP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, thread_idx, warp_id
from max.gpu.sync import barrier
from nn.attention.mha_utils import (
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
)
from max.gpu.compute.arch.tcgen05 import *
from std.utils.static_tuple import StaticTuple
from kv_cache.types import padded_depth

from nn.attention.gpu.nvidia.sm100.mla_prefill_utils import (
    MLAConfig,
    SM100MLA,
    MLAPositionSummary,
    MLAKVLayouts,
    select_mla_prefill_config,
    split_smem,
    TMAtoCvtPipeline,
    CvtToMMAPipeline,
    cvt_block_fp8_to_bf16_with_scale,
)
from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.nvidia.sm100.smem import SM100AttentionSMem


@fieldwise_init
struct WarpRole(Equatable, TrivialRegisterPassable):
    """Tags each warp with its specialized role in the warp-specialized MLA prefill kernel.
    """

    var _role: Int32
    comptime Softmax0 = Self(0)
    comptime Softmax1 = Self(1)
    comptime Correction = Self(2)
    comptime MMA = Self(3)
    comptime Load = Self(4)
    comptime CVTToBF16 = Self(5)
    comptime Empty = Self(6)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self == Self(Int32(other))


def warp_idx_to_role(warp_idx: UInt32) -> WarpRole:
    """Maps a global warp index to its `WarpRole` based on warpgroup assignment.

    Args:
        warp_idx: Global warp index within the CTA, where each warpgroup
            spans four consecutive warps. Determines the specialized role
            (softmax, correction, MMA, load, or FP8-to-BF16 conversion) via
            `warp_idx // 4` for warpgroup assignment and direct index checks
            for warps 12 and above.
    """
    var wg_idx = warp_idx // 4
    if wg_idx == 0:
        return WarpRole.Softmax0
    elif wg_idx == 1:
        return WarpRole.Softmax1
    elif wg_idx == 2:
        return WarpRole.Correction
    elif warp_idx == 12:
        return WarpRole.MMA
    elif warp_idx == 13:
        return WarpRole.Load
    elif warp_idx >= 14 and warp_idx < 16:
        return WarpRole.CVTToBF16
    else:
        return WarpRole.Empty


__extension SM100MLA:
    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_nope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"sm100_mla_prefill_blockscale_{Self.qkv_dtype}_{Self.output_dtype}_{blockwise_scale}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def mla_prefill_kernel_blockscale[
        blockwise_scale: Int = 0,
    ](
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.BK0,
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.ov_depth,  # V tile: ov_depth-wide
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_dtype,
            Self.config.output_swizzle_mode,
            # `// fa4_config.num_q` matches fa4_softmax's unified
            # 1Q/2Q signature; numerically `// 2` for num_q=2.
            BM=Self.config.fa4_config.BM // Self.config.fa4_config.num_q,
            BN=Self.config.fa4_config.ov_depth,
            middle_dim=Self.config.num_q_heads,
            group=config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
            # Concrete value: a GPU-kernel entry must be a fully-bound function.
            # Matches the created store.
            tma_blocks_per_op=o_store_tma_blocks_per_op[
                Self.output_dtype,
                Self.config.output_swizzle_mode,
                Self.config.fa4_config.ov_depth,
                config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
                depth_splits=2,
            ](),
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        scale: Float32,
        batch_size: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            Self.SinkType,
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
    ):
        comptime assert Self.KRopeType.dtype == config.rope_gmem_dtype
        comptime assert Self.KVLUTType.dtype == config.rope_mma_dtype
        comptime assert Self.MMA_M == 64 or Self.MMA_M == 128
        comptime assert Self.config.supported(), (
            "\nqk_depth = "
            + String(Self.config.qk_depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used)
            + "\n% smem_used = "
            + String(
                Float64(Self.config.smem_used)
                / Float64(Self.config.sm100_smem_carveout)
            )
            + "\nsmem_total = "
            + String(Self.config.sm100_smem_carveout)
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        var mask = pack.mask
        var scheduler = pack.scheduler
        var valid_length = pack.valid_length
        var max_seq_len = pack.max_seq_len
        var partition = pack.partition

        comptime num_q = Self.config.num_q()
        # TODO: We may want to support num_q>2 for depth=64?
        comptime assert (
            num_q == 1 or num_q == 2
        ), "Currently only support num_q == 1 or 2"
        # Blockscale uses its own extra barriers for the FP8→BF16 rope
        # conversion pipeline, so skip FA4MiscMBars rope barriers to stay
        # within the 32-barrier hardware limit.
        comptime SmemType = SM100AttentionSMem[Self.config.fa4_config]
        var attn_smem = SmemType()
        var misc_mbars = attn_smem.misc_mbars()
        var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.q_smem()
        )
        var k_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.v_smem_base()
        )
        var rope_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.rope_smem_base()
        )
        var ptr_tmem_addr = attn_smem.tmem_addr_ptr()

        # Extra blockscale barriers are placed after the SM100AttentionSMem layout.
        # Align to 8 bytes for SharedMemBarrier.
        comptime extra_barrier_offset = align_up(
            SmemType.smem_size(), size_of[SharedMemBarrier]()
        )
        var extra_base = (attn_smem.base + extra_barrier_offset).bitcast[
            SharedMemBarrier
        ]()
        comptime num_rope_bufs = Self.config.fa4_config.num_rope_buffers()
        var tma_to_cvt_producer_mbars = extra_base
        var tma_to_cvt_consumer_mbars = extra_base + num_rope_bufs
        var cvt_to_mma_producer_mbars = extra_base + 2 * num_rope_bufs
        var cvt_to_mma_consumer_mbars = extra_base + 2 * num_rope_bufs + 2

        var tma_to_cvt_pipeline = TMAtoCvtPipeline[
            num_rope_bufs,
            num_producer=1,
            num_consumer=64,
        ](tma_to_cvt_producer_mbars, tma_to_cvt_consumer_mbars)

        var cvt_to_mma_pipeline = CvtToMMAPipeline[
            2,
            num_producer=64,
            num_consumer=1,
        ](cvt_to_mma_producer_mbars, cvt_to_mma_consumer_mbars)

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 96
        comptime num_reg_other = 48
        comptime num_reg_empty = 24

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/KV/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
            tma_to_cvt_pipeline.init()
            cvt_to_mma_pipeline.init()
        elif warp_idx == 1:
            tcgen05_alloc[Self.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        elif warp_idx == 2:
            var e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_nope_tma_op.prefetch_descriptor()
            if e != 0:
                k_rope_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        barrier()

        # Read the TMEM base from SMEM ONCE here, post-barrier (alloc + this
        # barrier publish it), and carry it by register into the shared
        # fa4_softmax / fa4_correction consumers (see depth-512 fix).
        var tmem_addr = ptr_tmem_addr[0]

        var role = warp_idx_to_role(warp_idx)

        # warp group partitioning
        # Two QO:
        if role == WarpRole.Softmax0 or role == WarpRole.Softmax1:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            fa4_softmax[
                Self.KVLUTType,
                Self.config.fa4_config,
                Self.ValidLengthType,
                NullPointer[Self.output_dtype],
                False,
                Self.MaxSeqLenType,
            ](
                attn_smem,
                tmem_addr,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_dtype](),
                max_seq_len.as_uint32(),
                ragged_tma_store,
                NullPointer[Self.output_dtype](),
            )

        elif role == WarpRole.Correction:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)
            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            fa4_correction[
                Self.config.fa4_config,
                Self.page_size,
            ](
                attn_smem,
                tmem_addr,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        elif role == WarpRole.Load:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            Self.load(
                tma_to_cvt_pipeline,
                pos.score_row,
                pos.num_keys,
                seq_info,
                max_seq_len,
                mask,
                q_tma_op,
                k_nope_tma_op,
                k_rope_tma_op,
                v_tma_op,
                kv_lut,
                k_rope_lut,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
            )

        elif role == WarpRole.MMA:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                tcgen05_release_allocation_lock[Self.cta_group]()
                tcgen05_dealloc[Self.cta_group](
                    ptr_tmem_addr[0], Self.config.sm100_tmem_cols
                )
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            Self.mma(
                ptr_tmem_addr[0],
                cvt_to_mma_pipeline,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
            )
        elif role == WarpRole.CVTToBF16:
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            var iter_count: UInt32 = (
                mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                    seq_info.prompt_idx, pos.score_row, pos.num_keys
                )
                - 1
            )

            var local_thread_idx = UInt32(thread_idx.x - 14 * WARP_SIZE)

            Self.convert_fp8_to_bf16(
                iter_count,
                tma_to_cvt_pipeline,
                cvt_to_mma_pipeline,
                k_rope_lut,
                k_smem,
                rope_smem,
                seq_info,
                pos.num_keys,
                local_thread_idx,
            )

        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    def load(
        mut tma_to_cvt_pipeline: TMAtoCvtPipeline,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.BK0,  # padded depth -> 192
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.ov_depth,  # V tile: ov_depth-wide
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = MLAKVLayouts[
            Self.KVLUTType.dtype,
            Self.KRopeType.dtype,
            None,
            Self.config,
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        # Blockscale uses its own TMA-to-CVT/CVT-to-MMA barriers for rope
        # synchronization.
        comptime BlockscaleSmemType = SM100AttentionSMem[Self.config.fa4_config]
        var mbars = BlockscaleSmemType().misc_mbars()

        comptime SMemTensorLT[elems: Int] = TileTensor[
            Self.KVLUTType.dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        comptime q_elems = type_of(q_tma_op).tile_shape[0] * type_of(
            q_tma_op
        ).tile_shape[1] * type_of(q_tma_op).tile_shape[2]
        comptime QType = SMemTensorLT[q_elems]

        var k_rope_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)
        var kv_head_idx: UInt32 = seq_info.head_idx

        comptime q_elements = (Self.config.BM // 2) * Self.config.BK0
        comptime q_bytes = size_of[Self.qkv_dtype]() * q_elements

        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )
        var q_head_idx: UInt32 = seq_info.head_idx
        var e = elect()

        # Sub-tile paging: when page_size < BN, each BN-row load is split
        # into num_kv_pages sub-tile loads of kv_sub_BN rows each.
        comptime kv_sub_BN = kv_sub_tile_rows(Self.config.BN, Self.page_size)
        comptime num_kv_pages = kv_num_sub_tiles(Self.config.BN, Self.page_size)
        comptime rope_sub_BN = kv_sub_tile_rows(
            Self.config.BN, Self.KRopeType.page_size
        )
        comptime num_rope_pages = kv_num_sub_tiles(
            Self.config.BN, Self.KRopeType.page_size
        )
        comptime PagedRows = PagedRowIndices[Self.config.BN, Self.page_size]
        comptime RopePagedRows = PagedRowIndices[
            Self.config.BN, Self.KRopeType.page_size
        ]

        # Alignment of `kv_row` produced by mask-driven iteration. The
        # mask's `start_column` uses `Self.page_size`, so the resulting
        # alignment is what we promise to both `kv_lut.populate`
        # (page_size `Self.page_size`) and `k_rope_lut.populate`
        # (page_size `Self.KRopeType.page_size`).
        comptime base_alignment: Int = Self.MaskType.start_column_alignment[
            Self.BM, Self.BN, Self.page_size
        ]()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](seq_info.prompt_idx, score_row)
        var paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
            seq_info.prompt_idx, kv_row
        )
        var rope_paged_rows = k_rope_lut.populate[
            Self.config.BN, base_alignment
        ](seq_info.prompt_idx, kv_row)
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                seq_info.prompt_idx, score_row, num_keys
            )
            - 1
        )

        # Partial-page handling: when page_size < BN, runtime-bound the
        # K_nope/V/K_rope sub-tile loops via the `needs_partial=True`
        # overloads (mirrors FA4 `load_warp.mojo` and the analogous
        # `mla_prefill_generic.mojo` fix). K_nope/V use `Self.page_size`;
        # K_rope uses `KRopeType.page_size`. The flags are computed
        # independently so the iter_count peel below correctly gates on
        # *either* page table needing a runtime bound.
        comptime needs_partial_kv = (
            Self.page_size > 0 and Self.page_size < Self.config.BN
        )
        comptime needs_partial_rope = (
            Self.KRopeType.page_size > 0
            and Self.KRopeType.page_size < Self.config.BN
        )
        comptime needs_partial = needs_partial_kv or needs_partial_rope

        # Per-sub-page byte sizes for partial expect_bytes_pred. K_rope
        # uses `KRopeType.dtype` (FP8 here), NOT `qkv_dtype` — the K_rope
        # bytes go on `tma_to_cvt_pipeline.producer_mbar()`, not the K
        # barrier, but the byte math is the same.
        comptime k_nope_bytes_pp = (
            Self.nope_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )
        comptime k_rope_bytes_pp = (
            Self.rope_depth * rope_sub_BN * size_of[Self.KRopeType.dtype]()
        )
        # V sub-page bytes use the V head dim (`ov_depth`), not `nope_depth`.
        comptime v_bytes_pp = (
            Self.ov_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )

        @__parameter
        @always_inline
        def _k_num_valid_pages(current_kv_row: UInt32) -> UInt32:
            """Valid K_nope/V sub-tile pages at `current_kv_row`."""
            if current_kv_row >= num_keys:
                return UInt32(0)
            return min(
                UInt32(num_kv_pages),
                UInt32(ceildiv(Int(num_keys - current_kv_row), Int(kv_sub_BN))),
            )

        @__parameter
        @always_inline
        def _rope_num_valid_pages(current_kv_row: UInt32) -> UInt32:
            """Valid K_rope sub-tile pages at `current_kv_row`."""
            if current_kv_row >= num_keys:
                return UInt32(0)
            return min(
                UInt32(num_rope_pages),
                UInt32(
                    ceildiv(Int(num_keys - current_kv_row), Int(rope_sub_BN))
                ),
            )

        # ---- Mode-shared sub-tile constants ----
        # The K_rope sub-tile shape is identical in shared-KV and non-shared-KV
        # mode (`rope_depth * rope_sub_BN` FP8 elements). Hoist the
        # constants and the SmemTensor type so the unified `_produce_k_rope`
        # closure below works for both modes — the only mode-specific
        # piece is the smem base pointer, which the caller threads in.
        comptime k_rope_sub_elems = Self.rope_depth * rope_sub_BN
        comptime KRopeSubType = TileTensor[
            Self.KRopeType.dtype,
            type_of(tt_row_major[k_rope_sub_elems]()),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        # Full-tile byte counts (when no partial bound applies).
        comptime k_rope_full_bytes = (
            Self.rope_depth * Self.config.BN * size_of[Self.KRopeType.dtype]()
        )
        # V full-tile bytes use the V head dim (`ov_depth`), not `nope_depth`.
        comptime kv_data_full_bytes = (
            Self.ov_depth * Self.config.BN * size_of[Self.qkv_dtype]()
        )

        @__parameter
        @always_inline
        def _produce_k_rope[
            partial: Bool,
        ](
            rope_pages: type_of(rope_paged_rows),
            kv_row_base: UInt32,
            smem_base_ptr: SharedMemPointer[Scalar[Self.KRopeType.dtype]],
            rope_nvp: UInt32,
        ):
            """K_rope sub-tile TMA into smem starting at `smem_base_ptr`,
            signaling completion on the CVT producer mbar.

            `smem_base_ptr` is the FP8 base of this tile's K_rope smem
            region, the caller pre-bitcasts (shared-KV) or pre-rebounds
            (non-shared-KV) so this closure can advance by `_p *
            k_rope_sub_elems` in FP8-element units. Folds in
            `expect_bytes_pred` for the CVT producer mbar so the byte
            count and the `_p`-loop stay in sync. `partial=True` early-
            returns when `_p == rope_nvp`.
            """
            var rope_bytes_local: Int32
            comptime if partial:
                rope_bytes_local = Int32(k_rope_bytes_pp) * Int32(rope_nvp)
            else:
                rope_bytes_local = Int32(k_rope_full_bytes)
            expect_bytes_pred(
                tma_to_cvt_pipeline.producer_mbar(),
                rope_bytes_local,
                e,
            )
            comptime for _p in range(num_rope_pages):
                comptime if partial:
                    if UInt32(_p) == rope_nvp:
                        return
                # Belt-and-suspenders: post-fix this should be
                # unreachable on every config. Kept as a permanent
                # red-test for the partial bound.
                debug_assert(
                    kv_row_base + UInt32(_p * rope_sub_BN) < num_keys,
                    (
                        "MLA blockscale K_rope sub-tile TMA OOB after"
                        " partial bound: kv_row_base="
                    ),
                    kv_row_base,
                    " _p=",
                    _p,
                    " rope_sub_BN=",
                    rope_sub_BN,
                    " num_keys=",
                    num_keys,
                    " rope_nvp=",
                    rope_nvp,
                    " partial=",
                    partial,
                )
                var k_rope_coord = kv_coord[depth=Self.rope_depth,](
                    rope_pages.get_row(UInt32(_p * rope_sub_BN)),
                    k_rope_head_idx,
                )
                k_rope_coord[0] = UInt32(Self.cache_depth - Self.rope_depth)
                k_rope_tma_op.async_copy_elect(
                    KRopeSubType(
                        smem_base_ptr + _p * k_rope_sub_elems,
                        tt_row_major[k_rope_sub_elems](),
                    ),
                    tma_to_cvt_pipeline.producer_mbar()[],
                    k_rope_coord,
                    e,
                )

        @__parameter
        @always_inline
        def _produce_v[
            partial: Bool,
        ](
            paged: type_of(paged_rows),
            mbar: SharedMemPointer[SharedMemBarrier],
            smem_ptr: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
            v_nvp: UInt32 = UInt32(num_kv_pages),
        ):
            """V tile production at `mbar` / `smem_ptr`. Both modes pass
            the V destination smem pointer directly so this closure
            doesn't need to know about pipeline machinery
            (kv_pipeline vs pipeline_v).

            `partial=True` runtime-bounds the sub-tile loop and accounts
            only the bytes actually delivered.
            """
            var v_bytes_local: Int32
            comptime if partial:
                v_bytes_local = Int32(v_bytes_pp) * Int32(v_nvp)
            else:
                v_bytes_local = Int32(kv_data_full_bytes)
            expect_bytes_pred(mbar, v_bytes_local, e)
            paged.tma_copy_v[needs_partial=partial](
                v_tma_op,
                smem_ptr,
                mbar[],
                kv_head_idx=kv_head_idx,
                elect=e,
                num_valid_pages=v_nvp,
            )

        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        comptime if Self.config.fa4_config.use_shared_kv:
            # ---- Shared KV mode ----
            # K_nope/V alternate in the same circular buffer; a stage fits the
            # wider of the two (shared_kv_cols). K_rope (FP8) goes into the
            # separate rope smem buffer via tma_to_cvt_pipeline.
            comptime kv_stage_elems = (
                Self.config.fa4_config.shared_kv_cols() * Self.config.BN
            )
            comptime rope_stage_elems = (
                Self.config.rope_depth * Self.config.BN
            )

            comptime KVPipeProdType = StagedPipeline[
                Self.config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeProdType = {mbars.get_k_mbars()}
            kv_pipeline.state._phase = 1  # producer starts at phase 1

            @__parameter
            @always_inline
            def _fused_rope_smem_ptr() -> (
                SharedMemPointer[Scalar[Self.KRopeType.dtype]]
            ):
                """FP8 base of the current rope-buffer slot.

                Bug-B fix: bitcast the BF16 buffer base to FP8 BEFORE
                handing it to `_produce_k_rope`, so the closure's
                `_p * k_rope_sub_elems` advance is in FP8-element units
                (== bytes). Adding it to a BF16 pointer would double
                the stride and leave a gap that breaks the CVT consumer.
                """
                return (
                    rope_smem_base
                    + tma_to_cvt_pipeline.state.index()
                    * UInt32(rope_stage_elems)
                ).bitcast[Scalar[Self.KRopeType.dtype]]()

            @__parameter
            @always_inline
            def _produce_k_fused[
                partial: Bool,
                with_q: Bool = False,
            ](
                paged: type_of(paged_rows),
                rope_paged: type_of(rope_paged_rows),
                kv_row_local: UInt32,
                mbar: type_of(kv_pipeline.producer_mbar()),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q (if `with_q`) + K_nope onto `mbar`, then K_rope
                onto `tma_to_cvt_pipeline.producer_mbar()` via the
                shared `_produce_k_rope` closure.

                The K barrier carries Q + K_nope bytes; K_rope bytes are
                accounted on the CVT producer mbar inside
                `_produce_k_rope` so the two pipelines stay in sync.
                """
                var qk_bytes: Int32 = Int32(q_bytes) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                else:
                    qk_bytes += Int32(kv_data_full_bytes)
                expect_bytes_pred(mbar, qk_bytes, e)

                comptime if with_q:
                    q_tma_op.async_copy_elect(
                        QType(q_smem, tt_row_major[q_elems]()),
                        mbar[],
                        q_coord[
                            depth=Self.qk_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )
                paged.tma_copy_k[needs_partial=partial](
                    k_nope_tma_op,
                    k_smem_base
                    + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                    mbar[],
                    kv_head_idx=kv_head_idx,
                    elect=e,
                    k_num_valid_pages=k_nvp,
                )
                _produce_k_rope[partial=partial](
                    rope_paged,
                    kv_row_local,
                    _fused_rope_smem_ptr(),
                    rope_nvp,
                )

            @__parameter
            @always_inline
            def _fused_v_smem_ptr() -> (
                SharedMemPointer[Scalar[Self.KVLUTType.dtype]]
            ):
                """V destination smem ptr at the current KV stage."""
                return k_smem_base + kv_pipeline.state.index() * UInt32(
                    kv_stage_elems
                )

            # ---- Peeled: K0 + Q0 on same KV barrier ----
            var k0_mbar = kv_pipeline.producer_mbar()
            var k_nvp_0 = _k_num_valid_pages(kv_row)
            var rope_nvp_0 = _rope_num_valid_pages(kv_row)
            _produce_k_fused[partial=needs_partial, with_q=True](
                paged_rows,
                rope_paged_rows,
                kv_row,
                k0_mbar,
                k_nvp_0,
                rope_nvp_0,
            )
            tma_to_cvt_pipeline.step()
            kv_pipeline.state.step()  # step -> stage 1

            # ---- Q1 (separate barrier) ----
            var q1_mbar = mbars.q1_wait_mbar()
            expect_bytes_pred(q1_mbar, Int32(q_bytes), e)
            # Q1 — elect-predicated in-PTX via `_elect`.
            q_tma_op.async_copy_elect(
                QType(q_smem + q_elements, tt_row_major[q_elems]()),
                q1_mbar[0],
                q_coord[
                    depth=Self.qk_depth,
                    decoding=False,
                ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
                e,
            )

            # ---- V0 (reuses paged_rows from K0) ----
            kv_pipeline.producer_acquire()
            var v0_mbar = kv_pipeline.producer_mbar()
            _produce_v[partial=needs_partial](
                paged_rows, v0_mbar, _fused_v_smem_ptr(), k_nvp_0
            )
            kv_pipeline.state.step()

            # ---- KV producer loop ----
            # Main body always issues full tiles (partial=False). When
            # needs_partial, peel off the last iteration so its
            # populate/TMAs can be runtime-bounded.
            var main_iters = iter_count
            comptime if needs_partial:
                if main_iters > 0:
                    main_iters -= 1
            while main_iters != 0:
                main_iters -= 1
                kv_row += UInt32(Self.config.BN)

                comptime if check_mask:
                    if (
                        Self.mask_status(
                            mask, seq_info.prompt_idx, score_row, kv_row
                        )
                        == TileMaskStatus.FULL_MASK
                    ):
                        continue
                paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
                    seq_info.prompt_idx, kv_row
                )
                rope_paged_rows = k_rope_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row)

                # Produce K_nope_n + K_rope_n (full sub-tile loops)
                kv_pipeline.producer_acquire()
                var kn_mbar = kv_pipeline.producer_mbar()
                _produce_k_fused[partial=False](
                    paged_rows, rope_paged_rows, kv_row, kn_mbar
                )
                tma_to_cvt_pipeline.step()
                kv_pipeline.state.step()

                # Produce Vn (reuses paged_rows)
                kv_pipeline.producer_acquire()
                var vn_mbar = kv_pipeline.producer_mbar()
                _produce_v[partial=False](
                    paged_rows, vn_mbar, _fused_v_smem_ptr()
                )
                kv_pipeline.state.step()

            # ---- Peeled last iteration (partial-page bound) ----
            comptime if needs_partial:
                if iter_count > 0:
                    kv_row += UInt32(Self.config.BN)
                    var _skip_last = False
                    comptime if check_mask:
                        if (
                            Self.mask_status(
                                mask, seq_info.prompt_idx, score_row, kv_row
                            )
                            == TileMaskStatus.FULL_MASK
                        ):
                            _skip_last = True
                    if not _skip_last:
                        # Re-populate BOTH LUTs at the new kv_row.
                        paged_rows = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        rope_paged_rows = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        var k_nvp_last = _k_num_valid_pages(kv_row)
                        var rope_nvp_last = _rope_num_valid_pages(kv_row)
                        # Kn (partial) + K_rope_n (partial)
                        kv_pipeline.producer_acquire()
                        var kn_mbar_last = kv_pipeline.producer_mbar()
                        _produce_k_fused[partial=needs_partial](
                            paged_rows,
                            rope_paged_rows,
                            kv_row,
                            kn_mbar_last,
                            k_nvp_last,
                            rope_nvp_last,
                        )
                        tma_to_cvt_pipeline.step()
                        kv_pipeline.state.step()
                        # Vn (partial)
                        kv_pipeline.producer_acquire()
                        var vn_mbar_last = kv_pipeline.producer_mbar()
                        _produce_v[partial=needs_partial](
                            paged_rows,
                            vn_mbar_last,
                            _fused_v_smem_ptr(),
                            k_nvp_last,
                        )
                        kv_pipeline.state.step()

        else:
            # ---- Non-shared mode (original) ----

            # Separate K and V pipelines
            comptime VPipeType = VProducerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            var k_pipeline = StagedPipeline[
                Self.config.num_kv_stages, Self.config.num_qk_stages
            ](mbars.get_k_mbars())
            k_pipeline.state._phase = 1
            var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem_base}

            # K stage may contain mixed dtypes (e.g. FP8 nope + BF16 rope).
            # Compute byte size then convert to qkv_dtype element count. The
            # K_nope part is `padded_nope_depth` wide (non-shared-KV: V has its own
            # `pipeline_v`), so this is K-only.
            comptime k_stage_bytes = (
                Self.config.fa4_config.padded_nope_depth
                * Self.config.BN
                * Self.qkv_dt_size
                + Self.config.rope_depth
                * Self.config.BN
                * Self.config.rope_mma_dtype_size
            )
            comptime k_elements_per_stage = k_stage_bytes // Self.qkv_dt_size

            # Get K0 barrier (no wait needed for first iteration)
            var k0_mbar = k_pipeline.producer_mbar[qk_stage=0]()

            @__parameter
            @always_inline
            def _produce_k_split[
                partial: Bool,
                with_q: Bool = False,
            ](
                paged: type_of(paged_rows),
                rope_paged: type_of(rope_paged_rows),
                kv_row_local: UInt32,
                mbar: type_of(k0_mbar),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q (if `with_q`) + K_nope onto `mbar`, then K_rope
                onto `tma_to_cvt_pipeline.producer_mbar()` via the
                shared `_produce_k_rope` closure.

                Includes the `split_smem` decomposition into K_nope and
                K_rope smem regions. Note: unlike generic non-shared-KV,
                blockscale's K_rope bytes are NOT on the K barrier, so
                the K-barrier expect only carries Q + K_nope.
                """
                # Q + K_nope bytes go on the K barrier.
                var qk_bytes: Int32 = Int32(q_bytes) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                else:
                    qk_bytes += Int32(kv_data_full_bytes)
                expect_bytes_pred(mbar, qk_bytes, e)

                comptime if with_q:
                    q_tma_op.async_copy_elect(
                        QType(q_smem, tt_row_major[q_elems]()),
                        mbar[],
                        q_coord[
                            depth=Self.qk_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )
                var smem_ptr = k_smem_base + k_pipeline.state.index() * UInt32(
                    k_elements_per_stage
                )
                var k_nope_smem_local, k_rope_smem_local = split_smem[
                    KVPipeType.k_nope_tma_layout,
                    KVPipeType.k_rope_tma_layout,
                    Self.KVLUTType.dtype,
                    Self.KRopeType.dtype,
                ](
                    SMemTensorLT[KVPipeType.k_tma_layout](
                        smem_ptr, tt_row_major[KVPipeType.k_tma_layout]()
                    )
                )
                paged.tma_copy_k[needs_partial=partial](
                    k_nope_tma_op,
                    rebind[SharedMemPointer[Scalar[Self.KVLUTType.dtype]]](
                        k_nope_smem_local.ptr
                    ),
                    mbar[],
                    kv_head_idx=kv_head_idx,
                    elect=e,
                    k_num_valid_pages=k_nvp,
                )
                _produce_k_rope[partial=partial](
                    rope_paged,
                    kv_row_local,
                    rebind[SharedMemPointer[Scalar[Self.KRopeType.dtype]]](
                        k_rope_smem_local.ptr
                    ),
                    rope_nvp,
                )

            @__parameter
            @always_inline
            def _split_v_smem_ptr(
                pair: type_of(pipeline_v.get_tile[qk_stage=0]()),
            ) -> SharedMemPointer[Scalar[Self.KVLUTType.dtype]]:
                """V destination smem ptr for non-shared-KV's V pipeline pair.

                Note we switched from `pipeline_v.get_v(e)` (which auto-
                emits a fixed-size `expect_bytes`) to
                `pipeline_v.get_tile[qk_stage=0]()` so the unified
                `_produce_v` closure can emit a partial-aware
                `expect_bytes_pred` itself.
                """
                return rebind[SharedMemPointer[Scalar[Self.KVLUTType.dtype]]](
                    pair.smem.ptr
                )

            # ---- K0 + Q0 (combined K barrier; K_rope on CVT mbar) ----
            var k_nvp_0 = _k_num_valid_pages(kv_row)
            var rope_nvp_0 = _rope_num_valid_pages(kv_row)
            _produce_k_split[partial=needs_partial, with_q=True](
                paged_rows,
                rope_paged_rows,
                kv_row,
                k0_mbar,
                k_nvp_0,
                rope_nvp_0,
            )
            tma_to_cvt_pipeline.step()
            k_pipeline.state.step()

            # ---- Q1 (separate barrier) ----
            var q1_mbar = mbars.q1_wait_mbar()
            expect_bytes_pred(q1_mbar, Int32(q_bytes), e)
            # Q1 — elect-predicated in-PTX via `_elect`.
            q_tma_op.async_copy_elect(
                QType(q_smem + q_elements, tt_row_major[q_elems]()),
                q1_mbar[0],
                q_coord[
                    depth=Self.qk_depth,
                    decoding=False,
                ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
                e,
            )

            # ---- V0 (reuses paged_rows from K0) ----
            var mbarv0 = pipeline_v.get_tile[qk_stage=0]()
            _produce_v[partial=needs_partial](
                paged_rows, mbarv0.mbar, _split_v_smem_ptr(mbarv0), k_nvp_0
            )
            pipeline_v.commit_step()

            # ---- KV producer loop ----
            # Main body: always full tiles (partial=False). When
            # needs_partial, peel off the last iteration so its
            # populate/TMAs can be runtime-bounded.
            var main_iters = iter_count
            comptime if needs_partial:
                if main_iters > 0:
                    main_iters -= 1
            while main_iters != 0:
                main_iters -= 1
                kv_row += UInt32(Self.config.BN)

                comptime if check_mask:
                    if (
                        Self.mask_status(
                            mask, seq_info.prompt_idx, score_row, kv_row
                        )
                        == TileMaskStatus.FULL_MASK
                    ):
                        continue
                paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
                    seq_info.prompt_idx, kv_row
                )
                rope_paged_rows = k_rope_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row)
                # produce k (full sub-tile loops for paged KV)
                k_pipeline.producer_acquire[qk_stage=0]()
                var kn_mbar = k_pipeline.producer_mbar[qk_stage=0]()
                _produce_k_split[partial=False](
                    paged_rows, rope_paged_rows, kv_row, kn_mbar
                )
                tma_to_cvt_pipeline.step()
                k_pipeline.state.step()
                # produce v (reuses paged_rows)
                pipeline_v.acquire_v()
                var mbarvn = pipeline_v.get_tile[qk_stage=0]()
                _produce_v[partial=False](
                    paged_rows, mbarvn.mbar, _split_v_smem_ptr(mbarvn)
                )
                pipeline_v.commit_step()

            # ---- Peeled last iteration (partial-page bound) ----
            comptime if needs_partial:
                if iter_count > 0:
                    kv_row += UInt32(Self.config.BN)
                    var _skip_last = False
                    comptime if check_mask:
                        if (
                            Self.mask_status(
                                mask, seq_info.prompt_idx, score_row, kv_row
                            )
                            == TileMaskStatus.FULL_MASK
                        ):
                            _skip_last = True
                    if not _skip_last:
                        # Re-populate BOTH LUTs at the new kv_row.
                        paged_rows = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        rope_paged_rows = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        var k_nvp_last = _k_num_valid_pages(kv_row)
                        var rope_nvp_last = _rope_num_valid_pages(kv_row)
                        # produce k (partial)
                        k_pipeline.producer_acquire[qk_stage=0]()
                        var kn_mbar_last = k_pipeline.producer_mbar[
                            qk_stage=0
                        ]()
                        _produce_k_split[partial=needs_partial](
                            paged_rows,
                            rope_paged_rows,
                            kv_row,
                            kn_mbar_last,
                            k_nvp_last,
                            rope_nvp_last,
                        )
                        tma_to_cvt_pipeline.step()
                        k_pipeline.state.step()
                        # produce v (partial)
                        pipeline_v.acquire_v()
                        var mbarvn_last = pipeline_v.get_tile[qk_stage=0]()
                        _produce_v[partial=needs_partial](
                            paged_rows,
                            mbarvn_last.mbar,
                            _split_v_smem_ptr(mbarvn_last),
                            k_nvp_last,
                        )
                        pipeline_v.commit_step()

    @staticmethod
    @always_inline
    def convert_fp8_to_bf16(
        mut iter_count: UInt32,
        mut tma_to_cvt_pipeline: TMAtoCvtPipeline,
        mut cvt_to_mma_pipeline: CvtToMMAPipeline,
        k_rope: Self.KRopeType,
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        seq_info: SeqInfo,
        num_keys: UInt32,
        local_thread_idx: UInt32,
    ):
        var local_warp_idx, local_lane_idx = divmod(
            local_thread_idx, UInt32(WARP_SIZE)
        )

        var kv_start_tok: UInt32 = 0

        comptime swizzle_fp8 = make_swizzle[
            Self.KRopeType.dtype, TensorMapSwizzle.SWIZZLE_64B
        ]()
        comptime swizzle_bf16 = make_swizzle[
            Self.KVLUTType.dtype, TensorMapSwizzle.SWIZZLE_128B
        ]()

        # In non-shared mode, k_rope sits after k_nope within each K stage.
        # In shared mode, k_rope is in a separate rope smem buffer.
        comptime k_stage_stride = Self.config.padded_qk_depth * Self.config.BN
        comptime k_rope_offset = Self.config.BN * Self.nope_depth
        comptime rope_stage_elems = Self.config.rope_depth * Self.config.BN

        while True:
            tma_to_cvt_pipeline.consumer_wait()

            # In shared mode, k_rope is in a separate rope smem buffer.
            # In non-shared mode, k_rope sits after k_nope within each K stage.
            var k_rope_smem_ptr: SharedMemPointer[Scalar[Self.KVLUTType.dtype]]
            comptime if Self.config.fa4_config.use_shared_kv:
                k_rope_smem_ptr = (
                    rope_smem_base
                    + tma_to_cvt_pipeline.state.index()
                    * UInt32(rope_stage_elems)
                )
            else:
                k_rope_smem_ptr = (
                    k_smem_base
                    + tma_to_cvt_pipeline.state.index() * UInt32(k_stage_stride)
                    + k_rope_offset
                )

            comptime tile_rows = Self.config.BN // 2
            comptime tile_layout = tt_row_major[tile_rows, Self.rope_depth]()
            comptime SmemFP8 = TileTensor[
                Self.KRopeType.dtype,
                type_of(tile_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ]
            comptime SmemBF16 = TileTensor[
                Self.KVLUTType.dtype,
                type_of(tile_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ]
            var tile_offset = Int(local_warp_idx) * tile_rows * Self.rope_depth
            var fp8_base = k_rope_smem_ptr.bitcast[
                Scalar[Self.KRopeType.dtype]
            ]()
            var bf16_base = k_rope_smem_ptr.bitcast[
                Scalar[Self.KVLUTType.dtype]
            ]()
            var k_rope_tile_fp8 = SmemFP8(fp8_base + tile_offset, tile_layout)
            var k_rope_tile_bf16 = SmemBF16(
                bf16_base + tile_offset, tile_layout
            )

            cvt_block_fp8_to_bf16_with_scale[
                swizzle_fp8=swizzle_fp8,
                swizzle_bf16=swizzle_bf16,
            ](
                k_rope_tile_fp8,
                k_rope_tile_bf16,
                k_rope,
                seq_info,
                kv_start_tok + UInt32(Self.BN // 2) * local_warp_idx,
                num_keys,
                local_lane_idx,
            )

            tma_to_cvt_pipeline.step()
            cvt_to_mma_pipeline.producer_commit()

            if iter_count == 0:
                break
            iter_count -= 1
            kv_start_tok += UInt32(Self.BN)

    @staticmethod
    @always_inline
    def mma(
        tmem_addr: UInt32,
        mut cvt_to_mma_pipeline: CvtToMMAPipeline,
        seq_id: UInt32,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        # Construct SmemType to get MiscMBars (blockscale uses its own CVT
        # barriers for rope synchronization).
        comptime BlockscaleSmemType = SM100AttentionSMem[Self.config.fa4_config]
        var mbars = BlockscaleSmemType().misc_mbars()

        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var s1_tmem = tmem_addr + UInt32(Self.config.TMEM_S1)
        var o0_tmem = tmem_addr + UInt32(Self.config.TMEM_O0)
        var o1_tmem = tmem_addr + UInt32(Self.config.TMEM_O1)

        # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
        var pipeline_s0 = mbars.producer_s0()
        var pipeline_s1 = mbars.producer_s1()
        # Keep consumer pointers for acquire operations (shared phase tracking)
        var consumer_s0 = pipeline_s0.consumer_mbar_base
        var consumer_s1 = pipeline_s1.consumer_mbar_base

        # O pipelines (producer side only; consumer wait is merged into S barriers)
        var pipeline_o0 = mbars.producer_o0()
        var pipeline_o1 = mbars.producer_o1()

        comptime q0_size = (Self.config.BM // 2) * Self.config.padded_qk_depth
        comptime q0_bytes = UInt32(q0_size * size_of[Self.KVLUTType.dtype]())
        var q0 = Self.descriptor_q(q_smem)
        var q1 = q0 + q0_bytes

        comptime if Self.config.fa4_config.use_shared_kv:
            # ---- Shared KV mode ----
            # K_nope/V alternate in the same buffer; a stage fits the wider of
            # the two (shared_kv_cols). K_rope (BF16 after CVT) is in a separate
            # rope buffer.
            # Q@K' = Q_nope@K_nope (c_scale=0) + Q_rope@K_rope (c_scale=1).

            comptime kv_stage_bytes = (
                Self.config.fa4_config.shared_kv_cols()
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )
            comptime rope_stage_bytes = (
                Self.config.rope_depth
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )

            # K_nope descriptor: k_major for Q@K_nope'. Width is the K_nope
            # depth (`padded_nope_depth`), the Q@K' contraction dim.
            var kv_desc_k = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.fa4_config.padded_nope_depth,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=True,
            ](k_smem_base)
            # V descriptor: mn_major for P@V. Width is the V head dim
            # (`padded_ov_depth`).
            var kv_desc_v = smem_descriptor[
                BMN=Self.config.fa4_config.padded_ov_depth,
                BK=Self.config.BN,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=False,
            ](k_smem_base)
            # K_rope descriptor: k_major for Q@K_rope' (BF16 after CVT)
            var rope_desc = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.rope_depth,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=True,
            ](rope_smem_base)

            # Q_rope offset: Q_nope occupies first MMA_M * padded_v_depth
            # elements in Q's column-major tiled layout.
            comptime q_rope_off = UInt32(Self.q_rope_byte_offset)
            var q0_rope = q0 + q_rope_off
            var q1_rope = q1 + q_rope_off

            comptime KVPipeType = StagedPipeline[
                Self.config.fa4_config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

            var iter_count: UInt32 = (
                mask.total_iters[Self.BM, Self.BN, Self.page_size](
                    seq_id, score_row, num_keys
                )
                - 1
            )

            var e = elect()

            # Rope stage counter: tracks which cvt_to_mma stage contains the
            # current K_rope BF16 data. Cycles 0,1,...,num_rope_buffers-1.
            var rope_stage: UInt32 = 0
            comptime num_rope_stages = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )

            # ---- Peeled iteration ----
            # Stage 0 = K0 (K_nope0 + K_rope0 via CVT)
            kv_pipeline.consumer_wait()
            cvt_to_mma_pipeline.consumer_wait()
            var k0 = (
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            )
            Self.UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
            # K_rope0: from CVT pipeline at rope_stage=0
            var r0 = rope_desc + UInt32(rope_stage_bytes) * rope_stage
            Self.UMMA0RopeType.mma[stage_idx=0](
                q0_rope, r0, s0_tmem, elect=e, c_scale=1
            )
            pipeline_s0.commit_mma(e)

            # Q1 @ K0 (wait for Q1 first)
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].wait()
            Self.UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)
            Self.UMMA0RopeType.mma[stage_idx=0](
                q1_rope, r0, s1_tmem, elect=e, c_scale=1
            )
            kv_pipeline.consumer_release(e)  # release K0, step → stage 1
            cvt_to_mma_pipeline.consumer_release(e)
            rope_stage = (rope_stage + 1) % num_rope_stages
            pipeline_s1.commit_mma(e)

            # Stage 1 = V0
            kv_pipeline.consumer_wait()
            var v_prev_idx: UInt32 = kv_pipeline.state.index()
            var v0 = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s0[pv_stage].wait(0)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, v0, o0_tmem, elect=e, c_scale=0
                )
            pipeline_o0.commit_mma(e)
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # ---- Main loop ----
            while iter_count != 0:
                iter_count -= 1

                # Advance past held V to get to next K
                kv_pipeline.state.step()

                # Kn (K_nope_n via KV pipeline + K_rope_n via CVT pipeline)
                kv_pipeline.consumer_wait()
                cvt_to_mma_pipeline.consumer_wait()
                var kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                Self.UMMA0Type.mma[stage_idx=0](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
                var rn = rope_desc + UInt32(rope_stage_bytes) * rope_stage
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, rn, s0_tmem, elect=e, c_scale=1
                )
                pipeline_s0.commit_mma(e)

                # P1 @ V_{n-1}
                var v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
                c_scale = 1
                kv_pipeline.consumer_release_at(
                    v_prev_idx, e
                )  # release V_{n-1}

                # Q1 @ Kn
                Self.UMMA0Type.mma[stage_idx=0](
                    q1, kn, s1_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q1_rope, rn, s1_tmem, elect=e, c_scale=1
                )
                kv_pipeline.consumer_release(e)  # release Kn, step
                cvt_to_mma_pipeline.consumer_release(e)
                rope_stage = (rope_stage + 1) % num_rope_stages
                pipeline_s1.commit_mma(e)
                phase ^= 1

                # Vn
                kv_pipeline.consumer_wait()
                v_prev_idx = kv_pipeline.state.index()
                var vn = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vn, o0_tmem, elect=e, c_scale=1
                    )
                pipeline_o0.commit_mma(e)

            # ---- Epilogue ----
            var v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)
            kv_pipeline.consumer_release_at(v_prev_idx, e)  # release V_last

        else:
            # ---- Non-shared mode (original) ----

            # Separate K and V consumer pipelines
            comptime KConType = KConsumerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            comptime VConType = VConsumerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            var pipeline_k: KConType = {mbars.get_k_mbars(), k_smem_base}
            var pipeline_v: VConType = {mbars.get_v_mbars(), v_smem_base}

            # We peel the first iteration, as we want to wait on q1
            var iter_count: UInt32 = (
                mask.total_iters[Self.BM, Self.BN, Self.page_size](
                    seq_id, score_row, num_keys
                )
                - 1
            )

            # Q_0 @ K_0'
            # wait for CVT for fp8 case, else wait for producer
            pipeline_k.wait_k()
            cvt_to_mma_pipeline.consumer_wait()

            var k0 = pipeline_k.get_k()
            var e = elect()

            Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
            pipeline_s0.commit_mma(e)

            # Q_1 @ K_0'
            mbars.q1_wait_mbar()[0].wait()  # wait on Q1
            Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
            pipeline_s1.commit_mma(e)

            pipeline_k.release_k(e)  # release K0
            cvt_to_mma_pipeline.step()

            # Wait V0
            pipeline_v.wait_v()
            var vlatest = pipeline_v.get_v()
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s0[pv_stage].wait(0)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0
                )
            pipeline_o0.commit_mma(e)
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            while iter_count != 0:
                iter_count -= 1

                # Q_0 @ K_n'
                var kn = pipeline_k.get_k()
                pipeline_k.wait_k()
                cvt_to_mma_pipeline.consumer_wait()

                Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
                pipeline_s0.commit_mma(e)

                # O_1 + P_1 @ V_{n-1}
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
                c_scale = 1
                pipeline_v.release_v(e)

                # Q_1 @ K_n'
                Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
                pipeline_k.release_k(e)
                pipeline_s1.commit_mma(e)
                phase ^= 1

                cvt_to_mma_pipeline.step()

                # O_0 + P_0 @ V_n
                vlatest = pipeline_v.get_v()
                pipeline_v.wait_v()
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                    )
                pipeline_o0.commit_mma(e)

            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)


@always_inline
def mla_sm100_prefill_blockscale[
    output_dtype: DType,
    q_type: DType,
    KVType: MHAOperand,
    VType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
    v_depth: Int = -1,
](
    output: TileTensor[output_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: KVType,
    v: VType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    """Launches the SM100 MLA prefill kernel with blockwise FP8 scaling.

    Selects an FA4 MLA config, builds the TMA tiles for Q, K_nope, K_rope, and
    V, then enqueues the warp-specialized `mla_prefill_kernel_blockscale` kernel
    that converts block-scaled FP8 K_rope to BF16 before the MMA stage.

    Parameters:
        output_dtype: The element dtype of the output tensor.
        q_type: The element dtype of the query tensor.
        KVType: The `MHAOperand` type for K_nope, carrying gmem layout
            and page-size metadata.
        VType: The `MHAOperand` type for V, carrying gmem layout and
            page-size metadata.
        KRopeType: The `MHAOperand` type for K_rope, carrying gmem
            layout, page-size, and dtype (FP8 for blockscale).
        MaskType: The `MHAMask` type controlling causal and padding
            behavior.
        MaxPromptLenType: The `OptionallyStaticInt` type of
            `max_prompt_len`, either a compile-time constant or a
            runtime integer.
        config: The `MHAConfig` holding tile sizes, stage counts, and
            head counts (inferred).
        group: The GQA group size, namely the number of query heads per
            KV head (inferred).
        q_depth: The total query head dimension, combining nope and rope
            depth (inferred).
        cache_depth: The total KV cache depth, combining nope and rope
            depth (inferred).
        _ndbuffer_mha_operand: Whether the MHA operand uses the ndbuffer
            layout (inferred).
        blockwise_scale: The blockwise FP8 scaling mode flag (defaults
            to 0, inferred).
        v_depth: The V and output head dimension, or -1 for the DeepSeek
            shape where V width equals nope width (defaults to -1,
            inferred).

    Args:
        output: The output tensor of shape
            `[batch * num_rows, num_q_heads, ov_depth]`.
        q: The query tensor of shape
            `[batch * num_rows, num_q_heads, qk_depth]`.
        k: The K_nope operand for the non-rotary part of the KV cache.
        v: The V operand for the value part of the KV cache.
        k_rope: The K_rope operand for the rotary-position-encoded part
            of K, stored as block-scaled FP8.
        mask_functor: The attention mask functor applied to the score
            matrix.
        valid_length: The per-sequence valid length tensor of shape
            `[batch]` as `uint32`.
        max_prompt_len: The maximum prompt length across the batch, as
            a compile-time or runtime integer.
        scale: The softmax scale factor applied to the Q@K' dot product.
        batch_size: The number of sequences in the batch.
        ctx: The `DeviceContext` used to enqueue the kernel.
    """
    comptime assert (
        KVType.dtype == VType.dtype
    ), "k and v must share an element dtype for SM100 MLA prefill"
    # Select the supported config: the standard 2-O config first (byte-identical
    # to the pre-decoupling path when v_head_dim == qk_nope_head_dim), else a
    # single-O fallback for a wide V. `v_depth` (V/output head dim) is `-1` for
    # the DeepSeek shape (V width == nope width). Shared with the generic /
    # per-token-scale kernels so the fallback policy lives in one place.
    comptime fa4_config = select_mla_prefill_config[
        q_type,
        rope_gmem_dtype=KRopeType.dtype,
        rope_mma_dtype=q_type,
    ](
        num_q_heads=config.num_heads,
        group=group,
        depth=q_depth,
        page_size=KVType.page_size,
        v_depth=v_depth,
    )
    comptime assert fa4_config.supported()
    # V / output head dim (= v_head_dim).
    comptime ov_depth = fa4_config.fa4_config.ov_depth

    var num_rows_q = q_num_matrix_view_rows(q)

    # Batched O store: half-depth box (depth_splits=2, shared with the 1Q
    # combine) for SWIZZLE_NONE group==1; 0 (per-block) otherwise.
    comptime store_blocks_per_op = o_store_tma_blocks_per_op[
        output_dtype,
        fa4_config.output_swizzle_mode,
        ov_depth,
        1,
        depth_splits=2,
    ]()
    comptime RaggedStoreType = RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // fa4_config.fa4_config.num_q,
        BN=ov_depth,
        middle_dim=fa4_config.num_q_heads,
        tma_blocks_per_op=store_blocks_per_op,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q
    )

    var q_tma_op = q_tma[
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.qk_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
    ](
        ctx,
        q.ptr,
        num_rows_q,
    )

    # [batch_size * num_keys, num_heads, kv_depth]
    var k_nope_tma_op = k.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        depth=fa4_config.nope_depth,
    ](ctx)

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    var k_rope_tma_op = k_rope.create_tma_tile[
        fa4_config.rope_gmem_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KRopeType.page_size),
        depth=cache_depth,
        BK=fa4_config.rope_depth,
    ](ctx)

    # [batch_size * num_keys, num_heads, v_depth] — V gmem width is ov_depth.
    var v_tma_op = v.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        depth=ov_depth,
    ](ctx)

    # Rebind V to the dispatch's V tile type (distinct from k_nope when
    # ov_depth != nope_depth).
    _mla_prefill_sm100_valid_length_dispatch[
        fa4_config=fa4_config,
        cache_depth=cache_depth,
        _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        blockwise_scale=blockwise_scale,
    ](
        ragged_tma_store,
        q_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        rebind[
            KVTMATile[
                KVType.dtype,
                fa4_config.qkv_swizzle_mode,
                BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
                BK=padded_depth[
                    KVType.dtype, fa4_config.qkv_swizzle_mode, ov_depth
                ](),
            ]
        ](v_tma_op),
        k,
        k_rope,
        mask_functor,
        valid_length,
        max_prompt_len,
        scale,
        batch_size,
        ctx,
    )


@always_inline
def _mla_prefill_sm100_valid_length_dispatch[
    KVType: MHAOperand,
    output_dtype: DType,
    q_type: DType,
    MaskType: MHAMask,
    KRopeType: MHAOperand,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    fa4_config: MLAConfig,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
](
    ragged_tma_store: RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // fa4_config.fa4_config.num_q,
        BN=fa4_config.fa4_config.ov_depth,
        # Inferred from the created store; forwarded to the kernel impl.
        middle_dim=_,
        tma_blocks_per_op=_,
    ],
    q_tma_op: QTMATile[
        q_type,
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.qk_depth,
        group=fa4_config.group,
        decoding=False,
    ],
    k_nope_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        BK=padded_depth[
            KVType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    k_rope_tma_op: KVTMATile[
        KRopeType.dtype,
        fa4_config.rope_gmem_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KRopeType.page_size),
        BK=fa4_config.rope_depth,
    ],
    v_tma_op: KVTMATile[  # V tile: ov_depth-wide
        KVType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        BK=padded_depth[
            KVType.dtype,
            fa4_config.qkv_swizzle_mode,
            fa4_config.fa4_config.ov_depth,
        ](),
    ],
    kv_lut: KVType,
    k_rope_lut: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime SchedulerType = TransientScheduler[
        UInt32(fa4_config.BM),
        UInt32(fa4_config.num_q_heads),
        flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
    ]
    comptime ValidLengthType = NonNullPointer[.uint32]
    comptime SinkType = NullPointer[output_dtype]
    comptime KVRowOffsetsType = NullPointer[.uint32]
    comptime PartitionType = NoPartition[.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    comptime SM100MLAType = SM100MLA[
        KVType,
        KRopeType,
        output_dtype,
        MaskType,
        SchedulerType,
        fa4_config,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
        _ndbuffer_mha_operand,
    ]

    comptime kernel = SM100MLAType.mla_prefill_kernel_blockscale[
        blockwise_scale
    ]

    comptime PackType = Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
    ]

    var pack: PackType = {
        mask_functor,
        SchedulerType(),
        valid_len,
        SinkType(),
        KVRowOffsetsType(),
        max_prompt_len,
        PartitionType(),
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_prompt_len.as_uint32(), UInt32(fa4_config.BM)
    )
    var num_blocks: UInt32 = (
        max_num_prompt_tiles * PartitionType().num_partitions()
    )

    comptime num_threads = fa4_config.num_threads
    comptime SmemType = SM100AttentionSMem[fa4_config.fa4_config]
    comptime assert fa4_config.supported(), fa4_config.fa4_config.description()
    # Extra barriers for blockscale: tma_to_cvt (2*num_rope_bufs) + cvt_to_mma (4)
    comptime extra_barrier_count = 2 * fa4_config.fa4_config.num_rope_buffers() + 4
    comptime smem_use = align_up(
        SmemType.smem_size(), size_of[SharedMemBarrier]()
    ) + extra_barrier_count * size_of[SharedMemBarrier]()
    comptime assert SmemType.smem_size() == fa4_config.smem_used, String(
        "SmemType.smem_size() = ",
        SmemType.smem_size(),
        "\nfa4_config.smem_used = ",
        fa4_config.smem_used,
        "\n",
        fa4_config.fa4_config.description(),
    )
    comptime assert smem_use <= fa4_config.sm100_smem_carveout

    ctx.enqueue_function[kernel](
        q_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        v_tma_op,
        ragged_tma_store,
        kv_lut,
        k_rope_lut,
        scale,
        UInt32(batch_size),
        pack,
        grid_dim=SchedulerType.grid_dim(UInt32(batch_size), num_blocks),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )
