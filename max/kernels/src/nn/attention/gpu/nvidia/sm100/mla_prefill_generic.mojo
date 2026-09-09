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

from std.sys import size_of, get_defined_bool
from std.math import ceildiv
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    expect_bytes_pred,
    KConsumerPipeline,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    o_store_tma_blocks_per_op,
    PagedRowIndices,
    SharedMemPointer,
    StagedPipeline,
    VConsumerPipeline,
    VProducerPipeline,
)
from nn.attention.gpu.mha import q_num_matrix_view_rows
from nn.attention.gpu.nvidia.common import (
    get_seq_info,
    kv_coord,
    KVTMATile,
    NonNullPointer,
    NullPointer,
    Pack,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import RaggedTMA3DTile, SharedMemBarrier
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from max.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, thread_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.gpu.primitives.warp import broadcast
from nn.attention.mha_utils import (
    _is_decoding,
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
)
from max.gpu.compute.arch.tcgen05 import *
from linalg.arch.sm100.mma import smem_descriptor
from std.utils.static_tuple import StaticTuple
from kv_cache.types import padded_depth

from nn.attention.gpu.nvidia.sm100.mla_prefill_utils import (
    MLAConfig,
    MLAKVLayouts,
    MLAPositionSummary,
    SM100MLA,
    select_mla_prefill_config,
    split_smem,
)
from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.nvidia.sm100.smem import SM100AttentionSMem


# Programmatic Dependent Launch level for the SM100 MLA (depth-576) prefill
# kernel.  On by default so back-to-back attention grids in a stream overlap
# launch/prologue latency; disable with `-D MLA_PREFILL_PDL=false` (e.g. for
# A/B benchmarking).  When > OFF the kernel emits `wait_on_dependent_grids()` /
# `launch_dependent_grids()` and the dispatch attaches the
# PROGRAMMATIC_STREAM_SERIALIZATION launch attribute.  Separate from the MHA
# prefill flag (`MHA_PDL`) so the two kernels can be toggled independently.
comptime MLA_PREFILL_PDL_LEVEL = PDLLevel.OVERLAP_AT_END if get_defined_bool[
    "MLA_PREFILL_PDL", True
]() else PDLLevel.OFF


@fieldwise_init
struct WarpRole(Equatable, TrivialRegisterPassable):
    var _role: Int32
    comptime Softmax0 = Self(0)
    comptime Softmax1 = Self(1)
    comptime Correction = Self(2)
    comptime MMA = Self(3)
    comptime Load = Self(4)
    comptime Empty = Self(5)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self == Self(Int32(other))


def warp_idx_to_role[
    single_softmax_wg: Bool = False
](warp_idx: UInt32) -> WarpRole:
    var wg_idx = warp_idx // 4
    comptime if single_softmax_wg:
        # Generic single-O (wide-V): WG1's softmax is a full no-op, so it is
        # dropped and the roles pack into 3 warpgroups (384 threads). WG0 keeps
        # Softmax0; Correction moves down from WG2 into WG1; the MMA/Load warps
        # move down from WG3 (warps 12/13) into WG2 (warps 8/9). Correction /
        # MMA / Load bodies are position-independent (local row via
        # `thread_idx.x % WARPGROUP_SIZE`; MMA/Load `elect()` not absolute
        # warp index), so the remap is transparent.
        if wg_idx == 0:
            return WarpRole.Softmax0
        elif wg_idx == 1:
            return WarpRole.Correction
        elif warp_idx == 8:
            return WarpRole.MMA
        elif warp_idx == 9:
            return WarpRole.Load
        else:
            return WarpRole.Empty
    else:
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
            Int32(Self.config.launch_num_threads())
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"sm100_mla_prefill_generic_{Self.qkv_dtype}_{Self.output_dtype}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def mla_prefill_kernel_generic(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            # `BM // num_q` = 128 in both modes (one of two Q halves in
            # 2Q, the single full-BM Q tile in 1Q), so the TMA-op type
            # folds across the 1Q/2Q configs.
            BM=Self.config.q_tile_rows(),
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
            # 1Q/2Q signature; numerically `// 2` for the num_q=2 MLA
            # path.
            BM=Self.config.fa4_config.BM // Self.config.fa4_config.num_q,
            BN=Self.config.fa4_config.ov_depth,
            middle_dim=Self.config.num_q_heads,
            group=config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
            # Concrete value: a GPU-kernel entry (and the device impl it calls)
            # must be a fully-bound function. Matches the created store.
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
        # Thin entrypoint. Compute this tile's `SeqInfo` once and forward it
        # to `_kernel_impl` (every warp group shares it — it is derived from
        # blockIdx, not threadIdx, so it is identical on every lane/warp).
        #
        # When this 2Q config admits a type-compatible 1Q variant
        # (`can_switch_to_1q()`) AND the mask needs no runtime FULL_MASK slow
        # path (`not check_mask`), route per-tile work whose remaining valid
        # rows fit a single 1Q tile through the cheaper 1Q body. The grid is
        # 2Q-tiled, so `seq_info.prompt_offset` is a multiple of the 2Q `BM`
        # (256) — also a valid 1Q tile boundary (128) — and
        # `seq_len - prompt_offset` is this tile's remaining valid rows. Any
        # 2Q tile that would have an empty output half (<= 128 rows left) is
        # exactly the tile routed here, which is what lets the 2Q body fold
        # away its empty-half guard via `output_nonempty`. `broadcast` makes
        # the branch warp-uniform; the decision is identical on all 16 warps
        # because `seq_info` is, so no cross-warp sync is needed.
        var seq_info: SeqInfo = get_seq_info[
            Self.BM,
            Self.num_q_heads,
            Self.MaskType.get_type_name() == "CausalMask",
        ](batch_size, pack.max_seq_len, pack.valid_length, pack.partition)

        comptime cfg_1q = Self.config.switch_1q_config()
        # 1Q does not implement the runtime FULL_MASK slow path (e.g. what
        # MaterializedMask needs); gate the switch off it, mirroring the
        # dispatch heuristic and the `load`/`mma` 1Q backstops.
        comptime check_mask = (
            Self.MaskType.nonfull_sets[cfg_1q.BM, cfg_1q.BN]()[0]
            == TileMaskStatus.UNKNOWN_MASK
        )
        comptime if Self.config.can_switch_to_1q() and not check_mask:
            comptime Kernel1Q = SM100MLA[
                Self.KVLUTType,
                Self.KRopeType,
                Self.output_dtype,
                Self.MaskType,
                Self.SchedulerType,
                cfg_1q,
                Self.ValidLengthType,
                Self.SinkType,
                Self.KVRowOffsetsType,
                Self.MaxSeqLenType,
                Self.PartitionType,
                Self._ndbuffer_mha_operand,
            ]
            # The TMA-op types are spelled from the 2Q `Self.config`; the 1Q
            # types fold to identical values (Q TMA / ragged store
            # `BM // num_q` = 128 in both; `depth = BK0` matches because the
            # live dispatch-time path already feeds these same ops to a
            # kernel built from both configs; K_nope/K_rope/V shapes are
            # BM-independent), but the parser sees distinct parameter
            # expressions, so `rebind`.
            comptime Q1Q = QTMATile[
                Kernel1Q.KVLUTType.dtype,
                Kernel1Q.config.qkv_swizzle_mode,
                BM=Kernel1Q.config.q_tile_rows(),
                depth=Kernel1Q.config.BK0,
                group=Kernel1Q.config.group,
                decoding=False,
            ]
            comptime KNope1Q = KVTMATile[
                Kernel1Q.KVLUTType.dtype,
                Kernel1Q.config.qkv_swizzle_mode,
                BN=kv_sub_tile_rows(Kernel1Q.config.BN, Kernel1Q.page_size),
                BK=Kernel1Q.nope_depth,
            ]
            comptime KRope1Q = KVTMATile[
                Kernel1Q.KRopeType.dtype,
                Kernel1Q.config.rope_gmem_swizzle_mode,
                BN=kv_sub_tile_rows(
                    Kernel1Q.config.BN, Kernel1Q.KRopeType.page_size
                ),
                BK=Kernel1Q.rope_depth,
            ]
            comptime V1Q = KVTMATile[
                Kernel1Q.KVLUTType.dtype,
                Kernel1Q.config.qkv_swizzle_mode,
                BN=kv_sub_tile_rows(Kernel1Q.config.BN, Kernel1Q.page_size),
                BK=Kernel1Q.ov_depth,  # V tile: ov_depth-wide
            ]
            comptime O1Q = RaggedTMA3DTile[
                Kernel1Q.output_dtype,
                Kernel1Q.config.output_swizzle_mode,
                BM=Kernel1Q.config.fa4_config.BM
                // Kernel1Q.config.fa4_config.num_q,
                BN=Kernel1Q.config.fa4_config.ov_depth,
                middle_dim=Kernel1Q.config.num_q_heads,
                group=Kernel1Q.config.fa4_config.group if Kernel1Q.config.fa4_config.fuse_gqa else 1,
                # Rebind target: must match the batched store the kernel built.
                tma_blocks_per_op=o_store_tma_blocks_per_op[
                    Kernel1Q.output_dtype,
                    Kernel1Q.config.output_swizzle_mode,
                    Kernel1Q.config.fa4_config.ov_depth,
                    Kernel1Q.config.fa4_config.group if Kernel1Q.config.fa4_config.fuse_gqa else 1,
                    depth_splits=2,
                ](),
            ]
            if broadcast(seq_info.seq_len - seq_info.prompt_offset) <= UInt32(
                Kernel1Q.BM
            ):
                Kernel1Q._kernel_impl_generic(
                    rebind[Q1Q](q_tma_op),
                    rebind[KNope1Q](k_nope_tma_op),
                    rebind[KRope1Q](k_rope_tma_op),
                    rebind[V1Q](v_tma_op),
                    rebind[O1Q](ragged_tma_store),
                    kv_lut,
                    k_rope_lut,
                    scale,
                    pack,
                    seq_info,
                )
                return
        Self._kernel_impl_generic(
            q_tma_op,
            k_nope_tma_op,
            k_rope_tma_op,
            v_tma_op,
            ragged_tma_store,
            kv_lut,
            k_rope_lut,
            scale,
            pack,
            seq_info,
        )

    @staticmethod
    @always_inline
    def _kernel_impl_generic(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.q_tile_rows(),
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
            BM=Self.config.fa4_config.BM // Self.config.fa4_config.num_q,
            BN=Self.config.fa4_config.ov_depth,
            middle_dim=Self.config.num_q_heads,
            group=config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
            # Concrete value: a GPU-kernel entry (and the device impl it calls)
            # must be a fully-bound function. Matches the created store.
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
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            Self.SinkType,
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
        seq_info: SeqInfo,
    ):
        comptime assert Self.MMA_M == 64 or Self.MMA_M == 128
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), (
            "depth = "
            + String(Self.config.qk_depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used)
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        var mask = pack.mask
        var max_seq_len = pack.max_seq_len

        # Matches the thin entrypoint's switch predicate: True iff short
        # tiles are actually routed to the 1Q body, so the 2Q softmax may
        # fold away its empty-output-half guard. A 1Q instantiation has
        # `can_switch_to_1q() == False`, so it correctly stays False.
        comptime _cfg_1q = Self.config.switch_1q_config()
        comptime _check_mask = (
            Self.MaskType.nonfull_sets[_cfg_1q.BM, _cfg_1q.BN]()[0]
            == TileMaskStatus.UNKNOWN_MASK
        )
        comptime output_nonempty = (
            Self.config.can_switch_to_1q() and not _check_mask
        )

        comptime num_q = Self.config.num_q()
        # TODO: We may want to support num_q>2 for depth=64?
        comptime assert (
            num_q == 1 or num_q == 2
        ), "Currently only support num_q == 1 or 2"
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
        var rope_smem = rebind[SharedMemPointer[Scalar[KRopeType.dtype]]](
            attn_smem.rope_smem_base()
        )
        var ptr_tmem_addr = attn_smem.tmem_addr_ptr()

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        #
        # Non-single-O keeps the 4-WG (512-thread) split at 184. Generic
        # single-O drops WG1 (3 WGs / 384 threads), freeing the WG1 register
        # budget for the lone remaining softmax WG. 208 is NOT "give softmax the
        # max": `num_reg_softmax` is that WG's `setmaxnreg` peak, and ptxas folds
        # the peak into the whole-function allocation it runs for the shared
        # prologue, so the peak must clear the prologue WITHOUT spilling a live
        # scalar to local memory. That is non-monotonic -- measured on B200,
        # single-O spills at 184/192/216/232 and at the "obvious" max 256 (896
        # local-ld sectors), and is spill-free only on the narrow {200, 208}
        # island (208 chosen). A ptxas/toolchain bump can move the island;
        # re-verify that the single-O `mla_prefill_generic` launch reports ZERO
        # local traffic (else sweep `num_reg_softmax` by 8 to re-find it):
        #   ncu --target-processes all --kernel-name-base demangled \
        #       --kernel-name regex:"mla_prefill_generic" \
        #       --print-metric-instances values --metrics \
        #       l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum <binary>
        comptime num_reg_softmax = 208 if Self.config.fa4_config.single_o else 184
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

        # Programmatic Dependent Launch (PDL).  This is the only point every
        # thread of every CTA reaches before the warp-specialized early
        # returns below (invalid tiles bail per-role in the Softmax/Correction/
        # Load/MMA warps), so it is the only divergence-free place to honor the
        # contract that *every* CTA signal launch-dependents — otherwise a
        # back-to-back consumer grid's `wait` hangs (see MLA decode).  The
        # data-independent prologue above (barrier init, tmem alloc, TMA
        # descriptor prefetch) overlaps the predecessor grid's tail; `wait`
        # fences here before the data-dependent Q/K/V loads in `Self.load`;
        # `launch` lets the successor grid's prologue overlap our compute.
        comptime if MLA_PREFILL_PDL_LEVEL > PDLLevel.OFF:
            wait_on_dependent_grids()
            launch_dependent_grids()

        # Read the TMEM base from SMEM ONCE here, post-barrier (alloc + this
        # barrier publish it), and carry it by register into the shared
        # fa4_softmax / fa4_correction consumers. See the depth-512 fix: the
        # consumers must not re-read `tmem_addr_ptr()` in their bodies.
        var tmem_addr = ptr_tmem_addr[0]

        var role = warp_idx_to_role[Self.config.fa4_config.single_o](warp_idx)

        # warp group partitioning
        # Two QO:
        if role == WarpRole.Softmax0 or role == WarpRole.Softmax1:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()

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
                output_nonempty=output_nonempty,
                single_softmax_wg=Self.config.fa4_config.single_o,
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

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            Self.load(
                misc_mbars,
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
                misc_mbars,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
            )
        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    def load[
        KRopeType: MHAOperand
    ](
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.q_tile_rows(),
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
            KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.ov_depth,  # V tile: ov_depth-wide
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[KRopeType.dtype]],
    ):
        comptime num_q = Self.config.num_q()
        # 1Q does not implement mid-range FULL_MASK tile skipping (the
        # `check_mask` slow path that e.g. MaterializedMask requires):
        # the load/mma/softmax warps would disagree on tile counts.
        # Dispatch must route such masks to 2Q. Range-bounded
        # early-skipping (e.g. sliding-window via `start_column` /
        # `last_masked_set_end`) is supported.
        comptime if num_q == 1:
            comptime assert not (
                mask.nonfull_sets[Self.BM, Self.BN]()[0]
                == TileMaskStatus.UNKNOWN_MASK
            ), (
                "1Q MLA prefill does not support masks requiring runtime"
                " FULL_MASK checks"
            )
        comptime KVPipeType = MLAKVLayouts[
            Self.KVLUTType.dtype,
            KRopeType.dtype,
            None,
            Self.config,
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        # In 1Q shared-KV mode the pattern is instead
        # q & k0, k1, v0, v1, k2, k3, v2, v3, ... (matching the mma's
        # even/odd consumption). Non-shared-KV needs no producer reorder: K
        # and V live in independent pipelines, so only the Q1 issue is
        # gated on num_q == 2.
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

        # Per-TMA-call element count: one of two Q halves in 2Q, the
        # single full-BM Q tile in 1Q — 128 * BK0 in both modes.
        comptime q_elements = (Self.config.q_tile_rows()) * Self.config.BK0
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
            Self.config.BN, KRopeType.page_size
        )
        comptime num_rope_pages = kv_num_sub_tiles(
            Self.config.BN, KRopeType.page_size
        )
        comptime PagedRows = PagedRowIndices[Self.config.BN, Self.page_size]
        comptime RopePagedRows = PagedRowIndices[
            Self.config.BN, KRopeType.page_size
        ]

        # Alignment of `kv_row` produced by mask-driven iteration.
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
        # overloads (mirrors FA4 `load_warp.mojo`). K_nope/V use
        # `Self.page_size`; K_rope uses `KRopeType.page_size`. Compute
        # the flags independently — the iter_count peel below is gated
        # by either.
        comptime needs_partial_kv = (
            Self.page_size > 0 and Self.page_size < Self.config.BN
        )
        comptime needs_partial_rope = (
            KRopeType.page_size > 0 and KRopeType.page_size < Self.config.BN
        )
        comptime needs_partial = needs_partial_kv or needs_partial_rope

        # Per-sub-page byte sizes for partial expect_bytes_pred.
        comptime k_nope_bytes_pp = (
            Self.nope_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )
        comptime k_rope_bytes_pp = (
            Self.rope_depth * rope_sub_BN * size_of[KRopeType.dtype]()
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

        # ---- Mode-shared sub-tile constants & closures ----
        # The K_rope sub-tile shape and V tile shape are identical in
        # shared-KV and non-shared-KV modes (only the smem base pointer and
        # pipeline machinery differ). Hoist the constants and the unified
        # `_produce_k_rope` / `_produce_v` closures so both modes share
        # them — mirrors `mla_prefill_blockscale.mojo`'s pattern.
        comptime k_rope_sub_elems = Self.rope_depth * rope_sub_BN
        comptime KRopeSubType = TileTensor[
            KRopeType.dtype,
            type_of(tt_row_major[k_rope_sub_elems]()),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        # Full-tile byte counts (no partial bound applies).
        comptime k_nope_full_bytes = (
            Self.nope_depth * Self.config.BN * size_of[Self.qkv_dtype]()
        )
        comptime k_rope_full_bytes = (
            Self.rope_depth * Self.config.BN * size_of[KRopeType.dtype]()
        )
        # V full-tile bytes use the V head dim (`ov_depth`), which differs from
        # `k_nope_full_bytes` (nope width) when `v_head_dim != qk_nope_head_dim`.
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
            smem_base_ptr: SharedMemPointer[Scalar[KRopeType.dtype]],
            mbar: SharedMemPointer[SharedMemBarrier],
            rope_nvp: UInt32,
        ):
            """K_rope sub-tile TMA into smem starting at `smem_base_ptr`,
            signaling completion on `mbar`.

            Caller is responsible for the `expect_bytes_pred` covering
            the K barrier — K_rope shares the K barrier with K_nope (and
            Q on the prologue) in MLA generic, so the byte total is
            accounted at the call site (`_produce_k_fused` /
            `_produce_k_split`) alongside K_nope and Q.

            `partial=True` early-returns when `_p == rope_nvp`, mirroring
            `PagedRowIndices.tma_copy_k[needs_partial=True]`.
            `partial=False` collapses to the existing fully-unrolled body
            — codegen-identical pre-fix.
            """
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
                        "MLA K_rope sub-tile TMA OOB after partial"
                        " bound: kv_row_base="
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
                    mbar[],
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
            """V tile production at `mbar`/`smem_ptr`.

            Both modes pass the V destination smem pointer directly so
            this closure doesn't need to know about pipeline machinery
            (kv_pipeline vs pipeline_v). `partial=True` runtime-bounds
            the sub-tile loop and accounts only the bytes actually
            delivered.
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

        comptime if Self.config.fa4_config.use_shared_kv:
            # ---- Shared KV mode ----
            # Single StagedPipeline with alternating K_nope and V stages.
            # K_rope stored separately in rope_smem, protected by K barriers.
            # Stages: K_nope0, V0, K_nope1, V1, ...
            # Stage fits the wider of K_nope/V (shared_kv_cols).
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

            # Rope buffer index: cycles through ceildiv(num_kv_stages, 2)
            # independently from the shared KV pipeline, since only K stages
            # (every other shared stage) need rope.
            var rope_idx: UInt32 = 0
            comptime num_rope_bufs = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )

            @__parameter
            @always_inline
            def _fused_rope_smem_ptr(
                idx: UInt32,
            ) -> SharedMemPointer[Scalar[KRopeType.dtype]]:
                """Return the K_rope smem base for rope buffer slot `idx`.

                Bitcast preserves parallelism with blockscale's pattern
                where the rope buffer's underlying storage may be a
                different dtype than `KRopeType.dtype`. For generic where
                `rope_smem_base` is already `KRopeType`-typed the bitcast
                is a no-op.
                """
                return rope_smem_base.bitcast[
                    Scalar[KRopeType.dtype]
                ]() + idx * UInt32(rope_stage_elems)

            @__parameter
            @always_inline
            def _fused_v_smem_ptr() -> (
                SharedMemPointer[Scalar[Self.KVLUTType.dtype]]
            ):
                """V destination smem ptr at the current KV stage."""
                return k_smem_base + kv_pipeline.state.index() * UInt32(
                    kv_stage_elems
                )

            @__parameter
            @always_inline
            def _produce_k_fused[
                partial: Bool,
                with_q: Bool = False,
            ](
                paged: type_of(paged_rows),
                rope_paged: type_of(rope_paged_rows),
                kv_row_local: UInt32,
                rope_idx_local: UInt32,
                mbar: type_of(kv_pipeline.producer_mbar()),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q (if `with_q`) + K_nope + K_rope onto `mbar`.

                Mirrors FA4's `_produce_k_stage` pattern: one helper used
                across prologue (with_q=True), main loop (partial=False),
                and peeled-last (partial=needs_partial). The
                `expect_bytes_pred` accumulator branches on `partial`
                comptime; the runtime barrier hint is one PTX issue.
                """
                var qk_bytes: Int32 = Int32(q_bytes) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                    qk_bytes += Int32(k_rope_bytes_pp) * Int32(rope_nvp)
                else:
                    qk_bytes += Int32(k_nope_full_bytes + k_rope_full_bytes)
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
                    _fused_rope_smem_ptr(rope_idx_local),
                    mbar,
                    rope_nvp,
                )

            comptime if num_q == 1:
                # ---- 1Q shared-KV producer ----
                # MMA consumes K_e, K_o, V_e, V_o per logical iter;
                # produce in matching slot order (mirrors
                # load_warp.mojo's 1Q shared producer). No FULL_MASK
                # skipping here (see the `check_mask` assert at the top
                # of `load`).

                # Per-tile emit helpers bundling the KV producer-pipeline
                # lifecycle (acquire / mbar / produce / rope-buf cycle /
                # step). The first peeled K slot passes `acquire=False`
                # (initial producer phase = 1).
                @__parameter
                @always_inline
                def _emit_k_1q[
                    partial: Bool,
                    with_q: Bool = False,
                    acquire: Bool = True,
                ](
                    paged: type_of(paged_rows),
                    rope_paged: type_of(rope_paged_rows),
                    kv_row_local: UInt32,
                    k_nvp: UInt32,
                    rope_nvp: UInt32,
                ):
                    comptime if acquire:
                        kv_pipeline.producer_acquire()
                    var mbar = kv_pipeline.producer_mbar()
                    _produce_k_fused[partial=partial, with_q=with_q](
                        paged,
                        rope_paged,
                        kv_row_local,
                        rope_idx,
                        mbar,
                        k_nvp,
                        rope_nvp,
                    )
                    rope_idx = (rope_idx + 1) % num_rope_bufs
                    kv_pipeline.state.step()

                @__parameter
                @always_inline
                def _emit_v_1q[
                    partial: Bool
                ](paged: type_of(paged_rows), v_nvp: UInt32):
                    kv_pipeline.producer_acquire()
                    var mbar = kv_pipeline.producer_mbar()
                    _produce_v[partial=partial](
                        paged, mbar, _fused_v_smem_ptr(), v_nvp
                    )
                    kv_pipeline.state.step()

                # T is the total K-tile count (iter_count was peeled by 1
                # at the top of `load` for the 2Q flow).
                var T: UInt32 = iter_count + UInt32(1)
                var k_nvp_0 = _k_num_valid_pages(kv_row)
                var rope_nvp_0 = _rope_num_valid_pages(kv_row)

                # NOTE: single-O (1Q wide-V) never reaches the shared-KV
                # producer — `FA4Config` forces `use_shared_kv=False` for
                # single-O, so its per-tile production lives on non-shared-KV.

                # T == 1 fast path: produce K_e[0] (with Q) + V_e[0]
                # only. mma's matching fast path consumes those two
                # slots then returns.
                if T == UInt32(1):
                    _emit_k_1q[
                        partial=needs_partial, with_q=True, acquire=False
                    ](paged_rows, rope_paged_rows, kv_row, k_nvp_0, rope_nvp_0)
                    _emit_v_1q[partial=needs_partial](paged_rows, k_nvp_0)
                    return

                # ---- Peel (T >= 2): K_e[0], K_o[0], V_e[0], V_o[0] ----
                var kv_row_o = kv_row + UInt32(Self.config.BN)
                var k_nvp_o: UInt32 = UInt32(num_kv_pages)
                var rope_nvp_o: UInt32 = UInt32(num_rope_pages)
                comptime if needs_partial:
                    k_nvp_o = _k_num_valid_pages(kv_row_o)
                    rope_nvp_o = _rope_num_valid_pages(kv_row_o)
                var paged_rows_o = kv_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row_o)
                var rope_paged_rows_o = k_rope_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row_o)

                # K_e[0] with Q (initial slot; no acquire).
                _emit_k_1q[partial=needs_partial, with_q=True, acquire=False](
                    paged_rows, rope_paged_rows, kv_row, k_nvp_0, rope_nvp_0
                )
                # K_o[0]
                _emit_k_1q[partial=needs_partial](
                    paged_rows_o,
                    rope_paged_rows_o,
                    kv_row_o,
                    k_nvp_o,
                    rope_nvp_o,
                )
                # V_e[0] (reuses paged_rows)
                _emit_v_1q[partial=needs_partial](paged_rows, k_nvp_0)
                # V_o[0] (reuses paged_rows_o)
                _emit_v_1q[partial=needs_partial](paged_rows_o, k_nvp_o)

                # ---- Loop bookkeeping ----
                # Peel consumed K_e[0] + K_o[0] and V_e[0] + V_o[0]. Each
                # main-loop iter produces 2 K + 2 V (one full logical
                # iter). Tail (T odd) produces a trailing K_e + V_e only.
                var main_iters_1q: UInt32 = (T - UInt32(2)) >> UInt32(1)
                var has_tail: Bool = (T & UInt32(1)) == UInt32(1)
                var has_peeled_last_full: Bool = False
                comptime if needs_partial:
                    # When T is odd, the tail block (always run) handles
                    # the partial trailing K_e + V_e. When T is even and
                    # there is at least one main-loop pair, the terminal
                    # full pair is partial and reserved for peeled-last.
                    if not has_tail and main_iters_1q > UInt32(0):
                        main_iters_1q -= UInt32(1)
                        has_peeled_last_full = True

                # ---- Main loop (full tiles) ----
                while main_iters_1q != UInt32(0):
                    main_iters_1q -= UInt32(1)
                    kv_row += UInt32(2 * Self.config.BN)
                    kv_row_o = kv_row + UInt32(Self.config.BN)

                    # K_e[n] (full)
                    paged_rows = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    rope_paged_rows = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    _emit_k_1q[partial=False](
                        paged_rows,
                        rope_paged_rows,
                        kv_row,
                        UInt32(num_kv_pages),
                        UInt32(num_rope_pages),
                    )
                    # K_o[n] (full)
                    paged_rows_o = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row_o)
                    rope_paged_rows_o = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row_o)
                    _emit_k_1q[partial=False](
                        paged_rows_o,
                        rope_paged_rows_o,
                        kv_row_o,
                        UInt32(num_kv_pages),
                        UInt32(num_rope_pages),
                    )
                    # V_e[n] / V_o[n] (full, reuse rows)
                    _emit_v_1q[partial=False](paged_rows, UInt32(num_kv_pages))
                    _emit_v_1q[partial=False](
                        paged_rows_o, UInt32(num_kv_pages)
                    )

                # ---- Tail K_e + V_e (T odd, any needs_partial) ----
                # mma's break-check swaps s1/o1 onto s0/o0 and consumes
                # one trailing K + V; that K and V come from this block.
                if has_tail:
                    kv_row += UInt32(2 * Self.config.BN)
                    var k_nvp_t: UInt32 = UInt32(num_kv_pages)
                    var rope_nvp_t: UInt32 = UInt32(num_rope_pages)
                    comptime if needs_partial:
                        k_nvp_t = _k_num_valid_pages(kv_row)
                        rope_nvp_t = _rope_num_valid_pages(kv_row)
                    paged_rows = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    rope_paged_rows = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    _emit_k_1q[partial=needs_partial](
                        paged_rows, rope_paged_rows, kv_row, k_nvp_t, rope_nvp_t
                    )
                    _emit_v_1q[partial=needs_partial](paged_rows, k_nvp_t)

                # ---- Peeled-last full pair (needs_partial, T even) ----
                comptime if needs_partial:
                    if has_peeled_last_full:
                        kv_row += UInt32(2 * Self.config.BN)
                        kv_row_o = kv_row + UInt32(Self.config.BN)
                        var k_nvp_pe = _k_num_valid_pages(kv_row)
                        var rope_nvp_pe = _rope_num_valid_pages(kv_row)
                        var k_nvp_po = _k_num_valid_pages(kv_row_o)
                        var rope_nvp_po = _rope_num_valid_pages(kv_row_o)
                        # K_e (partial)
                        paged_rows = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        rope_paged_rows = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        _emit_k_1q[partial=True](
                            paged_rows,
                            rope_paged_rows,
                            kv_row,
                            k_nvp_pe,
                            rope_nvp_pe,
                        )
                        # K_o (partial)
                        paged_rows_o = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row_o)
                        rope_paged_rows_o = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row_o)
                        _emit_k_1q[partial=True](
                            paged_rows_o,
                            rope_paged_rows_o,
                            kv_row_o,
                            k_nvp_po,
                            rope_nvp_po,
                        )
                        # V_e / V_o (partial, reuse rows)
                        _emit_v_1q[partial=True](paged_rows, k_nvp_pe)
                        _emit_v_1q[partial=True](paged_rows_o, k_nvp_po)
            else:
                # ---- 2Q shared-KV producer (original) ----

                # ---- Peeled: K0 + Q0 on same barrier ----
                var k0_mbar = kv_pipeline.producer_mbar()
                var k_nvp_0 = _k_num_valid_pages(kv_row)
                var rope_nvp_0 = _rope_num_valid_pages(kv_row)
                _produce_k_fused[partial=needs_partial, with_q=True](
                    paged_rows,
                    rope_paged_rows,
                    kv_row,
                    rope_idx,
                    k0_mbar,
                    k_nvp_0,
                    rope_nvp_0,
                )
                rope_idx = (rope_idx + 1) % num_rope_bufs
                kv_pipeline.state.step()  # step -> stage 1

                # ---- Q1 (separate barrier) ----
                q_gmem_row += UInt32(Self.config.BM // 2)
                var q1_mbar = mbars.q1_wait_mbar()
                expect_bytes_pred(q1_mbar, Int32(q_bytes), e)
                # Elect-predicated in-PTX via `_elect`; no Mojo `if e != 0:`.
                q_tma_op.async_copy_elect(
                    QType(q_smem + q_elements, tt_row_major[q_elems]()),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.qk_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                    e,
                )

                # ---- V0 (reuses paged_rows from K0) ----
                kv_pipeline.producer_acquire()
                var v0_mbar = kv_pipeline.producer_mbar()
                _produce_v[partial=needs_partial](
                    paged_rows, v0_mbar, _fused_v_smem_ptr(), k_nvp_0
                )
                kv_pipeline.state.step()

                comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
                    0
                ] == TileMaskStatus.UNKNOWN_MASK

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
                    paged_rows = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    rope_paged_rows = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)

                    # Produce K_nope_n + K_rope_n (full sub-tile loops)
                    kv_pipeline.producer_acquire()
                    var kn_mbar = kv_pipeline.producer_mbar()
                    _produce_k_fused[partial=False](
                        paged_rows, rope_paged_rows, kv_row, rope_idx, kn_mbar
                    )
                    rope_idx = (rope_idx + 1) % num_rope_bufs
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
                                    mask,
                                    seq_info.prompt_idx,
                                    score_row,
                                    kv_row,
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
                            # Kn (partial)
                            kv_pipeline.producer_acquire()
                            var kn_mbar_last = kv_pipeline.producer_mbar()
                            _produce_k_fused[partial=needs_partial](
                                paged_rows,
                                rope_paged_rows,
                                kv_row,
                                rope_idx,
                                kn_mbar_last,
                                k_nvp_last,
                                rope_nvp_last,
                            )
                            rope_idx = (rope_idx + 1) % num_rope_bufs
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
            # K_nope part is `padded_nope_depth` wide (non-shared-KV mode: V has its
            # own `pipeline_v`), so this is K-only.
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
            def _split_v_smem_ptr(
                pair: type_of(pipeline_v.get_tile[qk_stage=0]()),
            ) -> SharedMemPointer[Scalar[Self.KVLUTType.dtype]]:
                """V destination smem ptr for non-shared-KV's V pipeline pair.

                Mirrors blockscale's `_split_v_smem_ptr` so the unified
                `_produce_v` closure can emit a partial-aware
                `expect_bytes_pred` itself (rather than relying on
                `pipeline_v.get_v(e)`'s fixed-size auto-expect).
                """
                return rebind[SharedMemPointer[Scalar[Self.KVLUTType.dtype]]](
                    pair.smem.ptr
                )

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
                """Q (if `with_q`) + K_nope + K_rope onto `mbar` (non-shared-KV).

                Includes the `split_smem` decomposition into K_nope and
                K_rope smem regions so the call sites only need to set
                up the barrier and pass paged-row indices.
                """
                var qk_bytes: Int32 = Int32(q_bytes) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                    qk_bytes += Int32(k_rope_bytes_pp) * Int32(rope_nvp)
                else:
                    qk_bytes += Int32(KVPipeType.k_bytes)
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
                    KRopeType.dtype,
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
                    rebind[SharedMemPointer[Scalar[KRopeType.dtype]]](
                        k_rope_smem_local.ptr
                    ),
                    mbar,
                    rope_nvp,
                )

            # ---- K0 + Q0 (combined barrier) ----
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
            k_pipeline.state.step()

            # ---- Q1 (separate barrier) ----
            # Skipped in 1Q: the peeled K0 issue above (with_q=True)
            # already loaded the full BM-row Q tile on the K mbar.
            comptime if num_q == 2:
                var q1_mbar = mbars.q1_wait_mbar()
                expect_bytes_pred(q1_mbar, Int32(q_bytes), e)
                # Q1 — elect-predicated in-PTX via `_elect`.
                q_tma_op.async_copy_elect(
                    QType(q_smem + q_elements, tt_row_major[q_elems]()),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.qk_depth,
                        decoding=False,
                    ](
                        q_gmem_row + UInt32(Self.config.BM // 2),
                        q_head_idx,
                    ),
                    e,
                )

            # ---- V0 (reuses paged_rows from K0) ----
            var mbarv0 = pipeline_v.get_tile[qk_stage=0]()
            _produce_v[partial=needs_partial](
                paged_rows, mbarv0.mbar, _split_v_smem_ptr(mbarv0), k_nvp_0
            )
            pipeline_v.commit_step()
            comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
                0
            ] == TileMaskStatus.UNKNOWN_MASK

            # kv producer loop. Main body: always full tiles
            # (partial=False). When needs_partial, peel off the last
            # iteration so its populate/TMAs can be runtime-bounded.
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
    def mma(
        tmem_addr: UInt32,
        mbars: Self.MiscMBarsType,
        seq_id: UInt32,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KRopeType.dtype]],
    ):
        # 2Q: two Q halves q0/q1 feed s0/s1 against the same K tile.
        # 1Q: the single Q feeds s0/s1 against alternating (even/odd) K
        # tiles, and o0/o1 accumulate alternating V tiles — mirroring
        # mma_warp.mojo's 1Q structure.
        comptime num_q = Self.config.num_q()
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

        # Per-Q-tile size: one of two Q halves in 2Q, the single full-BM
        # Q tile in 1Q — 128 * padded_qk_depth in both modes.
        comptime q0_size = (
            Self.config.q_tile_rows()
        ) * Self.config.padded_qk_depth
        comptime q0_bytes = UInt32(q0_size * size_of[Self.KVLUTType.dtype]())
        var q0 = Self.descriptor_q(q_smem)
        var q1 = q0 + q0_bytes

        var q0_rope: type_of(q0)
        var q1_rope: type_of(q1)
        comptime if Self.config.fa4_config.use_shared_kv:
            # ---- Shared KV mode ----
            # Single StagedPipeline alternating K_nope and V.
            # K_rope is in a separate smem region, protected by the same
            # K barrier (load warp puts both on the same mbarrier).
            # Q@K' = Q_nope@K_nope (c_scale=0) + Q_rope@K_rope (c_scale=1).

            # Shared buffer stage fits the wider of K_nope/V (shared_kv_cols).
            comptime kv_stage_bytes = (
                Self.config.fa4_config.shared_kv_cols()
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )
            # rope_smem holds the rope tiles in `rope_mma_dtype` (e.g. BF16
            # for per-token-scale FP8 KV), which may differ from the K/V
            # `KVLUTType.dtype` (FP8). Use the rope dtype size for the
            # per-buffer stride. (For generic same-dtype MLA the two
            # coincide.)
            comptime rope_stage_bytes = (
                Self.config.rope_depth
                * Self.config.BN
                * size_of[Self.rope_mma_dtype]()
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
            # K_rope descriptor: k_major for Q@K_rope'
            var rope_desc = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.rope_depth,
                swizzle_mode=Self.config.rope_mma_swizzle_mode,
                is_k_major=True,
            ](rope_smem_base)

            # Q_rope descriptor.
            # Q_nope occupies the first `q_rope_byte_offset` bytes of the Q
            # smem tile.
            #
            # Same-dtype MLA (generic): Q is loaded as ONE wide
            # (nope+rope) tile in `qkv_dtype` with a single swizzle, so
            # Q_rope is just a byte-offset into q0 sharing q0's descriptor
            # (swizzle + k-major stride).
            #
            # Mixed-dtype MLA (per-token-scale: FP8 nope + BF16 rope):
            # the load stores Q_nope and Q_rope as SEPARATE tiles with
            # different dtypes AND swizzles (FP8 64B nope, BF16 128B rope),
            # so `q0 + offset` would read the BF16 rope with the FP8
            # descriptor's swizzle — garbage. Build a dedicated rope
            # descriptor at the rope sub-tile, mirroring the non-shared
            # mixed-dtype path (`descriptor_q_rope`).
            comptime q_rope_off = UInt32(Self.q_rope_byte_offset)
            comptime if Self.qkv_dtype == Self.rope_mma_dtype:
                q0_rope = q0 + q_rope_off
                q1_rope = q1 + q_rope_off
            else:
                # Mixed-dtype is single-CTA num_q=1 only (per-token-scale
                # 2Q is non-shared-KV, never shared), so q1_rope is dead here;
                # define it from the same base for type consistency.
                q0_rope = Self.descriptor_q_rope(
                    (q_smem.bitcast[UInt8]() + q_rope_off).bitcast[
                        Scalar[Self.rope_mma_dtype]
                    ]()
                )
                q1_rope = q0_rope

            comptime KVPipeType = StagedPipeline[
                Self.config.fa4_config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

            # Rope buffer index: cycles independently through
            # ceildiv(num_kv_stages, 2) indices, one per K tile.
            var rope_idx: UInt32 = 0
            comptime num_rope_bufs = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )

            # We peel the first iteration, as we want to wait on q1.
            # 2Q: peel consumes 1 K_0 (shared); main loop decrements once
            # per iter. iter_count = total_iters - 1.
            # 1Q: peel consumes 2 K-tiles (K_e[0], K_o[0]) and 2 V-tiles
            # (V_e[0], V_o[0] held); main loop decrements once at top
            # (K_e consume) plus once inside a 1Q guard (K_o consume).
            # iter_count = total_iters - 2. 1Q at total_iters == 1 takes
            # the T==1 fast path below, so the wrap at T == 1 is never
            # read. Unified: subtract (3 - num_q).
            var total_iters_runtime: UInt32 = mask.total_iters[
                Self.BM, Self.BN, Self.page_size
            ](seq_id, score_row, num_keys)
            var iter_count: UInt32 = total_iters_runtime - UInt32(3 - num_q)

            var e = elect()

            # Release the KV slot at `release_idx`, advance to the next
            # stage, wait for it, and return its slot index (1Q only;
            # mirrors mma_warp.mojo's `_advance_kv`).
            @__parameter
            @always_inline
            def _advance_kv(release_idx: UInt32) -> UInt32:
                kv_pipeline.consumer_release_at(release_idx, e)
                kv_pipeline.state.step()
                kv_pipeline.consumer_wait()
                return kv_pipeline.state.index()

            # NOTE: single-O (1Q wide-V) never reaches the shared-KV branch —
            # `FA4Config` forces `use_shared_kv=False` for single-O so the
            # single-O serial P@V path lives solely on non-shared-KV (below).

            # ---- Peeled iteration ----
            # Stage 0 = K0 (K_nope0 + K_rope0)
            kv_pipeline.consumer_wait()
            var k0 = (
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            )
            Self.UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
            var r0 = rope_desc + UInt32(rope_stage_bytes) * rope_idx
            Self.UMMA0RopeType.mma[stage_idx=0](
                q0_rope, r0, s0_tmem, elect=e, c_scale=1
            )
            pipeline_s0.commit_mma(e)

            # 1Q: release K_e[0]; step to slot 1; wait. Slot 1 holds
            # K_o[0] for T >= 2 and V_e[0] for T == 1 -- diverge on
            # descriptor base only.
            comptime if num_q == 1:
                rope_idx = (rope_idx + 1) % num_rope_bufs
                var slot1 = _advance_kv(kv_pipeline.state.index())

                # T == 1 fast path: slot 1 holds V_e[0] (load produced
                # K_e[0] + V_e[0] only). Do P_e @ V_e[0] -> o0 and
                # return. Don't touch s1 / o1 -- softmax WG1 takes its
                # matching no-op path.
                if total_iters_runtime == UInt32(1):
                    var v0_t1 = kv_desc_v + UInt32(kv_stage_bytes) * slot1
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s0[pv_stage].wait(0)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s0_tmem, v0_t1, o0_tmem, elect=e, c_scale=0
                        )
                    pipeline_o0.commit_mma(e)
                    kv_pipeline.consumer_release(e)  # release V_e[0]
                    return

                k0 = kv_desc_k + UInt32(kv_stage_bytes) * slot1
                r0 = rope_desc + UInt32(rope_stage_bytes) * rope_idx

            # Q1 @ K0 (2Q, wait for Q1 first) / Q @ K_o[0] (1Q,
            # q0 + redefined k0/r0)
            comptime if num_q == 2:
                var q1_mbar = mbars.q1_wait_mbar()
                q1_mbar[0].wait()
                Self.UMMA0Type.mma[stage_idx=0](
                    q1, k0, s1_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q1_rope, r0, s1_tmem, elect=e, c_scale=1
                )
            else:
                Self.UMMA0Type.mma[stage_idx=0](
                    q0, k0, s1_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, r0, s1_tmem, elect=e, c_scale=1
                )
            kv_pipeline.consumer_release(e)  # release K (K0 / K_o[0]), step
            rope_idx = (rope_idx + 1) % num_rope_bufs
            pipeline_s1.commit_mma(e)

            # Stage 1 = V0 (2Q held for first main iter) / V_e[0] (1Q
            # single use, then V_o[0] loaded and held)
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

            # 1Q: release V_e[0] (single use); load V_o[0] and hold its
            # slot index in v_prev_idx for the first main-loop iter's
            # P_o @ V_o[0] MMA.
            comptime if num_q == 1:
                v_prev_idx = _advance_kv(v_prev_idx)

            # ---- Main loop ----
            while iter_count != 0:
                iter_count -= 1

                # Advance past held V to get to next K
                kv_pipeline.state.step()

                # Kn (K_nope_n + K_rope_n) / K_e[n] (1Q)
                kv_pipeline.consumer_wait()
                var kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                Self.UMMA0Type.mma[stage_idx=0](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
                var rn = rope_desc + UInt32(rope_stage_bytes) * rope_idx
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, rn, s0_tmem, elect=e, c_scale=1
                )
                pipeline_s0.commit_mma(e)

                # P1 @ V_{n-1} (2Q) / P_o @ V_o[n-1] (1Q)
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

                # 1Q: between K_e[n] and K_o[n] -- break-check for tail
                # iter when total K-tiles is odd, else consume K_o[n] by
                # releasing K_e[n] and reassigning kn/rn.
                comptime if num_q == 1:
                    rope_idx = (rope_idx + 1) % num_rope_bufs
                    if iter_count == 0:
                        # Tail iter (T odd): no K_o[k]. The remaining
                        # work -- P_e[k] @ V_e[k] -> o0 -- has the same
                        # shape as the epilogue's P_1 @ V_last -> o1.
                        # Rebind o1-side aliases to o0-side resources
                        # and fall through; the epilogue does the work
                        # unchanged. Release K_e[k], wait V_e[k].
                        v_prev_idx = _advance_kv(kv_pipeline.state.index())
                        s1_tmem = s0_tmem
                        o1_tmem = o0_tmem
                        consumer_s1 = consumer_s0
                        pipeline_o1 = pipeline_o0
                        phase ^= 1  # advance from this iter's K@s1 phase
                        # to the V@o0 phase the s0 wait needs.
                        break
                    iter_count -= 1
                    # release K_e[n], wait K_o[n]
                    kn = kv_desc_k + UInt32(kv_stage_bytes) * _advance_kv(
                        kv_pipeline.state.index()
                    )
                    rn = rope_desc + UInt32(rope_stage_bytes) * rope_idx

                # Q1 @ Kn (2Q, q1 + same kn/rn) / Q @ K_o[n] (1Q,
                # q0 + redefined kn/rn)
                comptime if num_q == 2:
                    Self.UMMA0Type.mma[stage_idx=0](
                        q1, kn, s1_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q1_rope, rn, s1_tmem, elect=e, c_scale=1
                    )
                else:
                    Self.UMMA0Type.mma[stage_idx=0](
                        q0, kn, s1_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q0_rope, rn, s1_tmem, elect=e, c_scale=1
                    )
                kv_pipeline.consumer_release(e)  # release Kn / K_o[n], step
                rope_idx = (rope_idx + 1) % num_rope_bufs
                pipeline_s1.commit_mma(e)
                phase ^= 1

                # Vn (2Q held for next iter) / V_e[n] (1Q single use,
                # then V_o[n] loaded and held)
                kv_pipeline.consumer_wait()
                v_prev_idx = kv_pipeline.state.index()
                var vn = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vn, o0_tmem, elect=e, c_scale=1
                    )
                pipeline_o0.commit_mma(e)

                # 1Q: release V_e[n] (single use); load V_o[n] and hold
                # its slot in v_prev_idx for the next iter / epilogue.
                comptime if num_q == 1:
                    v_prev_idx = _advance_kv(v_prev_idx)

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

            # We peel the first iteration, as we want to wait on q1.
            # 2Q: iter_count = total_iters - 1. 1Q: the peel consumes 2
            # K-tiles and 2 V-tiles; iter_count = total_iters - 2 (the
            # T==1 fast path returns before the wrap at T == 1 is read).
            var total_iters_runtime: UInt32 = mask.total_iters[
                Self.BM, Self.BN, Self.page_size
            ](seq_id, score_row, num_keys)
            var iter_count: UInt32 = total_iters_runtime - UInt32(3 - num_q)

            # vo_prev_idx tracks the held V_o slot index in 1Q (needed
            # by the deferred consumer_release_at). Unused in 2Q (held V
            # is at the current pipeline state).
            var vo_prev_idx: UInt32 = 0

            # Shared single-O P@V tail: fold one V tile into the single O0
            # (init on tile 0 via c_scale=0, then accumulate onto the
            # correction-rescaled O0 with c_scale=1) and advance the S/O
            # phase. Called from BOTH the same-dtype and mixed-dtype
            # single-O serial loops below; the body is the verbatim inline
            # tail both loops used, so `@always_inline` keeps them
            # byte-identical. Captures pipeline_v/consumer_s0/pipeline_o0/
            # s0_tmem/o0_tmem from this scope (mirrors `_advance_kv`).
            @__parameter
            @always_inline
            def _pv_into_o0(mut s_phase: UInt32, mut c_scale: UInt32, e: Int32):
                pipeline_v.wait_v()
                var vi = pipeline_v.get_v()
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(s_phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vi, o0_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o0.commit_mma(e)
                pipeline_v.release_v(e)
                c_scale = 1
                s_phase ^= 1

            # ---- single-O serial path (1Q wide-V), non-shared-KV, same-dtype ----
            # Wide V (e.g. v_head_dim=256) cannot hold the two per-WG O
            # partials in the 512-col TMEM (`2*BN + 2*padded_ov > 512`), so
            # single-O aliases O1 onto O0 and CANNOT run the two-WG even/odd
            # LSE-combine that the standard 1Q path uses (the softmax peer O
            # read would overrun TMEM). Instead ONE warp group folds EVERY
            # K/V tile serially into the single O0 (WG1 no-op; the softmax and
            # correction warps take matching single-O paths). Non-shared-KV has
            # separate K and V consumer pipelines; per tile do Q@K_i -> s0,
            # then P_i @ V_i -> O0 (init on tile 0, accumulate onto the
            # correction-rescaled O0 after). `consumer_s0` completion gates
            # both "P ready" and "O0 rescaled". `FA4Config` forces non-shared-KV
            # for single-O, so this (and the mixed-dtype variant below) are
            # the ONLY single-O MMA paths.
            comptime if Self.config.fa4_config.single_o and Self.fused_umma0:
                var e = elect()
                var t_left: UInt32 = total_iters_runtime
                var s_phase: UInt32 = 0
                var c_scale: UInt32 = 0
                while t_left != 0:
                    t_left -= 1
                    pipeline_k.wait_k()
                    var ki = pipeline_k.get_k()
                    Self.UMMA0Type.mma(q0, ki, s0_tmem, elect=e, c_scale=0)
                    pipeline_s0.commit_mma(e)
                    pipeline_k.release_k(e)

                    _pv_into_o0(s_phase, c_scale, e)
                return

            comptime if Self.fused_umma0:
                # Q_0 @ K_0' (2Q) / Q @ K_e[0]' (1Q)
                pipeline_k.wait_k()
                var k0 = pipeline_k.get_k()
                var e = elect()
                Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
                comptime if num_q == 1:
                    pipeline_k.release_k(e)  # K_e[0] single use
                pipeline_s0.commit_mma(e)

                # 1Q T==1 fast path: only one K-tile exists (K_e[0]), so
                # K_o[0] is never produced and Q @ K_o[0] would hang on
                # wait_k. Do the single P_e @ V_e[0] -> o0 MMA and exit;
                # softmax WG1 takes the matching no-op path.
                comptime if num_q == 1:
                    if total_iters_runtime == UInt32(1):
                        var vlatest_t1 = pipeline_v.get_v()
                        pipeline_v.wait_v()
                        comptime for pv_stage in range(
                            Self.config.num_pv_stages
                        ):
                            _ = consumer_s0[pv_stage].wait(0)
                            Self.UMMA1Type.mma[stage_idx=pv_stage](
                                s0_tmem, vlatest_t1, o0_tmem, elect=e, c_scale=0
                            )
                        pipeline_o0.commit_mma(e)
                        pipeline_v.release_v(e)
                        return

                # Q_1 @ K_0' (2Q, q1 half, same k0) / Q @ K_o[0]' (1Q,
                # q0 + redefined k0)
                comptime if num_q == 2:
                    mbars.q1_wait_mbar()[0].wait()  # wait on Q1
                    Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
                    pipeline_s1.commit_mma(e)
                    pipeline_k.release_k(e)  # release K0
                else:
                    k0 = pipeline_k.get_k()  # K_o[0]
                    pipeline_k.wait_k()
                    Self.UMMA0Type.mma(q0, k0, s1_tmem, elect=e, c_scale=0)
                    pipeline_k.release_k(e)
                    pipeline_s1.commit_mma(e)

                # Wait V0 (2Q held for first main iter) / V_e[0] (1Q
                # single use, then V_o[0] loaded and held)
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

                # 1Q: release V_e[0] (single use); advance to V_o[0];
                # load and HOLD vlatest = V_o[0] in vo_prev_idx for the
                # first main-loop iter's P_o @ V_o[0] MMA. State is
                # pre-advanced past the held slot so subsequent get_v()
                # returns V_e[1].
                comptime if num_q == 1:
                    var ve_idx = pipeline_v.pipeline.state.index()
                    pipeline_v.pipeline.consumer_release_at(ve_idx, e)
                    pipeline_v.pipeline.state.step()
                    vlatest = pipeline_v.get_v()  # V_o[0]
                    pipeline_v.wait_v()
                    vo_prev_idx = pipeline_v.pipeline.state.index()
                    pipeline_v.pipeline.state.step()  # advance; do NOT release

                while iter_count != 0:
                    iter_count -= 1

                    # Q_0 @ K_n' (2Q) / Q @ K_e[n]' (1Q)
                    var kn = pipeline_k.get_k()
                    pipeline_k.wait_k()
                    Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
                    comptime if num_q == 1:
                        pipeline_k.release_k(e)  # K_e[n] single use
                    pipeline_s0.commit_mma(e)

                    # O_1 + P_1 @ V_{n-1} (2Q) / O_o + P_o @ V_o[n-1] (1Q)
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s1[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                        )
                    pipeline_o1.commit_mma(e)
                    c_scale = 1
                    # Release V_{n-1} (2Q at current state) / V_o[n-1]
                    # (1Q at vo_prev_idx; state was pre-advanced when
                    # V_o was held).
                    comptime if num_q == 2:
                        pipeline_v.release_v(e)
                    else:
                        pipeline_v.pipeline.consumer_release_at(vo_prev_idx, e)

                    # 1Q: between K_e[n] and K_o[n] -- break-check for
                    # tail iter when total K-tiles is odd, else load
                    # K_o[n] by reassigning kn (K_e[n] already released
                    # above).
                    comptime if num_q == 1:
                        if iter_count == 0:
                            # Tail iter (T odd). Same alias-swap pattern
                            # as shared-KV: rebind o1-side aliases to
                            # o0-side resources and fall through to the
                            # epilogue.
                            vlatest = pipeline_v.get_v()  # V_e[k]
                            pipeline_v.wait_v()
                            vo_prev_idx = pipeline_v.pipeline.state.index()
                            pipeline_v.pipeline.state.step()
                            s1_tmem = s0_tmem
                            o1_tmem = o0_tmem
                            consumer_s1 = consumer_s0
                            pipeline_o1 = pipeline_o0
                            phase ^= 1
                            break
                        iter_count -= 1
                        kn = pipeline_k.get_k()  # K_o[n]
                        pipeline_k.wait_k()

                    # Q_1 @ K_n' (2Q, q1 + same kn) / Q @ K_o[n]' (1Q,
                    # q0 + redefined kn)
                    comptime if num_q == 2:
                        Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
                    else:
                        Self.UMMA0Type.mma(q0, kn, s1_tmem, elect=e, c_scale=0)
                    pipeline_k.release_k(e)
                    pipeline_s1.commit_mma(e)
                    phase ^= 1

                    # O_0 + P_0 @ V_n (2Q) / O_e + P_e @ V_e[n] (1Q)
                    vlatest = pipeline_v.get_v()
                    pipeline_v.wait_v()
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s0[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                        )
                    pipeline_o0.commit_mma(e)

                    # 1Q: release V_e[n] (single use); advance to V_o[n];
                    # redefine vlatest = V_o[n] and hold its slot index
                    # in vo_prev_idx for the next iter / epilogue.
                    comptime if num_q == 1:
                        var ve_idx2 = pipeline_v.pipeline.state.index()
                        pipeline_v.pipeline.consumer_release_at(ve_idx2, e)
                        pipeline_v.pipeline.state.step()
                        vlatest = pipeline_v.get_v()  # V_o[n]
                        pipeline_v.wait_v()
                        vo_prev_idx = pipeline_v.pipeline.state.index()
                        pipeline_v.pipeline.state.step()  # no release

                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
            else:
                # ---- Non-shared mode with separate nope/rope UMMAs ----
                # K_nope and K_rope have different dtypes (e.g. FP8 + BF16),
                # so Q@K' = Q_nope@K_nope (c_scale=0) + Q_rope@K_rope (c_scale=1).

                # ---- Q descriptor setup ----
                # Q smem uses interleaved layout:
                #   [Q0_nope][Q0_rope][Q1_nope][Q1_rope]
                # Each Q half is q_nope_bytes + q_rope_bytes. Q_nope width is
                # the non-rope Q/K depth (`padded_nope_depth`), not V's width.
                comptime q_nope_bytes = (
                    (Self.config.q_tile_rows())
                    * Self.config.fa4_config.padded_nope_depth
                    * Self.qkv_dt_size
                )
                comptime q_rope_bytes = (
                    (Self.config.q_tile_rows())
                    * Self.rope_depth
                    * Self.config.rope_mma_dtype_size
                )
                comptime q_half_bytes = UInt32(q_nope_bytes + q_rope_bytes)

                var q0_nope = Self.descriptor_q(q_smem)
                q0_rope = Self.descriptor_q_rope(
                    (q_smem + q_nope_bytes // Self.qkv_dt_size).bitcast[
                        Scalar[Self.rope_mma_dtype]
                    ]()
                )
                var q1_nope = q0_nope + q_half_bytes
                q1_rope = q0_rope + q_half_bytes

                # ---- K descriptor setup ----
                # pipeline_k is used for wait/release only; get_k() is NOT
                # used because its descriptor has BK=padded_qk_depth with a
                # single swizzle, which is wrong for mixed dtypes.
                comptime nope_stage_bytes = (
                    Self.config.fa4_config.padded_nope_depth
                    * Self.config.BN
                    * Self.qkv_dt_size
                )
                comptime k_stage_stride = UInt32(KConType.full_kv_bytes)

                var kv_desc_k_nope = smem_descriptor[
                    BMN=Self.config.BN,
                    BK=Self.nope_depth,
                    swizzle_mode=Self.config.qkv_swizzle_mode,
                    is_k_major=True,
                ](k_smem_base)

                var kv_desc_k_rope = smem_descriptor[
                    BMN=Self.config.BN,
                    BK=Self.rope_depth,
                    swizzle_mode=Self.config.rope_mma_swizzle_mode,
                    is_k_major=True,
                ](
                    (
                        k_smem_base + nope_stage_bytes // Self.qkv_dt_size
                    ).bitcast[Scalar[Self.rope_mma_dtype]]()
                )

                # ---- single-O serial path (1Q wide-V), non-shared-KV, mixed ----
                # Mixed-dtype (e.g. FP8 nope + BF16 rope) wide V forces
                # single-O (aliased O0): ONE warp group folds EVERY K/V tile
                # serially into O0 (WG1 no-op). Non-shared-KV has separate K and V
                # consumer pipelines and a two-part Q@K' (Q_nope@K_nope +
                # Q_rope@K_rope). Per tile: Q@K_i -> s0, then P_i @ V_i -> O0
                # (init on tile 0, accumulate onto the correction-rescaled O0
                # after). `consumer_s0` gates "P ready" + "O0 rescaled".
                comptime if Self.config.fa4_config.single_o:
                    var e = elect()
                    var t_left_m: UInt32 = total_iters_runtime
                    var s_phase_m: UInt32 = 0
                    var c_scale_m: UInt32 = 0
                    while t_left_m != 0:
                        t_left_m -= 1
                        pipeline_k.wait_k()
                        var kidx_m = pipeline_k.pipeline.state.index()
                        var kn_m = kv_desc_k_nope + k_stage_stride * kidx_m
                        var kr_m = kv_desc_k_rope + k_stage_stride * kidx_m
                        Self.UMMA0Type.mma[stage_idx=0](
                            q0_nope, kn_m, s0_tmem, elect=e, c_scale=0
                        )
                        Self.UMMA0RopeType.mma[stage_idx=0](
                            q0_rope, kr_m, s0_tmem, elect=e, c_scale=1
                        )
                        pipeline_s0.commit_mma(e)
                        pipeline_k.release_k(e)

                        _pv_into_o0(s_phase_m, c_scale_m, e)
                    return

                # ---- Peeled iteration ----
                # Q_0 @ K_0' (2Q) / Q @ K_e[0]' (1Q)
                pipeline_k.wait_k()
                var k_idx = pipeline_k.pipeline.state.index()
                var k0_nope = kv_desc_k_nope + k_stage_stride * k_idx
                var k0_rope = kv_desc_k_rope + k_stage_stride * k_idx
                var e = elect()
                Self.UMMA0Type.mma[stage_idx=0](
                    q0_nope, k0_nope, s0_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, k0_rope, s0_tmem, elect=e, c_scale=1
                )
                comptime if num_q == 1:
                    pipeline_k.release_k(e)  # K_e[0] single use
                pipeline_s0.commit_mma(e)

                # 1Q T==1 fast path: only one K-tile exists (K_e[0]); do
                # the single P_e @ V_e[0] -> o0 MMA and exit (see the
                # fused_umma0 branch for details).
                comptime if num_q == 1:
                    if total_iters_runtime == UInt32(1):
                        var vlatest_t1 = pipeline_v.get_v()
                        pipeline_v.wait_v()
                        comptime for pv_stage in range(
                            Self.config.num_pv_stages
                        ):
                            _ = consumer_s0[pv_stage].wait(0)
                            Self.UMMA1Type.mma[stage_idx=pv_stage](
                                s0_tmem, vlatest_t1, o0_tmem, elect=e, c_scale=0
                            )
                        pipeline_o0.commit_mma(e)
                        pipeline_v.release_v(e)
                        return

                # Q_1 @ K_0' (2Q, q1 half, same K) / Q @ K_o[0]' (1Q,
                # q0 + recomputed descriptors at the next K slot)
                comptime if num_q == 2:
                    mbars.q1_wait_mbar()[0].wait()
                    Self.UMMA0Type.mma[stage_idx=0](
                        q1_nope, k0_nope, s1_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q1_rope, k0_rope, s1_tmem, elect=e, c_scale=1
                    )
                    pipeline_s1.commit_mma(e)

                    pipeline_k.release_k(e)
                else:
                    var k_idx_o = pipeline_k.pipeline.state.index()
                    k0_nope = kv_desc_k_nope + k_stage_stride * k_idx_o
                    k0_rope = kv_desc_k_rope + k_stage_stride * k_idx_o
                    pipeline_k.wait_k()
                    Self.UMMA0Type.mma[stage_idx=0](
                        q0_nope, k0_nope, s1_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q0_rope, k0_rope, s1_tmem, elect=e, c_scale=1
                    )
                    pipeline_k.release_k(e)
                    pipeline_s1.commit_mma(e)

                # Wait V0 (2Q held) / V_e[0] (1Q single use, then V_o[0]
                # loaded and held)
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

                # 1Q: release V_e[0]; load and hold V_o[0] (see the
                # fused_umma0 branch for details).
                comptime if num_q == 1:
                    var ve_idx = pipeline_v.pipeline.state.index()
                    pipeline_v.pipeline.consumer_release_at(ve_idx, e)
                    pipeline_v.pipeline.state.step()
                    vlatest = pipeline_v.get_v()  # V_o[0]
                    pipeline_v.wait_v()
                    vo_prev_idx = pipeline_v.pipeline.state.index()
                    pipeline_v.pipeline.state.step()  # advance; do NOT release

                # ---- Main loop ----
                while iter_count != 0:
                    iter_count -= 1

                    # Q_0 @ K_n' (2Q) / Q @ K_e[n]' (1Q)
                    var kn_nope = (
                        kv_desc_k_nope
                        + k_stage_stride * pipeline_k.pipeline.state.index()
                    )
                    var kn_rope = (
                        kv_desc_k_rope
                        + k_stage_stride * pipeline_k.pipeline.state.index()
                    )
                    pipeline_k.wait_k()
                    Self.UMMA0Type.mma[stage_idx=0](
                        q0_nope, kn_nope, s0_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q0_rope, kn_rope, s0_tmem, elect=e, c_scale=1
                    )
                    comptime if num_q == 1:
                        pipeline_k.release_k(e)  # K_e[n] single use
                    pipeline_s0.commit_mma(e)

                    # O_1 + P_1 @ V_{n-1} (2Q) / O_o + P_o @ V_o[n-1] (1Q)
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s1[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                        )
                    pipeline_o1.commit_mma(e)
                    c_scale = 1
                    comptime if num_q == 2:
                        pipeline_v.release_v(e)
                    else:
                        pipeline_v.pipeline.consumer_release_at(vo_prev_idx, e)

                    # 1Q: break-check for tail iter (T odd), else load
                    # K_o[n] by recomputing the descriptors (K_e[n] was
                    # already released above).
                    comptime if num_q == 1:
                        if iter_count == 0:
                            vlatest = pipeline_v.get_v()  # V_e[k]
                            pipeline_v.wait_v()
                            vo_prev_idx = pipeline_v.pipeline.state.index()
                            pipeline_v.pipeline.state.step()
                            s1_tmem = s0_tmem
                            o1_tmem = o0_tmem
                            consumer_s1 = consumer_s0
                            pipeline_o1 = pipeline_o0
                            phase ^= 1
                            break
                        iter_count -= 1
                        var k_idx_n = pipeline_k.pipeline.state.index()
                        kn_nope = kv_desc_k_nope + k_stage_stride * k_idx_n
                        kn_rope = kv_desc_k_rope + k_stage_stride * k_idx_n
                        pipeline_k.wait_k()

                    # Q_1 @ K_n' (2Q, q1 + same K) / Q @ K_o[n]' (1Q,
                    # q0 + recomputed descriptors)
                    comptime if num_q == 2:
                        Self.UMMA0Type.mma[stage_idx=0](
                            q1_nope, kn_nope, s1_tmem, elect=e, c_scale=0
                        )
                        Self.UMMA0RopeType.mma[stage_idx=0](
                            q1_rope, kn_rope, s1_tmem, elect=e, c_scale=1
                        )
                    else:
                        Self.UMMA0Type.mma[stage_idx=0](
                            q0_nope, kn_nope, s1_tmem, elect=e, c_scale=0
                        )
                        Self.UMMA0RopeType.mma[stage_idx=0](
                            q0_rope, kn_rope, s1_tmem, elect=e, c_scale=1
                        )
                    pipeline_k.release_k(e)
                    pipeline_s1.commit_mma(e)
                    phase ^= 1

                    # O_0 + P_0 @ V_n (2Q) / O_e + P_e @ V_e[n] (1Q)
                    vlatest = pipeline_v.get_v()
                    pipeline_v.wait_v()
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s0[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                        )
                    pipeline_o0.commit_mma(e)

                    # 1Q: release V_e[n]; load and hold V_o[n].
                    comptime if num_q == 1:
                        var ve_idx2 = pipeline_v.pipeline.state.index()
                        pipeline_v.pipeline.consumer_release_at(ve_idx2, e)
                        pipeline_v.pipeline.state.step()
                        vlatest = pipeline_v.get_v()  # V_o[n]
                        pipeline_v.wait_v()
                        vo_prev_idx = pipeline_v.pipeline.state.index()
                        pipeline_v.pipeline.state.step()  # no release

                # ---- Epilogue ----
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)


@always_inline
def mla_sm100_prefill_generic[
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
    comptime assert (
        KVType.dtype == VType.dtype
    ), "k and v must share an element dtype for SM100 MLA prefill"
    # Select the supported config: the standard 2-O config first (byte-identical
    # to the pre-decoupling path when v_head_dim == qk_nope_head_dim), else a
    # single-O fallback for a wide V. `v_depth` (V/output head dim) is `-1` for
    # the DeepSeek shape (V width == nope width). Shared with the blockscale /
    # per-token-scale kernels so the fallback policy lives in one place.
    comptime fa4_config = select_mla_prefill_config[
        q_type,
        rope_gmem_dtype=KRopeType.dtype,
        rope_mma_dtype=KRopeType.dtype,
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
        BM=fa4_config.q_tile_rows(),
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
        BM=fa4_config.q_tile_rows(),
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
    comptime ValidLengthType = NonNullPointer[.uint32]
    comptime SinkType = NullPointer[output_dtype]
    comptime KVRowOffsetsType = NullPointer[.uint32]
    comptime PartitionType = NoPartition[.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    # Launch the kernel built from `cfg` (the 2Q `fa4_config` or its 1Q
    # variant). All TMA ops were created by the caller from `fa4_config`;
    # their types fold to identical values for both configs (Q TMA and
    # ragged store use `BM // num_q` = 128 in both modes; K/V/rope TMA
    # shapes are BM-independent), so they are passed through unchanged.
    @__parameter
    @always_inline
    def _launch[cfg: MLAConfig]() raises:
        comptime assert cfg.supported(), cfg.fa4_config.description()
        comptime SchedulerType = TransientScheduler[
            UInt32(cfg.BM),
            UInt32(cfg.num_q_heads),
            flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
        ]

        comptime SM100MLAType = SM100MLA[
            KVType,
            KRopeType,
            output_dtype,
            MaskType,
            SchedulerType,
            cfg,
            ValidLengthType,
            SinkType,
            KVRowOffsetsType,
            MaxPromptLenType,
            PartitionType,
            _ndbuffer_mha_operand,
        ]

        comptime kernel = SM100MLAType.mla_prefill_kernel_generic

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
            max_prompt_len.as_uint32(), UInt32(cfg.BM)
        )
        var num_blocks: UInt32 = (
            max_num_prompt_tiles * PartitionType().num_partitions()
        )

        comptime num_threads = cfg.launch_num_threads()
        # When the launched (2Q) kernel may dispatch to the 1Q body at
        # runtime, it builds the 1Q `SM100AttentionSMem` over the same
        # dynamic-smem region, so reserve the max of both footprints.
        comptime smem_use = cfg.launch_smem_used()

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
            attributes=pdl_launch_attributes(MLA_PREFILL_PDL_LEVEL),
        )

    # --- 1Q / 2Q dispatch ---
    # 1Q (BM=128) halves the per-CTA Q-tile so short prompts don't run
    # MMAs at <= 50% utilization, mirroring the MHA heuristic in
    # `dispatch.mojo`. 1Q requires a mask without the runtime FULL_MASK
    # slow path (`check_mask`); range-bounded skipping (sliding-window
    # `start_column`) is fine.
    # The runtime 2Q->1Q short-seq switch only applies when the selected config
    # is the standard 2Q one. When the single-O fallback already picked num_q=1
    # (wide-V / GLM), launch it directly — it is single-CTA 1Q by construction.
    comptime if fa4_config.fa4_config.num_q == 2:
        comptime cfg_1q = fa4_config.with_num_q(1)
        comptime can_use_1q: Bool = (
            cfg_1q.supported()
            and cfg_1q.fa4_config.supported()
            and mask_functor.nonfull_sets[cfg_1q.BM, cfg_1q.BN]()[0]
            != TileMaskStatus.UNKNOWN_MASK
        )
        comptime if can_use_1q:
            if fa4_config.prefer_1q(
                max_prompt_len.as_uint32(),
                UInt32(PartitionType().num_partitions()),
                UInt32(batch_size),
                ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
            ):
                _launch[cfg_1q]()
            else:
                _launch[fa4_config]()
        else:
            _launch[fa4_config]()
    else:
        _launch[fa4_config]()
