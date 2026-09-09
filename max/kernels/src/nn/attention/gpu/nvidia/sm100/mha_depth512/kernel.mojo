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
"""Kernel entry point for depth=256/512 pair-CTA SM100 (Blackwell) MHA prefill.

Two neighboring SMs cooperate via pair-CTA MMA (cta_group=2,
cluster_shape=(2,1,1)).

Depth-dependent geometry:
  depth=512: MMA_M=128, BM=64,  BN=256. O split into O_lo/O_hi.
  depth=256: MMA_M=256, BM=128, BN=128. Single O accumulator.

Warp assignment (384 threads = 12 warps, 3 warp groups of 128):
    Warps 0-3:   Softmax (warp group 0)
    Warps 4-7:   Correction (warp group 1)
    Warp 8:      MMA (leader CTA issues pair-CTA MMA; peer early-returns)
    Warp 9:      Load (both CTAs issue TMA multicast; leader calls expect_bytes)
    Warps 10-11: Spare (no-op)
"""

from std.math import align_up, ceildiv, min
from std.sys import size_of
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx,
    warp_id,
)
from max.gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import external_memory, fence_mbarrier_init
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    MBarType,
    elect,
    kv_sub_tile_rows,
    o_store_tma_blocks_per_op,
)
from nn.attention.gpu.nvidia.common import (
    get_seq_info,
    KVTMATile,
    MHAPosition,
    NullPointer,
    OptionalPointer,
    Pack,
    PositionSummary,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHATileScheduler,
    SeqInfo,
)
from nn.attention.mha_utils import (
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .correction_warp import depth512_correction
from .load_warp import depth512_load
from .mma_warp import depth512_mma
from .smem import Depth512AttentionSMem
from .softmax_warp import depth512_softmax


struct SM100MHADepth512[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: Depth512SM100Config[KVLUTType.dtype],
    ValidLengthType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](TrivialRegisterPassable):
    """Implements pair-CTA SM100 (Blackwell) multi-head attention prefill for depth=512.

    Dispatches the 12-warp (3 warp-group) schedule across two cooperating CTAs:
    softmax, correction, MMA, load, and spare warps each run a specialized
    sub-kernel that shares a single `Depth512AttentionSMem` allocation.

    Parameters:
        KVLUTType: MHA operand describing the KV cache lookup table; its
            `dtype` is the Q, K, V element type and its `page_size` drives
            the KV sub-tile row counts.
        output_type: Output `DType` of the attention result (O store).
        MaskType: Mask type applied to the attention score tiles, such as
            a causal mask.
        SchedulerType: Tile scheduler that assigns work tiles to CTAs and
            advances the per-CTA iteration state.
        config: Depth-512 SM100 tile and pipeline configuration
            (`Depth512SM100Config`), parameterized by the KV dtype. Holds
            `BM`, `BN`, head counts, depths, staging, and the TMEM/SMEM
            budget.
        ValidLengthType: Optional pointer type for the per-batch valid
            sequence lengths; when non-null the kernel runs in ragged
            mode.
        KVRowOffsetsType: Optional pointer type for the KV input row
            offsets; when non-null used to compute the per-batch KV
            sequence length.
        _is_cache_length_accurate: Whether the cache length is known to
            be accurate, letting the start position be zeroed in ragged
            mode.
        MaxSeqLenType: Optionally-static maximum sequence length type; a
            static value of 1 selects decoding mode (asserted false for
            depth-512).
        PartitionType: KV cache partition scheme; partitioning is
            asserted unsupported for the depth-512 pair-CTA kernel.
    """

    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32

    comptime cta_group = 2
    comptime BM = Self.config.BM  # 64 (d512) or 128 (d256) per CTA
    comptime PairBM = Self.BM * 2  # 128 (d512) or 256 (d256) across pair
    comptime BN = Self.config.BN
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime fuse_gqa = Self.config.fuse_gqa
    comptime BM_eff: Int = Self.config.BM_eff()
    comptime PairBM_mask: Int = Self.BM_eff * 2
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size
    comptime k_sub_BN: Int = kv_sub_tile_rows(Self.BN // 2, Self.page_size)
    comptime v_sub_BN: Int = kv_sub_tile_rows(Self.config.BK1, Self.page_size)

    comptime SmemType = Depth512AttentionSMem[Self.config]

    comptime PositionType = MHAPosition[
        Self.PairBM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.qk_depth,  # padded_qk_depth = qk_depth for depth512
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 1, 1))
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"sm100_mha_depth{Self.config.qk_depth}_{Self.qkv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM=Self.config.BM,
            depth=Self.config.qk_depth,
            group=Self.config.group,
            decoding=False,
            fuse_gqa=Self.fuse_gqa,
            num_qk_stages=Self.config.num_qk_stages,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN=Self.k_sub_BN,
            BK=Self.config.BK0,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN=Self.v_sub_BN,
            BK=Self.config.v_cols_per_cta,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            # O output store is row-major SWIZZLE_NONE (decoupled from the
            # swizzled Q/K/V/S/P buffers governed by `config.swizzle_mode`).
            TensorMapSwizzle.SWIZZLE_NONE,
            BM=Self.config.BM,
            BN=Self.config.ov_depth,
            middle_dim=Self.config.num_kv_heads if Self.fuse_gqa else Self.config.num_q_heads,
            group=Self.config.group if Self.fuse_gqa else 1,
            # Single issuer, no combine (depth_splits=1): full-depth box, one
            # batched rank-5 TMA. Must match dispatch.mojo's store.
            tma_blocks_per_op=o_store_tma_blocks_per_op[
                Self.output_type,
                TensorMapSwizzle.SWIZZLE_NONE,
                Self.config.ov_depth,
                Self.config.group if Self.fuse_gqa else 1,
                depth_splits=1,
            ](),
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            NullPointer[.float32],  # SinkType (unused for depth512)
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
    ):
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), Self.config.description()
        comptime assert (
            not Self.PartitionType.do_partition
        ), "Partitioning not supported with depth512 pair-CTA"

        var mask = pack.mask
        var valid_length = pack.valid_length
        var kv_input_row_offsets = pack.kv_input_row_offsets
        var max_seq_len = pack.max_seq_len
        var partition = pack.partition

        var smem = Self.SmemType()

        # Per-warpgroup register allocation. Depth-512 widens the per-WG
        # working set vs the depth ≤ 128 path (see `kernel.mojo`), so the
        # softmax (256) and correction (184) WGs get more registers than
        # the 192/88 split there; MMA + load warps run lean at "other"
        # (64), and the spare warps drop to the floor (24). Sum must fit
        # in the SM register budget; bump together if a path starts
        # spilling.
        comptime num_reg_softmax = 256
        comptime num_reg_correction = 184
        comptime num_reg_other = 64

        # ---- Initialization (per-CTA, then cluster sync) ----------------

        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
            # Initialize all barriers.
            Depth512MBars[Self.config.num_kv_stages, Self.config.split_o](
                smem.mbar_base()
            ).init(lane_idx=Int32(thread_idx.x))
        elif warp_idx == 1:
            # TMEM allocation (pair-CTA cooperative).
            tcgen05_alloc[Int32(Self.cta_group)](
                smem.tmem_addr_ptr(),
                UInt32(Self.config.sm100_tmem_cols),
            )
        elif warp_idx == 2:
            var e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        fence_mbarrier_init()
        cluster_sync()

        # Read the TMEM base from SMEM EXACTLY ONCE here, where the prologue
        # `cluster_sync()` (preceded by `tcgen05_alloc`'s SMEM store +
        # MEMBAR.ALL) has provably published it, and carry it in a register to
        # every consumer warp (softmax/correction/mma) as an explicit argument.
        # Do NOT let the consumers re-read `smem.tmem_addr_ptr()` in their
        # bodies: SASS showed the in-body re-reads (the P@V O-operand load deep
        # in the MMA loop) gated only on KV-pipeline barriers, not on the alloc
        # publish, so under SM co-residency with the TP `allreduce_1stage` grid
        # (graph capture) a re-read could observe a stale/pre-alloc slot value
        # -> garbage TMEM base -> invalid `UTCHMMA` operand ->
        # CUDA_ERROR_ILLEGAL_INSTRUCTION. Reading once post-barrier and passing
        # by register (matches the proven `SM100MHA2Q` FA4 structure) removes
        # every in-body slot reload. Value is identical to the old per-warp
        # reads (same published base), so single-shot is bit-identical.
        var tmem_addr: UInt32 = smem.tmem_addr_ptr()[]

        # ---- Warp dispatch -----------------------------------------------

        var cta_rank = UInt32(block_rank_in_cluster() % 2)

        if warp_idx < 4:
            # Softmax warp group (warps 0-3, 128 threads).
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM_mask,
                Self.num_q_heads
                // Self.group if Self.fuse_gqa else Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )

            # Compute per-CTA output write parameters.
            var gmem_row = Self.PositionType.get_q_gmem_row[ragged=Self.ragged](
                seq_info, max_seq_len.as_uint32()
            )
            var out_row_idx = gmem_row + cta_rank * UInt32(Self.BM_eff)
            var out_head_idx = seq_info.head_idx
            var num_output_rows = min(
                Int32(seq_info.seq_len)
                - Int32(seq_info.prompt_offset)
                - Int32(cta_rank) * Int32(Self.BM_eff),
                Int32(Self.BM_eff),
            )

            depth512_softmax[
                Self.MaskType,
                Self.qkv_type,
                Self.output_type,
                Self.config,
                Self.page_size,
            ](
                smem,
                tmem_addr,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
                scale.cast[Self.accum_type](),
                ragged_tma_store,
                num_output_rows,
                out_head_idx,
                out_row_idx,
            )

        elif warp_idx < 8:
            # Correction warp group (warps 4-7, 128 threads).
            warpgroup_reg_alloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM_mask,
                Self.num_q_heads
                // Self.group if Self.fuse_gqa else Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)
            if not seq_info.is_valid():
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )
            depth512_correction[
                Self.MaskType,
                Self.qkv_type,
                Self.config,
                Self.page_size,
            ](
                smem,
                tmem_addr,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
            )

        elif warp_idx == 8:
            # MMA warp (single warp; leader issues pair-CTA MMA, peer
            # early-returns inside depth512_mma).
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM_mask,
                Self.num_q_heads
                // Self.group if Self.fuse_gqa else Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                tcgen05_dealloc[Int32(Self.cta_group)](
                    tmem_addr, UInt32(Self.config.sm100_tmem_cols)
                )
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )
            depth512_mma[
                Self.MaskType,
                Self.qkv_type,
                Self.config,
                Self.page_size,
            ](
                smem,
                tmem_addr,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
            )

        elif warp_idx == 9:
            # Load warp (single warp; both CTAs issue TMA multicast).
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM_mask,
                Self.num_q_heads
                // Self.group if Self.fuse_gqa else Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )
            # Hoist the pair-CTA rank branch to dispatch to two
            # comptime specializations of `depth512_load` (is_leader=
            # True for the even CTA, False for the odd one). Mirrors
            # the pattern at `sm100/kernel.mojo:447-483`.
            if cta_rank == UInt32(0):
                depth512_load[
                    Self.KVLUTType,
                    Self.MaskType,
                    Self.qkv_type,
                    Self.config,
                    Self.ValidLengthType,
                    Self._is_cache_length_accurate,
                    Self.MaxSeqLenType,
                    is_leader=True,
                ](
                    smem,
                    pos.score_row,
                    pos.num_keys,
                    seq_info,
                    max_seq_len,
                    mask,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    kv_lut,
                )
            else:
                depth512_load[
                    Self.KVLUTType,
                    Self.MaskType,
                    Self.qkv_type,
                    Self.config,
                    Self.ValidLengthType,
                    Self._is_cache_length_accurate,
                    Self.MaxSeqLenType,
                    is_leader=False,
                ](
                    smem,
                    pos.score_row,
                    pos.num_keys,
                    seq_info,
                    max_seq_len,
                    mask,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    kv_lut,
                )

        else:
            # Spare warps 10-11 (no-op). 24 is the floor for
            # `setmaxnreg.dec` on SM90+ — drop these warps' allocation
            # to the minimum so the active WGs claim their share of the
            # SM register file.
            warpgroup_reg_dealloc[24]()

    @staticmethod
    @always_inline
    def mask_status(
        mask: Self.MaskType,
        seq_id: UInt32,
        score_row: UInt32,
        kv_row: UInt32,
    ) -> TileMaskStatus:
        return mask.status(
            seq_id,
            Index[dtype=DType.int32](
                Int(score_row),
                Int(kv_row),
            ),
            Index[dtype=DType.int32](Self.PairBM_mask, Self.BN),
        )
