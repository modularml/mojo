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

"""Implements the SM100 (Blackwell) warp-specialized FlashAttention-4 multi-head attention kernel with the two-query (2Q) variant.
"""

from std.math import align_up
from std.sys import simd_width_of, size_of
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from max.gpu.primitives.warp import broadcast
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from max.gpu.memory import fence_mbarrier_init
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from layout.tma_async import RaggedTMA3DTile
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import FA4Config, MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    SM100TensorAccumulator,
    elect,
    kv_sub_tile_rows,
    o_store_tma_blocks_per_op,
)
from nn.attention.gpu.nvidia.common import (
    get_seq_info,
    KVTMATile,
    MHAPosition,
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
from linalg.arch.sm100.mma import smem_descriptor
from .smem import SM100AttentionSMem
from .softmax_warp import fa4_softmax
from .correction_warp import fa4_correction
from .load_warp import fa4_load
from .mma_warp import fa4_mma


struct SM100MHA2Q[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: FA4Config[KVLUTType.dtype],
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](TrivialRegisterPassable):
    """Implements the two-query (2Q) FlashAttention-4 forward attention kernel for NVIDIA SM100 GPUs.

    Bundles the comptime tile configuration, TMA operand types, and warp-specialized dispatch (softmax, correction, load, and MMA warps) that together compute scaled dot-product attention over a KV cache. When the configuration admits a type-compatible one-query (1Q) variant, short sequences are routed to the cheaper 1Q body at runtime.

    Parameters:
        KVLUTType: MHA operand describing the KV cache lookup table and dtype.
        output_type: Output dtype of the attention result.
        MaskType: Causal or unmasked attention mask type.
        SchedulerType: Tile scheduler controlling iteration over KV tiles.
        config: Comptime FA4 tile and pipeline configuration.
        ValidLengthType: Optional pointer type for valid sequence lengths.
        SinkType: Optional pointer type for attention sink weights.
        KVRowOffsetsType: Optional pointer type for KV input row offsets.
        _is_cache_length_accurate: Whether the cache length is known to be accurate.
        MaxSeqLenType: Optionally-static maximum sequence length type.
        PartitionType: KV cache partition scheme.
    """

    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32
    comptime simd_size: Int = simd_width_of[Self.qkv_type]()

    comptime pair_cta: Bool = Self.config.pair_cta
    comptime cta_group: Int = 2 if Self.pair_cta else 1
    # CTAs per launch cluster = cta_group (pair-CTA 2-SM width) * splitk
    # partitions. Drives the static `nvvm.cluster_dim` metadata. Distinct from
    # `cta_group` (the UMMA 1-SM/2-SM width, always 1 or 2): split-K is P
    # independent single-CTA kernels (cta_group==1) co-launched in a cluster.
    comptime cluster_size: Int = Self.config.cluster_size()
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.qk_depth
    comptime padded_depth = Self.config.padded_qk_depth
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime fuse_gqa = Self.config.fuse_gqa
    # BM_eff: sequence positions per full tile (BM // group when fusing)
    comptime BM_eff: Int = Self.config.BM_eff()
    # BM_mask: the BM value passed to mask functions.
    # For pair-CTA, use PairBM so both CTAs make identical skip decisions.
    comptime BM_mask: Int = Self.config.PairBM_eff()

    # Effective cross-stage P for THIS instantiation. This is the only site
    # that sees the config, the mask type and the tile geometry at once, so
    # it is where the last conjuncts land. Threaded identically into the
    # load, MMA and softmax warps -- they must agree or the K-ahead/inplace
    # handshake deadlocks.
    #
    # `config.crossp_on()` carries the config-level support matrix and drives
    # the SMEM/mbar allocation. The extra mask conjunct can only narrow it,
    # so the warps never use a barrier the layout did not allocate; an
    # unused pair of mbars is harmless, the reverse is ILLEGAL_ADDRESS.
    # `UNKNOWN_MASK` means the mask needs the runtime FULL_MASK slow path,
    # which the K-ahead producer in `load_warp` does not implement.
    # Store-shape conjunct for `softmax_warp`'s per-tile cross-P store, which
    # assumes exactly 4 full batches. Mirrors that function's own arithmetic
    # (`exp_simd == 2`) off this file's `UMMA1Type`.
    comptime _crossp_vs_len: Int = (Self.config.BN // Self.config.m_pack) // 2
    comptime _crossp_batch: Int = 32 if Self.config.num_pv_stages == 1 else (
        Self._crossp_vs_len
        // (4 if Self.UMMA1Type.use_3_then_1_split else Self.num_pv_stages)
    )
    comptime CrossPStoreShapeOk: Bool = (
        Self._crossp_batch > 0
        and (Self._crossp_vs_len % Self._crossp_batch) == 0
        and (Self._crossp_vs_len // Self._crossp_batch) == 4
    )

    comptime CrossPEffective: Bool = (
        Self.config.crossp_on()
        and Self.MaskType.nonfull_sets[Self.BM_mask, Self.config.BN]()[0]
        != TileMaskStatus.UNKNOWN_MASK
        and Self.CrossPStoreShapeOk
    )
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.MMA_M  # 128 single-CTA, 256 pair-CTA
    comptime qo_elements = Self.padded_depth * Self.HalfBM
    comptime qkv_dt_size = size_of[Self.qkv_type]()
    comptime HalfBM = Self.BM // 2

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulator[
        Self.qkv_type,
        Self.accum_type,
        MMA_M=Self.MMA_M,  # 128 single-CTA, 256 pair-CTA
        MMA_N=Self.BN,
        BK=align_up(Self.depth, Self.config.MMA_K),  # BK in memory depth
        a_tmem=False,
        swizzle_a=Self.config.swizzle_mode,
        swizzle_b=Self.config.swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
        cta_group=Self.cta_group,
    ]
    # Second MMA is P@V (V not staged, but P writing can be staged)
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulator[
        Self.qkv_type,
        Self.accum_type,
        MMA_M=Self.MMA_M,
        MMA_N=Self.config.padded_ov_depth,
        BK=Self.BN,
        a_tmem=True,
        swizzle_b=Self.config.swizzle_mode,
        transpose_b=False,
        num_stages=Self.num_pv_stages,
        cta_group=Self.cta_group,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = UInt32(
        Self.swizzle_granularity * Self.config.BN
    )
    comptime qo_bytes: UInt32 = UInt32(Self.qkv_dt_size * Self.qo_elements)
    comptime k_bytes: UInt32 = UInt32(Self.qkv_dt_size) * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = UInt32(
        Self.qkv_dt_size * Self.MMA_K * Self.config.padded_ov_depth
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.padded_qk_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    comptime SmemType = SM100AttentionSMem[Self.config]

    # TMA-op types as seen by `kernel`/`_kernel_impl`. Defined once so the 2Q
    # signatures and the 1Q-variant rebinds (see `kernel`) share a single
    # definition.
    comptime QTMAOpType = QTMATile[
        Self.KVLUTType.dtype,
        Self.config.swizzle_mode,
        BM=Self.config.BM // Self.config.num_q,
        depth=Self.config.qk_depth,
        group=Self.config.group,
        decoding=False,
        fuse_gqa=Self.fuse_gqa,
        num_qk_stages=Self.config.num_qk_stages,
    ]
    comptime KTMAOpType = KVTMATile[
        Self.KVLUTType.dtype,
        Self.config.swizzle_mode,
        BN=kv_sub_tile_rows(Self.config.k_rows_per_cta(), Self.page_size),
        BK=Self.config.BK0,
    ]
    comptime VTMAOpType = KVTMATile[
        Self.KVLUTType.dtype,
        Self.config.swizzle_mode,
        # V TMA box geometry per layout via `v_tma_box_rows()` /
        # `v_tma_box_cols()` -- the shared selector (see their docstrings).
        # Must match the dispatch `create_tma_tile` box + the `fa4_load`
        # signature.
        BN=Self.config.v_tma_box_rows(Self.page_size),
        BK=Self.config.v_tma_box_cols(),
    ]
    comptime OTMAStoreType = RaggedTMA3DTile[
        Self.output_type,
        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # Q/K/V/S/P buffers governed by `config.swizzle_mode`).
        TensorMapSwizzle.SWIZZLE_NONE,
        # 2Q: BM=128 (each WG writes one of two Q halves).
        # 1Q: BM=128 (both WGs cover the full BM=128 Q rows and write
        # disjoint depth-column ranges).
        BM=Self.config.BM // Self.config.num_q,
        BN=Self.config.ov_depth,
        middle_dim=Self.config.num_kv_heads if Self.fuse_gqa else Self.config.num_q_heads,
        group=Self.config.group if Self.fuse_gqa else 1,
        # Batched rank-5 O store (must match dispatch.mojo's store) for every
        # non-split config; the 1Q split-K (reduce-scatter) config uses the
        # PER-BLOCK (rank-3) store because each partition TMA-stores only its own
        # depth band via `async_copy_from_col` at a non-{0,half} offset (see the
        # matching conditional + rationale in dispatch.mojo).
        # WS (MMA_M=32) also uses the per-block WG0 egress (fa4_tma_store_o_smem),
        # so it takes the rank-3 store like the 1Q split-K path.
        tma_blocks_per_op=0 if (
            Self.config.splitk_partitions > 1 or Self.config.use_ws
        ) else o_store_tma_blocks_per_op[
            Self.output_type,
            TensorMapSwizzle.SWIZZLE_NONE,
            Self.config.ov_depth,
            Self.config.group if Self.fuse_gqa else 1,
            depth_splits=2,
        ](),
    ]
    comptime PackType = Pack[
        Self.MaskType,
        Self.SchedulerType,
        Self.ValidLengthType,
        Self.SinkType,
        Self.KVRowOffsetsType,
        Self.MaxSeqLenType,
        Self.PartitionType,
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
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](Int32(Self.cluster_size), 1, 1)
    )
    @__name(
        t"sm100_mha_{Self.config.num_q}q_depth{Self.config.qk_depth}_{Self.qkv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        q_tma_op: Self.QTMAOpType,
        k_tma_op: Self.KTMAOpType,
        v_tma_op: Self.VTMAOpType,
        ragged_tma_store: Self.OTMAStoreType,
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Self.PackType,
        # Total query-row count; used only by the workspace (traditional/unfused)
        # split-K egress as the per-partition `o_partial`/`lse_partial` row
        # stride. Ignored by every non-workspace config.
        num_rows_q: UInt32,
    ):
        # Static-cluster entry: the `nvvm.cluster_dim` metadata above bakes a
        # *required* cluster size into the kernel. This covers every config:
        # pair-CTA (2-SM width), non-split (size 1), AND num_q==1 split-K, which
        # is compiled once per static partition count `P` (cluster size `P`) and
        # selected at dispatch (see `mha_sm100_dispatch`).
        Self._entry_body(
            q_tma_op,
            k_tma_op,
            v_tma_op,
            ragged_tma_store,
            kv_lut,
            scale,
            batch_size,
            num_keys_arg,
            pack,
            num_rows_q,
        )

    @staticmethod
    @always_inline
    def _entry_body(
        q_tma_op: Self.QTMAOpType,
        k_tma_op: Self.KTMAOpType,
        v_tma_op: Self.VTMAOpType,
        ragged_tma_store: Self.OTMAStoreType,
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Self.PackType,
        num_rows_q: UInt32,
    ):
        # When this 2Q config admits a type-compatible 1Q variant
        # (`can_switch_to_1q`), route any tile whose REMAINING valid rows fit a
        # single 1Q tile through the cheaper 1Q body:
        # `seq_len - prompt_offset <= Kernel1Q.BM_eff`.
        #
        # That width is LOAD-BEARING — do not narrow it to `prompt_offset == 0`.
        # It equals `softmax_warp`'s `wg_row_offset_seq`, so the predicate is
        # precisely "WG1's output half is empty"; the 2Q caller passes
        # `output_nonempty=can_switch_to_1q()` on the strength of it and the WS
        # epilogue then statically discharges its `num_output_rows > 0` guard.
        # Restricted to whole prompts, every 2Q tile with an empty second half
        # loses that guard and writes 128 rows past the end of the sequence.
        #
        # A tile with `prompt_offset > seq_len` underflows the UInt32 subtract,
        # fails the predicate, and falls to the 2Q body's `is_valid()` early-out
        # — correct, but by wraparound.
        var seq_info: SeqInfo = get_seq_info[
            Self.BM_mask,
            Self.config.num_kv_heads if Self.fuse_gqa else Self.num_q_heads,
            Self.MaskType.get_type_name() == "CausalMask",
            pair_cta=Self.pair_cta,
            splitk_partitions=UInt32(Self.config.splitk_partitions),
        ](batch_size, pack.max_seq_len, pack.valid_length, pack.partition)

        comptime if Self.config.can_switch_to_1q():
            comptime Kernel1Q = SM100MHA2Q[
                Self.KVLUTType,
                Self.output_type,
                Self.MaskType,
                Self.SchedulerType,
                Self.config.switch_1q_config(),
                Self.ValidLengthType,
                Self.SinkType,
                Self.KVRowOffsetsType,
                Self._is_cache_length_accurate,
                Self.MaxSeqLenType,
                Self.PartitionType,
            ]
            if broadcast(seq_info.seq_len - seq_info.prompt_offset) <= UInt32(
                Kernel1Q.BM_eff
            ):
                # The TMA ops are typed from the 2Q `Self.config`; the 1Q types
                # fold to identical values (`can_switch_to_1q()` guarantees
                # matching `num_qk_stages` ⇒ matching `BK0`; per-half BM=128,
                # BN/depth/group/swizzle already match), but the parser sees
                # distinct parameter expressions, so `rebind`.
                Kernel1Q._kernel_impl(
                    rebind[Kernel1Q.QTMAOpType](q_tma_op),
                    rebind[Kernel1Q.KTMAOpType](k_tma_op),
                    rebind[Kernel1Q.VTMAOpType](v_tma_op),
                    rebind[Kernel1Q.OTMAStoreType](ragged_tma_store),
                    kv_lut,
                    scale,
                    num_keys_arg,
                    pack,
                    seq_info,
                    num_rows_q,
                )
                return
        Self._kernel_impl(
            q_tma_op,
            k_tma_op,
            v_tma_op,
            ragged_tma_store,
            kv_lut,
            scale,
            num_keys_arg,
            pack,
            seq_info,
            num_rows_q,
        )

    @staticmethod
    @always_inline
    def _kernel_impl(
        q_tma_op: Self.QTMAOpType,
        k_tma_op: Self.KTMAOpType,
        v_tma_op: Self.VTMAOpType,
        ragged_tma_store: Self.OTMAStoreType,
        kv_lut: Self.KVLUTType,
        scale: Float32,
        num_keys_arg: UInt32,
        pack: Self.PackType,
        seq_info: SeqInfo,
        num_rows_q: UInt32,
    ):
        comptime assert (
            Self.MMA_M == 32
            or Self.MMA_M == 64
            or Self.MMA_M == 128
            or Self.MMA_M == 256
        )
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
        # The dynamic-smem carveout reserved at launch is `config.smem_used`
        # (`launch_smem_used()`), so it must be at least the real
        # `SM100AttentionSMem` byte footprint that the kernel actually writes.
        # Under-reserving (smem_used < smem_size) is an out-of-bounds __shared__
        # write bug (the trailing mbar / tmem_addr regions overflow the carveout
        # on init); over-reserving is safe. Equality is not required: the 2Q
        # fused-KV path legitimately over-reserves a few bytes.
        comptime assert (
            Self.config.smem_used >= Self.SmemType.smem_size()
        ), String(
            "config.smem_used = ",
            Self.config.smem_used,
            " must be >= SmemType.smem_size() = ",
            Self.SmemType.smem_size(),
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        var mask = pack.mask
        var sink_weights = pack.sink_weights
        var kv_input_row_offsets = pack.kv_input_row_offsets
        var max_seq_len = pack.max_seq_len

        comptime num_q = Self.config.num_q
        # TODO: We may want to support num_q>2 for depth=64?
        comptime assert (
            num_q == 1 or num_q == 2
        ), "Currently only support num_q == 1 or 2"
        var smem = Self.SmemType()
        var misc_mbars = smem.misc_mbars()

        # Per-warpgroup register allocation, mirroring CUTLASS's
        # `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`.  Softmax gets
        # the largest slice (192), correction the next (88), and the
        # MMA-leader/other path runs lean (40); inactive warps drop to
        # the minimum (24).  Sum × WG size must stay ≤ the SM register
        # file; bump together if a path starts spilling.
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 88
        comptime num_reg_other = 40

        # The 2Q FA4 body supports the traditional (workspace) split-K partition
        # scheme ONLY at `splitk_partitions == 1` (no launch cluster, no in-kernel
        # DSMEM combine): each partition CTA runs the ordinary single-partition
        # path and writes to a per-partition global workspace, merged by a separate
        # combine kernel. The cluster/DSMEM split-K (`splitk_partitions > 1`) uses
        # the in-kernel reduce-scatter and is incompatible with a `do_partition`
        # scheme; decoding is likewise unsupported (and is blocked at dispatch).
        comptime assert not (
            Self.PartitionType.do_partition
            and Self.config.splitk_partitions > 1
        ), (
            "The 2-q FA4 implementation supports a partitioning scheme only"
            " with splitk_partitions == 1 (traditional workspace split-K);"
            " cluster split-K (splitk_partitions > 1) and decoding are not"
            " supported with a partitioning scheme."
        )
        # The workspace egress lives on the 1Q store paths only: the 2Q store
        # (`softmax_warp.mojo`'s `config.num_q == 2` branch) neither shifts its
        # ragged row by `ws_o_row_off` nor writes the per-row LSE, so pairing a
        # 2Q config with a partitioning scheme would silently drop a partition's
        # results instead of failing. Unreachable today (every workspace
        # instantiation is 1Q), so this is a fence, not a behavior change.
        comptime assert not (
            Self.PartitionType.do_partition and Self.config.num_q == 2
        ), (
            "workspace split-K egress is 1Q-only: the 2Q store path drops"
            " ws_o_row_off and the per-row LSE write."
        )

        var warp_idx = UInt32(warp_id[broadcast=True]())
        # Range-led nest (like the `warp_idx < 8 / < 12 / == 13 / == 12` dispatch
        # below) rather than a flat `== 0 / == 1 / == 2` chain: three contiguous
        # equality cases get lowered by ptxas to a constant-memory jump table
        # (`LDC c[0x2]` + `BRX`) in the 2Q kernel (which inlines both the 1Q and 2Q
        # bodies); leading with `< 2` keeps every level to <= 2 equality cases so the
        # prologue dispatch stays a uniform predicate-branch chain. Same warp -> task
        # mapping: warp 0 inits barriers, warp 1 allocates TMEM, warp 2 prefetches TMA.
        if warp_idx < 2:
            if warp_idx == 0:
                # Initialize all barriers (S/C/order/Q1Sync/K/V/O) in one call
                misc_mbars.init(lane_idx=Int32(thread_idx.x))
                # BLASST: zero the skip-vote region ("don't skip") before it's
                # published CTA-wide; the peel writes no vote, so this makes the
                # peel's P@V never skip.
                comptime if Self.SmemType.blasst_vote_slots > 0:
                    var blasst_vote = smem.blasst_vote_smem()
                    var blasst_lane = UInt32(thread_idx.x)
                    if blasst_lane < UInt32(Self.SmemType.blasst_vote_slots):
                        blasst_vote[blasst_lane] = UInt8(0)
            else:  # warp_idx == 1
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

        # Cluster (pair-CTA or 1Q split-K): a `cluster_sync` (after
        # `fence_mbarrier_init`) guarantees every CTA has finished initializing
        # its barriers before any peer arrives on them cross-cluster. Pair-CTA
        # needs this so both CTAs see each other's barriers; 1Q split-K needs it
        # so the split-K publish mbarrier is init-visible before a peer's
        # `arrive_cluster` (an arrive-before-init silently hangs). Plain
        # single-CTA uses a plain `barrier()`.
        comptime cluster_discipline = Self.pair_cta or (
            Self.config.splitk_partitions > 1
        )
        comptime if cluster_discipline:
            fence_mbarrier_init()
            cluster_sync()
        else:
            barrier()

        # Read the TMEM base from SMEM EXACTLY ONCE here, post-barrier (the
        # alloc on warp 1 + this barrier publish it), and carry it by register
        # to every consumer (fa4_softmax / fa4_correction / fa4_mma). Mirrors
        # the depth-512 fix: the consumers must NOT re-read
        # `smem.tmem_addr_ptr()` in their bodies, because an in-body slot reload
        # gated only on a pipeline barrier (not the alloc publish) can observe a
        # stale/pre-alloc value under SM co-residency and feed a garbage TMEM
        # operand to `UTCHMMA`. Same published value -> single-shot bit-identical.
        var tmem_addr: UInt32 = smem.tmem_addr_ptr()[]

        # Programmatic Dependent Launch (PDL).  This is the only point every
        # thread of every CTA reaches before the warp-specialized early
        # returns below (invalid tiles bail in warps 0-13 while warps 14-15
        # fall through), so it is the only divergence-free place to honor the
        # contract that *every* CTA signal launch-dependents — otherwise a
        # back-to-back consumer grid's `wait` hangs (see MLA decode).  The
        # data-independent prologue above (barrier init, tmem alloc, TMA
        # descriptor prefetch) overlaps the predecessor grid's tail; `wait`
        # fences here before the data-dependent Q/K/V loads in `fa4_load`;
        # `launch` lets the successor grid's prologue overlap our compute.
        comptime if MHA_PDL_LEVEL > PDLLevel.OFF:
            wait_on_dependent_grids()
            # `do_partition` (workspace/unfused split-K) feeds a SEPARATE
            # combine consumer that reads `o_partial`/`lse_partial` only AFTER
            # this grid's egress store. A prologue launch-dependents would
            # release that consumer's `wait_on_dependent_grids()` at our START
            # (before the store) -> stale read. Suppress it for `do_partition`;
            # the combine's `wait` then releases on this grid's COMPLETION
            # (which orders after the store). Every other config (writes its
            # final output directly, no split-K combine) keeps the prologue
            # launch-dependents unchanged.
            comptime if not Self.PartitionType.do_partition:
                launch_dependent_grids()

        # Workspace (traditional/unfused) split-K knobs forwarded to the warps.
        # `ws_split` is comptime-true only for a `do_partition` scheme (which the
        # kernel restricts to `splitk_partitions == 1`); it enables the runtime
        # KV windowing in the load/mma/correction warps. The softmax warp derives
        # the same predicate from `ws_lse`'s type instead of being told, so its
        # egress cannot be enabled without a buffer to write to.
        comptime assert Self.PartitionType.do_partition == (
            not Self.PartitionType.LSEPointerType.is_null
        ), (
            "a partitioning scheme must own an LSE buffer and a"
            " non-partitioning one must not: `do_partition` and"
            " `LSEPointerType` are one fact"
        )
        comptime ws_split = Self.PartitionType.do_partition
        var ws_np: UInt32 = pack.partition.num_partitions()
        var ws_lse = pack.partition.lse_pointer()

        # warp group partitioning
        # Two QO:
        #
        # Pair-CTA AND 1Q split-K both run as a thread-block cluster and end on
        # a terminal `cluster_sync()` (a cluster-wide barrier every thread of
        # every CTA must reach). So their per-warp invalid-tile early returns are
        # replaced with fall-through; otherwise some warps return early while the
        # rest block at the terminal `cluster_sync()` forever. Plain single-CTA
        # (not pair-CTA, `splitk_partitions == 1`) keeps the early returns.
        # Within a cluster all P CTAs share `block_idx.x // cluster_size` -> the
        # same tile -> the same validity, so invalid clusters are all-or-none and
        # every CTA reaches the terminal sync together.
        # (`cluster_discipline` is defined above, at the prologue barrier.)
        if warp_idx < 8:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()

            comptime if not cluster_discipline:
                if not seq_info.is_valid():
                    return

            if seq_info.is_valid():
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

                fa4_softmax[
                    Self.KVLUTType,
                    Self.config,
                    Self.ValidLengthType,
                    Self.SinkType,
                    Self._is_cache_length_accurate,
                    Self.MaxSeqLenType,
                    # `kernel` routes tiles with
                    # `seq_len - prompt_offset <= Kernel1Q.BM_eff` to the
                    # 1Q body when the switch is compiled in, so the 2Q
                    # body's output halves are always non-empty.
                    output_nonempty=Self.config.can_switch_to_1q(),
                    crossp_effective=Self.CrossPEffective,
                ](
                    smem,
                    tmem_addr,
                    pos.score_row,
                    seq_info,
                    mask,
                    pos.num_keys,
                    scale.cast[Self.accum_type](),
                    max_seq_len.as_uint32(),
                    ragged_tma_store,
                    sink_weights,
                    ws_num_partitions=ws_np,
                    ws_lse_ptr=ws_lse,
                    ws_num_rows_q=num_rows_q,
                )

        elif warp_idx < 12:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            comptime if not cluster_discipline:
                if not seq_info.is_valid():
                    return

            if seq_info.is_valid():
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
                fa4_correction[
                    Self.config,
                    Self.page_size,
                    workspace_split=ws_split,
                ](
                    smem,
                    tmem_addr,
                    seq_info.prompt_idx,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                    ws_num_partitions=ws_np,
                )
        else:
            if warp_idx == 13:  # produce
                warpgroup_reg_dealloc[num_reg_other]()

                comptime if not cluster_discipline:
                    if not seq_info.is_valid():
                        return

                if seq_info.is_valid():
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
                    comptime if not Self.pair_cta:
                        fa4_load[
                            Self.config,
                            ValidLengthType=Self.ValidLengthType,
                            _is_cache_length_accurate=Self._is_cache_length_accurate,
                            is_leader=True,
                            workspace_split=ws_split,
                            crossp_effective=Self.CrossPEffective,
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
                            ws_num_partitions=ws_np,
                        )
                    else:
                        var cta_rank = block_rank_in_cluster() % 2
                        if cta_rank == 0:
                            fa4_load[
                                Self.config,
                                ValidLengthType=Self.ValidLengthType,
                                _is_cache_length_accurate=Self._is_cache_length_accurate,
                                is_leader=True,
                                crossp_effective=Self.CrossPEffective,
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
                            fa4_load[
                                Self.config,
                                ValidLengthType=Self.ValidLengthType,
                                _is_cache_length_accurate=Self._is_cache_length_accurate,
                                is_leader=False,
                                crossp_effective=Self.CrossPEffective,
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

            elif warp_idx == 12:  # Q @ K', P @ V
                warpgroup_reg_dealloc[num_reg_other]()

                comptime if not cluster_discipline:
                    if not seq_info.is_valid():
                        tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                        tcgen05_dealloc[Int32(Self.cta_group)](
                            tmem_addr, UInt32(Self.config.sm100_tmem_cols)
                        )
                        return
                var execute: Bool = seq_info.is_valid()
                comptime if Self.pair_cta:
                    # ---- Pair-CTA: leader-only guard ----
                    execute &= broadcast(block_rank_in_cluster()) % 2 == 0
                if execute:
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
                    fa4_mma[
                        Self.config,
                        page_size=Self.page_size,
                        workspace_split=ws_split,
                        crossp_effective=Self.CrossPEffective,
                    ](
                        smem,
                        tmem_addr,
                        seq_info.prompt_idx,
                        pos.score_row,
                        pos.num_keys,
                        mask,
                        ws_num_partitions=ws_np,
                    )
            else:
                # 24 is the floor for `setmaxnreg.dec` on SM90+ — drop
                # this warpgroup's allocation to the minimum so the
                # active WGs can claim its share of the SM register file.
                warpgroup_reg_dealloc[24]()

        # Cluster discipline (pair-CTA or 1Q split-K): a terminal cluster_sync
        # before dealloc so no CTA exits and breaks the cluster while a peer's
        # cross-CTA access is still in flight. Pair-CTA protects cluster-scoped
        # stmatrix; 1Q split-K protects the DSMEM peer reads of this CTA's smem
        # (M4 combine) -- it is now the SOLE cluster-wide fence guarding those
        # reads (the combine dropped its in-helper round-2 barrier by packing the
        # bf16 output into each partition's OWN-band dead f32 slice, which no peer
        # reads). All early returns above were converted to fall-through so every
        # thread reaches this sync point. TMEM dealloc is deferred here (out of
        # the warp bodies) for the same reason.
        comptime if cluster_discipline:
            cluster_sync()

            if warp_idx == 0:
                tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                tcgen05_dealloc[Int32(Self.cta_group)](
                    tmem_addr, UInt32(Self.config.sm100_tmem_cols)
                )

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
            Index[dtype=DType.int32](Self.BM_mask, Self.BN),
        )

    @staticmethod
    @always_inline
    def descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.BM // 2,
            BK=Self.config.BK0,
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,
        ](q_smem)
