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
"""Softmax warp group logic for depth=256/512 pair-CTA SM100 attention.

Computes online softmax over Q@K' scores (S in TMEM) and writes the
exponentiated result P to SMEM for SS MMA P@V consumption. Unlike FA4
where P lives in TMEM for TS MMA, this kernel must explicitly transfer
P from registers to SMEM with the correct swizzle layout.

Pair-CTA TMEM column layout (cta_group=2):
    For MMA output [BM, MMA_N]:
      Columns 0 : MMA_N//2       → TMEM rows 0..BM-1
      Columns MMA_N//2 : MMA_N   → TMEM rows BM..MMA_M-1

Depth-dependent behavior:
  split_o=True (d512, MMA_M=128, BM=64): Each M row is served by a thread
    pair (row m and row m+64). exchange_reduce combines cross-thread
    row_max/row_sum.
  split_o=False (d256, MMA_M=256, BM=128): Each thread covers a unique
    M-row with full BN columns. No exchange needed.
"""

from std.math import exp2, recip
from std.math.constants import log2e
from std.memory import bitcast
from std.sys import size_of
import max.gpu.primitives.warp as warp
from max.gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from max.gpu.memory import fence_async_view_proxy
from max.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    umma_arrive_leader_cta,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from max.gpu.primitives.cluster import block_rank_in_cluster
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAddress,
)
from layout import IntTuple
from layout.swizzle import make_swizzle
from layout.tensor_core_async import tile_layout_k_major
from layout.tma_async import RaggedTMA3DTile, SharedMemBarrier
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    SharedMemPointer,
    MBarType,
    TMemTile,
    llvm_opaque_tid,
    add_ftz,
    sub_ftz,
    mul_ftz,
    fma_ftz,
    max_ftz,
    maximum,
    apply_mask,
    peel_mask,
    scale_pack_o_row,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem


# Named barrier ID for softmax warp group cross-thread exchange.
# Must not conflict with barrier IDs used by other warp groups.
comptime _SOFTMAX_EXCHANGE_BARRIER: Int32 = 0


@always_inline
def depth512_scale_write_output[
    output_type: DType,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
    tma_bpo: Int = 0,
](
    tid: UInt32,
    m_row: UInt32,
    is_lower: Bool,
    inv_row_sum: Float32,
    smem: Depth512AttentionSMem[config=config],
    tmem_addr: UInt32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        TensorMapSwizzle.SWIZZLE_NONE,
        BM=config.BM,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # `tma_bpo` (blocks per batched TMA op) is inferred from the store arg:
        # full depth (one batched copy: rank-4 group==1, rank-5 group>1),
        # 0 => per-block (swizzled-output fallback).
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """Read O from TMEM, scale by inv_row_sum, write to SMEM, TMA store.

    split_o (d512): Two phases (O_lo, O_hi). Each thread processes ov_depth/4
      physical TMEM cols per phase, with is_lower determining output col base.
    !split_o (d256): Single phase. Each thread processes MMA_M*ov_depth/256
      physical TMEM cols. All threads write to col base 0.

    Parameters:
        output_type: DType of the output store to global memory.
        qkv_dtype: DType of the Q/K/V inputs; specializes the config.
        config: Depth512 SM100 kernel config struct (tile sizes, split_o).
        tma_bpo: Blocks per batched TMA op; 0 selects the per-block
            swizzled-output fallback, nonzero selects a full-depth batched
            copy (defaults to 0).

    Args:
        tid: Thread ID within the softmax warpgroup.
        m_row: M-row index within the BM tile (0..BM-1).
        is_lower: True for the lower half of paired threads in split_o
            mode; selects the O column base.
        inv_row_sum: Reciprocal of the total row sum used to normalize O.
        smem: Shared-memory allocator holding the O buffer.
        tmem_addr: Base TMEM address for the O accumulator tiles.
        ragged_tma_store: Ragged TMA descriptor for the output store.
        num_output_rows: Dynamic output row count for the TMA store; 0
            skips the store.
        out_head_idx: Output head index passed to the TMA store.
        out_row_idx: Output row index passed to the TMA store.
    """
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    # Physical TMEM cols per O phase.
    comptime o_cols_per_phase = (config.ov_depth // 4) if config.split_o else (
        config.MMA_M * config.ov_depth // 256
    )
    comptime batch_size = 16
    comptime num_batches = o_cols_per_phase // batch_size
    comptime assert o_cols_per_phase % batch_size == 0

    # `tmem_addr` passed in by register (read once post-`cluster_sync` in the
    # kernel prologue); do NOT re-read `smem.tmem_addr_ptr()` here.

    # Output SMEM base (reuses Q buffer).
    var o_smem = smem.o_smem[output_type]()
    # O SMEM is row-major (SWIZZLE_NONE): the gmem output is row-major and the
    # O accumulator is loaded one-row-per-thread, so no swizzle is needed and
    # the per-row writes stay bank-conflict-free (8 rows * 16 B = 128 B = all
    # 32 banks once). O's TMA store swizzle is decoupled from `config.swizzle_mode`
    # (which still governs the swizzled Q/K/V/S/P buffers).
    comptime o_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
    # O SMEM must match tile_layout_k_major[BM, ov_depth] for TMA store.
    # Decompose col into k-block + inner offset; SWIZZLE_NONE makes the inner
    # swizzle the identity, so the layout is plain row-major within each k-block.
    comptime o_swizzle = make_swizzle[output_type, o_swizzle_mode]()
    comptime o_sw_K = o_swizzle_mode.bytes() // size_of[output_type]()
    # ov_depth is a multiple of o_sw_K for every supported head size.
    comptime n_blocks = config.ov_depth // o_sw_K
    comptime batched = tma_bpo > 0
    comptime if batched:
        comptime assert (
            tma_bpo == n_blocks
        ), "batched depth512 store expects a full-depth box (single issuer)."

    # ---- Helper: load from TMEM, scale, write to SMEM --------------------
    @__parameter
    @always_inline
    def read_scale_write(
        o_tmem: TmemAddress,
        col_base: Int,
    ):
        comptime for b in range(num_batches):
            comptime col_offset = b * batch_size
            var o_vals = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=batch_size,
                dtype=accum_dtype,
                pack=False,
                width=batch_size,
            ]((o_tmem + col_offset).addr)

            # Scale+pack each group of 8 into one 16 B row-major (SWIZZLE_NONE)
            # store (f32x2 compute, wide store; see scale_pack_o_row).
            comptime for g in range(batch_size // 8):
                comptime base = g * 8
                var packed = scale_pack_o_row[output_type, w=8, start=base](
                    o_vals, inv_row_sum
                )

                var col = col_base + col_offset + base
                var o_k_block = col // o_sw_K
                var o_inner = Int(m_row) * o_sw_K + col % o_sw_K
                (o_smem + o_k_block * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
                    UInt32
                ]().store(packed)

    comptime if config.split_o:
        comptime ov_quarter = config.ov_depth // 4
        comptime ov_half = config.ov_depth // 2
        var o_lo_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O))
        var o_hi_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O_hi))
        # Lower threads write first half cols, upper threads write second half.
        var col_base_lo: Int = 0 if is_lower else ov_quarter
        var col_base_hi: Int = ov_half if is_lower else (ov_half + ov_quarter)
        read_scale_write(o_lo_tmem, col_base_lo)
        read_scale_write(o_hi_tmem, col_base_hi)
    else:
        var o_tmem = TmemAddress(tmem_addr + UInt32(config.TMEM_O))
        read_scale_write(o_tmem, 0)

    # Sync all 128 softmax threads before TMA store.
    # Reuse barrier ID 0 (safe: softmax exchange loop is complete).
    named_barrier[Int32(WARPGROUP_SIZE)](_SOFTMAX_EXCHANGE_BARRIER)

    # TMA store: one elected thread issues all column-chunk stores.
    var e = elect()
    var local_warp_idx = (tid // UInt32(WARP_SIZE)) % 4

    if local_warp_idx == 0:
        fence_async_view_proxy()
        comptime if batched:
            # Single full-depth batched copy (no combine path). Covers fused GQA
            # too: the RaggedTMA3DTile selector merge keeps group>1 within the
            # 5D limit (rank-5; rank-4 for group==1).
            ragged_tma_store.async_copy_batched[0](
                o_smem,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
        else:
            # tma_bpo == 0: swizzled-output fallback -> one per-block TMA each.
            comptime for col in range(n_blocks):
                ragged_tma_store.async_copy_from_col[col](
                    o_smem,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                    elect=e,
                )
        cp_async_bulk_commit_group()

    # Wait for all TMA stores to complete.
    cp_async_bulk_wait_group[0]()


@always_inline
def depth512_softmax[
    MaskType: MHAMask,
    qkv_dtype: DType,
    output_type: DType,
    config: Depth512SM100Config[qkv_dtype],
    page_size: Int,
](
    smem: Depth512AttentionSMem[config=config],
    tmem_addr: UInt32,
    seq_id: UInt32,
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
    scale: Float32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        TensorMapSwizzle.SWIZZLE_NONE,
        BM=config.BM,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # Inferred from the store the kernel built; forwarded to the writeback
        # helper, which infers its own `tma_bpo` from this arg.
        tma_blocks_per_op=_,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """Runs the online softmax warp group for pair-CTA SM100 attention.

    Loads Q@K' scores (S) from TMEM in pipelined batches, applies the causal
    mask, computes a running row max and exponentiated probabilities P, writes
    P to SMEM for the P@V SS MMA, accumulates the row sum, and finally scales
    the O accumulator by the inverse row sum and TMA-stores the result to
    global memory. For split_o (d512) configs, cross-thread partial max and
    sum values are combined via correction SMEM; for d256 each thread owns a
    unique M row and no exchange is needed.

    Parameters:
        MaskType: Compile-time mask type for causal/attention masking.
        qkv_dtype: DType of the Q/K/V inputs; specializes the config.
        output_type: DType of the output store to global memory.
        config: Depth512 SM100 kernel config struct (tile sizes, split_o).
        page_size: KV cache page size in tokens, used for paged-attention
            masking.

    Args:
        smem: Shared-memory allocator holding S/P/O buffers and barriers.
        tmem_addr: Base TMEM address for S and O tiles (read once post
            cluster_sync).
        seq_id: Sequence index for mask evaluation.
        score_row: Row offset of the query tile within the sequence.
        num_keys: Number of valid key columns for masking.
        mask: Causal/attention mask applied to score batches.
        scale: Softmax scale factor (applied in the log2 domain).
        ragged_tma_store: Ragged TMA descriptor for the output store.
        num_output_rows: Dynamic output row count for the TMA store.
        out_head_idx: Output head index for the TMA store.
        out_row_idx: Output row index for the TMA store.
    """
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    # Per-thread S columns: split_o halves the columns (exchange_reduce combines),
    # !split_o each thread sees the full BN (both happen to be 128 currently).
    comptime effective_bn = BN // 2 if config.split_o else BN
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    comptime BM_eff: Int = config.BM_eff()
    comptime PairBM_mask = BM_eff * 2
    comptime f32x2 = SIMD[.float32, 2]

    # Batch size for pipelined TMEM loads and exp computation.
    comptime batch_size = 32
    comptime has_remainder = (effective_bn % batch_size) != 0
    comptime first_cols = (
        effective_bn % batch_size
    ) if has_remainder else batch_size

    comptime max_unroll = 8

    # ---- Thread identity -------------------------------------------------
    var tid = llvm_opaque_tid()
    var row = tid % 128  # TMEM row (0-127)
    var m_row = row % UInt32(BM)  # M row index
    # split_o (d512): is_lower distinguishes paired threads (row<64 vs >=64)
    # !split_o (d256): BM=128, so is_lower is always True (all threads unique)
    var is_lower = row < UInt32(BM)

    var cta_rank = block_rank_in_cluster() % 2
    var per_thread_score_row: UInt32
    comptime if fuse_gqa:
        per_thread_score_row = (
            score_row
            + UInt32(cta_rank) * UInt32(BM_eff)
            + m_row // UInt32(group)
        )
    else:
        per_thread_score_row = score_row + cta_rank * UInt32(BM) + m_row

    # Column offset into the full BN score tile.
    # split_o: lower→0, upper→effective_bn. !split_o: always 0.
    var col_offset: UInt32 = 0 if is_lower else UInt32(effective_bn)

    # ---- TMEM addresses --------------------------------------------------
    # `tmem_addr` passed in by register (read once post-`cluster_sync` in the
    # kernel prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var s_even_tmem = tmem_addr + UInt32(config.TMEM_S_even)
    var s_odd_tmem = tmem_addr + UInt32(config.TMEM_S_odd)

    # ---- Scale -----------------------------------------------------------
    var scale_log2e: Scalar[accum_dtype] = scale * log2e

    # ---- FP8 P-scale / lazy-rescale knob ---------------------------------
    # Fixed P scale for FP8-QKV only. Lifts the un-normalized
    # softmax probabilities P = exp2(score - row_max) out of the e4m3
    # subnormal floor before the fp8 cast that feeds the P@V SS MMA, reducing
    # PV-GEMM quantization error. Since the softmax uses exp2, the scale is
    # exactly an additive +bias in the exp2 argument (added raw, NOT
    # multiplied by scale_log2e). row_sum is accumulated from the SAME scaled
    # P and the output is normalized by 1/row_sum, so the scale cancels
    # exactly -- no explicit descale. This path has no sink term.
    #
    # `p_fp8_bias` and the lazy-rescale gate `rescale_threshold` are the same
    # knob (both in the exp2/log2 domain), linked as
    # `p_fp8_bias = 8 + rescale_threshold`:
    #   fp8 : rescale_threshold = -2,  p_fp8_bias = 6   (a 64x P lift out of
    #         the e4m3 subnormal floor)
    #   bf16: rescale_threshold = -8,  p_fp8_bias = 0   (no bias)
    # The fp8 threshold of -2 (bias 6) was chosen as the perf sweet spot: a
    # prefill sweep of the threshold magnitude gained ~6% and saturated at
    # T=2, and it is accuracy-neutral vs the prior x256 default (bit-identical
    # tail-stress regression test + byte-identical Gemma-4-31B fp8-KV 16k
    # e2e). The bias is applied ONLY inside `comptime if p_fp8_bias != 0:` so
    # the bf16 codegen is byte-identical. Overflow-safe: the lazy-rescale
    # gate (threshold -2) lets a non-rescaled tile's max lag the true max
    # by up to 2 (log2), so max P = exp2(2 + 6) = 256 < 448 (e4m3 max).
    comptime rescale_threshold: Scalar[accum_dtype] = Scalar[accum_dtype](
        -8
    ) if size_of[qkv_dtype]() >= 2 else Scalar[accum_dtype](-2)
    comptime p_fp8_bias: Scalar[accum_dtype] = 8 + rescale_threshold

    # ---- Barriers --------------------------------------------------------
    var mbars = Depth512MBars[config.num_kv_stages, config.split_o](
        smem.mbar_base()
    )
    var pipeline_s_even = mbars.consumer_s_even()
    var pipeline_s_odd = mbars.consumer_s_odd()
    var pipeline_c = mbars.producer_c()
    var po_lo = mbars.po_lo_mbar()

    # ---- SMEM pointers ---------------------------------------------------
    var p_smem = smem.p_smem()
    var correction_smem = smem.correction_smem()
    # P SMEM must match tile_layout_k_major[BM, BN] for the MMA descriptor.
    # The layout is hierarchical: BN/sw_K outer blocks of [BM, sw_K] each.
    # Decompose col into k_block + inner, swizzle only the inner part.
    comptime p_swizzle = make_swizzle[qkv_dtype, config.swizzle_mode]()
    comptime p_sw_K = config.swizzle_mode.bytes() // size_of[qkv_dtype]()

    # ---- S register buffer -----------------------------------------------
    # Holds effective_bn f32 values per thread (128 for both d256 and d512).
    var s = Array[Scalar[accum_dtype], effective_bn](uninitialized=True)

    # ---- Iteration bounds (must match MMA and load warps) ----------------
    var kv_row: UInt32 = mask.start_column[PairBM_mask, BN, page_size](
        seq_id, score_row
    )

    comptime mask_sets = MaskType.nonfull_sets[PairBM_mask, BN]()
    comptime mask_strategies = MaskType.mask_strategies[PairBM_mask, BN]()
    comptime num_sets = len(mask_sets)

    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        var mask_ends = mask.masked_set_ends[
            BM=PairBM_mask, BN=BN, page_size=page_size
        ](seq_id, score_row, num_keys)
        mask_iters[0] = mask_ends[0]
        comptime for i in range(1, num_sets):
            mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

    comptime assert num_sets >= 1 and num_sets <= 3

    # ---- Inner helpers ---------------------------------------------------

    @__parameter
    @always_inline
    def s_load[i: Int]() -> f32x2:
        return f32x2(s[2 * i], s[2 * i + 1])

    @__parameter
    @always_inline
    def s_store[i: Int](v: f32x2):
        s[2 * i] = v[0]
        s[2 * i + 1] = v[1]

    @__parameter
    @always_inline
    def mask_batch[
        N: Int, //, mask_strategy: MaskStrategy
    ](mut batch: Array[Scalar[accum_dtype], N], kv_col: UInt32):
        """Apply mask to a batch of score elements."""
        apply_mask[
            mask_strategy=mask_strategy,
            skip_scale=True,
        ](
            batch,
            mask,
            scale_log2e,
            prompt_idx=UInt32(0),
            q_head_idx=UInt32(0),
            kv_tile_start_row=Int32(kv_col),
            max_seq_len=num_keys,
            num_keys=Int32(num_keys),
            score_row=Int32(per_thread_score_row),
        )

    @__parameter
    @always_inline
    def exchange_reduce[
        op: StringLiteral,  # "max" or "add"
    ](partial_val: Float32) -> Float32:
        """Exchange partial value between paired threads via correction_smem.

        Uses 2 named_barrier syncs. correction_smem must be free (ensured
        by calling pipeline_c.acquire() before this function).
        """
        # Step 1: Lower half writes its partial value.
        if is_lower:
            correction_smem[m_row] = partial_val
        named_barrier[Int32(WARPGROUP_SIZE)](_SOFTMAX_EXCHANGE_BARRIER)

        # Step 2: Upper half reads partner, computes combined, writes back.
        var combined: Float32
        if not is_lower:
            var partner_val = correction_smem[m_row]
            comptime if op == "max":
                combined = max_ftz(partial_val, partner_val)
            else:
                combined = partial_val + partner_val
            correction_smem[m_row] = combined
        else:
            combined = partial_val
        named_barrier[Int32(WARPGROUP_SIZE)](_SOFTMAX_EXCHANGE_BARRIER)

        # Step 3: Lower half reads combined result.
        if is_lower:
            combined = correction_smem[m_row]
        return combined

    # ---- load_mask_max: pipelined S load + mask + max --------------------
    # Follows FA4 pattern: double-buffer TMEM loads across batches so that
    # masking + max of batch N overlaps with the TMEM load of batch N+1.

    @__parameter
    @always_inline
    def load_mask_max_impl[
        *, mask_strategy: MaskStrategy
    ](s_tmem: UInt32, kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        """Load effective_bn columns of S from TMEM, apply mask, compute partial
        row_max as a StaticTuple for reduction.

        Each warp pair loads from the correct TMEM address range:
        - Lower (warps 0-1): columns 0..effective_bn-1 at s_tmem
        - Upper (warps 2-3): columns effective_bn..BN-1 at s_tmem + effective_bn
        """
        # Base TMEM address: both row groups read the same physical columns.
        # Pair-CTA layout maps rows 0-63 → first logical N half, rows 64-127 →
        # second logical N half, but both use the same MMA_N/2 physical cols.
        var base_tmem = s_tmem
        # KV column base for masking.
        var kv_col_base = kv_row + col_offset

        # --- Pipelined load: load batch 0, start batch 1, process batch 0 ---
        var s0 = TMemTile[accum_dtype, BM, first_cols](base_tmem).load_async()

        var s1 = TMemTile[accum_dtype, BM, batch_size](
            base_tmem + UInt32(first_cols)
        ).load_async()

        mask_batch[mask_strategy=mask_strategy](s0, kv_col_base)
        var vrow_max = maximum[width=max_unroll](s0)
        comptime for _i in range(first_cols):
            s[_i] = s0[_i]

        comptime cols = effective_bn - first_cols + batch_size

        comptime for i in range(cols // (2 * batch_size)):
            comptime offset0 = first_cols + batch_size * (2 * i)
            comptime offset1 = first_cols + batch_size * (2 * i + 1)
            comptime offset2 = first_cols + batch_size * (2 * i + 2)

            comptime if offset1 >= effective_bn:
                # Last batch: s1 is already loaded, just process it.
                mask_batch[mask_strategy=mask_strategy](
                    s1, kv_col_base + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)
                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]
            else:
                # Load next batch (s2) while processing current (s1).
                var s2 = TMemTile[accum_dtype, BM, batch_size](
                    base_tmem + UInt32(offset1)
                ).load_async()
                mask_batch[mask_strategy=mask_strategy](
                    s1, kv_col_base + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)
                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]

                comptime if offset2 < effective_bn:
                    s1 = TMemTile[accum_dtype, BM, batch_size](
                        base_tmem + UInt32(offset2)
                    ).load_async()
                mask_batch[mask_strategy=mask_strategy](
                    s2, kv_col_base + UInt32(offset1)
                )
                vrow_max = maximum(s2, vrow_max)
                comptime for _i in range(batch_size):
                    s[offset1 + _i] = s2[_i]

        return vrow_max

    @__parameter
    @always_inline
    def init_load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> Float32:
        """Load S, mask, return partial max (scalar)."""
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](s_even_tmem, kv_row)
        )

    @__parameter
    @always_inline
    def load_mask_max[
        mask_strategy: MaskStrategy
    ](s_tmem: UInt32, kv_row: UInt32, old_max: Float32) -> Float32:
        """Load S, mask, return partial max combined with old_max."""
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](s_tmem, kv_row),
            old_max,
        )

    # ---- store_exp: compute exp, write P to SMEM, return row_sum ---------
    # Follows FA4 pattern: interleave score_to_logit ahead of exp2 via
    # score_to_logit_ratio, then write P to SMEM in batches.

    @__parameter
    @always_inline
    def store_exp(row_max: Float32) -> f32x2:
        comptime exp_simd = 2
        comptime vs_len = effective_bn // exp_simd
        comptime score_to_logit_ratio: Int = 4

        # fp8 P scale: fold +p_fp8_bias into the exp2 bias via a single fused
        # multiply-add, matching score_to_logit's fma_ftz:
        # fma_ftz(-row_max, scale, p_fp8_bias) = -m*scale + p_fp8_bias, so
        # score_to_logit = fma(score, scale, -m*scale + p_fp8_bias). ftz is
        # harmless here -- any fp32 subnormal is lost when P truncates to fp8
        # anyway. `p_fp8_bias` is the unified P-scale/lazy-rescale knob defined
        # in the outer scope; `comptime if p_fp8_bias != 0` keeps the bf16
        # expression byte-identical (no `+ 0.0` instruction emitted).
        var vscale = f32x2(scale_log2e)
        var vneg_max_scaled: f32x2
        comptime if p_fp8_bias != 0:
            vneg_max_scaled = fma_ftz(
                f32x2(-row_max), f32x2(scale_log2e), f32x2(p_fp8_bias)
            )
        else:
            vneg_max_scaled = f32x2(-row_max * scale_log2e)

        @__parameter
        @always_inline
        def score_to_logit(score: f32x2) -> f32x2:
            return fma_ftz(score, vscale, vneg_max_scaled)

        # Interleaved exp: score_to_logit runs ahead by score_to_logit_ratio.
        @__parameter
        @always_inline
        def exp_iter[idx: Int]():
            comptime if idx < vs_len // score_to_logit_ratio:
                comptime for i in range(score_to_logit_ratio):
                    comptime j = score_to_logit_ratio * idx + i
                    s_store[j](score_to_logit(s_load[j]()))
            s_store[idx](exp2(s_load[idx]()))

        # --- Process in batches, write P to SMEM after each ---
        comptime p_batch = vs_len // 4 if vs_len >= 128 else vs_len // 2
        comptime p_batch_elems = p_batch * exp_simd
        comptime num_p_batches = vs_len // p_batch
        comptime p_remainder = vs_len % p_batch
        comptime assert num_p_batches >= 1

        # Helper to write a range of exp values from s[] to P SMEM.
        comptime p_elems_per_store: Int = 16 // size_of[qkv_dtype]()
        comptime assert (
            16 % size_of[qkv_dtype]() == 0
        ), "P store byte width (16) must be a multiple of dtype size"

        @__parameter
        @always_inline
        def write_p_batch[start_elem: Int, num_elems: Int]():
            comptime assert num_elems % p_elems_per_store == 0, (
                "write_p_batch num_elems must be a multiple of the per-store"
                " element count (16/size_of[dtype])"
            )
            comptime for c in range(0, num_elems, p_elems_per_store):
                comptime base = start_elem + c

                @__parameter
                @always_inline
                def pack_vals[n: Int]() -> SIMD[qkv_dtype, n]:
                    var vec = SIMD[accum_dtype, n](0)
                    comptime for k in range(n):
                        vec[k] = s[base + k]
                    return vec.cast[qkv_dtype]()

                var vals = pack_vals[p_elems_per_store]()
                var col = Int(col_offset) + base
                var p_k_block = col // p_sw_K
                comptime assert effective_bn % p_sw_K == 0
                comptime r = base % p_sw_K
                var p_inner = Int(m_row) * p_sw_K + r
                (p_smem + p_k_block * BM * p_sw_K + p_swizzle(p_inner)).bitcast[
                    UInt32
                ]().store(bitcast[.uint32, 4](vals))

        # Batch 0: compute exp.
        comptime for idx in range(p_batch):
            exp_iter[idx]()
        write_p_batch[0, p_batch_elems]()

        # Remaining batches: compute exp, then write to P SMEM.
        comptime for b in range(1, num_p_batches):
            comptime offset = p_batch * b
            comptime el_offset = offset * exp_simd
            comptime for idx in range(offset, offset + p_batch):
                exp_iter[idx]()
            write_p_batch[el_offset, p_batch_elems]()

        comptime if p_remainder > 0:
            comptime offset = p_batch * num_p_batches
            comptime el_offset = offset * exp_simd
            comptime for idx in range(offset, offset + p_remainder):
                exp_iter[idx]()
            write_p_batch[el_offset, p_remainder * exp_simd]()

        # P is fully written to SMEM. Fence + signal PO_lo so the MMA warp
        # can start P@V as early as possible (before the row-sum reduction).
        # Cluster-scope arrive so both CTAs' signals reach the leader.
        fence_async_view_proxy()
        umma_arrive_leader_cta(po_lo)

        # Row sum: 4-way unrolled accumulation over exp values in s.
        var acc0 = s_load[0]()
        var acc1 = s_load[1]()
        var acc2 = s_load[2]()
        var acc3 = s_load[3]()
        comptime for i in range(4, vs_len, 4):
            acc0 = add_ftz(acc0, s_load[i]())
            acc1 = add_ftz(acc1, s_load[i + 1]())
            acc2 = add_ftz(acc2, s_load[i + 2]())
            acc3 = add_ftz(acc3, s_load[i + 3]())
        return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

    # ---- Peeled first iteration ------------------------------------------

    pipeline_s_even.wait()
    tcgen05_fence_after()

    var partial_max: Float32 = peel_mask[
        rebind[StaticTuple[MaskStrategy, num_sets]](mask_strategies),
        init_load_mask_max,
    ](mask_iters, kv_row)

    umma_arrive_leader_cta(pipeline_s_even.consumer_mbar())
    pipeline_s_even.step()

    # pipeline_c.acquire() passes immediately on first use (buffer free).
    pipeline_c.acquire()
    # split_o: exchange_reduce combines paired threads' partial values.
    # !split_o: each thread has the full row, no exchange needed.
    var row_max: Float32
    comptime if config.split_o:
        row_max = exchange_reduce["max"](partial_max)
    else:
        row_max = partial_max

    # Compute exp, write P to SMEM (signals PO_lo inside), get partial sum.
    var partial_sum = store_exp(row_max)
    var global_sum: Float32
    comptime if config.split_o:
        global_sum = exchange_reduce["add"](partial_sum.reduce_add())
    else:
        global_sum = partial_sum.reduce_add()
    var row_sum = f32x2(global_sum, 0)

    # ---- Main loop (alternating S_even / S_odd) --------------------------

    var o_phase: UInt32 = 0

    var s_cur_pipeline = pipeline_s_odd
    var s_cur_tmem = s_odd_tmem
    var s_nxt_pipeline = pipeline_s_even
    var s_nxt_tmem = s_even_tmem

    @__parameter
    @always_inline
    def main_loop_body[mask_strategy: MaskStrategy]():
        """One iteration of the main softmax loop."""
        var old_max = row_max

        # Wait for S, load, mask, compute partial max.
        s_cur_pipeline.wait()
        tcgen05_fence_after()
        partial_max = load_mask_max[mask_strategy](s_cur_tmem, kv_row, old_max)
        umma_arrive_leader_cta(s_cur_pipeline.consumer_mbar())
        s_cur_pipeline.step()

        # Exchange max (correction_smem free after acquire).
        pipeline_c.acquire()
        var new_row_max: Float32
        comptime if config.split_o:
            new_row_max = exchange_reduce["max"](partial_max)
        else:
            new_row_max = partial_max

        var diff = sub_ftz(old_max, new_row_max)
        diff = mul_ftz(diff, scale_log2e)

        var correction: Float32
        comptime if rescale_threshold < 0:
            # Per-lane predicate, not a warp vote: under `fuse_gqa`,
            # `per_thread_score_row` packs multiple distinct query rows
            # (heads and/or seq positions) into one warp's 32 `m_row`
            # values, so an OR here would leak a sibling row's rescale
            # trajectory into this one's.
            if diff < rescale_threshold:
                row_max = new_row_max
                correction = exp2(diff)
            else:
                correction = 1
        else:
            row_max = new_row_max
            correction = exp2(diff)

        # Compute exp, write P (signals PO_lo inside), exchange sum.
        partial_sum = store_exp(row_max)
        var local_sum: Float32
        comptime if config.split_o:
            local_sum = exchange_reduce["add"](partial_sum.reduce_add())
        else:
            local_sum = partial_sum.reduce_add()

        # Write correction factor for the correction warp.
        # split_o: only lower half writes to avoid double-write (paired threads).
        # !split_o: all threads have unique m_rows.
        comptime if config.split_o:
            if is_lower:
                correction_smem[m_row] = correction
        else:
            correction_smem[m_row] = correction
        pipeline_c.commit()

        row_sum = fma_ftz(row_sum, f32x2(correction), f32x2(local_sum, 0))
        o_phase ^= 1

        # Swap S buffers.
        var tmp_pipeline = s_cur_pipeline
        s_cur_pipeline = s_nxt_pipeline
        s_nxt_pipeline = tmp_pipeline
        var tmp_tmem = s_cur_tmem
        s_cur_tmem = s_nxt_tmem
        s_nxt_tmem = tmp_tmem

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        comptime for i in range(num_sets):
            comptime mask_strategy = mask_strategies[i]
            var iters: UInt32
            iters = warp.broadcast(mask_iters[i])
            while iters != 0:
                iters -= 1
                kv_row += UInt32(BN)
                main_loop_body[mask_strategy]()
    else:
        while True:
            kv_row += UInt32(BN)
            if kv_row >= num_keys:
                break
            var cur_mask_status = mask.status(
                seq_id,
                Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                Index[dtype=DType.int32](PairBM_mask, BN),
            )
            if cur_mask_status == TileMaskStatus.FULL_MASK:
                continue
            if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                main_loop_body[
                    MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
                ]()
            else:
                main_loop_body[MaskStrategy.OUT_OF_BOUNDS]()

    # ---- Post-loop: wait for final O and write output --------------------

    # Wait for the last P@V to complete.
    # split_o: O_mma_hi fires after O_mma_lo, so waiting O_mma_hi ensures both.
    # !split_o: only O_mma_lo exists.
    comptime o_mma_done_offset = Depth512MBars[
        config.num_kv_stages, config.split_o
    ].O_mma_hi_offset if config.split_o else Depth512MBars[
        config.num_kv_stages, config.split_o
    ].O_mma_lo_offset
    var o_mma_done_mbar: MBarType = mbars.mbar_base + o_mma_done_offset
    o_mma_done_mbar[].wait(o_phase)
    tcgen05_fence_after()

    # Final scaling: inv_row_sum = 1 / total_row_sum.
    var inv_row_sum = recip(row_sum.reduce_add())

    # Scale O from TMEM by inv_row_sum, write to SMEM, TMA store to global.
    if num_output_rows > 0:
        depth512_scale_write_output[output_type, qkv_dtype, config](
            tid,
            m_row,
            is_lower,
            inv_row_sum,
            smem,
            tmem_addr,
            ragged_tma_store,
            num_output_rows,
            out_head_idx,
            out_row_idx,
        )

    # TMEM deallocation: all other warps (correction, MMA, load) are done
    # with TMEM by this point. Only warp 0 needs to deallocate.
    if tid // UInt32(WARP_SIZE) == 0:
        tcgen05_release_allocation_lock[Int32(config.cta_group)]()
        tcgen05_dealloc[Int32(config.cta_group)](
            tmem_addr, UInt32(config.sm100_tmem_cols)
        )
