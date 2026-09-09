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
"""Softmax warp group logic for FA4 (SM100 Flash Attention)."""

from std.math import ceildiv, exp2, log2, recip, align_up
from std.math.constants import log2e
from std.memory import bitcast
from std.utils.numerics import min_or_neg_inf
from std.sys import size_of, get_defined_int, get_defined_bool
from std.sys.info import _accelerator_arch
import max.gpu.primitives.warp as warp
from max.gpu import block_idx
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.memory import fence_async_view_proxy
from max.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    umma_arrive_leader_cta,
)
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    load_cluster_smem,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from max.gpu.primitives.warp import _vote_nvidia_helper
from layout.swizzle import make_swizzle
from layout.tma_async import RaggedTMA3DTile
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
    EnableEarlyAdd,
    ExplicitTMEMCrossP,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    add_ftz,
    apply_mask,
    blasst_vote_unanimous,
    combine_pack_o_row,
    elect,
    exp2_emulation,
    fma_ftz,
    llvm_opaque_tid,
    max_ftz,
    maximum,
    MBarType,
    mul_ftz,
    pack_row,
    peel_mask,
    scale_pack_o_row,
    SharedMemPointer,
    SM100TensorAccumulator,
    splitk_partition_idx,
    splitk_window,
    splitk_num_partitions,
    st_shared_v4_b32,
    store_p_quadrant,
    sub_ftz,
    TMemTile,
)
from nn.attention.gpu.nvidia.common import (
    MHAPosition,
    NullPointer,
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from nn.attention.mha_utils import OptionallyStaticInt, _is_decoding
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .smem import SM100AttentionSMem


comptime f32x2 = SIMD[.float32, 2]


@always_inline
def fa4_scale_write_output[
    output_type: DType,
    //,
    config: FA4Config,
    output_swizzle_mode: TensorMapSwizzle,
    tma_bpo: Int,
    zero_fill: Bool = False,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    inv_row_sum: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    o_tmem_arg: TMemTile[
        DType.float32, config.BM // config.num_q, config.padded_ov_depth
    ],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        # `config.BM // config.num_q` is "rows this WG writes": 128 in
        # 2Q (BM=256, two WGs split rows) and 128 in 1Q (BM=128, one WG
        # writes the full set in the T==1 fast path; the multi-tile 1Q
        # path uses fa4_lse_combine_write instead). Same numeric value
        # in both modes.
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # `tma_bpo` (blocks per batched TMA op) is inferred from the store arg:
        # >0 => one batched copy per phase (rank-4 group==1, rank-5 group>1),
        # 0 => per-block (swizzled-output fallback).
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    comptime accum_dtype = DType.float32
    # Rows this WG writes: 128 in both 2Q (BM=256, two WGs split rows in half)
    # and 1Q (BM=128, one WG owns all rows in the T==1 fast path). Matches the
    # TMA store descriptor's BM.
    comptime BM = config.BM // config.num_q
    comptime ov_depth = config.ov_depth

    # O SMEM is row-major (SWIZZLE_NONE): the O accumulator is loaded
    # one-row-per-thread via `tcgen05_ld[datapaths=32]` (warp w, lane l -> row
    # 32*w + l; 4 warps cover 128 rows), exactly the row ownership the S
    # reductions use. So `inv_row_sum` is already the rescale factor for the
    # row this thread writes -- no `warp.shuffle_idx` -- and the per-row 16 B
    # stores stay bank-conflict-free (8 rows * 16 B = 128 B = all 32 banks
    # once). The inner swizzle is the identity for SWIZZLE_NONE, so each
    # k-block is plain row-major [BM, o_sw_K].
    comptime o_swizzle = make_swizzle[output_type, output_swizzle_mode]()
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    # The group-of-8 / 16 B store path requires a 2-byte output element
    # (8 elems * 2 B = 16 B = uint32x4). This is the same constraint the
    # previous `output_reg_to_smem_st_matrix` path enforced.
    comptime assert (
        size_of[output_type]() == 2
    ), "fa4_scale_write_output requires a 2-byte output dtype (bf16/f16)"

    # Output column count is aligned to the OUTPUT swizzle granularity
    # (o_sw_K, the SWIZZLE_NONE 16 B box = 8 bf16), NOT the QKV swizzle's
    # `padded_ov_depth` (which aligns to SWIZZLE_128B = 64 elems). ov_depth is
    # already a multiple of o_sw_K for every supported head size, so this is
    # exact and needs no padding (e.g. depth 72 -> 9 boxes, not 128/64=2).
    comptime o_sw_blocks = align_up(ov_depth, o_sw_K) // o_sw_K
    comptime batched = tma_bpo > 0
    comptime if batched:
        comptime assert (
            tma_bpo == (o_sw_blocks + 1) // 2
        ), "batched scale_write expects a half-depth (ceil(blocks/2)) box."

    var e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Each thread owns output row `local_row` (= tid % 128). Load that row from
    # TMEM for one o_sw_K-wide block, scale+pack (f32x2 compute, wide store; see
    # `scale_pack_o_row`), and write one 16 B row-major store.
    @__parameter
    @always_inline
    def write_block[blk: Int]():
        comptime col = blk * o_sw_K
        comptime if zero_fill:
            # Empty (all-masked) row: emit zeros without reading the
            # never-produced O accumulator in TMEM.
            var o_vals = Array[Scalar[accum_dtype], o_sw_K](
                fill=Scalar[accum_dtype](0)
            )
            var packed = scale_pack_o_row[output_type, w=o_sw_K](
                o_vals, inv_row_sum
            )
            var o_inner = Int(local_row) * o_sw_K
            (o_smem_arg + blk * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
                UInt32
            ]().store(packed)
        else:
            var o_vals = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=o_sw_K,
                dtype=accum_dtype,
                pack=False,
                width=o_sw_K,
            ](o_tmem_arg.tmem_addr + UInt32(col))

            var packed = scale_pack_o_row[output_type, w=o_sw_K](
                o_vals, inv_row_sum
            )

            # Block `blk` is one k-block [BM, o_sw_K]; col % o_sw_K == 0.
            var o_inner = Int(local_row) * o_sw_K
            (o_smem_arg + blk * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
                UInt32
            ]().store(packed)

    comptime if batched:
        # Single issuer, 2-phase pipeline: write the first half to smem and kick
        # off its batched TMA, then write the second half (which overlaps the
        # first TMA's copy) and kick off its TMA. The two halves touch disjoint
        # smem blocks, so there is no read/write hazard. The batched store now
        # also covers fused GQA (group > 1): the (middle_dim, rows) selector
        # merge in RaggedTMA3DTile keeps the descriptor within the 5D limit
        # (rank-4 for group==1, rank-5 for group>1); write_block is unchanged
        # (its BM already includes group).
        comptime for blk in range(tma_bpo):
            write_block[blk]()
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        if local_warp_idx == 0:
            fence_async_view_proxy()
            ragged_tma_store.async_copy_batched[0](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )

        comptime for blk in range(tma_bpo, o_sw_blocks):
            write_block[blk]()
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        if local_warp_idx == 0:
            fence_async_view_proxy()
            # col_start = tma_bpo; the box may overhang the last block
            # for odd o_sw_blocks -> masked off by the TMA.
            ragged_tma_store.async_copy_batched[tma_bpo](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
            cp_async_bulk_commit_group()
    else:
        # tma_bpo == 0: swizzled-output callers (e.g. an MLA variant with a
        # SWIZZLE_128B output store) can't use the blocked-smem batched box, so
        # fall back to one per-block TMA each. (Fused GQA with SWIZZLE_NONE now
        # takes the batched branch above.)
        comptime for blk in range(o_sw_blocks):
            write_block[blk]()
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        if local_warp_idx == 0:
            fence_async_view_proxy()
            comptime for blk in range(o_sw_blocks):
                ragged_tma_store.async_copy_from_col[blk](
                    o_smem_arg,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                    elect=e,
                )
            cp_async_bulk_commit_group()
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_lse_combine_write[
    output_type: DType,
    //,
    config: FA4Config,
    wg_j_offset: Int,
    iters_per_wg: Int,
    output_swizzle_mode: TensorMapSwizzle,
    tma_bpo: Int,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    own_o_tmem: TMemTile[.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[.float32, config.BM, config.padded_ov_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        # 1Q only: equals config.BM (= 128); kept as `config.BM //
        # config.num_q` for typewise consistency with the fa4_softmax
        # signature and kernel.mojo construction (which use the same
        # expression to give 128 in both 2Q and 1Q).
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # `tma_bpo` (blocks per batched TMA op) is inferred from the store arg:
        # >0 => this WG issues one batched copy over its half (rank-4 group==1,
        # rank-5 group>1), 0 => per-block.
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """LSE-combine two TMEM_O fragments and TMA-store a depth-column slice.

    1Q-only sibling of `fa4_scale_write_output`. Each WG handles a disjoint
    range `j in [wg_j_offset, wg_j_offset + iters_per_wg)` of swizzle-block
    columns. For each `j`, the WG loads both its own and the peer's TMEM_O
    fragments, combines them in registers via per-row scales
    (`final_scale_local` for own, `final_scale_peer` for peer), writes the
    combined output to the shared `o_smem_arg` at the `j` slot, then
    TMA-stores that slot to gmem. Both WGs target the same `BM` Q rows but
    disjoint depth columns, so smem and gmem regions never overlap.

    The caller must have already waited on both `pipeline_o0` and
    `pipeline_o1` producer barriers (and issued `tcgen05_fence_after()`)
    before invoking this helper, so the TMEM fragments are visible.
    """
    comptime assert config.num_q == 1

    comptime accum_dtype = DType.float32
    # 1Q: config.BM == 128 == config.BM // config.num_q (matches the TMA
    # store descriptor's BM and the shared o_smem tile extent).
    comptime BM = config.BM // config.num_q

    # O SMEM is row-major (SWIZZLE_NONE): O is loaded one-row-per-thread via
    # `tcgen05_ld[datapaths=32]`, so `final_scale_local`/`final_scale_peer`
    # are already this thread's row scales (no `warp.shuffle_idx`), the combine
    # is a pure per-thread register op, and the per-row 16 B stores stay
    # bank-conflict-free (8 rows * 16 B = 128 B = all 32 banks once). The inner
    # swizzle is the identity for SWIZZLE_NONE, so each k-block is plain
    # row-major [BM, o_sw_K].
    comptime o_swizzle = make_swizzle[output_type, output_swizzle_mode]()
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime assert (
        size_of[output_type]() == 2
    ), "fa4_lse_combine_write requires a 2-byte output dtype (bf16/f16)"

    # Each WG handles a disjoint range of o_sw_K-wide column blocks
    # [wg_j_offset, wg_j_offset + iters_per_wg). `iters` matches the caller's
    # `iters_total` and is the output column count aligned to the OUTPUT
    # swizzle granularity (o_sw_K), NOT the QKV swizzle's `padded_ov_depth`
    # (which aligns to SWIZZLE_128B = 64 elems). ov_depth is already a multiple
    # of o_sw_K for every supported head size, so this is exact. Under
    # SWIZZLE_NONE the block size is small (o_sw_K = 8 for bf16), so depth=64
    # yields iters=8 and both WGs participate; the caller's `if iters_per_wg1
    # > 0` guard still skips a WG only when its range is empty.
    comptime iters = align_up(config.ov_depth, o_sw_K) // o_sw_K
    comptime assert iters_per_wg >= 1, (
        "fa4_lse_combine_write requires at least one column block per"
        " call; the caller must skip WG1 when iters_per_wg would be 0."
    )
    comptime assert wg_j_offset + iters_per_wg <= iters

    # Batched: each WG issues ONE TMA over its block range (rank-4 for group==1,
    # rank-5 for group>1 fused GQA, after the RaggedTMA3DTile selector merge).
    # The box is the half-depth `ceil(iters/2)`; WG0 (wg_j_offset=0) fills it
    # exactly, WG1 (wg_j_offset=ceil) overhangs the last block for odd `iters`,
    # which the TMA masks off.
    comptime batched = tma_bpo > 0
    comptime if batched:
        comptime assert (
            tma_bpo == (iters + 1) // 2
        ), "batched combine expects a half-depth (ceil(iters/2)) box."
        comptime assert (
            wg_j_offset == 0 or wg_j_offset == tma_bpo
        ), "batched combine expects wg_j_offset in {0, tma_bpo}."

    var e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Each thread owns output row `local_row` (= tid % 128). Combine own+peer
    # for this WG's block range and write row-major to SMEM.
    comptime for iter in range(iters_per_wg):
        comptime j = wg_j_offset + iter
        comptime col_start = j * o_sw_K
        var own_arr = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=o_sw_K,
            dtype=accum_dtype,
            pack=False,
            width=o_sw_K,
        ](own_o_tmem.tmem_addr + UInt32(col_start))
        var peer_arr = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=o_sw_K,
            dtype=accum_dtype,
            pack=False,
            width=o_sw_K,
        ](peer_o_tmem.tmem_addr + UInt32(col_start))

        # combined = own * final_scale_local + peer * final_scale_peer, packed
        # into one 16 B store (f32x2 compute, wide store; see combine_pack_o_row).
        var packed = combine_pack_o_row[output_type](
            own_arr, peer_arr, final_scale_local, final_scale_peer
        )

        # Block `j` is one k-block [BM, o_sw_K]; col % o_sw_K == 0.
        var o_inner = Int(local_row) * o_sw_K
        (o_smem_arg + j * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
            UInt32
        ]().store(packed)

    # Sync all WARPGROUP_SIZE threads before the TMA store.
    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

    # TMA store: one elected thread issues this WG's column-block stores.
    if local_warp_idx == 0:
        fence_async_view_proxy()
        comptime if batched:
            # One batched copy over this WG's half [wg_j_offset, wg_j_offset+tma_bpo).
            ragged_tma_store.async_copy_batched[wg_j_offset](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
        else:
            # tma_bpo == 0: swizzled-output fallback -> one per-block TMA each.
            comptime for iter in range(iters_per_wg):
                comptime j = wg_j_offset + iter
                ragged_tma_store.async_copy_from_col[j](
                    o_smem_arg,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                    elect=e,
                )
        cp_async_bulk_commit_group()

    # Wait for all TMA stores to complete.
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_ws_exchange4[
    op: StaticString,
    *,
    m_pack: Int,
    rows: Int,
](
    partial: Float32,
    local_row: UInt32,
    partition_g: UInt32,
    warp_group_idx: UInt32,
    exchange_seq: UInt32,
    xchg_smem: SharedMemPointer[Scalar[DType.float32]],
) -> Float32:
    """`m_pack`-way cross-warp reduction within one softmax warpgroup.

    The shared-key sub-mode (`config.ws_shared_key`) gives all `m_pack` warps
    of a warpgroup ONE key band and ONE shared O accumulator, so the warps
    must agree on the running row max every KV tile before they may rescale
    it. Warp shuffles cannot cross warps, so the reduction transits SMEM.

    "Everyone reduces": every thread writes its own partial, then every thread
    reads all `m_pack` of them. No broadcast step, no designated reducer --
    which is what makes ONE barrier enough (see below).

    Generalized from the depth512 `exchange_reduce` ancestor, which is 2-way
    and spends **two** barriers: it single-buffers into `correction_smem` and
    does a read-modify-write there, so the upper half must not overwrite the
    slot before the lower half has read it. This one has its own
    double-buffered region and no write-back, so one barrier is provably
    sufficient.

    **Why one barrier is race-free.** Let `p = exchange_seq & 1`. Exchange `i`
    writes buffer `p`, barriers, reads `p`. A warp that runs ahead writes
    `1 - p` at `i+1` -- a *different* buffer, so it cannot disturb a sibling
    still reading `p`. To touch `p` again it must reach `i+2`, which requires
    passing `i+1`'s barrier, which the slow sibling must also reach; and to
    reach it the sibling must already have finished reading `p`. So no warp is
    ever two exchanges ahead. **Do not add a second "to be safe" barrier** --
    it is pure critical-path cost on the per-KV-tile path.

    That argument depends on `exchange_seq` being a **monotone counter across
    every call in the warpgroup's lifetime**, never reset per phase. In
    particular the post-loop row-sum "add" must continue the loop's sequence
    rather than restart at 0: restarting could hand it the same parity a
    lagging warp is still reading from the final loop iteration.

    **Bit-identical across warps, by construction.** The fold is a fixed left
    fold over `g = 0 .. m_pack-1`, the same order on every thread, so all
    `m_pack` warps obtain the *same* f32 result -- not merely equal-up-to-
    rounding. The shared-key design's deferred row-sum rests on this: with an
    identical `row_max` the per-warp `correction = exp2(diff)` is identical,
    so the warp-uniform lazy-rescale vote resolves the same way everywhere and
    the four partial sums stay in one scale, making the final combine a plain
    add.

    Barrier id is `warp_group_idx` (0 or 1), the established WG-scoped idiom
    in this file. Two facts make it safe and neither is enforced by the type
    system:
    * The correction, MMA and load warps never call `named_barrier`. They do
      execute one full-CTA `barrier()` before the warp-role dispatch, hardware
      barrier 0 and the only `barrier()` in the kernel, so it is
      phase-separated from every 128-thread use of id 0, never concurrent.
      (Under pair-CTA / split-K that site is `cluster_sync()` instead, which
      does not touch the named-barrier id space.)
    * The two hard-coded `named_barrier[...](Int32(0))` sites sit past a
      `warp_group_idx != 0 -> return` guard, so only WG0 arrives at them.

    Parameters:
        op: `"max"` or `"add"`.
        m_pack: Warps per warpgroup sharing the reduction (2 or 4).
        rows: Query rows per warp (`BM`). `m_pack * rows` must be
            `WARPGROUP_SIZE`, since the slot map is a bijection from the
            warpgroup's 128 threads onto 128 Float32 slots.

    Args:
        partial: This thread's partial value.
        local_row: Query row `r` this thread owns, in `[0, rows)`.
        partition_g: This thread's warp index within the warpgroup,
            `[0, m_pack)`.
        warp_group_idx: 0 or 1; doubles as the named-barrier id.
        exchange_seq: Monotone per-warpgroup exchange counter (see above).
        xchg_smem: Base of `SM100AttentionSMem.ws_exchange_smem()`, i.e. all
            `2 * 2 * WARPGROUP_SIZE` slots; this function selects its own
            warpgroup's and parity's window.

    Returns:
        The reduction over all `m_pack` warps' partials for row `local_row`,
        identical in every warp.
    """
    comptime assert (
        op == "max" or op == "add"
    ), "fa4_ws_exchange4 op must be 'max' or 'add'"
    comptime assert (
        m_pack == 2 or m_pack == 4
    ), "fa4_ws_exchange4 is for the packed-TMEM WS datapath (m_pack 2 or 4)"
    # The slot map below (`partition_g * rows + local_row`) is a bijection onto
    # WARPGROUP_SIZE slots only under this identity. Violate it and two warps
    # alias one slot: a plausible-but-wrong row max, no fault.
    comptime assert (
        m_pack * rows == WARPGROUP_SIZE
    ), "fa4_ws_exchange4 slot map needs m_pack * rows == WARPGROUP_SIZE"

    # Warpgroup `wg` at parity `p` owns [(wg*2 + p)*WARPGROUP_SIZE, +128).
    var buf = xchg_smem + (
        (warp_group_idx * 2 + (exchange_seq & 1)) * UInt32(WARPGROUP_SIZE)
    )
    buf[partition_g * UInt32(rows) + local_row] = partial

    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

    var acc = buf[local_row]
    comptime for g in range(1, m_pack):
        var other = buf[UInt32(g * rows) + local_row]
        comptime if op == "max":
            acc = max_ftz(acc, other)
        else:
            acc = add_ftz(acc, other)
    return acc


@always_inline
def fa4_ws_level0_band[
    band_cols: Int,
](o_tmem: UInt32, mut o_band: Array[Scalar[DType.float32], band_cols],):
    """Level 0 of the shared-key combine: this warp's own O depth band, as-is.

    The shared-key counterpart of `fa4_ws_level1_combine`, and it is startling
    how little is left. Level 1 exists because the key-split gives each of a
    warpgroup's `m_pack` warps an INDEPENDENT key partition carrying a
    full-depth O partial, so it must stage all `m_pack` partials through SMEM,
    barrier, agree a cross-warp max, and fold them with per-partition scales.

    Under `config.ws_shared_key` there is one key band and ONE accumulator, and
    the packed-TMEM quarters are DEPTH BANDS of the same result. So:

    * **No fold.** Warp `g`'s band is already final -- there is no `O_p` from a
      sibling to add. The `m_pack`-way FMA chain, the `scale[]`/`lps[]` arrays
      and the whole staging round trip disappear.
    * **No barrier, and none is missing.** The `m_pack` warps read DISJOINT
      hardware subpartitions of one accumulator that a single MMA wrote. There
      is no cross-warp data flow here at all, so there is nothing to order.
      (The cross-warp traffic the mode does need is elsewhere: the per-KV-tile
      row-max and the once-per-tile row-sum, both via `fa4_ws_exchange4`.)
    * **No max.** Every warp already holds the same `row_max`, because the
      per-iteration `fa4_ws_exchange4["max"]` made it so. That agreement is the
      premise the deferred row-sum rests on -- see `fa4_ws_exchange4`'s
      bit-identity paragraph.

    Packed-TMEM addressing rule, stated here rather than cited: for a
    `tcgen05.mma.ws` accumulator with `MMA_M < 128` and `m_pack = 128 //
    MMA_M`, an `MMA_M x MMA_N` accumulator occupies only `MMA_N / m_pack`
    PHYSICAL TMEM columns, because the hardware subpartition does the lane
    folding. So all `m_pack` warps issue this `tcgen05_ld` at the SAME column
    address and the routing hands each its own column-quarter.
    Do NOT add a per-warp column offset; the per-warp *logical* depth
    base (`depth_base + partition_g * band_cols`) is a consumer-side quantity
    and belongs to Level 2, which takes it as `depth_base`/`band_cols`.

    Caller contract, identical to `fa4_ws_intracta_combine`: make `O` visible
    in TMEM (wait the O-producer barrier and issue `tcgen05_fence_after()`)
    before calling. No `tcgen05_load_wait()` here -- the loads are ordered by
    data dependency, and the wait belongs to whoever next OVERWRITES the TMEM.

    Parameters:
        band_cols: Physical columns of one accumulator this warp reads, i.e.
            `config.pv_mma_n() // config.m_pack`. Call once per output depth
            tile with `o_tmem` advanced by `band_cols`; do NOT pass the whole
            `o_phys_cols()` at `num_o_tiles() > 1`, or the register array grows
            past what `num_reg_softmax` affords (128 f32 at d512, against 192).

    Args:
        o_tmem: TMEM column address of THIS depth tile's accumulator, i.e.
            `tmem_addr + config.TMEM_O{wg} + t * band_cols`.
        o_band: Out-param receiving the band, unnormalized, in registers.
    """
    comptime accum_dtype = DType.float32
    # Same `tcgen05.ld` register-pressure cap as `fa4_ws_intracta_combine`
    # (`ptxas` C7602 -- a width-128 read needs 128 consecutive destination
    # registers). At the shared-key geometry `band_cols` is 64 and this folds
    # to a single read; the split only matters if a future `pv_mma_n` makes the
    # band wider.
    comptime ld_width = min(band_cols, 64)
    comptime ld_splits = band_cols // ld_width
    comptime assert band_cols % ld_width == 0

    comptime for ld in range(ld_splits):
        var c_frag = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=ld_width,
            dtype=accum_dtype,
            pack=False,
            width=ld_width,
        ](o_tmem + UInt32(ld * ld_width))
        comptime for i in range(ld_width):
            o_band[ld * ld_width + i] = c_frag[i]


@always_inline
def fa4_ws_intracta_combine[
    output_type: DType,
    //,
    m_pack: Int,
    rows: Int,
    ov_depth: Int,
    use_fma: Bool = True,
](
    local_row: UInt32,
    partition_g: UInt32,
    warp_group_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    o_tmem: UInt32,
    stage_smem: SharedMemPointer[Float32],
    maxsum_smem: SharedMemPointer[Float32],
    o_smem: SharedMemPointer[Scalar[output_type]],
) -> Tuple[Float32, Float32]:
    """Intra-CTA `m_pack`-way LSE combine for the `.ws` MMA_M=32 datapath.

    The `m_pack` packed-TMEM datapath quarters are `m_pack` independent
    key-partitions over the SAME `rows` query rows: warp `g = partition_g` holds
    partition `g`'s unnormalized `O_g` (`rows x ov_depth`) in its C-TMEM band
    (all warps share the base `o_tmem`; the hardware subpartition routes each
    warp its own quarter, so there is no per-warp column offset), plus its
    per-row `(m_g, l_g) = (own_max, own_sum)`. Because the partitions live in
    DIFFERENT warps, the merge is a cross-warp reduction and must transit SMEM
    (a warp shuffle cannot reach across warps).

    Merges, per query row `r`:

        m       = max_g m_g
        scale_g = exp2((m_g - m) * scale_log2e)   # *scale_log2e gated on use_fma
        O       = sum_g scale_g * O_g
        l       = sum_g scale_g * l_g
        out     = O * recip(l)

    Identical LSE math to `fa4_lse_combine_write` / `fa4_splitk_combine_write`;
    only the transport is intra-CTA across warp quarters. Assumes >= 1 non-empty
    partition (a launched tile always does >= 1 partition of real work). An
    individual empty partition supplies `m_g = -inf`, `l_g = 0`, and a FINITE
    zero-filled `O_g` (so `scale_g = 0` and `0 * 0 = 0`, never `0 * inf = NaN`);
    there is no all-empty guard, mirroring `fa4_splitk_combine_write`'s
    >=1-valid-key assumption.

    The caller must have made `O_g` visible in TMEM (wait the O-producer barrier
    and issue `tcgen05_fence_after()`) before this call, exactly as the
    `fa4_lse_combine_write` caller does. Output is written to `o_smem` in the
    SWIZZLE_NONE 16 B block-major layout (`[block, rows, o_sw_K]`,
    `o_sw_K = 16 // size_of[output_type]()`) that a SWIZZLE_NONE O-TMA store
    consumes; the caller performs the egress (the O TMA store in the kernel).

    Parameters:
        output_type: The `o_smem` element dtype (bf16 in-kernel; f32 for tests).
        m_pack: Number of key-partitions == datapath quarters (4 for MMA_M=32).
        rows: Query rows per tile (`config.BM`, 32 for MMA_M=32).
        ov_depth: Output value depth.
        use_fma: Gate the `* scale_log2e` step (matches `fa4_splitk_combine_write`).

    Args:
        local_row: This lane's query row (`tid & 31`).
        partition_g: This warp's partition / datapath quarter (`tid >> 5`).
        warp_group_idx: WG-scoped `named_barrier` id.
        own_max: This partition's per-row local max `m_g[local_row]` (raw units).
        own_sum: This partition's per-row local sum `l_g[local_row]`.
        scale_log2e: The softmax `log2(e)` scale.
        o_tmem: Base C-TMEM column address of the shared O band.
        stage_smem: SMEM scratch, `>= m_pack*rows*ov_depth` f32 (raw O staging,
            SWIZZLE_NONE 16 B block-major `[m_pack, stage_blocks, rows, 4]`).
        maxsum_smem: SMEM scratch, `>= m_pack*rows*2` f32 ((m, l) staging).
        o_smem: Output sink, SWIZZLE_NONE 16 B block-major
            `[block, rows, o_sw_K]` of `output_type`.
    """
    comptime accum_dtype = DType.float32
    # Physical band width per depth-tile == MMA_N_max(256, F16) / m_pack, and the
    # P@V MMA wrote `num_d_tiles` such tiles contiguously, so warp g's full-depth O
    # reads back as `num_d_tiles` per-tile `tcgen05_ld`s of `depth_tile` columns.
    # `min(_, ov_depth)`: Layout-G (m_pack==4) DEPTH-SCATTERS into `256//m_pack`
    # (=64)-wide tiles, always <= ov_depth. Layout-E (m_pack==2) instead
    # REDUCTION-ACCUMULATES into ONE `ov_depth`-wide tile (mma_warp.mojo's
    # `num_d_tiles = num_qk_stages`, `pv_mma_n = m_pack*ov_depth`); at ov_depth=64
    # the depth-scatter `256//2=128` would overrun (`num_d_tiles=0`, assert
    # fires). `256//m_pack == ov_depth` only coincidentally at Layout-E d128.
    comptime depth_tile = min(256 // m_pack, ov_depth)
    comptime num_d_tiles = ov_depth // depth_tile
    comptime own_cols = ov_depth // m_pack
    comptime assert ov_depth % depth_tile == 0
    comptime assert ov_depth % m_pack == 0

    # SWIZZLE_NONE 16 B block-major layout (matches `fa4_scale_write_output` /
    # `fa4_splitk_stage_partial`): both `stage_smem` (f32) and `o_smem`
    # (`output_type`) are `[block, rows, K]` per partition, one 16 B `STS.128`
    # per block, so consecutive rows land 16 B (4 banks) apart -- bank-conflict
    # free on both the stage write and the output store. `stage_K` is fixed at 4
    # (f32), `o_sw_K` widens with a narrower `output_type` (8 for bf16/f16).
    comptime stage_K = 16 // size_of[accum_dtype]()  # f32 -> 4
    comptime stage_blocks = ov_depth // stage_K
    comptime o_sw_K = 16 // size_of[output_type]()  # bf16 -> 8, f32 -> 4
    comptime own_oblocks = own_cols // o_sw_K
    comptime sub = o_sw_K // stage_K  # stage-blocks per output block (1 or 2)
    comptime assert ov_depth % stage_K == 0
    comptime assert own_cols % o_sw_K == 0
    comptime assert depth_tile % stage_K == 0

    var g = Int(partition_g)
    var r = Int(local_row)

    # ---- (1) stage this partition's RAW O_g band + (m_g, l_g) into SMEM ----
    # Raw (not pre-scaled): the per-row scale needs the cross-warp global max, so
    # staging raw lets every reader compute its own scales after ONE barrier.
    # Stored SWIZZLE_NONE block-major (`stage_smem[(g*stage_blocks + sblk)*rows
    # *stage_K + r*stage_K + k]`): each `depth_tile` `tcgen05_ld` fragment is
    # split into `stage_K`-wide sub-blocks, each a freshly-built `SIMD[f32,
    # stage_K]` whose `.store()` is one bank-conflict-free `STS.128` (as at the
    # split-K peer store below).
    comptime tile_sblocks = depth_tile // stage_K
    # `tcgen05.ld` register-pressure cap: a single width-128 read (Layout-E,
    # depth_tile=128) needs 128 consecutive destination registers, which can
    # exceed the kernel's register budget (`ptxas` C7602 "Insufficient
    # registers (128)... Try to compile with register target of 146 or
    # higher" -- confirmed via a local `ptxas` 12.9/PTX-8.8 run against the
    # emitted PTX, `tcgen05.ld.sync.aligned.32x32b.x128.b32` with a
    # 128-register destination list). Split into `ld_width`(<=64)-wide
    # sub-reads; Layout-G's existing width=64 read folds to `ld_splits==1`
    # (byte-identical -- the split only changes instruction-level register
    # footprint, not the staged bytes: two contiguous 64-wide reads followed
    # by staging cover the exact same `sblk` range, in the same order, as one
    # 128-wide read would).
    comptime ld_width = min(depth_tile, 64)
    comptime ld_splits = depth_tile // ld_width
    comptime ld_sblocks = ld_width // stage_K
    comptime assert depth_tile % ld_width == 0
    comptime for t in range(num_d_tiles):
        comptime for ld in range(ld_splits):
            var c_frag = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=ld_width,
                dtype=accum_dtype,
                pack=False,
                width=ld_width,
            ](o_tmem + UInt32(t * depth_tile + ld * ld_width))
            comptime for sb in range(ld_sblocks):
                comptime sblk = t * tile_sblocks + ld * ld_sblocks + sb
                var v = SIMD[accum_dtype, stage_K]()
                comptime for k in range(stage_K):
                    v[k] = c_frag[sb * stage_K + k]
                (
                    stage_smem
                    + (g * stage_blocks + sblk) * rows * stage_K
                    + r * stage_K
                ).store(v)
    maxsum_smem[(g * rows + r) * 2] = own_max
    maxsum_smem[(g * rows + r) * 2 + 1] = own_sum

    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

    # ---- (2) distributed combine: warp g owns cols [g*own_cols, +own_cols) ----
    var m: Float32
    comptime if m_pack == 2:
        m = max_ftz(maxsum_smem[r * 2], maxsum_smem[(rows + r) * 2])
    else:
        comptime assert m_pack == 4
        m = max_ftz(
            maxsum_smem[r * 2],
            maxsum_smem[(rows + r) * 2],
            maxsum_smem[(2 * rows + r) * 2],
        )
        m = max_ftz(m, maxsum_smem[(3 * rows + r) * 2])

    var scale = Array[Scalar[accum_dtype], m_pack](uninitialized=True)
    var lps = Array[Scalar[accum_dtype], m_pack](uninitialized=True)
    comptime for p in range(m_pack):
        var mp = maxsum_smem[(p * rows + r) * 2]
        var lp = maxsum_smem[(p * rows + r) * 2 + 1]
        var d = mp - m
        comptime if use_fma:
            d = mul_ftz(d, scale_log2e)
        var s = exp2(d)
        scale[p] = s
        lps[p] = lp
        # l_acc = s.fma(lp, l_acc)
    var l_acc: Float32
    comptime if m_pack == 2:
        l_acc = scale[1].fma(lps[1], scale[0] * lps[0])
    else:
        comptime assert m_pack == 4
        var l_acc_x2: f32x2 = f32x2(scale[0], scale[1]) * f32x2(lps[0], lps[1])
        l_acc_x2 = f32x2(scale[2], scale[3]).fma(
            f32x2(lps[2], lps[3]), l_acc_x2
        )
        l_acc = l_acc_x2[0] + l_acc_x2[1]
    var inv_l = recip(l_acc)

    # Warp g owns output blocks [g*own_oblocks, (g+1)*own_oblocks), which tile
    # [0, ov_depth) disjointly. Per block: an f32x2 `m_pack`-way reduction over
    # the block-major staged O (two f32x2 halves per `stage_K`-wide sub-block),
    # normalized by `inv_l`, packed into one 16 B `STS.128` (`st_shared_v4_b32`
    # forces the wide store; the F2FP pack outputs would otherwise scalarize to
    # 4x bank-conflicting STS.32 -- see its docstring).
    var col0 = g * own_cols
    comptime for ob in range(own_oblocks):
        var oblk = (col0 // o_sw_K) + ob  # global output-block index
        var packed = SIMD[.uint32, 4]()
        comptime for sb in range(sub):
            var sblk = oblk * sub + sb  # global stage-block index
            comptime for hc in range(2):  # two f32x2 halves per stage-block
                var acc = mul_ftz(
                    f32x2(scale[0]),
                    (
                        stage_smem
                        + (0 * stage_blocks + sblk) * rows * stage_K
                        + r * stage_K
                        + 2 * hc
                    ).load[width=2](),
                )
                comptime for p in range(1, m_pack):
                    acc = fma_ftz(
                        f32x2(scale[p]),
                        (
                            stage_smem
                            + (p * stage_blocks + sblk) * rows * stage_K
                            + r * stage_K
                            + 2 * hc
                        ).load[width=2](),
                        acc,
                    )
                acc = mul_ftz(acc, f32x2(inv_l))
                comptime cc = sb * 2 + hc  # f32x2 chunk within the output block
                comptime if size_of[output_type]() == 4:
                    # f32 output: each f32x2 chunk is two u32 store lanes.
                    var u2 = bitcast[.uint32, 2](acc)
                    packed[2 * cc] = u2[0]
                    packed[2 * cc + 1] = u2[1]
                else:
                    # bf16/f16 output: each f32x2 chunk packs into one u32 lane.
                    packed[cc] = bitcast[.uint32, 1](acc.cast[output_type]())
        st_shared_v4_b32(o_smem, oblk * rows * o_sw_K + r * o_sw_K, packed)

    # Per-row fully-reduced (M_cta, L_cta) (raw max, base-2 denominator),
    # identical across the `m_pack` warps. Ignored by the non-workspace caller;
    # the workspace (unfused split-K) egress feeds it to `_ws_write_lse`.
    return (m, l_acc)


@always_inline
def fa4_ws_level1_combine[
    m_pack: Int,
    rows: Int,
    ov_depth: Int,
    use_fma: Bool = True,
](
    local_row: UInt32,
    partition_g: UInt32,
    warp_group_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    o_tmem: UInt32,
    stage_smem: SharedMemPointer[Float32],
    maxsum_smem: SharedMemPointer[Float32],
    mut o_band: Array[Float32, ov_depth // m_pack],
) -> Tuple[Float32, Float32]:
    """Level 1 of the two-level WS combine: per-warpgroup `m_pack`-way merge of
    the packed-TMEM datapath quarters into an UNNORMALIZED partial.

    A fork of `fa4_ws_intracta_combine` that stops before normalization. It
    reuses the raw-O staging, the single WG-local barrier, the cross-warp
    `m = max_g m_g`, and the `scale`/`l_acc` accumulation, but DROPS the `recip`
    and, instead of dividing by `l` and packing to `o_smem`, returns the
    per-warpgroup unnormalized partial (per this lane's query row `r`):

        m_wg      = max_g m_g
        l_wg      = Σ_g exp2((m_g - m_wg) * scale_log2e) * l_g
        o_band[i] = Σ_g exp2((m_g - m_wg) * scale_log2e) * O_g[r, g*own_cols + i]

    Warp `g = partition_g` owns depth band `g` (`own_cols = ov_depth // m_pack`
    columns `[g*own_cols, (g+1)*own_cols)`) and returns THAT band, unnormalized,
    in the `o_band` register array. `(m_wg, l_wg)` are per-row and identical
    across the WG's `m_pack` warps (every warp reduces the same `maxsum_smem`).
    Level 2 (`fa4_ws_level2_reduce_scatter_write`) merges the two warpgroups'
    `(m_wg, l_wg, o_band)` and normalizes.

    Same caller contract, comptime params, and empty-partition neutral element
    (`m_g = -inf, l_g = 0, finite-0 O_g`) as `fa4_ws_intracta_combine`.

    Returns:
        `(m_wg, l_wg)` for this lane's row (raw-unit max, unnormalized sum).
    """
    comptime accum_dtype = DType.float32
    # `min(_, ov_depth)`: see `fa4_ws_intracta_combine` -- Layout-E (m_pack==2)
    # reduction-accumulates O into ONE `ov_depth`-wide tile, so the depth-scatter
    # `256//m_pack` would overrun at ov_depth=64 (num_d_tiles=0, assert fires);
    # byte-identical for Layout-G and Layout-E d128 (`256//m_pack <= ov_depth`).
    comptime depth_tile = min(256 // m_pack, ov_depth)
    comptime num_d_tiles = ov_depth // depth_tile
    comptime own_cols = ov_depth // m_pack
    comptime assert ov_depth % depth_tile == 0
    comptime assert ov_depth % m_pack == 0

    comptime stage_K = 16 // size_of[accum_dtype]()  # f32 -> 4
    comptime stage_blocks = ov_depth // stage_K
    comptime own_sblocks = own_cols // stage_K
    comptime assert ov_depth % stage_K == 0
    comptime assert own_cols % stage_K == 0
    comptime assert depth_tile % stage_K == 0

    var g = Int(partition_g)
    var r = Int(local_row)

    # ---- (1) stage this partition's RAW O_g band + (m_g, l_g) into SMEM ----
    # Identical staging to `fa4_ws_intracta_combine`: each warp's own C-TMEM
    # quarter (shared base, HW subpartition routing) reads back as `num_d_tiles`
    # `depth_tile`-column fragments, sub-split into bank-conflict-free 16 B
    # block-major `stage_smem` stores.
    comptime tile_sblocks = depth_tile // stage_K
    # `tcgen05.ld` register-pressure cap: see `fa4_ws_intracta_combine` (same
    # fix, same rationale -- a single width-128 read at Layout-E blows the
    # register budget; split into `ld_width`(<=64)-wide sub-reads, folds to
    # `ld_splits==1` (byte-identical) for Layout-G's width=64 read.
    comptime ld_width = min(depth_tile, 64)
    comptime ld_splits = depth_tile // ld_width
    comptime ld_sblocks = ld_width // stage_K
    comptime assert depth_tile % ld_width == 0
    comptime for t in range(num_d_tiles):
        comptime for ld in range(ld_splits):
            var c_frag = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=ld_width,
                dtype=accum_dtype,
                pack=False,
                width=ld_width,
            ](o_tmem + UInt32(t * depth_tile + ld * ld_width))
            comptime for sb in range(ld_sblocks):
                comptime sblk = t * tile_sblocks + ld * ld_sblocks + sb
                var v = SIMD[accum_dtype, stage_K]()
                comptime for k in range(stage_K):
                    v[k] = c_frag[sb * stage_K + k]
                (
                    stage_smem
                    + (g * stage_blocks + sblk) * rows * stage_K
                    + r * stage_K
                ).store(v)
    maxsum_smem[(g * rows + r) * 2] = own_max
    maxsum_smem[(g * rows + r) * 2 + 1] = own_sum

    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

    # ---- (2) cross-warp m = max_g m_g and the per-quarter scales / l_wg ----
    var m: Float32
    comptime if m_pack == 2:
        m = max_ftz(maxsum_smem[r * 2], maxsum_smem[(rows + r) * 2])
    else:
        comptime assert m_pack == 4
        m = max_ftz(
            maxsum_smem[r * 2],
            maxsum_smem[(rows + r) * 2],
            maxsum_smem[(2 * rows + r) * 2],
        )
        m = max_ftz(m, maxsum_smem[(3 * rows + r) * 2])

    var scale = Array[Scalar[accum_dtype], m_pack](uninitialized=True)
    var lps = Array[Scalar[accum_dtype], m_pack](uninitialized=True)
    comptime for p in range(m_pack):
        var mp = maxsum_smem[(p * rows + r) * 2]
        var lp = maxsum_smem[(p * rows + r) * 2 + 1]
        var d = mp - m
        comptime if use_fma:
            d = mul_ftz(d, scale_log2e)
        scale[p] = exp2(d)
        lps[p] = lp
    var l_acc: Float32
    comptime if m_pack == 2:
        l_acc = scale[1].fma(lps[1], scale[0] * lps[0])
    else:
        comptime assert m_pack == 4
        var l_acc_x2: f32x2 = f32x2(scale[0], scale[1]) * f32x2(lps[0], lps[1])
        l_acc_x2 = f32x2(scale[2], scale[3]).fma(
            f32x2(lps[2], lps[3]), l_acc_x2
        )
        l_acc = l_acc_x2[0] + l_acc_x2[1]

    # ---- (3) unnormalized own-band reduction O_band = Σ_p scale_p * O_p ----
    # Warp `g` owns local band [0, own_cols) == global depth stage-blocks
    # [g*own_sblocks, (g+1)*own_sblocks). NO recip / NO pack / NO store: the band
    # is returned in registers, unnormalized, for Level 2 to merge across WGs.
    comptime for lb in range(own_sblocks):
        var sblk = (
            g * own_sblocks + lb
        )  # global stage-block for this band (runtime g)
        comptime for hc in range(2):  # two f32x2 halves per stage-block
            var acc = mul_ftz(
                f32x2(scale[0]),
                (
                    stage_smem
                    + (0 * stage_blocks + sblk) * rows * stage_K
                    + r * stage_K
                    + 2 * hc
                ).load[width=2](),
            )
            comptime for p in range(1, m_pack):
                acc = fma_ftz(
                    f32x2(scale[p]),
                    (
                        stage_smem
                        + (p * stage_blocks + sblk) * rows * stage_K
                        + r * stage_K
                        + 2 * hc
                    ).load[width=2](),
                    acc,
                )
            comptime obase = lb * stage_K + 2 * hc
            o_band[obase] = acc[0]
            o_band[obase + 1] = acc[1]

    return (m, l_acc)


@always_inline
def fa4_ws_level2_reduce_scatter_write[
    output_type: DType,
    //,
    m_pack: Int,
    rows: Int,
    ov_depth: Int,
    band_cols: Int = ov_depth // m_pack,
    depth_base: Int = 0,
    single_wg: Bool = False,
    use_fma: Bool = True,
](
    local_row: UInt32,
    band_g: UInt32,
    warp_group_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    o_band: Array[Float32, band_cols],
    l2_stage_smem: SharedMemPointer[Float32],
    l2_maxsum_smem: SharedMemPointer[Float32],
    o_smem: SharedMemPointer[Scalar[output_type]],
) -> Tuple[Float32, Float32]:
    """Level 2 of the two-level WS combine: intra-CTA cross-warpgroup
    reduce-scatter of the two per-WG Level-1 partials into the normalized
    output. WG0-writes-all (§5c step 3 / §9.8).

    Both warpgroups arrive with their Level-1 partial `(m_wg, l_wg, o_band)`
    (`own_max`, `own_sum`, `o_band`) for band `g = band_g` (this warp) and query
    row `r = local_row` (this lane). The merge, per (band `g`, row `r`):

        M       = max(m_0, m_1)
        s_w     = exp2((m_w - M) * scale_log2e)
        L       = s_1.fma(l_1, s_0 * l_0)
        O_final = (s_1.fma(O_1[band g], s_0 * O_0[band g])) * recip(L)

    Transport (WG0-writes-all): WG1 scatters its `o_band` into `l2_stage_smem`
    (16 B block-major, mirroring the Level-1 stage) and its `(m_1, l_1)` into
    `l2_maxsum_smem`; a single `named_barrier[2*WARPGROUP_SIZE](3)` fences both
    for all 256 threads (a `bar.sync` fences the intra-CTA `st/ld.shared`, so no
    `fence_async_view_proxy` is needed); then each WG0 warp `g` reads its band's
    `O_1` from `l2_stage_smem`, keeps its own `O_0[band g]` in registers,
    combines + normalizes, and writes band `g` to `o_smem` (SWIZZLE_NONE 16 B
    block-major). WG0 uses its own `(m_0, l_0)` from registers, so only WG1's
    `(m_1, l_1)` transit `l2_maxsum_smem`. The TMA egress is a separate call
    (`fa4_tma_store_o_smem`). See §5c.

    Empty-partition neutral element (`m_w = -inf, l_w = 0, finite-0 O`) composes
    at this level too: an empty WG contributes `s_w = exp2(-inf) = 0`, so
    `s_w * O_w = 0` and it drops out of `L` -- relies on the >=1-active-partition
    invariant (`M > -inf`, `L > 0`).

    Two things about the tiled form are load-bearing:

    1. **Give each tile its own `l2_stage_smem` window.** This routine is
       `WG1 scatters -> barrier(3) -> WG0 gathers`, and NOTHING orders WG1's
       *next* scatter against WG0's *current* gather. Calling it twice on one
       window is a WAR race: at `t=1` WG1 can overwrite the staging while WG0
       is still reading `t=0`. Pass `l2_stage_smem + t * (m_pack * band_cols) *
       rows`. Summed over the tiles that is `padded_ov * rows` f32 -- exactly
       the single-call footprint, so the fix costs no SMEM and no barrier.
    2. **Only tile 0 transports `(m_1, l_1)`.** They are depth-independent, so
       the `l2_maxsum_smem` write is gated on `depth_base == 0` and later tiles
       read what tile 0 left. Re-writing it every tile would be a same-value
       store racing WG0's read for no reason.

    **Single-warpgroup invocation (`single_wg`).** There is no second partial to
    merge when only WG0 is alive -- `total_iters_combined == 1`, where WG1 has
    already returned through `named_barrier[256](2)`. Reaching the barrier at
    (2) below with 128 of its 256 threads gone is a HANG, so that case cannot
    simply call this with a neutral peer. `single_wg=True` drops the scatter,
    the barrier and the gather, leaving `O_final = O_0[band g] * recip(l_0)`.

    It is a parameter here rather than a fourth combine routine because the tail
    that survives -- the `col0` block walk, the f32x2 pack, the `st_shared_v4_b32`
    -- is the part with the swizzle geometry in it, and that part is already
    duplicated once (`fa4_ws_intracta_combine`'s tail is structurally the same
    code). One reduction tree, not a fork. `False` is the default and leaves
    every existing call byte-identical.

    Parameters:
        output_type: The `o_smem` element dtype (bf16 in-kernel; f32 for tests).
        m_pack: Number of datapath quarters per WG (4 for MMA_M=32).
        rows: Query rows per tile (`config.BM`, 32 for MMA_M=32).
        ov_depth: Output value depth; sets the `o_smem` block geometry.
        band_cols: Columns this warp owns in THIS call. Defaults to
            `ov_depth // m_pack` (the whole band, single-tile).
        depth_base: Global depth column where this tile starts. Defaults to 0.
        single_wg: WG0 is the only live warpgroup; skip the cross-WG transport
            entirely (no scatter, no barrier, no gather). Defaults to False.
        use_fma: Gate the `* scale_log2e` step (matches Level 1).

    Args:
        local_row: This lane's query row (`tid & 31`).
        band_g: This warp's depth band within its WG (`(tid % 128) >> 5`).
        warp_group_idx: 0 or 1 (`tid // 128`); selects scatter (1) vs combine (0).
        own_max: This WG's Level-1 `m_wg[local_row]`.
        own_sum: This WG's Level-1 `l_wg[local_row]`.
        scale_log2e: The softmax `log2(e)` scale.
        o_band: This warp's unnormalized band for this tile (`band_cols` f32).
        l2_stage_smem: SMEM scratch for THIS tile, `>= m_pack*band_cols*rows`
            f32 (WG1's O staging).
        l2_maxsum_smem: SMEM scratch, `>= rows*2` f32 (WG1's (m_1, l_1)).
        o_smem: Output sink, SWIZZLE_NONE 16 B block-major of `output_type`.
    """
    comptime accum_dtype = DType.float32
    comptime stage_K = 16 // size_of[accum_dtype]()  # f32 -> 4
    comptime own_sblocks = band_cols // stage_K
    comptime o_sw_K = 16 // size_of[output_type]()  # bf16 -> 8, f32 -> 4
    comptime own_oblocks = band_cols // o_sw_K
    comptime sub = o_sw_K // stage_K  # stage-blocks per output block (1 or 2)
    comptime assert ov_depth % m_pack == 0
    comptime assert band_cols % stage_K == 0
    comptime assert band_cols % o_sw_K == 0
    # `oblk` below is `(depth_base + g*band_cols) // o_sw_K`; an unaligned
    # `depth_base` would silently round the whole tile's output block index
    # DOWN onto the previous tile's blocks.
    comptime assert depth_base % o_sw_K == 0
    comptime assert depth_base + m_pack * band_cols <= ov_depth

    var g = Int(band_g)
    var r = Int(local_row)

    # ---- (1) scatter: WG1 stages its band's O into l2_stage + (m_1,l_1) ----
    # ---- (2) cross-WG rendezvous (fences l2_stage / l2_maxsum, all 256) ----
    # Both vanish under `single_wg`. Not as an optimization: there is no WG1 to
    # scatter, and a 256-count barrier with 128 of its threads already returned
    # is a HANG rather than a no-op.
    comptime if not single_wg:
        if warp_group_idx == UInt32(1):
            # `sblk` is WINDOW-local, not global: `l2_stage_smem` is THIS tile's
            # own window (see the tiled-invocation note above), so it spans only
            # `m_pack * band_cols` columns.
            comptime for lb in range(own_sblocks):
                var sblk = (
                    g * own_sblocks + lb
                )  # window stage-block (runtime g)
                var v = SIMD[accum_dtype, stage_K]()
                comptime for k in range(stage_K):
                    v[k] = o_band[lb * stage_K + k]
                (l2_stage_smem + sblk * rows * stage_K + r * stage_K).store(v)
            # `(m_1, l_1)` are depth-independent, so only the FIRST tile
            # transports them and later tiles read what it left. Folds to
            # unconditional at the default `depth_base == 0`.
            comptime if depth_base == 0:
                if band_g == UInt32(
                    0
                ):  # WG1 warp 0's lanes own all rows' (m,l)
                    l2_maxsum_smem[r * 2] = own_max
                    l2_maxsum_smem[r * 2 + 1] = own_sum

        named_barrier[Int32(2 * WARPGROUP_SIZE)](Int32(3))

    # ---- (3) WG0: gather WG1's band, 2-way combine, normalize, pack -> o_smem
    # Per-CTA (M_cta, L_cta) for the workspace (unfused split-K) egress: WG0
    # computes the true (gmax, l_acc); WG1 returns its own placeholder (unused --
    # `_ws_write_lse` is WG0-only). Mirrors `fa4_ws_level2_combine_partial`.
    var ret_m = own_max
    var ret_l = own_sum
    if warp_group_idx == UInt32(0):
        var m0 = own_max
        var l0 = own_sum
        # `single_wg` initializers -- and in that mode they are the ANSWER, not
        # placeholders: with no peer, M == m_0 exactly, so s_0 == 1 and L == l_0.
        # `s1 == 0` additionally folds the peer FMA out of the pack loop below.
        var gmax = m0
        var s0 = Float32(1)
        var s1 = Float32(0)
        var l_acc = l0
        comptime if not single_wg:
            var m1 = l2_maxsum_smem[r * 2]
            var l1 = l2_maxsum_smem[r * 2 + 1]
            gmax = max_ftz(m0, m1)
            var d0 = m0 - gmax
            var d1 = m1 - gmax
            comptime if use_fma:
                d0 = mul_ftz(d0, scale_log2e)
                d1 = mul_ftz(d1, scale_log2e)
            s0 = exp2(d0)
            s1 = exp2(d1)
            l_acc = s1.fma(l1, mul_ftz(s0, l0))
        var inv_l = recip(l_acc)
        ret_m = gmax
        ret_l = l_acc
        # GLOBAL depth column this warp's band starts at. `depth_base` is 0 for
        # every single-tile call, so this is literally the old `g * own_cols`.
        var col0 = depth_base + g * band_cols
        comptime for ob in range(own_oblocks):
            var oblk = (col0 // o_sw_K) + ob  # global output-block index
            var packed = SIMD[.uint32, 4]()
            comptime for sb in range(sub):
                comptime lsblk = ob * sub + sb  # local stage-block within band
                var sblk = g * own_sblocks + lsblk  # window stage-block
                comptime for hc in range(2):  # two f32x2 halves per stage-block
                    # The load must be SUPPRESSED under `single_wg`, not merely
                    # scaled by `s1 == 0`: nothing scattered into this window, so
                    # it still holds whatever the dead Q/KV span held, and
                    # `fma(0, NaN, acc)` is NaN. The dead initializer folds away
                    # off-mode, keeping the load in its original position.
                    var o1 = f32x2(0)
                    comptime if not single_wg:
                        o1 = (
                            l2_stage_smem
                            + sblk * rows * stage_K
                            + r * stage_K
                            + 2 * hc
                        ).load[width=2]()
                    comptime obase = lsblk * stage_K + 2 * hc
                    var o0 = f32x2(o_band[obase], o_band[obase + 1])
                    var acc = mul_ftz(f32x2(s0), o0)
                    acc = fma_ftz(f32x2(s1), o1, acc)
                    acc = mul_ftz(acc, f32x2(inv_l))
                    comptime cc = sb * 2 + hc  # f32x2 chunk within output block
                    comptime if size_of[output_type]() == 4:
                        # f32 output: each f32x2 chunk is two u32 store lanes.
                        var u2 = bitcast[.uint32, 2](acc)
                        packed[2 * cc] = u2[0]
                        packed[2 * cc + 1] = u2[1]
                    else:
                        # bf16/f16 output: each f32x2 chunk packs into one u32.
                        packed[cc] = bitcast[.uint32, 1](
                            acc.cast[output_type]()
                        )
            st_shared_v4_b32(o_smem, oblk * rows * o_sw_K + r * o_sw_K, packed)

    return (ret_m, ret_l)


@always_inline
def fa4_ws_intracta_combine_partial[
    m_pack: Int,
    rows: Int,
    ov_depth: Int,
    use_fma: Bool = True,
](
    local_row: UInt32,
    partition_g: UInt32,
    warp_group_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    o_tmem: UInt32,
    stage_smem: SharedMemPointer[Float32],
    maxsum_smem: SharedMemPointer[Float32],
    mut o_band_norm: Array[Float32, ov_depth // m_pack],
) -> Tuple[Float32, Float32]:
    """T==1 cross-CTA (split-K) partial: the 4-way intra-CTA combine emitting the
    NORMALIZED per-CTA band into `o_band_norm` (registers) + returning
    `(M_cta, L_cta)`, for the cross-CTA reduce-scatter third level, instead of
    packing to `o_smem`.

    Implemented as `fa4_ws_level1_combine` (the unnormalized `m_pack`-way merge,
    which writes warp `g`'s band `[g*own_cols, +own_cols)` and returns
    `(m_wg, l_wg)`) followed by an explicit `recip(l_wg)` normalize -- an
    identical result to `fa4_ws_intracta_combine` but emitted to registers. For
    T==1 only WG0 is active, so its `(m_wg, l_wg)` is the full-CTA `(M_cta, L_cta)`
    and its per-warp band is the normalized `O_cta` band the third level scatters.
    """
    var m_wg, l_wg = fa4_ws_level1_combine[
        m_pack, rows, ov_depth, use_fma=use_fma
    ](
        local_row,
        partition_g,
        warp_group_idx,
        own_max,
        own_sum,
        scale_log2e,
        o_tmem,
        stage_smem,
        maxsum_smem,
        o_band_norm,
    )
    var inv_l = recip(l_wg)
    comptime for i in range(ov_depth // m_pack):
        o_band_norm[i] = mul_ftz(o_band_norm[i], inv_l)
    return (m_wg, l_wg)


@always_inline
def fa4_ws_level2_combine_partial[
    m_pack: Int,
    rows: Int,
    ov_depth: Int,
    use_fma: Bool = True,
](
    local_row: UInt32,
    band_g: UInt32,
    warp_group_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    o_band: Array[Float32, ov_depth // m_pack],
    l2_stage_smem: SharedMemPointer[Float32],
    l2_maxsum_smem: SharedMemPointer[Float32],
    mut o_band_norm: Array[Float32, ov_depth // m_pack],
) -> Tuple[Float32, Float32]:
    """T>=2 cross-CTA (split-K) partial: a fork of
    `fa4_ws_level2_reduce_scatter_write` that emits the NORMALIZED per-CTA band
    into `o_band_norm` (registers) + returns `(M_cta, L_cta) = (gmax, l_acc)` for
    the cross-CTA reduce-scatter third level, instead of packing to `o_smem`.

    Same WG0-writes-all transport as the base (WG1 scatters its `o_band` +
    `(m_1, l_1)` into `l2_stage_smem`/`l2_maxsum_smem`, single
    `named_barrier[2*WARPGROUP_SIZE](3)`, WG0 gathers + 2-way combines +
    normalizes), only the sink differs (registers, not `o_smem`). WG1 returns a
    placeholder `(own_max, own_sum)` -- it does not feed the third level (WG0 owns
    every band's per-CTA partial). See §5c / Phase D of the living doc.
    """
    comptime accum_dtype = DType.float32
    comptime own_cols = ov_depth // m_pack
    comptime stage_K = 16 // size_of[accum_dtype]()  # f32 -> 4
    comptime own_sblocks = own_cols // stage_K
    comptime assert ov_depth % m_pack == 0
    comptime assert own_cols % stage_K == 0

    var g = Int(band_g)
    var r = Int(local_row)

    # ---- (1) scatter: WG1 stages its band's O into l2_stage + (m_1,l_1) ----
    if warp_group_idx == UInt32(1):
        comptime for lb in range(own_sblocks):
            var sblk = g * own_sblocks + lb  # global stage-block (runtime g)
            var v = SIMD[accum_dtype, stage_K]()
            comptime for k in range(stage_K):
                v[k] = o_band[lb * stage_K + k]
            (l2_stage_smem + sblk * rows * stage_K + r * stage_K).store(v)
        if band_g == UInt32(0):  # WG1 warp 0's 32 lanes own all 32 rows' (m,l)
            l2_maxsum_smem[r * 2] = own_max
            l2_maxsum_smem[r * 2 + 1] = own_sum

    # ---- (2) cross-WG rendezvous (fences l2_stage / l2_maxsum, all 256) ----
    named_barrier[Int32(2 * WARPGROUP_SIZE)](Int32(3))

    # ---- (3) WG0: gather WG1's band, 2-way combine, normalize -> registers ----
    var ret_m = own_max
    var ret_l = own_sum
    if warp_group_idx == UInt32(0):
        var m0 = own_max
        var l0 = own_sum
        var m1 = l2_maxsum_smem[r * 2]
        var l1 = l2_maxsum_smem[r * 2 + 1]
        var gmax = max_ftz(m0, m1)
        var d0 = m0 - gmax
        var d1 = m1 - gmax
        comptime if use_fma:
            d0 = mul_ftz(d0, scale_log2e)
            d1 = mul_ftz(d1, scale_log2e)
        var s0 = exp2(d0)
        var s1 = exp2(d1)
        var l_acc = s1.fma(l1, mul_ftz(s0, l0))
        var inv_l = recip(l_acc)
        ret_m = gmax
        ret_l = l_acc

        comptime for lb in range(own_sblocks):
            var sblk = g * own_sblocks + lb  # global stage-block for this band
            comptime for hc in range(2):  # two f32x2 halves per stage-block
                var o1 = (
                    l2_stage_smem + sblk * rows * stage_K + r * stage_K + 2 * hc
                ).load[width=2]()
                comptime obase = lb * stage_K + 2 * hc
                var o0 = f32x2(o_band[obase], o_band[obase + 1])
                var acc = mul_ftz(f32x2(s0), o0)
                acc = fma_ftz(f32x2(s1), o1, acc)
                acc = mul_ftz(acc, f32x2(inv_l))
                o_band_norm[obase] = acc[0]
                o_band_norm[obase + 1] = acc[1]

    return (ret_m, ret_l)


@always_inline
def fa4_ws_splitk_reduce_scatter_write[
    output_type: DType,
    //,
    config: FA4Config,
    m_pack: Int,
    rows: Int,
    ov_depth: Int,
    P: Int,
    use_fma: Bool,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
    tma_bpo: Int = 0,
](
    local_row: UInt32,
    band_g: UInt32,
    warp_group_idx: UInt32,
    num_partitions: UInt32,
    partition_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    o_band_norm: Array[Float32, ov_depth // m_pack],
    stage_smem: SharedMemPointer[Float32],
    maxsum_smem: SharedMemPointer[Float32],
    o_smem: SharedMemPointer[Scalar[output_type]],
    publish_mbar: MBarType,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """Third level (cross-CTA cluster split-K) reduce-scatter combine for the
    `.ws` MMA_M=32 datapath: merges the P partition CTAs' per-CTA normalized
    partials into the final output, each CTA writing only its OWN depth band.

    Runs over the `(band = warp, row = lane)` BM=32 WS grid -- the packed-TMEM
    transpose of `fa4_splitk_combine_write`'s `(row = thread)` grid. WG0-only:
    each of WG0's `m_pack` warps `g = band_g` arrives holding the NORMALIZED
    per-CTA band `g` (`o_band_norm`, `ov_depth // m_pack` f32, from
    `fa4_ws_intracta_combine_partial` at T==1 / `fa4_ws_level2_combine_partial`
    at T>=2) plus this partition's per-row `(M_cta, L_cta) = (own_max, own_sum)`.
    WG1 (active only at T>=2) returned a placeholder from Level 2 and does not
    feed this level; it skips straight to the caller's terminal cross-WG barrier.

    Partition `p = partition_idx` owns the P-way split of the `m_pack` warp-bands
    (`bpp = ceil(m_pack / P)` bands each: P=4 -> band `p`; P=2 -> bands
    `[2p, 2p+2)`; P > m_pack -> partitions `p >= m_pack` own no band, but they
    still partition the KV, stage their normalized partial, and are reduced into
    the owning bands like any other rank). Transport (mirrors
    `fa4_splitk_combine_write`, DSMEM edition):
      1. Every WG0 warp `g` stages its normalized band `g` into `stage_smem`
         (global slot `g`, 16 B block-major) so peers can DSMEM-read this CTA's
         contribution to the band THEY own; warp 0 writes `(M_cta, L_cta)` to
         `maxsum_smem`. (The own-band slot is staged too but read by no peer --
         a uniform all-band stage keeps the offset partition-independent.)
      2. `named_barrier[WARPGROUP_SIZE](0)` fences the `m_pack` warps' staging.
      3. Publish: WG0 warp 0 ONLY `arrive_cluster`s every peer, so exactly
         `BM * num_partitions == rows * P` arrivals reach each CTA's publish
         barrier (init count `config.BM * cluster_dim.x`). All `m_pack` warps
         arriving would `m_pack`x-overshoot -> deadlock/corruption.
      4. All entering WG0 threads `wait(0)` -- peers' staged bands + `(M,L)` are
         visible (phase 0; the publish barrier is used once).
      5. Each owning warp `g` gathers every partition's `(M,L)` via
         `load_cluster_smem`, forms `Gmax = max_p M_p` and
         `w_p = L_p*exp2((M_p-Gmax)*c)/Gsum`, seeds its band from `o_band_norm`
         scaled by `w[own]`, DSMEM-accumulates the other ranks' bands, packs into
         `o_smem`, then TMAs its band columns to gmem (per-block
         `async_copy_from_col`, matching the `tma_bpo == 0` store descriptor).

    Empty-partition neutral element: an empty CTA passes a zero `o_band_norm`
    with `own_max = -inf, own_sum = 0`, so `w[own] = 0*exp2(-inf) = 0` (never
    `0*inf`), its staged (finite-0) band drops out of every peer's combine, and
    it still writes its OWN band from the non-empty peers. Relies on the
    >=1-active-partition (rank 0 front-loaded) invariant so `Gmax`/`Gsum` are
    finite. No round-2 cluster barrier: `o_smem` is a dedicated region (not
    `stage_smem`), so the pack clobbers nothing peers read; the kernel's terminal
    `cluster_sync()` keeps the peer-read staged bands alive through the reads.
    """
    comptime accum_dtype = DType.float32
    comptime own_cols = ov_depth // m_pack
    comptime stage_K = 16 // size_of[accum_dtype]()  # f32 -> 4
    comptime own_sblocks = own_cols // stage_K
    comptime o_sw_K = 16 // size_of[output_type]()  # bf16 -> 8, f32 -> 4
    comptime own_oblocks = own_cols // o_sw_K
    comptime sub = o_sw_K // stage_K  # stage-blocks per output block (1 or 2)
    comptime assert (
        P == 2 or P == 4 or P == 6 or P == 8 or P == 10 or P == 16
    ), (
        "WS band-major combine: P must be an even split-K count in"
        " {2,4,6,8,10,16}"
    )
    comptime assert ov_depth % m_pack == 0
    comptime assert own_cols % stage_K == 0
    comptime assert own_cols % o_sw_K == 0

    var r = Int(local_row)

    # WG0-only. WG1 skips to the caller's terminal cross-WG barrier.
    if warp_group_idx != UInt32(0):
        return

    # ---- (1) stage ALL m_pack normalized bands + (M_cta, L_cta) ----
    comptime for lb in range(own_sblocks):
        var sblk = band_g * UInt32(own_sblocks) + UInt32(lb)
        var v = SIMD[accum_dtype, stage_K]()
        comptime for k in range(stage_K):
            v[k] = o_band_norm[lb * stage_K + k]
        (
            stage_smem
            + sblk * UInt32(rows * stage_K)
            + local_row * UInt32(stage_K)
        ).store(v)
    if band_g == UInt32(0):
        maxsum_smem[r * 2] = own_max
        maxsum_smem[r * 2 + 1] = own_sum

    # ---- (2) fence the m_pack warps' staging (WG0-local) ----
    named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))

    @__parameter
    @always_inline
    def reduce_scatter_p[P_static: Int]():
        comptime bpp = ceildiv(m_pack, P_static)
        var e = elect()
        # ---- (3) publish: WG0 warp 0 ONLY, exactly rows*P_static arrivals ----
        if band_g == UInt32(0):
            if e != 0:
                ragged_tma_store.prefetch_descriptor()
            comptime for pp in range(P_static):
                publish_mbar[].arrive_cluster(UInt32(pp))
        # ---- (4) all WG0 threads wait: peers' staged bands + (M,L) visible ----
        publish_mbar[].wait(UInt32(0))

        # ---- (5) combine: warp g owns band g in [p*bpp, +bpp) ----
        comptime for p_static in range(P_static):
            comptime own_lo = p_static * bpp
            comptime own_hi = min((p_static + 1) * bpp, m_pack)
            if partition_idx == UInt32(p_static):
                comptime for gg in range(own_lo, own_hi):
                    if band_g == UInt32(gg):
                        # Loop A: cluster LSE over the P partitions' (M,L).
                        var b_rank: Int = Int(partition_idx)
                        var pmax = Array[Float32, P](uninitialized=True)
                        var psum = Array[Float32, P](uninitialized=True)
                        var gmax: Float32 = own_max
                        pmax[0] = own_max
                        psum[0] = own_sum
                        comptime for pp in range(P_static - 1):
                            var rr = pp if pp < b_rank else pp + 1
                            var ms = load_cluster_smem[.float32, 2](
                                maxsum_smem + local_row * 2, UInt32(rr)
                            )
                            pmax[pp + 1] = ms[0]
                            psum[pp + 1] = ms[1]
                            gmax = max(gmax, pmax[pp + 1])
                        var w = Array[Float32, P](uninitialized=True)
                        var gsum: Float32 = 0
                        comptime for pp in range(P_static):
                            var d: Float32 = pmax[pp] - gmax
                            comptime if use_fma:
                                d = mul_ftz(d, scale_log2e)
                            w[pp] = mul_ftz(psum[pp], exp2(d))
                            gsum = gsum + w[pp]
                        var inv_gsum = recip(gsum)
                        comptime for pp in range(0, P_static, 2):
                            var wp = f32x2(w[pp], w[pp + 1]) * inv_gsum
                            w[pp] = wp[0]
                            w[pp + 1] = wp[1]
                        # Seed own band by rank b's weight (stored at slot 0).
                        var o_out = Array[Scalar[accum_dtype], own_cols](
                            uninitialized=True
                        )
                        var wb = w[0]
                        comptime for i in range(0, own_cols, 2):
                            var pair = (
                                f32x2(o_band_norm[i], o_band_norm[i + 1]) * wb
                            )
                            o_out[i] = pair[0]
                            o_out[i + 1] = pair[1]
                        # Accumulate the other np-1 ranks via DSMEM band reads.
                        for r_base in range(P_static - 1):
                            var rr = r_base if r_base < b_rank else r_base + 1
                            var wr: Float32 = 0
                            comptime for pp in range(P_static - 1):
                                wr = w[pp + 1] if pp == r_base else wr
                            comptime for lb in range(own_sblocks):
                                comptime sblk = gg * own_sblocks + lb
                                var vec = load_cluster_smem[
                                    DType.float32, stage_K
                                ](
                                    stage_smem
                                    + UInt32(sblk * rows * stage_K)
                                    + local_row * UInt32(stage_K),
                                    UInt32(rr),
                                )
                                comptime for c in range(stage_K // 2):
                                    comptime obase = lb * stage_K + 2 * c
                                    var acc = f32x2(
                                        o_out[obase], o_out[obase + 1]
                                    )
                                    acc = f32x2(vec[2 * c], vec[2 * c + 1]).fma(
                                        f32x2(wr), acc
                                    )
                                    o_out[obase] = acc[0]
                                    o_out[obase + 1] = acc[1]
                        # Pack the combined band -> o_smem (SWIZZLE_NONE 16 B
                        # block-major, mirrors fa4_ws_level2_reduce_scatter_write).
                        comptime col0 = gg * own_cols
                        comptime for ob in range(own_oblocks):
                            comptime oblk = (col0 // o_sw_K) + ob
                            var packed = SIMD[.uint32, 4]()
                            comptime for sb in range(sub):
                                comptime lsblk = ob * sub + sb
                                comptime for hc in range(2):
                                    comptime obase = lsblk * stage_K + 2 * hc
                                    var acc = f32x2(
                                        o_out[obase], o_out[obase + 1]
                                    )
                                    comptime cc = sb * 2 + hc
                                    comptime if size_of[output_type]() == 4:
                                        var u2 = bitcast[.uint32, 2](acc)
                                        packed[2 * cc] = u2[0]
                                        packed[2 * cc + 1] = u2[1]
                                    else:
                                        packed[cc] = bitcast[.uint32, 1](
                                            acc.cast[output_type]()
                                        )
                            st_shared_v4_b32(
                                o_smem,
                                oblk * rows * o_sw_K + r * o_sw_K,
                                packed,
                            )
                # ---- (6) fence o_smem writes across WG0 before the TMA ----
                named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
                # ---- (7) warp 0 TMAs this partition's owned band(s) ----
                if band_g == UInt32(0):
                    fence_async_view_proxy()
                    comptime for gg in range(own_lo, own_hi):
                        comptime col0 = gg * own_cols
                        comptime for ob in range(own_oblocks):
                            comptime j_global = (col0 // o_sw_K) + ob
                            ragged_tma_store.async_copy_from_col[j_global](
                                o_smem,
                                ragged_idx=out_row_idx,
                                dynamic_dim=UInt32(num_output_rows),
                                middle_idx=out_head_idx,
                                elect=e,
                            )
                    cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()

    # Static-per-P: each partition count P compiles its OWN kernel, so
    # `num_partitions == P == config.splitk_partitions` (see the call sites) and
    # the active count is comptime. Dispatch the band split on comptime `P` --
    # mirrors the 1Q `fa4_splitk_reduce_scatter_write` (reduce_scatter_p[P]()).
    reduce_scatter_p[P]()


@always_inline
def fa4_tma_store_o_smem[
    output_type: DType,
    //,
    ov_depth: Int,
    output_swizzle_mode: TensorMapSwizzle,
    bm: Int,
    middle_dim: Int,
    group: Int = 1,
    # Inferred from the store arg (like the split-K/scale-write helpers). This
    # egress only implements the per-block (`tma_bpo == 0`) copy, so callers must
    # hand it the rank-3 store: the WS combine forces `tma_blocks_per_op=0` at the
    # descriptor sites (dispatch.mojo / kernel.mojo), exactly as the 1Q split-K
    # path does. Declaring it here (vs a hard `0`) lets the call type-check inside
    # `fa4_softmax`'s generic body, where the store's `tma_blocks_per_op` is the
    # inferred `_` symbol rather than a literal.
    tma_bpo: Int = 0,
](
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=bm,
        BN=ov_depth,
        middle_dim=middle_dim,
        group=group,
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """TMA egress of a fully-written `o_smem` to gmem: the per-block
    (`tma_blocks_per_op == 0`) tail factored out of `fa4_scale_write_output`'s
    `else` branch.

    Unlike `fa4_scale_write_output`, this does NOT read TMEM or scale -- it
    assumes `o_smem` already holds the SWIZZLE_NONE 16 B block-major output (as
    the WS combine produces). One `named_barrier` fences the combine's `o_smem`
    writes; then warp 0 issues one `async_copy_from_col` per depth block and the
    WG waits the bulk group. See §5c step 4.
    """
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime o_sw_blocks = align_up(ov_depth, o_sw_K) // o_sw_K
    var e = elect()
    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
    if local_warp_idx == 0:
        fence_async_view_proxy()
        comptime for blk in range(o_sw_blocks):
            ragged_tma_store.async_copy_from_col[blk](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
        cp_async_bulk_commit_group()
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_splitk_stage_partial[
    band_cols: Int,
    //,
    output_type: DType,
    config: FA4Config,
    own_off: Int,
    own_iters: Int,
    single_source: Bool = False,
    zero_fill: Bool = False,
    do_peers: Bool = True,
    do_own: Bool = True,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
](
    local_row: UInt32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    stage_smem: SharedMemPointer[Float32],
    own_o_tmem: TMemTile[.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[.float32, config.BM, config.padded_ov_depth],
    mut o_final: Array[Float32, band_cols],
):
    """Split-K pass 1: LSE-combine the two WG O fragments into this partition's
    NORMALIZED `O_cta` (f32) over the FULL `[0, iters)` depth range, routing each
    block by destination. The OWN band `[own_off, own_off+own_iters)` is written
    into the `o_final` register array UNSCALED (the combine scales it by `w[b]`
    after the publish barrier); every other (PEER-visible) block is stored to
    `stage_smem` in the SWIZZLE_NONE block-major, bank-conflict-free layout
    `fa4_scale_write_output` uses, at f32 width.

    The own-band smem columns are read by no peer (each peer reads only ITS band
    from us, and the bands partition `[0, iters)` disjointly), so keeping them in
    registers elides a pure self round-trip (the old code wrote them to smem and
    the combine read them straight back). The per-block normalized `O_cta` value
    is byte-identical to what the smem path stored; only the destination (regs vs
    smem) and WHEN `w[b]` is applied differ.

    `do_peers`/`do_own` split this into two passes so the caller can publish the
    peer bands (`arrive_cluster`) BEFORE the own band is computed: the peers pass
    (`do_own=False`) writes only the smem-routed blocks; the own pass
    (`do_peers=False`) writes only `o_final`. Computing the own band LAST (after
    the publish) overlaps its TMEM load with the cluster sync latency and keeps
    `o_final` out of the peer-staging live range. Both default True (single pass).

    Each thread owns row `local_row` (= tid%128, via `tcgen05_ld[datapaths=32]`).
    For each 4-element f32 block `fblk` a smem-routed write is one
    `st.shared.v4.b32` (16 B) at `stage_smem[fblk*BM*F + local_row*F + k]`,
    `F = 16/size_of[f32] = 4`; the offset is a pure function of (fblk, local_row,
    k) identical across partitions, so a peer's same-offset DSMEM read lands in
    the matching slot. Each output block (`o_sw_K` bf16 cols) maps to `o_sw_K//F`
    f32 blocks. `final_scale_*` are already this thread's per-row scales (NO
    `warp.shuffle_idx`).

    `single_source` (the T==1 split-K path): only `o0` exists (WG1 returned
    early; `peer_o_tmem`/`o1` is never produced), so skip the peer load and just
    scale `o0` by `final_scale_local` (= `inv_row_sum`). `final_scale_peer`/
    `peer_o_tmem` are unused.

    `zero_fill` (the empty-partition path): this partition has no tiles, so no O
    was produced -- write finite 0.0 at every block (own band -> `o_final`,
    peers -> smem) so the writer's weight-0 DSMEM read of this peer is a finite
    zero (avoids `0*inf = NaN`). No TMEM is read; `final_scale_*`/`own_o_tmem`/
    `peer_o_tmem` are all unused.

    `own_iters == 0` (owns no band, e.g. non-pow2 depth at high P): the own band
    is empty so the complement is the FULL range -- every block is staged to
    smem (peers still need this partition's O_cta for THEIR bands). `o_final` is
    zero-length and untouched.

    `output_swizzle_mode` MUST be the OUTPUT store's swizzle (SWIZZLE_NONE for
    MHA split-K), NOT the QKV `config.swizzle_mode`; the caller passes
    `type_of(ragged_tma_store).swizzle_mode`. `o_sw_K`/`iters` then match the
    caller's `iters_total` (the compile assert below catches a mismatch).
    """
    comptime assert config.num_q == 1
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime iters = align_up(config.ov_depth, o_sw_K) // o_sw_K
    comptime assert own_iters == 0 or own_off + own_iters <= iters
    comptime assert band_cols == own_iters * o_sw_K
    comptime BM = config.BM
    # 16 B / 4 B = 4 f32 per `st.shared.v4.b32` (the bank-conflict-free 16 B
    # store width; an 8-f32 store would be two instructions). Each output block
    # (o_sw_K bf16 cols) is `subblocks` f32 blocks.
    comptime F = 16 // size_of[DType.float32]()
    comptime assert o_sw_K % F == 0
    comptime subblocks = o_sw_K // F
    comptime accum_dtype = DType.float32

    var row_F = local_row * UInt32(F)

    # Stage the FULL O_cta over ALL `iters` blocks; route OWN band -> o_final
    # (regs, unscaled), every other block -> stage_smem. The f32x2 combine feeds
    # the destination register directly -- no `o_vals` round-trip.
    comptime for j in range(iters):
        comptime in_own = own_iters > 0 and j >= own_off and j < (
            own_off + own_iters
        )
        # `do_peers`/`do_own` split staging into two passes so the orchestrator
        # can publish (`arrive_cluster`) the peer bands BEFORE the own band is
        # computed: peers pass (`do_own=False`) writes only smem-routed blocks,
        # own pass (`do_peers=False`) writes only `o_final`. Gating the whole
        # block (incl. the `tcgen05_ld`) means each pass touches only its subset;
        # total loads/stores are unchanged. Defaults run both (single pass).
        comptime process = (in_own and do_own) or ((not in_own) and do_peers)
        comptime if process:
            comptime col = j * o_sw_K
            var vblk = Array[SIMD[accum_dtype, F], subblocks](
                uninitialized=True
            )
            comptime if zero_fill:
                # No TMEM read: finite 0.0 (peers' weight-0 DSMEM read must be
                # finite, else 0*inf=NaN; the own band's 0 is scaled by w[b]==0).
                comptime for sb in range(subblocks):
                    vblk[sb] = SIMD[accum_dtype, F](0)
            else:
                var own = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=o_sw_K,
                    dtype=accum_dtype,
                    pack=False,
                    width=o_sw_K,
                ](own_o_tmem.tmem_addr + UInt32(col))
                comptime if single_source:
                    # T==1: only o0; normalize by inv_row_sum (final_scale_local).
                    comptime for sb in range(subblocks):
                        var v = SIMD[accum_dtype, F]()
                        comptime for c in range(F // 2):
                            comptime e = sb * F + 2 * c
                            var pair = (
                                SIMD[accum_dtype, 2](own[e], own[e + 1])
                                * final_scale_local
                            )
                            v[2 * c] = pair[0]
                            v[2 * c + 1] = pair[1]
                        vblk[sb] = v
                else:
                    var peer = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=o_sw_K,
                        dtype=accum_dtype,
                        pack=False,
                        width=o_sw_K,
                    ](peer_o_tmem.tmem_addr + UInt32(col))
                    # own*final_scale_local + peer*final_scale_peer (f32x2 fma),
                    # identical to combine_pack_o_row modulo f32 not bf16.
                    comptime for sb in range(subblocks):
                        var v = SIMD[accum_dtype, F]()
                        comptime for c in range(F // 2):
                            comptime e = sb * F + 2 * c
                            var own_c = SIMD[accum_dtype, 2](own[e], own[e + 1])
                            var peer_c = SIMD[accum_dtype, 2](
                                peer[e], peer[e + 1]
                            )
                            var comb = peer_c.fma(
                                SIMD[accum_dtype, 2](final_scale_peer),
                                own_c * final_scale_local,
                            )
                            v[2 * c] = comb[0]
                            v[2 * c + 1] = comb[1]
                        vblk[sb] = v

            comptime if in_own:
                # Own band -> o_final registers (UNSCALED; combine applies w[b]).
                comptime local_iter = j - own_off
                comptime for sb in range(subblocks):
                    comptime ocol = local_iter * o_sw_K + sb * F
                    comptime for k in range(F):
                        o_final[ocol + k] = vblk[sb][k]
            else:
                # Peer-visible block -> stage_smem (16 B v4 f32, block-major).
                comptime for sb in range(subblocks):
                    comptime fblk = j * subblocks + sb
                    (stage_smem + UInt32(fblk * BM * F) + row_F).store(vblk[sb])


@always_inline
def fa4_splitk_combine_write[
    output_type: DType,
    band_cols: Int,
    //,
    config: FA4Config,
    P: Int,
    wg_j_offset: Int,
    iters_per_wg: Int,
    use_fma: Bool,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
    tma_bpo: Int = 0,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    own_rank: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    stage_smem: SharedMemPointer[Float32],
    maxsum_smem: SharedMemPointer[Float32],
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    publish_mbar: MBarType,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
    mut o_final: Array[Float32, band_cols],
):
    """Split-K pass 2 (reduce-scatter): EVERY partition's WG0 owns the
    depth-column band `[wg_j_offset, wg_j_offset+iters_per_wg)` (output blocks).
    It reads every partition's `O_cta` FOR ITS BAND + every partition's
    `(max,sum)`, forms the cluster LSE `Gmax`/`Gsum` and per-row weights
    `w_p = sum_p*exp2((max_p-Gmax)*c)/Gsum` (per-thread scalars -- this thread
    owns its row, NO `warp.shuffle_idx`), weight-sums
    `O_final = Σ_p O_cta_p*w_p` (the `sum_p` cancels the `O_cta` normalization),
    casts to bf16, and TMA-stores its band columns to gmem.

    `o_final` arrives PRE-SEEDED by `fa4_splitk_stage_partial` with THIS
    partition's own-rank `b` NORMALIZED `O_cta` band (UNSCALED). The own band is
    read by no peer (each peer reads only ITS band from us), so it is handed off
    in registers rather than round-tripped through `stage_smem`. This pass scales
    that seed in place by `w[b]`, then accumulates the other np-1 ranks via DSMEM
    reads of their `stage_smem` bands.

    Layout is the SWIZZLE_NONE row-per-thread pattern (mirrors
    `fa4_scale_write_output`): the f32 stage is 16 B (`F=4`) block-major, read in
    bulk v4 DSMEM loads; the bf16 write is 16 B (`o_sw_K=8`) block-major.

    NO round-2 cluster barrier: the bf16 pack is retargeted into THIS partition's
    OWN-band f32 stage slice (which no peer reads -- each peer reads only ITS band
    from us), so the write clobbers nothing peers are reading. The peer-read PEER
    bands of `stage_smem` are kept alive until every partition finishes reading by
    the kernel's terminal `cluster_sync()` (kernel.mojo) -- the same trailing-sync
    discipline the DSMEM spike validated. Only round-1 (publish_mbar phase 0, in
    the caller) is needed, to make peers' staged O_cta + (max,sum) visible before
    the DSMEM reads. (An earlier reduce-scatter revision aliased the bf16 write
    over peer-read f32 bytes and so needed a phase-1 barrier; the own-slice
    retarget removes that hazard. Writer-combines-all needed only a WG-local
    barrier because peers never wrote.)
    """
    comptime assert config.num_q == 1
    # The SIMD-2 weight-normalize loop below strides `range(0, P, 2)` and reads
    # `w[p+1]`, so P must be EVEN; the split-K whitelist
    # (FA4Config.supported()) admits only even P (2,4,6,8,10,16). Fail loudly if
    # a future odd P ever slips through instead of reading `w[P]` OOB.
    comptime assert (
        P % 2 == 0
    ), "split-K combine requires an even partition count P"
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime iters = align_up(config.ov_depth, o_sw_K) // o_sw_K
    comptime assert iters_per_wg >= 1
    comptime assert wg_j_offset + iters_per_wg <= iters
    comptime BM = config.BM
    # 16 B / 4 B = 4 f32 per v4 load (the staging block width); each output
    # block (o_sw_K bf16 cols) is `subblocks` f32 blocks.
    comptime F = 16 // size_of[DType.float32]()
    comptime assert o_sw_K % F == 0
    comptime subblocks = o_sw_K // F
    comptime assert band_cols == iters_per_wg * o_sw_K
    comptime accum_dtype = DType.float32
    comptime o_swizzle = make_swizzle[output_type, output_swizzle_mode]()

    var e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Per-thread (max,sum) slot. combine runs WG0-only, so tid == local_row;
    # kept general to match how the caller wrote maxsum (`maxsum[tid*2]`).
    var tid = warp_group_idx * UInt32(WARPGROUP_SIZE) + local_row

    # --- Loop A: gather every partition's (max,sum) for THIS thread's row,
    # indexed by GLOBAL cluster rank r in [0, np). r == own_rank uses this
    # partition's own (max,sum) from regs; every other rank is a DSMEM read of
    # that rank's slot. (Reduce-scatter: EVERY partition runs this, so "own" is
    # rank `b`, not rank 0 -- reading absolute ranks 1..np-1 would skip rank 0
    # and double-count `b` for b != 0.) `P` (comptime ceiling) sizes the
    # buffers; loops run to RUNTIME `np`, so slots [np, P) are never read.
    var b: Int = Int(own_rank)
    var pmax = Array[Float32, P](uninitialized=True)
    var psum = Array[Float32, P](uninitialized=True)
    var gmax: Float32 = own_max
    # Own partition's (max,sum) goes to comptime slot 0 (not slot `b`); the
    # other ranks fill comptime slots `p+1`. This keeps every pmax/psum index
    # static so the arrays stay in registers (dynamic indexing would spill to
    # local memory). The slot<->rank mapping is just a permutation; the
    # reductions below are order-independent, and the actual rank `r` is still
    # tracked for the DSMEM read.
    pmax[0] = own_max
    psum[0] = own_sum
    comptime for p in range(P - 1):
        var r = p if p < b else p + 1
        var ms = load_cluster_smem[.float32, 2](
            maxsum_smem + tid * 2, UInt32(r)
        )
        pmax[p + 1] = ms[0]
        psum[p + 1] = ms[1]
    # Cluster-global max via 3-arg `max_ftz` (one FMNMX3 per two partitions)
    # rather than a 2-arg chain. `gmax` seeds at pmax[0] (own_max); fold the
    # remaining P-1 slots two at a time. P is even (split-K whitelist), so the
    # final odd slot folds alone. Order-independent: max is exact/associative.
    comptime for p in range(1, P, 2):
        comptime if p + 1 < P:
            gmax = max_ftz(gmax, pmax[p], pmax[p + 1])
        else:
            gmax = max_ftz(gmax, pmax[p])

    var w = Array[Float32, P](uninitialized=True)
    var gsum: Float32 = 0
    comptime for p in range(P):
        var d: Float32 = pmax[p] - gmax
        comptime if use_fma:
            d = d * scale_log2e
        w[p] = psum[p] * exp2(d)
        gsum = gsum + w[p]
    var inv_gsum = recip(gsum)
    comptime for p in range(0, P, 2):
        var wp = SIMD[.float32, 2](w[p], w[p + 1]) * inv_gsum
        w[p] = wp[0]
        w[p + 1] = wp[1]

    # --- Pass 2a: O_final[band] = Σ_r O_cta_r[row, band] * w_r over GLOBAL
    # cluster ranks r in [0, np). `w_r` is a per-thread scalar (this thread's
    # row). The seed (rank `b`, this partition's own band) arrives in `o_final`
    # from staging UNSCALED -- scale it in place by `w[0]` (rank `b`'s weight,
    # stored at slot 0; NO stage_smem
    # read-back; those columns are read by no peer). Then accumulate the other
    # np-1 ranks, skipping `b`, via 16 B v4 DSMEM reads of their `stage_smem`.
    # The partition loop is the outermost runtime loop so each rank's unrolled v4
    # reads issue back-to-back (DSMEM-latency MLP). `o_final` spans the FULL band
    # and is written once per block, so the depth-128 column-clobber is
    # impossible.
    var row_F = local_row * UInt32(F)

    # Seed scale: o_final holds the own-rank `b` normalized O_cta band; * its
    # weight, which lives at comptime slot 0 (own rank was stored there above).
    var wb = w[0]
    comptime for i in range(0, band_cols, 2):
        var pair = SIMD[accum_dtype, 2](o_final[i], o_final[i + 1]) * wb
        o_final[i] = pair[0]
        o_final[i + 1] = pair[1]

    # Accumulate the other np-1 ranks (partition loop outermost), skipping the
    # local partition `b` that already seeded `o_final`.
    for r_base in range(P - 1):
        var r = r_base if r_base < b else r_base + 1
        # Rank `r` was stored at comptime slot `r_base + 1` in Loop A (same
        # `p<->r_base` enumeration), so its weight is `w[r_base + 1]`. Select
        # it via a static-index mux so `w` stays in registers (a dynamic `w[r]`
        # would spill to local memory). The `= 0` init is dead (exactly one
        # branch matches, since r_base in [0, np-1) is within [0, P-1)) but is
        # needed for definite assignment.
        var wr: Float32 = 0
        comptime for p in range(P - 1):
            wr = w[p + 1] if p == r_base else wr
        comptime for iter in range(iters_per_wg):
            comptime j = wg_j_offset + iter
            comptime for sb in range(subblocks):
                comptime fblk = j * subblocks + sb
                comptime ocol = iter * o_sw_K + sb * F
                var vec: SIMD[accum_dtype, F] = load_cluster_smem[
                    DType.float32, F
                ](stage_smem + UInt32(fblk * BM * F) + row_F, UInt32(r))
                comptime for c in range(F // 2):
                    var acc = SIMD[accum_dtype, 2](
                        o_final[ocol + 2 * c], o_final[ocol + 2 * c + 1]
                    )
                    var pair = SIMD[accum_dtype, 2](
                        vec[2 * c], vec[2 * c + 1]
                    ).fma(SIMD[accum_dtype, 2](wr), acc)
                    o_final[ocol + 2 * c] = pair[0]
                    o_final[ocol + 2 * c + 1] = pair[1]

    # --- Pass 2b: pack O_final band -> bf16 row-major -> o_smem (scale 1.0,
    # already normalized), then per-block TMA of the band columns to gmem.
    #
    # NO round-2 cluster barrier here (the old `publish_mbar.wait(1)`): the bf16
    # pack is retargeted into THIS partition's OWN-band f32 stage slice, which no
    # peer ever reads (each peer reads only ITS band from us; the bands partition
    # `[0, iters)` disjointly). So the write clobbers nothing peers are reading
    # and needs no cluster-wide fence -- the kernel's terminal `cluster_sync()`
    # (kernel.mojo) keeps the peer-read PEER bands alive until every partition
    # finishes reading. The own band's f32 data starts at f32-block
    # `wg_j_offset*subblocks`, so shift the bf16 base by
    # `wg_j_offset*(subblocks-1)*BM*o_sw_K` output elements; bf16 block
    # `j == wg_j_offset+iter` then lands at byte `(wg_j_offset*subblocks+iter)*
    # BM*16`, strictly inside the dead own slice (no overlap with peer bands above
    # or below). `async_copy_from_col[j]` reads smem at `base + j*BM*o_sw_K` but
    # its gmem coordinate is `j*swizzle_granularity` (independent of `base`), so
    # the shifted base + global `j` still stores to the correct global gmem
    # column. (SWIZZLE_NONE -> simple block stride; the shift is block-aligned, so
    # the within-block `o_swizzle(o_inner)` row layout is undisturbed.)
    comptime o_own_shift = wg_j_offset * (subblocks - 1) * BM * o_sw_K
    var o_smem_own = o_smem_arg + o_own_shift
    comptime for iter in range(iters_per_wg):
        comptime j = wg_j_offset + iter
        var packed = pack_row[output_type, w=o_sw_K, start=iter * o_sw_K](
            o_final
        )
        var o_inner = Int(local_row) * o_sw_K
        # Explicit v4.b32 (STS.128): a plain `.store(packed)` scalarizes to 4x
        # STS.32 here because `packed` is packed from the long-lived `o_final`
        # accumulator (non-contiguous in-place F2FP pack outputs), which ptxas
        # can't fuse -- the 4 B stores then hit only every 4th bank (4-way
        # conflict). Forcing the wide store keeps it one bank-conflict-free 16 B
        # transaction (see `st_shared_v4_b32`).
        st_shared_v4_b32(
            o_smem_own, j * BM * o_sw_K + o_swizzle(o_inner), packed
        )

    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
    if local_warp_idx == 0:
        fence_async_view_proxy()
        comptime for iter in range(iters_per_wg):
            comptime j_global = wg_j_offset + iter
            ragged_tma_store.async_copy_from_col[j_global](
                o_smem_own,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
        cp_async_bulk_commit_group()
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_splitk_reduce_scatter_write[
    output_type: DType,
    //,
    config: FA4Config,
    P: Int,
    use_fma: Bool,
    single_source: Bool = False,
    zero_fill: Bool = False,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
    tma_bpo: Int = 0,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    partition_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    stage_smem: SharedMemPointer[Float32],
    maxsum_smem: SharedMemPointer[Float32],
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    publish_mbar: MBarType,
    own_o_tmem: TMemTile[.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[.float32, config.BM, config.padded_ov_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """Split-K pass 2 dispatch (reduce-scatter): pick THIS partition's
    depth-column band from the STATIC partition count `P` and the runtime
    `partition_idx`, then STAGE + publish + combine over it. `P` is
    `config.splitk_partitions` (each split-K kernel is compiled once per static
    `P`), so the band's `wg_j_offset`/`iters_per_wg` are already comptime; only
    `partition_idx` is lifted via a comptime-for over the `P` partitions -- one
    runtime branch taken, the band params comptime per branch.

    Per band (WG0), staging split so the publish lands between peers and own:
      1. `fa4_splitk_stage_partial[do_own=False]` stages only the PEER-visible
         blocks to `stage_smem` (peers read those for THEIR bands).
      2. Phase-0 publish: WG0 `arrive_cluster`s every peer (peers read only our
         peer bands + maxsum, never our own band, so publish now).
      3. `fa4_splitk_stage_partial[do_peers=False]` stages the OWN band into the
         register array `o_final` (UNSCALED) LAST -- its TMEM load overlaps the
         cluster sync, and `o_final` stays out of the peer-staging live range.
      4. ALL entering WGs `wait(0)` (peers' staged `O_cta`/`(max,sum)` visible),
         then `fa4_splitk_combine_write` (WG0) scales the `o_final` seed by
         `w[b]`, DSMEM-accumulates the other ranks' bands, and TMA-stores its
         band -- packing the bf16 into its OWN-band dead f32 slice (no peer reads
         it), so no round-2 cluster barrier is needed (the terminal
         `cluster_sync()` keeps the peer-read bands alive through peers' reads).

    Called by BOTH warp groups (WG1 only participates in the phase-0 wait; WG0
    does the staging/combine). EVERY partition -- empty or not -- joins round-1
    (publish_mbar phase 0), so the publish_mbar count always completes. The
    own-no-band partition (non-pow2 depth at high P) still stages its FULL `O_cta`
    to smem (its complement is the full range) so peers can read it for their
    bands; it writes no bf16 output, so it just falls through to the kernel's
    terminal `cluster_sync()` after round-1.
    """
    comptime assert config.num_q == 1
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime iters_total = align_up(config.ov_depth, o_sw_K) // o_sw_K

    # `P` (== `config.splitk_partitions`, the static cluster size) is EVEN, in
    # {2,4,6,8,10,16}; only `partition_idx` (runtime) needs lifting to comptime,
    # via the comptime-for below. The band split `bpp = ceildiv(iters_total, P)`
    # is P-general: a non-pow2 P (6, 10) just yields uneven and/or empty
    # trailing bands, both of which are already handled below.

    @__parameter
    @always_inline
    def reduce_scatter_p[P_static: Int]():
        comptime bpp = ceildiv(iters_total, P_static)
        comptime for p_static in range(P_static):
            comptime ob = p_static * bpp
            comptime ipw = min(bpp, iters_total - ob) if ob < iters_total else 0
            if partition_idx == UInt32(p_static):
                comptime if ipw > 0:
                    comptime band_cols = ipw * o_sw_K
                    var o_final = Array[Float32, band_cols](uninitialized=True)
                    if warp_group_idx == UInt32(0):
                        # 1. Stage PEER bands -> smem (own band skipped).
                        fa4_splitk_stage_partial[
                            output_type,
                            config,
                            own_off=ob,
                            own_iters=ipw,
                            single_source=single_source,
                            zero_fill=zero_fill,
                            do_own=False,
                            output_swizzle_mode=output_swizzle_mode,
                        ](
                            local_row,
                            final_scale_local,
                            final_scale_peer,
                            stage_smem,
                            own_o_tmem,
                            peer_o_tmem,
                            o_final,
                        )
                        # 2. Phase-0 publish (per-row, every peer): peers
                        # read only our peer bands + maxsum, never our
                        # own band, so publish before computing own.
                        comptime for pp in range(P_static):
                            publish_mbar[].arrive_cluster(UInt32(pp))
                        # 3. Stage OWN band -> o_final LAST: its TMEM
                        # load overlaps the cluster sync, and o_final
                        # stays out of the peer-staging live range.
                        fa4_splitk_stage_partial[
                            output_type,
                            config,
                            own_off=ob,
                            own_iters=ipw,
                            single_source=single_source,
                            zero_fill=zero_fill,
                            do_peers=False,
                            output_swizzle_mode=output_swizzle_mode,
                        ](
                            local_row,
                            final_scale_local,
                            final_scale_peer,
                            stage_smem,
                            own_o_tmem,
                            peer_o_tmem,
                            o_final,
                        )
                    # 4. Both WGs wait; then combine over the band (WG0).
                    publish_mbar[].wait(UInt32(0))
                    if warp_group_idx == UInt32(0):
                        fa4_splitk_combine_write[
                            config,
                            P=P_static,
                            wg_j_offset=ob,
                            iters_per_wg=ipw,
                            use_fma=use_fma,
                            output_swizzle_mode=output_swizzle_mode,
                            tma_bpo=tma_bpo,
                        ](
                            local_row,
                            local_warp_idx,
                            UInt32(0),
                            partition_idx,
                            own_max,
                            own_sum,
                            scale_log2e,
                            stage_smem,
                            maxsum_smem,
                            o_smem_arg,
                            publish_mbar,
                            ragged_tma_store,
                            num_output_rows,
                            out_head_idx,
                            out_row_idx,
                            o_final,
                        )
                else:
                    # Owns no band (non-pow2 depth at high P): the
                    # complement is the FULL range, so still stage the
                    # whole O_cta to smem (peers read it for THEIR bands)
                    # and publish (round-1). No bf16 write here -> no
                    # round-2 needed; the staged O_cta stays alive for
                    # peers via the kernel's terminal `cluster_sync()`.
                    var o_final_empty = Array[Float32, 0](uninitialized=True)
                    if warp_group_idx == UInt32(0):
                        fa4_splitk_stage_partial[
                            output_type,
                            config,
                            own_off=0,
                            own_iters=0,
                            single_source=single_source,
                            zero_fill=zero_fill,
                            output_swizzle_mode=output_swizzle_mode,
                        ](
                            local_row,
                            final_scale_local,
                            final_scale_peer,
                            stage_smem,
                            own_o_tmem,
                            peer_o_tmem,
                            o_final_empty,
                        )
                        comptime for pp in range(P_static):
                            publish_mbar[].arrive_cluster(UInt32(pp))
                    publish_mbar[].wait(UInt32(0))

    reduce_scatter_p[P]()


@always_inline
def fa4_softmax[
    QScaleType: OptionalPointer,
    KScaleType: OptionalPointer,
    WSLSEType: OptionalPointer,
    qkv_dtype: DType,
    rope_dtype_: Optional[DType],
    scale_dtype_: Optional[DType],
    output_type: DType,
    MaskType: MHAMask,
    //,
    KVLUTType: MHAOperand,
    config: FA4Config[
        qkv_dtype, rope_dtype_=rope_dtype_, scale_dtype_=scale_dtype_
    ],
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    # Statically guarantees `num_output_rows > 0` for every warp group,
    # folding the 2Q output guard away. Only pass True when the calling
    # kernel routes every tile short enough for an empty row half
    # (`seq_len - prompt_offset <= wg_row_offset_seq`) to the 1Q body
    # instead (see `can_switch_to_1q()` use in `kernel.mojo` for MHA and the
    # thin `mla_prefill_kernel_generic` / `_per_token_scale` entrypoints for
    # MLA). A pure-2Q kernel (or one whose mask needs the runtime FULL_MASK
    # slow path, where MLA cannot switch) must leave this False.
    output_nonempty: Bool = False,
    # The generic MLA-prefill single-O path physically drops the 2nd softmax
    # warpgroup (launches 3 WGs / 384 threads). When True, WG0 is the ONLY
    # softmax WG, so the WG0<->WG1 pair rendezvous `named_barrier[2*WG](2)`
    # before the terminal TMEM dealloc must be skipped (no WG1 arrives ->
    # otherwise a 256-thread barrier with only 128 arrivals hangs). Default
    # False keeps the per-token-scale / blockscale siblings (still 2 softmax
    # WGs even for their single-O configs) and every non-single-O path
    # byte-identical.
    single_softmax_wg: Bool = False,
    # Effective cross-stage-P switch, computed once at the kernel level where
    # config, MaskType and the store shape are all visible. Defaults True so
    # the config-level gate alone decides for callers that do not thread it;
    # the MHA kernel passes its own narrower predicate.
    crossp_effective: Bool = True,
](
    smem: SM100AttentionSMem[config],
    tmem_addr: UInt32,
    score_row: UInt32,
    seq_info: SeqInfo,
    mask: MaskType,
    num_keys: UInt32,
    scale: Float32,
    max_seq_len: UInt32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        _,
        # 2Q: BM=128 (one Q-half per WG). 1Q: BM=128 (both WGs cover the
        # full BM=128 and write disjoint depth-column ranges). Use
        # `config.BM // config.num_q` so the type is consistent across
        # both modes; in 2Q this equals the historical `config.BM // 2`.
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # Inferred from the store the kernel built; forwarded to the writeback
        # helpers, which infer their own `tma_bpo` from this arg.
        tma_blocks_per_op=_,
    ],
    sink_weights: SinkType,
    q_scale: QScaleType = NullPointer[.float32, AddressSpace.SHARED](),
    k_scale: KScaleType = NullPointer[.float32, AddressSpace.SHARED](),
    # Workspace (traditional/unfused) split-K egress. A non-null `ws_lse_ptr` is
    # what turns the egress on: the KV is windowed by `ws_num_partitions`, the O
    # store is redirected into this partition's slice of the workspace
    # (`out_row_idx += p*ws_num_rows_q`, descriptor built over `o_partial`), and
    # the fused per-row LSE is `st.global`ed into `ws_lse_ptr`, whose layout is
    # `[P, num_rows_q, num_q_heads]`. `ws_num_rows_q` is the partition stride.
    # Defaulting the pointer to null keeps every non-workspace caller (2Q, MLA)
    # byte-identical without them naming any of these.
    ws_num_partitions: UInt32 = 1,
    ws_lse_ptr: WSLSEType = NullPointer[.float32](),
    ws_num_rows_q: UInt32 = 0,
):
    # Local aliases matching SM100MHA2Q comptime members
    comptime qkv_type = KVLUTType.dtype
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    comptime BM_mask: Int = config.PairBM_eff()
    comptime padded_ov_depth = config.padded_ov_depth
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group = config.cta_group()
    # The workspace split-K egress is on exactly when a buffer was handed in, so
    # the flag cannot disagree with the pointer.
    comptime workspace_split = not WSLSEType.is_null
    comptime assert (
        WSLSEType.is_null or WSLSEType.dtype == .float32
    ), "the workspace split-K LSE buffer is f32"
    # Warp-specialized packed-TMEM (1x4 / Layout-G) per-quarter score geometry.
    # `score_cols` is the physical S/P column extent one warp reads/writes;
    # `score_rows` is its query-row extent. Non-WS (m_pack == 1): score_cols ==
    # BN and score_rows == BM // 2, so every substitution below folds to the
    # historical (byte-identical) instruction stream.
    comptime score_cols = config.BN // config.m_pack
    comptime score_rows = config.MMA_M if config.use_ws else config.BM // 2

    var mbars = smem.misc_mbars()
    comptime MiscMBarsType = type_of(mbars)

    # MMA types for TMEM access
    comptime UMMA0Type = SM100TensorAccumulator[
        qkv_type,
        accum_dtype,
        MMA_M=config.MMA_M,
        MMA_N=BN,
        BK=align_up(config.qk_depth, config.MMA_K),
        a_tmem=False,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        num_stages=config.num_qk_stages,
        cta_group=cta_group,
    ]
    # `MMA_N` DIVERGES from the authoritative P@V declaration in
    # `mma_warp.mojo` (`MMA_N=pv_mma_n`). Inert: only `AType` and
    # `use_3_then_1_split` are read off this alias, and neither depends on
    # `MMA_N`. Fixing it in place would duplicate `mma_warp`'s three-way ladder;
    # the exit is one shared accumulator alias replacing both declarations.
    comptime UMMA1Type = SM100TensorAccumulator[
        qkv_type,
        accum_dtype,
        MMA_M=config.MMA_M,
        MMA_N=padded_ov_depth,
        BK=BN,
        a_tmem=True,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        num_stages=config.num_pv_stages,
        cta_group=cta_group,
    ]
    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.qk_depth,
        config.padded_qk_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    # `tmem_addr` passed in by register (read once post-barrier in the kernel
    # prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var o_smem = smem.o_smem[output_type]()
    var o_prod_mbar: MBarType = (
        mbars.mbar_base + MiscMBarsType.O_producer_offset
    )
    var s_tmem: UInt32 = tmem_addr + UInt32(config.TMEM_S0)

    # var tid = UInt32(thread_idx.x)
    var tid = llvm_opaque_tid()
    var row = tid % 128
    var warp_idx: UInt32 = warp.broadcast(tid // 32)
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    # WS {warp, lane} -> {key-partition, query-row} map. Layout-G (m_pack==4):
    # `warps_per_partition == 1`, so `partition_g == warp_idx & 3` and
    # `row_half` is always 0 -- every substitution below folds to the
    # historical (byte-identical) `warp_idx & 3` / `warp_idx % 4`. Layout-E
    # (m_pack==2): `warps_per_partition == 2`; the HIGH bit of `warp_idx & 3`
    # selects the key-partition (warp-pairs {0,1} -> p0, {2,3} -> p1) and the
    # LOW bit selects the 32-row half of the 64-row tile -- CONTIGUOUS, not
    # interleaved (corroborated by `thread_tile_row = tid % config.BM`, which
    # already yields `(warp_idx&1)*32+lane` at BM=64, see `:2218` below). Only
    # meaningful inside `config.use_ws` blocks. Non-WS (m_pack==1) never reads
    # this (dead, no crash risk either way since `warps_per_partition` would
    # be 4 there too, folding `partition_g` to 0 -- see `attention.mojo`'s
    # `use_ws` gate).
    comptime warps_per_partition = 4 // config.m_pack
    var partition_g: UInt32 = (warp_idx & 3) // UInt32(warps_per_partition)
    # Per-thread BM row within the current Q tile.
    # 2Q (BM = 256): WG0 covers BM rows [0, 128) and WG1 covers
    # [128, 256), so `tid` directly indexes the BM row.
    # 1Q (BM = 128): both WGs share the same BM rows [0, 128); folding
    # WG1's `tid` (128..255) back to [0, 128) via `tid % BM` ensures
    # the per-thread (Q row, head) mapping is identical across WGs.
    # Using bare `tid` in 1Q would shift WG1's score_row by `BM_eff`,
    # which leaks OOB K positions into the softmax for tiles whose
    # `score_row + BM_eff` exceeds `num_keys` (the OOB columns are not
    # masked by SlidingWindow's UPPER|LOWER strategy and TMA-padded
    # K=0 / V=0 then dilutes the output toward 0).
    var thread_tile_row: UInt32 = tid % UInt32(config.BM)

    var cta_q_offset: UInt32 = 0
    comptime if config.pair_cta:
        cta_q_offset = UInt32(
            warp.broadcast(block_rank_in_cluster()) % 2
        ) * UInt32(config.BM_eff())

    # 2-Q / two-pipeline path: WG1's S region starts one S-block in. The block
    # width is s_cols = BN // m_pack == TMEM_S1 - TMEM_S0 (== BN for the non-WS
    # m_pack==1 layout, so byte-identical). Under packed WS TMEM all m_pack warps
    # share this base; the HW subpartition routes each warp its column-quarter,
    # so there is NO per-warp column offset here.
    s_tmem += UInt32(config.TMEM_S1 - config.TMEM_S0) * warp_group_idx

    # Cross-stage P (2Q + non-WS only): write this WG's P into the OTHER
    # stage's free S columns instead of self-aliased s_tmem, so exp2 no
    # longer overwrites the S it just read -- letting S release right after
    # LDTM, before exp2. Default OFF => p_tmem == s_tmem, byte-identical.
    comptime CrossP = crossp_effective and MiscMBarsType.CrossP_enabled
    var p_tmem: UInt32
    comptime if CrossP:
        p_tmem = tmem_addr + (
            UInt32(config.TMEM_P0) if warp_group_idx
            == UInt32(0) else UInt32(config.TMEM_P1)
        )
    else:
        p_tmem = s_tmem
    var s_tile = UMMA0Type.CType(s_tmem)
    var p_tile = UMMA1Type.AType(p_tmem)

    var pipeline_s = mbars.consumer_s(warp_group_idx)
    var pipeline_c = mbars.producer_c(warp_group_idx)
    # Cross-stage P inplace handshake (2Q + non-WS only): `inplace_cons` waits
    # the peer's "S{1-wg} window free" signal before storing P{wg} there;
    # `inplace_prod` commits it after each LDTM (fire-and-forget, no acquire).
    # Depth-4 so the producer's bounded lead over the consumer never laps a
    # single-stage mbar.
    var inplace_cons = mbars.inplace_consumer(UInt32(1) - warp_group_idx)
    var inplace_prod = mbars.inplace_producer(warp_group_idx)
    var order_phase: UInt32 = 1 - warp_group_idx

    var order_s_wait: Optional[MBarType] = None
    var order_s_arrive: Optional[MBarType] = None
    comptime if EnableForcedOrdering:
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)

    # When fuse_gqa, head_idx is a kv_head_idx
    # the output will match, so `head_idx` is what we use for writing
    # sink and mask want q_head_idx
    var head_idx: UInt32 = seq_info.head_idx
    var q_head_idx: UInt32 = head_idx
    comptime if config.fuse_gqa:
        q_head_idx = UInt32(config.group) * head_idx + row % UInt32(
            config.group
        )

    var scale_log2e: Scalar[accum_dtype] = scale
    var correction_smem = smem.correction_smem() + tid

    comptime if not MaskType.apply_log2e_after_mask:
        scale_log2e *= log2e

    # Fuse scale*log2e multiplication and row_max subtraction into a
    # single FMA in store_exp. Only valid on the default scaling path
    # where apply_log2e_after_mask is off.
    # Disabled when sink weights are used because the sink logit lives
    # in a different domain (scaled by log2e only, not scale*log2e).
    # To disable for NaN debugging, set use_fma = False.
    comptime use_fma = (
        not MaskType.apply_log2e_after_mask
    ) and SinkType.is_null and QScaleType.is_null

    # Fixed P scale for FP8-QKV only. The un-normalized softmax
    # probabilities P = exp2(score - row_max) sit in the e4m3 subnormal floor;
    # lifting them before the fp8 cast that feeds the P@V GEMM reduces PV-GEMM
    # quantization error. Because the softmax uses exp2, the scale is exactly
    # an additive +bias in the exp2 argument (added raw, NOT multiplied by
    # scale_log2e). The final output is normalized by 1/row_sum, and row_sum
    # is accumulated from the SAME scaled P (plus the sink mass, also biased),
    # so the scale factor appears in both numerator and denominator and
    # cancels exactly -- no explicit descale.
    #
    # `p_fp8_bias` and the online-softmax lazy-rescale gate `rescale_threshold`
    # are the same knob (both live in the exp2/log2 domain), linked as
    # `p_fp8_bias = 8 + rescale_threshold`:
    #   fp8 : rescale_threshold = -2,  p_fp8_bias = 6   (a 64x P lift out of
    #         the e4m3 subnormal floor)
    #   bf16: rescale_threshold = -8,  p_fp8_bias = 0   (no bias)
    # The fp8 threshold of -2 (bias 6) was chosen as the perf sweet spot: a
    # prefill sweep of the threshold magnitude gained ~6% and saturated at
    # T=2, and it is accuracy-neutral vs the prior x256 default (bit-identical
    # tail-stress regression test + byte-identical Gemma-4-31B fp8-KV 16k
    # e2e). The bias is applied ONLY inside `comptime if p_fp8_bias != 0:`
    # branches below, so the bf16 codegen is byte-identical (a `+ 0.0` would
    # otherwise survive as a real fadd). Overflow-safe: the lazy-rescale
    # gate (threshold -2) lets a non-rescaled tile's max lag the true max
    # by up to 2 (log2), so max P = exp2(2 + 6) = 256 < 448 (e4m3 max).
    comptime rescale_threshold: Scalar[accum_dtype] = Scalar[accum_dtype](
        -8
    ) if size_of[qkv_type]() >= 2 else Scalar[accum_dtype](-2)
    comptime p_fp8_bias: Scalar[accum_dtype] = 8 + rescale_threshold

    # BLASST (arXiv 2512.12087): a block is a skip candidate iff all 32 lanes
    # agree its tile-local max sits `blasst_log_threshold` below the true
    # running max `m_true`. `and not config.use_ws`: main's `_pv_ws` P@V path
    # has no BLASST analog, so stripping would drop P while the MMA still ran
    # P@V (wrong O).
    comptime ENABLE_BLASST = get_defined_bool[
        "ENABLE_BLASST", False
    ]() and not config.use_ws
    # Negative log2-domain threshold; the compiler rejects negative int
    # defines, so it's passed as a positive milli-unit magnitude via
    # -D BLASST_LOG_THRESHOLD_MAG and negated here.
    comptime blasst_log_threshold: Scalar[accum_dtype] = -Float32(
        get_defined_int["BLASST_LOG_THRESHOLD_MAG", 1_000_000_000]()
    ) * Float32(0.001)

    @__parameter
    @always_inline
    def mask_row[
        BN: Int, //, mask_strategy: MaskStrategy
    ](mut s: Array[Scalar[accum_dtype], BN], kv_row: UInt32):
        apply_mask[
            mask_strategy=mask_strategy,
            skip_scale=use_fma,
        ](
            s,
            mask,
            scale_log2e,
            prompt_idx=seq_info.prompt_idx,
            q_head_idx=q_head_idx,
            kv_tile_start_row=Int32(kv_row),
            max_seq_len=max_seq_len,
            num_keys=Int32(num_keys),
            score_row=Int32(
                score_row
                + cta_q_offset
                + (
                    thread_tile_row
                    // UInt32(group) if fuse_gqa else thread_tile_row
                )
            ),
        )

    # while waiting, offset output
    #
    # Q-tile geometry:
    # - `per_qo_BM` is the row count of one output tile. 2Q emits two
    #   BM/2-row outputs (one per WG); 1Q emits one full-BM output
    #   combined across both WGs. Both modes have per_qo_BM == 128.
    # - `wg_row_offset` is the gap between WG0's and WG1's row ranges
    #   in BM-direct units (used for q_scale indexing). 2Q: BM/2. 1Q: 0
    #   (both WGs share the same Q rows).
    # - `wg_row_offset_seq` is the same gap in seq-space units
    #   (fuse_gqa-aware, used for num_output_rows / gmem_row).
    comptime per_qo_BM = BM // config.num_q
    comptime per_qo_BM_seq = per_qo_BM // group if fuse_gqa else per_qo_BM
    comptime wg_row_offset: Int = (BM // 2) if config.num_q == 2 else 0
    comptime wg_row_offset_seq: Int = (
        wg_row_offset // group if fuse_gqa else wg_row_offset
    )
    var num_output_rows = min(
        Int32(seq_info.seq_len)
        - Int32(seq_info.prompt_offset)
        - Int32(cta_q_offset)
        - Int32(warp_group_idx) * Int32(wg_row_offset_seq),
        Int32(per_qo_BM_seq),
    )

    var gmem_row = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )

    # ---- Workspace (traditional/unfused) split-K egress bookkeeping ----
    # `ws_part_idx` is this CTA's partition index p (= block_idx.x % P). The O
    # store (already producing locally-normalized O_i/l_i) is redirected into
    # this partition's slice of `o_partial` by shifting the ragged row by
    # `ws_o_row_off = p * ws_num_rows_q` (the descriptor, built over `o_partial`
    # with a P-extended row axis, then lands at [p, token, q_head, :]). Both fold
    # to 0 for every non-workspace config, so the O store is byte-identical.
    var ws_part_idx: UInt32 = 0
    var ws_o_row_off: UInt32 = 0
    comptime if workspace_split:
        ws_part_idx = splitk_partition_idx(ws_num_partitions)
        ws_o_row_off = ws_part_idx * ws_num_rows_q

    @__parameter
    @always_inline
    def _ws_write_lse(mx: Scalar[accum_dtype], sm: Scalar[accum_dtype]):
        # Store this thread's fused per-row LSE (log2 domain) into
        # `ws_lse_ptr[p, token_row, q_head]` (layout [P, num_rows_q,
        # num_q_heads]) for the separate combine kernel. WG0 only (LSE is
        # per-row, not per-depth-column); one thread per owned (seq, q-head).
        # Base-2, matching the FA4 softmax: under `use_fma` the running max is
        # carried in RAW score units, so scale it by `scale_log2e` here; the sum
        # is already Sum exp2(...). An empty partition window (T < P) is handled
        # by the caller below, which reaches this same helper with
        # `(-inf, 0) -> lse = -inf`; between that exit and the non-empty ones,
        # every row this CTA owns gets an LSE. `dispatch.mojo`'s no-init contract
        # for `lse_partial` rests on that being exhaustive.
        comptime if workspace_split:
            if warp_group_idx == UInt32(0):
                # The query row is `thread_tile_row` (`tid % BM`). For the WS
                # BM=32 datapath the `m_pack` warps share the same query rows, so
                # `row = tid % 128` folds in the key-partition quarter and is NOT
                # the query row here; `thread_tile_row` is (matching the mask's
                # `score_row` at the `mask_row` closure). `thread_tile_row == row`
                # for the non-WS BM=128 config, so this is identical there.
                var _seq_off: UInt32 = (
                    thread_tile_row // UInt32(group)
                ) if fuse_gqa else thread_tile_row
                if Int32(_seq_off) < num_output_rows:
                    var _qh: UInt32 = (
                        UInt32(group) * head_idx
                        + thread_tile_row % UInt32(group)
                    ) if fuse_gqa else head_idx
                    var _token: UInt32 = gmem_row + cta_q_offset + _seq_off
                    var _lse: Scalar[accum_dtype]
                    comptime if use_fma:
                        _lse = log2(sm) + mx * scale_log2e
                    else:
                        _lse = log2(sm) + mx
                    var _off: UInt32 = (
                        ws_part_idx * ws_num_rows_q * UInt32(config.num_q_heads)
                        + _token * UInt32(config.num_q_heads)
                        + _qh
                    )
                    # `OptionalPointer.value()` is immutable — its other users
                    # are all read-only inputs. This is the one egress, and the
                    # enclosing `comptime if workspace_split` already proves the
                    # pointer is the non-null conformer. `rebind` reconciles
                    # `WSLSEType.dtype` with `accum_dtype`, which the assert at
                    # the top of the function pins to the same f32.
                    var _ws_lse = rebind[
                        UnsafePointer[Scalar[accum_dtype], MutAnyOrigin]
                    ](ws_lse_ptr.value().unsafe_mut_cast[True]())
                    _ws_lse[Int(_off)] = _lse

    # Per-row score scratch: one warp's own quarter is `score_cols` wide
    # (== config.BN for the non-WS m_pack==1 layout).
    var s = Array[Scalar[accum_dtype], score_cols](uninitialized=True)

    # Per-token k_scale buffer offset. The load warp cycles k_scale through
    # num_k_scale_bufs staged buffers (each BN elements wide). The softmax
    # must advance this offset after each K tile to read the correct buffer.
    # 1Q: each softmax WG consumes every other K tile (WG0 even, WG1 odd),
    # so WG1 starts one buffer in and both advance by TWO buffers per
    # processed tile (see the stride-2 advance in load_mask_max_impl).
    var k_scale_off: UInt32 = 0
    comptime if config.num_q == 1 and not KScaleType.is_null:
        k_scale_off = warp_group_idx * UInt32(config.BN)
    comptime k_scale_wrap = config.num_k_scale_bufs() * config.BN
    comptime assert KScaleType.is_null == (k_scale_wrap == 0), String(
        "KScaleType.is_null = ",
        KScaleType.is_null,
        "\nconfig.num_k_scale_bufs() = ",
        config.num_k_scale_bufs(),
        "\nBN = ",
        config.BN,
    )

    comptime max_unroll = 8

    @__parameter
    @always_inline
    def apply_k_scale[
        N: Int, //, offset: Int
    ](mut s0: Array[Float32, N], k_scale_off: UInt32):
        comptime if not KScaleType.is_null:
            comptime for n in range(0, N, 2):
                var k_sc: f32x2 = (
                    k_scale.value()
                    .load[width=2](k_scale_off + UInt32(n + offset))
                    .cast[accum_dtype]()
                )
                var sn = mul_ftz(k_sc, f32x2(s0[n], s0[n + 1]))
                s0[n] = sn[0]
                s0[n + 1] = sn[1]

    @__parameter
    @always_inline
    def load_mask_max_impl[
        *, mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        comptime if EnableForcedOrdering:
            order_s_wait.unsafe_value()[].wait(order_phase)
        # break up into sets of 32
        # minimize wait time by using smallest first
        comptime BM = score_rows
        comptime batch_size = 32
        comptime has_remainder = (score_cols % batch_size) != 0
        comptime first_cols = (
            score_cols % batch_size
        ) if has_remainder else batch_size
        var s0 = TMemTile[accum_dtype, BM, first_cols](s_tmem).load_async()
        apply_k_scale[0](s0, k_scale_off)
        var s1 = TMemTile[accum_dtype, BM, batch_size](
            s_tmem + UInt32(first_cols)
        ).load_async()
        mask_row[mask_strategy=mask_strategy](s0, kv_row)
        var vrow_max = maximum[width=max_unroll](s0)

        comptime for _i in range(first_cols):
            s[_i] = s0[_i]
        comptime cols = score_cols - first_cols + batch_size

        comptime for i in range(cols // (2 * batch_size)):
            comptime offset0 = first_cols + batch_size * (2 * i)
            comptime offset1 = first_cols + batch_size * (2 * i + 1)
            comptime offset2 = first_cols + batch_size * (2 * i + 2)

            comptime if offset1 >= score_cols:
                apply_k_scale[offset0](s1, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]
            else:
                var s2 = TMemTile[accum_dtype, BM, batch_size](
                    s_tmem + UInt32(offset1)
                ).load_async()
                apply_k_scale[offset0](s1, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]

                comptime if offset2 < score_cols:
                    s1 = TMemTile[accum_dtype, BM, batch_size](
                        s_tmem + UInt32(offset2)
                    ).load_async()
                apply_k_scale[offset1](s2, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s2, kv_row + UInt32(offset1)
                )
                vrow_max = maximum(s2, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset1 + _i] = s2[_i]

        comptime if not KScaleType.is_null:
            comptime if config.num_q == 1:
                # Stride 2 buffers per processed tile (each WG sees every
                # other K tile). `num_k_scale_bufs` may be odd, so use a
                # modular wrap rather than the equality trick below.
                k_scale_off += UInt32(2 * config.BN)
                if k_scale_off >= UInt32(k_scale_wrap):
                    k_scale_off -= UInt32(k_scale_wrap)
            else:
                k_scale_off = (k_scale_off + UInt32(config.BN)) if (
                    k_scale_off != UInt32(k_scale_wrap - config.BN)
                ) else 0
        comptime if CrossP:
            # Cross-stage P early release: S{wg} is now fully read into `s`,
            # so signal both the MMA (may overwrite S{wg}) and the peer WG
            # (S{wg}'s columns are free for P{1-wg}). `load_wait`+`fence_before`
            # order this ahead of the mbar arrives so the MMA can't race the read.
            tcgen05_load_wait()
            tcgen05_fence_before()
            var sfree_p = mbars.sfree_producer(warp_group_idx)
            sfree_p.commit()
            inplace_prod.commit()
        return vrow_max

    @__parameter
    @always_inline
    def init_load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> Float32:
        return maximum(load_mask_max_impl[mask_strategy=mask_strategy](kv_row))

    @__parameter
    @always_inline
    def load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32, old_max: Float32) -> Float32:
        pipeline_s.wait()
        tcgen05_fence_after()
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](kv_row), old_max
        )

    # Same wait+fence+load as `load_mask_max`, but returns the pure tile-local
    # max (no `old_max` fusion) so one reduction feeds both the running max
    # and the BLASST vote.
    @__parameter
    @always_inline
    def load_mask_tile_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        pipeline_s.wait()
        tcgen05_fence_after()
        return load_mask_max_impl[mask_strategy=mask_strategy](kv_row)

    @__parameter
    @always_inline
    def store_exp(max_term: Float32) -> f32x2:
        comptime exp_simd = 2
        comptime vs_len = score_cols // exp_simd  # score_cols // 2
        comptime assert (vs_len % config.num_pv_stages) == 0
        comptime use_3_then_1_split = UMMA1Type.use_3_then_1_split
        comptime batch_size = 32 if config.num_pv_stages == 1 else vs_len // (
            4 if use_3_then_1_split else config.num_pv_stages
        )
        comptime num_batch_iters, remainder = divmod(vs_len, batch_size)
        comptime assert num_batch_iters > 0
        comptime BatchTileType = TMemTile[
            qkv_type, score_rows, batch_size * exp_simd
        ]
        comptime RemainderTileType = TMemTile[
            qkv_type, score_rows, remainder * exp_simd
        ]
        comptime assert (score_cols % exp_simd) == 0

        @__parameter
        @always_inline
        def s_load[i: Int]() -> f32x2:
            return f32x2(s[2 * i], s[2 * i + 1])

        @__parameter
        @always_inline
        def s_store[i: Int](v: f32x2):
            s[2 * i] = v[0]
            s[2 * i + 1] = v[1]

        var vrow_max: f32x2
        var vscale: f32x2
        var vneg_max_scaled: f32x2

        comptime if use_fma:
            vscale = f32x2(scale_log2e)
            # `max_term` arrives fully formed as the negated log2-domain max
            # `-row_max*scale_log2e` (+ `p_fp8_bias` already folded in for fp8,
            # via a single fused fma at max-definition time — see
            # `neg_scaled_max`). Computed once by the caller and carried across
            # the K-loop, so store_exp neither recomputes `-row_max*scale` (one
            # fewer scalar FMUL per K-iter) nor re-adds the bias here.
            vneg_max_scaled = f32x2(max_term)
            vrow_max = f32x2(0)  # unused
        else:
            # non-fma: `max_term` is the raw running max.
            comptime if p_fp8_bias != 0:
                vrow_max = f32x2(max_term - p_fp8_bias)
            else:
                vrow_max = f32x2(max_term)
            vscale = f32x2(0)  # unused
            vneg_max_scaled = f32x2(0)  # unused

        @__parameter
        @always_inline
        def score_to_logit(score: f32x2) -> f32x2:
            comptime if use_fma:
                return fma_ftz(score, vscale, vneg_max_scaled)
            else:
                return sub_ftz(score, vrow_max)

        # --- Experiment parameters ---
        # Schedule the score-to-logit conversion `ratio` iterations ahead
        # of its corresponding exp2 to hide latency.  1 = strict
        # interleave, 4 = ~4-iteration prefetch (current tuned value).
        comptime score_to_logit_ratio: Int = 4
        # Number of exp2s per pass to route through the polynomial
        # emulation path (`exp2_emulation`) rather than hardware
        # `ex2.approx`.  Default 16 on sm_100; disabled on sm_103 where
        # the emulation does not pay off.
        comptime default_emulate_count: Int = 0 if "sm_103" in _accelerator_arch() else 16
        # `default_emulate_count` is calibrated at vs_len=64; the
        # `// 64` normalizes it back to that reference so non-default
        # vs_len scales the count proportionally.  Override at compile
        # time with `-D EXP2_EMULATE_COUNT=N`.
        comptime num_emulated: Int = (
            get_defined_int["EXP2_EMULATE_COUNT", default_emulate_count]()
            * vs_len
        ) // 64  # target emulated exp2s out of vs_len
        comptime emulation_start: Int = batch_size  # emulation window start
        # comptime emulation_start: Int = vs_len // score_to_logit_ratio  # emulation window start
        comptime emulation_end: Int = 0 if num_emulated == 0 else vs_len  # emulation window end
        comptime order_arrive_offset: Int = batch_size - 1  # within last batch
        # Derived: stride to distribute ~num_emulated across [emul_start, emul_end)
        comptime emulation_window: Int = 0 if num_emulated == 0 else emulation_end - emulation_start
        comptime emulation_stride_freq = 1 if num_emulated == 0 else emulation_window // num_emulated
        # num_emulated = emulation_window_freq / emulation_stride_freq
        # +  (emulation_window - emulation_window_freq) / (emulation_stride_freq + 1)
        #
        # num_emulated * emulation_stride_freq * (emulation_stride_freq + 1)
        #   = (emulation_stride_freq + 1)*emulation_window_freq
        #    + emulation_stride_freq * (emulation_window - emulation_window_freq)
        #   = (emulation_stride_freq + 1)*emulation_window_freq
        #    + emulation_stride_freq * emulation_window
        #    - emulation_stride_freq * emulation_window_freq
        #   = emulation_window_freq + emulation_stride_freq * emulation_window
        #
        # Thus:
        comptime emulation_window_freq = num_emulated * emulation_stride_freq * (
            emulation_stride_freq + 1
        ) - emulation_stride_freq * emulation_window
        comptime emulation_window_unfreq_start = emulation_start + emulation_window_freq
        comptime assert vs_len % score_to_logit_ratio == 0
        comptime assert (
            num_emulated >= 0
            and num_emulated <= emulation_window
            and emulation_window >= 0
        )
        comptime assert (
            num_emulated
            == emulation_window_freq // emulation_stride_freq
            + (emulation_window - emulation_window_freq)
            // (emulation_stride_freq + 1)
        )

        @__parameter
        @always_inline
        def exp_iter[idx: Int]():
            comptime if idx < vs_len // score_to_logit_ratio:
                comptime for i in range(score_to_logit_ratio):
                    comptime j = score_to_logit_ratio * idx + i
                    s_store[j](score_to_logit(s_load[j]()))

            var x = s_load[idx]()
            comptime if (
                (
                    idx >= emulation_start
                    and (idx < emulation_window_unfreq_start)
                    and ((idx - emulation_start) % emulation_stride_freq == 0)
                )
                or (
                    idx >= emulation_window_unfreq_start
                    and (idx < emulation_end)
                    and (
                        (idx - emulation_start) % (emulation_stride_freq + 1)
                        == 0
                    )
                )
            ):
                x = exp2_emulation(x)
            else:
                x = exp2(x)
            s_store[idx](x)

        # Cross-stage P store: per-tile asymmetric static schedule (`ahead`
        # batches of exp2 buffered before the inplace wait, tuned per-tile
        # since each WG's consume signal arrives at a different phase) with
        # the row-sum interleaved per-batch. Same 4-acc every-4th reduction
        # tree/pairings as the aliased path below => bit-exact, no reassociation.
        comptime if CrossP:
            comptime assert not EnableForcedOrdering, (
                "FA4_TMEM_CROSS_P per-tile store is untested with"
                " FA4ForcedSoftmaxOrdering"
            )
            comptime assert not (
                ExplicitTMEMCrossP
                and not (remainder == 0 and num_batch_iters == 4)
            ), (
                "FA4_TMEM_CROSS_P=true was requested but the per-tile store"
                " assumes 4 full batches (headline shape)"
            )

            # single inplace consume (instant at this static position)
            inplace_cons.wait()
            inplace_cons.step()
            # init 0 only to satisfy definite-init across the comptime-if
            # branches; batch 0 ASSIGNS (overwrites) below, so the 4-acc
            # every-4th tree/pairings are unchanged => bit-exact.
            var r0 = f32x2(0)
            var r1 = f32x2(0)
            var r2 = f32x2(0)
            var r3 = f32x2(0)
            comptime for b in range(num_batch_iters):
                # 3-then-1: release consumer_s[0] (P[0:96)) at the b==3
                # boundary, after batches 0,1,2 are stored.
                comptime if 4 * b == 3 * num_batch_iters:
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                    comptime if config.pair_cta:
                        umma_arrive_leader_cta(pipeline_s.consumer_mbar[0]())
                    else:
                        pipeline_s.release_no_step[0]()
                comptime for idx in range(b * batch_size, (b + 1) * batch_size):
                    exp_iter[idx]()
                comptime if b == 0:
                    BatchTileType(p_tmem).store_async(s)
                else:
                    comptime el = batch_size * b * exp_simd
                    comptime to = (el * size_of[qkv_type]()) // size_of[
                        accum_dtype
                    ]()
                    BatchTileType(p_tmem + UInt32(to)).store_async[
                        src_offset=el
                    ](s)
                # interleave the row-sum for batch b (same 4-acc every-4th
                # tree as the aliased path => bit-exact)
                comptime if b == 0:
                    r0 = s_load[0]()
                    r1 = s_load[1]()
                    r2 = s_load[2]()
                    r3 = s_load[3]()
                    comptime for k in range(4, batch_size, 4):
                        r0 = add_ftz(r0, s_load[k]())
                        r1 = add_ftz(r1, s_load[k + 1]())
                        r2 = add_ftz(r2, s_load[k + 2]())
                        r3 = add_ftz(r3, s_load[k + 3]())
                else:
                    comptime for k in range(
                        b * batch_size, (b + 1) * batch_size, 4
                    ):
                        r0 = add_ftz(r0, s_load[k]())
                        r1 = add_ftz(r1, s_load[k + 1]())
                        r2 = add_ftz(r2, s_load[k + 2]())
                        r3 = add_ftz(r3, s_load[k + 3]())
            tcgen05_store_wait()
            tcgen05_fence_before()
            comptime if config.pair_cta:
                umma_arrive_leader_cta(
                    pipeline_s.consumer_mbar[config.num_pv_stages - 1]()
                )
                pipeline_s.step()
            else:
                pipeline_s.release[config.num_pv_stages - 1]()
            pipeline_c.acquire()
            return add_ftz(add_ftz(r0, r1), add_ftz(r2, r3))

        # --- Batch 0 ---
        comptime for idx in range(batch_size):
            exp_iter[idx]()

        var acc = s_load[0]()
        comptime if EnableEarlyAdd:
            comptime for i in range(1, batch_size // 2):
                acc = add_ftz(acc, s_load[i]())

        # Shared-key sinks P to `p_smem` instead, in ONE whole-quadrant store
        # at the tail (see the `store_p_quadrant` call below). The per-batch
        # TMEM stores are the off-mode path and are unchanged.
        comptime if not config.ws_shared_key:
            BatchTileType(p_tmem).store_async(s)

        comptime for b in range(1, num_batch_iters):
            comptime offset = batch_size * b

            # Shared-key releases NOTHING mid-tile. Its P sub-stage split is
            # per-warp along each warp's OWN column range, so `consumer_s[k]`
            # would mean "the first k+1 sub-stages of EVERY warp's columns" --
            # a striped set, not any contiguous key chunk, and no key-chunk
            # P@V MMA may legally start after it. The releases move to the
            # tail; they cannot simply be dropped, or stages
            # `0 ..< num_pv_stages-1` would never be released and the MMA's
            # `consumer_wait` would hang.
            #
            # Guarded around BOTH arms, not on each one: the local
            # `use_3_then_1_split` reads this file's inert `UMMA1Type`, so it is
            # True for shared-key too, and guarding only the `elif` would leave
            # its stage-0 release live while the tail released stage 0 a SECOND
            # time -- an extra `consumer_mbar[0]` arrival that compiles clean
            # and corrupts at runtime. The MMA that actually runs under the mode
            # (`mma_warp`'s `UMMA1SKType`) has the flag False; editing this
            # alias to match would move the S-release cadence of every shipping
            # non-shared-key config.
            comptime if not config.ws_shared_key:
                comptime if use_3_then_1_split:
                    comptime if 4 * b == 3 * num_batch_iters:
                        tcgen05_store_wait()
                        tcgen05_fence_before()
                        comptime if config.pair_cta:
                            umma_arrive_leader_cta(
                                pipeline_s.consumer_mbar[0]()
                            )
                        else:
                            pipeline_s.release_no_step[0]()
                elif config.num_pv_stages > 1:
                    comptime assert config.num_pv_stages == num_batch_iters
                    tcgen05_store_wait()
                    tcgen05_fence_before()

                    comptime assert config.num_pv_stages == num_batch_iters
                    comptime if config.pair_cta:
                        umma_arrive_leader_cta(
                            pipeline_s.consumer_mbar[b - 1]()
                        )
                    else:
                        pipeline_s.release_no_step[b - 1]()

            comptime for idx in range(offset, offset + batch_size):
                exp_iter[idx]()
                comptime if (
                    EnableForcedOrdering
                    and b == max(1, num_batch_iters - 1)
                    and idx == offset + order_arrive_offset
                ):
                    _ = order_s_arrive.unsafe_value()[].arrive()
                    order_phase ^= 1

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_dtype
            ]()
            comptime if not config.ws_shared_key:
                BatchTileType(p_tmem + UInt32(tmem_offset)).store_async[
                    src_offset=el_offset
                ](s)

        comptime if remainder > 0:
            comptime offset = batch_size * num_batch_iters

            comptime for idx in range(offset, offset + remainder):
                exp_iter[idx]()

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_dtype
            ]()
            comptime if not config.ws_shared_key:
                RemainderTileType(p_tmem + UInt32(tmem_offset)).store_async[
                    src_offset=el_offset
                ](s)

        comptime if config.ws_shared_key:
            # `tcgen05_load_wait`, NOT `tcgen05_store_wait`. Off-mode the WAR
            # against the MMA's next Q@K' is covered only TRANSITIVELY: P's
            # `tcgen05.st` targets the same TMEM addresses the S `load_async`
            # read, so store-completion implies the read happened. With P in
            # SMEM there is no such store, and `tcgen05_store_wait()` here
            # would be a no-op that still LOOKS like a guarantee -- while
            # `load_async` emits a bare `tcgen05.ld` that never waits and the
            # release below lets the MMA overwrite S.
            tcgen05_load_wait()
            store_p_quadrant[BN=config.BN, p_tile_rows=config.BM](
                smem.p_smem() + warp_group_idx * UInt32(config.BM * config.BN),
                s,
                thread_tile_row,
                partition_g,
            )
            # P left through the generic proxy; the SS P@V MMA reads it through
            # the async proxy. `tcgen05_fence_before()` does not cover that.
            fence_async_view_proxy()
        else:
            tcgen05_store_wait()
        tcgen05_fence_before()
        comptime if config.pair_cta:
            comptime if config.ws_shared_key:
                comptime for pv_stage in range(config.num_pv_stages - 1):
                    umma_arrive_leader_cta(pipeline_s.consumer_mbar[pv_stage]())
            umma_arrive_leader_cta(
                pipeline_s.consumer_mbar[config.num_pv_stages - 1]()
            )
            pipeline_s.step()
        else:
            comptime if config.ws_shared_key:
                comptime for pv_stage in range(config.num_pv_stages - 1):
                    pipeline_s.release_no_step[pv_stage]()
            pipeline_s.release[config.num_pv_stages - 1]()

        pipeline_c.acquire()
        # now we can sum the remaining elements of `acc`
        comptime add_offset = batch_size // 2 if EnableEarlyAdd else 0
        var acc0: f32x2
        var acc1: f32x2
        var acc2: f32x2
        var acc3: f32x2

        comptime if EnableEarlyAdd:
            acc0 = acc
            acc1 = s_load[batch_size // 2]()
            acc2 = s_load[batch_size // 2 + 1]()
            acc3 = add_ftz(
                s_load[batch_size // 2 + 2](),
                s_load[batch_size // 2 + 3](),
            )
        else:
            acc0 = acc
            acc1 = s_load[1]()
            acc2 = s_load[2]()
            acc3 = s_load[3]()

        comptime for i in range(add_offset + 4, vs_len, 4):
            acc0 = add_ftz(acc0, s_load[i]())
            acc1 = add_ftz(acc1, s_load[i + 1]())
            acc2 = add_ftz(acc2, s_load[i + 2]())
            acc3 = add_ftz(acc3, s_load[i + 3]())
        return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

    # Stripped sibling of store_exp for a WG-unanimous skip: no exp2/P-store/
    # row-sum, but reproduces every barrier arrival so the MMA and correction
    # warps stay in lockstep.
    @__parameter
    @always_inline
    def store_exp_skip():
        # Match store_exp's tail ordering (no P store to fence, but keep the
        # same tcgen05 wait/fence discipline before releasing S-consumer).
        tcgen05_store_wait()
        tcgen05_fence_before()
        # Cross-stage P: the peer's inplace producer commits every block
        # regardless of skip, so this consumer phase must advance here too
        # (BLASST skip) or it drifts out of lockstep with the peer.
        comptime if CrossP:
            inplace_cons.wait()
            inplace_cons.step()
        comptime if config.pair_cta:
            comptime for pv_stage in range(config.num_pv_stages):
                umma_arrive_leader_cta(pipeline_s.consumer_mbar[pv_stage]())
            pipeline_s.step()
        else:
            comptime for pv_stage in range(config.num_pv_stages - 1):
                pipeline_s.release_no_step[pv_stage]()
            pipeline_s.release[config.num_pv_stages - 1]()
        comptime if EnableForcedOrdering:
            _ = order_s_arrive.unsafe_value()[].arrive()
            order_phase ^= 1
        pipeline_c.acquire()

    var kv_row: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_info.prompt_idx, score_row
    )
    # 1Q: WG0 takes even-indexed K/V tiles (start = kv_row); WG1 takes
    # odd-indexed (+BN). Both advance by 2*BN per main-loop iter (set
    # below). 2Q: both WGs share the same kv_row stride of BN.
    # single-O (1Q wide-V): WG0 owns EVERY tile (WG1 no-op), so no
    # per-WG start offset and a stride of BN (set below).
    comptime if config.num_q == 1 and not config.single_o:
        kv_row += warp_group_idx * UInt32(config.BN)
        # Packed WS: within a 256-key tile, this warp's partition `g` owns the
        # logical key band [g*score_cols, g*score_cols+score_cols) -- a
        # quarter for Layout-G (m_pack==4), a half for Layout-E (m_pack==2).
        # Fold that per-partition base into `kv_row` so it flows through every
        # masking call below (`mask_row` itself stays per-warp-invariant).
        # Inert non-WS (m_pack==1 -> no inner block elaborated).
        comptime if config.use_ws:
            kv_row += partition_g * UInt32(score_cols)
    comptime mask_sets = MaskType.nonfull_sets[BM_mask, BN]()
    comptime mask_strategies = MaskType.mask_strategies[BM_mask, BN]()
    comptime num_sets = len(mask_strategies)
    comptime assert len(mask_sets) == num_sets

    var row_max: Float32
    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    # `total_iters_combined` is the combined K-tile count across both
    # WGs in 1Q (= MMA's `mask.total_iters` view). Needed for the peer
    # `o_prod_mbar` wait phase in the 1Q LSE combine below.
    var total_iters_combined: UInt32 = 0

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        var mask_ends = mask.masked_set_ends[
            BM=BM_mask, BN=BN, page_size=page_size
        ](seq_info.prompt_idx, score_row, num_keys)

        # Split-K (1Q): restrict this CTA to a balanced sub-window
        # [part_cb, part_ce) of the combined tile range [0, T) (T =
        # masked_set_ends[-1]) and offset kv_row to its first tile. Same window
        # as load/mma/correction -- masked_set_ends[-1] == total_iters for
        # check_mask==False masks.
        var part_cb: UInt32 = 0
        var part_ce: UInt32 = mask_ends[num_sets - 1]
        comptime if config.num_q == 1 and (
            config.splitk_partitions > 1 or workspace_split
        ):
            var _np = splitk_num_partitions[config](ws_num_partitions)
            var _w = splitk_window(
                mask_ends[num_sets - 1],
                _np,
                splitk_partition_idx(_np),
            )
            part_cb = _w[0]
            part_ce = _w[1]
            kv_row += part_cb * UInt32(config.BN)

        # Per-set tile counts within [part_cb, part_ce): clamp each set end
        # into the window (prev starts at part_cb). Folds to the full
        # per-set counts when split-K is off (part_cb=0, part_ce=T).
        var _prev: UInt32 = part_cb
        comptime for i in range(num_sets):
            var _e = max(part_cb, min(mask_ends[i], part_ce))
            mask_iters[i] = _e - _prev
            _prev = _e

        comptime if config.num_q == 1:
            total_iters_combined = part_ce - part_cb
            # single-O: WG0 owns EVERY tile (WG1 no-op), so keep the full
            # combined `mask_iters` (no per-WG split).
            comptime if not config.single_o:
                # Per-WG split with cumulative-parity carry. WG0 owns
                # combined indices with parity 0 (even cumulative position);
                # WG1 owns parity 1. Within set i starting at cumulative
                # combined index `cum`:
                #   parity=0: WG0 takes ceil(iters_combined_i/2), WG1 floor.
                #   parity=1: WG0 takes floor, WG1 ceil.
                # Parity is local to the partition (cumulative starts at 0),
                # consistent with the unconditional WG offset (+warp_group_idx
                # *BN) applied to kv_row above.
                var cumulative: UInt32 = 0
                comptime for i in range(num_sets):
                    var iters_combined_i = mask_iters[i]
                    var parity = cumulative & UInt32(1)
                    if warp_group_idx == UInt32(0):
                        mask_iters[i] = (
                            iters_combined_i + UInt32(1) - parity
                        ) // UInt32(2)
                    else:
                        mask_iters[i] = (iters_combined_i + parity) // UInt32(2)
                    cumulative += iters_combined_i
    else:
        comptime if config.num_q == 1:
            # Unmasked-only path has no precomputed mask_ends. Derive
            # the combined K-tile count from the [start_column, num_keys)
            # range, matching MMA's `mask.total_iters` view.
            total_iters_combined = mask.total_iters[BM_mask, BN, page_size](
                seq_info.prompt_idx, score_row, num_keys
            )

    comptime assert num_sets >= 1 and num_sets <= 3
    comptime assert num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK
    # M2 split-K slices the contiguous masked-set range only. UNKNOWN_MASK
    # (Materialized/And/Or) iterates via a runtime FULL_MASK skip whose
    # tile count is an upper bound -- partitioning it correctly needs the
    # per-partition skip (M5). Block the combo so a future caller cannot
    # silently get redundant full-range partials (which the combine would
    # then sum P times).
    comptime assert (
        config.splitk_partitions == 1
        or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK
    ), (
        "split-K (M2) supports only check_mask==False masks; UNKNOWN_MASK"
        " (Materialized/And/Or) masks are M5"
    )

    # Split-K M4 combine activation, hoisted above the T<=1 / empty guards (it
    # gates the empty-partition neutral-stage path below as well as the T==1 /
    # T>=2 combine paths in the 1Q output phase). Unconditional whenever split-K
    # is active (`splitk_partitions > 1`); folds away when split-K is off.
    comptime splitk_combine_active = config.splitk_partitions > 1

    # 1Q T<=1: WG1 owns the odd-indexed K-tiles. At total_iters_combined == 1
    # (only K_e[0]) AND == 0 (empty partition -- front-load leaves only TRAILING
    # partitions empty, never the rank-0 writer; also covers M6 idle CTAs) WG1
    # has zero work and MMA never commits s1, so pipeline_s.wait() below would
    # hang; peel_mask (num_sets==1) would also underflow mask_iters[0]. Skip
    # everything WG1 would do and drop to the final cross-WG sync that gates
    # TMEM dealloc (the dealloc, `warp_idx == 0`, runs at the kernel terminal).
    #
    # single-O (wide-V fallback): the two per-WG O partials do NOT fit in
    # the 512-col TMEM (`2*BN + 2*padded_ov > 512`), so single-O aliases
    # O1 onto O0 and cannot run the two-WG even/odd LSE-combine (the peer
    # read would overrun TMEM). Instead WG0 processes ALL K-tiles serially
    # into the single O0 accumulator and WG1 is a full no-op for every T
    # (not just T==1) — mirroring the T==1 fast path, generalized. The MMA
    # and correction warps take matching single-O single-WG paths.
    comptime if config.num_q == 1:
        # WS two-pipeline un-no-op: the packed WS config has single_o == False,
        # so this guard reduces to "WG1 returns only when total_iters_combined
        # <= 1" (i.e. WG1 is allotted floor(T/2) == 0 odd tiles). For T >= 2 WG1
        # runs its odd-tile stream normally -- no code change needed to activate
        # the second pipeline.
        if (
            config.single_o or total_iters_combined <= UInt32(1)
        ) and warp_group_idx == UInt32(1):
            named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            return

        # Empty partition (total_iters_combined == 0; WG0 only -- WG1 returned
        # just above). No scores/O were produced, so skip pipeline_s.wait +
        # peel + main loop + LSE combine. When the combine is active, stage a
        # NEUTRAL softmax identity (max = -inf, sum = 0, O = 0) and publish it
        # (so peers' weight-0 DSMEM read of this partition is a finite 0). Under
        # REDUCE-SCATTER this partition STILL owns a depth band, so it then runs
        # the same pass-2 helper as the non-empty partitions: it reads its band
        # from all peers (own weight 0; the non-empty peers carry the data) and
        # writes it into its own dead band slice. No round-2 cluster barrier (the
        # kernel's terminal `cluster_sync()` gates peer reads). No sink fold here
        # (partition-0 folds it once).
        comptime if config.splitk_partitions > 1:
            if total_iters_combined == UInt32(0):
                comptime if config.use_ws:
                    # WS empty partition: neutral third-level element. A zero band
                    # + `(M = -inf, L = 0)` makes this CTA contribute weight 0 to
                    # every peer's combine (never `0*inf`), yet it STILL writes
                    # its OWN depth band from the non-empty peers. The stage /
                    # maxsum regions MUST match the non-empty WS reduce-scatter
                    # (`ws_l2_stage` / `ws_maxsum1`) so the DSMEM offsets line up
                    # across partitions. No non-WS `fa4_splitk_reduce_scatter_write`
                    # here (it assumes the BM=128 row=thread TMEM grid).
                    comptime WS_STAGE = config.ws_epilogue_stage_f32()
                    comptime WS_ML = config.ws_epilogue_ml_f32()
                    comptime WS_L2STAGE = config.ov_depth * config.BM
                    comptime WS_L2MS = config.BM * 2
                    var ws_f32e = smem.o_smem[.float32]()
                    var ws_maxsum1e = ws_f32e + (2 * WS_STAGE + WS_ML)
                    var ws_l2_stagee = ws_f32e + (2 * WS_STAGE + 2 * WS_ML)
                    var ws_o_e = (
                        ws_l2_stagee + (WS_L2STAGE + WS_L2MS)
                    ).bitcast[Scalar[output_type]]()
                    var o_band_zero = Array[
                        Float32,
                        config.ov_depth // config.m_pack,
                    ](fill=Float32(0))
                    fa4_ws_splitk_reduce_scatter_write[
                        config,
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        P=config.splitk_partitions,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        UInt32(config.splitk_partitions),
                        splitk_partition_idx(UInt32(config.splitk_partitions)),
                        min_or_neg_inf[.float32](),
                        Float32(0),
                        scale_log2e,
                        o_band_zero,
                        ws_l2_stagee,
                        ws_maxsum1e,
                        ws_o_e,
                        smem.misc_mbars().publish_mbar(),
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset,
                    )
                elif splitk_combine_active:
                    var stage = smem.o_smem[.float32]()
                    var maxsum = stage + (config.BM * padded_ov_depth)
                    # Neutral (max,sum): -inf never wins the cluster Gmax, and
                    # sum 0 zeroes this partition's combine weight regardless.
                    maxsum[row * 2] = min_or_neg_inf[.float32]()
                    maxsum[row * 2 + 1] = Float32(0)
                    # Zero-fill all depth columns (WG0 covers the full range at
                    # 1Q) so peers' weight-0 DSMEM read of this partition is a
                    # finite 0.
                    var own_o_empty = TMemTile[
                        accum_dtype, config.BM, padded_ov_depth
                    ](tmem_addr + UInt32(config.TMEM_O0))
                    # Reduce-scatter: stage this (empty) partition's neutral
                    # O_cta (zero_fill), publish, then read its OWN band from all
                    # peers (own weight 0; non-empty peers carry the data) and
                    # write it into its own dead band slice. No round-2 cluster
                    # barrier (terminal `cluster_sync()` gates peer reads).
                    fa4_splitk_reduce_scatter_write[
                        config,
                        P=config.splitk_partitions,
                        use_fma=use_fma,
                        single_source=True,
                        zero_fill=True,
                    ](
                        row,
                        warp_idx & 3,
                        warp_group_idx,
                        splitk_partition_idx(UInt32(config.splitk_partitions)),
                        min_or_neg_inf[.float32](),
                        Float32(0),
                        scale_log2e,
                        Float32(0),
                        Float32(0),
                        stage,
                        maxsum,
                        o_smem,
                        smem.misc_mbars().publish_mbar(),
                        own_o_empty,
                        own_o_empty,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset,
                    )
                # Release WG1 (parked on this 2*WG barrier above) before the
                # kernel terminal cluster_sync + TMEM dealloc.
                named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
                return

        # Workspace (traditional/unfused) split-K empty partition. The workspace
        # path keeps `splitk_partitions == 1`, so the cluster block above does not
        # fire; an empty partition (its balanced KV window holds zero active tiles
        # -- reachable once windowed / chunked masks leave trailing partitions
        # empty) must be handled here, or WG0 hangs on `pipeline_s.wait()` below
        # (no scores were ever produced). Write `lse_p = -inf` (skip the O store;
        # the separate `fa4_splitk_combine` folds `exp2(lse_p - m*) = 0`), then
        # rendezvous with WG1 (parked on the 2*WG barrier at its early return for
        # `total_iters_combined <= 1`) exactly as the T==1 non-empty exit does.
        # `_ws_write_lse(-inf, 0)` -> `log2(0) + (-inf)*scale_log2e = -inf`.
        #
        # This is the ONLY egress that writes LSE without an O store, which is
        # exactly what `dispatch.mojo`'s no-init contract for `o_partial` is
        # stated against: leaving O unwritten is safe only while the paired LSE is
        # `-inf`, so the combine's `scale != 0` select sees weight 0. Do not add an
        # O-less exit at a finite LSE, and do not drop the `-inf` write here.
        comptime if workspace_split:
            if total_iters_combined == UInt32(0):
                _ws_write_lse(
                    min_or_neg_inf[accum_dtype](), Scalar[accum_dtype](0)
                )
                comptime if not single_softmax_wg:
                    named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
                # Free the TMEM this CTA allocated in the prologue. The non-empty
                # workspace exit deallocs at the end of its store path (which this
                # early return skips), and the MMA warp likewise returns for
                # `total_iters == 0` WITHOUT issuing any MMA (`mma_warp.mojo`'s
                # empty-partition guard), so no TMEM op is in flight -- warp 0
                # frees it here, mirroring the non-empty exit. Omitting it leaks
                # the CTA's TMEM (CUDA_ERROR_TENSOR_MEMORY_LEAK).
                if warp_idx == 0:
                    tcgen05_release_allocation_lock[Int32(cta_group)]()
                    tcgen05_dealloc[Int32(cta_group)](
                        tmem_addr, UInt32(config.sm100_tmem_cols)
                    )
                return
    # All-masked row (valid_length 0): no scores produced, so `pipeline_s.wait()`
    # below would hang. Write a deterministic zero output row (the O accumulator
    # was never produced, so `zero_fill` skips the TMEM read), then take WG0's
    # normal terminal exit -- rendezvous on the 2*WG barrier (unless there is no
    # second WG) and free the TMEM warp 0 owns on the single-CTA path; a bare
    # return would leak it and wedge the next launch.
    comptime if config.num_q == 1 and not (config.splitk_partitions > 1):
        if total_iters_combined == UInt32(0):
            fa4_scale_write_output[config, zero_fill=True](
                row,
                warp_idx & 3,
                UInt32(0),
                Float32(0),
                o_smem,
                TMemTile[accum_dtype, BM // config.num_q, padded_ov_depth](
                    tmem_addr + UInt32(config.TMEM_O0)
                ),
                ragged_tma_store,
                num_output_rows,
                head_idx,
                gmem_row + cta_q_offset,
            )
            comptime if not single_softmax_wg:
                named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            comptime if not config.pair_cta and config.splitk_partitions == 1:
                if warp_idx == 0:
                    tcgen05_release_allocation_lock[Int32(cta_group)]()
                    tcgen05_dealloc[Int32(cta_group)](
                        tmem_addr, UInt32(config.sm100_tmem_cols)
                    )
            return

    pipeline_s.wait()
    tcgen05_fence_after()
    # Apply per-token q_scale
    comptime if not QScaleType.is_null:
        scale_log2e *= q_scale.value()[
            warp_group_idx * UInt32(wg_row_offset) + row
        ].cast[accum_dtype]()

    var row_max: Float32 = peel_mask[
        rebind[StaticTuple[MaskStrategy, num_sets]](mask_strategies),
        init_load_mask_max,
    ](mask_iters, kv_row)
    var sink_weight: Scalar[accum_dtype]

    # Split-K: the sink mass must land in the *cluster-global* sum exactly
    # once per Q row, not once per partition CTA. Each partition's `sum_cta`
    # is rescaled into `global_sum_cluster` by `exp2((max_cta - Gmax)*c)` at
    # the terminal combine, so folding the sink into partition 0's `sum_cta`
    # (and clamping its `row_max` so `max_cta` already accounts for the sink)
    # makes the sink contribute `exp2((sink - Gmax)*c)` exactly once. Gate
    # both the clamp and the fold on partition 0; folds to current behavior
    # when split-K is off. (Mirrors the WG0-only intra-CTA sink rule and
    # `mla_decode_combine.mojo`'s once-per-row sink accounting.)
    var fold_sink: Bool = True
    comptime if config.splitk_partitions > 1:
        fold_sink = splitk_partition_idx(
            UInt32(config.splitk_partitions)
        ) == UInt32(0)

    # Workspace (unfused) split-K keeps `splitk_partitions == 1`, so the cluster
    # gate above is a no-op and `fold_sink` would stay True for ALL `P` runtime
    # partitions -- the separate `fa4_splitk_combine` would then fold the sink
    # mass `P` times. Gate on the RUNTIME partition index `ws_part_idx`
    # (= block_idx.x % P) so the sink mass lands in partition 0's `lse_partial`
    # exactly once; the combine rescales it into `exp2((sink - Gmax)*c)` via its
    # `exp2(lse_p - m*)` weighting. Mirrors the cluster partition-0 discipline
    # above and the WG0-only intra-CTA sink rule. Folds to `True` (no-op) for
    # every non-workspace config.
    comptime if workspace_split:
        fold_sink = ws_part_idx == UInt32(0)

    # WS key split: WG0's warps cover the SAME query rows but disjoint key
    # partitions (quarters for Layout-G, halves for Layout-E). The intra-CTA
    # mass add below is WG0-gated (warp_group_idx == 0) but NOT per-partition,
    # so every WG0 warp would each fold the sink -> an m_pack-x over-count in
    # the combined denominator. Gate on partition 0 (`partition_g == 0`) --
    # exactly the split-K partition-0 discipline above. The two-level LSE
    # combine then rescales WG0's partial (which carries `m >= sink` and
    # `l += exp2(sink - m)`) into `exp2(sink - Gmax)` exactly once.
    # `partition_g == 0` covers ALL `warps_per_partition` warps of partition 0
    # -- both row-halves at Layout-E, not just `warp_idx == 0`.
    #
    # Precise about WHICH half this gates. `fold_sink` carries no
    # `warp_group_idx` term, so the CLAMP below fires in WG1's partition-0 warp
    # too; only the MASS add is WG0-gated, at its own site. That asymmetry is
    # benign and stays benign: the two-level combine is exactly scale-invariant
    # in `m_wg` (a larger `m_wg` shrinks `l_wg` by the same factor), so WG1
    # carrying `m >= sink` with no matching mass changes nothing. Do not "fix"
    # it to match the comment without re-deriving the combine -- and note the
    # shared-key exchange propagates WG1's clamped max to all four of WG1's
    # warps, which is still scale-invariant.
    comptime if config.use_ws:
        fold_sink = fold_sink and partition_g == UInt32(0)

    # Monotone `fa4_ws_exchange4` counter for this warpgroup. Declared ahead of
    # the sink clamp because the PEEL exchange below needs it, and shared from
    # here through the KV loop and into the epilogue -- ONE counter, because its
    # double-buffering proof (see that function's docstring) needs the sequence
    # consecutive across the warpgroup's whole lifetime: exchange `i` writes
    # parity `i & 1`, and a warp two exchanges ahead would land on the parity a
    # lagging sibling is still reading. So the epilogue's row-sum must be
    # exactly `last + 1`, never a restart at 0 and never a jump.
    #
    # Skipping or repeating a value is not a near-miss: it puts two successive
    # exchanges on the SAME buffer, which is a plausible-but-wrong row max --
    # no fault, no hang. Nothing machine-checks it either: not the helper, not
    # `supported()`, and not `test_ws_exchange_reduce4.mojo`, whose every arm
    # passes a clean `range`. `sk_shared_max` is the mechanism instead -- it is
    # the only caller of the max exchange AND the only writer of this counter,
    # so the two cannot drift. Do not open-code either half.
    var ws_exchange_seq: UInt32 = 0

    @__parameter
    @always_inline
    def sk_shared_max(m: Float32) -> Float32:
        """Shared-key: agree the running row max across the warpgroup's warps.

        Off-mode this is the identity and elaborates to nothing.

        On-mode the `m_pack` warps of a warpgroup share ONE O accumulator, so
        they must rescale it as a unit, which means agreeing the max every KV
        tile. Their local maxima genuinely differ: each warp still owns a
        disjoint key quarter (`kv_row += partition_g * score_cols`), and that
        disjointness is the entire reason this exists.

        Only the MAX is shared. Row sums stay per-warp and are summed once in
        the epilogue -- which is exact precisely BECAUSE the maxima agree. With
        a shared `old_max` and a shared `new_row_max` every warp computes the
        same `diff`, resolves the lazy-rescale vote the same way, and gets a
        bit-identical `correction` (the helper's fold order is fixed), so the
        four per-quarter sums stay in one scale and the combine is a plain add.
        """
        comptime if config.ws_shared_key:
            var shared = fa4_ws_exchange4[
                "max", m_pack=config.m_pack, rows=config.BM
            ](
                m,
                thread_tile_row,
                partition_g,
                warp_group_idx,
                ws_exchange_seq,
                smem.ws_exchange_smem(),
            )
            ws_exchange_seq += 1
            return shared
        else:
            return m

    comptime if not SinkType.is_null:
        var sink_weights_ptr = rebind[
            UnsafePointer[Scalar[qkv_type], ImmutAnyOrigin]
        ](sink_weights.value())

        comptime if use_fma:
            sink_weight = sink_weights_ptr[q_head_idx].cast[accum_dtype]()
        else:
            sink_weight = (
                sink_weights_ptr[q_head_idx].cast[accum_dtype]() * log2e
            )
        if fold_sink:
            row_max = max(row_max, sink_weight)
    else:
        sink_weight = 0.0

    # PEEL EXCHANGE -- the first of two, and the one the sink ordering
    # constraint actually governs.
    #
    # `peel_mask` already folded the first tile into `row_max`, and the
    # `store_exp` further below turns it into P. Off-mode that P is per-warp and
    # correct. On-mode the warps share one O accumulator, so the peel's P has to
    # go out at ONE exponent scale like every later tile's -- an in-loop
    # exchange alone would leave exactly the first tile wrong.
    #
    # It runs AFTER the clamp above, and that order is load-bearing. The clamp
    # is warp-0-only (`fold_sink and partition_g == 0`), so in this order the
    # sink enters through warp 0's operand and the `max` reduction hands
    # `max(all four maxima, sink)` to all four warps -- exact, and free. Reverse
    # them and only warp 0 carries the sink: warps 1-3 write P at a different
    # scale and the epilogue's deferred row-sum is wrong by `exp2((m_0 - m_g) *
    # s)` per warp. A silent wrong answer, INVISIBLE without a sink, since
    # `fold_sink == False` makes the clamp a no-op and the two orders agree. The
    # dataflow is the enforcement: this call consumes the clamped value.
    #
    # A full-warpgroup barrier HERE is safe because every `return` that can
    # precede it is warpgroup-uniform: the 1Q T<=1 exit parks the whole of WG1,
    # and the three empty-partition / all-masked exits are gated on
    # `total_iters_combined`, a per-CTA quantity (each of those blocks arrives at
    # a 256-count `named_barrier(2)` before returning, so a per-warp predicate
    # there would already hang today). This exchange adds NO uniformity
    # obligation the function did not already carry. A future early return
    # between `peel_mask` and here WOULD: make it warpgroup-uniform, or skip the
    # exchange for the whole warpgroup.
    row_max = sk_shared_max(row_max)

    # use_fma carries the running max in the negated log2-domain
    # (`neg_max_scaled = -row_max*scale_log2e`, with `p_fp8_bias` folded in for
    # fp8). Computing it once and carrying it lets the K-loop share this
    # `*scale_log2e` between the online-softmax correction diff and store_exp's
    # bias term, dropping one scalar FMUL per iter. `neg_scale_log2e` hoists the
    # sign flip out of the loop. use_fma implies null sink/qscale, so
    # `row_max`/`scale_log2e` are final here.
    # Function-scoped so the carry survives across K-loop iterations (a var
    # first-assigned inside a `comptime if` would be block-scoped). Dead `0.0`
    # init covers the non-fma path (unused there, DCE'd), as with `wr` above.
    var neg_scale_log2e: Float32 = 0.0
    var neg_max_scaled: Float32 = 0.0
    comptime if use_fma:
        neg_scale_log2e = -scale_log2e

    @__parameter
    @always_inline
    def neg_scaled_max(m: Float32) -> Float32:
        # `-m*scale_log2e`, with the fp8 `p_fp8_bias` folded in via one fused
        # fma -- so store_exp needs no separate bias add and, since the bias is
        # common to every max, it cancels in the correction diff
        # `nms_new - nms_old`. bf16 (bias == 0) uses a plain ftz mul (no fma) to
        # avoid a spurious `+ 0.0`; the fp8 form is bit-identical to the old
        # `fma(-row_max, scale_log2e, p_fp8_bias)` store_exp bias.
        comptime if p_fp8_bias != 0:
            return fma_ftz(m, neg_scale_log2e, p_fp8_bias)
        else:
            return mul_ftz(m, neg_scale_log2e)

    # TRUE per-lane running max, distinct from the lazy/stale `row_max`;
    # orthogonal to the #92318 scaled-domain carry (raw units).
    var m_true: Float32 = row_max

    # Vote-ALL (not vote-ANY): skipping needs every lane's agreement, since
    # P@V is monolithic over the WG and one dissenting row's contribution
    # would be lost. Published to smem keyed by (wg, S-consumer phase) --
    # double-buffered since softmax can run one block ahead of the MMA's
    # P@V read; the peel's zero-inited region reads as "don't skip".
    var blasst_vote = smem.blasst_vote_smem()
    var blasst_warp_in_wg: UInt32 = warp_idx & UInt32(3)
    var blasst_lane0: Bool = (tid % UInt32(32)) == UInt32(0)

    @__parameter
    @always_inline
    def blasst_observe(tile_max: Float32) -> Bool:
        m_true = max_ftz(m_true, tile_max)
        var diff_true = sub_ftz(tile_max, m_true)
        comptime if use_fma:
            diff_true = mul_ftz(diff_true, scale_log2e)
        var would_skip = _vote_nvidia_helper(
            diff_true < blasst_log_threshold
        ) == UInt32(0xFFFFFFFF)
        # `pipeline_s.state.phase()` is the S-consumer generation of the block
        # softmax is currently producing (store_exp steps it AFTER this).
        var vote_phase: UInt32 = pipeline_s.state.phase()
        var vote_base: UInt32 = (
            warp_group_idx * UInt32(2) + vote_phase
        ) * UInt32(4)
        if blasst_lane0:
            blasst_vote[vote_base + blasst_warp_in_wg] = UInt8(
                1
            ) if would_skip else UInt8(0)
        # Strip only on the WG-unanimous vote (P@V is monolithic over the WG;
        # one warp skipping while a sibling keeps would multiply stale P).
        # Barrier id = warp_group_idx so the two WGs never cross-sync.
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        return blasst_vote_unanimous(blasst_vote, warp_group_idx, vote_phase)

    # Cross-stage P JIT-P1 seed: WG1 consumes one extra s0_p1 token, giving
    # WG0 a one-block lead so WG1's P1(m) store waits on WG0's block-(m+1) S0
    # release -- the WAR that makes the QK0-first MMA order legal (see
    # mma_warp cross-P block). WG0 needs no seed (covered by MMA program
    # order). Balanced by the WG0 drain after the main loop below.
    comptime if CrossP:
        if warp_group_idx == UInt32(1):
            inplace_cons.wait()
            inplace_cons.step()

    var row_sum: f32x2
    comptime if use_fma:
        neg_max_scaled = neg_scaled_max(row_max)
        row_sum = store_exp(neg_max_scaled)
    else:
        row_sum = store_exp(row_max)

    var o_phase: UInt32 = 0  # initial wait is phase 0

    comptime if not SinkType.is_null:
        # The sink mass must land in `global_sum` exactly once per Q row.
        #
        # 2Q: each WG owns a disjoint set of Q rows, so adding the sink to
        # every WG's `row_sum` already contributes it once per row.
        #
        # 1Q: both WGs cover the SAME Q rows but stride over disjoint halves
        # of the K/V stream (`kv_row_stride = 2*BN`), then LSE-combine their
        # `row_sum`s (`global_sum = row_sum_total*scale_local +
        # peer_sum*scale_peer`, ~L1548). Adding the sink to BOTH WGs would
        # double-count it in `global_sum`, inflating the denominator and
        # shrinking every output (the gpt-oss-20b sink bug). Add it in WG0
        # only. WG0 must be the carrier because the T==1 fast path returns
        # WG1 early (~L1257) before any LSE exchange, so WG0 always survives
        # to fold the sink into the combined denominator.
        # The sink mass must enter row_sum in the SAME scale as the P values
        # stored by store_exp, so it cancels through the final 1/row_sum
        # normalize. fp8 adds the same +p_fp8_bias as store_exp; the
        # `comptime if p_fp8_bias != 0` keeps the bf16 sink expression
        # byte-identical.
        @__parameter
        @always_inline
        def sink_mass() -> Float32:
            comptime if use_fma:
                comptime if p_fp8_bias != 0:
                    return exp2(
                        (sink_weight - row_max) * scale_log2e + p_fp8_bias
                    )
                else:
                    return exp2((sink_weight - row_max) * scale_log2e)
            else:
                comptime if p_fp8_bias != 0:
                    return exp2(sink_weight - row_max + p_fp8_bias)
                else:
                    return exp2(sink_weight - row_max)

        comptime if config.num_q == 1:
            # WG0-only (intra-CTA once-per-row) AND, under split-K,
            # partition-0-only (once-per-cluster). `fold_sink` carries the
            # partition-0 gate; folds to `True` when split-K is off.
            if warp_group_idx == UInt32(0) and fold_sink:
                row_sum[0] += sink_mass()
        else:
            row_sum[0] += sink_mass()

    # Lazy-rescale gate for online softmax: only re-scale the accumulator
    # (and adopt the new running max) when `old_max - new_row_max <
    # rescale_threshold` in the log2 domain. Below that, we keep the stale max
    # and skip the rescale; the new exp2(score - old_max) terms stay within
    # 2^|rescale_threshold| of the existing scale, which fp32 accumulation can
    # absorb without meaningful loss. bf16 uses -8 (256x); fp8 uses -2 (a
    # tighter gate that rescales sooner, chosen with `p_fp8_bias` above -- the
    # knob it is linked to via `p_fp8_bias = 8 + rescale_threshold`).

    # 1Q advances kv_row by 2*BN (each WG strides over its half of the
    # K/V stream); 2Q advances by BN (each WG processes every K tile).
    # single-O (1Q wide-V): WG0 processes CONSECUTIVE tiles (WG1 no-op),
    # so it strides by BN like 2Q.
    comptime kv_row_stride: Int = (
        2
        * config.BN if (
            config.num_q == 1 and not config.single_o
        ) else config.BN
    )

    var diff: Float32
    var local_rowsum: f32x2
    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        comptime for i in range(num_sets):
            comptime mask_status = mask_sets[i]
            comptime mask_strategy = mask_strategies[i]
            var iters: UInt32

            iters = warp.broadcast(mask_iters[i])
            while iters != 0:
                iters -= 1
                kv_row += UInt32(kv_row_stride)
                # calculate rowmax
                var old_max = row_max
                var new_row_max: Float32
                # WG-unanimous skip decision (comptime-False -> byte-identical
                # OFF).
                var blasst_wg_skip: Bool = False
                comptime if ENABLE_BLASST:
                    # One reduction feeds both: pure `tile_max` for the vote
                    # (folding old_max in would suppress the skips BLASST
                    # targets), and `new_row_max` via max_ftz (bitwise-identical
                    # to the init-fused form, fewer instructions).
                    var tile_max = maximum(
                        load_mask_tile_max[mask_strategy](kv_row)
                    )
                    new_row_max = max_ftz(tile_max, old_max)
                    blasst_wg_skip = blasst_observe(tile_max)
                else:
                    new_row_max = load_mask_max[mask_strategy](kv_row, old_max)

                # PER-ITERATION EXCHANGE. Placed before `diff` and not merely
                # before `store_exp`, because the lazy-rescale gate below is a
                # warp VOTE: only a shared `new_row_max` (against an
                # already-shared `old_max`) makes the vote resolve identically
                # in all four warps, and hence `correction` bit-identical. Move
                # it after the vote and the warps disagree about whether they
                # rescaled -- their O quarters drift apart with no diagnostic.
                # Unconditional in the iteration, so the counter advances in
                # lockstep across the warpgroup whatever the skip vote says.
                new_row_max = sk_shared_max(new_row_max)

                if not blasst_wg_skip:
                    # Loop-body scoped so `nms_new` reaches the gate's adoption
                    # below (dead `0.0` for non-fma; DCE'd).
                    var nms_new: Float32 = 0.0
                    comptime if use_fma:
                        # scaled-domain carry: one FMUL (nms_new) feeds both
                        # the correction diff and store_exp's bias.
                        nms_new = neg_scaled_max(new_row_max)
                        diff = sub_ftz(nms_new, neg_max_scaled)
                    else:
                        diff = sub_ftz(old_max, new_row_max)
                    var correction: Float32

                    comptime if rescale_threshold < 0:
                        # old_max - new_row_max < -8
                        # 8 < new_row_max - old_max
                        #
                        # Per-lane predicate, not a warp vote: thread_tile_row
                        # = tid % BM packs unrelated Q rows into one warp, so
                        # an OR here would leak a sibling row's rescale.
                        if diff < rescale_threshold:
                            row_max = new_row_max
                            comptime if use_fma:
                                neg_max_scaled = nms_new
                            correction = exp2(diff)
                        else:
                            correction = 1
                    else:
                        row_max = new_row_max
                        comptime if use_fma:
                            neg_max_scaled = nms_new
                        correction = exp2(diff)
                    correction_smem[] = correction
                    pipeline_c.commit()
                    # update s->p
                    comptime if use_fma:
                        local_rowsum = store_exp(neg_max_scaled)
                    else:
                        local_rowsum = store_exp(row_max)
                    row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
                else:
                    # Stripped: no exp2/P-store/row-sum; identity correction
                    # leaves prior-block O untouched.
                    correction_smem[] = 1.0
                    pipeline_c.commit()
                    store_exp_skip()
                o_phase ^= 1
    else:
        # The shared-key row-max exchange cannot be wired into THIS arm, and
        # the combination is refused rather than left to hang.
        #
        # The `break` and `continue` below are driven by `kv_row` and by a
        # per-tile mask status, and `kv_row` is PER WARP (`kv_row +=
        # partition_g * score_cols`), so the warps of a warpgroup can take
        # different trip counts. `fa4_ws_exchange4` arrives at a 128-count
        # `named_barrier`; different trip counts on a full-warpgroup barrier is
        # a hang, not a wrong answer.
        #
        # Stated as a build error (not a runtime assert) because UNKNOWN_MASK
        # dispatch is currently gated OFF, and this assert is the obligation any
        # future ungating inherits. The work then is making the trip counts
        # warp-uniform (or hoisting the exchange out of the skip), not deleting
        # this assert.
        comptime assert not config.ws_shared_key, (
            "shared-key + UNKNOWN_MASK: this runtime mask-skip loop gives the"
            " warps of a warpgroup different trip counts, which hangs the"
            " row-max exchange's warpgroup barrier"
        )
        while True:
            kv_row += UInt32(kv_row_stride)
            if kv_row >= num_keys:
                break
            var cur_mask_status = mask.status(
                seq_info.prompt_idx,
                Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                Index[dtype=DType.int32](BM_mask, BN),
            )
            if cur_mask_status == TileMaskStatus.FULL_MASK:
                continue
            # calculate rowmax
            var old_max = row_max
            var new_row_max: Scalar[accum_dtype]
            # WG-unanimous skip decision (comptime-False -> byte-identical OFF).
            var blasst_wg_skip: Bool = False
            comptime if ENABLE_BLASST:
                # See the main-loop note above: one reduction feeds both the
                # pure tile_max (vote) and new_row_max (max_ftz fold).
                var tile_max: Float32
                if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                    tile_max = maximum(
                        load_mask_tile_max[
                            MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
                        ](kv_row)
                    )
                else:
                    tile_max = maximum(
                        load_mask_tile_max[MaskStrategy.OUT_OF_BOUNDS](kv_row)
                    )
                new_row_max = max_ftz(tile_max, old_max)
                blasst_wg_skip = blasst_observe(tile_max)
            else:
                if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                    new_row_max = load_mask_max[
                        MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
                    ](kv_row, old_max)
                else:
                    new_row_max = load_mask_max[MaskStrategy.OUT_OF_BOUNDS](
                        kv_row, old_max
                    )

            if not blasst_wg_skip:
                var nms_new: Float32 = 0.0
                comptime if use_fma:
                    # scaled-domain carry: one FMUL (nms_new) feeds both the
                    # correction diff and store_exp's bias (see pre-loop note).
                    nms_new = neg_scaled_max(new_row_max)
                    diff = sub_ftz(nms_new, neg_max_scaled)
                else:
                    diff = sub_ftz(old_max, new_row_max)
                var correction: Float32

                comptime if rescale_threshold < 0:
                    # old_max - new_row_max < -8
                    # 8 < new_row_max - old_max
                    # Per-lane predicate, not a warp vote -- see the gate above.
                    if diff < rescale_threshold:
                        row_max = new_row_max
                        comptime if use_fma:
                            neg_max_scaled = nms_new
                        correction = exp2(diff)
                    else:
                        correction = 1
                else:
                    row_max = new_row_max
                    comptime if use_fma:
                        neg_max_scaled = nms_new
                    correction = exp2(diff)
                correction_smem[] = correction
                pipeline_c.commit()
                # update s->p
                comptime if use_fma:
                    local_rowsum = store_exp(neg_max_scaled)
                else:
                    local_rowsum = store_exp(row_max)
                row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
            else:
                # Stripped (see main-loop note): no exp2/P-store/row-sum.
                correction_smem[] = 1.0
                pipeline_c.commit()
                store_exp_skip()
            o_phase ^= 1

    # Cross-stage P JIT-P1 drain: WG0-only extra s0_p1 commit that balances
    # the WG1 seed above, consumed by WG1's final P store.
    comptime if CrossP:
        if warp_group_idx == UInt32(0):
            inplace_prod.commit()

    # Do the final correction and write.
    comptime assert size_of[output_type]() >= size_of[qkv_type]()

    comptime if config.num_q == 2:
        # 2Q: each WG writes its row half independently.
        var inv_row_sum = recip(row_sum.reduce_add())
        # `BM // config.num_q` matches the helper's signature
        # (`config.BM // config.num_q`) at the comptime-expression level;
        # numerically identical to HalfBM = BM // 2 inside this 2Q branch.
        # The `TMemTile` type param stays `padded_ov_depth` while the stride
        # comes from the TMEM addresses: the two are equal here (shared-key
        # requires `num_q == 1`), and the 14 `TMemTile` declarations form one
        # type-identity set that must move together.
        var o_tile = TMemTile[accum_dtype, BM // config.num_q, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + warp_group_idx * UInt32(config.TMEM_O1 - config.TMEM_O0)
        )

        # wait on the o_pipeline producer
        @__parameter
        @always_inline
        def wait_and_write_output():
            o_prod_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?

            fa4_scale_write_output[config](
                row,
                warp_idx & 3,
                warp_group_idx,
                inv_row_sum,
                o_smem + warp_group_idx * UInt32(HalfBM * padded_ov_depth),
                o_tile,
                ragged_tma_store,
                num_output_rows,
                head_idx,
                gmem_row
                + cta_q_offset
                + warp_group_idx
                * UInt32(HalfBM // group if fuse_gqa else HalfBM),
            )

        # `output_nonempty` statically discharges the guard: the entry
        # kernel routed every tile with
        # `seq_len - prompt_offset <= wg_row_offset_seq` to the 1Q body,
        # so both WGs' halves are non-empty whenever this 2Q path runs.
        comptime if output_nonempty:
            debug_assert(
                num_output_rows > 0,
                "1Q switch must take every tile with an empty output half",
            )
            wait_and_write_output()
        else:
            if num_output_rows > 0:
                wait_and_write_output()
    else:
        # 1Q output. T==1 takes a fast path (WG0 has the full output
        # in TMEM_O0 and WG1 has no work / already returned); T>=2
        # combines per-WG partials via LSE exchange.
        #
        # No `num_output_rows > 0` guards on the 1Q write paths: 1Q has no
        # per-WG row offset (`wg_row_offset == 0`) and is single-CTA
        # (`supported()` forbids pair_cta), so `num_output_rows =
        # min(seq_len - prompt_offset, per_qo_BM_seq)`, which is >= 1
        # because every caller dispatches the softmax warps only for tiles
        # with `seq_info.is_valid()` (seq_len > prompt_offset).
        debug_assert(
            num_output_rows > 0,
            "1Q tiles always have output rows (is_valid() holds)",
        )
        # Split-K (num_q==1): each partition computes its OWN K-slice, so its
        # normalized output is a per-partition partial. `splitk_combine_active`
        # (hoisted above the T<=1 guard, since the empty-partition neutral-stage
        # path needs it too) gates the M4 writer-combines-all combine, which is
        # unconditional whenever split-K is active (`splitk_partitions > 1`):
        # every partition stages its partial + (max,sum), and the writer
        # DSMEM-reads all P partitions and writes the FULL combined O. The test
        # compares that combined O against `mha_gpu_naive` over the full key
        # range. `is_writer` (= `splitk_partition_idx == FA4_1Q_SPLITK_WRITER`,
        # default partition 0) selects the writer for the T==1 combine path; it
        # folds to a constant `True` when split-K is off (`splitk_partitions ==
        # 1`), leaving pair-CTA and single-CTA unchanged. `block_rank_in_cluster()`
        # is block-uniform, so gating on `is_writer` does not diverge the
        # `named_barrier` calls below.
        var is_writer: Bool = True
        comptime if config.splitk_partitions > 1:
            comptime SPLITK_WRITER = get_defined_int[
                "FA4_1Q_SPLITK_WRITER", 0
            ]()
            # Same `splitk_num_partitions` the mma/load/correction warps use --
            # they slice the same window, so the count must match by
            # construction, not by two sites agreeing.
            is_writer = splitk_partition_idx(
                splitk_num_partitions[config](ws_num_partitions)
            ) == UInt32(SPLITK_WRITER)
        # === Warp-specialized (BM=32, m_pack=4) epilogue: two-level combine. ===
        # (Md) The WS packed-TMEM combine is handled HERE and RETURNS before the
        # non-WS 1Q epilogue below. That non-WS code is comptime-elaborated even
        # under a WS instantiation (it is dead for WS but type-checks -- its only
        # comptime asserts, `num_q==1`/2-byte output/`iters_per_wg` bounds, all
        # hold for WS), so no `comptime if not use_ws` guard is needed. For a
        # non-WS config this whole block is pruned -> byte-identical. WS is
        # single-CTA with `splitk_partitions == 1`.
        comptime if config.use_ws:
            # SMEM scratch carved from the Q+KV span. Layout mirrors the
            # B200-proven `test_ws_intracta_combine`: [stage0|stage1|maxsum0|
            # maxsum1|l2_stage|l2_maxsum|o]. `ws_o` (the bf16 output) sits past
            # the f32 regions; the TMA egress reads whatever pointer it is handed.
            #
            # HAZARD: `ws_f32 == smem.o_smem() == the Q base
            # (q_byte_offset == 0)`, and for BM=32 the Q region is tiny
            # (q_bytes == BM*padded_qk_depth*2 == 8192 B) while `ws_stage0` alone
            # is m_pack*BM*ov_depth*4 == 65536 B, so the staging spills FAR past
            # Q into the LIVE KV ring slots. The budget assert below only proves
            # it FITS q_bytes+kv_bytes -- it does NOT prove the KV ring is
            # drained. Every staging write MUST therefore be gated behind the O
            # producer(s) that drain the ring: T==1 waits o0 (MMA returns after
            # o0); T>=2 waits BOTH o0 and o1 (the peer wait -- the MMA's epilogue
            # reads ring-wrapped V_o[0] between the o0 and o1 commits). Do not
            # stage before those waits.
            #
            # The hazard is `ws_f32`'s, not `ws_o`'s: `ws_o` is derived BELOW
            # every f32 region and is the last thing in the carve. It applies
            # under shared-key too, where the f32 regions collapse to
            # `ws_l2_stage` alone -- still `ov_depth * BM` f32 (32 KiB at d256)
            # against a `q_bytes` of 16 KiB, still written by WG1, so still
            # spilling into live ring slots. A 6x smaller carve is not no carve.
            #
            # TWO epilogues share this carve, selected by `ws_shared_key`.
            #
            # The KEY-SPLIT one (the bulk of this block, past the shared-key
            # early return) folds the four packed-TMEM key partitions through
            # `ws_stage*`, because each of a warpgroup's `m_pack` warps carries
            # an independent full-depth O partial.
            #
            # SHARED-KEY has no such fold: one key band, one accumulator, and
            # the quarters are DEPTH bands, so warp `g`'s TMEM subpartition is
            # already its final band. Its `ws_epilogue_stage_f32()` and
            # `ws_epilogue_ml_f32()` are therefore 0, collapsing the carve to
            # `~48 KiB` and so fitting depth 256/512. What survives is
            # `[l2_stage | l2_maxsum | ws_o]` -- the same three regions, at the
            # same expressions, since the two zeroed terms drop out of the
            # running sum below. The carve is written ONCE, here, for both modes.
            comptime WS_STAGE = config.ws_epilogue_stage_f32()
            comptime WS_ML = config.ws_epilogue_ml_f32()
            comptime WS_L2STAGE = config.ov_depth * config.BM
            comptime WS_L2MS = config.BM * 2
            comptime assert (
                config.ws_epilogue_f32_slots() * size_of[DType.float32]()
                + config.BM * config.ov_depth * size_of[output_type]()
                <= type_of(smem).q_bytes + type_of(smem).kv_bytes
            ), "WS two-level combine staging must fit the dead Q+KV span"
            var ws_f32 = smem.o_smem[.float32]()
            var ws_stage0 = ws_f32
            var ws_stage1 = ws_stage0 + WS_STAGE
            var ws_maxsum0 = ws_stage1 + WS_STAGE
            var ws_maxsum1 = ws_maxsum0 + WS_ML
            var ws_l2_stage = ws_maxsum1 + WS_ML
            var ws_l2_maxsum = ws_l2_stage + WS_L2STAGE
            var ws_o = (ws_l2_maxsum + WS_L2MS).bitcast[Scalar[output_type]]()

            var ws_row_sum = row_sum.reduce_add()

            comptime if config.ws_shared_key:
                # ---- Shared-key epilogue ----
                # Early `return` rather than an `else` around the key-split code
                # below, to keep that block un-reindented.
                #
                # There is no Level 1: warp `g`'s packed-TMEM subpartition IS
                # depth band `g` of the one shared accumulator, so
                # `fa4_ws_level0_band` reads it straight to registers -- no
                # staging, no fold, no barrier. Level 2 stays, because it is the
                # cross-WARPGROUP key merge (WG0 even tiles, WG1 odd), which
                # this mode does not touch.
                #
                # Barrier budget of everything below, and it is the spec:
                #   1 (row-sum) + num_o_tiles() (Level 2) + 1 (egress)
                #   + 1 (terminal)
                # -- five at d512. Level 0 needs none, and the tiles need none
                # between them: Level 2's own barrier(3) is the next sync point
                # either way.
                comptime SK_BAND = config.pv_mma_n() // config.m_pack

                # The walk must cover the accumulator exactly: too few tiles
                # leaves the top of O unwritten, too many run past the TMEM
                # allocation. `num_o_tiles()` is DEFINED by this invariant, so
                # this guards the definition drifting, it does not restate it.
                comptime assert (
                    config.num_o_tiles() * SK_BAND == config.o_phys_cols()
                ), "shared-key epilogue does not tile o_phys_cols() exactly"
                # In-warp mirror of `supported()` conjunct (5): the per-tile L2
                # windows sum to `padded_ov_depth * BM` f32 while the carve
                # provides `ov_depth * BM`. Equal only when the swizzle does not
                # pad; unequal means the last window runs over `ws_l2_maxsum`
                # and the output tile, silently.
                comptime assert (
                    config.num_o_tiles() * config.pv_mma_n() * config.BM
                    <= WS_L2STAGE
                ), "shared-key per-tile L2 windows overrun the epilogue carve"
                # In-warp mirror of `supported()` conjunct (4). Both fused
                # split-K arms stage the per-CTA (M, L) in `ws_maxsum1`, which
                # is the SAME pointer as `ws_l2_stage` once
                # `ws_epilogue_ml_f32()` is 0. Rejected at dispatch; asserted
                # here so a hand-built config is a build error, not corruption.
                comptime assert config.splitk_partitions == 1, (
                    "shared-key + fused split-K: ws_maxsum1 aliases ws_l2_stage"
                    " once ws_epilogue_ml_f32() == 0 -- re-carve them first"
                )
                # `single_o` parks WG1 for EVERY T, not just T==1 (see the
                # num_q==1 early-out), so the T>=2 arm's Level 2 would wait a
                # 256-count barrier(3) that 128 threads never reach -- a hang.
                # The early-out's own comment asserts "the packed WS config has
                # single_o == False"; this makes that claim checked rather than
                # stated. (The key-split T>=2 arm below has the same exposure
                # and the same protection-by-comment -- not this task's to fix,
                # but do not read its silence as a proof.)
                comptime assert not config.single_o, (
                    "shared-key + single_o parks WG1 for every T, so Level 2's"
                    " 256-count barrier(3) would hang"
                )

                # ONE `m_pack`-way add, OUTSIDE the tile loop. Each warp
                # accumulated the row sum over its own key quadrant, and the
                # total is a plain add ONLY because all four already share
                # `row_max` -- which `sk_shared_max` establishes at the peel and
                # again every KV tile. Remove either of those calls and this
                # add silently stops being a sum of comparable quantities.
                #
                # `ws_exchange_seq` arrives here as `1 + <loop iterations>`,
                # continuing the sequence rather than restarting: the counter is
                # written in exactly one place, and only by `sk_shared_max`.
                #
                # Ordered BEFORE the O-producer waits on purpose:
                # `ws_exchange_smem()` is carved PAST `kv_bytes`, so it is the
                # one epilogue SMEM touch that does not alias the live KV ring,
                # and overlapping it with the MMA drain costs nothing.
                var sk_l_wg = fa4_ws_exchange4[
                    "add", m_pack=config.m_pack, rows=config.BM
                ](
                    ws_row_sum,
                    thread_tile_row,
                    partition_g,
                    warp_group_idx,
                    ws_exchange_seq,
                    smem.ws_exchange_smem(),
                )

                @__parameter
                @always_inline
                def sk_epilogue[single_wg: Bool]():
                    # (M, L) are depth-independent, so every tile computes the
                    # same pair; keep tile 0's. They are carried OUT of the loop
                    # rather than used inside it because the egress below must
                    # not run until every tile has packed its bands.
                    var sk_m = Float32(0)
                    var sk_l = Float32(0)
                    comptime for t in range(config.num_o_tiles()):
                        # ONE tile's band, never `o_phys_cols()`: at d512 the
                        # latter is 128 live f32 against num_reg_softmax = 192.
                        var o_band = Array[Scalar[DType.float32], SK_BAND](
                            uninitialized=True
                        )
                        # All `m_pack` warps issue at the SAME column address;
                        # the subpartition routes each its own quarter. No
                        # per-warp column offset.
                        fa4_ws_level0_band[SK_BAND](
                            tmem_addr
                            + UInt32(config.TMEM_O0)
                            + warp_group_idx
                            * UInt32(config.TMEM_O1 - config.TMEM_O0)
                            + UInt32(t * SK_BAND),
                            o_band,
                        )
                        var m_t, l_t = fa4_ws_level2_reduce_scatter_write[
                            config.m_pack,
                            config.BM,
                            config.ov_depth,
                            band_cols=SK_BAND,
                            depth_base=t * config.pv_mma_n(),
                            single_wg=single_wg,
                            use_fma=use_fma,
                        ](
                            thread_tile_row,
                            partition_g,
                            warp_group_idx,
                            row_max,
                            sk_l_wg,
                            scale_log2e,
                            o_band,
                            # Each tile gets its OWN window. Level 2 is
                            # scatter -> barrier -> gather and NOTHING orders
                            # WG1's next scatter against WG0's current gather,
                            # so one shared window is a WAR race. Summed over
                            # the tiles this is exactly `WS_L2STAGE` -- the fix
                            # costs no SMEM and no barrier.
                            ws_l2_stage + t * config.pv_mma_n() * config.BM,
                            ws_l2_maxsum,
                            ws_o,
                        )
                        comptime if t == 0:
                            sk_m = m_t
                            sk_l = l_t

                    # AFTER the loop, not inside it at `t == 0`. Level 2 packs
                    # only THIS tile's `pv_mma_n()` output columns into `ws_o`,
                    # so at `num_o_tiles() > 1` an egress from inside the loop
                    # TMA-stores a half-written tile -- the depth columns of
                    # every later tile would still hold whatever the dead Q/KV
                    # span held. Invisible at d256 (one tile, so the loop body
                    # IS the whole epilogue), silently wrong at d512.
                    #
                    # `fa4_tma_store_o_smem` fences with its own
                    # `named_barrier[128](warp_group_idx)`, and only WG0 ever
                    # writes `ws_o` (Level 2's gather arm), so a WG-scoped
                    # barrier is the right width here.
                    if warp_group_idx == UInt32(0):
                        fa4_tma_store_o_smem(
                            warp_idx & 3,
                            warp_group_idx,
                            ws_o,
                            ragged_tma_store,
                            num_output_rows,
                            head_idx,
                            gmem_row + cta_q_offset + ws_o_row_off,
                        )
                    _ws_write_lse(sk_m, sk_l)

                if total_iters_combined == UInt32(1):
                    # WG0 alone: WG1 returned at the num_q==1 early-out having
                    # already taken `named_barrier[256](2)`, so Level 2's
                    # 256-count barrier(3) would HANG. `single_wg` drops the
                    # whole cross-WG transport rather than feeding it a neutral
                    # peer, which would not fix the barrier count.
                    o_prod_mbar[0].wait(o_phase)
                    tcgen05_fence_after()
                    sk_epilogue[True]()
                    named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
                else:
                    # BOTH waits, for the two distinct reasons the key-split
                    # T>=2 arm documents below. The OWN wait gates this WG's
                    # TMEM read. The PEER wait is the SMEM/KV happens-before for
                    # `ws_l2_stage`, which under this mode is still
                    # `ov_depth * BM` f32 -- 32 KiB at d256 against a `q_bytes`
                    # of 16 KiB -- so it still spills into live ring slots that
                    # the MMA's epilogue reads between the o0 and o1 commits.
                    # Dropping WS_STAGE shrank the hazard; it did not remove it.
                    o_prod_mbar[warp_group_idx].wait(o_phase)
                    tcgen05_fence_after()
                    o_prod_mbar[UInt32(1) - warp_group_idx].wait(
                        o_phase ^ (total_iters_combined & UInt32(1))
                    )
                    sk_epilogue[False]()
                    named_barrier[Int32(2 * WARPGROUP_SIZE)](4)
                comptime if (
                    not config.pair_cta and config.splitk_partitions == 1
                ):
                    if warp_idx == 0:
                        tcgen05_release_allocation_lock[Int32(cta_group)]()
                        tcgen05_dealloc[Int32(cta_group)](
                            tmem_addr, UInt32(config.sm100_tmem_cols)
                        )
                return

            if total_iters_combined == UInt32(1):
                # T==1 fast path: WG0 alone holds all 4 quarters in TMEM_O0 (WG1
                # returned at the num_q==1 early-out). 4-way combine + egress.
                o_prod_mbar[0].wait(o_phase)
                tcgen05_fence_after()
                comptime if splitk_combine_active:
                    # T==1 cross-CTA split-K: WG0's 4-way intra-CTA combine emits
                    # the NORMALIZED per-CTA band to regs (+ (M_cta, L_cta)), then
                    # the third-level reduce-scatter combines across the P CTAs
                    # and TMA-stores this CTA's OWN depth band. Uses the dead
                    # `ws_l2_stage`/`ws_maxsum1` for the cross-CTA staging (the
                    # intra-CTA `ws_stage0`/`ws_maxsum0` are consumed above).
                    var o_band_norm_t1 = Array[
                        Float32,
                        config.ov_depth // config.m_pack,
                    ](uninitialized=True)
                    var m_cta_t1, l_cta_t1 = fa4_ws_intracta_combine_partial[
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        row_max,
                        ws_row_sum,
                        scale_log2e,
                        tmem_addr + UInt32(config.TMEM_O0),
                        ws_stage0,
                        ws_maxsum0,
                        o_band_norm_t1,
                    )
                    fa4_ws_splitk_reduce_scatter_write[
                        config,
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        P=config.splitk_partitions,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        UInt32(config.splitk_partitions),
                        splitk_partition_idx(UInt32(config.splitk_partitions)),
                        m_cta_t1,
                        l_cta_t1,
                        scale_log2e,
                        o_band_norm_t1,
                        ws_l2_stage,
                        ws_maxsum1,
                        ws_o,
                        smem.misc_mbars().publish_mbar(),
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset,
                    )
                else:
                    # Plain single-CTA WS combine (also the workspace / unfused
                    # split-K egress: `ws_o_row_off` redirects the O TMA into
                    # this partition's `o_partial` slice and `_ws_write_lse`
                    # writes the per-row LSE; both fold to the non-split store
                    # when `workspace_split` is off).
                    var m_cta_t1_ws, l_cta_t1_ws = fa4_ws_intracta_combine[
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        row_max,
                        ws_row_sum,
                        scale_log2e,
                        tmem_addr + UInt32(config.TMEM_O0),
                        ws_stage0,
                        ws_maxsum0,
                        ws_o,
                    )
                    if warp_group_idx == UInt32(0):
                        fa4_tma_store_o_smem(
                            warp_idx & 3,
                            warp_group_idx,
                            ws_o,
                            ragged_tma_store,
                            num_output_rows,
                            head_idx,
                            gmem_row + cta_q_offset + ws_o_row_off,
                        )
                    _ws_write_lse(m_cta_t1_ws, l_cta_t1_ws)
                # WG1 already hit `named_barrier[256](2)` and returned; WG0 hits
                # it here so the pair-WG sync resolves before TMEM dealloc.
                named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            else:
                # T>=2: hierarchical two-level combine (both WGs active). Wait
                # BOTH o_prod producers. The OWN wait gates this WG's Level-1
                # TMEM-O read (fenced below). The PEER wait is REQUIRED for a
                # separate reason: ws_stage0/ws_stage1 (carved from o_smem == the
                # BM=32 Q base, q_bytes=8192) spill far past Q into the LIVE KV
                # ring slots. WG0 is released by its OWN o_prod (the PEELED o0),
                # which the MMA commits BEFORE its epilogue reads V_o[0] (WG1's
                # odd-tile V, ring-wrapped into KV slots 0/1) to produce o1.
                # Without the peer wait, WG0's ws_stage write races that V_o[0]
                # read -> corrupt V -> inf o1 (the peer O flows through l2_stage
                # smem so there is still no peer TMEM read; this wait is purely
                # the SMEM/KV happens-before). The peer commit means the MMA has
                # drained the KV ring, so the staging overwrite is safe. Parity
                # phase (odd-T) mirrors the non-WS 1Q combine. (T==1 needs none:
                # the MMA fast-path returns after o0 with no epilogue KV read.)
                o_prod_mbar[warp_group_idx].wait(o_phase)
                tcgen05_fence_after()
                var ws_peer_wg = UInt32(1) - warp_group_idx
                var ws_peer_phase = o_phase ^ (total_iters_combined & UInt32(1))
                o_prod_mbar[ws_peer_wg].wait(ws_peer_phase)
                var ws_stage = (
                    ws_stage0 if warp_group_idx == UInt32(0) else ws_stage1
                )
                var ws_maxsum = (
                    ws_maxsum0 if warp_group_idx == UInt32(0) else ws_maxsum1
                )
                var o_band = Array[Float32, config.ov_depth // config.m_pack](
                    uninitialized=True
                )
                var m_wg, l_wg = fa4_ws_level1_combine[
                    config.m_pack, config.BM, config.ov_depth, use_fma=use_fma
                ](
                    thread_tile_row,
                    partition_g,
                    warp_group_idx,
                    row_max,
                    ws_row_sum,
                    scale_log2e,
                    tmem_addr
                    + UInt32(config.TMEM_O0)
                    + warp_group_idx * UInt32(config.TMEM_O1 - config.TMEM_O0),
                    ws_stage,
                    ws_maxsum,
                    o_band,
                )
                comptime if splitk_combine_active:
                    # T>=2 cross-CTA split-K: Level 2 emits the NORMALIZED
                    # per-CTA band to regs (WG0; + (M_cta, L_cta)), then the
                    # third-level reduce-scatter combines across the P CTAs and
                    # TMA-stores this CTA's OWN depth band. `ws_l2_stage` is reused
                    # for the cross-CTA staging (each warp reads then rewrites its
                    # own band slot -- program-order safe); `ws_maxsum1` (WG1's
                    # dead Level-1 maxsum, ordered by Level 2's barrier 3) holds
                    # the per-CTA (M,L).
                    var o_band_norm = Array[
                        Float32,
                        config.ov_depth // config.m_pack,
                    ](uninitialized=True)
                    var m_cta, l_cta = fa4_ws_level2_combine_partial[
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        m_wg,
                        l_wg,
                        scale_log2e,
                        o_band,
                        ws_l2_stage,
                        ws_l2_maxsum,
                        o_band_norm,
                    )
                    fa4_ws_splitk_reduce_scatter_write[
                        config,
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        P=config.splitk_partitions,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        UInt32(config.splitk_partitions),
                        splitk_partition_idx(UInt32(config.splitk_partitions)),
                        m_cta,
                        l_cta,
                        scale_log2e,
                        o_band_norm,
                        ws_l2_stage,
                        ws_maxsum1,
                        ws_o,
                        smem.misc_mbars().publish_mbar(),
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset,
                    )
                else:
                    # Plain single-CTA two-level WS combine (also the workspace /
                    # unfused split-K egress: `ws_o_row_off` redirects the O TMA
                    # into this partition's `o_partial` slice and `_ws_write_lse`
                    # writes the per-row LSE; both fold to the non-split store
                    # when `workspace_split` is off). Level 2 returns the
                    # per-CTA (M_cta, L_cta) on WG0 (WG1's value is unused --
                    # `_ws_write_lse` is WG0-only).
                    var m_cta_ws, l_cta_ws = fa4_ws_level2_reduce_scatter_write[
                        config.m_pack,
                        config.BM,
                        config.ov_depth,
                        use_fma=use_fma,
                    ](
                        thread_tile_row,
                        partition_g,
                        warp_group_idx,
                        m_wg,
                        l_wg,
                        scale_log2e,
                        o_band,
                        ws_l2_stage,
                        ws_l2_maxsum,
                        ws_o,
                    )
                    if warp_group_idx == UInt32(0):
                        fa4_tma_store_o_smem(
                            warp_idx & 3,
                            warp_group_idx,
                            ws_o,
                            ragged_tma_store,
                            num_output_rows,
                            head_idx,
                            gmem_row + cta_q_offset + ws_o_row_off,
                        )
                    _ws_write_lse(m_cta_ws, l_cta_ws)
                named_barrier[Int32(2 * WARPGROUP_SIZE)](4)
            comptime if not config.pair_cta and config.splitk_partitions == 1:
                if warp_idx == 0:
                    tcgen05_release_allocation_lock[Int32(cta_group)]()
                    tcgen05_dealloc[Int32(cta_group)](
                        tmem_addr, UInt32(config.sm100_tmem_cols)
                    )
            return

        if config.single_o or total_iters_combined == UInt32(1):
            # (WS handled above and returned; this path is non-WS or dead-for-WS.)
            # T==1 fast path AND the single-O all-T path: skip the
            # LSE-exchange entirely and reuse the 2Q row-scale + stmatrix
            # + TMA helper directly. No peer partial to combine; no per-WG
            # smem/gmem-row offsets. `BM // config.num_q` is the helper's
            # expected row count and numerically equals config.BM (= 128)
            # in 1Q. For single-O, WG0 has accumulated ALL K-tiles' P@V
            # into the single O0 and holds the full `row_sum`, so this
            # writer produces the complete output; WG1 already returned.
            var row_sum_total = row_sum.reduce_add()
            var inv_row_sum = recip(row_sum_total)
            var o_tile = TMemTile[
                accum_dtype, BM // config.num_q, padded_ov_depth
            ](tmem_addr + UInt32(config.TMEM_O0))
            # Only o0 is produced (MMA skipped the o1 commit at T==1).
            o_prod_mbar[0].wait(o_phase)
            tcgen05_fence_after()
            comptime if splitk_combine_active:
                # T==1 split-K combine (reduce-scatter): WG1 returned early, so
                # WG0 alone stages its single normalized `O_cta` (= o0*inv_row_sum)
                # over ALL depth columns + this partition's (row_max, row_sum),
                # publishes via the cluster mbar, then EVERY partition's WG0 reads
                # its OWN depth band from all peers and writes it. (The non-empty
                # partition here carries the data; trailing empty partitions take
                # the empty-partition path above and combine their bands from it.)
                var stage = smem.o_smem[.float32]()
                var maxsum = stage + (config.BM * padded_ov_depth)
                # o0 tile typed with `config.BM` exactly — `o_tile`'s
                # `BM // config.num_q` does not fold to `config.BM` at parse
                # time, so it can't convert to stage_partial's param type.
                var own_o_t1 = TMemTile[
                    accum_dtype, config.BM, padded_ov_depth
                ](tmem_addr + UInt32(config.TMEM_O0))
                # Publish this partition's (row_max, row_sum). WG0-only at
                # T==1, so `row` (in-WG tid) is the full softmax tid; partition
                # 0 already folded the sink into row_max/row_sum above.
                maxsum[row * 2] = row_max
                maxsum[row * 2 + 1] = row_sum_total
                # Single-source O (o0*inv_row_sum): staged (own band -> regs,
                # peers -> smem), published, then combined over this partition's
                # own band.
                fa4_splitk_reduce_scatter_write[
                    config,
                    P=config.splitk_partitions,
                    use_fma=use_fma,
                    single_source=True,
                ](
                    row,
                    warp_idx & 3,
                    warp_group_idx,
                    splitk_partition_idx(UInt32(config.splitk_partitions)),
                    row_max,
                    row_sum_total,
                    scale_log2e,
                    inv_row_sum,
                    Float32(0),
                    stage,
                    maxsum,
                    o_smem,
                    smem.misc_mbars().publish_mbar(),
                    own_o_t1,
                    own_o_t1,
                    ragged_tma_store,
                    num_output_rows,
                    head_idx,
                    gmem_row + cta_q_offset,
                )
            else:
                if is_writer:
                    fa4_scale_write_output[config](
                        row,
                        warp_idx & 3,
                        UInt32(0),
                        inv_row_sum,
                        o_smem,
                        o_tile,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset + ws_o_row_off,
                    )
                # Workspace split-K: emit the fused per-row LSE alongside the
                # per-partition O (T==1: WG0 owns the full row_max/row_sum).
                _ws_write_lse(row_max, row_sum_total)
            # WG1 already participated in `named_barrier[2*WG](2)` and
            # returned; WG0 must hit it here so the pair-WG sync resolves
            # before TMEM dealloc. Mirrors the unconditional sync below.
            # When the generic single-O path drops WG1 entirely
            # (`single_softmax_wg`), there is no peer WG to rendezvous with, so
            # this 256-thread barrier would hang with only WG0's 128 arrivals.
            # The correction / MMA -> dealloc ordering is carried by the
            # o-producer mbars (unchanged), not this softmax-only barrier.
            comptime if not single_softmax_wg:
                named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            # Pair-CTA and 1Q split-K defer dealloc to the kernel terminal
            # (after the cluster_sync); only plain single-CTA deallocs inline.
            comptime if not config.pair_cta and config.splitk_partitions == 1:
                if warp_idx == 0:
                    tcgen05_release_allocation_lock[Int32(cta_group)]()
                    tcgen05_dealloc[Int32(cta_group)](
                        tmem_addr, UInt32(config.sm100_tmem_cols)
                    )
            return

        # (WS T>=2 two-level combine handled above and returned; the non-WS
        # LSE-exchange below is dead-but-type-checks for a WS instantiation.)

        # 1Q: LSE-combine both WGs' TMEM_O fragments into the shared
        # o_smem in depth-column slices, then both WGs TMA-store
        # disjoint column ranges to gmem. Both WGs cover the same Q
        # rows; no per-WG row offset on the write side.

        # 1. WG-local LSE reduce.
        var row_sum_total = row_sum.reduce_add()

        # 2. Wait on OWN pipeline_o producer. After this, MMA1 has finished
        # its last V·P, so the last P in own's s_tmem has been consumed and
        # the slot is safe to repurpose for the cross-WG LSE exchange below.
        # MMA1 always runs and the TMEM reuse requires this wait.
        o_prod_mbar[warp_group_idx].wait(o_phase)
        tcgen05_fence_after()

        # 3. LSE exchange through the (now-dead) s_tmem slot. Each WG writes
        # (row_max, row_sum_total) into the first two TMEM columns of its
        # s_tmem slot; the peer reads those two columns from the other WG's
        # slot. Replaces an earlier smem-aliased exchange buffer that had
        # to overlay the K region and could collide with the load warp's K
        # TMA writes.
        # TMEM layout (per Q row r):
        #     col TMEM_S0+0     = WG0 row_max
        #     col TMEM_S0+1     = WG0 row_sum_total
        #     col TMEM_S0+BN+0  = WG1 row_max
        #     col TMEM_S0+BN+1  = WG1 row_sum_total
        var own_lse: Array[Scalar[accum_dtype], 2] = [
            row_max,
            row_sum_total,
        ]
        TMemTile[accum_dtype, BM, 2](s_tmem).store_async(own_lse)
        tcgen05_store_wait()
        tcgen05_fence_before()
        named_barrier[Int32(2 * WARPGROUP_SIZE)](5)
        tcgen05_fence_after()

        # 4. Read peer's slice from the peer WG's s_tmem.
        var peer_wg = UInt32(1) - warp_group_idx
        var peer_s_tmem: UInt32 = (tmem_addr + UInt32(config.TMEM_S0)) + UInt32(
            config.BN
        ) * peer_wg
        var peer_lse = TMemTile[accum_dtype, BM, 2](peer_s_tmem).load_async()
        var peer_max = peer_lse[0]
        var peer_sum = peer_lse[1]

        var global_max = max(row_max, peer_max)
        # Match the per-WG online softmax convention: when `use_fma`,
        # `row_max` is tracked in raw (unscaled) score units and the
        # inner-loop correction diff is formed in the `scale_log2e` (log2)
        # domain before `exp2` (the running max is carried scaled as
        # `neg_max_scaled = -row_max * scale_log2e`). The LSE combine must
        # apply the same conversion, otherwise the cross-WG weights are
        # `exp2(raw_diff)` instead of `exp2(raw_diff * scale_log2e)` and the
        # 1Q output drifts ~1 ULP whenever the two WGs' raw maxes differ.
        # Without this scaling the bug is masked when K is constant (raw maxes
        # equal across WGs) or V is constant (per-WG O ∝ row_sum so
        # the wrong weights cancel through global_sum normalization).
        # Vectorize the (local, peer) pair: one f32x2 sub / mul / exp2 in
        # place of two scalars each (matching `store_exp`'s f32x2 `exp2`),
        # and fuse the denominator's two products into a single FMA. The
        # `*= scale_log2e` here is `mul_ftz` (ftz), matching the inner loop's
        # log2-domain correction diff.
        var diffs: f32x2 = f32x2(row_max, peer_max) - f32x2(global_max)
        comptime if use_fma:
            diffs = mul_ftz(diffs, f32x2(scale_log2e))
        var scales: f32x2 = exp2(diffs)  # (scale_local, scale_peer)
        # global_sum = row_sum_total*scale_local + peer_sum*scale_peer, fused.
        var global_sum = peer_sum.fma(scales[1], row_sum_total * scales[0])
        var final_scales: f32x2 = scales * recip(global_sum)
        var final_scale_local = final_scales[0]
        var final_scale_peer = final_scales[1]

        # 5. Wait on PEER pipeline_o producer so peer's TMEM_O is safe to
        # read. Per-pipeline iter counts differ by
        # `total_iters_combined & 1` for odd combined-T, so peer's phase
        # XORs in that bit. (Own's producer was already waited on above
        # before the LSE exchange.)
        var peer_phase = o_phase ^ (total_iters_combined & UInt32(1))
        o_prod_mbar[peer_wg].wait(peer_phase)
        tcgen05_fence_after()

        # 5. Build own + peer TMEM tiles at full-BM extent.
        var own_o_tile = TMemTile[accum_dtype, BM, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + warp_group_idx * UInt32(config.TMEM_O1 - config.TMEM_O0)
        )
        var peer_o_tile = TMemTile[accum_dtype, BM, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + peer_wg * UInt32(config.TMEM_O1 - config.TMEM_O0)
        )

        # 6. Per-WG comptime j-range specialization for the helper.
        # Ceil/floor split: WG0 takes ceil(iters/2) blocks starting
        # at j=0, WG1 takes floor(iters/2) starting at j=ceil(iters/2).
        # Under SWIZZLE_NONE the block is small (o_sw_K = 8 bf16), so
        # iters = ov_depth/8 >= 8 for every supported head size and both
        # WGs always participate (iters_per_wg1 >= 4 > 0). When batched,
        # the store descriptor's box is the half-depth ceil(iters/2): WG0
        # fills it exactly and WG1, for odd iters (e.g. depth=72 -> 9),
        # overhangs the last block, which the TMA masks off. The
        # `iters_per_wg1 > 0` guard below is now always true but kept for
        # safety.
        #
        # The block size must come from the OUTPUT store's swizzle, not
        # `config.swizzle_mode`: fa4_lse_combine_write infers its
        # `output_swizzle_mode` from `ragged_tma_store`, and the two
        # differ for FP8-QKV MLA (64B QKV swizzle, 128B BF16 output
        # store). For MHA the store is built with `config.swizzle_mode`,
        # so this folds to the previous expression.
        comptime swizzle_granularity = (
            type_of(ragged_tma_store).swizzle_mode.bytes()
            // size_of[output_type]()
        )
        # Block count is the output column count aligned to the OUTPUT swizzle
        # granularity (SWIZZLE_NONE -> 8 bf16), NOT the QKV swizzle's
        # `padded_ov_depth` (aligned to 64). ov_depth is already a multiple of
        # the output granularity for every supported head size.
        comptime iters_total = (
            align_up(config.ov_depth, swizzle_granularity)
            // swizzle_granularity
        )
        comptime iters_per_wg0 = (iters_total + 1) // 2
        comptime iters_per_wg1 = iters_total // 2
        # In 1Q both WGs write the same Q rows; no per-WG gmem-row
        # offset (the depth column j drives the gmem position). Workspace
        # split-K shifts the ragged row into this partition's `o_partial` slice
        # (`ws_o_row_off` folds to 0 otherwise).
        var out_row_idx = gmem_row + cta_q_offset + ws_o_row_off
        # Workspace split-K: emit the fused per-row LSE (WG0 only; global_max /
        # global_sum are the per-partition stats after the cross-WG exchange).
        _ws_write_lse(global_max, global_sum)
        comptime if splitk_combine_active:
            # Reduce-scatter: every partition stages its FULL normalized O_cta
            # (f32) + per-row (max,sum) into the dead Q+KV span and publishes via
            # the round-1 cluster mbarrier; then (Pass 2) every partition's WG0
            # reads its OWN depth band from all P peers and TMA-stores that band
            # (see fa4_splitk_reduce_scatter_write below). Zero per-CTA smem
            # growth (reuses dead Q+KV) — growing it hangs P>=4.
            comptime assert (
                config.BM * config.padded_ov_depth * 4 + 2 * 256 * 4
                <= type_of(smem).q_bytes + type_of(smem).kv_bytes
            ), "split-K f32 O+(max,sum) staging must fit in the dead Q+KV span"
            var stage = smem.o_smem[.float32]()
            var maxsum = stage + (config.BM * config.padded_ov_depth)

            # --- Publish (max,sum) for both WGs, then stage + reduce-scatter. ---
            tid = warp_group_idx * UInt32(WARPGROUP_SIZE) + row
            maxsum[tid * 2] = global_max
            maxsum[tid * 2 + 1] = global_sum
            # Reduce-scatter (both WGs call; WG0 stages + combines, WG1 only
            # joins the phase-0 wait inside the helper): WG0 stages this CTA's
            # FULL O_cta -- its OWN depth band into the `o_final` registers, every
            # other block to the `stage` smem (peers read those for THEIR bands)
            # -- combining its own o0 with the peer o1 it already waited on
            # (single_source=False). Then every partition's WG0 reads its OWN band
            # from all P partitions and writes the combined bf16 into its OWN-band
            # f32 stage slice (no peer reads it), so NO round-2 cluster barrier is
            # needed -- the kernel's terminal `cluster_sync()` keeps the peer-read
            # PEER bands alive until every partition finishes reading. Band
            # ownership in Pass 2 is across PARTITIONS, not WGs, so WG1 stages
            # nothing (keeping the per-partition combine WG0-only avoids the known
            # WG1-owns-a-band deadlock).
            # MUST-VERIFY-ON-B200: no `fence_async_view_proxy()` between the
            # plain-`st.shared` stage dump and the publish arrive (both now inside
            # the helper) — proven (test_cluster_publish_smoke) unnecessary for
            # the peer `ld.shared::cluster` reads. If P>=4 hangs or reads stale O,
            # the first fix to try is a fence before the arrive in
            # `fa4_splitk_reduce_scatter_write`.
            fa4_splitk_reduce_scatter_write[
                config,
                P=config.splitk_partitions,
                use_fma=use_fma,
            ](
                row,
                warp_idx & 3,
                warp_group_idx,
                splitk_partition_idx(UInt32(config.splitk_partitions)),
                global_max,
                global_sum,
                scale_log2e,
                final_scale_local,
                final_scale_peer,
                stage,
                maxsum,
                o_smem,
                smem.misc_mbars().publish_mbar(),
                own_o_tile,
                peer_o_tile,
                ragged_tma_store,
                num_output_rows,
                head_idx,
                out_row_idx,
            )
        else:
            if warp_group_idx == UInt32(0):
                if is_writer:
                    fa4_lse_combine_write[
                        config,
                        wg_j_offset=0,
                        iters_per_wg=iters_per_wg0,
                    ](
                        row,
                        warp_idx & 3,
                        warp_group_idx,
                        final_scale_local,
                        final_scale_peer,
                        o_smem,
                        own_o_tile,
                        peer_o_tile,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        out_row_idx,
                    )
            else:
                comptime if iters_per_wg1 > 0:
                    if is_writer:
                        fa4_lse_combine_write[
                            config,
                            wg_j_offset=iters_per_wg0,
                            iters_per_wg=iters_per_wg1,
                        ](
                            row,
                            warp_idx & 3,
                            warp_group_idx,
                            final_scale_local,
                            final_scale_peer,
                            o_smem,
                            own_o_tile,
                            peer_o_tile,
                            ragged_tma_store,
                            num_output_rows,
                            head_idx,
                            out_row_idx,
                        )
    named_barrier[Int32(2 * WARPGROUP_SIZE)](4)
    # Pair-CTA and 1Q split-K defer dealloc to the kernel after cluster_sync so
    # that no CTA exits while a peer's cluster-scoped access is in flight
    # (stmatrix for pair-CTA; the DSMEM `(max,sum)` peer reads for split-K).
    # Only plain single-CTA deallocs inline here.
    comptime if not config.pair_cta and config.splitk_partitions == 1:
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Int32(cta_group)]()
            tcgen05_dealloc[Int32(cta_group)](
                tmem_addr, UInt32(config.sm100_tmem_cols)
            )
