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
"""MMA warp logic for FA4 (SM100 Flash Attention)."""

from std.math import align_up, ceildiv
from std.sys import size_of, get_defined_bool
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    ExplicitTMEMCrossP,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    blasst_vote_unanimous,
    elect,
    elect_mma_arrive,
    SM100TensorAccumulator,
    KConsumerPipeline,
    VConsumerPipeline,
    StagedPipeline,
    MBarType,
    splitk_window,
    splitk_partition_idx,
    splitk_num_partitions,
)
from nn.attention.mha_mask import MHAMask
from linalg.arch.sm100.mma import smem_descriptor
from layout import IntTuple
from layout.tensor_core_async import tile_layout_k_major
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from .smem import SM100AttentionSMem


@always_inline
def fa4_mma[
    MaskType: MHAMask,
    //,
    config: FA4Config,
    *,
    page_size: Int,
    # Workspace (traditional/unfused) split-K: window the KV by a RUNTIME
    # partition count even at `config.splitk_partitions == 1`. Defaulted so
    # every non-workspace caller is byte-identical.
    workspace_split: Bool = False,
    # Effective cross-stage-P switch, computed once at the kernel level where
    # config, MaskType and the store shape are all visible. Defaults True so
    # the config-level gate alone decides for callers that do not thread it;
    # the MHA kernel passes its own narrower predicate.
    crossp_effective: Bool = True,
](
    smem: SM100AttentionSMem[config],
    tmem_addr: UInt32,
    seq_id: UInt32,
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
    ws_num_partitions: UInt32 = 1,
):
    """Executes the FA4 MMA warp loop for SM100 Flash Attention.

    Computes Q@K' into score TMEM (S0/S1) and P@V into output TMEM (O0/O1)
    across the KV tile sequence, coordinating with the load and softmax
    warps via barrier pipelines. Supports fused-KV and split-KV modes,
    single-Q and two-Q tile configurations, and partial-K contraction for
    paged sub-tiles (`page_size` zero disables paging; sub-BN enables
    partial-K).

    Args:
        smem: Shared memory allocator holding Q, K, V tiles and barriers.
        tmem_addr: Base TMEM address for this CTA's S/O accumulators.
        seq_id: Sequence index for the current query batch.
        score_row: Row offset of the query tile within the sequence.
        num_keys: Number of valid key columns to attend to.
        mask: Attention mask controlling tile iteration and masking bounds.
        ws_num_partitions: Runtime split-K partition count that windows the
            KV range when `workspace_split` is `True`; `1` disables windowing.
    """
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime num_q: Int = config.num_q
    comptime BM_mask: Int = config.PairBM_eff()
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime cta_group: Int = config.cta_group()

    # Mb: WS packed P@V. Layout-G (MMA_M=32, m_pack=4) DEPTH-SCATTERS: packs
    # `m_pack` key-quarters x `depth_tile` output cols into one MMA_N=256 tile,
    # so `v_subslots` depth-tiled MMAs each write a DISJOINT output column range
    # at C stride `depth_tile`. Layout-E (MMA_M=64, m_pack=2) REDUCTION-SPLITS:
    # V is split along the KEY axis into `num_qk_stages` chunks that all
    # accumulate into the SAME `o_tmem`; `v_subslots` is the reduction-chunk trip
    # count. Non-WS (m_pack==1) folds to today's literal, byte-identical.
    comptime is_reduction_split = config.use_ws and config.m_pack == 2
    comptime depth_tile = (
        256 // config.m_pack
    ) if config.use_ws else config.padded_ov_depth
    # V ring SUB-SLOTS one V tile occupies -- how many ring positions `_pv_ws`
    # walks, NOT a depth-tile count. They coincide on Layout-G (so "depth/64"
    # leaked into the ring span model), not on Layout-E -- one depth tile,
    # several sub-slots. The depth-scatter arm reads `num_o_tiles()`
    # (`ceildiv(o_phys_cols(), pv_mma_n() // m_pack)`); Layout-E keeps
    # `num_qk_stages`, a different quantity.
    comptime v_subslots = (
        config.num_qk_stages if is_reduction_split else config.num_o_tiles()
    )
    # Via the accessor: a local copy drifts to `padded_ov_depth`, which
    # disagrees on BOTH WS layouts.
    comptime pv_mma_n = config.pv_mma_n()
    # Layout-E's reduction-chunk width: a partition's `BN // m_pack` keys split
    # `num_qk_stages` ways. Exactly the V TMA box's KEY-row count.
    comptime pv_bk_chunk = config.v_e_chunk_rows()
    # Keys contracted by ONE P@V MMA = `UMMA1Type`'s BK. `m_pack` is the wrong
    # axis to derive this from -- shared-key cuts V along KEYS, single
    # partition -- so a `BN // m_pack` ladder mis-sizes the MMA's B operand on
    # any cell where `pv_mma_n() != m_pack * BK0`.
    comptime pv_bk = config.pv_key_chunk()
    # Layout-E ONLY: P/S occupies `BN//m_pack` columns but `UMMA1Type`'s BK
    # covers only ONE reduction chunk. `stage_idx` (p_stage) offsets the A (P)
    # operand WITHIN a `pv_bk_chunk`-wide window at a manually-passed base
    # (`attention_utils.mojo`'s `a_tmem_offset`), NOT by reduction chunk --
    # without also advancing the BASE per `r`, every chunk silently re-reads
    # chunk 0's P columns. Dead for Layout-G (`s_tmem` never advances per
    # `v_stage` there).
    comptime s_tmem_chunk_words = (pv_bk * size_of[config.qkv_dtype]()) // 4
    # Shared-key V walk: key chunks outer, `num_o_tiles()` depth tiles inner.
    # `v_sk_blocks_per_chunk` maps the tile-wide `valid_k_mmas` (call-site) to
    # per-chunk counts one MMA's B operand can express (= `UMMA1Type.MMA_K`
    # units, since the MMA's BK IS `pv_bk`).
    comptime v_sk_chunks = config.pv_reduction_chunks()
    comptime v_sk_blocks_per_chunk = pv_bk // UMMA1Type.MMA_K
    # Byte stride between adjacent V depth-tile regions in SMEM (one packed
    # `[pv_mma_n, pv_bk]` mn-major region). Only referenced by `_pv`'s
    # per-`v_stage` offset, and `_pv` runs only where `v_subslots == 1`
    # (non-WS), so that offset is always zero. = 32768 @depth128.
    comptime v_region_bytes = pv_mma_n * pv_bk * size_of[config.qkv_dtype]()
    comptime if config.use_ws and not is_reduction_split:
        comptime assert (
            config.padded_ov_depth % depth_tile == 0
        ), "Mb: padded_ov_depth must be a multiple of depth_tile"

    var mbars = smem.misc_mbars()
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if config.qkv_dtype.is_float8() else UMMAKind.KIND_F16
    )

    # MMA types
    comptime UMMA0Type = SM100TensorAccumulator[
        config.qkv_dtype,
        accum_type,
        MMA_M=config.MMA_M,
        MMA_N=BN,
        BK=align_up(config.qk_depth, config.MMA_K),
        a_tmem=False,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        cta_group=cta_group,
        num_stages=num_qk_stages,
        mma_kind=mma_kind,
        b_page_dense=config.k_row_major(),
    ]
    # The P@V accumulator, in the TWO forms its A operand can take -- identical
    # but for `a_tmem` and the A swizzle. Two aliases, not one with
    # `a_tmem=not config.ws_shared_key`: `AInput` is a conditional type and
    # Mojo folds it only on a LITERAL, so a config expression leaves it
    # unresolved and rejects both arms.
    comptime UMMA1Type = SM100TensorAccumulator[
        config.qkv_dtype,
        accum_type,
        MMA_M=config.MMA_M,
        MMA_N=pv_mma_n,
        BK=pv_bk,
        # Key-split layouts: each warp's packed TMEM quarter is the A operand
        # of its OWN MMA, which is exactly what the `.ws` TS path gives (no
        # cross-subpartition mixing of A).
        a_tmem=True,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        cta_group=cta_group,
        num_stages=num_pv_stages,
        mma_kind=mma_kind,
        b_page_dense=config.v_row_major(),
        # Layout-E's reduction-split P@V wants an EVEN 2-then-2 split of each
        # reduction chunk's own BK (so the P sub-stage chunks align with the
        # V reduction chunks); Layout-G keeps the default 3-then-1 (unchanged).
        allow_3_then_1_split=not is_reduction_split,
    ]
    # Shared-key: ONE MMA whose A operand is the full `BM x BN` P tile spanning
    # all four quarters, which no packed per-warp TMEM operand can express.
    # Producer: `store_p_quadrant` -> `smem.p_smem()`.
    #
    # `swizzle_a` is SWIZZLE_NONE and that is NOT a simplification:
    # `store_p_quadrant` is hard-specialized to it (`sw_K = 16//size_of[p_type]()`),
    # so P's layout is independent of `config.swizzle_mode` (64B/128B for Q/K/V) --
    # building the descriptor with the config's mode instead scrambles warps 1-3's
    # P into a plausible-but-wrong result rather than crashing.
    #
    # `cta_group` is pinned to 1 rather than inherited: no FA4 config is
    # pair-CTA (`use_ws` requires `not pair_cta`), and the literal makes the
    # declaration inert by construction. The shared-key arm's
    # `comptime assert cta_group == 1` pins the MMA agreement.
    comptime UMMA1SKType = SM100TensorAccumulator[
        config.qkv_dtype,
        accum_type,
        MMA_M=config.MMA_M,
        MMA_N=pv_mma_n,
        BK=pv_bk,
        a_tmem=False,
        swizzle_a=TensorMapSwizzle.SWIZZLE_NONE,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        cta_group=1,
        num_stages=num_pv_stages,
        mma_kind=mma_kind,
        b_page_dense=config.v_row_major(),
        allow_3_then_1_split=not is_reduction_split,
    ]

    # Runtime-k partial-page gate. Only the last KV tile can be partially
    # loaded (paged sub-tiles, page_size < BN), and skipping its unloaded V
    # tail in P@V avoids reading uninitialized SMEM (`0 * NaN = NaN`).
    # supported() guarantees page_size % MMA_K == 0 here, so the loaded
    # boundary is MMA_K-aligned and the cut is exact.
    comptime PARTIAL_K = page_size > 0 and page_size < BN

    # Whether the LAST KV tile's P@V contraction is CUT to the blocks that
    # actually hold keys. `PARTIAL_K` is NOT that question: it asks whether a
    # tile can be partially LOADED (page geometry), while the cut is needed
    # whenever the last tile is SHORT (`num_keys % BN != 0`, no page-size term).
    # At `page_size >= BN` they diverge: `PARTIAL_K` goes False and the
    # contraction runs over the whole `BN` rows -- past `align_up(num_keys,
    # MMA_K)`, into V rows that belong to no sequence. P is +0.0 there, so
    # `0 * garbage`: harmless while the garbage is finite, NaN the moment it
    # is not.
    #
    # Under shared-key the cut is therefore unconditional. Producer and
    # consumer must cut the SAME key chunks or the ring desynchronizes; the
    # identity that makes them agree,
    # `ceildiv(ceildiv(a, m), n) == ceildiv(a, m * n)`, is page-size-free.
    # Key-split and non-WS keep `PARTIAL_K`.
    comptime CUT_LAST_TILE = PARTIAL_K or config.ws_shared_key

    # `tmem_addr` passed in by register (read once post-barrier in the kernel
    # prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var q_smem = smem.q_smem()

    var s0_tmem = tmem_addr + UInt32(config.TMEM_S0)
    var s1_tmem = tmem_addr + UInt32(config.TMEM_S1)
    var o0_tmem = tmem_addr + UInt32(config.TMEM_O0)
    var o1_tmem = tmem_addr + UInt32(config.TMEM_O1)

    # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
    var pipeline_s0 = mbars.producer_s0()
    var pipeline_s1 = mbars.producer_s1()
    # Keep consumer pointers for acquire operations (shared phase tracking)
    var consumer_s0 = pipeline_s0.consumer_mbar_base
    var consumer_s1 = pipeline_s1.consumer_mbar_base

    # O pipelines (producer side only; consumer wait is merged into S barriers)
    var pipeline_o0 = mbars.producer_o0()
    var pipeline_o1 = mbars.producer_o1()

    # Per-Q-tile element/byte counts.
    # 2Q: HalfBM * padded_qk_depth = 128 * d (one of two Q halves).
    # 1Q: BM * padded_qk_depth = 128 * d (the single full-BM Q tile).
    # Numerically identical because BM // num_q == 128 in both modes
    # (mirrors the load_warp.mojo Q-TMA invariance).
    comptime q0_size = (BM // num_q) * config.padded_qk_depth
    comptime q0_bytes = q0_size * size_of[config.qkv_dtype]()
    var q0 = smem_descriptor[
        BMN=config.BM // config.num_q,
        BK=config.BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](q_smem)
    var q1 = q0 + UInt32(q0_bytes)

    comptime q_sub_bytes = HalfBM * config.BK0 * size_of[config.qkv_dtype]()

    # WS Q depth-chunk stride: the full Q tile (q0_bytes) split into
    # `num_qk_stages` equal depth chunks. In the shared sub-tile ring the two
    # K depth-halves live in separate ring slots, so Q@K' is two
    # `mma[stage_idx=0]` and the second reads Q at `q0 + q_chunk_bytes`. Folds
    # to `q0_bytes` for num_qk_stages==1 (unused there). NOT `q_sub_bytes`
    # (=HalfBM*BK0*size, the 2Q row-half stride on the wrong axis).
    comptime q_chunk_bytes = q0_bytes // config.num_qk_stages

    var e = elect()

    # BLASST (arXiv 2512.12087): skip P@V (UMMA1) when all 4 of a WG's softmax
    # warps voted to skip; QK' (UMMA0) and the V load are never skipped. `and
    # not config.use_ws` since main's `_pv_ws` P@V path has no BLASST analog.
    comptime ENABLE_BLASST = get_defined_bool[
        "ENABLE_BLASST", False
    ]() and not config.use_ws
    var blasst_vote = smem.blasst_vote_smem()

    # The 1Q shared-KV schedule emits and consumes KV in CONSUMPTION order
    # (K_e[n] | V_o[n-1] | K_o[n] | V_e[n]) rather than block order, taking the
    # ring SPAN floor from `2*num_qk_stages` to `num_qk_stages`. Producer and
    # consumer rotate together (`load_warp.mojo`'s `_shared_1q` is the other
    # half) -- correct only as a pair. The tightened `ring_slots_needed()` bound
    # is UNSOUND on the unrotated schedule, so the rotation cannot be disabled
    # without hanging any config with `num_kv_stages` in [S, 2*S).

    # Cross-stage P (BLASST enabler, 2Q + non-WS only): the MMA runs the
    # QK0-first schedule (see the gated block below) and reads each stage's P
    # from the OTHER stage's S columns (TMEM_P{wg}). Shared-KV path only (the
    # headline config); split-KV is rejected at compile time.
    comptime CrossP = crossp_effective and type_of(mbars).CrossP_enabled
    # Fires only for an explicit `-D FA4_TMEM_CROSS_P=true` on a config outside
    # the support matrix. The ON-by-default path resolves `crossp_on()` to
    # False on those configs instead, so it never reaches here.
    comptime assert not (ExplicitTMEMCrossP and not config.crossp_on()), (
        "FA4_TMEM_CROSS_P=true was requested on a config that does not"
        " support cross-stage P. Supported: 2Q, non-warp-specialized,"
        " single-CTA, shared-KV, MHA (no rope split), full pages. See"
        " FA4Config.crossp_supported"
    )

    # Called after consumer_s[*].wait(phase), which orders the vote write
    # (softmax) before this read. Peel writes no vote -> reads 0 -> never skips.
    @__parameter
    @always_inline
    def blasst_should_skip(wg: UInt32, phase: UInt32) -> Bool:
        return blasst_vote_unanimous(blasst_vote, wg, phase)

    @__parameter
    @always_inline
    def _commit(
        mbar: UnsafePointer[address_space=.SHARED, ...],
    ):
        """Arrive at mbar: multicast for pair-CTA, local elect for single."""
        comptime if config.pair_cta:
            if e != 0:
                # cta_mask = 0b11 = (1 << cta_group) - 1 for cta_group=2:
                # arrive on both CTAs' instance of the barrier.
                mma_arrive_multicast[cta_group](mbar, UInt16(0x3))
        else:
            elect_mma_arrive(mbar, e)

    # Sliding-window / any non-zero `start_column` mask: the contraction runs
    # in the `[start_column, num_keys)` frame -- the producer iterates V tiles
    # from `kv_row = start_column` (load_warp.mojo) and softmax mirrors it. The
    # partial-K `valid_k_mmas` (`vkm`) below must use the SAME frame: the last
    # loaded tile sits at `start_column + (total_iters - 1) * BN`, so the loaded
    # MMA_K-block count is measured against `num_keys - start_column`, NOT
    # `num_keys`. Omitting `start_column` over-counts `vkm` by
    # `start_column // MMA_K`, so P@V runs blocks over V pages the producer never
    # loaded (`0 * stale-NaN = NaN`). For causal `start_column == 0`, so
    # `v_eff_keys == num_keys` and every `vkm` site below is bit-identical.
    #
    # 1Q split-K narrows the frame again: the `vkm` sites count tiles from the
    # partition's own window, so `v_eff_keys` is rebased by the window start
    # where split-K slices it below.
    var v_start_col: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_id, score_row
    )
    var v_eff_keys: UInt32 = num_keys - v_start_col

    # `valid_k_mmas` for a P@V whose V tile is only partially loaded. Which of
    # the two frame functions a site needs is decided by WHICH TILE it contracts
    # -- tile 0 vs. the last tile (see the docstrings).
    #
    # The frame is passed IN rather than captured. Both arms below rebase
    # `v_eff_keys` into their split-K window (`v_eff_keys -= _w[0] * BN`), and
    # these are defined above both rebases -- a nested `def` is not scoped to
    # the `comptime if` arm it sits in, so one definition per arm is a
    # redefinition error. An explicit argument makes the value read at the
    # CALL, the only thing that makes the hoist safe: a captured `v_eff_keys`
    # read pre-rebase would over-count by the window start and run P@V over V
    # pages the producer never loaded (`0 * stale-NaN = NaN`).
    @__parameter
    @always_inline
    def _vkm_tile0(v_eff: UInt32) -> UInt32:
        """Loaded `MMA_K` blocks of KV tile 0.

        Self-clamps to a full tile whenever `v_eff >= BN`, so a caller that
        contracts tile 0 and is only *sometimes* also contracting the last tile
        (`total_iters == 1`) needs no branch of its own.
        """
        return ceildiv(min(v_eff, UInt32(BN)), UInt32(UMMA1Type.MMA_K))

    @__parameter
    @always_inline
    def _vkm_final(v_eff: UInt32, total_iters: UInt32) -> UInt32:
        """Loaded `MMA_K` blocks of the LAST KV tile.

        The last tile starts at `(total_iters - 1) * BN` inside the frame, so
        the count is measured from there. Spelling such a site `_vkm_tile0`
        would over-count by exactly that offset.
        """
        return ceildiv(
            min(v_eff - (total_iters - UInt32(1)) * UInt32(BN), UInt32(BN)),
            UInt32(UMMA1Type.MMA_K),
        )

    # P@V contraction, written once. `partial` cuts the contraction at the
    # loaded-V boundary of a partially-loaded / short last KV tile; the plain
    # form is the hot interior path. NOT one body always calling
    # `mma_maybe_partial_k` with a saturated count: the two are numerically
    # identical, but the partial form still EMITS the `setp` + per-block
    # predicate, and the interior path runs `T - 1` times out of `T`.
    #
    # Non-WS only -- every warp-specialized site routes through `_pv_ws` via
    # `_body_1q`'s `_pv_kv`. Two things ride on that: `v_subslots == 1` here,
    # and `CUT_LAST_TILE == PARTIAL_K` for every caller of the two gated
    # wrappers below (`CUT_LAST_TILE` adds only `ws_shared_key`, which implies
    # `use_ws`). Assert it rather than comment it.
    @__parameter
    @always_inline
    def _pv[
        partial: Bool = False
    ](
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
        *,
        valid_k_mmas: UInt32 = 0,
        wg: UInt32 = 0,
    ):
        comptime assert not config.use_ws, (
            "_pv is the non-WS P@V; warp-specialized configs route through"
            " _pv_ws"
        )
        # BLASST (gated): the wait always runs -- it orders the vote read --
        # and only the P@V UMMA is skipped on a unanimous vote, so O keeps its
        # prior value. The waits hoist above the walk so the vote is read once
        # they retire; the early return keeps the walk below un-nested.
        comptime if ENABLE_BLASST:
            comptime for p_stage in range(num_pv_stages):
                _ = consumer_s[p_stage].wait(wait_phase)
            if blasst_should_skip(wg, wait_phase):
                return
        # V-smem tile outer; P-tmem stage inner. V-smem is the binding
        # constraint, so each V region is fully contracted before the next.
        # `valid_k_mmas` is the KEY count, identical across V regions, so it
        # passes through unchanged for every `v_stage`.
        comptime for v_stage in range(v_subslots):
            comptime for p_stage in range(num_pv_stages):
                # First V pass: wait each P sub-stage exactly once (P stays
                # live in TMEM across all V regions; preserves the 3/4 + 1/4
                # P-stage overlap). Hoisted above instead under BLASST.
                comptime if v_stage == 0 and not ENABLE_BLASST:
                    _ = consumer_s[p_stage].wait(wait_phase)
                comptime if partial:
                    UMMA1Type.mma_maybe_partial_k[stage_idx=p_stage](
                        s_tmem,
                        v + UInt32(v_stage * v_region_bytes),
                        o_tmem + UInt32(v_stage * depth_tile),
                        c_scale=c_scale,
                        elect=e,
                        valid_k_mmas=valid_k_mmas,
                    )
                else:
                    UMMA1Type.mma[stage_idx=p_stage](
                        s_tmem,
                        v + UInt32(v_stage * v_region_bytes),
                        o_tmem + UInt32(v_stage * depth_tile),
                        c_scale=c_scale,
                        elect=e,
                    )

    # The last-tile cut, gated ONCE instead of at every call site. The frame
    # (`v_eff`, `total_iters`) is passed in and the `_vkm_*` call sits INSIDE
    # the comptime-true arm, so an uncut build never evaluates it -- identity
    # by construction rather than by trusting DCE. WHICH of the two a site
    # wants is decided by which tile it contracts (see the `_vkm_*`
    # docstrings), so that choice stays at the call site; only the gate moves.
    #
    # `CUT_LAST_TILE`, not `PARTIAL_K`: the two are equal wherever these are
    # instantiated (see `_pv`'s assert), and one constant beats two.
    @__parameter
    @always_inline
    def _pv_tile0(
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
        v_eff: UInt32,
        *,
        wg: UInt32 = 0,
    ):
        comptime if CUT_LAST_TILE:
            _pv[partial=True](
                s_tmem,
                v,
                o_tmem,
                consumer_s,
                wait_phase,
                c_scale,
                valid_k_mmas=_vkm_tile0(v_eff),
                wg=wg,
            )
        else:
            _pv(s_tmem, v, o_tmem, consumer_s, wait_phase, c_scale, wg=wg)

    # `is_last` is RUNTIME with a constant default: the 2Q main-loop sites
    # contract the last tile only on their final trip and pass
    # `iter_count == 0`, taking a full contraction otherwise; epilogue sites
    # take the default and the ternary folds.
    @__parameter
    @always_inline
    def _pv_final(
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
        v_eff: UInt32,
        total_iters: UInt32,
        *,
        wg: UInt32 = 0,
        is_last: Bool = True,
    ):
        comptime if CUT_LAST_TILE:
            var vkm = _vkm_final(v_eff, total_iters) if is_last else UInt32(
                UMMA1Type.num_k_mmas
            )
            _pv[partial=True](
                s_tmem,
                v,
                o_tmem,
                consumer_s,
                wait_phase,
                c_scale,
                valid_k_mmas=vkm,
                wg=wg,
            )
        else:
            _pv(s_tmem, v, o_tmem, consumer_s, wait_phase, c_scale, wg=wg)

    # `num_q` selects the whole schedule, so the 1Q and 2Q bodies are separate
    # `def`s. `use_ws` implies `num_q == 1` AND `use_shared_kv`, so every
    # warp-specialized path lives in `_body_1q`'s shared-KV arm -- which makes
    # the co-rotation a local edit instead of one threaded through the 2Q
    # schedule.
    #
    # Both bodies must be at FUNCTION scope with DISTINCT names: a nested `def`
    # is not scoped to the `comptime if` arm containing it, so two same-named
    # defs in the two arms is a redefinition error.
    #
    # The bodies contain bare `return`s (degenerate-shape early exits). Those
    # now return from the body rather than from `fa4_mma`, equivalent ONLY
    # because the dispatch below is the last statement in the function -- keep
    # it last if anything is ever appended here.

    @__parameter
    @always_inline
    def _body_1q():
        # Cross-stage P is 2Q-only (`CrossP_enabled` requires `num_q == 2`
        # and `not use_ws`), so its QK0-first schedule and `rel_slot` release
        # cursor live only in `_body_2q`. Assert it rather than comment it: if
        # that ever stops holding, this body silently drops the schedule
        # CrossP depends on.
        comptime assert not CrossP, "CrossP requires num_q == 2"

        comptime if config.use_shared_kv:
            # ---- Shared KV mode ----
            # In shared mode, K_nope and V alternate in a single StagedPipeline.
            # Stages: K0, V0, K1, V1, ...

            var kv_smem = (
                smem.k_smem_base()
            )  # same as v_smem_base in shared mode
            # Per-slot byte stride = ONE ring slot. Derived from the SMEM struct
            # so it can NEVER drift from the reservation/producer. WS: 32768
            # (one 256x64 sub-tile); non-WS shared: 65536 (one full-depth tile).
            # The old `v_cols_per_cta()*BN*size` literal equalled the non-WS
            # value but strode 2x the WS reservation -> latent overrun; this
            # reconciles it.
            comptime kv_stage_bytes = SM100AttentionSMem[config].k_stage_bytes
            comptime if config.use_ws:
                comptime assert (
                    kv_stage_bytes
                    == config.BN
                    * (config.shared_kv_cols() // config.num_qk_stages)
                    * size_of[config.qkv_dtype]()
                ), "WS kv_stage_bytes must equal one 256x64 sub-tile"
                # Uniform shared-ring sub-tile premise (mirrors the
                # m_pack-selected invariant already enforced at
                # `attention.mojo`'s `supported()`): Layout-G (m_pack==4) keeps
                # its depth-scatter checks; Layout-E (m_pack==2) asserts its
                # reduction-split V sub-tile divides the partition's keys evenly
                # AND is byte-equal to one K/V ring slot.
                comptime if is_reduction_split:
                    comptime assert (
                        config.BN // config.m_pack
                    ) % config.num_qk_stages == 0, (
                        "WS Layout-E: (BN // m_pack) must divide evenly by"
                        " num_qk_stages"
                    )
                    comptime assert (
                        pv_mma_n * pv_bk_chunk * size_of[config.qkv_dtype]()
                        == kv_stage_bytes
                    ), (
                        "WS Layout-E: reduction sub-tile bytes must equal one"
                        " K/V ring slot"
                    )
                elif config.ws_shared_key:
                    # Shared-key's V positions are (key chunk, depth tile)
                    # pairs, so the position count is their PRODUCT -- neither
                    # `v_subslots` (depth tiles only) nor Layout-E's partition
                    # split. This is the consumer-side statement of
                    # producer/consumer agreement, and its failure mode is a
                    # hang, not a wrong answer: `_emit_v` emits exactly
                    # `num_qk_stages` positions per full V tile and this walk
                    # must consume exactly that many.
                    comptime assert (
                        config.v_ring_positions_per_tile()
                        == config.num_qk_stages
                    ), (
                        "shared-key V walk must occupy exactly the ring"
                        " positions the producer emits, or the CTA hangs"
                    )
                    comptime assert (
                        pv_mma_n * pv_bk * size_of[config.qkv_dtype]()
                        == kv_stage_bytes
                    ), (
                        "shared-key: one (key chunk, depth tile) pair must be"
                        " exactly one K/V ring slot"
                    )
                    comptime assert pv_bk % UMMA1Type.MMA_K == 0, (
                        "shared-key: a key chunk must be a whole number of"
                        " MMA_K blocks, or the per-chunk valid_k_mmas split is"
                        " not expressible"
                    )
                    # ...and NOT zero blocks. `x % 0 == 0` in Mojo (`Int` is
                    # `Scalar[DType.int]` and `SIMD.__mod__` zero-guards), so
                    # the assert above passes at `pv_bk == 0`. There the MMA
                    # emits zero instructions and `v_sk_chunks` is 0, so the
                    # whole walk is empty: O keeps whatever stale value TMEM
                    # held, with no fault and no build error.
                    comptime assert (
                        v_sk_blocks_per_chunk >= 1 and v_sk_chunks >= 1
                    ), (
                        "shared-key: degenerate key-chunk geometry -- the P@V"
                        " walk would emit no MMAs and leave O uninitialized"
                    )
                    # `v_sk_blocks_per_chunk` sizes TWO things now: the
                    # `chunk_vkm` denominator AND the block count one
                    # `UMMA1SKType` call emits over a chunk. Its own definition
                    # states this in prose; bind it, because a `pv_key_chunk()` change
                    # that left `MMA_K` alone would make the emitted
                    # contraction cover a different key span than `chunk_vkm`
                    # predicates -- the ring stays in step and the data is
                    # wrong.
                    comptime assert (
                        v_sk_blocks_per_chunk == UMMA1SKType.num_k_mmas
                    ), (
                        "shared-key: the per-chunk block count must be the"
                        " MMA's own num_k_mmas"
                    )
                    # ---- The SS A operand's preconditions --------------------
                    # `a_bmn` is `align_up(MMA_M // cta_group, 8)`, i.e. the
                    # MMA's own A row count -- while the descriptor and
                    # `store_p_quadrant`'s `p_tile_rows` both describe a
                    # `config.BM`-row physical tile. They coincide only via
                    # `m_pack * BM == WARPGROUP_SIZE` (`supported()`) and
                    # `MMA_M = 128 // m_pack`. Relax either and
                    # `tile_layout_k_major`'s k-chunk stride (`BM * sw_K`)
                    # stops matching the stride the MMA walks, so A-blocks 1..n
                    # read the wrong 16 B atoms: plausible-looking, wrong O.
                    comptime assert config.BM == config.MMA_M, (
                        "shared-key: the P SMEM tile height (config.BM) must"
                        " equal MMA_M, or the A descriptor and the MMA"
                        " disagree on the k-chunk stride"
                    )
                    # `padded_BK` is `align_up(BK, max(swizzle_a, swizzle_b) /
                    # dtsize)` on the SS path. It lands on `pv_bk` at every
                    # admitted cell (64 keys against a 64-element granularity,
                    # both dtypes); if it ever padded, `a_layout` would describe
                    # a WIDER tile than `store_p_quadrant` writes.
                    comptime assert UMMA1SKType.padded_BK == pv_bk, (
                        "shared-key: the SS A operand must not pad BK -- the P"
                        " tile is exactly BN keys wide"
                    )
                    # The B operand is spelled `pv_mma_n`, matching
                    # `kv_desc_v` exactly. That is the TS-quadrant carve-out in
                    # `SM100TensorAccumulator.b_bmn`; the SS rule is
                    # `MMA_N // cta_group`, and the two agree only at
                    # cta_group == 1. `use_ws` forces that, and `supported()`
                    # forces `not pair_cta` -- assert rather than re-spell, so
                    # the descriptor and the builder keep ONE tuple.
                    comptime assert cta_group == 1, (
                        "shared-key: the SS P@V assumes single-CTA; at"
                        " cta_group == 2 the B layout's chunk-crossing K"
                        " offsets double and read past the per-CTA tile"
                    )
                    # `use_3_then_1_split` conjoins `a_tmem`
                    # (`attention_utils.mojo`), so `UMMA1SKType` takes the even
                    # 2-then-2 P sub-stage k-split where `UMMA1Type` takes
                    # 3-then-1 over the same four k-blocks.
                    # That is INERT here and only here: `_pv_ws` hoists all
                    # `num_pv_stages` waits above the walk, so both spellings
                    # issue the same four k-blocks of a chunk before the
                    # position is released, and softmax releases every
                    # sub-stage at the tail. Asserted rather than left silent
                    # because the 3:1 split becomes load-bearing again the
                    # moment the waits are un-hoisted to overlap P production
                    # with P@V -- that work owns re-enabling it explicitly.
                    comptime assert not UMMA1SKType.use_3_then_1_split, (
                        "shared-key: the SS P@V takes the even P sub-stage"
                        " split; re-enabling 3-then-1 means un-hoisting the"
                        " consumer_s waits in the same change"
                    )
                    # The ring floor is EARNED here, not merely consumed by
                    # `supported()`. `_pv_ws`'s key-chunk walk releases each
                    # position the instant its MMAs are issued, so `R_V == 1`
                    # and `ring_slots_needed()` says 1. Hold a position across
                    # another position's MMAs -- restore a block release, widen
                    # a chunk past one slot, or hoist the `_commit` out of the
                    # `(chunk, depth tile)` body -- and the accessor is a lie
                    # that `supported()` will honour: bf16 d512 has 4 slots and
                    # cannot afford a whole-tile residency at ANY dicing, since
                    # one K tile is 8 * 32768 B against a 232448 B carveout.
                    # Assert here so that regression is a build error naming the
                    # walk, rather than a hang on the one cell that cannot
                    # absorb it.
                    comptime assert config.ring_slots_needed() == 1, (
                        "shared-key ring floor must be the per-position"
                        " residency (1); a larger value means this walk stopped"
                        " releasing each position as its MMAs issue"
                    )
                else:
                    comptime assert (
                        v_subslots == config.num_qk_stages
                    ), "WS uniform sub-tile: v_subslots == num_qk_stages"
                    comptime assert (
                        256 // config.m_pack
                    ) == config.BK0, "WS uniform sub-tile: depth_tile == BK0"

            # K descriptor: k_major for Q@K' (BK0=64 -> one depth-half
            # sub-tile).
            var kv_desc_k = smem_descriptor[
                BMN=config.k_rows_per_cta(),
                BK=config.BK0,
                swizzle_mode=config.swizzle_mode,
                is_k_major=True,
                page_dense=config.k_row_major(),
            ](kv_smem)
            # V descriptor: mn_major for P@V. WS uses the packed [pv_mma_n,
            # pv_bk] box -- Layout-G's proven one-depth-tile box (256x64) or
            # Layout-E's reduction-chunk box (256x64, same numbers @depth128,
            # different meaning); non-WS keeps the full [v_cols_per_cta, BN]
            # tile (byte-identical).
            comptime v_desc_bmn = (
                pv_mma_n if config.use_ws else config.v_cols_per_cta()
            )
            comptime v_desc_bk = pv_bk if config.use_ws else config.BN
            var kv_desc_v = smem_descriptor[
                BMN=v_desc_bmn,
                BK=v_desc_bk,
                swizzle_mode=config.swizzle_mode,
                is_k_major=False,
                page_dense=config.v_row_major(),
            ](kv_smem)

            comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
            var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

            # We peel the first iteration, as we want to wait on q1.
            # Peel consumes 2 K-tiles (K_e[0], K_o[0]) and 2 V-tiles
            # (V_e[0], V_o[0] held); the main loop decrements once at the
            # top (K_e consume) and once more past the break-check
            # (K_o consume). iter_count = total_iters - 2.
            # 1Q at total_iters == 1 takes the T==1 fast path below after the
            # Q @ K_e[0] staged MMA; the iter_count underflow at T == 1
            # (1u32 - 2u32 wraps) is never read.
            var total_iters_runtime: UInt32 = mask.total_iters[
                BM_mask, BN, page_size
            ](seq_id, score_row, num_keys)
            # Split-K (1Q): slice the combined tile count to this partition's
            # window. mma needs only the per-partition COUNT -- load_warp emits
            # the matching cb-offset tiles that this warp consumes in order.
            # `total_iters == last_masked_set_end` for check_mask==False masks
            # (mha_mask.mojo derives the split-K window identically across all
            # four warps), so all four warps derive the same window.
            comptime if config.splitk_partitions > 1 or workspace_split:
                var _np = splitk_num_partitions[config](ws_num_partitions)
                var _w = splitk_window(
                    total_iters_runtime,
                    _np,
                    splitk_partition_idx(_np),
                )
                total_iters_runtime = _w[1] - _w[0]
                # Empty partition (front-load trailing window, T <
                # num_partitions; or an M6 idle CTA): no tiles, so load_warp
                # produces no K0 and the peeled `consumer_wait()` below would
                # hang. Skip all MMA work -- the empty partition's softmax
                # stages a neutral identity, and the kernel terminal
                # `cluster_sync()` still runs after this return.
                if total_iters_runtime == 0:
                    return
                # Rebase into this partition's tile window (see the `vkm` note).
                v_eff_keys -= _w[0] * UInt32(BN)
            # All-masked row (valid_length 0): total_iters == 0, so the `- 2`
            # below underflows and the MMA warp spins, hanging the pipeline.
            # Split-K is guarded above; guard the non-split path here.
            if total_iters_runtime == 0:
                return
            var iter_count: UInt32 = total_iters_runtime - UInt32(2)

            # Release the KV slot at `release_idx`, advance to the next stage,
            # wait for it, and return its slot index. Bundles the
            # release/step/wait/capture idiom repeated across the 1Q path.
            # `consumer_mbar(idx)` with the current index is identical to the
            # no-arg `consumer_mbar()` (which forwards `state.index()`).
            @__parameter
            @always_inline
            def _advance_kv(release_idx: UInt32) -> UInt32:
                _commit(kv_pipeline.consumer_mbar(release_idx))
                kv_pipeline.state.step()
                kv_pipeline.consumer_wait()
                return kv_pipeline.state.index()

            # ---- WS sub-tile ring helpers (all fold to no-ops for non-WS) ----
            # `_qk_extra`: after the d=0 Q@K' mma at the first K sub-slot, do
            # the remaining num_qk_stages-1 depth-half sub-slots. Each is its
            # OWN ring slot (NOT contiguous), so we use two `mma[stage_idx=0]`
            # with manual K/Q bases and c_scale 0->1 -- NOT `mma[stage_idx=1]`,
            # which adds an internal +BK0 offset assuming one contiguous tile
            # (would read OOB). Empty (no-op) when num_qk_stages==1.
            @__parameter
            @always_inline
            def _qk_extra(q_base: MMASmemDescriptorPair, s_tmem: UInt32):
                comptime for d in range(1, config.num_qk_stages):
                    var kd_idx = _advance_kv(kv_pipeline.state.index())
                    UMMA0Type.mma[stage_idx=0](
                        q_base + UInt32(d * q_chunk_bytes),
                        kv_desc_k + UInt32(kv_stage_bytes) * kd_idx,
                        s_tmem,
                        elect=e,
                        c_scale=1,
                    )

            # The three V lifecycle helpers below all assume ONE fixed-size,
            # ring-CONTIGUOUS group of V sub-slots per call site: `v_subslots`
            # of them, starting at `v_idx0`. Every one of those assumptions is
            # false under shared-key, where the group is a RUNTIME count of live
            # key chunks crossed with the depth tiles -- so all three are
            # comptime-EMPTY there and `_pv_ws` owns the whole
            # wait/MMA/release lifecycle inline instead.
            #
            # This is not stylistic. `_v_release_rest` walks a fixed span from
            # the anchor, so under a skip it would arrive on the consumer
            # barrier of a slot the producer never filled. That frees the
            # producer to overwrite a slot the MMA warp has not yet read:
            # silent WAR corruption, NOT a hang, and therefore invisible to
            # every timeout-based rung of the bring-up ladder. Emptying them is
            # the guard.
            comptime v_sk_inline_lifecycle = config.ws_shared_key

            # `_v_wait_rest`: wait the remaining v_subslots-1 V ring
            # sub-slots (produced consecutively right after the first). Leaves
            # the ring state at the last of them. Empty when v_subslots==1.
            @__parameter
            @always_inline
            def _v_wait_rest():
                comptime if not v_sk_inline_lifecycle:
                    comptime for d in range(1, v_subslots):
                        kv_pipeline.state.step()
                        kv_pipeline.consumer_wait()

            # `_v_release_rest`: release the v_subslots-1 V sub-slots after
            # v_idx0 (ring-adjacent). Empty when v_subslots==1.
            @__parameter
            @always_inline
            def _v_release_rest(v_idx0: UInt32):
                comptime if not v_sk_inline_lifecycle:
                    comptime for d in range(1, v_subslots):
                        _commit(
                            kv_pipeline.consumer_mbar(
                                (v_idx0 + UInt32(d))
                                % UInt32(config.num_kv_stages)
                            )
                        )

            # `_v_release_first`: release the V group's FIRST sub-slot -- the
            # one the call site waited before entering `_pv_ws`. Its five call
            # sites used to spell this as a bare `_commit(consumer_mbar(idx))`;
            # it is a named helper only so shared-key can empty it in one place
            # beside its two siblings. There, `_pv_ws` has already released that
            # slot per-position, and a second arrival would over-credit the
            # producer by exactly one slot -- the desync class that lands on the
            # right slot with the right phase and returns WRONG DATA instead of
            # hanging.
            @__parameter
            @always_inline
            def _v_release_first(v_idx0: UInt32):
                comptime if not v_sk_inline_lifecycle:
                    _commit(kv_pipeline.consumer_mbar(v_idx0))

            # `_pv_ws`: P@V (WS only). TWO arms disagree about where the A
            # operand lives: the key-split arm passes a raw TMEM address to
            # `UMMA1Type` (`a_tmem=True`), the shared-key arm passes an
            # `MMASmemDescriptorPair` over `smem.p_smem()` to `UMMA1SKType`
            # (`a_tmem=False`). `s_tmem` is read by ONE arm, `wg` (which selects
            # that warpgroup's P tile) by the other; `wg` takes the same position
            # and values as `_pv`'s `wg`.
            #
            # KEY-SPLIT arm (the `else`): P@V across `v_subslots` ring slots;
            # `v_idx0` is the first slot's index, the rest ring-adjacent.
            # V-slot outer, P-stage inner (mirrors `_pv`); wait each P
            # sub-stage once on the first slot only (P stays live in TMEM across
            # slots). Layout-G (m_pack==4, depth-scatter): `v_stage` selects a
            # DISJOINT output depth-tile (`o_tmem + v_stage*depth_tile`) from its
            # own slot base (NOT the `+d*v_region_bytes` intra-slot offset the
            # non-WS path uses for its contiguous V buffer); `c_scale` applies to
            # every v_stage unchanged (each writes independent columns).
            # Layout-E (m_pack==2, reduction-accumulate): `v_stage` is a
            # reduction (key) chunk `r` -- every chunk MMAs into the SAME
            # (un-offset) `o_tmem`, so only `v_stage==0` uses the caller's
            # `c_scale`; later chunks accumulate (scale=1), composing with
            # `UMMA1Type`'s own per-`p_stage` internal `stage_idx==0 -> c_scale
            # else 1` handling.
            #
            # SHARED-KEY arm: key chunks outer, depth tiles inner, all P waits
            # hoisted, each ring position released as its MMAs issue, `v_idx0`
            # unread -- the caller's wait leaves the cursor on (0, 0) and every
            # later one is a bare step off it. See its own comment block below.
            @__parameter
            @always_inline
            def _pv_ws[
                partial: Bool
            ](
                s_tmem: UInt32,
                v_idx0: UInt32,
                o_tmem: UInt32,
                consumer_s: MBarType,
                wait_phase: UInt32,
                c_scale: UInt32,
                valid_k_mmas: UInt32,
                wg: UInt32 = 0,
            ):
                comptime if v_sk_inline_lifecycle:
                    # ---- Shared-key: key chunks outer, depth tiles inner ----
                    #
                    # The caller has waited position (0, 0) and left the ring
                    # cursor on it; `v_idx0` is that index. Every later position
                    # is reached by a bare step + wait off the RUNNING cursor,
                    # NOT `(v_idx0 + n) % num_kv_stages`: the group is
                    # ring-contiguous only when complete, and under a skip it
                    # is a runtime-length prefix -- anchoring indices is how a
                    # skip turns into a read of the wrong slot.
                    #
                    # Each position is released the instant its MMAs issue.
                    # `_commit` is `elect_mma_arrive` (a `tcgen05.commit`), so
                    # the arrival is ordered BEHIND every MMA this warp issued --
                    # the release cannot outrun the operand read, and V liveness
                    # drops to one in-flight slot. That drop is the whole point:
                    # it takes the ring's V residency from `num_o_tiles()` to 1.
                    #
                    # P availability is hoisted out of the walk. The old
                    # `comptime if v_stage == 0` spelling cannot survive here --
                    # the first chunk index is runtime -- and hoisting is
                    # strictly more conservative than waiting inside.
                    comptime for p_stage in range(num_pv_stages):
                        _ = consumer_s[p_stage].wait(wait_phase)

                    # Live key chunks. `valid_k_mmas` is the tile-wide count of
                    # MMA_K blocks, so `ceildiv` by the per-chunk block count is
                    # the chunk count -- the SAME quantity the producer derives
                    # in key units as `ceildiv(num_keys - base, pv_key_chunk())`,
                    # because `ceildiv(ceildiv(a, m), n) == ceildiv(a, m*n)`.
                    # Non-partial sites know the tile is full, so the bound
                    # folds to a constant and the walk fully unrolls.
                    var live_chunks = UInt32(v_sk_chunks)
                    comptime if partial:
                        debug_assert(
                            valid_k_mmas > UInt32(0),
                            (
                                "shared-key V: valid_k_mmas == 0 -- chunk 0 is"
                                " never dead (I1), so this is a frame error"
                            ),
                        )
                        live_chunks = ceildiv(
                            valid_k_mmas, UInt32(v_sk_blocks_per_chunk)
                        )
                    # ---- The A operand: P, from SMEM ----------------------
                    # `store_p_quadrant` writes warpgroup `wg`'s `BM x BN` P tile
                    # at `p_smem() + wg*BM*BN`, in the SWIZZLE_NONE k-major layout
                    # `element (r, k) -> (k // sw_K)*(BM*sw_K) + r*sw_K`. The
                    # `tile_layout_k_major(BM, BN, SWIZZLE_NONE)` IS that map, so
                    # the descriptor, the MMA's internal per-k-block advance, and
                    # the per-chunk advance below all read off one layout.
                    #
                    # `wg` is RUNTIME, not comptime, and that is load-bearing:
                    # the odd-`T` tail rebinds `s1_tmem = s0_tmem` at runtime, so
                    # the epilogue serves WG1 on even `T`, WG0 on odd `T`. A
                    # comptime `1` there would read WG1's P against WG0's V while
                    # still waiting WG0's barrier -- wrong output on odd `T`
                    # only. Same position and values as `_pv`'s `wg`.
                    comptime P_SWIZZLE = TensorMapSwizzle.SWIZZLE_NONE
                    comptime p_layout = tile_layout_k_major[
                        config.qkv_dtype, BM, BN, P_SWIZZLE
                    ]()
                    # Bytes from one key chunk's P columns to the next, taken
                    # from the layout. Closed form `BM * pv_bk * dtsize`; assert
                    # so a layout change cannot silently walk the descriptor off
                    # the tile.
                    comptime p_chunk_bytes = (
                        p_layout(IntTuple(0, pv_bk))
                        * size_of[config.qkv_dtype]()
                    )
                    comptime assert (
                        p_chunk_bytes
                        == BM * pv_bk * size_of[config.qkv_dtype]()
                    ), (
                        "shared-key: the P key-chunk stride must be one"
                        " BM x pv_bk sub-tile"
                    )
                    var p_desc = smem_descriptor[
                        BMN=BM,
                        BK=BN,
                        swizzle_mode=P_SWIZZLE,
                        is_k_major=True,
                    ](smem.p_smem() + wg * UInt32(BM * BN))

                    for c in range(live_chunks):
                        # P columns advance by ONE key chunk per `c`;
                        # `UMMA1SKType`'s internal `stage_idx` offset only covers
                        # a single `pv_bk`-wide window relative to this base.
                        # NOT `bulk_mma_ws`'s `k_start`: that is comptime while `c`
                        # is a runtime induction var, and it is a SHARED slice index
                        # that would move the B operand too -- B is a per-ring-
                        # position descriptor holding exactly this one chunk's keys.
                        var chunk_p_desc = p_desc + c * UInt32(p_chunk_bytes)
                        # Chunks REDUCE into each output tile, so the caller's
                        # `c_scale` applies once per tile (first chunk, whatever
                        # `t` is) and later chunks accumulate.
                        var chunk_c_scale: UInt32 = (
                            1 if c != UInt32(0) else c_scale
                        )
                        var chunk_vkm: UInt32 = UInt32(v_sk_blocks_per_chunk)
                        comptime if partial:
                            # `c < live_chunks` guarantees this subtraction is
                            # positive, which is why it is evaluated HERE, not
                            # once per tile outside the loop: the dead chunks would
                            # wrap it, and `min(huge, bpc)` would hand back a FULL
                            # contraction of a dead chunk.
                            chunk_vkm = min(
                                valid_k_mmas
                                - c * UInt32(v_sk_blocks_per_chunk),
                                UInt32(v_sk_blocks_per_chunk),
                            )
                        comptime for t in range(config.num_o_tiles()):
                            # Position (c, t). The caller already waited (0, 0);
                            # every other position is a bare step + wait off the
                            # RUNNING cursor.
                            #
                            # Written as one predicate, not a `comptime if t != 0 /
                            # else` pair, on purpose. At `num_o_tiles() == 1` --
                            # every geometry reachable from the
                            # `depth in {64, 128}` force intercept -- a `t != 0`
                            # arm is comptime-false, parsed but never
                            # instantiated, so it ships having never reached the
                            # type checker; the first thing to exercise it would
                            # be d512. This spelling folds to `c == 0` at t == 0
                            # and to `False` at t != 0 while being type-checked
                            # once, for every geometry.
                            var first_pos = (
                                c == UInt32(0)
                            ) if t == 0 else False
                            if not first_pos:
                                kv_pipeline.state.step()
                                kv_pipeline.consumer_wait()
                            var sk_idx = kv_pipeline.state.index()
                            var sk_slot = (
                                kv_desc_v + UInt32(kv_stage_bytes) * sk_idx
                            )
                            # O tile stride is `pv_mma_n() // m_pack`, NOT the
                            # `depth_tile` (`256 // m_pack`) the other two WS
                            # arms use. `num_o_tiles()`'s invariant is
                            # `num_o_tiles() * (pv_mma_n() // m_pack) ==
                            # o_phys_cols()`, and shared-key's `pv_mma_n()` is
                            # `min(padded_ov_depth, 256)`, so the two AGREE at
                            # d256/d512 (256) and DISAGREE at d128 (`pv_mma_n()`
                            # 128 vs `depth_tile` 64). Invisible today only
                            # because d128 has `num_o_tiles() == 1` and the
                            # offset is `* 0` -- the wrong spelling would pass
                            # the only geometry that can compile it and scatter
                            # O to the wrong columns on d512.
                            comptime sk_o_stride = pv_mma_n // config.m_pack
                            var sk_o_tmem = o_tmem + UInt32(t * sk_o_stride)
                            comptime for p_stage in range(num_pv_stages):
                                comptime if partial:
                                    UMMA1SKType.mma_maybe_partial_k[
                                        stage_idx=p_stage
                                    ](
                                        chunk_p_desc,
                                        sk_slot,
                                        sk_o_tmem,
                                        c_scale=chunk_c_scale,
                                        elect=e,
                                        valid_k_mmas=chunk_vkm,
                                    )
                                else:
                                    # A full tile takes the plain contraction
                                    # rather than `mma_maybe_partial_k` with a
                                    # saturated count. The two are numerically
                                    # identical (every `@!%pv` guard is
                                    # never-true), but the partial form still
                                    # EMITS the `setp` + predicate on every
                                    # block, and this is the interior-tile path
                                    # that runs `T - 1` times out of `T`.
                                    UMMA1SKType.mma[stage_idx=p_stage](
                                        chunk_p_desc,
                                        sk_slot,
                                        sk_o_tmem,
                                        c_scale=chunk_c_scale,
                                        elect=e,
                                    )
                            _commit(kv_pipeline.consumer_mbar(sk_idx))
                else:
                    # Key-split (Layout-G depth-scatter / Layout-E
                    # reduction-split / non-WS). An `else` rather than trailing
                    # code after a `return`: Mojo ELABORATES statements following a
                    # comptime-true arm, so as trailing code this body would be
                    # type-checked and its `v_subslots` loop instantiated in every
                    # shared-key build too. It passes a raw TMEM address to
                    # `UMMA1Type` (`AInput` is `UInt32` in both modes), so it is
                    # not load-bearing for the types -- just for saying exactly
                    # one of these arms exists per config.
                    comptime for v_stage in range(v_subslots):
                        var v_idx_d: UInt32
                        comptime if v_stage == 0:
                            v_idx_d = v_idx0
                        else:
                            v_idx_d = (v_idx0 + UInt32(v_stage)) % UInt32(
                                config.num_kv_stages
                            )
                        var v_slot = (
                            kv_desc_v + UInt32(kv_stage_bytes) * v_idx_d
                        )
                        # Per-`v_stage` O / P-column offsets, exactly one nonzero
                        # per layout. Layout-G scatters O by depth-tile, never
                        # advances P. Layout-E accumulates into un-offset O but must
                        # advance the P base by `v_stage` reduction chunks
                        # (`s_tmem_chunk_words` TMEM cols each): `UMMA1Type.mma`'s
                        # internal offset only covers ONE `pv_bk_chunk`-wide window
                        # relative to this base, so without it every chunk silently
                        # re-reads chunk 0's P columns against a DIFFERENT V key
                        # range. `v_stage` is comptime, so both fold to constants.
                        comptime o_col_offset = (
                            0 if is_reduction_split else v_stage * depth_tile
                        )
                        comptime s_col_offset = (
                            v_stage
                            * s_tmem_chunk_words if is_reduction_split else 0
                        )
                        var stage_o_tmem = o_tmem + UInt32(o_col_offset)
                        var stage_s_tmem = s_tmem + UInt32(s_col_offset)
                        var stage_c_scale: UInt32 = c_scale
                        comptime if is_reduction_split and v_stage != 0:
                            stage_c_scale = 1
                        comptime for p_stage in range(num_pv_stages):
                            comptime if v_stage == 0:
                                _ = consumer_s[p_stage].wait(wait_phase)
                            comptime if partial:
                                UMMA1Type.mma_maybe_partial_k[
                                    stage_idx=p_stage
                                ](
                                    stage_s_tmem,
                                    v_slot,
                                    stage_o_tmem,
                                    c_scale=stage_c_scale,
                                    elect=e,
                                    valid_k_mmas=valid_k_mmas,
                                )
                            else:
                                UMMA1Type.mma[stage_idx=p_stage](
                                    stage_s_tmem,
                                    v_slot,
                                    stage_o_tmem,
                                    elect=e,
                                    c_scale=stage_c_scale,
                                )

            # ONE P@V entry point for the shared-KV 1Q schedule. WS takes the
            # ring-walking `_pv_ws`, non-WS the flat `_pv` over this slot's
            # descriptor; the two disagree only on how a ring index becomes an
            # operand, so the dispatch lives here rather than at all five call
            # sites. Takes the ring INDEX -- the one form both arms express.
            @__parameter
            @always_inline
            def _pv_kv[
                partial: Bool = False
            ](
                s_tmem: UInt32,
                v_idx: UInt32,
                o_tmem: UInt32,
                consumer_s: MBarType,
                wait_phase: UInt32,
                c_scale: UInt32,
                *,
                valid_k_mmas: UInt32 = 0,
                wg: UInt32 = 0,
            ):
                comptime if config.use_ws:
                    _pv_ws[partial](
                        s_tmem,
                        v_idx,
                        o_tmem,
                        consumer_s,
                        wait_phase,
                        c_scale,
                        valid_k_mmas,
                        wg,
                    )
                else:
                    _pv[partial](
                        s_tmem,
                        kv_desc_v + UInt32(kv_stage_bytes) * v_idx,
                        o_tmem,
                        consumer_s,
                        wait_phase,
                        c_scale,
                        valid_k_mmas=valid_k_mmas,
                        wg=wg,
                    )

            # The last-tile cut for this arm: `_pv_tile0` / `_pv_final`'s gate,
            # forwarding through `_pv_kv`. It is spelled `CUT_LAST_TILE` and
            # not `PARTIAL_K` because it must ALSO fire unconditionally under
            # shared-key, where producer and consumer have to cut the same key
            # chunks -- which is exactly why this arm needs its own pair rather
            # than reusing the fa4_mma-scope wrappers.
            @__parameter
            @always_inline
            def _pv_kv_tile0(
                s_tmem: UInt32,
                v_idx: UInt32,
                o_tmem: UInt32,
                consumer_s: MBarType,
                wait_phase: UInt32,
                c_scale: UInt32,
                v_eff: UInt32,
                *,
                wg: UInt32 = 0,
            ):
                comptime if CUT_LAST_TILE:
                    _pv_kv[partial=True](
                        s_tmem,
                        v_idx,
                        o_tmem,
                        consumer_s,
                        wait_phase,
                        c_scale,
                        valid_k_mmas=_vkm_tile0(v_eff),
                        wg=wg,
                    )
                else:
                    _pv_kv(
                        s_tmem,
                        v_idx,
                        o_tmem,
                        consumer_s,
                        wait_phase,
                        c_scale,
                        wg=wg,
                    )

            @__parameter
            @always_inline
            def _pv_kv_final(
                s_tmem: UInt32,
                v_idx: UInt32,
                o_tmem: UInt32,
                consumer_s: MBarType,
                wait_phase: UInt32,
                c_scale: UInt32,
                v_eff: UInt32,
                total_iters: UInt32,
                *,
                wg: UInt32 = 0,
            ):
                comptime if CUT_LAST_TILE:
                    _pv_kv[partial=True](
                        s_tmem,
                        v_idx,
                        o_tmem,
                        consumer_s,
                        wait_phase,
                        c_scale,
                        valid_k_mmas=_vkm_final(v_eff, total_iters),
                        wg=wg,
                    )
                else:
                    _pv_kv(
                        s_tmem,
                        v_idx,
                        o_tmem,
                        consumer_s,
                        wait_phase,
                        c_scale,
                        wg=wg,
                    )

            # ---- Peeled iteration ----
            # Stage 0 = K0 (K_e[0]_d0)
            kv_pipeline.consumer_wait()
            var k0 = (
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            )
            UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
            _qk_extra(q0, s0_tmem)  # WS: K_e[0]_d1..; non-WS: no-op
            _commit(pipeline_s0.producer_mbar())

            # 1Q: release K_e[0] (last sub-slot); step to the next tile's first
            # sub-slot; wait. It holds K_o[0]_d0 for T >= 2 and V_e[0]_d0 for
            # T == 1 -- diverge on descriptor base only.
            var slot1_offset = UInt32(kv_stage_bytes) * _advance_kv(
                kv_pipeline.state.index()
            )

            # T == 1 fast path: slot 1 holds V_e[0] (only K_e[0] + V_e[0] were
            # produced, = 2*num_qk_stages sub-slots). P_e @ V_e[0] -> o0 and
            # return; don't touch s1/o1 (softmax WG1 takes the matching no-op).
            if total_iters_runtime == UInt32(1):
                var ve_idx0 = kv_pipeline.state.index()  # V_e[0]_d0
                _v_wait_rest()  # WS: wait V_e[0]_d1..
                # Tile 0 is also the LAST tile here, so it takes the cut;
                # `_vkm_tile0` self-clamps to a full tile when it is not short.
                _pv_kv_tile0(
                    s0_tmem, ve_idx0, o0_tmem, consumer_s0, 0, 0, v_eff_keys
                )
                _commit(pipeline_o0.producer_mbar())
                _v_release_first(ve_idx0)  # V_e[0]_d0
                _v_release_rest(ve_idx0)  # WS: release V_e[0]_d1..
                return

            k0 = kv_desc_k + slot1_offset  # K_o[0]_d0

            # Q @ K_o[0] (q0 + redefined k0), depth-split across
            # sub-slots for WS.
            UMMA0Type.mma[stage_idx=0](q0, k0, s1_tmem, elect=e, c_scale=0)
            _qk_extra(q0, s1_tmem)  # WS: K_o[0]_d1..

            # Release K (K_o[0], last sub-slot) and advance.
            _commit(kv_pipeline.consumer_mbar())
            kv_pipeline.state.step()
            _commit(pipeline_s1.producer_mbar())

            # Stage 1 = V_e[0] (single use; we then load V_o[0] and hold
            # it for the first main-loop iter).
            kv_pipeline.consumer_wait()
            var v_prev_idx: UInt32 = kv_pipeline.state.index()  # V_e[0]_d0
            _v_wait_rest()  # WS: wait V_e[0]_d1..
            _pv_kv(s0_tmem, v_prev_idx, o0_tmem, consumer_s0, 0, 0)
            _commit(pipeline_o0.producer_mbar())
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # The epilogue's softmax WARPGROUP INDEX. Epilogue P@V is WG1 by
            # default; the 1Q odd-T tail re-points the s1 aliases to WG0, so
            # flip this there too. It must stay a RUNTIME value: that rebind is
            # a runtime `if`, so both parities reach the epilogue.
            #
            # TWO consumers now, which is why the name under-describes it. Off
            # the shared-key path it is BLASST's vote index and is DCE'd when
            # BLASST is off; under `ws_shared_key` it ALSO selects which of the
            # two `p_smem` tiles the SS P@V takes as its A operand. Getting that
            # wrong is silent -- the epilogue still waits WG0's barrier
            # successfully and simply contracts WG1's P, so only odd `T` is
            # wrong. A rename wants a sweep of the three sibling copies in arms
            # this change does not touch.
            var blasst_epi_wg: UInt32 = 1

            # 1Q: release V_e[0] (all sub-slots). V_o[0]'s acquire is deferred
            # into the first main-loop trip, so `v_prev_idx` stays on V_e[0]
            # until the loop rebinds it -- nothing reads it in between, and the
            # ring cursor is left on V_e[0]'s LAST sub-slot. The `state.step()`
            # at the loop top is what reaches K_e[1]_d0.
            #
            # `_v_release_first` is load-bearing: it is the d0 arrival that
            # `_advance_kv` carries as a side effect on the paths that use it,
            # and `_v_release_rest` only covers d1..
            _v_release_first(v_prev_idx)  # V_e[0]_d0
            _v_release_rest(v_prev_idx)  # WS: release V_e[0]_d1..

            # ---- Main loop ----
            while iter_count != 0:
                iter_count -= 1

                # Advance past held V to get to next K
                kv_pipeline.state.step()

                # Kn (K_e[n]_d0, depth-split across sub-slots for WS)
                kv_pipeline.consumer_wait()
                var kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                UMMA0Type.mma[stage_idx=0](q0, kn, s0_tmem, elect=e, c_scale=0)
                _qk_extra(q0, s0_tmem)  # WS: K_e[n]_d1..
                # Shared-key: this commit is ALSO what makes the single-buffered
                # `p_smem` safe, and nothing else says so. Per-WG issue order is
                # `... PV_wg(n) ... QK_wg(n+1); _commit(producer_mbar)`, and UMMA
                # retires in issue order, so "S(n+1) ready" => PV_wg(n) complete =>
                # WG0's P tile free. That is the WAR against softmax overwriting
                # P(n) before the P@V reads it -- no barrier behind it. Reorder
                # so a QK commit precedes the PV it must trail, and P corrupts
                # silently. (Live only once P@V's A operand moved to SMEM; the T==1
                # fast path and epilogue end at `_commit(pipeline_o*.producer)`
                # instead, after which the tile is not reused.)
                _commit(pipeline_s0.producer_mbar())

                # V_o[n-1] sits behind K_e[n]. This releases K_e[n]'s LAST sub-slot
                # before the V wait -- the early-K release that takes the span
                # floor S+1 -> S. Must stay AFTER the s0 producer commit, or
                # softmax WG0 parks behind this KV stall.
                #
                # DO NOT demote to a bare `state.step()`. One of three sites that
                # retire the preceding K tile's last sub-slot BEFORE a V d0 wait.
                # Once V is released per key chunk instead of per group, all three
                # look vestigial -- and they are the entire difference between a
                # floor of O(1) and S+1. Dropping any re-hangs bf16 d512 at
                # BN=256, where only 4 ring slots are affordable.
                v_prev_idx = _advance_kv(kv_pipeline.state.index())
                _v_wait_rest()  # WS: wait V_o[n-1]_d1..

                # P1 @ V_{n-1} (held V_o[n-1]). Per-depth-tile ring slots for
                # WS.
                _pv_kv(
                    s1_tmem,
                    v_prev_idx,
                    o1_tmem,
                    consumer_s1,
                    phase,
                    c_scale,
                    wg=1,  # WG1's P tile
                )
                _commit(pipeline_o1.producer_mbar())
                c_scale = 1
                _v_release_first(v_prev_idx)  # V_{n-1}_d0
                _v_release_rest(v_prev_idx)  # WS: release V_{n-1}_d1..

                # Between K_e[n] and K_o[n] -- break-check for tail
                # iter when total K-tiles is odd, else consume K_o[n] by
                # releasing K_e[n] and reassigning kn = K_o[n].
                if iter_count == 0:
                    # Tail iter (T odd): no K_o[k]. The remaining work
                    # -- P_e[k] @ V_e[k] -> o0_tmem -- has the same shape as the
                    # epilogue's P_1 @ V_last -> o1_tmem. Rebind o1-side aliases to
                    # o0-side resources and fall through; epilogue does the work
                    # unchanged. K_e[k]'s last-sub-slot release happened above (the
                    # V_o[k-1] acquire) and V_o[k-1] was released after its P@V --
                    # nothing left to release here, and the V_e[k] acquire moves
                    # to the single post-loop acquire that serves both T parities.
                    s1_tmem = s0_tmem
                    o1_tmem = o0_tmem
                    consumer_s1 = consumer_s0
                    pipeline_o1 = pipeline_o0
                    blasst_epi_wg = 0  # aliased to WG0 side
                    phase ^= 1  # advance from this iter's K@s1 phase
                    # to the V@o0 phase the s0 wait needs.
                    break
                iter_count -= 1
                # The cursor sits on V_o[n-1]_d(S-1), ALREADY released by
                # `_v_release_rest` above, so `_advance_kv` here would arrive on
                # its consumer mbar a SECOND time. Live WAR hazard now that
                # `ring_slots_needed()` admits N == S -- keep this a bare
                # step + wait.
                kv_pipeline.state.step()
                kv_pipeline.consumer_wait()
                kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )

                # Q @ K_o[n] (q0 + redefined kn), depth-split across
                # sub-slots for WS.
                UMMA0Type.mma[stage_idx=0](q0, kn, s1_tmem, elect=e, c_scale=0)
                _qk_extra(q0, s1_tmem)  # WS: K_o[n]_d1..
                _commit(
                    kv_pipeline.consumer_mbar()
                )  # release K_n / K_o[n] last
                kv_pipeline.state.step()
                # WG1's half of the implicit P WAR -- see the WG0 commit above.
                _commit(pipeline_s1.producer_mbar())
                phase ^= 1

                # V_e[n] (single use), then V_o[n] loaded and held.
                kv_pipeline.consumer_wait()
                v_prev_idx = kv_pipeline.state.index()  # V_e[n]_d0
                _v_wait_rest()  # WS: wait V_e[n]_d1..
                _pv_kv(s0_tmem, v_prev_idx, o0_tmem, consumer_s0, phase, 1)
                _commit(pipeline_o0.producer_mbar())

                # 1Q: release V_e[n] (all sub-slots). V_o[n]'s acquire is
                # deferred to the NEXT trip's V_o[n-1] site -- or, on the last
                # trip, to the single post-loop acquire below.
                _v_release_first(v_prev_idx)  # V_e[n]_d0
                _v_release_rest(v_prev_idx)  # WS: release V_e[n]_d1..

            # ONE unconditional acquire serving BOTH T parities. T odd ->
            # V_e[k] from the producer's tail block; T even -> the terminal
            # drain's V_o[m]. Both are "the next group after the last one the
            # loop consumed", and in both cases everything below it is already
            # released -- so this is a bare step + wait, not `_advance_kv`,
            # which would double-arrive. Unreachable for T == 1 (fast path
            # returns above).
            kv_pipeline.state.step()
            kv_pipeline.consumer_wait()
            v_prev_idx = kv_pipeline.state.index()
            _v_wait_rest()

            # ---- Epilogue ----
            _pv_kv_final(
                s1_tmem,
                v_prev_idx,
                o1_tmem,
                consumer_s1,
                phase,
                c_scale,
                v_eff_keys,
                total_iters_runtime,
                wg=blasst_epi_wg,
            )
            _commit(pipeline_o1.producer_mbar())
            _v_release_first(v_prev_idx)  # V_last_d0
            _v_release_rest(v_prev_idx)  # WS: release V_last_d1..

        else:
            # ---- Non-shared mode (original) ----

            var k_smem = rebind[SharedMemPointer[Scalar[config.qkv_dtype]]](
                smem.k_smem_base()
            )
            var v_smem = rebind[SharedMemPointer[Scalar[config.qkv_dtype]]](
                smem.v_smem_base()
            )
            comptime KPipeType = KConsumerPipeline[config.qkv_dtype, config]
            comptime VPipeType = VConsumerPipeline[config.qkv_dtype, config]
            var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
            var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

            # We peel the first iteration, as we want to wait on q1.
            # Peel consumes 2 K-tiles (K_e[0], K_o[0]) and 2 V-tiles
            # (V_e[0], V_o[0] held); the main loop decrements once at the
            # top (K_e consume) and once more past the break-check
            # (K_o consume). iter_count = total_iters - 2.
            # At total_iters == 1 an early-return fast path runs after the
            # Q @ K_e[0] staged MMA below, so the iter_count underflow at
            # T == 1 (1u32 - 2u32 wraps) is never read. Keep the raw value in
            # `total_iters_runtime` for the runtime branch.
            var total_iters_runtime: UInt32 = mask.total_iters[
                BM_mask, BN, page_size
            ](seq_id, score_row, num_keys)
            # Split-K (1Q): slice the combined tile count to this partition's
            # window. mma needs only the per-partition COUNT -- load_warp emits
            # the matching cb-offset tiles that this warp consumes in order.
            # `total_iters == last_masked_set_end` for check_mask==False masks
            # (mha_mask.mojo derives the split-K window identically across all
            # four warps), so all four warps derive the same window.
            comptime if config.splitk_partitions > 1 or workspace_split:
                var _np = splitk_num_partitions[config](ws_num_partitions)
                var _w = splitk_window(
                    total_iters_runtime,
                    _np,
                    splitk_partition_idx(_np),
                )
                total_iters_runtime = _w[1] - _w[0]
                # Empty partition (front-load trailing window, T < num_partitions;
                # or an M6 idle CTA): no tiles, so load_warp produces no K0 and the
                # peeled `pipeline_k.wait_k` below would hang. Skip all MMA work --
                # the empty partition's softmax stages a neutral identity, and the
                # kernel terminal `cluster_sync()` still runs after this.
                if total_iters_runtime == 0:
                    return
                # Rebase into this partition's tile window (see the `vkm` note).
                v_eff_keys -= _w[0] * UInt32(BN)
            # All-masked row (valid_length 0): total_iters == 0, so the `- 2`
            # below underflows and the MMA warp spins, hanging the pipeline.
            # Split-K is guarded above; guard the non-split path here.
            if total_iters_runtime == 0:
                return
            var iter_count: UInt32 = total_iters_runtime - UInt32(2)

            # Q @ K_e[0]', staged over num_qk_stages
            var k0 = pipeline_k.get_k()

            comptime for qk_stage in range(num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, k0, s0_tmem, elect=e, c_scale=0
                )
                # 1Q: release K_e[0] stage (single use); step at last.
                _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                comptime if qk_stage == num_qk_stages - 1:
                    pipeline_k.pipeline.state.step()
            _commit(pipeline_s0.producer_mbar())

            # 1Q T==1 fast path. Only one K-tile in the sequence (K_e[0]), so
            # K_o[0] is never produced by load_warp and Q @ K_o[0] would hang on
            # `pipeline_k.wait_k`. Do the single P_e @ V_e[0] -> o0 MMA and exit.
            # Skip Q @ K_o[0] -> s1, V_o[0] hold, the main loop, and the epilogue --
            # no mbar on s1/o1 touched here; softmax WG1 takes the matching no-op
            # so the s1/o1 producer-consumer balance is preserved.
            if total_iters_runtime == UInt32(1):
                var vlatest_t1 = pipeline_v.get_v()
                pipeline_v.wait_v()
                # Tile 0 is also the LAST tile here; `_vkm_tile0` self-clamps.
                _pv_tile0(
                    s0_tmem, vlatest_t1, o0_tmem, consumer_s0, 0, 0, v_eff_keys
                )
                _commit(pipeline_o0.producer_mbar())
                var ve_idx = pipeline_v.pipeline.state.index()
                pipeline_v.pipeline.consumer_release_at(ve_idx, e)
                return

            # 1Q: redefine k0 = K_o[0] for the s1 staged loop below.
            k0 = pipeline_k.get_k()

            # Q @ K_o[0]' (q0 + redefined k0), staged over num_qk_stages.
            comptime for qk_stage in range(num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, k0, s1_tmem, elect=e, c_scale=0
                )
                _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                comptime if qk_stage == num_qk_stages - 1:
                    pipeline_k.pipeline.state.step()
            _commit(pipeline_s1.producer_mbar())

            # V_e[0] (single use), then V_o[0] loaded and held.
            var vlatest = pipeline_v.get_v()  # [kv1]
            pipeline_v.wait_v()  # [kv1]

            # For the first V tile in the current KV stage buffer:
            # Use the SAME base pointer you used for K (no manual offset).
            _pv(s0_tmem, vlatest, o0_tmem, consumer_s0, 0, 0)
            _commit(pipeline_o0.producer_mbar())
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # BLASST: see the fused-mode `blasst_epi_wg` above -- same 1Q tail
            # flip.
            var blasst_epi_wg: UInt32 = 1

            # Release V_e[0] (single use); advance to V_o[0]; load
            # and HOLD vlatest = V_o[0] in vo_prev_idx for the first
            # main-loop iter's P_o @ V_o[0] MMA. State is pre-advanced
            # past the held slot so subsequent get_v() returns V_e[1].
            var ve_idx = pipeline_v.pipeline.state.index()
            pipeline_v.pipeline.consumer_release_at(ve_idx, e)
            pipeline_v.pipeline.state.step()
            vlatest = pipeline_v.get_v()  # V_o[0]
            pipeline_v.wait_v()
            # vo_prev_idx tracks the held V_o slot index (needed by the
            # deferred consumer_release_at). Declared here, in the peel that
            # first holds a V_o, and read by the main loop's V_{n-1} release
            # and V_o[n] hold.
            var vo_prev_idx: UInt32 = pipeline_v.pipeline.state.index()
            pipeline_v.pipeline.state.step()  # advance; do NOT release
            while iter_count != 0:
                iter_count -= 1
                # Q @ K_e[n]', staged over num_qk_stages.
                var kn = pipeline_k.get_k()  # kv_{2n-1}->[kv_{2n}]

                comptime for qk_stage in range(num_qk_stages):
                    pipeline_k.wait_k[
                        qk_stage=qk_stage
                    ]()  # kv_{2n-1}->[kv_{2n}]
                    UMMA0Type.mma[stage_idx=qk_stage](
                        q0, kn, s0_tmem, elect=e, c_scale=0
                    )
                    # 1Q: release K_e[n] stage (single use); step at last.
                    _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                    comptime if qk_stage == num_qk_stages - 1:
                        pipeline_k.pipeline.state.step()
                _commit(pipeline_s0.producer_mbar())

                # O_o + P_o @ V_o[n-1].
                _pv(
                    s1_tmem, vlatest, o1_tmem, consumer_s1, phase, c_scale, wg=1
                )
                _commit(pipeline_o1.producer_mbar())
                c_scale = 1
                # Release V_o[n-1] at vo_prev_idx (state was pre-advanced
                # when V_o was held).
                pipeline_v.pipeline.consumer_release_at(vo_prev_idx, e)

                # Between K_e[n] and K_o[n] -- break-check for tail
                # iter when total K-tiles is odd, else load K_o[n] by
                # reassigning kn (K_e[n] was already released per-stage
                # in the Q@K_e[n] staged loop above).
                if iter_count == 0:
                    # Tail iter (T odd). Same alias-swap pattern as
                    # shared-KV. K_e[k] was already released per
                    # qk_stage inside the Q@K_e[k] staged loop above,
                    # so no K release is needed here.
                    vlatest = pipeline_v.get_v()  # V_e[k]
                    pipeline_v.wait_v()
                    vo_prev_idx = pipeline_v.pipeline.state.index()
                    pipeline_v.pipeline.state.step()
                    s1_tmem = s0_tmem
                    o1_tmem = o0_tmem
                    consumer_s1 = consumer_s0
                    pipeline_o1 = pipeline_o0
                    blasst_epi_wg = 0  # aliased to WG0 side
                    phase ^= 1
                    break
                iter_count -= 1
                kn = pipeline_k.get_k()  # kn = K_o[n]

                # Q @ K_o[n]' (q0 + redefined kn), staged over num_qk_stages.
                comptime for qk_stage in range(num_qk_stages):
                    pipeline_k.wait_k[qk_stage=qk_stage]()
                    UMMA0Type.mma[stage_idx=qk_stage](
                        q0, kn, s1_tmem, elect=e, c_scale=0
                    )
                    _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                    comptime if qk_stage == num_qk_stages - 1:
                        pipeline_k.pipeline.state.step()  # [kv_{2n}]->kv_{2n+1}
                _commit(pipeline_s1.producer_mbar())
                phase ^= 1

                # O_e + P_e @ V_e[n]
                vlatest = pipeline_v.get_v()  # [kv_{2n+1}]
                pipeline_v.wait_v()  # [kv_{2n+1}]

                _pv(s0_tmem, vlatest, o0_tmem, consumer_s0, phase, 1)
                _commit(pipeline_o0.producer_mbar())

                # 1Q: release V_e[n] (single use); advance to V_o[n];
                # redefine vlatest = V_o[n] and hold its slot index in
                # vo_prev_idx for the next iter / epilogue. State is
                # pre-advanced past the held slot.
                var ve_idx = pipeline_v.pipeline.state.index()
                pipeline_v.pipeline.consumer_release_at(ve_idx, e)
                pipeline_v.pipeline.state.step()
                vlatest = pipeline_v.get_v()  # V_o[n]
                pipeline_v.wait_v()
                vo_prev_idx = pipeline_v.pipeline.state.index()
                pipeline_v.pipeline.state.step()  # advance; do NOT release

            _pv_final(
                s1_tmem,
                vlatest,
                o1_tmem,
                consumer_s1,
                phase,
                c_scale,
                v_eff_keys,
                total_iters_runtime,
                wg=blasst_epi_wg,
            )
            _commit(pipeline_o1.producer_mbar())

    @__parameter
    @always_inline
    def _body_2q():
        comptime if config.use_shared_kv:
            # ---- Shared KV mode ----
            # In shared mode, K_nope and V alternate in a single StagedPipeline.
            # Stages: K0, V0, K1, V1, ...

            var kv_smem = (
                smem.k_smem_base()
            )  # same as v_smem_base in shared mode
            # Per-slot byte stride = ONE ring slot. Derived from the SMEM struct
            # so it can NEVER drift from the reservation/producer. WS: 32768
            # (one 256x64 sub-tile); non-WS shared: 65536 (one full-depth tile).
            # The old `v_cols_per_cta()*BN*size` literal equalled the non-WS
            # value but strode 2x the WS reservation -> latent overrun; this
            # reconciles it.
            comptime kv_stage_bytes = SM100AttentionSMem[config].k_stage_bytes

            # K descriptor: k_major for Q@K' (BK0=64 -> one depth-half
            # sub-tile).
            var kv_desc_k = smem_descriptor[
                BMN=config.k_rows_per_cta(),
                BK=config.BK0,
                swizzle_mode=config.swizzle_mode,
                is_k_major=True,
                page_dense=config.k_row_major(),
            ](kv_smem)
            # V descriptor: mn_major for P@V. WS uses the packed [pv_mma_n,
            # pv_bk] box -- Layout-G's proven one-depth-tile box (256x64) or
            # Layout-E's reduction-chunk box (256x64, same numbers @depth128,
            # different meaning); non-WS keeps the full [v_cols_per_cta, BN]
            # tile (byte-identical).
            comptime v_desc_bmn = (
                pv_mma_n if config.use_ws else config.v_cols_per_cta()
            )
            comptime v_desc_bk = pv_bk if config.use_ws else config.BN
            var kv_desc_v = smem_descriptor[
                BMN=v_desc_bmn,
                BK=v_desc_bk,
                swizzle_mode=config.swizzle_mode,
                is_k_major=False,
                page_dense=config.v_row_major(),
            ](kv_smem)

            comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
            var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

            # We peel the first iteration, as we want to wait on q1.
            # Peel consumes 1 K_0 (shared by both Q halves); the main loop
            # decrements once per iter. iter_count = total_iters - 1.
            # 1Q at total_iters == 1 takes the T==1 fast path below after the
            # Q @ K_e[0] staged MMA; the iter_count underflow at T == 1
            # (1u32 - 2u32 wraps) is never read.
            var total_iters_runtime: UInt32 = mask.total_iters[
                BM_mask, BN, page_size
            ](seq_id, score_row, num_keys)
            # All-masked row (valid_length 0): total_iters == 0, so the `- 1`
            # below underflows and the MMA warp spins, hanging the pipeline.
            # (Split-K is 1Q-only, so there is no split-K window to slice
            # here -- see `_body_1q`.)
            if total_iters_runtime == 0:
                return
            var iter_count: UInt32 = total_iters_runtime - UInt32(1)

            # ---- Cross-stage P QK0-first JIT schedule + K-ahead KV ring ----
            # Two cursors over the shared K0,K1,V0,K2,V1,... ring: ACQUIRE walks
            # position order every step; RELEASE (`rel_slot`) lags by <=2 but
            # fires in the SAME producer-fill order, so per-slot mbar parity
            # never skips. Per-iter MMA order
            # QK0(n)->PV0(n-1)->QK1(n)->PV1(n-1): QK0(n) overwrites S0 (incl.
            # P1's window S0[32:96]) before PV1(n-1) reads P1(n-1) from there --
            # a WAR hazard made safe by the softmax JIT-P1 seed, which delays
            # P1(n-1)'s store to land after QK0(n). 2Q + non-WS only; aliased
            # path below is byte-identical when off.
            comptime if CrossP:
                var p0_tmem = tmem_addr + UInt32(
                    config.TMEM_P0
                )  # P0 in S1 window
                var p1_tmem = tmem_addr + UInt32(
                    config.TMEM_P1
                )  # P1 in S0 window
                # MMA-side sfree consumers: QK{wg}(n) waits until softmax
                # finished reading S{wg}(n-1) before overwriting it. Peel QKs
                # skip this (no prior S). Persistent phase.
                var sfree0 = mbars.sfree_consumer(0)
                var sfree1 = mbars.sfree_consumer(1)

                # Private in-order release cursor (slot = pos % num_kv_stages).
                # No jump-back: releases fire in strict producer-fill order.
                var rel_slot: UInt32 = 0
                comptime last_stage = config.num_kv_stages - 1

                # ---- Peel: acquire K0 (pos0); QK0(0); QK1(0); release K0. ----
                kv_pipeline.consumer_wait()  # pos0 = K0
                var kc = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                kv_pipeline.state.step()
                UMMA0Type.mma[stage_idx=0](q0, kc, s0_tmem, elect=e, c_scale=0)
                _commit(pipeline_s0.producer_mbar())
                var q1m = mbars.q1_wait_mbar()
                q1m[0].wait()
                UMMA0Type.mma[stage_idx=0](q1, kc, s1_tmem, elect=e, c_scale=0)
                _commit(pipeline_s1.producer_mbar())
                # release K0 (pos0)
                _commit(kv_pipeline.consumer_mbar(rel_slot))
                rel_slot = 0 if rel_slot == UInt32(last_stage) else rel_slot + 1

                var xphase: UInt32 = 0
                var xcscale: UInt32 = 0

                # ---- Main loop: iter n (n=1..NT-1) makes S(n), drains P(n-1).
                # ----
                while iter_count != 0:
                    iter_count -= 1

                    # Acquire K_n (pos 2n-1).
                    kv_pipeline.consumer_wait()
                    var kn_idx = kv_pipeline.state.index()
                    kv_pipeline.state.step()
                    kc = kv_desc_k + UInt32(kv_stage_bytes) * kn_idx

                    # QK0(n): acquire sfree0 (softmax read S0(n-1)); q0*K_n ->
                    # S0.
                    sfree0.wait()
                    sfree0.step()
                    UMMA0Type.mma[stage_idx=0](
                        q0, kc, s0_tmem, elect=e, c_scale=0
                    )
                    _commit(pipeline_s0.producer_mbar())

                    # Acquire V_{n-1} (pos 2n).
                    kv_pipeline.consumer_wait()
                    var vn1_idx = kv_pipeline.state.index()
                    kv_pipeline.state.step()
                    var vn1 = kv_desc_v + UInt32(kv_stage_bytes) * vn1_idx

                    # PV0(n-1): P0(n-1) * V_{n-1} -> O0 (cross p0_tmem in S1
                    # window).
                    _pv(p0_tmem, vn1, o0_tmem, consumer_s0, xphase, xcscale)
                    _commit(pipeline_o0.producer_mbar())

                    # QK1(n): acquire sfree1; q1*K_n -> S1.
                    sfree1.wait()
                    sfree1.step()
                    UMMA0Type.mma[stage_idx=0](
                        q1, kc, s1_tmem, elect=e, c_scale=0
                    )
                    _commit(pipeline_s1.producer_mbar())

                    # Release K_n (pos 2n-1) -- BEFORE the deferred PV1 (FI
                    # 1317).
                    _commit(kv_pipeline.consumer_mbar(rel_slot))
                    rel_slot = (
                        0 if rel_slot == UInt32(last_stage) else rel_slot + 1
                    )

                    # PV1(n-1): P1(n-1) * V_{n-1} -> O1 (cross p1_tmem in S0
                    # window; JIT-protected -- softmax stored P1(n-1) after
                    # QK0(n)).
                    _pv(
                        p1_tmem,
                        vn1,
                        o1_tmem,
                        consumer_s1,
                        xphase,
                        xcscale,
                        wg=1,
                    )
                    _commit(pipeline_o1.producer_mbar())

                    # Release V_{n-1} (pos 2n) -- after PV1 (FI 1333).
                    _commit(kv_pipeline.consumer_mbar(rel_slot))
                    rel_slot = (
                        0 if rel_slot == UInt32(last_stage) else rel_slot + 1
                    )

                    xcscale = 1
                    xphase ^= 1

                # ---- Epilogue: acquire V_{NT-1} (pos 2NT-1); drain P0/P1. ----
                kv_pipeline.consumer_wait()
                var vl_idx = kv_pipeline.state.index()
                kv_pipeline.state.step()
                var vl = kv_desc_v + UInt32(kv_stage_bytes) * vl_idx
                _pv_final(
                    p0_tmem,
                    vl,
                    o0_tmem,
                    consumer_s0,
                    xphase,
                    xcscale,
                    v_eff_keys,
                    total_iters_runtime,
                )
                _commit(pipeline_o0.producer_mbar())
                _pv_final(
                    p1_tmem,
                    vl,
                    o1_tmem,
                    consumer_s1,
                    xphase,
                    xcscale,
                    v_eff_keys,
                    total_iters_runtime,
                    wg=1,
                )
                _commit(pipeline_o1.producer_mbar())
                # release V_{NT-1} (pos 2NT-1)
                _commit(kv_pipeline.consumer_mbar(rel_slot))
                return

            # ---- Peeled iteration ----
            # Stage 0 = K0 (K_e[0]_d0)
            kv_pipeline.consumer_wait()
            var k0 = (
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            )
            UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
            _commit(pipeline_s0.producer_mbar())

            # Q_1 @ K_0 (q1 half, same K).
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].wait()
            UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)

            # Release K_0 and advance.
            _commit(kv_pipeline.consumer_mbar())
            kv_pipeline.state.step()
            _commit(pipeline_s1.producer_mbar())

            # Stage 1 = V_0.
            kv_pipeline.consumer_wait()
            var v_prev_idx: UInt32 = kv_pipeline.state.index()  # V_e[0]_d0
            # 2Q peeled o0 contracts tile 0, which is the last (and only) tile
            # only when total_iters == 1; `_vkm_tile0` self-clamps to full
            # (v_eff_keys >= BN) otherwise. 2Q shared is non-WS.
            _pv_tile0(
                s0_tmem,
                kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                o0_tmem,
                consumer_s0,
                0,
                0,
                v_eff_keys,
            )
            _commit(pipeline_o0.producer_mbar())
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # BLASST: epilogue P@V is WG1 by default; the 1Q odd-T tail
            # re-points s1 aliases to WG0, so flip this there too (DCE'd when
            # off).
            var blasst_epi_wg: UInt32 = 1

            # ---- Main loop ----
            while iter_count != 0:
                iter_count -= 1

                # Advance past held V to get to next K
                kv_pipeline.state.step()

                # Kn (K_e[n]_d0, depth-split across sub-slots for WS)
                kv_pipeline.consumer_wait()
                var kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                UMMA0Type.mma[stage_idx=0](q0, kn, s0_tmem, elect=e, c_scale=0)
                _commit(pipeline_s0.producer_mbar())

                # P1 @ V_{n-1} (held V_o[n-1]). Per-depth-tile ring slots for
                # WS.
                _pv(
                    s1_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o1_tmem,
                    consumer_s1,
                    phase,
                    c_scale,
                    wg=1,
                )
                _commit(pipeline_o1.producer_mbar())
                c_scale = 1
                _commit(kv_pipeline.consumer_mbar(v_prev_idx))  # V_{n-1}_d0

                # Q_1 @ K_n (q1 + same kn).
                UMMA0Type.mma[stage_idx=0](q1, kn, s1_tmem, elect=e, c_scale=0)
                _commit(
                    kv_pipeline.consumer_mbar()
                )  # release K_n / K_o[n] last
                kv_pipeline.state.step()
                _commit(pipeline_s1.producer_mbar())
                phase ^= 1

                # Vn, held for the next iter.
                kv_pipeline.consumer_wait()
                v_prev_idx = kv_pipeline.state.index()  # V_e[n]_d0
                # 2Q: Vn is the last tile exactly when iter_count == 0 (the
                # final main-loop iteration); otherwise full.
                _pv_final(
                    s0_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o0_tmem,
                    consumer_s0,
                    phase,
                    1,
                    v_eff_keys,
                    total_iters_runtime,
                    is_last=(iter_count == UInt32(0)),
                )
                _commit(pipeline_o0.producer_mbar())

            # ---- Epilogue ----
            _pv_final(
                s1_tmem,
                kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                o1_tmem,
                consumer_s1,
                phase,
                c_scale,
                v_eff_keys,
                total_iters_runtime,
                wg=blasst_epi_wg,
            )
            _commit(pipeline_o1.producer_mbar())
            _commit(kv_pipeline.consumer_mbar(v_prev_idx))  # V_last_d0

        else:
            # ---- Non-shared mode (original) ----

            var k_smem = rebind[SharedMemPointer[Scalar[config.qkv_dtype]]](
                smem.k_smem_base()
            )
            var v_smem = rebind[SharedMemPointer[Scalar[config.qkv_dtype]]](
                smem.v_smem_base()
            )
            comptime KPipeType = KConsumerPipeline[config.qkv_dtype, config]
            comptime VPipeType = VConsumerPipeline[config.qkv_dtype, config]
            var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
            var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

            # We peel the first iteration, as we want to wait on q1.
            # Peel consumes 1 K_0 (shared by both Q halves); the main loop
            # decrements once per iter. iter_count = total_iters - 1.
            var total_iters_runtime: UInt32 = mask.total_iters[
                BM_mask, BN, page_size
            ](seq_id, score_row, num_keys)
            # All-masked row (valid_length 0): total_iters == 0, so the `- 1`
            # below underflows and the MMA warp spins, hanging the pipeline.
            # (Split-K is 1Q-only, so there is no split-K window to slice
            # here -- see `_body_1q`.)
            if total_iters_runtime == 0:
                return
            var iter_count: UInt32 = total_iters_runtime - UInt32(1)

            # Q_0 @ K_0', staged over num_qk_stages
            var k0 = pipeline_k.get_k()

            comptime for qk_stage in range(num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, k0, s0_tmem, elect=e, c_scale=0
                )
            _commit(pipeline_s0.producer_mbar())

            # 1Q T==1 fast path. Only one K-tile in the sequence (K_e[0]),
            # so K_o[0] is never produced by load_warp and Q @ K_o[0] would
            # hang on pipeline_k.wait_k. Do the single P_e @ V_e[0] -> o0
            # MMA and exit. Skip Q @ K_o[0] -> s1, V_o[0] hold, the main
            # loop, and the epilogue P @ V_held -> o1. No mbar on s1 / o1
            # is touched here; softmax_warp.mojo's WG1 takes the matching
            # no-op path so the s1/o1 producer-consumer balance is
            # preserved.

            # Q_1 @ K_0' (q1 half, same k0), staged over num_qk_stages.
            comptime for qk_stage in range(num_qk_stages):
                var q1_mbar = mbars.q1_wait_mbar()
                q1_mbar[qk_stage].wait()  # wait on Q1
                UMMA0Type.mma[stage_idx=qk_stage](
                    q1, k0, s1_tmem, elect=e, c_scale=0
                )
                _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                comptime if qk_stage == num_qk_stages - 1:
                    pipeline_k.pipeline.state.step()
            _commit(pipeline_s1.producer_mbar())

            # V_0, held for the first main-loop iter.
            var vlatest = pipeline_v.get_v()  # [kv1]
            pipeline_v.wait_v()  # [kv1]

            # For the first V tile in the current KV stage buffer:
            # Use the SAME base pointer you used for K (no manual offset).
            # 2Q peeled o0 contracts tile 0; last (and only) tile only when
            # total_iters == 1 -- `_vkm_tile0` self-clamps to full otherwise.
            _pv_tile0(s0_tmem, vlatest, o0_tmem, consumer_s0, 0, 0, v_eff_keys)
            _commit(pipeline_o0.producer_mbar())
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # BLASST: see the fused-mode `blasst_epi_wg` above.
            var blasst_epi_wg: UInt32 = 1

            while iter_count != 0:
                iter_count -= 1
                # Q_0 @ K_n', staged over num_qk_stages.
                var kn = pipeline_k.get_k()  # kv_{2n-1}->[kv_{2n}]

                comptime for qk_stage in range(num_qk_stages):
                    pipeline_k.wait_k[
                        qk_stage=qk_stage
                    ]()  # kv_{2n-1}->[kv_{2n}]
                    UMMA0Type.mma[stage_idx=qk_stage](
                        q0, kn, s0_tmem, elect=e, c_scale=0
                    )
                _commit(pipeline_s0.producer_mbar())

                # O_1 + P_1 @ V_{n-1}.
                _pv(
                    s1_tmem, vlatest, o1_tmem, consumer_s1, phase, c_scale, wg=1
                )
                _commit(pipeline_o1.producer_mbar())
                c_scale = 1
                # Release V_{n-1} at the current pipeline state.
                _commit(pipeline_v.pipeline.consumer_mbar[0]())
                pipeline_v.pipeline.state.step()  # [kv_{2n-1}]

                # Q_1 @ K_n' (q1 + same kn), staged over num_qk_stages.
                comptime for qk_stage in range(num_qk_stages):
                    UMMA0Type.mma[stage_idx=qk_stage](
                        q1, kn, s1_tmem, elect=e, c_scale=0
                    )
                    _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                    comptime if qk_stage == num_qk_stages - 1:
                        pipeline_k.pipeline.state.step()  # [kv_{2n}]->kv_{2n+1}
                _commit(pipeline_s1.producer_mbar())
                phase ^= 1

                # O_0 + P_0 @ V_n
                vlatest = pipeline_v.get_v()  # [kv_{2n+1}]
                pipeline_v.wait_v()  # [kv_{2n+1}]

                # 2Q: Vn is the last tile exactly when iter_count == 0.
                _pv_final(
                    s0_tmem,
                    vlatest,
                    o0_tmem,
                    consumer_s0,
                    phase,
                    1,
                    v_eff_keys,
                    total_iters_runtime,
                    is_last=(iter_count == UInt32(0)),
                )
                _commit(pipeline_o0.producer_mbar())

            _pv_final(
                s1_tmem,
                vlatest,
                o1_tmem,
                consumer_s1,
                phase,
                c_scale,
                v_eff_keys,
                total_iters_runtime,
                wg=blasst_epi_wg,
            )
            _commit(pipeline_o1.producer_mbar())

    comptime if num_q == 1:
        _body_1q()
    else:
        _body_2q()
