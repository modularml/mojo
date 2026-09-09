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
"""TMA load warp logic for depth=256/512 pair-CTA SM100 attention.

Each CTA in the pair loads its own half of K/V data into its local SMEM.
The pair-CTA MMA instruction reads from both SMs' SMEM to combine the halves.

K is split along BN rows: even CTA (`is_leader=True`) loads K[0:BN//2, :],
odd CTA (`is_leader=False`) loads K[BN//2:BN, :].

V loading depends on split_o (depth-dependent):
  split_o=True (depth=512): V split into V_lo and V_hi (separate pipeline slots):
    V_lo: even loads V[:, 0:ov_depth//4], odd loads V[:, ov_depth//4:ov_depth//2]
    V_hi: even loads V[:, ov_depth//2:3*ov_depth//4], odd loads V[:, 3*ov_depth//4:ov_depth]
  split_o=False (depth=256): Single V (no V_hi):
    V: even loads V[:, 0:ov_depth//2], odd loads V[:, ov_depth//2:ov_depth]

Q is per-CTA: even loads Q[0:BM, :], odd loads Q[BM:PairBM, :].

All TMA loads are non-multicast. `cta_group=2` on the K/V TMAs (inside
`tma_copy_k` / `tma_copy_v`) tells the shared cluster barrier to
accumulate bytes from both CTAs. Only the leader CTA calls
`expect_bytes` and waits.

Mask computations use PairBM (BM*2) so both CTAs make identical skip
decisions. If one CTA skips a tile and the other doesn't, barriers desync.
"""

from std.math import ceildiv
from std.sys import size_of
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import SharedMemBarrier
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem
from .barriers import Depth512MBars
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    expect_bytes_pred,
    StagedPipeline,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    kv_tma_fold_chunks,
)
from nn.attention.gpu.nvidia.common import (
    KVTMATile,
    MHAPosition,
    OptionalPointer,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from nn.attention.mha_utils import OptionallyStaticInt, _is_decoding
from std.utils.index import Index


@always_inline
def depth512_load[
    KVLUTType: MHAOperand,
    MaskType: MHAMask,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    is_leader: Bool,
](
    smem: Depth512AttentionSMem[config=config],
    score_row: UInt32,
    num_keys: UInt32,
    seq_info: SeqInfo,
    max_seq_len: MaxSeqLenType,
    mask: MaskType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BM=config.BM,
        depth=config.qk_depth,
        group=config.group,
        decoding=False,
        fuse_gqa=config.fuse_gqa,
        num_qk_stages=config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=kv_sub_tile_rows(config.BN // 2, KVLUTType.page_size),
        BK=config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=kv_sub_tile_rows(config.BK1, KVLUTType.page_size),
        BK=config.v_cols_per_cta,
    ],
    kv_lut: KVLUTType,
):
    """Issues TMA loads for one pair-CTA attention tile's Q, K, and V data.

    Drives the producer side of the staged KV pipeline for a single
    attention tile, loading Q (per-CTA), K (per-CTA half of BN), and V
    (depth-split sub-tiles) into SMEM via non-multicast TMA copies.
    The peeled-first iteration co-arrives Q with the first K depth stage
    on a shared barrier so the MMA can start immediately; the main loop
    issues full-tile K/V loads; the peeled-last iteration handles
    partial pages when `page_size < BN`. Mask checks use `PairBM` so both
    CTAs make identical skip decisions and pipeline barriers stay
    synchronized.

    Parameters:
        KVLUTType: Paged key-value lookup table type supplying the element
            `dtype`, `page_size`, and row-to-page mappings used by `kv_lut`.
        MaskType: Attention mask type driving per-tile skip and load
            decisions via `start_column`, `last_masked_set_end`, and
            `status` queries.
        qkv_dtype: Element `DType` of the Q, K, and V tensors; must match
            `KVLUTType.dtype` and `config.qkv_dtype`.
        config: Depth-512 SM100 pair-CTA attention configuration providing
            tile sizes (`BM`, `BN`, `BK0`, `BK1`), stage counts, swizzle
            mode, GQA grouping, and `split_o` controls.
        ValidLengthType: Optional pointer type for per-sequence valid
            lengths; `is_null` is `False` for ragged variable-length
            sequences.
        _is_cache_length_accurate: Whether the reported KV cache length
            exactly matches the count of valid tokens.
        MaxSeqLenType: Type of the maximum sequence length, either a
            comptime static `Int` or a runtime value; selects the decoding
            vs. prefill path via `_is_decoding`.
        is_leader: Whether this CTA is the leader (even-ranked) CTA in
            the pair-CTA cluster; the leader issues `expect_bytes` and
            selects the first half of K/V rows.

    Args:
        smem: Shared-memory allocator and stage buffers for the kernel.
        score_row: Starting row of the Q tile within the current sequence.
        num_keys: Number of valid KV keys for the current sequence.
        seq_info: Per-sequence metadata (prompt index, head index, etc.).
        max_seq_len: Maximum sequence length (static or dynamic).
        mask: Attention mask driving per-tile skip/load decisions.
        q_tma_op: TMA descriptor for Q tile loads.
        k_tma_op: TMA descriptor for K tile loads (per-CTA half).
        v_tma_op: TMA descriptor for V tile loads (per-sub-tile).
        kv_lut: Paged key-value lookup table supplying row-to-page mappings.
    """
    comptime assert KVLUTType.dtype == config.qkv_dtype
    comptime qkv_type = KVLUTType.dtype
    comptime BM = config.BM
    comptime BN = config.BN
    comptime BK0 = config.BK0
    comptime BK1 = config.BK1
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime num_kv_stages = config.num_kv_stages
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    comptime BM_eff: Int = config.BM_eff()
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group = config.cta_group
    comptime qkv_size = size_of[qkv_type]()

    # Unified paged-rows covering the FULL BN. K uses its per-CTA half
    # internally (selected by is_leader); V uses sub-tile selection
    # (num_v_sub_tiles=num_pv_stages) to carve out each pv_stage's rows.
    # Storage is `BN / eff_page` entries per instance — populated once
    # per tile by `kv_lut.populate[BN]` and reused by every subsequent
    # K and V TMA in the same tile.
    comptime KVPagedRows = PagedRowIndices[
        BN=BN,
        page_size=page_size,
        pair_cta=True,
        is_leader=is_leader,
    ]

    # Full pair-CTA M dimension for mask computations.
    # CRITICAL: Both CTAs must use the same M so they make identical
    # skip/load decisions. If one CTA skips a tile and the other doesn't,
    # pipeline barriers desync and the kernel hangs.
    # When fuse_gqa, the tile covers fewer seq positions (BM_eff per CTA).
    comptime PairBM = BM * 2
    comptime PairBM_mask = BM_eff * 2

    # Alignment of `kv_row` produced by mask-driven iteration.
    # Used by `kv_lut.populate` to pick the largest legal SIMD chunk.
    comptime base_alignment: Int = MaskType.start_column_alignment[
        PairBM_mask, BN, page_size
    ]()

    comptime PositionType = MHAPosition[
        PairBM,
        BN,
        config.qk_depth,
        config.qk_depth,  # padded_qk_depth = qk_depth for depth512
        config.num_q_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ]

    # ---- CTA identity (comptime) --------------------------------------------
    # is_leader hoisted from kernel.mojo; cta_rank derived as a comptime.
    comptime cta_rank: Int = 0 if is_leader else 1

    # ---- TileTensor types for TMA destinations ------------------------------
    # TMA only uses .ptr — flat row_major TileTensor is sufficient.

    comptime q_elems = type_of(q_tma_op).tile_shape[0] * type_of(
        q_tma_op
    ).tile_shape[1] * type_of(q_tma_op).tile_shape[2]
    comptime QType = TileTensor[
        KVLUTType.dtype,
        type_of(tt_row_major[q_elems]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    # kv_elems: per-stage SMEM element count, used for pipeline stage indexing.
    # Each slot holds EITHER a K sub-tile OR a V sub-tile (same size):
    #   K: (BN//2) rows × BK0 cols per CTA
    #   V: BK1 rows × v_cols_per_cta cols per CTA
    # Both equal (BN//2)*BK0 because BK1 = BN//2 and v_cols_per_cta = BK0 in
    # this config family (see Depth512SM100Config).
    comptime kv_elems = (BN // 2) * BK0

    # ---- Byte sizes for expect_bytes ----------------------------------------

    comptime q_stage_elements = BM * BK0
    comptime q_stage_bytes = q_stage_elements * qkv_size
    # Per-CTA bytes: K tile is BN//2 rows × BK0 cols, V tile is BK1 rows
    # × v_cols_per_cta cols. cta_group multiplier below accounts for
    # both CTAs in the cluster.
    comptime k_stage_bytes = (BN // 2) * BK0 * qkv_size
    comptime v_stage_bytes = BK1 * config.v_cols_per_cta * qkv_size
    comptime qk_expect_bytes = cta_group * (q_stage_bytes + k_stage_bytes)
    comptime k_expect_bytes = cta_group * k_stage_bytes
    comptime v_expect_bytes = cta_group * v_stage_bytes

    # Partial-page handling: when page_size < BN, a tile may span more
    # pages than the sequence has allocated. K per CTA covers BK1 rows
    # starting at kv_row + cta_rank*BK1; V at pv_stage `s` covers BK1
    # rows starting at kv_row + s*BK1. Either half can exceed num_keys
    # on the last tile even when page_size >= BK1 (num_pages == 1).
    comptime k_needs_partial = page_size > 0 and page_size < BN
    comptime v_needs_partial = page_size > 0 and page_size < BN
    comptime needs_partial = k_needs_partial or v_needs_partial
    # Per-page byte sizes for partial expect_bytes. Use the TMA tile's
    # row count per sub-tile (= min(BK1, page_size)), which matches
    # what `tma_copy_k` / `tma_copy_v` issue per TMA.
    comptime k_tma_tile_rows = kv_sub_tile_rows(BN // 2, page_size)
    comptime v_tma_tile_rows = kv_sub_tile_rows(BK1, page_size)
    comptime k_bytes_pp = BK0 * k_tma_tile_rows * qkv_size
    comptime v_bytes_pp = config.v_cols_per_cta * v_tma_tile_rows * qkv_size

    # Depth-chunk TMA fold (SM100). MUST match `mha_sm100_depth512_dispatch`
    # which builds `k_tma_op` / `v_tma_op` with the same `kv_tma_fold_chunks`
    # values (single source of truth). The matching descriptor rank and
    # issue-coord rank are guaranteed because the same comptime predicate drives
    # both builder and issue site. The per-issue byte total is unchanged: one
    # folded TMA carries the same bytes as the N per-chunk TMAs.
    #
    # K: box_rows == k_tma_tile_rows, smem_BN == BN // 2 (K's per-CTA tile_rows),
    # mirroring the issue site's `smem_BN=BN // 2`.
    comptime k_fold_chunks = kv_tma_fold_chunks[
        qkv_type,
        config.swizzle_mode,
        BK=BK0,
        head_size=config.qk_depth,
        box_rows=k_tma_tile_rows,
        smem_BN=BN // 2,
        page_size=page_size,
    ]()
    # V: folds the `v_cols_per_cta` depth columns. box_rows == v_tma_tile_rows,
    # smem_BN == BK1 (V's per-sub-tile tile_rows = BN // num_pv_stages); the fold
    # is byte-equivalent because `_tma_copy_kv_impl` writes V chunks at stride
    # `tile_rows * gran == BK1 * gran` and the rank-4 box reproduces that stride.
    comptime v_fold_chunks = kv_tma_fold_chunks[
        qkv_type,
        config.swizzle_mode,
        BK=config.v_cols_per_cta,
        head_size=config.ov_depth,
        box_rows=v_tma_tile_rows,
        smem_BN=BK1,
        page_size=page_size,
    ]()

    # ---- SMEM pointers ------------------------------------------------------

    var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.q_smem()
    )
    var kv_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.kv_smem_base()
    )

    # ---- Pipeline setup -----------------------------------------------------

    var mbars = Depth512MBars[num_kv_stages, config.split_o](smem.mbar_base())
    comptime KVPipeType = StagedPipeline[num_kv_stages, 1]
    var kv_pipeline: KVPipeType = {mbars.get_kv_mbars()}
    kv_pipeline.state._phase = 1  # producer starts at phase 1

    # ---- GMEM coordinates ---------------------------------------------------

    var q_gmem_row: UInt32 = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )
    # Each CTA loads its own BM rows of Q.
    # With fuse_gqa, BM_eff seq positions per CTA (not BM physical rows).
    q_gmem_row += UInt32(cta_rank) * UInt32(BM_eff)

    var q_head_idx: UInt32 = seq_info.head_idx
    var kv_head_idx: UInt32
    comptime if fuse_gqa:
        kv_head_idx = seq_info.head_idx
    else:
        kv_head_idx = seq_info.head_idx // UInt32(group)

    var e = elect()

    @__parameter
    @always_inline
    def _kv_num_valid_pages(current_kv_row: UInt32) -> UInt32:
        """Valid paged entries in a BK1-row range starting at `current_kv_row`.

        Used for both K's per-CTA half and V's per-pv_stage half (both
        are BK1 rows wide). When `num_pages == 1` (page_size >= BN),
        returns 0 or 1 depending on whether this BK1 range intersects
        `num_keys`; when `num_pages >= 2`, returns min(half_num_pages,
        ceildiv(remaining, eff_page)).
        """
        # Half count: num_pages // 2 when >= 2; 1 when num_pages == 1
        # (single page shared between halves via intra-page offset).
        comptime half_num_pages: Int = (
            KVPagedRows.num_pages // 2 if KVPagedRows.num_pages >= 2 else 1
        )
        if current_kv_row >= num_keys:
            return UInt32(0)
        return min(
            UInt32(half_num_pages),
            UInt32(
                ceildiv(Int(num_keys - current_kv_row), KVPagedRows.eff_page)
            ),
        )

    var kv_row: UInt32 = mask.start_column[PairBM_mask, BN, page_size](
        seq_info.prompt_idx, score_row
    )
    var iter_count: UInt32 = (
        mask.last_masked_set_end[PairBM_mask, BN, page_size](
            seq_info.prompt_idx, score_row, num_keys
        )
        - 1
    )

    # V depth-column offsets (comptime, depend on cta_rank and split_o).
    comptime v_lo_col_offset: Int = cta_rank * config.v_cols_per_cta
    comptime v_hi_col_offset: Int = config.ov_depth // 2 + cta_rank * (
        config.v_cols_per_cta
    )
    comptime v_col_offset: Int = cta_rank * config.v_cols_per_cta

    # Per-half valid-page counts. Same on both CTAs; leader uses
    # kv_nvp_0 for its K, peer uses kv_nvp_1. V sub-tile `s` uses
    # kv_nvp_0 if s==0 else kv_nvp_1. In the non-partial path these
    # are unused (TMA loops are comptime-unrolled) — but we still
    # materialize them so the partial-path ternaries compile.
    comptime half_num_pages_ct: Int = (
        KVPagedRows.num_pages // 2 if KVPagedRows.num_pages >= 2 else 1
    )
    var kv_nvp_0: UInt32
    var kv_nvp_1: UInt32
    comptime if needs_partial:
        kv_nvp_0 = _kv_num_valid_pages(kv_row)
        kv_nvp_1 = _kv_num_valid_pages(kv_row + UInt32(BK1))
    else:
        kv_nvp_0 = UInt32(half_num_pages_ct)
        kv_nvp_1 = UInt32(half_num_pages_ct)

    # One paged-rows instance per tile; populated above each K loop
    # via `kv_lut.populate[BN]` and reused by every K depth stage and
    # the subsequent V loop. Declared here (without init) so
    # `_load_v_stage` can capture it; first assignment happens before
    # any read.
    var kv_paged_rows: KVPagedRows

    # Mask check uses PairBM (= BM*2 or BM_eff*2 for fuse_gqa) so both
    # CTAs make identical skip decisions. If one CTA skips a tile and
    # the other doesn't, the pipeline barriers desync and the kernel
    # hangs or produces wrong results.
    comptime check_mask = mask.nonfull_sets[PairBM_mask, BN]()[
        0
    ] == TileMaskStatus.UNKNOWN_MASK

    # ---- V load helper (peeled + loop share this) ----------------------------

    @__parameter
    @always_inline
    def _load_v_stage[
        pv_stage: Int
    ](depth_col_offset: Int, v_nvp: UInt32,):
        """Load one V pv_stage using the shared kv_paged_rows.

        With `oob_fill_pages=True` on the partial path, OOB-coord TMAs
        zero-fill the SMEM rows past `v_nvp` valid pages, so the full
        BN-row V tile holds finite data before the MMA reads it. This
        prevents `0 * non-finite = NaN` propagation in `O += P * V`
        when masked V rows would otherwise contain stale or
        uninitialized SMEM (most common when this is the very first
        write to the SMEM slot, typically the only iter is partial,
        i.e. `seq_len <= BN`). Because each OOB TMA still arrives at
        `mbar` with its byte count, we always set the full
        `v_expect_bytes` regardless of `v_needs_partial`.
        """
        kv_pipeline.producer_acquire()
        var mbar = kv_pipeline.producer_mbar()

        comptime if is_leader:
            expect_bytes_pred(mbar, Int32(v_expect_bytes), e)

        kv_paged_rows.tma_copy_v[
            needs_partial=v_needs_partial,
            num_v_sub_tiles=num_pv_stages,
            v_sub_tile_idx=pv_stage,
            oob_fill_pages=v_needs_partial,
            fold_chunks=v_fold_chunks,
        ](
            v_tma_op,
            kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            num_valid_pages=v_nvp,
            depth_offset=UInt32(depth_col_offset),
        )
        kv_pipeline.state.step()

    # ---- K load helper (peeled-first, main, peeled-last share this) ---------

    @__parameter
    @always_inline
    def _produce_k[
        partial: Bool,
        qk_stage: Int = 0,
        with_q: Bool = False,
    ](paged_rows: KVPagedRows, kv_nvp_0: UInt32 = 0, kv_nvp_1: UInt32 = 0,):
        """Produce one K depth stage.

        `partial`: forward to `tma_copy_k`; partial-page TMA when True.
        `qk_stage`: depth stage index, controlling the BK0-multiple
            depth offset.
        `with_q`: True only at the peeled-first iteration where Q
            co-arrives on the same mbar. Implies (a) Q stage bytes
            bundle into `expect_bytes_pred`, and (b) `producer_acquire()`
            is skipped (init pre-acquired this slot; calling acquire
            here would phase-mismatch). The Q TMA itself stays at the
            call site since `fuse_gqa` branching is per-call.

        `paged_rows`: paged-rows handle from `kv_lut.populate(...)`.
            Caller hoists the populate ABOVE the qk_stage loop and passes
            the result to every helper call in the loop.

        `kv_nvp_0`, `kv_nvp_1`: per-CTA-half valid-page counts. Used for
            `expect_bytes_pred` when partial=True (sum) and for the
            per-CTA `k_num_valid_pages` (selected by `is_leader`).
            Defaults to 0, fine for partial=False since both are unused
            (full-tile bytes from `k_expect_bytes`; TMA loop is
            comptime-unrolled).
        """
        comptime d_idx = qk_stage * BK0
        comptime if not with_q:
            kv_pipeline.producer_acquire()
        var mbar = kv_pipeline.producer_mbar()

        comptime if is_leader:
            var bytes: Int32 = Int32(
                cta_group * q_stage_bytes
            ) if with_q else Int32(0)
            comptime if partial:
                bytes += Int32(k_bytes_pp) * Int32(kv_nvp_0 + kv_nvp_1)
            else:
                bytes += Int32(k_expect_bytes)
            expect_bytes_pred(mbar, bytes, e)

        var k_nvp = kv_nvp_0 if is_leader else kv_nvp_1
        paged_rows.tma_copy_k[
            needs_partial=partial,
            smem_BN=BN // 2,
            fold_chunks=k_fold_chunks,
        ](
            k_tma_op,
            kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            k_num_valid_pages=k_nvp,
            depth_offset=UInt32(d_idx),
        )
        kv_pipeline.state.step()

    # ---- Peeled first iteration: Q + K depth stages -------------------------
    # Q is loaded once (num_qk_stages depth stages) co-arrived with K depth
    # stages on the same barriers. The MMA can start Q@K' as soon as the
    # first Q+K stage is ready.

    # Populate kv_paged_rows once for this tile; reused by every K
    # depth stage below and by the subsequent V loop (which captures
    # kv_paged_rows from this scope).
    kv_paged_rows = kv_lut.populate[BN, base_alignment, True, is_leader](
        seq_info.prompt_idx, kv_row
    )
    comptime for qk_stage in range(num_qk_stages):
        comptime d_idx = qk_stage * BK0
        var mbar = kv_pipeline.producer_mbar()

        # Both CTAs: load Q depth stage (non-multicast — each CTA
        # loads its own BM rows). `cta_group=2` is critical: it
        # makes the PTX emit `cp.async.bulk.tensor.Nd.cta_group::2`,
        # which routes the mbarrier arrival to the LEADER CTA's
        # barrier regardless of which CTA issued the TMA. Without
        # it the peer's Q bytes deposit to peer's own barrier and
        # the leader's `expect_bytes(cta_group * q_stage_bytes)`
        # wait hangs.
        # Elect-predicated in-PTX via the _elect overload; no Mojo-level
        # `if e != 0:` branch here — the TMA fires only on the elected lane.
        # Q TMA must run before `_produce_k`'s `state.step()` since
        # both writes target the same mbar; `with_q=True` skips
        # producer_acquire and bundles Q bytes into expect_bytes_pred.
        comptime if fuse_gqa:
            q_tma_op.async_copy_4d_elect[cta_group=cta_group](
                QType(
                    q_smem + q_stage_elements * qk_stage,
                    tt_row_major[q_elems](),
                ),
                mbar[],
                (d_idx, 0, Int(kv_head_idx), Int(q_gmem_row)),
                e,
            )
        else:
            q_tma_op.async_copy_3d_elect[cta_group=cta_group](
                QType(
                    q_smem + q_stage_elements * qk_stage,
                    tt_row_major[q_elems](),
                ),
                mbar[],
                (d_idx, Int(q_head_idx), Int(q_gmem_row)),
                e,
            )

        _produce_k[
            partial=k_needs_partial,
            qk_stage=qk_stage,
            with_q=True,
        ](kv_paged_rows, kv_nvp_0, kv_nvp_1)

    # ---- Peeled first iteration: V BN stages -----------------------------------
    # split_o: V_lo and V_hi occupy separate pipeline slots, each
    # [BK1, v_cols_per_cta]. !split_o: single V.

    comptime for pv_stage in range(num_pv_stages):
        var v_nvp = kv_nvp_0 if pv_stage == 0 else kv_nvp_1
        comptime if config.split_o:
            _load_v_stage[pv_stage](v_lo_col_offset, v_nvp)
        else:
            _load_v_stage[pv_stage](v_col_offset, v_nvp)

    # ---- Peeled first iteration: V_hi BN stages (split_o only) ---------------

    comptime if config.split_o:
        comptime for pv_stage in range(num_pv_stages):
            var v_nvp = kv_nvp_0 if pv_stage == 0 else kv_nvp_1
            _load_v_stage[pv_stage](v_hi_col_offset, v_nvp)

    # ---- Main KV producer loop ----------------------------------------------

    var main_iters = iter_count
    comptime if needs_partial:
        if main_iters > 0:
            main_iters -= 1

    while main_iters != 0:
        main_iters -= 1
        kv_row += UInt32(BN)

        # Mask check: skip fully-masked tiles.
        # CRITICAL: Uses PairBM (BM*2), NOT per-CTA BM.
        # Both CTAs must make identical skip/load decisions to stay
        # synchronized, since the pipeline barriers and MMA are coordinated
        # across the pair. If one CTA skips and the other doesn't, barriers
        # desync and the kernel hangs or produces wrong results.
        comptime if check_mask:
            if (
                mask.status(
                    seq_info.prompt_idx,
                    Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                    Index[dtype=DType.int32](PairBM_mask, BN),
                )
                == TileMaskStatus.FULL_MASK
            ):
                continue

        # ---- K depth stages (num_qk_stages loads) ----
        # Populate once per tile; reused by every K depth stage and the
        # subsequent V loop. Full tile (no partial) in main loop.
        kv_paged_rows = kv_lut.populate[BN, base_alignment, True, is_leader](
            seq_info.prompt_idx, kv_row
        )
        comptime for qk_stage in range(num_qk_stages):
            _produce_k[partial=False, qk_stage=qk_stage](kv_paged_rows)

        # ---- V BN stages (reuse kv_paged_rows) ----
        comptime for pv_stage in range(num_pv_stages):
            comptime if config.split_o:
                _load_v_stage[pv_stage](
                    v_lo_col_offset,
                    UInt32(
                        KVPagedRows.num_pages // num_pv_stages
                    ) if KVPagedRows.num_pages
                    >= num_pv_stages else UInt32(1),
                )
            else:
                _load_v_stage[pv_stage](
                    v_col_offset,
                    UInt32(
                        KVPagedRows.num_pages // num_pv_stages
                    ) if KVPagedRows.num_pages
                    >= num_pv_stages else UInt32(1),
                )

        # ---- V_hi BN stages (split_o only) ----
        comptime if config.split_o:
            comptime for pv_stage in range(num_pv_stages):
                _load_v_stage[pv_stage](
                    v_hi_col_offset,
                    UInt32(
                        KVPagedRows.num_pages // num_pv_stages
                    ) if KVPagedRows.num_pages
                    >= num_pv_stages else UInt32(1),
                )

    # ---- Peeled last iteration (partial pages possible) ---------------------

    comptime if needs_partial:
        if iter_count > 0:
            kv_row += UInt32(BN)
            var _skip_last = False
            comptime if check_mask:
                if (
                    mask.status(
                        seq_info.prompt_idx,
                        Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                        Index[dtype=DType.int32](PairBM_mask, BN),
                    )
                    == TileMaskStatus.FULL_MASK
                ):
                    _skip_last = True
            if not _skip_last:
                # Recompute per-half valid-page counts for the last tile.
                kv_nvp_0 = _kv_num_valid_pages(kv_row)
                kv_nvp_1 = _kv_num_valid_pages(kv_row + UInt32(BK1))

                # K: populate once per tile; reused by every K depth
                # stage and the subsequent V loop.
                kv_paged_rows = kv_lut.populate[
                    BN, base_alignment, True, is_leader
                ](seq_info.prompt_idx, kv_row)
                comptime for qk_stage in range(num_qk_stages):
                    _produce_k[
                        partial=k_needs_partial,
                        qk_stage=qk_stage,
                    ](kv_paged_rows, kv_nvp_0, kv_nvp_1)

                # V: partial, reuse kv_paged_rows.
                comptime for pv_stage in range(num_pv_stages):
                    var v_nvp = kv_nvp_0 if pv_stage == 0 else kv_nvp_1
                    comptime if config.split_o:
                        _load_v_stage[pv_stage](v_lo_col_offset, v_nvp)
                    else:
                        _load_v_stage[pv_stage](v_col_offset, v_nvp)

                comptime if config.split_o:
                    comptime for pv_stage in range(num_pv_stages):
                        var v_nvp = kv_nvp_0 if pv_stage == 0 else kv_nvp_1
                        _load_v_stage[pv_stage](v_hi_col_offset, v_nvp)
