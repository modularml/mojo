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
"""TMA load warp logic for FA4 (SM100 Flash Attention)."""

from std.math import ceildiv
from std.sys import size_of
from max.gpu.memory import CacheEviction
from layout.tma_async import SharedMemBarrier
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    ExplicitTMEMCrossP,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    expect_bytes_pred,
    KProducerPipeline,
    VProducerPipeline,
    StagedPipeline,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    splitk_window,
    splitk_partition_idx,
    splitk_num_partitions,
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
from std.utils.static_tuple import StaticTuple
from .smem import SM100AttentionSMem


@always_inline
def fa4_load[
    KVLUTType: MHAOperand,
    MaxSeqLenType: OptionallyStaticInt,
    MaskType: MHAMask,
    //,
    config: FA4Config,
    *,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    is_leader: Bool,
    # Workspace (traditional/unfused) split-K: when True, window the KV by a
    # RUNTIME partition count (`ws_num_partitions`) even though
    # `config.splitk_partitions == 1` (no launch cluster). Defaulted so every
    # non-workspace caller is byte-identical.
    workspace_split: Bool = False,
    # Effective cross-stage-P switch, computed once at the kernel level where
    # config, MaskType and the store shape are all visible. Defaults True so
    # the config-level gate alone decides for callers that do not thread it;
    # the MHA kernel passes its own narrower predicate.
    crossp_effective: Bool = True,
](
    smem: SM100AttentionSMem[config],
    score_row: UInt32,
    num_keys: UInt32,
    seq_info: SeqInfo,
    max_seq_len: MaxSeqLenType,
    mask: MaskType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BM=config.BM // config.num_q,
        depth=config.qk_depth,
        group=config.group,
        decoding=False,
        fuse_gqa=config.fuse_gqa,
        num_qk_stages=config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=kv_sub_tile_rows(config.k_rows_per_cta(), KVLUTType.page_size),
        BK=config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        # V TMA box geometry per layout via `v_tma_box_rows()` /
        # `v_tma_box_cols()` -- the one selector shared with dispatch's
        # `create_tma_tile` and the kernel `VTMAOpType`, so the box shape folds
        # identically at every V TMA type-param site.
        BN=config.v_tma_box_rows(KVLUTType.page_size),
        BK=config.v_tma_box_cols(),
    ],
    kv_lut: KVLUTType,
    ws_num_partitions: UInt32 = 1,
):
    """Issues the TMA loads for the Q, K, and V tiles consumed by one FA4
    attention warp group on SM100.

    Drives the producer side of the K/V pipeline across both fused-KV and
    split-KV modes, handling pair-CTA half-tile offsets, split-K windowing,
    partial-page bounds for sub-page KV tiles, and mask-driven iteration.
    Q is issued on the K barriers (fused with the first K stage) and on a
    separate Q1 barrier in two-Q mode. K and V TMA coordinates are produced
    from a shared `kv_lut.populate` so V reuses K's paged-row indices without
    a second LUT lookup.

    Parameters:
        KVLUTType: Paged KV-cache lookup table type supplying the element
            `dtype`, `page_size`, and row-to-page mappings used by
            `kv_lut` (inferred).
        MaxSeqLenType: Type of the maximum sequence length, either a comptime
            static `Int` or a runtime value; selects the decoding vs.
            prefill path via `_is_decoding` (inferred).
        MaskType: Attention mask type driving per-tile skip and load
            decisions via `start_column`, `last_masked_set_end`, and
            `status` queries (inferred).
        config: SM100 FA4 attention configuration providing tile sizes
            (`BM`, `BN`, `BK0`), stage counts, swizzle mode, GQA
            grouping, and split-K controls.
        ValidLengthType: Optional pointer type for per-sequence valid
            lengths; `is_null` is `False` for ragged variable-length
            sequences.
        _is_cache_length_accurate: Whether the reported KV cache length
            exactly matches the count of valid tokens.
        is_leader: Whether this CTA is the leader (even-ranked) CTA in a
            pair-CTA cluster, or always `True` in single-CTA mode; the
            leader issues `expect_bytes` and selects the first half of
            K/V rows.
        workspace_split: When `True`, windows the KV range by the runtime
            `ws_num_partitions` count for workspace (unfused) split-K even
            though `config.splitk_partitions == 1`; defaulted so every
            non-workspace caller is byte-identical.
        crossp_effective: Whether cross-stage P is enabled for this
            instantiation. Computed once at the kernel level from the
            config, mask type, and store shape, then threaded
            identically into the load, MMA, and softmax warps.

    Args:
        smem: Shared-memory allocator for Q, K, V, and barrier storage.
        score_row: Row index of this tile within the score matrix.
        num_keys: Number of valid KV keys in the sequence.
        seq_info: Per-sequence metadata (prompt index, head index, etc.).
        max_seq_len: Maximum sequence length, optionally a compile-time constant.
        mask: Attention mask governing which KV tiles are skipped.
        q_tma_op: TMA descriptor for the Q tile load.
        k_tma_op: TMA descriptor for the K tile load.
        v_tma_op: TMA descriptor for the V tile load.
        kv_lut: Paged KV-cache lookup table producing per-tile row indices.
        ws_num_partitions: Runtime split-K partition count that windows the
            KV range when `workspace_split` is `True`; `1` disables windowing.
    """
    comptime assert KVLUTType.dtype == config.qkv_dtype
    comptime qkv_type = KVLUTType.dtype
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime num_q: Int = config.num_q
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    # For pair-CTA, use PairBM so both CTAs make identical mask decisions.
    comptime BM_mask: Int = config.PairBM_eff()
    comptime page_size = KVLUTType.page_size
    # Alignment of `kv_row` values produced by mask-driven iteration.
    # Used by `kv_lut.populate` to pick the largest legal SIMD chunk.
    comptime base_alignment: Int = MaskType.start_column_alignment[
        BM_mask, BN, page_size
    ]()
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group: Int = config.cta_group()
    comptime pair_cta: Bool = config.pair_cta
    comptime assert pair_cta or is_leader

    # Unified paged-rows type shared by shared-KV and non-shared-KV: populate
    # covers V's full tile so V can consume K's pre-populated indices
    # with no lazy LUT lookup. In non-pair-CTA mode `num_pages` is
    # simply `BN / eff_page`. In pair-CTA mode the struct's
    # `is_leader`/`pair_cta` params select K's half at comptime (index
    # shift for `num_pages >= 2`, intra-page row shift for
    # `num_pages == 1`); storage is always V-sized so V reuses the same
    # array without re-LUT.
    comptime KVPagedRows = PagedRowIndices[
        BN=BN,
        page_size=page_size,
        pair_cta=pair_cta,
        is_leader=is_leader,
    ]
    # Both V producers that need a SUB-RANGE of the tile (Layout-E's partitions,
    # shared-key's key chunks) carve it with `sub_rows`, which needs this LUT's
    # own populate base to be page-aligned. It always is: every mask's
    # `start_column_alignment` returns `BN` or `min(page_size, BN)` and must
    # divide `BN`, while `eff_page` is `BN` (`page_size` 0 or `>= BN`) or
    # `page_size`. A `page_size` breaking this already breaks `populate` itself
    # (`num_pages * eff_page != BN`), so make it a build error rather than
    # silently wrong keys.
    comptime assert (
        base_alignment % KVPagedRows.eff_page == 0
    ), "V sub-range reuse needs a page-aligned tile LUT base"
    # K's per-CTA TMA page count. Must derive from K's tile size
    # (k_rows_per_cta) rather than `num_pages // 2`: when
    # `page_size >= BN` (e.g. ps256 hs128), `num_pages = 1` but K's
    # TMA still issues once per CTA (it covers BN/2 rows from a single
    # page), so `k_pages_per_cta = 1`, not 0.
    comptime k_pages_per_cta = kv_num_sub_tiles(
        config.k_rows_per_cta(), page_size
    )

    var mbars = smem.misc_mbars()
    # Cross-stage P (2Q shared-KV, non-WS): fills FlashInfer's K-ahead sequence
    # K0,K1,V0,K2,V1,...  (upstream FlashInfer `fmha.py`; no line anchor, it
    # is another project's file and would rot) instead of the strict
    # K0,V0,K1,V1,... interleave, so K is always one BN ahead of its V.
    # `crossp_effective` is the kernel-level predicate: it already folds in
    # everything this file's regime guard below demands, so the default path
    # can never reach that guard. All three warps take the SAME value.
    comptime CrossP = crossp_effective and type_of(mbars).CrossP_enabled

    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.qk_depth,
        config.padded_qk_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    comptime KPipeType = KProducerPipeline[KVLUTType.dtype, config]
    comptime VPipeType = VProducerPipeline[KVLUTType.dtype, config]

    # If two-qo, we produce qkv in a pattern of
    # q0 & k0, q1, v0, k1, v1, k2, v2...
    # TMA only uses .ptr — flat row_major TileTensor is sufficient.
    # Per-TMA-call element count. In 2Q (num_q=2) this is HalfBM * BK0
    # (one of two Q-half TMAs); in 1Q (num_q=1) this is BM * BK0 (the
    # single full-Q TMA). The two numerically coincide at 128 * BK0
    # because BM=128 in 1Q and HalfBM=128 in 2Q, so q_bytes and the K0
    # barrier's expect_bytes math are invariant between modes.
    # (fused: BM//(2*group) * group * BK0 = HalfBM * BK0)
    comptime q_elements = (BM // num_q) * config.BK0
    comptime QType = TileTensor[
        KVLUTType.dtype,
        type_of(tt_row_major[q_elements]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    var kv_head_idx: UInt32
    comptime if fuse_gqa:
        kv_head_idx = seq_info.head_idx
    else:
        kv_head_idx = seq_info.head_idx // UInt32(group)

    var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.q_smem()
    )
    comptime q_bytes = size_of[qkv_type]() * q_elements

    var q_gmem_row: UInt32 = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )

    # Pair-CTA: each CTA loads different Q positions and half of K/V.
    var k_row_offset = UInt32(0)
    var v_col_offset = 0
    comptime if not is_leader:
        q_gmem_row += UInt32(config.BM_eff())
        k_row_offset = UInt32(BN // 2)
        v_col_offset = config.v_cols_per_cta()

    var e = elect()

    # Default eviction_policy: pair-CTA disallows cache_hint with
    # cta_group=2 in the stdlib TMA intrinsic, so we fall back to
    # EVICT_NORMAL there. Non-pair-CTA keeps EVICT_FIRST (Q is read
    # once per stage then discarded).
    comptime q_default_eviction: CacheEviction = (
        CacheEviction.EVICT_NORMAL if pair_cta else CacheEviction.EVICT_FIRST
    )

    @__parameter
    @always_inline
    def q_async_copy[
        eviction_policy: CacheEviction = q_default_eviction,
    ](
        smem_dst: QType,
        ref[AddressSpace.SHARED] mbar: SharedMemBarrier,
        depth_idx: UInt32 = 0,
    ):
        """Issue Q TMA elect-predicated on `e`. Caller no longer needs
        `if e != 0:` around the call; the TMA fires only on the elected
        lane via the PTX predicate inside `_elect`."""
        comptime if fuse_gqa:
            q_tma_op.async_copy_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                smem_dst,
                mbar,
                StaticTuple[UInt32, 4](depth_idx, 0, kv_head_idx, q_gmem_row),
                e,
            )
        else:
            q_tma_op.async_copy_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                smem_dst,
                mbar,
                StaticTuple[UInt32, 3](
                    depth_idx, seq_info.head_idx, q_gmem_row
                ),
                e,
            )

    # Partial-page handling: when page_size < BN, a BN-sized tile may span
    # more pages than the sequence has allocated. We detect this and use
    # `tma_copy_{k,v}[needs_partial=True]` with a runtime-bounded page
    # count to avoid OOB page lookups.
    comptime needs_partial = page_size > 0 and page_size < BN
    # KV ring co-rotation: emit each odd V one K group later, so the producer
    # emits in the MMA warp's *consumption* order. The ring bound is a SPAN,
    # not a liveness count -- block order puts V_o[n-1] immediately behind
    # K_e[n], forcing N >= 2 * num_qk_stages; the rotation takes that to
    # num_qk_stages, which is what `ring_slots_needed()` now returns.
    #
    # `mma_warp.mojo`'s `_body_1q` is the consumer half. The two must rotate
    # TOGETHER: rotating one file alone is a hang -- or, at T <= 2 where block
    # and consumption order coincide, a green that proves nothing.
    # WS shared sub-tile ring: V depth-tile width (256x64 sub-tile). Folds to the
    # full V width for non-WS (which loads V whole). All V byte/box constants
    # below route through this so the WS path emits one TMA per V depth-tile.
    comptime v_sub_cols = config.v_box_cols()
    comptime k_bytes_pp = config.BK0 * KVPagedRows.eff_page * size_of[
        qkv_type
    ]()
    comptime v_bytes_pp = v_sub_cols * KVPagedRows.eff_page * size_of[
        qkv_type
    ]()

    # Depth-chunk TMA fold (SM100): fold a tile's depth chunks into one rank-4
    # TMA when byte-equivalent (same total bytes as unfolded). MUST match the
    # `create_tma_tile[..., fold_chunks=...]` args in `mha_sm100_dispatch`
    # (single source of truth), or the baked descriptor rank and issue-coord
    # rank disagree.
    #   K: smem_BN == k_rows_per_cta (K's per-CTA tile_rows; == BN single-CTA,
    #      BN//2 pair-CTA).
    comptime k_row_major = config.k_row_major()
    comptime k_fold_chunks = kv_tma_fold_chunks[
        qkv_type,
        config.swizzle_mode,
        BK=config.BK0,
        head_size=config.qk_depth,
        box_rows=kv_sub_tile_rows(config.k_rows_per_cta(), page_size),
        smem_BN=config.k_rows_per_cta(),
        page_size=page_size,
        row_major=k_row_major,
    ]()
    #   V: same `v_tma_box_cols()` / `v_tma_box_rows()` / `v_tma_tile_rows()`
    #      accessors `mha_sm100_dispatch` uses to build `v_tma_op` -- else the
    #      baked descriptor rank and issue-coord rank disagree.
    comptime v_desc_cols = config.v_tma_box_cols()
    comptime v_desc_rows = config.v_tma_box_rows(page_size)
    comptime v_row_major = False if config.m_pack == 2 else config.v_row_major()
    comptime v_fold_chunks = 1 if config.m_pack == 2 else kv_tma_fold_chunks[
        qkv_type,
        config.swizzle_mode,
        BK=v_desc_cols,
        head_size=config.ov_depth,
        box_rows=v_desc_rows,
        smem_BN=config.v_tma_tile_rows(),
        page_size=page_size,
        row_major=v_row_major,
    ]()

    @__parameter
    @always_inline
    def _k_num_valid_pages(current_kv_row: UInt32) -> UInt32:
        """Valid K sub-tile pages at `current_kv_row` (per-CTA range)."""
        if current_kv_row >= num_keys:
            return UInt32(0)
        return min(
            UInt32(k_pages_per_cta),
            UInt32(
                ceildiv(Int(num_keys - current_kv_row), KVPagedRows.eff_page)
            ),
        )

    @__parameter
    @always_inline
    def _v_num_valid_pages(current_kv_row: UInt32) -> UInt32:
        """Valid V sub-tile pages at `current_kv_row` (full BN range)."""
        return min(
            UInt32(KVPagedRows.num_pages),
            UInt32(
                ceildiv(Int(num_keys - current_kv_row), KVPagedRows.eff_page)
            ),
        )

    var kv_row: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_info.prompt_idx, score_row
    )
    var last_masked_set_end: UInt32 = mask.last_masked_set_end[
        BM_mask, BN, page_size
    ](seq_info.prompt_idx, score_row, num_keys)
    # All-masked row (valid_length 0): last_masked_set_end == 0, so the
    # `- 1` below underflows and the load warp over-produces, hanging the KV
    # pipeline. Split-K is guarded below; guard the non-split path here.
    comptime if not (config.num_q == 1 and config.splitk_partitions > 1):
        if last_masked_set_end == UInt32(0):
            return
    var iter_count: UInt32 = last_masked_set_end - 1

    # Split-K (1Q): shift this CTA to its own tile window [cb, ce) of the
    # combined range. Offset kv_row by cb*BN BEFORE the first-tile
    # valid-page counts below (they key on kv_row), and stash the local
    # tile count for the 1Q peel at the `T` site below. Same window as the
    # other warps -- last_masked_set_end == total_iters for check_mask==False
    # masks.
    var part_first_tile: UInt32 = 0
    var part_local_iters: UInt32 = 0
    comptime if config.num_q == 1 and (
        config.splitk_partitions > 1 or workspace_split
    ):
        var _gT: UInt32 = mask.last_masked_set_end[BM_mask, BN, page_size](
            seq_info.prompt_idx, score_row, num_keys
        )
        var _np = splitk_num_partitions[config](ws_num_partitions)
        var _w = splitk_window(
            _gT,
            _np,
            splitk_partition_idx(_np),
        )
        part_first_tile = _w[0]
        part_local_iters = _w[1] - _w[0]
        kv_row += part_first_tile * UInt32(BN)
        # Empty partition (front-load trailing window, T < num_partitions; or an
        # M6 idle CTA): no tiles to load. Return before the peeled Q+K load and
        # the first-tile valid-page counts -- the mma/correction warps take
        # their matching empty-partition returns and softmax stages a neutral
        # identity. The kernel terminal `cluster_sync()` still runs.
        if part_local_iters == 0:
            return
        # Window the non-shared-KV producer's loop count too. The shared-KV 1Q path
        # below uses `T = part_local_iters` directly, but the non-shared-KV path
        # (num_qk_stages > 1, i.e. depth >= 128) drives its peel + main loop
        # off `iter_count` (set above to the FULL last_masked_set_end - 1).
        # Without windowing it, that producer emits the whole [0, T) range while
        # mma/softmax/correction consume only this partition's part_local_iters
        # tiles (they all window total_iters) -> the K/V pipeline producer
        # over-fills and deadlocks on producer_acquire (the depth-128 split-K
        # hang). `iter_count` is the post-first-peel main count, so set it to
        # part_local_iters - 1: total emitted = first-peel(1) + main(iter_count)
        # [+ last-peel(1) when needs_partial] == part_local_iters either way.
        iter_count = part_local_iters - UInt32(1)

    # Valid page counts for the first tile (shared between shared-KV and
    # non-shared-KV). When `needs_partial`, these reflect how many sub-tile
    # pages are actually in-bounds for the sequence; otherwise every
    # sub-tile is assumed fully populated.
    #
    # `k_nvp` is this CTA's count (the half it will TMA). `k_nvp_peer`
    # holds the *other* CTA's count and is read only by the leader when
    # it accumulates `expect_bytes` across the cluster's shared barrier.
    # Non-leader keeps it at 0; the peer's `expect_bytes` branch is
    # comptime-pruned so the value is dead there.
    var k_nvp: UInt32 = UInt32(k_pages_per_cta)
    var k_nvp_peer: UInt32 = UInt32(0)
    var v_nvp: UInt32 = UInt32(KVPagedRows.num_pages)
    comptime if needs_partial:
        k_nvp = _k_num_valid_pages(kv_row + k_row_offset)
        comptime if is_leader and pair_cta:
            k_nvp_peer = _k_num_valid_pages(
                kv_row + UInt32(config.k_rows_per_cta())
            )
        v_nvp = _v_num_valid_pages(kv_row)

    # Full-tile expect_bytes (accounts for both CTAs in pair mode).
    # Shared between shared-KV and non-shared-KV: `KPipeType.bytes` and
    # `VPipeType.bytes` resolve to the same per-CTA values, so we
    # define them once here.
    comptime k_per_cta_bytes = (
        config.BK0 * config.k_rows_per_cta() * size_of[qkv_type]()
    )
    comptime v_per_cta_bytes = (v_sub_cols * BN * size_of[qkv_type]())
    comptime k_expect_bytes = cta_group * k_per_cta_bytes
    comptime v_expect_bytes = cta_group * v_per_cta_bytes
    comptime qk_expect_bytes = cta_group * (q_bytes + k_per_cta_bytes)

    # Mode-shared K/V producer closures. These cover both shared-KV and
    # non-shared-KV call sites (and the inlined main-loop / peeled-last V in
    # non-shared mode). Captures: `is_leader`, `e`, `cta_group`, `q_bytes`,
    # `q_smem`, `q_elements`, `tt_row_major`, `QType`, `q_async_copy`,
    # `k_bytes_pp`, `v_bytes_pp`, `k_per_cta_bytes`, `v_expect_bytes`,
    # `k_nvp_peer`, `kv_head_idx`, `v_col_offset`, `k_tma_op`,
    # `v_tma_op`, `config`. Caller owns `populate`, `smem_ptr`, and the
    # producer-pipeline acquire/step lifecycle.
    @__parameter
    @always_inline
    def _produce_k[
        partial: Bool,
        qk_stage: Int = 0,
        with_q: Bool = False,
    ](
        kv_paged_rows: KVPagedRows,
        smem_ptr: SharedMemPointer[Scalar[qkv_type]],
        mbar: SharedMemPointer[SharedMemBarrier],
        k_num_valid_pages: UInt32,
    ):
        comptime d_idx = qk_stage * config.BK0
        comptime if is_leader:
            comptime q_term = (cta_group * q_bytes if with_q else 0)
            var bytes: Int32 = Int32(q_term)
            comptime if partial:
                bytes += Int32(k_bytes_pp) * Int32(
                    k_num_valid_pages + k_nvp_peer
                )
            else:
                bytes += Int32(cta_group * k_per_cta_bytes)
            expect_bytes_pred(mbar, bytes, e)
        comptime if with_q:
            # Elect-predicated in-PTX by q_async_copy; no if-guard here.
            q_async_copy(
                QType(
                    q_smem + q_elements * qk_stage,
                    tt_row_major[q_elements](),
                ),
                mbar[],
                depth_idx=UInt32(d_idx),
            )
        kv_paged_rows.tma_copy_k[
            needs_partial=partial,
            smem_BN=config.k_rows_per_cta(),
            fold_chunks=k_fold_chunks,
            row_major=k_row_major,
        ](
            k_tma_op,
            smem_ptr,
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            k_num_valid_pages=k_num_valid_pages,
            depth_offset=UInt32(d_idx),
        )

    @__parameter
    @always_inline
    def _produce_v[
        partial: Bool,
        d_tile: Int = 0,
    ](
        kv_paged_rows: KVPagedRows,
        smem_ptr: SharedMemPointer[Scalar[qkv_type]],
        mbar: SharedMemPointer[SharedMemBarrier],
        v_num_valid_pages: UInt32,
    ):
        # Shared-key uses `_produce_v_sk` instead; assert the prune rather than
        # trust it, so an instantiation here is a build error, not a rank-3
        # coord handed to a rank-4 descriptor.
        comptime assert not config.ws_shared_key, (
            "_produce_v is shared-key-dead; the shared-key V producer is"
            " _produce_v_sk"
        )
        # WS: one TMA per V depth-tile; `d_tile` selects the 256x64 sub-tile at
        # gmem depth `d_tile * v_sub_cols` (mirrors K's `qk_stage * BK0`).
        # Non-WS keeps d_tile==0 -> depth_offset == v_col_offset (unchanged).
        comptime v_depth_stage_offset = d_tile * v_sub_cols
        # WS short-KV correctness: OOB-zero-fill the beyond-num_keys page slots
        # of a partial V sub-tile so the FULL 256x64 sub-tile holds finite data
        # before the packed .ws P@V reads it. The .ws P@V contracts the whole
        # sub-tile: a fully-masked quarter's per-quarter row_max == MASK_VALUE
        # gives P==1 (empty quarter), and masked keys in a live quarter give
        # P==0; either way the two-level combine multiplies by scale==0, so
        # stale/uninitialized SMEM propagates `0/1 * non-finite = NaN` (only on
        # the first write to the slot, typically seq_len <= BN). Mirrors the
        # depth-512 FA4 fix (mha_depth512/load_warp.mojo) + types.mojo
        # `tma_copy_v` `oob_fill_pages`. WS-gated -> non-WS is byte-identical.
        comptime v_oob_fill = partial and config.use_ws
        comptime if is_leader:
            var v_bytes: Int32
            comptime if partial:
                comptime if config.use_ws:
                    # OOB-fill issues all v_pages_per_sub_tile TMAs (valid +
                    # OOB), each arriving at the mbar, so expect the FULL bytes.
                    v_bytes = Int32(v_expect_bytes)
                else:
                    v_bytes = Int32(
                        cta_group * v_bytes_pp * Int(v_num_valid_pages)
                    )
            else:
                v_bytes = Int32(v_expect_bytes)
            expect_bytes_pred(mbar, v_bytes, e)
        kv_paged_rows.tma_copy_v[
            needs_partial=partial,
            oob_fill_pages=v_oob_fill,
            fold_chunks=v_fold_chunks,
            row_major=v_row_major,
        ](
            v_tma_op,
            smem_ptr,
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            num_valid_pages=v_num_valid_pages,
            depth_offset=UInt32(v_depth_stage_offset + v_col_offset),
        )

    # Layout-E (m_pack==2) reduction-chunk V producer: `r` selects the
    # reduction-chunk ring slot (one of `num_qk_stages`, mirrors K's
    # `qk_stage`); within it, TWO per-partition natural TMA loads (full
    # depth, `pv_bk_chunk` keys each) land at `p * partition_region_elems`
    # sub-offsets, filling the 32 KB slot as a single CONTIGUOUS B-operand in
    # `mn = p*ov_depth + d` order -- partition-major, depth-minor, no gap
    # between the two partitions' regions.
    #
    # Carves each partition's OWN `partition_keys`-row range out of the SHARED
    # `BN=config.BN` row LUT with `sub_rows` (single-CTA: `pair_cta=False,
    # is_leader=True`, matching Layout-E's `supported()` invariant), rather than
    # selecting the partition through `num_v_sub_tiles`. `PagedRowIndices`'s
    # sub-tile row math (`kv_cache/types.mojo`) is documented correct only for
    # `num_pages >= num_v_sub_tiles` or `num_pages == 1` (the depth-512 MLA
    # precedent); one BN=256 LUT split `m_pack * num_qk_stages` ways would fall
    # in NEITHER case at e.g. `page_size=128` (`num_pages=2 < 4`). A
    # `partition_keys`-row sub-range keeps the
    # `num_v_sub_tiles=num_qk_stages` reduction split inside one of those two
    # documented cases for any `page_size` -- and costs no LUT read, since
    # `sub_rows` reads only rows K already populated.
    comptime partition_keys = config.BN // config.m_pack
    # Per-partition SMEM footprint WITHIN one reduction-chunk ring slot: one
    # reduction chunk (`v_e_chunk_rows()` chunk-keys x `v_e_box_cols()` full
    # depth) = `pv_bk_chunk * padded_ov_depth` elems. This is the REGION size,
    # deliberately NOT clamped by `page_size` the way the TMA box row count is:
    # when paging splits the chunk, `_tma_copy_kv_impl` issues `pages_per_iter`
    # boxes at stride `tma_per_issue_rows * gran` which together tile exactly
    # this region. `tma_copy_v`'s `num_v_sub_tiles`/`v_sub_tile_idx` offset only
    # the SOURCE KV rows, never the destination, so partition `p`'s region is
    # written at `smem_ptr + p*partition_region_elems`; partition 1 sits directly
    # above partition 0 (= 8192 elems @d128 = 16 KB), together filling the 32 KB
    # slot. NOT `partition_keys * ov_depth` -- that is a partition's TOTAL keys
    # (all reduction chunks), which would overrun into the next ring slot and
    # leave this slot's upper half unwritten.
    comptime partition_region_elems = (
        config.v_e_chunk_rows() * config.v_e_box_cols()
    )
    # Layout-E's own page frame. `tma_copy_v` re-derives its per-issue rows and
    # `pages_per_iter` from the PER-PARTITION LUT below (`kv_cache/types.mojo`'s
    # `tma_per_issue_rows` / `pages_per_iter` off `Self.BN == partition_keys`
    # and `num_v_sub_tiles == num_qk_stages`), so a valid-page count handed to
    # it must be measured in THAT frame. The shared BN-wide `KVPagedRows` frame
    # `_v_num_valid_pages` uses (`min(KVPagedRows.num_pages, ...)`) is a
    # different, larger one -- feeding it here lands outside
    # `[0, pages_per_iter]`, the partial dispatch chain's equality tests never
    # match, and every slot loads real page-table data instead of OOB-filling.
    comptime v_e_rows_per_page = config.v_tma_box_rows(page_size)
    comptime v_e_pages_per_chunk = config.v_e_chunk_rows() // v_e_rows_per_page

    @__parameter
    @always_inline
    def _produce_v_e[
        partial: Bool,
        r: Int,
    ](
        rows: KVPagedRows,
        kv_row_base: UInt32,
        smem_ptr: SharedMemPointer[Scalar[qkv_type]],
        mbar: SharedMemPointer[SharedMemBarrier],
    ):
        # `_produce_v_e` is Layout-E-only (m_pack==2 => use_ws), so the non-WS
        # per-page byte count `_produce_v` carries never applies: V bytes are
        # always the full `v_expect_bytes`, and OOB-fill reduces to `partial`.
        comptime v_oob_fill = partial
        comptime if is_leader:
            expect_bytes_pred(mbar, Int32(v_expect_bytes), e)
        comptime for p in range(config.m_pack):
            var p_base = kv_row_base + UInt32(p * partition_keys)
            var p_rows = rows.sub_rows[partition_keys](
                UInt32(p * partition_keys)
            )
            # Each partition owns a DIFFERENT key range, and sub-tile `r` sits
            # `r` reduction chunks into it, so the valid-page count is per
            # (partition, sub-tile) -- a single tile-wide count would leave the
            # upper partition loading pages past `num_keys`. The P@V trim cannot
            # rescue that: `mma_maybe_partial_k` predicates k-blocks on the
            # MMA's shared BK axis while Layout-E's partitions are packed along
            # MMA_N, so ALL packs get one cut. OOB zero-fill is the only
            # protection (`0 * P == 0`), and it engages only when this count is
            # exact. 0 is legal and means "fill every slot" (V's dispatch chain
            # starts at slot 0).
            var chunk_base = p_base + UInt32(r * config.v_e_chunk_rows())
            var p_valid = UInt32(0) if chunk_base >= num_keys else min(
                UInt32(v_e_pages_per_chunk),
                UInt32(ceildiv(Int(num_keys - chunk_base), v_e_rows_per_page)),
            )
            p_rows.tma_copy_v[
                needs_partial=partial,
                num_v_sub_tiles=config.num_qk_stages,
                v_sub_tile_idx=r,
                oob_fill_pages=v_oob_fill,
            ](
                v_tma_op,
                smem_ptr + UInt32(p) * UInt32(partition_region_elems),
                mbar[],
                kv_head_idx=kv_head_idx,
                elect=e,
                num_valid_pages=p_valid,
            )

    # ---- Shared-key (deep heads) key-chunked V producer ----
    # V is cut along KEYS into `v_sk_chunks` chunks of `v_sk_key_chunk` keys,
    # each crossed with `num_o_tiles()` output depth tiles of `v_sk_depth_tile`
    # columns. One (chunk, tile) pair is exactly one ring slot -- the byte
    # identity `pv_key_chunk() * pv_mma_n() == BN * BK0` -- and exactly one P@V
    # MMA's B operand, which is what drops V liveness to a single in-flight slot.
    comptime v_sk_key_chunk = config.pv_key_chunk()
    comptime v_sk_chunks = config.pv_reduction_chunks()
    comptime v_sk_depth_tile = config.pv_mma_n()
    comptime v_sk_rows_per_page = config.v_tma_box_rows(page_size)
    comptime v_sk_pages_per_chunk = config.v_pages_per_chunk(page_size)
    comptime v_sk_oob_fill = config.v_oob_fill_needed(page_size)
    # A key chunk's OWN row LUT type -- `v_sk_key_chunk` rows, single-CTA. NOT
    # `KVPagedRows` (BN rows) directly: selecting the chunk out of the BN-wide
    # LUT with `num_v_sub_tiles = v_sk_chunks` is the documented-wrong case in
    # `kv_cache/types.mojo`, whose sub-tile row math holds only for
    # `num_pages >= num_v_sub_tiles` or `num_pages == 1`. At `page_size=128`,
    # `num_pages == 2 < 4`, so both ternaries there take the wrong arm and
    # chunks 2-3 address past the end of page 0 -- silent wrong data, no assert.
    # `sub_rows` carves the chunk into its own LUT instead, which keeps
    # `num_v_sub_tiles == 1` -- inside the documented case for every
    # `page_size` -- and reads no page table of its own.
    comptime VSKPagedRows = PagedRowIndices[
        BN=v_sk_key_chunk,
        page_size=page_size,
        pair_cta=False,
        is_leader=True,
    ]

    @__parameter
    @always_inline
    def _produce_v_sk[
        partial: Bool,
        t: Int,
    ](
        chunk_rows: VSKPagedRows,
        smem_ptr: SharedMemPointer[Scalar[qkv_type]],
        mbar: SharedMemPointer[SharedMemBarrier],
        chunk_valid_pages: UInt32,
    ):
        # Always the FULL slot: a chunk that reaches here is live, and at the
        # production page sizes `v_sk_pages_per_chunk == 1`, so one issue
        # delivers the whole box. The dead chunks contribute no bytes because
        # they contribute no ring positions at all -- the skip replaces the byte
        # accounting rather than complicating it.
        #
        # Shared-key's box is 4 swizzle chunks wide (vs K's 1), so `fold_chunks`
        # saves more here than elsewhere. `expect_bytes` above is a
        # transaction-byte count, so folding doesn't change it.
        comptime if is_leader:
            expect_bytes_pred(mbar, Int32(v_expect_bytes), e)
        chunk_rows.tma_copy_v[
            needs_partial=partial,
            num_v_sub_tiles=1,
            v_sub_tile_idx=0,
            oob_fill_pages=v_sk_oob_fill and partial,
            fold_chunks=v_fold_chunks,
            row_major=v_row_major,
        ](
            v_tma_op,
            smem_ptr,
            mbar[],
            kv_head_idx=kv_head_idx,
            elect=e,
            num_valid_pages=chunk_valid_pages,
            depth_offset=UInt32(t * v_sk_depth_tile + v_col_offset),
        )

    comptime if config.use_shared_kv:
        # ---- Shared KV mode ----
        # Single StagedPipeline alternating K and V stages.
        # 2Q (num_q=2): K0, V0, K1, V1, ... (one K/V per logical iter).
        # 1Q (num_q=1): K_e[0], K_o[0], V_e[0], V_o[0], K_e[1], K_o[1], ...
        # (two K + two V per logical iter, matching mma_warp's
        # Q@K_e->s0 / Q@K_o->s1 / P_e@V_e->o0 / P_o@V_o->o1 pattern).
        # For MHA: padded_qk_depth == padded_ov_depth, rope_depth == 0.
        # non-WS shared: num_qk_stages==1, one full-depth tile per slot.
        # WS shared sub-tile ring: each K/V slot is ONE 256x64 sub-tile, so
        # `_emit_k`/`_emit_v` each emit num_qk_stages sub-tiles -> the 1Q iter
        # becomes K_e_d0,K_e_d1,K_o_d0,K_o_d1,V_e_d0,V_e_d1,V_o_d0,V_o_d1 (8 slots).

        var kv_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.k_smem_base()
        )
        # Per-CTA SMEM per ring slot. WS: one 256x64 sub-tile (matches
        # smem.ws_subtile_bytes); non-WS shared: one full-depth K/V tile.
        comptime kv_stage_elems = (
            (config.shared_kv_cols() // config.num_qk_stages) * BN // cta_group
        ) if config.use_ws else (config.padded_ov_depth * BN // cta_group)
        # Anti-drift: the producer per-slot stride MUST equal the struct sub-tile
        # size the reservation + consumer (mma_warp) use. A mismatch is exactly
        # the class of latent 2x-overrun this migration fixed. (cta_group==1 for
        # WS, so per-CTA == full.) A non-WS-shared analog would need the per-CTA
        # form `kv_stage_elems * size == k_stage_bytes` (no *cta_group) to stay
        # pair-CTA-correct; left as a follow-up since nope==ov holds on the
        # reachable fa4 path today.
        comptime if config.use_ws:
            comptime assert (
                kv_stage_elems * cta_group * size_of[qkv_type]()
                == SM100AttentionSMem[config].k_stage_bytes
            ), "WS producer kv_stage_elems must match the struct sub-tile size"

        comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
        var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}
        kv_pipeline.state._phase = 1  # producer starts at phase 1

        # Per-stage emit closures. Each bundles the KV producer-pipeline
        # lifecycle (acquire / mbar / smem-offset / produce / step) that
        # is otherwise repeated verbatim at every K and V slot in both
        # the 1Q and 2Q producers. `producer_acquire` only reads `state`
        # (never mutates it), so reading `producer_mbar()` / `state.index()`
        # after the acquire is identical to the prior inline order. The
        # first peeled K slot passes `acquire=False` (initial phase=1).
        # The caller still owns `populate` (K computes `rows`, V reuses it)
        # and any interleaved Q TMA.
        @__parameter
        @always_inline
        def _emit_k[
            partial: Bool,
            with_q: Bool = False,
            acquire: Bool = True,
        ](rows: KVPagedRows, k_num_valid_pages: UInt32):
            # WS shared sub-tile ring: emit num_qk_stages K depth-half sub-tiles
            # (each a 32768-B ring slot with its own barrier); Q (when with_q)
            # rides every K sub-tile (q_elements is per-sub-tile). Folds to one
            # slot for non-WS shared (num_qk_stages==1). The acquire=False peel
            # applies ONLY to the very first sub-tile (initial phase=1); later
            # sub-tiles acquire like any subsequent ring slot.
            comptime for qk_stage in range(config.num_qk_stages):
                comptime if acquire or qk_stage != 0:
                    kv_pipeline.producer_acquire()
                var mbar = kv_pipeline.producer_mbar()
                var smem_ptr = kv_smem + kv_pipeline.state.index() * UInt32(
                    kv_stage_elems
                )
                _produce_k[partial=partial, qk_stage=qk_stage, with_q=with_q](
                    rows, smem_ptr, mbar, k_num_valid_pages
                )
                kv_pipeline.state.step()

        @__parameter
        @always_inline
        def _emit_v[
            partial: Bool, acquire: Bool = True
        ](
            rows: KVPagedRows,
            v_num_valid_pages: UInt32,
            # Layout-E (m_pack==2) ONLY: this reduction chunk's base kv_row
            # (used to derive each partition's own populate range). Unused
            # (dead) for Layout-G / non-WS / 2Q -- default keeps every
            # existing (2-arg) call site byte-identical.
            kv_row_base: UInt32 = 0,
        ):
            # WS shared sub-tile ring: emit num_qk_stages V sub-tiles (each a
            # 32768-B ring slot with its own barrier). Folds to one slot for
            # non-WS shared (num_qk_stages==1). The ring scaffolding
            # (acquire/mbar/smem_ptr/step) is layout-independent; only the
            # producer differs: Layout-G (m_pack==4) depth-splits
            # (`_produce_v`, indexes `rows` whole), Layout-E (m_pack==2)
            # reduction/key-splits (`_produce_v_e`, one `rows.sub_rows`
            # partition each), shared-key key-splits (one `rows.sub_rows` chunk
            # each). All three read the SAME `rows` K populated.
            comptime if config.ws_shared_key:
                # Shared-key: a RUNTIME chunk count, the only sub-tile walk in
                # this kernel whose trip count is not comptime. `live` is the
                # number of key chunks that hold at least one real key; the rest
                # emit nothing at all -- no TMA, no MMA, no ring position.
                #
                # It is computed from the tile's own base row at EVERY site, not
                # only the ones spelled `partial=True`. That is deliberate: the
                # formula returns `v_sk_chunks` for any full tile, so the
                # interior sites are unchanged by construction, and the walk
                # stops depending on a site-by-site "is this the last tile?"
                # classification that `partial` does NOT actually encode
                # (`partial` is paging, and it is True on the known-FULL
                # second-to-last tile at `_emit_v`'s peeled-last-full site).
                comptime assert acquire, (
                    "shared-key V: the acquire=False peel is K-only; a skipped"
                    " first chunk would leave the peeled slot unacquired"
                )
                # `kv_row_base` is this V group's ABSOLUTE base row; the
                # `live` count below and the per-chunk OOB page count both
                # measure against it. All nine 1Q call sites supply it; the 2Q
                # ones do not and take the 0 default, which would walk every
                # chunk from row 0 -- silently, since the ring would stay in
                # step. `FA4Config.__init__` already forces
                # `ws_shared_key => num_q == 1`, so this is unreachable; assert
                # it here because THIS is the code that depends on it.
                comptime assert config.num_q == 1, (
                    "shared-key V: the key-chunk walk needs `kv_row_base`,"
                    " which only the 1Q producer threads through"
                )
                # Every mask clamps `end_col <= num_cols`, so the last tile
                # always holds >= 1 key and chunk 0 is never dead. A base at or
                # past `num_keys` would wrap this subtraction and `min` would
                # silently restore a full walk, so assert rather than defend.
                debug_assert(
                    kv_row_base < num_keys,
                    (
                        "shared-key V: tile base at/past num_keys -- the"
                        " producer was asked for an entirely dead tile"
                    ),
                )
                var live = min(
                    UInt32(v_sk_chunks),
                    UInt32(
                        ceildiv(Int(num_keys - kv_row_base), v_sk_key_chunk)
                    ),
                )
                for c in range(live):
                    var chunk_base = kv_row_base + c * UInt32(v_sk_key_chunk)
                    # Hoisted out of the depth-tile loop: all `num_o_tiles()`
                    # tiles of a chunk read the SAME key rows and differ only in
                    # `depth_offset`. This is the payoff of chunk-outer order.
                    var chunk_rows = rows.sub_rows[v_sk_key_chunk](
                        c * UInt32(v_sk_key_chunk)
                    )
                    var chunk_nvp = UInt32(v_sk_pages_per_chunk)
                    comptime if v_sk_oob_fill and partial:
                        # Only reachable at `page_size < pv_key_chunk()`, where
                        # a LIVE chunk can still contain dead page slots. The
                        # subtraction is safe because `c < live` implies
                        # `chunk_base < num_keys`, and `live` really was
                        # truncated above.
                        chunk_nvp = min(
                            UInt32(v_sk_pages_per_chunk),
                            UInt32(
                                ceildiv(
                                    Int(num_keys - chunk_base),
                                    v_sk_rows_per_page,
                                )
                            ),
                        )
                    comptime for t in range(config.num_o_tiles()):
                        kv_pipeline.producer_acquire()
                        var mbar_sk = kv_pipeline.producer_mbar()
                        var smem_sk = kv_smem + kv_pipeline.state.index() * (
                            UInt32(kv_stage_elems)
                        )
                        _produce_v_sk[partial=partial, t=t](
                            chunk_rows, smem_sk, mbar_sk, chunk_nvp
                        )
                        kv_pipeline.state.step()
                return

            comptime for stage in range(config.num_qk_stages):
                comptime if acquire or stage != 0:
                    kv_pipeline.producer_acquire()
                var mbar = kv_pipeline.producer_mbar()
                var smem_ptr = kv_smem + kv_pipeline.state.index() * UInt32(
                    kv_stage_elems
                )
                comptime if config.m_pack == 2:
                    # No `v_num_valid_pages`: Layout-E derives its own count
                    # per (partition, sub-tile) in the per-partition LUT's page
                    # frame, which the caller's tile-wide count cannot express.
                    _produce_v_e[partial=partial, r=stage](
                        rows, kv_row_base, smem_ptr, mbar
                    )
                else:
                    _produce_v[partial=partial, d_tile=stage](
                        rows, smem_ptr, mbar, v_num_valid_pages
                    )
                kv_pipeline.state.step()

        comptime if num_q == 1:
            # ---- 1Q shared-KV producer ----
            # MMA consumes K_e, K_o, V_e, V_o per logical iter. Produce
            # in matching slot order. No FULL_MASK skip in this path
            # (deferred; standard masks in 1Q-eligible regimes have no
            # mid-range FULL_MASK).
            # pair_cta is always False in 1Q (dispatch guards), so
            # k_row_offset == 0 and k_nvp_peer == 0; the peer branches
            # of _produce_k comptime-prune.

            var T: UInt32
            comptime if config.splitk_partitions > 1 or workspace_split:
                # Per-partition local tile count (window + kv_row offset
                # computed above). T==1 here means a single-tile partition.
                # Covers BOTH cluster split-K and the workspace (do_partition)
                # path -- the shared-KV (num_qk_stages==1, depth<128) producer
                # loop must window here too, else it emits the full range while
                # mma/softmax/correction consume only `part_local_iters` tiles
                # (producer over-fills -> deadlock; cf. the two-load-path trap).
                T = part_local_iters
            else:
                T = mask.last_masked_set_end[BM_mask, BN, page_size](
                    seq_info.prompt_idx, score_row, num_keys
                )

            # T == 1 fast path: produce K_e[0] (with Q) + V_e[0] only.
            # mma_warp's matching T==1 fast path consumes those two slots then
            # returns; softmax_warp's WG1 takes its no-op path.
            if T == UInt32(1):
                var rows_e_t1 = kv_lut.populate[
                    BN, base_alignment, pair_cta, is_leader
                ](seq_info.prompt_idx, kv_row)
                _emit_k[partial=needs_partial, with_q=True, acquire=False](
                    rows_e_t1, k_nvp
                )
                _emit_v[partial=needs_partial](rows_e_t1, v_nvp, kv_row)
                return

            # ---- Peel (T >= 2): K_e[0], K_o[0], V_e[0], V_o[0] ----
            # k_nvp / v_nvp from the parent scope cover the K_e/V_e
            # half (computed at kv_row + k_row_offset == kv_row).
            # Compute the K_o/V_o (kv_row + BN) counts here.
            var k_nvp_o: UInt32 = UInt32(k_pages_per_cta)
            var v_nvp_o: UInt32 = UInt32(KVPagedRows.num_pages)
            comptime if needs_partial:
                k_nvp_o = _k_num_valid_pages(kv_row + UInt32(BN))
                v_nvp_o = _v_num_valid_pages(kv_row + UInt32(BN))

            # K_e[0] with Q (initial slot; no acquire — initial phase=1).
            var rows_e = kv_lut.populate[
                BN, base_alignment, pair_cta, is_leader
            ](seq_info.prompt_idx, kv_row)
            _emit_k[partial=needs_partial, with_q=True, acquire=False](
                rows_e, k_nvp
            )

            # K_o[0]
            var rows_o = kv_lut.populate[
                BN, base_alignment, pair_cta, is_leader
            ](seq_info.prompt_idx, kv_row + UInt32(BN))
            _emit_k[partial=needs_partial](rows_o, k_nvp_o)

            # V_e[0] (reuses rows_e)
            _emit_v[partial=needs_partial](rows_e, v_nvp, kv_row)

            # V_o[0] (reuses rows_o).
            # `v_o_row` is the row of the odd V that is currently OWED. It is
            # assigned at every site that would have emitted an odd V, so the
            # rotated sites below never re-derive it from `kv_row` -- which by
            # then has advanced by 2*BN, making the correct expression
            # `kv_row - BN` at three sites and `kv_row + BN` at the fourth.
            # That sign is the one mistake this whole rotation can make
            # silently: `_emit_v`'s row argument is dead on Layout-G but live on
            # Layout-E (`_produce_v_e`'s partition bases) and on the shared-key
            # arm (its `live` chunk count and OOB page counts). The shared-key
            # arm IS auto-dispatched at depth 256/512, so a wrong sign now reads
            # the wrong keys on a production route rather than lying dormant.
            var v_o_row: UInt32 = kv_row + UInt32(BN)

            # ---- Loop bookkeeping ----
            # T is total K-tiles. Peel consumed K_e[0] + K_o[0] (2 tiles)
            # and V_e[0] + V_o[0] (2 tiles). Each main-loop iter
            # consumes 2 K + 2 V (one full logical iter). Tail (T odd)
            # produces a trailing K_e + V_e only.
            var main_iters: UInt32 = (T - UInt32(2)) >> UInt32(1)
            var has_tail: Bool = (T & UInt32(1)) == UInt32(1)
            var has_peeled_last_full: Bool = False
            comptime if needs_partial:
                # When T is odd, the tail block (always run) handles
                # the partial trailing K_e + V_e. When T is even and
                # there is at least one main-loop pair, the terminal
                # full pair is partial and reserved for peeled-last.
                if not has_tail and main_iters > UInt32(0):
                    main_iters -= UInt32(1)
                    has_peeled_last_full = True

            # ---- Main loop (full tiles) ----
            while main_iters != UInt32(0):
                main_iters -= UInt32(1)
                kv_row += UInt32(2 * BN)

                # K_e[n] (full)
                rows_e = kv_lut.populate[
                    BN, base_alignment, pair_cta, is_leader
                ](seq_info.prompt_idx, kv_row)
                _emit_k[partial=False](rows_e, UInt32(KVPagedRows.num_pages))

                # Owed V_o[n-1] (tile 2n-1), one K group later than block order.
                # MUST precede the K_o[n] `rows_o` populate just below -- that
                # is what still makes `rows_o` tile 2n-1. Always full: this site
                # needs main_iters > 0, i.e. T >= 4, so tile 2n-1 <= T-3 is
                # interior.
                _emit_v[partial=False](
                    rows_o, UInt32(KVPagedRows.num_pages), v_o_row
                )

                # K_o[n] (full)
                rows_o = kv_lut.populate[
                    BN, base_alignment, pair_cta, is_leader
                ](seq_info.prompt_idx, kv_row + UInt32(BN))
                _emit_k[partial=False](rows_o, UInt32(KVPagedRows.num_pages))

                # V_e[n] (full, reuses rows_e)
                _emit_v[partial=False](
                    rows_e, UInt32(KVPagedRows.num_pages), kv_row
                )

                # V_o[n] (full, reuses rows_o) is NOT emitted here -- the
                # rotation defers it to the next block, so only the owed row
                # carries forward.
                v_o_row = kv_row + UInt32(BN)

            # ---- Tail K_e + V_e (T odd, any needs_partial) ----
            # mma_warp's break-check swaps s1/o1 onto s0/o0 and consumes one
            # trailing K + V; that K and V come from this block.
            if has_tail:
                kv_row += UInt32(2 * BN)
                var k_nvp_t: UInt32 = UInt32(k_pages_per_cta)
                var v_nvp_t: UInt32 = UInt32(KVPagedRows.num_pages)
                comptime if needs_partial:
                    k_nvp_t = _k_num_valid_pages(kv_row)
                    v_nvp_t = _v_num_valid_pages(kv_row)

                var rows_t = kv_lut.populate[
                    BN, base_alignment, pair_cta, is_leader
                ](seq_info.prompt_idx, kv_row)
                _emit_k[partial=needs_partial](rows_t, k_nvp_t)

                # Owed V_o (tile T-2). Full: has_tail => T odd >= 3, so T-2 is
                # interior. This block's own V below is tile T-1, the terminal
                # one, so the drain is suppressed here.
                _emit_v[partial=False](
                    rows_o, UInt32(KVPagedRows.num_pages), v_o_row
                )

                _emit_v[partial=needs_partial](rows_t, v_nvp_t, kv_row)

            # ---- Peeled-last full pair (needs_partial && T even) ----
            comptime if needs_partial:
                if has_peeled_last_full:
                    kv_row += UInt32(2 * BN)
                    var k_nvp_pe = _k_num_valid_pages(kv_row)
                    var k_nvp_po = _k_num_valid_pages(kv_row + UInt32(BN))
                    var v_nvp_pe = _v_num_valid_pages(kv_row)
                    var v_nvp_po = _v_num_valid_pages(kv_row + UInt32(BN))

                    # K_e (partial)
                    var rows_pe = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, kv_row)
                    _emit_k[partial=True](rows_pe, k_nvp_pe)

                    # Owed V_o (tile T-3). Full: this block itself emits T-2 and
                    # T-1, so T-3 is interior. MUST precede the `rows_po`
                    # populate below, which does not touch `rows_o` -- but the
                    # owed tile is `rows_o`'s.
                    _emit_v[partial=False](
                        rows_o, UInt32(KVPagedRows.num_pages), v_o_row
                    )

                    # K_o (partial)
                    var rows_po = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, kv_row + UInt32(BN))
                    _emit_k[partial=True](rows_po, k_nvp_po)

                    # V_e (partial, reuses rows_pe)
                    _emit_v[partial=True](rows_pe, v_nvp_pe, kv_row)

                    # V_o (partial, reuses rows_po). Already the terminal V of
                    # this branch and already in rotated position -- unmoved,
                    # which is why the drain below must skip this cell.
                    _emit_v[partial=True](
                        rows_po, v_nvp_po, kv_row + UInt32(BN)
                    )

            # ---- Terminal V_o drain ----
            # The rotation leaves exactly one V owed at the end: the peel owes
            # one, and every block that emits a K pair flushes one and owes one.
            # The tail (T odd) and peeled-last (needs_partial, T even) blocks
            # each end on their own terminal V, so this fires only when neither
            # ran -- T even and either needs_partial is False, or T == 2. It
            # pairs with the consumer's single unconditional post-loop acquire;
            # omitting it, or widening the guard, hangs the CTA rather than
            # corrupting output.
            #
            # `partial=needs_partial` is exact, not conservative: with
            # needs_partial True the guard forces main_iters == 0 and T even,
            # i.e. T == 2, where the peel's odd tile IS terminal and the peel's
            # `v_nvp_o` is its runtime count. With it False nothing is ever
            # partial and `v_nvp_o` is literally num_pages.
            #
            # Must stay the last statement of this arm: the T == 1 fast path
            # returns above and owes nothing.
            if not has_tail and not has_peeled_last_full:
                _emit_v[partial=needs_partial](rows_o, v_nvp_o, v_o_row)

        else:
            comptime if CrossP:
                # ---- 2Q shared-KV K-ahead producer (cross-stage P enabler) ----
                # The MMA consumer walks this same K-ahead order with a private
                # cursor, stepping once per position so slot/phase stay in sync.
                # Headline regime only (no partial pages, no mid-range FULL_MASK
                # skip) -- the same regime the MMA cross-P block runs.
                comptime check_mask_crossp = mask.nonfull_sets[BM_mask, BN]()[
                    0
                ] == TileMaskStatus.UNKNOWN_MASK
                comptime assert not (
                    ExplicitTMEMCrossP and (needs_partial or check_mask_crossp)
                ), (
                    "FA4_TMEM_CROSS_P=true was requested but the K-ahead load"
                    " is implemented for the headline regime only (no partial"
                    " pages, no mid-range FULL_MASK skip)"
                )
                # NT = total K-tiles (== total V-tiles). iter_count is the last
                # masked set end minus one; for the dense contiguous regime that is
                # NT-1, so NT = iter_count + 1.
                var NT: UInt32 = iter_count + UInt32(1)

                # ---- Peel: K0 (+Q0), Q1, [K1 if NT>1], V0 ----
                var rows0 = kv_lut.populate[
                    BN, base_alignment, pair_cta, is_leader
                ](seq_info.prompt_idx, kv_row)
                _emit_k[partial=False, with_q=True, acquire=False](
                    rows0, UInt32(KVPagedRows.num_pages)
                )  # pos 0 = K0

                # Q1 (separate barrier), same as the strict-interleave peel.
                comptime if fuse_gqa:
                    q_gmem_row += UInt32(HalfBM // group)
                else:
                    q_gmem_row += UInt32(HalfBM)
                var q1_mbar = mbars.q1_wait_mbar()
                comptime if is_leader:
                    expect_bytes_pred(q1_mbar, Int32(cta_group * q_bytes), e)
                comptime q1_smem_offset = q_elements * 1  # num_qk_stages=1
                q_async_copy(
                    QType(q_smem + q1_smem_offset, tt_row_major[q_elements]()),
                    q1_mbar[0],
                )

                # v_row lags k_row by one BN (K is one block ahead of its V).
                var v_row_c: UInt32 = kv_row  # V0 row
                var k_row_c: UInt32 = kv_row + UInt32(BN)  # K1 row

                if NT > UInt32(1):
                    var rows_k1 = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, k_row_c)
                    _emit_k[partial=False](
                        rows_k1, UInt32(KVPagedRows.num_pages)
                    )  # pos 1 = K1
                _emit_v[partial=False](
                    rows0, UInt32(KVPagedRows.num_pages)
                )  # pos 2 = V0 (reuses rows0)

                # ---- Loop: emit (K_m, V_{m-1}) for m = 2 .. NT-1 ----
                var m_c: UInt32 = 2
                while m_c < NT:
                    k_row_c += UInt32(BN)  # K_m
                    var rows_km = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, k_row_c)
                    _emit_k[partial=False](
                        rows_km, UInt32(KVPagedRows.num_pages)
                    )
                    v_row_c += UInt32(BN)  # V_{m-1}
                    var rows_vm = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, v_row_c)
                    _emit_v[partial=False](
                        rows_vm, UInt32(KVPagedRows.num_pages)
                    )
                    m_c += UInt32(1)

                # ---- Trailing V_{NT-1} (NT >= 2) ----
                if NT > UInt32(1):
                    v_row_c += UInt32(BN)
                    var rows_vt = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, v_row_c)
                    _emit_v[partial=False](
                        rows_vt, UInt32(KVPagedRows.num_pages)
                    )
            else:
                # ---- 2Q shared-KV producer (existing path, unchanged) ----

                # ---- Peeled: K0 + Q0 on same barrier ----
                var kv_paged_rows = kv_lut.populate[
                    BN, base_alignment, pair_cta, is_leader
                ](seq_info.prompt_idx, kv_row)
                _emit_k[partial=needs_partial, with_q=True, acquire=False](
                    kv_paged_rows, k_nvp
                )  # step -> stage 1

                # ---- Q1 (separate barrier) ----
                comptime if fuse_gqa:
                    q_gmem_row += UInt32(HalfBM // group)
                else:
                    q_gmem_row += UInt32(HalfBM)
                var q1_mbar = mbars.q1_wait_mbar()
                comptime if is_leader:
                    expect_bytes_pred(q1_mbar, Int32(cta_group * q_bytes), e)
                # Elect-predicated in-PTX by q_async_copy; no if-guard here.
                comptime q1_smem_offset = q_elements * 1  # num_qk_stages=1
                q_async_copy(
                    QType(q_smem + q1_smem_offset, tt_row_major[q_elements]()),
                    q1_mbar[0],
                )

                # ---- V0 (reuses kv_paged_rows from K0's populate) ----
                _emit_v[partial=needs_partial](kv_paged_rows, v_nvp)

                comptime check_mask = mask.nonfull_sets[BM_mask, BN]()[
                    0
                ] == TileMaskStatus.UNKNOWN_MASK

                # ---- KV producer loop ----
                # Main body: always full-page (partial=False). When
                # needs_partial, peel off the last iteration for
                # runtime-bounded populate/TMA.
                var main_iters = iter_count
                comptime if needs_partial:
                    if main_iters > 0:
                        main_iters -= 1
                while main_iters != 0:
                    main_iters -= 1
                    kv_row += UInt32(config.BN)

                    comptime if check_mask:
                        if (
                            mask.status(
                                seq_info.prompt_idx,
                                Index[dtype=DType.int32](
                                    Int(score_row), Int(kv_row)
                                ),
                                Index[dtype=DType.int32](BM_mask, BN),
                            )
                            == TileMaskStatus.FULL_MASK
                        ):
                            continue
                    # Produce Kn (full, populate + K TMA)
                    kv_paged_rows = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, kv_row)
                    _emit_k[partial=False](
                        kv_paged_rows, UInt32(KVPagedRows.num_pages)
                    )
                    # Produce Vn (full, reuses kv_paged_rows)
                    _emit_v[partial=False](
                        kv_paged_rows, UInt32(KVPagedRows.num_pages)
                    )

                # ---- Peeled last iteration (partial pages) ----
                comptime if needs_partial:
                    if iter_count > 0:
                        kv_row += UInt32(config.BN)
                        var _skip_last = False
                        comptime if check_mask:
                            if (
                                mask.status(
                                    seq_info.prompt_idx,
                                    Index[dtype=DType.int32](
                                        Int(score_row), Int(kv_row)
                                    ),
                                    Index[dtype=DType.int32](BM_mask, BN),
                                )
                                == TileMaskStatus.FULL_MASK
                            ):
                                _skip_last = True
                        if not _skip_last:
                            k_nvp = _k_num_valid_pages(kv_row + k_row_offset)
                            comptime if is_leader and pair_cta:
                                k_nvp_peer = _k_num_valid_pages(
                                    kv_row + UInt32(config.k_rows_per_cta())
                                )
                            v_nvp = _v_num_valid_pages(kv_row)
                            # Produce Kn (partial, populate + K TMA)
                            kv_paged_rows = kv_lut.populate[
                                BN, base_alignment, pair_cta, is_leader
                            ](seq_info.prompt_idx, kv_row)
                            _emit_k[partial=True](kv_paged_rows, k_nvp)
                            # Produce Vn (partial, reuses kv_paged_rows)
                            _emit_v[partial=True](kv_paged_rows, v_nvp)

    else:
        # ---- Non-shared mode ----
        # One `populate` per outer iteration yields a shared
        # `kv_paged_rows` whose row-indices feed both the K and V TMA
        # copies (`tma_copy_k` / `tma_copy_v`). K's per-CTA half and
        # pair-CTA peer offsets are derived at comptime from
        # `is_leader` inside KVPagedRows.

        var k_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.v_smem_base()
        )
        var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
        var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

        # ---- First tile: K stage 0 (Q0 + populate + K TMA) ----
        var mbark0 = pipeline_k.get_k[qk_stage=0]()  # no wait
        var kv_paged_rows = kv_lut.populate[
            BN, base_alignment, pair_cta, is_leader
        ](seq_info.prompt_idx, kv_row)
        _produce_k[partial=needs_partial, qk_stage=0, with_q=True](
            kv_paged_rows, mbark0.smem.ptr, mbark0.mbar, k_nvp
        )

        # ---- First tile: K stages 1..num_qk_stages-1 (Q + K TMA) ----
        comptime for qk_stage in range(1, config.num_qk_stages):
            var mbark = pipeline_k.get_k[qk_stage=qk_stage]()  # no wait
            _produce_k[partial=needs_partial, qk_stage=qk_stage, with_q=True](
                kv_paged_rows, mbark.smem.ptr, mbark.mbar, k_nvp
            )

        pipeline_k.commit_step()

        # Q1 (separate barriers, one per qk_stage).
        # Skipped in 1Q: the peeled K0 issues above (with_q=True for
        # stages 0..num_qk_stages-1) already loaded the full BM-row Q
        # tile on the K mbars.
        comptime if num_q == 2:
            comptime if fuse_gqa:
                q_gmem_row += UInt32(HalfBM // group)
            else:
                q_gmem_row += UInt32(HalfBM)
            var q1_mbar = mbars.q1_wait_mbar()

            comptime for qk_stage in range(config.num_qk_stages):
                comptime q_smem_offset = q_elements * (
                    config.num_qk_stages + qk_stage
                )
                comptime d_idx = qk_stage * config.BK0
                comptime if is_leader:
                    expect_bytes_pred(
                        q1_mbar + qk_stage, Int32(cta_group * q_bytes), e
                    )
                # Elect-predicated in-PTX by q_async_copy; no if-guard here.
                q_async_copy(
                    QType(q_smem + q_smem_offset, tt_row_major[q_elements]()),
                    q1_mbar[qk_stage],
                    depth_idx=UInt32(d_idx),
                )

        # ---- V0 (reuses kv_paged_rows from stage-0 populate) ----
        var mbarv0 = pipeline_v.get_tile[qk_stage=0]()
        _produce_v[partial=needs_partial](
            kv_paged_rows, mbarv0.smem.ptr, mbarv0.mbar, v_nvp
        )
        pipeline_v.commit_step()

        comptime check_mask = mask.nonfull_sets[BM_mask, BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        # ---- Main body + peeled last iteration ----
        # Main body: always full tiles (partial=False). When
        # needs_partial, peel off the last iteration for
        # runtime-bounded populate/TMA.
        var main_iters = iter_count
        comptime if needs_partial:
            if main_iters > 0:
                main_iters -= 1
        while main_iters != 0:
            main_iters -= 1
            kv_row += UInt32(config.BN)

            comptime if check_mask:
                if (
                    mask.status(
                        seq_info.prompt_idx,
                        Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                        Index[dtype=DType.int32](BM_mask, BN),
                    )
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            # K stage 0 (full, populate + K TMA, no Q).
            pipeline_k.acquire_k[qk_stage=0]()
            mbark0 = pipeline_k.get_k[qk_stage=0]()
            kv_paged_rows = kv_lut.populate[
                BN, base_alignment, pair_cta, is_leader
            ](seq_info.prompt_idx, kv_row)
            _produce_k[partial=False, qk_stage=0](
                kv_paged_rows,
                mbark0.smem.ptr,
                mbark0.mbar,
                UInt32(k_pages_per_cta),
            )
            # K stages 1..num_qk_stages-1 (full, no Q, reuse rows).
            comptime for k_stage in range(1, config.num_qk_stages):
                pipeline_k.acquire_k[qk_stage=k_stage]()
                var mbarkn = pipeline_k.get_k[qk_stage=k_stage]()
                _produce_k[partial=False, qk_stage=k_stage](
                    kv_paged_rows,
                    mbarkn.smem.ptr,
                    mbarkn.mbar,
                    UInt32(k_pages_per_cta),
                )
            pipeline_k.commit_step()

            # V (full, reuses rows from K stage 0's populate).
            pipeline_v.acquire_v()
            var mbarvn = pipeline_v.get_tile[qk_stage=0]()
            _produce_v[partial=False](
                kv_paged_rows,
                mbarvn.smem.ptr,
                mbarvn.mbar,
                UInt32(KVPagedRows.num_pages),
            )
            pipeline_v.commit_step()

        # ---- Peeled last iteration (partial pages) ----
        comptime if needs_partial:
            if iter_count > 0:
                kv_row += UInt32(config.BN)
                var _skip_last = False
                comptime if check_mask:
                    if (
                        mask.status(
                            seq_info.prompt_idx,
                            Index[dtype=DType.int32](
                                Int(score_row), Int(kv_row)
                            ),
                            Index[dtype=DType.int32](BM_mask, BN),
                        )
                        == TileMaskStatus.FULL_MASK
                    ):
                        _skip_last = True
                if not _skip_last:
                    k_nvp = _k_num_valid_pages(kv_row + k_row_offset)
                    comptime if is_leader and pair_cta:
                        k_nvp_peer = _k_num_valid_pages(
                            kv_row + UInt32(config.k_rows_per_cta())
                        )
                    v_nvp = _v_num_valid_pages(kv_row)
                    # K stage 0 (partial, populate + K TMA, no Q).
                    pipeline_k.acquire_k[qk_stage=0]()
                    mbark0 = pipeline_k.get_k[qk_stage=0]()
                    kv_paged_rows = kv_lut.populate[
                        BN, base_alignment, pair_cta, is_leader
                    ](seq_info.prompt_idx, kv_row)
                    _produce_k[partial=True, qk_stage=0](
                        kv_paged_rows, mbark0.smem.ptr, mbark0.mbar, k_nvp
                    )
                    # K stages 1+ (partial, no Q).
                    comptime for k_stage in range(1, config.num_qk_stages):
                        pipeline_k.acquire_k[qk_stage=k_stage]()
                        var mbarkn = pipeline_k.get_k[qk_stage=k_stage]()
                        _produce_k[partial=True, qk_stage=k_stage](
                            kv_paged_rows,
                            mbarkn.smem.ptr,
                            mbarkn.mbar,
                            k_nvp,
                        )
                    pipeline_k.commit_step()

                    # V (partial, reuses rows).
                    pipeline_v.acquire_v()
                    var mbarvn = pipeline_v.get_tile[qk_stage=0]()
                    _produce_v[partial=True](
                        kv_paged_rows,
                        mbarvn.smem.ptr,
                        mbarvn.mbar,
                        v_nvp,
                    )
                    pipeline_v.commit_step()
