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
"""SM100 (B200) tensor-core FP8 MLA lightning-indexer score kernel.

Computes the same per-(query token, key) logit as the scalar
`nn.index_fp8.fp8_index_kernel`, but runs the depth-128 dot product on the
tcgen05 tensor cores instead of a serial FMA loop:

    score[token, key] = k_scale[key]
                        * Σ_head relu(q[token, head] · k[key]) * q_scale[token, head]

Q is `[total_seq, num_heads, depth]` fp8-e4m3, K is paged `[keys, depth]` fp8-e4m3
with a per-token `k_scale`, and the head reduction is a **sum** over `num_heads`.

Layout (crux), cloning the shipped MSA prefill scorer
(`Kernels/lib/msa/sparse_indexer_prefill.mojo`) with the operand roles inverted:

- **MMA_M = key tile** (`BM_key`): A operand = this CTA's K tile `[BM_key, depth]`,
  loaded once and reused across the MTP query tokens.
- **MMA_N = 128 = (query-token × head)**: B operand = a pair of query tokens'
  `[N_TOKENS * num_heads, depth]`, `transpose_b=True` -> `S^T = K @ Q^T =
  [key, (token, head)]`. This is DeepGEMM's `sm100_mqa_logits` packing
  (`BLOCK_Q = 128 / num_heads`, so `N_TOKENS = 2` at `num_heads = 64`).
- **MMA_K = depth = 128** contraction, fp8 in / f32 TMEM accumulation.

The epilogue drains TMEM one row per thread (`tcgen05.ld.32x32b`, warp `w` /
lane `l` -> accumulator row `32 * w + l`), so a thread owns a whole key row and
the head reduction never leaves its registers. Per column it applies the
branchless relu `(x + |x|) * 0.5` and multiplies by `q_scale[token, head]`,
**summing** over each token's head columns; then it multiplies by
`k_scale[key]` and writes one f32 per (token, key). Because a warp holds 32
consecutive keys at a fixed token, each store is one fully-active 128B
transaction.

Grid `(batch, ceil(num_keys / BM_key), seq_slices)`: key tiles and token tiles
are independent outputs, so there is no split-K and no cross-CTA reduction.
`BM_key = MMA_M = 128` keeps the standard (non-`.ws`) tcgen05 datapath -- the
packed `.ws` form engages at `MMA_M <= 64`, and the non-ws form there would
leave half the datapaths idle -- and matches the 128 TMEM lanes this CTA's four
warps drain. grid.z splits a sequence's token tiles across CTAs when the
key-tile grid alone underfills the machine (low-key prefill); decode always
launches one slice.

Token tiles are software-pipelined: Q and its scales are double-buffered so
tile nt+1's TMA and q_scale loads fly under tile nt's MMA and epilogue. The
TMEM accumulator stays single-stage (the drain is a TMEM->register copy that
precedes the epilogue math, so the next MMA already overlaps the math; a
second stage measured as a pure loss by halving TMEM-limited CTAs/SM on large
grids). Decode launches allocate only the SMEM prefix (Q buffer 1 is last in
the layout and unreachable at a single token tile), keeping decode occupancy
unchanged.

Prefill / causal masking and the `-inf` tail-fill fusion are Slice 2/3 (not here);
this kernel is a drop-in for the score buffer the top-k stage consumes.

NVIDIA SM100 only (SS-UMMA / TMA / tcgen05). Verified against
`nn.index_fp8.fp8_index_naive` via `test_index_fp8` and end-to-end top-k set
match via `test_mla_index_fp8`.
"""

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.math import align_up, ceildiv
from std.sys import get_defined_int, has_nvidia_gpu_accelerator, size_of
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple

from layout import (
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    create_split_tma,
)

from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    TMemTile,
    elect,
    elect_mma_arrive,
)
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.sparse_index_fp8_sm100_prefill import (
    _KEYSPLIT_MAX_TOKEN_TILES,
    _PREFILL_MIN_TOKEN_TILES,
    _ctas_per_sm,
    _is_keysplit_shape,
    fp8_index_score_sm100_prefill,
)


comptime _INDEX_SWIZZLE = TensorMapSwizzle.SWIZZLE_128B

# MMA_M, and the number of TMEM lanes the epilogue drains. 128 keeps the
# standard (non-`.ws`) tcgen05 datapath: `SM100TensorAccumulator` switches to the
# packed `.ws` form at `MMA_M <= 64`, and the non-ws form at 64 would leave half
# the datapaths idle. It is also exactly the lane count a 4-warp consumer covers
# one-row-per-thread, which is what makes the epilogue's head sum thread-local.
comptime _BM_KEY = 128

# One consumer warpgroup. `tcgen05_ld` picks its TMEM sub-partition
# warpgroup-relative, so 4 warps x 32 lanes span all `_BM_KEY` accumulator rows.
comptime _NTHREADS = 128

# Accumulator stages and Q buffers.
comptime _N_S_STAGES = 1
comptime _N_Q_BUF = 2
# k(1) + Q buffers + (full, empty) per accumulator stage + the tcgen05 base slot.
comptime _N_MBARS = 1 + _N_Q_BUF + 2 * _N_S_STAGES + 1

# Column chunk for the TMEM->register drain: one `tcgen05.ld.32x32b.x{chunk}` per
# chunk, so this is the drain's live register footprint. Must stay <= 64 -- a
# 128-wide read needs 128 consecutive destination registers and trips ptxas C7602.
comptime _EPILOGUE_CHUNK = 16

# Token-slice split targets, used only by the launcher's `num_slices` heuristic.
# The grid aims for `base_ctas * num_slices ~= _SLICE_CTAS_PER_SM * sm_count`,
# and no slice may own fewer than `_SLICE_MIN_TOKEN_TILES` token tiles (below
# that the per-slice pipeline cannot amortize re-staging the K tile).
comptime _SLICE_CTAS_PER_SM = 2
comptime _SLICE_MIN_TOKEN_TILES = 2

# Token-block count above which nh=32 pure-prefill routes to the K-streaming
# prefill kernel (see the prefill route in `fp8_index_score_sm100`). Much higher
# than nh=64's `_PREFILL_MIN_TOKEN_TILES` (16): measured B200 causal cache=0
# crossover reaches a safe >=12% win only at 448 tiles (seq ~1792).
comptime _PREFILL_MIN_TOKEN_TILES_NH32 = 448

# The `N_TOKENS_ALT` hint every production caller passes: a speculative-decode tile
# sized to divide a GLM 5.x MTP step (6 tokens = num_draft_tokens + 1). At nh=32 that
# is 96 MMA columns, which tiles the step exactly while staying on the 2-CTA/SM side
# of the TMEM step function.
#
# Shared rather than spelled at each call site because a test that scores through its
# own wrapper must instantiate the SAME tile production does, or the alt tile has no
# value-level gate at all.
comptime SPEC_DECODE_N_TOKENS_ALT = 3

# Measurement overrides for the prefill route's alternate N-tile. All three default to
# the production behaviour, so an unset build instantiates exactly the shipped kernel
# set, and none of them can reach a tile that would cost residency. `_ALT_OFF` needs
# its own switch because 0 already means "use the caller's hint" and `get_defined_int`
# rejects a negative value.
comptime _ALT_NTOK_FORCE = get_defined_int["FP8_INDEX_ALT_NTOK", 0]()
comptime _ALT_FORCE_ANY = get_defined_int["FP8_INDEX_ALT_FORCE", 0]()
comptime _ALT_OFF = get_defined_int["FP8_INDEX_ALT_OFF", 0]()


# Q buffer 1 sits at the END of the SMEM layout so a decode launch (every
# batch entry a single token tile, so buffer 1 is never touched) can allocate
# only the prefix and keep the un-pipelined kernel's CTAs/SM. 128B alignment
# (not the 1KB swizzle atom) suffices: MMASmemDescriptor.create preserves the
# full byte address, so the TMA write and the MMA read share the swizzle
# phase off the same base.
comptime _Q1SmemOffset[
    dtype: DType, BM_key: Int, MMA_N: Int, depth: Int
] = align_up(
    (BM_key + MMA_N) * depth * size_of[Scalar[dtype]]()
    + (_N_Q_BUF * MMA_N + BM_key) * size_of[Float32]()
    + _N_MBARS * size_of[SharedMemBarrier](),
    128,
)


comptime QTMATileT[
    dtype: DType, MMA_N: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(MMA_N, 1, depth), _INDEX_SWIZZLE]
comptime KTMATileT[
    dtype: DType, BM_key: Int, depth: Int
] = SplitLastDimTMATensorTile[dtype, Index(BM_key, 1, depth), _INDEX_SWIZZLE]


@always_inline
def _fp8_index_body[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    epilogue_chunk: Int = _EPILOGUE_CHUNK,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
    q_s: TileTensor[.float32, QSLT, ImmutAnyOrigin, Engine=QSEngine],
    output: TileTensor[.float32, OutLT, MutAnyOrigin, Engine=OutEngine],
    max_num_keys: Int,
    causal: Int,
    nt_start: Int,
    n_local: Int,
    out_row_begin: Int,
    out_row_end: Int,
):
    comptime assert valid_length.flat_rank == 1
    comptime MMA_N = N_TOKENS * num_heads
    comptime assert (
        MMA_N == 128
    ), "MMA_N must pack to 128 (N_TOKENS * num_heads); got " + String(MMA_N)
    comptime AT = DType.float32
    comptime SW = _INDEX_SWIZZLE
    comptime NTHREADS = _NTHREADS
    comptime assert BM_key == NTHREADS, (
        "the epilogue drains one TMEM lane per thread, so BM_key must equal the"
        " consumer thread count"
    )
    comptime assert (
        MMA_N % epilogue_chunk == 0
    ), "epilogue_chunk must divide MMA_N; got " + String(epilogue_chunk)
    comptime assert epilogue_chunk <= 64, (
        "a tcgen05.ld wider than 64 needs 64+ consecutive destination registers"
        " and trips ptxas C7602"
    )
    # The fold walks columns in groups of four (one 16-byte q-scale load feeding
    # two f32x2 FFMAs), so a group must sit wholly inside one token and its base
    # must be 16-byte aligned. Every admitted num_heads is a multiple of 4, but
    # nothing in the body enforced it.
    comptime assert epilogue_chunk % 4 == 0 and num_heads % 4 == 0, (
        "the epilogue folds columns in groups of four, so both epilogue_chunk"
        " and num_heads must be multiples of 4 for a group never to straddle a"
        " token boundary"
    )

    var tid = thread_idx.x
    var b = block_idx.x
    var key_start = Int(block_idx.y) * BM_key

    var start_of_seq = Int(valid_length[b])
    var end_of_seq = Int(valid_length[b + 1])
    var seq_len = end_of_seq - start_of_seq

    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    # `num_keys` counts tokens, because the causal bound is expressed in
    # tokens. `num_rows` counts what the cache holds, one pooled key per
    # `kpool` tokens. The two are equal when `kpool == 1`.
    var num_rows = num_keys // kpool

    # This launch owns global token rows `[out_row_begin, out_row_end)` and writes
    # them to `output` rows `[0, out_row_end - out_row_begin)`, so a caller that
    # cannot afford the whole score matrix fills it a row-window at a time. Token
    # tiles are counted from `tok_lo`, so the window costs the epilogue only a
    # different liveness bound. `seq_len` stays the TRUE length -- the causal
    # bound is an absolute position and must not see the window.
    var tok_lo = max(0, out_row_begin - start_of_seq)
    var tok_hi = min(seq_len, out_row_end - start_of_seq)
    var out_row0 = start_of_seq - out_row_begin

    # SINGLE source of the local token origin. The Q tile and its scales are
    # staged for tile 0 in the prologue and for tile it+1 inside the loop, so the
    # origin is spelled in three places; when the window shifted only the loop's
    # copy, tile 0 kept scoring the entry's first tokens while the epilogue filed
    # the result under the window's -- wrong scores that still look structurally
    # valid. Derive all three from this.
    var tok_base = tok_lo + nt_start * N_TOKENS

    # FA4 stateless S = K @ Q^T accumulator. `MMA_M = BM_key = 128 > 64`, so
    # `use_ws` is False and this is the standard (non-packed) TMEM datapath.
    # It carries no handshake -- the mbars below are driven by this kernel.
    # `mma_kind` has no dtype-derived default here (unlike the accumulator this
    # replaced), so an fp8 operand MUST select the f8f6f4 instruction family.
    comptime QK = SM100TensorAccumulator[
        dtype,
        AT,
        MMA_M=BM_key,
        MMA_N=MMA_N,
        BK=depth,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F8F6F4 if dtype.is_float8() else UMMAKind.KIND_F16,
        swizzle_a=SW,
        swizzle_b=SW,
        transpose_b=True,
        cta_group=1,
        num_stages=1,
    ]
    # Operand descriptor K extent, padded to the MMA_K granularity (equals the
    # accumulator's internal `padded_BK` at depth == 128).
    comptime compute_BK = align_up(depth, 16)

    comptime k_elems = BM_key * depth
    comptime q_elems = MMA_N * depth
    var smem = external_memory[
        Scalar[dtype],
        address_space=.SHARED,
        alignment=128,
        name="fp8_index_sm100_smem",
    ]()
    var k_smem = smem
    var q_smem = smem + k_elems
    var qs_smem = (smem + k_elems + q_elems).bitcast[Float32]()
    var ks_smem = qs_smem + _N_Q_BUF * MMA_N
    # mbar[0] = K staging-done (one-shot); then one Q staging-done per Q buffer;
    # then the accumulator handshake, a (full, empty) pair per stage; the last
    # slot holds the tcgen05 TMEM base address.
    var mbar = (ks_smem + BM_key).bitcast[SharedMemBarrier]()
    var k_mbar = mbar
    var q_mbar = mbar + 1
    var s_full = mbar + 1 + _N_Q_BUF
    var s_empty = s_full + _N_S_STAGES
    var ptr_tmem = (mbar + _N_MBARS - 1).bitcast[UInt32]()
    var q1_smem = smem + _Q1SmemOffset[dtype, BM_key, MMA_N, depth]

    comptime k_flat_layout = tt_row_major[k_elems]()
    comptime q_flat_layout = tt_row_major[q_elems]()

    if tid == 0:
        k_mbar[0].init()
        comptime for i in range(_N_Q_BUF):
            q_mbar[i].init()
        comptime for i in range(_N_S_STAGES):
            # `s_full` is armed by a single tcgen05 commit; `s_empty` is the WAR
            # release, so every consumer thread arrives on it once per tile.
            s_full[i].init()
            s_empty[i].init(Int32(NTHREADS))
    named_barrier[Int32(NTHREADS)]()

    # tcgen05 alloc is warp-collective (.sync.aligned): exactly one warp.
    # Release the lock right after so other CTAs sharing the SM can allocate.
    comptime S_COLS = align_up(MMA_N, 32)
    comptime TMEM_COLS = UInt32(_N_S_STAGES * S_COLS)
    if warp_id() == 0:
        tcgen05_alloc[1](ptr_tmem, TMEM_COLS)
        tcgen05_release_allocation_lock[1]()
    named_barrier[Int32(NTHREADS)]()
    var tmem_addr: UInt32 = ptr_tmem[0]

    # --- Stage the K tile once (A operand). A full BM_key-row tile is always
    # staged; rows past this batch's num_keys hold unrelated pool data (the
    # paged descriptor's bound is the whole pool, not num_keys, so it does not
    # zero them). Correctness comes from the epilogue's `key_local < num_rows`
    # guard, which drops those rows' scores before any write to `output`.
    comptime k_bytes = k_elems * size_of[dtype]()
    comptime q_bytes = q_elems * size_of[dtype]()
    if tid == 0:
        var k_row0 = Int(k_operand.row_idx(UInt32(b), UInt32(key_start)))
        var k_dst = TileTensor[
            dtype, type_of(k_flat_layout), address_space=.SHARED
        ](k_smem, k_flat_layout)
        k_mbar[0].expect_bytes(Int32(k_bytes))
        k_tma.async_copy_3d(k_dst, k_mbar[0], (0, 0, k_row0))
        # The first token tile's Q TMA is independent of K, so it rides
        # alongside the K TMA instead of stalling behind the k_mbar wait. This
        # prologue arm is the sole producer of q_mbar[0]'s first completion;
        # the loop only ever issues tile nt+1 into the other buffer.
        var q_dst = TileTensor[
            dtype, type_of(q_flat_layout), address_space=.SHARED
        ](q_smem, q_flat_layout)
        q_mbar[0].expect_bytes(Int32(q_bytes))
        q_tma.async_copy_3d(
            q_dst,
            q_mbar[0],
            (0, 0, (start_of_seq + tok_base) * num_heads),
        )

    # k_scale depends only on this CTA's resident K rows, so stage all BM_key
    # scales once while the TMAs are in flight; the epilogue would otherwise
    # re-load them from global memory in every token tile's critical path.
    if Int(tid) < BM_key:
        var key_local = key_start + Int(tid)
        if key_local < num_rows:
            var ks_ptr = ks_operand.block_paged_ptr[1](
                UInt32(b), UInt32(key_local), UInt32(0), UInt32(0)
            )
            ks_smem[tid] = ks_ptr[0].cast[.float32]()
        else:
            ks_smem[tid] = 0.0

    # Tile 0's q_scale staging: the loop below only stages tile nt+1 during
    # iteration nt, so the first tile's buffer must be filled here (published
    # by the pre-loop named_barrier).
    if tok_base + Int(tid) // num_heads < seq_len:
        qs_smem[tid] = q_s[
            start_of_seq + tok_base + Int(tid) // num_heads,
            Int(tid) % num_heads,
        ][0]
    else:
        qs_smem[tid] = 0.0
    k_mbar[0].wait(0)

    # `tcgen05_ld[datapaths=32]` maps warp w, lane l -> accumulator row
    # `WARP_SIZE * w + l`, picking its TMEM sub-partition warpgroup-relative, so
    # this CTA's 4 warps cover all BM_key rows one-per-thread. Owning a whole key
    # row is what makes the head sum below thread-local (no cross-lane
    # reduction), and it puts 32 consecutive `key_local` in each warp, so each
    # score store is one fully-active 128B transaction.
    var row = Int(warp_id()) * WARP_SIZE + Int(lane_id())
    var key_local = key_start + row

    # `elect()` is a warp-collective `elect.sync` (membermask -1): every lane of
    # every warp must execute it, so it cannot sit inside the producer branch.
    var e = elect()

    # Pre-arm the WAR release so the first MMA's `s_empty` wait falls through.
    comptime for i in range(_N_S_STAGES):
        _ = s_empty[i].arrive()
    named_barrier[Int32(NTHREADS)]()

    # Producer (warp 0) and consumer (every thread) views of the accumulator
    # ring. At one stage both indices are 0 and only the phase alternates.
    var p_state = PipelineState[_N_S_STAGES]()
    var c_state = PipelineState[_N_S_STAGES]()

    # Loop-invariant: `row` is fixed and `ks_smem` is written once above (K is
    # resident for the whole CTA), so this hoists out of the tile loop. It has to
    # stay live across the fold either way now that the stores are interleaved
    # into it.
    var k_scale = ks_smem[row]

    # Software pipeline over token tiles: everything tile nt+1 needs that does
    # not depend on tile nt's results (its Q TMA into the other Q buffer, the
    # q_scale global loads) is issued before tile nt's MMA, so it is in flight
    # underneath the MMA wait and the epilogue. The q_scale SMEM store is
    # deferred to after the epilogue (load-early / store-late) so the thread
    # never stalls on the dependent store. The MMA <-> epilogue ordering rides
    # entirely on the accumulator mbars; the one named_barrier per iteration
    # publishes the cross-thread q_scale staging and keeps its buffers exactly
    # one tile deep.
    for it in range(n_local):
        var tok0 = tok_base + it * N_TOKENS
        var q_buf = it & 1
        var q_next = 1 - q_buf
        var has_next = it + 1 < n_local

        if has_next and tid == 0:
            var q_row0 = (start_of_seq + tok0 + N_TOKENS) * num_heads
            var q_dst = TileTensor[
                dtype, type_of(q_flat_layout), address_space=.SHARED
            ](q1_smem if q_next == 1 else q_smem, q_flat_layout)
            q_mbar[q_next].expect_bytes(Int32(q_bytes))
            q_tma.async_copy_3d(q_dst, q_mbar[q_next], (0, 0, q_row0))

        var qs_tok = Int(tid) // num_heads
        var qs_head = Int(tid) % num_heads
        var qs_next: Float32 = 0.0
        if has_next:
            var qs_local = tok0 + N_TOKENS + qs_tok
            if qs_local < seq_len:
                qs_next = q_s[start_of_seq + qs_local, qs_head][0]

        # Buffer q_buf completes once per round trip of the ring: local
        # iteration it is its (it // 2)-th completion, hence the wait parity.
        q_mbar[q_buf].wait(UInt32((it >> 1) & 1))

        if warp_id() == 0:
            var st = Int(p_state.index())
            # WAR: hold the MMA until every consumer has drained the stage it is
            # about to overwrite. Pre-armed above, so tile 0 falls through.
            s_empty[st].wait(p_state.phase())
            QK.mma(
                smem_descriptor[
                    BMN=BM_key,
                    BK=compute_BK,
                    swizzle_mode=SW,
                    is_k_major=True,
                ](k_smem),
                smem_descriptor[
                    BMN=MMA_N,
                    BK=compute_BK,
                    swizzle_mode=SW,
                    is_k_major=True,
                ](q1_smem if q_buf == 1 else q_smem),
                tmem_addr + UInt32(st * S_COLS),
                c_scale=0,
                elect=e,
            )
            elect_mma_arrive(s_full + st, elect=e)
            p_state.step()

        var cs = Int(c_state.index())
        s_full[cs].wait(c_state.phase())
        var s_it = tmem_addr + UInt32(cs * S_COLS)

        # Drain this thread's key row in `epilogue_chunk`-column chunks. A token owns a
        # CONTIGUOUS column range (`col // num_heads`), so its sum is final at its last
        # column: store it right there and the accumulator collapses from
        # `SIMD[AT, N_TOKENS]` to one f32x2, rather than keeping every token sum and all
        # 128 q-scales live at once (~250 registers, which spilled 1.8M times and cost
        # 2 CTAs/SM of occupancy). The prefill twin's epilogue folds identically; see
        # `sparse_index_fp8_sm100_prefill.mojo` for the register-pressure variant.
        #
        # The chunk loads carry no wait between them, so they pipeline against the
        # folds: `tcgen05.ld` register outputs are automatically ordered, and the single
        # `tcgen05_load_wait` below is only the WAR fence before the stage is released.
        #
        # `num_heads` and `epilogue_chunk` are both powers of two, so a chunk never
        # straddles a token partway and the completion test stays comptime. A *runtime*
        # token index would spill the accumulator to local memory (measured 18x slower
        # at num_heads == 4 / N_TOKENS == 32).
        #
        # Columns fold two at a time so the multiply-accumulate is a single packed
        # `fma.rn.f32x2` (SASS FFMA2); `.fma()` is required over `a * b + c`, which LLVM
        # does not contract for f32 pairs. The relu is a vector op but PTX has no
        # `max.f32x2` at any ISA version, so it lowers to one FMNMX per column. The
        # q-scales are read FOUR at a time: adjacent scalar reads already coalesce into
        # one 16-byte access and an explicit width-2 load does not re-merge.
        var acc = SIMD[AT, 2](0)
        comptime for c in range(MMA_N // epilogue_chunk):
            var frag = TMemTile[AT, BM_key, epilogue_chunk](
                s_it + UInt32(c * epilogue_chunk)
            ).load_async()
            comptime for i in range(epilogue_chunk // 4):
                comptime col4 = c * epilogue_chunk + 4 * i
                # `q_buf * MMA_N` is 0 or 512 B and `col4` is a multiple of 4,
                # so the group is 16-byte aligned; `q_buf` is runtime, so the
                # alignment cannot be inferred and is stated.
                var qs4 = qs_smem.unsafe_load[width=4, alignment=16](
                    q_buf * MMA_N + col4
                )
                comptime for h in range(2):
                    comptime col = col4 + 2 * h
                    var raw = SIMD[AT, 2](
                        frag[4 * i + 2 * h], frag[4 * i + 2 * h + 1]
                    )
                    acc = max(raw, SIMD[AT, 2](0)).fma(
                        qs4.slice[2, offset=2 * h](), acc
                    )
                    comptime if (col + 2) % num_heads == 0:
                        comptime t = col // num_heads
                        var tok_local = tok0 + t
                        # Fused causal mask: token tok_local sees keys up to
                        # cache_len + tok_local (cache_len = num_keys -
                        # seq_len), so forbidden slots are never written --
                        # the bounded top-k reads only `[0, key_bound)` --
                        # and the separate mask pass over the whole score
                        # buffer is skipped. Branchless (causal is 0 or 1): a
                        # branch here in the unrolled token loop measured +4-9%
                        # on the non-causal path from codegen alone.
                        var key_bound = (
                            num_keys - (seq_len - 1 - tok_local) * causal
                        ) // kpool
                        if key_local < key_bound and tok_local < tok_hi:
                            var out_row = out_row0 + tok_local
                            output.raw_store(
                                out_row * max_num_keys + key_local,
                                k_scale * (acc[0] + acc[1]),
                            )
                        acc = SIMD[AT, 2](0)
        tcgen05_load_wait()
        _ = s_empty[cs].arrive()
        c_state.step()

        if has_next:
            qs_smem[q_next * MMA_N + Int(tid)] = qs_next

        named_barrier[Int32(NTHREADS)]()

    if warp_id() == 0:
        tcgen05_dealloc[1](tmem_addr, TMEM_COLS)


@__name(t"fp8_index_score_sm100_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_NTHREADS))
)
def _fp8_index_score_kernel_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
    q_s: TileTensor[.float32, QSLT, ImmutAnyOrigin, Engine=QSEngine],
    output: TileTensor[.float32, OutLT, MutAnyOrigin, Engine=OutEngine],
    max_num_keys_dev: Int32,
    causal_dev: Int32,
    out_row_begin_dev: Int32,
    out_row_end_dev: Int32,
):
    var max_num_keys = Int(max_num_keys_dev)
    var causal = Int(causal_dev)
    var b = block_idx.x
    var key_start = Int(block_idx.y) * BM_key
    var start_of_seq = Int(valid_length[b])
    var seq_len = Int(valid_length[b + 1]) - start_of_seq
    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len
    var num_rows = num_keys // kpool

    # Tokens this entry contributes to the caller's row window (see the body).
    var out_row_begin = Int(out_row_begin_dev)
    var out_row_end = Int(out_row_end_dev)
    var win_tokens = min(seq_len, out_row_end - start_of_seq) - max(
        0, out_row_begin - start_of_seq
    )

    # Bail uniformly (every thread) before the helper's first collective op
    # (TMA mbar / tcgen05 alloc); a divergent early return would deadlock them.
    # OOB keys keep the caller's `-inf` fill.
    if key_start >= num_rows or win_tokens <= 0:
        return

    # Flat launch covers every windowed token tile of this sequence (grid.z == 1).
    _fp8_index_body[
        dtype,
        KOperand,
        KSOperand,
        VLLT,
        QSLT,
        OutLT,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
        kpool,
        VLEngine=VLEngine,
        QSEngine=QSEngine,
        OutEngine=OutEngine,
    ](
        q_tma,
        k_tma,
        k_operand,
        ks_operand,
        valid_length,
        q_s,
        output,
        max_num_keys,
        causal,
        0,
        ceildiv(win_tokens, N_TOKENS),
        out_row_begin,
        out_row_end,
    )


@__name(t"fp8_index_score_sm100_split_{dtype}")
@__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_NTHREADS))
)
def _fp8_index_score_kernel_sm100_split[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    VLLT: TensorLayout,
    QSLT: TensorLayout,
    OutLT: TensorLayout,
    num_heads: Int,
    depth: Int,
    BM_key: Int,
    N_TOKENS: Int,
    _is_cache_length_accurate: Bool,
    kpool: Int = 1,
    *,
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    QSEngine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    q_tma: QTMATileT[dtype, N_TOKENS * num_heads, depth],
    k_tma: KTMATileT[dtype, BM_key, depth],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[.uint32, VLLT, ImmutAnyOrigin, Engine=VLEngine],
    q_s: TileTensor[.float32, QSLT, ImmutAnyOrigin, Engine=QSEngine],
    output: TileTensor[.float32, OutLT, MutAnyOrigin, Engine=OutEngine],
    max_num_keys_dev: Int32,
    causal_dev: Int32,
    out_row_begin_dev: Int32,
    out_row_end_dev: Int32,
):
    var max_num_keys = Int(max_num_keys_dev)
    var causal = Int(causal_dev)
    var b = block_idx.x
    var key_start = Int(block_idx.y) * BM_key
    var start_of_seq = Int(valid_length[b])
    var seq_len = Int(valid_length[b + 1]) - start_of_seq
    var num_keys = Int(k_operand.cache_length(b))
    comptime if not _is_cache_length_accurate:
        num_keys += seq_len
    var num_rows = num_keys // kpool

    # Tokens this entry contributes to the caller's row window (see the body).
    var out_row_begin = Int(out_row_begin_dev)
    var out_row_end = Int(out_row_end_dev)
    var win_tokens = min(seq_len, out_row_end - start_of_seq) - max(
        0, out_row_begin - start_of_seq
    )

    # grid.z splits this sequence's windowed token tiles across CTAs; bounds are
    # uniform per CTA.
    var n_token_tiles = ceildiv(win_tokens, N_TOKENS)
    var tiles_per_slice = ceildiv(n_token_tiles, Int(grid_dim.z))
    var nt_start = Int(block_idx.z) * tiles_per_slice

    # Bail uniformly (every thread) before the helper's first collective op
    # (TMA mbar / tcgen05 alloc); a divergent early return would deadlock them.
    # OOB keys keep the caller's `-inf` fill.
    if key_start >= num_rows or win_tokens <= 0 or nt_start >= n_token_tiles:
        return
    var n_local = min(tiles_per_slice, n_token_tiles - nt_start)

    _fp8_index_body[
        dtype,
        KOperand,
        KSOperand,
        VLLT,
        QSLT,
        OutLT,
        num_heads,
        depth,
        BM_key,
        N_TOKENS,
        _is_cache_length_accurate,
        kpool,
        VLEngine=VLEngine,
        QSEngine=QSEngine,
        OutEngine=OutEngine,
    ](
        q_tma,
        k_tma,
        k_operand,
        ks_operand,
        valid_length,
        q_s,
        output,
        max_num_keys,
        causal,
        nt_start,
        n_local,
        out_row_begin,
        out_row_end,
    )


@always_inline
def fp8_index_score_sm100[
    dtype: DType,
    KOperand: MHAOperand,
    KSOperand: MHAOperand,
    num_heads: Int,
    depth: Int,
    _is_cache_length_accurate: Bool,
    N_TOKENS_ALT: Int = 0,
    kpool: Int = 1,
](
    output: TileTensor[.float32, ...],
    q: TileTensor[mut=False, dtype, ...],
    q_s: TileTensor[mut=False, .float32, ...],
    k_operand: KOperand,
    ks_operand: KSOperand,
    valid_length: TileTensor[mut=False, .uint32, ...],
    batch_size: Int,
    max_seq_len: Int,
    max_num_keys: Int,
    causal: Bool,
    ctx: DeviceContext,
    out_row_begin: Int = 0,
) raises:
    """Launch the SM100 tensor-core FP8 indexer scorer into `output`.

    NVIDIA SM100 only: uses SS-UMMA, tcgen05 TMEM, and TMA staging. Writes the
    same `[total_seq, max_num_keys]` score buffer as the scalar
    `nn.index_fp8.fp8_index_kernel`. Out-of-range keys are left untouched (the
    caller's `-inf` fill covers them).

    Parameters:
        dtype: FP8 element type of Q and K (float8_e4m3fn).
        KOperand: `MHAOperand` type for the K values.
        KSOperand: `MHAOperand` type for the per-token K scales.
        num_heads: Query index heads (must be 64 so N_TOKENS * num_heads == 128).
        depth: Head dimension (contraction, must be 128).
        _is_cache_length_accurate: When False, `num_keys = cache_length + seq_len`;
            when True, `cache_length` already includes the new tokens.
        N_TOKENS_ALT: Optional alternate N-tile, in TOKENS, for the prefill
            route. 0 disables it and reproduces the default instantiation set
            exactly. Taken when `N_TOKENS_ALT * num_heads` is a legal UMMA N that
            keeps the default tile's CTAs/SM, and `N_TOKENS_ALT` divides
            `max_seq_len` while the default `128 // num_heads` does not -- the
            default then spends whole MMA columns and TMEM reads on dead tokens.
            Silently inert at head counts and sequence lengths where any of that
            fails, so one constant is safe across TP ranks. Intended value: the
            speculative decode width (`num_draft_tokens + 1`) or a divisor of it.
        kpool: Tokens per pooled cache row. `1` scores one row per token;
            `k > 1` scores one pooled key per `k` consecutive tokens, so
            every candidate count and the caller's `top_k` are
            pool-granular.

    Args:
        output: Score buffer `[total_seq, max_num_keys]`, f32.
        q: Query tensor `[total_seq, num_heads, depth]`, fp8.
        q_s: Query scales `[total_seq, num_heads]`, f32.
        k_operand: K values as an `MHAOperand`.
        ks_operand: K scales as an `MHAOperand`.
        valid_length: Ragged query-token offsets `[batch + 1]`.
        batch_size: Batch size.
        max_seq_len: Upper bound on any batch entry's query-token count; when
            it fits a single token tile, the launch allocates only the SMEM
            prefix (Q buffer 1 is unreachable).
        max_num_keys: Row stride of `output` (>= every per-batch key count).
        causal: Apply the causal mask in the epilogue store guard (token t
            sees keys up to cache_len + t); forbidden slots are never
            written, and the bounded top-k never reads them.
        ctx: Device context.
        out_row_begin: Global token row that `output` row 0 holds. `output`
            covers `[out_row_begin, out_row_begin + output.dim[0]())`, so a
            caller that cannot afford the whole `total_seq x max_num_keys`
            matrix scores it a row-window at a time. The default covers every
            row, and every quantity the window touches then reduces to its
            unwindowed form.
    """
    # The N-tile packs N_TOKENS = 128 // num_heads whole query tokens, so any
    # divisor of 128 is structurally admissible; the gate lists the validated
    # counts (64 = DeepSeek V3.2 replicated; 32 = GLM 5.x replicated; 8 / 4 =
    # TP-head-sharded indexers, e.g. GLM's 32 heads over 4 or 8 ranks).
    comptime assert num_heads in (64, 32, 8, 4), (
        "SM100 FP8 indexer scorer requires num_heads in {4, 8, 32, 64} (N-tile"
        " of 128)"
    )
    comptime assert (
        depth == 128
    ), "SM100 FP8 indexer scorer requires depth == 128"
    comptime BM_key = _BM_KEY
    # The K tile is staged with ONE TMA copy of BM_key contiguous physical rows
    # at row_idx(b, key_start), which is only correct when BM_key virtual key
    # rows never straddle a page boundary: page_size must be 0 (contiguous /
    # ragged) or a multiple of BM_key. The dispatch sites (index_fp8 /
    # mla_index_fp8) route any other page_size to the scalar kernel; this is the
    # backstop so a future caller cannot reintroduce the wrong-page read.
    comptime assert (
        KOperand.page_size == 0 or KOperand.page_size % BM_key == 0
    ), (
        "SM100 FP8 indexer scorer requires K-cache page_size == 0 or a multiple"
        " of BM_key"
    )
    comptime MMA_N = 128
    comptime N_TOKENS = MMA_N // num_heads

    # Rows of the score matrix this launch fills, and the global window they map
    # to. Grid sizing below uses the window; ROUTING deliberately does not, so
    # chunking a batch cannot silently move it to a different kernel.
    var out_rows = Int(output.dim[0]())
    var win_seq_len = min(max_seq_len, out_rows)

    var total_q_rows = Int(q.dim[0]()) * num_heads
    var q_tma_tile = create_split_tma[
        Index(MMA_N, 1, depth),
        Index(UNKNOWN_VALUE, 1, depth),
        _INDEX_SWIZZLE,
    ](
        ctx,
        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](q.ptr),
        total_q_rows,
    )
    var k_tma_tile = k_operand.create_tma_tile[
        _INDEX_SWIZZLE,
        BN=BM_key,
        depth=depth,
        BK=depth,
    ](ctx)

    # Prefill route: the warp-specialized K-streaming kernel (Q resident, causal
    # triangle trim) vs the K-resident scorer. Both fold the same (token, key, head)
    # triples, so the epilogue instruction count is identical and the trade is per-SM
    # consumer warps against per-CTA prologue cost.
    #
    # The thresholds were measured when the K-streaming kernel ran at 1 CTA/SM with
    # 232-reg consumers -- ~2x lower issue utilization than the scorer's 4-5 CTA/SM --
    # which made it lose everywhere except long prefill: nh=64 won broadly (+7-44%,
    # growing with seq) hence `_PREFILL_MIN_TOKEN_TILES` (16), while nh=32 won only for
    # long PURE-prefill under a causal mask (cache=0 crossover >=384 tiles; a cached
    # prefix or a NULL mask regressed it). The penalty is per-CTA occupancy, not grid
    # underfill.
    #
    # TODO(cme): the kernel now targets 2 CTAs/SM, halving that deficit, so the nh=64
    # threshold is stale in the conservative direction -- re-measure it. nh in {4, 8}
    # reach only the key-split arm below.
    #
    # At nh=32 the thresholds are gone entirely: `UNIFY` sends every shape to the
    # warp-specialized kernel. What they excluded was not a slow corner but a hole --
    # `max_num_keys <= max_seq_len` is false the moment a request has any cached
    # prefix, so every chunked-prefill continuation fell to the scorer, which is
    # 1.37-2.00x slower there, and the 448-tile floor sent 256/512/1024-token chunks
    # the same way at 1.25x. The two shapes the scorer still wins are a 3-token tail
    # over a shallow cache (~1.1x on a ~6 us call) and shallow-cache decode (a wash);
    # neither justifies keeping a second kernel family alive.
    comptime UNIFY = num_heads == 32
    comptime if (
        num_heads == 64 or num_heads == 32 or num_heads == 8 or num_heads == 4
    ):
        var token_tiles = ceildiv(max_seq_len, N_TOKENS)
        var to_prefill = UNIFY
        comptime if num_heads == 64:
            comptime min_tiles = (
                _PREFILL_MIN_TOKEN_TILES if num_heads
                == 64 else _PREFILL_MIN_TOKEN_TILES_NH32
            )
            to_prefill = token_tiles >= min_tiles
            comptime if num_heads == 32:
                to_prefill = (
                    to_prefill and causal and max_num_keys <= max_seq_len
                )
        # ... and the opposite corner, which the thresholds above exclude but
        # the key split reopens: too few token blocks to fill the grid, with a
        # cache deep enough that splitting the key range fills it instead
        # (decode / MTP-decode). The K-resident scorer is at its worst here --
        # one CTA per key tile means a full CTA prologue per single MMA, 6704
        # CTAs for a batch-8 GLM MTP step at 107K keys. Deliberately NOT gated
        # on `causal`: `fp8_index` hard-codes it to False, so a causal gate
        # would leave the benchmark measuring the route it is meant to replace.
        to_prefill = to_prefill or _is_keysplit_shape[num_heads, BM_key](
            max_seq_len, max_num_keys
        )
        if to_prefill:
            # Optional alternate N-tile, taken when it divides the run and the default
            # does not. The default packs `128 // num_heads` whole tokens, so an MTP
            # step of seq_len 6 at nh=32 runs two 4-token blocks: 256 MMA columns for
            # 192 live ones. What that recovers is NOT the fold (the FMNMX / FFMA2 / LDS
            # counts measure identical across tile widths) but the MMA and the TMEM
            # reads, which are issued for dead columns too.
            #
            # Prefer a DIVISOR over a multiple. Both tile the run exactly, but
            # `_S_TMEM_STAGES * align_up(MMA_N, 32)` rounded up to a power of two is
            # what buys or loses the second co-resident CTA: 6 tokens at nh=32 needs
            # 384 -> 512 columns and drops to 1 CTA/SM, while 3 tokens needs 192 -> 256
            # and keeps 2. This arm therefore stays on the 2-CTA/SM side. See
            # `_ctas_per_sm` in the prefill file.
            #
            # A 192-column WIDE tile covering the whole 6-token step used to claim
            # this width ahead of here, buying tile exactness with residency and
            # handing the consumer warps back through a second warpgroup. Measured
            # against it on B200, min over 5 reps x 3 processes: the 96-column
            # divisor is faster on every decode cell that moves, 1.07x at (b8, 32K
            # keys) to 1.28x at (b32, 8K), geomean 1.17x over those five and 1.12x
            # over all seven. Two things it wins that the wide tile could not: the
            # second co-resident CTA, and the q-scale hoist, which keys on the
            # column count and so fits at 96 -- worth 48 `ld.shared` per key tile
            # per thread, 8.8% of the wide tile's loop body. Streaming the key range
            # twice, once per 3-token block, costs nothing measurable: `grid_dim` is
            # (batch, token_block, key_part), so an entry's two blocks land in the
            # same key-part slab and share L2.
            #
            # The hint is a TOKEN count, so one caller-side constant reaches every head
            # count and is inert where it cannot apply: 3 is 24 columns at nh=8, not a
            # legal UMMA N, and at nh=64 the default 2 already divides any even
            # speculative width. A guard rather than an assert, so one constant works
            # for all TP ranks. Each clause below earns its keep:
            #
            # * `_ctas_per_sm[MMA_N_ALT] == _ctas_per_sm[MMA_N]` never trades residency
            #   for tile exactness, and keeps a TOKEN-count hint from leaking across
            #   head counts -- 3 tokens is 96 columns at nh=32 but 192 at nh=64, which
            #   without this clause dropped nh=64's long-prefill route to 1 CTA/SM.
            # * `MMA_N_ALT >= 64` keeps a pathological hint (1 token = 32 columns at
            #   nh=32, which divides everything) off a tile far below the shape where
            #   the tensor pipe is efficient.
            # * The block-count clause is checked against the CEILING the routing arm
            #   tested, not the default tile's own block count: a narrower tile
            #   legitimately uses more blocks, and comparing against `token_tiles`
            #   rejected exactly the configuration this exists to allow.
            #
            # The three `FP8_INDEX_ALT_*` overrides all default to the production
            # behaviour, so an unset build is byte-identical. None of them relaxes the
            # residency clause -- the wider tile stays unreachable by knob, because that
            # trade was measured a loss.
            comptime N_ALT = 0 if _ALT_OFF != 0 else (
                _ALT_NTOK_FORCE if _ALT_NTOK_FORCE > 0 else N_TOKENS_ALT
            )
            comptime MMA_N_ALT = N_ALT * num_heads
            comptime if (
                N_ALT != N_TOKENS
                and MMA_N_ALT % 16 == 0
                and MMA_N_ALT >= 64
                and MMA_N_ALT <= 256
                and _ctas_per_sm[MMA_N_ALT]() == _ctas_per_sm[MMA_N]()
            ):
                if (
                    max_seq_len % N_ALT == 0
                    and (max_seq_len % N_TOKENS != 0 or _ALT_FORCE_ANY != 0)
                    and max_seq_len // N_ALT
                    <= max(token_tiles, _KEYSPLIT_MAX_TOKEN_TILES)
                ):
                    var q_tma_alt = create_split_tma[
                        Index(MMA_N_ALT, 1, depth),
                        Index(UNKNOWN_VALUE, 1, depth),
                        _INDEX_SWIZZLE,
                    ](
                        ctx,
                        rebind[UnsafePointer[Scalar[dtype], ImmutAnyOrigin]](
                            q.ptr
                        ),
                        total_q_rows,
                    )
                    fp8_index_score_sm100_prefill[
                        dtype,
                        KOperand,
                        KSOperand,
                        num_heads,
                        depth,
                        BM_key,
                        N_ALT,
                        _is_cache_length_accurate,
                        kpool,
                        VLEngine=type_of(valid_length.as_immut()).Engine,
                        QSEngine=q_s.Engine,
                        OutEngine=output.Engine,
                    ](
                        rebind[QTMATileT[dtype, MMA_N_ALT, depth]](q_tma_alt),
                        rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
                        k_operand,
                        ks_operand,
                        valid_length,
                        q_s,
                        output,
                        batch_size,
                        max_seq_len,
                        max_num_keys,
                        Int(causal),
                        ctx,
                        out_row_begin,
                    )
                    return
            fp8_index_score_sm100_prefill[
                dtype,
                KOperand,
                KSOperand,
                num_heads,
                depth,
                BM_key,
                N_TOKENS,
                _is_cache_length_accurate,
                kpool,
                VLEngine=type_of(valid_length.as_immut()).Engine,
                QSEngine=q_s.Engine,
                OutEngine=output.Engine,
            ](
                rebind[QTMATileT[dtype, N_TOKENS * num_heads, depth]](
                    q_tma_tile
                ),
                rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
                k_operand,
                ks_operand,
                valid_length,
                q_s,
                output,
                batch_size,
                max_seq_len,
                max_num_keys,
                Int(causal),
                ctx,
                out_row_begin,
            )
            return

    # The K-resident scorer, and the token-slice split that feeds it. Guarded
    # rather than merely unreachable: dead straight-line code still instantiates
    # its kernels, so without this `comptime if` a unified nh=32 build would
    # keep compiling two kernel families it can never launch. With it, "the
    # scorer never runs for GLM-5.2" is checkable from the emitted blob set.
    comptime if not UNIFY:
        comptime kernel_flat = _fp8_index_score_kernel_sm100[
            dtype,
            KOperand,
            KSOperand,
            type_of(valid_length.as_immut()).LayoutType,
            type_of(q_s).LayoutType,
            type_of(output).LayoutType,
            num_heads,
            depth,
            BM_key,
            N_TOKENS,
            _is_cache_length_accurate,
            kpool,
            VLEngine=type_of(valid_length.as_immut()).Engine,
            QSEngine=q_s.Engine,
            OutEngine=output.Engine,
        ]
        comptime kernel_split = _fp8_index_score_kernel_sm100_split[
            dtype,
            KOperand,
            KSOperand,
            type_of(valid_length.as_immut()).LayoutType,
            type_of(q_s).LayoutType,
            type_of(output).LayoutType,
            num_heads,
            depth,
            BM_key,
            N_TOKENS,
            _is_cache_length_accurate,
            kpool,
            VLEngine=type_of(valid_length.as_immut()).Engine,
            QSEngine=q_s.Engine,
            OutEngine=output.Engine,
        ]

        comptime q1_offset = _Q1SmemOffset[dtype, BM_key, MMA_N, depth]
        comptime smem_bytes = q1_offset + MMA_N * depth * size_of[
            Scalar[dtype]
        ]()
        var smem_bytes_rt = q1_offset if win_seq_len <= N_TOKENS else smem_bytes

        # Split each sequence's token tiles over grid.z only when the key-tile
        # grid leaves SMs idle (batch-1 prefill at 2K keys is 32 CTAs on ~148
        # SMs), targeting `_SLICE_CTAS_PER_SM` CTAs per SM. Slices re-stage the
        # (L2-hot) K tile and halve the per-slice pipeline depth, so a grid already
        # at one wave never splits (a 256-CTA GLM MTP-decode shape measured -29%
        # when split), and a slice never gets fewer than `_SLICE_MIN_TOKEN_TILES`.
        #
        # The fill arm FLOORS, for the same reason as the prefill kernel's twin
        # (`sparse_index_fp8_sm100_prefill.mojo`): it wants the largest slice count
        # whose grid still fits the target, and `ceildiv` overshoots it by
        # construction, buying a second wave that carries a handful of CTAs.
        comptime sm_count = ctx.default_device_info.sm_count
        var base_ctas = batch_size * ceildiv(max_num_keys, BM_key)
        var num_slices = 1
        if base_ctas < sm_count:
            num_slices = max(
                1,
                min(
                    (_SLICE_CTAS_PER_SM * sm_count) // base_ctas,
                    ceildiv(
                        ceildiv(win_seq_len, N_TOKENS), _SLICE_MIN_TOKEN_TILES
                    ),
                ),
            )

        if num_slices > 1:
            ctx.enqueue_function[kernel_split](
                rebind[QTMATileT[dtype, MMA_N, depth]](q_tma_tile),
                rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
                k_operand,
                ks_operand,
                valid_length.as_immut(),
                q_s,
                output,
                Int32(max_num_keys),
                Int32(causal),
                Int32(out_row_begin),
                Int32(out_row_begin + out_rows),
                grid_dim=(
                    batch_size,
                    ceildiv(max_num_keys, BM_key),
                    num_slices,
                ),
                block_dim=_NTHREADS,
                shared_mem_bytes=smem_bytes_rt,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_bytes)
                ),
            )
        else:
            ctx.enqueue_function[kernel_flat](
                rebind[QTMATileT[dtype, MMA_N, depth]](q_tma_tile),
                rebind[KTMATileT[dtype, BM_key, depth]](k_tma_tile),
                k_operand,
                ks_operand,
                valid_length.as_immut(),
                q_s,
                output,
                Int32(max_num_keys),
                Int32(causal),
                Int32(out_row_begin),
                Int32(out_row_begin + out_rows),
                grid_dim=(batch_size, ceildiv(max_num_keys, BM_key), 1),
                block_dim=_NTHREADS,
                shared_mem_bytes=smem_bytes_rt,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_bytes)
                ),
            )
