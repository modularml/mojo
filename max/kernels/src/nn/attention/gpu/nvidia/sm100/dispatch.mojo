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
Provides the SM100 (Blackwell) flash-attention host-side dispatch entry point
that selects a 1Q or 2Q FA4 kernel configuration, builds the Q/K/V/O TMA
descriptors and tile scheduler, and enqueues the kernel onto the device.
"""

from std.collections import OptionalReg
from std.math import ceildiv, clamp, nan
from std.sys import get_defined_int
from max.gpu.primitives.grid_controls import pdl_launch_attributes
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    Dim,
    FuncAttribute,
)
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from layout import TensorEngine
from layout.tma_async import RaggedTMA3DTile
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.logger import Logger
from nn.attention.gpu.nvidia.sm100.attention import FA4Config, MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.common import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    q_tma,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    SplitKPartition,
    StaticInt,
)
from .attention_utils import (
    clusters_per_wave,
    kv_sub_tile_rows,
    kv_tma_fold_chunks,
    o_store_tma_blocks_per_op,
    splitk_p_ladder,
)
from .kernel import SM100MHA2Q
from .fa4_splitk_combine import P_MAX, fa4_splitk_combine

comptime logger = Logger()

# Debug instrument for the unfused (workspace) split-K path -- see
# `launch_workspace` below for what it grades and why it must exist.
comptime FA4_WS_POISON: Bool = get_defined_int["FA4_WS_POISON", 0]() != 0


@always_inline
def _bucket_ws[sm_count: Int](n: Int, p_max: Int, raw_grid: Int) -> Int:
    """Snap a desired workspace split-K partition count UP to the nearest
    `splitk_p_ladder` rung, then CAP at `p_max` (`ws_p_ceiling`'s
    capture-invariant ceiling).

    The cap is NOT snapped: when `p_max` binds, `P` is off-rung and the combine
    takes its generic runtime-`P` kernel (the common case on B200). The ladder is
    shared with `fa4_splitk_combine`, which specializes per rung.

    `raw_grid` is invariant within a CUDA-graph capture, so `p_max` is fixed per
    capture and the ladder only bounds the `grid.x` values the `num_keys`-driven
    `by_cache` ramp produces below it. Over-bucketing past `by_cache` (never past
    `p_max`) is correctness-safe: surplus partitions get empty KV windows, egress
    `-inf`, and the combine folds them. Mirrors `_bucket_num_partitions` in
    `mla_decode_dispatch.mojo`.

    Rungs below 12 (`2,4,6,8,10`) keep a short cache at a mid/high-batch
    `raw_grid` from jumping straight to 12 partitions; `2`, `4`, and `10` double
    as the fill-all-SM cluster candidates (`_cluster_splitk_candidates`).

    A partition is useful only if it fills an idle SM. An FA4 CTA uses the full
    TMEM of one SM. A `raw_grid` of `sm_count` or more thus fills every wave
    but the last one. The last wave leaves some SMs idle when `raw_grid` is not
    a multiple of `sm_count`.

    The guard below returns one partition for a short cache. But it keeps the
    split while the last wave is half full or less
    (`0 < raw_grid % sm_count <= sm_count // 2`). A split then fills SMs that
    are idle. A fuller last wave has too few idle SMs, and the split costs a
    wave instead.
    """
    comptime _LADDER = splitk_p_ladder[sm_count]()
    comptime first_rung: Int = _LADDER[0]
    # A split to `first_rung` divides the per-CTA work by `first_rung`, so it
    # pays only if it cuts the wave count by more than that same factor.
    var waves_unsplit = ceildiv(raw_grid, sm_count)
    var waves_split = ceildiv(first_rung * raw_grid, sm_count)
    var split_saves_a_wave = waves_split < first_rung * waves_unsplit

    var short_cache = n < first_rung
    var grid_saturated = raw_grid >= sm_count
    if short_cache and grid_saturated and not split_saves_a_wave:
        return 1
    comptime for v in _LADDER:
        if n <= v:
            return min(v, p_max)
    return min(sm_count, p_max)


# `raw_grid` below which `ws_p_ceiling` follows one GPC wave's SM-fill before
# its measured plateau.
comptime WS_RAW_GRID_CLAMP: UInt32 = 4

# Last `raw_grid` covered by the B200 ragged partition sweep.
comptime WS_SWEEP_MAX_RAW_GRID: UInt32 = 24


@always_inline
def ws_p_ceiling[sm_count: Int](raw_grid: UInt32) -> UInt32:
    """Capture-invariant partition-count ceiling for the split-K crossover:
    follows one GPC wave's SM-fill, holds the plateau through
    `WS_SWEEP_MAX_RAW_GRID`, then decays toward one partition.

    `raw_grid` is capture-invariant (see `_bucket_ws`), so this is a single
    fixed value per CUDA-graph capture.

    `WS_RAW_GRID_CLAMP` clamps `raw_grid` instead of multiplying the wave-fill:
    a flat multiple reproduces the moderate-batch optimum but also scales the
    already-good small-`raw_grid` case into a measured regression, so clamping
    leaves small grids untouched while lifting a small one-wave toward the
    plateau. Beyond the sweep, the inverse target keeps the split grid near a
    fixed wave budget rather than growing with `raw_grid`.
    """
    comptime assert (
        WS_SWEEP_MAX_RAW_GRID % WS_RAW_GRID_CLAMP == 0
    ), "WS_RAW_GRID_CLAMP must divide WS_SWEEP_MAX_RAW_GRID exactly"
    comptime oversub = WS_SWEEP_MAX_RAW_GRID // WS_RAW_GRID_CLAMP
    var g = max(raw_grid, 1)
    return clamp(
        UInt32(sm_count) * oversub // g,
        1,
        UInt32(sm_count) // min(g, WS_RAW_GRID_CLAMP),
    )


@always_inline
def _raw_grid[
    dtype: DType, //, cfg: FA4Config[dtype]
](max_prompt_len: UInt32, batch_size: UInt32) -> UInt32:
    """Work items the launch needs: prompt tiles x scheduled heads x batch."""
    comptime heads_sched = UInt32(
        cfg.num_kv_heads if cfg.fuse_gqa else cfg.num_q_heads
    )
    return (
        ceildiv(max_prompt_len, UInt32(cfg.PairBM_eff()))
        * heads_sched
        * batch_size
    )


# Keys per shared-key workspace partition. Depth-free, unlike the 1Q/WS-G
# `512 * depth // 128` divisor which under-splits at d512; only one swept shape
# is bound by this and it admits a wide band, so the exact value is not
# load-bearing.
comptime SK_WS_KEYS_PER_PARTITION: UInt32 = 384


@always_inline
def _sk_ws_partitions[
    sm_count: Int
](visible_keys: UInt32, raw_grid: UInt32) -> UInt32:
    """Workspace split-K `P` for the shared-key (deep-head) route.

    Separate from `ws_p_ceiling` / `_bucket_ws`: two of their choices are
    measurably wrong here, and changing them in place would retune the other
    three routes off this route's sweep.

    Two rules (measured at d256 and d512):

    1. The binding constraint is the WAVE, so the ceiling is
       `sm_count // raw_grid` -- WITHOUT `ws_p_ceiling`'s clamp floor, which
       overshoots a non-monotone cliff here.
    2. Snap DOWN to a `splitk_p_ladder` rung, never return the raw ceiling: an
       off-rung `P` falls back to the generic runtime-`P` combine, which costs
       more than the extra wave, and the raw ceiling sits on a bistable wave
       edge.

    `SK_WS_KEYS_PER_PARTITION` bounds the shallow-key case from over-splitting.
    Evidence is four d512 and five d256 shapes (batch in {1,4}, keys in
    {4096,8192,32768}); outside that box is extrapolation.
    """
    # `visible_keys`, not the raw cache length: a windowed mask over a long
    # cache would otherwise be split into partitions that see nothing. Those
    # are correctness-safe (they egress `-inf` and the combine folds them) but
    # pure launch cost. Inert for Null/Causal, where `start_column == 0`.
    var by_keys: UInt32 = max(
        UInt32(1), visible_keys // SK_WS_KEYS_PER_PARTITION
    )
    var by_wave: UInt32 = max(
        UInt32(1), UInt32(sm_count) // max(raw_grid, UInt32(1))
    )
    var want: UInt32 = min(by_keys, by_wave)
    # Ascending ladder, so the last rung at or below `want` wins -- i.e. snap
    # DOWN. `1` is not a rung and is the correct answer when nothing fits: the
    # caller's `> 1` test then takes the plain single-CTA launch.
    var p: UInt32 = 1
    comptime for v in splitk_p_ladder[sm_count]():
        if UInt32(v) <= want:
            p = UInt32(v)
    return p


@always_inline
def _visible_keys[
    MaskType: MHAMask, //, BM_mask: Int, BN: Int, page_size: Int
](mask: MaskType, num_keys: UInt32) -> UInt32:
    """Size the split-K ladder against keys the mask leaves visible, not the
    raw cache length.

    Split-K `P` divides the key range a single query tile iterates, so the
    right measure is that band width, `num_keys - start_column` -- the same
    quantity `mha.mojo`'s FA2 decode correction and the FA4 warps work in
    (`load_warp.mojo` / `mma_warp.mojo`).

    Evaluated at the LAST query row (no prefill/decode distinction):

    * `Null` / `Causal` / `CausalPadding`: `start_column == 0`, so the result is
      exactly `num_keys` -- the correction is inert.
    * `SlidingWindow*` / `Chunked`: the band is ~`window` wide at every row, and
      the last row reports it faithfully. The first row would report
      `window + max_prompt_len` and under-correct long windowed prefill.

    So there is deliberately no `max_prompt_len` test here.

    `start_column` (not `total_iters`) because `total_iters` is NOT host-callable
    for every mask: `CausalPaddingMask.total_iters` dereferences its
    `valid_lengths` tensor in DEVICE memory, so calling it from this host-side
    dispatch segfaults. `start_column` is host-safe for every mask in the tree,
    and `mha.mojo` already depends on that. Using `num_keys` as the right edge is
    conservative (never under-partitions).

    O(1) for every mask that reaches here: the only masks whose `start_column`
    still scans (`MaterializedMask` / `AndMask`) report `UNKNOWN_MASK`, which
    every caller gates on before calling this, so split-K is comptime-pruned for
    them. `OrMask` (hence `ChunkedCausalMask`, the Llama4 production mask) is
    closed-form.

    Parameters:
        MaskType: Concrete `MHAMask` implementation to query (inferred).
        BM_mask: Query tile height the kernel masks against (route's
            `config.PairBM_eff()`).
        BN: Key tile width in columns.
        page_size: KV cache page size in key columns (0 or 1 if unpaged).

    Args:
        mask: The mask functor.
        num_keys: Raw batch-max KV length (`max_cache_valid_length`).

    Returns:
        Visible key count in `[0, num_keys]`, for `P` sizing ONLY. The kernel
        still receives the RAW `num_keys` as its key range.
    """
    if num_keys == 0:
        return 0
    # `start_column <= num_keys - 1` for every mask, so this cannot underflow
    # and is always >= 1.
    return num_keys - mask.start_column[BM_mask, BN, page_size](
        # Pre-launch dispatch is batch-aggregate; no per-sequence id is
        # available. Masks whose `start_column` depends on `seq_id` must not size
        # `P` through here (the same caveat the FA2 decode correction in
        # `mha.mojo` carries).
        UInt32(0),
        num_keys - 1,
    )


@always_inline
def _cluster_splitk_candidates[sm_count: Int]() -> List[Int]:
    """Cluster/DSMEM split-K candidate set: partition counts whose per-GPC
    tiling wastes zero SMs (`clusters_per_wave[C] * C == sm_count`).

    A cluster/DSMEM launch is GPC-resident, so a `C` that doesn't evenly divide
    every GPC size leaves idle SMs past one wave. Restricting to zero-waste sizes
    means oversubscribing across waves is as cheap as the workspace route, so
    partition-count and mechanism become independent: pick `P` via the shared
    `ws_p_ceiling` / `_bucket_ws`, use cluster/DSMEM iff `P` is one of these,
    else workspace. This replaces the old per-route ladders and their
    `fits_wave` scans.

    B200 (148 SMs): only `C=2` divides every GPC size. B300 (160 SMs,
    8x20 GPCs): keep the validated even candidates `{2,4,10}`. Every candidate
    is already on some existing per-route ladder, so this only prunes the
    auto-set -- no new kernel instantiations. B300 is untested here (no B300
    hardware), matching `clusters_per_wave`'s unverified B300 branch; the
    `comptime assert`s keep it self-consistent with that model.
    """
    comptime if sm_count == 148:
        comptime assert (
            clusters_per_wave[2, sm_count]() * 2 == sm_count
        ), "B200 cluster candidate 2 does not fill all SMs"
        return [2]
    elif sm_count == 160:
        comptime for c in [2, 4, 10]:
            comptime assert (
                clusters_per_wave[c, sm_count]() * c == sm_count
            ), "B300 cluster candidate does not fill all SMs"
        return [2, 4, 10]
    else:
        comptime assert (
            False
        ), "_cluster_splitk_candidates: only B200 (148) / B300 (160) modeled"


@always_inline
def mha_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    KVRowOffsetsEngine: TensorEngine,
    SinkEngine: TensorEngine,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: UnsafePointer[Scalar[q_type], _],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    valid_length: UnsafePointer[UInt32, _],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        ImmutTileTensor1D[.uint32, Engine=KVRowOffsetsEngine]
    ],
    batch_size_arg: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type, Engine=SinkEngine]],
    # Caller-supplied EXACT split-K partition count (`mha.mojo`'s
    # `num_partitions`). `0` => auto (the `ws_p_ceiling` / `_bucket_ws` ladder
    # picks `P`); non-zero is honored verbatim rather than bucketed -- pinning
    # `P` is the whole point (determinism tests, partition sweeps). Honoring it
    # also forces BM < 256, since the 2Q route has no split-K at all.
    num_partitions_override: Int = 0,
) raises:
    """Dispatches the SM100 FA4 flash-attention kernel for a prefill or decode workload.

    Selects between the 1Q split-K and 2Q FA4 configurations based on a
    occupancy and prompt-length heuristic, constructs the Q/K/V/O TMA tile
    descriptors and transient tile scheduler, threads optional ragged
    valid-length, KV-row-offset, and sink-attention arguments through to the
    compiled kernel, and enqueues the launch onto the supplied device context.

    Decode is not a separate route: a single-token prompt is just the shortest
    prefill, so it takes the same ladder and lands on the smallest tile that
    single-tiles it. `max_prompt_len_arg` must be dynamic even when it is 1 --
    see the note on the missing `_is_decoding` guard in the body.

    Parameters:
        q_type: Element type of the query tensor (inferred).
        KVType: Key/value operand descriptor with dtype and page size
            (inferred).
        MaskType: Attention mask scheme applied to the Q@K' scores (inferred).
        output_type: Element type of the attention output buffer (inferred).
        MaxPromptLenType: Optionally-static type encoding the maximum prompt
            length (inferred).
        KVRowOffsetsEngine: `TensorEngine` policy of `kv_input_row_offsets`
            (inferred).
        SinkEngine: `TensorEngine` policy of `sink_weights` (inferred).
        config: MHA configuration supplying dtype, head count, depth, and
            swizzle mode.
        group: Number of query heads per KV head (GQA group size).
        ragged: Whether to dispatch the variable-length valid-length path.
        sink: Whether to thread sink-attention weights into the kernel.
        _is_cache_length_accurate: Whether the supplied cache length is
            accurate, threaded to the compiled kernel.

    Args:
        output: Device buffer that receives the attention output rows.
        q_arg: Pointer to the query tensor data.
        k: Key operand descriptor.
        v: Value operand descriptor.
        num_rows_q: Number of query rows to attend over.
        mask: Attention mask applied to the Q@K' scores.
        valid_length: Per-row valid KV length pointer, used when `ragged` is set.
        max_prompt_len_arg: Maximum prompt length, optionally static.
        max_cache_valid_length_arg: Maximum valid KV cache length across the batch.
        scale: Scalar applied to the Q@K' product before softmax.
        kv_input_row_offsets: Optional per-row KV input offsets for ragged layouts.
        batch_size_arg: Number of sequences in the batch.
        ctx: Device context used to build TMA descriptors and enqueue the kernel.
        sink_weights: Optional sink-attention weights used when `sink` is set.
        num_partitions_override: Exact split-K partition count to use, or `0`
            for the automatic ladder. A non-zero value is honored verbatim
            (cluster/DSMEM when it is a `CLUSTER_SPLITK_CANDIDATES` member,
            else the workspace route, which admits any `P`) and additionally
            forces a BM < 256 route, since 2Q cannot split at all. Overriding
            bypasses the capture-invariant `ws_p_ceiling` ceiling and the
            measured Layout-E workspace guards.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    # NOTE: deliberately NO `_is_decoding[MaxPromptLenType]()` guard here.
    # `mha.mojo` routes every SM100 `depth <= 128` half-float/fp8 shape here
    # regardless of `is_token_generation`; a `seq_len == 1` prompt arrives as a
    # `DynamicInt(1)` and lands on the smallest single-tiling tile. What's
    # unsupported is a *statically* known `1`: the Q TMA builder hard-codes
    # `decoding=False`, so a `StaticInt[1]` caller would silently get the wrong
    # descriptor. `kernel.mojo`'s `_is_decoding(...) == False` assert is the
    # fail-loud backstop -- don't reinstate a guard here, it would re-block the
    # runtime-1 decode route.
    comptime fa4_config_2q = FA4Config[KVType.dtype](
        num_q_heads=config.num_heads,
        group=group,
        qk_depth=config.depth,
        ov_depth=config.depth,
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
        is_mla=False,
    )
    # `ws_shared_key` deep-arm admission. A DEPTH-LEVEL predicate
    # (`config.depth in (256, 512)`); `mha.mojo`'s `fa4_route` is the sole feeder
    # of d256/d512 here -- it admits decode at every depth and short prefill
    # (`max_prompt_len * group <= 32`), routing large d256/d512 prefill
    # upstream, and only for ragged/unpadded callers (FA4 drops `valid_length`
    # when `ragged` is False, so a padded non-ragged caller would attend its pad
    # rows).
    # Load-bearing: it gates the 2Q `supported()` assert below AND the num_q
    # dispatch's first arm -- the 2Q config is degenerate at these depths
    # (`BN == 0` at d256, negative at d512), so an unconditional assert would
    # fail before the deep arm ran. If `fa4_route` and this predicate ever
    # disagree, that assert fires loudly.
    comptime deep_ws_shared_key = config.depth in (256, 512)
    comptime if not deep_ws_shared_key:
        comptime assert fa4_config_2q.supported(), fa4_config_2q.description()

    var q = q_arg.bitcast[Scalar[KVType.dtype]]().unsafe_origin_cast[
        q_arg.origin
    ]()

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)

    @__parameter
    @always_inline
    def with_fa4_config[
        NumPartitionsType: OptionallyStaticInt,
        PartitionType: MHAPartitionScheme,
        OWorkspaceType: OptionalPointer,
        //,
        fa4_config: FA4Config[KVType.dtype],
    ](
        num_partitions: NumPartitionsType,
        partition: PartitionType,
        # Workspace (traditional/unfused) split-K O target. Supplied only when
        # `PartitionType.do_partition`: the O store is redirected here (a
        # `[P, num_rows_q, num_q_heads, ov_depth]` buffer) instead of `output`,
        # and a separate combine kernel merges it into `output`. Every other
        # launch omits it and infers the null conformer from this default.
        o_workspace_ptr: OWorkspaceType = NullPointer[output_type](),
    ) raises:
        # `num_partitions` is ALWAYS static: the 2Q config (and its in-kernel 1Q
        # switch) carry a static 1, and each num_q==1 split-K config is compiled
        # once per static partition count `P` (cluster size `P`) — the dispatch
        # picks which `P` kernel to launch at runtime. So the cluster size is a
        # comptime constant that matches the kernel's static `nvvm.cluster_dim`.
        comptime assert Bool(
            NumPartitionsType.static_value
        ), "split-K num_partitions must be static (compiled once per P)"
        # The O-store redirect and the partition scheme are one decision:
        # workspace split-K writes partials to `o_partial`, every other route
        # writes straight to `output`. Biconditional, so BOTH halves of a
        # mismatch are a compile error rather than a TMA descriptor built over a
        # dangling address.
        comptime assert PartitionType.do_partition == (
            not OWorkspaceType.is_null
        ), (
            "workspace split-K must supply the O-partial target, and only it"
            " may: `do_partition` and the O pointer must agree"
        )
        comptime assert (
            OWorkspaceType.is_null or OWorkspaceType.dtype == output_type
        ), "the O-partial workspace must match the kernel's output dtype"
        comptime swizzle_mode = fa4_config.swizzle_mode
        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # Q/K/V/S/P buffers governed by `swizzle_mode`). The softmax warp loads
        # O one-row-per-thread and writes it row-major, avoiding cross-thread
        # shuffles and swizzling while staying bank-conflict-free.
        comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
        comptime BM = fa4_config.BM
        comptime fuse_gqa = fa4_config.fuse_gqa
        comptime num_threads = fa4_config.num_threads
        # `MMA_M // cta_group` drives q_tma_op and ragged_tma_store BM under a
        # unified expression: 128 for 2Q single-CTA (128 // 1), 2Q pair-CTA
        # (256 // 2), and 1Q single-CTA (128 // 1); 32 for the WS BM=32 config
        # (MMA_M=32 // 1).
        comptime BM_per_mma = fa4_config.MMA_M // fa4_config.cta_group()
        comptime assert BM == 32 or BM == 64 or BM == 128 or BM == 256

        # Batch the O store into one TMA per issuer: the box covers
        # `ceil(n_blocks/2)` swizzle blocks, so single-issuer writeback emits 2
        # copies and the 1Q combine 1 per WG (vs `n_blocks` per-block). Fused
        # GQA batches via the RaggedTMA3DTile merge to stay within the 5D TMA
        # limit; only swizzled-output callers fall back to per-block (0).
        # 1Q split-K (reduce-scatter) and the WS (MMA_M=32) combine both
        # TMA-store from a per-block egress whose band offset isn't a {0, half}
        # batched box, so they need the per-block (rank-3) descriptor -- `0`
        # here flags that to `fa4_splitk_combine_write`. Every non-split config
        # keeps the batched store.
        comptime store_blocks_per_op = 0 if (
            fa4_config.splitk_partitions > 1 or fa4_config.use_ws
        ) else o_store_tma_blocks_per_op[
            output_type,
            output_swizzle_mode,
            fa4_config.ov_depth,
            fa4_config.group if fuse_gqa else 1,
            depth_splits=2,
        ]()

        comptime RaggedStoreType = RaggedTMA3DTile[
            output_type,
            output_swizzle_mode,
            BM=BM_per_mma,
            BN=fa4_config.ov_depth,
            middle_dim=fa4_config.num_kv_heads if fuse_gqa else fa4_config.num_q_heads,
            group=fa4_config.group if fuse_gqa else 1,
            tma_blocks_per_op=store_blocks_per_op,
        ]

        # Workspace (traditional/unfused) split-K redirects the O store to the
        # per-partition workspace `o_partial` with a P-extended row axis; the
        # kernel writes partition `p`'s tile at ragged row `p*num_rows_q + token`
        # and a separate combine kernel merges into `output`. Every other config
        # stores straight to `output`. `do_partition` is comptime, so exactly one
        # branch is codegen'd.
        var ragged_tma_store: RaggedStoreType
        comptime if PartitionType.do_partition:
            ragged_tma_store = RaggedStoreType.create(
                ctx,
                # `OptionalPointer.value()` is immutable — its other users are
                # all read-only inputs, whereas this is a store target. `rebind`
                # reconciles `OWorkspaceType.dtype` with `output_type`, which the
                # assert above pins to the same dtype (the tie is not expressible
                # through the trait).
                rebind[UnsafePointer[Scalar[output_type], MutAnyOrigin]](
                    o_workspace_ptr.value().unsafe_mut_cast[True]()
                ),
                rows=Int(partition.max_num_partitions()) * num_rows_q,
            )
        else:
            ragged_tma_store = RaggedStoreType.create(
                ctx,
                output.unsafe_ptr(),
                rows=num_rows_q,
            )

        var q_tma_op = q_tma[
            swizzle_mode,
            BM=BM_per_mma,
            depth=fa4_config.qk_depth,
            q_num_heads=fa4_config.num_q_heads,
            group=fa4_config.group,
            decoding=False,
            fuse_gqa=fuse_gqa,
            num_qk_stages=fa4_config.num_qk_stages,
        ](ctx, q, num_rows_q)
        # Depth-chunk TMA fold (SM100): fold the BK0 (K) / v_cols_per_cta (V)
        # depth chunks into one rank-4 TMA when byte-equivalent. Each
        # `kv_tma_fold_chunks` is the single source of truth shared with the
        # `tma_copy_k` / `tma_copy_v` issue sites in `load_warp.mojo`; the builder
        # and issue site must pass identical args so the baked descriptor rank and
        # issue-coord rank agree.
        #   K: smem_BN == k_rows_per_cta (K's per-CTA tile_rows); box_rows ==
        #      kv_sub_tile_rows(k_rows_per_cta, page_size).
        #   V: smem_BN == BN (V's tile_rows, num_v_sub_tiles == 1); box_rows ==
        #      kv_sub_tile_rows(BN, page_size).
        comptime k_sub_BN = kv_sub_tile_rows(
            fa4_config.k_rows_per_cta(), KVType.page_size
        )
        comptime k_row_major = fa4_config.k_row_major()
        comptime k_fold_chunks = kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=fa4_config.BK0,
            head_size=fa4_config.qk_depth,
            box_rows=k_sub_BN,
            smem_BN=fa4_config.k_rows_per_cta(),
            page_size=KVType.page_size,
            row_major=k_row_major,
        ]()
        # Producer/consumer agreement: if the Q@K' consumer reads the page-dense
        # (row-major) K layout, the producer MUST have actually folded it
        # (`k_fold_chunks >= 2`). `k_row_major()` mirrors this predicate, so a
        # mismatch here means the two drifted.
        comptime assert (not k_row_major) or (
            k_fold_chunks > 1
        ), "k_row_major() implies the K row-major fold; predicate drift"
        var k_tma_op = k.create_tma_tile[
            fa4_config.swizzle_mode,
            BN=k_sub_BN,
            depth=fa4_config.qk_depth,
            BK=fa4_config.BK0,
            fold_chunks=k_fold_chunks,
            row_major=k_row_major,
        ](ctx)
        # V TMA box geometry per layout comes from `v_tma_box_rows()` /
        # `v_tma_box_cols()` -- the one selector dispatch, kernel, and the
        # `fa4_load` signature all route through (see their docstrings). Byte
        # layout for Layout-E is CONTIGUOUS `mn = p*ov_depth + d`.
        #
        # `v_row_major` / `v_fold_chunks` stay inline scalar ternaries (NOT a
        # `comptime if`/`else` branching the whole `create_tma_tile`): Mojo
        # unifies a `var`'s type structurally across both branches, so two
        # different `TMATensorTile` shapes assigned to `v_tma_op` in separate
        # branches fail to parse. Layout-E forces them False/1 (Phase 1:
        # correctness-only, no row-major page-dense fold).
        comptime v_sub_BN = fa4_config.v_tma_box_rows(KVType.page_size)
        comptime v_sub_cols = fa4_config.v_tma_box_cols()
        comptime v_row_major = (
            False if fa4_config.m_pack == 2 else fa4_config.v_row_major()
        )
        comptime v_fold_chunks = 1 if fa4_config.m_pack == 2 else kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=v_sub_cols,
            head_size=fa4_config.ov_depth,
            box_rows=v_sub_BN,
            smem_BN=fa4_config.v_tma_tile_rows(),
            page_size=KVType.page_size,
            row_major=v_row_major,
        ]()
        # Producer/consumer agreement: if the P@V consumer reads the page-dense
        # (row-major) V layout, the producer MUST have actually folded it
        # (`v_fold_chunks >= 2`). `v_row_major()` mirrors this predicate, so a
        # mismatch here means the two drifted. Trivially holds for Layout-E
        # (`v_row_major` forced False).
        comptime assert (not v_row_major) or (
            v_fold_chunks > 1
        ), "v_row_major() implies the V row-major fold; predicate drift"
        # Non-vacuity: a geometry change that silently drops this fold back to
        # 1 is a 4x TMA-issue regression with a green build and green tests.
        comptime v_sk_fold_must_engage = fa4_config.ws_shared_key and (
            KVType.page_size == 0
            or KVType.page_size >= fa4_config.v_tma_tile_rows()
        )
        comptime assert (
            not v_sk_fold_must_engage
        ) or v_fold_chunks > 1, (
            "shared-key V depth-chunk TMA fold did not engage"
        )
        var v_tma_op = v.create_tma_tile[
            fa4_config.swizzle_mode,
            BN=v_sub_BN,
            depth=fa4_config.ov_depth,
            BK=v_sub_cols,
            fold_chunks=v_fold_chunks,
            row_major=v_row_major,
        ](ctx)
        comptime PairBM_eff = fa4_config.PairBM_eff()
        comptime SchedulerType = TransientScheduler[
            UInt32(PairBM_eff),
            UInt32(
                fa4_config.num_kv_heads if fuse_gqa else fa4_config.num_q_heads
            ),
            flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
            pair_cta=fa4_config.pair_cta,
            splitk_partitions=UInt32(fa4_config.splitk_partitions),
        ]
        var scheduler: SchedulerType = SchedulerType()

        @__parameter
        @always_inline
        def with_sink[SinkType: OptionalPointer](sink_ptr: SinkType) raises:
            @__parameter
            @always_inline
            def with_kv_offsets[
                KVRowOffsetsType: OptionalPointer
            ](kv_row_offsets: KVRowOffsetsType) raises:
                @__parameter
                @always_inline
                def with_valid_length[
                    ValidLengthType: OptionalPointer
                ](valid_len: ValidLengthType) raises:
                    # the pack contains all possibly 0-sized objects
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
                        mask,
                        scheduler,
                        valid_len,
                        sink_ptr,
                        kv_row_offsets,
                        max_prompt_len_arg,
                        partition,
                    }

                    var max_num_prompt_tiles: UInt32 = ceildiv(
                        max_prompt_len_arg.as_uint32(), UInt32(PairBM_eff)
                    )
                    # Over-launch the prompt-tile (x) axis by the partition count
                    # for the traditional (workspace) split-K scheme. For
                    # `NoPartition` this is `*1` (byte-identical to the non-split
                    # launch); for `SplitKPartition` it widens x by `P` so the grid
                    # has one CTA per (tile, partition). This matches `get_seq_info`,
                    # which inflates the scheduler tile space by the runtime
                    # `partition.num_partitions()` (see `common.mojo`). No cluster is
                    # formed (workspace split-K keeps `splitk_partitions == 1`, so
                    # `cluster_size()==1` and `cluster_dim=None`); over-launched CTAs
                    # that map past the real prompt are marked invalid by
                    # `SeqInfo.is_valid` and early-return.
                    var block_x: UInt32 = (
                        max_num_prompt_tiles * partition.max_num_partitions()
                    )
                    logger.info(
                        "------ Dispatching to SM100 FMHA-",
                        fa4_config.num_q,
                        "Q ------",
                    )
                    logger.info(
                        "QKV Type:",
                        KVType.dtype,
                        "Depth:",
                        fa4_config.qk_depth,
                        "Number of Q // KV Heads:",
                        fa4_config.num_q_heads,
                        "//",
                        fa4_config.num_kv_heads,
                        "Batch Size:",
                        batch_size,
                        "Max Num Prompt Tiles:",
                        max_num_prompt_tiles,
                    )

                    # Covers the in-kernel 1Q/2Q switch: when
                    # `can_switch_to_1q()` the kernel constructs the 1Q smem
                    # layout over the same dynamic smem region, so this is the
                    # max of both footprints (see `FA4Config.launch_smem_used`).
                    comptime smem_use = fa4_config.launch_smem_used()

                    comptime KernelStruct = SM100MHA2Q[
                        KVType,
                        output_type,
                        MaskType,
                        SchedulerType,
                        fa4_config,
                        ValidLengthType,
                        SinkType,
                        KVRowOffsetsType,
                        _is_cache_length_accurate,
                        MaxPromptLenType,
                        PartitionType,
                    ]
                    # Every config uses the static `kernel` entry; the cluster
                    # size is baked into its `nvvm.cluster_dim` metadata.
                    comptime kernel = KernelStruct.kernel

                    var cluster_dim: OptionalReg[Dim] = None
                    # Unifies pair-CTA (cluster_size==2) and num_q==1 split-K
                    # (cluster_size==P): both are the comptime `cluster_size()`,
                    # matching the static `nvvm.cluster_dim` metadata on the
                    # `kernel` entry (split-K is compiled once per static P).
                    comptime if fa4_config.cluster_size() > 1:
                        cluster_dim = Dim(fa4_config.cluster_size(), 1, 1)
                    comptime name = String(
                        "nq",
                        fa4_config.num_q,
                        "d",
                        fa4_config.qk_depth,
                        "qh",
                        fa4_config.num_q_heads,
                        "kvh",
                        fa4_config.num_kv_heads,
                        ".",
                    )
                    ctx.enqueue_function[kernel](
                        q_tma_op,
                        k_tma_op,
                        v_tma_op,
                        ragged_tma_store,
                        k,
                        scale,
                        batch_size,
                        max_cache_valid_length,
                        pack,
                        # Total query-row count, used only by the workspace
                        # (traditional/unfused) split-K egress as the per-partition
                        # `o_partial`/`lse_partial` row stride; harmless (unused)
                        # for every other config.
                        UInt32(num_rows_q),
                        grid_dim=SchedulerType.grid_dim(
                            batch_size, block_x, num_partitions.as_uint32()
                        ),
                        block_dim=(num_threads, 1, 1),
                        cluster_dim=cluster_dim,
                        shared_mem_bytes=smem_use,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                            UInt32(smem_use)
                        ),
                        attributes=pdl_launch_attributes(MHA_PDL_LEVEL),
                    )

                # --- ragged dispatch ---
                comptime if ragged:
                    with_valid_length[NonNullPointer[.uint32]](
                        {valid_length.as_imm().as_unsafe_any_origin()}
                    )
                else:
                    with_valid_length[NullPointer[.uint32]]({})

            # --- kv_input_row_offsets dispatch ---
            if kv_input_row_offsets:
                with_kv_offsets[NonNullPointer[.uint32]](
                    {kv_input_row_offsets.value().ptr}
                )
            else:
                with_kv_offsets[NullPointer[.uint32]]({})

        # --- sink dispatch ---
        comptime if sink:
            with_sink[NonNullPointer[KVType.dtype]](
                {
                    rebind[UnsafePointer[Scalar[KVType.dtype], ImmutAnyOrigin]](
                        sink_weights.value().ptr
                    )
                }
            )
        else:
            with_sink[NullPointer[KVType.dtype]]({})

    @__parameter
    @always_inline
    def launch_workspace[fa4_config: FA4Config[KVType.dtype]](p: UInt32) raises:
        """Traditional (unfused / workspace) split-K launch: run the plain 1Q
        config over an over-launched grid that writes each partition's
        locally-normalized `O_p/l_p` + fused per-row LSE to a transient
        `o_partial`/`lse_partial` global workspace, then merge them into `output`
        with a separate `fa4_splitk_combine` kernel (same stream, so it runs after
        the attention completes). No launch cluster, no in-kernel DSMEM combine.

        `p` is both the grid over-launch factor (`block_x *= p`) and the
        workspace row extent, and every launched CTA writes a real or
        `-inf`-empty row, so it is also the count the combine reduces.
        `SplitKPartition` keeps `num`/`max` as separate fields (FA2 decode does
        distinguish them); should this route ever over-launch, split this
        parameter rather than the struct.

        The config itself stays at `splitk_partitions == 1` (so
        `cluster_size() == 1` and `cluster_dim=None`) and `P` is carried purely
        at runtime by the `SplitKPartition`. That partition SCHEME is a comptime
        TYPE flowing into `with_fa4_config`, which is why this is a separate
        generic `def` rather than a `comptime if` selecting a shared
        `var partition`: Mojo unifies a `var`'s type across both branches, so
        both ctors would have to type-check. Same idiom as
        `with_valid_length[NonNullPointer]` vs `[NullPointer]`.
        """
        # `o_partial`: `[p, num_rows_q, num_q_heads, ov_depth]` locally-
        # normalized partials. `lse_partial`: one fused-LSE f32 per (partition,
        # row, q-head), `[p, num_rows_q, num_q_heads]` -- matching the O-store
        # TMA row extent.
        var rows_x_heads = num_rows_q * fa4_config.num_q_heads
        var lse_partial = ctx.enqueue_create_buffer[.float32](
            Int(p) * rows_x_heads
        )
        var o_partial = ctx.enqueue_create_buffer[output_type](
            Int(p) * rows_x_heads * fa4_config.ov_depth
        )
        # `-D FA4_WS_POISON=1` supplies the garbage that grades
        # `fa4_splitk_combine.mojo`'s no-initialization contract. The defect it
        # catches is `0 * garbage`: an empty partition skips its O store, and
        # fresh device memory is usually zeroed, so a missing `scale != 0`
        # select or a dropped `-inf` LSE write looks correct. Poisoning
        # `lse_partial` is the load-bearing half -- `-inf` there is the only
        # thing that makes `scale` exactly 0. Off by default and
        # comptime-pruned, so production codegen is untouched.
        comptime if FA4_WS_POISON:
            ctx.enqueue_memset(lse_partial, nan[.float32]())
            ctx.enqueue_memset(o_partial, nan[output_type]())
        var partition = SplitKPartition[.float32](
            lse_partial.unsafe_ptr().as_unsafe_any_origin(),
            p,
            p,
        )
        # Attention: per-partition partials -> workspace (O store redirected to
        # `o_partial` via `o_workspace_ptr`; fused LSE via the `SplitKPartition`
        # pointer = `lse_partial`).
        with_fa4_config[fa4_config](
            StaticInt[1](),
            partition,
            NonNullPointer[output_type](o_partial),
        )
        # Combine: reduce the partials into `output`. Bind to
        # `partition.num_partitions()` (not `max_num_partitions()`) so an
        # over-launching caller would need no combine-side change.
        fa4_splitk_combine[output_type, fa4_config.ov_depth](
            ctx,
            output.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
            o_partial.unsafe_ptr().as_unsafe_any_origin(),
            lse_partial.unsafe_ptr().as_unsafe_any_origin(),
            partition.num_partitions(),
            UInt32(num_rows_q),
            UInt32(fa4_config.num_q_heads),
        )
        _ = o_partial^
        _ = lse_partial^

    # --- num_q dispatch ---
    comptime if deep_ws_shared_key:
        comptime fa4_config_ws_sk = fa4_config_2q.ws_shared_key_vehicle()
        comptime ws_sk_mask_ok = MaskType.nonfull_sets[
            fa4_config_ws_sk.PairBM_eff(), fa4_config_ws_sk.BN
        ]()[0] != TileMaskStatus.UNKNOWN_MASK
        comptime assert fa4_config_ws_sk.supported() and ws_sk_mask_ok, (
            "ws_shared_key deep route cannot fire at this shape: "
            + fa4_config_ws_sk.description()
        )
        # UNFUSED (workspace) split-K at a runtime-pinned `P`. This is the arm
        # `supported()`'s shared-key conjunct (4) deliberately carves out: it
        # rejects the FUSED cross-CTA split-K, whose `ws_maxsum1` and
        # `ws_l2_stage` collapse onto one pointer once `ws_epilogue_ml_f32()`
        # is 0 -- while `ws_o_row_off` plus `_ws_write_lse` ride the plain
        # epilogue arm untouched, which is the route production decode takes.
        #
        # `fa4_config_ws_sk` therefore keeps BOTH invariants `launch_workspace`
        # needs and `supported()` already guarantees: `splitk_partitions == 1`
        # (required by `kernel.mojo`'s `not (do_partition and splitk_partitions
        # > 1)`) and `num_q == 1` (required by its "workspace egress is 1Q-only"
        # fence). `P` rides at runtime on the `SplitKPartition`, and every warp
        # recovers its window from `splitk_window` via the grid index.
        #
        # `P` selection. An explicit `num_partitions` is honored VERBATIM --
        # pinning it is the whole point for determinism tests and partition
        # sweeps, and bucketing it to a rung would defeat the caller. `0`
        # (auto) goes through `_sk_ws_partitions`, which is this route's own
        # measured formula rather than the `ws_p_ceiling` / `_bucket_ws` pair
        # the other three routes share; see its docstring for the two places
        # those two are measurably wrong here.
        var sk_p: UInt32
        if num_partitions_override > 0:
            sk_p = UInt32(num_partitions_override)
        else:
            sk_p = _sk_ws_partitions[ctx.default_device_info.sm_count](
                _visible_keys[
                    fa4_config_ws_sk.PairBM_eff(),
                    fa4_config_ws_sk.BN,
                    KVType.page_size,
                ](mask, max_cache_valid_length),
                _raw_grid[fa4_config_ws_sk](
                    max_prompt_len_arg.as_uint32(), batch_size
                ),
            )
            # Range guard on the AUTO path only -- an explicit override is the
            # caller's business and `mha.mojo` already bounds it.
            #
            # This is the only instrument that can catch a bad auto-selection,
            # and the reason is the same one that makes this whole route hard
            # to test: split-K is a work DECOMPOSITION, so a wrong-but-valid
            # `P` still computes the right answer and no oracle can see it. A
            # `P` outside this range is the one auto-selection failure that is
            # NOT invisible -- `0` would silently demote to the single-CTA
            # launch, and anything above `P_MAX` overruns
            # `fa4_splitk_combine`'s per-lane LSE array. Both are bounded by
            # construction today (`_sk_ws_partitions` floors at 1 and its top
            # rung is `sm_count`), which is exactly why the bound wants an
            # assertion rather than a sentence in a docstring.
            debug_assert[assert_mode="safe"](
                sk_p >= UInt32(1) and sk_p <= UInt32(P_MAX),
                (
                    "shared-key auto-selected a partition count outside"
                    " [1, FA4_COMBINE_P_MAX]"
                ),
            )
        if sk_p > UInt32(1):
            launch_workspace[fa4_config_ws_sk](sk_p)
        else:
            # Reached either by auto-selection returning 1 (short cache,
            # or `raw_grid` already filling the machine) or by an explicit
            # `num_partitions <= 1`. NOT reachable with an override >= 2:
            # workspace split-K is unconditionally available on this route, so
            # such an override always takes the branch above. Kept as a forward
            # guard -- any future comptime gate on that call would silently
            # serve `P == 1` and let a caller believe it pinned a count it
            # never got. This is the only instrument that can catch that, since
            # split-K is a work decomposition and a demoted cell still computes
            # the right answer.
            debug_assert[assert_mode="safe"](
                num_partitions_override <= 1,
                (
                    "num_partitions override could not be honored on the"
                    " shared-key route: workspace split-K is unavailable"
                ),
            )
            with_fa4_config[fa4_config_ws_sk](
                StaticInt[1](), NoPartition[DType.float32]()
            )
    else:
        # Static per-C split-K: the num_q==1 split-K kernel is compiled ONCE per
        # candidate partition count `P` (== cluster size) in
        # `CLUSTER_SPLITK_CANDIDATES` (defined below, shared by all three
        # split-K routes -- see `_cluster_splitk_candidates`); the dispatch
        # picks which `P` to LAUNCH from occupancy + KV length.
        # `num_partitions` is an `OptionallyStaticInt` carrying a static `P` (or
        # a static `1` for the 2Q config and its in-kernel 1Q switch), so the
        # cluster size / grid / combine are all comptime constants -- there is no
        # runtime cluster dimension.
        comptime fa4_config_1q = fa4_config_2q.with_num_q(1)

        # The GPC fragmentation model (`clusters_per_wave`) covers B200 (148 SMs)
        # and B300 (160 SMs); its LUT folds at comptime keyed on
        # `ctx.default_device_info.sm_count` (a comptime value). The helper's own
        # `else` branch is the single source of truth for an unsupported chip, so
        # no standalone assert is needed here.

        # 1Q-vs-2Q gate (unchanged): pick 1Q when (a) max_prompt_len fits one 1Q
        # tile (2Q's BM=256 would waste >= 50% of Q rows) or (b) the unclamped 2Q
        # grid only fills <= half the SMs, so halving BM doubles the grid without
        # oversubscribing.
        var max_prompt_len_u32: UInt32 = max_prompt_len_arg.as_uint32()
        var raw_grid_2q: UInt32 = _raw_grid[fa4_config_2q](
            max_prompt_len_u32, batch_size
        )
        # SM count is the comptime compile-target value (same source the
        # `clusters_per_wave` wave-fit uses below via `default_device_info`),
        # not a runtime `get_attribute` query -- keeps the 1Q/2Q grid gate and
        # the split-K wave-fit reasoning off one consistent occupancy model.
        comptime sm_count: UInt32 = UInt32(ctx.default_device_info.sm_count)
        comptime grid_threshold: UInt32 = sm_count // 2
        # `BM_eff`/`PairBM_eff` are P-independent (split-K forces cta_group==1),
        # so the 1Q geometry uses `fa4_config_1q` directly.
        comptime bm_eff_1q = UInt32(fa4_config_1q.BM_eff())
        # Shared cluster/DSMEM split-K candidate set (M4): only the
        # partition counts that fill every SM with zero GPC fragmentation
        # (see `_cluster_splitk_candidates`). Computed once and shared by the
        # 1Q, WS-G, and WS-E crossovers below -- with this restriction, the
        # per-route "predict what cluster would pick" scans and their
        # `clusters_per_wave`/`fits_wave` wave-fit gates are no longer
        # needed: a bucketed `P` either IS one of these (use cluster/DSMEM,
        # cheaper) or it ISN'T (use workspace, which has no such
        # restriction).
        comptime CLUSTER_SPLITK_CANDIDATES = _cluster_splitk_candidates[
            ctx.default_device_info.sm_count
        ]()
        # Warp-specialized BM=32 packed-TMEM datapath for very short prompts. The
        # BM=32 tile holds `group` q-heads x BM_eff seq positions under fuse_gqa
        # (BM_eff = 32 // group; 8 at group=4), and the 8-way intra-CTA split-K
        # (2 softmax WGs x 4 packed-TMEM quarters) extracts the parallelism from
        # the KV reduction.
        # Scoped to `depth in {64, 128}` ONLY: the WS per-warp band-packing
        # combine helpers impose `ov_depth % depth_tile == 0` where
        # `depth_tile = 256 // m_pack` = 64 for Layout-G (m_pack=4), so only
        # 64/128 qualify; depths 72/80/96 fail this and route to the 1Q carve
        # (whose `fa4_splitk_combine` handles them via its scalar lane-strided
        # fallback). Sinks ARE supported: the intra-CTA fold is confined to WG0
        # quarter 0 (`fold_sink and warp_idx == 0` in softmax_warp.mojo), so the
        # 8-way split adds the sink mass exactly once. `supported()` prunes
        # rope/KV-scale/non-uniform-sub-tile shapes. Routed BEFORE the 1Q
        # split-K carve so WS-eligible shapes take WS first.
        comptime if config.depth == 128 or config.depth == 64:
            # The shared-key deep decode intercept used to live here, gated by
            # this same `depth in {64,128}` block -- permanently dead, since
            # `ws_shared_key_vehicle()` needs depth 256/512. It moved to the
            # reachable `deep_ws_shared_key` arm at the top of the num_q
            # dispatch, so there is now exactly one.
            comptime fa4_config_ws = fa4_config_2q.with_bm(32)
            # Cross-CTA cluster split-K over the WS BM=32 config: each P groups
            # P single-CTA WS kernels in a launch cluster that partition the KV
            # and DSMEM-combine (a third level on top of the 8-way intra-CTA
            # split). Each P compiles its own static kernel (`StaticInt[P]`,
            # `cluster_size() == P`); production auto-sizes P from the single-tile
            # scan below. Reuses `CLUSTER_SPLITK_CANDIDATES`: for P > m_pack (==4)
            # only `m_pack` partitions own a depth band, but all P partition the
            # KV and are reduced into them, so wider P raises SM utilization like
            # the 1Q path.
            comptime ws_mask_ok = MaskType.nonfull_sets[
                fa4_config_ws.PairBM_eff(), fa4_config_ws.BN
            ]()[0] != TileMaskStatus.UNKNOWN_MASK
            # WS is validated only for masks whose visible range is statically
            # known and contiguous (`nonfull_sets[0] != UNKNOWN_MASK`: Null,
            # Causal, Chunked, SlidingWindow). Materialized/And/Or masks report
            # `UNKNOWN_MASK`; their WS softmax `mask.status(...)` over the
            # 256-wide tile window is unverified on WS, so route them to the
            # proven non-WS path. (Belt-and-suspenders: the 1Q split-K
            # `UNKNOWN_MASK` comptime assert in `fa4_softmax` already blocks
            # them from this dispatch; this keeps them off WS if that's ever
            # lifted.)
            # No cache-length gate. The WS 1Q shared-ring main loop
            # (`main_iters >= 1`) is correct for short T: the correction-SMEM
            # region is sized by the softmax-thread count `2*WARPGROUP_SIZE`,
            # not `BM` (smem.mojo), so a short prompt + long cache routes to
            # single-CTA WS here. The perf guard that keeps a huge-cache /
            # short-prompt shape off single-CTA WS is the `ws_grid < sm_count`
            # occupancy rule below.
            comptime if (fa4_config_ws.supported() and ws_mask_ok):
                # Production auto: route WS iff the whole prompt fits ONE WS
                # tile -- `BM_eff >= max_prompt_len`, where fuse_gqa packs
                # `group` q-heads into the BM=32 tile so one tile spans
                # `BM_eff = BM // group` SEQ positions (8 for group=4), not
                # 32. Beyond one tile the prompt shatters into
                # `ceildiv(seq, BM_eff)` WS tiles vs baseline's 1-2 BM=32
                # tiles; at long cache those extra KV passes lose to baseline
                # even when the grid is small, so an occupancy-only check
                # over-routed the mid-seq / small-batch corner. Single-tile
                # WS avoids that shatter tax and still fills the SMs via the
                # cross-CTA split-K scan below.
                var raw_grid_ws: UInt32 = _raw_grid[fa4_config_ws](
                    max_prompt_len_u32, batch_size
                )
                if UInt32(fa4_config_ws.BM_eff()) >= max_prompt_len_u32:
                    # Sink is NOT a gate on either mechanism here: cluster/DSMEM
                    # always supported it, and the workspace fallback gained it
                    # via the `fold_sink` partition-0 gate (see the Layout-E
                    # note at the matching site). So an explicit
                    # `num_partitions` may force workspace on a sink shape.
                    var p_bucket_ws: UInt32
                    if num_partitions_override > 0:
                        p_bucket_ws = UInt32(num_partitions_override)
                    else:
                        # Size the ladder against the keys the mask leaves
                        # VISIBLE, not the raw cache length: a windowed mask
                        # over a long cache would otherwise be split into
                        # partitions that see nothing (correctness-safe --
                        # they egress `-inf` and the combine folds them --
                        # but pure waste). Inert for non-windowing masks;
                        # see `_visible_keys`.
                        var by_cache_ws: UInt32 = _visible_keys[
                            fa4_config_ws.PairBM_eff(),
                            fa4_config_ws.BN,
                            KVType.page_size,
                        ](mask, max_cache_valid_length) // UInt32(
                            512 * config.depth // 128
                        )
                        var p_ceiling_ws: UInt32 = ws_p_ceiling[
                            ctx.default_device_info.sm_count
                        ](raw_grid_ws)
                        # `_bucket_ws` buckets UP then caps at the ceiling,
                        # so handing it `by_cache` directly is identical to
                        # pre-clamping against the ceiling first.
                        p_bucket_ws = UInt32(
                            _bucket_ws[ctx.default_device_info.sm_count](
                                Int(by_cache_ws),
                                Int(p_ceiling_ws),
                                Int(raw_grid_ws),
                            )
                        )
                    if p_bucket_ws > UInt32(1):
                        comptime for C in CLUSTER_SPLITK_CANDIDATES:
                            comptime ws_splitk_cfg = fa4_config_ws.with_splitk(
                                C
                            )
                            comptime assert (
                                ws_splitk_cfg.supported()
                            ), ws_splitk_cfg.description()
                            if p_bucket_ws == UInt32(C):
                                with_fa4_config[ws_splitk_cfg](
                                    StaticInt[C](),
                                    NoPartition[.float32](),
                                )
                                return
                        launch_workspace[fa4_config_ws](p_bucket_ws)
                        return
                    with_fa4_config[fa4_config_ws](
                        StaticInt[1](), NoPartition[.float32]()
                    )
                    return
            # Layout-E (BM=64, m_pack=2) production route: one rung down the
            # tile ladder from Layout-G, reached when the prompt did NOT
            # single-tile into BM=32. Routes BM=64 iff the prompt single-tiles
            # into it (`BM_eff_e >= max_prompt_len`, i.e. 32 < group*seq <= 64),
            # then fills the SMs via the shared `CLUSTER_SPLITK_CANDIDATES` /
            # workspace crossover. Falls through (no return) to the 1Q/2Q carve
            # when a bigger tile is needed.
            comptime fa4_config_ws_e = fa4_config_2q.with_bm(64)
            comptime ws_e_mask_ok = MaskType.nonfull_sets[
                fa4_config_ws_e.PairBM_eff(), fa4_config_ws_e.BN
            ]()[0] != TileMaskStatus.UNKNOWN_MASK
            comptime if (fa4_config_ws_e.supported() and ws_e_mask_ok):
                var raw_grid_ws_e: UInt32 = _raw_grid[fa4_config_ws_e](
                    max_prompt_len_u32, batch_size
                )
                if UInt32(fa4_config_ws_e.BM_eff()) >= max_prompt_len_u32:
                    # Unified split-K choice (mirrors WS-G / 1Q), with three
                    # WS-E-specific measured guards on top of the shared formula:
                    #  (1) `ws_e_min_keys`: workspace's fixed combine + round-trip
                    #      cost isn't amortized below a depth-keyed cache floor
                    #      (8192 @ d128, 65536 @ d64 -- d64's half-FLOP/key makes
                    #      that fixed cost a bigger fraction); below it, skip to
                    #      cluster/plain.
                    #  (2) d64 workspace is unproven beyond raw_grid<=2 (it goes
                    #      non-monotonic and regresses there on the same one-wave
                    #      computation); d128 has no such evidence. So d64 stays
                    #      restricted to raw_grid<=2 regardless of bucketed `P`.
                    #  (3) `ws_e_p_cap=64`: past ~64 partitions the per-partition
                    #      combine overhead dominates; cap the ceiling there.
                    # `sink` isn't a gate -- workspace supports it via the
                    # `fold_sink` partition-0 gate.
                    comptime ws_e_min_keys = UInt32(
                        8192 if config.depth >= 128 else 65536
                    )
                    comptime ws_e_p_cap = UInt32(64)
                    comptime ws_e_max_raw_grid_d64 = UInt32(2)
                    var p_bucket_ws_e: UInt32
                    if num_partitions_override > 0:
                        p_bucket_ws_e = UInt32(num_partitions_override)
                    else:
                        # Visible-key sizing, as in the Layout-G route above.
                        var by_cache_ws_e: UInt32 = _visible_keys[
                            fa4_config_ws_e.PairBM_eff(),
                            fa4_config_ws_e.BN,
                            KVType.page_size,
                        ](mask, max_cache_valid_length) // UInt32(
                            512 * config.depth // 128
                        )
                        var p_ceiling_ws_e: UInt32 = min(
                            ws_p_ceiling[ctx.default_device_info.sm_count](
                                raw_grid_ws_e
                            ),
                            ws_e_p_cap,
                        )
                        p_bucket_ws_e = UInt32(
                            _bucket_ws[ctx.default_device_info.sm_count](
                                Int(by_cache_ws_e),
                                Int(p_ceiling_ws_e),
                                Int(raw_grid_ws_e),
                            )
                        )
                    if p_bucket_ws_e > UInt32(1):
                        comptime for C in CLUSTER_SPLITK_CANDIDATES:
                            comptime ws_e_splitk_cfg = fa4_config_ws_e.with_splitk(
                                C
                            )
                            comptime assert (
                                ws_e_splitk_cfg.supported()
                            ), ws_e_splitk_cfg.description()
                            if p_bucket_ws_e == UInt32(C):
                                with_fa4_config[ws_e_splitk_cfg](
                                    StaticInt[C](),
                                    NoPartition[.float32](),
                                )
                                return
                        # An explicit override bypasses both measured perf
                        # guards: they are heuristics about when workspace
                        # split-K PAYS, and a caller naming `P` has already
                        # decided. `ws_e_min_keys` deliberately keeps reading
                        # the RAW cache length -- it is a fixed-cost
                        # amortization threshold, not a P-sizing input.
                        if num_partitions_override > 0 or (
                            max_cache_valid_length >= ws_e_min_keys
                            and (
                                config.depth >= 128
                                or raw_grid_ws_e <= ws_e_max_raw_grid_d64
                            )
                        ):
                            launch_workspace[fa4_config_ws_e](p_bucket_ws_e)
                            return
                    # Unreachable with an override >= 2: workspace split-K is
                    # unconditionally available on this route, so such an
                    # override always returns above (cluster when `P` is a
                    # candidate, workspace otherwise). Kept as a forward guard --
                    # any future comptime gate on the workspace fallback would
                    # silently serve `P == 1` and let a caller believe it pinned
                    # a partition count it never got.
                    debug_assert[assert_mode="safe"](
                        num_partitions_override <= 1,
                        (
                            "num_partitions override could not be honored on"
                            " the Layout-E route: workspace split-K is"
                            " unavailable"
                        ),
                    )
                    with_fa4_config[fa4_config_ws_e](
                        StaticInt[1](), NoPartition[.float32]()
                    )
                    return
        # An explicit `num_partitions` forces BM < 256. The 2Q arms below launch
        # `StaticInt[1]()` / `NoPartition` unconditionally -- 2Q has NO split-K
        # mechanism at all -- so letting an override fall through there would
        # silently pin `P == 1`. Taking the 1Q carve instead is correctness-safe
        # at any prompt length (the 1Q `_raw_grid` below shatters a long
        # prompt across BM=128 tiles); the 2Q preference is purely a
        # large-tile perf heuristic. Both configs are already compiled, so this
        # adds no instantiation.
        if (
            num_partitions_override > 0
            or max_prompt_len_u32 <= bm_eff_1q
            or raw_grid_2q <= grid_threshold
        ):
            # One split-K CLUSTER per work item; each is a size-`P` cluster
            # occupying `P` SMs within a single GPC.
            var raw_grid_1q: UInt32 = _raw_grid[fa4_config_1q](
                max_prompt_len_u32, batch_size
            )
            # Don't split a short KV into more partitions than there is K/V to
            # go around (min keys / partition); avoids mostly-empty clusters. A
            # d64 partition does half the per-partition depth work of a d128 one,
            # so it amortizes the cross-partition combine at half the keys (256
            # vs 512) and can split 2x further -- the flat-512 floor
            # under-parallelized d64 prefill. Scoped to d64
            # (`256 if depth==64 else 512`, NOT a general `512*depth//128`) so
            # only the measured depth changes; the other 1Q-legal depths in
            # [64, 256] keep 512 pending their own sweep.
            comptime one_q_min_kpp = 256 if config.depth == 64 else 512

            # Unified split-K partition-count + mechanism choice: pick one `P`
            # from the shared `ws_p_ceiling` / `_bucket_ws` formula, then launch
            # cluster/DSMEM iff `P` is in `CLUSTER_SPLITK_CANDIDATES` (the
            # zero-SM-waste sizes -- cheaper, no combine kernel / round-trip),
            # else workspace, which admits ANY `P`. Once cluster/DSMEM is
            # restricted to zero-GPC-fragmentation sizes, oversubscribing it
            # past one wave is as cheap as workspace, so one shared `P` + a
            # mechanism check replaces the old two-scan design.
            #
            # Gated on `ws1q_mask_ok`: UNKNOWN_MASK masks trip the 1Q split-K
            # `fa4_softmax` comptime assert for either mechanism. Sink IS
            # supported on the workspace route -- the `fold_sink` partition-0
            # gate (softmax_warp.mojo) folds the sink mass into partition 0's
            # `lse_partial` once and `fa4_splitk_combine` rescales it; a sink
            # shape whose bucketed `P` isn't cluster-compatible falls back to
            # `P==1` (accepted; the cluster path is the preferred sink vehicle).
            #
            # Workspace is restricted to `depth in [64, 128]` (the
            # `launch_workspace` call below). Outside that the 1Q workspace
            # split-K combine is unverified: the vectorized combine path's
            # `vec_size = depth // ...` misaligns at non-power-of-2 widths --
            # depth=96's `vec_size=3` lowers a 3-wide bf16 load to a 4-byte op at
            # a 2-byte-aligned address (`CUDA_ERROR_MISALIGNED_ADDRESS`). The
            # fix gates the vectorized path on a power-of-2 `vec_size` (depths
            # 96/160/192/224 take the scalar lane-strided fallback); cluster
            # split-K and the `P==1` fallback are unaffected.
            comptime ws1q_mask_ok = (
                MaskType.nonfull_sets[
                    fa4_config_1q.PairBM_eff(), fa4_config_1q.BN
                ]()[0]
                != TileMaskStatus.UNKNOWN_MASK
            )
            comptime if ws1q_mask_ok:
                var p_bucket: UInt32
                if num_partitions_override > 0:
                    p_bucket = UInt32(num_partitions_override)
                else:
                    # Visible-key sizing (see `_visible_keys`). Computed INSIDE
                    # the `ws1q_mask_ok` gate on purpose: for an `UNKNOWN_MASK`
                    # mask `start_column` falls back to
                    # `naively_get_first_nonempty_mask_col`, which steps
                    # `status()` -- and `status()` is exactly the method that
                    # reads DEVICE memory for a padding mask, so calling it from
                    # this host-side dispatch would segfault. Split-K is
                    # comptime-pruned for those masks right here, so the value
                    # would be unused anyway; not computing it is what keeps
                    # this host-safe. `MaterializedMask` and `AndMask` are the
                    # only two masks that still scan, and both are exactly the
                    # `UNKNOWN_MASK` set this gate excludes, so every mask that
                    # does reach `_visible_keys` resolves in O(1).
                    var by_cache: UInt32 = _visible_keys[
                        fa4_config_1q.PairBM_eff(),
                        fa4_config_1q.BN,
                        KVType.page_size,
                    ](mask, max_cache_valid_length) // UInt32(one_q_min_kpp)
                    var p_ceiling: UInt32 = ws_p_ceiling[
                        ctx.default_device_info.sm_count
                    ](raw_grid_1q)
                    p_bucket = UInt32(
                        _bucket_ws[ctx.default_device_info.sm_count](
                            Int(by_cache),
                            Int(p_ceiling),
                            Int(raw_grid_1q),
                        )
                    )
                if p_bucket > UInt32(1):
                    comptime for C in CLUSTER_SPLITK_CANDIDATES:
                        comptime splitk_cfg = fa4_config_1q.with_splitk(C)
                        comptime assert (
                            splitk_cfg.supported()
                        ), splitk_cfg.description()
                        if p_bucket == UInt32(C):
                            with_fa4_config[splitk_cfg](
                                StaticInt[C](), NoPartition[.float32]()
                            )
                            return
                    comptime if config.depth >= 64 and config.depth <= 128:
                        launch_workspace[fa4_config_1q](p_bucket)
                        return
            # P==1 (nothing to combine, or mask/depth excluded split-K):
            # launch the 1Q single-partition config -- no cluster, no DSMEM,
            # no combine.
            #
            # An override lands here when split-K is comptime-unavailable: the
            # mask reports `UNKNOWN_MASK` (`ws1q_mask_ok`), or `P` missed every
            # cluster candidate at a depth outside the workspace range
            # [64, 128]. Silently serving `P == 1` would let a test believe it
            # pinned a partition count it never got.
            debug_assert[assert_mode="safe"](
                num_partitions_override <= 1,
                (
                    "num_partitions override could not be honored on the 1Q"
                    " route: split-K unavailable (UNKNOWN_MASK, or depth"
                    " outside [64, 128])"
                ),
            )
            with_fa4_config[fa4_config_1q](
                StaticInt[1](), NoPartition[.float32]()
            )
        else:
            # Not reachable with an override: the gate above forces the 1Q carve
            # whenever `num_partitions_override > 0`.
            with_fa4_config[fa4_config_2q](
                StaticInt[1](), NoPartition[.float32]()
            )
