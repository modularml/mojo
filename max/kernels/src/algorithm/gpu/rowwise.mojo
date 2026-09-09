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
"""GPU row-wise reduction scaffolder.

The author writes one `body[ctx: Context](row_coords)` callable,
taking a comptime `Context` (the dispatch tier) and the row's
coords. The body composes a few helpers from this module:

- `reduce` — tier-aware iteration over the reduce axis, then a
  cross-thread join of the body's accumulators. Takes a per-tile
  callback `tile_fn[ws]` (which closes over input closures +
  local monoid states) and a variadic of `ReduceOp` states to pjoin
  after the loop. Zero states runs iteration only — used by
  2-pass algorithms' second pass.
- `once` — run a closure exactly once per (logical) output row.
  Picks the canonical writer thread; the body never sees
  `lane_id()` / `thread_idx.x`.
- `simd` — width-polymorphic `SIMD` constructor from a per-lane
  callback; degenerates cleanly at `w = 1`.

`launch` is the top-level entry point. It picks the tier from
shape and axis, then instantiates the matching kernel template.

No GPU primitives leak into the body: it sees `ctx`, lambdas, and
the helpers above.
"""

from std.atomic import Atomic, Ordering, fence
from std.bit import log2_floor, next_power_of_two
from max.gpu import (
    WARP_SIZE,
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu import barrier, syncwarp
from max.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from max.gpu.primitives import block
from max.gpu.intrinsics import threadfence
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.math import ceildiv
from std.math.uutils import udivmod
from std.memory import UnsafePointer, stack_allocation
from std.sys import simd_width_of, size_of, get_defined_int
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    is_apple_gpu,
)
from std.utils.coord import Coord, DynamicCoord, coord_to_index_list
from std.utils.static_tuple import StaticTuple

from algorithm.cpu.rowwise import SerialReducer
from algorithm.rowwise_types import (
    Context,
    ContextParams,
    ReduceTier,
    RowBody,
    RowCoord,
    _num_outputs_excluding_axis,
)
from std.algorithm.backend.unswitch import unswitch
from algorithm.reduce_op import ReduceOp, Reducer
from max.algorithm.reduction import _get_nd_indices_from_flat_index


comptime _PDL_LEVEL = PDLLevel.ON
comptime _SM_OVERPROVISION = 32

# Output rows per SM the long-reduce-axis tiled kernel needs to beat
# cooperative.
comptime _THREAD_SAT_OUTPUTS_PER_SM = 256

# Minimum output rows per SM for the short-reduce-axis tiled tier: a
# measured cooperative-vs-tiled crossover, not an SM-coverage guarantee
# (both tiers are latency-bound in this band). The MI355 value doubles
# as the default for unmeasured devices.
comptime _TILED_MIN_ROWS_DEFAULT = 16
comptime _TILED_MIN_ROWS_NVIDIA = 1

# Row-size cutoff below which the non-inner tiled kernel beats the
# cooperative one, scaled at the use site by cooperative's wave count.
# Measured crossovers; the MI355 value doubles as the default.
comptime _ROW_SIZE_CUTOFF_NVIDIA = 32
comptime _ROW_SIZE_CUTOFF_DEFAULT = 8

# Inner-axis block-tier SIMD-full thresholds. Switch to the SIMD
# path only with enough `iters_full` axis steps to amortize address
# math and the SIMD-width remainder branch. Two thresholds because
# under-saturated grids (`num_rows < sm_count`) need a slightly
# higher bar to offset one SIMD load fetching fewer outputs.
comptime _SIMD_FULL_ITERS_UNDER_SAT = 2
comptime _SIMD_FULL_ITERS_SAT = 2

# Split-K inner-axis tier gating. Block-per-row already saturates
# bandwidth on wide-SIMD dtypes (bf16 / fp16, simd_width = 8) even
# when `num_rows < sm_count`; split-K only pays off (over its
# cross-block atomic-finish overhead) on narrow-SIMD dtypes
# (fp32 / int32, simd_width <= 4), where one-block-per-row leaves
# the row bandwidth-bound. Gated on:
#   - `num_rows < sm_count` (block-per-row would leave SMs idle),
#   - `row_size >= _SPLITK_MIN_ROW` (amortizes the atomic finish),
#   - `effective_simd <= _SPLITK_MAX_SIMD` (narrow-SIMD dtype).
comptime _SPLITK_MIN_ROW = 32768

# Per-element-output split-K (normalize-shaped bodies) engages only when the row's
# byte volume clears this floor. Below it the occupancy-starved single-block
# kernel is already overhead-bound, so the tier's K extra launches cost more
# than the occupancy they recover. Byte-based, not element-based like
# `_SPLITK_MIN_ROW`, so an fp32 row engages at half the element count of a
# bf16 one: the fp32 163840-vocab Kimi sampling row (655 KB) clears it.
#
# Re-measured on B200 (fixed `num_rows=8`, forced split-K on/off A/B on the
# `softmax` Row body): the single-block path stays ahead through ~176 KB
# (0.92x), crosses over at ~192 KB (1.07x), and is a clean win by 224 KB
# (1.37x) and beyond (2-2.5x by 640 KB-1 MB) -- the previous 512 KB floor left
# the bf16 163840-vocab shape (320 KB) on the single-block path even though
# splitting already wins there. 224 KB keeps a margin above the measured
# crossover for run-to-run noise while still clearing that shape.
#
comptime _SPLITK_MIN_ROW_BYTES = 224 * 1024
comptime _SPLITK_MAX_SIMD = 4

# Threads per block for the PER-ELEMENT-OUTPUT split-K tier. Matches the
# legacy `twophase_reduce_kernel`'s `unsaturated_block_size`. Smaller than
# `BLOCK_SIZE = 256` so more blocks fit per SM under the scattered
# global-load atomic-finish pattern.
comptime _SPLITK_BLOCK_SIZE = 128

# The per-row split-K tuning below — finish ordering, block size, blocks per
# row, wide stripe — was measured on MI355X only. Other targets keep the
# pre-tuning path until swept there; the sibling per-element tier's own B200
# A/B (see `_SPLITK_TOTAL_BLOCKS_TARGET`) changes sign past ~20 rows, so the
# direction does not transfer for free.
comptime _SPLITK_ROW_TIER_TUNED = has_amd_gpu_accelerator()

# Threads per block for the PER-ROW split-K tier — each block pays a fixed
# cross-block-finish toll, so it wants few wide blocks. Kept separate from
# `_SPLITK_BLOCK_SIZE`, whose 128 was tuned against its own B200 A/B.
comptime _SPLITK_ROW_BLOCK_SIZE = (
    256 if _SPLITK_ROW_TIER_TUNED else _SPLITK_BLOCK_SIZE
)

# Blocks per row once the rows alone saturate the device: past a handful, extra
# blocks buy only more cross-block finishes (at 16-128 rows, 128/row -> 8 is
# 2.4-4.1x). Rows still get more when too few of them to fill the device.
comptime _SPLITK_TARGET_BLOCKS_PER_ROW = 8

# Damping on the device-fill term above: 1-row and 16-row shapes want opposite
# block counts. 2 costs ~6% at 16 rows, against the 20% undamped costs at 1 row.
comptime _SPLITK_FILL_DIVISOR = 2

# Total-block target for the PER-ELEMENT-OUTPUT split-K tier
# (normalize-shaped bodies' `num_splits`, below), independent of the per-row
# `_SPLITK_BLOCK_SIZE` cap. Without this, `num_splits = min(_SPLITK_BLOCK_SIZE,
# (sm_count*_SM_OVERPROVISION)//num_rows)` stays pinned at `_SPLITK_BLOCK_SIZE`
# for any `num_rows` up to ~`sm_count*_SM_OVERPROVISION/_SPLITK_BLOCK_SIZE`
# (~37 on B200), so `num_rows * num_splits` (the actual launched block count)
# grows LINEARLY with `num_rows` instead of staying near the total that
# empirically works. Fresh B200 A/B (legacy / split-off / split-on, all three
# measured back-to-back in one process, `num_rows` swept 8-64 at a fixed
# 163840-col bf16 row): capping `num_splits` so `num_rows * num_splits`
# tracks the validated `num_rows=8` total (1024) instead of growing with
# `num_rows` turns `num_rows=16` from a slight loss (branch on/off 0.98x,
# uncapped) into a clean win (1.13x, capped) -- but does NOT fully recover
# `num_rows` past ~20 (branch on/off crosses back under 1.0 there: 20 ->
# 1.00x, 24 -> 0.94x, 32 -> 0.79x, 64 -> 0.50x, all still capped). The
# residual loss past `num_rows≈20` is a distinct tension in the K+1-launch
# design itself (capping total blocks means fewer, FATTER per-block chunks,
# so each of the 3 phases re-reads more of the row per block; the launch/
# combine overhead this cap avoids is traded for more redundant HBM reads
# per phase) -- not fixable by retuning this constant alone. See
# `_SPLITK_MAX_ROWS_FOR_SPLIT` below for how the remaining range is handled.
comptime _SPLITK_TOTAL_BLOCKS_TARGET = 1024

# Upper bound on `num_rows` for engaging the per-element-output split-K tier
# at all -- ADDITIONAL to the byte floor and `num_rows < sm_count` occupancy
# gate below. Necessary because `_SPLITK_TOTAL_BLOCKS_TARGET` alone doesn't
# recover the whole `num_rows < sm_count` range (see its comment): past
# `num_rows≈20` on B200, split-K is a genuine loss vs the single-block path
# no matter how `num_splits` is capped, for reasons inherent to the K+1-
# launch schedule, not the sizing constant. Gating on `num_rows` directly
# guarantees no regression for excluded row counts BY CONSTRUCTION -- they
# fall through to the unchanged single-block path, identical to what ran
# before any pointwise-split-K tuning touched this file. `16` (not `20`)
# leaves measured margin over the observed crossover (~20, branch on/off
# 1.00x) for run-to-run noise.
comptime _SPLITK_MAX_ROWS_FOR_SPLIT = 16

# Bytes reserved per partial in the split-K scratch buffer. Typed as
# `uint8`; `rowwise.reduce` casts to `UnsafePointer[State]` at the
# body's call site (State known there, opaque to `launch`). Sized to
# cover every common monoid (`ReduceSum` / `ReduceMax` ≤ 8 bytes,
# `OnlineLogSumExp` ≤ 16, Welford ≤ 24).
comptime _SPLITK_STATE_BYTES = 128


# ===-----------------------------------------------------------------------===#
# GPU-side `Reducer` impls — used by `reduce`'s `pjoin` step.
# ===-----------------------------------------------------------------------===#

comptime _WARP_TIER_CHUNK_CAP = get_defined_int["warp_cap", 8]()
"""Max SIMD chunks one warp grid-strides over on the warp tier. `1` is the
legacy single-pass behavior (warp tier only for `cols <= WARP_SIZE*simd`).
`> 1` lets the warp tier cover longer (divisible) rows — a cheap warp-shuffle
reduce with many warps/block, instead of one over-reducing block per row.

Default `8` from a B200 crossover sweep: one-warp-per-row beats
one-block-per-row up to ~8 chunks (e.g. 16384x2048 reduce_sum 0.68x -> 0.89x),
but past ~16 chunks the per-warp serial work — amplified by phases — loses
badly for multi-phase ops (softmax 4096x8192 1.84x -> 0.71x), so the cap keeps
longer rows on the block tier."""

comptime _WARP_SHUFFLE_MAX_WORDS = 4
"""ReduceOp states up to this many `uint32` words use the register-only
warp-shuffle butterfly inside `generic`; larger states fall back to the shmem
tree, where the per-word shuffle + register pressure would dominate. `ReduceOp`
is `TrivialRegisterPassable`, so any state is byte-shuffle-safe; this is purely
a size cutoff (Welford = 3 words → shuffle; ArgMax/ArgMin → tree)."""


@always_inline
def _state_fits_warp_shuffle[S: ReduceOp]() -> Bool:
    """Returns `True` when `S` is small enough that the register-only
    warp-shuffle butterfly beats the shmem tree in `generic`.

    `_warp_shuffle_combine` exchanges whole `uint32` words via a raw
    pointer bitcast, so it is only byte-safe when `size_of[S]()` is an
    *exact* multiple of 4 -- a state smaller than one word (e.g. a
    width-1 `ReduceProduct[float16 | bfloat16]`, 2 bytes, or a bool
    `ReduceMax`/`ReduceMin`, 1 byte) would have its last word's high
    bytes read from -- and, on the write-back half of the exchange,
    stomped onto -- whatever memory follows the state on the stack.
    Round *down* here (not `ceildiv`) so a non-multiple size falls
    through to the shmem tree, which uses a typed `S`-element array and
    is size-agnostic.

    Packing such a state into a padded tail word to keep it on the
    register path was measured on B200 and is NOT a win: it buys
    `reduce_product` fp16/bf16 8-14% but costs `bool` `reduce_max`/
    `reduce_min` 5-24% (both at narrow rows), for a geomean of 1.00.
    """
    return (
        size_of[S]() % 4 == 0 and size_of[S]() // 4 <= _WARP_SHUFFLE_MAX_WORDS
    )


@always_inline
def _warp_shuffle_combine[S: ReduceOp](mut state: S):
    """Within-warp butterfly all-reduce over `state.join`, exchanging
    the multi-field state between lanes as `uint32` words. Register-only
    (`log2(WARP_SIZE)` levels, no shmem); on return **every** lane holds
    the warp-wide combined state.

    Uses `shuffle_xor` (not `shuffle_down`) for a true all-reduce: at
    each level lane `L` and lane `L ^ stride` swap and both `join`, so
    every lane ends with the full result. A `shuffle_down` tree would
    leave only lane 0 correct, breaking warp-tier callers that read
    each lane's own state. Requires `join` to be commutative and
    associative (true for `Welford` — Chan's combine is symmetric).

    Used by `BlockReducer` / `WarpReducer` as the within-warp portion.
    Bytes are exchanged word-by-word (one `shuffle_xor` per word per
    level), so cost grows with state size — hence the ~4-word cutoff.

    Only ever called with a state whose size is an exact multiple of 4
    (`_state_fits_warp_shuffle` gates that), so the word exchange never
    reads or writes past `state`.
    """
    comptime n_words = size_of[S]() // 4

    comptime for step in reversed(range(log2_floor(WARP_SIZE))):
        var stride = UInt32(1 << step)
        var partner = S()
        var state_ptr = UnsafePointer(to=state).bitcast[UInt32]()
        var partner_ptr = UnsafePointer(to=partner).bitcast[UInt32]()

        comptime for word in range(n_words):
            partner_ptr[word] = warp.shuffle_xor(state_ptr[word], stride)
        state.join(partner)


struct BlockReducer[BLOCK_SIZE: Int](Reducer, TrivialRegisterPassable):
    """Reduces a scalar across `BLOCK_SIZE` threads in a block.
    Broadcasts the result to every thread.

    Parameters:
        BLOCK_SIZE: Number of threads in the launching block.
    """

    @always_inline
    def __init__(out self):
        """Default-initializes a `BlockReducer`."""
        pass

    @always_inline
    def sum[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the sum of `val` across the block.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-thread value.

        Returns:
            The block-wide sum, broadcast to every thread.
        """
        return block.sum[block_size=Self.BLOCK_SIZE, broadcast=True](val)

    @always_inline
    def max[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the maximum of `val` across the block.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-thread value.

        Returns:
            The block-wide maximum, broadcast to every thread.
        """
        return block.max[block_size=Self.BLOCK_SIZE, broadcast=True](val)

    @always_inline
    def min[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the minimum of `val` across the block.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-thread value.

        Returns:
            The block-wide minimum, broadcast to every thread.
        """
        return block.min[block_size=Self.BLOCK_SIZE, broadcast=True](val)

    @always_inline
    def generic[S: ReduceOp](self, mut state: S):
        """Block-wide all-reduce over `state.join`; on return every
        thread holds the combined value.

        Dispatches on state size (a perf choice — either path is
        byte-safe since `ReduceOp` is `TrivialRegisterPassable`):

        - **Small** (`<= _WARP_SHUFFLE_MAX_WORDS` uint32 words): a
          register-only within-warp warp-shuffle butterfly, then a
          sparse one-entry-per-warp shmem store + a second shuffle
          over warp 0 (mirrors legacy `welford_block_all_reduce`) —
          `BLOCK_SIZE/WARP_SIZE` stores + one barrier, ~30x less shmem
          traffic than the tree.
        - **Large**: a `log2(BLOCK_SIZE)`-step block-wide shmem tree.
          Past a few words the per-word shuffle + register pressure
          makes the tree comparable or better.

        Parameters:
            S: The monoid type being combined.

        Args:
            state: The per-thread state; on return, holds the
                block-wide combined value on every thread.
        """
        comptime if _state_fits_warp_shuffle[S]():
            # Within-warp shuffle butterfly, then sparse cross-warp shmem.
            _warp_shuffle_combine(state)

            comptime n_warps = Self.BLOCK_SIZE // WARP_SIZE
            var shmem = stack_allocation[n_warps, S, address_space=.SHARED]()
            var w_idx = Int(warp_id())
            var l_idx = Int(lane_id())

            # Lane 0 of each warp publishes its warp-combined state.
            if l_idx == 0:
                shmem[w_idx] = state
            barrier()

            # Warp 0 combines the per-warp results (lanes beyond
            # `n_warps` get identity); lane 0 broadcasts via shmem[0].
            if w_idx == 0:
                if l_idx < n_warps:
                    state = shmem[l_idx]
                else:
                    state = S()
                _warp_shuffle_combine(state)
                if l_idx == 0:
                    shmem[0] = state
            barrier()
            state = shmem[0]
            # The broadcast read above is the last shmem access of this
            # combine, with no barrier after it — so when a kernel calls
            # `generic` again (the block tier grid-strides several rows per
            # block), a fast warp's next-call publish can overwrite `shmem`
            # while a lagging warp still reads it. NVIDIA/AMD warps never
            # drift that far in practice; Metal loses this race readily
            # (measured: layer_norm block tier, run-to-run divergent bytes
            # on M5, frequency scaling with rows per block). Close the
            # reuse window on Apple only, keeping other targets' codegen
            # byte-identical.
            comptime if is_apple_gpu():
                barrier()
        else:
            # Block-wide shmem tree: log2(BLOCK_SIZE) halving steps.
            var shmem = stack_allocation[
                Self.BLOCK_SIZE, S, address_space=.SHARED
            ]()
            var tid = Int(thread_idx.x)
            shmem[tid] = state
            barrier()

            comptime n_steps = log2_floor(Self.BLOCK_SIZE)

            comptime for inv_step in range(n_steps):
                comptime step = n_steps - 1 - inv_step
                comptime stride = 1 << step
                if tid < stride:
                    var partner = shmem[tid + stride]
                    var local = shmem[tid]
                    local.join(partner)
                    shmem[tid] = local
                barrier()

            state = shmem[0]
            # Same cross-call shmem-reuse race as the small-state path
            # above; same Apple-only close.
            comptime if is_apple_gpu():
                barrier()


struct WarpReducer[WARPS_PER_BLOCK: Int = 1](Reducer, TrivialRegisterPassable):
    """Reduces a scalar across all lanes in a warp. Broadcasts the
    result to every lane.

    Only `generic` uses `WARPS_PER_BLOCK`; the scalar paths use
    hardware warp ops and are warp-count-oblivious. Default `1` keeps
    the type usable in single-warp-block kernels.

    Parameters:
        WARPS_PER_BLOCK: Number of warps in the launching block.
    """

    @always_inline
    def __init__(out self):
        """Default-initializes a `WarpReducer`."""
        pass

    @always_inline
    def sum[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the sum of `val` across the warp.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-lane value.

        Returns:
            The warp-wide sum, broadcast to every lane.
        """
        return warp.sum(val)

    @always_inline
    def max[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the maximum of `val` across the warp.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-lane value.

        Returns:
            The warp-wide maximum, broadcast to every lane.
        """
        return warp.max(val)

    @always_inline
    def min[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns the minimum of `val` across the warp.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The per-lane value.

        Returns:
            The warp-wide minimum, broadcast to every lane.
        """
        return warp.min(val)

    @always_inline
    def generic[S: ReduceOp](self, mut state: S):
        """Warp-wide all-reduce over `state.join`; on return every lane
        holds the combined value.

        Dispatches on state size (a perf choice — either path is
        byte-safe): small states (`<= _WARP_SHUFFLE_MAX_WORDS` uint32
        words) use a register-only `shuffle_xor` butterfly (no shmem);
        larger states use a per-warp shmem tree over the warp's
        `[warp_id*WARP_SIZE, ...)` slice, so concurrent warps don't
        clobber each other.

        The tree's `syncwarp()`s are load-bearing, not defensive: each
        step reads the slot its partner lane wrote in the previous one,
        and lanes diverge on `lid < stride`, so without them the
        compiler is free to reorder the shared accesses (and Volta+
        lanes need not reconverge). `barrier()` -- what `BlockReducer`'s
        tree uses -- is not an option here: a warp whose `row_idx`
        exceeds `num_rows` skips the body entirely, so a block-wide
        barrier inside it would deadlock.

        Parameters:
            S: The monoid type being combined.

        Args:
            state: The per-lane state; on return, holds the warp-wide
                combined value on every lane.
        """
        comptime if _state_fits_warp_shuffle[S]():
            _warp_shuffle_combine(state)
        else:
            var shmem = stack_allocation[
                Self.WARPS_PER_BLOCK * WARP_SIZE,
                S,
                address_space=.SHARED,
            ]()
            var warp_base = Int(warp_id()) * WARP_SIZE
            var lid = Int(lane_id())
            shmem[warp_base + lid] = state
            syncwarp()

            comptime n_steps = log2_floor(WARP_SIZE)

            comptime for inv_step in range(n_steps):
                comptime step = n_steps - 1 - inv_step
                comptime stride = 1 << step
                if lid < stride:
                    var partner = shmem[warp_base + lid + stride]
                    var local = shmem[warp_base + lid]
                    local.join(partner)
                    shmem[warp_base + lid] = local
                syncwarp()

            state = shmem[warp_base]


# ===-----------------------------------------------------------------------===#
# Body-facing primitives: `reduce`, `once`, `simd`.
# ===-----------------------------------------------------------------------===#


@always_inline
def reduce[
    params: ContextParams,
    rank: Int,
    //,
    TileFn: ImplicitlyCopyable & (def[ws: Int](RowCoord[rank]) -> None),
](
    row_coords: RowCoord[rank],
    axis_size: Int,
    mut ctx: Context[params],
    tile_fn: TileFn,
):
    """Drives the tier-aware iteration over the reduce axis, with no
    monoid state — pure per-tile iteration for map/emit terminals (see
    the state-carrying overload below for reduce phases).

    `tile_fn[ws]` is invoked per tile with the scaffolder's SIMD
    width and the row's coords. It's a value closure (its copy-captured
    state — input/output closures — rides the value).

    `reduce` runs **iteration only**; the body calls `rowwise.pjoin`
    separately for each state needing a cross-thread combine.
    Two-pass algorithms (softmax pass 2, layernorm rewrite) use
    `reduce` with no `pjoin` — pure map-style iteration.

    Tier semantics, picked from `ctx`:

    - **Warp tier** (cooperative, `emit_tile_width == 1`): lane `i` runs
      one scalar `tile_fn[1]` on element `i`
      (`axis_size <= WARP_SIZE`).
    - **Block tier** (cooperative, `emit_tile_width == 1`): threads
      grid-stride by `BLOCK_SIZE * simd_width` along the axis; full
      SIMD `tile_fn[simd_width]` on aligned chunks, scalar
      `tile_fn[1]` on the tail.
    - **Tiled tier** (`emit_tile_width > 1`): the thread owns
      `emit_tile_width` adjacent rows. Each axis step issues one
      `tile_fn[emit_tile_width]`; the body dispatches each lane to its
      own state. No cross-thread combine — `rowwise.pjoin` no-ops.
    - **Serial tier** (one thread, one row): scalar `tile_fn[1]` per
      axis step.
    - **Split-K tier** (`num_rows < sm_count`, large row): each
      block handles a slice of one row; `rowwise.pjoin` coordinates
      across blocks.

    Parameters:
        params: The comptime dispatch parameters.
        TileFn: The value-closure type of `tile_fn`.

    Args:
        row_coords: The current row's coords. `ctx.axis` is the
            reduce axis; other dims are pinned.
        axis_size: Length of the reduce axis.
        ctx: The dispatch bundle.
        tile_fn: Per-tile callback; closes over input/output closures.
    """
    var coords = row_coords

    comptime if ctx._tier == ReduceTier.Serial:
        for k in range(axis_size):
            coords.write_axis[ctx.axis](k)
            tile_fn[1](coords)
    elif ctx._tier == ReduceTier.Warp:
        # One warp covers the row, grid-striding by `WARP_SIZE * sw` to
        # span up to `_WARP_TIER_CHUNK_CAP` SIMD chunks; each lane handles
        # `sw` consecutive elements per chunk at `chunk_base + lid*sw`.
        # For `axis_size <= WARP_SIZE * sw` the loop runs once (legacy
        # single-pass). The tier picker only routes divisible rows here
        # past one pass, so the `lane_count == sw` fast path holds except
        # for a possible scalar tail on a non-divisible single-pass row.
        comptime sw = ctx.simd_width
        comptime warp_span = WARP_SIZE * sw
        var lid = lane_id()
        for chunk_base in range(0, axis_size, warp_span):
            var lane_base = chunk_base + Int(lid) * sw
            if lane_base < axis_size:
                var lane_count = min(axis_size - lane_base, sw)
                if lane_count == sw:
                    coords.write_axis[ctx.axis](lane_base)
                    tile_fn[sw](coords)
                else:
                    for j in range(lane_count):
                        coords.write_axis[ctx.axis](lane_base + j)
                        tile_fn[1](coords)
    elif ctx.emit_tile_width > 1:
        # Tiled (SIMD-on-outputs). One thread owns `emit_tile_width`
        # adjacent rows; per axis step, one SIMD load on innermost
        # dim — the body's tile_fn dispatches each lane.
        comptime W = ctx.emit_tile_width
        for k in range(axis_size):
            coords.write_axis[ctx.axis](k)
            tile_fn[W](coords)
    elif ctx._tier == ReduceTier.Splitk:
        # Split-K iteration: threads stripe `sw`-wide tiles across
        # `blocks_per_row * BLOCK_SIZE` lanes, so adjacent threads load
        # adjacent tiles (coalesced per-warp transactions).
        comptime sw = ctx.simd_width
        var blocks_per_row = Int(ctx._blocks_per_row)
        var block_in_row = Int(ctx._block_in_row)
        var tid = thread_idx.x
        var row_tid = block_in_row * ctx.BLOCK_SIZE + Int(tid)
        var row_total_threads = blocks_per_row * ctx.BLOCK_SIZE

        # `sw > 1` is dispatched only when `sw` divides the row, so the stripe
        # needs no ragged tail: every in-range `elem_idx` has a full tile behind
        # it.
        for elem_idx in range(row_tid * sw, axis_size, row_total_threads * sw):
            coords.write_axis[ctx.axis](elem_idx)
            tile_fn[sw](coords)
    else:
        # Block tier (cooperative, simd along axis).
        comptime BLOCK_SPAN = ctx.BLOCK_SIZE * ctx.simd_width
        var tid = thread_idx.x
        for tile_base in range(0, axis_size, BLOCK_SPAN):
            var lane_base = tile_base + tid * ctx.simd_width
            if lane_base < axis_size:
                var lane_count = min(axis_size - lane_base, ctx.simd_width)
                if lane_count == ctx.simd_width:
                    coords.write_axis[ctx.axis](lane_base)
                    tile_fn[ctx.simd_width](coords)
                else:
                    for j in range(lane_count):
                        coords.write_axis[ctx.axis](lane_base + j)
                        tile_fn[1](coords)


@always_inline
def reduce[
    State: ReduceOp,
    params: ContextParams,
    rank: Int,
    //,
    TileFn: ImplicitlyCopyable
    & (def[ws: Int](mut State, RowCoord[rank]) -> None),
](
    row_coords: RowCoord[rank],
    axis_size: Int,
    mut ctx: Context[params],
    mut state: State,
    tile_fn: TileFn,
):
    """State-carrying overload of `reduce`: same tier-aware iteration as
    above, but `tile_fn` takes the caller's monoid `state` as a `mut`
    argument (rather than capturing it — a captured accumulator can't be
    mutated through a value closure) and folds each tile into it.

    Parameters:
        State: The monoid type being accumulated.
        params: The comptime dispatch parameters.
        TileFn: The value-closure type of `tile_fn`.

    Args:
        row_coords: The current row's coords. `ctx.axis` is the
            reduce axis; other dims are pinned.
        axis_size: Length of the reduce axis.
        ctx: The dispatch bundle.
        state: The caller's monoid accumulator, threaded through by
            `mut` reference and mutated in place by `tile_fn`.
        tile_fn: Per-tile callback; closes over input closures and
            folds each tile into `state`.
    """
    var coords = row_coords

    comptime if ctx._tier == ReduceTier.Serial:
        for k in range(axis_size):
            coords.write_axis[ctx.axis](k)
            tile_fn[1](state, coords)
    elif ctx._tier == ReduceTier.Warp:
        comptime sw = ctx.simd_width
        comptime warp_span = WARP_SIZE * sw
        var lid = lane_id()
        for chunk_base in range(0, axis_size, warp_span):
            var lane_base = chunk_base + Int(lid) * sw
            if lane_base < axis_size:
                var lane_count = min(axis_size - lane_base, sw)
                if lane_count == sw:
                    coords.write_axis[ctx.axis](lane_base)
                    tile_fn[sw](state, coords)
                else:
                    for j in range(lane_count):
                        coords.write_axis[ctx.axis](lane_base + j)
                        tile_fn[1](state, coords)
    elif ctx.emit_tile_width > 1:
        comptime W = ctx.emit_tile_width
        for k in range(axis_size):
            coords.write_axis[ctx.axis](k)
            tile_fn[W](state, coords)
    elif ctx._tier == ReduceTier.Splitk:
        comptime sw = ctx.simd_width
        var blocks_per_row = Int(ctx._blocks_per_row)
        var block_in_row = Int(ctx._block_in_row)
        var tid = thread_idx.x
        var row_tid = block_in_row * ctx.BLOCK_SIZE + Int(tid)
        var row_total_threads = blocks_per_row * ctx.BLOCK_SIZE

        for elem_idx in range(row_tid * sw, axis_size, row_total_threads * sw):
            coords.write_axis[ctx.axis](elem_idx)
            tile_fn[sw](state, coords)
    else:
        comptime BLOCK_SPAN = ctx.BLOCK_SIZE * ctx.simd_width
        var tid = thread_idx.x
        for tile_base in range(0, axis_size, BLOCK_SPAN):
            var lane_base = tile_base + tid * ctx.simd_width
            if lane_base < axis_size:
                var lane_count = min(axis_size - lane_base, ctx.simd_width)
                if lane_count == ctx.simd_width:
                    coords.write_axis[ctx.axis](lane_base)
                    tile_fn[ctx.simd_width](state, coords)
                else:
                    for j in range(lane_count):
                        coords.write_axis[ctx.axis](lane_base + j)
                        tile_fn[1](state, coords)


@always_inline
def pjoin[
    State: ReduceOp,
    params: ContextParams,
    //,
](mut state: State, mut ctx: Context[params]):
    """Cross-thread join for a single monoid state, tier-appropriately.

    Called by the body **after** `rowwise.reduce`, once per state. The
    cooperative tiers (warp, block) emit a hardware-backed warp/block
    reduce; the tiled and serial tiers no-op (one thread per output).
    Split-K does a within-block reduce then a cross-block
    partials-buffer join, setting `ctx._is_last_block` so
    `rowwise.once` gates emission.

    This is a **separate** call rather than a parameter to
    `rowwise.reduce` because a `mut *states: *States` variadic copies
    arguments into the pack at the call site, so mutations inside
    `reduce` wouldn't propagate back. Single `mut state: State` passes
    by mut reference cleanly.

    Parameters:
        State: The monoid type being joined.
        params: The comptime dispatch parameters.

    Args:
        state: The body's monoid accumulator. On return, holds the
            cross-thread-joined value on every thread (cooperative
            tiers) or its own value (tiled/serial tiers).
        ctx: The dispatch bundle. Mutated for the split-K case so
            `rowwise.once` knows which block is the canonical writer.
    """
    comptime if ctx._tier == ReduceTier.Serial:
        # No cross-thread participants; state is already final.
        pass
    elif ctx._tier == ReduceTier.Warp:
        comptime warps_per_block = ctx.BLOCK_SIZE // WARP_SIZE
        # Collapse the `W` lane-wise partials to a width-1 scalar, warp-combine
        # that small state (cheap register shuffle vs a W-wide shmem tree), then
        # broadcast back across the lanes. Branchless: `reduce` is identity at
        # `W == 1` (empty fold) and the broadcast ctor is a no-op there.
        var s = state.reduce()
        s.join_parallel(WarpReducer[warps_per_block]())
        state = State(s)
    elif ctx.emit_tile_width > 1:
        # Tiled tier: the `W` lanes are independent outputs — no
        # collapse, no cross-thread join. Per-lane results stand.
        pass
    elif ctx._tier == ReduceTier.Splitk:
        # Within-block join on the collapsed width-1 scalar, broadcast back.
        var s = state.reduce()
        s.join_parallel(BlockReducer[ctx.BLOCK_SIZE]())
        state = State(s)
        # Cross-block finish: thread 0 publishes the partial and
        # atomically increments the per-row counter; the atomic's
        # release semantics make the partial write visible to the
        # block whose increment last reaches `blocks_per_row`.
        # Release-only, not seq_cst: only the last arriver reads the partials,
        # so the acquire rides a fence in that branch instead of costing a
        # `buffer_inv` per CTA. Off the tuned path the counter stays seq_cst,
        # which subsumes both halves — except on Apple GPU, which rejects
        # `seq_cst` at lowering, so it keeps the relaxed ordering the stdlib
        # defaults to there. The tier never launches on Metal (gated in
        # `launch`), but the phased split-K params still instantiate this.
        comptime _finish_ordering = (
            Ordering.RELEASE if _SPLITK_ROW_TIER_TUNED else (
                Ordering.RELAXED if is_apple_gpu() else Ordering.SEQUENTIAL
            )
        )
        # The partials buffer uses fixed `_SPLITK_STATE_BYTES` slots
        # (independent of State's size), so address each slot via byte
        # arithmetic + bitcast — `UnsafePointer[State]` indexing would
        # stride by `size_of[State]()` and walk off the slot bounds.
        #
        # The slot holds `State.Single`, not the `W`-lane broadcast: the block's
        # result is a scalar, which holds the second `BlockReducer` tree's shmem
        # at width-1 whatever `simd_width` the tier runs at.
        comptime assert (
            size_of[State.Single]() <= _SPLITK_STATE_BYTES
        ), "split-K partial slot too small for this monoid's width-1 state"
        var blocks_per_row_ = Int(ctx._blocks_per_row)
        var row_idx_ = Int(ctx._row_idx)
        var block_in_row_ = Int(ctx._block_in_row)
        var row_base_bytes = (
            ctx._partials_base
            + row_idx_ * blocks_per_row_ * _SPLITK_STATE_BYTES
        )
        var counter_ptr = ctx._counters_base + row_idx_
        var last_shmem = stack_allocation[1, Bool, address_space=.SHARED]()
        var tid_ = thread_idx.x
        if tid_ == 0:
            var slot_ptr = (
                row_base_bytes + block_in_row_ * _SPLITK_STATE_BYTES
            ).bitcast[State.Single]()
            slot_ptr[0] = s
            var prev = Atomic[Int32].fetch_add[ordering=_finish_ordering](
                counter_ptr, Int32(1)
            )
            last_shmem[0] = Int(prev) + 1 == blocks_per_row_
        barrier()
        ctx._is_last_block = last_shmem[0]

        if ctx._is_last_block:
            # Acquire half of the `fetch_add`'s RELEASE — must cover every
            # thread that loads a partial below, so it cannot be narrowed to
            # thread 0. On gfx9 `barrier()` invalidates no cache, so it is no
            # substitute. Redundant when the counter is seq_cst.
            comptime if _SPLITK_ROW_TIER_TUNED:
                fence[Ordering.ACQUIRE]()
            # Cross-block join: the first `blocks_per_row` threads each
            # load one partial, the rest pad with the monoid identity,
            # then one block-wide combine folds them.
            #
            # Goes through `join_parallel`, not `BlockReducer.generic`
            # directly: `generic` is only the *default* implementation of
            # the cross-thread step, and a monoid that overrides it does
            # so because a field-wise `join` alone does not finish the
            # combine. `ArgMax`/`ArgMin` are the case in point -- their
            # override publishes the winning index into `acc_indices[0]`,
            # which is the field the body's emit reads, after resetting
            # the SIMD acc that `join` compares. Calling `generic` here
            # skipped that publish and left every partial's acc tied at
            # the identity, so the lowest per-block index won the
            # tie-break instead of the row's argmax.
            var local = State.Single()
            if Int(tid_) < blocks_per_row_:
                var slot_ptr = (
                    row_base_bytes + Int(tid_) * _SPLITK_STATE_BYTES
                ).bitcast[State.Single]()
                local = slot_ptr[0]
            local.join_parallel(BlockReducer[ctx.BLOCK_SIZE]())
            state = State(local)
    else:
        # Block tier: collapse to width-1, block-join the small scalar (register
        # shuffle instead of a W-wide shmem tree), broadcast back.
        var s = state.reduce()
        s.join_parallel(BlockReducer[ctx.BLOCK_SIZE]())
        state = State(s)


@always_inline
def once[
    Emit: ImplicitlyCopyable & RegisterPassable & (def() -> None),
    params: ContextParams,
    //,
](emit: Emit, ctx: Context[params]):
    """Runs `emit` exactly once per (logical) output row.

    - Warp tier: only lane 0 runs `emit`.
    - Block tier: only thread 0 runs `emit`.
    - Tiled / serial tier: each thread owns its own output(s), so
      every thread that runs the body also runs `emit`.
    - Split-K tier: only thread 0 of the last-arriving block (per
      `ctx._is_last_block`, set by `rowwise.reduce`) runs `emit`.

    `emit` is a value closure (its copy-captured state rides the value);
    this is an in-kernel thread predicate, not a launch boundary.

    Parameters:
        Emit: The value-closure type of `emit`.
        params: The comptime dispatch parameters.

    Args:
        emit: Callback to run on the canonical writer.
        ctx: The dispatch bundle passed to the body. The split-K
            branch reads `ctx._is_last_block` to gate emission.

    The body never observes `lane_id()` / `thread_idx.x`.
    """
    comptime if ctx._tier == ReduceTier.Serial:
        emit()
    elif ctx._tier == ReduceTier.Warp:
        if lane_id() == 0:
            emit()
    elif ctx.emit_tile_width > 1:
        emit()
    elif ctx._tier == ReduceTier.Splitk:
        if ctx._is_last_block and thread_idx.x == 0:
            emit()
    else:
        if thread_idx.x == 0:
            emit()


# ===-----------------------------------------------------------------------===#
# Per-element-output split-K: partial store + cross-block combine.
#
# The phase-aware K+1-launch split-K (see `_PointwiseSplitkKernel` and the
# `Row`'s pointwise-split-K branch) reuses the single-kernel split-K's
# chunk-striding and block reduce, but the cross-block join happens across
# LAUNCHES — the launch boundary is the sync — so there is no atomic counter and
# EVERY block combines (not just the last). The scratch is
# `[num_rows x num_reduces x num_splits x _SPLITK_STATE_BYTES]`; slot
# `[row, reduce_index, split]` is written once (partial pass of reduce
# `reduce_index`) and read in every later phase (combine).
# ===-----------------------------------------------------------------------===#


@always_inline
def _pointwise_splitk_slot_base[
    params: ContextParams, //
](ctx: Context[params], reduce_index: Int) -> Int:
    # Byte offset of `scratch[row, reduce_index, 0]`. `K` (the number of
    # dependent reduce phases, i.e. `num_phases - 1`, the final write phase
    # excluded) is comptime; the rest are per-block runtime values carried
    # in `ctx`.
    comptime K = params._num_phases - 1
    var num_splits = Int(ctx._blocks_per_row)
    return (
        (Int(ctx._row_idx) * K + reduce_index) * num_splits
    ) * _SPLITK_STATE_BYTES


@always_inline
def _pointwise_splitk_store_partial[
    State: ReduceOp, params: ContextParams, //
](mut state: State, ctx: Context[params], reduce_index: Int):
    """PARTIAL pass: block-reduces this block's chunk state and thread 0
    stores it to `scratch[row, reduce_index, block_in_row]`. No atomic /
    counter — the launch boundary makes the write visible to later phases.
    """
    var s = state.reduce()
    s.join_parallel(BlockReducer[params.BLOCK_SIZE]())
    state = State(s)
    if thread_idx.x == 0:
        var off = (
            _pointwise_splitk_slot_base(ctx, reduce_index)
            + Int(ctx._block_in_row) * _SPLITK_STATE_BYTES
        )
        (ctx._partials_base + off).bitcast[State]()[0] = state


@always_inline
def _pointwise_splitk_combine[
    params: ContextParams, //, State: ReduceOp
](ctx: Context[params], reduce_index: Int) -> State:
    """COMBINE pass: fold the row's `num_splits` partials for reduce
    `reduce_index` into the global state, broadcast to every thread. The
    first `num_splits` threads each load one partial (capped at
    `_SPLITK_BLOCK_SIZE == BLOCK_SIZE`), the rest pad with identity, then
    `BlockReducer.generic` reduces across the block. Every block runs this
    independently and gets the same global result.
    """
    var num_splits = Int(ctx._blocks_per_row)
    var base = _pointwise_splitk_slot_base(ctx, reduce_index)
    var tid = Int(thread_idx.x)
    var local = State()
    if tid < num_splits:
        var off = base + tid * _SPLITK_STATE_BYTES
        local = (ctx._partials_base + off).bitcast[State]()[0]
    BlockReducer[params.BLOCK_SIZE]().generic(local)
    return local


# ===-----------------------------------------------------------------------===#
# Kernels — one per tier; each binds `ctx` and calls the body.
# ===-----------------------------------------------------------------------===#


# The four tier kernels are STRUCTS that hold the per-row `body` as a
# value field (not a comptime `capturing` parameter). `enqueue_function`
# takes the struct instance, so the compiler serializes `body`'s
# copy-captured state (the fused `input_fn` / `output_fn` value closures,
# `axis_size`) into the kernel's parameter block — the only mechanism
# that survives the launch for value closures (a comptime-`body` closure
# copy-capturing value closures compiles but faults at run:
# CUDA_ERROR_ILLEGAL_ADDRESS). The `MAX_THREADS_PER_BLOCK` occupancy hint
# moves onto `__call__` (mirrors `linalg/fp8_quantization.mojo`).


struct _BlockKernel[rank: Int, params: ContextParams, Body: RowBody](
    ImplicitlyCopyable, RegisterPassable, def() -> None
):
    """Block-per-row kernel: one block per row, grid-strided over rows."""

    var body: Self.Body
    var shape: DynamicCoord[.int64, Self.rank]

    @always_inline
    def __init__(
        out self,
        body: Self.Body,
        shape: DynamicCoord[.int64, Self.rank],
    ):
        # `body` borrowed + copied (not consumed) so the launch's dispatch
        # closures can construct the kernel without moving out of a
        # copy-captured value.
        self.body = body
        self.shape = shape

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.params.BLOCK_SIZE)
        )
    )
    def __call__(self) capturing:
        var num_rows = _num_outputs_excluding_axis[Self.params.axis](self.shape)

        with PDL():
            var ctx = Context[Self.params].empty()
            for row_idx in range(block_idx.x, num_rows, grid_dim.x):
                var row_coords = _get_nd_indices_from_flat_index(
                    row_idx, self.shape, Self.params.axis
                )
                self.body[Self.params](row_coords, ctx)


struct _WarpKernel[rank: Int, params: ContextParams, Body: RowBody](
    ImplicitlyCopyable, RegisterPassable, def() -> None
):
    """Warp-per-row kernel: one warp per row, multiple rows per block,
    grid-strided over row groups."""

    var body: Self.Body
    var shape: DynamicCoord[.int64, Self.rank]

    @always_inline
    def __init__(
        out self,
        body: Self.Body,
        shape: DynamicCoord[.int64, Self.rank],
    ):
        self.body = body
        self.shape = shape

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.params.BLOCK_SIZE)
        )
    )
    def __call__(self) capturing:
        comptime warps_per_block = Self.params.BLOCK_SIZE // WARP_SIZE
        var num_rows = _num_outputs_excluding_axis[Self.params.axis](self.shape)

        with PDL():
            var ctx = Context[Self.params].empty()
            for block_base in range(
                block_idx.x * warps_per_block,
                num_rows,
                grid_dim.x * warps_per_block,
            ):
                var row_idx = block_base + warp_id()
                if row_idx < num_rows:
                    var row_coords = _get_nd_indices_from_flat_index(
                        row_idx, self.shape, Self.params.axis
                    )
                    self.body[Self.params](row_coords, ctx)


struct _TiledKernel[rank: Int, params: ContextParams, Body: RowBody](
    ImplicitlyCopyable, RegisterPassable, def() -> None
):
    """Tiled outer-axis kernel: each thread handles `params.emit_tile_width`
    consecutive flat-output positions of the innermost non-axis dim.
    Grid-strided over output tiles."""

    var body: Self.Body
    var shape: DynamicCoord[.int64, Self.rank]

    @always_inline
    def __init__(
        out self,
        body: Self.Body,
        shape: DynamicCoord[.int64, Self.rank],
    ):
        self.body = body
        self.shape = shape

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.params.BLOCK_SIZE)
        )
    )
    def __call__(self) capturing:
        # The body needs one of these to take the one-thread-per-output
        # branches; without either it compiles fine and reduces across
        # output rows.
        comptime assert (
            Self.params.emit_tile_width > 1
            or Self.params._tier == ReduceTier.Serial
        ), "tiled kernel needs a one-thread-per-output tier"

        var num_outputs = _num_outputs_excluding_axis[Self.params.axis](
            self.shape
        )

        # Index math is not data-dependent — compute it before the PDL
        # wait so it overlaps with the prior grid's tail.
        var stride = (
            Int(grid_dim.x)
            * Self.params.BLOCK_SIZE
            * Self.params.emit_tile_width
        )
        var base = (
            Int(block_idx.x) * Self.params.BLOCK_SIZE + Int(thread_idx.x)
        ) * Self.params.emit_tile_width
        with PDL():
            var ctx = Context[Self.params].empty()
            while base < num_outputs:
                var row_coords = _get_nd_indices_from_flat_index(
                    base, self.shape, Self.params.axis
                )
                self.body[Self.params](row_coords, ctx)
                base += stride


struct _SplitkKernel[rank: Int, params: ContextParams, Body: RowBody](
    ImplicitlyCopyable, RegisterPassable, def() -> None
):
    """Split-K inner-axis kernel: `blocks_per_row` blocks cooperate on
    one row. Per-block runtime values (partials/counters pointers +
    block position) ride in the `Context`'s var fields, so the body
    forwards a single `ctx` to `rowwise.reduce` / `rowwise.once`
    without seeing the split-K plumbing."""

    var body: Self.Body
    var shape: DynamicCoord[.int64, Self.rank]
    var partials: UnsafePointer[UInt8, MutUntrackedOrigin]
    var counters: UnsafePointer[Int32, MutUntrackedOrigin]
    var blocks_per_row: Int32

    @always_inline
    def __init__(
        out self,
        body: Self.Body,
        shape: DynamicCoord[.int64, Self.rank],
        partials: UnsafePointer[UInt8, MutUntrackedOrigin],
        counters: UnsafePointer[Int32, MutUntrackedOrigin],
        blocks_per_row: Int32,
    ):
        self.body = body
        self.shape = shape
        self.partials = partials
        self.counters = counters
        self.blocks_per_row = blocks_per_row

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.params.BLOCK_SIZE)
        )
    )
    def __call__(self) capturing:
        var num_rows = _num_outputs_excluding_axis[Self.params.axis](self.shape)

        var qr = udivmod(Int(block_idx.x), Int(self.blocks_per_row))
        var row_idx_ = qr[0]
        var block_in_row_ = qr[1]
        if row_idx_ >= num_rows:
            return

        # `row_coords` is not data-dependent — compute it before the PDL
        # wait so it overlaps with the prior grid's tail.
        var row_coords = _get_nd_indices_from_flat_index(
            row_idx_, self.shape, Self.params.axis
        )
        with PDL():
            var ctx = Context[Self.params](
                partials_base=self.partials,
                counters_base=self.counters,
                blocks_per_row=self.blocks_per_row,
                block_in_row=Int32(block_in_row_),
                row_idx=Int32(row_idx_),
                is_last_block=False,
            )
            self.body[Self.params](row_coords, ctx)


struct _PointwiseSplitkKernel[rank: Int, params: ContextParams, Body: RowBody](
    ImplicitlyCopyable, RegisterPassable, def() -> None
):
    """Per-element-output split-K kernel: `num_splits` blocks cooperate on
    one row for phase `phase` of the K+1-launch schedule. No PDL and no
    atomic counter — the launch boundary is the cross-block sync, so a
    later phase reads the prior phase's partials directly. The body runs
    unchanged; the `Row`'s pointwise-split-K branch reads `ctx._phase`."""

    var body: Self.Body
    var shape: DynamicCoord[.int64, Self.rank]
    var partials: UnsafePointer[UInt8, MutUntrackedOrigin]
    var num_splits: Int32
    var phase: Int32

    @always_inline
    def __init__(
        out self,
        body: Self.Body,
        shape: DynamicCoord[.int64, Self.rank],
        partials: UnsafePointer[UInt8, MutUntrackedOrigin],
        num_splits: Int32,
        phase: Int32,
    ):
        self.body = body
        self.shape = shape
        self.partials = partials
        self.num_splits = num_splits
        self.phase = phase

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.params.BLOCK_SIZE)
        )
    )
    def __call__(self) capturing:
        var num_rows = _num_outputs_excluding_axis[Self.params.axis](self.shape)

        var qr = udivmod(Int(block_idx.x), Int(self.num_splits))
        var row_idx_ = qr[0]
        var split_ = qr[1]
        if row_idx_ >= num_rows:
            return

        var row_coords = _get_nd_indices_from_flat_index(
            row_idx_, self.shape, Self.params.axis
        )
        var ctx = Context[Self.params](
            partials_base=self.partials,
            counters_base=UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=1
            ),
            blocks_per_row=self.num_splits,
            block_in_row=Int32(split_),
            row_idx=Int32(row_idx_),
            is_last_block=False,
            phase=self.phase,
        )
        self.body[Self.params](row_coords, ctx)


# ===-----------------------------------------------------------------------===#
# `launch` — the single top-level scaffolder.
# ===-----------------------------------------------------------------------===#


def launch[
    Body: RowBody,
    //,
    axis: Int,
    simd_width: Int,
    supports_splitk: Bool = True,
    computationally_expensive: Bool = False,
    BLOCK_SIZE: Int = 256,
    COOPERATIVE_BLOCK_SIZE: Int = 128,
    WARP_BLOCK_WARPS: Int = 4,
    cache_dtype: Optional[DType] = None,
    cache_count: Int = 0,
    num_phases: Int = 0,
    dtype_size: Int = 0,
](body: Body, shape: Coord, ctx: DeviceContext) raises:
    """Top-level scaffolder. Picks the tier from shape + axis,
    instantiates the matching kernel, and launches.

    Tier choice:

    - **Inner axis** (`axis == rank - 1`) — contiguous reduce axis.
      Warp-per-row when `row_size <= WARP_SIZE`, else block-per-row
      with a 2D SIMD-width heuristic over
      `(num_rows, row_size, sm_count)`.
    - **Non-inner axis**, SIMD-tileable, and either (the reduce axis
      is short and the output rows clear the per-SM floor) or the
      output count saturates the device — tiled kernel (one thread per
      row tile, coalesced SIMD load + store on the innermost non-axis
      dim).
    - **Non-inner axis**, short reduce axis clearing the same floor,
      not SIMD-tileable — scalar (`W = 1`) tiled kernel: same
      one-thread-per-output layout, without the vectorized load.
    - **Non-inner axis** otherwise — block-per-output cooperative
      (one block per output row, threads collaborate on the strided
      reduce axis with `simd_width = 1`).

    Parameters:
        axis: Axis being reduced.
        simd_width: The SIMD width the scaffolder uses for tile
            dispatch. Bodies compute this via
            `rowwise.pick_simd_width[...]` and pass it directly.
        supports_splitk: Whether the body can run in the split-K
            tier. Default `True` — single-state reductions opt in
            implicitly. Bodies that maintain multiple monoid states
            in one `rowwise.reduce` call (fused dual-reduce, fused
            mean+M2, ...) must set this `False` until the split-K
            partials buffer can hold N states per slot.
        computationally_expensive: Author hint that per-element work is heavy
            (`exp`, `tanh`, `sqrt`, a normalize step), so per-load
            address math is in the noise. When `True`, the inner-axis
            block tier picks SIMD-full as soon as alignment permits,
            bypassing the `iters_full` threshold that amortizes
            address math for cheap reductions. Softmax / log-softmax
            / layernorm set this; plain sum / max do not.
        BLOCK_SIZE: Threads per block for the inner-axis block tier.
        COOPERATIVE_BLOCK_SIZE: Threads per block for the non-inner
            cooperative tier.
        WARP_BLOCK_WARPS: Warps per block for the warp tier.
        cache_dtype: Optional persistent-cache dtype for the warp-tier
            multi-row scratch (`None` for no cache).
        cache_count: Number of cache entries per row.
        num_phases: Per-element-output split-K opt-in for
            normalize-shaped bodies. `<= 1` disables it (comptime-dead,
            byte-identical codegen). `N > 1` = the body has `N - 1`
            dependent `row.reduce` phases before its final per-element
            write; on an under-occupied inner-axis shape (`num_rows <
            sm_count`, `num_rows <= _SPLITK_MAX_ROWS_FOR_SPLIT`, and row
            bytes `>= _SPLITK_MIN_ROW_BYTES`) the row is split across many
            blocks via a phase-aware K+1-launch schedule (softmax /
            log-softmax: `N = 3`). Otherwise falls through to the warp /
            block tiers unchanged.
        dtype_size: Byte size of the body's primary dtype, used only by the
            per-element-output split-K byte gate above (`0` when the tier is
            off). Bodies pass `size_of[dtype]()`.

    Args:
        body: The per-row computation. Receives a `Context` and the
            row's `Coord`; uses `reduce` / `map` / `once` / `simd`
            to compose the algorithm.
        shape: Tensor shape.
        ctx: Device context.

    Raises:
        If the underlying GPU kernel launch fails.
    """
    comptime rank = shape.rank
    comptime sm_count = ctx.default_device_info.sm_count
    comptime effective_simd = simd_width

    # One wave per block: the smallest tiled block that leaves no lane idle.
    comptime tiled_block_size = ctx.default_device_info.warp_size

    comptime is_nvidia_device = ctx.default_device_info.api == "cuda"
    comptime tiled_min_rows_per_sm = (
        _TILED_MIN_ROWS_NVIDIA if is_nvidia_device else _TILED_MIN_ROWS_DEFAULT
    )
    comptime tiled_row_size_cutoff = (
        _ROW_SIZE_CUTOFF_NVIDIA if is_nvidia_device else _ROW_SIZE_CUTOFF_DEFAULT
    )

    var shape_il = coord_to_index_list(shape)
    var shape_dc = Coord(shape_il)
    var row_size = shape_il[axis]
    # `num_rows` is the product of every dim other than `axis`, so it stays
    # well-defined when the reduce axis itself is empty (`row_size == 0`) —
    # unlike `flattened_length() // row_size`, which is a `0 // 0` form in
    # that case. A reduce-shaped body still owns one output per row when
    # the axis is empty (the monoid identity), so only `num_rows == 0`
    # (no rows at all) means there is truly nothing to launch.
    var num_rows = _num_outputs_excluding_axis[axis](shape_dc)
    if num_rows == 0:
        return

    # ----- Non-inner axis ------------------------------------------------
    comptime if axis != rank - 1:
        var innermost = shape_il[rank - 1]
        var sat_threshold = sm_count * _THREAD_SAT_OUTPUTS_PER_SM
        # The row size tiled can afford scales with cooperative's wave count.
        var coop_waves = ceildiv(num_rows, sm_count * _SM_OVERPROVISION)
        var short_axis_ok = (
            row_size <= tiled_row_size_cutoff * coop_waves
            and num_rows >= sm_count * tiled_min_rows_per_sm
        )
        var occupancy_ok = short_axis_ok or num_rows >= sat_threshold
        # Tiled non-inner tier: the monoid is W-wide (lanes = the W
        # adjacent output columns the thread owns), `join_parallel` /
        # `reduce` no-op here, and the body's `emit` writes
        # `emit_tile_width` lanes.
        var use_full_tiled = (
            effective_simd > 1
            and innermost >= effective_simd
            and innermost % effective_simd == 0
            and num_rows % effective_simd == 0
            and occupancy_ok
        )

        if use_full_tiled:
            comptime W = effective_simd
            comptime tiled_params = ContextParams(
                axis=axis,
                emit_tile_width=W,
                BLOCK_SIZE=tiled_block_size,
                simd_width=W,
                target="gpu",
            )
            var num_blocks = min(
                ceildiv(num_rows, tiled_block_size * W),
                sm_count * _SM_OVERPROVISION,
            )
            ctx.enqueue_function(
                _TiledKernel[rank, tiled_params, Body](body, shape_dc),
                grid_dim=num_blocks,
                block_dim=tiled_block_size,
                attributes=pdl_launch_attributes(_PDL_LEVEL),
            )
            return

        # Scalar (W=1) tiled fallback for short-axis shapes the
        # full-width tier can't tile evenly. `tier=Serial` marks the
        # one-thread-per-output contract that `emit_tile_width == 1`
        # can't carry.
        # TODO: retry with W/2, W/4, ... before giving up on
        # vectorization — shapes that miss at W often divide at a
        # smaller power of two, and scalar loses ~2.4x to the vector
        # tier at large sizes. Deferred: each extra width instantiates
        # _TiledKernel for every rowwise body, and the halved widths
        # are unmeasured.
        if effective_simd > 1 and short_axis_ok:
            comptime w1_params = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=tiled_block_size,
                simd_width=1,
                target="gpu",
                tier=ReduceTier.Serial,
            )
            var w1_num_blocks = min(
                ceildiv(num_rows, tiled_block_size),
                sm_count * _SM_OVERPROVISION,
            )
            ctx.enqueue_function(
                _TiledKernel[rank, w1_params, Body](body, shape_dc),
                grid_dim=w1_num_blocks,
                block_dim=tiled_block_size,
                attributes=pdl_launch_attributes(_PDL_LEVEL),
            )
            return

        # Cooperative non-inner: block-per-output, simd_width=1.
        comptime coop_params = ContextParams(
            axis=axis,
            emit_tile_width=1,
            BLOCK_SIZE=COOPERATIVE_BLOCK_SIZE,
            simd_width=1,
            target="gpu",
        )
        var coop_num_blocks = min(num_rows, sm_count * _SM_OVERPROVISION)
        ctx.enqueue_function(
            _BlockKernel[rank, coop_params, Body](body, shape_dc),
            grid_dim=coop_num_blocks,
            block_dim=COOPERATIVE_BLOCK_SIZE,
            attributes=pdl_launch_attributes(_PDL_LEVEL),
        )
        return

    # ----- Inner axis: per-element-output split-K (opt-in). -------------
    # Phase-aware K+1-launch split-K for normalize-shaped bodies with
    # dependent reduces (softmax's max-then-sum). Occupancy gate like the
    # single-kernel reduce split-K (`num_rows < sm_count`), but on a tuned BYTE
    # floor (`_SPLITK_MIN_ROW_BYTES`) rather than the reduce tier's element
    # floor: small rows are overhead-bound, so the K extra launches would lose.
    # Per-element output needs every block to see the joined stat, so it uses
    # K+1 launches instead of an atomic finish. Comptime-off (byte-identical)
    # when the opt-in is `<= 1`.
    comptime if num_phases > 1:
        if (
            num_rows < sm_count
            and num_rows <= _SPLITK_MAX_ROWS_FOR_SPLIT
            and row_size * dtype_size >= _SPLITK_MIN_ROW_BYTES
        ):
            # `num_splits` capped at `_SPLITK_BLOCK_SIZE` (the combine loads
            # one partial per thread over `BLOCK_SIZE` threads) AND at
            # `_SPLITK_TOTAL_BLOCKS_TARGET // num_rows` (keeps the actual
            # launched block count, `num_rows * num_splits`, from scaling
            # linearly with `num_rows` — see the comment on
            # `_SPLITK_TOTAL_BLOCKS_TARGET`).
            var num_splits = min(
                _SPLITK_BLOCK_SIZE,
                max(1, (sm_count * _SM_OVERPROVISION) // num_rows),
                max(1, _SPLITK_TOTAL_BLOCKS_TARGET // num_rows),
            )
            var total_blocks = num_rows * num_splits

            # Scratch: [num_rows x num_reduces x num_splits] fixed-size
            # slots, one per dependent reduce (`num_phases - 1`; the final
            # write phase needs no slot of its own). No memset — slot
            # [row, reduce, split] is written in its reduce's partial phase
            # before any combine reads it.
            var partials_buf = ctx.enqueue_create_buffer[.uint8](
                num_rows * (num_phases - 1) * num_splits * _SPLITK_STATE_BYTES
            )
            var partials_ptr = partials_buf.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()

            comptime splitk_params = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=_SPLITK_BLOCK_SIZE,
                tier=ReduceTier.Splitk,
                simd_width=1,
                target="gpu",
                num_phases=num_phases,
            )

            # K+1 launches on this stream (`num_phases == K + 1`). Phase
            # p = 0..K: same-stream ordering makes launch p's global writes
            # visible to p+1.
            for phase in range(num_phases):
                ctx.enqueue_function(
                    _PointwiseSplitkKernel[rank, splitk_params, Body](
                        body,
                        shape_dc,
                        partials_ptr,
                        Int32(num_splits),
                        Int32(phase),
                    ),
                    grid_dim=total_blocks,
                    block_dim=_SPLITK_BLOCK_SIZE,
                )

            _ = partials_buf
            return

    # ----- Inner axis: warp / split-K / block tiers. --------------------
    # Warp tier. SIMD-width is taken only on evenly-divisible rows: then
    # `row_stride == row_size` is a multiple of `effective_simd`, so
    # (device allocs being over-aligned) every row base is SIMD-aligned
    # and a body's `ws > 1` tile lands on an aligned offset — the
    # alignment contract, matching the block tier's `use_simd_full`.
    # Odd rows fall to scalar (`sw = 1`).
    var warp_simd = effective_simd > 1 and row_size % effective_simd == 0
    # The warp reduce (`rowwise.reduce`'s `ReduceTier.Warp` branch) grid-strides
    # up to `_WARP_TIER_CHUNK_CAP` chunks regardless of `sw`: at `sw = 1` each
    # chunk is one scalar element per lane, still coalesced (adjacent lanes
    # read adjacent elements) and correctly masked for a non-multiple-of-
    # WARP_SIZE tail (`lane_count < sw` branch). So a non-divisible row need
    # not drop past the warp tier at `WARP_SIZE` (32) elements — it can stay
    # on the warp tier, scalar, up to the same `_WARP_TIER_CHUNK_CAP` chunks
    # the divisible case gets (just narrower ones). Not doing this cost
    # `reduce_min_and_max` a whole tier at cols=129 (0.87x vs legacy; the
    # cols=128 divisible neighbor won 2.44x) — see
    # `Kernels/claude_kb/entries/known-limitations/rowwise-gpu-warp-tier-simd-divisibility-cliffs.md`.
    var warp_max = (
        WARP_SIZE * effective_simd * _WARP_TIER_CHUNK_CAP
    ) if warp_simd else WARP_SIZE * _WARP_TIER_CHUNK_CAP
    if row_size <= warp_max:
        comptime WARP_BLOCK_SIZE = WARP_SIZE * WARP_BLOCK_WARPS
        var num_blocks = min(
            ceildiv(num_rows, WARP_BLOCK_WARPS),
            sm_count * _SM_OVERPROVISION,
        )

        @__parameter
        @__copy_capture(shape_il, num_blocks, body)
        def dispatch_warp[sw: Int]() raises:
            comptime warp_params = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=WARP_BLOCK_SIZE,
                tier=ReduceTier.Warp,
                simd_width=sw,
                target="gpu",
            )
            ctx.enqueue_function(
                _WarpKernel[rank, warp_params, Body](body, shape_dc),
                grid_dim=num_blocks,
                block_dim=WARP_BLOCK_SIZE,
                attributes=pdl_launch_attributes(_PDL_LEVEL),
            )

        # Clamp the per-lane SIMD width so one warp pass keeps all WARP_SIZE
        # lanes busy: `WARP_SIZE * sw <= row_size`. A tiny row otherwise
        # over-provisions the width — e.g. cols=32 with effective_simd=8 gives
        # `warp_span = 32*8 = 256 >> 32`, so only 4 of 32 lanes load (8x
        # waste). A grid-striding row (`row_size >= WARP_SIZE * effective_simd`)
        # keeps the full width; a non-divisible row stays scalar (`sw = 1`).
        # Halving keeps `warp_sw` a power of two dividing `effective_simd`, so
        # the row (a multiple of `effective_simd`) stays a multiple of
        # `warp_sw` — no scalar tail.
        var warp_sw = 1
        if warp_simd:
            warp_sw = effective_simd
            while warp_sw > 1 and WARP_SIZE * warp_sw > row_size:
                warp_sw //= 2

        comptime for i in range(log2_floor(effective_simd) + 1):
            comptime cand = 1 << i
            if warp_sw == cand:
                dispatch_warp[cand]()
                return
        dispatch_warp[1]()
        return

    # Split-K: under-saturated grid + large row + narrow-SIMD dtype.
    # Each row splits across `blocks_per_row` blocks; the last-arriving
    # block (per a per-row atomic counter) joins the partials and emits.
    # Gated on `supports_splitk` — multi-pass bodies need every block to
    # see the joined state, which split-K can't provide (normalize-shaped
    # bodies set it `False`).
    #
    # Also off on Metal. The finish relies on the counter's release/acquire
    # pair publishing each block's partial to the last arriver, which needs the
    # partials buffer marked `coherent(device)` and a device-scoped
    # `atomic_thread_fence`. Metal has both; the backend does not emit them
    # yet, so nothing orders the partial stores and the join folds whatever
    # happens to be in the (un-memset) scratch — the row reduces to a
    # nondeterministic fraction of its true value. Falling through to the block
    # tier is correct, just single-block on an under-saturated shape.
    # TODO(KERN-3391): re-enable when backend supports coherent(device)
    # buffer plus atomic_thread_fence with thread_scope_device
    comptime if (
        supports_splitk
        and effective_simd <= _SPLITK_MAX_SIMD
        and not has_apple_gpu_accelerator()
    ):
        if num_rows < sm_count and row_size >= _SPLITK_MIN_ROW:
            # Cap `blocks_per_row` at `_SPLITK_ROW_BLOCK_SIZE`: the
            # last-arriving block folds one partial per thread, so past
            # `BLOCK_SIZE` the reduce silently drops partials. Below the cap,
            # keep splitting while the rows alone cannot fill the device — a
            # single huge row needs every block it can get.
            var blocks_per_row: Int
            comptime if _SPLITK_ROW_TIER_TUNED:
                blocks_per_row = min(
                    _SPLITK_ROW_BLOCK_SIZE,
                    max(
                        _SPLITK_TARGET_BLOCKS_PER_ROW,
                        ceildiv(sm_count, num_rows * _SPLITK_FILL_DIVISOR),
                    ),
                )
            else:
                blocks_per_row = min(
                    _SPLITK_ROW_BLOCK_SIZE,
                    max(1, (sm_count * _SM_OVERPROVISION) // num_rows),
                )
            var total_blocks = num_rows * blocks_per_row

            var partials_buf = ctx.enqueue_create_buffer[.uint8](
                total_blocks * _SPLITK_STATE_BYTES
            )
            var counters_buf = ctx.enqueue_create_buffer[.int32](num_rows)
            ctx.enqueue_memset(counters_buf, Int32(0))

            var partials_ptr = partials_buf.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
            var counters_ptr = counters_buf.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()

            def dispatch_splitk[
                use_simd: Bool
            ]() raises {
                var shape_il,
                var body,
                var partials_ptr,
                var counters_ptr,
                var blocks_per_row,
                var total_blocks,
                imm,
            }:
                comptime sw = effective_simd if use_simd else 1
                comptime splitk_params = ContextParams(
                    axis=axis,
                    emit_tile_width=1,
                    BLOCK_SIZE=_SPLITK_ROW_BLOCK_SIZE,
                    tier=ReduceTier.Splitk,
                    simd_width=sw,
                    target="gpu",
                )
                ctx.enqueue_function(
                    _SplitkKernel[rank, splitk_params, Body](
                        body,
                        shape_dc,
                        partials_ptr,
                        counters_ptr,
                        Int32(blocks_per_row),
                    ),
                    grid_dim=total_blocks,
                    block_dim=_SPLITK_ROW_BLOCK_SIZE,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

            # Same two conditions as the block tier's `use_simd_full`: the width
            # must divide the row (the alignment a body's `ws > 1` load is
            # promised), and one full stripe pass must fit the row.
            var use_simd_full = (
                effective_simd > 1
                and row_size % effective_simd == 0
                and row_size
                >= blocks_per_row * _SPLITK_ROW_BLOCK_SIZE * effective_simd
            )
            # `unswitch` instantiates both arms, so gate it comptime rather than
            # folding the tier flag into the predicate: off the tuned path the
            # wide-stripe kernel could never launch but would still be compiled.
            comptime if _SPLITK_ROW_TIER_TUNED:
                unswitch(use_simd_full, dispatch_splitk)
            else:
                dispatch_splitk[False]()

            _ = partials_buf
            _ = counters_buf
            return

    # Block tier: 2D SIMD-width heuristic with per-shape BLOCK_SIZE.
    # For expensive per-element work (softmax, layernorm, ...) the
    # address-math amortization behind the multi-pass threshold no
    # longer applies — drop to 1 so SIMD-full kicks in as soon as
    # alignment permits.
    comptime simd_iters_under_sat = (
        1 if computationally_expensive else _SIMD_FULL_ITERS_UNDER_SAT
    )
    comptime simd_iters_sat = (
        1 if computationally_expensive else _SIMD_FULL_ITERS_SAT
    )
    var num_blocks = min(num_rows, sm_count * _SM_OVERPROVISION)
    # The smaller-BS one-pass path is only taken when
    # `num_rows >= sm_count`: with few rows a smaller block loses too
    # much thread parallelism (fewer warps = less latency hiding) and
    # the GPU can't compensate with more blocks (one block per row).
    # With many rows there are plenty of blocks to schedule, so a
    # smaller block that one-pass-covers the row at full SIMD wins.
    var rows_saturate_gpu = num_rows >= sm_count

    def dispatch_bs_one_pass[
        BS: Int
    ]() raises {var shape_il, var num_blocks, var body, imm}:
        # Smaller-BS one-pass path: `BS * simd >= row_size`. Active
        # threads do one SIMD tile; tail threads stay idle (contribute
        # identity to the block reduce). Reached only when
        # `num_rows >= sm_count`, so the GPU stays saturated.
        def dispatch[use_simd: Bool]() raises {imm}:
            comptime sw = effective_simd if use_simd else 1
            comptime block_params = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=BS,
                simd_width=sw,
                target="gpu",
            )
            comptime if cache_count > 0:
                var shmem_bytes = (
                    row_size * cache_count * size_of[cache_dtype.value()]()
                )
                ctx.enqueue_function(
                    _BlockKernel[rank, block_params, Body](body, shape_dc),
                    grid_dim=num_blocks,
                    block_dim=BS,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                    shared_mem_bytes=shmem_bytes,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(shmem_bytes)
                    ),
                )
            else:
                ctx.enqueue_function(
                    _BlockKernel[rank, block_params, Body](body, shape_dc),
                    grid_dim=num_blocks,
                    block_dim=BS,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

        var use_simd_full = (
            effective_simd > 1 and row_size % effective_simd == 0
        )
        unswitch(use_simd_full, dispatch)

    def dispatch_bs_default() raises {
        var shape_il, var num_blocks, var body, imm
    }:
        # Default block tier: BS = launched BLOCK_SIZE, multi-pass via
        # grid stride. Under-utilization gate (#104) drops to sw=1 when
        # `row_size < BLOCK_SIZE * simd` so all threads stay busy with
        # scalar grid-stride loads rather than 1/N of them doing SIMD.
        var iters_full = ceildiv(row_size, BLOCK_SIZE * effective_simd)
        var enough_work_for_simd = row_size >= BLOCK_SIZE * effective_simd
        var use_simd_full = (
            effective_simd > 1
            and row_size % effective_simd == 0
            and enough_work_for_simd
            and (
                iters_full >= simd_iters_under_sat
                or (num_rows >= sm_count and iters_full >= simd_iters_sat)
            )
        )

        def dispatch[use_simd: Bool]() raises {imm}:
            comptime sw = effective_simd if use_simd else 1
            comptime block_params = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=BLOCK_SIZE,
                simd_width=sw,
                target="gpu",
            )
            comptime if cache_count > 0:
                var shmem_bytes = (
                    row_size * cache_count * size_of[cache_dtype.value()]()
                )
                ctx.enqueue_function(
                    _BlockKernel[rank, block_params, Body](body, shape_dc),
                    grid_dim=num_blocks,
                    block_dim=BLOCK_SIZE,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                    shared_mem_bytes=shmem_bytes,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(shmem_bytes)
                    ),
                )
            else:
                ctx.enqueue_function(
                    _BlockKernel[rank, block_params, Body](body, shape_dc),
                    grid_dim=num_blocks,
                    block_dim=BLOCK_SIZE,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

        unswitch(use_simd_full, dispatch)

    def dispatch_bs_large[
        BS: Int
    ]() raises {var shape_il, var num_blocks, var body, imm}:
        # Larger-BS path: few rows leave the GPU under-saturated even at
        # `BLOCK_SIZE=256`, so grow BS to make each row's block fat
        # enough to fill its SM — the counterpart of
        # `dispatch_bs_one_pass` (which shrinks BS to fit many small
        # rows). Reached when `num_rows < sm_count` and the row is big
        # enough that `BS * effective_simd` keeps threads busy (no
        # idle-tail).
        def dispatch[use_simd: Bool]() raises {imm}:
            comptime sw = effective_simd if use_simd else 1
            comptime block_params = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=BS,
                simd_width=sw,
                target="gpu",
            )
            comptime if cache_count > 0:
                var shmem_bytes = (
                    row_size * cache_count * size_of[cache_dtype.value()]()
                )
                ctx.enqueue_function(
                    _BlockKernel[rank, block_params, Body](body, shape_dc),
                    grid_dim=num_blocks,
                    block_dim=BS,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                    shared_mem_bytes=shmem_bytes,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(shmem_bytes)
                    ),
                )
            else:
                ctx.enqueue_function(
                    _BlockKernel[rank, block_params, Body](body, shape_dc),
                    grid_dim=num_blocks,
                    block_dim=BS,
                    attributes=pdl_launch_attributes(_PDL_LEVEL),
                )

        # SIMD width only on SIMD-divisible rows so a body's `ws > 1`
        # tile lands on an aligned offset (the alignment contract, same
        # guard as the other dispatchers); odd rows fall to scalar.
        unswitch(
            effective_simd > 1 and row_size % effective_simd == 0, dispatch
        )

    # Pick BS adaptively:
    # - many rows + small row → shrink BS for one-pass coverage, more
    #   blocks resident per SM (`dispatch_bs_one_pass[64 / 128]`).
    # - few rows + big row → grow BS so each row's lone block fills
    #   its SM (`dispatch_bs_large[1024]`).
    # - otherwise → launched `BLOCK_SIZE` with multi-pass grid stride.
    if rows_saturate_gpu and row_size <= 64 * effective_simd:
        dispatch_bs_one_pass[64]()
    elif rows_saturate_gpu and row_size <= 128 * effective_simd:
        dispatch_bs_one_pass[128]()
    elif not rows_saturate_gpu and row_size >= 1024 * effective_simd:
        dispatch_bs_large[1024]()
    else:
        dispatch_bs_default()
