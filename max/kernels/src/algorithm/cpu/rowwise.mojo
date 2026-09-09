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
"""CPU row-wise reduction scaffolder.

Mirrors the body-facing surface of `algorithm.gpu.rowwise`
so the same `body[params: ContextParams](row_coords, mut ctx)` works
for both targets. Reuses GPU's `ContextParams` / `Context` types
verbatim; the GPU-only runtime fields (split-K pointers and counters)
get sentinels and are never read.

Tier mapping CPU ↔ GPU:

- **Inner-axis cooperative** (`emit_tile_width == 1`, `axis == rank - 1`):
  one worker per row, vectorize the row at `simd_width`. Mirrors the
  GPU block tier with `BLOCK_SIZE = 1`.
- **Non-inner-axis tiled** (`emit_tile_width > 1`, `supports_tiled`): one
  worker per slice; tile the innermost-non-axis dim with
  `W = simd_width` independent accumulators. Mirrors the GPU tiled
  tier.
- **Non-inner-axis cooperative** (`emit_tile_width == 1`, `!supports_tiled`):
  one worker per output, scalar walk over the reduce axis. Mirrors
  the GPU cooperative-non-inner tier.

`pjoin` is a no-op on CPU and `once` runs unconditionally: one worker
owns each output, so the monoid state is already final after `reduce`.
"""

from std.atomic import Atomic
from std.math import align_down, ceildiv
from std.math.math import min as _min
from std.memory import UnsafePointer, unsafe_memset_zero
from std.memory import alloc as ptr_alloc
from std.sys import size_of
from std.sys.info import simd_width_of

from std.utils.coord import Coord, coord_to_index_list

from algorithm.rowwise_types import (
    Context,
    ContextParams,
    ReduceTier,
    RowBody,
    RowCoord,
    _num_outputs_excluding_axis,
)
from max.algorithm.backend.cpu.parallelize import (
    _get_num_workers,
    sync_parallelize,
)
from algorithm.reduce_op import ReduceOp, Reducer
from max.algorithm.reduction import _get_nd_indices_from_flat_index


# Bytes per partial-state slot in the split-axis scratch buffer. `launch`
# allocates the buffer before the body picks its concrete monoid, so the
# slot is a fixed worst-case size. Slots hold the lane-collapsed
# `State.Single`, so 128 bytes covers every monoid; `pjoin` asserts the
# fit at compile time.
comptime _SPLITK_STATE_BYTES = 128

# Gating constants for the CPU axis-split tier. Enable only when the
# worker pool is significantly under-saturated (`num_outputs * 4 <
# num_workers`) AND the axis is large enough that a single-worker walk
# would exceed L2/L3 cache or take longer than the multi-worker fan-out
# overhead. Empirically a single-worker walk on a 128K-element row
# (~256KB bf16) is ~20μs, faster than spawning 7 workers + atomic
# finish; split-K only wins above that, so set the threshold there.
# `_CPU_SPLITK_MIN_AXIS_PER_SPLIT` caps the split factor so each split
# keeps enough work to amortize.
comptime _CPU_SPLITK_MIN_AXIS = 262144
comptime _CPU_SPLITK_MIN_AXIS_PER_SPLIT = 32768
comptime _CPU_SPLITK_MIN_SPLITS = 2


# Inner-axis cooperative tier, wide-accumulator (opt-in via `associative`).
# The `reduce` else-branch issues `UNROLL` `state.accumulate` calls per block
# into ONE monoid state, forming a depth-`UNROLL` serial `acc += v_u`
# dependency chain (add-latency-bound → ~half bandwidth on large contiguous
# rows). Widening the monoid *width* (NOT its dtype — accumulation stays in the
# input dtype) to `_CPU_ILP_ACCUMULATORS * simd_width` makes each link
# `_CPU_ILP_ACCUMULATORS`-way lane-parallel, recovering the ILP the legacy
# inner reducer got from its `SIMD[dtype, UNROLL*simd_width]` fat vector. Only
# additive reductions (sum/mean) opt in — the wider bf16 reassociation stays in
# tolerance (and is closer to legacy's own wide bf16 accumulator); product's
# bf16 numerics can't widen safely, and max/min gain nothing. Row-size gated so
# small rows keep native width; decoupled from the non-inner tiled tier.
comptime _CPU_ILP_ACCUMULATORS = 4
comptime _CPU_WIDE_ACC_MIN_ROW = 2048


# ===-----------------------------------------------------------------------===#
# `SerialReducer` — CPU single-participant `Reducer`.
# ===-----------------------------------------------------------------------===#


struct SerialReducer(Reducer, TrivialRegisterPassable):
    """Single-participant `Reducer` for CPU's serial / cooperative
    tier. Nothing to combine across (one worker per output), so
    `sum`/`max`/`min`/`generic` are identity. Routing
    `cpu_rowwise.pjoin` through it rather than skipping the call still
    invokes the monoid's own `pjoin` body, which may do per-state
    finalization (e.g. `ReduceSum` horizontally reducing its SIMD-wide
    accumulator into the scalar `acc` field exposed to bodies)."""

    @always_inline
    def __init__(out self):
        """Initializes the stateless single-participant reducer."""
        pass

    @always_inline
    def sum[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns `val` unchanged — nothing to sum across on CPU.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The participant's value.

        Returns:
            `val` unchanged.
        """
        return val

    @always_inline
    def max[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns `val` unchanged — nothing to combine across on CPU.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The participant's value.

        Returns:
            `val` unchanged.
        """
        return val

    @always_inline
    def min[dtype: DType](self, val: Scalar[dtype]) -> Scalar[dtype]:
        """Returns `val` unchanged — nothing to combine across on CPU.

        Parameters:
            dtype: The scalar dtype.

        Args:
            val: The participant's value.

        Returns:
            `val` unchanged.
        """
        return val

    @always_inline
    def generic[S: ReduceOp](self, mut state: S):
        """No-op cross-thread combine — one worker per output on CPU.

        Parameters:
            S: The monoid type.

        Args:
            state: The monoid state (left unchanged).
        """
        pass


# ===-----------------------------------------------------------------------===#
# Body-facing primitives: `reduce`, `pjoin`, `once`, `simd`.
# ===-----------------------------------------------------------------------===#


@always_inline
def _simd_walk_unrolled[
    W: Int,
    rank: Int,
    //,
    axis: Int,
    TileFn: ImplicitlyCopyable & (def[ws: Int](RowCoord[rank]) -> None),
](mut coords: RowCoord[rank], lo: Int, hi: Int, tile_fn: TileFn):
    """8-way-unrolled SIMD walk over `[lo, hi)` along `axis`, then SIMD
    and scalar tails. Shared by the split-axis tier (`[lo, hi)` a
    per-worker stripe) and the inner-axis cooperative tier
    (`[0, axis_size)`, the whole row) — the two tiers differ only in
    which interval they walk, not in how they walk it.
    """
    comptime UNROLL = 8
    comptime W_U = W * UNROLL
    var span = hi - lo
    var simd_unrolled = lo + align_down(span, W_U)
    var simd_aligned = lo + align_down(span, W)
    var k = lo
    while k < simd_unrolled:
        comptime for u in range(UNROLL):
            coords.write_axis[axis](k + u * W)
            tile_fn[W](coords)
        k += W_U
    while k < simd_aligned:
        coords.write_axis[axis](k)
        tile_fn[W](coords)
        k += W
    while k < hi:
        coords.write_axis[axis](k)
        tile_fn[1](coords)
        k += 1


@always_inline
def _simd_walk_unrolled[
    State: ReduceOp,
    W: Int,
    rank: Int,
    //,
    axis: Int,
    TileFn: ImplicitlyCopyable
    & (def[ws: Int](mut State, RowCoord[rank]) -> None),
](
    mut coords: RowCoord[rank],
    lo: Int,
    hi: Int,
    mut state: State,
    tile_fn: TileFn,
):
    """State-carrying overload of `_simd_walk_unrolled` — see there for
    the tier-sharing rationale."""
    comptime UNROLL = 8
    comptime W_U = W * UNROLL
    var span = hi - lo
    var simd_unrolled = lo + align_down(span, W_U)
    var simd_aligned = lo + align_down(span, W)
    var k = lo
    while k < simd_unrolled:
        comptime for u in range(UNROLL):
            coords.write_axis[axis](k + u * W)
            tile_fn[W](state, coords)
        k += W_U
    while k < simd_aligned:
        coords.write_axis[axis](k)
        tile_fn[W](state, coords)
        k += W
    while k < hi:
        coords.write_axis[axis](k)
        tile_fn[1](state, coords)
        k += 1


@always_inline
def reduce[
    params: ContextParams,
    rank: Int,
    //,
    TileFn: ImplicitlyCopyable & (def[ws: Int](RowCoord[rank]) -> None),
](
    row_coords: RowCoord[rank],
    axis_size: Int,
    ctx: Context[params],
    tile_fn: TileFn,
):
    """Drives `tile_fn` over the reduce axis, CPU-side, with no monoid
    state — pure per-tile iteration for map/emit terminals (see the
    state-carrying overload below for reduce phases).

    Tiers mirror the state-carrying overload; see there for the
    per-tier walk description.

    Parameters:
        params: Comptime dispatch parameters.
        TileFn: The value-closure type of `tile_fn`.

    Args:
        row_coords: The current row's coords.
        axis_size: Length of the reduce axis.
        ctx: The dispatch bundle (unused on CPU beyond comptime
            params reads).
        tile_fn: Per-tile callback; closes over input/output closures.
    """
    var coords = row_coords

    comptime if ctx._tier == ReduceTier.Splitk:
        var num_splits = Int(ctx._blocks_per_row)
        var split = Int(ctx._block_in_row)
        var stripe = ceildiv(axis_size, num_splits)
        var lo = split * stripe
        var hi = _min((split + 1) * stripe, axis_size)
        _simd_walk_unrolled[W=ctx.simd_width, rank=rank, axis=ctx.axis](
            coords, lo, hi, tile_fn
        )
    elif ctx.emit_tile_width > 1:
        comptime W = ctx.emit_tile_width
        for k in range(axis_size):
            coords.write_axis[ctx.axis](k)
            tile_fn[W](coords)
    else:
        _simd_walk_unrolled[W=ctx.simd_width, rank=rank, axis=ctx.axis](
            coords, 0, axis_size, tile_fn
        )


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
    ctx: Context[params],
    mut state: State,
    tile_fn: TileFn,
):
    """Drives `tile_fn` over the reduce axis, CPU-side.

    Two tiers, picked from `params.emit_tile_width`:

    - `emit_tile_width == 1`: walk axis in `simd_width` chunks, 8-way
      unrolled (matches legacy `_reduce_along_inner_dimension`), then
      SIMD and scalar tails. `tile_fn[ws]` is called with
      `ws ∈ {simd_width, 1}`; the monoid's width-polymorphic
      `reduce[w]` collapses each tile.
    - `emit_tile_width > 1`: scalar axis walk, `tile_fn[W]` per step. The
      body holds `W` parallel accumulators (the monoid `M` at width `W`); each step
      issues one SIMD load on the innermost-non-axis dim, spreading
      lanes across the accumulators.

    `tile_fn` is a value closure (its copy-captured state — input
    closures, side-loaded tensors — rides the value); it takes the
    caller's monoid `state` as a `mut` argument rather than capturing it,
    since a captured accumulator can't be mutated through a value
    closure (captures are immutable) but an `inout` argument can.

    Parameters:
        State: The monoid type being accumulated.
        params: Comptime dispatch parameters.
        TileFn: The value-closure type of `tile_fn`.

    Args:
        row_coords: The current row's coords.
        axis_size: Length of the reduce axis.
        ctx: The dispatch bundle (unused on CPU beyond comptime
            params reads).
        state: The caller's monoid accumulator, threaded through by
            `mut` reference and mutated in place by `tile_fn`.
        tile_fn: Per-tile callback; closes over input closures and
            folds each tile into `state`.
    """
    var coords = row_coords

    comptime if ctx._tier == ReduceTier.Splitk:
        # Split-axis tier: each worker walks ONE stripe of the reduce
        # axis, sized `ceildiv(axis_size, blocks_per_row)`. Stripes are
        # contiguous (not strided) on CPU — locality matters more than
        # the cross-worker cache-line interleave that helps GPU warps.
        # SIMD-vectorized within the stripe, 8-way unrolled like the
        # cooperative tier (`_simd_walk_unrolled` — shared with it).
        var num_splits = Int(ctx._blocks_per_row)
        var split = Int(ctx._block_in_row)
        var stripe = ceildiv(axis_size, num_splits)
        var lo = split * stripe
        var hi = _min((split + 1) * stripe, axis_size)
        _simd_walk_unrolled[W=ctx.simd_width, rank=rank, axis=ctx.axis](
            coords, lo, hi, state, tile_fn
        )
    elif ctx.emit_tile_width > 1:
        comptime W = ctx.emit_tile_width
        for k in range(axis_size):
            coords.write_axis[ctx.axis](k)
            tile_fn[W](state, coords)
    else:
        # 8-way unrolled SIMD walk along the whole axis (unroll factor
        # from `_reduce_along_inner_dimension`); the unroll lets the OoO
        # core overlap loads + reduces. Shared with the split-axis tier
        # above via `_simd_walk_unrolled` — same walk, `[0, axis_size)`.
        _simd_walk_unrolled[W=ctx.simd_width, rank=rank, axis=ctx.axis](
            coords, 0, axis_size, state, tile_fn
        )


@always_inline
def pjoin[
    State: ReduceOp,
    params: ContextParams,
    //,
](mut state: State, mut ctx: Context[params]):
    """CPU: drives the monoid's `pjoin` body via `SerialReducer` in the
    cooperative tier, or merges the cross-worker partials buffer in the
    split-axis tier.

    Three tiers, picked from `params`:

    - Cooperative (`emit_tile_width == 1`, `!is_splitk_tier`): call
      `state.pjoin(SerialReducer())`. No cross-worker combine (one
      worker per output), but monoids may carry per-state finalization
      in `pjoin` (e.g. `ReduceSum` horizontally reducing its SIMD
      accumulator into the scalar `acc` field).
    - Tiled (`emit_tile_width > 1`): per-output accumulators already final;
      each lane is a separate output; skip the cross-participant join.
    - Split-axis (`is_splitk_tier`): write this worker's width-1
      partial to its slot in `ctx._partials_base`, atomically increment
      `counters_base[row_idx]`, and — only the worker whose increment
      brings the counter to `blocks_per_row` — gather all partials,
      `join` them, run `join_parallel` once on the merged result, and
      set `ctx._is_last_block` so `rowwise.once` gates emission.

    Parameters:
        State: The monoid type being joined.
        params: Comptime dispatch parameters.

    Args:
        state: The body's monoid accumulator. On return, holds the
            finalized value.
        ctx: The dispatch bundle. Mutated in the split-axis tier so
            `rowwise.once` knows which worker emits.
    """
    comptime if ctx._tier == ReduceTier.Splitk:
        comptime assert (
            size_of[State.Single]() <= _SPLITK_STATE_BYTES
        ), "split-K partial slot too small for this monoid's width-1 state"
        # Collapse this worker's lanes to width 1 and publish the raw
        # width-1 partial, so it stays `join`-able for the cross-worker
        # merge. Finalization happens once, on the fully merged state.
        var sc = state.reduce()
        var num_splits = Int(ctx._blocks_per_row)
        var split = Int(ctx._block_in_row)
        var row_idx_ = Int(ctx._row_idx)
        var row_base_bytes = (
            ctx._partials_base + row_idx_ * num_splits * _SPLITK_STATE_BYTES
        )
        var slot_ptr = (row_base_bytes + split * _SPLITK_STATE_BYTES).bitcast[
            State.Single
        ]()
        slot_ptr[0] = sc
        var counter_ptr = ctx._counters_base + row_idx_
        # `fetch_add` returns the prior value; `prev + 1 == num_splits`
        # means this worker arrived last.
        var prev = Atomic[Int32].fetch_add(counter_ptr, Int32(1))
        var is_last = Int(prev) + 1 == num_splits
        ctx._is_last_block = is_last
        if is_last:
            # Merge other workers' partials into self.
            for s in range(num_splits):
                if s == split:
                    continue
                var other_ptr = (
                    row_base_bytes + s * _SPLITK_STATE_BYTES
                ).bitcast[State.Single]()
                sc.join(other_ptr[0])
            sc.join_parallel(SerialReducer())
        state = State(sc)
    elif ctx.emit_tile_width == 1:
        # Cooperative: collapse lanes to a width-1 scalar, finalize, broadcast
        # back (branchless — identity at W == 1).
        var sc = state.reduce()
        sc.join_parallel(SerialReducer())
        state = State(sc)


@always_inline
def once[
    Emit: ImplicitlyCopyable & RegisterPassable & (def() -> None),
    params: ContextParams,
    //,
](emit: Emit, ctx: Context[params]):
    """CPU: runs `emit` unconditionally on the cooperative / tiled
    tiers (one worker per output), or gated on `ctx._is_last_block` on
    the split-axis tier (only the last-arriving worker emits).

    `emit` is a value closure (its copy-captured state rides the value).

    Parameters:
        Emit: The value-closure type of `emit`.
        params: Comptime dispatch parameters.

    Args:
        emit: Callback to run.
        ctx: The dispatch bundle.
    """
    comptime if ctx._tier == ReduceTier.Splitk:
        if ctx._is_last_block:
            emit()
    else:
        emit()


# ===-----------------------------------------------------------------------===#
# `launch` — top-level CPU scaffolder.
# ===-----------------------------------------------------------------------===#


def launch[
    Body: RowBody,
    //,
    axis: Int,
    simd_width: Int,
    supports_tiled: Bool = False,
    associative: Bool = False,
](body: Body, shape: Coord) raises:
    """Top-level CPU scaffolder. Picks the tier from `axis` and
    `supports_tiled`, parallelizes over outputs (or output tiles), and
    invokes the body once per output (or tile).

    Tier choice:

    - **Inner axis** (`axis == rank - 1`) — one worker per row,
      vectorize within the row at `simd_width`. `emit_tile_width = 1`.
    - **Non-inner axis** with `supports_tiled` — one worker per slice;
      each walks the innermost-non-axis dim in `simd_width` tiles, with
      a scalar tail for the remainder. `emit_tile_width = simd_width`.
    - **Non-inner axis** without `supports_tiled` — one worker per
      output, scalar reduce. `emit_tile_width = 1`.

    Parameters:
        axis: Axis being reduced.
        simd_width: SIMD width for tile dispatch. Bodies compute it via
            `rowwise.pick_simd_width[...]` and pass it directly; the
            scaffolder no longer derives it from a dtype.
        supports_tiled: Whether the body can run in the tiled tier
            (`emit_tile_width > 1`). Single-output reductions (`reduce_sum`,
            …) set this `True`; multi-output kernels (softmax,
            layernorm) set it `False`.
        associative: Opt in the inner-axis cooperative tier's wide
            (`_CPU_ILP_ACCUMULATORS * simd_width`) same-dtype accumulator
            for rows `>= _CPU_WIDE_ACC_MIN_ROW`, breaking the serial
            accumulate dependency chain. Additive reductions
            (`reduce_sum`, `reduce_mean`) set this `True`; everything
            else keeps the native width.

    Args:
        body: The per-row computation. Receives a `Context` and the
            row's `Coord`; uses `reduce` / `pjoin` / `once` / `simd`
            to compose the algorithm.
        shape: Tensor shape.

    Raises:
        If a worker raises.
    """
    comptime rank = shape.rank
    comptime effective_simd = simd_width

    var shape_il = coord_to_index_list(shape)
    var axis_size = shape_il[axis]
    var total_size = shape_il.flattened_length()
    # `num_outputs` is the product of every dim other than `axis`, so it
    # stays well-defined when the reduce axis itself is empty
    # (`axis_size == 0`) — unlike `total_size // axis_size`, which is a
    # `0 // 0` form in that case. A reduce-shaped body still owns one
    # output per row when the axis is empty (the monoid identity), so
    # only `num_outputs == 0` (no rows at all) means there is truly
    # nothing to run.
    var num_outputs = _num_outputs_excluding_axis[axis](shape)
    if num_outputs == 0:
        return

    var num_workers = _get_num_workers(total_size)

    comptime params_scalar = ContextParams(
        axis=axis,
        emit_tile_width=1,
        BLOCK_SIZE=1,
        simd_width=effective_simd,
        target="cpu",
    )

    comptime if axis == rank - 1:
        # Inner-axis: pick between cooperative (one worker per row) and
        # split-axis (multiple workers per row, fanning out across the
        # axis) by how saturated the worker pool is. Require ≥4×
        # under-subscription before split-K fires, so its
        # sync_parallelize cost has idle workers to soak up. Then cap
        # the split factor at `axis_size / MIN_AXIS_PER_SPLIT` so each
        # split keeps enough work to amortize.
        var splits_pool = num_workers // num_outputs if num_outputs > 0 else 1
        var splits_axis = axis_size // _CPU_SPLITK_MIN_AXIS_PER_SPLIT
        var splits_per_row = _min(splits_pool, splits_axis)
        if (
            num_outputs * 4 <= num_workers
            and splits_per_row >= _CPU_SPLITK_MIN_SPLITS
            and axis_size >= _CPU_SPLITK_MIN_AXIS
        ):
            # ---- split-axis tier ----
            # Each `(row, split)` pair is one sync_parallelize worker.
            # Per-worker pjoin writes its partial to `partials_base`,
            # atomically increments `counters_base`; the last-arriving
            # worker gathers all partials, joins them, and triggers
            # `rowwise.once`'s emit gate.
            comptime params_splitk = ContextParams(
                axis=axis,
                emit_tile_width=1,
                BLOCK_SIZE=1,
                tier=ReduceTier.Splitk,
                simd_width=effective_simd,
                target="cpu",
            )
            var num_splits = splits_per_row
            var total_workers = num_outputs * num_splits
            # Heap scratch: per-row counter + per-(row,split) partial
            # slot. Both freed at the end of the function.
            var partials_size = num_outputs * num_splits * _SPLITK_STATE_BYTES
            var partials_buf = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](
                ptr_alloc[UInt8](partials_size)
            )
            unsafe_memset_zero(partials_buf, partials_size)
            var counters_buf = rebind[UnsafePointer[Int32, MutUntrackedOrigin]](
                ptr_alloc[Int32](num_outputs)
            )
            unsafe_memset_zero(counters_buf, num_outputs)

            @always_inline
            def split_worker(
                w: Int,
            ) {
                var partials_buf,
                var counters_buf,
                var num_splits,
                var body,
                imm,
            }:
                var row_idx = w // num_splits
                var split_idx = w % num_splits
                var row_coords = _get_nd_indices_from_flat_index(
                    row_idx, shape_il, axis
                )
                var ctx = Context[params_splitk](
                    partials_base=partials_buf,
                    counters_base=counters_buf,
                    blocks_per_row=Int32(num_splits),
                    block_in_row=Int32(split_idx),
                    row_idx=Int32(row_idx),
                    is_last_block=False,
                )
                body[params_splitk](Coord(row_coords.canonicalize()), ctx)

            sync_parallelize(split_worker, total_workers)
            partials_buf.free()
            counters_buf.free()
        else:
            # ---- cooperative tier (one worker per row) ----
            # Clamp worker count to `num_outputs` to avoid paying
            # sync_parallelize per-worker setup for empty slots (legacy
            # `_argn` does this too). Without it, (32, 128256) on a
            # 96-thread pool launches 96 workers, 64 of which return
            # immediately — ~700μs of empty-worker setup, larger than
            # the reduction itself.
            var actual_workers = _min(num_workers, num_outputs)
            var chunk = ceildiv(num_outputs, actual_workers)

            @always_inline
            def run_coop[params: ContextParams]() {imm}:
                @always_inline
                def row_worker(w: Int) {var body, imm}:
                    var start = w * chunk
                    var end = _min((w + 1) * chunk, num_outputs)
                    if start >= end:
                        return
                    var ctx = Context[params].empty()
                    for flat_idx in range(start, end):
                        var row_coords = _get_nd_indices_from_flat_index(
                            flat_idx, shape_il, axis
                        )
                        body[params](Coord(row_coords.canonicalize()), ctx)

                sync_parallelize(row_worker, actual_workers)

            # Opt-in wide-bf16-accumulator path for large rows (`associative`,
            # sum/mean only): a `_CPU_ILP_ACCUMULATORS * simd_width`-wide monoid
            # (same dtype) breaks the serial `acc += v_u` chain into lane-
            # parallel adds, recovering bandwidth. Small rows and non-opted-in
            # ops (product/max/min/arg) run the native `params_scalar` width.
            comptime if associative and effective_simd > 1 and _CPU_ILP_ACCUMULATORS > 1:
                comptime params_wide = ContextParams(
                    axis=axis,
                    emit_tile_width=1,
                    BLOCK_SIZE=1,
                    simd_width=effective_simd * _CPU_ILP_ACCUMULATORS,
                    target="cpu",
                )
                if axis_size >= _CPU_WIDE_ACC_MIN_ROW:
                    run_coop[params_wide]()
                    return
            run_coop[params_scalar]()
    else:
        comptime inner_axis = rank - 1
        var inner_dim = shape_il[inner_axis]
        # `num_outputs // inner_dim` rather than `total_size // (axis_size *
        # inner_dim)`: the latter is `0 // 0` when `axis_size == 0`.
        # `inner_dim` is one of the dims `num_outputs` multiplies over (it
        # isn't `axis` in this non-inner-axis branch), so `num_outputs > 0`
        # (checked above) guarantees `inner_dim > 0` here.
        var slice_size = num_outputs // inner_dim
        var chunk = ceildiv(slice_size, num_workers)

        # NON-inner axis. Two sub-tiers, mirroring the GPU non-inner split:
        #
        # - Tiled (`supports_tiled`, `W > 1`): the innermost-non-axis dim is
        #   contiguous, so vectorize ACROSS it — one thread owns `W` adjacent
        #   output columns. Each axis step issues one contiguous SIMD load of
        #   those `W` columns into `W` independent monoid lanes (scalar walk
        #   over the strided reduce axis), and the body's `emit` writes the `W`
        #   columns via `output_fn[W]`. Mirrors the GPU `_tiled_kernel` and the
        #   legacy `_reduce_along_outer_dimension`'s `vectorize[simd_width]`.
        #   `emit_tile_width = W` routes `reduce` through its tiled branch and makes
        #   `pjoin` a no-op (the `W` lanes are independent outputs). Only the
        #   aligned bulk `align_down(inner_dim, W)` is tiled; the odd remainder
        #   falls to the scalar tail.
        # - Scalar-strided (`!supports_tiled`, `W == 1`, or the remainder):
        #   `simd_width = 1`, one output column at a time. A SIMD-width
        #   contiguous load along the strided axis would span neighboring rows,
        #   so multi-output kernels (softmax / layernorm) stay here.
        comptime params_scalar_strided = ContextParams(
            axis=axis,
            emit_tile_width=1,
            BLOCK_SIZE=1,
            simd_width=1,
            target="cpu",
        )
        comptime W = effective_simd
        comptime params_tiled = ContextParams(
            axis=axis,
            emit_tile_width=W,
            BLOCK_SIZE=1,
            simd_width=W,
            target="cpu",
        )

        @always_inline
        def slice_worker(w: Int) {var body, imm}:
            var s_start = w * chunk
            var s_end = _min((w + 1) * chunk, slice_size)
            if s_start >= s_end:
                return
            var ctx_scalar = Context[params_scalar_strided].empty()
            for s in range(s_start, s_end):
                # Compute the slice's coords ONCE, then mutate
                # `coords[inner_axis]` per tile/column. Flat encoding matches
                # the legacy outer-dim reducer: `flat = s * inner_dim` with
                # skip_dim=axis decodes to `coords[axis] = 0`,
                # `coords[inner_axis] = 0`, other dims from the slice.
                var slice_coords = _get_nd_indices_from_flat_index(
                    s * inner_dim, shape_il, axis
                ).canonicalize()
                var k = 0
                comptime if supports_tiled and W > 1:
                    var ctx_tiled = Context[params_tiled].empty()
                    var bulk = align_down(inner_dim, W)
                    while k < bulk:
                        slice_coords[inner_axis] = k
                        body[params_tiled](Coord(slice_coords), ctx_tiled)
                        k += W
                # Scalar tail: the odd `inner_dim % W` remainder, or every
                # column when the tiled tier is comptime-disabled.
                while k < inner_dim:
                    slice_coords[inner_axis] = k
                    body[params_scalar_strided](Coord(slice_coords), ctx_scalar)
                    k += 1

        sync_parallelize(slice_worker, num_workers)
