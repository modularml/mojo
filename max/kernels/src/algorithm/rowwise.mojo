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
"""Unified row-wise reduction scaffolder.

Author surface for kernels targeting both CPU and GPU with one body.
Re-exports `ContextParams` and `Context` from
`algorithm.rowwise_types`, and provides comptime-dispatched
`reduce` / `pjoin` / `once` / `simd` / `launch` that pick the CPU or
GPU backend from `params.target` (a comptime `StaticString` field on
`ContextParams`).

Author contract — one body, two targets:

```
def reduce_sum[...](shape, target="cpu", ctx=None) raises:
    var axis_size: DType = ...

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {var axis_size}:
        comptime W = params.emit_tile_width
        comptime rank = row_coords.rank
        var state = ReduceSum[accum_type, W]()

        @always_inline
        def tile_fn[
            ws: Int
        ](mut state: ReduceSum[accum_type, W], coords: RowCoord[rank]):
            ...
            state.accumulate[dtype, ws](...)

        rowwise.reduce(row_coords, axis_size, ctx, state, tile_fn)
        rowwise.pjoin(state, ctx)

        @always_inline
        def emit() {...}:
            ...

        rowwise.once(emit, ctx)

    # `num_phases=1` if the output collapses the axis (one value per
    # row); `num_phases>1` if the output keeps the input shape
    # (per-element emission along axis).
    rowwise.launch[body, axis=..., simd_width=..., target=target, num_phases=1, ...](shape, ctx)
```

The body never branches on `target` directly. `rowwise.reduce` /
`pjoin` / `once` / `simd` each comptime-dispatch via
`comptime if is_cpu[params.target]():`, so GPU primitives never appear
in CPU codegen and vice versa. `tile_fn` and `emit` are value closures
(their copy-captured state rides the value); a reduce phase's `state`
is threaded through `reduce` as a `mut` argument rather than captured,
since a captured accumulator can't be mutated through a value closure.

`computationally_expensive` is a GPU-only author hint, forwarded to the
GPU backend and ignored by the CPU backend. `BLOCK_SIZE` /
`COOPERATIVE_BLOCK_SIZE` / `WARP_BLOCK_WARPS` are GPU launch-geometry
constants private to `launch` — no caller has ever needed to override
them, so they are not part of its public signature (see the
tier-dispatch notes above `launch` for what they do). The CPU backend
runs `sync_parallelize` over output rows / slices (no `DeviceContext`);
the GPU backend launches kernels through the provided `DeviceContext`.
"""

from algorithm.cpu.rowwise import (
    launch as _cpu_launch,
    once as _cpu_once,
    pjoin as _cpu_pjoin,
    reduce as _cpu_reduce,
)
from algorithm.gpu.rowwise import (
    launch as _gpu_launch,
    once as _gpu_once,
    pjoin as _gpu_pjoin,
    reduce as _gpu_reduce,
    _pointwise_splitk_combine,
    _pointwise_splitk_store_partial,
)
from algorithm.rowwise_types import (
    Context,
    ContextParams,
    ReduceTier,
    RowBody,
    RowCoord,
    tile_alignment,
)
from algorithm.reduce_op import ReduceOp, Reducer
from std.collections import Optional
from max.gpu import (
    WARP_SIZE,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu import barrier, syncwarp
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.memory import external_memory
from max.gpu.host.info import is_cpu, is_gpu
from std.math import ceildiv, iota
from std.memory import bitcast, stack_allocation
from std.memory.unsafe_pointer import UnsafePointer
from std.sys.info import (
    _current_target,
    align_of,
    bit_width_of,
    is_apple_gpu,
    simd_width_of,
    size_of,
)
from std.sys.intrinsics import strided_load as _strided_load
from std.utils.coord import Coord, CoordLike
from std.utils.static_tuple import StaticTuple


# ===-----------------------------------------------------------------------===#
# Body-facing primitives.
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
    """Drives `tile_fn` over the reduce axis on the target backend, with
    no monoid state — pure per-tile iteration. See the state-carrying
    overload below for reduce phases.

    Parameters:
        params: Comptime dispatch parameters (target + tier).
        TileFn: The value-closure type of `tile_fn`.

    Args:
        row_coords: The current row's coords.
        axis_size: Length of the reduce axis.
        ctx: The dispatch bundle.
        tile_fn: Per-tile callback; closes over input/output closures.
    """
    comptime if is_cpu[params.target]():
        _cpu_reduce(row_coords, axis_size, ctx, tile_fn)
    else:
        comptime assert is_gpu[params.target](), "unknown rowwise target"
        _gpu_reduce(row_coords, axis_size, ctx, tile_fn)


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
    """Drives `tile_fn` over the reduce axis on the target backend.

    See the CPU and GPU `reduce` impls for per-target tier semantics;
    both call `tile_fn[ws]` per axis tile, folding into `state`.
    `tile_fn` is a value closure taking `state` as a `mut` argument
    (not a capture — a captured accumulator can't be mutated through a
    value closure).

    Parameters:
        State: The monoid type being accumulated.
        params: Comptime dispatch parameters (target + tier).
        TileFn: The value-closure type of `tile_fn`.

    Args:
        row_coords: The current row's coords.
        axis_size: Length of the reduce axis.
        ctx: The dispatch bundle.
        state: The caller's monoid accumulator, threaded through by
            `mut` reference and mutated in place by `tile_fn`.
        tile_fn: Per-tile callback; closes over input closures and
            folds each tile into `state`.
    """
    comptime if is_cpu[params.target]():
        _cpu_reduce(row_coords, axis_size, ctx, state, tile_fn)
    else:
        comptime assert is_gpu[params.target](), "unknown rowwise target"
        _gpu_reduce(row_coords, axis_size, ctx, state, tile_fn)


@always_inline
def pjoin[
    State: ReduceOp,
    params: ContextParams,
    //,
](mut state: State, mut ctx: Context[params]):
    """Cross-thread join for a single monoid state.

    GPU: cooperative warp/block reduce (warp / block tiers) or
    cross-block atomic finish (split-K); no-op for the tiled and serial
    tiers. CPU: drives the monoid's own `pjoin` body via a
    single-participant `SerialReducer` — no cross-thread combine, but
    monoids may run per-state finalization here (e.g. `ReduceSum`
    horizontally reducing its SIMD accumulator into the scalar `acc`
    field bodies read).

    Parameters:
        State: The monoid type being joined.
        params: Comptime dispatch parameters.

    Args:
        state: The body's monoid accumulator.
        ctx: The dispatch bundle.
    """
    comptime if is_cpu[params.target]():
        _cpu_pjoin(state, ctx)
    else:
        comptime assert is_gpu[params.target](), "unknown rowwise target"
        _gpu_pjoin(state, ctx)


@always_inline
def once[
    Emit: ImplicitlyCopyable & RegisterPassable & (def() -> None),
    params: ContextParams,
    //,
](emit: Emit, ctx: Context[params]):
    """Runs `emit` exactly once per logical output row.

    GPU: picks the canonical writer (lane 0 / thread 0 / last-block
    thread 0, depending on tier). CPU: every worker emits its own
    outputs (one worker per row).

    `emit` is a value closure (its copy-captured state rides the value);
    `once` is an in-kernel thread predicate — not a launch boundary — so it
    is invoked directly under the canonical-writer guard.

    Parameters:
        Emit: The value-closure type of `emit`.
        params: Comptime dispatch parameters.

    Args:
        emit: Callback to run on the canonical writer.
        ctx: The dispatch bundle.
    """
    comptime if is_cpu[params.target]():
        _cpu_once(emit, ctx)
    else:
        comptime assert is_gpu[params.target](), "unknown rowwise target"
        _gpu_once(emit, ctx)


@always_inline
# ===-----------------------------------------------------------------------===#
# `pick_simd_width` — body-side SIMD width selection.
# ===-----------------------------------------------------------------------===#


@always_inline
def pick_simd_width[
    M: ReduceOp,
    target: StaticString = "cpu",
    PER_THREAD_BUDGET: Int = 64,
    *Ts: DType,
]() -> Int:
    """Picks the SIMD width a rowwise body should use throughout.

    Mirrors elementwise's "pick the smallest dtype's natural SIMD
    width" rule: the widest register that fits the smallest element,
    with any wider-element dtype expressed via paired SIMD ops at no
    perf cost. Computed as `max(simd_width_of[T] for T in Ts)`.

    On GPU only, caps the result down when `M`'s storage at `W=1`
    (`sizeof[M]`) times the max SIMD width would exceed
    `PER_THREAD_BUDGET` bytes per thread. This protects heavy-state
    monoids (ArgMax/ArgMin with int64 indices, OnlineLogSumExp /
    future Welford with multi-field state) from the GPU `BLOCK_SIZE`
    multiplier blowing past the register budget. Single-field monoids
    (Sum/Max/Min/Product) almost always stay under the cap.

    Why `M` at `W=1`: bodies declare states like `M[dtype, 1]` for the
    budget check; the helper assumes state size scales roughly linearly
    with `W` (true for SIMD field accumulators). Bodies wanting a
    sub-1 width or a custom rule can compute `W` themselves.

    Parameters:
        M: The monoid type at `W=1` (`sizeof` proxy for the GPU
            storage-budget check).
        target: `"cpu"` or `"gpu"` (anything non-CPU = GPU).
        PER_THREAD_BUDGET: GPU per-thread storage cap in bytes.
            Tunable; default 64B is an empirically reasonable
            register-pressure ceiling on B200-class GPUs.
        Ts: Variadic pack of dtypes involved in the body (input,
            accumulator, output, gamma weights, ...); the helper takes
            the max natural SIMD width across them.

    Returns:
        The SIMD width for the monoid `M[dtype, W]`, `simd_width=W` in
        `ContextParams`, and `tile_fn[ws]` calls. Always at least `1`.
    """
    comptime is_gpu_target = (target != StaticString("cpu"))
    # `simd_width_of` defaults to `_current_target()` (host = CPU),
    # giving wrong widths for a GPU-bound body. Resolve against the
    # actual launch target.
    comptime resolved_target = get_gpu_target() if is_gpu_target else _current_target()

    var w_out = 1
    comptime for T in Ts:
        comptime wi = simd_width_of[T, resolved_target]()
        if wi > w_out:
            w_out = wi

    comptime if not is_gpu_target:
        return w_out

    # GPU: shrink W to the largest width whose per-thread state still
    # fits the budget. Halving (not collapsing straight to 1) keeps
    # vector loads alive for heavy-state monoids on narrow dtypes — e.g.
    # Welford (12B) + bf16 would otherwise pick width 1: scalar `LDG`s
    # plus a division per element. `w_out` is a power of two, so this
    # lands on the largest fitting power of two.
    comptime state_at_W1 = size_of[M]()
    while w_out > 1 and state_at_W1 * w_out > PER_THREAD_BUDGET:
        w_out //= 2
    return w_out


# ===-----------------------------------------------------------------------===#
# `strided_load` — compile-time-stride side-input load (gamma/beta/cos/sin).
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def strided_load[
    dtype: DType,
    //,
    simd_width: SIMDLength,
    stride: Int,
    *,
    alignment: Int = align_of[Scalar[dtype]](),
    invariant: Bool = True,
](
    addr: UnsafePointer[mut=False, Scalar[dtype], ...],
    mask: SIMD[.bool, simd_width] = SIMD[.bool, simd_width](fill=True),
) -> SIMD[dtype, simd_width]:
    """Loads `simd_width` values from `addr` with a compile-time `stride`.

    Knowing the stride at compile time lets the two common cases avoid the
    `gather` used by the runtime-stride overload (which lowers to scattered
    per-element loads on NVIDIA GPUs): a contiguous `stride == 1` load becomes a
    single coalesced vector load, and a `stride == 0` broadcast becomes one
    scalar load splatted across lanes. Any other stride (or a partially-masked
    load) delegates to the runtime-stride overload.

    Body authors use this for side-input loads (gamma/beta/cos/sin) along the
    reduced axis: inner axis gives `stride=1`, a non-inner axis gives
    `stride=0` (one value per row, splatted across the tile).

    `invariant` defaults `True`: `addr` is always `mut=False` here, and every
    current call site loads a read-only weight/side input (never aliased by
    an output of the same kernel), so the load-invariant hint is safe by
    construction. Pass `invariant=False` explicitly if a future call site's
    memory isn't provably invariant for the kernel's duration.

    Parameters:
      dtype: DType of the loaded value.
      simd_width: The width of the SIMD vector.
      stride: The compile-time stride, in elements, between lanes.
      alignment: Alignment in bytes of the contiguous `stride == 1` load;
        defaults to the element alignment. Pass the SIMD-natural alignment when
        `addr` is known to be vector-aligned to fold the load into one access.
      invariant: Whether the memory is load invariant.

    Args:
      addr: The memory location to load data from.
      mask: A binary vector which prevents memory access to certain lanes of
        the result.

    Returns:
      A vector containing the loaded data.
    """
    comptime if simd_width == 1:
        return addr.load[invariant=invariant]() if mask else Scalar[dtype]()

    # Contiguous / broadcast fast paths. Both touch every lane, so they require
    # a fully-active mask; otherwise fall through to the masked gather. With the
    # default all-true mask the guard constant-folds away.
    comptime if not is_apple_gpu() and stride == 1:
        if Bool(mask.reduce_and()):
            return addr.load[
                width=simd_width, alignment=alignment, invariant=invariant
            ]()
    comptime if not is_apple_gpu() and stride == 0:
        if Bool(mask.reduce_and()):
            return SIMD[dtype, simd_width](addr.load[invariant=invariant]())

    # General case (Apple GPU, |stride| > 1, or a partially-masked load):
    # delegate to the runtime-stride overload — no logic duplicated.
    return _strided_load[simd_width, invariant=invariant](addr, stride, mask)


# ===-----------------------------------------------------------------------===#
# Top-level entry point: `launch`.
#
# One entry point, driven by `num_phases`, partitions row-wise bodies by
# output shape:
#
# - `num_phases == 1`: reduce-shaped — output collapses the reduced axis,
#   one value per row (e.g. `reduce_sum`, `reduce_max`, `arg_max`,
#   `row_mean_of_squares`). Body emits via `rowwise.emit`/`rowwise.once`. The
#   GPU backend may pick the tiled tier (one thread per output, SIMD-on-
#   outputs along the innermost non-axis dim) for a non-inner axis, and the
#   single-state split-K tier (many blocks per row, partials-buffer join)
#   when the row is too large for a single block.
#
# - `num_phases > 1`: normalize-shaped — output keeps the input shape,
#   per-element emission along the axis (e.g. `softmax`, `layer_norm`,
#   `rms_norm`, `rms_norm_rope`, `rms_norm_fused_residual_add`). Body writes
#   via `rowwise.elementwise` inside `rowwise.reduce` over the row. The
#   reduce-shaped single-launch split-K (partials-buffer join) does not
#   apply here — its last-block-only emission can't produce per-element
#   results (`_gpu_launch` gets `supports_splitk=False`). It has its own,
#   different split-K variant, driven by `num_phases` itself — a
#   phase-aware K+1-launch schedule, where `num_phases - 1` is the number
#   of dependent `row.reduce` phases before the final per-element write.
#
# `num_phases` counts phases uniformly across both shapes: the 8 pure
# reductions plus `row_mean_of_squares` pass `1` (the reduce's own result
# IS the final output — no separate normalize phase); `rms_norm` /
# `layer_norm` / `rms_norm_rope` / `rms_norm_fused_residual_add` /
# `rms_norm_fused_quantize_dynamic_scaled_fp8` pass `2` (one reduce, then a
# normalizing per-element write); `softmax` / `log_softmax` pass `3` (two
# dependent reduces — max, then sum(exp(x-max)) — then the normalizing
# write). Asserted `> 0`; a reduce-shaped body's `num_phases == 1` never
# reaches the phase-aware split-K branch (that branch is gated on
# `num_phases > 1`), so `associative` (reduce-shaped's CPU-only accumulator
# knob) and `dtype_size` (normalize-shaped's split-K byte gate) are each
# meaningless — and simply left at their defaults — for the other shape.
# (`BLOCK_SIZE` / `COOPERATIVE_BLOCK_SIZE` / `WARP_BLOCK_WARPS` are internal
# GPU launch-geometry constants, not part of the public signature — no caller
# anywhere in the repo has ever overridden them. `supports_tiled` is likewise
# hardcoded per shape: every real reduction body, `row_mean_of_squares`
# included, supports the tiled tier.)
#
# GPU tier-dispatch decision tree — which tier actually runs, and why.
# A body author never picks a tier directly; the launch heuristic below
# (in `gpu/rowwise.mojo`) picks one from `axis`, the body's capability
# flags (`supports_splitk` / `num_phases`), and the runtime shape
# (row_size = elements per row, num_rows = rows, both vs. the device's
# SM count):
#
#   axis == rank - 1 (inner / contiguous reduce axis)?
#     |
#     +-- NO (non-inner axis) --------------------------------------------
#     |     |
#     |     +-- simd_width > 1 AND row/col counts are SIMD-aligned AND
#     |     |   ((short reduce axis AND enough rows for the per-SM
#     |     |   floor) OR enough rows to saturate the device)
#     |     |                                   --> TILED tier
#     |     |   (one thread per output row; each thread's `w` SIMD lanes
#     |     |   are `w` adjacent output columns, not partials of one row
#     |     |   — the body's terminal decides what those `w` lanes mean:
#     |     |   `emit` collapses each to one value, `elementwise` re-walks
#     |     |   each of the `w` rows per-element)
#     |     |
#     |     +-- simd_width > 1 AND short reduce axis AND enough rows
#     |     |   for the same floor, SIMD-misaligned
#     |     |                                   --> TILED tier, scalar W=1
#     |     |
#     |     +-- otherwise            --------------------------------------
#     |                                         --> BLOCK (cooperative) tier
#     |         (one block per output row; threads cooperate on the
#     |         strided reduce axis, simd_width=1)
#     |
#     +-- YES (inner axis) ------------------------------------------------
#           |
#           +-- num_phases > 1 (a normalize-shaped body with N - 1
#           |   sequential `row.reduce` phases before its final
#           |   per-element write — softmax's max-then-sum sets N=3)
#           |   AND the grid is under-occupied (few rows, big rows)
#           |                                   --> SPLIT-K (pointwise) tier
#           |   (K+1 launches on one stream; every block sees the
#           |   joined stat from the prior phase before the final write)
#           |
#           +-- row_size fits one warp's reach (<= WARP_SIZE * simd_width
#           |   * the warp-tier chunk cap)       --> WARP tier
#           |   (one warp per row; grid-strides the row in chunks;
#           |   scalar SIMD on a non-`simd_width`-divisible row)
#           |
#           +-- supports_splitk AND simd_width small enough AND the grid
#           |   is under-occupied
#           |                                   --> SPLIT-K (single-state) tier
#           |   (the row splits across many blocks; the last-arriving
#           |   block — tracked by a per-row atomic counter — joins the
#           |   partials and emits; reduce-shaped bodies only, since every
#           |   other block's result is otherwise discarded — normalize-
#           |   shaped bodies set `supports_splitk=False`)
#           |
#           +-- otherwise            --------------------------------------
#                                                --> BLOCK tier
#               (one block per row; a 2D SIMD-width heuristic picks the
#               block size from row_size / num_rows / SM count)
#
# What actually drives the choice (the only things a body controls):
#   - `axis` (comptime): inner vs. non-inner selects the two halves above.
#   - `supports_splitk`: does this body carry exactly one monoid state
#     through a single `reduce` call? A body that reduces multiple
#     states in one call (mean + M2 fused, a dual sum-of-squares) must
#     say no until the split-K partials buffer can hold N states per
#     slot instead of one. Reduce-shaped (`num_phases == 1`) bodies say
#     yes; normalize-shaped bodies always say no (they have their own
#     split-K variant instead).
#   - `num_phases`: for normalize-shaped bodies — `num_phases - 1` is the
#     number of dependent `row.reduce` calls before the final per-element
#     write. `<= 1` disables this tier entirely (comptime-dead,
#     byte-identical codegen to before it existed).
#   - the runtime shape: row_size and num_rows are not a body knob, but
#     they are what every threshold above actually tests (a large,
#     under-occupied row earns split-K; a small row stays on warp/block).
#
# `ReduceTier` (`rowwise_types.mojo`) defines a fourth value, `Serial`,
# used by the non-inner tiled tier's scalar (`emit_tile_width == 1`)
# fallback: one thread per output, no cross-thread join. CPU ignores
# `_tier` altogether (always the `ReduceTier.Block` default) and picks
# its own split-axis-vs-cooperative tiering independently in
# `cpu/rowwise.mojo`.
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# What a backend actually has to provide.
#
# There is no trait tying `algorithm.cpu.rowwise` and `algorithm.gpu.rowwise`
# together — a real one was explored (item 90691-19) and abandoned: it de-risks
# the Mojo mechanism (a functor + an associated-type trait can dispatch a
# closure-carrying reduce phase through one generic call site) but doesn't
# remove the `comptime if is_cpu[...]` at the call site, which was the actual
# point, so it wasn't worth the added indirection. This section is the
# fallback: the contract written down in plain language, verified against
# what the two backend modules actually export today, so "does my new backend
# plug in" has a checklist even without the type system enforcing it.
#
# Each backend module must export these 5 functions, under these exact names
# (this file imports them aliased `_cpu_X` / `_gpu_X` — see the imports above):
#
#   reduce[params: ContextParams, rank: Int, //, TileFn: ...](
#       row_coords: RowCoord[rank], axis_size: Int, ctx: Context[params],
#       tile_fn: TileFn)
#       -- stateless tile walk: calls `tile_fn[ws](coords)` once per tile
#       on the target's tiers. No monoid; used by `elementwise`/`emit`-only
#       bodies.
#
#   reduce[State: ReduceOp, params: ContextParams, rank: Int, //,
#          TileFn: ...](
#       row_coords: RowCoord[rank], axis_size: Int, mut ctx: Context[params],
#       mut state: State, tile_fn: TileFn)
#       -- state-carrying tile walk: calls `tile_fn[ws](state, coords)`,
#       folding into `state` in place (a `mut` argument, not a capture — see
#       the module docstring above on why). This is the overload every
#       reduction body actually goes through.
#
#   pjoin[State: ReduceOp, params: ContextParams, //](
#       mut state: State, mut ctx: Context[params])
#       -- cross-thread combine for one monoid state, called once per state
#       after `reduce` returns. What "cross-thread" means is entirely
#       backend-defined: CPU has no threads to combine within one worker (a
#       single-participant `SerialReducer` still drives the monoid's own
#       `pjoin`/finalization body); GPU does a real warp/block shuffle-reduce
#       or a cross-block atomic finish depending on tier. Both backends'
#       per-tier combiners (`SerialReducer`, `BlockReducer`, `WarpReducer`)
#       conform to the pre-existing `Reducer` trait from
#       `algorithm.reduce_op` — the one piece of this contract the type
#       system genuinely checks today.
#
#   once[Emit: ... (def() -> None), params: ContextParams, //](
#       emit: Emit, ctx: Context[params])
#       -- runs `emit` on whichever thread(s) are the "canonical writer" for
#       this tier: every worker on CPU's cooperative/tiled tiers (one worker
#       already owns one output) or the last-arriving worker on its
#       split-axis tier; lane/thread 0 on GPU's warp/block tiers, every
#       thread on GPU's tiled/serial tiers. The body never knows which case
#       applies — it just calls `once` and trusts the backend to run `emit`
#       the right number of times.
#
#   launch[Body: RowBody, //, axis: Int, simd_width: Int, ...](
#       body: Body, shape: Coord, ...) raises
#       -- the actual entry point: picks a tier from `axis`/shape/the body's
#       capability flags, then drives `body[params](row_coords, ctx)` once
#       per row (or output, or tile) on that tier. This is the one function
#       whose signature is allowed to diverge completely between backends —
#       see the "why two entry points" note above for why GPU's tuning
#       knobs (`BLOCK_SIZE`, ...) have no CPU equivalent at all, and why
#       that's fine.
#
# Implicit in all five: every argument/return type that crosses one of these
# calls (`Coord`, `Context[params]`, `State: ReduceOp`, `TileFn`, `Emit`,
# `Body: RowBody`) must be `ImplicitlyCopyable & RegisterPassable` (or
# conform to `RowBody`/`ReduceOp` directly) — a body-authored value closure,
# not a `capturing` comptime parameter, because the GPU backend's `launch`
# serializes `body`'s copy-captured state into a kernel struct field to
# survive the host/device launch boundary (see the comment above
# `_BlockKernel` in `gpu/rowwise.mojo`); a `capturing` closure held as a
# comptime param does not survive that boundary. Nothing enforces this for a
# new backend beyond "the compiler will reject a call site whose types don't
# line up" — there is no single trait a backend author conforms to and gets
# checked against before writing the first line of `reduce`.
# ===-----------------------------------------------------------------------===#


def launch[
    Body: RowBody,
    //,
    axis: Int,
    simd_width: Int,
    target: StaticString,
    num_phases: Int,
    computationally_expensive: Bool = False,
    associative: Bool = False,
    dtype_size: Int = 0,
](body: Body, shape: Coord, ctx: Optional[DeviceContext] = None) raises:
    """Top-level scaffolder. `num_phases` picks reduce-shaped
    (`num_phases == 1`, output collapses the reduced axis, terminal
    `emit`) vs. normalize-shaped (`num_phases > 1`, output keeps the
    input shape, terminal `elementwise`) — see the module note above for
    the full picture, including the GPU tier-dispatch decision tree.

    Parameters:
        axis: Axis being reduced.
        simd_width: SIMD width for tile dispatch. Bodies compute this
            via `rowwise.pick_simd_width[...]` and pass it directly.
        target: `"cpu"` or `"gpu"` (anything non-CPU routes through the
            GPU backend).
        num_phases: Total phase count — `1` for reduce-shaped bodies
            (the reduce's own result is the final output); `> 1` for
            normalize-shaped bodies (`num_phases - 1` dependent
            `row.reduce` phases before a final per-element write).
            Asserted `> 0`.
        computationally_expensive: GPU author hint — flips the block tier's
            SIMD-full threshold for compute-heavy bodies. CPU ignores.
        associative: Reduce-shaped (`num_phases == 1`) only, CPU-only. Set
            `True` to widen the accumulator on long rows so the running
            total doesn't serialize into one add-per-element chain
            (recovers memory bandwidth). Only correct when reordering the
            accumulation doesn't change the result — additive reductions
            like `reduce_sum` / `reduce_mean`. Leave `False` for anything
            where element order matters (`product`'s narrower-dtype
            rounding, tie-breaking in `arg_max` / `arg_min`, ...). Ignored
            on GPU and meaningless for normalize-shaped bodies.
        dtype_size: Normalize-shaped (`num_phases > 1`) only, GPU only.
            Byte size of the body's primary dtype (`size_of[dtype]()`),
            used only by the phase-aware split-K byte gate; `0` (the
            default) leaves that tier's runtime condition permanently
            false, so the extra codegen it introduces is unreachable
            (softmax / log-softmax pass their dtype's size to actually
            engage it). Meaningless for reduce-shaped bodies.

    Args:
        body: The per-row computation. Receives a `Context` and the
            row's `Coord`; uses `reduce` / `pjoin` / `once` / `simd`
            to compose the algorithm.
        shape: Tensor shape.
        ctx: Optional `DeviceContext`; required for `target="gpu"`,
            unused on CPU.

    Raises:
        On launch or worker failure.
    """
    comptime assert num_phases > 0, "num_phases must be > 0"

    comptime if num_phases == 1:
        # Reduce-shaped: output collapses the reduced axis, one value per
        # row. Terminal is `emit`.
        comptime if is_cpu[target]():
            _cpu_launch[
                axis=axis,
                simd_width=simd_width,
                supports_tiled=True,
                associative=associative,
            ](body, shape)
        else:
            comptime assert is_gpu[target](), "unknown rowwise target"
            # GPU launch geometry: no caller has ever needed a different
            # block/warp size than these, so they are internal to this
            # function rather than forwarded from the public signature.
            comptime BLOCK_SIZE = 256
            comptime COOPERATIVE_BLOCK_SIZE = 128
            comptime WARP_BLOCK_WARPS = 4
            _gpu_launch[
                axis=axis,
                simd_width=simd_width,
                supports_splitk=True,
                computationally_expensive=computationally_expensive,
                BLOCK_SIZE=BLOCK_SIZE,
                COOPERATIVE_BLOCK_SIZE=COOPERATIVE_BLOCK_SIZE,
                WARP_BLOCK_WARPS=WARP_BLOCK_WARPS,
            ](body, shape, ctx.value())
    else:
        # Normalize-shaped: output keeps the input shape, per-element
        # emission along the axis. Terminal is `elementwise`.
        comptime if is_cpu[target]():
            _cpu_launch[
                axis=axis,
                simd_width=simd_width,
                supports_tiled=False,
            ](body, shape)
        else:
            comptime assert is_gpu[target](), "unknown rowwise target"
            # GPU launch geometry: no caller has ever needed a different
            # block/warp size than these, so they are internal to this
            # function rather than forwarded from the public signature.
            comptime BLOCK_SIZE = 256
            comptime COOPERATIVE_BLOCK_SIZE = 128
            comptime WARP_BLOCK_WARPS = 4
            _gpu_launch[
                axis=axis,
                simd_width=simd_width,
                supports_splitk=False,
                computationally_expensive=computationally_expensive,
                BLOCK_SIZE=BLOCK_SIZE,
                COOPERATIVE_BLOCK_SIZE=COOPERATIVE_BLOCK_SIZE,
                WARP_BLOCK_WARPS=WARP_BLOCK_WARPS,
                num_phases=num_phases,
                dtype_size=dtype_size,
            ](body, shape, ctx.value())


# ===-----------------------------------------------------------------------===#
# `RowCache` — a cached per-row value the body computes once and later
# phases consume. Produced by `Row.cache`; see there for rationale.
#
# Two backings, comptime-selected, invisible to the body:
#  - Registers (per participant): each participant's own tiles live in `_owned`,
#    read back by the cached overloads of `Row.reduce` / `Row.elementwise` at
#    the participant's own columns — cross-phase reuse (e.g. the fused
#    residual-add intermediate).
#  - Shared memory (opt in, `shared=True`): the row is *also* published to a
#    per-row shmem strip so any column is readable via `load` from a *different*
#    participant — cross-participant reuse (e.g. rope's rotate-half partner).
#
# Off a cache-eligible tier (`fuse=False`: CPU, dynamic/odd cols, oversized
# rows) both backings are empty and every access falls back to `recompute`
# (re-run `compute` on a freshly loaded input tile) — always correct, no traffic
# saved. `compute` may side-effect (e.g. emit a secondary output); on the
# fallback path it can run more than once per element, so its stores must be
# idempotent (the fused body re-emits the same residual value each time).
# ===-----------------------------------------------------------------------===#


# Named shared-memory global for the staged row. `cache` and `load`
# both `stack_allocation` under this name so they resolve to the *same* per-row
# strip without threading a pointer through the handle (a raw
# `unsafe_from_address` pointer would trip the non-null comptime check when the
# handle's `shared` branch is comptime-dead). One shared `cache` per body.
comptime _STAGED_SHMEM_NAME: StaticString = "rowwise_materialized_row"


struct RowCache[
    params: ContextParams,
    T: DType,
    dtype: DType,
    axis: Int,
    rank: Int,
    NCH: Int,
    W: Int,
    cols: Int,
    fuse: Bool,
    shared: Bool,
](ImplicitlyCopyable, Movable):
    """A row value cached once by `Row.cache` and read
    by later phases. See the module note above for the two backings.

    Carries no closures of its own: `recompute` / `load` take the
    owning view's `input_fn` / `compute` as fresh value-closure ARGUMENTS
    (a "forwarding bridge") rather than storing them as typed fields. A
    field whose type is a trait-bound closure referencing a sibling
    struct parameter (`dtype`/`T`) can't be named generically from
    another struct's method signature in current Mojo (`over:
    RowCache[..., InputFn, Compute]` fails to elaborate with "lacking
    evidence to prove correctness" — a struct-level, not method-level,
    limitation); passing the closures as per-call arguments instead
    keeps `RowCache`'s own type free of them entirely.

    Parameters:
        params: The owning view's comptime tier parameters.
        T: The cached value's dtype.
        dtype: The primary input dtype (for the recompute fallback).
        axis: Reduced axis.
        rank: Tensor rank.
        NCH: Per-participant chunk count (mirrors `Row._NCH`).
        W: SIMD width (mirrors `Row._W`).
        cols: Reduced-axis length (mirrors `Row._cols`).
        fuse: Whether the owning view is on a cache-eligible tier.
        shared: Whether the row is also published to shared memory.
    """

    var _owned: Array[SIMD[Self.T, Self.W], Self.NCH]
    """This participant's cached tiles (valid when `fuse`)."""

    var _shmem_addr: Int
    """Base address of the per-row shmem strip that `cache` wrote
    (valid when `fuse and shared`); `load` rebuilds a SHARED pointer from
    it. Carrying the address in the handle guarantees `load` reads the
    exact buffer `cache` allocated."""

    var row_coord: RowCoord[Self.rank]
    """The row's coords (reduced axis pinned to its base)."""

    @always_inline
    def __init__(out self, *, copy: Self):
        """Explicit copy constructor: `Array` lost `ImplicitlyCopyable`
        conformance, so `_owned` can't be auto-derived and needs `.copy()`.

        Args:
            copy: The handle to copy.
        """
        self._owned = copy._owned.copy()
        self._shmem_addr = copy._shmem_addr
        self.row_coord = copy.row_coord

    @always_inline
    def __init__(out self, row_coord: RowCoord[Self.rank]):
        """Builds an empty handle; `cache` fills the backing.

        Args:
            row_coord: The row's coords.
        """
        self.row_coord = row_coord
        self._owned = Array[SIMD[Self.T, Self.W], Self.NCH](
            fill=SIMD[Self.T, Self.W](0)
        )
        # Seed from a runtime value (never a literal): on `shared=False`
        # handles `load`'s shmem branch is comptime-dead but still
        # type-checked, and a provably-constant address would fold into
        # `UnsafePointer(unsafe_from_address=...)`'s IntLiteral overload,
        # whose comptime null check rejects it. `cache` overwrites this
        # with the real base when `shared`.
        self._shmem_addr = Int(row_coord.coord[0].value())

    @always_inline
    def recompute[
        w: Int,
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
        Compute: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                w: Int
            ](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> SIMD[Self.T, w]
        ),
    ](
        self,
        coord: RowCoord[Self.rank],
        input_fn: InputFn,
        compute: Compute,
    ) -> SIMD[Self.T, w]:
        """Recomputes the value at `coord` from a fresh global load —
        the fallback when the row was not staged.

        `w` is given explicitly (`recompute[w](...)`); `InputFn` /
        `Compute` are inferred from `input_fn` / `compute`.

        Parameters:
            w: SIMD width.
            InputFn: The value-closure type of `input_fn`.
            Compute: The value-closure type of `compute`.

        Args:
            coord: The element coord.
            input_fn: The owning view's primary-input loader, as a value.
            compute: The per-element producer, as a value.

        Returns:
            `compute(input_fn(coord), coord)`.
        """
        comptime al = tile_alignment[Self.dtype, w, Self.params.target]()
        return compute[w](input_fn[w, al](coord), coord)

    @always_inline
    def load[
        w: Int,
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
        Compute: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                w: Int
            ](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> SIMD[Self.T, w]
        ),
    ](
        self,
        coord: RowCoord[Self.rank],
        input_fn: InputFn,
        compute: Compute,
    ) -> SIMD[Self.T, w]:
        """Reads the cached value at any column `coord` — from the
        per-row shmem strip when staged (cross-participant), else the
        recompute fallback.

        `w` is given explicitly (`load[w](...)`); `InputFn` / `Compute`
        are inferred from `input_fn` / `compute`.

        Parameters:
            w: SIMD width.
            InputFn: The value-closure type of `input_fn`.
            Compute: The value-closure type of `compute`.

        Args:
            coord: The element coord (any column of this row).
            input_fn: The owning view's primary-input loader, as a value
                (fallback path only).
            compute: The per-element producer, as a value (fallback
                path only).

        Returns:
            The cached value at `coord`.
        """
        comptime if Self.fuse and Self.shared:
            comptime al = align_of[SIMD[Self.T, Self.W]]()
            var warp_off = (
                Int(warp_id()) * Self.cols
            ) if Self.params._tier == ReduceTier.Warp else 0
            var sh = UnsafePointer[
                Scalar[Self.T], MutUntrackedOrigin, address_space=.SHARED
            ](unsafe_from_address=self._shmem_addr)
            return sh.load[width=w, alignment=al](
                warp_off + Int(coord.coord[Self.axis].value())
            )
        else:
            return self.recompute[w](coord, input_fn, compute)


# ===-----------------------------------------------------------------------===#
# Free-form row-wise body layer: `Row`
#
# The body authors a row mirroring the GC IR: N `row.reduce` phases (each names
# a `ReduceOp` + a free-form contribution closure), ordinary Mojo scalar math on
# the joined results in between, then a terminal — `row.elementwise`
# (per-element output: normalizing reductions) or `row.emit` (one value per
# row: true reductions).
#
# The scaffolder owns the per-row register cache. On a cache-eligible tier
# (`is_cached`, GPU block tier, inner axis, statically-known SIMD-aligned
# cols) it stages the row's input strip once into registers and replays
# it across every phase + the elementwise terminal — so an N-phase reduction
# loads each element once, not N+1 times. Otherwise it streams (re-loads):
# always correct, just slower. The body is identical either way and never
# branches on tier/target.
#
# Single primary (axis-walked) input for now; side inputs (gamma/beta/cos/sin,
# typically broadcast) load on demand inside the body's closures via ordinary
# captures, as the GC IR does (`iter.load %weight[...]`). The heterogeneous
# multi-primary-input tile tuple is a forward-compatible extension (the
# closures already receive the full coord).
# ===-----------------------------------------------------------------------===#


struct Row[
    params: ContextParams,
    accum: DType,
    dtype: DType,
    axis: Int,
    rank: Int,
    AxisSize: CoordLike,
    is_cached: Bool,
](Copyable, Movable):
    """The body's handle to one logical row. See the module-level note.

    Carries no `input_fn` field: it's threaded as a fresh value-closure
    ARGUMENT to every method that needs it (a "forwarding bridge"),
    rather than stored as a typed field. A field whose type is a
    trait-bound closure that references a sibling struct parameter
    (`dtype`) hits a Mojo limitation the moment the struct itself is
    named generically elsewhere (e.g. as another method's parameter or
    return type) — "lacking evidence to prove correctness" even when
    the constraint holds by construction. Method-level trait-bound
    params (fresh per call, not part of `Self`'s persistent identity)
    don't hit this; every other closure here (`Contribute`, `G`,
    `Write`, ...) already uses that shape.

    Parameters:
        params: Comptime tier parameters (target, tier, widths).
        accum: Accumulator dtype the reduce closures compute in.
        dtype: The primary (axis-walked) input dtype.
        axis: Reduced axis (statically known).
        rank: Tensor rank.
        AxisSize: The reduced-axis length's `CoordLike` type —
            `ComptimeInt[N]` when known at comptime (enables the
            register-resident `_fuse` path) or a dynamic `Scalar` type
            otherwise. One type carries both the comptime-known-ness
            and (via `.value()`) the runtime length, so the caller
            supplies a single `AxisSize`-typed value instead of a
            separate comptime "is it static" signal.
        is_cached: Whether this view may cache the row — true for
            normalizing multi-phase bodies; false for single-pass true
            reductions where caching buys nothing.
    """

    comptime _W = Self.params.simd_width
    # Participant stride: one warp (WARP_SIZE lanes) per row on the warp
    # tier, one block (BLOCK_SIZE threads) per row on the block tier.
    # Participant owns elements [pid*_W + chunk*_PSTRIDE ...] over `_chunks`.
    comptime _PSTRIDE = (
        WARP_SIZE if Self.params._tier
        == ReduceTier.Warp else Self.params.BLOCK_SIZE
    ) * Self._W
    comptime _cols = Self.AxisSize.static_value if Self.AxisSize.is_static_value else 0
    # Per-participant chunk count if we staged the row strip.
    comptime _chunks = ceildiv(Self._cols, Self._PSTRIDE)
    # Register-residency budget: only keep registers when the staged strip fits
    # a small per-thread footprint. Beyond it (very large cols) the cache
    # would spill, so fall back to streaming (always correct). Scaffolder-
    # internal; the body never sees it. Independent of `_W`: at a fixed
    # `_cols`, `_chunks * size_of[SIMD[dtype, _W]]` is the same total byte
    # count regardless of width (`_PSTRIDE` scales with `_W` too), so this
    # budget caps register pressure the same way whether the tier vectorized
    # or not.
    comptime _FUSE_BYTE_BUDGET = 256
    comptime _fuse = (
        Self.is_cached
        and not is_cpu[Self.params.target]()
        and Self.params._tier != ReduceTier.Splitk
        and Self.axis == Self.rank - 1
        and Self._cols > 0
        and Self._cols % Self._W == 0
        and Self._chunks * size_of[SIMD[Self.dtype, Self._W]]()
        <= Self._FUSE_BYTE_BUDGET
    )
    comptime _NCH = Self._chunks if Self._fuse else 1
    # Warps per block (warp tier packs this many rows per block). Shmem
    # staging (`cache[shared=True]`) allocates one row strip per warp
    # on the warp tier, one per block otherwise.
    comptime _WARPS_PER_BLOCK = Self.params.BLOCK_SIZE // WARP_SIZE
    comptime _shmem_elems = (
        Self._WARPS_PER_BLOCK if Self.params._tier == ReduceTier.Warp else 1
    ) * Self._cols

    var row_coord: RowCoord[Self.rank]
    """The row's coords (reduced axis pinned to its base)."""
    var axis_size: Self.AxisSize
    var ctx: Context[Self.params]
    var _row_regs: Array[SIMD[Self.dtype, Self._W], Self._NCH]
    # 0-based order index of the next `reduce` call, incremented per call.
    # Only read on the pointwise-split-K tier (to pick combine / partial /
    # no-op against `ctx._phase`); unused and DCE'd on every other tier.
    var _reduce_index: Int

    @always_inline
    def __init__[
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
    ](
        out self,
        row_coords: Coord,
        axis_size: Self.AxisSize,
        ctx: Context[Self.params],
        input_fn: InputFn,
    ):
        """Builds the view, staging the row strip into registers
        on cache-eligible tiers.

        Parameters:
            InputFn: The value-closure type of `input_fn`.

        Args:
            row_coords: The row's coords.
            axis_size: Length of the reduce axis, as an `AxisSize`-typed
                value (`ComptimeInt[N]()` when statically known, else a
                dynamic `Scalar`).
            ctx: The dispatch bundle.
            input_fn: Loads a tile of the primary input at a coord. Used
                only here (fuse-path staging); not retained as a field —
                later non-fuse-path methods take it again as an argument.
        """
        self.row_coord = row_coords
        self.axis_size = axis_size
        self.ctx = ctx
        self._row_regs = Array[SIMD[Self.dtype, Self._W], Self._NCH](
            fill=SIMD[Self.dtype, Self._W](0)
        )
        self._reduce_index = 0

        comptime if Self._fuse:
            var participant = Self._participant()
            var base = self.row_coord
            comptime al = ctx.element_alignment[Self.dtype, Self._W]()

            comptime for chunk in range(Self._NCH):
                var pos = participant * Self._W + chunk * Self._PSTRIDE
                if pos < Self._cols:
                    var idx = base.at_axis[Self.axis](pos)
                    self._row_regs[chunk] = input_fn[Self._W, al](idx)

    @staticmethod
    @always_inline
    def _participant() -> Int:
        # Row-local participant index (comptime-resolved): lane within the
        # warp (warp tier) or thread within the block (block tier).
        comptime if Self.params._tier == ReduceTier.Warp:
            return Int(lane_id())
        else:
            return Int(thread_idx.x)

    @staticmethod
    @always_inline
    def _index_vector[w: Int](pos: Int) -> SIMD[.int64, w]:
        # Per-lane axis positions for a tile starting at `pos`. Lane stride
        # is the tier discriminator (comptime): `1` when lanes are
        # consecutive axis positions (cooperative tiers), `0` when they are
        # independent output columns sharing one axis position (tiled tier).
        # Keeping this `iota`/splat choice (pure layout) in the scaffolder
        # leaves the monoid's `reduce` purely elementwise.
        comptime lane_stride = 0 if Self.params.emit_tile_width > 1 else 1
        return SIMD[.int64, w](pos) + iota[.int64, w]() * Int64(lane_stride)

    @always_inline
    def reduce[
        M: ReduceOp,
        Contribute: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                w: Int
            ](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> SIMD[Self.accum, w]
        ),
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
    ](mut self, contribute: Contribute, input_fn: InputFn) -> M:
        """Runs one reduction phase: folds `contribute(tile, idx)` into
        `M` across the row, cross-thread-joins, and returns the joined
        state. The body reads `M`'s result fields (e.g. `ReduceSum.acc[0]`,
        `Welford.mean[0]`, `ArgMin.best_idx`).

        `contribute` is a value closure (its copy-captured state rides the
        value); the tier loop's `reduce_tile` captures it directly.
        `input_fn` (the primary-input loader) is likewise a fresh value
        argument, used only on the non-fuse (streaming) path — see the
        module note above `Row` on why it isn't a stored field.

        Parameters:
            M: The monoid for this phase (instantiated at the view's
                SIMD width, e.g. `ReduceSum[accum, params.simd_width]`).
            Contribute: The value-closure type of `contribute`.
            InputFn: The value-closure type of `input_fn`.

        Args:
            contribute: Per-element math; maps the loaded primary tile +
                its coord to the accumulator-dtype contribution.
            input_fn: Loads a tile of the primary input at a coord
                (non-fuse path only).

        Returns:
            The joined `M`, broadcast to every participant.
        """
        comptime if Self.params._num_phases > 1:
            # Phase-aware split-K: this reduce's 0-based order index `i`
            # against the launch phase `p`. `i < p` combine the row's
            # stored partials into the global stat; `i == p` reduce this
            # block's chunk and store the partial (its return is unused
            # this launch); `i > p` no-op (default). Reuses the split-K
            # chunk-striding via `_gpu_reduce` (this tier sets
            # `_tier = ReduceTier.Splitk`), and the block reduce + store / combine
            # helpers. The launch boundary is the cross-block sync.
            var i = self._reduce_index
            self._reduce_index = i + 1
            var phase = Int(self.ctx._phase)
            if i < phase:
                return _pointwise_splitk_combine[M](self.ctx, i)
            if i > phase:
                return M()

            var partial = M()
            var ctx = self.ctx

            @always_inline
            def splitk_reduce_tile[
                ws: Int
            ](mut state: M, coords: RowCoord[Self.rank]) {
                var input_fn, var contribute, var ctx
            }:
                comptime al = ctx.element_alignment[Self.dtype, ws]()
                var pos = Int(coords.coord[Self.axis].value())
                state.accumulate[Self.accum, ws](
                    contribute[ws](input_fn[ws, al](coords), coords),
                    Self._index_vector[ws](pos),
                )

            _gpu_reduce(
                self.row_coord,
                Int(self.axis_size.value()),
                self.ctx,
                partial,
                splitk_reduce_tile,
            )
            _pointwise_splitk_store_partial(partial, self.ctx, i)
            return partial

        var state = M()

        comptime if Self._fuse:
            var participant = Self._participant()
            var base = self.row_coord

            comptime for chunk in range(Self._NCH):
                var pos = participant * Self._W + chunk * Self._PSTRIDE
                if pos < Self._cols:
                    var idx = base.at_axis[Self.axis](pos)
                    state.accumulate[Self.accum, Self._W](
                        contribute[Self._W](self._row_regs[chunk], idx),
                        Self._index_vector[Self._W](pos),
                    )
        else:
            var ctx = self.ctx

            @always_inline
            def reduce_tile[
                ws: Int
            ](mut state: M, coords: RowCoord[Self.rank]) {
                var input_fn, var contribute, var ctx
            }:
                comptime al = ctx.element_alignment[Self.dtype, ws]()
                var pos = Int(coords.coord[Self.axis].value())
                state.accumulate[Self.accum, ws](
                    contribute[ws](input_fn[ws, al](coords), coords),
                    Self._index_vector[ws](pos),
                )

            comptime if is_cpu[Self.params.target]():
                _cpu_reduce(
                    self.row_coord,
                    Int(self.axis_size.value()),
                    self.ctx,
                    state,
                    reduce_tile,
                )
            else:
                comptime assert is_gpu[
                    Self.params.target
                ](), "unknown rowwise target"
                _gpu_reduce(
                    self.row_coord,
                    Int(self.axis_size.value()),
                    self.ctx,
                    state,
                    reduce_tile,
                )

        pjoin(state, self.ctx)
        return state

    @always_inline
    def elementwise[
        G: ImplicitlyCopyable
        & RegisterPassable
        & (def[w: Int](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> None),
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
    ](mut self, g: G, input_fn: InputFn):
        """Per-element terminal (normalizing reductions): runs `g(tile,
        idx)` over the row; `g` computes and stores each output element.
        Reuses the register cache when present.

        `g` is a value closure (its copy-captured state rides the value);
        the tier loop's `emit_tile` captures it directly. `input_fn` is
        likewise a fresh value argument (non-fuse path only) — see the
        module note above `Row`.

        Parameters:
            G: The value-closure type of `g`.
            InputFn: The value-closure type of `input_fn`.

        Args:
            g: Per-element write; closes over the joined results and side
                inputs, and stores via the body's output closure.
            input_fn: Loads a tile of the primary input at a coord
                (non-fuse path only).
        """
        comptime if Self.params._num_phases > 1:
            # Split-K terminal: write only in the final phase (`p == N`,
            # every reduce already combined to its global stat; `N` is the
            # number of dependent reduces, `Self.params._num_phases - 1`).
            # Each block writes only its own chunk via the split-K striding
            # of `_gpu_reduce`; the blocks partition the row exactly.
            # Earlier phases no-op.
            if Int(self.ctx._phase) == Self.params._num_phases - 1:
                var ctx = self.ctx

                @always_inline
                def splitk_emit_tile[
                    ws: Int
                ](coords: RowCoord[Self.rank]) {var input_fn, var g, var ctx}:
                    comptime al = ctx.element_alignment[Self.dtype, ws]()
                    g[ws](input_fn[ws, al](coords), coords)

                _gpu_reduce(
                    self.row_coord,
                    Int(self.axis_size.value()),
                    self.ctx,
                    splitk_emit_tile,
                )
            return

        comptime if Self._fuse:
            var participant = Self._participant()
            var base = self.row_coord

            comptime for chunk in range(Self._NCH):
                var pos = participant * Self._W + chunk * Self._PSTRIDE
                if pos < Self._cols:
                    var idx = base.at_axis[Self.axis](pos)
                    g[Self._W](
                        self._row_regs[chunk],
                        idx,
                    )
        else:
            var ctx = self.ctx

            @always_inline
            def emit_tile[
                ws: Int
            ](coords: RowCoord[Self.rank]) {var input_fn, var g, var ctx}:
                comptime al = ctx.element_alignment[Self.dtype, ws]()
                g[ws](input_fn[ws, al](coords), coords)

            comptime if is_cpu[Self.params.target]():
                _cpu_reduce(
                    self.row_coord,
                    Int(self.axis_size.value()),
                    self.ctx,
                    emit_tile,
                )
            else:
                comptime assert is_gpu[
                    Self.params.target
                ](), "unknown rowwise target"
                _gpu_reduce(
                    self.row_coord,
                    Int(self.axis_size.value()),
                    self.ctx,
                    emit_tile,
                )

    @always_inline
    def emit[
        Write: ImplicitlyCopyable
        & RegisterPassable
        & (def(RowCoord[Self.rank]) -> None),
    ](mut self, write: Write):
        """Per-row terminal (true reductions): runs `write(out_coord)`
        once on the canonical writer for this row. `write` closes over the
        joined results and stores them, and is taken as a value arg (its
        copy-captured state rides the value) rather than a comptime
        `capturing` parameter.

        Unlike the per-element terminal (`elementwise`, plain or cached),
        `emit` needs no comptime-form + value-shim pair: `once` is an
        in-kernel thread predicate (not a launch boundary and not a
        per-element loop), so the `emit_once` wrapper (which pins the
        reduced axis and calls `write`) is itself a value closure passed
        straight to `once`, with no per-element env-hoist risk.

        Parameters:
            Write: The value-closure type of `write`.

        Args:
            write: Per-row terminal store, passed as a value. Stores the
                row's output(s) at the collapsed coordinate (reduced axis
                pinned to 0).
        """
        var row_coord = self.row_coord

        @always_inline
        def emit_once() {var write, var row_coord}:
            var oc = row_coord.at_axis[Self.axis](0)
            write(oc)

        once(emit_once, self.ctx)

    @staticmethod
    @always_inline
    def _row_sync():
        # Row-local barrier for shmem staging: warp-level on the warp tier
        # (one warp per row — a block `barrier()` would deadlock when the
        # last block's warps cover fewer rows than `WARPS_PER_BLOCK`),
        # block-wide on the block tier (one block per row).
        comptime if Self.params._tier == ReduceTier.Warp:
            syncwarp()
        else:
            barrier()

    @always_inline
    def cache[
        T: DType,
        Compute: ImplicitlyCopyable
        & RegisterPassable
        & (def[w: Int](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> SIMD[T, w]),
        shared: Bool = False,
    ](mut self, compute: Compute) -> RowCache[
        Self.params,
        T,
        Self.dtype,
        Self.axis,
        Self.rank,
        Self._NCH,
        Self._W,
        Self._cols,
        Self._fuse,
        shared,
    ]:
        """Caches a per-row value the later phases reuse: runs
        `compute(tile, idx)` once over the row and stages the result so
        the cached overloads of `reduce` / `elementwise` (and, with
        `shared=True`, `load` at any column) read it back instead of
        recomputing.

        On a cache-eligible tier this writes each participant's tiles into
        registers (and, when `shared`, publishes the row to a shmem strip
        with one row-local sync); otherwise the returned handle recomputes
        on access (always correct) — the caller re-supplies `input_fn` /
        `compute` to the cached `reduce` / `elementwise` overloads on that
        path, since `RowCache` doesn't carry them as fields (see its
        module note).
        `compute` runs exactly once per element on the fast path but may
        re-run on the fallback, so any side effect it performs (e.g. a
        residual emit) must be idempotent.

        Parameters:
            T: The cached value's dtype.
            Compute: The value-closure type of `compute`.
            shared: Publish the row to shmem for cross-participant `load`.

        Args:
            compute: Per-element producer from the primary tile + coord.

        Returns:
            A `RowCache` handle over the cached row.
        """
        var out = RowCache[
            Self.params,
            T,
            Self.dtype,
            Self.axis,
            Self.rank,
            Self._NCH,
            Self._W,
            Self._cols,
            Self._fuse,
            shared,
        ](self.row_coord)

        comptime if Self._fuse:
            var participant = Self._participant()
            var base = self.row_coord

            comptime if shared:
                var sh = stack_allocation[
                    Self._shmem_elems,
                    Scalar[T],
                    name=_STAGED_SHMEM_NAME,
                    alignment=align_of[SIMD[T, Self._W]](),
                    address_space=.SHARED,
                ]()
                var warp_off = (
                    Int(warp_id()) * Self._cols
                ) if Self.params._tier == ReduceTier.Warp else 0
                comptime al = align_of[SIMD[T, Self._W]]()
                # Protect the previous grid-stride row's readers before we
                # overwrite this warp/block's strip.
                Self._row_sync()

                comptime for chunk in range(Self._NCH):
                    var pos = participant * Self._W + chunk * Self._PSTRIDE
                    if pos < Self._cols:
                        var idx = base.at_axis[Self.axis](pos)
                        var v = compute[Self._W](
                            self._row_regs[chunk],
                            idx,
                        )
                        out._owned[chunk] = v
                        sh.store[alignment=al](warp_off + pos, v)
                out._shmem_addr = Int(sh)
                # Publish this row's writes to all participants.
                Self._row_sync()
            else:
                comptime for chunk in range(Self._NCH):
                    var pos = participant * Self._W + chunk * Self._PSTRIDE
                    if pos < Self._cols:
                        var idx = base.at_axis[Self.axis](pos)
                        out._owned[chunk] = compute[Self._W](
                            self._row_regs[chunk],
                            idx,
                        )
        return out

    @always_inline
    def reduce[
        T: DType,
        shared: Bool,
        //,
        M: ReduceOp,
        Contribute: ImplicitlyCopyable
        & RegisterPassable
        & (def[w: Int](SIMD[T, w], RowCoord[Self.rank]) -> SIMD[Self.accum, w]),
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
        Compute: ImplicitlyCopyable
        & RegisterPassable
        & (def[w: Int](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> SIMD[T, w]),
    ](
        mut self,
        over: RowCache[
            Self.params,
            T,
            Self.dtype,
            Self.axis,
            Self.rank,
            Self._NCH,
            Self._W,
            Self._cols,
            Self._fuse,
            shared,
        ],
        contribute: Contribute,
        input_fn: InputFn,
        compute: Compute,
    ) -> M:
        """Cached overload of `reduce`: folds `contribute` over a
        previously `cache`d row instead of the primary input — reads the
        staged value at each column (registers on the fast path, recompute
        fallback otherwise). Used for a phase that reduces a computed
        cross-phase value (e.g. sum of the fused intermediate squared).

        Resolved at compile time from plain Mojo function overloading —
        whether the call site passes `over` — not a runtime-checked
        `Optional`, so the cached-vs-plain choice costs nothing at
        runtime.

        `contribute` is a value closure (its copy-captured state rides the
        value); the tier loop's `reduce_tile` captures it directly.
        `input_fn` / `compute` (the same producer passed to the `cache`
        call that built `over`) are likewise fresh value arguments, used
        only on the non-fuse (recompute) path — `RowCache` doesn't carry
        them as fields (see its module note). `axis_size` / `ctx` are read
        from `self` rather than threaded as separate arguments — the
        owning view already holds both.

        Parameters:
            T: The staged value's dtype (inferred from `over`).
            shared: The staged handle's shmem flag (inferred).
            M: The monoid for this phase.
            Contribute: The value-closure type of `contribute`.
            InputFn: The value-closure type of `input_fn`.
            Compute: The value-closure type of `compute`.

        Args:
            over: The cached row to reduce.
            contribute: Per-element math over the staged tile, as a value.
            input_fn: Loads a tile of the primary input at a coord
                (non-fuse path only).
            compute: The producer that built `over` (non-fuse path only).

        Returns:
            The joined `M`, broadcast to every participant.
        """
        var state = M()

        comptime if Self._fuse:
            var participant = Self._participant()
            var base = over.row_coord

            comptime for chunk in range(Self._NCH):
                var pos = participant * Self._W + chunk * Self._PSTRIDE
                if pos < Self._cols:
                    var idx = base.at_axis[Self.axis](pos)
                    state.accumulate[Self.accum, Self._W](
                        contribute[Self._W](over._owned[chunk], idx),
                        Self._index_vector[Self._W](pos),
                    )
        else:
            var over_ = over
            var axis_size = Int(self.axis_size.value())

            @always_inline
            def reduce_tile[
                ws: Int
            ](mut state: M, coords: RowCoord[Self.rank]) {
                var over_,
                var contribute,
                var input_fn,
                var compute,
            }:
                var pos = Int(coords.coord[Self.axis].value())
                state.accumulate[Self.accum, ws](
                    contribute[ws](
                        over_.recompute[ws](coords, input_fn, compute),
                        coords,
                    ),
                    Self._index_vector[ws](pos),
                )

            comptime if is_cpu[Self.params.target]():
                _cpu_reduce(
                    over.row_coord,
                    axis_size,
                    self.ctx,
                    state,
                    reduce_tile,
                )
            else:
                comptime assert is_gpu[
                    Self.params.target
                ](), "unknown rowwise target"
                _gpu_reduce(
                    over.row_coord,
                    axis_size,
                    self.ctx,
                    state,
                    reduce_tile,
                )

        pjoin(state, self.ctx)
        return state

    @always_inline
    def elementwise[
        T: DType,
        shared: Bool,
        //,
        G: ImplicitlyCopyable
        & (def[w: Int](SIMD[T, w], RowCoord[Self.rank]) -> None),
        InputFn: ImplicitlyCopyable
        & RegisterPassable
        & (
            def[
                width: Int, alignment: Int
            ](RowCoord[Self.rank]) -> SIMD[Self.dtype, width]
        ),
        Compute: ImplicitlyCopyable
        & RegisterPassable
        & (def[w: Int](SIMD[Self.dtype, w], RowCoord[Self.rank]) -> SIMD[T, w]),
    ](
        mut self,
        over: RowCache[
            Self.params,
            T,
            Self.dtype,
            Self.axis,
            Self.rank,
            Self._NCH,
            Self._W,
            Self._cols,
            Self._fuse,
            shared,
        ],
        g: G,
        input_fn: InputFn,
        compute: Compute,
    ):
        """Cached overload of `elementwise`: the per-element terminal `g`
        receives the staged value at each column (from `over`) instead of
        the primary input — registers on the fast path, recompute
        fallback otherwise. `g` may additionally read other columns of
        `over` via `over.load` (e.g. rope's rotate-half partner, staged
        in shmem).

        Resolved at compile time from plain Mojo function overloading —
        whether the call site passes `over` — not a runtime-checked
        `Optional`, so the cached-vs-plain choice costs nothing at
        runtime.

        `g` is a value closure (its copy-captured state rides the value);
        the tier loop's `emit_tile` captures it directly. `input_fn` /
        `compute` (the same producer passed to the `cache` call that
        built `over`) are likewise fresh value arguments, used only on
        the non-fuse (recompute) path — `RowCache` doesn't carry them as
        fields (see its module note). `axis_size` / `ctx` are read from
        `self` rather than threaded as separate arguments — the owning
        view already holds both.

        The bound on `G` is `ImplicitlyCopyable` (not `RegisterPassable`
        like the other value overloads) because `g` may copy-capture
        non-register-passable state (e.g. `over` itself, for a terminal
        that also reads a different column via `over.load`); this overload
        runs inside the launched kernel (not across a launch boundary),
        so register-passability is not required.

        Parameters:
            T: The staged value's dtype (inferred from `over`).
            shared: The staged handle's shmem flag (inferred).
            G: The value-closure type of `g`.
            InputFn: The value-closure type of `input_fn`.
            Compute: The value-closure type of `compute`.

        Args:
            over: The cached row to walk.
            g: Per-element write over the staged tile, as a value.
            input_fn: Loads a tile of the primary input at a coord
                (non-fuse path only).
            compute: The producer that built `over` (non-fuse path only).
        """
        comptime if Self._fuse:
            var participant = Self._participant()
            var base = over.row_coord

            comptime for chunk in range(Self._NCH):
                var pos = participant * Self._W + chunk * Self._PSTRIDE
                if pos < Self._cols:
                    var idx = base.at_axis[Self.axis](pos)
                    g[Self._W](
                        over._owned[chunk],
                        idx,
                    )
        else:
            var over_ = over
            var axis_size = Int(self.axis_size.value())

            @always_inline
            def emit_tile[
                ws: Int
            ](coords: RowCoord[Self.rank]) {
                var over_,
                var g,
                var input_fn,
                var compute,
            }:
                g[ws](
                    over_.recompute[ws](coords, input_fn, compute),
                    coords,
                )

            comptime if is_cpu[Self.params.target]():
                _cpu_reduce(over.row_coord, axis_size, self.ctx, emit_tile)
            else:
                comptime assert is_gpu[
                    Self.params.target
                ](), "unknown rowwise target"
                _gpu_reduce(over.row_coord, axis_size, self.ctx, emit_tile)
