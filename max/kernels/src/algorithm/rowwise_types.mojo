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
"""Target-neutral `ContextParams` + `Context` types for the row-wise
reduction scaffolder. In its own module so the CPU and GPU backends
can both import it without an import cycle through the unified
`rowwise` dispatcher.

`ContextParams.target` (comptime `StaticString`) is the dispatcher's
switch: `"cpu"` selects the CPU backend, anything else the GPU backend.
The GPU-only runtime fields on `Context` (split-K plumbing) hold
sentinels on CPU and are never read — every access is behind a
`comptime if is_cpu[params.target]:` branch that dead-code-eliminates.

Member-naming convention: members a body reads have plain names
(`axis`, `emit_tile_width`, `simd_width`, `BLOCK_SIZE`, `target`,
`accumulator_width`); scaffolder-internal members — the tier
discriminators and split-K plumbing — are `_`-prefixed.
"""

from std.memory import UnsafePointer
from max.gpu.host.info import is_cpu
from std.utils.coord import Coord, DynamicCoord


# ===-----------------------------------------------------------------------===#
# `tile_alignment`
# ===-----------------------------------------------------------------------===#


@always_inline
def tile_alignment[dtype: DType, ws: Int, target: StaticString]() -> Int:
    """Alignment, **in elements**, a row-wise body's store — or its
    `input_fn`'s load — may assume for a width-`ws` tile of `dtype` on
    `target`.

    The unit is elements, not bytes: every caller forwards this value into
    the `element_alignment` parameter of a tensor load/store, which
    multiplies it by `align_of[dtype]()` to get the byte alignment it
    promises the backend. Returning a byte count here therefore squares
    the intended alignment, and an alignment the address does not keep is
    a fault rather than a missed optimization — a `float64` tile promised
    64 bytes loads through `vmovapd`.

    Single source of truth for the rule. `1` on CPU (element-natural: the
    CPU tiers walk a row at the tensor's own stride, and an input tensor
    may be an imported buffer, so neither a tile base nor the step between
    tiles is guaranteed SIMD-aligned). `ws` on GPU (SIMD-natural —
    `ws * align_of[dtype]()` bytes: folds the access into one vector
    transaction (`LDG.128`/`STG.128`), safe because a body only ever gets
    a `ws > 1` tile on a SIMD-aligned offset: the block, warp and per-row
    split-K tiers all gate a wide tile on `row_size % simd == 0`, and fall
    back to a narrower path when it fails). The two coincide at `ws == 1`.

    Both `Context.element_alignment` (the store) and the public-op
    wrappers' synthesized `input_fn` (the load) call this, so the rule
    lives in one place.

    Parameters:
        dtype: The tile's element dtype.
        ws: The tile's SIMD width.
        target: `"cpu"` or `"gpu"`.

    Returns:
        The alignment in elements.
    """
    comptime if is_cpu[target]():
        return 1
    else:
        return ws


# ===-----------------------------------------------------------------------===#
# `ContextParams`
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ReduceTier(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """GPU tier discriminator for the row-wise scaffolder. Mutually
    exclusive by construction (a single field, not independent bools):
    `Block` is the default (also every CPU tier, where only
    `emit_tile_width`/`simd_width` matter). Scaffolder-internal — a body
    never reads `Context._tier`.
    """

    var value: Int

    comptime Block = ReduceTier(0)
    """Block-per-row (or block-per-output) cooperative tier — the
    default. Also the tag used for every CPU tier."""

    comptime Warp = ReduceTier(1)
    """Warp-per-row: one warp covers the whole row (GPU only)."""

    comptime Serial = ReduceTier(2)
    """One thread per row: it walks the whole reduce axis and owns the
    output, with no cross-thread join. Used by the non-inner tiled
    tier's scalar fallback (GPU only)."""

    comptime Splitk = ReduceTier(3)
    """Multiple blocks cooperate on one row when `num_rows < sm_count`
    would otherwise leave SMs idle (GPU only)."""


struct ContextParams(TrivialRegisterPassable):
    """Comptime half of `Context`. Bundled into one template parameter
    so callers write `Context[params]` instead of threading eight
    separate comptime args.

    Tier discriminator (GPU): `_tier` (a `ReduceTier`) picks exactly one
    of warp / serial / split-K, or `Block` (also every CPU tier).
    `emit_tile_width > 1` further marks the tiled (SIMD-on-outputs) tier,
    independent of `_tier`. The same fields exist on CPU but only
    `emit_tile_width` and `simd_width` matter; `_tier` is always `Block`.
    `_tier` is a `_`-prefixed scaffolder internal (set by the tier
    picker) — a body never reads it.

    `target` (comptime) is the dispatcher's switch — `"cpu"` selects
    the CPU backend, anything else the GPU backend.
    """

    var axis: Int
    """Axis being reduced."""

    var emit_tile_width: Int
    """Rows-per-thread. `1` for warp/block/serial/split-K tiers, `> 1`
    for the tiled tier (SIMD-on-outputs)."""

    var BLOCK_SIZE: Int
    """Threads per block (kernel-launch shape). GPU-only; `1` on CPU."""

    var _tier: ReduceTier
    """Tier discriminator (GPU only; always `ReduceTier.Block` on CPU).
    Scaffolder-internal."""

    var simd_width: Int
    """SIMD width. On GPU, the block tier's axis-direction load width
    (ignored by tiled/warp/serial). On CPU, the width the scaffolder
    walks the axis with."""

    var target: StaticString
    """`"cpu"` or `"gpu"` — picks which backend the unified
    `rowwise.{reduce, pjoin, once, launch}` dispatches to."""

    var _num_phases: Int
    """Per-element-output split-K opt-in (GPU only). `<= 1` disables it —
    the whole tier is comptime-dead and codegen is byte-identical to a
    a plain normalize-shaped launch. `N > 1` enables the phase-aware K+1-launch
    split-K for a body with `N - 1` dependent `row.reduce` phases before
    its final per-element write (softmax / log-softmax: `N = 3`),
    splitting each row across many blocks to fix under-occupancy on
    few-rows-by-very-long-inner shapes. Scaffolder-internal; set only by
    the tier picker, never read by a body."""

    @always_inline
    def __init__(
        out self,
        axis: Int,
        emit_tile_width: Int,
        BLOCK_SIZE: Int,
        simd_width: Int,
        target: StaticString,
        tier: ReduceTier = ReduceTier.Block,
        num_phases: Int = 0,
    ):
        """Initializes a `ContextParams` from per-tier comptime values.

        Args:
            axis: Axis being reduced.
            emit_tile_width: Rows-per-thread (`1` for cooperative tiers).
            BLOCK_SIZE: Threads per block.
            simd_width: SIMD width for the block tier's axis load.
            target: `"cpu"` or `"gpu"`.
            tier: Tier discriminator (`ReduceTier.Block` by default —
                every CPU tier and the GPU tiled/cooperative tiers).
            num_phases: Per-element-output split-K total phase count
                (`<= 1` disables the tier).
        """
        debug_assert(
            not is_cpu(target) or BLOCK_SIZE == 1,
            "CPU backend requires BLOCK_SIZE == 1",
        )
        self.axis = axis
        self.emit_tile_width = emit_tile_width
        self.BLOCK_SIZE = BLOCK_SIZE
        self._tier = tier
        self.simd_width = simd_width
        self.target = target
        self._num_phases = num_phases


# ===-----------------------------------------------------------------------===#
# `Context`
# ===-----------------------------------------------------------------------===#


struct Context[params: ContextParams](TrivialRegisterPassable):
    """Per-block dispatch bundle. Comptime tier identity and widths
    flow through `params`; runtime per-block plumbing (split-K scratch
    + position) sits in `var` fields. The body forwards `ctx` to
    `rowwise.reduce` / `rowwise.once` and treats it as opaque, reading
    only the plainly-named members (`ctx.axis`,
    `ctx.accumulator_width`, ...). The `_`-prefixed members (`ctx._tier`,
    `ctx._partials_base`, ...) are scaffolder internals.

    On CPU the runtime fields hold sentinels via `Context.empty()` and
    are never read.

    Parameters:
        params: Comptime tier parameters (axis, BLOCK_SIZE, tier
            discriminator, SIMD widths, target).
    """

    comptime axis = Self.params.axis
    """Axis being reduced (mirrors `Self.params.axis`)."""

    comptime emit_tile_width = Self.params.emit_tile_width
    """Rows-per-thread (mirrors `Self.params.emit_tile_width`)."""

    comptime BLOCK_SIZE = Self.params.BLOCK_SIZE
    """Threads per block (mirrors `Self.params.BLOCK_SIZE`)."""

    comptime _tier = Self.params._tier
    """Tier discriminator (mirrors `Self.params._tier`)."""

    comptime simd_width = Self.params.simd_width
    """SIMD width (mirrors `Self.params.simd_width`)."""

    comptime target = Self.params.target
    """Backend target (mirrors `Self.params.target`)."""

    comptime _num_phases = Self.params._num_phases
    """Per-element-output split-K total phase count (mirrors
    `Self.params._num_phases`; `<= 1` = tier off)."""

    comptime accumulator_width = 1 if Self.params.emit_tile_width > 1 else Self.params.simd_width
    """Reduce-op accumulator width a body should use for its
    `M[dtype, accumulator_width]` state: `1` on the
    tiled tier (each output lane is a separate accumulator), else the
    full `simd_width`. Centralized here so every reduction picks the
    same width."""

    @staticmethod
    @always_inline
    def element_alignment[dtype: DType, ws: Int]() -> Int:
        """Store/load alignment for a width-`ws` tile of `dtype` on this
        `Context`'s target — element-natural on CPU, SIMD-natural on GPU.

        A `@staticmethod` so it's usable in a `comptime` initializer:
        the receiver in `ctx.element_alignment[dtype, ws]()` is only a
        type carrier (runtime value never read), which `comptime`
        allows. Delegates to the module-level `tile_alignment`, the
        single source of truth shared with the public-op wrappers'
        loads.

        Parameters:
            dtype: The tile's element dtype.
            ws: The tile's SIMD width.

        Returns:
            The alignment in elements -- matches `element_alignment` in
            `graph_compiler/extensibility/tensor_arg_traits.mojo` and
            `managed_tensor_slice.mojo`, which multiply this by
            `align_of[dtype]()` to get the byte alignment passed to the
            backend. Named to match that convention explicitly, since a
            caller that instead hands this straight to a raw byte-level
            primitive (e.g. `TileTensor.raw_load`/`raw_store`, which
            wants bytes) without that multiplication under-claims
            alignment -- silently harmless for 4-byte-and-wider dtypes,
            a real miscompilation risk for narrower ones.
        """
        return tile_alignment[dtype, ws, Self.target]()

    var _partials_base: UnsafePointer[UInt8, MutUntrackedOrigin]
    """Split-K partials scratch (GPU only; sentinel on CPU)."""

    var _counters_base: UnsafePointer[Int32, MutUntrackedOrigin]
    """Split-K per-row counters (GPU only; sentinel on CPU)."""

    var _blocks_per_row: Int32
    """Blocks cooperating on this row (GPU split-K only)."""

    var _block_in_row: Int32
    """This block's position in its row (GPU split-K only)."""

    var _row_idx: Int32
    """The row this block is reducing (GPU split-K only)."""

    var _is_last_block: Bool
    """Set by `rowwise.reduce`'s split-K branch after the atomic
    finish — True on the one block per row that brought the counter to
    `_blocks_per_row`."""

    var _phase: Int32
    """Per-element-output split-K launch phase, `0..N` (GPU only; `0`
    on every other tier). In phase `p` the `Row` combines reduce
    phases `< p`, runs the partial for reduce `p`, no-ops reduces
    `> p`, and writes the output only when `p == N`. Set per launch by
    `_PointwiseSplitkKernel` from the K+1-launch loop."""

    @always_inline
    def __init__(
        out self,
        partials_base: UnsafePointer[UInt8, MutUntrackedOrigin],
        counters_base: UnsafePointer[Int32, MutUntrackedOrigin],
        blocks_per_row: Int32,
        block_in_row: Int32,
        row_idx: Int32,
        is_last_block: Bool,
        phase: Int32 = Int32(0),
    ):
        """Initializes a `Context` with the per-block runtime values.

        Args:
            partials_base: Split-K partials scratch (or sentinel).
            counters_base: Split-K per-row counters (or sentinel).
            blocks_per_row: Blocks cooperating on this row (or 0).
            block_in_row: This block's position in its row (or 0).
            row_idx: The row this block is reducing (or 0).
            is_last_block: Initialized to False; set by
                `rowwise.reduce` after the atomic finish (GPU).
            phase: Per-element-output split-K launch phase (or 0).
        """
        self._partials_base = partials_base
        self._counters_base = counters_base
        self._blocks_per_row = blocks_per_row
        self._block_in_row = block_in_row
        self._row_idx = row_idx
        self._is_last_block = is_last_block
        self._phase = phase

    @staticmethod
    @always_inline
    def empty() -> Self:
        """Returns a `Context` with zeroed runtime fields. Used by CPU
        bodies and non-split-K GPU kernels.

        Returns:
            A `Context[params]` with sentinel pointers and zero counters.
        """
        return Self(
            partials_base=UnsafePointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=1
            ),
            counters_base=UnsafePointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=1
            ),
            blocks_per_row=Int32(0),
            block_in_row=Int32(0),
            row_idx=Int32(0),
            is_last_block=False,
        )


# ===-----------------------------------------------------------------------===#
# `RowBody` — value-closure bound for the per-row body.
# ===-----------------------------------------------------------------------===#

comptime RowBody = ImplicitlyCopyable & RegisterPassable & (
    def[_p: ContextParams](Coord, mut Context[_p]) -> None
)
"""Bound for the per-row body threaded through the launch as a VALUE.

The body is a value closure (not a comptime `capturing` parameter): the
GPU backend crosses it into the kernel as a struct field, so its
copy-captured state (the fused `input_fn` / `output_fn` value closures,
`axis_size`) is serialized across the launch. `ImplicitlyCopyable &
RegisterPassable` lets the kernel struct hold it as a register-passable
field; the callable trait is non-`capturing` (captures ride the value)."""


# TODO(MOCO-4713): Remove this and use a DynamicCoord with
# wrapper functions where needed
@fieldwise_init
struct RowCoord[rank: Int](
    ImplicitlyCopyable, Movable, TrivialRegisterPassable
):
    """A row's coordinates, as a non-variadic handle.

    Wraps an all-dynamic `Coord`: the reduce axis is overwritten per tile,
    which a statically typed element cannot hold.

    The wrapper is not cosmetic. A `Coord` parameterized by a symbolic rank
    expands to a variadic pack (`TypeList.splat`), and such a pack does not
    survive substitution into a closure-type trait bound.

    Parameters:
        rank: The coordinate's rank.
    """

    var coord: DynamicCoord[DType.int64, Self.rank]
    """The wrapped coordinates."""

    @always_inline
    @implicit
    def __init__(out self, row_coords: Coord):
        """Builds a handle from any `Coord`, dropping static dims.

        Args:
            row_coords: The row's coords, static dims permitted.
        """
        self.coord = rebind[DynamicCoord[DType.int64, Self.rank]](
            row_coords.make_dynamic[DType.int64]()
        )

    @always_inline
    def write_axis[axis: Int](mut self, pos: Int):
        """Sets the reduce axis to `pos`, in place.

        Parameters:
            axis: The reduce axis to overwrite.

        Args:
            pos: The column position along the reduce axis.
        """
        Pointer(to=self.coord[axis]).write(
            rebind[type_of(self.coord[axis])](Scalar[DType.int64](pos))
        )

    @always_inline
    def at_axis[axis: Int](self, pos: Int) -> Self:
        """A copy with the reduce axis set to `pos`.

        Parameters:
            axis: The reduce axis to overwrite.

        Args:
            pos: The column position along the reduce axis.

        Returns:
            A copy of `self` with element `axis` set to `pos`.
        """
        var result = self
        result.write_axis[axis](pos)
        return result


@always_inline
def _num_outputs_excluding_axis[axis: Int](shape: Coord) -> Int:
    """Product of `shape`'s dims other than `axis`.

    `total_size // axis_size` cannot supply that count when the reduce
    axis is empty (`axis_size == 0`). Not because it faults: Mojo's `//`
    inserts a zero-guard, so `0 // 0` yields `0` — indistinguishable from
    the `0` a shape with no rows at all produces, which is exactly why an
    empty axis used to read as "nothing to do". A reduce-shaped body still
    owns one output per row when the axis is empty (the monoid identity,
    from an axis walk of zero elements), so `launch` counts outputs
    without dividing by the (possibly empty) axis. Shared by the CPU and
    GPU backends (`cpu/rowwise.mojo`, `gpu/rowwise.mojo`); it lives here,
    rather than in `rowwise.mojo`, so both backends can import it without
    an import cycle through the unified dispatcher.
    """
    var num_outputs = 1
    comptime for i in range(shape.rank):
        if i != axis:
            num_outputs *= Int(shape[i].value())
    return num_outputs
