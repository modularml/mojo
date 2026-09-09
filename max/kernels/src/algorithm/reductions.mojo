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
"""Row-based row-wise reductions (free-form body layer).

Pure-reduction entry points (`reduce_sum` / `reduce_max` / `reduce_min` /
`reduce_product` / `reduce_mean` / `reduce_argmax` / `reduce_argmin` /
`reduce_min_and_max`) built on the `algorithm.rowwise` scaffolder and the
`algorithm.reduce_op` monoid library. Each authors one free-form body — a
single reduce phase plus a per-row `emit` — and runs on both CPU and GPU.
"""

from max.gpu.host import DeviceContext
from std.sys.info import has_apple_gpu_accelerator
from std.utils.coord import Coord

from algorithm import rowwise
from algorithm.rowwise_types import RowCoord
from algorithm.reduce_op import (
    ArgMax,
    ArgMin,
    MinMax,
    ReduceMax,
    ReduceMin,
    ReduceProduct,
    ReduceSum,
)


def reduce_sum[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    """Sums input values along `reduce_dim` via the Row free-form
    body layer: one `ReduceSum` phase plus a per-row `emit`.

    `input_fn` / `output_fn` are value-based closures (a type parameter
    bound to a callable trait + a value argument), not comptime
    `capturing[_]` parameters."""
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: sum via the ReduceSum monoid.
        @always_inline
        def add[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var acc = row.reduce[ReduceSum[dtype, params.simd_width]](add, load).acc

        # Emit: per-row output — the reduced scalar.
        @always_inline
        def write(oc: RowCoord[rank]) {var acc, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                acc.slice[params.emit_tile_width](),
            )

        # `acc`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=True,
    ](body, input_shape, context)


# ===----------------------------------------------------------------------=== #
# More Row-based reductions: max / min / product / mean. Same free-form
# body as reduce_sum; only the monoid (and the result field read) changes.
# ===----------------------------------------------------------------------=== #


def reduce_max[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceMax[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: max via the ReduceMax monoid.
        @always_inline
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var acc = row.reduce[ReduceMax[dtype, params.simd_width]](val, load).acc

        # Emit: per-row output — the reduced scalar.
        @always_inline
        def write(oc: RowCoord[rank]) {var acc, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                acc.slice[params.emit_tile_width](),
            )

        # `acc`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, input_shape, context)


def reduce_min[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceMin[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: min via the ReduceMin monoid.
        @always_inline
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var acc = row.reduce[ReduceMin[dtype, params.simd_width]](val, load).acc

        # Emit: per-row output — the reduced scalar.
        @always_inline
        def write(oc: RowCoord[rank]) {var acc, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                acc.slice[params.emit_tile_width](),
            )

        # `acc`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, input_shape, context)


def reduce_product[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceProduct[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: product via the ReduceProduct monoid.
        @always_inline
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var acc = row.reduce[ReduceProduct[dtype, params.simd_width]](
            val, load
        ).acc

        # Emit: per-row output — the reduced scalar.
        @always_inline
        def write(oc: RowCoord[rank]) {var acc, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                acc.slice[params.emit_tile_width](),
            )

        # `acc`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, input_shape, context)


def reduce_mean[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: sum via the ReduceSum monoid.
        @always_inline
        def add[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var acc = row.reduce[ReduceSum[dtype, params.simd_width]](add, load).acc

        # Emit: per-row output — mean = sum / axis_size. Match legacy
        # `mean` scaling: floats multiply by an f64 (f32 on Apple) reciprocal
        # cast to `dtype`; ints integer-divide by the axis length.
        # Accumulation stays in `dtype` (no upcast).
        @always_inline
        def write(
            oc: RowCoord[rank],
        ) {var acc, var axis_size, var output_fn}:
            var total = acc.slice[params.emit_tile_width]()

            comptime if dtype.is_floating_point():
                comptime float_type = DType.float32 if has_apple_gpu_accelerator() else DType.float64
                # `axis_size == 0` gives `recip = inf` and `total` (the
                # `ReduceSum` identity) is `0`, so `total * recip` is the
                # IEEE-754 `0 * inf = NaN` — the same "no data" signal
                # `numpy.mean` reports (via upcast-to-float) for an empty
                # reduction. No explicit guard needed for this branch.
                var recip = (
                    Scalar[float_type](1) / Scalar[float_type](axis_size)
                ).cast[dtype]()
                output_fn[params.emit_tile_width](
                    oc.coord,
                    total * recip,
                )
            else:
                var divisor = axis_size if axis_size != 0 else 1
                output_fn[params.emit_tile_width](
                    oc.coord,
                    total
                    / SIMD[dtype, params.emit_tile_width](
                        Scalar[dtype](divisor)
                    ),
                )

        # `acc`/`axis_size`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=True,
    ](body, input_shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based argument reductions: argmax / argmin. Same free-form body;
# the ArgMax / ArgMin monoid also tracks the winning index.
# ===----------------------------------------------------------------------=== #


def reduce_argmin[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[.int64, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    """Finds the argmin along `reduce_dim` via the Row layer: one
    `ArgMin` phase (native dtype, lower index wins ties) plus a per-row
    `emit` of the int64 index."""
    comptime simd_width = rowwise.pick_simd_width[
        ArgMin[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: argmin via the ArgMin monoid (lower index wins ties).
        @always_inline
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var indices = row.reduce[ArgMin[dtype, params.simd_width]](
            val, load
        ).acc_indices

        # Emit: per-row output — the winning index.
        @always_inline
        def write(oc: RowCoord[rank]) {var indices, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                indices.slice[params.emit_tile_width](),
            )

        # `indices`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, input_shape, context)


def reduce_argmax[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[.int64, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_fn: OutputFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ArgMax[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: argmax via the ArgMax monoid (lower index wins ties).
        @always_inline
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var indices = row.reduce[ArgMax[dtype, params.simd_width]](
            val, load
        ).acc_indices

        # Emit: per-row output — the winning index.
        @always_inline
        def write(oc: RowCoord[rank]) {var indices, var output_fn}:
            output_fn[params.emit_tile_width](
                oc.coord,
                indices.slice[params.emit_tile_width](),
            )

        # `indices`/`output_fn` ride `write`'s capture list into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, input_shape, context)


# ===----------------------------------------------------------------------=== #
# Row-based fused min-and-max: one axis walk, two outputs via the
# MinMax monoid (min_acc / max_acc).
# ===----------------------------------------------------------------------=== #


def reduce_min_and_max[
    dtype: DType,
    InputFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: Int, alignment: Int](Coord) -> SIMD[dtype, width]),
    OutputMinFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    OutputMaxFn: ImplicitlyCopyable
    & RegisterPassable
    & (def[width: SIMDLength](Coord, SIMD[dtype, width]) -> None),
    /,
    target: StaticString,
    *,
    reduce_dim: Int,
](
    input_fn: InputFn,
    output_min_fn: OutputMinFn,
    output_max_fn: OutputMaxFn,
    input_shape: Coord,
    context: Optional[DeviceContext] = None,
) raises:
    """Computes min and max in one axis walk (`MinMax` monoid), two
    outputs."""
    comptime simd_width = rowwise.pick_simd_width[
        MinMax[dtype, 1], target, 64, dtype
    ]()
    comptime assert (
        0 <= reduce_dim < input_shape.rank
    ), "reduce_dim must index input_shape"
    comptime assert input_shape.is_flat, "input_shape must be flat"
    var axis_size = Int(input_shape[reduce_dim].value())

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx: rowwise.Context[params]) {
        var axis_size, var input_fn, var output_min_fn, var output_max_fn
    }:
        comptime rank = row_coords.rank

        # Load: fuses the caller's input closure into the row's primary load.
        @always_inline
        def load[
            width: Int, alignment: Int
        ](idx: RowCoord[rank]) {var input_fn} -> SIMD[dtype, width]:
            return input_fn[width, alignment](idx.coord)

        # Prepare Row: build the row view over the axis size (always
        # dynamic here — these entry points take no static_cols param).
        var row = rowwise.Row[
            params, dtype, dtype, reduce_dim, rank, is_cached=False
        ](row_coords, Int(axis_size), ctx, load)

        # Reduce: min and max in one pass via the MinMax monoid.
        @always_inline
        def val[
            width: Int
        ](tile: SIMD[dtype, width], idx: RowCoord[rank]) {} -> SIMD[
            dtype, width
        ]:
            return tile

        var stats = row.reduce[MinMax[dtype, params.simd_width]](val, load)
        var mn = stats.min_acc
        var mx = stats.max_acc

        # Emit: per-row output — both the min and the max.
        @always_inline
        def write(
            oc: RowCoord[rank],
        ) {var mn, var mx, var output_min_fn, var output_max_fn,}:
            output_min_fn[params.emit_tile_width](
                oc.coord,
                mn.slice[params.emit_tile_width](),
            )
            output_max_fn[params.emit_tile_width](
                oc.coord,
                mx.slice[params.emit_tile_width](),
            )

        # `mn`/`mx`/`output_min_fn`/`output_max_fn` ride into `emit`.
        row.emit(write)

    rowwise.launch[
        axis=reduce_dim,
        simd_width=simd_width,
        target=target,
        num_phases=1,
    ](body, input_shape, context)
