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
"""Provides a single-pass GPU argmax/argmin kernel over the inner-most dimension."""


from std.bit import next_power_of_two
from max.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.math import align_down, ceildiv, iota
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import has_apple_gpu_accelerator

from max.gpu.host import DeviceContext, get_gpu_target
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from layout import TensorLayout, TensorEngine, TileTensor, coord_to_index_list
from nn.topk import TopK_2, _block_reduce_topk, _topk_dead_val
from std.utils.index import IndexList


# Elements each thread pulls per loop trip, as a multiple of the vector width.
# The (value, index) update is loop-carried, so without a widened trip a warp
# keeps only one load in flight and the scan runs at HBM-latency rate rather
# than at bandwidth. Four 128-bit loads per trip is enough to saturate a
# streaming CTA while staying well inside the 64-register budget at 1024
# threads.
comptime _ARGMAXMIN_UNROLL = 4

# Row bytes per scan block. One block saturates at the per-SM load rate, so
# long rows have to be split across blocks to reach device bandwidth; the
# split costs one extra (tiny) combine launch, which is why short rows are
# left on a single block.
comptime _ARGMAXMIN_BYTES_PER_BLOCK = 32 * 1024

# Cap on slices per row so the combine stays one block of at most 1024
# threads and the scratch stays negligible.
comptime _ARGMAXMIN_MAX_SPLITS = 64


@always_inline
def _argmaxmin_vec_update[
    dtype: DType, largest: Bool, simd_width: Int
](
    mut best_vals: SIMD[dtype, simd_width],
    mut best_idxs: SIMD[.int32, simd_width],
    vals: SIMD[dtype, simd_width],
    base: Int,
):
    """Folds one loaded vector into the per-lane running (extremum, index).

    The comparison is strict so that, scanning indices in increasing order,
    the lowest index wins a tie. NaN never compares greater and is therefore
    ignored, matching `TopK_2.insert`.
    """
    comptime lane_offsets = iota[.int32, simd_width]()
    var better: SIMD[.bool, simd_width]
    comptime if largest:
        better = vals.gt(best_vals)
    else:
        better = vals.lt(best_vals)
    best_vals = better.select(vals, best_vals)
    best_idxs = better.select(lane_offsets + Int32(base), best_idxs)


@always_inline
def _argmaxmin_scan[
    dtype: DType,
    largest: Bool,
    simd_width: Int,
    unroll: Int,
    alignment: Int,
](
    row: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
    tid: Int,
    block_size: Int,
    mut best_vals: SIMD[dtype, simd_width],
    mut best_idxs: SIMD[.int32, simd_width],
):
    """Streams one row once, accumulating a per-lane (extremum, index)."""
    var lane_stride = block_size * simd_width
    var trip = lane_stride * unroll
    var num_trips = num_elements // trip
    var vec_end = align_down(num_elements, simd_width)

    for t in range(num_trips):
        var offset = t * trip + tid * simd_width
        var staged = Array[SIMD[dtype, simd_width], unroll](
            fill=SIMD[dtype, simd_width](_topk_dead_val[dtype, largest]())
        )
        comptime for u in range(unroll):
            staged[u] = row.unsafe_load[width=simd_width, alignment=alignment](
                offset + u * lane_stride
            )
        comptime for u in range(unroll):
            _argmaxmin_vec_update[dtype, largest](
                best_vals, best_idxs, staged[u], offset + u * lane_stride
            )

    var i = num_trips * trip + tid * simd_width
    while i + simd_width <= vec_end:
        _argmaxmin_vec_update[dtype, largest](
            best_vals,
            best_idxs,
            row.unsafe_load[width=simd_width, alignment=alignment](i),
            i,
        )
        i += lane_stride


@always_inline
def _argmaxmin_block_partial[
    dtype: DType, largest: Bool, simd_width: Int, unroll: Int
](
    row: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    begin: Int,
    count: Int,
    aligned: Bool,
    tid: Int,
    block_size: Int,
) -> TopK_2[dtype, largest]:
    """Reduces `row[begin : begin + count]` to one (extremum, global index)."""
    var best_vals = SIMD[dtype, simd_width](_topk_dead_val[dtype, largest]())
    var best_idxs = SIMD[.int32, simd_width](0)
    var chunk = row.unsafe_offset(begin)

    if aligned:
        _argmaxmin_scan[
            dtype,
            largest,
            simd_width,
            unroll,
            align_of[SIMD[dtype, simd_width]](),
        ](chunk, count, tid, block_size, best_vals, best_idxs)
    else:
        _argmaxmin_scan[
            dtype, largest, simd_width, unroll, align_of[Scalar[dtype]]()
        ](chunk, count, tid, block_size, best_vals, best_idxs)

    # Collapse the lanes. Lanes hold indices from different trips, so the
    # tie-break has to compare indices explicitly rather than rely on lane
    # order.
    var partial = TopK_2[dtype, largest]()
    comptime for lane in range(simd_width):
        var val = best_vals[lane]
        var idx = Int(best_idxs[lane])
        var better: Bool
        comptime if largest:
            better = val > partial.u
        else:
            better = val < partial.u
        if better or (val == partial.u and idx < partial.p):
            partial.u = val
            partial.p = idx

    # Ragged tail: fewer than `simd_width` elements, all at higher indices
    # than anything scanned above, so a strict insert keeps first-index.
    var vec_end = align_down(count, simd_width)
    if tid < count - vec_end:
        partial.insert(chunk.unsafe_offset(vec_end + tid)[], vec_end + tid)

    partial.p += begin
    return partial


@__name(t"argmaxmin_scan_{dtype}_{largest}")
def _argmaxmin_scan_kernel[
    dtype: DType,
    largest: Bool,
    simd_width: Int,
    unroll: Int,
    InputLayoutType: TensorLayout,
    InputEngine: TensorEngine,
](
    input: TileTensor[
        dtype, InputLayoutType, ImmutAnyOrigin, Engine=InputEngine
    ],
    part_vals: MutPointer[Scalar[dtype], MutAnyOrigin],
    part_idxs: MutPointer[Int32, MutAnyOrigin],
    num_elements_arg: Int32,
    split_len_arg: Int32,
    num_splits_arg: Int32,
    aligned_arg: Int32,
):
    """Streams one contiguous slice of one row and emits its local winner.

    Target families: NVIDIA (SM90/SM100), AMD CDNA, Apple silicon. Grid is
    `(rows, num_splits)`; each block reads its slice exactly once with
    vector loads and keeps a per-lane running `(extremum, index)` in
    registers. There is no input copy and no re-scan.

    Parameters:
        dtype: Element type of the input.
        largest: Select the maximum (argmax) or the minimum (argmin).
        simd_width: Elements per vector load.
        unroll: Vector loads issued back to back per loop trip.
        InputLayoutType: Layout of the input.
        InputEngine: Engine of the input.

    Args:
        input: Input rows, contiguous with `num_elements` per row.
        part_vals: Per-slice winning values, `rows * num_splits`.
        part_idxs: Per-slice winning indices, row-relative.
        num_elements_arg: Length of the reduced inner-most dimension.
        split_len_arg: Elements per slice, a multiple of `simd_width`.
        num_splits_arg: Slices per row.
        aligned_arg: Non-zero when vector loads may assume vector alignment.
    """
    var num_elements = Int(num_elements_arg)
    # Rows ride the x dimension: it is the only one whose extent is not
    # capped at 65535, and a batch can have millions of rows.
    var row_id = Int(block_idx.x)
    var split = Int(block_idx.y)
    var tid = Int(thread_idx.x)

    var begin = min(split * Int(split_len_arg), num_elements)
    var count = min(Int(split_len_arg), num_elements - begin)

    with PDL():
        var partial = _argmaxmin_block_partial[
            dtype, largest, simd_width, unroll
        ](
            input.ptr.unsafe_offset(row_id * num_elements),
            begin,
            count,
            aligned_arg != 0,
            tid,
            Int(block_dim.x),
        )
        var total = _block_reduce_topk[ascending=largest](partial)

        if tid == 0:
            var slot = row_id * Int(num_splits_arg) + split
            part_vals.unsafe_offset(slot)[] = total.u
            part_idxs.unsafe_offset(slot)[] = Int32(total.p)


@__name(t"argmaxmin_combine_{dtype}_{out_idx_type}_{largest}")
def _argmaxmin_combine_kernel[
    dtype: DType,
    out_idx_type: DType,
    largest: Bool,
    OutIdxLayoutType: TensorLayout,
    OutIdxEngine: TensorEngine,
](
    out_idxs: TileTensor[
        mut=True,
        out_idx_type,
        OutIdxLayoutType,
        MutAnyOrigin,
        Engine=OutIdxEngine,
    ],
    part_vals: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    part_idxs: ImmPointer[Int32, ImmutAnyOrigin],
    num_splits_arg: Int32,
):
    """Reduces one row's per-slice winners to the final index.

    Parameters:
        dtype: Element type of the values.
        out_idx_type: Output index dtype.
        largest: Select the maximum (argmax) or the minimum (argmin).
        OutIdxLayoutType: Layout of the output indices.
        OutIdxEngine: Engine of the output indices.

    Args:
        out_idxs: One index per row.
        part_vals: Per-slice winning values from the scan kernel.
        part_idxs: Per-slice winning indices from the scan kernel.
        num_splits_arg: Slices per row.
    """
    var num_splits = Int(num_splits_arg)
    var row_id = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = row_id * num_splits

    with PDL():
        # Slices are scanned in increasing index order and carry global
        # indices, so a strict insert plus the block reduce's lowest-index
        # tie-break reproduces a single-pass first-index scan.
        var partial = TopK_2[dtype, largest]()
        for i in range(tid, num_splits, Int(block_dim.x)):
            partial.insert(
                part_vals.unsafe_offset(base + i)[],
                Int(part_idxs.unsafe_offset(base + i)[]),
            )

        var total = _block_reduce_topk[ascending=largest](partial)

        if tid == 0:
            out_idxs.ptr.unsafe_offset(row_id)[] = Int(total.p).cast[
                out_idx_type
            ]()


def argmaxmin_gpu[
    dtype: DType, output_type: DType, largest: Bool
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_type, ...],
) raises:
    """
    Reduces the inner-most dimension to the index of its largest/smallest
    element with a single streaming pass.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        output_type: DType - The data dtype of the output tensor.
        largest: Bool - Whether to perform argmax or argmin.
    Args:
        ctx: Device context for launching the reduction kernel.
        input: Input tensor reduced along its inner-most dimension. Must have
            positive rank and the same rank as `output`.
        output: Output tensor receiving the index of the selected extreme
            value along the inner-most dimension. Must have the same rank as
            `input`.
    """
    comptime assert input.rank > 0, "Input rank must be positive"
    comptime assert (
        input.rank == output.rank
    ), "Input and output rank must be the same"

    var in_shape = rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("input", in_shape, dtype),
                    "largest=" + String(largest),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "argmaxmin_gpu",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var num_elements = in_shape[input.rank - 1]
        var num_rows = in_shape.flattened_length() // num_elements
        if num_rows == 0 or num_elements == 0:
            return

        # Per-lane indices are tracked as int32 to halve the register cost of
        # the running argmax; a row that long is > 4GB and unreachable here.
        if num_elements > Int(Int32.MAX):
            raise Error(
                "argmaxmin_gpu: inner dimension exceeds the int32 index range"
            )

        comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

        # A single block streaming a whole row tops out at the per-SM load
        # rate (~44 GB/s measured on B200), so a 512KB row would take ~12us.
        # Split the row across blocks until each holds roughly
        # `_ARGMAXMIN_BYTES_PER_BLOCK`, but never past the point where the
        # batch alone already fills the device.
        var row_bytes = num_elements * size_of[Scalar[dtype]]()
        var num_splits = min(
            ceildiv(row_bytes, _ARGMAXMIN_BYTES_PER_BLOCK),
            max(Int(ctx.default_device_info.sm_count) // num_rows, 1),
        )
        num_splits = min(max(num_splits, 1), _ARGMAXMIN_MAX_SPLITS)

        var split_len = (
            ceildiv(ceildiv(num_elements, num_splits), simd_width) * simd_width
        )

        # A slice base is `input.ptr + (row * num_elements + k * split_len)`,
        # and `split_len` is a multiple of `simd_width`, so vector loads are
        # aligned exactly when the base pointer and the row pitch both are.
        # Views into a larger buffer need not satisfy either.
        comptime vec_bytes = align_of[SIMD[dtype, simd_width]]()
        var aligned = (
            num_elements % simd_width == 0 and Int(input.ptr) % vec_bytes == 0
        )

        # Size the block to its slice so short rows do not launch mostly-idle
        # blocks. `_block_reduce_topk` is single-warp on Apple.
        var block_size = min(
            next_power_of_two(ceildiv(split_len, simd_width)), 1024
        )
        block_size = max(block_size, WARP_SIZE)
        var combine_block_size = max(next_power_of_two(num_splits), WARP_SIZE)
        comptime if has_apple_gpu_accelerator():
            block_size = WARP_SIZE
            combine_block_size = WARP_SIZE

        var part_vals = ctx.enqueue_create_buffer[dtype](num_rows * num_splits)
        var part_idxs = ctx.enqueue_create_buffer[.int32](num_rows * num_splits)

        comptime scan_kernel = _argmaxmin_scan_kernel[
            dtype,
            largest,
            simd_width,
            _ARGMAXMIN_UNROLL,
            input.LayoutType,
            type_of(input).Engine,
        ]
        ctx.enqueue_function[scan_kernel](
            input.as_immut(),
            part_vals,
            part_idxs,
            Int32(num_elements),
            Int32(split_len),
            Int32(num_splits),
            Int32(1) if aligned else Int32(0),
            grid_dim=(num_rows, num_splits),
            block_dim=block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        comptime combine_kernel = _argmaxmin_combine_kernel[
            dtype,
            output_type,
            largest,
            output.LayoutType,
            type_of(output).Engine,
        ]
        ctx.enqueue_function[combine_kernel](
            output,
            part_vals,
            part_idxs,
            Int32(num_splits),
            grid_dim=num_rows,
            block_dim=combine_block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

        _ = part_vals^
        _ = part_idxs^


def argmax_gpu[
    dtype: DType, output_type: DType
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_type, ...],
) raises:
    """
    Computes the indices of the maximum values along the inner-most dimension of the input tensor on the GPU.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        output_type: DType - The data dtype of the output tensor.
    Args:
        ctx: Device context for allocating intermediate buffers and
            launching the reduction kernel.
        input: Input tensor whose inner-most dimension is searched for the
            maximum value. Must have positive rank and the same rank as
            `output`.
        output: Output tensor receiving the index of the maximum value along
            the inner-most dimension. Must have the same rank as `input`.
    """
    argmaxmin_gpu[largest=True](ctx, input, output)


def argmin_gpu[
    dtype: DType, output_type: DType
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_type, ...],
) raises:
    """
    Computes the indices of the minimum values along the inner-most dimension of the input tensor on the GPU.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        output_type: DType - The data dtype of the output tensor.
    Args:
        ctx: Device context for allocating intermediate buffers and
            launching the reduction kernel.
        input: Input tensor whose inner-most dimension is searched for the
            minimum value. Must have positive rank and the same rank as
            `output`.
        output: Output tensor receiving the index of the minimum value along
            the inner-most dimension. Must have the same rank as `input`.
    """
    argmaxmin_gpu[largest=False](ctx, input, output)
