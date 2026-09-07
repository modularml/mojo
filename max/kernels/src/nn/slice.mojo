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
"""Implements the ONNX Slice operator, selecting sub-tensors along specified axes with start, stop, and step."""

from std.math import clamp

from max.algorithm import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from layout import Coord, TileTensor, coord_to_index_list
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.sys.info import simd_width_of, _current_target

from std.utils._select import _select_register_value as select
from std.utils.index import IndexList


@always_inline("nodebug")
def _normalize_and_clamp_dim(start: Int, step: Int, dim_i: Int) -> Int:
    # Normalize the start/stop indices
    var normalized_idx = select(start < 0, start + dim_i, start)

    # Compute the min/max for clamping start/end
    var idx_min = select(step > 0, 0, -1)
    var idx_max = select(step > 0, dim_i, dim_i - 1)

    # Allow start and stop to truncate like numpy and torch allow.
    return clamp(normalized_idx, idx_min, idx_max)


# ===-----------------------------------------------------------------------===#
# slice_dim_as_view
# ===-----------------------------------------------------------------------===#


@always_inline
def slice_dim_as_view[
    dtype: DType, dim: Int
](
    tensor: TileTensor[dtype, ...], start: Int, end: Int, step: Int
) -> TileTensor[
    dtype,
    Layout[
        shape_types=DynamicCoord[.int64, tensor.rank].element_types,
        stride_types=DynamicCoord[.int64, tensor.rank].element_types,
    ],
    tensor.origin,
    Engine=tensor.Engine.OffsetResultType[
        TypeList.of[Scalar[tensor.linear_idx_type]]()
    ],
    address_space=tensor.address_space,
]:
    """Returns a view of `tensor` sliced along a single dimension.

    The returned view shares the underlying data with `tensor` but adjusts the
    offset, stride, and extent of `dim` to reflect the normalized `start`,
    `end`, and `step` range.

    Args:
        tensor: Source tensor to slice.
        start: Starting index along `dim` (negative values wrap from the end).
        end: Stopping index along `dim` (exclusive; negative values wrap from the end).
        step: Stride between selected indices along `dim`; must be non-zero.
    """
    var new_shape = coord_to_index_list(tensor.layout.shape_coord())
    var new_stride = coord_to_index_list(tensor.layout.stride_coord())

    var dim_i = Int(tensor.dim(dim))
    var old_stride = Int(tensor.dynamic_stride(dim))

    # Normalize the start/stop indices
    var clamped_start = _normalize_and_clamp_dim(start, step, dim_i)
    var clamped_stop = _normalize_and_clamp_dim(end, step, dim_i)

    var new_offset = clamped_start * old_stride

    # Stride == number of elements to the next index in this dimension.
    # So to step we can just increase the stride.
    new_stride[dim] = old_stride * step

    # If the steps are positive we traverse from start, if negative from
    # stop.
    new_shape[dim] = len(range(clamped_start, clamped_stop, step))

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor._offset_storage(
        Scalar[tensor.linear_idx_type](new_offset)
    )
    return {
        new_data,
        Layout(
            Coord(rebind[IndexList[tensor.rank]](new_shape)),
            Coord(rebind[IndexList[tensor.rank]](new_stride)),
        ),
    }


# ===-----------------------------------------------------------------------===#
# slice_as_view
# ===-----------------------------------------------------------------------===#


@always_inline
def slice_as_view[
    dtype: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
](
    tensor: TileTensor[dtype, ...],
    starts: TileTensor[mut=False, start_type, ...],
    ends: TileTensor[mut=False, end_type, ...],
    steps: TileTensor[mut=False, step_type, ...],
) -> TileTensor[
    dtype,
    Layout[
        shape_types=DynamicCoord[.int64, tensor.rank].element_types,
        stride_types=DynamicCoord[.int64, tensor.rank].element_types,
    ],
    tensor.origin,
    Engine=tensor.Engine.OffsetResultType[
        TypeList.of[Scalar[tensor.linear_idx_type]]()
    ],
    address_space=tensor.address_space,
]:
    """Returns a view of `tensor` sliced along every dimension.

    For each axis, the corresponding entries in `starts`, `ends`, and `steps`
    are normalized and clamped, then the offset, stride, and extent of that
    dimension are adjusted to produce a zero-copy view of the source data.

    Args:
        tensor: Source tensor to slice.
        starts: One-dimensional tensor of starting indices, one per rank.
        ends: One-dimensional tensor of stopping indices (exclusive), one per rank.
        steps: One-dimensional tensor of strides, one per rank; each must be non-zero.
    """
    comptime assert starts.flat_rank == 1
    comptime assert ends.flat_rank == 1
    comptime assert steps.flat_rank == 1

    var new_shape = IndexList[tensor.rank]()
    var new_stride = IndexList[tensor.rank]()

    # The data does not change however we will be addressing a different
    # offset of the data; accumulate that offset and apply it once at the
    # end.
    var total_offset = Scalar[tensor.linear_idx_type](0)

    comptime for i in range(tensor.rank):
        var start = Int(starts[i])
        var stop = Int(ends[i])
        var step = Int(steps[i])
        var dim_i = Int(tensor.dim(i))
        var stride_i = Int(tensor.dynamic_stride(i))

        # Normalize the start/stop indices
        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        total_offset += Scalar[tensor.linear_idx_type](start * stride_i)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = stride_i * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = len(range(start, stop, step))

    var new_data = tensor._offset_storage(total_offset)
    return {
        new_data,
        Layout(
            Coord(new_shape),
            Coord(rebind[type_of(new_shape)](new_stride)),
        ),
    }


# ===-----------------------------------------------------------------------===#
# copy_to_slice
# ===-----------------------------------------------------------------------===#


@always_inline
def copy_to_slice[
    dtype: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    target: StaticString = "cpu",
](
    buffer: TileTensor[mut=True, dtype, ...],
    in_slice: TileTensor[mut=False, dtype, ...],
    start: TileTensor[mut=False, start_type, ...],
    end: TileTensor[mut=False, end_type, ...],
    step: TileTensor[mut=False, step_type, ...],
    context: DeviceContext,
) raises:
    """Copies `in_slice` into the slice of `buffer` defined by `start`, `end`, and `step`.

    The shape of `in_slice` must match the shape produced by slicing `buffer`
    with the given indices; otherwise an error is raised. The copy is performed
    element-wise over the sliced view of `buffer`.

    Args:
        buffer: Mutable destination tensor whose slice is overwritten.
        in_slice: Source tensor whose data is copied into the slice.
        start: One-dimensional tensor of starting indices, one per rank.
        end: One-dimensional tensor of stopping indices (exclusive), one per rank.
        step: One-dimensional tensor of strides, one per rank; each must be non-zero.
        context: Device context for the target execution backend.
    """
    var expected_shape = slice_shape(buffer, start, end, step)

    if expected_shape != rebind[IndexList[buffer.rank]](
        coord_to_index_list(in_slice.layout.shape_coord())
    ):
        raise Error(
            "Shape mismatch for mo.mutable.store.slice: expected 'slice'",
            " operand to have shape: ",
            expected_shape,
            " but got: ",
            coord_to_index_list(in_slice.layout.shape_coord()),
        )

    var buffer_slice_view = slice_as_view(buffer, start, end, step)

    @always_inline
    def copy[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        buffer_slice_view.store[width=simd_width](
            idx, in_slice.load[width=simd_width](idx)
        )

    elementwise[1, target=target, _trace_description="slice_copy"](
        copy,
        buffer_slice_view.layout.shape_coord(),
        context,
    )


# ===-----------------------------------------------------------------------===#
# slice_as_copy
# ===-----------------------------------------------------------------------===#


@always_inline
def slice_as_copy[
    dtype: DType,
    index_type: DType,
](
    output: TileTensor[mut=True, dtype, ...],
    tensor: TileTensor[mut=False, dtype, ...],
    start: TileTensor[mut=False, index_type, ...],
    end: TileTensor[mut=False, index_type, ...],
    step: TileTensor[mut=False, index_type, ...],
    ctx: DeviceContext,
) raises:
    """Copies a slice of `tensor` into `output` using the given start, end, and step indices.

    The slice of `tensor` is materialized into `output` by loading from a
    temporary view and storing element-wise; `output` must have the same rank
    as `tensor`.

    Args:
        output: Mutable destination tensor that receives the sliced data.
        tensor: Source tensor to slice.
        start: One-dimensional tensor of starting indices, one per rank.
        end: One-dimensional tensor of stopping indices (exclusive), one per rank.
        step: One-dimensional tensor of strides, one per rank; each must be non-zero.
        ctx: Device context for the target execution backend.
    """
    comptime assert output.flat_rank == tensor.flat_rank
    # Apply slice to the tensor
    var sliced = slice_as_view(tensor, start, end, step)

    # Copy lambda sliced view into output buffer.
    @always_inline
    def copy[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        output.store[width=simd_width](idx, sliced.load[width=simd_width](idx))

    # Invoke copy.
    elementwise[1](copy, output.layout.shape_coord(), ctx)


# ===-----------------------------------------------------------------------===#
# slice_shape
# ===-----------------------------------------------------------------------===#


@always_inline
def slice_shape[
    input_type: DType,
    start_type: DType,
    stop_type: DType,
    step_type: DType,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    start_buf: TileTensor[mut=False, start_type, ...],
    stop_buf: TileTensor[mut=False, stop_type, ...],
    step_buf: TileTensor[mut=False, step_type, ...],
) raises -> IndexList[input_buf.rank]:
    """Computes the shape that results from slicing `input_buf` with the given start, stop, and step tensors.

    Validates that the index tensors each have one entry per rank of
    `input_buf` and that no step is zero, then normalizes and clamps the
    indices per axis to determine the output extent along each dimension.

    Args:
        input_buf: Source tensor whose slice shape is computed.
        start_buf: One-dimensional tensor of starting indices, one per rank.
        stop_buf: One-dimensional tensor of stopping indices (exclusive), one per rank.
        step_buf: One-dimensional tensor of strides, one per rank; each must be non-zero.

    Returns:
        An `IndexList` holding the extent of each dimension after slicing.
    """
    comptime assert start_buf.flat_rank == 1, "start_buf.rank must be 1"
    comptime assert stop_buf.flat_rank == 1, "stop_buf.rank must be 1"
    comptime assert step_buf.flat_rank == 1, "step_buf.rank must be 1"

    if input_buf.rank != Int(start_buf.dim[0]()):
        raise Error("[slice] start indices size must equal input rank")
    if input_buf.rank != Int(stop_buf.dim[0]()):
        raise Error("[slice] stop indices size must equal input rank")
    if input_buf.rank != Int(step_buf.dim[0]()):
        raise Error("[slice] step indices size must equal input rank")

    for axis in range(input_buf.rank):
        if step_buf[axis] == 0:
            raise Error("[slice] step must be non-zero")

    var output_shape = IndexList[input_buf.rank]()

    for i in range(input_buf.rank):
        var start = Int(start_buf[i])
        var stop = Int(stop_buf[i])
        var step = Int(step_buf[i])
        var dim_i = Int(input_buf.dim(i))

        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        if step > 0 and stop < start:
            raise Error(
                "[slice] normalized stop cannot be smaller than start for"
                " positive step"
            )

        if step < 0 and start < stop:
            raise Error(
                "[slice] normalized start cannot be smaller than stop for"
                " negative step"
            )

        output_shape[i] = len(range(start, stop, step))

    return output_shape


# ===-----------------------------------------------------------------------===#
# sliced_add
# ===-----------------------------------------------------------------------===#


def sliced_add[
    dtype: DType,
    //,
    target: StaticString,
](
    c: TileTensor[mut=True, dtype, ...],
    a: TileTensor[mut=False, dtype, ...],
    b: TileTensor[mut=False, dtype, ...],
    lora_end_idx: TileTensor[mut=False, .int64, ...],
    ctx: DeviceContext,
) raises:
    """Adds tensors a and b element-wise for rows < lora_end_idx, otherwise copies a.

    This is used for LoRA where only some sequences have LoRA applied.
    For rows in [0, lora_end_idx): c = a + b
    For rows in [lora_end_idx, batch_seq_len): c = a

    Args:
        c: Output tensor.
        a: First input tensor.
        b: Second input tensor.
        lora_end_idx: Scalar tensor with end index of LoRA token portion (rows to apply add).
        ctx: Device context for GPU operations.
    """
    comptime assert lora_end_idx.flat_rank == 1

    var batch_end_idx = Int(lora_end_idx[0])

    def _sliced_add[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var out_val: SIMD[dtype, width]

        if Int(idx[0].value()) >= batch_end_idx:
            out_val = a.load[width](idx)
        else:
            var a_val = a.load[width](idx)
            var b_val = b.load[width](idx)
            out_val = a_val + b_val

        c.store[width](idx, out_val)

    comptime if target == "gpu":
        comptime compile_target = get_gpu_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[
            simd_width,
            target=target,
            _trace_description="slice_add",
        ](
            _sliced_add,
            c.layout.shape_coord(),
            ctx,
        )
    else:
        comptime compile_target = _current_target()
        comptime simd_width = simd_width_of[dtype, target=compile_target]()

        elementwise[simd_width, target=target](
            _sliced_add, c.layout.shape_coord(), ctx
        )
