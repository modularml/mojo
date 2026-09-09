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
"""Implements `repeat_interleave`, which repeats each tensor element a specified number of times along an axis."""

from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major

from std.utils import IndexList


def _collapse_dims_around_axis(
    shape: IndexList, axis: Int, out result: IndexList[3]
) raises:
    if axis >= shape.size:
        raise Error("axis larger than provided shape")

    comptime if shape.size == 0:
        return IndexList[3](1, 1, 1)

    var split = shape[axis]

    var before = 1
    for i in range(axis):
        before *= shape[i]

    var after = 1
    for i in range(axis + 1, shape.size):
        after *= shape[i]

    return IndexList[3](before, split, after)


@always_inline
def repeat_interleave[
    dtype: DType,
    type_repeats: DType,
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """
    Fill `output` by repeating values from `input` along `axis` based on the
    values in `repeats` buffer.

    This is intended to implement the same functionality as torch.repeat:
    https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

    Args:
        input: The input buffer.
        repeats: The number of repetitions each element in input.
        axis: The axis along which to repeat values.
        output: The output buffer.
        ctx: Device context.
    """
    # comptime assert (is_row_major[input.rank](input.layout)) and (
    #     is_row_major[output.rank](output.layout)
    # )
    comptime assert input.flat_rank == output.flat_rank
    comptime assert repeats.flat_rank == 1

    # Compute the shape of the input and result buffers.
    # These are the shapes of the buffers we will be working on.
    var collapsed_input_shape = _collapse_dims_around_axis(
        coord_to_index_list(input.layout.shape_coord()), axis
    )
    var collapsed_output_shape = _collapse_dims_around_axis(
        coord_to_index_list(output.layout.shape_coord()), axis
    )

    assert collapsed_output_shape[0] == collapsed_input_shape[0]
    assert collapsed_output_shape[2] == collapsed_input_shape[2]

    var collapsed_input = TileTensor(
        input.ptr,
        row_major(Coord(collapsed_input_shape)),
    )

    var collapsed_output = TileTensor(
        output.ptr,
        row_major(Coord(collapsed_output_shape)),
    )

    var input_repeat_dim = collapsed_input_shape[1]
    var output_repeat_dim = collapsed_output_shape[1]
    var repeat_stride = Int(repeats.dim[0]() > 1)

    # Mapping from offsets in the input tensor to offsets in the output tensor
    # along the repeat axis.
    var offset_mapping = List[Int](unsafe_uninit_length=output_repeat_dim)

    var repeat_offset = 0
    var output_offset = 0
    for input_offset in range(input_repeat_dim):
        var repeat_val = Int(repeats[repeat_offset])
        if repeat_val < 0:
            raise Error("all repeat values must be non-negative")

        for _ in range(repeat_val):
            offset_mapping[output_offset] = input_offset
            output_offset += 1

        repeat_offset += repeat_stride

    # `offset_mapping` is a `List` (not register-passable), so it cannot be
    # captured directly by the unified closure. Capture a `Span` view instead;
    # the `Span` carries the list's origin, keeping it alive through the
    # closure's last use. Everything else is captured by `mut`, matching the
    # original implicit-capture behavior.
    var offset_mapping_view = Span(offset_mapping)

    @always_inline
    def func[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var offset_mapping_view, mut}:
        var output_index = rebind[IndexList[3]](coord_to_index_list(idx))
        var input_index = output_index
        input_index[1] = offset_mapping_view[output_index[1]]

        var input_idx = collapsed_input.layout(Coord(input_index))
        var input_value = collapsed_input.raw_load[width=width](input_idx)

        var output_idx = collapsed_output.layout(Coord(output_index))
        collapsed_output.raw_store(output_idx, input_value)

    elementwise[simd_width_of[output.dtype]()](
        func, collapsed_output.layout.shape_coord(), ctx
    )


@always_inline
def repeat_interleave_shape[
    type_repeats: DType,
](
    input: TileTensor,
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
) raises -> IndexList[input.rank]:
    """
    Computes the output shape of `repeat_interleave` for the given `input`,
    `repeats`, and `axis`.

    The returned `IndexList` matches `input`'s rank with the size along
    `axis` replaced by the summed `repeats` values, or by `input[axis]`
    multiplied by the single repeat value when `repeats` is a size-1
    broadcast.

    Args:
        input: The input tensor whose shape is being transformed.
        repeats: A one-dimensional integral tensor of per-element repeat
            counts, either size 1 or equal to `input.dim(axis)`.
        axis: The axis along which elements are repeated.

    Returns:
        An `IndexList` matching `input.rank` with the `axis` dimension
        updated to the total repeated size.
    """
    comptime assert type_repeats.is_integral()
    comptime assert repeats.flat_rank == 1

    var repeats_size = repeats.dim[0]()
    if repeats_size != 1 and repeats_size != Scalar[repeats.linear_idx_type](
        Int(input.dim(axis))
    ):
        raise Error(
            "repeat_interleave: repeats must be size 1 or equal to "
            "the size of input[axis]"
        )

    var total_repeats = 0
    comptime assert repeats_size.dtype.is_integral()
    for i in range(repeats_size):
        total_repeats += Int(repeats[i])

    var result = coord_to_index_list(input.layout.shape_coord())

    # If the repeats is size 1, the repeat is treated as a broadcast
    if repeats_size == 1:
        result[axis] *= total_repeats
    else:
        result[axis] = total_repeats

    return rebind[IndexList[input.rank]](result)
