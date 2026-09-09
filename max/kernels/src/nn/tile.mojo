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
"""Implements the ONNX Tile operator, which replicates a tensor a specified number of times along each axis."""

from layout import TileTensor
from std.memory import unsafe_memcpy

from std.utils import IndexList

# TODO: This implementation supports up to 4 dimensions.

# Note: ONNX spec specifies that `tile` behaves like Numpy's tile, but without
#       broadcast. This means that `repeats` is a 1D int64 tensor of the SAME
#       length as input's rank (unlike Numpy that allows `repeats` to have more
#       or less elements than the input's rank).


@always_inline
def tile[
    dtype: DType, type_repeats: DType
](
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    repeats: TileTensor[type_repeats, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
) raises:
    """
    Implements the `Tile` operator from the ONNX spec. This behaves like Numpy
    tile, but without broadcast.

    Parameters:
        dtype: Type of the input and output tensors.
        type_repeats: Type of the repeats tensor.

    Args:
        input: The input tensor. Currently <= 4 dimensions are supported.
        repeats: One-dimensional tensor that specifies the number of repeated
                 copies along each of the input's dimensions. Length equals
                 input tensor rank.
        output: The output tensor. Has the same dimensions and type as input.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output must have the same rank"

    comptime assert (
        input.rank <= 4
    ), "Currently only inputs of up to three dimensions are supported."

    comptime assert (
        repeats.flat_rank == 1 and type_repeats == .int64
    ), "Rank of repeats tensor needs to be one-dimensional and of int64 type."

    if input.rank != Int(repeats.dim(0)):
        raise Error(
            "Length of repeats tensor should be equal to the rank of the input"
            " tensor."
        )

    var num_dp_input = 1
    var num_depth_input = 1
    var num_rows_input = 1

    comptime if input.rank == 4:
        num_dp_input = Int(input.dim(input.rank - 4))

    comptime if input.rank >= 3:
        num_depth_input = Int(input.dim(input.rank - 3))

    comptime if input.rank >= 2:
        num_rows_input = Int(input.dim(input.rank - 2))
    var num_cols_input = Int(input.dim(input.rank - 1))

    var repeats_len = Int(repeats.dim(0))

    # Pre-compute repeat values, using 1 for dimensions that don't exist
    # to avoid negative index access when rank < 4
    var repeat_cols = Int(repeats[repeats_len - 1])
    var repeat_rows = 1
    var repeat_depth = 1
    var repeat_dp = 1

    comptime if input.rank >= 2:
        repeat_rows = Int(repeats[repeats_len - 2])

    comptime if input.rank >= 3:
        repeat_depth = Int(repeats[repeats_len - 3])

    comptime if input.rank >= 4:
        repeat_dp = Int(repeats[repeats_len - 4])

    # Initializes output by first copying in the original input to the
    # appropriate output elements, and then handles tiling across the column
    # (last) dimension.
    # e.g., for:
    #   input:  [[1, 2, 3],
    #            [4, 5, 6]]
    #   and repeats = [2,2], the below will handle:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [X, X, X, X, X, X],
    #            [X, X, X, X, X, X]]
    #   where 'X' denotes parts of the output that are not yet calculated.
    #   These (i.e., for dimensions beyond the innermost) are handled later.
    for dp in range(num_dp_input):
        for d in range(num_depth_input):
            for r in range(num_rows_input):
                # print(dp, d, r)
                var input_src_index = (
                    dp * num_depth_input * num_rows_input * num_cols_input
                    + d * num_rows_input * num_cols_input
                    + r * num_cols_input
                )
                var output_src_index = (
                    dp
                    * num_depth_input
                    * repeat_depth
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                    + d
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                    + r * num_cols_input * repeat_cols
                )
                var output_src_stride = num_cols_input
                var count = output_src_stride
                for rep in range(repeat_cols):
                    var src_ptr = input.ptr + input_src_index
                    var dst_ptr = output.ptr + (
                        output_src_index + rep * output_src_stride
                    )
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=count)

    # Handles tiling across the second lowest dimension (if tensor rank >= 2).
    # Continuing with the example above, this will handle the 'X's, which
    # correspond to the first '2' in the repeats = [2, 2] tensor.
    # The result is:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6]]
    # Moving from the inner to the outermost dimension, we can unsafe_memcpy to
    # replicate contiguous memory areas (representing a dimension to be tiled).
    comptime if input.rank >= 2:
        var src_index_stride = num_rows_input * num_cols_input * repeat_cols
        var count = src_index_stride
        for dp in range(num_dp_input):
            for d in range(num_depth_input):
                var src_index = (
                    dp
                    * num_depth_input
                    * repeat_depth
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                    + d
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                )
                for rep in range(repeat_rows - 1):
                    var src_ptr = output.ptr + src_index
                    var dst_ptr = output.ptr + (
                        src_index + (rep + 1) * src_index_stride
                    )
                    # dst_ptr and src_ptr are non-overlapping slices of the
                    # same `output` buffer (shared origin). Opt out of
                    # exclusivity with an unsafe any-origin:
                    # unsafe_memcpy's non-overlap
                    # is a caller contract the checker can't prove.
                    unsafe_memcpy(
                        dest=dst_ptr,
                        src=src_ptr.as_unsafe_any_origin(),
                        count=count,
                    )

    # Handles tiling across the third dimension from the end (if tensor rank >= 3)
    comptime if input.rank >= 3:
        var src_index_stride = (
            num_depth_input
            * repeat_rows
            * num_rows_input
            * num_cols_input
            * repeat_cols
        )
        var count = src_index_stride
        for dp in range(num_dp_input):
            var src_index = (
                dp
                * num_depth_input
                * repeat_depth
                * num_rows_input
                * repeat_rows
                * num_cols_input
                * repeat_cols
            )
            for rep in range(repeat_depth - 1):
                var src_ptr = output.ptr + src_index
                var dst_ptr = output.ptr + (
                    src_index + (rep + 1) * src_index_stride
                )
                # dst_ptr and src_ptr are non-overlapping slices of the
                # same `output` buffer (shared origin). Opt out of
                # exclusivity with an unsafe any-origin:
                # unsafe_memcpy's non-overlap
                # is a caller contract the checker can't prove.
                unsafe_memcpy(
                    dest=dst_ptr,
                    src=src_ptr.as_unsafe_any_origin(),
                    count=count,
                )

    # Handles tiling across the fourth dimension from the end (if tensor rank == 4)
    comptime if input.rank == 4:
        var src_index_stride = (
            num_dp_input
            * repeat_depth
            * num_depth_input
            * repeat_rows
            * num_rows_input
            * num_cols_input
            * repeat_cols
        )
        var count = src_index_stride
        var src_index = 0
        for rep in range(repeat_dp - 1):
            var src_ptr = output.ptr + src_index
            var dst_ptr = output.ptr + (
                src_index + (rep + 1) * src_index_stride
            )
            # dst_ptr and src_ptr are non-overlapping slices of the
            # same `output` buffer (shared origin). Opt out of
            # exclusivity with an unsafe any-origin:
            # unsafe_memcpy's non-overlap
            # is a caller contract the checker can't prove.
            unsafe_memcpy(
                dest=dst_ptr,
                src=src_ptr.as_unsafe_any_origin(),
                count=count,
            )


@always_inline
def tile_shape[
    input_type: DType,
    repeats_type: DType,
](
    input_buf: TileTensor[input_type, ...],
    repeats_buf: TileTensor[repeats_type, ...],
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a `tile` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        repeats_type: Type of the repeats tensor.

    Args:
        input_buf: The input tensor.
        repeats_buf: The repeats tensor.

    Returns:
        The output shape.
    """
    comptime assert repeats_buf.flat_rank == 1, "repeats_buf must be of rank 1"

    # TODO add runtime test once we support dynamic rank execution, currently
    # MLIR verifier of `MO::TileOp` prevents testing this with static rank.
    if Int(repeats_buf.dim(0)) != input_buf.rank:
        raise Error("[tile] requires (len(repeats) == input_rank)")

    # Compute and return the output shape.
    var output_shape = IndexList[input_buf.rank]()

    comptime for i in range(input_buf.rank):
        output_shape[i] = Int(input_buf.dim(i)) * Int(repeats_buf[i])

    return output_shape
