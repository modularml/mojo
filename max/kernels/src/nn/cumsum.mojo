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
"""Implements the ONNX CumSum operator, computing prefix sums along a specified tensor axis."""

from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major

from std.utils.numerics import get_accum_type


@always_inline
def cumsum[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
](output: TileTensor[mut=True, dtype, ...], input: TileTensor[dtype, ...],):
    """
    Implements the CumSum operator from the ONNX spec:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
    Computes cumulative sum of the input elements along the given axis.
    Cumulative sum can be inclusive or exclusive of the top element, and
    normal or reverse (direction along a given axis).

    Parameters:
        dtype: Type of the input and output tensors.
        exclusive: If set to True, return exclusive sum (top element not included).
        reverse: If set to True, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation.

    Args:
        output: The output tensor.
        input: The input tensor.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output should have the same rank."

    comptime accum_type = DType.float64 if dtype == DType.float32 else get_accum_type[
        dtype
    ]()
    comptime assert (
        -input.rank <= axis < input.rank
    ), "Axis value must be in range [-rank, rank)"
    comptime axis_pos = axis if axis >= 0 else axis + input.rank

    var shape = coord_to_index_list(input.layout.shape_coord())

    var inner = 1
    var outer = 1
    var depth = 1
    for i in range(input.rank):
        if i < axis_pos:
            inner *= shape[i]
        elif i > axis_pos:
            outer *= shape[i]
        else:
            depth = shape[i]

    var output_data = TileTensor(
        output.ptr,
        row_major(Coord(output.num_elements())),
    )
    var input_data = TileTensor(
        input.ptr,
        row_major(Coord(input.num_elements())),
    )

    for outer_index in range(outer):
        var outer_index_adj: Int

        comptime if reverse:
            outer_index_adj = (outer - 1) - outer_index
        else:
            outer_index_adj = outer_index

        for inner_index in range(inner):
            var accumulator: Scalar[accum_type] = 0
            var inner_index_adj: Int

            comptime if reverse:
                inner_index_adj = (inner - 1) - inner_index
            else:
                inner_index_adj = inner_index

            for depth_index in range(depth):
                var depth_index_adj: Int

                comptime if reverse:
                    depth_index_adj = (depth - 1) - depth_index
                else:
                    depth_index_adj = depth_index

                var index = (
                    outer_index_adj
                    + inner_index_adj * depth * outer
                    + depth_index_adj * outer
                )

                comptime if exclusive:
                    output_data[index] = accumulator.cast[dtype]()
                    accumulator = (
                        accumulator + input_data[index].cast[accum_type]()
                    )
                else:
                    accumulator = (
                        accumulator + input_data[index].cast[accum_type]()
                    )
                    output_data[index] = accumulator.cast[dtype]()
