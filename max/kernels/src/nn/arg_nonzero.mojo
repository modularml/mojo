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
"""Implements `arg_nonzero`, which returns the indices of all nonzero elements in a tensor."""


from max.algorithm.functional import _get_start_indices_of_nth_subvolume
from layout import Coord, Idx, TileTensor, coord_to_index_list
from max.runtime.tracing import Trace, TraceLevel

from std.utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# arg_nonzero
# ===-----------------------------------------------------------------------===#


@always_inline
def arg_nonzero[
    dtype: DType,
    output_type: DType,
](
    input_buffer: TileTensor[dtype, ...],
    output_buffer: TileTensor[mut=True, output_type, ...],
) raises:
    """Gather the indices of all non-zero elements in input buffer storing
    the indices in the output_buffer.

    Parameters:
        dtype: The element dtype.
        output_type: The integer dtype to store the indices in.

    Args:
        input_buffer: The tensor to count the non-zeros in.
        output_buffer: The indices of all non-zero elements.
    """
    comptime assert (
        output_buffer.flat_rank == 2
    ), "output_buffer must be of rank 2"
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...) stores below.
    comptime assert output_buffer.flat_rank >= 2

    with Trace[TraceLevel.OP, target=StaticString("cpu")]("arg_nonzero"):
        var numel = input_buffer.num_elements()
        if numel == 0:
            return

        var j: Int = 0
        for i in range(numel):
            var indices = _get_start_indices_of_nth_subvolume[0](
                i, coord_to_index_list(input_buffer.layout.shape_coord())
            )
            var offset = input_buffer.layout(Coord(indices))
            if input_buffer.raw_load(offset) != 0:
                var out_indices = IndexList[2]()
                out_indices[0] = j
                j += 1

                # Get the coordinates for the current index
                var coords = input_buffer.layout.idx2crd(Int(offset))

                # Write each coordinate to the output buffer
                comptime for k in range(input_buffer.rank):
                    out_indices[1] = k
                    output_buffer.store(
                        Coord(out_indices[0], out_indices[1]),
                        output_buffer.ElementType(coords[k].value()),
                    )


# Where has the shape 2D shape [NumNonZeros, InputRank]
@always_inline
def arg_nonzero_shape[
    dtype: DType
](input_buffer: TileTensor[dtype, ...]) -> IndexList[2]:
    """Return [NumNonZeros, InputRank] where NumNonZeros are the number of
    non-zero elements in the input.

    Parameters:
        dtype: The element dtype.

    Args:
        input_buffer: The tensor to count the non-zeros in.

    Returns:
        Shape of the arg_nonzero kernel for this input [NumNonZeros, InputRank].
    """

    var shape = IndexList[2]()
    shape[1] = input_buffer.rank

    var numel = input_buffer.num_elements()

    var j: Int = 0
    for i in range(numel):
        var offset = input_buffer.layout(i)
        if input_buffer.raw_load(offset) != 0:
            j += 1

    shape[0] = j
    return shape
