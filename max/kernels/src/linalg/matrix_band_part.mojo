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
"""The module implements matrix band part functions."""


from std.algorithm.functional import unswitch

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import TileTensor


from std.utils.coord import Coord
from std.utils.index import IndexList


@always_inline
def matrix_band_part[
    dtype: DType,
    int_type: DType,
    cond_type: DType,
    rank: Int,
    simd_width: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    target: StaticString = "cpu",
](
    input_0_fn: InputFnType,
    input_shape: IndexList[rank],
    num_lower: TileTensor[int_type, ...],
    num_upper: TileTensor[int_type, ...],
    exclude: TileTensor[cond_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Copies a band of `input_0_fn` into `output`, zeroing elements outside the band defined by `num_lower` and `num_upper`.

    Reads the lower and upper diagonal counts and the `exclude` flag from the
    supplied tensors, then dispatches to the implementation with the exclude
    flag specialized at compile time via `unswitch`.

    Parameters:
        dtype: Element `DType` of the input and output tensors.
        int_type: `DType` of the `num_lower` and `num_upper` count tensors.
        cond_type: `DType` of the `exclude` flag tensor.
        rank: Number of dimensions in the input tensor (must be at least 2).
        simd_width: SIMD width for the elementwise kernel launch.
        InputFnType: Callable type returning the input element at a given
            index as a `SIMD` vector.
        target: Target device for the elementwise kernel (defaults to
            `"cpu"`).

    Args:
        input_0_fn: Function returning the input element at a given index.
        input_shape: Shape of the input tensor.
        num_lower: Scalar tensor giving the number of lower subdiagonals to keep.
        num_upper: Scalar tensor giving the number of upper superdiagonals to keep.
        exclude: Scalar tensor flag that, when nonzero, inverts the band mask.
        output: Mutable output tensor receiving the band-part result.
        ctx: Device context used to launch the elementwise kernel.
    """
    var lower_diagonal_index = Int(num_lower.load_linear[1](IndexList[1](0)))
    var upper_diagonal_index = Int(num_upper.load_linear[1](IndexList[1](0)))

    def dispatch[
        exclude: Bool
    ]() raises {
        var input_shape,
        var lower_diagonal_index,
        var upper_diagonal_index,
        var output,
        var input_0_fn,
        imm,
    }:
        _matrix_band_part_impl[
            dtype,
            int_type,
            cond_type,
            rank,
            simd_width,
            exclude=exclude,
            target=target,
        ](
            input_0_fn,
            input_shape,
            lower_diagonal_index,
            upper_diagonal_index,
            output,
            ctx,
        )

    unswitch(exclude.load_linear[1](IndexList[1](0)) != 0, dispatch)


@always_inline
def _matrix_band_part_impl[
    dtype: DType,
    int_type: DType,
    cond_type: DType,
    rank: Int,
    simd_width: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    exclude: Bool,
    target: StaticString = "cpu",
](
    input_0_fn: InputFnType,
    input_shape: IndexList[rank],
    lower_diagonal_index: Int,
    upper_diagonal_index: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Implements the elementwise band-part copy with the `exclude` flag specialized at compile time.

    For each element, keeps it when its diagonal offset lies within
    `[lower_diagonal_index, upper_diagonal_index]` (inclusive), otherwise stores
    zero; when `exclude` is set, the mask is inverted.

    Args:
        input_0_fn: Function returning the input element at a given index.
        input_shape: Shape of the input tensor.
        lower_diagonal_index: Number of lower subdiagonals to keep (negative keeps all below).
        upper_diagonal_index: Number of upper superdiagonals to keep (negative keeps all above).
        output: Mutable output tensor receiving the band-part result.
        ctx: Device context used to launch the elementwise kernel.
    """
    comptime assert rank >= 2, "Matrix band only supports rank >=2"

    @always_inline
    def func[simd_width: Int, alignment: Int = 1](index: Coord) {var}:
        var idx = IndexList[rank]()
        comptime for i in range(rank):
            idx[i] = Int(index[i].value())

        var row = idx[rank - 2]
        var col = idx[rank - 1]

        var in_band = (
            lower_diagonal_index < 0 or (row - col) <= lower_diagonal_index
        ) and (upper_diagonal_index < 0 or (col - row) <= upper_diagonal_index)

        comptime if exclude:
            in_band = not in_band

        if in_band:
            output.store_linear(idx, input_0_fn[1, rank](idx))
        else:
            output.store_linear(idx, Scalar[dtype](0))

    elementwise[
        simd_width=1,
        target=target,
    ](func, output.layout.shape_coord(), ctx)
