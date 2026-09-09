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
"""CPU implementation of reduction operations."""

from std.math import align_down, ceildiv
from std.sys.info import simd_width_of

from std.algorithm import vectorize

from max.algorithm import sync_parallelize
from max.algorithm.functional import _get_num_workers
from std.math.math import min as _min

from std.utils.index import IndexList, StaticTuple
from std.utils.coord import Coord, coord_to_index_list

from max.algorithm.reduction import _get_nd_indices_from_flat_index


# ===-----------------------------------------------------------------------===#
# CPU reduce implementation
# ===-----------------------------------------------------------------------===#


@always_inline
def _reduce_generator_cpu[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: SIMDLength, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
    *,
    reduce_dim: Int,
](shape_coord: Coord, init: StaticTuple[Scalar[init_type], num_reductions],):
    """Reduce the given tensor using the given reduction function on CPU. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        reduce_dim: The dimension we are reducing.

    Args:
        shape_coord: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
    """

    # The inner/outer reduction helpers index the shape by a runtime
    # `reduce_dim`, which a `Coord` does not support, so convert to an
    # `IndexList` at the boundary.
    var shape = coord_to_index_list(shape_coord)

    comptime rank = shape.size

    comptime reduce_dim_normalized = (
        rank + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    comptime if shape.size == 1:
        _reduce_along_inner_dimension[
            num_reductions,
            init_type,
            input_0_fn,
            output_0_fn,
            reduce_function,
            reduce_dim=reduce_dim_normalized,
        ](shape, init)
    else:
        comptime if rank - 1 == reduce_dim_normalized:
            _reduce_along_inner_dimension[
                num_reductions,
                init_type,
                input_0_fn,
                output_0_fn,
                reduce_function,
                reduce_dim=reduce_dim_normalized,
            ](shape, init)
        else:
            _reduce_along_outer_dimension[
                num_reductions,
                init_type,
                input_0_fn,
                output_0_fn,
                reduce_function,
                reduce_dim=reduce_dim_normalized,
            ](shape, init)


def _reduce_along_inner_dimension[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: SIMDLength, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
    *,
    reduce_dim: Int,
](
    shape: IndexList[_, element_type=.int64],
    init_value: StaticTuple[Scalar[init_type], num_reductions],
):
    """Reduces the innermost (or specified) dimension of a tensor using SIMD-
    vectorized accumulation with optional parallelism across rows.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The dtype of the initial accumulator value.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to store to the output tensor.
        reduce_function: The lambda implementing the reduction.
        reduce_dim: The dimension being reduced.

    Args:
        shape: The shape of the tensor being reduced.
        init_value: The initial accumulator value for each reduction.
    """
    var total_size: Int = shape.flattened_length()
    if total_size == 0:
        return

    var reduce_dim_size = shape[reduce_dim]

    var parallelism_size: Int = total_size // reduce_dim_size

    var num_workers = _get_num_workers(total_size)

    var chunk_size = ceildiv(parallelism_size, num_workers)

    comptime unroll_factor = 8
    comptime simd_width = simd_width_of[init_type]()
    comptime unrolled_simd_width = simd_width * unroll_factor

    var unrolled_simd_compatible_size = align_down(
        reduce_dim_size, unrolled_simd_width
    )
    var simd_compatible_size = align_down(reduce_dim_size, simd_width)

    @always_inline
    @__parameter
    def simd_reduce_helper_fn[
        in_width: SIMDLength,
        out_width: Int,
    ](
        in_acc_tup: StaticTuple[SIMD[init_type, in_width], num_reductions]
    ) -> StaticTuple[SIMD[init_type, out_width], num_reductions]:
        var out_acc_tup = StaticTuple[
            SIMD[init_type, out_width], num_reductions
        ]()

        comptime for i in range(num_reductions):
            out_acc_tup[i] = in_acc_tup[i].reduce[
                reduce_function[init_type, reduction_idx=i, ...], out_width
            ]()

        return out_acc_tup

    @always_inline
    @__parameter
    def reduce_rows_unrolled(start_row: Int, end_row: Int):
        # Iterate over the non reduced dimensions.
        for flat_index in range(start_row, end_row):
            # In normal elementwise get_nd_indices skips the last dimension as
            # it is the dimension being iterated over. In our case we don't know
            # this yet so we do have to calculate the extra one.
            var indices = _get_nd_indices_from_flat_index(
                flat_index, shape, reduce_dim
            )

            @always_inline
            @__parameter
            def unrolled_reduce_helper_fn[
                width: SIMDLength,
            ](
                start: Int,
                finish: Int,
                init: StaticTuple[SIMD[init_type, width], num_reductions],
            ) -> StaticTuple[SIMD[init_type, width], num_reductions]:
                var acc = init
                for idx in range(start, finish, width):
                    indices[reduce_dim] = idx
                    var load_value = input_0_fn[init_type, width](indices)

                    comptime for i in range(num_reductions):
                        acc[i] = reduce_function[init_type, width, i](
                            load_value, acc[i]
                        )

                return acc

            # initialize our accumulator
            var acc_unrolled_simd_tup = StaticTuple[
                SIMD[
                    init_type,
                    unrolled_simd_width,
                ],
                num_reductions,
            ]()

            comptime for i in range(num_reductions):
                acc_unrolled_simd_tup[i] = SIMD[
                    init_type,
                    unrolled_simd_width,
                ](init_value[i])

            # Loop over unroll_factor*simd_width chunks.
            acc_unrolled_simd_tup = unrolled_reduce_helper_fn[
                unrolled_simd_width
            ](0, unrolled_simd_compatible_size, acc_unrolled_simd_tup)

            # Reduce to simd_width
            var acc_simd_tup = simd_reduce_helper_fn[
                unrolled_simd_width,
                simd_width,
            ](acc_unrolled_simd_tup)

            # Loop over tail simd_width chunks
            acc_simd_tup = unrolled_reduce_helper_fn[simd_width](
                unrolled_simd_compatible_size,
                simd_compatible_size,
                acc_simd_tup,
            )

            # Reduce to scalars
            var acc_scalar_tup = simd_reduce_helper_fn[
                simd_width,
                1,
            ](acc_simd_tup)

            # Loop over tail scalars
            acc_scalar_tup = unrolled_reduce_helper_fn[1](
                simd_compatible_size, reduce_dim_size, acc_scalar_tup
            )

            # Store the result back to the output.
            indices[reduce_dim] = 0
            output_0_fn(indices, acc_scalar_tup)

    @always_inline
    def reduce_rows(i: Int) {imm}:
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        var length = end_parallel_offset - start_parallel_offset
        if length <= 0:
            return

        reduce_rows_unrolled(start_parallel_offset, end_parallel_offset)

    sync_parallelize(reduce_rows, num_workers)
    _ = reduce_dim_size
    _ = parallelism_size
    _ = chunk_size
    _ = unrolled_simd_compatible_size
    _ = simd_compatible_size


def _reduce_along_outer_dimension[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: SIMDLength, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
    *,
    reduce_dim: Int,
](
    shape: IndexList[_, element_type=.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
):
    """Reduce the given tensor using the given reduction function. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to execute in parallel.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        reduce_dim: The dimension we are reducing.

    Args:
        shape: The shape of the tensor we are reducing
        init: The value to start the reduction from.
    """
    comptime rank = shape.size
    comptime dtype = init.element_type

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    comptime simd_width = simd_width_of[dtype]()

    var total_size: Int = shape.flattened_length()
    if total_size == 0:
        return

    var reduce_dim_size = shape[reduce_dim]
    var inner_dim = shape[rank - 1]

    # parallelize across slices of the input, where a slice is [reduce_dim, inner_dim]
    # the slice is composed of [reduce_dim, simd_width] chunks
    # these chunks are reduced simultaneously across the reduce_dim using simd instructions
    # and accumulation
    var parallelism_size: Int = total_size // (reduce_dim_size * inner_dim)

    var num_workers = _get_num_workers(total_size)

    var chunk_size = ceildiv(parallelism_size, num_workers)

    def reduce_slices(i: Int) {imm}:
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = _min((i + 1) * chunk_size, parallelism_size)

        var length = end_parallel_offset - start_parallel_offset

        if length <= 0:
            return

        for var slice_idx in range(start_parallel_offset, end_parallel_offset):

            @always_inline
            def reduce_chunk[simd_width: Int](inner_dim_idx: Int) {imm}:
                var acc_simd_tup = StaticTuple[
                    SIMD[init_type, simd_width], num_reductions
                ]()

                comptime for i in range(num_reductions):
                    acc_simd_tup[i] = SIMD[init_type, simd_width](init[i])

                var reduce_vector_idx = slice_idx * inner_dim + inner_dim_idx
                var indices = _get_nd_indices_from_flat_index(
                    reduce_vector_idx, shape, reduce_dim
                )
                for reduce_dim_idx in range(reduce_dim_size):
                    indices[reduce_dim] = reduce_dim_idx
                    var load_value = input_0_fn[
                        init_type, simd_width, shape.size
                    ](indices)

                    comptime for i in range(num_reductions):
                        acc_simd_tup[i] = reduce_function[
                            init_type, simd_width, i
                        ](load_value, acc_simd_tup[i])

                indices[reduce_dim] = 0
                output_0_fn[init_type, simd_width, indices.size](
                    indices, acc_simd_tup
                )

            vectorize[simd_width](inner_dim, reduce_chunk)

    sync_parallelize(reduce_slices, num_workers)
