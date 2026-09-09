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
"""CPU implementation of stencil computation."""

from std.math import ceildiv

from std.utils.index import IndexList
from std.algorithm.backend.vectorize import vectorize

from .parallelize import _get_num_workers, sync_parallelize
from max.algorithm.functional import _get_start_indices_of_nth_subvolume


# ===-----------------------------------------------------------------------===#
# stencil CPU implementation
# ===-----------------------------------------------------------------------===#


def _stencil_impl_cpu[
    shape_element_type: DType,
    input_shape_element_type: DType,
    //,
    rank: Int,
    stencil_rank: Int,
    stencil_axis: IndexList[stencil_rank, element_type=_],
    simd_width: Int,
    dtype: DType,
    map_fn: def(IndexList[stencil_rank, element_type=_]) -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ],
    map_strides: def(dim: Int) -> Int,
    load_fn: def[simd_width: Int, dtype: DType](
        IndexList[rank, element_type=_]
    ) -> SIMD[dtype, simd_width],
    compute_init_fn: def[simd_width: Int]() -> SIMD[dtype, simd_width],
    compute_fn: def[simd_width: SIMDLength](
        IndexList[rank, element_type=_],
        SIMD[dtype, simd_width],
        SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width],
    compute_finalize_fn: def[simd_width: SIMDLength](
        IndexList[rank, element_type=_], SIMD[dtype, simd_width]
    ) -> None,
](
    shape: IndexList[rank, element_type=shape_element_type],
    input_shape: IndexList[rank, element_type=input_shape_element_type],
    map_fn_closure: map_fn,
    map_strides_closure: map_strides,
    load_fn_closure: load_fn,
    compute_init_fn_closure: compute_init_fn,
    compute_fn_closure: compute_fn,
    compute_finalize_fn_closure: compute_finalize_fn,
):
    """Computes stencil operation in parallel.

    Computes output as a function that processes input stencils, stencils are
    computed as a continuous region for each output point that is determined
    by map_fn : map_fn(y) -> lower_bound, upper_bound. The boundary conditions
    for regions that fail out of the input domain are handled by load_fn.


    Parameters:
        shape_element_type: The element dtype of the shape.
        input_shape_element_type: The element dtype of the input shape.
        rank: Input and output domain rank.
        stencil_rank: Rank of stencil subdomain slice.
        stencil_axis: Stencil subdomain axes.
        simd_width: The SIMD vector width to use.
        dtype: The input and output data dtype.
        map_fn: A function that a point in the output domain to the input co-domain.
        map_strides: A function that returns the stride for the dim.
        load_fn: A function that loads a vector of simd_width from input.
        compute_init_fn: A function that initializes vector compute over the stencil.
        compute_fn: A function the process the value computed for each point in the stencil.
        compute_finalize_fn: A function that finalizes the computation of a point in the output domain given a stencil.

    Args:
        shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
        map_fn_closure: Closure mapping output points to input co-domain bounds.
        map_strides_closure: Closure returning the stride for a given dimension.
        load_fn_closure: Closure loading a SIMD vector from input.
        compute_init_fn_closure: Closure initializing the stencil accumulator.
        compute_fn_closure: Closure processing each stencil point.
        compute_finalize_fn_closure: Closure finalizing the output value.
    """
    comptime assert rank == 4, "Only stencil of rank-4 supported"
    comptime assert (
        stencil_axis[0] == 1 and stencil_axis[1] == 2
    ), "Only stencil spatial axes [1, 2] are supported"

    # If we know we will have no work, return early
    if shape[rank - 1] == 0:
        return

    var total_size = shape.flattened_length()

    var num_workers = _get_num_workers(total_size)
    var parallelism_size = total_size // shape[rank - 1]
    var chunk_size = ceildiv(parallelism_size, num_workers)

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    @always_inline
    def task_func(i: Int) {imm}:
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        var len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            var indices = _get_start_indices_of_nth_subvolume(
                parallel_offset, shape
            )

            @always_inline
            def func_wrapper[
                simd_width: Int
            ](idx: Int) {
                mut indices,
                imm input_shape,
                imm map_fn_closure,
                imm map_strides_closure,
                imm load_fn_closure,
                imm compute_init_fn_closure,
                imm compute_fn_closure,
                imm compute_finalize_fn_closure,
            }:
                indices[rank - 1] = idx
                var stencil_indices = IndexList[
                    stencil_rank, element_type=stencil_axis.element_type
                ](indices[stencil_axis[0]], indices[stencil_axis[1]])
                var bounds = map_fn_closure(stencil_indices)
                var lower_bound = bounds[0]
                var upper_bound = bounds[1]
                var step_i = map_strides_closure(0)
                var step_j = map_strides_closure(1)
                var result = compute_init_fn_closure[simd_width]()
                var input_height = input_shape[1]
                var input_width = input_shape[2]

                # In the below loops, each part corresponds to either a padded
                # side (A, B, C, D), or the main part of the input buffer (X)
                # For the padded parts we do not need to load pad value
                # (e.g., neginf for max_pool, or 0 for avg_pool) because result
                # is initialized by compute_init_fn() to the appropriate value
                # (e.g., neginf or 0 in the max_pool/avg_pool cases).
                # Loading and calculating the result for boundary locations
                # (A-D) as in:
                #   var val = pad_value
                #   result = compute_fn[simd_width](point_idx, result, val)
                # would therefore make no difference to not doing it at
                # all.

                # NOTE: The above works when padding is constant and invariant
                # to input coordinates.

                #  AAAAAAAA
                #  AAAAAAAA
                #  BBXXXXDD
                #  BBXXXXDD
                #  BBXXXXDD
                #  BBXXXXDD
                #  CCCCCCCC
                #  CCCCCCCC

                # Calculation for lower_bound below takes into account dilation
                # across rows dimension.
                # Will be the zero if dilation is 1, or the closest point >0 if
                # dilation > 1
                if lower_bound[0] < 0:
                    var mul_i = ceildiv(-lower_bound[0], step_i)
                    lower_bound[0] = lower_bound[0] + mul_i * step_i
                if lower_bound[1] < 0:
                    var mul_j = ceildiv(-lower_bound[1], step_j)
                    lower_bound[1] = lower_bound[1] + mul_j * step_j

                # Part X (inner part)
                for i in range(
                    lower_bound[0],
                    min(input_height, upper_bound[0]),
                    step_i,
                ):
                    for j in range(
                        lower_bound[1],
                        min(input_width, upper_bound[1]),
                        step_j,
                    ):
                        var point_idx = IndexList[
                            rank, element_type=shape_element_type
                        ](indices[0], i, j, indices[3])

                        var val = load_fn_closure[simd_width, dtype](point_idx)
                        result = compute_fn_closure[simd_width](
                            point_idx, result, val
                        )

                compute_finalize_fn_closure[simd_width](indices, result)

            vectorize[simd_width, unroll_factor=unroll_factor](
                shape[rank - 1], func_wrapper
            )

    sync_parallelize(task_func, num_workers)
