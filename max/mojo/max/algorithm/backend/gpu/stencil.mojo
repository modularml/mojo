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
"""GPU implementation of stencil computation."""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.math.uutils import udivmod
from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# stencil GPU implementation
# ===-----------------------------------------------------------------------===#


def _stencil_impl_gpu[
    shape_element_type: DType,
    input_shape_element_type: DType,
    //,
    rank: Int,
    stencil_rank: Int,
    stencil_axis: IndexList[stencil_rank, element_type=_],
    simd_width: Int,
    dtype: DType,
    MapFnType: ImplicitlyCopyable
    & RegisterPassable
    & def(IndexList[stencil_rank, element_type=_]) -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ],
    MapStridesType: ImplicitlyCopyable
    & RegisterPassable
    & def(dim: Int) -> Int,
    LoadFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: Int, dtype: DType](
        IndexList[rank, element_type=_]
    ) -> SIMD[dtype, simd_width],
    ComputeInitFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: Int]() -> SIMD[dtype, simd_width],
    ComputeFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: SIMDLength](
        IndexList[rank, element_type=_],
        SIMD[dtype, simd_width],
        SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width],
    ComputeFinalizeFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: SIMDLength](
        IndexList[rank, element_type=_], SIMD[dtype, simd_width]
    ) -> None,
](
    ctx: DeviceContext,
    shape: IndexList[rank, element_type=shape_element_type],
    input_shape: IndexList[rank, element_type=input_shape_element_type],
    map_func: MapFnType,
    map_strides_func: MapStridesType,
    load_func: LoadFnType,
    compute_init_func: ComputeInitFnType,
    compute_func: ComputeFnType,
    compute_finalize_func: ComputeFinalizeFnType,
) raises:
    """(Naive implementation) Computes stencil operation in parallel on GPU.

    Parameters:
        shape_element_type: The element dtype of the shape.
        input_shape_element_type: The element dtype of the input shape.
        rank: Input and output domain rank.
        stencil_rank: Rank of stencil subdomain slice.
        stencil_axis: Stencil subdomain axes.
        simd_width: The SIMD vector width to use.
        dtype: The input and output data dtype.
        MapFnType: Maps a point in the output domain to input co-domain bounds.
        MapStridesType: Returns the stride for each dimension.
        LoadFnType: Loads a SIMD vector from input.
        ComputeInitFnType: Initializes the stencil accumulator.
        ComputeFnType: Processes the value computed for each stencil point.
        ComputeFinalizeFnType: Finalizes the output value from the stencil result.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
        map_func: Closure mapping output points to input co-domain bounds.
        map_strides_func: Closure returning the stride for a given dimension.
        load_func: Closure loading a SIMD vector from input.
        compute_init_func: Closure initializing the stencil accumulator.
        compute_func: Closure processing each stencil point.
        compute_finalize_func: Closure finalizing the output value.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert rank == 4, "Only stencil of rank-4 supported"
    comptime assert (
        stencil_axis[0] == 1 and stencil_axis[1] == 2
    ), "Only stencil spatial axes [1, 2] are supported"

    # GPU kernel implementation
    @always_inline
    def stencil_kernel() {
        var shape,
        var input_shape,
        var map_func,
        var map_strides_func,
        var load_func,
        var compute_init_func,
        var compute_func,
        var compute_finalize_func,
    }:
        # Get thread indices
        var tid_x = thread_idx.x
        var tid_y = thread_idx.y
        var bid_x = block_idx.x
        var bid_y = block_idx.y
        var bid_z = block_idx.z

        # Calculate global indices
        var x = bid_x * block_dim.x + tid_x
        var y = bid_y * block_dim.y + tid_y

        # Calculate batch and channel from bid_z
        var batch_idx, channel = udivmod(bid_z, shape[3])

        # Early exit if outside bounds
        if x >= shape[2] or y >= shape[1]:
            return

        # Create output point indices with computed batch and channel
        var indices = IndexList[rank, element_type=shape_element_type](
            batch_idx, y, x, channel
        )

        # Process stencil for this point
        var stencil_indices = IndexList[
            stencil_rank, element_type=stencil_axis.element_type
        ](indices[stencil_axis[0]], indices[stencil_axis[1]])
        var bounds = map_func(stencil_indices)
        var lower_bound = bounds[0]
        var upper_bound = bounds[1]
        var step_i = map_strides_func(0)
        var step_j = map_strides_func(1)
        var result = compute_init_func[simd_width]()
        var input_height = input_shape[1]
        var input_width = input_shape[2]

        # Handle boundary conditions
        if lower_bound[0] < 0:
            var mul_i = ceildiv(-lower_bound[0], step_i)
            lower_bound[0] = lower_bound[0] + mul_i * step_i
        if lower_bound[1] < 0:
            var mul_j = ceildiv(-lower_bound[1], step_j)
            lower_bound[1] = lower_bound[1] + mul_j * step_j

        # Process stencil window
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
                var val = load_func[simd_width, dtype](point_idx)
                result = compute_func[simd_width](point_idx, result, val)

        compute_finalize_func[simd_width](indices, result)

    # Calculate grid and block dimensions
    var block_dim = (32, 32, 1)
    var grid_dim = (
        ceildiv(shape[2], block_dim[0]),  # width
        ceildiv(shape[1], block_dim[1]),  # height
        shape[0] * shape[3],  # batch_size * num_channels
    )

    # Compile and launch kernel
    ctx.enqueue_function(stencil_kernel, grid_dim=grid_dim, block_dim=block_dim)
