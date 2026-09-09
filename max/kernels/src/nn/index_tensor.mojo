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
"""Implements numpy-style advanced tensor indexing (getitem and setitem) for CPU and GPU."""

from std.math import ceildiv
from std.sys import simd_width_of
from std.sys.info import _current_target

from nn.reshape import reshape
from max.algorithm import elementwise, sync_parallelize
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu
from layout import Coord, Idx, TileTensor, coord_to_index_list
from max.runtime.asyncrt import parallelism_level

from std.utils import IndexList


@always_inline
def index_tensor_shape[
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    batch_dims: Int,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    indices_buf: TileTensor[mut=False, indices_type, ...],
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `index_tensor` operation, and assert the
    inputs are compatible.

    Parameters:
        output_rank: Rank of the output tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        batch_dims: Batch dimensions.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.

    Returns:
        The output shape.
    """

    # TODO: Revisit when we generalize (e.g.res[: indA] vs. res[:, indA, indB]).
    if input_buf.rank <= 1 or indices_buf.rank <= 1:
        raise Error("[index_tensor] input_rank and indices_rank must be >= 2")
    if batch_dims + indices_buf.rank != input_buf.rank:
        raise Error(
            "Sum of batch_dims and indices_rank needs to equal input_rank"
        )

    # Since we pass indices without the batch_dims dimensions (since they do
    # not need to be materialized), we need to construct the indices_shape as
    # follows for the purposes of calculating the index.tensor shape:
    comptime combined_indices_rank = batch_dims + indices_buf.rank
    var indices_shape = IndexList[combined_indices_rank]()

    comptime for i in range(batch_dims):
        indices_shape[i] = Int(input_buf.layout.shape[i]().value())

    comptime for i in range(indices_buf.rank):
        indices_shape[batch_dims + i] = Int(
            indices_buf.layout.shape[i]().value()
        )

    var index_size = indices_shape[combined_indices_rank - 1]
    # TODO: Revisit when we generalize (see above TODO).
    if index_size < 2 or input_buf.rank - batch_dims < index_size:
        raise Error(
            "[index_tensor] index size must be within range [2, input_rank -"
            " batch_dims]"
        )
    # TODO: Revisit keeping when we generalize.
    if batch_dims >= combined_indices_rank:
        raise Error(
            "[index_tensor] requires (batch_dims < indices_rank + batch_dims)"
        )

    # compute and return the output shape
    var output_shape = IndexList[output_rank]()
    var next_out_dim = 0

    var input_shape = coord_to_index_list(input_buf.layout.shape_coord())

    comptime for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    comptime for i in range(batch_dims, combined_indices_rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    if indices_shape[combined_indices_rank - 1] == input_buf.rank - batch_dims:
        return output_shape

    # TODO: Revisit cases where/if this applies for generalized index_tensor.
    for i in range(
        batch_dims + indices_shape[combined_indices_rank - 1],
        len(input_shape),
    ):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===-----------------------------------------------------------------------===#
# index_tensor
# ===-----------------------------------------------------------------------===#

# TODO:
# Need to limit to cases where : is in consecutive dimensions starting from 0th.
# (so it does NOT work with non-contiguous case).
# This needs to get the TWO indices as part of the OP itself (?)
#   See if it makes sense to leave like this to be generic and lowering can
#   deal with subcases (e.g., N-D indices).
# FOLLOW-UP: Revisit all constrained and raises in dimensions and values of indices.
#        When we support more general cases like below.
# FOLLOW-UP: See if it works with 2D indices case.
# FOLLOW-UP: See example with [:, indA] indexing.
# FOLLOW-UP: Simplify if not needed to be that complex.
# Note: We could have used original gather_nd but then would need to materialize
# an unneeded huge index tensor (would broadcast to : dimension(s)).
# Note: Currently, the `_index_tensor_1d` is retained as the CPU implementation
# (see PR #38365). The `_index_tensor_impl` is introduced as the gpu implementation.
# We intend to merge `index_tensor` with the `gather_nd` operations in the future.


def index_tensor[
    dtype: DType,
    indices_type: DType,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """
    Index_tensor operation; based on modified implementation of gather_nd.

    Parameters:
        dtype: Type of data tensor.
        indices_type: Type of indices tensor.
        batch_dims: Number of batch dimensions. The gather of indexing
                    starts from dimension of data[batch_dims:].
        target: The target architecture to execute on.

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank >= 1. All index values are expected
                 to be within bounds [-s, s-1] along axis of size s. It is an
                 error if any of the index values are out of bounds.
        output: Tensor of rank data_rank + indices_rank - indices_shape[-1] - 1 - b.
        ctx: The device context as prepared by the graph compiler.

    """

    comptime if is_cpu[target]():
        return _index_tensor_1d[
            batch_dims,
            target=target,
        ](data, indices, output, Optional[DeviceContext](ctx))
    else:
        return _index_tensor_impl[
            batch_dims,
            target=target,
        ](data, indices, output, ctx)


# Note: this is an extremely specialized version of the kernel that only handles
# the [:, :, x, y] case where x and y are 1D tensors.
# Batch dims refer to the number of sliced dimensions at the beginning
def _index_tensor_1d[
    dtype: DType,
    indices_type: DType,
    //,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: Optional[DeviceContext] = None,
):
    comptime assert (
        data.flat_rank >= 2 and indices.flat_rank == 2
    ), "Constraint: data_rank >= 2 and indices_rank == 2"
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...) loads below.
    comptime assert indices.flat_rank >= 2

    var last_index_dim = Int(indices.dim(indices.rank - 1))

    assert (
        last_index_dim + batch_dims == data.rank
    ), "kernel doesn't support slicing after specified dims"

    var data_shape = coord_to_index_list(data.layout.shape_coord())
    var batch_volume: Int = 1

    comptime for i in range(batch_dims):
        batch_volume *= data_shape[i]

    # Flatten data to array of shape (batch_dim_size, data.shape[batch_dims:])
    comptime reshaped_data_rank = data.rank - batch_dims + 1
    var reshaped_data_tuple = IndexList[reshaped_data_rank]()

    reshaped_data_tuple[0] = batch_volume
    var counter = 1
    for i in range(batch_dims, data.rank):
        reshaped_data_tuple[counter] = data_shape[i]
        counter += 1

    var reshaped_data = reshape[reshaped_data_rank](
        data.make_dynamic[.int64](),
        reshaped_data_tuple,
    )

    # TODO: Find a heuristic to replace the magic number
    #       to also take into account the data size per line.
    comptime MIN_LINES = 32
    var num_threads = parallelism_level(ctx)
    var num_tasks = min(
        ceildiv(
            batch_volume,
            MIN_LINES,
        ),
        num_threads,
    )
    var work_per_thread = ceildiv(batch_volume, num_tasks)

    def calc_batch_dim(
        task_id: Int,
    ) {var work_per_thread, var batch_volume, var last_index_dim, imm}:
        # each thread gets a chunk of output embedding vectors to avoid inter-thread reduction
        var work_start = task_id * work_per_thread
        var work_end = min((task_id + 1) * work_per_thread, batch_volume)

        for i in range(work_start, work_end):
            for j in range(Int(indices.dim(0))):
                var data_coord = IndexList[reshaped_data_rank]()
                data_coord[0] = i
                for k in range(last_index_dim):
                    data_coord[k + 1] = Int(indices.load[width=1](Coord(j, k)))

                var rd_coord = Coord(data_coord)
                output.raw_store(
                    i * Int(indices.dim(0)) + j,
                    reshaped_data.load[width=1](rd_coord),
                )

    sync_parallelize(calc_batch_dim, num_tasks, ctx)


def _index_tensor_impl[
    dtype: DType,
    indices_type: DType,
    //,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime assert (
        data.flat_rank >= 2 and indices.flat_rank >= 2
    ), "Constraint: data_rank >= 2 and indices_rank >= 2"

    # This is modeled as an elementwise function mapping an index in the
    # output to an index in the input
    def index_tensor_elementwise_fn[
        simd_width: Int, alignment: Int = 1
    ](output_idx_arg: Coord) {var}:
        var output_idx = IndexList[output.rank]()
        comptime for i in range(output.rank):
            output_idx[i] = Int(output_idx_arg[i].value())
        var data_idx = IndexList[data.rank]()
        var indices_idx = IndexList[indices.rank]()
        var indices_last_dim = Int(indices.dim[indices.rank - 1]())

        # Fill in the known dimensions in our batch_dim
        comptime for i in range(batch_dims):
            data_idx[i] = output_idx[i]

        # Start filling in the index into the indices buffer
        comptime for i in range(0, indices.rank - 1):
            indices_idx[i] = output_idx[batch_dims + i]

        # walk the last dimensions, which are the slices we're gathering
        for i in range(indices_last_dim):
            indices_idx[indices.rank - 1] = i
            var coord = Coord(indices_idx)
            data_idx[batch_dims + i] = Int(indices.load[width=1](coord))

        # fill in the last slices in the input
        var num_tail_elems = data.rank - batch_dims - indices_last_dim
        var output_start = output.rank - num_tail_elems
        var src_start = indices_last_dim + batch_dims
        for i in range(0, num_tail_elems):
            data_idx[src_start + i] = output_idx[output_start + i]

        var data_coord = Coord(data_idx)
        var out_coord = Coord(output_idx)
        output.store[width=simd_width, alignment=1](
            out_coord, data.load[width=simd_width, alignment=1](data_coord)
        )

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()

    # Only use SIMD if:
    #   - the input data is contiguous
    #   - the slices at the end of the input are not scalars
    #   - the last dimension of the slices are evenly divisible by simd_width
    var slice_rank = (
        Scalar[indices.linear_idx_type](data.rank - batch_dims)
        - indices.dim[indices.rank - 1]()
    )
    var slice_last_dim = output.dim[output.rank - 1]() if slice_rank > 0 else 1

    comptime assert data.rank > 0
    var use_simd = (
        data.static_stride[data.rank - 1] == 1
        and (slice_last_dim % Scalar[output.linear_idx_type](target_simd_width))
        == 0
    )

    comptime if is_cpu[target]():
        var cpu_ctx = DeviceContext(api="cpu")
        if use_simd:
            elementwise[
                target_simd_width,
                target=target,
            ](index_tensor_elementwise_fn, output.layout.shape_coord(), cpu_ctx)
        else:
            elementwise[
                1,
                target=target,
            ](index_tensor_elementwise_fn, output.layout.shape_coord(), cpu_ctx)
    else:
        assert Bool(ctx), "Must provide DeviceContext if executing on GPU."
        var cuda_ctx = ctx.value()
        if use_simd:
            elementwise[
                target_simd_width,
                target=target,
            ](
                index_tensor_elementwise_fn,
                output.layout.shape_coord(),
                cuda_ctx,
            )
        else:
            elementwise[
                1,
                target=target,
            ](
                index_tensor_elementwise_fn,
                output.layout.shape_coord(),
                cuda_ctx,
            )


# ===-----------------------------------------------------------------------===#
# Advanced Indexing
# ===-----------------------------------------------------------------------===#
@always_inline
def _advanced_indexing_use_simd[
    start_axis: Int, num_index_tensors: Int, input_rank: Int
](read_strides: IndexList, write_strides: IndexList) -> Bool:
    """Return whether we can use vectorized loads/stores for advanced indexing

    Parameters:
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.
        input_rank: The rank of the tensor being indexed.

    Args:
        read_strides: The stride of the tensor being read from in advanced indexing.
            In `getitem` this is `input_tensor`, in `setitem` it is `update_tensor`.
        write_strides: The strides of the tensor being written to in advanced indexing.
            In `getitem` this is `out_tensor`, in `setitem` it is `input_tensor`
    """
    # We can vectorize the assignment only if:
    # - The tensors we are reading and writing to are contiguous in inner dimension
    # - We are not directly indexing the inner dimension of input
    comptime inner_dim_not_indexed = (start_axis + num_index_tensors - 1) < (
        input_rank - 1
    )
    var read_contiguous = read_strides[read_strides.size - 1] == 1
    var write_contiguous = write_strides[write_strides.size - 1] == 1
    return inner_dim_not_indexed and read_contiguous and write_contiguous


@always_inline
def advanced_indexing_getitem[
    input_rank: Int,
    index_rank: Int,
    input_type: DType,
    //,
    start_axis: Int,
    num_index_tensors: Int,
    target: StaticString,
    trace_description: StaticString,
    InputTensorFn: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: Int](IndexList[input_rank]) -> SIMD[
        dtype, width
    ],
    IndicesFn: ImplicitlyCopyable
    & RegisterPassable
    & def[indices_index: Int](IndexList[index_rank]) -> Int,
](
    out_tensor: TileTensor[mut=True, input_type, ...],
    in_tensor_strides: IndexList[input_rank],
    ctx: DeviceContext,
    input_tensor_fn: InputTensorFn,
    indices_fn: IndicesFn,
) raises:
    """Implement basic numpy-style advanced indexing.

    This is designed to be fused with other view-producing operations to
    implement full numpy-indexing semantics.

    This assumes the dimensions in `input_tensor` not indexed by index tensors
    are ":", ie selecting all indices along the slice. For example in numpy:

    ```
    # rank(indices1) == 3
    # rank(indices2) == 3
    out_tensor = input_tensor[:, :, :, indices1, indices2, :, :]
    ```

    We calculate the following for all valid valued indexing variables:

    ```
    out_tensor[a, b, c, i, j, k, d, e] = input_tensor[
        a, b, c,
        indices1[i, j, k],
        indices2[i, j, k],
        d, e
    ]
    ```

    In this example `start_axis = 3` and `num_index_tensors = 2`.

    Parameters:
        input_rank: The rank of the input tensor.
        index_rank: The rank of the indexing tensors.
        input_type: The dtype of the input tensor.
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.
        target: The target architecture to operation on.
        trace_description: For profiling, the trace name the operation will
            appear under.
        InputTensorFn: The type of the input-tensor fusion lambda.
        IndicesFn: The type of the indices fusion lambda.

    Args:
        out_tensor: The output tensor to write to.
        in_tensor_strides: The strides of the input tensor.
        ctx: The device context as prepared by the graph compiler.
        input_tensor_fn: Fusion lambda for the input tensor.
        indices_fn: Fusion lambda for the indices tensors.

    Note:
        Currently supports contiguous indexing tensors only; boolean tensor
        masks and view-fusion are not yet implemented.
    """
    comptime assert (
        out_tensor.rank == input_rank + index_rank - num_index_tensors
    )

    @always_inline
    def elementwise_fn_wrapper[
        width: Int,
        alignment: Int = 1,
    ](output_index: Coord) {var}:
        var input_index = IndexList[input_rank]()

        # Find the associated output index from input index
        comptime for input_dim in range(input_rank):
            comptime if input_dim < start_axis:
                input_index[input_dim] = Int(output_index[input_dim].value())
            elif input_dim >= start_axis + num_index_tensors:
                input_index[input_dim] = Int(
                    output_index[
                        input_dim - num_index_tensors + index_rank
                    ].value()
                )
            else:
                comptime index_tensor_offset = input_dim - start_axis
                var index_tensor_indices = IndexList[index_rank]()

                comptime for offset in range(index_rank):
                    index_tensor_indices[offset] = Int(
                        output_index[offset + start_axis].value()
                    )
                input_index[input_dim] = Int(
                    indices_fn[index_tensor_offset](index_tensor_indices)
                )

        out_tensor.store[width=width, alignment=1](
            output_index,
            input_tensor_fn[input_type, width=width](input_index),
        )

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[
        input_type, target=compile_target
    ]()
    var use_simd = _advanced_indexing_use_simd[
        start_axis, num_index_tensors, input_rank
    ](
        read_strides=in_tensor_strides,
        write_strides=coord_to_index_list(out_tensor.layout.stride_coord()),
    )
    if use_simd:
        elementwise[
            target_simd_width,
            target=target,
            _trace_description=trace_description,
        ](elementwise_fn_wrapper, out_tensor.layout.shape_coord(), ctx)
    else:
        elementwise[
            1,
            target=target,
            _trace_description=trace_description,
        ](elementwise_fn_wrapper, out_tensor.layout.shape_coord(), ctx)


@always_inline
def advanced_indexing_getitem_shape[
    input_rank: Int,
    index_rank: Int,
    //,
    start_axis: Int,
    num_index_tensors: Int,
](
    input_shape: IndexList[input_rank],
    index_shape: IndexList[index_rank],
) -> IndexList[input_rank + index_rank - num_index_tensors]:
    """Calculate the output shape from advanced indexing.

    Parameters:
        input_rank: The rank of the input tensor.
        index_rank: The rank of the indexing tensors.
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.

    Args:
        input_shape: The shape of the input tensor in the operation.
        index_shape: The shape of the indexing tensors in the operation.
    """
    comptime output_rank = input_rank + index_rank - num_index_tensors
    var answer = IndexList[output_rank]()

    comptime for i in range(output_rank):
        if i < start_axis:
            answer[i] = input_shape[i]
        elif i >= start_axis + index_rank:
            answer[i] = input_shape[i - index_rank + num_index_tensors]
        else:
            answer[i] = index_shape[i - start_axis]

    return answer


@always_inline
def advanced_indexing_setitem_inplace[
    index_rank: Int,
    updates_rank: Int,
    input_type: DType,
    //,
    start_axis: Int,
    num_index_tensors: Int,
    target: StaticString,
    trace_description: StaticString,
    UpdatesTensorFn: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: Int](IndexList[updates_rank]) -> SIMD[
        dtype, width
    ],
    IndicesFn: ImplicitlyCopyable
    & RegisterPassable
    & def[indices_index: Int](IndexList[index_rank]) -> Int,
](
    input_tensor: TileTensor[mut=True, input_type, ...],
    index_tensor_shape: IndexList[index_rank],
    updates_tensor_strides: IndexList[updates_rank],
    ctx: DeviceContext,
    updates_tensor_fn: UpdatesTensorFn,
    indices_fn: IndicesFn,
) raises:
    """Implement basic numpy-style advanced indexing with assignment.

    This is designed to be fused with other view-producing operations to
    implement full numpy-indexing semantics.

    This assumes the dimensions in `input_tensor` not indexed by index tensors
    are ":", ie selecting all indices along the slice. For example in numpy:

    ```
    # rank(indices1) == 2
    # rank(indices2) == 2
    # rank(updates) == 2
    input_tensor[:, :, :, indices1, indices2, :, :] = updates
    ```

    We calculate the following for all valid valued indexing variables:

    ```
    input_tensor[
        a, b, c,
        indices1[i, j],
        indices2[i, j],
        d, e
    ] = updates[i, j]
    ```

    In this example `start_axis = 3` and `num_index_tensors = 2`.

    In terms of implementation details, our strategy is to iterate over
    all indices over a common iteration range. The idea is we can map
    indices in this range to the write location in `input_tensor` as well
    as the data location in `updates`. An update can illustrate how this is
    possible best:

    Imagine the `input_tensor` shape is [A, B, C, D] and we have indexing
    tensors I1 and I2 with shape [M, N, K]. Assume I1 and I2 are applied
    to dimensions 1 and 2.

    I claim an appropriate common iteration range is then (A, M, N, K, D).
    Note we expect `updates` to be the shape [A, M, N, K, D]. We will show
    this by providing the mappings into `updates` and `input_tensor`:

    Consider an arbitrary set of indices in this range (a, m, n, k, d):
        - The index into `updates` is (a, m, n, k, d).
        - The index into `input_tensor` is (a, I1[m, n, k], I2[m, n, k], d).

    Parameters:
        index_rank: The rank of the indexing tensors.
        updates_rank: The rank of the updates tensor.
        input_type: The dtype of the input tensor.
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.
        target: The target architecture to operation on.
        trace_description: For profiling, the trace name the operation will
            appear under.
        UpdatesTensorFn: The type of the updates-tensor fusion lambda.
        IndicesFn: The type of the indices fusion lambda.

    Args:
        input_tensor: The input tensor being indexed into and modified in-place.
        index_tensor_shape: The shape of each index tensor.
        updates_tensor_strides: The strides of the update tensor.
        ctx: The device context as prepared by the graph compiler.
        updates_tensor_fn: Fusion lambda for the update tensor.
        indices_fn: Fusion lambda for the indices tensors.

    Note:
        Currently supports contiguous indexing tensors only; boolean tensor
        masks, view-fusion, and a unified getitem/setitem interface are not
        yet implemented.
    """

    # First calculate
    comptime iteration_rank = input_tensor.rank + index_rank - num_index_tensors
    comptime assert iteration_rank == updates_rank
    var iteration_shape = IndexList[iteration_rank]()

    # Find the common iteration space
    comptime for i in range(iteration_rank):
        comptime if i < start_axis:
            iteration_shape[i] = Int(input_tensor.layout.shape[i]().value())
        elif i >= start_axis + index_rank:
            iteration_shape[i] = Int(
                input_tensor.layout.shape[
                    i - index_rank + num_index_tensors
                ]().value()
            )
        else:
            iteration_shape[i] = index_tensor_shape[i - start_axis]

    @always_inline
    def elementwise_fn_wrapper[
        width: Int, alignment: Int = 1
    ](iteration_indices: Coord) {var}:
        var index_tensor_indices = IndexList[index_rank]()

        # Find the index into the indexing tensors from the common index
        comptime for i in range(index_rank):
            index_tensor_indices[i] = Int(
                iteration_indices[i + start_axis].value()
            )

        # Find the index into the inputs from the common index
        var input_tensor_indices = IndexList[input_tensor.rank]()

        comptime for i in range(input_tensor.rank):
            comptime if i < start_axis:
                input_tensor_indices[i] = Int(iteration_indices[i].value())
            elif i >= start_axis + num_index_tensors:
                input_tensor_indices[i] = Int(
                    iteration_indices[
                        i - num_index_tensors + index_rank
                    ].value()
                )

            else:
                comptime index_tensor_offset = i - start_axis
                input_tensor_indices[i] = Int(
                    indices_fn[index_tensor_offset](index_tensor_indices)
                )

        var input_tensor_coord = Coord(input_tensor_indices)
        var updates_indices = IndexList[updates_rank]()
        comptime for i in range(updates_rank):
            updates_indices[i] = Int(iteration_indices[i].value())
        input_tensor.store[width=width, alignment=1](
            input_tensor_coord,
            updates_tensor_fn[input_type, width=width](updates_indices),
        )

    # We can vectorize the assignment only if we are
    # not indexing in the last dimension of input.
    comptime last_indexed_dim = start_axis + num_index_tensors - 1
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[
        input_type, target=compile_target
    ]()
    var use_simd = _advanced_indexing_use_simd[
        start_axis, num_index_tensors, input_tensor.rank
    ](
        read_strides=updates_tensor_strides,
        write_strides=coord_to_index_list(input_tensor.layout.stride_coord()),
    )
    if use_simd:
        elementwise[
            target_simd_width,
            target=target,
            _trace_description=trace_description,
        ](elementwise_fn_wrapper, Coord(iteration_shape), ctx)
    else:
        elementwise[
            1,
            target=target,
            _trace_description=trace_description,
        ](elementwise_fn_wrapper, Coord(iteration_shape), ctx)
