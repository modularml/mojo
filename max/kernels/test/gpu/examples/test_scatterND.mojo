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

from std.math import ceildiv

from max.gpu import block_dim, global_idx
from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from layout.tile_tensor import stack_allocation
from std.testing import assert_false

from std.utils import IndexList

# This is DeviceAttribute.MAX_THREADS_PER_BLOCK (in ONNXRT it is a global
# with value of 256).
comptime MAX_THREADS_PER_BLOCK = 256


# TODO: Follow-up: Eliminate offsets calculations and use tensors directly.
def scatter_nd_gpu[
    dtype: DType,
    indices_type: DType,
](
    output_data_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    indices_data_ptr: MutPointer[Scalar[indices_type], MutAnyOrigin],
    element_counts_and_input_dims_ptr: MutPointer[Int64, MutAnyOrigin],
    updates_data_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    num_indices: Int,
    last_index_dimension: Int,
    num_updates_elements: Int,
):
    var id: Int = global_idx.x
    if id >= num_indices:
        return

    var element_counts_and_input_dims = TileTensor(
        element_counts_and_input_dims_ptr,
        row_major(last_index_dimension * 2),
    )

    var data_offset = 0

    var indices_start = last_index_dimension * id
    var indices_end = indices_start + last_index_dimension

    for i in range(indices_start, indices_end):
        var index = Int(indices_data_ptr.load(i))
        var element_count_dim = Int(
            element_counts_and_input_dims[i - indices_start]
        )
        var dim_value = Int(
            element_counts_and_input_dims[
                i - indices_start + last_index_dimension
            ]
        )

        # Clamp the index if out of range.
        # This would have been an error in the CPU kernel, but throwing in the CUDA EP
        # is hard. This is the approach taken by other frameworks for out of bound indices
        # in their corresponding GPU backends as well.
        # index >= -dim_value && index < dim_value
        if index >= 0:
            if index >= dim_value:
                index = dim_value - 1
        else:
            if index < -dim_value:
                index = 0
            else:
                index += dim_value

        data_offset += index * element_count_dim

    # Set updates_data_base to appropriate offset (from where to copy).
    var updates_data_base = updates_data_ptr + num_updates_elements * id
    # Set output_data_base to appropriate offset (where to copy).
    var output_data_base = output_data_ptr + data_offset

    # Start copying appropriate amount of elements.
    for i in range(num_updates_elements):
        output_data_base[i] = updates_data_base[i]


# TODO: Extend for using reduce function if needed.
def scatter_nd[
    dtype: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    updates_rank: Int,
](
    data: TileTensor[dtype, ...],
    indices: TileTensor[indices_type, ...],
    updates: TileTensor[dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """
    Implements ONNX ScatterND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND.

    Parameters:
        dtype: Type of data, updates, and output tensors.
        indices_type: Type of the indices tensor.
        data_rank: Rank of input (data) tensor (data_rank >= 1).
        indices_rank: Rank of input (data) tensor (indices_rank >= 1).
        updates_rank: Rank of updates tensor (updates_rank = data_rank +
                      indices_rank - indices_shape[-1] - 1).

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank containing indices for the scatter
                 operation.
        updates: Tensor containing values to update output tensor based on
                 indices tensor.
        output: Tensor of rank data_rank, shaped the same as data tensor.
        ctx: DeviceContext.
    """
    # Build shape arrays for iteration.
    var data_shape = IndexList[data_rank]()
    comptime for i in range(data_rank):
        data_shape[i] = Int(data.dim[i]())
    var indices_shape = IndexList[indices_rank]()
    comptime for i in range(indices_rank):
        indices_shape[i] = Int(indices.dim[i]())
    var updates_shape = IndexList[updates_rank]()
    comptime for i in range(updates_rank):
        updates_shape[i] = Int(updates.dim[i]())

    var output_shape = IndexList[data_rank]()
    comptime for i in range(data_rank):
        output_shape[i] = Int(output.dim[i]())
    if data_shape != output_shape:
        print("Input and output shapes in scatter_nd must be the same.")

    var last_shape_of_indices = indices_shape[indices_rank - 1]

    if updates_rank != data_rank + indices_rank - last_shape_of_indices - 1:
        print(
            "updates rank must be: data_rank + indices_rank -"
            " indices_shape[-1] - 1"
        )

    # Copy input data to output (appropriate elements will be updated as needed
    # by the end of scatternd kernel).
    for _i in range(output.num_elements()):
        output.ptr[_i] = data.ptr[_i]

    # Depending on r_minus_m = data_rank - last_shape_of_indices,
    # we will be copying:
    #   element (r_minus_m = 0),
    #   row (r_minus_m = 1),
    #   sheet (r_minus_m = 2),
    #   cuboid (r_minus_m = 3), etc.
    var r_minus_m = data_rank - last_shape_of_indices
    # Calculate how many elements to copy/scatter (this is from the innermost
    # dimensions, and is contiguous memory locations).
    var count_copy = 1
    for i in range(r_minus_m):
        count_copy = count_copy * data_shape[data_rank - 1 - i]

    # Calculate number of (input) data elements to copy to GPU.
    var data_count_copy = 1
    for i in range(data_rank):
        data_count_copy = data_count_copy * data_shape[data_rank - 1 - i]

    # Calculate number of indices elements to copy to GPU.
    var indices_count_copy = 1
    for i in range(indices_rank):
        indices_count_copy = (
            indices_count_copy * indices_shape[indices_rank - 1 - i]
        )

    # Calculate number of updates elements to copy to GPU.
    var updates_count_copy = 1
    for i in range(updates_rank):
        updates_count_copy = (
            updates_count_copy * updates_shape[updates_rank - 1 - i]
        )

    # Buffer below will store both input_strides and data dimensions.
    # (combine both in one to reduce number of memcpy from H->D).
    var ptr = ctx.enqueue_create_host_buffer[.int64](last_shape_of_indices * 2)
    ctx.synchronize()

    # input_strides
    # e.g., for a shape of 2, 3, 4, 5
    #       input_strides --> [3*4*5, 4*5, 5, 1]
    def input_strides_at(i: Int) {imm} -> Int64:
        var total_stride = 1
        for j in range(i + 1, data_rank):
            total_stride *= data_shape[j]
        return Int64(total_stride)

    var input_strides = Array[_, data_rank](fill_with=input_strides_at)

    for i in range(last_shape_of_indices):
        ptr[i] = input_strides[i]
        ptr[i + last_shape_of_indices] = data_shape[i]

    # Allocate and copy output data, elements_counts_and_input_dims, updates,
    # indices to GPU.
    var output_device = ctx.enqueue_create_buffer[dtype](data_count_copy)
    var element_counts_and_input_dims_device = ctx.enqueue_create_buffer[
        DType.int64
    ](last_shape_of_indices * 2)
    var updates_device = ctx.enqueue_create_buffer[dtype](updates_count_copy)
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_count_copy
    )
    ctx.enqueue_copy(output_device, output.ptr)
    ctx.enqueue_copy(element_counts_and_input_dims_device, ptr)
    ctx.enqueue_copy(updates_device, updates.ptr)
    ctx.enqueue_copy(indices_device, indices.ptr)

    # Number of indices (that is without last dimension).
    # Each thread will handle one index.
    # e.g., 3,2,3 ==> 6
    var num_indices = 1
    for i in range(indices_rank - 1):
        num_indices *= indices_shape[i]

    var num_updates_elements = count_copy
    comptime kernel = scatter_nd_gpu[dtype=dtype, indices_type=indices_type]

    ctx.enqueue_function[kernel](
        output_device,
        indices_device,
        element_counts_and_input_dims_device,
        updates_device,
        num_indices,
        last_shape_of_indices,
        num_updates_elements,
        grid_dim=(ceildiv(num_indices, MAX_THREADS_PER_BLOCK)),
        block_dim=(MAX_THREADS_PER_BLOCK),
    )

    # Copy back output data from GPU to CPU.
    ctx.enqueue_copy(output.ptr, output_device)
    ctx.synchronize()

    _ = output_device
    _ = element_counts_and_input_dims_device
    _ = updates_device
    _ = indices_device


def linear_fill[
    dtype: DType
](buf: TileTensor[mut=True, dtype, ...], elems: Span[Scalar[dtype], _]):
    assert buf.num_elements() == len(elems), "must fill all elements of tensor"

    for i in range(buf.num_elements()):
        buf.ptr[i] = elems[i]


def test_case[
    dtype: DType,
    d0: Int,
    d1: Int,
    d2: Int,
    id0: Int,
    id1: Int,
    ud0: Int,
    ud1: Int,
    ud2: Int,
](
    data_vals: Span[Scalar[dtype], _],
    indices_vals: Span[Int64, _],
    updates_vals: Span[Scalar[dtype], _],
    output_ref_vals: Span[Scalar[dtype], _],
) raises:
    var data = stack_allocation[dtype=dtype](row_major[d0, d1, d2]())
    linear_fill(data, data_vals)
    var indices = stack_allocation[dtype=DType.int64](row_major[id0, id1]())
    linear_fill(indices, indices_vals)
    var updates = stack_allocation[dtype=dtype](row_major[ud0, ud1, ud2]())
    linear_fill(updates, updates_vals)
    var output = stack_allocation[dtype=dtype](row_major[d0, d1, d2]())

    with DeviceContext() as ctx:
        scatter_nd[
            dtype, DType.int64, data_rank=3, indices_rank=2, updates_rank=3
        ](data, indices, updates, output, ctx)

    _ = data
    _ = indices
    _ = updates

    var output_ref = stack_allocation[dtype=dtype](row_major[d0, d1, d2]())
    linear_fill(output_ref, output_ref_vals)

    for i in range(output.num_elements()):
        if output_ref.ptr[i] != output.ptr[i]:
            print("FAILURE: Mismatch at idx: ", end="")
            print(i)
            assert_false(True)


def main():
    def test_scatternd_gpu():
        print("== test_scatternd_gpu")
        var data: List[Float32] = [
            # fmt: off
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            # fmt: on
        ]

        var indices: List[Int64] = [0, 2]

        var updates: List[Float32] = [
            # fmt: off
            5, 5, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
            # fmt: on
        ]

        var output_ref: List[Float32] = [
            # fmt: off
            5, 5, 5, 5,
            6, 6, 6, 6,
            7, 7, 7, 7,
            8, 8, 8, 8,
            1, 2, 3, 4,
            5, 6, 7, 8,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4, 4, 4, 4,
            8, 7, 6, 5,
            4, 3, 2, 1,
            1, 2, 3, 4,
            5, 6, 7, 8,
            # fmt: on
        ]

        _ = test_case[
            DType.float32,
            d0=4,
            d1=4,
            d2=4,
            id0=2,
            id1=1,
            ud0=2,
            ud1=4,
            ud2=4,
        ]
        (
            data,
            indices,
            updates,
            output_ref,
        )

    test_scatternd_gpu()
