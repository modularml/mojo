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

from max.gpu.host import DeviceContext
from std.random import random_ui64

from layout import Coord, TileTensor, Idx, row_major
from nn.gather_scatter import gather, gather_nd, gather_nd_shape, gather_shape
from nn.index_tensor import (
    _index_tensor_1d,
    _index_tensor_impl,
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
    index_tensor_shape,
)
from std.math import align_up

from std.sys import simd_width_of
from std.testing import assert_equal

from std.utils import IndexList


# TODO: It is like example 5 ONNX.
# CHECK-LABEL: test_index_tensor_DLRM
def test_index_tensor_DLRM() raises:
    print("== test_index_tensor_DLRM")

    comptime input_type = DType.int32
    comptime dim_0 = 4096
    comptime dim_1 = 9
    comptime dim_2 = 9

    comptime batch_dims = 1
    comptime index_len = 45

    comptime input_rank = 3
    comptime indices_rank = 2
    comptime output_rank = 2

    # dim_0 x dim_1 x dim_2 input tensor.
    comptime input_layout = row_major[dim_0, dim_1, dim_2]()
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var index_a = TileTensor(a_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_1) since it points do dim_1 of
    # input.
    for i in range(index_len):
        a_stack[i] = random_ui64(0, dim_1 - 1)

    # index_len-element input tensor.
    var b_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var index_b = TileTensor(b_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_2) since it points do dim_2 of
    # input.
    for i in range(index_len):
        b_stack[i] = random_ui64(0, dim_2 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n], Z[n]],
    # where x = [0, input.dim(0)), n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len.
    comptime ref_layout = row_major[dim_0, index_len]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](uninitialized=True)
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(dim_0):
        for j in range(index_len):
            ref_output[i, j] = input[i, index_a[j], index_b[j]]

    # Convert index_a, index_b (each of 1D size index_len) to a
    # 2D index_len x 2 indices TileTensor.
    # TODO: This needs to be part of the OP itself.
    var indices_stack = Array[
        UInt64, align_up(index_len * 2, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    var input_dyn = input.make_dynamic[.int64]()
    var indices_dyn = indices.make_dynamic[.int64]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        .uint64,
        batch_dims,
    ](input_dyn, indices_dyn)

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(dim_0 * index_len, simd_width_of[input_type]()),
    ](uninitialized=True)
    comptime output_static_layout = row_major[dim_0, index_len]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    _index_tensor_1d[batch_dims](input_dyn, indices_dyn, output_dyn)

    for i in range(dim_0):
        for j in range(index_len):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# Example with batch_dim = 2 (i.e., result[:, :, indexA, indexB])
# CHECK-LABEL: test_index_tensor_DLRM_batch
def test_index_tensor_DLRM_batch() raises:
    print("== test_index_tensor_DLRM_batch")

    comptime input_type = DType.int32

    comptime dim_0 = 2
    comptime dim_1 = 2
    comptime dim_3 = 3
    comptime dim_4 = 4

    comptime batch_dims = 2
    comptime index_len = 5

    comptime input_rank = 4
    comptime indices_rank = 2
    comptime output_rank = 3

    # dim_0 x dim_1 x dim_3 x dim_4 input tensor.
    comptime input_layout = row_major[dim_0, dim_1, dim_3, dim_4]()
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have two 1D tensors with index_len elements each.

    # index_len-element input tensor.
    var a_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var index_a = TileTensor(a_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_3)
    for i in range(index_len):
        a_stack[i] = random_ui64(0, dim_3 - 1)

    # index_len-element input tensor.
    var b_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var index_b = TileTensor(b_stack, row_major[index_len]())
    # Initialize with random values within [0-dim_4)
    for i in range(index_len):
        b_stack[i] = random_ui64(0, dim_4 - 1)

    # The two 1D tensors are used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, y, n] = input[x, y, Y[n], Z[n]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)),
    # n = [0, index_a.dim(0))

    # Reference output of shape dim_0 x index_len
    comptime ref_layout = row_major[dim_0, dim_1, index_len]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](uninitialized=True)
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(dim_0):
        for j in range(dim_1):
            for k in range(index_len):
                ref_output[i, j, k] = input[i, j, index_a[k], index_b[k]]

    # Convert index_a, index_b (each of 1D size index_len) to a 2D index_len x 2
    # indices TileTensor.
    var indices_stack = Array[
        UInt64, align_up(index_len * 2, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    for i in range(index_len):
        indices[i, 0] = index_a[i]
        indices[i, 1] = index_b[i]

    var input_dyn = input.make_dynamic[.int64]()
    var indices_dyn = indices.make_dynamic[.int64]()
    var output_shape = index_tensor_shape[
        output_rank,
        input_type,
        .uint64,
        batch_dims,
    ](input_dyn, indices_dyn)

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(dim_0 * dim_1 * index_len, simd_width_of[input_type]()),
    ](uninitialized=True)
    comptime output_static_layout = row_major[dim_0, dim_1, index_len]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    _index_tensor_impl[batch_dims](input_dyn, indices_dyn, output_dyn)

    for i in range(dim_0):
        for j in range(dim_1):
            for k in range(index_len):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


# TODO: It is like example 3 ONNX gather_nd.
# CHECK-LABEL: test_index_tensor_CLIPVIT
def test_index_tensor_CLIPVIT() raises:
    print("== test_index_tensor_CLIPVIT")

    comptime input_type = DType.int32
    comptime dim_0 = 2
    comptime dim_1 = 2
    comptime dim_2 = 768

    comptime batch_dims = 0
    comptime index_len = 2

    comptime input_rank = 3
    comptime indices_rank = 2
    comptime output_rank = 2

    # dim_0 x dim_1 x dim_2 input tensor.
    comptime input_layout = row_major[dim_0, dim_1, dim_2]()
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have two 2D tensors with 1 element each.

    # 1-element input tensor.
    # Initialize with [0,1]
    var a_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (i: Int) -> UInt64: UInt64(i))
    var index_a = TileTensor(a_stack, row_major[index_len]())

    # 1-element input tensor.
    # Initialize with [1,0]
    var b_stack = Array[
        UInt64, align_up(index_len, simd_width_of[DType.uint64]())
    ](fill_with=lambda (i: Int) -> UInt64: 1 if i == 0 else 0)
    var index_b = TileTensor(b_stack, row_major[index_len]())

    # Reference output of shape dim_0 x dim_2

    comptime ref_layout = row_major[dim_0, dim_2]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](
        fill_with=lambda (idx: Int) -> Scalar[input_type]: (
            input[Int(index_a[0]), Int(index_a[1]), idx % dim_2] if idx // dim_2
            == 0 else input[Int(index_b[0]), Int(index_b[1]), idx % dim_2]
        )
    )
    var ref_output = TileTensor(ref_stack, ref_layout)

    # TODO:
    # See how I need to convert separate indices to
    # combined indices TileTensor
    # to be as input to gather_nd.
    # See if it works with 2D indices case.
    # See if it works with non-contiguous case.

    # Convert index_a, index_b (each of 1D size 2) to a
    # 2D indices_len x 2 indices TileTensor
    var indices_stack = Array[
        UInt64, align_up(index_len * 2, simd_width_of[DType.uint64]())
    ](uninitialized=True)
    var indices = TileTensor(indices_stack, row_major[index_len, 2]())
    indices[0, 0] = index_a[0]
    indices[0, 1] = index_b[0]
    indices[1, 0] = index_a[1]
    indices[1, 1] = index_b[1]
    # TODO: Or index_a[0], index_a[1] and index_b[0], index_b[1]???

    var input_dyn = input.make_dynamic[.int64]()
    var indices_dyn = indices.make_dynamic[.int64]()
    var output_shape = gather_nd_shape[
        output_rank,
        input_type,
        .uint64,
        0,
    ](
        input_dyn,
        indices_dyn,
    )

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(dim_0 * dim_2, simd_width_of[input_type]()),
    ](uninitialized=True)
    comptime output_static_layout = row_major[dim_0, dim_2]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    # TODO: index_tensor works too. For batch_dims = 0 only.
    gather_nd[batch_dims, target="cpu"](
        input_dyn,
        indices_dyn,
        output_dyn,
        DeviceContext(api="cpu"),
    )

    for i in range(dim_0):
        for j in range(dim_2):
            assert_equal(output_data_buffer[i, j], ref_output[i, j])


# CHECK-LABEL: test_index_tensor_llama2_mistral
def test_index_tensor_llama2_mistral() raises:
    print("== test_index_tensor_llama2_mistral")

    comptime input_type = DType.int32
    comptime index_type = DType.uint64
    comptime dim_0 = 257
    comptime dim_1 = 128

    comptime batch_dims = 0
    comptime index_dim_0 = 1
    comptime index_dim_1 = 1

    comptime input_rank = 2
    comptime index_rank = 2
    comptime output_rank = 3

    # dim_0 x dim_1 input tensor.
    comptime input_layout = row_major[dim_0, dim_1]()
    # Initialize with sequential data for test purposes.
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_layout.product(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    var input = TileTensor(input_stack, input_layout)

    # We have one 2D tensor with index_len elements each.

    # index_len-element input tensor.
    # Initialize with one.
    comptime index_layout = row_major[index_dim_0, index_dim_1]()
    var a_stack = Array[
        UInt64, align_up(index_layout.product(), simd_width_of[DType.uint64]())
    ](fill=1)
    var index_a = TileTensor(a_stack, index_layout)

    # This is effectively a gather operation.

    # Reference output of shape index_dim_0 x index_dim_1 x dim_1.
    comptime ref_layout = row_major[index_dim_0, index_dim_1, dim_1]()
    var ref_stack = Array[
        Scalar[input_type],
        align_up(ref_layout.product(), simd_width_of[input_type]()),
    ](uninitialized=True)
    var ref_output = TileTensor(ref_stack, ref_layout)
    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                ref_output[i, j, k] = input[Int(index_a[i, j]), k]

    var input_dyn = input.make_dynamic[.int64]()
    var index_a_dyn = index_a.make_dynamic[.int64]()
    var output_shape = gather_shape[output_rank, input_type, index_type](
        input_dyn,
        index_a_dyn,
        0,
    )

    var output_data_stack = Array[
        Scalar[input_type],
        align_up(
            index_dim_0 * index_dim_1 * dim_1, simd_width_of[input_type]()
        ),
    ](uninitialized=True)
    comptime output_static_layout = row_major[index_dim_0, index_dim_1, dim_1]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    gather[axis=0](
        output_dyn,
        input_dyn,
        index_a_dyn,
        context=DeviceContext(api="cpu"),
    )

    for i in range(index_dim_0):
        for j in range(index_dim_1):
            for k in range(dim_1):
                assert_equal(output_data_buffer[i, j, k], ref_output[i, j, k])


# CHECK-LABEL: test_advanced_indexing_getitem
# Matches equivalent numpy: input[:, :, index_a, index_b]
def test_advanced_indexing_getitem(ctx: DeviceContext) raises:
    print("== test_advanced_indexing_getitem")

    # Initialize input with sequential data for test purposes.
    comptime input_type = DType.int32
    comptime input_rank = 4
    comptime input_shape = IndexList[input_rank](2, 3, 5, 6)
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_shape.flattened_length(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(i))
    comptime input_static_layout = row_major[2, 3, 5, 6]()
    var input_buffer = TileTensor(input_stack, input_static_layout)
    var input_dyn = input_buffer.make_dynamic[.int64]()

    # Create tensors for indexing in a somewhat predictable pattern
    comptime index_rank = 2
    comptime index_shape = IndexList[index_rank](2, 3)
    comptime index_type = DType.uint64
    var a_stack = Array[
        Scalar[index_type],
        align_up(index_shape.flattened_length(), simd_width_of[index_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[index_type]: UInt64(i % 5))
    var b_stack = Array[
        Scalar[index_type],
        align_up(index_shape.flattened_length(), simd_width_of[index_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[index_type]: UInt64((i + 1) % 5))
    comptime index_static_layout = row_major[2, 3]()
    var index_a = TileTensor(a_stack, index_static_layout)
    var index_b = TileTensor(b_stack, index_static_layout)
    var _index_a_dyn = index_a.make_dynamic[.int64]()
    var _index_b_dyn = index_b.make_dynamic[.int64]()
    # Create output tensor
    comptime output_rank = input_rank + index_rank - num_index_tensors
    comptime ref_shape = IndexList[output_rank](2, 3, 2, 3)
    comptime start_axis = 2
    comptime num_index_tensors = 2
    comptime output_shape = advanced_indexing_getitem_shape[
        start_axis=start_axis, num_index_tensors=num_index_tensors
    ](input_shape, index_shape)
    var output_data_stack = Array[
        Scalar[input_type],
        align_up(output_shape.flattened_length(), simd_width_of[input_type]()),
    ](uninitialized=True)
    comptime output_static_layout = row_major[2, 3, 2, 3]()
    var output_data_buffer = TileTensor(output_data_stack, output_static_layout)
    var output_dyn = output_data_buffer.make_dynamic[.int64]()

    @always_inline
    def input_tensor_fn[
        dtype: DType, width: Int
    ](idx: IndexList[input_rank]) {var input_dyn} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            input_dyn.load[width=width, alignment=1](Coord(idx))
        )

    @always_inline
    def indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) {
        var _index_a_dyn, var _index_b_dyn
    } -> Int:
        comptime if indices_index == 0:
            return Int(_index_a_dyn.load[width=1](Coord(coordinates)))
        else:
            return Int(_index_b_dyn.load[width=1](Coord(coordinates)))

    # Build input strides IndexList manually from layout
    var in_strides = IndexList[input_rank](
        Int(input_dyn.dynamic_stride(0)),
        Int(input_dyn.dynamic_stride(1)),
        Int(input_dyn.dynamic_stride(2)),
        Int(input_dyn.dynamic_stride(3)),
    )

    advanced_indexing_getitem[
        input_rank=input_rank,
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target="cpu",
        trace_description="test_advanced_indexing_getitem",
    ](
        output_dyn,
        in_strides,
        ctx,
        input_tensor_fn,
        indices_fn,
    )

    var output_stack = Array[
        Scalar[input_type],
        align_up(output_shape.flattened_length(), simd_width_of[input_type]()),
    ](uninitialized=True)
    var reference_output = TileTensor(output_stack, output_static_layout)

    reference_output[0, 0, 0, 0] = 1
    reference_output[0, 0, 0, 1] = 8
    reference_output[0, 0, 0, 2] = 15
    reference_output[0, 0, 1, 0] = 22
    reference_output[0, 0, 1, 1] = 24
    reference_output[0, 0, 1, 2] = 1

    reference_output[0, 1, 0, 0] = 31
    reference_output[0, 1, 0, 1] = 38
    reference_output[0, 1, 0, 2] = 45
    reference_output[0, 1, 1, 0] = 52
    reference_output[0, 1, 1, 1] = 54
    reference_output[0, 1, 1, 2] = 31

    reference_output[0, 2, 0, 0] = 61
    reference_output[0, 2, 0, 1] = 68
    reference_output[0, 2, 0, 2] = 75
    reference_output[0, 2, 1, 0] = 82
    reference_output[0, 2, 1, 1] = 84
    reference_output[0, 2, 1, 2] = 61

    reference_output[1, 0, 0, 0] = 91
    reference_output[1, 0, 0, 1] = 98
    reference_output[1, 0, 0, 2] = 105
    reference_output[1, 0, 1, 0] = 112
    reference_output[1, 0, 1, 1] = 114
    reference_output[1, 0, 1, 2] = 91

    reference_output[1, 1, 0, 0] = 121
    reference_output[1, 1, 0, 1] = 128
    reference_output[1, 1, 0, 2] = 135
    reference_output[1, 1, 1, 0] = 142
    reference_output[1, 1, 1, 1] = 144
    reference_output[1, 1, 1, 2] = 121

    reference_output[1, 2, 0, 0] = 151
    reference_output[1, 2, 0, 1] = 158
    reference_output[1, 2, 0, 2] = 165
    reference_output[1, 2, 1, 0] = 172
    reference_output[1, 2, 1, 1] = 174
    reference_output[1, 2, 1, 2] = 151

    for i in range(output_shape.flattened_length()):
        assert_equal(output_data_stack[i], output_stack[i])
    _ = b_stack^
    _ = a_stack^


# CHECK-LABEL: test_advanced_indexing_setitem_inplace
# Matches equivalent numpy: input[:, :, index_a, index_b] = updates
def test_advanced_indexing_setitem_inplace(ctx: DeviceContext) raises:
    print("== test_advanced_indexing_setitem_inplace")

    # Create input vector
    comptime input_type = DType.int32
    comptime input_rank = 4
    comptime input_shape = IndexList[input_rank](2, 2, 4, 4)
    # Fill with zeros
    var input_stack = Array[
        Scalar[input_type],
        align_up(input_shape.flattened_length(), simd_width_of[input_type]()),
    ](fill=0)
    comptime input_static_layout = row_major[2, 2, 4, 4]()
    var input_buffer = TileTensor(input_stack, input_static_layout)
    var input_dyn = input_buffer.make_dynamic[.int64]()

    # Create indexing tensors, ensure no pair of indices point to the same
    # location in `input` to avoid nondeterministic behavior.
    comptime index_rank = 2
    comptime num_index_tensors = 2
    comptime index_shape = IndexList[index_rank](2, 2)
    comptime index_type = DType.uint64

    var a_stack = Array[
        Scalar[index_type],
        align_up(index_shape.flattened_length(), simd_width_of[index_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[index_type]: UInt64(i % 4))
    var b_stack = Array[
        Scalar[index_type],
        align_up(index_shape.flattened_length(), simd_width_of[index_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[index_type]: UInt64((i + 1) % 4))
    comptime index_static_layout = row_major[2, 2]()
    var index_a = TileTensor(a_stack, index_static_layout)
    var index_b = TileTensor(b_stack, index_static_layout)
    var _index_a_dyn = index_a.make_dynamic[.int64]()
    var _index_b_dyn = index_b.make_dynamic[.int64]()

    # Create the updates list and set it sequential data to make it easy to read
    comptime updates_rank = 4
    comptime updates_shape = IndexList[updates_rank](2, 2, 2, 2)
    var updates_stack = Array[
        Scalar[input_type],
        align_up(updates_shape.flattened_length(), simd_width_of[input_type]()),
    ](fill_with=lambda (i: Int) -> Scalar[input_type]: Int32(1 + i))
    comptime updates_static_layout = row_major[2, 2, 2, 2]()
    var updates = TileTensor(updates_stack, updates_static_layout)
    var updates_dyn = updates.make_dynamic[.int64]()

    @always_inline
    def updates_tensor_fn[
        dtype: DType, width: Int
    ](idx: IndexList[updates_rank]) {var updates_dyn} -> SIMD[dtype, width]:
        return rebind[SIMD[dtype, width]](
            updates_dyn.load[width=width, alignment=1](Coord(idx))
        )

    @always_inline
    def indices_fn[
        indices_index: Int,
    ](coordinates: IndexList[index_rank]) {
        var _index_a_dyn, var _index_b_dyn
    } -> Int:
        comptime if indices_index == 0:
            return Int(_index_a_dyn.load[width=1](Coord(coordinates)))
        else:
            return Int(_index_b_dyn.load[width=1](Coord(coordinates)))

    # Build index shape and updates strides manually
    var idx_shape = IndexList[index_rank](
        Int(_index_a_dyn.dim(0)), Int(_index_a_dyn.dim(1))
    )
    var upd_strides = IndexList[updates_rank](
        Int(updates_dyn.dynamic_stride(0)),
        Int(updates_dyn.dynamic_stride(1)),
        Int(updates_dyn.dynamic_stride(2)),
        Int(updates_dyn.dynamic_stride(3)),
    )

    comptime start_axis = 2
    advanced_indexing_setitem_inplace[
        index_rank=index_rank,
        start_axis=start_axis,
        num_index_tensors=num_index_tensors,
        target="cpu",
        trace_description="test_advanced_indexing_setitem_inplace",
    ](
        input_dyn,
        idx_shape,
        upd_strides,
        ctx,
        updates_tensor_fn,
        indices_fn,
    )

    # Fill with zeros
    var output_stack = Array[
        Scalar[input_type],
        align_up(input_shape.flattened_length(), simd_width_of[input_type]()),
    ](fill=0)
    var reference_output = TileTensor(output_stack, input_static_layout)

    reference_output[0, 0, 0, 1] = 1
    reference_output[0, 0, 1, 2] = 2
    reference_output[0, 0, 2, 3] = 3
    reference_output[0, 0, 3, 0] = 4

    reference_output[0, 1, 0, 1] = 5
    reference_output[0, 1, 1, 2] = 6
    reference_output[0, 1, 2, 3] = 7
    reference_output[0, 1, 3, 0] = 8

    reference_output[1, 0, 0, 1] = 9
    reference_output[1, 0, 1, 2] = 10
    reference_output[1, 0, 2, 3] = 11
    reference_output[1, 0, 3, 0] = 12

    reference_output[1, 1, 0, 1] = 13
    reference_output[1, 1, 1, 2] = 14
    reference_output[1, 1, 2, 3] = 15
    reference_output[1, 1, 3, 0] = 16

    for i in range(input_shape.flattened_length()):
        assert_equal(input_stack[i], output_stack[i])

    _ = _index_a_dyn
    _ = _index_b_dyn
    _ = updates_dyn
    _ = a_stack^
    _ = b_stack^
    _ = updates_stack^
    _ = input_stack^


def main() raises:
    test_index_tensor_DLRM()
    test_index_tensor_DLRM_batch()
    test_index_tensor_CLIPVIT()
    test_index_tensor_llama2_mistral()
    with DeviceContext(api="cpu") as ctx:
        test_advanced_indexing_getitem(ctx)
        test_advanced_indexing_setitem_inplace(ctx)
