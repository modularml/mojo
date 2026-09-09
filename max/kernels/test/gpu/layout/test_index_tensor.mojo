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

from std.random import random_ui64

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Idx, TileTensor, coord_to_index_list, row_major
from nn.index_tensor import _index_tensor_impl
from std.testing import assert_equal, assert_true

from std.utils import IndexList


def execute_index_tensor_test[
    data_type: DType,
    //,
    batch_dims: Int,
](
    data_device: TileTensor[data_type, address_space=.GENERIC, ...],
    indices_device: TileTensor[address_space=.GENERIC, ...],
    expected_output_device: TileTensor[data_type, address_space=.GENERIC, ...],
    expected_output_device_buffer: DeviceBuffer[data_type],
    ctx: DeviceContext,
) raises:
    # execute the kernel
    var actual_output_device = ctx.enqueue_create_buffer[
        expected_output_device.dtype
    ](expected_output_device.num_elements())
    var actual_output_tensor = TileTensor(
        actual_output_device,
        row_major(
            expected_output_device.layout.shape_coord().make_dynamic[
                DType.int64
            ]()
        ),
    )
    # Convert all tensors to dynamic layouts before calling the kernel
    _index_tensor_impl[batch_dims, target="gpu"](
        data_device.make_dynamic[.int64](),
        indices_device.make_dynamic[.int64](),
        actual_output_tensor,
        ctx,
    )

    ctx.synchronize()

    # check that our shapes are consistent and that the contents of the output are consistent
    assert_true(
        rebind[IndexList[actual_output_tensor.rank]](
            coord_to_index_list(actual_output_tensor.layout.shape_coord())
        )
        == rebind[IndexList[actual_output_tensor.rank]](
            coord_to_index_list(expected_output_device.layout.shape_coord())
        )
    )
    with actual_output_device.map_to_host() as actual_output_host:
        with expected_output_device_buffer.map_to_host() as expected_output_host:
            for i in range(len(actual_output_host)):
                assert_equal(actual_output_host[i], expected_output_host[i])


def test_index_tensor_DLRM(ctx: DeviceContext) raises:
    print("== test_index_tensor_DLRM")

    comptime input_type = DType.int32
    comptime dim_0 = 4096
    comptime dim_1 = 9
    comptime dim_2 = 9

    comptime batch_dims = 1
    comptime index_len = 45

    # dim_0 x dim_1 x dim_2 input tensor.
    comptime input_layout = row_major((Idx[dim_0], Idx[dim_1], Idx[dim_2]))
    var input = ctx.enqueue_create_buffer[input_type](dim_0 * dim_1 * dim_2)
    var input_tensor = TileTensor(input, input_layout)

    # Initialize with sequential data for test purposes.
    with input.map_to_host() as input_host:
        for i in range(dim_0 * dim_1 * dim_2):
            input_host[i] = Int32(i)

    # We have a 2D tensor of shape (index_len, 2).
    comptime indices_layout = row_major(Idx[index_len], Idx[2])
    var indices = ctx.enqueue_create_buffer[.uint64](index_len * 2)
    with indices.map_to_host() as indices_host:
        var indices_host_tensor = TileTensor(indices_host, indices_layout)
        for i in range(index_len):
            indices_host_tensor[i, 0] = random_ui64(0, dim_1 - 1)
            indices_host_tensor[i, 1] = random_ui64(0, dim_1 - 1)
    var indices_tensor = TileTensor(indices, indices_layout)

    # The 2D tensor is used as coordinates to dimensions 1 and 2 in the
    # dim_0 x dim_1 x dim_1 input tensor. Dimension 0 is preserved.
    # output[x, n] = input[x, Y[n, 0], Y[n, 1]],
    # where x = [0, input.dim(0)), n = [0, indices.dim(0))

    # Reference output of shape dim_0 x index_len.
    comptime output_layout = row_major(Idx[dim_0], Idx[index_len])
    var ref_output = ctx.enqueue_create_buffer[input_type](dim_0 * index_len)
    with ref_output.map_to_host() as ref_output_host:
        with input.map_to_host() as input_host:
            with indices.map_to_host() as indices_host:
                var indices_tensor_host = TileTensor(
                    indices_host, indices_layout
                )
                var input_tensor_host = TileTensor(input_host, input_layout)
                var ref_output_host_tensor = TileTensor(
                    ref_output_host, output_layout
                )
                for i in range(dim_0):
                    for j in range(index_len):
                        ref_output_host_tensor[i, j] = input_tensor_host[
                            i,
                            Int(indices_tensor_host[j, 0]),
                            Int(indices_tensor_host[j, 1]),
                        ]
    var ref_output_tensor = TileTensor(ref_output, output_layout)
    execute_index_tensor_test[batch_dims](
        input_tensor,
        indices_tensor,
        ref_output_tensor.as_immut(),
        ref_output,
        ctx,
    )


def test_index_tensor_DLRM_batch(ctx: DeviceContext) raises:
    print("== test_index_tensor_DLRM_batch")

    comptime input_type = DType.int32

    comptime dim_0 = 2
    comptime dim_1 = 2
    comptime dim_2 = 3
    comptime dim_3 = 4

    comptime batch_dims = 2
    comptime index_len = 5

    # dim_0 x dim_1 x dim_2 x dim_3 input tensor.
    comptime input_layout = row_major(
        (Idx[dim_0], Idx[dim_1], Idx[dim_2], Idx[dim_3])
    )
    var input = ctx.enqueue_create_buffer[input_type](
        dim_0 * dim_1 * dim_2 * dim_3
    )
    var input_tensor = TileTensor(input, input_layout)

    # Initialize with sequential data for test purposes.
    with input.map_to_host() as input_host:
        for i in range(dim_0 * dim_1 * dim_2 * dim_3):
            input_host[i] = Int32(i)

    # We have a 2D tensor of shape (index_len, 2).
    comptime indices_layout = row_major(Idx[index_len], Idx[2])
    var indices = ctx.enqueue_create_buffer[.uint64](index_len * 2)
    with indices.map_to_host() as indices_host:
        var indices_host_tensor = TileTensor(indices_host, indices_layout)
        for i in range(index_len):
            indices_host_tensor[i, 0] = random_ui64(0, dim_2 - 1)
            indices_host_tensor[i, 1] = random_ui64(0, dim_3 - 1)
    var indices_tensor = TileTensor(indices, indices_layout)

    # The 2D tensor is used as coordinates to dimensions 2 and 3 in the
    # dim_0 x dim_1 x dim_2 x dim_3 input tensor. Dimension 0, 1 is preserved.
    # output[x, y, n] = input[x, y, Z[n, 0], Z[n, 1]],
    # where x = [0, input.dim(0)), y = [0, input.dim(1)) and n = [0, indices.dim(0))

    # Reference output of shape dim_0 x dim_1 x index_len.
    comptime output_layout = row_major((Idx[dim_0], Idx[dim_1], Idx[index_len]))
    var ref_output = ctx.enqueue_create_buffer[input_type](
        dim_0 * dim_1 * index_len
    )
    with ref_output.map_to_host() as ref_output_host:
        with input.map_to_host() as input_host:
            with indices.map_to_host() as indices_host:
                var indices_tensor_host = TileTensor(
                    indices_host, indices_layout
                )
                var input_tensor_host = TileTensor(input_host, input_layout)
                var ref_output_host_tensor = TileTensor(
                    ref_output_host, output_layout
                )
                for i in range(dim_0):
                    for j in range(dim_1):
                        for k in range(index_len):
                            ref_output_host_tensor[i, j, k] = input_tensor_host[
                                i,
                                j,
                                Int(indices_tensor_host[k, 0]),
                                Int(indices_tensor_host[k, 1]),
                            ]
    var ref_output_tensor = TileTensor(ref_output, output_layout)
    execute_index_tensor_test[batch_dims](
        input_tensor,
        indices_tensor,
        ref_output_tensor.as_immut(),
        ref_output,
        ctx,
    )


def main() raises:
    with DeviceContext() as ctx:
        test_index_tensor_DLRM(ctx)
        test_index_tensor_DLRM_batch(ctx)
