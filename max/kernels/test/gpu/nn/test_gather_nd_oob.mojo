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
from layout import Coord, TileTensor, row_major
from nn.gather_scatter import gather_nd, gather_nd_shape

from std.utils import IndexList


def execute_gather_nd_test[
    data_type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    batch_dims: Int,
](
    data_host_ptr: Pointer[Scalar[data_type], _],
    data_shape: IndexList[data_rank],
    indices_host_ptr: Pointer[Scalar[indices_type], _],
    indices_shape: IndexList[indices_rank],
    ctx: DeviceContext,
) raises:
    # Compute sizes
    var data_size = 1
    for i in range(data_rank):
        data_size *= data_shape[i]
    var indices_size = 1
    for i in range(indices_rank):
        indices_size *= indices_shape[i]

    # Create host TileTensors
    var data_host_tensor = TileTensor(
        data_host_ptr, row_major(Coord(data_shape))
    )
    var indices_host_tensor = TileTensor(
        indices_host_ptr, row_major(Coord(indices_shape))
    )

    # Create device buffers and copy data to them
    var data_device = ctx.enqueue_create_buffer[data_type](data_size)
    var indices_device = ctx.enqueue_create_buffer[indices_type](indices_size)

    comptime output_rank = 1

    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](
        data_host_tensor,
        indices_host_tensor,
    )

    # Compute output size
    var output_size = 1
    for i in range(output_shape.size):
        output_size *= output_shape[i]

    var actual_output_device = ctx.enqueue_create_buffer[data_type](output_size)

    ctx.enqueue_copy(data_device, data_host_ptr)
    ctx.enqueue_copy(indices_device, indices_host_ptr)

    # Create device TileTensors
    var data_device_tensor = TileTensor(
        data_device, row_major(Coord(data_shape))
    )
    var indices_device_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )
    var actual_output_tensor = TileTensor(
        actual_output_device, row_major(Coord(output_shape))
    )

    # execute the kernel
    gather_nd[batch_dims, target="gpu"](
        data_device_tensor,
        indices_device_tensor,
        actual_output_tensor,
        ctx,
    )
    # Give the kernel an opportunity to raise the error before finishing the test.
    ctx.synchronize()

    # Cleanup device buffers
    _ = data_device^
    _ = indices_device^
    _ = actual_output_device^


def test_gather_nd_oob(ctx: DeviceContext) raises:
    # Example 1
    comptime batch_dims = 0
    comptime data_rank = 2
    comptime data_type = DType.int32
    var data_shape = IndexList[data_rank](2, 2)
    var data_size = 4
    var data_host_ptr = ctx.enqueue_create_host_buffer[data_type](data_size)
    var data_tensor = TileTensor(data_host_ptr, row_major(Coord(data_shape)))

    data_tensor[0, 0] = 0
    data_tensor[0, 1] = 1
    data_tensor[1, 0] = 2
    data_tensor[1, 1] = 3

    comptime indices_rank = 2
    var indices_shape = IndexList[indices_rank](2, 2)
    var indices_size = 4
    var indices_host_ptr = ctx.enqueue_create_host_buffer[.int64](indices_size)
    var indices_tensor = TileTensor(
        indices_host_ptr, row_major(Coord(indices_shape))
    )

    indices_tensor[0, 0] = 0
    indices_tensor[0, 1] = 0
    indices_tensor[1, 0] = 1
    indices_tensor[1, 1] = 100  # wildly out of bounds

    execute_gather_nd_test[
        data_type, DType.int64, data_rank, indices_rank, batch_dims
    ](
        data_host_ptr.unsafe_ptr(),
        data_shape,
        indices_host_ptr.unsafe_ptr(),
        indices_shape,
        ctx,
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        # CHECK: {{.*}}data index out of bounds{{.*}}
        test_gather_nd_oob(ctx)
