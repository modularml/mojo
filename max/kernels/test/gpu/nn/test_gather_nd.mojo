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

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor, row_major
from nn.gather_scatter import gather_nd, gather_nd_shape
from std.testing import assert_equal

from std.utils import IndexList


def execute_gather_nd_test[
    data_type: DType,
    indices_type: DType,
    batch_dims: Int,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
](
    data_device: DeviceBuffer[data_type],
    data_shape: IndexList[data_rank],
    indices_device: DeviceBuffer[indices_type],
    indices_shape: IndexList[indices_rank],
    expected_output_device: DeviceBuffer[data_type],
    output_shape: IndexList[output_rank],
    ctx: DeviceContext,
) raises:
    # Create output device buffer
    var actual_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )

    # Create layout tensors for GPU operations
    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )
    var actual_output_tensor = TileTensor(
        actual_output_device, row_major(Coord(output_shape))
    )

    # execute the kernel
    gather_nd[batch_dims, target="gpu"](
        data_tensor,
        indices_tensor,
        actual_output_tensor,
        ctx,
    )
    ctx.synchronize()

    # check that the contents of the output are consistent
    with actual_output_device.map_to_host() as actual_host:
        with expected_output_device.map_to_host() as expected_host:
            for i in range(output_shape.flattened_length()):
                assert_equal(actual_host[i], expected_host[i])


def test_gather_nd_eg1(ctx: DeviceContext) raises:
    # Example 1
    comptime batch_dims = 0
    comptime data_type = DType.int32
    comptime indices_type = DType.int64

    var data_shape = IndexList[2](2, 2)
    var indices_shape = IndexList[2](2, 2)

    # Create device buffers
    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    # Initialize data on host
    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 2]())
        data_tensor[0, 0] = 0
        data_tensor[0, 1] = 1
        data_tensor[1, 0] = 2
        data_tensor[1, 1] = 3

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 2]())
        indices_tensor[0, 0] = 0
        indices_tensor[0, 1] = 0
        indices_tensor[1, 0] = 1
        indices_tensor[1, 1] = 1

    # Compute output shape
    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 1
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0] = 0
        expected_tensor[1] = 3

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 2, 2, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg2(ctx: DeviceContext) raises:
    # Example 2
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    var data_shape = IndexList[2](2, 2)
    var indices_shape = IndexList[2](2, 1)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 2]())
        data_tensor[0, 0] = 0
        data_tensor[0, 1] = 1
        data_tensor[1, 0] = 2
        data_tensor[1, 1] = 3

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 1]())
        indices_tensor[0, 0] = 1
        indices_tensor[1, 0] = 0

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0] = 2
        expected_tensor[0, 1] = 3
        expected_tensor[1, 0] = 0
        expected_tensor[1, 1] = 1

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 2, 2, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg3(ctx: DeviceContext) raises:
    # Example 3
    comptime batch_dims = 0
    comptime data_type = DType.float32
    comptime indices_type = DType.int64

    var data_shape = IndexList[3](2, 2, 2)
    var indices_shape = IndexList[2](2, 2)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 2, 2]())
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 2]())
        indices_tensor[0, 0] = 0
        indices_tensor[0, 1] = 1
        indices_tensor[1, 0] = 1
        indices_tensor[1, 1] = 0

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0] = 2
        expected_tensor[0, 1] = 3
        expected_tensor[1, 0] = 4
        expected_tensor[1, 1] = 5

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 3, 2, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg4(ctx: DeviceContext) raises:
    # Example 4
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    var data_shape = IndexList[3](2, 2, 2)
    var indices_shape = IndexList[3](2, 1, 2)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 2, 2]())
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 1, 2]())
        indices_tensor[0, 0, 0] = 0
        indices_tensor[0, 0, 1] = 1
        indices_tensor[1, 0, 0] = 1
        indices_tensor[1, 0, 1] = 0

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 3
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0, 0] = 2
        expected_tensor[0, 0, 1] = 3
        expected_tensor[1, 0, 0] = 4
        expected_tensor[1, 0, 1] = 5

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 3, 3, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg5(ctx: DeviceContext) raises:
    # Example 5
    comptime batch_dims = 1
    comptime data_type = DType.int32
    comptime indices_type = DType.int64

    var data_shape = IndexList[3](2, 2, 2)
    var indices_shape = IndexList[2](2, 1)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 2, 2]())
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 1]())
        indices_tensor[0, 0] = 1
        indices_tensor[1, 0] = 0

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0] = 2
        expected_tensor[0, 1] = 3
        expected_tensor[1, 0] = 4
        expected_tensor[1, 1] = 5

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 3, 2, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg6(ctx: DeviceContext) raises:
    # Example 6
    comptime batch_dims = 2
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    var data_shape = IndexList[3](2, 3, 4)
    var indices_shape = IndexList[4](2, 3, 1, 1)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 3, 4]())
        data_tensor[0, 0, 0] = 1
        data_tensor[0, 0, 1] = 2
        data_tensor[0, 0, 2] = 3
        data_tensor[0, 0, 3] = 4
        data_tensor[0, 1, 0] = 5
        data_tensor[0, 1, 1] = 6
        data_tensor[0, 1, 2] = 7
        data_tensor[0, 1, 3] = 8
        data_tensor[0, 2, 0] = 9
        data_tensor[0, 2, 1] = 10
        data_tensor[0, 2, 2] = 11
        data_tensor[0, 2, 3] = 12
        data_tensor[1, 0, 0] = 13
        data_tensor[1, 0, 1] = 14
        data_tensor[1, 0, 2] = 15
        data_tensor[1, 0, 3] = 16
        data_tensor[1, 1, 0] = 17
        data_tensor[1, 1, 1] = 18
        data_tensor[1, 1, 2] = 19
        data_tensor[1, 1, 3] = 20
        data_tensor[1, 2, 0] = 21
        data_tensor[1, 2, 1] = 22
        data_tensor[1, 2, 2] = 23
        data_tensor[1, 2, 3] = 24

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 3, 1, 1]())
        indices_tensor[0, 0, 0, 0] = 1
        indices_tensor[0, 1, 0, 0] = 0
        indices_tensor[0, 2, 0, 0] = 2
        indices_tensor[1, 0, 0, 0] = 0
        indices_tensor[1, 1, 0, 0] = 2
        indices_tensor[1, 2, 0, 0] = 2

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 3
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0, 0] = 2
        expected_tensor[0, 1, 0] = 5
        expected_tensor[0, 2, 0] = 11
        expected_tensor[1, 0, 0] = 13
        expected_tensor[1, 1, 0] = 19
        expected_tensor[1, 2, 0] = 23

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 3, 4, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg7(ctx: DeviceContext) raises:
    # Example 7
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    var data_shape = IndexList[3](2, 2, 2)
    var indices_shape = IndexList[3](2, 1, 1)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 2, 2]())
        data_tensor[0, 0, 0] = 0
        data_tensor[0, 0, 1] = 1
        data_tensor[0, 1, 0] = 2
        data_tensor[0, 1, 1] = 3
        data_tensor[1, 0, 0] = 4
        data_tensor[1, 0, 1] = 5
        data_tensor[1, 1, 0] = 6
        data_tensor[1, 1, 1] = 7

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 1, 1]())
        indices_tensor[0, 0, 0] = 0
        indices_tensor[1, 0, 0] = 1

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 4
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0, 0, 0] = 0
        expected_tensor[0, 0, 0, 1] = 1
        expected_tensor[0, 0, 1, 0] = 2
        expected_tensor[0, 0, 1, 1] = 3
        expected_tensor[1, 0, 0, 0] = 4
        expected_tensor[1, 0, 0, 1] = 5
        expected_tensor[1, 0, 1, 0] = 6
        expected_tensor[1, 0, 1, 1] = 7

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 3, 3, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def test_gather_nd_eg8(ctx: DeviceContext) raises:
    # Example 8
    comptime batch_dims = 0
    comptime data_type = DType.int8
    comptime indices_type = DType.int64

    var data_shape = IndexList[2](2, 3)
    var indices_shape = IndexList[2](2, 1)

    var data_device = ctx.enqueue_create_buffer[data_type](
        data_shape.flattened_length()
    )
    var indices_device = ctx.enqueue_create_buffer[indices_type](
        indices_shape.flattened_length()
    )

    with data_device.map_to_host() as data_host:
        var data_tensor = TileTensor(data_host, row_major[2, 3]())
        data_tensor[0, 0] = 0
        data_tensor[0, 1] = 1
        data_tensor[0, 2] = 2
        data_tensor[1, 0] = 3
        data_tensor[1, 1] = 4
        data_tensor[1, 2] = 5

    with indices_device.map_to_host() as indices_host:
        var indices_tensor = TileTensor(indices_host, row_major[2, 1]())
        indices_tensor[0, 0] = 1
        indices_tensor[1, 0] = 0

    var data_tensor = TileTensor(data_device, row_major(Coord(data_shape)))
    var indices_tensor = TileTensor(
        indices_device, row_major(Coord(indices_shape))
    )

    comptime output_rank = 2
    var output_shape = gather_nd_shape[
        output_rank,
        data_type,
        indices_type,
        batch_dims,
    ](data_tensor, indices_tensor)

    var expected_output_device = ctx.enqueue_create_buffer[data_type](
        output_shape.flattened_length()
    )
    with expected_output_device.map_to_host() as expected_host:
        var expected_tensor = TileTensor(
            expected_host, row_major(Coord(output_shape))
        )
        expected_tensor[0, 0] = 3
        expected_tensor[0, 1] = 4
        expected_tensor[0, 2] = 5
        expected_tensor[1, 0] = 0
        expected_tensor[1, 1] = 1
        expected_tensor[1, 2] = 2

    execute_gather_nd_test[
        data_type, indices_type, batch_dims, 2, 2, output_rank
    ](
        data_device,
        data_shape,
        indices_device,
        indices_shape,
        expected_output_device,
        output_shape,
        ctx,
    )


def main() raises:
    """
    Note: Examples 1-5 are from:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND.
    """

    with DeviceContext() as ctx:
        test_gather_nd_eg1(ctx)
        test_gather_nd_eg2(ctx)
        test_gather_nd_eg3(ctx)
        test_gather_nd_eg4(ctx)
        test_gather_nd_eg5(ctx)
        test_gather_nd_eg6(ctx)
        test_gather_nd_eg7(ctx)
        test_gather_nd_eg8(ctx)
