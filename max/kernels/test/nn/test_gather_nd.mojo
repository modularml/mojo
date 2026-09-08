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


# CHECK-LABEL: test_gather_nd
def main():
    """Note: Examples 1-5 are from.

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    """

    print("test_gather_nd")

    def test_gather_nd_eg1() raises:
        # Example 1
        comptime batch_dims = 0
        comptime data_type = DType.int32
        var data_stack: Array[Scalar[data_type], _] = [0, 1, 2, 3]
        var data = TileTensor(data_stack, row_major[2, 2]())

        var indices_stack: Array[Int64, _] = [0, 0, 1, 1]
        var indices = TileTensor(indices_stack, row_major[2, 2]())

        comptime output_rank = 1
        var output_shape = gather_nd_shape[
            output_rank, data_type, .int64, batch_dims
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 2](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:", output_data_buffer[0], ",", output_data_buffer[1]
        )

    def test_gather_nd_eg2() raises:
        # Example 2
        comptime batch_dims = 0
        comptime data_type = DType.int8
        var data_stack: Array[Scalar[data_type], _] = [0, 1, 2, 3]
        var data = TileTensor(data_stack, row_major[2, 2]())

        var indices_stack: Array[Int64, _] = [1, 0]
        var indices = TileTensor(indices_stack, row_major[2, 1]())

        comptime output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank,
            data_type,
            .int64,
            batch_dims,
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 4](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
        )

    def test_gather_nd_eg3() raises:
        # Example 3
        comptime batch_dims = 0
        comptime data_type = DType.float32
        var data_stack: Array[Scalar[data_type], _] = [0, 1, 2, 3, 4, 5, 6, 7]
        var data = TileTensor(data_stack, row_major[2, 2, 2]())

        var indices_stack = Array[Int64, 4](fill={})
        var indices = TileTensor(indices_stack, row_major[2, 2]())

        indices[0, 0] = 0
        indices[0, 1] = 1
        indices[1, 0] = 1
        indices[1, 1] = 0

        comptime output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank, data_type, .int64, batch_dims
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 4](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
        )

    def test_gather_nd_eg4() raises:
        # Example 4
        comptime batch_dims = 0
        comptime data_type = DType.int8
        var data_stack = Array[Scalar[data_type], 8](
            fill_with=lambda (i: Int) -> Scalar[data_type]: Scalar[data_type](i)
        )
        var data = TileTensor(data_stack, row_major[2, 2, 2]())

        var indices_stack = Array[Int64, 4](fill={})
        var indices = TileTensor(indices_stack, row_major[2, 1, 2]())

        indices[0, 0, 0] = 0
        indices[0, 0, 1] = 1
        indices[1, 0, 0] = 1
        indices[1, 0, 1] = 0

        comptime output_rank = 3
        var output_shape = gather_nd_shape[
            output_rank, data_type, .int64, batch_dims
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 4](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0, 0],
            ",",
            output_data_buffer[0, 0, 1],
            ",",
            output_data_buffer[1, 0, 0],
            ",",
            output_data_buffer[1, 0, 1],
        )

    def test_gather_nd_eg5() raises:
        # Example 5
        comptime batch_dims = 1
        comptime data_type = DType.int32
        var data_stack = Array[Scalar[data_type], 8](
            fill_with=lambda (i: Int) -> Scalar[data_type]: Scalar[data_type](i)
        )
        var data = TileTensor(data_stack, row_major[2, 2, 2]())

        var indices_stack: Array[Int64, _] = [1, 0]
        var indices = TileTensor(indices_stack, row_major[2, 1]())

        comptime output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank, data_type, .int64, batch_dims
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 4](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
        )

    def test_gather_nd_eg6() raises:
        # Example 6
        comptime batch_dims = 2
        comptime data_type = DType.int8
        var data_stack = Array[Scalar[data_type], 2 * 3 * 4](
            fill_with=lambda (i: Int) -> Scalar[data_type]: Scalar[data_type](
                i + 1
            )
        )
        var data = TileTensor(data_stack, row_major[2, 3, 4]())

        var indices_stack = Array[Int64, 2 * 3](fill={})
        var indices = TileTensor(indices_stack, row_major[2, 3, 1, 1]())

        indices[0, 0, 0, 0] = 1
        indices[0, 1, 0, 0] = 0
        indices[0, 2, 0, 0] = 2
        indices[1, 0, 0, 0] = 0
        indices[1, 1, 0, 0] = 2
        indices[1, 2, 0, 0] = 2

        comptime output_rank = 3
        var output_shape = gather_nd_shape[
            output_rank, data_type, .int64, batch_dims
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 6](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0, 0],
            ",",
            output_data_buffer[0, 1, 0],
            ",",
            output_data_buffer[0, 2, 0],
            ",",
            output_data_buffer[1, 0, 0],
            ",",
            output_data_buffer[1, 1, 0],
            ",",
            output_data_buffer[1, 2, 0],
        )

    def test_gather_nd_eg7() raises:
        # Example 4
        comptime batch_dims = 0
        comptime data_type = DType.int8
        var data_stack = Array[Scalar[data_type], 8](
            fill_with=lambda (i: Int) -> Scalar[data_type]: Scalar[data_type](i)
        )
        var data = TileTensor(data_stack, row_major[2, 2, 2]())

        var indices_stack: Array[Int64, _] = [0, 1]
        var indices = TileTensor(indices_stack, row_major[2, 1, 1]())

        comptime output_rank = 4
        var output_shape = gather_nd_shape[
            output_rank, data_type, .int64, batch_dims
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 8](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )

        print(
            "Output buffer:",
            output_data_buffer[0, 0, 0, 0],
            ",",
            output_data_buffer[0, 0, 0, 1],
            ",",
            output_data_buffer[0, 0, 1, 0],
            ",",
            output_data_buffer[0, 0, 1, 1],
            ",",
            output_data_buffer[1, 0, 0, 0],
            ",",
            output_data_buffer[1, 0, 0, 1],
            ",",
            output_data_buffer[1, 0, 1, 0],
            ",",
            output_data_buffer[1, 0, 1, 1],
        )

    def test_gather_nd_eg8() raises:
        # Example 2
        comptime batch_dims = 0
        comptime data_type = DType.int8
        var data_stack = Array[Scalar[data_type], 6](
            fill_with=lambda (i: Int) -> Scalar[data_type]: Scalar[data_type](i)
        )
        var data = TileTensor(data_stack, row_major[2, 3]())

        var indices_stack: Array[Int64, _] = [1, 0]
        var indices = TileTensor(indices_stack, row_major[2, 1]())

        comptime output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank,
            data_type,
            .int64,
            batch_dims,
        ](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
        )
        print("Output shape: ", output_shape)

        var output_data_data = Array[Scalar[data_type], 6](fill={})
        var output_data_buffer = TileTensor(
            output_data_data, row_major(Coord(output_shape))
        )
        gather_nd[batch_dims](
            data.make_dynamic[.int64](),
            indices.make_dynamic[.int64](),
            output_data_buffer,
            DeviceContext(api="cpu"),
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[0, 2],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
            ",",
            output_data_buffer[1, 2],
        )

    try:
        # CHECK: Output shape:  (2,)
        # CHECK: Output buffer: 0 , 3
        test_gather_nd_eg1()
        # CHECK: Output shape:  (2, 2)
        # CHECK: Output buffer: 2 , 3 , 0 , 1
        test_gather_nd_eg2()
        # CHECK: Output shape:  (2, 2)
        # CHECK: Output buffer: 2.0 , 3.0 , 4.0 , 5.0
        test_gather_nd_eg3()
        # CHECK: Output shape:  (2, 1, 2)
        # CHECK: Output buffer: 2 , 3 , 4 , 5
        test_gather_nd_eg4()
        # CHECK: Output shape:  (2, 2)
        # CHECK: Output buffer: 2 , 3 , 4 , 5
        test_gather_nd_eg5()
        # CHECK: Output shape:  (2, 3, 1)
        # CHECK: Output buffer: 2 , 5 , 11 , 13 , 19 , 23
        test_gather_nd_eg6()
        # CHECK: Output shape:  (2, 1, 2, 2)
        # CHECK: Output buffer: 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7
        test_gather_nd_eg7()
        # CHECK: Output shape:  (2, 3)
        # CHECK: Output buffer: 3 , 4 , 5 , 0 , 1 , 2
        test_gather_nd_eg8()
    except e:
        print(e)
