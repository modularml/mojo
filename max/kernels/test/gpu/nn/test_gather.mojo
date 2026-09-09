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
from layout import TileTensor, row_major
from nn.gather_scatter import gather


# CHECK-LABEL: test_gather
def test_gather(ctx: DeviceContext) raises:
    print("== test_gather")

    @no_inline
    @__parameter
    def _test_gather[indices_type: DType]() raises:
        comptime num_rows = 16
        comptime row_size = 4
        comptime num_indices = 16

        # Create device buffers
        var input_device = ctx.enqueue_create_buffer[.float32](
            num_rows * row_size
        )
        var indices_device = ctx.enqueue_create_buffer[indices_type](
            num_indices
        )
        var output_device = ctx.enqueue_create_buffer[.float32](
            num_indices * row_size
        )

        # Initialize input data
        with input_device.map_to_host() as input_host:
            var input_tensor = TileTensor(
                input_host, row_major[num_rows, row_size]()
            )
            for i in range(num_rows):
                for j in range(row_size):
                    input_tensor[i, j] = Float32(i)

        # Initialize indices
        with indices_device.map_to_host() as indices_host:
            var indices_tensor = TileTensor(
                indices_host, row_major[num_indices]()
            )
            for i in range(num_indices):
                indices_tensor[i] = Scalar[indices_type](i // 2)
            indices_tensor[0] = -1
            indices_tensor[1] = -num_rows

        # Create TileTensors for GPU operations
        var input_tensor = TileTensor(
            input_device, row_major[num_rows, row_size]()
        )
        var indices_tensor = TileTensor(
            indices_device, row_major[num_indices]()
        )
        var output_tensor = TileTensor(
            output_device, row_major[num_indices, row_size]()
        )

        gather[axis=0, target="gpu"](
            output_tensor.make_dynamic[.int64](),
            input_tensor.make_dynamic[.int64](),
            indices_tensor.make_dynamic[.int64](),
            context=ctx,
        )
        ctx.synchronize()

        # Read back and print results
        with output_device.map_to_host() as output_host:
            var output_tensor_host = TileTensor(
                output_host, row_major[num_indices, row_size]()
            )
            print(output_tensor_host[0, 0])
            print(output_tensor_host[1, 0])
            print(output_tensor_host[2, 0])
            print(output_tensor_host[6, 0])
            print(output_tensor_host[15, 0])

    # CHECK: 15.0
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[.int64]()


def main() raises:
    with DeviceContext() as ctx:
        test_gather(ctx)
