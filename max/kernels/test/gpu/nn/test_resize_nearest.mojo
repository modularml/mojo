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

from layout import TileTensor, coord, row_major
from max.gpu.host import DeviceContext
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_nearest_neighbor,
)
from std.testing import assert_equal


def main() raises:
    with DeviceContext() as ctx:
        comptime input_shape = coord[1, 1, 2, 2]
        comptime output_shape = coord[1, 1, 4, 4]

        var input_host = ctx.enqueue_create_host_buffer[DType.float32](4)
        var output_host = ctx.enqueue_create_host_buffer[DType.float32](16)
        ctx.synchronize()

        for i in range(4):
            input_host[i] = Scalar[DType.float32](i + 1)

        var input_device = ctx.enqueue_create_buffer[DType.float32](4)
        var output_device = ctx.enqueue_create_buffer[DType.float32](16)
        ctx.enqueue_copy(input_device, input_host)

        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel,
            RoundMode.HalfDown,
            target="gpu",
        ](
            TileTensor(input_device, row_major(input_shape)),
            TileTensor(output_device, row_major(output_shape)),
            ctx,
        )

        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # A 2x upsample repeats every source value in a 2x2 block.
        for row in range(4):
            for col in range(4):
                var expected = Scalar[DType.float32](
                    (row // 2) * 2 + col // 2 + 1
                )
                assert_equal(output_host[row * 4 + col], expected)

        # Half-pixel coordinates begin at -0.25 during a 2x upsample. Floor
        # rounds that to -1, so the kernel must clamp it to the first element.
        resize_nearest_neighbor[
            CoordinateTransformationMode.HalfPixel,
            RoundMode.Floor,
            target="gpu",
        ](
            TileTensor(input_device, row_major(input_shape)),
            TileTensor(output_device, row_major(output_shape)),
            ctx,
        )

        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        for row in range(4):
            for col in range(4):
                var expected = Scalar[DType.float32](
                    (row // 3) * 2 + col // 3 + 1
                )
                assert_equal(output_host[row * 4 + col], expected)

        _ = input_device
        _ = output_device
