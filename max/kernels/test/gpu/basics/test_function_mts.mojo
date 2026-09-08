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

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from extensibility import InputTensor, OutputTensor, StaticTensorSpec
from std.testing import TestSuite, assert_equal
from std.utils import IndexList

comptime WIDTH = 5
comptime HEIGHT = 10
comptime NUM_CHANNELS = 3

comptime int_dtype = DType.uint8
comptime float_dtype = DType.float32
comptime rgb_layout_orig = Layout.row_major(HEIGHT, WIDTH, NUM_CHANNELS)
comptime gray_layout_orig = Layout.row_major(HEIGHT, WIDTH)
comptime rgb_spec = StaticTensorSpec[int_dtype, 3, ...].get_unknown()
comptime rgb_layout = rgb_spec.to_layout()
comptime gray_spec = StaticTensorSpec[int_dtype, 2, ...].get_unknown()
comptime gray_layout = gray_spec.to_layout()


def color_to_grayscale(
    rgb_tensor: LayoutTensor[int_dtype, rgb_layout, MutAnyOrigin],
    gray_tensor: LayoutTensor[int_dtype, gray_layout, MutAnyOrigin],
):
    """Converting each RGB pixel to grayscale, parallelized across the output tensor on the GPU.
    """
    var row = global_idx.y
    var col = global_idx.x

    if col < WIDTH and row < HEIGHT:
        var red = rgb_tensor[row, col, 0].cast[float_dtype]()
        var green = rgb_tensor[row, col, 1].cast[float_dtype]()
        var blue = rgb_tensor[row, col, 2].cast[float_dtype]()
        var gray = 0.21 * red + 0.71 * green + 0.07 * blue

        gray_tensor[row, col] = gray.cast[int_dtype]()


def print_image(
    gray_tensor: LayoutTensor[int_dtype, gray_layout_orig, ...]
) raises:
    """A helper function to print out the grayscale channel intensities."""
    for row in range(HEIGHT):
        for col in range(WIDTH):
            var v = gray_tensor[row, col]
            if v < 100:
                print(" ", end="")
                if v < 10:
                    print(" ", end="")
            print(v, " ", end="")
        print("")


def test_color_to_grayscale() raises:
    with DeviceContext() as ctx:
        var rgb_buffer = ctx.enqueue_create_buffer[int_dtype](
            comptime (rgb_layout_orig.size())
        )
        var gray_buffer = ctx.enqueue_create_buffer[int_dtype](
            comptime (gray_layout_orig.size())
        )

        var rgb_tensor = InputTensor[static_spec=rgb_spec](
            rgb_buffer.unsafe_ptr(), IndexList[3](HEIGHT, WIDTH, NUM_CHANNELS)
        )

        # Map device buffer to host to initialize values from CPU
        with rgb_buffer.map_to_host() as host_buffer:
            var rgb_tensor = InputTensor[static_spec=rgb_spec](
                host_buffer.unsafe_ptr(),
                IndexList[3](HEIGHT, WIDTH, NUM_CHANNELS),
            ).to_layout_tensor()
            # Fill the image with initial colors.
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    rgb_tensor[row, col, 0] = UInt8(row + col)
                    rgb_tensor[row, col, 1] = UInt8(row + col + 20)
                    rgb_tensor[row, col, 2] = UInt8(row + col + 40)

        var gray_tensor = OutputTensor[static_spec=gray_spec](
            gray_buffer.unsafe_ptr(), IndexList[2](HEIGHT, WIDTH)
        )

        # The grid is divided up into blocks, making sure there's an extra
        # full block for any remainder. This hasn't been tuned for any specific
        # GPU.
        comptime BLOCK_SIZE = 16
        var num_col_blocks = ceildiv(WIDTH, BLOCK_SIZE)
        var num_row_blocks = ceildiv(HEIGHT, BLOCK_SIZE)

        # Launch the compiled function on the GPU. The target device is specified
        # first, followed by all function arguments. The last two named parameters
        # are the dimensions of the grid in blocks, and the block dimensions.
        ctx.enqueue_function[color_to_grayscale](
            rgb_tensor,
            gray_tensor,
            grid_dim=(num_col_blocks, num_row_blocks),
            block_dim=(BLOCK_SIZE, BLOCK_SIZE),
        )

        with gray_buffer.map_to_host() as host_buffer:
            var host_tensor = LayoutTensor[int_dtype, gray_layout_orig, ...](
                host_buffer
            )
            print("Resulting grayscale image:")
            print_image(host_tensor)
            assert_equal(host_tensor[0, 0], 17)
            assert_equal(host_tensor[0, 3], 19)
            assert_equal(host_tensor[5, 4], 25)
            assert_equal(host_tensor[7, 1], 24)
            assert_equal(host_tensor[9, 4], 29)

        _ = rgb_buffer
        _ = gray_buffer


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_color_to_grayscale]()

    suite^.run()
