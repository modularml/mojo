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
from std.sys import has_accelerator

from std.complex import ComplexSIMD, ComplexScalar
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major

comptime GRID_WIDTH = 60
comptime GRID_HEIGHT = 25

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32

comptime MIN_X: Scalar[float_dtype] = -2.0
comptime MAX_X: Scalar[float_dtype] = 0.7
comptime MIN_Y: Scalar[float_dtype] = -1.12
comptime MAX_Y: Scalar[float_dtype] = 1.12

comptime MAX_ITERATIONS = 100

comptime layout = row_major[GRID_HEIGHT, GRID_WIDTH]()


def main() raises:
    comptime assert (
        has_accelerator()
    ), "This example requires a supported accelerator"

    # Get the context for the attached GPU
    var ctx = DeviceContext()

    # Allocate a tensor on the target device to hold the resulting set.
    var dev_buf = ctx.enqueue_create_buffer[int_dtype](comptime (layout.size()))
    var out_tensor = TileTensor(dev_buf, layout)

    # Compute how many blocks are needed in each dimension to fully cover the grid,
    # rounding up to ensure even partially filled blocks are launched.
    comptime BLOCK_SIZE = 16
    comptime COL_BLOCKS = ceildiv(GRID_WIDTH, BLOCK_SIZE)
    comptime ROW_BLOCKS = ceildiv(GRID_HEIGHT, BLOCK_SIZE)

    # Launch the Mandelbrot kernel on the GPU with a 2D grid of thread blocks.
    ctx.enqueue_function[mandelbrot](
        out_tensor,
        grid_dim=(COL_BLOCKS, ROW_BLOCKS),
        block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )
    ctx.synchronize()

    # Map the output tensor data to CPU so that we can read the results.
    with dev_buf.map_to_host() as host_buf:
        var host_tensor = TileTensor(host_buf, layout)
        draw_mandelbrot(host_tensor)


def mandelbrot(
    tensor: TileTensor[int_dtype, type_of(layout), MutAnyOrigin],
):
    """The per-element calculation of iterations to escape in the Mandelbrot set.
    """
    # Obtain the position in the grid from the X, Y thread locations.
    var row = global_idx.y
    var col = global_idx.x

    comptime SCALE_X = (MAX_X - MIN_X) / GRID_WIDTH
    comptime SCALE_Y = (MAX_Y - MIN_Y) / GRID_HEIGHT

    # Calculate the complex C corresponding to that grid location.
    var cx = MIN_X + Float32(col) * SCALE_X
    var cy = MIN_Y + Float32(row) * SCALE_Y
    var c = ComplexScalar[float_dtype](cx, cy)

    # Perform the Mandelbrot iteration loop calculation.
    var z = ComplexScalar[float_dtype](0, 0)
    var iters = Scalar[int_dtype](0)

    var in_set_mask = Scalar[.bool](True)
    for _ in range(MAX_ITERATIONS):
        if not any(in_set_mask):
            break
        in_set_mask = z.squared_norm().le(4)
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    # Write out the resulting iterations to escape.
    tensor[row, col] = iters


def draw_mandelbrot(tensor: TileTensor[int_dtype, type_of(layout), ...]) raises:
    """A helper function to visualize the Mandelbrot set in ASCII art."""
    comptime sr = StringSlice("....,c8M@jawrpogOQEPGJ")
    for row in range(GRID_HEIGHT):
        for col in range(GRID_WIDTH):
            var v = tensor[row, col]
            if v < MAX_ITERATIONS:
                var idx = Int(v % Int32(sr.byte_length()))
                var p = sr[byte=idx]
                print(p, end="")
            else:
                print(" ", end="")
        print("")
