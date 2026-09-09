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

# Image blur kernel (Fig 3.8 from PMPP)
# Applies a box blur filter to an image

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.itertools import product

# ========================== KERNEL CODE ==========================


def blur_kernel(
    input: UnsafePointer[UInt8, MutUntrackedOrigin],
    output: UnsafePointer[UInt8, MutUntrackedOrigin],
    m_dev: Int32,
    n_dev: Int32,
):
    """GPU kernel for image blur.

    Args:
        input: Input image (device).
        output: Output blurred image (device).
        m_dev: Image height (number of rows).
        n_dev: Image width (number of columns).
    """
    # Int is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    comptime BLUR_SIZE = 3

    var row = global_idx.y
    var col = global_idx.x
    var idx = row * n + col

    if row < m and col < n:
        var pixval = 0
        var total_pixels = 0

        for drow in range(-BLUR_SIZE, BLUR_SIZE + 1):
            for dcol in range(-BLUR_SIZE, BLUR_SIZE + 1):
                var cur_row = row + drow
                var curr_col = col + dcol

                if (
                    0 <= cur_row
                    and cur_row < m
                    and 0 <= curr_col
                    and curr_col < n
                ):
                    pixval += Int(input[cur_row * n + curr_col])
                    total_pixels += 1

        output[idx] = UInt8(Float32(pixval) / Float32(total_pixels))


# ========================== TEST CODE ==========================


def cpu_blur(
    h_in: UnsafePointer[UInt8, MutUntrackedOrigin],
    h_ref: UnsafePointer[UInt8, MutUntrackedOrigin],
    n: Int,
    m: Int,
):
    """CPU reference implementation for image blur.

    Args:
        h_in: Input image.
        h_ref: Output reference blurred image.
        n: Image width (number of columns).
        m: Image height (number of rows).
    """
    comptime BLUR_SIZE = 3

    for i, j in product(range(m), range(n)):
        # Have to compute the average of the surrounding pixels
        var curr_idx = i * n + j
        var pixval = 0
        var total_pixels = 0

        for drow, dcol in product(
            range(-BLUR_SIZE, BLUR_SIZE + 1),
            range(-BLUR_SIZE, BLUR_SIZE + 1),
        ):
            # Get new row and column indices
            var n_row = i + drow
            var n_col = j + dcol

            # Check this is in range
            if 0 <= n_row and n_row < m and 0 <= n_col and n_col < n:
                pixval += Int(h_in[n_row * n + n_col])
                total_pixels += 1

        var avg = UInt8(Float32(pixval) / Float32(total_pixels))
        h_ref[curr_idx] = avg


def initialize(h_in: UnsafePointer[UInt8, MutUntrackedOrigin], size: Int):
    """Initialize input array with test data.

    Args:
        h_in: Input array to initialize.
        size: Size of the array.
    """
    for i in range(size):
        h_in[i] = UInt8(i % 256)


def main() raises:
    var m = 2048  # Height
    var n = 1024  # Width

    var total_pixels = m * n

    # Allocate host memory
    var h_in = alloc[UInt8](total_pixels)
    var h_out = alloc[UInt8](total_pixels)
    var h_ref = alloc[UInt8](total_pixels)

    # Initialize input
    initialize(h_in, total_pixels)

    # Run GPU blur
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8

        # Allocate device memory
        var d_in = ctx.enqueue_create_buffer[dtype](total_pixels)
        var d_out = ctx.enqueue_create_buffer[dtype](total_pixels)

        # Copy data from host to device
        ctx.enqueue_copy(d_in, h_in)

        # Configure kernel launch parameters
        var block_dim_x = 16
        var block_dim_y = 16
        var grid_dim_x = ceildiv(n, block_dim_x)
        var grid_dim_y = ceildiv(m, block_dim_y)

        # Launch kernel
        ctx.enqueue_function[blur_kernel](
            d_in,
            d_out,
            Int32(m),
            Int32(n),
            grid_dim=(grid_dim_x, grid_dim_y, 1),
            block_dim=(block_dim_x, block_dim_y, 1),
        )

        # Copy result from device to host
        ctx.enqueue_copy(h_out, d_out)

        # Synchronize to ensure completion
        ctx.synchronize()

    # Run CPU blur
    cpu_blur(h_in, h_ref, n, m)

    # Verify results
    var errors = 0
    for i in range(total_pixels):
        if h_out[i] != h_ref[i]:
            if errors < 10:
                print("Error at index", i, ":", h_out[i], "!=", h_ref[i])
            errors += 1

    if errors > 0:
        print("Total errors:", errors)
    else:
        print("Success")

    # Cleanup
    h_in.free()
    h_out.free()
    h_ref.free()
