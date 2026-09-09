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

# Color to grayscale conversion kernel (Fig 3.4 from PMPP)
# The input image is encoded as unsigned chars [0, 255]
# Each pixel is 3 consecutive chars for the 3 channels (RGB)

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.itertools import product

# ========================== KERNEL CODE ==========================


def color_to_grayscale_kernel(
    p_out: UnsafePointer[UInt8, MutUntrackedOrigin],
    p_in: UnsafePointer[UInt8, MutUntrackedOrigin],
    width_dev: Int32,
    height_dev: Int32,
):
    """GPU kernel for color to grayscale conversion.

    Args:
        p_out: Output grayscale image (device).
        p_in: Input RGB image (device).
        width_dev: Image width in pixels.
        height_dev: Image height in pixels.
    """
    # Int is not device-passable; widen the fixed-width args.
    var width = Int(width_dev)
    var height = Int(height_dev)
    comptime CHANNELS = 3

    var col = global_idx.x
    var row = global_idx.y

    if col < width and row < height:
        # Get 1D offset for the grayscale image
        var gray_offset = row * width + col
        # One can think of the RGB image having CHANNELS
        # times more columns than the gray scale image
        var rgb_offset = gray_offset * CHANNELS
        var r = p_in[rgb_offset].cast[.float32]()  # Red value
        var g = p_in[rgb_offset + 1].cast[.float32]()  # Green value
        var b = p_in[rgb_offset + 2].cast[.float32]()  # Blue value
        # Perform the rescaling and store it
        # We multiply by floating point constants
        p_out[gray_offset] = UInt8(0.21 * r + 0.71 * g + 0.07 * b)


# ========================== TEST CODE ==========================


def color_to_grayscale(
    h_input: UnsafePointer[UInt8, MutUntrackedOrigin],
    h_output: UnsafePointer[UInt8, MutUntrackedOrigin],
    width: Int,
    height: Int,
    ctx: DeviceContext,
) raises:
    """Host function for color to grayscale conversion.

    Args:
        h_input: Input RGB image (host).
        h_output: Output grayscale image (host).
        width: Image width in pixels.
        height: Image height in pixels.
        ctx: Device context for GPU operations.
    """
    comptime CHANNELS = 3
    comptime dtype = DType.uint8

    var rgb_size = width * height * CHANNELS
    var gray_size = width * height

    # Allocate device memory
    var d_input = ctx.enqueue_create_buffer[dtype](rgb_size)
    var d_output = ctx.enqueue_create_buffer[dtype](gray_size)

    # Copy data from host to device
    ctx.enqueue_copy(d_input, h_input)

    # Configure kernel launch parameters
    var block_dim_x = 16
    var block_dim_y = 16
    var grid_dim_x = ceildiv(width, block_dim_x)
    var grid_dim_y = ceildiv(height, block_dim_y)

    # Launch kernel
    ctx.enqueue_function[color_to_grayscale_kernel](
        d_output,
        d_input,
        Int32(width),
        Int32(height),
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(block_dim_x, block_dim_y, 1),
    )

    # Copy result from device to host
    ctx.enqueue_copy(h_output, d_output)

    # Synchronize to ensure completion
    ctx.synchronize()


def color_to_grayscale_cpu(
    input: UnsafePointer[UInt8, MutUntrackedOrigin],
    output: UnsafePointer[UInt8, MutUntrackedOrigin],
    width: Int,
    height: Int,
):
    """CPU reference implementation for color to grayscale.

    Args:
        input: Input RGB image.
        output: Output grayscale image.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    comptime CHANNELS = 3

    for row, col in product(range(height), range(width)):
        var gray_offset = row * width + col
        var rgb_offset = gray_offset * CHANNELS
        var r = input[rgb_offset].cast[.float32]()
        var g = input[rgb_offset + 1].cast[.float32]()
        var b = input[rgb_offset + 2].cast[.float32]()
        output[gray_offset] = UInt8(0.21 * r + 0.71 * g + 0.07 * b)


def initialize_image(
    image: UnsafePointer[UInt8, MutUntrackedOrigin], width: Int, height: Int
):
    """Initialize a test image with gradient pattern.

    Args:
        image: Output image buffer.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    comptime CHANNELS = 3

    for row, col in product(range(height), range(width)):
        var idx = (row * width + col) * CHANNELS
        # Create a simple gradient pattern
        image[idx] = UInt8((col * 255) // width)  # R
        image[idx + 1] = UInt8((row * 255) // height)  # G
        image[idx + 2] = UInt8(((col + row) * 128) // (width + height))  # B


def verify_results(
    cpu_output: UnsafePointer[UInt8, MutUntrackedOrigin],
    gpu_output: UnsafePointer[UInt8, MutUntrackedOrigin],
    width: Int,
    height: Int,
) -> Bool:
    """Verify results between CPU and GPU implementations.

    Args:
        cpu_output: CPU computed grayscale image.
        gpu_output: GPU computed grayscale image.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        True if results match, False otherwise.
    """
    var errors = 0
    var total_pixels = width * height

    for i in range(total_pixels):
        # Allow 1 unit difference due to floating point rounding
        var diff = abs(Int(cpu_output[i]) - Int(gpu_output[i]))
        if diff > 1:
            if errors < 10:
                print(
                    "Mismatch at index",
                    i,
                    ": CPU =",
                    cpu_output[i],
                    ", GPU =",
                    gpu_output[i],
                )
            errors += 1

    if errors > 0:
        print("Total errors:", errors, "out of", total_pixels, "pixels")

    return errors == 0


def main() raises:
    # Default image dimensions
    var width = 512
    var height = 512

    print("\n=== Testing Color to Grayscale Conversion (Fig 3.4) ===")
    print("Image size:", width, "x", height, "pixels")

    comptime CHANNELS = 3
    var rgb_size = width * height * CHANNELS
    var gray_size = width * height

    print("RGB input size:", rgb_size, "bytes")
    print("Grayscale output size:", gray_size, "bytes")

    # Allocate host memory
    var h_rgb_input = alloc[UInt8](rgb_size)
    var h_gray_output_gpu = alloc[UInt8](gray_size)
    var h_gray_output_cpu = alloc[UInt8](gray_size)

    # Initialize test image
    initialize_image(h_rgb_input, width, height)

    # Run CPU version
    color_to_grayscale_cpu(h_rgb_input, h_gray_output_cpu, width, height)

    # Run GPU version
    with DeviceContext() as ctx:
        color_to_grayscale(h_rgb_input, h_gray_output_gpu, width, height, ctx)

    # Verify results
    print("Verifying results...")
    var success = verify_results(
        h_gray_output_cpu, h_gray_output_gpu, width, height
    )

    if success:
        print("✓ PASSED: GPU results match CPU results!")
    else:
        print("✗ FAILED: GPU results do not match CPU results!")

    # Cleanup
    h_rgb_input.free()
    h_gray_output_gpu.free()
    h_gray_output_cpu.free()

    print("\n===========================================")
    if success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("===========================================")
