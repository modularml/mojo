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

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.atomic import Atomic
from std.random import random_ui64

# ========================== CONFIGURATION ==========================
comptime NUM_BINS = 256


# ========================== KERNEL CODE ==========================
def histogram_kernel(
    image: UnsafePointer[UInt8, ImmutAnyOrigin],
    bins: UnsafePointer[UInt32, MutAnyOrigin],
    width: UInt32,
    height: UInt32,
):
    """GPU histogram kernel using device-level atomic operations.

    Args:
        image: Input image data.
        bins: Output histogram bins.
        width: Image width.
        height: Image height.
    """
    var i = UInt32(block_idx.x) * UInt32(256) + UInt32(thread_idx.x)

    if i < width * height:
        var b = image[Int(i)]
        _ = Atomic.fetch_add(bins + Int(b), UInt32(1))


# ========================== TEST CODE ==========================
def cpu_histogram(
    image: UnsafePointer[mut=False, UInt8, _],
    bins: UnsafePointer[mut=True, UInt32, _],
    width: UInt32,
    height: UInt32,
):
    """CPU reference histogram implementation.

    Args:
        image: Input image data.
        bins: Output histogram bins.
        width: Image width.
        height: Image height.
    """
    for i in range(Int(width * height)):
        bins[Int(image[i])] += 1


def main() raises:
    """Example usage of GPU histogram with device-level atomics."""
    var width = UInt32(1024)
    var height = UInt32(1024)
    var total_pixels = Int(width * height)

    # Host memory allocation
    var h_image = alloc[UInt8](total_pixels)
    var h_bins = alloc[UInt32](NUM_BINS)
    var h_ref = alloc[UInt32](NUM_BINS)

    # Initialize image with random values
    for i in range(total_pixels):
        h_image[i] = UInt8(random_ui64(0, NUM_BINS - 1))

    # Initialize bins to zero
    for i in range(NUM_BINS):
        h_bins[i] = 0
        h_ref[i] = 0

    # Launch kernel
    var threads_per_block = 256
    var blocks = ceildiv(total_pixels, threads_per_block)

    print(
        "Launching histogram kernel (Fig 9.6) with",
        blocks,
        "blocks and",
        threads_per_block,
        "threads",
    )

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_image = ctx.enqueue_create_buffer[.uint8](total_pixels)
        var d_bins = ctx.enqueue_create_buffer[.uint32](NUM_BINS)

        # Copy data to device
        ctx.enqueue_copy(d_image, h_image)
        ctx.enqueue_copy(d_bins, h_bins)

        # Launch kernel
        ctx.enqueue_function[histogram_kernel](
            d_image,
            d_bins,
            width,
            height,
            grid_dim=(blocks, 1, 1),
            block_dim=(threads_per_block, 1, 1),
        )

        # Copy result back to host
        ctx.enqueue_copy(h_bins, d_bins)
        ctx.synchronize()

    # Run CPU reference
    cpu_histogram(h_image, h_ref, width, height)

    # Verify results
    var errors = 0
    for i in range(NUM_BINS):
        if h_ref[i] != h_bins[i]:
            if errors < 10:
                print(
                    "Error at bin", i, ": CPU=", h_ref[i], ", GPU=", h_bins[i]
                )
            errors += 1

    # Cleanup
    h_image.free()
    h_bins.free()
    h_ref.free()

    if errors == 0:
        print("SUCCESS: All values match!")
    else:
        print("FAILED: Found", errors, "errors")
