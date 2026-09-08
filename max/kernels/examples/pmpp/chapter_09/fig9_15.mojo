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
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.math import ceildiv
from std.atomic import Atomic
from std.random import random_ui64

# ========================== CONFIGURATION ==========================
comptime NUM_BINS = 256
comptime COARSE_FACTOR = 4


# ========================== KERNEL CODE ==========================
def histogram_kernel(
    image: UnsafePointer[UInt8, ImmutAnyOrigin],
    bins: UnsafePointer[UInt32, MutAnyOrigin],
    width: UInt32,
    height: UInt32,
):
    """GPU histogram kernel with thread-level privatization.

    Args:
        image: Input image data.
        bins: Output histogram bins (global).
        width: Image width.
        height: Image height.
    """
    # Allocate shared memory for private bins
    var bins_s = unsafe_stack_allocation[
        NUM_BINS,
        UInt32,
        address_space=.SHARED,
    ]()

    # Initialize shared memory bins to zero
    var b = UInt32(thread_idx.x)
    while b < NUM_BINS:
        bins_s[Int(b)] = UInt32(0)
        b += UInt32(256)

    barrier()

    # Thread-level privatization with aggregation
    var segment = COARSE_FACTOR * UInt32(block_idx.x) * UInt32(256)
    if segment + UInt32(thread_idx.x) < width * height:
        var bin = image[Int(segment + UInt32(thread_idx.x))]
        var bin_r = UInt32(1)

        for c in range(1, COARSE_FACTOR):
            var i = segment + UInt32(c) * UInt32(256) + UInt32(thread_idx.x)
            if i < width * height:
                var bin_next = image[Int(i)]
                if bin_next == bin:
                    bin_r += 1
                else:
                    _ = Atomic.fetch_add(bins_s + Int(bin), bin_r)
                    bin = bin_next
                    bin_r = 1

        # Add final accumulated count
        _ = Atomic.fetch_add(bins_s + Int(bin), bin_r)

    barrier()

    # Aggregate shared bins to global bins
    b = UInt32(thread_idx.x)
    while b < NUM_BINS:
        if bins_s[Int(b)] > 0:
            _ = Atomic.fetch_add(bins + Int(b), bins_s[Int(b)])
        b += UInt32(256)


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
    """Example usage of GPU histogram with thread-level privatization."""
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

    # Launch kernel with thread-level privatization
    var threads_per_block = 256
    var blocks = ceildiv(total_pixels, threads_per_block * COARSE_FACTOR)
    print(
        "Launching histogram kernel (Fig 9.15) with",
        blocks,
        "blocks and",
        threads_per_block,
        "threads",
    )
    print(
        "Thread-level privatization (COARSE_FACTOR="
        + String(COARSE_FACTOR)
        + ")"
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
