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

from max.gpu import block_idx, thread_idx, block_dim, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from max.gpu.primitives.warp import shuffle_down
from max.gpu.primitives.id import (
    lane_id,
    warp_id,
)
from std.atomic import Atomic
from std.random import random_float64
from std.math import abs, ceildiv
from std.bit import log2_floor

# ========================== CONFIGURATION ==========================
comptime BLOCK_DIM = 256


# ========================== KERNEL CODE ==========================
def warp_reduce(val: Float32) -> Float32:
    """Warp-level reduction using shuffle operations (loop unrolled).

    Args:
        val: Value from current thread.

    Returns:
        Sum of values across the warp (only valid in lane 0).
    """
    var partial_sum = val

    # Unrolled warp reduction
    comptime LOG2_WARP_SIZE = log2_floor(WARP_SIZE)

    comptime for i in range(LOG2_WARP_SIZE):
        comptime offset = 1 << (LOG2_WARP_SIZE - 1 - i)
        partial_sum += shuffle_down(partial_sum, UInt32(offset))

    return partial_sum


def reduce_kernel(
    input: UnsafePointer[Float32, ImmutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: UInt32,
):
    """Multi-block sum reduction kernel using atomics.

    Args:
        input: Input array (in global memory).
        output: Output scalar for the sum result.
        N: Total number of elements.
    """
    var segment = 2 * UInt32(block_dim.x) * UInt32(block_idx.x)
    var i = segment + UInt32(thread_idx.x)

    # Each thread loads and does initial reduction
    var partial_sum = input[Int(i)] + input[Int(i + BLOCK_DIM)]
    partial_sum = warp_reduce(partial_sum)

    # Allocate shared memory for partial sums from each warp
    var partial_sums_s = unsafe_stack_allocation[
        BLOCK_DIM // WARP_SIZE,
        Float32,
        address_space=.SHARED,
    ]()

    # Store warp results to shared memory
    if lane_id() == 0:
        partial_sums_s[warp_id()] = partial_sum

    barrier()

    # First warp reduces and atomically adds to output
    if UInt32(thread_idx.x) < UInt32(WARP_SIZE):
        partial_sum = partial_sums_s[thread_idx.x]
        partial_sum = warp_reduce(partial_sum)
        if thread_idx.x == 0:
            _ = Atomic.fetch_add(output, partial_sum)


# ========================== TEST CODE ==========================
def cpu_sum(
    input: UnsafePointer[mut=False, Float32, _],
    output: UnsafePointer[mut=True, Float32, _],
    N: Int,
):
    """CPU reference sum implementation.

    Args:
        input: Input array.
        output: Output scalar for the sum result.
        N: Number of elements.
    """
    var sum = Float32(0.0)
    for i in range(N):
        sum += input[i]
    output[0] = sum


def main() raises:
    """Example usage of multi-block sum reduction kernel."""
    comptime N = 65536  # Large array

    # Host memory allocation
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](1)
    var h_ref = alloc[Float32](1)

    # Initialize input with random values
    for i in range(N):
        h_input[i] = random_float64().cast[.float32]()

    # Initialize output to zero
    h_output[0] = 0.0

    # Calculate number of blocks
    var blocks = ceildiv(N, BLOCK_DIM * 2)

    print(
        "Launching multi-block sum reduction kernel (Fig 10.18) with",
        blocks,
        "blocks and",
        BLOCK_DIM,
        "threads",
    )
    print("Uses atomic operations to combine results from multiple blocks")

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[.float32](N)
        var d_output = ctx.enqueue_create_buffer[.float32](1)

        # Copy data to device
        ctx.enqueue_copy(d_input, h_input)
        ctx.enqueue_copy(d_output, h_output)

        # Launch kernel (multiple blocks)
        ctx.enqueue_function[reduce_kernel](
            d_input,
            d_output,
            UInt32(N),
            grid_dim=(blocks, 1, 1),
            block_dim=(BLOCK_DIM, 1, 1),
        )

        # Copy result back to host
        ctx.enqueue_copy(h_output, d_output)
        ctx.synchronize()

    # Run CPU reference
    cpu_sum(h_input, h_ref, N)

    # Verify results
    var diff = abs(h_ref[0] - h_output[0])
    var rel = diff / (abs(h_ref[0]) + 1e-7)

    if rel > 1e-4:
        print(
            "FAILED: CPU=",
            h_ref[0],
            ", GPU=",
            h_output[0],
            ", diff=",
            diff,
            ", rel=",
            rel,
        )
    else:
        print("SUCCESS: CPU=", h_ref[0], ", GPU=", h_output[0], "match!")

    # Cleanup
    h_input.free()
    h_output.free()
    h_ref.free()
