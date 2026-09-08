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

from max.gpu import block_idx, thread_idx, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from max.gpu.primitives.id import lane_id, warp_id
from max.gpu.primitives.warp import shuffle_up

from std.math import abs
from std.atomic import Atomic

# ========================== CONFIGURATION ==========================
comptime BLOCK_DIM = 256
comptime NUM_WARPS = BLOCK_DIM // WARP_SIZE


# ========================== DEVICE FUNCTIONS ==========================
def warp_scan(val: Float32) -> Float32:
    """Warp-level inclusive scan using shuffle operations.

    Args:
        val: Value from current thread.

    Returns:
        Inclusive scan result for this thread.
    """
    var result = val

    # Kogge-Stone scan within a warp
    var stride = UInt32(1)
    while stride < UInt32(WARP_SIZE):
        var left_val = shuffle_up(result, stride)
        if UInt32(lane_id()) >= stride:
            result += left_val
        stride *= 2

    return result


# ========================== TEST CODE ==========================
def test_interblock_scan(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    partial_sums: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    N: UInt32,
):
    """Test kernel for inter-block scan.

    Args:
        input: Input array.
        output: Output array for scan result.
        partial_sums: Array to store partial sums from each block.
        flags: Array of flags for block synchronization.
        N: Number of elements.
    """
    var i = block_idx.x * BLOCK_DIM + thread_idx.x
    var val = Float32(0.0) if i >= Int(N) else input[i]

    # Allocate shared memory
    var warp_sums = unsafe_stack_allocation[
        NUM_WARPS,
        Float32,
        address_space=.SHARED,
    ]()
    var prev_block_sum_ptr = unsafe_stack_allocation[
        1,
        Float32,
        address_space=.SHARED,
    ]()

    # Block-level scan using warp primitives
    var result = warp_scan(val)

    # Collect warp sums
    if UInt32(lane_id()) == UInt32(WARP_SIZE - 1):
        warp_sums[warp_id()] = result

    barrier()

    # Scan warp sums (only first warp participates)
    if warp_id() == 0:
        var warp_sum = Float32(0.0)
        if thread_idx.x < NUM_WARPS:
            warp_sum = warp_sums[thread_idx.x]
        warp_sum = warp_scan(warp_sum)
        if thread_idx.x < NUM_WARPS:
            warp_sums[thread_idx.x] = warp_sum

    barrier()

    # Add previous warp's scanned sum
    if warp_id() > 0:
        result += warp_sums[warp_id() - 1]

    # Inter-block scan - all threads call but only last thread does the work
    if thread_idx.x == BLOCK_DIM - 1:
        var bid = UInt32(block_idx.x)
        if bid > 0:
            # Wait for previous block to pass partial sum
            while Atomic.load(flags + Int(bid - 1)) == 0:
                pass

            # Read previous block's partial sum
            prev_block_sum_ptr[0] = partial_sums[Int(bid - 1)]
        else:
            prev_block_sum_ptr[0] = 0.0

        # Write this block's partial sum
        partial_sums[Int(bid)] = prev_block_sum_ptr[0] + result

        # Set this block's flag
        _ = Atomic.fetch_add(flags + Int(bid), UInt32(1))

    barrier()

    # Add previous blocks' sum
    val = result + prev_block_sum_ptr[0]

    if i < Int(N):
        output[i] = val


def cpu_scan(
    input: UnsafePointer[mut=False, Float32, _],
    output: UnsafePointer[mut=True, Float32, _],
    N: UInt32,
):
    """CPU reference scan implementation.

    Args:
        input: Input array.
        output: Output array for scan result.
        N: Number of elements.
    """
    output[0] = input[0]
    for i in range(1, Int(N)):
        output[i] = output[i - 1] + input[i]


def main() raises:
    """Example usage of inter-block scan."""
    comptime num_blocks = 4
    comptime N = BLOCK_DIM * num_blocks
    comptime dtype = DType.float32

    # Allocate host memory
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](N)
    var h_ref = alloc[Float32](N)

    # Initialize input
    for i in range(N):
        h_input[i] = 1.0

    print(
        "Launching inter-block scan (Fig 11.17) with",
        num_blocks,
        "blocks and",
        BLOCK_DIM,
        "threads/block",
    )

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[dtype](N)
        var d_output = ctx.enqueue_create_buffer[dtype](N)
        var d_partial_sums = ctx.enqueue_create_buffer[dtype](num_blocks)
        var d_flags = ctx.enqueue_create_buffer[.uint32](num_blocks)

        # Copy data to device and initialize flags to 0
        ctx.enqueue_copy(d_input, h_input)

        # Initialize flags to 0
        var h_flags = alloc[UInt32](num_blocks)
        for i in range(num_blocks):
            h_flags[i] = 0
        ctx.enqueue_copy(d_flags, h_flags)
        h_flags.free()

        # Launch kernel
        ctx.enqueue_function[test_interblock_scan](
            d_input,
            d_output,
            d_partial_sums,
            d_flags,
            UInt32(N),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(BLOCK_DIM, 1, 1),
        )

        # Copy result back to host
        ctx.enqueue_copy(h_output, d_output)
        ctx.synchronize()

    # Run CPU reference
    cpu_scan(h_input, h_ref, UInt32(N))

    # Verify results
    var errors = 0
    for i in range(N):
        if abs(h_output[i] - h_ref[i]) > 1e-5:
            if errors < 10:
                print(
                    "Error at index",
                    i,
                    ": CPU=",
                    h_ref[i],
                    ", GPU=",
                    h_output[i],
                )
            errors += 1

    # Cleanup
    h_input.free()
    h_output.free()
    h_ref.free()

    if errors == 0:
        print("SUCCESS: All values match!")
    else:
        print("FAILED: Found", errors, "errors")
