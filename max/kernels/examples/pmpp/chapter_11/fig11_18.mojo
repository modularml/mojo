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

from max.gpu import thread_idx, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from max.gpu.primitives.id import lane_id, warp_id
from max.gpu.primitives.warp import shuffle_up

from std.math import abs
from std.atomic import Atomic
from std.utils import StaticTuple

# ========================== CONFIGURATION ==========================
comptime BLOCK_DIM = 256
comptime COARSE_FACTOR = 4
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


def block_scan(val: Float32) -> Float32:
    """Block-level inclusive scan using hierarchical warp scans.

    Args:
        val: Value from current thread.

    Returns:
        Inclusive scan result for this thread.
    """
    # Step 1: Warp-level scan
    var result = warp_scan(val)

    # Allocate shared memory for warp sums
    var warp_sums_s = unsafe_stack_allocation[
        NUM_WARPS,
        Float32,
        address_space=.SHARED,
    ]()

    # Step 2: Collect warp sums
    if lane_id() == WARP_SIZE - 1:
        warp_sums_s[warp_id()] = result

    barrier()

    # Step 3: Scan warp sums (only first warp participates)
    if warp_id() == 0:
        var warp_sum = Float32(0.0)
        if thread_idx.x < NUM_WARPS:
            warp_sum = warp_sums_s[thread_idx.x]
        warp_sum = warp_scan(warp_sum)
        if thread_idx.x < NUM_WARPS:
            warp_sums_s[thread_idx.x] = warp_sum

    barrier()

    # Step 4: Add previous warp's scanned sum
    if warp_id() > 0:
        result += warp_sums_s[warp_id() - 1]

    return result


def inter_block_scan(
    val: Float32,
    bid: UInt32,
    partial_sums: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
) -> Float32:
    """Inter-block scan using atomic operations for synchronization.

    Args:
        val: Scanned value from this block.
        bid: Block ID.
        partial_sums: Array to store partial sums from each block.
        flags: Array of flags for block synchronization.

    Returns:
        Sum from all previous blocks.
    """
    # Allocate shared memory for previous block's partial sum
    var prev_block_partial_sum_s = unsafe_stack_allocation[
        1,
        Float32,
        address_space=.SHARED,
    ]()

    if thread_idx.x == BLOCK_DIM - 1:
        if bid == 0:
            prev_block_partial_sum_s[0] = 0.0
        else:
            # Wait for previous block to pass partial sum
            while Atomic.load(flags + Int(bid - 1)) == 0:
                pass

            # Read previous block's partial sum
            prev_block_partial_sum_s[0] = partial_sums[Int(bid - 1)]

        # Write this block's partial sum
        partial_sums[Int(bid)] = prev_block_partial_sum_s[0] + val

        # Set this block's flag
        _ = Atomic.fetch_add(flags + Int(bid), UInt32(1))

    barrier()
    return prev_block_partial_sum_s[0]


# ========================== KERNEL CODE ==========================
def scan_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    block_counter: UnsafePointer[UInt32, MutAnyOrigin],
    partial_sums: UnsafePointer[Float32, MutAnyOrigin],
    flags: UnsafePointer[UInt32, MutAnyOrigin],
    N: UInt32,
):
    """Full scan kernel with coarsening, register tiling, and dynamic block assignment.

    Args:
        input: Input array.
        output: Output array for scan result.
        block_counter: Counter for dynamic block assignment.
        partial_sums: Array to store partial sums from each block.
        flags: Array of flags for block synchronization.
        N: Number of elements.
    """
    # Assign block index dynamically
    var bid_s = unsafe_stack_allocation[
        1,
        UInt32,
        address_space=.SHARED,
    ]()

    if thread_idx.x == 0:
        bid_s[0] = Atomic.fetch_add(block_counter, UInt32(1))

    barrier()
    var bid = bid_s[0]
    var block_segment = Int(bid) * COARSE_FACTOR * BLOCK_DIM

    # Allocate shared memory for buffer
    var buffer_s = unsafe_stack_allocation[
        COARSE_FACTOR * BLOCK_DIM,
        Float32,
        address_space=.SHARED,
    ]()

    # Load data to shared memory
    for c in range(COARSE_FACTOR):
        var idx = block_segment + c * BLOCK_DIM + thread_idx.x
        var val = Float32(0.0) if idx >= Int(N) else input[idx]
        buffer_s[c * BLOCK_DIM + thread_idx.x] = Float32(val)

    barrier()

    # Scan thread subsegment using registers
    var thread_segment = thread_idx.x * COARSE_FACTOR
    var buffer_r = StaticTuple[Float32, COARSE_FACTOR]()
    buffer_r[0] = Float32(buffer_s[thread_segment])

    comptime for c in range(1, COARSE_FACTOR):
        buffer_r[c] = Float32(buffer_s[thread_segment + c]) + buffer_r[c - 1]

    # Block-wide scan of thread sums
    var thread_sum = buffer_r[COARSE_FACTOR - 1]
    thread_sum = block_scan(thread_sum)

    # Collect thread partial sums
    var thread_sums = unsafe_stack_allocation[
        BLOCK_DIM,
        Float32,
        address_space=.SHARED,
    ]()
    thread_sums[thread_idx.x] = thread_sum

    barrier()

    # Add previous thread's partial sums
    var prev_partial_sum = Float32(0.0)
    if thread_idx.x > 0:
        prev_partial_sum = thread_sums[thread_idx.x - 1]

    comptime for c in range(COARSE_FACTOR):
        buffer_r[c] += prev_partial_sum

    # Scan block partial sums
    var prev_block_partial_sum = inter_block_scan(
        buffer_r[COARSE_FACTOR - 1], bid, partial_sums, flags
    )

    # Add previous block's partial sum
    comptime for c in range(COARSE_FACTOR):
        buffer_s[thread_segment + c] = Float32(
            buffer_r[c] + prev_block_partial_sum
        )

    barrier()

    # Write output
    for c in range(COARSE_FACTOR):
        var idx = block_segment + c * BLOCK_DIM + thread_idx.x
        if idx < Int(N):
            output[idx] = Float32(buffer_s[c * BLOCK_DIM + thread_idx.x])


# ========================== TEST CODE ==========================
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
    """Example usage of full scan kernel."""
    comptime num_blocks = 4
    comptime N = COARSE_FACTOR * BLOCK_DIM * num_blocks
    comptime dtype = DType.float32

    # Allocate host memory
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](N)
    var h_ref = alloc[Float32](N)

    # Initialize input
    for i in range(N):
        h_input[i] = 1.0

    print(
        "Launching full scan kernel (Fig 11.18) with",
        num_blocks,
        "blocks,",
        BLOCK_DIM,
        "threads/block, COARSE_FACTOR=",
        COARSE_FACTOR,
    )
    print("Processing", N, "elements")

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[dtype](N)
        var d_output = ctx.enqueue_create_buffer[dtype](N)
        var d_block_counter = ctx.enqueue_create_buffer[.uint32](1)
        var d_partial_sums = ctx.enqueue_create_buffer[dtype](num_blocks)
        var d_flags = ctx.enqueue_create_buffer[.uint32](num_blocks)

        # Copy data to device
        ctx.enqueue_copy(d_input, h_input)

        # Initialize block counter and flags to 0
        var h_counter = alloc[UInt32](1)
        var h_flags = alloc[UInt32](num_blocks)
        h_counter[0] = 0
        for i in range(num_blocks):
            h_flags[i] = 0
        ctx.enqueue_copy(d_block_counter, h_counter)
        ctx.enqueue_copy(d_flags, h_flags)
        h_counter.free()
        h_flags.free()

        # Launch kernel
        ctx.enqueue_function[scan_kernel](
            d_input,
            d_output,
            d_block_counter,
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
    cpu_scan(h_input, h_ref, N)

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
