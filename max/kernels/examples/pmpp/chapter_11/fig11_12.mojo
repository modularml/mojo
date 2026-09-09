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


# ========================== KERNEL CODE ==========================
def scan_kernel(
    input: UnsafePointer[Float32, ImmutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: UInt32,
):
    """Scan kernel with thread coarsening.

    Args:
        input: Input array.
        output: Output array for scan result.
        N: Number of elements.
    """
    var block_segment = block_idx.x * COARSE_FACTOR * BLOCK_DIM

    # Allocate shared memory
    var buffer_s = unsafe_stack_allocation[
        COARSE_FACTOR * BLOCK_DIM,
        Float32,
        address_space=.SHARED,
    ]()
    var warp_sums = unsafe_stack_allocation[
        NUM_WARPS,
        Float32,
        address_space=.SHARED,
    ]()
    var thread_sums = unsafe_stack_allocation[
        BLOCK_DIM,
        Float32,
        address_space=.SHARED,
    ]()

    # Load data to shared memory
    for c in range(COARSE_FACTOR):
        buffer_s[c * BLOCK_DIM + thread_idx.x] = input[
            block_segment + c * BLOCK_DIM + thread_idx.x
        ]

    barrier()

    # Scan thread subsegment
    var thread_segment = thread_idx.x * COARSE_FACTOR
    for c in range(1, COARSE_FACTOR):
        buffer_s[thread_segment + c] += buffer_s[thread_segment + c - 1]

    # Block-wide scan of thread sums using warp primitives
    var thread_sum = buffer_s[thread_segment + COARSE_FACTOR - 1]

    # Inline block scan with the pre-allocated warp_sums
    var result = warp_scan(thread_sum)

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

    # Store thread partial sums
    thread_sums[thread_idx.x] = result

    barrier()

    # Add previous thread's partial sums
    if thread_idx.x > 0:
        var prev_partial_sum = thread_sums[thread_idx.x - 1]
        for c in range(COARSE_FACTOR):
            buffer_s[thread_segment + c] += prev_partial_sum

    barrier()

    # Write output
    for c in range(COARSE_FACTOR):
        output[block_segment + c * BLOCK_DIM + thread_idx.x] = buffer_s[
            c * BLOCK_DIM + thread_idx.x
        ]


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
    """Example usage of scan kernel with thread coarsening."""
    comptime N = COARSE_FACTOR * BLOCK_DIM
    comptime dtype = DType.float32

    # Allocate host memory
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](N)
    var h_ref = alloc[Float32](N)

    # Initialize input
    for i in range(N):
        h_input[i] = 1.0

    print(
        "Launching scan with thread coarsening (Fig 11.12) with 1 block,",
        BLOCK_DIM,
        "threads, COARSE_FACTOR=",
        COARSE_FACTOR,
    )
    print("Processing", N, "elements")

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[dtype](N)
        var d_output = ctx.enqueue_create_buffer[dtype](N)

        # Copy data to device
        ctx.enqueue_copy(d_input, h_input)

        # Launch kernel (single block with thread coarsening)
        ctx.enqueue_function[scan_kernel](
            d_input,
            d_output,
            UInt32(N),
            grid_dim=(1, 1, 1),
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
