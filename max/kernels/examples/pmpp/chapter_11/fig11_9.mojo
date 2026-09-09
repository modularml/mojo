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
comptime NUM_WARPS = BLOCK_DIM // WARP_SIZE


# ========================== KERNEL CODE ==========================
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


# ========================== TEST CODE ==========================
def test_block_scan(
    input: UnsafePointer[Float32, ImmutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: UInt32,
):
    """Test kernel for block-level scan.

    Args:
        input: Input array.
        output: Output array for scan result.
        N: Number of elements.
    """
    var i = block_idx.x * BLOCK_DIM + thread_idx.x
    var val = Float32(0.0) if i >= Int(N) else input[i]
    val = block_scan(val)
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
    """Example usage of block-level scan."""
    comptime N = BLOCK_DIM
    comptime dtype = DType.float32

    # Allocate host memory
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](N)
    var h_ref = alloc[Float32](N)

    # Initialize input
    for i in range(N):
        h_input[i] = 1.0

    print(
        "Launching block-level scan (Fig 11.9) with 1 block and",
        BLOCK_DIM,
        "threads",
    )

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[dtype](N)
        var d_output = ctx.enqueue_create_buffer[dtype](N)

        # Copy data to device
        ctx.enqueue_copy(d_input, h_input)

        # Launch kernel (single block)
        ctx.enqueue_function[test_block_scan](
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
