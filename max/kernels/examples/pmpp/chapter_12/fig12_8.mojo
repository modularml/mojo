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

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from max.gpu.primitives.id import (
    lane_id,
    warp_id,
)
from max.gpu.primitives.warp import shuffle_up


# ========================== CONFIGURATION ==========================
comptime BLOCK_DIM = 256
comptime NUM_WARPS = BLOCK_DIM // WARP_SIZE


# ========================== DEVICE FUNCTIONS ==========================
@always_inline
def cond(val: UInt32) -> Bool:
    """Filter: keep only even numbers.

    Args:
        val: Value to test.

    Returns:
        True if value passes filter (even number).
    """
    return (val % 2) == 0


def warp_scan(val: UInt32) -> UInt32:
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


def block_scan(
    val: UInt32,
    warp_sums: UnsafePointer[mut=True, UInt32, _, address_space=.SHARED],
) -> UInt32:
    """Block-level inclusive scan.

    Args:
        val: Value from current thread.
        warp_sums: Shared memory for warp partial sums.

    Returns:
        Inclusive scan result for this thread.
    """
    var result = warp_scan(val)

    # Store warp sums
    if lane_id() == WARP_SIZE - 1:
        warp_sums[warp_id()] = result

    barrier()

    # Scan warp sums
    if warp_id() == 0:
        var warp_sum: UInt32 = 0
        if thread_idx.x < NUM_WARPS:
            warp_sum = warp_sums[thread_idx.x]
        warp_sum = warp_scan(warp_sum)
        if thread_idx.x < NUM_WARPS:
            warp_sums[thread_idx.x] = warp_sum

    barrier()

    # Add previous warp sum
    if warp_id() > 0:
        result += warp_sums[warp_id() - 1]

    return result


def block_exclusive_scan(
    val: UInt32,
    warp_sums: UnsafePointer[mut=True, UInt32, _, address_space=.SHARED],
    temp: UnsafePointer[mut=True, UInt32, _, address_space=.SHARED],
) -> UInt32:
    """Convert inclusive scan to exclusive scan.

    Args:
        val: Value from current thread.
        warp_sums: Shared memory for warp sums.
        temp: Temporary shared memory.

    Returns:
        Exclusive scan result for this thread.
    """
    var inclusive = block_scan(val, warp_sums)
    temp[thread_idx.x] = inclusive
    barrier()

    var exclusive: UInt32 = 0
    if thread_idx.x > 0:
        exclusive = temp[thread_idx.x - 1]

    return exclusive


# ========================== KERNEL CODE ==========================
def filter_kernel(
    input: UnsafePointer[UInt32, MutAnyOrigin],
    output: UnsafePointer[UInt32, MutAnyOrigin],
    output_size: UnsafePointer[UInt32, MutAnyOrigin],
    N: UInt32,
):
    """Figure 12.8: Simple stable filter kernel using scan.

    Args:
        input: Input array.
        output: Output array.
        output_size: Pointer to output size counter.
        N: Number of elements.
    """
    # Allocate shared memory for scan operations
    var warp_sums = unsafe_stack_allocation[
        NUM_WARPS,
        UInt32,
        address_space=.SHARED,
    ]()
    var temp = unsafe_stack_allocation[
        BLOCK_DIM,
        UInt32,
        address_space=.SHARED,
    ]()

    var i = block_idx.x * BLOCK_DIM + thread_idx.x
    var val: UInt32 = 0
    var keep: UInt32 = 0
    if UInt32(i) < N:
        val = input[i]
        if cond(val):
            keep = 1

    # Exclusive scan to get output position
    var offset = block_exclusive_scan(keep, warp_sums, temp)

    # Write output if this thread's element passes the filter
    if keep == 1:
        output[Int(offset)] = val

    # Last thread updates the total output size
    if i == Int(N) - 1:
        output_size[0] = offset + keep


def main() raises:
    """Test the stable filter kernel."""
    # Use single block for simplified demo
    comptime N = BLOCK_DIM
    comptime dtype = DType.uint32

    with DeviceContext() as ctx:
        # Allocate device memory
        var d_input = ctx.enqueue_create_buffer[dtype](N)
        var d_output = ctx.enqueue_create_buffer[dtype](N)
        var d_output_size = ctx.enqueue_create_buffer[dtype](1)

        # Initialize input and output size
        with d_input.map_to_host() as h_input:
            for i in range(N):
                h_input[i] = UInt32(i)

        with d_output_size.map_to_host() as h_size:
            h_size[0] = UInt32(0)

        # Launch kernel (single block for simplified demo)
        print(
            "Launching simple stable filter kernel (Fig 12.8) with 1 block and",
            BLOCK_DIM,
            "threads",
        )
        print("Input size:", N, "elements")

        comptime kernel = filter_kernel
        ctx.enqueue_function[kernel](
            d_input,
            d_output,
            d_output_size,
            UInt32(N),
            grid_dim=(1, 1, 1),
            block_dim=(BLOCK_DIM, 1, 1),
        )
        ctx.synchronize()

        # Get output size
        var output_count: Int
        with d_output_size.map_to_host() as h_size:
            output_count = Int(h_size[0])

        print("GPU output size:", output_count)

        # Copy results and verify
        var errors = 0
        with d_input.map_to_host() as h_input:
            with d_output.map_to_host() as h_output:
                # Create reference on CPU
                var expected_count = 0
                for i in range(N):
                    if cond(UInt32(i)):
                        expected_count += 1

                print("CPU output size:", expected_count)

                if output_count != expected_count:
                    print("FAILED: Output sizes don't match!")
                    return

                # No need to sort - this is a STABLE filter, order is preserved
                # Verify all even numbers are in order
                for i in range(output_count):
                    var expected = UInt32(i * 2)  # Even numbers: 0, 2, 4, ...
                    if h_output[i] != expected:
                        if errors < 10:
                            print(
                                "Error at index",
                                i,
                                ": expected=",
                                expected,
                                ", got=",
                                h_output[i],
                            )
                        errors += 1

        if errors == 0:
            print("SUCCESS: All values match in order!")
            print("Note: This is a STABLE filter - output order preserved")
        else:
            print("FAILED: Found", errors, "errors")
