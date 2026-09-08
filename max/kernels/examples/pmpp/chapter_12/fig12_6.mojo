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
from std.atomic import Atomic


# ========================== CONFIGURATION ==========================
comptime BLOCK_DIM = 256


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


# ========================== KERNEL CODE ==========================
def filter_kernel(
    input: UnsafePointer[UInt32, MutAnyOrigin],
    output: UnsafePointer[UInt32, MutAnyOrigin],
    output_size: UnsafePointer[UInt32, MutAnyOrigin],
    N: UInt32,
):
    """Figure 12.6: Unstable filter kernel with privatization.

    Args:
        input: Input array.
        output: Output array.
        output_size: Pointer to output size counter.
        N: Number of elements.
    """
    # Allocate shared memory for block-private output list
    var output_s = unsafe_stack_allocation[
        BLOCK_DIM,
        UInt32,
        address_space=.SHARED,
    ]()
    var output_size_s = unsafe_stack_allocation[
        1,
        UInt32,
        address_space=.SHARED,
    ]()

    if thread_idx.x == 0:
        output_size_s[0] = UInt32(0)

    barrier()

    # Filter in the private lists (shared memory)
    var i = block_idx.x * BLOCK_DIM + thread_idx.x
    if UInt32(i) < N:
        var val = input[i]
        if cond(val):
            # Block-scoped atomic add to shared memory
            var j = Atomic.fetch_add(output_size_s, UInt32(1))
            output_s[Int(j)] = val

    barrier()

    # Update the public counter (one thread per block)
    var j_shared = unsafe_stack_allocation[
        1,
        UInt32,
        address_space=.SHARED,
    ]()
    if thread_idx.x == 0:
        var local_size = output_size_s[0]
        var j = Atomic.fetch_add(output_size, local_size)
        j_shared[0] = j

    barrier()

    # Write to the public list
    var local_output_size = Int(output_size_s[0])
    if thread_idx.x < local_output_size:
        var j_val = j_shared[0]
        output[Int(j_val) + thread_idx.x] = output_s[thread_idx.x]


def main() raises:
    """Test the privatization filter kernel."""
    comptime N = 10000
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

        # Launch kernel
        var num_blocks = (N + BLOCK_DIM - 1) // BLOCK_DIM
        print(
            (
                "Launching unstable filter kernel with privatization (Fig 12.6)"
                " with"
            ),
            num_blocks,
            "blocks and",
            BLOCK_DIM,
            "threads/block",
        )
        print("Input size:", N, "elements")

        comptime kernel = filter_kernel
        ctx.enqueue_function[kernel](
            d_input,
            d_output,
            d_output_size,
            UInt32(N),
            grid_dim=(num_blocks, 1, 1),
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

                # Sort GPU output for comparison (unstable filter)
                for i in range(output_count):
                    for j in range(output_count - 1 - i):
                        if h_output[j] > h_output[j + 1]:
                            var temp = h_output[j]
                            h_output[j] = h_output[j + 1]
                            h_output[j + 1] = temp

                # Verify all even numbers are present
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
            print("SUCCESS: All values match (after sorting)!")
            print("Note: This is an UNSTABLE filter with privatization")
        else:
            print("FAILED: Found", errors, "errors")
