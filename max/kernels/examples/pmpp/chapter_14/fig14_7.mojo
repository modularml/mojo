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

"""Figure 14.7: Radix sort iteration kernel implementation in Mojo."""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.random import random_ui64


def grid_exclusive_scan(bits: UnsafePointer[mut=True, UInt32, _], N: Int):
    """CPU-based exclusive scan for simplicity.

    Args:
        bits: Array to scan.
        N: Size of array.
    """
    var sum = UInt32(0)
    for i in range(N):
        var tmp = bits[i]
        bits[i] = sum
        sum += tmp


def radix_sort_iter(
    input: UnsafePointer[UInt32, ImmutAnyOrigin],
    output: UnsafePointer[UInt32, MutAnyOrigin],
    bits: UnsafePointer[UInt32, MutAnyOrigin],
    N_dev: Int32,
    iter_dev: Int32,
    block_dim_x_dev: Int32,
):
    """Radix sort iteration kernel - extracts bits.

    Args:
        input: Input array.
        output: Output array.
        bits: Bit array.
        N_dev: Size of array.
        iter_dev: Current bit position.
        block_dim_x_dev: Block dimension.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var N = Int(N_dev)
    var iter = Int(iter_dev)
    var block_dim_x = Int(block_dim_x_dev)
    var i = block_idx.x * block_dim_x + thread_idx.x

    if i < N:
        var key = input[i]
        var bit = (key >> UInt32(iter)) & 1
        bits[i] = bit


def radix_scatter(
    input: UnsafePointer[UInt32, ImmutAnyOrigin],
    output: UnsafePointer[UInt32, MutAnyOrigin],
    bits: UnsafePointer[UInt32, MutAnyOrigin],
    scanResult: UnsafePointer[UInt32, ImmutAnyOrigin],
    N_dev: Int32,
    iter_dev: Int32,
    block_dim_x_dev: Int32,
):
    """Scatter elements based on scan results.

    Args:
        input: Input array.
        output: Output array.
        bits: Bit array.
        scanResult: Scan result array.
        N_dev: Size of array.
        iter_dev: Current bit position.
        block_dim_x_dev: Block dimension.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var N = Int(N_dev)
    var iter = Int(iter_dev)
    var block_dim_x = Int(block_dim_x_dev)
    var i = block_idx.x * block_dim_x + thread_idx.x

    if i < N:
        var key = input[i]
        var bit = (key >> UInt32(iter)) & 1
        var numOnesBefore = scanResult[i]
        var numOnesTotal = scanResult[N]
        var dst = (i - Int(numOnesBefore)) if bit == 0 else (
            N - Int(numOnesTotal) + Int(numOnesBefore)
        )
        output[dst] = key


def cpu_radix_sort(
    data: UnsafePointer[mut=True, UInt32, _], N: Int, num_bits: Int
):
    """CPU radix sort for verification.

    Args:
        data: Array to sort.
        N: Size of array.
        num_bits: Number of bits to sort.
    """
    var tmp = alloc[UInt32](N)
    var count0 = alloc[UInt32](N)
    var count1 = alloc[UInt32](N)

    for bit in range(num_bits):
        var zeros = 0
        var ones = 0

        # Count 0s and 1s
        for i in range(N):
            if ((data[i] >> UInt32(bit)) & 1) == 0:
                count0[zeros] = data[i]
                zeros += 1
            else:
                count1[ones] = data[i]
                ones += 1

        # Place back: 0s first, then 1s
        for i in range(zeros):
            data[i] = count0[i]
        for i in range(ones):
            data[zeros + i] = count1[i]

    tmp.free()
    count0.free()
    count1.free()


def find_max(data: UnsafePointer[UInt32, _], N: Int) -> UInt32:
    """Find maximum value in array.

    Args:
        data: Array to search.
        N: Size of array.

    Returns:
        Maximum value.
    """
    var max_val = UInt32(0)
    for i in range(N):
        if data[i] > max_val:
            max_val = data[i]
    return max_val


def count_bits(value: UInt32) -> Int:
    """Count number of bits needed to represent value.

    Args:
        value: Value to analyze.

    Returns:
        Number of bits needed.
    """
    var bits = 0
    var v = value
    while v > 0:
        bits += 1
        v >>= 1
    return bits


def main() raises:
    var N = 10000

    # Allocate host memory
    var h_data = alloc[UInt32](N)
    var h_ref = alloc[UInt32](N)
    var h_bits = alloc[UInt32](N + 1)  # +1 for total count

    # Initialize with random data
    print("Initializing array with", N, "random elements")
    var max_val = UInt32(0)
    for i in range(N):
        var val = UInt32(
            random_ui64(0, 65536)
        )  # Limit to 16-bit for faster testing
        h_data[i] = val
        h_ref[i] = val
        if val > max_val:
            max_val = val

    var num_bits = count_bits(max_val)
    print("Maximum value:", max_val, ", requires", num_bits, "bits")

    # Print first 10 elements before sorting
    print("First 10 elements before sorting: ", end="")
    for i in range(10):
        print(h_data[i], " ", end="")
    print("...")

    # Create device context
    var ctx = DeviceContext()

    # Allocate device memory
    var d_input = ctx.enqueue_create_buffer[.uint32](N)
    var d_output = ctx.enqueue_create_buffer[.uint32](N)
    var d_bits = ctx.enqueue_create_buffer[.uint32](N + 1)

    # Copy to device
    ctx.enqueue_copy(d_input, h_data)

    # Launch configuration
    var threads_per_block = 256
    var num_blocks = (N + threads_per_block - 1) // threads_per_block

    print("\nLaunching radix sort iteration kernel (Fig 14.7)")
    print("Array size:", N, "elements")
    print(
        "Config:",
        num_blocks,
        "blocks x",
        threads_per_block,
        "threads =",
        num_blocks * threads_per_block,
        "threads total",
    )
    print("Number of iterations (bits):", num_bits)

    # Radix sort main loop
    var current_input = d_input
    var current_output = d_output

    for iter in range(num_bits):
        # Extract bits
        ctx.enqueue_function[radix_sort_iter](
            current_input,
            current_output,
            d_bits,
            Int32(N),
            Int32(iter),
            Int32(threads_per_block),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(threads_per_block, 1, 1),
        )
        ctx.synchronize()

        # Copy bits to host for CPU scan (in real implementation, use GPU scan)
        ctx.enqueue_copy(h_bits, d_bits)
        ctx.synchronize()

        # Perform exclusive scan on CPU and count total ones
        var total_ones = UInt32(0)
        for j in range(N):
            total_ones += h_bits[j]
        grid_exclusive_scan(h_bits, N)
        h_bits[N] = total_ones  # Store total count of 1s

        # Copy scan results back
        ctx.enqueue_copy(d_bits, h_bits)
        ctx.synchronize()

        # Scatter based on scan results
        ctx.enqueue_function[radix_scatter](
            current_input,
            current_output,
            d_bits,
            d_bits,
            Int32(N),
            Int32(iter),
            Int32(threads_per_block),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(threads_per_block, 1, 1),
        )
        ctx.synchronize()

        # Swap buffers
        var tmp = current_input
        current_input = current_output
        current_output = tmp

        if iter % 4 == 0:
            print("Completed iteration", iter, "...")

    print("GPU radix sort completed")

    # Copy result back (might be in either buffer)
    ctx.enqueue_copy(h_data, current_input)
    ctx.synchronize()

    # Print first 10 elements after sorting
    print("\nFirst 10 elements after GPU sorting: ", end="")
    for i in range(10):
        print(h_data[i], " ", end="")
    print("...")

    # Verify GPU result is sorted
    var gpu_sorted = True
    for i in range(N - 1):
        if h_data[i] > h_data[i + 1]:
            print(
                "ERROR: Array not sorted at index",
                i,
                ":",
                h_data[i],
                ">",
                h_data[i + 1],
            )
            gpu_sorted = False
            break

    if gpu_sorted:
        print("SUCCESS: GPU array is correctly sorted!")
    else:
        print("ERROR: GPU array is not sorted correctly!")

    # Run CPU reference for comparison
    print("\nRunning CPU radix sort reference...")
    cpu_radix_sort(h_ref, N, num_bits)
    print("CPU radix sort completed")

    # Verify CPU result matches GPU result
    var errors = 0
    for i in range(N):
        if h_data[i] != h_ref[i]:
            if errors < 10:
                print(
                    "Mismatch at index",
                    i,
                    ": GPU=",
                    h_data[i],
                    ", CPU=",
                    h_ref[i],
                )
            errors += 1

    if errors == 0:
        print("SUCCESS: GPU and CPU results match!")
    else:
        print("ERROR: Found", errors, "mismatches between GPU and CPU")

    # Cleanup
    h_data.free()
    h_ref.free()
    h_bits.free()
