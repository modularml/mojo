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

"""Figure 14.2: Parallel odd-even sort kernel implementation in Mojo."""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.random import random_ui64


def sort_kernel(
    data: UnsafePointer[UInt32, MutAnyOrigin],
    hasChanged: UnsafePointer[UInt32, MutAnyOrigin],
    N_dev: Int32,
    isOddStep_dev: Int32,
    block_dim_x_dev: Int32,
):
    """Parallel odd-even sort kernel.

    Args:
        data: Array to sort.
        hasChanged: Flag to indicate if any swaps were made.
        N_dev: Size of array.
        isOddStep_dev: 1 for odd phase, 0 for even phase.
        block_dim_x_dev: Block dimension.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var N = Int(N_dev)
    var isOddStep = Int(isOddStep_dev)
    var block_dim_x = Int(block_dim_x_dev)
    var i = 2 * (block_idx.x * block_dim_x + thread_idx.x) + (
        1 if isOddStep == 1 else 0
    )

    if i < N - 1:
        if data[i] > data[i + 1]:
            var tmp = data[i]
            data[i] = data[i + 1]
            data[i + 1] = tmp
            hasChanged[0] = 1


def cpu_odd_even_sort(data: UnsafePointer[mut=True, UInt32, _], N: Int):
    """CPU reference implementation of odd-even sort.

    Args:
        data: Array to sort.
        N: Size of array.
    """
    var sorted = False
    while not sorted:
        sorted = True

        # Odd phase
        for i in range(1, N - 1, 2):
            if data[i] > data[i + 1]:
                var tmp = data[i]
                data[i] = data[i + 1]
                data[i + 1] = tmp
                sorted = False

        # Even phase
        for i in range(0, N - 1, 2):
            if data[i] > data[i + 1]:
                var tmp = data[i]
                data[i] = data[i + 1]
                data[i + 1] = tmp
                sorted = False


def verify_sorted(data: UnsafePointer[UInt32, _], N: Int) -> Bool:
    """Verify if array is sorted.

    Args:
        data: Array to verify.
        N: Size of array.

    Returns:
        True if sorted, False otherwise.
    """
    for i in range(N - 1):
        if data[i] > data[i + 1]:
            return False
    return True


def main() raises:
    var N = 10000

    # Allocate host memory
    var h_data = alloc[UInt32](N)
    var h_ref = alloc[UInt32](N)

    # Initialize with random data
    print("Initializing array with", N, "random elements")
    for i in range(N):
        var val = UInt32(random_ui64(0, 10000))
        h_data[i] = val
        h_ref[i] = val

    # Print first 10 elements before sorting
    print("First 10 elements before sorting: ", end="")
    for i in range(10):
        print(h_data[i], " ", end="")
    print("...")

    # Create device context
    var ctx = DeviceContext()

    # Allocate device memory
    var d_data = ctx.enqueue_create_buffer[.uint32](N)
    var d_hasChanged = ctx.enqueue_create_buffer[.uint32](1)

    # Copy to device
    ctx.enqueue_copy(d_data, h_data)

    # Launch configuration
    var threads_per_block = 256
    var num_blocks = (N // 2 + threads_per_block - 1) // threads_per_block

    print("\nLaunching odd-even sort kernel (Fig 14.2)")
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
    print("Each thread handles 2 adjacent elements")

    # Odd-even sort main loop
    var iteration = 0
    var max_iterations = N
    var h_hasChanged = alloc[UInt32](1)

    while iteration < max_iterations:
        # Reset hasChanged flag
        h_hasChanged[0] = 0
        ctx.enqueue_copy(d_hasChanged, h_hasChanged)

        # Odd phase
        ctx.enqueue_function[sort_kernel](
            d_data,
            d_hasChanged,
            Int32(N),
            Int32(1),
            Int32(threads_per_block),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(threads_per_block, 1, 1),
        )

        # Even phase
        ctx.enqueue_function[sort_kernel](
            d_data,
            d_hasChanged,
            Int32(N),
            Int32(0),
            Int32(threads_per_block),
            grid_dim=(num_blocks, 1, 1),
            block_dim=(threads_per_block, 1, 1),
        )

        # Check if any changes were made
        ctx.enqueue_copy(h_hasChanged, d_hasChanged)
        ctx.synchronize()

        iteration += 1
        if iteration % 10 == 0:
            print("Iteration", iteration, "...")

        if h_hasChanged[0] == 0:
            break

    print("GPU sorting completed in", iteration, "iterations")

    # Copy result back
    ctx.enqueue_copy(h_data, d_data)
    ctx.synchronize()

    # Print first 10 elements after sorting
    print("\nFirst 10 elements after GPU sorting: ", end="")
    for i in range(10):
        print(h_data[i], " ", end="")
    print("...")

    # Verify GPU result is sorted
    if verify_sorted(h_data, N):
        print("SUCCESS: GPU array is correctly sorted!")
    else:
        print("ERROR: GPU array is not sorted correctly!")

    # Run CPU reference for comparison
    print("\nRunning CPU reference implementation...")
    cpu_odd_even_sort(h_ref, N)
    print("CPU sorting completed")

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
    h_hasChanged.free()
