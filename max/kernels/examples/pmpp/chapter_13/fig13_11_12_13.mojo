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

"""Figures 13.11, 13.12, 13.13: Tiled merge kernel implementation in Mojo."""

from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.math import min, max


def ceildiv(a: Int, b: Int) -> Int:
    """Ceiling division."""
    return (a + b - 1) // b


def co_rank(
    k: Int,
    A: UnsafePointer[Int32, _],
    m: Int,
    B: UnsafePointer[Int32, _],
    n: Int,
) -> Int:
    """Find the co-rank of k in the merged sequence of A and B.

    Args:
        k: Target position in merged sequence.
        A: Sorted array A.
        m: Length of array A.
        B: Sorted array B.
        n: Length of array B.

    Returns:
        The number of elements from A that should come before position k.
    """
    var i = min(k, m)
    var j = k - i
    var i_low = max(k - n, 0)
    var j_low = max(k - m, 0)
    var delta: Int

    while True:
        if i > 0 and j < n and A[i - 1] > B[j]:
            delta = ceildiv(i - i_low, 2)
            j_low = j
            j = j + delta
            i = i - delta
        elif j > 0 and i < m and B[j - 1] >= A[i]:
            delta = ceildiv(j - j_low, 2)
            i_low = i
            i = i + delta
            j = j - delta
        else:
            return i


def co_rank_shared(
    k: Int,
    A: UnsafePointer[Int32, _, address_space=.SHARED],
    m: Int,
    B: UnsafePointer[Int32, _, address_space=.SHARED],
    n: Int,
) -> Int:
    """Find the co-rank for shared memory arrays.

    Args:
        k: Target position in merged sequence.
        A: Sorted array A in shared memory.
        m: Length of array A.
        B: Sorted array B in shared memory.
        n: Length of array B.

    Returns:
        The number of elements from A that should come before position k.
    """
    var i = min(k, m)
    var j = k - i
    var i_low = max(k - n, 0)
    var j_low = max(k - m, 0)
    var delta: Int

    while True:
        if i > 0 and j < n and Int32(A[i - 1]) > Int32(B[j]):
            delta = ceildiv(i - i_low, 2)
            j_low = j
            j = j + delta
            i = i - delta
        elif j > 0 and i < m and Int32(B[j - 1]) >= Int32(A[i]):
            delta = ceildiv(j - j_low, 2)
            i_low = i
            i = i + delta
            j = j - delta
        else:
            return i


def merge_sequential(
    A: UnsafePointer[Int32, _],
    m: Int,
    B: UnsafePointer[Int32, _],
    n: Int,
    C: UnsafePointer[Int32, MutAnyOrigin],
):
    """Sequential merge of two sorted arrays.

    Args:
        A: First sorted array.
        m: Length of A.
        B: Second sorted array.
        n: Length of B.
        C: Output array.
    """
    var i = 0
    var j = 0
    var k = 0

    while i < m and j < n:
        if A[i] <= B[j]:
            C[k] = A[i]
            k += 1
            i += 1
        else:
            C[k] = B[j]
            k += 1
            j += 1

    while i < m:
        C[k] = A[i]
        k += 1
        i += 1

    while j < n:
        C[k] = B[j]
        k += 1
        j += 1


def merge_sequential_shared(
    A: UnsafePointer[Int32, _, address_space=.SHARED],
    m: Int,
    B: UnsafePointer[Int32, _, address_space=.SHARED],
    n: Int,
    C: UnsafePointer[Int32, MutAnyOrigin],
):
    """Sequential merge of two sorted arrays from shared memory.

    Args:
        A: First sorted array in shared memory.
        m: Length of A.
        B: Second sorted array in shared memory.
        n: Length of B.
        C: Output array in global memory.
    """
    var i = 0
    var j = 0
    var k = 0

    while i < m and j < n:
        if Int32(A[i]) <= Int32(B[j]):
            C[k] = Int32(A[i])
            k += 1
            i += 1
        else:
            C[k] = Int32(B[j])
            k += 1
            j += 1

    while i < m:
        C[k] = Int32(A[i])
        k += 1
        i += 1

    while j < n:
        C[k] = Int32(B[j])
        k += 1
        j += 1


def merge_tiled_kernel(
    A: UnsafePointer[Int32, ImmutAnyOrigin],
    m_dev: Int32,
    B: UnsafePointer[Int32, ImmutAnyOrigin],
    n_dev: Int32,
    C: UnsafePointer[Int32, MutAnyOrigin],
    tile_size_dev: Int32,
):
    """Tiled merge kernel using shared memory.

    Args:
        A: First sorted array.
        m_dev: Length of A.
        B: Second sorted array.
        n_dev: Length of B.
        C: Output merged array.
        tile_size_dev: Size of each tile.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var tile_size = Int(tile_size_dev)
    # Allocate shared memory
    var A_S = unsafe_stack_allocation[
        1024,
        Int32,
        address_space=.SHARED,
    ]()
    var B_S = unsafe_stack_allocation[
        1024,
        Int32,
        address_space=.SHARED,
    ]()

    # Figure 13.11: Part 1 - Block-level co-rank
    var C_curr = block_idx.x * ceildiv(
        m + n, Int(16)
    )  # 16 blocks hardcoded for now
    var C_next = min((block_idx.x + 1) * ceildiv(m + n, Int(16)), m + n)

    if thread_idx.x == 0:
        # Store co-rank values in first two elements of A_S
        A_S[0] = Int32(co_rank(C_curr, A, m, B, n))
        A_S[1] = Int32(co_rank(C_next, A, m, B, n))

    barrier()

    var A_curr = Int32(A_S[0])
    var A_next = Int32(A_S[1])
    var B_curr = C_curr - Int(A_curr)
    var B_next = C_next - Int(A_next)

    barrier()

    # Calculate total work for this block
    var counter = 0
    var C_length = C_next - C_curr
    var A_length = Int(A_next) - Int(A_curr)
    var B_length = B_next - B_curr
    var total_iteration = ceildiv(C_length, tile_size)
    var C_completed = 0
    var A_consumed = 0
    var B_consumed = 0

    # Figure 13.12 & 13.13: Iterative tiled merge
    while counter < total_iteration:
        # Figure 13.12: Part 2 - Loading A and B elements into shared memory
        var leftover_B = B_length - B_consumed
        var leftover_A = A_length - A_consumed

        for i in range(0, tile_size, Int(128)):  # blockDim.x = 128
            if i + thread_idx.x < leftover_A:
                A_S[i + thread_idx.x] = Int32(
                    A[Int(A_curr) + A_consumed + i + thread_idx.x]
                )

        for i in range(0, tile_size, Int(128)):  # blockDim.x = 128
            if i + thread_idx.x < leftover_B:
                B_S[i + thread_idx.x] = Int32(
                    B[B_curr + B_consumed + i + thread_idx.x]
                )

        barrier()

        # Figure 13.13: Part 3 - All threads merge their individual subarrays
        var c_curr = thread_idx.x * (tile_size // Int(128))  # blockDim.x = 128
        var c_next = (thread_idx.x + 1) * (tile_size // Int(128))
        var leftover_c = C_length - C_completed

        c_curr = min(c_curr, leftover_c)
        c_next = min(c_next, leftover_c)

        # Find co-rank for this thread's portion in shared memory
        var a_curr = co_rank_shared(
            c_curr,
            A_S,
            min(tile_size, leftover_A),
            B_S,
            min(tile_size, leftover_B),
        )
        var b_curr = c_curr - a_curr
        var a_next = co_rank_shared(
            c_next,
            A_S,
            min(tile_size, leftover_A),
            B_S,
            min(tile_size, leftover_B),
        )
        var b_next = c_next - a_next

        # All threads call the sequential merge function
        merge_sequential_shared(
            A_S + a_curr,
            a_next - a_curr,
            B_S + b_curr,
            b_next - b_curr,
            C + C_curr + C_completed + c_curr,
        )

        # Update the number of A and B elements that have been consumed thus far
        counter += 1
        var tile_elements = min(tile_size, leftover_c)
        var tile_A_consumed = co_rank_shared(
            tile_elements,
            A_S,
            min(tile_size, leftover_A),
            B_S,
            min(tile_size, leftover_B),
        )
        C_completed += tile_elements
        A_consumed += tile_A_consumed
        B_consumed += tile_elements - tile_A_consumed

        barrier()


def cpu_merge(
    A: UnsafePointer[Int32, _],
    m: Int,
    B: UnsafePointer[Int32, _],
    n: Int,
    C: UnsafePointer[mut=True, Int32, _],
):
    """CPU reference implementation of merge.

    Args:
        A: First sorted array.
        m: Length of A.
        B: Second sorted array.
        n: Length of B.
        C: Output merged array.
    """
    var i = 0
    var j = 0
    var k = 0

    while i < m and j < n:
        if A[i] <= B[j]:
            C[k] = A[i]
            k += 1
            i += 1
        else:
            C[k] = B[j]
            k += 1
            j += 1

    while i < m:
        C[k] = A[i]
        k += 1
        i += 1

    while j < n:
        C[k] = B[j]
        k += 1
        j += 1


def main() raises:
    var m = 10000
    var n = 10000
    var total = m + n

    # Allocate host memory
    var h_A = alloc[Int32](m)
    var h_B = alloc[Int32](n)
    var h_C = alloc[Int32](total)
    var h_ref = alloc[Int32](total)

    # Initialize sorted arrays
    for i in range(m):
        h_A[i] = Int32(i * 2)  # Even numbers

    for i in range(n):
        h_B[i] = Int32(i * 2 + 1)  # Odd numbers

    # Create device context
    var ctx = DeviceContext()

    # Allocate device memory
    var d_A = ctx.enqueue_create_buffer[.int32](m)
    var d_B = ctx.enqueue_create_buffer[.int32](n)
    var d_C = ctx.enqueue_create_buffer[.int32](total)

    # Copy to device
    ctx.enqueue_copy(d_A, h_A)
    ctx.enqueue_copy(d_B, h_B)

    # Launch tiled merge kernel
    var tile_size = 1024
    var threads_per_block = 128
    var num_blocks = 16

    print("Launching tiled merge kernel (Fig 13.11-13)")
    print("Arrays: A[", m, "], B[", n, "] -> C[", total, "]")
    print("Config: ", num_blocks, " blocks x ", threads_per_block, " threads")
    print("Tile size: ", tile_size)

    ctx.enqueue_function[merge_tiled_kernel](
        d_A,
        Int32(m),
        d_B,
        Int32(n),
        d_C,
        Int32(tile_size),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(threads_per_block, 1, 1),
    )

    # Copy result back
    ctx.enqueue_copy(h_C, d_C)
    ctx.synchronize()

    # Run CPU reference
    cpu_merge(h_A, m, h_B, n, h_ref)

    # Print sample
    print("\nFirst 20 merged: ", end="")
    for i in range(min(20, total)):
        print(h_C[i], " ", end="")
    print("...")

    # Verify results
    var errors = 0
    for i in range(total):
        if h_C[i] != h_ref[i]:
            if errors < 10:
                print(
                    "Error at index ", i, ": CPU=", h_ref[i], ", GPU=", h_C[i]
                )
            errors += 1

    # Free memory
    h_A.free()
    h_B.free()
    h_C.free()
    h_ref.free()

    if errors == 0:
        print("SUCCESS: All values match!")
        print("Merged ", total, " elements correctly using tiled approach")
    else:
        print("FAILED: Found ", errors, " errors")
