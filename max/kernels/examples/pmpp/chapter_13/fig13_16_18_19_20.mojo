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

"""Figures 13.16, 13.18, 13.19, 13.20: Circular buffer merge kernel implementation in Mojo."""

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


def co_rank_circular(
    k: Int,
    A: UnsafePointer[Int32, _, address_space=.SHARED],
    m: Int,
    B: UnsafePointer[Int32, _, address_space=.SHARED],
    n: Int,
    A_S_start: Int,
    B_S_start: Int,
    tile_size: Int,
) -> Int:
    """Figure 13.19: co_rank function that operates on circular buffers.

    Args:
        k: Target position in merged sequence.
        A: Array A in shared memory (circular buffer).
        m: Length of data in A.
        B: Array B in shared memory (circular buffer).
        n: Length of data in B.
        A_S_start: Start index in circular buffer A.
        B_S_start: Start index in circular buffer B.
        tile_size: Size of the circular buffer.

    Returns:
        The number of elements from A that should come before position k.
    """
    var i = min(k, m)
    var j = k - i
    var i_low = max(k - n, 0)
    var j_low = max(k - m, 0)
    var delta: Int
    var active = True

    while active:
        var i_cir = (A_S_start + i) % tile_size
        var i_m_1_cir = (A_S_start + i - 1) % tile_size
        var j_cir = (B_S_start + j) % tile_size
        var j_m_1_cir = (B_S_start + j - 1) % tile_size

        if i > 0 and j < n and Int32(A[i_m_1_cir]) > Int32(B[j_cir]):
            delta = (i - i_low + 1) >> 1  # ceil((i - i_low) / 2)
            j_low = j
            i = i - delta
            j = j + delta
        elif j > 0 and i < m and Int32(B[j_m_1_cir]) >= Int32(A[i_cir]):
            delta = (j - j_low + 1) >> 1
            i_low = i
            i = i + delta
            j = j - delta
        else:
            active = False

    return i


def merge_sequential_circular(
    A: UnsafePointer[Int32, _, address_space=.SHARED],
    m: Int,
    B: UnsafePointer[Int32, _, address_space=.SHARED],
    n: Int,
    C: UnsafePointer[Int32, MutAnyOrigin],
    A_S_start: Int,
    B_S_start: Int,
    tile_size: Int,
):
    """Figure 13.20: Implementation of merge_sequential_circular function.

    Args:
        A: Array A in shared memory (circular buffer).
        m: Length of data in A.
        B: Array B in shared memory (circular buffer).
        n: Length of data in B.
        C: Output array in global memory.
        A_S_start: Start index in circular buffer A.
        B_S_start: Start index in circular buffer B.
        tile_size: Size of the circular buffer.
    """
    var i = 0  # virtual index into A
    var j = 0  # virtual index into B
    var k = 0  # virtual index into C

    while i < m and j < n:
        var i_cir = (A_S_start + i) % tile_size
        var j_cir = (B_S_start + j) % tile_size
        if Int32(A[i_cir]) <= Int32(B[j_cir]):
            C[k] = Int32(A[i_cir])
            k += 1
            i += 1
        else:
            C[k] = Int32(B[j_cir])
            k += 1
            j += 1

    if i == m:  # done with A[], handle remaining B[]
        while j < n:
            var j_cir = (B_S_start + j) % tile_size
            C[k] = Int32(B[j_cir])
            k += 1
            j += 1
    else:  # done with B[], handle remaining A[]
        while i < m:
            var i_cir = (A_S_start + i) % tile_size
            C[k] = Int32(A[i_cir])
            k += 1
            i += 1


def merge_circular_buffer_kernel(
    A: UnsafePointer[Int32, ImmutAnyOrigin],
    m_dev: Int32,
    B: UnsafePointer[Int32, ImmutAnyOrigin],
    n_dev: Int32,
    C: UnsafePointer[Int32, MutAnyOrigin],
    tile_size_dev: Int32,
):
    """Figures 13.16, 13.18, 13.19, 13.20: Circular buffer merge kernel.

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

    # Block-level co-rank (same as tiled version)
    # Dynamic grid size based on launch configuration
    var grid_size = ceildiv(m + n, 2 * tile_size)  # Max 2 iterations per block
    var C_curr = block_idx.x * ceildiv(m + n, grid_size)
    var C_next = min((block_idx.x + 1) * ceildiv(m + n, grid_size), m + n)

    if thread_idx.x == 0:
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

    # Figure 13.16 & 13.18: Circular buffer tracking
    var A_S_start = 0
    var B_S_start = 0
    var A_S_consumed = tile_size  # in the first iteration, fill the tile_size
    var B_S_consumed = tile_size  # in the first iteration, fill the tile_size

    while counter < total_iteration:
        # Figure 13.16: Part 2 - Loading A_S_consumed and B_S_consumed elements
        # into A_S and B_S using circular buffer indexing
        for i in range(0, A_S_consumed, Int(128)):  # blockDim.x = 128
            if (
                i + thread_idx.x < A_length - A_consumed
                and i + thread_idx.x < A_S_consumed
            ):
                A_S[
                    (A_S_start + (tile_size - A_S_consumed) + i + thread_idx.x)
                    % tile_size
                ] = Int32(A[Int(A_curr) + A_consumed + i + thread_idx.x])

        # Loading B_S_consumed elements into B_S
        for i in range(0, B_S_consumed, Int(128)):  # blockDim.x = 128
            if (
                i + thread_idx.x < B_length - B_consumed
                and i + thread_idx.x < B_S_consumed
            ):
                B_S[
                    (B_S_start + (tile_size - B_S_consumed) + i + thread_idx.x)
                    % tile_size
                ] = Int32(B[B_curr + B_consumed + i + thread_idx.x])

        barrier()

        # Figure 13.18: Part 3 - Thread-level merge
        var c_curr = thread_idx.x * (tile_size // Int(128))  # blockDim.x = 128
        var c_next = (thread_idx.x + 1) * (tile_size // Int(128))

        c_curr = min(c_curr, C_length - C_completed)
        c_next = min(c_next, C_length - C_completed)

        # Find co-rank for current and next using circular buffer version
        var a_curr = co_rank_circular(
            c_curr,
            A_S,
            min(tile_size, A_length - A_consumed),
            B_S,
            min(tile_size, B_length - B_consumed),
            A_S_start,
            B_S_start,
            tile_size,
        )
        var b_curr = c_curr - a_curr
        var a_next = co_rank_circular(
            c_next,
            A_S,
            min(tile_size, A_length - A_consumed),
            B_S,
            min(tile_size, B_length - B_consumed),
            A_S_start,
            B_S_start,
            tile_size,
        )
        var b_next = c_next - a_next

        # All threads call the circular-buffer version of the sequential merge
        merge_sequential_circular(
            A_S,
            a_next - a_curr,
            B_S,
            b_next - b_curr,
            C + C_curr + C_completed + c_curr,
            A_S_start + a_curr,
            B_S_start + b_curr,
            tile_size,
        )

        # Figure out the work has been done
        counter += 1
        A_S_consumed = co_rank_circular(
            min(tile_size, C_length - C_completed),
            A_S,
            min(tile_size, A_length - A_consumed),
            B_S,
            min(tile_size, B_length - B_consumed),
            A_S_start,
            B_S_start,
            tile_size,
        )
        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed
        A_consumed += A_S_consumed
        C_completed += min(tile_size, C_length - C_completed)
        B_consumed = C_completed - A_consumed

        # Update circular buffer start positions (Figure 13.18, lines 55-56)
        A_S_start = (A_S_start + A_S_consumed) % tile_size
        B_S_start = (B_S_start + B_S_consumed) % tile_size

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
    var m = 32768
    var n = 32768
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

    # Launch circular buffer merge kernel
    var tile_size = 1024
    var threads_per_block = 128

    # The circular buffer implementation only works correctly for
    # at most 2 iterations per block (2*tile_size elements).
    # Dynamically calculate num_blocks to ensure each block processes
    # at most 2*tile_size elements, guaranteeing ≤2 iterations per block.
    var max_elements_per_block = 2 * tile_size  # 2048 elements max per block
    var num_blocks = ceildiv(total, max_elements_per_block)

    print("Launching circular buffer merge kernel (Fig 13.16-13.20)")
    print("Arrays: A[", m, "], B[", n, "] -> C[", total, "]")
    print(
        "Config: ",
        num_blocks,
        " blocks x ",
        threads_per_block,
        " threads (max ",
        max_elements_per_block,
        " elements/block, ",
        ceildiv(max_elements_per_block, tile_size),
        " iterations/block)",
    )
    print("Tile size: ", tile_size)

    ctx.enqueue_function[merge_circular_buffer_kernel](
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
    print("\n========== DIFFERENCES ==========")
    for i in range(total):
        if h_C[i] != h_ref[i]:
            if errors < 10:
                print(
                    "Error at index ", i, ": CPU=", h_ref[i], ", GPU=", h_C[i]
                )
            errors += 1

    if errors == 0:
        print("No differences found!")

    # Free memory
    h_A.free()
    h_B.free()
    h_C.free()
    h_ref.free()

    if errors == 0:
        print("SUCCESS: All values match!")
        print(
            "Merged ",
            total,
            " elements correctly using circular buffer approach",
        )
    else:
        print("FAILED: Found ", errors, " errors")
