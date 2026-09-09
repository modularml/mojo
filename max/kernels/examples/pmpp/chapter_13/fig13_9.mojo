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

"""Figure 13.9: Basic merge kernel implementation in Mojo."""

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import min, max


def ceildiv(a: Int, b: Int) -> Int:
    """Ceiling division."""
    return (a + b - 1) // b


def co_rank(
    k: Int,
    m: Int,
    n: Int,
    A: UnsafePointer[Int32, _],
    B: UnsafePointer[Int32, _],
) -> Int:
    """Find the co-rank of k in the merged sequence of A and B.

    Args:
        k: Target position in merged sequence.
        m: Length of array A.
        n: Length of array B.
        A: Sorted array A.
        B: Sorted array B.

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
            active = False

    return i


def local_subset_merge(
    A: UnsafePointer[Int32, _],
    m_local: Int,
    B: UnsafePointer[Int32, _],
    n_local: Int,
    C: UnsafePointer[Int32, MutAnyOrigin],
):
    """Sequential merge of two sorted subarrays.

    Args:
        A: First sorted array.
        m_local: Length of A.
        B: Second sorted array.
        n_local: Length of B.
        C: Output array.
    """
    var i = 0
    var j = 0
    var k = 0
    var total = m_local + n_local

    for idx in range(total):
        if i < m_local and (j >= n_local or A[i] <= B[j]):
            C[k] = A[i]
            k += 1
            i += 1
        else:
            C[k] = B[j]
            k += 1
            j += 1


def merge_basic_kernel(
    A: UnsafePointer[Int32, ImmutAnyOrigin],
    m_dev: Int32,
    B: UnsafePointer[Int32, ImmutAnyOrigin],
    n_dev: Int32,
    C: UnsafePointer[Int32, MutAnyOrigin],
    block_dim_x_dev: Int32,
    grid_dim_x_dev: Int32,
):
    """Basic merge kernel using co-rank to distribute work among threads.

    Args:
        A: First sorted array.
        m_dev: Length of A.
        B: Second sorted array.
        n_dev: Length of B.
        C: Output merged array.
        block_dim_x_dev: Block dimension (threads per block).
        grid_dim_x_dev: Grid dimension (number of blocks).
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var m = Int(m_dev)
    var n = Int(n_dev)
    var block_dim_x = Int(block_dim_x_dev)
    var grid_dim_x = Int(grid_dim_x_dev)
    var tid = block_idx.x * block_dim_x + thread_idx.x
    var total_elements = m + n
    var total_threads = block_dim_x * grid_dim_x
    var elements_per_thread = ceildiv(total_elements, total_threads)

    var k_begin = tid * elements_per_thread

    var a_idx = co_rank(k_begin, m, n, A, B)
    var b_idx = k_begin - a_idx

    # Calculate remaining elements in each array from this point
    var m_local = m - a_idx
    var n_local = n - b_idx

    # Perform local merge
    local_subset_merge(A + a_idx, m_local, B + b_idx, n_local, C + k_begin)


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
    var m = 5000  # Size of A
    var n = 5000  # Size of B
    var total = m + n

    # Allocate host memory
    var h_A = alloc[Int32](m)
    var h_B = alloc[Int32](n)
    var h_C = alloc[Int32](total)
    var h_ref = alloc[Int32](total)

    # Initialize sorted arrays
    for i in range(m):
        h_A[i] = Int32(i * 2)  # Even numbers: 0, 2, 4, 6, ...

    for i in range(n):
        h_B[i] = Int32(i * 2 + 1)  # Odd numbers: 1, 3, 5, 7, ...

    # Create device context
    var ctx = DeviceContext()

    # Allocate device memory
    var d_A = ctx.enqueue_create_buffer[.int32](m)
    var d_B = ctx.enqueue_create_buffer[.int32](n)
    var d_C = ctx.enqueue_create_buffer[.int32](total)

    # Copy to device
    ctx.enqueue_copy(d_A, h_A)
    ctx.enqueue_copy(d_B, h_B)

    # Launch kernel with 4-5 elements per thread
    var threads_per_block = 256
    var elements_per_thread = 4
    var total_threads = ceildiv(total, elements_per_thread)
    var num_blocks = ceildiv(total_threads, threads_per_block)

    print("Launching merge kernel (Fig 13.9)")
    print("Arrays: A[", m, "], B[", n, "] -> C[", total, "]")
    print(
        "Config: ",
        num_blocks,
        " blocks x ",
        threads_per_block,
        " threads = ",
        num_blocks * threads_per_block,
        " threads total",
    )
    print("Elements per thread: ", elements_per_thread)

    ctx.enqueue_function[merge_basic_kernel](
        d_A,
        Int32(m),
        d_B,
        Int32(n),
        d_C,
        Int32(threads_per_block),
        Int32(num_blocks),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(threads_per_block, 1, 1),
    )

    # Copy result back
    ctx.enqueue_copy(h_C, d_C)
    ctx.synchronize()

    # Run CPU reference
    cpu_merge(h_A, m, h_B, n, h_ref)

    # Print sample to verify
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
        print("Merged ", total, " elements correctly")
    else:
        print("FAILED: Found ", errors, " errors")
