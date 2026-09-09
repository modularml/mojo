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

"""Figure 15.14: Matrix multiplication with double buffering (software pipelining)."""

from std.math import ceildiv
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.itertools import product
from std.memory import unsafe_stack_allocation

comptime bM = 64
comptime bN = 64
comptime bK = 32  # Reduced from 64 to fit double-buffered shared memory
comptime tM = 8
comptime tN = 4
comptime NUM_THREADS_PER_BLOCK = 128


@always_inline
def loadTile(
    T: UnsafePointer[Float32, _],
    lda: Int,
    maxRow: Int,
    maxCol: Int,
    T_s: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
    ldas: Int,
    height: Int,
    width: Int,
):
    """Load tile from global memory to shared memory.

    Args:
        T: Pointer to global memory tile.
        lda: Leading dimension of T (stride).
        maxRow: Maximum valid row to load.
        maxCol: Maximum valid column to load.
        T_s: Pointer to shared memory tile.
        ldas: Leading dimension of T_s (stride).
        height: Height of tile to load.
        width: Width of tile to load.
    """
    var num_rows_per_tile = NUM_THREADS_PER_BLOCK // width
    var num_subtiles = height // num_rows_per_tile

    var tx = thread_idx.x

    for subtile in range(num_subtiles):
        var row, col = divmod(tx, width)
        row += subtile * num_rows_per_tile

        if row < maxRow and col < maxCol:
            T_s[row * ldas + col] = Float32(T[row * lda + col])
        else:
            T_s[row * ldas + col] = Float32(0.0)


def mm_tiled_kernel_double_buffer(
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    B: UnsafePointer[Float32, ImmutAnyOrigin],
    C: UnsafePointer[Float32, MutAnyOrigin],
    M: UInt32,
    N: UInt32,
    K: UInt32,
):
    """Double-buffered tiled matrix multiplication kernel with software pipelining.

    This version uses double buffering to overlap computation with memory loads,
    hiding global memory latency and improving performance.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Number of columns in A and rows in B.
    """
    var bx = block_idx.x
    var by = block_idx.y
    var tx = thread_idx.x

    var tiles_across_x = bN // tN
    var tile_x, tile_y = divmod(tx, tiles_across_x)

    var bRow = bx * bM
    var bCol = by * bN
    var tRow = tile_x * tM
    var tCol = tile_y * tN

    # Register accumulator
    var Cr = SIMD[.float32, tM * tN](0.0)

    # Allocate double-buffered shared memory (2 sets of tiles)
    var A_s = unsafe_stack_allocation[
        2 * bM * bK,
        Float32,
        address_space=.SHARED,
    ]()
    var B_s = unsafe_stack_allocation[
        2 * bK * bN,
        Float32,
        address_space=.SHARED,
    ]()

    # Pointers to current and next buffers (using offsets)
    var A_curr_offset = 0
    var A_next_offset = bM * bK
    var B_curr_offset = 0
    var B_next_offset = bK * bN

    var numTiles = ceildiv(Int(K), bK)

    # Pre-fetch first iteration tiles
    var curr_offset = 0
    var addr_A = A + bRow * Int(K) + curr_offset
    var addr_B = B + curr_offset * Int(N) + bCol

    loadTile(
        addr_A,
        Int(K),
        Int(M) - bRow,
        Int(K) - curr_offset,
        A_s + A_curr_offset,
        bK,
        bM,
        bK,
    )
    loadTile(
        addr_B,
        Int(N),
        Int(K) - curr_offset,
        Int(N) - bCol,
        B_s + B_curr_offset,
        bN,
        bK,
        bN,
    )

    barrier()

    # Software pipeline: overlap compute with memory loads
    for tile in range(numTiles - 1):
        # Pre-fetch next iteration tiles while computing current
        curr_offset = (tile + 1) * bK
        addr_A = A + bRow * Int(K) + curr_offset
        addr_B = B + curr_offset * Int(N) + bCol

        loadTile(
            addr_A,
            Int(K),
            Int(M) - bRow,
            Int(K) - curr_offset,
            A_s + A_next_offset,
            bK,
            bM,
            bK,
        )
        loadTile(
            addr_B,
            Int(N),
            Int(K) - curr_offset,
            Int(N) - bCol,
            B_s + B_next_offset,
            bN,
            bK,
            bN,
        )

        # Compute with current iteration shared memory tiles
        for row in range(tM):
            for col in range(tN):
                for k in range(bK):
                    var a_idx = A_curr_offset + (tRow + row) * bK + k
                    var b_idx = B_curr_offset + k * bN + (tCol + col)
                    Cr[row * tN + col] += Float32(A_s[a_idx]) * Float32(
                        B_s[b_idx]
                    )

        barrier()

        # Swap double buffers (swap offsets)
        var tmp_A = A_curr_offset
        A_curr_offset = A_next_offset
        A_next_offset = tmp_A

        var tmp_B = B_curr_offset
        B_curr_offset = B_next_offset
        B_next_offset = tmp_B

    # Compute with last iteration shared memory tiles
    for row in range(tM):
        for col in range(tN):
            for k in range(bK):
                var a_idx = A_curr_offset + (tRow + row) * bK + k
                var b_idx = B_curr_offset + k * bN + (tCol + col)
                Cr[row * tN + col] += Float32(A_s[a_idx]) * Float32(B_s[b_idx])

    # Write results to global memory
    for row in range(tM):
        for col in range(tN):
            var out_row = bRow + tRow + row
            var out_col = bCol + tCol + col

            if out_row < Int(M) and out_col < Int(N):
                C[out_row * Int(N) + out_col] = Cr[row * tN + col]


def cpu_mm(
    A: UnsafePointer[Float32, _],
    B: UnsafePointer[Float32, _],
    C: UnsafePointer[mut=True, Float32, _],
    M: Int,
    N: Int,
    K: Int,
):
    """CPU matrix multiplication for verification.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Number of columns in A and rows in B.
    """
    for i, j in product(range(M), range(N)):
        var sum = Float32(0.0)
        for k in range(K):
            sum += A[i * K + k] * B[k * N + j]
        C[i * N + j] = sum


def main() raises:
    comptime M = 1024
    comptime N = 1024
    comptime K = 1024

    print(
        "Testing double-buffered matrix multiplication:",
        M,
        "x",
        K,
        "*",
        K,
        "x",
        N,
        "=",
        M,
        "x",
        N,
    )
    print("Using software pipelining to overlap computation and memory loads")

    # Allocate host memory
    var h_A = alloc[Float32](M * K)
    var h_B = alloc[Float32](K * N)
    var h_C_gpu = alloc[Float32](M * N)
    var h_C_cpu = alloc[Float32](M * N)

    # Initialize matrices
    print("Initializing matrices...")
    for i in range(M * K):
        h_A[i] = Float32(i + 1)
    for i in range(K * N):
        h_B[i] = Float32(i + 1)

    # GPU computation
    with DeviceContext() as ctx:
        comptime dtype = DType.float32

        # Allocate device memory
        var d_A = ctx.enqueue_create_buffer[dtype](M * K)
        var d_B = ctx.enqueue_create_buffer[dtype](K * N)
        var d_C = ctx.enqueue_create_buffer[dtype](M * N)

        # Copy data to device
        print("Copying data to GPU...")
        ctx.enqueue_copy(d_A, h_A)
        ctx.enqueue_copy(d_B, h_B)

        # Launch kernel
        var grid_x = ceildiv(M, bM)
        var grid_y = ceildiv(N, bN)

        print(
            "Launching GPU kernel with grid(",
            grid_x,
            ",",
            grid_y,
            ") and block(",
            NUM_THREADS_PER_BLOCK,
            ")...",
        )

        ctx.enqueue_function[mm_tiled_kernel_double_buffer](
            d_A,
            d_B,
            d_C,
            UInt32(M),
            UInt32(N),
            UInt32(K),
            grid_dim=(grid_x, grid_y, 1),
            block_dim=(NUM_THREADS_PER_BLOCK, 1, 1),
        )

        # Copy result back
        ctx.enqueue_copy(h_C_gpu, d_C)
        ctx.synchronize()

    # CPU verification
    print("Running CPU verification...")
    cpu_mm(h_A, h_B, h_C_cpu, M, N, K)
    print("CPU computation complete")

    # Verify results
    print("Verifying results...")
    var max_error = Float32(0.0)
    var avg_error = Float32(0.0)
    var num_errors = 0
    comptime tolerance = 1e-2

    for i in range(M * N):
        var error = abs(h_C_gpu[i] - h_C_cpu[i])
        var relative_error = error / (abs(h_C_cpu[i]) + 1e-7)

        if relative_error > tolerance:
            num_errors += 1
            if num_errors <= 10:
                print(
                    "Mismatch at index",
                    i,
                    ": GPU=",
                    h_C_gpu[i],
                    "CPU=",
                    h_C_cpu[i],
                    "error=",
                    error,
                )

        if error > max_error:
            max_error = error
        avg_error += error

    avg_error /= Float32(M * N)

    print("\n=== Results ===")
    print("Maximum error:", max_error)
    print("Average error:", avg_error)
    print(
        "Number of mismatches (tolerance",
        tolerance,
        "):",
        num_errors,
        "out of",
        M * N,
    )

    if num_errors == 0:
        print("✓ SUCCESS: All results match within tolerance!")
    else:
        print("✗ FAILED: Results do not match!")

    # Cleanup
    h_A.free()
    h_B.free()
    h_C_gpu.free()
    h_C_cpu.free()
