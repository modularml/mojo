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
from layout.layout_tensor import Layout, LayoutTensor

comptime bM = 64
comptime bN = 64
comptime bK = 32  # Reduced from 64 to fit double-buffered shared memory
comptime tM = 8
comptime tN = 4
comptime NUM_THREADS = 128


def mm_tiled_kernel_double_buffer(
    A: UnsafePointer[Float32, ImmutAnyOrigin],
    B: UnsafePointer[Float32, ImmutAnyOrigin],
    C: UnsafePointer[Float32, MutAnyOrigin],
    M: UInt32,
    N: UInt32,
    K: UInt32,
):
    """Double-buffered tiled matrix multiplication kernel with software pipelining.

    Uses LayoutTensor for shared/local memory management to demonstrate idiomatic
    Mojo GPU programming while maintaining CUDA algorithm structure.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        M: Number of rows in A and C.
        N: Number of columns in B and C.
        K: Number of columns in A and rows in B.
    """
    comptime dtype = DType.float32

    var bx = block_idx.x
    var by = block_idx.y
    var tx = thread_idx.x

    var tiles_across_x = bN // tN
    var tile_x, tile_y = divmod(tx, tiles_across_x)

    var bRow = bx * bM
    var bCol = by * bN
    var tRow = tile_x * tM
    var tCol = tile_y * tN

    # Allocate register tile using LayoutTensor (LOCAL address space)
    var dst_reg = LayoutTensor[
        dtype,
        Layout.row_major(tM, tN),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    # Initialize to zero
    for i in range(tM):
        for j in range(tN):
            dst_reg[i, j] = 0.0

    # Allocate double-buffered shared memory using LayoutTensor (SHARED address space)
    var a_smem_0 = LayoutTensor[
        dtype,
        Layout.row_major(bM, bK),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()
    var a_smem_1 = LayoutTensor[
        dtype,
        Layout.row_major(bM, bK),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    var b_smem_0 = LayoutTensor[
        dtype,
        Layout.row_major(bK, bN),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()
    var b_smem_1 = LayoutTensor[
        dtype,
        Layout.row_major(bK, bN),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    var numTiles = ceildiv(Int(K), bK)
    var curr_uses_0 = True

    # Pre-fetch first iteration tiles (tile 0) - cooperative loading
    var curr_offset = 0
    var num_rows_per_tile_a = NUM_THREADS // bK
    var num_subtiles_a = bM // num_rows_per_tile_a

    for subtile in range(num_subtiles_a):
        var row, col = divmod(tx, bK)
        row += subtile * num_rows_per_tile_a
        var global_row = bRow + row
        var global_col = curr_offset + col

        if global_row < Int(M) and global_col < Int(K):
            a_smem_0[row, col] = A[global_row * Int(K) + global_col]
        else:
            a_smem_0[row, col] = 0.0

    var num_rows_per_tile_b = NUM_THREADS // bN
    var num_subtiles_b = bK // num_rows_per_tile_b

    for subtile in range(num_subtiles_b):
        var row, col = divmod(tx, bN)
        row += subtile * num_rows_per_tile_b
        var global_row = curr_offset + row
        var global_col = bCol + col

        if global_row < Int(K) and global_col < Int(N):
            b_smem_0[row, col] = B[global_row * Int(N) + global_col]
        else:
            b_smem_0[row, col] = 0.0

    barrier()

    # Software pipeline: overlap compute with memory loads
    for tile in range(numTiles - 1):
        # Pre-fetch next iteration tiles
        curr_offset = (tile + 1) * bK

        # Load next tiles into alternate buffer
        if curr_uses_0:
            # Load into buffer 1
            for subtile in range(num_subtiles_a):
                var row, col = divmod(tx, bK)
                row += subtile * num_rows_per_tile_a
                var global_row = bRow + row
                var global_col = curr_offset + col

                if global_row < Int(M) and global_col < Int(K):
                    a_smem_1[row, col] = A[global_row * Int(K) + global_col]
                else:
                    a_smem_1[row, col] = 0.0

            for subtile in range(num_subtiles_b):
                var row, col = divmod(tx, bN)
                row += subtile * num_rows_per_tile_b
                var global_row = curr_offset + row
                var global_col = bCol + col

                if global_row < Int(K) and global_col < Int(N):
                    b_smem_1[row, col] = B[global_row * Int(N) + global_col]
                else:
                    b_smem_1[row, col] = 0.0

            # Compute with buffer 0
            for row in range(tM):
                for col in range(tN):
                    for k in range(bK):
                        var a_val = a_smem_0[tRow + row, k]
                        var b_val = b_smem_0[k, tCol + col]
                        dst_reg[row, col] = dst_reg[row, col] + a_val * b_val
        else:
            # Load into buffer 0
            for subtile in range(num_subtiles_a):
                var row, col = divmod(tx, bK)
                row += subtile * num_rows_per_tile_a
                var global_row = bRow + row
                var global_col = curr_offset + col

                if global_row < Int(M) and global_col < Int(K):
                    a_smem_0[row, col] = A[global_row * Int(K) + global_col]
                else:
                    a_smem_0[row, col] = 0.0

            for subtile in range(num_subtiles_b):
                var row, col = divmod(tx, bN)
                row += subtile * num_rows_per_tile_b
                var global_row = curr_offset + row
                var global_col = bCol + col

                if global_row < Int(K) and global_col < Int(N):
                    b_smem_0[row, col] = B[global_row * Int(N) + global_col]
                else:
                    b_smem_0[row, col] = 0.0

            # Compute with buffer 1
            for row in range(tM):
                for col in range(tN):
                    for k in range(bK):
                        var a_val = a_smem_1[tRow + row, k]
                        var b_val = b_smem_1[k, tCol + col]
                        dst_reg[row, col] = dst_reg[row, col] + a_val * b_val

        barrier()
        curr_uses_0 = not curr_uses_0

    # Compute with last iteration tiles
    if curr_uses_0:
        for row in range(tM):
            for col in range(tN):
                for k in range(bK):
                    var a_val = a_smem_0[tRow + row, k]
                    var b_val = b_smem_0[k, tCol + col]
                    dst_reg[row, col] = dst_reg[row, col] + a_val * b_val
    else:
        for row in range(tM):
            for col in range(tN):
                for k in range(bK):
                    var a_val = a_smem_1[tRow + row, k]
                    var b_val = b_smem_1[k, tCol + col]
                    dst_reg[row, col] = dst_reg[row, col] + a_val * b_val

    # Write results to global memory
    for row in range(tM):
        for col in range(tN):
            var out_row = bRow + tRow + row
            var out_col = bCol + tCol + col

            if out_row < Int(M) and out_col < Int(N):
                var val = dst_reg[row, col]
                C[out_row * Int(N) + out_col] = val[0]


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
    comptime dtype = DType.float32

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
    print("Using LayoutTensor for shared/local memory management")

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
    for i in range(M * N):
        h_C_gpu[i] = 0.0

    # GPU computation
    with DeviceContext() as ctx:
        # Allocate device memory
        var d_A = ctx.enqueue_create_buffer[dtype](M * K)
        var d_B = ctx.enqueue_create_buffer[dtype](K * N)
        var d_C = ctx.enqueue_create_buffer[dtype](M * N)

        # Copy data to device
        print("Copying data to GPU...")
        ctx.enqueue_copy(d_A, h_A)
        ctx.enqueue_copy(d_B, h_B)
        ctx.enqueue_copy(d_C, h_C_gpu)

        # Launch kernel
        var grid_x = ceildiv(M, bM)
        var grid_y = ceildiv(N, bN)

        print(
            "Launching GPU kernel with grid(",
            grid_x,
            ",",
            grid_y,
            ") and block(",
            NUM_THREADS,
            ")...",
        )
        print("Using LayoutTensor-based double buffering")

        ctx.enqueue_function[mm_tiled_kernel_double_buffer](
            d_A,
            d_B,
            d_C,
            UInt32(M),
            UInt32(N),
            UInt32(K),
            grid_dim=(grid_x, grid_y, 1),
            block_dim=(NUM_THREADS, 1, 1),
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
