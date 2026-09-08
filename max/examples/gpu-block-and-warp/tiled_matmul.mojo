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

# DOC: max/gpu/block-and-warp.mdx

"""
Tiled Matrix Multiplication GPU Kernel Example.

This example demonstrates a basic tiled matrix multiplication GPU kernel
implementation in Mojo, with a specific focus on using gpu.sync.barrier()
for block-level synchronization. It serves as an educational example for
developers learning GPU programming in Mojo.

The implementation shows:
- Basic tiling techniques for improved memory bandwidth utilization
- Proper usage of gpu.sync.barrier() for thread synchronization
- TileTensor usage for matrix representation
- Shared memory optimization patterns

This example uses only open source Mojo libraries.
"""

from std.math import ceildiv
from std.sys import exit, has_accelerator

# GPU programming imports from open source stdlib and max accelerator library
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import thread_idx, block_idx

# TileTensor support from open source layout package
from layout import TileTensor, stack_allocation
from layout.tile_layout import row_major

# Data type selection: float32 provides good balance of precision and performance
comptime float_dtype = DType.float32

# Matrix dimensions: chosen to be small enough for easy understanding
# while still demonstrating tiling concepts effectively
comptime MATRIX_SIZE = 64  # 64x64 matrices
comptime MATRIX_M = MATRIX_SIZE  # Number of rows in matrices A and C
comptime MATRIX_N = MATRIX_SIZE  # Number of columns in matrices B and C
comptime MATRIX_K = MATRIX_SIZE  # Shared dimension (A cols = B rows)

# Tile dimensions: chosen to fit comfortably in GPU shared memory
# and demonstrate clear blocking behavior
comptime TILE_SIZE = 16  # 16x16 tiles balance memory usage and parallelism
comptime TILE_M = TILE_SIZE  # Tile height for matrix A and C
comptime TILE_N = TILE_SIZE  # Tile width for matrix B and C
comptime TILE_K = TILE_SIZE  # Tile depth for the K dimension

# Derived constants
comptime NUM_TILES_PER_SIDE = MATRIX_SIZE // TILE_SIZE  # Number of tiles per matrix side (4)
comptime THREADS_PER_TILE = TILE_SIZE * TILE_SIZE  # Threads needed per tile (256)
comptime TOTAL_TILES_TO_PROCESS = NUM_TILES_PER_SIDE  # Tiles to process in K dimension

# TileTensor provides type-safe multi-dimensional data access with automatic memory layout handling
# Layout definitions using example matrix dimensions
comptime matrix_a_layout = row_major[MATRIX_M, MATRIX_K]()  # A: M x K
comptime matrix_b_layout = row_major[MATRIX_K, MATRIX_N]()  # B: K x N
comptime matrix_c_layout = row_major[MATRIX_M, MATRIX_N]()  # C: M x N

# Layout definitions for tile access
comptime tile_a_layout = row_major[TILE_M, TILE_K]()
comptime tile_b_layout = row_major[TILE_K, TILE_N]()


def tiled_matmul_kernel(
    matrix_a: TileTensor[float_dtype, type_of(matrix_a_layout), MutAnyOrigin],
    matrix_b: TileTensor[float_dtype, type_of(matrix_b_layout), MutAnyOrigin],
    matrix_c: TileTensor[float_dtype, type_of(matrix_c_layout), MutAnyOrigin],
):
    # Thread and block indices
    var thread_x = thread_idx.x
    var thread_y = thread_idx.y
    var block_x = block_idx.x
    var block_y = block_idx.y

    # Global matrix coordinates
    var global_row = block_y * TILE_M + thread_y
    var global_col = block_x * TILE_N + thread_x

    # Tile starting positions
    var tile_row_start = block_y * TILE_M
    var tile_col_start = block_x * TILE_N

    # Allocate shared memory tiles for fast on-chip access
    var tile_a_shared = stack_allocation[float_dtype, address_space=.SHARED](
        tile_a_layout
    )

    var tile_b_shared = stack_allocation[float_dtype, address_space=.SHARED](
        tile_b_layout
    )

    # Initialize accumulator and start tiling loop
    var accumulator: matrix_c.ElementType = 0.0

    # Iterate through tiles along K dimension
    # Use comptime for to unroll the loop at compile time
    comptime for k_tile in range(0, MATRIX_K, TILE_K):
        # Cooperative tile loading
        # Calculate global coordinates for tile loading
        var a_global_row = tile_row_start + thread_y
        var a_global_col = k_tile + thread_x
        var b_global_row = k_tile + thread_y
        var b_global_col = tile_col_start + thread_x

        # Bounds checking
        var load_a_valid = (a_global_row < MATRIX_M) and (
            a_global_col < MATRIX_K
        )
        var load_b_valid = (b_global_row < MATRIX_K) and (
            b_global_col < MATRIX_N
        )

        # Load tiles into shared memory with bounds checking
        if load_a_valid:
            tile_a_shared[thread_y, thread_x] = matrix_a[
                a_global_row, a_global_col
            ]
        else:
            tile_a_shared[thread_y, thread_x] = 0.0

        if load_b_valid:
            tile_b_shared[thread_y, thread_x] = matrix_b[
                b_global_row, b_global_col
            ]
        else:
            tile_b_shared[thread_y, thread_x] = 0.0

        # Ensure all threads finish loading tiles before any thread starts computing
        barrier()

        # Compute dot product using shared memory tiles
        comptime for k in range(TILE_K):
            var a_element = tile_a_shared[thread_y, k]
            var b_element = tile_b_shared[k, thread_x]
            accumulator += a_element * b_element

        # Ensure all threads finish computing before any thread loads next tiles
        barrier()

    # Write final result to global memory with bounds checking
    if (global_row < MATRIX_M) and (global_col < MATRIX_N):
        matrix_c[global_row, global_col] = accumulator


def main() raises:
    # Check for GPU availability
    comptime if not has_accelerator():
        print("No GPU detected - this example requires a supported GPU")
    else:
        print("Tiled Matrix Multiplication GPU Example")
        print(
            String(
                "Matrix size: ",
                MATRIX_SIZE,
                "x",
                MATRIX_SIZE,
                ", Tile size: ",
                TILE_SIZE,
                "x",
                TILE_SIZE,
            )
        )

        # Initialize GPU device context
        var ctx = DeviceContext()

        # Allocate device buffers
        var matrix_a_buffer = ctx.enqueue_create_buffer[float_dtype](
            comptime (matrix_a_layout.size())
        )
        var matrix_b_buffer = ctx.enqueue_create_buffer[float_dtype](
            comptime (matrix_b_layout.size())
        )
        var matrix_c_buffer = ctx.enqueue_create_buffer[float_dtype](
            comptime (matrix_c_layout.size())
        )

        # Create host buffers and initialize input data
        var host_matrix_a_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            comptime (matrix_a_layout.size())
        )
        var host_matrix_b_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            comptime (matrix_b_layout.size())
        )
        ctx.synchronize()

        var host_matrix_a = TileTensor(host_matrix_a_buffer, matrix_a_layout)
        for i in range(MATRIX_M):
            for j in range(MATRIX_K):
                host_matrix_a[i, j] = Float32(i + 1)
        print("Matrix A initialized with simple test pattern (A[i,j] = i + 1)")

        var host_matrix_b = TileTensor(host_matrix_b_buffer, matrix_b_layout)
        for i in range(MATRIX_K):
            for j in range(MATRIX_N):
                host_matrix_b[i, j] = Float32(j + 1)
        print("Matrix B initialized with simple test pattern (B[i,j] = j + 1)")

        # Transfer data to device
        ctx.enqueue_copy(src_buf=host_matrix_a_buffer, dst_buf=matrix_a_buffer)
        ctx.enqueue_copy(src_buf=host_matrix_b_buffer, dst_buf=matrix_b_buffer)
        print("Input data transferred from host to device")

        # Wrap device buffers in TileTensors
        var device_matrix_a = TileTensor(matrix_a_buffer, matrix_a_layout)
        var device_matrix_b = TileTensor(matrix_b_buffer, matrix_b_layout)
        var device_matrix_c = TileTensor(matrix_c_buffer, matrix_c_layout)

        # Calculate grid and block dimensions
        comptime num_blocks_x = ceildiv(MATRIX_N, TILE_N)
        comptime num_blocks_y = ceildiv(MATRIX_M, TILE_M)
        comptime block_dim_x = TILE_N
        comptime block_dim_y = TILE_M

        print(
            String(
                "Launching kernel with grid (",
                num_blocks_x,
                ", ",
                num_blocks_y,
                ") and block (",
                block_dim_x,
                ", ",
                block_dim_y,
                ")",
            )
        )

        # Launch the kernel
        ctx.enqueue_function[tiled_matmul_kernel](
            device_matrix_a,
            device_matrix_b,
            device_matrix_c,
            grid_dim=(num_blocks_x, num_blocks_y),
            block_dim=(block_dim_x, block_dim_y),
        )

        # Retrieve results from device
        var host_matrix_c_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            comptime (matrix_c_layout.size())
        )
        ctx.enqueue_copy(src_buf=matrix_c_buffer, dst_buf=host_matrix_c_buffer)
        ctx.synchronize()

        var result_matrix = TileTensor(host_matrix_c_buffer, matrix_c_layout)

        print("Matrix multiplication completed!")
        print("Result matrix C (first few elements):")

        # Display results with expected values
        var display_size = min(4, MATRIX_M)
        for i in range(display_size):
            var row_str = String("Row ", i, ": ")
            for j in range(min(4, MATRIX_N)):
                row_str += String(result_matrix[i, j], " ")
            print(row_str)

            var expected_str = String("Expected: ")
            for j in range(min(4, MATRIX_N)):
                var expected_val = Float32((i + 1) * MATRIX_K * (j + 1))
                expected_str += String(expected_val, " ")
            print(expected_str)

        comptime if MATRIX_SIZE > 4:
            print("... (remaining elements not displayed)")

        print(
            "Note: Expected formula is C[i,j] = (i+1) * MATRIX_K * (j+1) ="
            " (i+1) * 64 * (j+1)"
        )

        # Validate GPU results against CPU reference implementation
        # This ensures our tiled GPU kernel produces correct results
        print("Validating GPU results against CPU reference...")

        # Perform simple validation by checking a few known values
        # Given A[i,j] = i+1 and B[i,j] = j+1, we expect C[i,j] = (i+1) * MATRIX_K * (j+1)
        print("Performing pattern-based validation...")

        var validation_errors = 0
        comptime TOLERANCE: Float32 = 1e-5

        # Check a few key positions manually for validation
        var test_coords_i: List[Int] = [0, 0, 1, 1, 3]
        var test_coords_j: List[Int] = [0, 1, 0, 1, 3]

        for idx in range(len(test_coords_i)):
            var i = test_coords_i[idx]
            var j = test_coords_j[idx]
            var expected = Float32((i + 1) * MATRIX_K * (j + 1))
            var actual = result_matrix[i, j].cast[.float32]()
            var diff = abs(actual - expected)

            if diff > TOLERANCE:
                validation_errors += 1
                print(
                    String(
                        "Error at [",
                        i,
                        ",",
                        j,
                        "]: expected=",
                        expected,
                        " actual=",
                        actual,
                    )
                )
            else:
                print(
                    String(
                        "✓ [",
                        i,
                        ",",
                        j,
                        "]: expected=",
                        expected,
                        " actual=",
                        actual,
                    )
                )

        if validation_errors == 0:
            print("✓ Validation PASSED: GPU results match expected pattern!")
            print("Tiled matrix multiplication example completed successfully!")
        else:
            print("✗ Validation FAILED: ", validation_errors, " errors found!")
            exit(1)
