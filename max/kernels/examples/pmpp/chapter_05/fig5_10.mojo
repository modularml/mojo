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

# Figure 5.10: Tiled matrix multiplication using shared memory (from PMPP Chapter 5)
# Demonstrates how to use shared memory to optimize matrix multiplication

from std.math import ceildiv
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

# ========================== KERNEL CODE ==========================


def matrix_mul_tiled_kernel(
    M: UnsafePointer[Float32, MutUntrackedOrigin],
    N: UnsafePointer[Float32, MutUntrackedOrigin],
    P: UnsafePointer[Float32, MutUntrackedOrigin],
    width_dev: Int32,
):
    """Tiled matrix multiplication kernel using shared memory.

    Args:
        M: Input matrix M (device).
        N: Input matrix N (device).
        P: Output matrix P = M * N (device).
        width_dev: Matrix dimension (Width x Width matrices).
    """
    # Int is not device-passable; widen the fixed-width arg.
    var Width = Int(width_dev)
    comptime TILE_WIDTH = 16

    # Allocate shared memory tiles
    var sA = unsafe_stack_allocation[
        TILE_WIDTH * TILE_WIDTH,
        Float32,
        address_space=.SHARED,
    ]()
    var sB = unsafe_stack_allocation[
        TILE_WIDTH * TILE_WIDTH,
        Float32,
        address_space=.SHARED,
    ]()

    # Get global and shared indices
    var g_row = block_idx.y * TILE_WIDTH + thread_idx.y
    var g_col = block_idx.x * TILE_WIDTH + thread_idx.x
    var s_row = thread_idx.y
    var s_col = thread_idx.x

    var tmp = Float32(0)

    # Go through tiles
    for ph in range(Width // TILE_WIDTH):
        # Load tiles into shared memory
        sA[s_row * TILE_WIDTH + s_col] = Float32(
            M[g_row * Width + (ph * TILE_WIDTH + s_col)]
        )
        sB[s_row * TILE_WIDTH + s_col] = Float32(
            N[(s_row + ph * TILE_WIDTH) * Width + g_col]
        )
        barrier()

        # Compute matrix multiplication for this tile
        for k in range(TILE_WIDTH):
            tmp += Float32(sA[s_row * TILE_WIDTH + k]) * Float32(
                sB[k * TILE_WIDTH + s_col]
            )
        barrier()

    # Write result to global memory
    P[g_row * Width + g_col] = tmp


def matmul_tiled(
    h_a: UnsafePointer[Float32, MutUntrackedOrigin],
    h_b: UnsafePointer[Float32, MutUntrackedOrigin],
    h_c: UnsafePointer[Float32, MutUntrackedOrigin],
    width: Int,
    ctx: DeviceContext,
) raises:
    """Host function for tiled matrix multiplication.

    Args:
        h_a: Input matrix A (host).
        h_b: Input matrix B (host).
        h_c: Output matrix C = A * B (host).
        width: Matrix dimension.
        ctx: Device context for GPU operations.
    """
    comptime TILE_WIDTH = 16
    comptime dtype = DType.float32

    var size = width * width

    # Allocate device memory
    var d_a = ctx.enqueue_create_buffer[dtype](size)
    var d_b = ctx.enqueue_create_buffer[dtype](size)
    var d_c = ctx.enqueue_create_buffer[dtype](size)

    # Copy data to device
    ctx.enqueue_copy(d_a, h_a)
    ctx.enqueue_copy(d_b, h_b)

    # Configure kernel launch
    var block_dim_x = TILE_WIDTH
    var block_dim_y = TILE_WIDTH
    var grid_dim_x = ceildiv(width, TILE_WIDTH)
    var grid_dim_y = ceildiv(width, TILE_WIDTH)

    # Launch kernel
    ctx.enqueue_function[matrix_mul_tiled_kernel](
        d_a,
        d_b,
        d_c,
        Int32(width),
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(block_dim_x, block_dim_y, 1),
    )

    # Copy result back to host
    ctx.enqueue_copy(h_c, d_c)
    ctx.synchronize()


# ========================== TEST CODE ==========================


def matmul_cpu(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    c: UnsafePointer[Float32, MutUntrackedOrigin],
    width: Int,
):
    """CPU reference implementation for matrix multiplication.

    Args:
        a: Input matrix A.
        b: Input matrix B.
        c: Output matrix C = A * B.
        width: Matrix dimension.
    """
    for row in range(width):
        for col in range(width):
            var out_idx = row * width + col
            c[out_idx] = 0
            for k in range(width):
                c[out_idx] += a[row * width + k] * b[k * width + col]


def initialize(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    width: Int,
):
    """Initialize input matrices with test data.

    Args:
        a: Input matrix A to initialize.
        b: Input matrix B to initialize.
        width: Matrix dimension.
    """
    for i in range(width * width):
        a[i] = Float32(i)
        b[i] = Float32(i)


def main() raises:
    var width = 256
    var size = width * width

    # Allocate host memory
    var h_a = alloc[Float32](size)
    var h_b = alloc[Float32](size)
    var h_c = alloc[Float32](size)
    var h_ref = alloc[Float32](size)

    # Initialize input matrices
    initialize(h_a, h_b, width)

    # Run GPU matrix multiplication
    with DeviceContext() as ctx:
        matmul_tiled(h_a, h_b, h_c, width, ctx)

    # Run CPU reference
    matmul_cpu(h_a, h_b, h_ref, width)

    # Compare results
    var errors = 0
    for i in range(size):
        var diff = abs(h_ref[i] - h_c[i])
        var rel_error = diff / (abs(h_ref[i]) + 1e-7)
        if rel_error > 1e-4:
            if errors < 10:
                print(
                    "Error at index",
                    i,
                    ", host val",
                    h_ref[i],
                    "and gpu val",
                    h_c[i],
                    "with diff",
                    diff,
                )
            errors += 1

    if errors > 0:
        print("Total errors:", errors)
    else:
        print("Success!")

    # Cleanup
    h_a.free()
    h_b.free()
    h_c.free()
    h_ref.free()
