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

# Figure 5.14: Tiled matrix multiplication with boundary checking (from PMPP Chapter 5)
# Handles matrix dimensions that are not divisible by tile width

from std.math import ceildiv
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

# ========================== KERNEL CODE ==========================


def matrix_mul_tiled_boundary_kernel(
    M: UnsafePointer[Float32, MutUntrackedOrigin],
    N: UnsafePointer[Float32, MutUntrackedOrigin],
    P: UnsafePointer[Float32, MutUntrackedOrigin],
    width_dev: Int32,
):
    """Tiled matrix multiplication kernel with boundary checking.

    Handles matrices where dimensions are not divisible by tile width.

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
    var Mds = unsafe_stack_allocation[
        TILE_WIDTH * TILE_WIDTH,
        Float32,
        address_space=.SHARED,
    ]()
    var Nds = unsafe_stack_allocation[
        TILE_WIDTH * TILE_WIDTH,
        Float32,
        address_space=.SHARED,
    ]()

    var tx = thread_idx.x
    var ty = thread_idx.y
    var Row = TILE_WIDTH * block_idx.y + ty
    var Col = TILE_WIDTH * block_idx.x + tx

    var Pvalue = Float32(0)

    # Calculate number of phases
    var num_phases = ceildiv(Width, TILE_WIDTH)

    for ph in range(num_phases):
        # Collaborative loading of M and N tiles into shared memory with boundary checks
        if Row < Width and (ph * TILE_WIDTH + tx) < Width:
            Mds[ty * TILE_WIDTH + tx] = Float32(
                M[Row * Width + ph * TILE_WIDTH + tx]
            )
        else:
            Mds[ty * TILE_WIDTH + tx] = Float32(0.0)

        if (ph * TILE_WIDTH + ty) < Width and Col < Width:
            Nds[ty * TILE_WIDTH + tx] = Float32(
                N[(ph * TILE_WIDTH + ty) * Width + Col]
            )
        else:
            Nds[ty * TILE_WIDTH + tx] = Float32(0.0)

        barrier()

        # Compute partial dot product
        for k in range(TILE_WIDTH):
            Pvalue += Float32(Mds[ty * TILE_WIDTH + k]) * Float32(
                Nds[k * TILE_WIDTH + tx]
            )

        barrier()

    # Write result to global memory with boundary check
    if Row < Width and Col < Width:
        P[Row * Width + Col] = Pvalue


def matmul_tiled_boundary(
    h_a: UnsafePointer[Float32, MutUntrackedOrigin],
    h_b: UnsafePointer[Float32, MutUntrackedOrigin],
    h_c: UnsafePointer[Float32, MutUntrackedOrigin],
    width: Int,
    ctx: DeviceContext,
) raises:
    """Host function for tiled matrix multiplication with boundary checking.

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

    # Initialize d_c to zero
    var zeros = alloc[Float32](size)
    for i in range(size):
        zeros[i] = 0.0

    # Copy data to device
    ctx.enqueue_copy(d_a, h_a)
    ctx.enqueue_copy(d_b, h_b)
    ctx.enqueue_copy(d_c, zeros)

    # Configure kernel launch
    var block_dim_x = TILE_WIDTH
    var block_dim_y = TILE_WIDTH
    var grid_dim_x = ceildiv(width, TILE_WIDTH)
    var grid_dim_y = ceildiv(width, TILE_WIDTH)

    # Launch kernel
    ctx.enqueue_function[matrix_mul_tiled_boundary_kernel](
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

    # Cleanup
    zeros.free()


# ========================== TEST CODE ==========================


def cpu_matmul(
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
            c[row * width + col] = 0
            for k in range(width):
                c[row * width + col] += a[row * width + k] * b[k * width + col]


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
    # Use width not divisible by TILE_WIDTH (16) to test boundary handling
    var width = 280
    var size = width * width

    # Allocate host memory
    var h_a = alloc[Float32](size)
    var h_b = alloc[Float32](size)
    var h_c = alloc[Float32](size)
    var h_ref = alloc[Float32](size)

    # Initialize buffers
    initialize(h_a, h_b, width)
    for i in range(size):
        h_c[i] = 0
        h_ref[i] = 0

    # Run GPU matrix multiplication
    with DeviceContext() as ctx:
        matmul_tiled_boundary(h_a, h_b, h_c, width, ctx)

    # Run CPU reference
    cpu_matmul(h_a, h_b, h_ref, width)

    # Compare results
    var errors = 0
    for i in range(size):
        var diff = abs(h_c[i] - h_ref[i])
        var rel = diff / (abs(h_ref[i]) + 1e-8)
        if rel > 1e-3:
            if errors < 10:
                print(
                    "error at index",
                    i,
                    ", reference is",
                    h_ref[i],
                    ", kernel is",
                    h_c[i],
                    ", diff is",
                    diff,
                )
            errors += 1

    if errors > 0:
        print("Total errors:", errors)
    else:
        print("Success")

    # Cleanup
    h_a.free()
    h_b.free()
    h_c.free()
    h_ref.free()
