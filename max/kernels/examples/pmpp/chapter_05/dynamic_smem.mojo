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

# Dynamic shared memory matrix multiplication kernel
# Translates the CUDA dynamic_smem.cu example to Mojo
# Uses parameterized tile width with shared memory

from std.math import ceildiv, sqrt
from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import external_memory

# ========================== KERNEL CODE ==========================


def matrixMulKernel(
    M: UnsafePointer[Float32, MutUntrackedOrigin],
    N: UnsafePointer[Float32, MutUntrackedOrigin],
    P: UnsafePointer[Float32, MutUntrackedOrigin],
    width_dim: Int32,
    tile_width_dim: Int32,
):
    """Matrix multiplication kernel with parameterized tile width.

    Args:
        M: Input matrix M (device).
        N: Input matrix N (device).
        P: Output matrix P = M * N (device).
        width_dim: Matrix dimension (Width x Width matrices).
        tile_width_dim: Tile width for this execution.
    """
    # `Int` is not device-passable; widen the fixed-width args for indexing.
    var Width = Int(width_dim)
    var tile_width = Int(tile_width_dim)
    # Allocate shared memory using external_memory (following working pattern)
    # Use max tile size of 32x32 for allocation
    var Mds = rebind[
        UnsafePointer[Float32, MutUntrackedOrigin, address_space=.SHARED]
    ](
        external_memory[
            Float32,
            address_space=.SHARED,
            alignment=16,
            name="shared_dynamic_memory",
        ]()
    )
    var Nds = Mds + 32 * 32  # Use max size for allocation

    var tx = thread_idx.x
    var ty = thread_idx.y
    var Row = tile_width * block_idx.y + ty
    var Col = tile_width * block_idx.x + tx

    var Pvalue = Float32(0)

    for ph in range(ceildiv(Width, tile_width)):
        # Collaborative loading of M and N tiles into shared memory
        if (Row < Width) and (ph * tile_width + tx) < Width:
            Mds[ty * tile_width + tx] = Float32(
                M[Row * Width + ph * tile_width + tx]
            )
        else:
            Mds[ty * tile_width + tx] = Float32(0.0)

        if (ph * tile_width + ty) < Width and Col < Width:
            Nds[ty * tile_width + tx] = Float32(
                N[(ph * tile_width + ty) * Width + Col]
            )
        else:
            Nds[ty * tile_width + tx] = Float32(0.0)

        barrier()

        for k in range(tile_width):
            Pvalue += Float32(Mds[ty * tile_width + k]) * Float32(
                Nds[k * tile_width + tx]
            )

        barrier()

    if (Row < Width) and (Col < Width):
        P[Row * Width + Col] = Pvalue


# ========================== Test CODE ==========================


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
    # Simulate device properties
    # Using reasonable values for a modern GPU
    var max_threads = 1024
    var max_smem = 49152  # bytes

    var thread_dim = Int(
        sqrt(Float64(max_threads // 2))
    )  # threads across 2 tiles
    var smem_per_tile = (
        max_smem // 4 // 2
    )  # sizeof(float) = 4, divide by 2 for two tiles
    var smem_dim = Int(sqrt(Float64(smem_per_tile)))

    var tile_width = min(smem_dim, thread_dim)

    # not divisible by tile width
    var width = 280
    var total_bytes = width * width
    comptime dtype = DType.float32

    # declare and allocate buff mem
    var h_a = alloc[Float32](total_bytes)
    var h_b = alloc[Float32](total_bytes)
    var h_c = alloc[Float32](total_bytes)
    var h_ref = alloc[Float32](total_bytes)

    # initialize and copy bufs
    initialize(h_a, h_b, width)
    for i in range(total_bytes):
        h_c[i] = 0
        h_ref[i] = 0

    # Run GPU matrix multiplication
    with DeviceContext() as ctx:
        var size = width * width

        # Allocate device memory
        var d_a = ctx.enqueue_create_buffer[dtype](size)
        var d_b = ctx.enqueue_create_buffer[dtype](size)
        var d_c = ctx.enqueue_create_buffer[dtype](size)

        # Copy data to device
        ctx.enqueue_copy(d_a, h_a)
        ctx.enqueue_copy(d_b, h_b)
        ctx.enqueue_copy(d_c, h_c)

        # Configure kernel launch
        var block_dim_x = tile_width
        var block_dim_y = tile_width
        var grid_dim_x = ceildiv(width, tile_width)
        var grid_dim_y = ceildiv(width, tile_width)

        # Calculate shared memory size needed (2 tiles of 32x32 Float32s for max size)
        var shared_mem_bytes = (
            2 * 32 * 32 * 4
        )  # 4 bytes per Float32, use max size

        # Launch kernel with shared memory configuration
        # Note: func_attribute is now automatically inferred from shared_mem_bytes
        ctx.enqueue_function[matrixMulKernel](
            d_a,
            d_b,
            d_c,
            Int32(width),
            Int32(tile_width),
            grid_dim=(grid_dim_x, grid_dim_y, 1),
            block_dim=(block_dim_x, block_dim_y, 1),
            shared_mem_bytes=shared_mem_bytes,
        )

        # Copy result back to host
        ctx.enqueue_copy(h_c, d_c)
        ctx.synchronize()

    # call cpu ref
    cpu_matmul(h_a, h_b, h_ref, width)

    # compare error
    for i in range(width * width):
        var diff = abs(h_c[i] - h_ref[i])
        var rel = diff / (h_ref[i] + 1e-8)
        if rel > 1e-3:
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

    # free everything
    h_a.free()
    h_b.free()
    h_c.free()
    h_ref.free()

    print("Success")
