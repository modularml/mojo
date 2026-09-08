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

# Matrix multiplication kernel (Fig 3.11 from PMPP)
# Basic matrix multiplication implementation

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.itertools import product

# ========================== KERNEL CODE ==========================


def matrix_mul_kernel(
    M: UnsafePointer[Float32, MutUntrackedOrigin],
    N: UnsafePointer[Float32, MutUntrackedOrigin],
    P: UnsafePointer[Float32, MutUntrackedOrigin],
    width_dev: Int32,
):
    """GPU kernel for matrix multiplication.

    Args:
        M: Input matrix M (device).
        N: Input matrix N (device).
        P: Output matrix P = M * N (device).
        width_dev: Matrix dimension (Width x Width matrices).
    """
    # Int is not device-passable; widen the fixed-width arg.
    var Width = Int(width_dev)
    var row = global_idx.y
    var col = global_idx.x

    if row < Width and col < Width:
        var Pvalue = Float32(0)
        for k in range(Width):
            Pvalue += M[row * Width + k] * N[k * Width + col]
        P[row * Width + col] = Pvalue


# ========================== TEST CODE ==========================


def matmul_cpu(
    A_host: UnsafePointer[mut=False, Float32, _],
    B_host: UnsafePointer[mut=False, Float32, _],
    C_ref: UnsafePointer[mut=True, Float32, _],
    Width: Int,
):
    """CPU reference implementation for matrix multiplication.

    Args:
        A_host: Input matrix A.
        B_host: Input matrix B.
        C_ref: Output matrix C = A * B.
        Width: Matrix dimension (Width x Width matrices).
    """
    for i, j in product(range(Width), range(Width)):
        var out_idx = i * Width + j
        var tmp = Float32(0)
        for k in range(Width):
            tmp += A_host[i * Width + k] * B_host[k * Width + j]
        C_ref[out_idx] = tmp


def initialize_host(
    A_host: UnsafePointer[Float32, MutUntrackedOrigin],
    B_host: UnsafePointer[Float32, MutUntrackedOrigin],
    Width: Int,
):
    """Initialize input matrices with test data.

    Args:
        A_host: Input matrix A to initialize.
        B_host: Input matrix B to initialize.
        Width: Matrix dimension.
    """
    for i, j in product(range(Width), range(Width)):
        var idx = i * Width + j
        A_host[idx] = Float32(idx)
        B_host[idx] = Float32(idx)


def main() raises:
    # Matrix dimensions
    var Width = 512
    var num_elements = Width * Width

    # Allocate host memory
    var A_h = alloc[Float32](num_elements)
    var B_h = alloc[Float32](num_elements)
    var C_h = alloc[Float32](num_elements)
    var C_ref = alloc[Float32](num_elements)

    # Initialize host arrays
    initialize_host(A_h, B_h, Width)

    # Run GPU matrix multiplication
    with DeviceContext() as ctx:
        comptime dtype = DType.float32

        # Allocate device memory
        var A_d = ctx.enqueue_create_buffer[dtype](num_elements)
        var B_d = ctx.enqueue_create_buffer[dtype](num_elements)
        var C_d = ctx.enqueue_create_buffer[dtype](num_elements)

        # Copy data from host to device
        ctx.enqueue_copy(A_d, A_h)
        ctx.enqueue_copy(B_d, B_h)

        # Configure kernel launch parameters
        var block_dim_x = 16
        var block_dim_y = 16
        var grid_dim_x = ceildiv(Width, block_dim_x)
        var grid_dim_y = ceildiv(Width, block_dim_y)

        # Launch kernel
        ctx.enqueue_function[matrix_mul_kernel](
            A_d,
            B_d,
            C_d,
            Int32(Width),
            grid_dim=(grid_dim_x, grid_dim_y, 1),
            block_dim=(block_dim_x, block_dim_y, 1),
        )

        # Copy result from device to host
        ctx.enqueue_copy(C_h, C_d)

        # Synchronize to ensure completion
        ctx.synchronize()

    # Run CPU matrix multiplication
    matmul_cpu(A_h, B_h, C_ref.as_unsafe_any_origin(), Width)

    # Compare results (allow small floating point error)
    var errors = 0
    for i in range(num_elements):
        var diff = abs(C_h[i] - C_ref[i])
        var tolerance = abs(C_ref[i]) * 1e-5
        if diff > tolerance:
            if errors < 10:
                print(
                    "Error at index",
                    i,
                    ":",
                    C_h[i],
                    "!=",
                    C_ref[i],
                    "(diff:",
                    diff,
                    ", tolerance:",
                    tolerance,
                    ")",
                )
            errors += 1

    if errors > 0:
        print("Total errors:", errors)
    else:
        print("Success")

    # Cleanup
    A_h.free()
    B_h.free()
    C_h.free()
    C_ref.free()
