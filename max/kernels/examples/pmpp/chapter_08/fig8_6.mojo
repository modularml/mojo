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

# Figure 8.6: 3D Stencil Kernel (from PMPP Chapter 8)
# Demonstrates basic 3D stencil computation (7-point stencil)

from std.random import random_float64
from std.math import ceildiv
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.itertools import product

# ========================== KERNEL CODE ==========================


def stencil_kernel(
    in_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_dim: Int32,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    c4: Float32,
    c5: Float32,
    c6: Float32,
):
    """3D stencil kernel computing 7-point stencil (center + 6 neighbors).

    Args:
        in_ptr: Input 3D array (device).
        out_ptr: Output 3D array (device).
        n_dim: Dimension size (N x N x N volume).
        c0: Coefficient for center element.
        c1: Coefficient for k-1 neighbor.
        c2: Coefficient for k+1 neighbor.
        c3: Coefficient for j-1 neighbor.
        c4: Coefficient for j+1 neighbor.
        c5: Coefficient for i-1 neighbor.
        c6: Coefficient for i+1 neighbor.
    """
    comptime BLOCK_DIM = 8
    # `Int` is not device-passable; widen the fixed-width arg back for indexing.
    var N = Int(n_dim)
    var i = block_idx.z * BLOCK_DIM + thread_idx.z
    var j = block_idx.y * BLOCK_DIM + thread_idx.y
    var k = block_idx.x * BLOCK_DIM + thread_idx.x

    if i >= N or j >= N or k >= N:
        return

    if 1 <= i < N - 1 and 1 <= j < N - 1 and 1 <= k < N - 1:
        var idx = i * N * N + j * N + k
        out_ptr[idx] = (
            c0 * in_ptr[idx]  # center
            + c1 * in_ptr[i * N * N + j * N + (k - 1)]  # k-1
            + c2 * in_ptr[i * N * N + j * N + (k + 1)]  # k+1
            + c3 * in_ptr[i * N * N + (j - 1) * N + k]  # j-1
            + c4 * in_ptr[i * N * N + (j + 1) * N + k]  # j+1
            + c5 * in_ptr[(i - 1) * N * N + j * N + k]  # i-1
            + c6 * in_ptr[(i + 1) * N * N + j * N + k]  # i+1
        )
    else:
        # Boundary elements - set to zero
        out_ptr[i * N * N + j * N + k] = 0.0


def stencil_3d(
    h_in: UnsafePointer[mut=False, Float32, _],
    h_out: UnsafePointer[mut=True, Float32, _],
    N: Int,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    c4: Float32,
    c5: Float32,
    c6: Float32,
    ctx: DeviceContext,
) raises:
    """Host function for 3D stencil computation.

    Args:
        h_in: Input array (host).
        h_out: Output array (host).
        N: Dimension size.
        c0: Coefficient for center element.
        c1: Coefficient for k-1 neighbor.
        c2: Coefficient for k+1 neighbor.
        c3: Coefficient for j-1 neighbor.
        c4: Coefficient for j+1 neighbor.
        c5: Coefficient for i-1 neighbor.
        c6: Coefficient for i+1 neighbor.
        ctx: Device context for GPU operations.
    """
    comptime dtype = DType.float32

    var total_elements = N * N * N

    # Allocate device memory
    var d_in = ctx.enqueue_create_buffer[dtype](total_elements)
    var d_out = ctx.enqueue_create_buffer[dtype](total_elements)

    # Copy data to device
    ctx.enqueue_copy(d_in, h_in)

    # Configure kernel launch with 3D grid
    comptime BLOCK_DIM = 8
    var grid_dim_x = ceildiv(N, BLOCK_DIM)
    var grid_dim_y = ceildiv(N, BLOCK_DIM)
    var grid_dim_z = ceildiv(N, BLOCK_DIM)

    # Launch kernel
    ctx.enqueue_function[stencil_kernel](
        d_in,
        d_out,
        Int32(N),
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM),
    )

    # Copy result back to host
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()


# ========================== TEST CODE ==========================


def cpu_3d_stencil(
    in_ptr: UnsafePointer[mut=False, Float32, _],
    out_ptr: UnsafePointer[mut=True, Float32, _],
    N: Int,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    c4: Float32,
    c5: Float32,
    c6: Float32,
):
    """CPU reference implementation for 3D stencil.

    Args:
        in_ptr: Input array.
        out_ptr: Output array.
        N: Dimension size.
        c0: Coefficient for center element.
        c1: Coefficient for k-1 neighbor.
        c2: Coefficient for k+1 neighbor.
        c3: Coefficient for j-1 neighbor.
        c4: Coefficient for j+1 neighbor.
        c5: Coefficient for i-1 neighbor.
        c6: Coefficient for i+1 neighbor.
    """
    for i, j, k in product(range(N), range(N), range(N)):
        var idx = i * N * N + j * N + k
        if 1 <= i < N - 1 and 1 <= j < N - 1 and 1 <= k < N - 1:
            out_ptr[idx] = (
                c0 * in_ptr[idx]
                + c1 * in_ptr[i * N * N + j * N + (k - 1)]
                + c2 * in_ptr[i * N * N + j * N + (k + 1)]
                + c3 * in_ptr[i * N * N + (j - 1) * N + k]
                + c4 * in_ptr[i * N * N + (j + 1) * N + k]
                + c5 * in_ptr[(i - 1) * N * N + j * N + k]
                + c6 * in_ptr[(i + 1) * N * N + j * N + k]
            )
        else:
            out_ptr[idx] = 0.0


def main() raises:
    var N = 64  # 64x64x64 volume
    var total_elements = N * N * N

    # Define stencil coefficients
    # c0 = center, c1 = k-1, c2 = k+1, c3 = j-1, c4 = j+1, c5 = i-1, c6 = i+1
    var c0 = Float32(1.0)
    var c1 = Float32(1.0)
    var c2 = Float32(1.0)
    var c3 = Float32(1.0)
    var c4 = Float32(1.0)
    var c5 = Float32(1.0)
    var c6 = Float32(1.0)

    # Allocate host memory
    var h_in = alloc[Float32](total_elements)
    var h_out = alloc[Float32](total_elements)
    var h_ref = alloc[Float32](total_elements)

    # Initialize input with random values
    for i in range(total_elements):
        h_in[i] = random_float64().cast[.float32]()

    # Initialize output to zero
    for i in range(total_elements):
        h_out[i] = 0.0

    # Run GPU stencil
    print("Launching 3D stencil kernel on", N, "x", N, "x", N, "volume")
    with DeviceContext() as ctx:
        stencil_3d(h_in, h_out, N, c0, c1, c2, c3, c4, c5, c6, ctx)

    # Run CPU reference
    cpu_3d_stencil(h_in, h_ref, N, c0, c1, c2, c3, c4, c5, c6)

    # Verify results
    var errors = 0
    for i, j, k in product(range(N), range(N), range(N)):
        var idx = i * N * N + j * N + k
        var diff = abs(h_ref[idx] - h_out[idx])
        var rel = diff / (abs(h_ref[idx]) + 1e-7)
        if rel > 1e-5:
            if errors < 10:
                print(
                    "Error at (",
                    i,
                    ",",
                    j,
                    ",",
                    k,
                    "): CPU=",
                    h_ref[idx],
                    ", GPU=",
                    h_out[idx],
                    ", diff=",
                    diff,
                )
            errors += 1

    if errors > 0:
        print("FAILED: Total errors:", errors)
    else:
        print("SUCCESS: All values match!")

    # Cleanup
    h_in.free()
    h_out.free()
    h_ref.free()
