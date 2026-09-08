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

from std.random import random_float64
from std.itertools import product

from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

# ========================== TILING CONFIGURATION ==========================
comptime STENCIL_WIDTH = 7
comptime STENCIL_ORDER = 3
comptime OUT_TILE_DIM = 4
comptime IN_TILE_DIM = OUT_TILE_DIM + ((STENCIL_WIDTH - 1) // STENCIL_ORDER)


# ========================== KERNEL CODE ==========================
def stencil_kernel(
    d_in: UnsafePointer[Float32, ImmutAnyOrigin],
    d_out: UnsafePointer[Float32, MutAnyOrigin],
    d_c: UnsafePointer[Float32, ImmutAnyOrigin],  # Stencil coefficients
    n_dev: Int32,
):
    """3D stencil kernel with shared memory tiling and halo cells.

    Args:
        d_in: Input 3D array (flattened).
        d_out: Output 3D array (flattened).
        d_c: Stencil coefficients [7] (center, k-1, k+1, j-1, j+1, i-1, i+1).
        n_dev: Dimension size (N x N x N).
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var N = Int(n_dev)
    # Allocate shared memory tile (includes halo cells)
    var in_s = unsafe_stack_allocation[
        IN_TILE_DIM * IN_TILE_DIM * IN_TILE_DIM,
        Float32,
        address_space=.SHARED,
    ]()

    # Get thread and block indices
    var tx = thread_idx.x
    var ty = thread_idx.y
    var tz = thread_idx.z

    # Calculate global indices with -1 offset to load halo cells
    var i = block_idx.z * OUT_TILE_DIM + tz - 1
    var j = block_idx.y * OUT_TILE_DIM + ty - 1
    var k = block_idx.x * OUT_TILE_DIM + tx - 1

    # Load data into shared memory (all threads participate)
    if i >= 0 and i < N and j >= 0 and j < N and k >= 0 and k < N:
        var global_idx = i * N * N + j * N + k
        var shared_idx = tz * IN_TILE_DIM * IN_TILE_DIM + ty * IN_TILE_DIM + tx
        in_s[shared_idx] = Float32(d_in[global_idx])

    # Wait for all threads to finish loading
    barrier()

    # Only interior threads compute output (to avoid accessing invalid neighbors)
    if 1 <= i < N - 1 and 1 <= j < N - 1 and 1 <= k < N - 1:
        if (
            tz >= 1
            and tz <= OUT_TILE_DIM
            and ty >= 1
            and ty <= OUT_TILE_DIM
            and tx >= 1
            and tx <= OUT_TILE_DIM
        ):
            # Calculate shared memory indices for stencil access
            var center_idx = (
                tz * IN_TILE_DIM * IN_TILE_DIM + ty * IN_TILE_DIM + tx
            )
            var k_minus = center_idx - 1
            var k_plus = center_idx + 1
            var j_minus = center_idx - IN_TILE_DIM
            var j_plus = center_idx + IN_TILE_DIM
            var i_minus = center_idx - IN_TILE_DIM * IN_TILE_DIM
            var i_plus = center_idx + IN_TILE_DIM * IN_TILE_DIM

            # Compute 7-point stencil using shared memory
            var result = Float32(0)
            result += d_c[0] * Float32(in_s[center_idx])  # center
            result += d_c[1] * Float32(in_s[k_minus])  # k-1
            result += d_c[2] * Float32(in_s[k_plus])  # k+1
            result += d_c[3] * Float32(in_s[j_minus])  # j-1
            result += d_c[4] * Float32(in_s[j_plus])  # j+1
            result += d_c[5] * Float32(in_s[i_minus])  # i-1
            result += d_c[6] * Float32(in_s[i_plus])  # i+1

            # Write result to global memory
            var output_idx = i * N * N + j * N + k
            d_out[output_idx] = result


# ========================== TEST CODE ==========================
def cpu_3d_stencil(
    h_in: UnsafePointer[mut=False, Float32, _],
    h_out: UnsafePointer[mut=True, Float32, _],
    N: Int,
    coeffs: UnsafePointer[mut=False, Float32, _],
):
    """CPU reference implementation of 3D stencil."""
    for i, j, k in product(range(N), range(N), range(N)):
        var idx = i * N * N + j * N + k
        if 1 <= i < N - 1 and 1 <= j < N - 1 and 1 <= k < N - 1:
            h_out[idx] = (
                coeffs[0] * h_in[idx]
                + coeffs[1] * h_in[i * N * N + j * N + (k - 1)]
                + coeffs[2] * h_in[i * N * N + j * N + (k + 1)]
                + coeffs[3] * h_in[i * N * N + (j - 1) * N + k]
                + coeffs[4] * h_in[i * N * N + (j + 1) * N + k]
                + coeffs[5] * h_in[(i - 1) * N * N + j * N + k]
                + coeffs[6] * h_in[(i + 1) * N * N + j * N + k]
            )
        else:
            h_out[idx] = Float32(0.0)


def main() raises:
    comptime dtype = DType.float32
    var N = 64  # Must be divisible by OUT_TILE_DIM
    var total_elements = N * N * N

    print("3D Tiled Stencil Kernel with Shared Memory")
    print("=" * 50)
    print("Grid size:", N, "x", N, "x", N)
    print("OUT_TILE_DIM:", OUT_TILE_DIM)
    print("IN_TILE_DIM:", IN_TILE_DIM, "(includes halo cells)")
    print()

    # Define stencil coefficients
    # c[0] = center, c[1] = k-1, c[2] = k+1, c[3] = j-1, c[4] = j+1, c[5] = i-1, c[6] = i+1
    var h_c = alloc[Float32](7)
    h_c[0] = 1.0  # center
    h_c[1] = 1.0  # k-1
    h_c[2] = 1.0  # k+1
    h_c[3] = 1.0  # j-1
    h_c[4] = 1.0  # j+1
    h_c[5] = 1.0  # i-1
    h_c[6] = 1.0  # i+1

    # Host memory allocation
    var h_in = alloc[Float32](total_elements)
    var h_out = alloc[Float32](total_elements)
    var h_ref = alloc[Float32](total_elements)

    # Initialize input with random values
    print("Initializing input data...")
    for i in range(total_elements):
        h_in[i] = Float32(random_float64(0, 1.0))

    # Initialize output to zero
    for i in range(total_elements):
        h_out[i] = Float32(0.0)
        h_ref[i] = Float32(0.0)

    # Create GPU context
    var ctx = DeviceContext()

    # Allocate device memory
    var d_in = ctx.enqueue_create_buffer[dtype](total_elements)
    var d_out = ctx.enqueue_create_buffer[dtype](total_elements)
    var d_c = ctx.enqueue_create_buffer[dtype](7)

    # Copy data to device
    ctx.enqueue_copy(d_in, h_in)
    ctx.enqueue_copy(d_out, h_out)
    ctx.enqueue_copy(d_c, h_c)

    # Configure kernel launch
    var grid_dim_x = N // OUT_TILE_DIM
    var grid_dim_y = N // OUT_TILE_DIM
    var grid_dim_z = N // OUT_TILE_DIM
    var block_dim_x = IN_TILE_DIM
    var block_dim_y = IN_TILE_DIM
    var block_dim_z = IN_TILE_DIM

    print("Launching GPU kernel...")
    print(
        "Grid:",
        "(",
        grid_dim_x,
        ",",
        grid_dim_y,
        ",",
        grid_dim_z,
        ")",
    )
    print(
        "Block:",
        "(",
        block_dim_x,
        ",",
        block_dim_y,
        ",",
        block_dim_z,
        ")",
    )
    print()

    # Launch kernel
    ctx.enqueue_function[stencil_kernel](
        d_in,
        d_out,
        d_c,
        Int32(N),
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_dim_x, block_dim_y, block_dim_z),
    )
    ctx.synchronize()

    # Copy result back
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()

    # Run CPU reference
    print("Running CPU reference...")
    cpu_3d_stencil(h_in, h_ref, N, h_c)
    print()

    # Verify results
    print("Verifying results...")
    var errors = 0
    var max_errors = 10
    for i, j, k in product(range(N), range(N), range(N)):
        if errors >= max_errors:
            break
        var idx = i * N * N + j * N + k
        var diff = abs(h_ref[idx] - h_out[idx])
        var rel = diff / (abs(h_ref[idx]) + 1e-7)
        if rel > 1e-5:
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

    if errors == 0:
        print("✓ SUCCESS: All values match!")
    else:
        print("✗ FAILED: Found", errors, "errors")

    # Cleanup
    h_in.free()
    h_out.free()
    h_ref.free()
    h_c.free()
