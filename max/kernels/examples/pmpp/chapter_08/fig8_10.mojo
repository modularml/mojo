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


# ========================== KERNEL CODE (THREAD COARSENING) ==========================
def stencil_kernel(
    d_in: UnsafePointer[Float32, ImmutAnyOrigin],
    d_out: UnsafePointer[Float32, MutAnyOrigin],
    d_c: UnsafePointer[Float32, ImmutAnyOrigin],  # Stencil coefficients
    n_dev: Int32,
):
    """3D stencil kernel with thread coarsening in z-direction.

    Each thread block uses only 2D threads (x-y plane) and iterates through
    the z-direction. Only 3 z-planes are kept in shared memory at a time,
    reducing memory requirements and enabling larger tiles.

    Args:
        d_in: Input 3D array (flattened).
        d_out: Output 3D array (flattened).
        d_c: Stencil coefficients [7] (center, k-1, k+1, j-1, j+1, i-1, i+1).
        n_dev: Dimension size (N x N x N).
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var N = Int(n_dev)
    # Allocate shared memory for three 2D planes
    var prev_in_s = unsafe_stack_allocation[
        IN_TILE_DIM * IN_TILE_DIM,
        Float32,
        address_space=.SHARED,
    ]()
    var curr_in_s = unsafe_stack_allocation[
        IN_TILE_DIM * IN_TILE_DIM,
        Float32,
        address_space=.SHARED,
    ]()
    var next_in_s = unsafe_stack_allocation[
        IN_TILE_DIM * IN_TILE_DIM,
        Float32,
        address_space=.SHARED,
    ]()

    # Get thread and block indices (only x and y, no z)
    var tx = thread_idx.x
    var ty = thread_idx.y

    # Calculate global row and column indices
    var in_col = block_idx.x * OUT_TILE_DIM + tx - 1
    var in_row = block_idx.y * OUT_TILE_DIM + ty - 1
    var i_start = block_idx.z * OUT_TILE_DIM
    var i_prev = i_start - 1

    # Shared memory index for this thread
    var shared_idx = ty * IN_TILE_DIM + tx

    # Load prev plane (i_start - 1)
    if (
        i_prev >= 0
        and i_prev < N
        and in_row >= 0
        and in_row < N
        and in_col >= 0
        and in_col < N
    ):
        var global_idx = i_prev * N * N + in_row * N + in_col
        prev_in_s[shared_idx] = Float32(d_in[global_idx])

    # Load curr plane (i_start)
    if (
        i_start >= 0
        and i_start < N
        and in_row >= 0
        and in_row < N
        and in_col >= 0
        and in_col < N
    ):
        var global_idx = i_start * N * N + in_row * N + in_col
        curr_in_s[shared_idx] = Float32(d_in[global_idx])

    barrier()

    # Thread coarsening: iterate through z direction
    for in_z in range(i_start, i_start + OUT_TILE_DIM):
        # Load next z-plane
        var next_z = in_z + 1
        if (
            next_z >= 0
            and next_z < N
            and in_row >= 0
            and in_row < N
            and in_col >= 0
            and in_col < N
        ):
            var global_idx = next_z * N * N + in_row * N + in_col
            next_in_s[shared_idx] = Float32(d_in[global_idx])

        barrier()

        # Compute output for interior points only
        if 1 <= in_z < N - 1 and 1 <= in_row < N - 1 and 1 <= in_col < N - 1:
            if (
                tx >= 1
                and tx <= OUT_TILE_DIM
                and ty >= 1
                and ty <= OUT_TILE_DIM
            ):
                # Calculate stencil using current plane neighbors and prev/next planes
                var result = Float32(0)
                result += d_c[0] * Float32(curr_in_s[shared_idx])  # center
                result += d_c[1] * Float32(curr_in_s[shared_idx - 1])  # k-1
                result += d_c[2] * Float32(curr_in_s[shared_idx + 1])  # k+1
                result += d_c[3] * Float32(
                    curr_in_s[shared_idx - IN_TILE_DIM]
                )  # j-1
                result += d_c[4] * Float32(
                    curr_in_s[shared_idx + IN_TILE_DIM]
                )  # j+1
                result += d_c[5] * Float32(prev_in_s[shared_idx])  # i-1
                result += d_c[6] * Float32(next_in_s[shared_idx])  # i+1

                # Write result to global memory
                var output_idx = in_z * N * N + in_row * N + in_col
                d_out[output_idx] = result

        barrier()

        # Rotate buffers: prev <- curr, curr <- next
        # Copy data between planes
        prev_in_s[shared_idx] = curr_in_s[shared_idx]
        curr_in_s[shared_idx] = next_in_s[shared_idx]


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

    print("3D Stencil with THREAD COARSENING")
    print("=" * 50)
    print("Grid size:", N, "x", N, "x", N)
    print("OUT_TILE_DIM:", OUT_TILE_DIM)
    print("IN_TILE_DIM:", IN_TILE_DIM)
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

    # Configure kernel launch with 2D blocks (thread coarsening in z)
    var grid_dim_x = N // OUT_TILE_DIM
    var grid_dim_y = N // OUT_TILE_DIM
    var grid_dim_z = N // OUT_TILE_DIM
    var block_dim_x = IN_TILE_DIM
    var block_dim_y = IN_TILE_DIM
    var block_dim_z = 1  # Only 2D blocks!

    print("Launching GPU kernel with thread coarsening...")
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
    print(
        "Threads per block:",
        block_dim_x * block_dim_y,
        "(each iterates",
        OUT_TILE_DIM,
        "times in z)",
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
