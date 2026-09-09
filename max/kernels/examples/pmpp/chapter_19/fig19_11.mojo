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

"""Figure 19.11: Tiled Matrix Multiplication Kernel for Convolution.

This kernel performs GEMM implicitly transforming X to matrix form (im2col on
the fly).

Matrix Dimensions:
- Matrix A (Filters F): (M, C*K*K)
- Matrix B (Input X unrolled): (C*K*K, H_out*W_out)
- Matrix C (Output Y): (M, H_out*W_out)

The kernel computes C_sub = A_sub * B_sub using tiled matrix multiplication
with shared memory. The key optimization is that B is loaded from X using
complex indexing that implicitly performs the im2col transformation.
"""

from max.gpu import block_idx, thread_idx, block_dim, grid_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

from conv_utils import idx_x, idx_y, conv_cpu, init_data, verify_results

comptime TILE_WIDTH = 16


def conv_layer_mm_kernel(
    C_dev: Int32,
    M_dev: Int32,
    H_dev: Int32,
    W_dev: Int32,
    K_dev: Int32,
    N_batch_dev: Int32,
    F: UnsafePointer[Float32, ImmutAnyOrigin],
    X: UnsafePointer[Float32, ImmutAnyOrigin],
    Y: UnsafePointer[Float32, MutAnyOrigin],
):
    """Tiled matrix multiplication kernel for convolution.

    This kernel performs the convolution using a matrix multiplication
    approach with shared memory tiling. The input is implicitly transformed
    to matrix form (im2col) during the load to shared memory.

    Args:
        C_dev: Input channels.
        M_dev: Output channels.
        H_dev: Input height.
        W_dev: Input width.
        K_dev: Kernel size.
        N_batch_dev: Batch size.
        F: Filter tensor pointer.
        X: Input tensor pointer.
        Y: Output tensor pointer.
    """
    # Int is not device-passable; widen the fixed-width args.
    var C = Int(C_dev)
    var M = Int(M_dev)
    var H = Int(H_dev)
    var W = Int(W_dev)
    var K = Int(K_dev)
    var N_batch = Int(N_batch_dev)
    var H_out = H - K + 1
    var W_out = W - K + 1

    # Allocate shared memory for tiles
    # Fds[TILE_WIDTH][TILE_WIDTH] + Bds[TILE_WIDTH][TILE_WIDTH]
    var Fds = unsafe_stack_allocation[
        TILE_WIDTH * TILE_WIDTH,
        Float32,
        address_space=.SHARED,
    ]()
    var Bds = unsafe_stack_allocation[
        TILE_WIDTH * TILE_WIDTH,
        Float32,
        address_space=.SHARED,
    ]()

    var bx = block_idx.x
    var by = block_idx.y
    var bz = block_idx.z  # Batch index

    var tx = thread_idx.x
    var ty = thread_idx.y

    # Identify row/column of Y element
    # Row corresponds to Output Channel m
    var Row = by * TILE_WIDTH + ty
    # Col corresponds to Output Pixel Index (h_out * W_out + w_out)
    var Col = bx * TILE_WIDTH + tx

    var Pvalue: Float32 = 0.0

    # The width of F matrix is C*K*K
    var width_A = C * K * K

    # Loop over F and B tiles
    var num_phases = (width_A + TILE_WIDTH - 1) // TILE_WIDTH

    for ph in range(num_phases):
        # Load F tile into shared memory
        # F is (M, C*K*K)
        # Row is 'm', (ph*TILE_WIDTH + tx) is column index in F
        var f_col = ph * TILE_WIDTH + tx
        if Row < M and f_col < width_A:
            # F[Row, f_col] = F[Row * width_A + f_col]
            Fds[ty * TILE_WIDTH + tx] = F[Row * width_A + f_col]
        else:
            Fds[ty * TILE_WIDTH + tx] = 0.0

        # Load B tile (from X) into shared memory
        # B is (C*K*K, H_out*W_out)
        # 'u' is row in B -> input feature index
        # 'v' is col in B -> output pixel index
        var u = ph * TILE_WIDTH + ty
        var v = Col

        if u < width_A and v < H_out * W_out:
            # Mapping from Figure 19.11:
            # u = c * K * K + p * K + q
            # c = u / (K * K)
            # p = (u % (K * K)) / K
            # q = (u % (K * K)) % K
            #
            # v = h_out * W_out + w_out
            # h_out = v / W_out
            # w_out = v % W_out
            #
            # X index: (n, c, h_in, w_in)
            # h_in = h_out + p
            # w_in = w_out + q

            var c, u_rem = divmod(u, K * K)
            var p, q = divmod(u_rem, K)
            var h_out, w_out = divmod(v, W_out)

            var x_idx = idx_x(bz, c, h_out + p, w_out + q, N_batch, C, H, W)
            Bds[ty * TILE_WIDTH + tx] = X[x_idx]
        else:
            Bds[ty * TILE_WIDTH + tx] = 0.0

        barrier()

        # Compute dot product for this tile
        for k in range(TILE_WIDTH):
            Pvalue += Fds[ty * TILE_WIDTH + k] * Bds[k * TILE_WIDTH + tx]

        barrier()

    # Write result to output
    if Row < M and Col < H_out * W_out:
        var h_out, w_out = divmod(Col, W_out)
        var y_idx = idx_y(bz, Row, h_out, w_out, N_batch, M, H_out, W_out)
        Y[y_idx] = Pvalue


def main() raises:
    print("Figure 19.11: Tiled Matrix Multiplication Kernel for Convolution")

    # Parameters
    var N = 4  # Batch size
    var M = 16  # Output channels
    var C = 3  # Input channels
    var H = 64  # Input height
    var W = 64  # Input width
    var K = 5  # Kernel size

    var H_out = H - K + 1
    var W_out = W - K + 1

    var size_X = N * C * H * W
    var size_F = M * C * K * K
    var size_Y = N * M * H_out * W_out

    # Allocate host memory
    var h_X = alloc[Float32](size_X)
    var h_F = alloc[Float32](size_F)
    var h_Y_cpu = alloc[Float32](size_Y)
    var h_Y_gpu = alloc[Float32](size_Y)

    # Initialize data
    init_data(h_X, size_X)
    init_data(h_F, size_F)

    # Initialize output
    for i in range(size_Y):
        h_Y_cpu[i] = 0.0
        h_Y_gpu[i] = 0.0

    # Create device context
    var ctx = DeviceContext()

    # Allocate device memory
    var d_X = ctx.enqueue_create_buffer[.float32](size_X)
    var d_F = ctx.enqueue_create_buffer[.float32](size_F)
    var d_Y = ctx.enqueue_create_buffer[.float32](size_Y)

    # Copy to device
    ctx.enqueue_copy(d_X, h_X)
    ctx.enqueue_copy(d_F, h_F)

    # MM Kernel Launch Configuration
    # Grid X: Covers output pixels (Columns of Matrix C) -> (H_out * W_out)
    # Grid Y: Covers output channels (Rows of Matrix C) -> M
    # Grid Z: Batch size
    var num_output_pixels = H_out * W_out

    var grid_x = (num_output_pixels + TILE_WIDTH - 1) // TILE_WIDTH
    var grid_y = (M + TILE_WIDTH - 1) // TILE_WIDTH

    print("Launching MM Kernel with Grid(", grid_x, ",", grid_y, ",", N, ")")

    ctx.enqueue_function[conv_layer_mm_kernel](
        Int32(C),
        Int32(M),
        Int32(H),
        Int32(W),
        Int32(K),
        Int32(N),
        d_F,
        d_X,
        d_Y,
        grid_dim=(grid_x, grid_y, N),
        block_dim=(TILE_WIDTH, TILE_WIDTH, 1),
    )

    # Copy results back
    ctx.enqueue_copy(h_Y_gpu, d_Y)
    ctx.synchronize()

    # CPU verification
    conv_cpu(N, M, C, H, W, K, h_X, h_F, h_Y_cpu)

    if verify_results(h_Y_cpu, h_Y_gpu, N, M, H_out, W_out):
        print("Figure 19.11 Kernel Passed!")
    else:
        print("Figure 19.11 Kernel Failed!")

    # Cleanup
    h_X.free()
    h_F.free()
    h_Y_cpu.free()
    h_Y_gpu.free()
