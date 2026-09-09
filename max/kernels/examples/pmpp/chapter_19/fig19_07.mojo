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

"""Figure 19.7: Kernel for the forward path of a convolutional layer.

This kernel implements the basic forward pass of a convolutional layer
using a direct convolution approach. Each thread computes one output
pixel in the output feature map.

The grid dimensions are organized as:
- blockIdx.x: Output feature map index (m)
- blockIdx.y: Tile index (T = H_grid * W_grid)
- blockIdx.z: Batch index (n)
"""

from max.gpu import block_idx, thread_idx, grid_dim
from max.gpu.host import DeviceContext

from conv_utils import idx_x, idx_f, idx_y, conv_cpu, init_data, verify_results

comptime TILE_WIDTH = 16


def conv_layer_forward_kernel(
    N_dev: Int32,
    C_dev: Int32,
    H_dev: Int32,
    W_dev: Int32,
    K_dev: Int32,
    W_grid_dev: Int32,
    X: UnsafePointer[Float32, ImmutAnyOrigin],
    F: UnsafePointer[Float32, ImmutAnyOrigin],
    Y: UnsafePointer[Float32, MutAnyOrigin],
):
    """Convolution layer forward kernel.

    Each thread computes one element of the output feature map.
    The tile-based indexing allows processing large feature maps.

    Args:
        N_dev: Batch size.
        C_dev: Input channels.
        H_dev: Input height.
        W_dev: Input width.
        K_dev: Kernel size.
        W_grid_dev: Width grid dimension (W_out / TILE_WIDTH).
        X: Input tensor pointer.
        F: Filter tensor pointer.
        Y: Output tensor pointer.
    """
    # Int is not device-passable; widen the fixed-width args.
    var N = Int(N_dev)
    var C = Int(C_dev)
    var H = Int(H_dev)
    var W = Int(W_dev)
    var K = Int(K_dev)
    var W_grid = Int(W_grid_dev)
    var m = block_idx.x  # Output feature map index
    var h_grid, w_grid = divmod(block_idx.y, W_grid)
    var h = h_grid * TILE_WIDTH + thread_idx.y  # Output H index
    var w = w_grid * TILE_WIDTH + thread_idx.x  # Output W index
    var n = block_idx.z  # Batch index

    var H_out = H - K + 1
    var W_out = W - K + 1
    var M = grid_dim.x  # Number of output channels

    if n < N and m < M and h < H_out and w < W_out:
        var acc: Float32 = 0.0

        # Sum over all input channels
        for c in range(C):
            # Loop over KxK filter
            for p in range(K):
                for q in range(K):
                    var x_idx = idx_x(n, c, h + p, w + q, N, C, H, W)
                    var f_idx = idx_f(m, c, p, q, M, C, K)
                    acc += X[x_idx] * F[f_idx]

        var y_idx = idx_y(n, m, h, w, N, M, H_out, W_out)
        Y[y_idx] = acc


def main() raises:
    print("Figure 19.7: Convolution Layer Forward Kernel")

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

    # Figure 19.5: Host code for kernel launch
    var W_grid = (W_out + TILE_WIDTH - 1) // TILE_WIDTH  # Ceiling division
    var H_grid = (H_out + TILE_WIDTH - 1) // TILE_WIDTH
    var T = H_grid * W_grid

    print(
        "Launching Kernel with Grid(",
        M,
        ",",
        T,
        ",",
        N,
        "), Block(",
        TILE_WIDTH,
        ",",
        TILE_WIDTH,
        ", 1)",
    )

    ctx.enqueue_function[conv_layer_forward_kernel](
        Int32(N),
        Int32(C),
        Int32(H),
        Int32(W),
        Int32(K),
        Int32(W_grid),
        d_X,
        d_F,
        d_Y,
        grid_dim=(M, T, N),
        block_dim=(TILE_WIDTH, TILE_WIDTH, 1),
    )

    # Copy results back
    ctx.enqueue_copy(h_Y_gpu, d_Y)
    ctx.synchronize()

    # CPU verification
    conv_cpu(N, M, C, H, W, K, h_X, h_F, h_Y_cpu)

    if verify_results(h_Y_cpu, h_Y_gpu, N, M, H_out, W_out):
        print("Figure 19.7 Kernel Passed!")
    else:
        print("Figure 19.7 Kernel Failed!")

    # Cleanup
    h_X.free()
    h_F.free()
    h_Y_cpu.free()
    h_Y_gpu.free()
