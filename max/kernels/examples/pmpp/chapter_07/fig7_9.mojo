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

# Figure 7.9: 2D Convolution with Constant Memory (from PMPP Chapter 7)
# Demonstrates 2D convolution with filter in constant memory
# Note: Mojo version uses global memory for filter since constant memory API
# may differ from CUDA. The structure remains the same.

from std.random import random_float64
from std.math import ceildiv
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.itertools import product

# ========================== KERNEL CODE ==========================


def convolution_2D_const_mem_kernel(
    N: UnsafePointer[Float32, ImmutAnyOrigin],
    F: UnsafePointer[Float32, ImmutAnyOrigin],
    P: UnsafePointer[Float32, MutAnyOrigin],
    r_dev: Int32,
    width_dev: Int32,
    height_dev: Int32,
):
    """2D convolution kernel using constant memory for filter.

    Args:
        N: Input array (device).
        F: Filter array in constant/global memory (device).
        P: Output array (device).
        r_dev: Filter radius (filter size = 2*r + 1).
        width_dev: Input width.
        height_dev: Input height.
    """
    # `Int` is not device-passable; widen the fixed-width args.
    var r = Int(r_dev)
    var width = Int(width_dev)
    var height = Int(height_dev)
    comptime BLOCK_DIM = 16
    var outCol = block_idx.x * BLOCK_DIM + thread_idx.x
    var outRow = block_idx.y * BLOCK_DIM + thread_idx.y

    if outRow >= height or outCol >= width:
        return

    var Pvalue = Float32(0.0)

    for fRow in range(2 * r + 1):
        for fCol in range(2 * r + 1):
            var inRow = outRow - r + fRow
            var inCol = outCol - r + fCol
            if inRow >= 0 and inRow < height and inCol >= 0 and inCol < width:
                # Access filter from constant/global memory
                Pvalue += (
                    F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol]
                )

    P[outRow * width + outCol] = Pvalue


def convolution_2d_const_mem(
    h_in: UnsafePointer[mut=False, Float32, _],
    h_filter: UnsafePointer[mut=False, Float32, _],
    h_out: UnsafePointer[mut=True, Float32, _],
    r: Int,
    width: Int,
    height: Int,
    ctx: DeviceContext,
) raises:
    """Host function for 2D convolution with constant memory.

    Args:
        h_in: Input array (host).
        h_filter: Filter array (host).
        h_out: Output array (host).
        r: Filter radius.
        width: Input width.
        height: Input height.
        ctx: Device context for GPU operations.
    """
    comptime dtype = DType.float32

    var in_size = width * height
    var filter_width = 2 * r + 1
    var filter_size = filter_width * filter_width

    # Allocate device memory
    var d_in = ctx.enqueue_create_buffer[dtype](in_size)
    var d_filter = ctx.enqueue_create_buffer[dtype](filter_size)
    var d_out = ctx.enqueue_create_buffer[dtype](in_size)

    # Copy data to device (in CUDA this would use cudaMemcpyToSymbol for constant mem)
    ctx.enqueue_copy(d_in, h_in)
    ctx.enqueue_copy(d_filter, h_filter)

    # Configure kernel launch
    comptime BLOCK_DIM = 16
    var grid_dim_x = ceildiv(width, BLOCK_DIM)
    var grid_dim_y = ceildiv(height, BLOCK_DIM)

    # Launch kernel
    ctx.enqueue_function[convolution_2D_const_mem_kernel](
        d_in,
        d_filter,
        d_out,
        Int32(r),
        Int32(width),
        Int32(height),
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    # Copy result back to host
    ctx.enqueue_copy(h_out, d_out)
    ctx.synchronize()


# ========================== TEST CODE ==========================


def cpu_2d_conv(
    inarr: UnsafePointer[mut=False, Float32, _],
    outarr: UnsafePointer[mut=True, Float32, _],
    width: Int,
    height: Int,
    h_filter: UnsafePointer[mut=False, Float32, _],
    filter_width: Int,
    r: Int,
):
    """CPU reference implementation for 2D convolution.

    Args:
        inarr: Input array.
        outarr: Output array.
        width: Input width.
        height: Input height.
        h_filter: Filter array.
        filter_width: Filter width.
        r: Filter radius.
    """
    for row, col in product(range(height), range(width)):
        var out = row * width + col
        outarr[out] = 0.0
        for dr, dc in product(range(-r, r + 1), range(-r, r + 1)):
            var nr = row + dr
            var fr = dr + r
            var nc = col + dc
            var fc = dc + r
            if 0 <= nr and nr < height and 0 <= nc and nc < width:
                outarr[out] += (
                    h_filter[fr * filter_width + fc] * inarr[nr * width + nc]
                )


def main() raises:
    comptime FILTER_RADIUS = 2
    var w = 128
    var h = 256
    var in_elements = h * w
    var filter_width = 2 * FILTER_RADIUS + 1
    var filter_elements = filter_width * filter_width

    # Allocate host memory
    var h_in = alloc[Float32](in_elements)
    var h_filter = alloc[Float32](filter_elements)
    var h_out = alloc[Float32](in_elements)
    var h_ref = alloc[Float32](in_elements)

    # Initialize input and filter with random values
    for i in range(in_elements):
        h_in[i] = random_float64().cast[.float32]()
    for i in range(filter_elements):
        h_filter[i] = random_float64().cast[.float32]()

    # Run GPU convolution
    with DeviceContext() as ctx:
        convolution_2d_const_mem(
            h_in, h_filter, h_out, FILTER_RADIUS, w, h, ctx
        )

    # Run CPU reference
    cpu_2d_conv(h_in, h_ref, w, h, h_filter, filter_width, FILTER_RADIUS)

    # Verify results
    var errors = 0
    for i, j in product(range(h), range(w)):
        var idx = i * w + j
        var diff = abs(h_ref[idx] - h_out[idx])
        var rel = diff / (abs(h_ref[idx]) + 1e-7)
        if rel > 1e-3:
            if errors < 10:
                print(
                    "Error at (",
                    i,
                    ",",
                    j,
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
    h_filter.free()
    h_out.free()
    h_ref.free()
