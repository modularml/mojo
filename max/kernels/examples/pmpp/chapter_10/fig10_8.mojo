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

from max.gpu import thread_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.random import random_float64
from std.math import abs
from std.bit import log2_floor

# ========================== CONFIGURATION ==========================
comptime BLOCK_DIM = 256


# ========================== KERNEL CODE ==========================
def convergent_sum_reduction_kernel(
    input: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
):
    """Convergent sum reduction with less control divergence (loop unrolled).

    Args:
        input: Input array (in global memory, modified in-place).
        output: Output scalar for the sum result.
    """
    var i = UInt32(thread_idx.x)

    # Unrolled reduction: strides 256, 128, 64, 32, 16, 8, 4, 2, 1
    comptime NUM_ITERATIONS = log2_floor(BLOCK_DIM) + 1

    comptime for s in range(NUM_ITERATIONS):
        comptime stride = BLOCK_DIM >> s
        if UInt32(thread_idx.x) < UInt32(stride):
            input[Int(i)] += input[Int(i + UInt32(stride))]
        barrier()

    if thread_idx.x == 0:
        output[0] = input[0]


# ========================== TEST CODE ==========================
def cpu_sum(
    input: UnsafePointer[mut=False, Float32, _],
    output: UnsafePointer[mut=True, Float32, _],
    N: Int,
):
    """CPU reference sum implementation.

    Args:
        input: Input array.
        output: Output scalar for the sum result.
        N: Number of elements.
    """
    var sum = Float32(0.0)
    for i in range(N):
        sum += input[i]
    output[0] = sum


def main() raises:
    """Example usage of convergent sum reduction kernel."""
    comptime N = BLOCK_DIM * 2

    # Host memory allocation
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](1)
    var h_ref = alloc[Float32](1)

    # Initialize input with random values
    for i in range(N):
        h_input[i] = random_float64().cast[.float32]()

    print(
        "Launching convergent sum reduction kernel (Fig 10.8) with 1 block and",
        BLOCK_DIM,
        "threads",
    )
    print("Less control divergence and improved execution resource utilization")

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[.float32](N)
        var d_output = ctx.enqueue_create_buffer[.float32](1)

        # Copy data to device
        ctx.enqueue_copy(d_input, h_input)

        # Launch kernel (single block)
        ctx.enqueue_function[convergent_sum_reduction_kernel](
            d_input,
            d_output,
            grid_dim=(1, 1, 1),
            block_dim=(BLOCK_DIM, 1, 1),
        )

        # Copy result back to host
        ctx.enqueue_copy(h_output, d_output)
        ctx.synchronize()

    # Run CPU reference
    cpu_sum(h_input, h_ref, N)

    # Verify results
    var diff = abs(h_ref[0] - h_output[0])
    var rel = diff / (abs(h_ref[0]) + 1e-7)

    if rel > 1e-4:
        print(
            "FAILED: CPU=",
            h_ref[0],
            ", GPU=",
            h_output[0],
            ", diff=",
            diff,
            ", rel=",
            rel,
        )
    else:
        print("SUCCESS: CPU=", h_ref[0], ", GPU=", h_output[0], "match!")

    # Cleanup
    h_input.free()
    h_output.free()
    h_ref.free()
