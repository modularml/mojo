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

from max.gpu import block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation

from std.math import abs

# ========================== CONFIGURATION ==========================
comptime SEG_SIZE = 256


# ========================== KERNEL CODE ==========================
def scan_kernel(
    input: UnsafePointer[Float32, ImmutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    N: UInt32,
):
    """Kogge-Stone scan kernel using shared memory.

    Args:
        input: Input array.
        output: Output array for scan result.
        N: Number of elements.
    """
    # Allocate shared memory
    var buffer_s = unsafe_stack_allocation[
        SEG_SIZE,
        Float32,
        address_space=.SHARED,
    ]()

    var tx = thread_idx.x
    var idx = block_idx.x * SEG_SIZE + tx

    # Read everything into shared buffer
    buffer_s[tx] = Float32(input[idx])
    barrier()

    # Kogge-Stone scan
    var stride = 1
    while stride < Int(N):
        # Read current input from shared memory
        var curr = Float32(buffer_s[tx])
        barrier()

        if tx + stride < SEG_SIZE:
            # Add it to buffer position
            buffer_s[tx + stride] = Float32(
                Float32(buffer_s[tx + stride]) + curr
            )
        barrier()
        stride *= 2

    output[idx] = Float32(buffer_s[tx])


# ========================== TEST CODE ==========================
def cpu_scan(
    input: UnsafePointer[mut=False, Float32, _],
    output: UnsafePointer[mut=True, Float32, _],
    N: UInt32,
):
    """CPU reference scan implementation.

    Args:
        input: Input array.
        output: Output array for scan result.
        N: Number of elements.
    """
    output[0] = input[0]
    for i in range(1, Int(N)):
        output[i] = output[i - 1] + input[i]


def main() raises:
    """Example usage of Kogge-Stone scan kernel."""
    comptime N = SEG_SIZE
    comptime dtype = DType.float32

    # Allocate host memory
    var h_input = alloc[Float32](N)
    var h_output = alloc[Float32](N)
    var h_ref = alloc[Float32](N)

    # Initialize input
    for i in range(N):
        h_input[i] = 1.0

    print(
        "Launching Kogge-Stone scan kernel (Fig 11.3) with 1 block and",
        SEG_SIZE,
        "threads",
    )

    with DeviceContext() as ctx:
        # Device memory allocation
        var d_input = ctx.enqueue_create_buffer[dtype](N)
        var d_output = ctx.enqueue_create_buffer[dtype](N)

        # Copy data to device
        ctx.enqueue_copy(d_input, h_input)

        # Launch kernel (single block)
        ctx.enqueue_function[scan_kernel](
            d_input,
            d_output,
            UInt32(N),
            grid_dim=(1, 1, 1),
            block_dim=(SEG_SIZE, 1, 1),
        )

        # Copy result back to host
        ctx.enqueue_copy(h_output, d_output)
        ctx.synchronize()

    # Run CPU reference
    cpu_scan(h_input, h_ref, N)

    # Verify results
    var errors = 0
    for i in range(N):
        if abs(h_output[i] - h_ref[i]) > 1e-5:
            if errors < 10:
                print(
                    "Error at index",
                    i,
                    ": CPU=",
                    h_ref[i],
                    ", GPU=",
                    h_output[i],
                )
            errors += 1

    # Cleanup
    h_input.free()
    h_output.free()
    h_ref.free()

    if errors == 0:
        print("SUCCESS: All values match!")
    else:
        print("FAILED: Found", errors, "errors")
