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
# DOC: max/gpu/fundamentals.mdx

from std.math import iota
from std.sys import exit, has_accelerator

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx

comptime num_elements = 20


def scalar_add(
    vector: Pointer[Float32, MutAnyOrigin], size: Int32, scalar: Float32
):
    """
    Kernel function to add a scalar to all elements of a vector.

    This kernel function adds a scalar value to each element of a vector stored
    in GPU memory. The input vector is modified in place.

    Args:
        vector: Pointer to the input vector.
        size: Number of elements in the vector.
        scalar: Scalar to add to the vector.

    """

    # Calculate the global thread index within the entire grid. Each thread
    # processes one element of the vector.
    #
    # block_idx.x: index of the current thread block.
    # block_dim.x: number of threads per block.
    # thread_idx.x: index of the current thread within its block.
    var idx = block_idx.x * block_dim.x + thread_idx.x

    # Bounds checking: ensure we don't access memory beyond the vector size.
    # This is crucial when the number of threads doesn't exactly match vector
    # size.
    if idx < Int(size):
        # Each thread adds the scalar to its corresponding vector element
        # This operation happens in parallel across all GPU threads
        vector[unsafe_offset=idx] += scalar


def main() raises:
    comptime if not has_accelerator():
        print("No GPUs detected")
        exit(0)
    else:
        # Initialize GPU context for device 0 (default GPU device).
        var ctx = DeviceContext()

        # Create a buffer in host (CPU) memory to store our input data
        var host_buffer = ctx.enqueue_create_host_buffer[.float32](num_elements)

        # Wait for buffer creation to complete.
        ctx.synchronize()

        # Fill the host buffer with sequential numbers (0, 1, 2, ..., size-1).
        iota(host_buffer.as_span())
        print("Original host buffer:", host_buffer)

        # Create a buffer in device (GPU) memory to store data for computation.
        var device_buffer = ctx.enqueue_create_buffer[.float32](num_elements)

        # Copy data from host memory to device memory for GPU processing.
        ctx.enqueue_copy(src_buf=host_buffer, dst_buf=device_buffer)

        # Compile the scalar_add kernel function for execution on the GPU.
        var scalar_add_kernel = ctx.compile_function[scalar_add]()

        # Launch the GPU kernel with the following arguments:
        #
        # - device_buffer: GPU memory containing our vector data
        # - num_elements: number of elements in the vector
        # - Float32(20.0): the scalar value to add to each element
        # - grid_dim=1: use 1 thread block
        # - block_dim=num_elements: use 'num_elements' threads per block (one
        #   thread per vector element)
        ctx.enqueue_function(
            scalar_add_kernel,
            device_buffer,
            Int32(num_elements),
            Float32(20.0),
            grid_dim=1,
            block_dim=num_elements,
        )

        # Copy the computed results back from device memory to host memory.
        ctx.enqueue_copy(src_buf=device_buffer, dst_buf=host_buffer)

        # Wait for all GPU operations to complete.
        ctx.synchronize()

        # Display the final results after GPU computation.
        print("Modified host buffer:", host_buffer)
