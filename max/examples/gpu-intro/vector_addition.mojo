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

# DOC: max/gpu/intro-tutorial.mdx

from std.math import ceildiv
from std.sys import has_accelerator

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, row_major

# Vector data type and size
comptime float_dtype = DType.float32
comptime vector_size = 1000
comptime layout = row_major[vector_size]()

# Calculate the number of thread blocks needed by dividing the vector size
# by the block size and rounding up.
comptime block_size = 256
comptime num_blocks = ceildiv(vector_size, block_size)


def vector_addition(
    lhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    rhs_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
    out_tensor: TileTensor[float_dtype, type_of(layout), MutAnyOrigin],
):
    """Calculate the element-wise sum of two vectors on the GPU."""

    # Calculate the index of the vector element for the thread to process
    var tid = block_idx.x * block_dim.x + thread_idx.x

    # Don't process out of bounds elements
    if tid < vector_size:
        out_tensor[tid] = lhs_tensor[tid] + rhs_tensor[tid]


def main() raises:
    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        # Get the context for the attached GPU
        var ctx = DeviceContext()

        # Create HostBuffers for input vectors
        var lhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )
        var rhs_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )
        ctx.synchronize()

        # Initialize the input vectors
        for i in range(vector_size):
            lhs_host_buffer[i] = Float32(i)
            rhs_host_buffer[i] = Float32(Float64(i) * 0.5)

        print("LHS buffer: ", lhs_host_buffer)
        print("RHS buffer: ", rhs_host_buffer)

        # Create DeviceBuffers for the input vectors
        var lhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](
            vector_size
        )
        var rhs_device_buffer = ctx.enqueue_create_buffer[float_dtype](
            vector_size
        )

        # Copy the input vectors from the HostBuffers to the DeviceBuffers
        ctx.enqueue_copy(dst_buf=lhs_device_buffer, src_buf=lhs_host_buffer)
        ctx.enqueue_copy(dst_buf=rhs_device_buffer, src_buf=rhs_host_buffer)

        # Create a DeviceBuffer for the result vector
        var result_device_buffer = ctx.enqueue_create_buffer[float_dtype](
            vector_size
        )

        # Wrap the DeviceBuffers in TileTensors
        var lhs_tensor = TileTensor(lhs_device_buffer, layout)
        var rhs_tensor = TileTensor(rhs_device_buffer, layout)
        var result_tensor = TileTensor(result_device_buffer, layout)

        # Compile and enqueue the kernel
        ctx.enqueue_function[vector_addition](
            lhs_tensor,
            rhs_tensor,
            result_tensor,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        # Create a HostBuffer for the result vector
        var result_host_buffer = ctx.enqueue_create_host_buffer[float_dtype](
            vector_size
        )

        # Copy the result vector from the DeviceBuffer to the HostBuffer
        ctx.enqueue_copy(
            dst_buf=result_host_buffer, src_buf=result_device_buffer
        )

        # Finally, synchronize the DeviceContext to run all enqueued operations
        ctx.synchronize()

        print("Result vector:", result_host_buffer)
