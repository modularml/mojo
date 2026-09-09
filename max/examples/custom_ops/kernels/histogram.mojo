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

import extensibility

from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.atomic import Atomic

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    global_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host.info import is_cpu
from max.gpu.host import DeviceBuffer
from std.memory import unsafe_stack_allocation

from extensibility import InputTensor, ManagedTensorSlice, OutputTensor

from std.utils import StaticTuple

comptime bin_width = Int(UInt8.MAX)


def _histogram_cpu(output: ManagedTensorSlice, input: ManagedTensorSlice):
    for i in range(output.dim_size(0)):
        output[i] = 0

    for i in range(input.dim_size(0)):
        output[Int(input[i])] += 1


def _histogram_gpu(
    output: ManagedTensorSlice,
    input: ManagedTensorSlice,
    ctx_ptr: DeviceContext,
) raises:
    comptime bin_width = Int(UInt8.MAX) + 1
    comptime block_dim = bin_width

    # Set the maximum number of threads per block to the block dimension.
    # This is equivalent to the `__launch_bounds__` attribute in CUDA.
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_dim))
    )
    def kernel(
        output: Pointer[Int64, MutAnyOrigin],
        input: Pointer[UInt8, MutAnyOrigin],
        n_dev: Int32,
    ):
        var n = Int(n_dev)
        var tid = global_idx.x

        if tid >= n:
            return

        # Allocate shared memory for the histogram
        var shared_mem = unsafe_stack_allocation[
            bin_width, Int64, address_space=.SHARED
        ]()

        # Initialize the shared memory to 0
        shared_mem[unsafe_offset=thread_idx.x] = 0

        # Synchronize all threads to ensure that the shared memory is initialized
        barrier()

        # Increment the shared memory for the current thread
        _ = Atomic.fetch_add(
            shared_mem.unsafe_offset(Int(input[unsafe_offset=tid])), 1
        )

        # Synchronize all threads to ensure that the shared memory is updated
        barrier()

        # Increment the output for the current thread
        _ = Atomic.fetch_add(
            output.unsafe_offset(thread_idx.x),
            shared_mem[unsafe_offset=thread_idx.x],
        )

    var n = input.dim_size(0)

    var grid_dim = ceildiv(n, block_dim)

    var ctx = ctx_ptr

    var output_device = output.to_device_buffer(ctx)
    var input_device = input.to_device_buffer(ctx)

    # Zero initialize the output buffer
    ctx.enqueue_memset(output_device, 0)

    ctx.enqueue_function[kernel](
        output_device,
        input_device,
        Int32(n),
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@extensibility.register("histogram")
struct Histogram:
    @staticmethod
    def execute[
        target: StaticString
    ](
        output: OutputTensor[dtype=.int64, rank=1, ...],
        input: InputTensor[dtype=.uint8, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if is_cpu[target]():
            _histogram_cpu(output, input)
        else:
            _histogram_gpu(output, input, ctx)
