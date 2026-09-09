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

# DOC: max/develop/build-custom-ops.mdx
import extensibility

from max.gpu.host import DeviceContext
from std.math import ceildiv

from max.gpu import block_dim, block_idx, thread_idx

from extensibility import InputTensor, ManagedTensorSlice, OutputTensor

from std.utils.index import IndexList


def _vector_addition_cpu(
    output: ManagedTensorSlice[mut=True, ...],
    lhs: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
    rhs: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
    ctx: DeviceContext,
):
    # Warning: This is an extremely inefficient implementation! It's merely an
    # instructional example of how a dedicated CPU-only path can be specified
    # for basic vector addition.
    var vector_length = output.dim_size(0)
    for i in range(vector_length):
        var idx = IndexList[output.rank](i)
        var result = lhs.load[1](idx) + rhs.load[1](idx)
        output.store[1](idx, result)


def _vector_addition_gpu(
    output: ManagedTensorSlice[mut=True, ...],
    lhs: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
    rhs: ManagedTensorSlice[dtype=output.dtype, rank=output.rank, ...],
    ctx: DeviceContext,
) raises:
    # Note: The following has not been tuned for any GPU hardware, and is an
    # instructional example for how a simple GPU function can be constructed
    # and dispatched.
    comptime BLOCK_SIZE = 16
    var gpu_ctx = ctx
    var vector_length = output.dim_size(0)

    # The function that will be launched and distributed across GPU threads.
    @__parameter
    def vector_addition_gpu_kernel(length_dev: Int32):
        var length = Int(length_dev)
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < length:
            var idx = IndexList[output.rank](tid)
            var result = lhs.load[1](idx) + rhs.load[1](idx)
            output.store[1](idx, result)

    # The vector is divided up into blocks, making sure there's an extra
    # full block for any remainder.
    var num_blocks = ceildiv(vector_length, BLOCK_SIZE)

    # The GPU function is compiled and enqueued to run on the GPU across the
    # 1-D vector, split into blocks of `BLOCK_SIZE` width.
    gpu_ctx.enqueue_function[vector_addition_gpu_kernel](
        Int32(vector_length), grid_dim=num_blocks, block_dim=BLOCK_SIZE
    )


@extensibility.register("vector_addition")
struct VectorAddition:
    @staticmethod
    def execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor[rank=1, ...],
        lhs: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        rhs: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        # the context is needed for some GPU calls
        ctx: DeviceContext,
    ) raises:
        # For a simple elementwise operation like this, the `foreach` function
        # does much more rigorous hardware-specific tuning. We recommend using
        # that abstraction, with this example serving purely as an illustration
        # of how lower-level functions can be used to program GPUs via Mojo.

        # At graph compilation time, we will know what device we are compiling
        # this operation for, so we can specialize it for the target hardware.
        comptime if target == "cpu":
            _vector_addition_cpu(output, lhs, rhs, ctx)
        elif target == "gpu":
            _vector_addition_gpu(output, lhs, rhs, ctx)
        else:
            raise Error("No known target:", target)
