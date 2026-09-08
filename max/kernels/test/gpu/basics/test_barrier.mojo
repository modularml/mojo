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

import max.gpu.primitives.warp as warp
from max.gpu import global_idx
from max.gpu.sync import barrier
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def kernel[
    dtype: DType
](
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    shared_data: MutPointer[Scalar[dtype], MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var global_tid = global_idx.x
    if global_tid >= size:
        return
    shared_data[global_tid] = input[global_tid]

    barrier()

    var result = shared_data[global_tid] + shared_data[0]

    output[global_tid] = result


def test_barrier[dtype: DType](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size
    comptime constant_add: Scalar[dtype] = 42
    var input_host = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    var output_host = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    var shared_host = ctx.enqueue_create_host_buffer[dtype](buffer_size)

    for i in range(buffer_size):
        input_host[i] = Scalar[dtype](i) + constant_add
        output_host[i] = -1.0
        shared_host[i] = -999.0

    var input_buffer = ctx.enqueue_create_buffer[dtype](buffer_size)
    var output_buffer = ctx.enqueue_create_buffer[dtype](buffer_size)
    var shared_buffer = ctx.enqueue_create_buffer[dtype](buffer_size)

    ctx.enqueue_copy(input_buffer, input_host)
    ctx.enqueue_copy(output_buffer, output_host)
    ctx.enqueue_copy(shared_buffer, shared_host)

    ctx.enqueue_function[kernel[dtype]](
        input_buffer,
        output_buffer,
        shared_buffer,
        Int32(buffer_size),
        grid_dim=1,
        block_dim=block_size,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.enqueue_copy(shared_host, shared_buffer)
    ctx.synchronize()

    for i in range(buffer_size):
        assert_equal(output_host[i], 2 * constant_add + Scalar[dtype](i))
        assert_equal(shared_host[i], constant_add + Scalar[dtype](i))

    _ = input_buffer
    _ = shared_buffer


def main() raises:
    with DeviceContext() as ctx:
        test_barrier[.float32](ctx)
