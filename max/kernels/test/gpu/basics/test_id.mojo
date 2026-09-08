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

from max.gpu import block_idx, global_idx, thread_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal

comptime buffer_size = 1024
comptime block_dim = 32


def test_fill_thread_idx(ctx: DeviceContext) raises:
    var output_host = ctx.enqueue_create_host_buffer[.int](buffer_size)
    var output_buffer = ctx.enqueue_create_buffer[.int](buffer_size)
    output_buffer.enqueue_fill(9)

    def kernel(output: MutPointer[Int, MutAnyOrigin]):
        output[global_idx.x] = Int(thread_idx.x)

    ctx.enqueue_function[kernel](
        output_buffer,
        grid_dim=buffer_size // block_dim,
        block_dim=block_dim,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(0, buffer_size, block_dim):
        for j in range(block_dim):
            assert_equal(output_host[i + j], Int(j))


def test_fill_block_idx(ctx: DeviceContext) raises:
    var output_host = ctx.enqueue_create_host_buffer[.int](buffer_size)
    var output_buffer = ctx.enqueue_create_buffer[.int](buffer_size)
    output_buffer.enqueue_fill(9)

    def kernel(output: MutPointer[Int, MutAnyOrigin]):
        output[global_idx.x] = Int(block_idx.x)

    ctx.enqueue_function[kernel](
        output_buffer,
        grid_dim=buffer_size // block_dim,
        block_dim=block_dim,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    for i in range(0, buffer_size, block_dim):
        for j in range(block_dim):
            assert_equal(output_host[i + j], Int(i // block_dim))


def main() raises:
    with DeviceContext() as ctx:
        test_fill_thread_idx(ctx)
        test_fill_block_idx(ctx)
