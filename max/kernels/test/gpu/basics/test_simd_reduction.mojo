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
# This test checks that we can perform simd load and simd reductions correctly
# on GPUs.


from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal

comptime buffer_size = 1024
comptime block_dim = 32
comptime simd_width = 4


def test_simd_reduction(ctx: DeviceContext) raises:
    var input_host = ctx.enqueue_create_host_buffer[.int](buffer_size)
    var output_host = ctx.enqueue_create_host_buffer[.int](
        buffer_size // simd_width
    )

    for i in range(0, buffer_size, simd_width):
        for j in range(simd_width):
            # Fill with sensible values, but make sure the addition does not
            # blow up.
            if j == 0:
                input_host[i + j] = Int(i)
            else:
                input_host[i + j] = Int(j)

    var input_buffer = ctx.enqueue_create_buffer[.int](buffer_size)

    var output_buffer = ctx.enqueue_create_buffer[.int](
        buffer_size // simd_width
    )
    output_buffer.enqueue_fill(9)

    ctx.enqueue_copy(input_buffer, input_host)

    def kernel(
        output: MutPointer[Int, MutAnyOrigin],
        input: ImmPointer[Int, ImmutAnyOrigin],
    ):
        output[global_idx.x] = input.load[width=simd_width](
            simd_width * global_idx.x
        ).reduce_add()

    ctx.enqueue_function[kernel](
        output_buffer,
        input_buffer,
        grid_dim=buffer_size // (block_dim * simd_width),
        block_dim=block_dim,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    var simd_sum = 0
    for i in range(simd_width):
        simd_sum += i

    for i in range(buffer_size // simd_width):
        assert_equal(output_host[i], Int(i * simd_width + simd_sum))


def main() raises:
    with DeviceContext() as ctx:
        test_simd_reduction(ctx)
