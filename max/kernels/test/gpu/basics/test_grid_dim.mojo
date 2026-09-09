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
from max.gpu import global_idx, grid_dim
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def kernel(
    output: MutPointer[Float32, MutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var global_tid = global_idx.x
    if global_tid >= size:
        return
    if global_tid & 3 == 0:
        output[global_tid] = Float32(grid_dim.x)
    elif global_tid & 3 == 1:
        output[global_tid] = Float32(grid_dim.y)
    elif global_tid & 3 == 2:
        output[global_tid] = Float32(grid_dim.z)


def test_grid_dim(ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size
    var output_host = ctx.enqueue_create_host_buffer[.float32](buffer_size)
    for i in range(buffer_size):
        output_host[i] = -1.0

    var output_buffer = ctx.enqueue_create_buffer[.float32](buffer_size)

    ctx.enqueue_copy(output_buffer, output_host)

    ctx.enqueue_function[kernel](
        output_buffer,
        Int32(buffer_size),
        grid_dim=(20, 15, 10),
        block_dim=20,
    )

    ctx.enqueue_copy(output_host, output_buffer)
    ctx.synchronize()

    assert_equal(output_host[0], 20)
    assert_equal(output_host[1], 15)
    assert_equal(output_host[2], 10)


def main() raises:
    with DeviceContext() as ctx:
        test_grid_dim(ctx)
