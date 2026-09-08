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

"""Capturing `enqueue_function` encodes `DevicePointerEngine` tiles.

A unified closure that captures a `TileTensor` backed by
`DevicePointerEngine` must go through `DevicePassable` at
`DeviceContext.enqueue_function(kernel, grid_dim=, block_dim=)`. Packing the
host bytes would leave a host `DevicePointer` on the device; encoding writes
the real device address so the kernel store is visible after copy-back.
"""

from layout import TileTensor, row_major
from layout.tensor_engine import DevicePointerEngine

from max.gpu.host import DeviceContext
from std.testing import assert_equal


def test_capturing_enqueue_encodes_device_pointer_tile(
    ctx: DeviceContext,
) raises:
    var buf = ctx.enqueue_create_buffer[.float32](1)
    buf.enqueue_fill(Float32(0))

    var tile = TileTensor(buf.device_ptr(), row_major[1]())
    comptime assert tile.Engine == DevicePointerEngine[element_width=1]

    def kernel() {var tile}:
        tile[0] = 42.0

    ctx.enqueue_function(kernel, grid_dim=1, block_dim=1)

    var host = ctx.enqueue_create_host_buffer[.float32](1)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    assert_equal(host[0], Float32(42.0))


def main() raises:
    with DeviceContext() as ctx:
        test_capturing_enqueue_encodes_device_pointer_tile(ctx)
