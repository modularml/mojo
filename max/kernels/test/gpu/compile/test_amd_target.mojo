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

from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, grid_dim, thread_idx


# CHECK-LABEL: test_amd_dims
# CHECK: 14 15 16
# CHECK: 2 3 4
def test_amd_dims(ctx: DeviceContext) raises:
    print("== test_amd_dims")

    def test_dims_kernel():
        if (
            block_idx.x == 0
            and block_idx.y == 0
            and block_idx.z == 0
            and thread_idx.x == 0
            and thread_idx.y == 0
            and thread_idx.z == 0
        ):
            print(grid_dim.x, grid_dim.y, grid_dim.z)
            print(block_dim.x, block_dim.y, block_dim.z)

    comptime kernel = test_dims_kernel
    ctx.enqueue_function[kernel](
        grid_dim=(14, 15, 16),
        block_dim=(2, 3, 4),
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        test_amd_dims(ctx)
