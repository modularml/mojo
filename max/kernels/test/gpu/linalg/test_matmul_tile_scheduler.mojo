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
from max.gpu import block_idx
from linalg.matmul.gpu.tile_scheduler import TileScheduler

from std.utils.index import Index


def test_kernel():
    var problem_shape = Index(12, 12, 20)

    var scheduler = TileScheduler[
        problem_shape=Index(12, 12, 20),
        tile_shape=Index(4, 4, 4),
        grid_shape=Index(2, 2),
    ](problem_shape)

    var num_output_tiles = scheduler.num_output_tiles()
    var work_info = scheduler.get_current_work_info()

    for _ in range(num_output_tiles):
        print(block_idx.x, work_info)
        work_info = scheduler.fetch_next_work()


# CHECK-DAG: 0 (0, 0, 0, 5, True)
# CHECK-DAG: 1 (0, 4, 0, 5, True)
# CHECK-DAG: 2 (4, 0, 0, 5, True)
# CHECK-DAG: 3 (4, 4, 0, 5, True)
# CHECK-DAG: 1 (0, 12, 0, 5, False)
# CHECK-DAG: 0 (0, 8, 0, 5, True)
# CHECK-DAG: 2 (4, 8, 0, 5, True)
# CHECK-DAG: 3 (4, 12, 0, 5, False)
# CHECK-DAG: 0 (8, 0, 0, 5, True)
# CHECK-DAG: 1 (8, 4, 0, 5, True)
# CHECK-DAG: 2 (12, 0, 0, 5, False)
# CHECK-DAG: 3 (12, 4, 0, 5, False)
# CHECK-DAG: 0 (8, 8, 0, 5, True)
# CHECK-DAG: 1 (8, 12, 0, 5, False)
# CHECK-DAG: 2 (12, 8, 0, 5, False)
# CHECK-DAG: 3 (12, 12, 0, 5, False)
def test(ctx: DeviceContext) raises:
    comptime kernel = test_kernel

    ctx.enqueue_function[kernel](
        grid_dim=(4),
        block_dim=(1),
    )

    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        test(ctx)
