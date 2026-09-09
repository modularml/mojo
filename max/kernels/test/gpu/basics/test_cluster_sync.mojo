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

from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from max.gpu.host import DeviceContext, Dim
from max.gpu import cluster_dim


def test_cluster_sync_kernel():
    var block_rank = block_rank_in_cluster()
    var num_blocks_in_cluster = cluster_dim.x * cluster_dim.y * cluster_dim.z

    for i in range(num_blocks_in_cluster):
        if block_rank == UInt32(i):
            print(block_rank)
        cluster_sync()


# CHECK-LABEL: test_cluster_sync
# CHECK: 0
# CHECK: 1
# CHECK: 2
# CHECK: 3
# CHECK: 4
# CHECK: 5
# CHECK: 6
# CHECK: 7
def test_cluster_sync(ctx: DeviceContext) raises:
    print("== test_cluster_sync")
    comptime kernel = test_cluster_sync_kernel
    ctx.enqueue_function[kernel](
        grid_dim=(2, 2, 2),
        block_dim=(1),
        cluster_dim=Dim((2, 2, 2)),
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        test_cluster_sync(ctx)
