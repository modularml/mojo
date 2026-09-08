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

from std.collections import OptionalReg

from max.gpu.primitives.cluster import block_rank_in_cluster
from max.gpu.host import DeviceContext, Dim
from max.gpu import block_idx, cluster_idx


def test_thread_block_cluster():
    var rank = block_rank_in_cluster()
    print(
        "cluster_ids=(",
        cluster_idx.x,
        ",",
        cluster_idx.y,
        ",",
        cluster_idx.z,
        ")",
        "block_ids=(",
        block_idx.x,
        ",",
        block_idx.y,
        ")",
        "block_rank=(",
        rank,
        ")",
    )


# CHECK-LABEL: test_tbc_launch_config_2x1x1
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 1 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 1 ) block_rank=( 1 )
def test_tbc_launch_config_2x1x1(ctx: DeviceContext) raises:
    print("== test_tbc_launch_config_2x1x1")
    comptime kernel = test_thread_block_cluster
    ctx.enqueue_function[kernel](
        grid_dim=(2, 2),
        block_dim=(1),
        cluster_dim=OptionalReg[Dim]((2, 1, 1)),
    )
    ctx.synchronize()


# CHECK-LABEL: test_tbc_launch_config_1x2x1
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 1 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 1 , 1 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 0 )
def test_tbc_launch_config_1x2x1(ctx: DeviceContext) raises:
    print("== test_tbc_launch_config_1x2x1")
    comptime kernel = test_thread_block_cluster
    ctx.enqueue_function[kernel](
        grid_dim=(2, 2),
        block_dim=(1),
        cluster_dim=OptionalReg[Dim]((1, 2, 1)),
    )
    ctx.synchronize()


# CHECK-LABEL: test_tbc_launch_config_2x2x2
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 1 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 1 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 1 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 1 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 3 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 0 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 3 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 3 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 3 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 2 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 1 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 1 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 2 ) block_rank=( 4 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 2 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 0 , 2 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 0 , 1 , 0 ) block_ids=( 1 , 2 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 3 , 0 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 1 , 0 , 0 ) block_ids=( 2 , 0 ) block_rank=( 4 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 3 ) block_rank=( 6 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 3 ) block_rank=( 7 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 2 ) block_rank=( 0 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 2 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 3 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 1 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 1 ) block_rank=( 2 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 1 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 1 , 0 ) block_rank=( 5 )
# CHECK-DAG: cluster_ids=( 0 , 0 , 0 ) block_ids=( 0 , 0 ) block_rank=( 4 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 3 , 3 ) block_rank=( 3 )
# CHECK-DAG: cluster_ids=( 1 , 1 , 0 ) block_ids=( 2 , 2 ) block_rank=( 4 )
def test_tbc_launch_config_2x2x2(ctx: DeviceContext) raises:
    print("== test_tbc_launch_config_2x2x2")
    comptime kernel = test_thread_block_cluster
    ctx.enqueue_function[kernel](
        grid_dim=(4, 4, 2),
        block_dim=(1),
        cluster_dim=OptionalReg[Dim]((2, 2, 2)),
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        test_tbc_launch_config_2x1x1(ctx)
        test_tbc_launch_config_1x2x1(ctx)
        test_tbc_launch_config_2x2x2(ctx)
