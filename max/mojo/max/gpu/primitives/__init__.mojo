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
"""GPU primitives package - warp, block, and cluster operations.

This package provides low-level GPU execution primitives at various levels
of the GPU hierarchy:

- **warp**: Warp-level operations (shuffle, reduce, broadcast)
- **block**: Block-level operations (reductions across thread blocks)
- **cluster**: Cluster-level synchronization (SM90+)
- **grid_controls**: Grid dependency control (Hopper PDL)
- **id**: Thread/block/grid indexing and dimensions

These primitives form the foundation for GPU kernel development.
"""

# Thread/block/grid indexing
from .id import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    cluster_dim,
    cluster_idx,
    global_idx,
    grid_dim,
    lane_id,
    sm_id,
    thread_idx,
    warp_id,
)

# Grid control operations (Hopper PDL)
from .grid_controls import (
    PDL,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

# Cluster operations (SM90+)
from .cluster import (
    block_rank_in_cluster,
    cluster_arrive,
    cluster_arrive_relaxed,
    cluster_sync,
    cluster_sync_relaxed,
    cluster_wait,
    elect_one_sync,
)
