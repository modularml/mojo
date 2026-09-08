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
"""GPU programming primitives.

These low level constructs allow you to write code that runs on the GPU with
traditional programming style--partitioning work across threads that are mapped
onto 1-, 2-, or 3-dimensional blocks. The thread blocks can subsequently be
grouped into a grid of thread blocks.

A _kernel_ is a function that runs on the GPU in parallel across many threads.
Currently, the
[`DeviceContext`](/api/mojo/max/gpu/host/device_context/DeviceContext/) struct
provides the interface for compiling and launching GPU kernels inside MAX
[custom operations](https://max.modular.com/develop/custom-ops/).

The [`gpu.host`](/api/mojo/max/gpu/host/) package includes APIs to manage
interaction between the _host_ (that is, the CPU) and _device_ (that is, the GPU
or accelerator).

The `gpu` package exports aliases you can use to access information about the
grid and the current thread, including block dimensions, block index in the grid,
and thread index. Import these directly from `gpu`:

```mojo
from max.gpu import block_dim, block_idx, thread_idx, global_idx
```

For an example of launching a GPU kernel from a MAX custom operation, see the
[vector addition example](https://github.com/modular/modular/blob/main/max/examples/custom_ops/kernels/vector_addition.mojo)
in the MAX repo.
"""


@__doc_inline
from std._gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
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

from .primitives import (
    block_rank_in_cluster,
    cluster_arrive,
    cluster_arrive_relaxed,
    cluster_sync,
    cluster_sync_relaxed,
    cluster_wait,
    elect_one_sync,
    PDL,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

from .memory import (
    AddressSpace,
    CacheEviction,
    CacheOperation,
    Consistency,
    Fill,
    ReduceOp,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_global_shared_cta_elect,
    cp_async_bulk_tensor_reduce_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_shared_cluster_global_multicast,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
    fence_proxy_tensormap_generic_sys_acquire,
    fence_proxy_tensormap_generic_sys_release,
    load,
    multimem_ld_reduce,
    multimem_st,
)

from .sync import (
    NamedBarrierSemaphore,
    Semaphore,
    AMDScheduleBarrierMask,
    async_copy_arrive,
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_test_wait,
    mbarrier_try_wait_parity_shared,
    named_barrier,
    schedule_barrier,
    schedule_group_barrier,
    syncwarp,
    s_waitcnt,
    s_waitcnt_barrier,
)
