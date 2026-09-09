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

# Re-exported contents of `std._gpu` so `max.gpu` is ready to become the single
# public source for these names. The definitions stay in the stdlib, and each
# re-export repeats its source docstring so the `max.gpu` API reference
# documents it in place rather than publishing a bare name.
import std._gpu

comptime MAX_THREADS_PER_BLOCK_METADATA = std._gpu.MAX_THREADS_PER_BLOCK_METADATA
"""This is metadata tag that is used in conjunction with __llvm_metadata to
give a hint to the compiler about the max threads per block that's used."""

comptime WARP_SIZE = std._gpu.WARP_SIZE
"""The number of threads that execute in lockstep within a warp on the GPU.

This constant represents the hardware warp size, which is the number of threads that execute
instructions synchronously as a unit. The value is architecture-dependent:
- 32 threads per warp on NVIDIA GPUs
- 32 threads per warp on AMD RDNA GPUs
- 64 threads per warp on AMD CDNA GPUs
- 0 if no GPU is detected

The warp size is a fundamental parameter that affects:
- Thread scheduling and execution
- Memory access coalescing
- Synchronization primitives
- Overall performance optimization
"""

comptime block_dim = std._gpu.block_dim
"""Contains the dimensions of the block as `x`, `y`, and `z` values.

For example: `block_dim.y`."""

comptime block_id_in_cluster = std._gpu.block_id_in_cluster
"""Contains the block id of the threadblock within a cluster, as `x`, `y`, and `z` values."""

comptime block_idx = std._gpu.block_idx
"""Contains the block index in the grid, as `x`, `y`, and `z` values."""

comptime cluster_dim = std._gpu.cluster_dim
"""Contains the dimensions of the cluster, as `x`, `y`, and `z` values."""

comptime cluster_idx = std._gpu.cluster_idx
"""Contains the cluster index in the grid, as `x`, `y`, and `z` values."""

comptime global_idx = std._gpu.global_idx
"""Contains the global offset of the kernel launch, as `x`, `y`, and `z`
values."""

comptime grid_dim = std._gpu.grid_dim
"""Provides accessors for getting the `x`, `y`, and `z`
dimensions of a grid."""


@always_inline("nodebug")
def lane_id() -> Int:
    """Returns the lane ID of the current thread within its warp.

    The lane ID is a unique identifier for each thread within a warp, ranging from 0 to
    WARP_SIZE-1. This ID is commonly used for warp-level programming and thread
    synchronization within a warp.

    Returns:
        The lane ID (0 to WARP_SIZE-1) of the current thread.
    """
    return std._gpu.lane_id()


@always_inline("nodebug")
def sm_id() -> Int:
    """Returns the Streaming Multiprocessor (SM) ID of the current thread.

    The SM ID uniquely identifies which physical streaming multiprocessor the thread is
    executing on. This is useful for SM-level optimizations and understanding hardware
    utilization.

    If called on non-NVIDIA GPUs, this function aborts as this functionality
    is only supported on NVIDIA hardware.

    Returns:
        The SM ID of the current thread.
    """
    return std._gpu.sm_id()


comptime thread_idx = std._gpu.thread_idx
"""Contains the thread index in the block, as `x`, `y`, and `z` values."""


@always_inline("nodebug")
def warp_id[*, broadcast: Bool = False]() -> Int:
    """Returns the warp ID of the current thread within its block.
    The warp ID is a unique identifier for each warp within a block, ranging
    from 0 to BLOCK_SIZE/WARP_SIZE-1. This ID is commonly used for warp-level
    programming and synchronization within a block.

    Parameters:
        broadcast: If true, broadcasts the warp ID to all threads in the warp,
                   ensuring that all threads in the same warp have the same
                   value. This can be useful for certain warp-level algorithms.

    Returns:
        The warp ID (0 to BLOCK_SIZE/WARP_SIZE-1) of the current thread.
    """
    return std._gpu.warp_id[broadcast=broadcast]()


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
