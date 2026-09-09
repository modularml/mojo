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
"""GPU memory operations package.

This package provides GPU memory operations including:

- **async_copy**: Asynchronous memory copy operations
- **cp_async_bulk**: Bulk tensor memory copy operations (TMA)
- **load/store**: Memory access operations with cache control
- **fence operations**: Memory ordering and synchronization
- **mbarrier**: Memory barrier operations
- **edge-masked loads**: Apple M5 (AGX3) hardware bounds-checked vector loads

These operations enable efficient data movement between different
GPU memory spaces and coordination of memory operations across threads.
"""

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
    cp_async_bulk_global_shared_cta,
    cp_async_bulk_prefetch,
    cp_async_bulk_reduce_global_shared_cta,
    cp_async_bulk_shared_cluster_global,
    cp_async_bulk_tensor_2d_gather4,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_global_shared_cta_elect,
    cp_async_bulk_tensor_reduce_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_shared_cluster_global_elect,
    cp_async_bulk_tensor_shared_cluster_global_im2col,
    cp_async_bulk_tensor_shared_cluster_global_im2col_multicast,
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
from .masked_load_apple import (
    build_edge_mask,
    edge_masked_load,
    gmem_edge_masked_load,
    smem_edge_masked_load,
)
