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
"""GPU synchronization primitives package.

This package provides GPU synchronization operations including:

- **barrier**: Block-level synchronization barriers
- **syncwarp**: Warp-level synchronization
- **mbarrier**: Memory barrier operations (arrive/wait)
- **named_barrier**: Named barriers for flexible synchronization
- **schedule_barrier**: AMD instruction scheduling barriers
- **Semaphore**: Device-wide semaphore implementation
- **cp_async_bulk**: Bulk async copy synchronization

These primitives enable coordination of execution and memory operations
across threads, warps, and blocks in GPU kernels.
"""

# Semaphore operations
from .semaphore import NamedBarrierSemaphore, Semaphore

# Synchronization operations
from .sync import (
    AMDScheduleBarrierMask,
    async_copy_arrive,
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx_relaxed,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_test_wait,
    mbarrier_try_wait_parity_shared,
    named_barrier,
    named_barrier_arrive,
    schedule_barrier,
    schedule_group_barrier,
    syncwarp,
    umma_arrive_leader_cta,
    umma_arrive_peer_cta,
    s_waitcnt,
    s_waitcnt_barrier,
)
