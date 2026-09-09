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
"""Barrier abstractions for SM100 structured matmul kernels.

This module provides type-safe wrappers around low-level barrier primitives,
improving code readability and reducing error potential.
"""

from max.gpu.sync import named_barrier, named_barrier_arrive
from layout.tma_async import SharedMemBarrier

from .smem_types import SMemArray


# =============================================================================
# SmemBarriers - Composable barrier storage for SMEM structs
# =============================================================================


struct SmemBarriers[
    num_group_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_clc_pipeline_stages: Int,
]:
    """Composable barrier storage for SM100 matmul SMEM structs.

    This struct consolidates all barrier-related storage and accessors,
    enabling code reuse across MatmulSmem, BlockScaledSmem, and
    BlockwiseFP8Smem through composition.

    Parameters:
        num_group_pipeline_stages: Number of K-group pipeline stages.
        num_accum_pipeline_stages: Number of accumulator pipeline stages.
        num_clc_pipeline_stages: Number of CLC pipeline stages.

    Usage:
        Compose this struct into SMEM structs and delegate accessors:

        ```
        struct MySmem[...]:
            var barriers: SmemBarriers[num_group, num_accum, num_clc]

            def input_barriers(ref[AddressSpace.SHARED] self):
                return self.barriers.input_barriers()
        ```
    """

    # ========== Type Aliases ==========
    comptime InputBarriers = SMemArray[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]
    comptime AccumBarriers = SMemArray[
        SharedMemBarrier, Self.num_accum_pipeline_stages * 2
    ]
    comptime ClcBarriers = SMemArray[
        SharedMemBarrier, Self.num_clc_pipeline_stages
    ]
    comptime ClcThrottleBarriers = SMemArray[
        SharedMemBarrier, Self.num_clc_pipeline_stages * 2
    ]
    comptime ClcResponse = SMemArray[UInt128, Self.num_clc_pipeline_stages]
    comptime TmemDealloc = SMemArray[SharedMemBarrier, 1]
    comptime TmemAddr = SMemArray[UInt32, 1]

    # ========== Storage Fields ==========
    var input_barriers_storage: Self.InputBarriers.Storage
    var accum_barriers_storage: Self.AccumBarriers.Storage
    var clc_full_storage: Self.ClcBarriers.Storage
    var clc_empty_storage: Self.ClcBarriers.Storage
    var clc_throttle_storage: Self.ClcThrottleBarriers.Storage
    var clc_response_storage: Self.ClcResponse.Storage
    var tmem_dealloc_storage: Self.TmemDealloc.Storage
    var tmem_addr_storage: Self.TmemAddr.Storage

    # ========== Accessors ==========
    @always_inline
    def input_barriers(ref[AddressSpace.SHARED] self) -> Self.InputBarriers:
        """Returns input tile pipeline barriers (2 per group stage)."""
        return Self.InputBarriers(self.input_barriers_storage)

    @always_inline
    def accum_barriers(ref[AddressSpace.SHARED] self) -> Self.AccumBarriers:
        """Returns accumulator pipeline barriers (2 per accum stage)."""
        return Self.AccumBarriers(self.accum_barriers_storage)

    @always_inline
    def clc_full(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        """Returns CLC full barriers (1 per CLC stage)."""
        return Self.ClcBarriers(self.clc_full_storage)

    @always_inline
    def clc_empty(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        """Returns CLC empty barriers (1 per CLC stage)."""
        return Self.ClcBarriers(self.clc_empty_storage)

    @always_inline
    def clc_throttle(ref[AddressSpace.SHARED] self) -> Self.ClcThrottleBarriers:
        """Returns CLC throttle barriers (2 per CLC stage)."""
        return Self.ClcThrottleBarriers(self.clc_throttle_storage)

    @always_inline
    def clc_response(ref[AddressSpace.SHARED] self) -> Self.ClcResponse:
        """Returns CLC response storage (1 UInt128 per CLC stage)."""
        return Self.ClcResponse(self.clc_response_storage)

    @always_inline
    def tmem_dealloc(ref[AddressSpace.SHARED] self) -> Self.TmemDealloc:
        """Returns TMEM deallocation barrier."""
        return Self.TmemDealloc(self.tmem_dealloc_storage)

    @always_inline
    def tmem_addr(ref[AddressSpace.SHARED] self) -> Self.TmemAddr:
        """Returns TMEM address storage."""
        return Self.TmemAddr(self.tmem_addr_storage)


struct WarpGroupBarrier[num_threads: Int, barrier_id: Int = 0](
    TrivialRegisterPassable
):
    """Named barrier for warp group synchronization.

    Wraps `named_barrier` and `named_barrier_arrive` with compile-time
    thread count and barrier ID for type-safe synchronization.

    Parameters:
        num_threads: Number of threads participating in the barrier.
        barrier_id: Hardware named-barrier ID to use (defaults to 0).
    """

    @staticmethod
    @always_inline
    def arrive():
        """Signal arrival without blocking (non-blocking arrive)."""
        named_barrier_arrive[Int32(Self.num_threads)](Int32(Self.barrier_id))

    @staticmethod
    @always_inline
    def wait():
        """Block until all threads have arrived."""
        named_barrier[Int32(Self.num_threads)](Int32(Self.barrier_id))

    @staticmethod
    @always_inline
    def sync():
        """Full barrier: arrive and wait for all threads."""
        named_barrier[Int32(Self.num_threads)](Int32(Self.barrier_id))
