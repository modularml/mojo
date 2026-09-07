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
"""Shared kernel components for SM100 warp-specialized matmul kernels.

This module contains common components used by all SM100 matmul kernel variants:
- WarpRole: Warp specialization roles for 4-warp kernels (MMA, Load, Scheduler, Epilogue)
- WarpRole1D1D: Warp specialization roles for 3-warp kernels (MMA, Load, Epilogue)
- KernelContext: Common kernel state (election vars, CTA coords, masks)
- Barrier init helpers: compute_input_consumer_count, init_core_barriers, init_clc_barriers
- _Batched3DLayout / _to_batched_3d: Reshape 2D TileTensor to 3D (batch=1)
"""

from max.gpu import WARP_SIZE, lane_id, thread_idx
from max.gpu import warp_id as get_warp_id
from max.gpu import block_id_in_cluster
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    elect_one_sync_with_mask,
)
from std.math.uutils import udivmod, ufloordiv
from layout.tma_async import SharedMemBarrier
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    RowMajorLayout,
    TensorLayout,
    TileTensor,
    row_major,
)

from std.utils.static_tuple import StaticTuple

from .smem_types import SMemPtr, SMemArray
from .pipeline import ProducerConsumerPipeline

comptime MbarPtr = SMemPtr[SharedMemBarrier]


# =============================================================================
# WarpRole - Warp specialization roles
# =============================================================================


@fieldwise_init
struct WarpRole(TrivialRegisterPassable):
    """Warp role identifiers for SM100 warp-specialized kernel.

    Warp assignment (8 warps total = 256 threads):
    - Epilogue: warp IDs 0-3 (4 warps, 128 threads)
    - Scheduler: warp ID 4 (1 warp, 32 threads)
    - MainLoad: warp ID 5 (1 warp, 32 threads)
    - Mma: warp ID 6 (1 warp, 32 threads)
    - EpilogueLoad: warp ID 7 (1 warp, 32 threads) - loads source C for residual

    Note: When epilogue load is not needed (no residual), warp 7 exits early.
    """

    var _role: Int32

    comptime EpilogueLoad = Self(7)
    comptime Mma = Self(6)
    comptime MainLoad = Self(5)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self._role == Int32(other)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    def __ge__(self, other: Int) -> Bool:
        return self._role >= Int32(other)

    @staticmethod
    @always_inline
    def is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    def is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    def is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()

    @staticmethod
    @always_inline
    def is_scheduler() -> Bool:
        return Self.Scheduler == get_warp_id()

    @staticmethod
    @always_inline
    def is_epilogue_load() -> Bool:
        """Check if current warp is the epilogue load warp (loads source C)."""
        return Self.EpilogueLoad == get_warp_id()


# =============================================================================
# WarpRole1D1D - 3-warp specialization (no scheduler)
# =============================================================================


struct WarpRole1D1D[has_sfb: Bool = False, num_epi_warps: Int = 4](
    TrivialRegisterPassable
):
    """Warp role for 1D-1D kernels with warp specialization.

    Parameterized on `has_sfb` so the SFB TMA-load / TMEM-load warps (and the
    scheduler's warp index) compile out cleanly on the MMA_N >= 64 path, and
    on `num_epi_warps` so kernels with heavier consumer phases can grow the
    pool without affecting other kernels.

    Default layout (`has_sfb=False, num_epi_warps=4`: 224 threads with
    scheduler, MMA_N >= 64):
    - Warps 0-3 (threads 0-127): Epilogue
    - Warp 4 (threads 128-159): TMA Load
    - Warp 5 (threads 160-191): MMA
    - Warp 6 (threads 192-223): Scheduler

    Extended layout (`has_sfb=True, num_epi_warps=4`: 384 threads with
    scheduler, MMA_N < 64):
    - Warps 0-3 (threads 0-127):    Epilogue
    - Warp 4 (threads 128-159):     TMA Load (A, B, SFA)
    - Warp 5 (threads 160-191):     MMA
    - Warp 6 (threads 192-223):     SFB TMA Load
    - Warps 7-10 (threads 224-351): SFB TMEM Load
    - Warp 11 (threads 352-383):    Scheduler

    The epilogue warps being at 0..NUM_EPILOGUE_THREADS-1 is important
    because TMAStoreCoords uses `warp_id == 0` for election.

    Parameters:
        has_sfb: Whether SFB TMA-load and TMEM-load warps are present,
            enabled on the MMA_N < 64 path (defaults to `False`).
        num_epi_warps: Number of epilogue warps in the warp pool, grown
            for kernels with heavier consumer phases (defaults to 4).
    """

    comptime NUM_EPILOGUE_THREADS = Self.num_epi_warps * 32
    comptime NUM_LOAD_THREADS = 32
    comptime NUM_MMA_THREADS = 32
    comptime NUM_SFB_TMA_LOAD_THREADS = 32  # 1 warp
    comptime NUM_SFB_LOAD_THREADS = 128  # 4 warps
    comptime NUM_SCHEDULER_THREADS = 32

    comptime EPILOGUE_WARP_START = 0
    comptime LOAD_WARP_START = Self.NUM_EPILOGUE_THREADS
    comptime MMA_WARP_START = (Self.LOAD_WARP_START + Self.NUM_LOAD_THREADS)
    comptime SFB_TMA_LOAD_WARP_START = (
        Self.MMA_WARP_START + Self.NUM_MMA_THREADS
    )
    comptime SFB_LOAD_WARP_START = (
        Self.SFB_TMA_LOAD_WARP_START + Self.NUM_SFB_TMA_LOAD_THREADS
    )

    # Scheduler warp sits right after the SFB warps when they exist, otherwise
    # directly after the MMA warp. Launching the SFB warps on MMA_N >= 64 would
    # burn ~7.5K registers on idle threads, so we compile them out there.
    comptime SCHEDULER_WARP_START = (
        Self.SFB_LOAD_WARP_START
        + Self.NUM_SFB_LOAD_THREADS if Self.has_sfb else Self.SFB_TMA_LOAD_WARP_START
    )

    comptime TOTAL_THREADS = (
        Self.NUM_EPILOGUE_THREADS + Self.NUM_LOAD_THREADS + Self.NUM_MMA_THREADS
    )
    comptime TOTAL_THREADS_WITH_SFB = (
        Self.TOTAL_THREADS
        + Self.NUM_SFB_TMA_LOAD_THREADS
        + Self.NUM_SFB_LOAD_THREADS
    )
    comptime TOTAL_THREADS_WITH_SCHED = (
        Self.SCHEDULER_WARP_START + Self.NUM_SCHEDULER_THREADS
    )

    @staticmethod
    @always_inline
    def is_epilogue() -> Bool:
        """Returns True if current thread is in an epilogue warp (warps 0-3)."""
        return thread_idx.x < Self.LOAD_WARP_START

    @staticmethod
    @always_inline
    def is_load() -> Bool:
        """Returns True if current thread is in the TMA load warp (warp 4)."""
        return (
            thread_idx.x >= Self.LOAD_WARP_START
            and thread_idx.x < Self.MMA_WARP_START
        )

    @staticmethod
    @always_inline
    def is_mma() -> Bool:
        """Returns True if current thread is in the MMA warp (warp 5)."""
        return (
            thread_idx.x >= Self.MMA_WARP_START
            and thread_idx.x < Self.SFB_TMA_LOAD_WARP_START
        )

    @staticmethod
    @always_inline
    def is_sfb_tma_load() -> Bool:
        """Returns True if current thread is in the SFB TMA load warp (warp 6).

        Only meaningful when `has_sfb` (i.e. MMA_N < 64). Callers gate this
        behind `comptime if Self.MMA_N < 64` so the check is unreachable on
        the no-SFB path, where the same threads host the scheduler warp.
        """
        return (
            thread_idx.x >= Self.SFB_TMA_LOAD_WARP_START
            and thread_idx.x < Self.SFB_LOAD_WARP_START
        )

    @staticmethod
    @always_inline
    def is_sfb_load() -> Bool:
        """Returns True if current thread is in an SFB TMEM load warp (warps 7-10).

        Only meaningful when `has_sfb` (i.e. MMA_N < 64); callers gate the
        check with `comptime if Self.MMA_N < 64`.
        """
        return (
            thread_idx.x >= Self.SFB_LOAD_WARP_START
            and thread_idx.x < Self.SCHEDULER_WARP_START
        )

    @staticmethod
    @always_inline
    def is_scheduler() -> Bool:
        """Returns True if current thread is in the scheduler warp.

        Scheduler = warp 6 when `has_sfb = False`, else warp 11. The scheduler
        warp precomputes tile info into SMEM for consumer warps.
        """
        return thread_idx.x >= Self.SCHEDULER_WARP_START

    @staticmethod
    @always_inline
    def epilogue_role_index() -> Int:
        """Returns this thread's warp index WITHIN the epilogue pool.

        Role-relative (`0 .. num_epi_warps - 1`), so callers electing
        "the first epilogue warp" no longer have to know where the pool
        sits in the block. Only meaningful when `is_epilogue()`.
        """
        return ufloordiv(thread_idx.x - Self.EPILOGUE_WARP_START, WARP_SIZE)

    @staticmethod
    @always_inline
    def is_epilogue_leader() -> Bool:
        """Returns True for the single leader thread of the epilogue pool.

        This is the election callers mean when they write
        `thread_idx.x == 0`: lane 0 of the pool's first warp. Writing it
        as a role predicate keeps the election correct if the pool ever
        stops starting at thread 0.
        """
        return (
            Self.is_epilogue()
            and Self.epilogue_role_index() == 0
            and lane_id() == 0
        )


# =============================================================================
# KernelContext - Common state for kernel entry points
# =============================================================================


struct KernelContext[
    num_clc_pipeline_stages: Int,
    cta_group: Int,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](Copyable, Movable):
    """Shared kernel state: election vars, CTA coords, multicast masks, pipeline states.

    Parameters:
        num_clc_pipeline_stages: Number of CLC pipeline stages managed by
            the scheduler work iterator.
        cta_group: Number of CTAs that cooperate as a single group, either
            1 or 2 for dual-CTA (2SM) mode.
        CLUSTER_M: Number of CTAs in the cluster along the M dimension.
        CLUSTER_N: Number of CTAs in the cluster along the N dimension.
    """

    # ===== Election Variables =====
    var elect_one_warp: Bool
    var elect_one_thread: Bool
    var elect_one_cta: Bool
    var is_first_cta_in_cluster: Bool
    var warp_id: UInt32

    # ===== CTA Coordinates =====
    var rank_m: Int
    var rank_n: Int
    var peer_cta_coord: Tuple[Int, Int, Int]

    # ===== Multicast Masks =====
    var a_multicast_mask: UInt16
    var b_multicast_mask: UInt16
    var mma_complete_mask: Int

    # Note: Pipeline states (producer and consumer) are now managed by
    # SchedulerWorkIterator and WorkIterator respectively.

    # ===== TMEM Pointer =====
    comptime TmemAddrArray = SMemArray[UInt32, 1]
    var ptr_tmem_addr: SMemPtr[UInt32]

    @always_inline
    def __init__(out self, ptr_tmem_addr: SMemPtr[UInt32]):
        """Initialize context from TMEM pointer; computes all derived state.

        Args:
            ptr_tmem_addr: Shared-memory pointer to the TMEM address used by
                the epilogue warps.
        """
        # Election variables
        self.warp_id = UInt32(get_warp_id())
        self.elect_one_warp = self.warp_id == 0
        self.elect_one_thread = elect_one_sync_with_mask()
        self.elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )
        self.is_first_cta_in_cluster = block_rank_in_cluster() == 0

        # CTA coordinates
        self.rank_m = block_id_in_cluster.x
        self.rank_n = block_id_in_cluster.y

        # Peer CTA coordinate: (peer_id, mma_coord_m, mma_coord_n)
        var cta_quotient, cta_remainder = udivmod(self.rank_m, Self.cta_group)
        self.peer_cta_coord = (
            cta_remainder,
            cta_quotient,
            self.rank_n,
        )

        # Compute multicast masks
        self.a_multicast_mask = 0x0
        self.b_multicast_mask = 0x0

        comptime for i in range(Self.CLUSTER_N):
            self.a_multicast_mask |= UInt16(1 << (i * Self.CLUSTER_M))

        comptime for i in range(Self.CLUSTER_M // Self.cta_group):
            self.b_multicast_mask |= UInt16(1 << (i * Self.cta_group))

        self.a_multicast_mask <<= UInt16(self.rank_m)
        self.b_multicast_mask <<= UInt16(self.peer_cta_coord[0])
        self.b_multicast_mask <<= UInt16(self.rank_n * Self.CLUSTER_M)

        # MMA completion mask for barrier synchronization
        # For 2SM: peer is the other CTA in the cluster (XOR with 1)
        var self_mask = 1 << Int(block_rank_in_cluster())
        var peer_rank = (
            block_rank_in_cluster() ^ 1 if Self.cta_group
            == 2 else block_rank_in_cluster()
        )
        var peer_mask = 1 << Int(peer_rank)
        self.mma_complete_mask = self_mask | peer_mask

        # TMEM pointer
        self.ptr_tmem_addr = ptr_tmem_addr

    @always_inline
    def __init__(out self, tmem_addr: Self.TmemAddrArray):
        """Initialize context from typed TMEM address array.

        Args:
            tmem_addr: Typed TMEM address array to initialize the context
                from.
        """
        self = Self(tmem_addr.ptr)


# =============================================================================
# TMA tile dimension and barrier count helpers
# =============================================================================


@always_inline
def compute_tma_tile_dims[
    BM: Int,
    BN: Int,
    MMA_M: Int,
    OutputM: Int,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
    cta_group: Int,
    AB_swapped: Bool = False,
]() -> StaticTuple[Int, 3]:
    """Compute TMA tile dimensions (a_tile_dim0, b_tile_dim0, c_tile_dim0).

    Parameters:
        BM: Block size along the M dimension of the matmul tile.
        BN: Block size along the N dimension of the matmul tile.
        MMA_M: M dimension of the MMA instruction shape, used to select
            the output tile strategy.
        OutputM: M dimension of the output tensor, used as the C tile
            size when the MMA shape or CTA group allows it.
        CLUSTER_M: Number of CTAs in the cluster along the M dimension.
        CLUSTER_N: Number of CTAs in the cluster along the N dimension.
        cta_group: Number of cooperating CTAs in a group, either 1 or 2
            for dual-CTA mode.
        AB_swapped: Whether the A and B operands are swapped, selecting
            the full-output tile path (defaults to `False`).

    Returns:
        StaticTuple of (a_tile_dim0, b_tile_dim0, c_tile_dim0).
    """
    comptime a_tile_dim0 = BM // CLUSTER_N
    comptime b_tile_dim0 = BN // (CLUSTER_M // cta_group)
    comptime c_tile_dim0 = OutputM if (
        MMA_M == 256 or cta_group == 1 or AB_swapped
    ) else 64
    return StaticTuple[Int, 3](a_tile_dim0, b_tile_dim0, c_tile_dim0)


@always_inline
def compute_clc_barrier_counts[
    SCHEDULER_THREADS: Int,
    TMA_LOAD_THREADS: Int,
    MMA_THREADS: Int,
    EPILOGUE_THREADS: Int,
    CLUSTER_SIZE: Int,
    cta_group: Int,
]() -> StaticTuple[Int, 4]:
    """Compute CLC barrier arrival counts.

    Parameters:
        SCHEDULER_THREADS: Number of scheduler threads.
        TMA_LOAD_THREADS: Number of TMA load threads.
        MMA_THREADS: Number of MMA threads.
        EPILOGUE_THREADS: Number of epilogue threads.
        CLUSTER_SIZE: Number of CTAs in the cluster.
        cta_group: Number of cooperating CTAs in a group, either 1 or 2.

    Returns:
        StaticTuple of (producer, consumer, throttle_producer, throttle_consumer).
    """
    return StaticTuple[Int, 4](
        1,  # clc_producer_arv_count
        SCHEDULER_THREADS
        + CLUSTER_SIZE
        * (
            TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
        ),  # clc_consumer_arv_count
        TMA_LOAD_THREADS,  # clc_throttle_producer_arv_count
        SCHEDULER_THREADS,  # clc_throttle_consumer_arv_count
    )


@always_inline
def compute_accum_barrier_counts[
    EPILOGUE_THREADS: Int,
    cta_group: Int,
]() -> StaticTuple[Int, 2]:
    """Compute accumulator pipeline barrier arrival counts.

    Parameters:
        EPILOGUE_THREADS: Number of epilogue threads per CTA.
        cta_group: Number of cooperating CTAs in a group, either 1 or 2.

    Returns:
        StaticTuple of (producer_arv_count, consumer_arv_count).
    """
    return StaticTuple[Int, 2](
        1,  # accum_pipeline_producer_arv_count (MMA warp via mma_arrive)
        cta_group * EPILOGUE_THREADS,  # accum_pipeline_consumer_arv_count
    )


# =============================================================================
# Barrier initialization helpers
# =============================================================================


@always_inline
def compute_input_consumer_count[
    CLUSTER_M: Int,
    CLUSTER_N: Int,
    cta_group: Int,
    CLUSTER_SIZE: Int = 0,
    epilogue_threads: Int = 0,
]() -> Int:
    """Compute input pipeline barrier consumer count.

    For standard kernels, consumers are the MMA warps across the cluster.
    For blockwise FP8 kernels, epilogue warps also consume input tiles
    (A-scales), so pass CLUSTER_SIZE and epilogue_threads to include them.

    Parameters:
        CLUSTER_M: Number of CTAs in the cluster along the M dimension.
        CLUSTER_N: Number of CTAs in the cluster along the N dimension.
        cta_group: Number of cooperating CTAs in a group, either 1 or 2.
        CLUSTER_SIZE: Number of CTAs in the cluster (defaults to 0).
        epilogue_threads: Number of epilogue threads that consume input
            tiles, included when greater than 0 (defaults to 0).
    """
    comptime base = CLUSTER_M // cta_group + CLUSTER_N - 1

    comptime if epilogue_threads > 0:
        return base + CLUSTER_SIZE * (epilogue_threads // 32)
    else:
        return base


@always_inline
def init_core_barriers[
    num_input_stages: Int,
    num_accum_stages: Int,
](
    input_barriers_ptr: MbarPtr,
    input_consumer_count: Int32,
    accum_barriers_ptr: MbarPtr,
    accum_producer_arv_count: Int32,
    accum_consumer_arv_count: Int32,
    tmem_dealloc_ptr: MbarPtr,
    tmem_dealloc_thread_count: Int32,
):
    """Initialize input, output, and TMEM deallocation barriers.

    Called inside the elect_one_warp && elect_one_thread guard.
    Handles the three barrier init steps shared by all SM100 kernels.

    Parameters:
        num_input_stages: Number of input pipeline stages to initialize
            barriers for.
        num_accum_stages: Number of accumulator pipeline stages to
            initialize barriers for.

    Args:
        input_barriers_ptr: Shared-memory pointer to the input pipeline
            barriers.
        input_consumer_count: Number of consumer arrivals on each input
            barrier.
        accum_barriers_ptr: Shared-memory pointer to the accumulator
            pipeline barriers.
        accum_producer_arv_count: Number of producer arrivals on each
            accumulator barrier.
        accum_consumer_arv_count: Number of consumer arrivals on each
            accumulator barrier.
        tmem_dealloc_ptr: Shared-memory pointer to the TMEM deallocation
            barrier.
        tmem_dealloc_thread_count: Number of threads that arrive on the
            TMEM deallocation barrier.
    """
    ProducerConsumerPipeline[num_input_stages](input_barriers_ptr).init_mbars(
        Int32(1), input_consumer_count
    )
    ProducerConsumerPipeline[num_accum_stages](accum_barriers_ptr).init_mbars(
        accum_producer_arv_count, accum_consumer_arv_count
    )
    tmem_dealloc_ptr[].init(tmem_dealloc_thread_count)


@always_inline
def init_clc_barriers[
    num_clc_stages: Int
](
    clc_full_ptr: MbarPtr,
    clc_empty_ptr: MbarPtr,
    clc_producer_arv_count: Int32,
    clc_consumer_arv_count: Int32,
):
    """Initialize CLC full/empty barrier pairs.

    Called inside the elect_one_warp && elect_one_thread guard for
    CLC-enabled kernels (default, block_scaled, blockwise_fp8, grouped_2sm).

    Parameters:
        num_clc_stages: Number of CLC pipeline stages to initialize
            barriers for.

    Args:
        clc_full_ptr: Shared-memory pointer to the CLC full barriers,
            one per pipeline stage.
        clc_empty_ptr: Shared-memory pointer to the CLC empty barriers,
            one per pipeline stage.
        clc_producer_arv_count: Number of arrivals the producer makes on
            each CLC full barrier.
        clc_consumer_arv_count: Number of arrivals the consumer makes on
            each CLC empty barrier.
    """
    comptime for i in range(num_clc_stages):
        clc_full_ptr[i].init(clc_producer_arv_count)
        clc_empty_ptr[i].init(clc_consumer_arv_count)


# =============================================================================
# _Batched3DLayout / _to_batched_3d - 2D → 3D TileTensor reshape
# =============================================================================


comptime _Batched3DLayout[L: TensorLayout] = RowMajorLayout[
    *Coord[ComptimeInt[1], L._shape_types[0], L._shape_types[1]].element_types,
]
"""3D batched layout from a 2D layout: prepend batch=1, preserve shape types."""


def _to_batched_3d(
    tensor: TileTensor,
) -> tensor.ViewType[_Batched3DLayout[type_of(tensor).LayoutType]]:
    """Reshape 2D TileTensor to 3D by prepending batch=1: (M, K) -> (1, M, K).

    The input must be rank 2. Shape types (static/dynamic) are preserved.
    """
    comptime L = type_of(tensor).LayoutType
    comptime assert L.rank == 2, "expected rank-2 TileTensor"
    return tensor.reshape(
        row_major(
            Coord(
                Idx[1],
                tensor.layout.shape[0](),
                tensor.layout.shape[1](),
            )
        )
    )
