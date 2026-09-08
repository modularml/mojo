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
"""Grouped tile scheduler for SM100 structured block-scaled GEMM.

This scheduler extends the SM100 TileScheduler to support grouped GEMM with
variable problem sizes per group. It uses linear tile iteration instead of CLC
(Cluster Launch Control) to map a global linear tile index to group-specific
coordinates.

Key features:
- GroupedWorkInfo: Extends WorkInfo with group_idx, k_tile_count, group_changed
- delinearize_to_group(): Maps linear tile index to group + local coordinates
- Supports variable M, N, K per group
- Compatible with dynamic tensormap updates

Usage:
    var scheduler = GroupedTileScheduler[...](problem_sizes, tile_shape)
    var work_iter = scheduler.work_iterator()
    for current in work_iter:
        if current.group_changed:
            update_tensormaps(current.group_idx)
        process_tile(current)
"""

from std.math import ceildiv
from std.math.uutils import ufloordiv

from max.gpu import block_idx, grid_dim
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.memory import fence_async_view_proxy
from layout.tma_async import PipelineState, SharedMemBarrier
from .grouped_block_scaled_matmul_kernel import _ProblemSizesTile

from std.utils.static_tuple import StaticTuple

from linalg.structuring import SMemPtr
from structured_kernels.pipeline import ProducerConsumerPipeline


# =============================================================================
# GroupedWorkInfo - Extended work info for grouped GEMM
# =============================================================================


@fieldwise_init
struct GroupedWorkInfo(
    ImplicitlyCopyable, Movable, TrivialRegisterPassable, Writable
):
    """Work info for grouped GEMM with group-specific metadata.

    Extends the base WorkInfo with:
    - group_idx: Current group index
    - k_tile_count: Number of K tiles for this group
    - group_changed: True if group changed since last tile (triggers tensormap update)
    """

    # Base coordinates (compatible with WorkInfo)
    var m: UInt32
    """M-coordinate of tile within current group."""
    var n: UInt32
    """N-coordinate of tile within current group."""
    var k_start: UInt32
    """Starting K index (always 0 for grouped GEMM)."""
    var is_valid_tile: Bool
    """Whether this work tile is valid (not OOB)."""

    # Grouped extensions
    var group_idx: UInt32
    """Current group index."""
    var k_tile_count: UInt32
    """Number of K tiles for this group."""
    var group_changed: Bool
    """True if group changed since last tile (triggers tensormap update)."""

    @always_inline
    def __init__(out self):
        """Create an invalid/empty work info."""
        self.m = 0
        self.n = 0
        self.k_start = 0
        self.is_valid_tile = False
        self.group_idx = 0
        self.k_tile_count = 0
        self.group_changed = False

    @always_inline
    def is_valid(self) -> Bool:
        """Check if this work tile is valid."""
        return self.is_valid_tile

    @always_inline
    def coord(self) -> Tuple[Int, Int]:
        """Get (m, n) tile coordinates as a tuple."""
        return (Int(self.m), Int(self.n))

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "GroupedWorkInfo(m=",
            self.m,
            ", n=",
            self.n,
            ", k_start=",
            self.k_start,
            ", valid=",
            self.is_valid_tile,
            ", group=",
            self.group_idx,
            ", k_tiles=",
            self.k_tile_count,
            ", changed=",
            self.group_changed,
            ")",
        )


# =============================================================================
# GroupedWorkIterator - Per-warp iterator for grouped GEMM
# =============================================================================


struct GroupedWorkIterator[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    cta_group: Int = 1,
](Copyable, Iterable, Iterator, RegisterPassable):
    """Per-warp work iterator for grouped GEMM using __next__-style iteration.

    This iterator traverses tiles across all groups, tracking when groups change
    to trigger tensormap updates. It uses linear iteration instead of CLC.

    For 2SM (cta_group=2), both CTAs in a cluster work on the same logical tile.
    The cluster index (block_idx.x // cta_group) is used for tile assignment,
    and advance step is grid_dim.x // cta_group (number of clusters).

    Parameters:
        tile_m: M dimension of output tiles.
        tile_n: N dimension of output tiles.
        tile_k: K dimension of input tiles.
        max_groups: Maximum number of groups.
        cta_group: Number of CTAs cooperating per tile (1 or 2 for 2SM).

    Usage:
        var work_iter = scheduler.work_iterator()
        for current in work_iter:
            if current.group_changed:
                update_tensormaps(current.group_idx)
            process_tile(current)
    """

    comptime Element = GroupedWorkInfo

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var work_info: GroupedWorkInfo
    """Current work item."""
    var linear_tile_idx: UInt32
    """Current linear tile index (across all groups)."""
    var total_tiles: UInt32
    """Total number of tiles across all groups."""
    var prev_group_idx: UInt32
    """Previous group index for detecting group changes."""

    # Group metadata (cumulative tile counts)
    var cumulative_tiles: StaticTuple[UInt32, Self.max_groups + 1]
    """Cumulative tile count at the start of each group."""
    var problem_m: StaticTuple[UInt32, Self.max_groups]
    """M dimension for each group."""
    var problem_n: StaticTuple[UInt32, Self.max_groups]
    """N dimension for each group."""
    var problem_k: StaticTuple[UInt32, Self.max_groups]
    """K dimension for each group."""
    var num_groups: UInt32
    """Number of active groups."""

    @always_inline
    def __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
        grid_size: UInt32,
    ):
        """Initialize work iterator with problem sizes.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
            grid_size: Number of blocks in the grid.
        """
        # Initialize all fields first to satisfy compiler
        self.work_info = GroupedWorkInfo()
        self.linear_tile_idx = UInt32(0)
        self.total_tiles = UInt32(0)
        self.prev_group_idx = UInt32(0)
        self.cumulative_tiles = StaticTuple[UInt32, Self.max_groups + 1]()
        self.problem_m = StaticTuple[UInt32, Self.max_groups]()
        self.problem_n = StaticTuple[UInt32, Self.max_groups]()
        self.problem_k = StaticTuple[UInt32, Self.max_groups]()
        self.num_groups = UInt32(num_groups)

        # Compute cumulative tile counts
        # Explicitly zero-initialize ALL slots to avoid stale memory issues
        comptime for i in range(Self.max_groups + 1):
            self.cumulative_tiles[i] = 0

        var cumsum: UInt32 = 0
        # cumulative_tiles[0] is already 0 from the loop above

        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var k = UInt32(Int(problem_sizes[g, 2]))

            self.problem_m[g] = m
            self.problem_n[g] = n
            self.problem_k[g] = k

            # Compute tiles for this group
            var m_tiles = ceildiv(Int(m), Self.tile_m)
            var n_tiles = ceildiv(Int(n), Self.tile_n)
            var group_tiles = UInt32(m_tiles * n_tiles)
            cumsum += group_tiles
            self.cumulative_tiles[g + 1] = cumsum

        # Initialize remaining slots
        for g in range(num_groups, Self.max_groups):
            self.problem_m[g] = 0
            self.problem_n[g] = 0
            self.problem_k[g] = 0
            self.cumulative_tiles[g + 1] = cumsum

        self.total_tiles = cumsum

        # Start at this cluster's first tile
        # For 2SM (cta_group=2), both CTAs in a cluster work on the same tile
        # Use cluster index = block_idx.x // cta_group
        self.linear_tile_idx = UInt32(ufloordiv(block_idx.x, Self.cta_group))

        # Delinearize initial position
        self.work_info = self._delinearize_to_group(self.linear_tile_idx)
        self.work_info.group_changed = True  # First tile always triggers update

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> GroupedWorkInfo:
        """Return current work item, deferring advance to next call.

        Raises:
            StopIteration: When there is no more work to process.
        """
        if not self.work_info.is_valid():
            raise StopIteration()
        var current = self.work_info

        # Advance to next tile
        self.prev_group_idx = self.work_info.group_idx
        self.linear_tile_idx += UInt32(ufloordiv(grid_dim.x, Self.cta_group))

        if self.linear_tile_idx >= self.total_tiles:
            self.work_info = GroupedWorkInfo()  # Invalid
        else:
            self.work_info = self._delinearize_to_group(self.linear_tile_idx)
            self.work_info.group_changed = (
                self.work_info.group_idx != self.prev_group_idx
            )

        return current

    @always_inline
    def _delinearize_to_group(self, linear_idx: UInt32) -> GroupedWorkInfo:
        """Map linear tile index to group + local coordinates.

        Uses binary search to find the group containing this tile index.
        """
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()

        # Binary search for group
        var lo: UInt32 = 0
        var hi: UInt32 = self.num_groups

        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < self.cumulative_tiles[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        # Local tile index within group
        var local_idx = linear_idx - self.cumulative_tiles[Int(group_idx)]

        # Get group dimensions
        var m = self.problem_m[Int(group_idx)]
        var n = self.problem_n[Int(group_idx)]
        var k = self.problem_k[Int(group_idx)]

        var m_tiles = ceildiv(Int(m), Self.tile_m)
        _ = ceildiv(Int(n), Self.tile_n)  # n_tiles used only for validation
        var k_tiles = ceildiv(Int(k), Self.tile_k)

        # Convert to M, N tile coordinates (row-major within group)
        # These are tile indices, not global coordinates
        # (load_input_tiles will multiply by BM/BN to get global coords)
        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,  # Tile index, not global coordinate
            n=n_tile,  # Tile index, not global coordinate
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=False,  # Caller sets this
        )


# =============================================================================
# GroupedTileScheduler - Main scheduler for grouped GEMM
# =============================================================================


struct GroupedTileScheduler[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    num_stages: Int = 0,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """Tile scheduler for grouped block-scaled GEMM.

    Uses linear tile iteration to map tiles across groups. Does not use CLC
    (Cluster Launch Control) since work distribution is deterministic.

    Parameters:
        tile_m: M dimension of output tiles.
        tile_n: N dimension of output tiles.
        tile_k: K dimension of input tiles.
        max_groups: Maximum number of groups.
        num_stages: Pipeline stages (0 = single wave).
        cta_group: Number of CTAs cooperating per tile (1 or 2 for 2SM).
    """

    var num_groups: Int
    """Number of active groups."""

    @__allow_legacy_any_origin_fields
    var problem_sizes: _ProblemSizesTile[Self.max_groups]
    """Problem sizes tensor (num_groups, 4) with [M, N, K, L] per group."""

    @always_inline
    def __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
    ):
        """Initialize scheduler with problem sizes.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
        """
        self.problem_sizes = problem_sizes
        self.num_groups = num_groups

    @always_inline
    def work_iterator(
        self,
    ) -> GroupedWorkIterator[
        Self.tile_m, Self.tile_n, Self.tile_k, Self.max_groups, Self.cta_group
    ]:
        """Create a per-warp work iterator.

        Each warp should create its own work iterator. The iterator owns
        work_info and cumulative tile counts internally.

        For 2SM (cta_group=2), the iterator uses cluster-based indexing.
        """
        return GroupedWorkIterator[
            Self.tile_m,
            Self.tile_n,
            Self.tile_k,
            Self.max_groups,
            Self.cta_group,
        ](
            self.problem_sizes,
            self.num_groups,
            UInt32(grid_dim.x),
        )

    @always_inline
    def total_tiles(self) -> Int:
        """Compute total number of tiles across all groups."""
        var total = 0
        for g in range(self.num_groups):
            var m = Int(self.problem_sizes[g, 0])
            var n = Int(self.problem_sizes[g, 1])
            var m_tiles = ceildiv(m, Self.tile_m)
            var n_tiles = ceildiv(n, Self.tile_n)
            total += m_tiles * n_tiles
        return total


# =============================================================================
# GroupedCLCWorkIterator - Per-warp iterator with CLC barrier support
# =============================================================================


struct GroupedCLCWorkIterator[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    num_clc_stages: Int,
    cta_group: Int = 2,
](Copyable, Iterable, Iterator, RegisterPassable):
    """Per-warp work iterator for grouped GEMM with CLC barrier support.

    This iterator combines grouped GEMM features with CLC-based synchronization
    for 2SM support. It uses CLC barriers to ensure both CTAs in a cluster
    process the same tile at the same time.

    Parameters:
        tile_m: M dimension of output tiles.
        tile_n: N dimension of output tiles.
        tile_k: K dimension of input tiles.
        max_groups: Maximum number of groups.
        num_clc_stages: Number of CLC pipeline stages for barrier-based
            synchronization.
        cta_group: Number of CTAs cooperating per tile (1 or 2 for 2SM).

    Usage:
        var work_iter = scheduler.clc_work_iterator()
        for current in work_iter:
            if current.group_changed:
                update_tensormaps(current.group_idx)
            process_tile(current)
    """

    comptime Element = GroupedWorkInfo

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime ThrottlePipeline = ProducerConsumerPipeline[Self.num_clc_stages]

    var work_info: GroupedWorkInfo
    """Current work item."""
    var consumer_state: PipelineState[Self.num_clc_stages]
    """CLC consumer pipeline state."""
    var throttle_pipeline: Self.ThrottlePipeline
    """Throttle pipeline for load/scheduler sync."""

    # CLC barrier pointers
    var full_mbar: SMemPtr[SharedMemBarrier]
    """CLC full barriers (signaled by scheduler when work is ready)."""
    var empty_mbar: SMemPtr[SharedMemBarrier]
    """CLC empty barriers (signaled by workers when done)."""
    var clc_response: SMemPtr[UInt128]
    """CLC response storage (contains work info)."""

    # Group metadata (cumulative tile counts)
    var cumulative_tiles: StaticTuple[UInt32, Self.max_groups + 1]
    """Cumulative tile count at the start of each group."""
    var problem_m: StaticTuple[UInt32, Self.max_groups]
    """M dimension for each group."""
    var problem_n: StaticTuple[UInt32, Self.max_groups]
    """N dimension for each group."""
    var problem_k: StaticTuple[UInt32, Self.max_groups]
    """K dimension for each group."""
    var num_groups: UInt32
    """Number of active groups."""
    var total_tiles: UInt32
    """Total tiles across all groups."""

    var use_clc_fetch: Bool
    """If True, __next__ waits on CLC barriers (for MMA warp)."""

    @always_inline
    def __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
        full_mbar: SMemPtr[SharedMemBarrier],
        empty_mbar: SMemPtr[SharedMemBarrier],
        clc_response: SMemPtr[UInt128],
        throttle_ptr: SMemPtr[SharedMemBarrier],
        initial_work: GroupedWorkInfo,
        use_clc_fetch: Bool = False,
    ):
        """Initialize CLC work iterator.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
            full_mbar: CLC full barrier pointer.
            empty_mbar: CLC empty barrier pointer.
            clc_response: CLC response storage pointer.
            throttle_ptr: Throttle pipeline barrier pointer.
            initial_work: Initial work item (first tile).
            use_clc_fetch: If True, __next__ waits on CLC barriers (for MMA warp).
        """
        self.work_info = initial_work
        self.use_clc_fetch = use_clc_fetch
        self.consumer_state = PipelineState[Self.num_clc_stages]()
        self.throttle_pipeline = Self.ThrottlePipeline(throttle_ptr)
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar
        self.clc_response = clc_response
        self.num_groups = UInt32(num_groups)
        self.cumulative_tiles = StaticTuple[UInt32, Self.max_groups + 1]()
        self.problem_m = StaticTuple[UInt32, Self.max_groups]()
        self.problem_n = StaticTuple[UInt32, Self.max_groups]()
        self.problem_k = StaticTuple[UInt32, Self.max_groups]()

        # Initialize cumulative tiles
        comptime for i in range(Self.max_groups + 1):
            self.cumulative_tiles[i] = 0

        var cumsum: UInt32 = 0
        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var k = UInt32(Int(problem_sizes[g, 2]))
            self.problem_m[g] = m
            self.problem_n[g] = n
            self.problem_k[g] = k
            var m_tiles = ceildiv(Int(m), Self.tile_m)
            var n_tiles = ceildiv(Int(n), Self.tile_n)
            cumsum += UInt32(m_tiles * n_tiles)
            self.cumulative_tiles[g + 1] = cumsum

        for g in range(num_groups, Self.max_groups):
            self.problem_m[g] = 0
            self.problem_n[g] = 0
            self.problem_k[g] = 0
            self.cumulative_tiles[g + 1] = cumsum

        self.total_tiles = cumsum

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> GroupedWorkInfo:
        """Return current work item and advance.

        When use_clc_fetch is True (MMA warp), waits on CLC barriers for
        synchronization. Otherwise uses simple linear advance.

        Raises:
            StopIteration: When there is no more work to process.
        """
        if not self.work_info.is_valid():
            raise StopIteration()
        var current = self.work_info
        if self.use_clc_fetch:
            self.work_info = self._fetch_next_work()
            self.consumer_state.step()
        else:
            self.work_info = self._compute_next_work()
        return current

    @always_inline
    def _fetch_next_work(self) -> GroupedWorkInfo:
        """Fetch next work item with CLC barrier synchronization.

        This is the key synchronization point - waits on CLC full barrier
        to ensure all CTAs receive the same work at the same time.

        Note: For 2SM clusters, each CTA has separate SMEM so we can't read
        the clc_response written by the scheduler (CTA 0). Instead, all CTAs
        compute work locally (they get the same result since they use the same
        formula), and use barriers just for synchronization.
        """
        # Wait for scheduler to signal work is ready
        self.full_mbar[self.consumer_state.index()].wait(
            self.consumer_state.phase()
        )

        # Compute next work locally instead of reading from CLC response
        # (CLC response is in CTA 0's SMEM, not accessible to other CTAs)
        var linear_idx = self._current_linear_idx() + UInt32(
            ufloordiv(grid_dim.x, Self.cta_group)
        )

        fence_async_view_proxy()

        # Signal that we've consumed this work item (all CTAs arrive on CTA 0)
        self.empty_mbar[self.consumer_state.index()].arrive_cluster(0)

        # Delinearize to grouped work info
        return self._delinearize_to_group(linear_idx, self.work_info.group_idx)

    @always_inline
    def _read_linear_idx_from_clc(self) -> UInt32:
        """Read linear tile index from CLC response.

        The CLC response contains: (linear_idx, 0, 0, is_valid).
        """
        var response_ptr = self.clc_response + self.consumer_state.index()
        # Read the 128-bit response and extract linear_idx (first 32 bits)
        var response = response_ptr[].cast[.uint32]()
        return response
        # Note: The working kernel uses inline assembly here, but for simplicity
        # we just cast the UInt128 to UInt32 to get the first component

    @always_inline
    def _compute_next_work(self) -> GroupedWorkInfo:
        """Compute next work item without CLC wait (for non-MMA warps)."""
        # Simple linear advance
        var linear_idx = self._current_linear_idx() + UInt32(
            ufloordiv(grid_dim.x, Self.cta_group)
        )
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()  # Invalid
        return self._delinearize_to_group(linear_idx, self.work_info.group_idx)

    @always_inline
    def _current_linear_idx(self) -> UInt32:
        """Compute current linear tile index from work_info."""
        var g = Int(self.work_info.group_idx)
        var m_tiles = ceildiv(Int(self.problem_m[g]), Self.tile_m)
        return (
            self.cumulative_tiles[g]
            + self.work_info.n * UInt32(m_tiles)
            + self.work_info.m
        )

    @always_inline
    def _delinearize_to_group(
        self, linear_idx: UInt32, prev_group_idx: UInt32
    ) -> GroupedWorkInfo:
        """Map linear tile index to group + local coordinates."""
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()

        # Binary search for group
        var lo: UInt32 = 0
        var hi: UInt32 = self.num_groups
        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < self.cumulative_tiles[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        # Local tile index within group
        var local_idx = linear_idx - self.cumulative_tiles[Int(group_idx)]

        # Get group dimensions
        var m = self.problem_m[Int(group_idx)]
        var k = self.problem_k[Int(group_idx)]
        var m_tiles = ceildiv(Int(m), Self.tile_m)
        var k_tiles = ceildiv(Int(k), Self.tile_k)

        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,
            n=n_tile,
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=(group_idx != prev_group_idx),
        )


# =============================================================================
# GroupedCLCSchedulerIterator - For scheduler warp with CLC
# =============================================================================


struct GroupedCLCSchedulerIterator[
    tile_m: Int,
    tile_n: Int,
    tile_k: Int,
    max_groups: Int,
    num_clc_stages: Int,
    cta_group: Int = 2,
](Copyable, Iterable, Iterator, RegisterPassable):
    """Scheduler warp iterator for grouped GEMM with CLC.

    The scheduler warp produces work items for other warps via CLC.
    It iterates through all tiles across all groups and signals CLC barriers.

    Parameters:
        tile_m: M dimension of output tiles.
        tile_n: N dimension of output tiles.
        tile_k: K dimension of input tiles.
        max_groups: Maximum number of groups.
        num_clc_stages: Number of CLC pipeline stages for barrier-based
            synchronization.
        cta_group: Number of CTAs cooperating per tile (1 or 2 for 2SM).

    Usage:
        var sched_iter = scheduler.scheduler_iterator()
        for _ in sched_iter:
            sched_iter.signal_and_advance()
        sched_iter.drain()
    """

    comptime Element = GroupedWorkInfo

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime ThrottlePipeline = ProducerConsumerPipeline[Self.num_clc_stages]

    var work_info: GroupedWorkInfo
    """Current work item."""
    var linear_tile_idx: UInt32
    """Current linear tile index."""
    var consumer_state: PipelineState[Self.num_clc_stages]
    var producer_state: PipelineState[Self.num_clc_stages]
    var throttle_pipeline: Self.ThrottlePipeline

    # CLC barrier pointers
    var full_mbar: SMemPtr[SharedMemBarrier]
    var empty_mbar: SMemPtr[SharedMemBarrier]
    var clc_response: SMemPtr[UInt128]

    # Group metadata
    var cumulative_tiles: StaticTuple[UInt32, Self.max_groups + 1]
    var problem_m: StaticTuple[UInt32, Self.max_groups]
    var problem_n: StaticTuple[UInt32, Self.max_groups]
    var problem_k: StaticTuple[UInt32, Self.max_groups]
    var num_groups: UInt32
    var total_tiles: UInt32
    var signal_count: UInt32
    """Number of signals sent (for pipeline fill tracking)."""

    @always_inline
    def __init__(
        out self,
        problem_sizes: _ProblemSizesTile[Self.max_groups],
        num_groups: Int,
        full_mbar: SMemPtr[SharedMemBarrier],
        empty_mbar: SMemPtr[SharedMemBarrier],
        clc_response: SMemPtr[UInt128],
        throttle_ptr: SMemPtr[SharedMemBarrier],
        initial_work: GroupedWorkInfo,
    ):
        """Initialize scheduler iterator.

        Args:
            problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
            num_groups: Number of active groups.
            full_mbar: CLC full barrier pointer.
            empty_mbar: CLC empty barrier pointer.
            clc_response: CLC response storage pointer.
            throttle_ptr: Throttle pipeline barrier pointer.
            initial_work: Initial work item (first tile).
        """
        self.work_info = initial_work
        # Each cluster starts at its own linear tile index
        # block_idx.x // cta_group gives the cluster's starting tile
        self.linear_tile_idx = UInt32(block_idx.x // Self.cta_group)
        self.consumer_state = PipelineState[Self.num_clc_stages]()
        self.producer_state = PipelineState[Self.num_clc_stages](0, 1, 0)
        self.throttle_pipeline = Self.ThrottlePipeline(throttle_ptr)
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar
        self.clc_response = clc_response
        self.num_groups = UInt32(num_groups)
        self.cumulative_tiles = StaticTuple[UInt32, Self.max_groups + 1]()
        self.problem_m = StaticTuple[UInt32, Self.max_groups]()
        self.problem_n = StaticTuple[UInt32, Self.max_groups]()
        self.problem_k = StaticTuple[UInt32, Self.max_groups]()
        self.signal_count = UInt32(0)

        # Initialize cumulative tiles
        comptime for i in range(Self.max_groups + 1):
            self.cumulative_tiles[i] = 0

        var cumsum: UInt32 = 0
        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var k = UInt32(Int(problem_sizes[g, 2]))
            self.problem_m[g] = m
            self.problem_n[g] = n
            self.problem_k[g] = k
            var m_tiles = ceildiv(Int(m), Self.tile_m)
            var n_tiles = ceildiv(Int(n), Self.tile_n)
            cumsum += UInt32(m_tiles * n_tiles)
            self.cumulative_tiles[g + 1] = cumsum

        for g in range(num_groups, Self.max_groups):
            self.problem_m[g] = 0
            self.problem_n[g] = 0
            self.problem_k[g] = 0
            self.cumulative_tiles[g + 1] = cumsum

        self.total_tiles = cumsum

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> GroupedWorkInfo:
        """Return current work item, deferring advance to next call.

        Raises:
            StopIteration: When there is no more work to process.
        """
        if not self.work_info.is_valid():
            raise StopIteration()
        var current = self.work_info

        # Advance to next tile
        var num_clusters = UInt32(ufloordiv(grid_dim.x, Self.cta_group))
        var next_linear_idx = self.linear_tile_idx + num_clusters

        if next_linear_idx >= self.total_tiles:
            self.work_info = GroupedWorkInfo()
        else:
            self.work_info = self._delinearize_to_group(
                next_linear_idx, current.group_idx
            )
        self.linear_tile_idx = next_linear_idx

        return current

    @always_inline
    def signal_and_advance(mut self):
        """Signal CLC throttle and produce next work request.

        This is called inside the work loop after processing current work.
        It signals that we've consumed the throttle and produces the next
        work item for all CTAs.

        NOTE: We skip the throttle_pipeline.consumer_signal_and_step() call
        that the hardware CLC version uses. For software CLC simulation,
        the clc_full/clc_empty barriers provide sufficient synchronization.
        The throttle pattern causes a deadlock because:
        - Scheduler waits for TMA Load via throttle full barrier
        - TMA Load waits for Scheduler via throttle empty barrier
        - Both block on first iteration since barriers start at phase 0
        """

        # Produce next work item: write linear_idx to CLC response
        # For 2SM: advance by number of clusters (each cluster processes different tiles)
        # Always signal, even for "no more work" (consumer detects via total_tiles)
        var num_clusters = UInt32(ufloordiv(grid_dim.x, Self.cta_group))
        var next_linear_idx = self.linear_tile_idx + num_clusters

        # Wait for empty signal (consumers done with previous)
        # Skip wait during pipeline fill phase (first num_clc_stages iterations)
        # to avoid deadlock - no previous data to wait for yet.
        if self.signal_count >= UInt32(Self.num_clc_stages):
            self.empty_mbar[self.producer_state.index()].wait(
                self.producer_state.phase()
            )

        # Write next linear index to CLC response (only one thread writes)
        # If next_linear_idx >= total_tiles, consumer will detect "no more work"
        # Signal full (work is ready) to all CTAs in the cluster
        # For 2SM: scheduler runs on CTA 0 but must signal all CTAs' barriers
        # via arrive_cluster so each CTA's MMA warp can proceed
        #
        # CRITICAL: Use elect_one_sync() to ensure only ONE thread in the warp
        # signals the barriers. Without this, all 32 threads would each do
        # arrive_cluster(), causing 32 arrivals on barriers expecting only 1.
        if elect_one_sync():
            var response_ptr = self.clc_response + self.producer_state.index()
            response_ptr[] = UInt128(Int(next_linear_idx))

            comptime for cta in range(Self.cta_group):
                self.full_mbar[self.producer_state.index()].arrive_cluster(
                    UInt32(cta)
                )
        self.signal_count += 1

        self.producer_state.step()

        # Update iterator state so has_work() eventually returns False
        self.linear_tile_idx = next_linear_idx
        if next_linear_idx >= self.total_tiles:
            self.work_info = GroupedWorkInfo()  # Invalid - no more work
        else:
            self.work_info = self._delinearize_to_group(
                next_linear_idx, self.work_info.group_idx
            )

    @always_inline
    def drain(mut self):
        """Drain all pending CLC requests before kernel exit.

        Only waits for slots that were actually signaled to avoid deadlock
        when workload is smaller than pipeline depth.

        Note: After signaling, producer_state has stepped to the NEXT stage.
        We need to wait on stages 0..slots_to_drain-1, not from producer_state.
        """
        # Number of slots actually used is min(signals_made, num_stages)
        var slots_to_drain = min(Int(self.signal_count), Self.num_clc_stages)

        # Drain stages starting from 0 (the first stage we used)
        # Phase alternates: stage 0 uses phase 0, stages wrap at num_stages
        for i in range(slots_to_drain):
            var stage = i % Self.num_clc_stages
            # Phase is 0 for first num_stages iterations, then 1, etc.
            var phase = UInt32(i // Self.num_clc_stages) & 1
            self.empty_mbar[stage].wait(phase)

    @always_inline
    def _delinearize_to_group(
        self, linear_idx: UInt32, prev_group_idx: UInt32
    ) -> GroupedWorkInfo:
        """Map linear tile index to group + local coordinates."""
        if linear_idx >= self.total_tiles:
            return GroupedWorkInfo()

        var lo: UInt32 = 0
        var hi: UInt32 = self.num_groups
        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < self.cumulative_tiles[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        var local_idx = linear_idx - self.cumulative_tiles[Int(group_idx)]
        var m = self.problem_m[Int(group_idx)]
        var k = self.problem_k[Int(group_idx)]
        var m_tiles = ceildiv(Int(m), Self.tile_m)
        var k_tiles = ceildiv(Int(k), Self.tile_k)

        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,
            n=n_tile,
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=(group_idx != prev_group_idx),
        )
