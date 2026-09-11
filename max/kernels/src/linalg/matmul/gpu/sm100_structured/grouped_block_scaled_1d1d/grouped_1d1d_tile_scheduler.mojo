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
"""Work scheduler for grouped 1D-1D block-scaled SM100 matmul.

Provides work iteration using offset-based addressing for the 1D-1D tensor layout.
This is a port of the TileScheduler from grouped_matmul_tile_scheduler.mojo
to the structured kernels architecture with context manager patterns.

Key characteristics:
- Uses a_offsets tensor for group boundaries (prefix sum of token counts)
- Each iteration returns (m_coord, n_coord, expert_id, expert_scale)
- 3-warp specialization (no scheduler warp)
"""


from std.math import ceildiv, divmod
from std.math.uutils import ufloordiv

from std.bit import count_trailing_zeros
from max.gpu import WARP_SIZE, block_idx, grid_dim, lane_id
import max.gpu.primitives.warp as warp
from layout import DefaultEngine, TensorEngine, TileTensor

from structured_kernels.tile_types import GMEMLayout1D

from std.utils.fast_div import FastDiv
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


# ===----------------------------------------------------------------------=== #
# Shared Work Geometry
# ===----------------------------------------------------------------------=== #


trait GroupedWorkGeometry1D1D:
    """Mapping from linear block index to tile coordinate.

    Every scheduler path built on these members enumerates the same
    tiles in the same order.
    """

    comptime cta_group_tile_shape: IndexList[2]
    """CTA-group tile extent, dynamic dimension first."""

    comptime div_dynamic_block: FastDiv[.uint32]
    """Fast divisor for the dynamic-dimension tile extent."""

    comptime num_static_dim_blocks: UInt32
    """Number of blocks along the static (N) dimension."""

    comptime OffsetsStoragePolicy: TensorEngine
    """Storage policy of the group-offsets TileTensor."""

    comptime ExpertIdsStoragePolicy: TensorEngine
    """Storage policy of the expert-IDs TileTensor."""

    comptime ExpertScalesStoragePolicy: TensorEngine
    """Storage policy of the expert-scales TileTensor."""


# ===----------------------------------------------------------------------=== #
# Work Info for 1D-1D Grouped Matmul
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct GroupedWorkInfo1D1D(TrivialRegisterPassable, Writable):
    """Work tile information for 1D-1D grouped matmul.

    Contains the coordinates and metadata for a single work tile:
    - m, n: Output tile coordinates (m is in contiguous token space)
    - group_idx: Index into active experts (for a_offsets indexing)
    - expert_id: The actual expert ID for B tensor lookup
    - is_valid_tile: Whether this tile contains valid work
    - terminate: Whether the scheduler has no more work
    """

    var m: UInt32
    var n: UInt32
    var group_idx: UInt32
    var expert_id: Int32
    var is_valid_tile: Bool
    var terminate: Bool
    var m_start: UInt32  # Expert's start offset in contiguous token space

    @always_inline
    def __init__(out self):
        self.m = 0
        self.n = 0
        self.group_idx = 0
        self.expert_id = 0
        self.is_valid_tile = False
        self.terminate = False
        self.m_start = 0

    @staticmethod
    @always_inline
    def terminal() -> Self:
        """Returns the end-of-work marker: no tile, no expert."""
        return Self(0, 0, 0, -1, False, True, 0)

    @always_inline
    def is_valid(self) -> Bool:
        """Returns True if this work tile has valid work to do."""
        return self.is_valid_tile

    @always_inline
    def is_done(self) -> Bool:
        """Returns True if the scheduler has no more work."""
        return self.terminate

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "GroupedWorkInfo1D1D(m=",
            self.m,
            ", n=",
            self.n,
            ", group_idx=",
            self.group_idx,
            ", expert_id=",
            self.expert_id,
            ", valid=",
            self.is_valid_tile,
            ", terminate=",
            self.terminate,
            ", m_start=",
            self.m_start,
            ")",
        )


# ===----------------------------------------------------------------------=== #
# Work Context for Context Manager Pattern
# ===----------------------------------------------------------------------=== #


struct GroupedWorkContext1D1D(ImplicitlyCopyable, Movable):
    """Context for current work tile, used with context manager pattern.

    Provides access to work tile info and expert scale factor.
    """

    var m_coord: UInt32
    var n_coord: UInt32
    var group_idx_val: UInt32
    var expert_id_val: Int32
    var m_start_coord: UInt32
    var expert_scale: Float32
    var m_end: UInt32  # End offset for bounds checking (exclusive upper bound)
    var is_valid_tile: Bool
    var terminate: Bool

    @always_inline
    def __init__(
        out self,
        info: GroupedWorkInfo1D1D,
        expert_scale: Float32,
        m_end: UInt32,
    ):
        self.m_coord = info.m
        self.n_coord = info.n
        self.group_idx_val = info.group_idx
        self.expert_id_val = info.expert_id
        self.m_start_coord = info.m_start
        self.expert_scale = expert_scale
        self.m_end = m_end
        self.is_valid_tile = info.is_valid_tile
        self.terminate = info.terminate

    @always_inline
    def __init__(
        out self,
        m: UInt32,
        n: UInt32,
        group_idx: UInt32,
        expert_id: Int32,
        m_start: UInt32,
        expert_scale: Float32,
        m_end: UInt32,
    ):
        self.m_coord = m
        self.n_coord = n
        self.group_idx_val = group_idx
        self.expert_id_val = expert_id
        self.m_start_coord = m_start
        self.expert_scale = expert_scale
        self.m_end = m_end
        self.is_valid_tile = True
        self.terminate = False

    @staticmethod
    @always_inline
    def terminal() -> Self:
        """Returns the context that marks the end of the work.

        `is_done()` is True and `expert_id()` is negative, so both
        termination tests agree.
        """
        return Self(GroupedWorkInfo1D1D.terminal(), Float32(1.0), UInt32(0))

    @always_inline
    def m(self) -> UInt32:
        """M coordinate in contiguous token space."""
        return self.m_coord

    @always_inline
    def m_start(self) -> UInt32:
        """Expert's start token offset in contiguous token space."""
        return self.m_start_coord

    @always_inline
    def n(self) -> UInt32:
        """N coordinate in output space."""
        return self.n_coord

    @always_inline
    def group_idx(self) -> UInt32:
        """Index into active experts list."""
        return self.group_idx_val

    @always_inline
    def expert_id(self) -> Int32:
        """Expert ID for B tensor indexing."""
        return self.expert_id_val

    @always_inline
    def is_valid(self) -> Bool:
        """Whether this tile has valid work."""
        return self.is_valid_tile

    @always_inline
    def is_done(self) -> Bool:
        """Whether the scheduler has no more work."""
        return self.terminate


# ===----------------------------------------------------------------------=== #
# Work Iterator for 1D-1D Grouped Matmul
# ===----------------------------------------------------------------------=== #


struct GroupedWorkIterator1D1D[
    static_N: Int,  # N dimension (expert output dim, static)
    tile_shape: IndexList[3],  # Block tile shape (BM, BN, BK)
    cluster: IndexList[3] = Index(1, 1, 1),
    cta_group: Int = 1,
    AB_swapped: Bool = False,
    OffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ExpertIdsEngine: TensorEngine = DefaultEngine[element_width=1],
    ExpertScalesEngine: TensorEngine = DefaultEngine[element_width=1],
](Copyable, GroupedWorkGeometry1D1D, Iterable, Iterator):
    """Work iterator for 1D-1D grouped block-scaled matmul.

    Iterates through work tiles using offset-based addressing:
    - a_offsets: Prefix sum of token counts per active expert
    - expert_ids: Mapping from active expert index to actual expert ID
    - expert_scales: Per-expert output scaling factors

    Yields only valid work tiles, skipping invalid ones internally.

    Parameters:
        static_N: Size of the N dimension, the expert output dimension of
            the B weight matrix, known at compile time.
        tile_shape: Block tile shape as `(BM, BN, BK)` controlling the
            per-CTA output tile and reduction tile sizes.
        cluster: CTA cluster shape as `(M, N, K)` for threadblock cluster
            multicast. Only the M component may exceed 1; N and K must be
            1 (defaults to `Index(1, 1, 1)`).
        cta_group: Number of CTAs cooperating per work tile. Must equal
            `cluster[0]` (defaults to 1).
        AB_swapped: Whether the A and B operands are swapped. When true,
            the M (token) dimension strides by `BN` and the N (weight)
            dimension strides by `BM`; otherwise M strides by `BM` and N
            strides by `BN` (defaults to False).
        OffsetsEngine: Engine of the group-offsets `TileTensor`.
        ExpertIdsEngine: Engine of the expert-IDs `TileTensor`.
        ExpertScalesEngine: Engine of the expert-scales `TileTensor`.

    Usage:
        for ctx in work_iter:
            process_tile(ctx)
    """

    comptime Element = GroupedWorkContext1D1D

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    # 1D TileTensor types: dynamic shape, stride 1 (flat arrays)
    comptime OffsetsTile = TileTensor[
        .uint32, GMEMLayout1D, MutAnyOrigin, Engine=Self.OffsetsEngine
    ]
    comptime ExpertIdsTile = TileTensor[
        .int32, GMEMLayout1D, MutAnyOrigin, Engine=Self.ExpertIdsEngine
    ]
    comptime ExpertScalesTile = TileTensor[
        .float32, GMEMLayout1D, MutAnyOrigin, Engine=Self.ExpertScalesEngine
    ]

    var num_active_experts: Int

    @__allow_legacy_any_origin_fields
    var group_offsets: Self.OffsetsTile

    @__allow_legacy_any_origin_fields
    var expert_ids: Self.ExpertIdsTile

    @__allow_legacy_any_origin_fields
    var expert_scales: Self.ExpertScalesTile

    # Iteration state
    var current_iter: Int32
    var current_group_idx: UInt32
    var current_dynamic_dim_cumsum: UInt32
    var block_idx_start: UInt32

    # Derived constants
    # For AB_swapped: m=tokens strides by MMA_N (=BN*cta_group),
    # n=weights strides by MMA_M (=BM*cta_group).
    # For non-swapped: m=tokens strides by BM, n=weights strides by MMA_N.
    comptime cta_group_tile_shape = Index(
        Self.tile_shape[1] * Self.cta_group,
        Self.tile_shape[0] * Self.cta_group,
    ) if Self.AB_swapped else Index(
        Self.tile_shape[0],
        Self.tile_shape[1] * Self.cta_group,
    )
    comptime div_dynamic_block = FastDiv[.uint32](
        Self.cta_group_tile_shape[0]  # M dimension is dynamic
    )
    comptime num_static_dim_blocks: UInt32 = UInt32(
        ceildiv(Self.static_N, Self.cta_group_tile_shape[1])
    )

    # Params are not trait members; alias them for
    # GroupedWorkGeometry1D1D.
    comptime OffsetsStoragePolicy = Self.OffsetsEngine
    comptime ExpertIdsStoragePolicy = Self.ExpertIdsEngine
    comptime ExpertScalesStoragePolicy = Self.ExpertScalesEngine

    @always_inline
    def __init__(
        out self,
        num_active_experts: Int,
        group_offsets: Self.OffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
    ):
        comptime assert (
            Self.cluster[1] == Self.cluster[2] == 1
        ), "Currently multicasting along non-M dimension is not supported"
        comptime assert Self.cta_group == Self.cluster[0], (
            "cta_group must be equal to cluster M size. Got cta_group = "
            + String(Self.cta_group)
            + " and cluster M size = "
            + String(Self.cluster[0])
        )

        self.num_active_experts = num_active_experts
        self.group_offsets = group_offsets
        self.expert_ids = expert_ids
        self.expert_scales = expert_scales
        self.current_iter = -1
        self.current_group_idx = 0
        self.current_dynamic_dim_cumsum = 0
        self.block_idx_start = 0

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> GroupedWorkContext1D1D:
        """Return next valid work tile, skipping invalid ones.

        Raises:
            StopIteration: When all work is done.
        """
        while True:
            var ctx = self.next()
            if ctx.is_done():
                raise StopIteration()
            if ctx.is_valid():
                return ctx

    @always_inline
    def next(mut self) -> GroupedWorkContext1D1D:
        """Fetch next work tile and return context with work info and scale."""
        var info, m_end = self._fetch_next_work()
        var expert_scale: Float32 = 1.0
        if info.is_valid():
            expert_scale = self.expert_scales[Int(info.expert_id)][0]
        return GroupedWorkContext1D1D(info, expert_scale, m_end)

    @always_inline
    def _fetch_next_work(mut self) -> Tuple[GroupedWorkInfo1D1D, UInt32]:
        """Internal method to compute next work tile."""
        self.current_iter += 1
        # Normalize by cta_group so all CTAs in a cluster get the same
        # work tile. For cta_group==1 this is a no-op.
        var next_block_idx = UInt32(self.current_iter) * UInt32(
            ufloordiv(grid_dim.x, Self.cta_group)
        ) + UInt32(ufloordiv(block_idx.x, Self.cta_group))
        var start_idx = self.group_offsets[Int(self.current_group_idx)][0]
        var end_idx: UInt32 = 0
        var num_dynamic_dim_blocks: UInt32 = 0
        var current_dynamic_dim: UInt32 = 0

        # Advance to the correct group
        while True:
            if self.current_group_idx >= UInt32(self.num_active_experts):
                # Finished all groups
                return (GroupedWorkInfo1D1D.terminal(), UInt32(0))

            end_idx = self.group_offsets[Int(self.current_group_idx + 1)][0]

            current_dynamic_dim = end_idx - start_idx

            # Fast-skip inactive experts (expert_id < 0) and groups with
            # zero tokens. No A, B, or scale-factor loads should happen.
            var group_expert_id = self.expert_ids[Int(self.current_group_idx)][
                0
            ]
            if group_expert_id < 0 or current_dynamic_dim <= 0:
                self.current_group_idx += 1
                start_idx = end_idx
                continue
            num_dynamic_dim_blocks = UInt32(
                rebind[Scalar[Self.div_dynamic_block.uint_type]](
                    current_dynamic_dim
                    + UInt32(Self.cta_group_tile_shape[0] - 1)
                )
                / Self.div_dynamic_block
            )
            var current_dynamic_dim_block_cumsum = (
                self.current_dynamic_dim_cumsum + num_dynamic_dim_blocks
            )
            var current_dynamic_dim_block_idx_start = (
                current_dynamic_dim_block_cumsum * Self.num_static_dim_blocks
            )
            if next_block_idx < current_dynamic_dim_block_idx_start:
                break
            self.current_group_idx += 1
            self.current_dynamic_dim_cumsum = current_dynamic_dim_block_cumsum
            self.block_idx_start = current_dynamic_dim_block_idx_start
            start_idx = end_idx

        var group_local_block_idx = next_block_idx - self.block_idx_start
        var is_valid = (
            group_local_block_idx
            < num_dynamic_dim_blocks * Self.num_static_dim_blocks
        )
        if not is_valid:
            return (
                GroupedWorkInfo1D1D(
                    0, 0, self.current_group_idx, 0, False, False, 0
                ),
                end_idx,
            )

        # Get expert_id for this group
        var expert_id = self.expert_ids[Int(self.current_group_idx)][0]

        # Row-major block order: M first, then N. The divisor is a
        # runtime value used once, so a FastDiv costs more than the
        # divmod it saves.
        var m_block_idx = group_local_block_idx % num_dynamic_dim_blocks
        var n_block_idx = group_local_block_idx / num_dynamic_dim_blocks

        # Compute actual coordinates
        # M is in contiguous token space, offset by start_idx
        var m = m_block_idx * UInt32(Self.cta_group_tile_shape[0]) + start_idx
        var n = n_block_idx * UInt32(Self.cta_group_tile_shape[1])

        return (
            GroupedWorkInfo1D1D(
                m,
                n,
                self.current_group_idx,
                expert_id,
                True,
                False,
                start_idx,
            ),
            end_idx,
        )

    @always_inline
    def current_expert_id(self) -> Int32:
        """Get the expert ID for the current group."""
        return self.expert_ids[Int(self.current_group_idx)][0]


# ===----------------------------------------------------------------------=== #
# Warp-Cooperative Work Lookup for 1D-1D Grouped Matmul
# ===----------------------------------------------------------------------=== #


struct GroupedWorkLookup1D1D[
    Iter: GroupedWorkGeometry1D1D,
    groups_per_lane: Int = 8,
    walk_cache_slots: Int = 0,
]:
    """Warp-cooperative work-tile lookup for 1D-1D grouped matmul.

    Resolves a linear block index to its group in constant time. Each
    lane keeps a contiguous segment of `groups_per_lane` slots in
    registers: token offsets, expert IDs, scales and M-block counts. A
    warp prefix sum makes the segment bases warp-wide, so a lookup is a
    ballot, a register scan on the owning lane, and a shuffle.

    Every method is warp-collective: all 32 lanes call it together with
    the same arguments. At most `MAX_GROUPS` slots are addressable;
    priming past that is a precondition failure.

    Parameters:
        Iter: Work iterator whose block geometry this lookup mirrors.
        groups_per_lane: Slots held by each lane. 8 covers 256 slots
            across a warp, the largest per-rank expert count this
            kernel serves.
        walk_cache_slots: Slot count at or below which the scan runs
            from SMEM. `can_handle` declines those launches.
    """

    comptime GROUPS_PER_LANE: Int = Self.groups_per_lane
    comptime MAX_GROUPS: Int = Self.GROUPS_PER_LANE * WARP_SIZE

    # Sparsity threshold in tokens per pinned slot, and the tile depth
    # in k-iterations below which sparsity stops mattering.
    comptime LOOKUP_TOKENS_PER_SLOT_NUM: Int = 2
    comptime LOOKUP_TOKENS_PER_SLOT_DEN: Int = 5
    comptime LOOKUP_MAX_K_ITERS: Int = 4

    # Block geometry and storage types come from `Iter`.
    comptime cta_group_tile_shape = Self.Iter.cta_group_tile_shape
    comptime div_dynamic_block = Self.Iter.div_dynamic_block
    comptime num_static_dim_blocks = Self.Iter.num_static_dim_blocks

    comptime OffsetsTile = TileTensor[
        .uint32,
        GMEMLayout1D,
        MutAnyOrigin,
        Engine=Self.Iter.OffsetsStoragePolicy,
    ]
    comptime ExpertIdsTile = TileTensor[
        .int32,
        GMEMLayout1D,
        MutAnyOrigin,
        Engine=Self.Iter.ExpertIdsStoragePolicy,
    ]
    comptime ExpertScalesTile = TileTensor[
        .float32,
        GMEMLayout1D,
        MutAnyOrigin,
        Engine=Self.Iter.ExpertScalesStoragePolicy,
    ]

    var g0: UInt32
    """First group slot of this lane's segment."""

    var cum: StaticTuple[UInt32, Self.GROUPS_PER_LANE + 1]
    """M-block prefix before each of this lane's segment boundaries."""

    var starts: StaticTuple[UInt32, Self.GROUPS_PER_LANE + 1]
    """Token start offset of each group in this lane's segment."""

    var mb: StaticTuple[UInt32, Self.GROUPS_PER_LANE]
    """M-block count of each group in this lane's segment (0 if empty)."""

    var eids: StaticTuple[Int32, Self.GROUPS_PER_LANE]
    """Expert ID of each group in this lane's segment (-1 if unused)."""

    var scales: StaticTuple[Float32, Self.GROUPS_PER_LANE]
    """Expert scale of each group in this lane's segment (1.0 if unused)."""

    @staticmethod
    @always_inline
    def needs_total_m(num_active_experts: Int, num_k_iters: Int) -> Bool:
        """Whether `can_handle` reads `total_m` for this launch.

        Only the density term needs a device-side value. When this is
        False the caller may pass `total_m = 0` and skip the load.

        Args:
            num_active_experts: Number of group slots.
            num_k_iters: K-tiles each work tile contracts over.

        Returns:
            True when the density term decides this launch.
        """
        return (
            Self.walk_cache_slots < num_active_experts <= Self.MAX_GROUPS
            and num_k_iters > Self.LOOKUP_MAX_K_ITERS
        )

    @staticmethod
    @always_inline
    def can_handle(
        num_active_experts: Int, total_m: UInt32, num_k_iters: Int
    ) -> Bool:
        """Whether the lookup is the cheaper path for this launch.

        Priming pays for itself on the empty slots the lookup skips, so
        it needs a sparse launch or a tile too thin in K to hide a scan
        behind. Occupied slot count is not visible here, so tokens per
        pinned slot stands in for it. An imbalanced launch defeats that
        proxy and loses a win. Both paths enumerate the same tiles, so
        correctness holds either way.

        Thresholds come from B200 measurements over 128 and 256 pinned
        slots: the lookup wins at any depth at or below 0.4 tokens per
        slot, and at any density at or below 4 k-iterations. Both
        curves are shallow, so a misjudged launch costs a few percent.
        Launches at or below `walk_cache_slots` are refused at any
        sparsity: that scan runs from SMEM with no dependent loads to
        remove.

        Args:
            num_active_experts: Number of group slots.
            total_m: Token total across all slots, `group_offsets[n]`.
                May be 0 when `needs_total_m` is False.
            num_k_iters: K-tiles each work tile contracts over.

        Returns:
            True when the lookup should produce this launch's work.
        """
        var sparse = (
            Int(total_m) * Self.LOOKUP_TOKENS_PER_SLOT_DEN
            < num_active_experts * Self.LOOKUP_TOKENS_PER_SLOT_NUM
        )
        var thin_tile = num_k_iters <= Self.LOOKUP_MAX_K_ITERS
        return (
            Self.walk_cache_slots < num_active_experts <= Self.MAX_GROUPS
            and (sparse or thin_tile)
        )

    @always_inline
    def __init__(
        out self,
        num_active_experts: Int,
        group_offsets: Self.OffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
    ):
        """Prime this lane's share of the warp-wide block prefix.

        Warp-collective. `num_active_experts` must fit `MAX_GROUPS`;
        `can_handle` tests that.

        Args:
            num_active_experts: Number of group slots.
            group_offsets: Prefix sum of token counts, `[slots + 1]`.
            expert_ids: Expert ID per slot, negative for unused slots.
            expert_scales: Output scale per expert ID.
        """
        debug_assert(
            ceildiv(num_active_experts, WARP_SIZE) <= Self.GROUPS_PER_LANE,
            "primed past lookup capacity; call can_handle() first",
        )

        self.cum = StaticTuple[UInt32, Self.GROUPS_PER_LANE + 1]()
        self.starts = StaticTuple[UInt32, Self.GROUPS_PER_LANE + 1](fill=0)
        self.mb = StaticTuple[UInt32, Self.GROUPS_PER_LANE]()
        self.eids = StaticTuple[Int32, Self.GROUPS_PER_LANE]()
        self.scales = StaticTuple[Float32, Self.GROUPS_PER_LANE]()

        # Shortest segment that still covers every slot across the
        # warp, so a smaller launch primes fewer slots per lane.
        var depth = ceildiv(num_active_experts, WARP_SIZE)
        var g0 = Int(lane_id()) * depth
        self.g0 = UInt32(g0)

        # A segment running past the last slot clamps to the final
        # offset. Such slots take zero blocks and own no work.
        comptime for i in range(Self.GROUPS_PER_LANE + 1):
            if i <= depth:
                self.starts[i] = group_offsets[min(g0 + i, num_active_experts)][
                    0
                ]

        var seg_blocks: UInt32 = 0
        comptime for i in range(Self.GROUPS_PER_LANE):
            var eid: Int32 = -1
            var group_blocks: UInt32 = 0
            var scale: Float32 = 1.0
            if i < depth and g0 + i < num_active_experts:
                eid = expert_ids[g0 + i][0]
                var tokens = self.starts[i + 1] - self.starts[i]
                if eid >= 0 and tokens > 0:
                    group_blocks = ceildiv(
                        tokens, UInt32(Self.cta_group_tile_shape[0])
                    )
                    scale = expert_scales[Int(eid)][0]
            self.eids[i] = eid
            self.scales[i] = scale
            self.mb[i] = group_blocks
            self.cum[i] = seg_blocks
            seg_blocks += group_blocks
        self.cum[Self.GROUPS_PER_LANE] = seg_blocks

        # Exclusive warp prefix sum: per-lane boundaries become
        # warp-wide ones.
        var inclusive = warp.prefix_sum(seg_blocks)
        var base = inclusive - seg_blocks
        comptime for i in range(Self.GROUPS_PER_LANE + 1):
            self.cum[i] += base

    @always_inline
    def lookup(self, nbi: UInt32) -> GroupedWorkContext1D1D:
        """Return the work context for linear block index `nbi`.

        Warp-collective; every lane passes the same `nbi`. An index at
        or past the end of the work yields the terminal context.

        Args:
            nbi: Linear block index: m-major within a group, groups in
                slot order, empty slots absent.

        Returns:
            The work context for that block.
        """
        # The first lane whose prefix passes `nbi` owns the block. An
        # empty segment leaves the prefix flat, so it never wins.
        var mask = warp.vote[.uint32](
            self.cum[Self.GROUPS_PER_LANE] * Self.num_static_dim_blocks > nbi
        )
        if mask == 0:
            return GroupedWorkContext1D1D.terminal()
        var owner = count_trailing_zeros(mask)

        var m: UInt32 = 0
        var n: UInt32 = 0
        var group_idx: UInt32 = 0
        var expert_id: Int32 = -1
        var m_start: UInt32 = 0
        var m_end: UInt32 = 0
        var scale: Float32 = 1.0

        if Int(lane_id()) == Int(owner):
            comptime for i in range(Self.GROUPS_PER_LANE):
                if (
                    self.mb[i] > 0
                    and nbi
                    < (self.cum[i] + self.mb[i]) * Self.num_static_dim_blocks
                ):
                    var local = nbi - self.cum[i] * Self.num_static_dim_blocks
                    var n_blk, m_blk = divmod(local, self.mb[i])
                    m = (
                        m_blk * UInt32(Self.cta_group_tile_shape[0])
                        + self.starts[i]
                    )
                    n = n_blk * UInt32(Self.cta_group_tile_shape[1])
                    group_idx = self.g0 + UInt32(i)
                    expert_id = self.eids[i]
                    m_start = self.starts[i]
                    m_end = self.starts[i + 1]
                    scale = self.scales[i]
                    break

        return GroupedWorkContext1D1D(
            warp.shuffle_idx(m, owner),
            warp.shuffle_idx(n, owner),
            warp.shuffle_idx(group_idx, owner),
            warp.shuffle_idx(expert_id, owner),
            warp.shuffle_idx(m_start, owner),
            warp.shuffle_idx(scale, owner),
            warp.shuffle_idx(m_end, owner),
        )
