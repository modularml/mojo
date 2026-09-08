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
- Supports block swizzling for L2 cache efficiency
- 3-warp specialization (no scheduler warp)
"""

from std.math import ceildiv
from std.math.uutils import ufloordiv

from max.gpu import block_idx, grid_dim
from layout import DefaultEngine, TensorEngine, TileTensor

from structured_kernels.tile_types import GMEMLayout1D

from std.utils.fast_div import FastDiv
from std.utils.index import Index, IndexList


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
    swizzle: Bool = False,
    AB_swapped: Bool = False,
    OffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ExpertIdsEngine: TensorEngine = DefaultEngine[element_width=1],
    ExpertScalesEngine: TensorEngine = DefaultEngine[element_width=1],
](Copyable, Iterable, Iterator):
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
        swizzle: Whether to swizzle block iteration order for L2 cache
            reuse across CTAs (defaults to False).
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
    comptime kNum1DBlocksPerGroup: UInt32 = 16

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
                return (
                    GroupedWorkInfo1D1D(0, 0, 0, 0, False, True, 0),
                    UInt32(0),
                )

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

        # Compute swizzled block indices
        var num_n_blocks = Self.num_static_dim_blocks
        var m_block_idx, n_block_idx = self._get_swizzled_block_idx(
            num_n_blocks, group_local_block_idx, num_dynamic_dim_blocks
        )

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
    def _get_swizzled_block_idx(
        self,
        num_n_blocks: UInt32,
        _block_idx: UInt32,
        num_dynamic_dim_blocks: UInt32,
    ) -> Tuple[UInt32, UInt32]:
        """Compute swizzled (m_block_idx, n_block_idx) for L2 cache efficiency.
        """
        var primary_num_blocks = num_dynamic_dim_blocks  # M blocks
        var div_primary_num_blocks = FastDiv[.uint32](Int(primary_num_blocks))
        comptime uint_type = div_primary_num_blocks.uint_type
        var block_idx_val = rebind[Scalar[uint_type]](_block_idx)

        if not Self.swizzle:
            # Row-major order: iterate M first, then N
            return (
                UInt32(block_idx_val % div_primary_num_blocks),
                UInt32(block_idx_val / div_primary_num_blocks),
            )

        # Swizzle for better L2 usage
        var secondary_num_blocks = num_n_blocks
        var num_blocks_per_group = (
            secondary_num_blocks * Self.kNum1DBlocksPerGroup
        )
        var div_num_blocks_per_group = FastDiv[.uint32](
            Int(num_blocks_per_group)
        )
        var group_idx = UInt32(block_idx_val / div_num_blocks_per_group)
        var first_block_idx = group_idx * Self.kNum1DBlocksPerGroup
        var in_group_idx = block_idx_val % div_num_blocks_per_group
        var num_blocks_in_group = min(
            Self.kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx
        )
        var div_num_blocks_in_group = FastDiv[.uint32](Int(num_blocks_in_group))
        comptime uint_type2 = div_num_blocks_in_group.uint_type
        var m_block_idx = first_block_idx + UInt32(
            rebind[Scalar[uint_type2]](in_group_idx) % div_num_blocks_in_group
        )
        var n_block_idx = UInt32(
            rebind[Scalar[uint_type2]](in_group_idx) / div_num_blocks_in_group
        )

        return (m_block_idx, n_block_idx)

    @always_inline
    def current_expert_id(self) -> Int32:
        """Get the expert ID for the current group."""
        return self.expert_ids[Int(self.current_group_idx)][0]
