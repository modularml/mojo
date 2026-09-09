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
"""Tile scheduler for split-K matmul kernels on SM100 structured hardware.

Wraps the B200 tile scheduler to partition the K dimension into multiple
splits, coordinating per-split workspace reductions and lock-based
synchronization across CTAs that contribute to the same output tile.
"""

from .tile_scheduler import TileScheduler as B200TileScheduler
from .tile_scheduler import WorkInfo as B200WorkInfo
from linalg.matmul.gpu.tile_scheduler import RasterOrder
from layout import Coord, Idx, Layout, TensorLayout, TileTensor, row_major
from std.math import align_up, ceildiv
from layout.tma_async import SharedMemBarrier, PipelineState
from std.utils.static_tuple import StaticTuple
from structured_kernels.tile_types import (
    static_row_major,
    _StridedLayout,
    _strided_layout,
)
from max.gpu import WARP_SIZE, grid_dim, lane_id
from max.gpu.sync import NamedBarrierSemaphore
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.compute.arch.tcgen05 import *
from std.bit import prev_power_of_two
from std.math.uutils import ufloordiv, umod
from std.utils.index import Index, IndexList

from linalg.structuring import SMemPtr
from .tmem import TmemAddress, TmemTensor


@fieldwise_init
struct WorkInfo(TrivialRegisterPassable, Writable):
    """Describes a unit of split-K work for a single output tile.

    Holds the output-tile coordinates (m, n), the starting K index and
    number of K tiles this split covers, and whether the tile is in bounds.
    """

    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    var num_k_tiles: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    comptime INVALID_WORK_INFO = Self(0, 0, 0, 0, False)

    @always_inline
    def is_valid(self) -> Bool:
        """Returns whether this work tile is in bounds."""
        return self.is_valid_tile

    @always_inline
    def is_final_split(self, k_tiles_per_output_tile: UInt32) -> Bool:
        """Returns whether this split covers the final K tiles of the output tile.

        Args:
            k_tiles_per_output_tile: Total number of K tiles for the output tile.
        """
        return (self.k_start + self.num_k_tiles) == k_tiles_per_output_tile

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes a parenthesized summary of this work info to the given writer.

        Args:
            writer: Sink the parenthesized `(m, n, k_start, is_valid_tile)`
                summary is written to.
        """
        writer.write(
            "(",
            self.m,
            ", ",
            self.n,
            ", ",
            self.k_start,
            ", ",
            self.is_valid_tile,
            ")",
        )


struct WaitAndAdvanceContextSplitK[
    work_origin: MutOrigin,
](TrivialRegisterPassable):
    """Context for waiting on CLC barrier and advancing work iterator (Split-K).

    Encapsulates the CLC response barrier synchronization:
    - Construction: Waits for CLC response, fetches next work
    - __enter__: Returns current work_info for processing
    - __exit__: Assigns fetched work as current

    Parameters:
        work_origin: Memory origin of the work info pointer (inferred).
    """

    var work_info_ptr: Pointer[WorkInfo, Self.work_origin]
    var next_work: WorkInfo

    @always_inline
    def __init__(
        out self,
        work_info_ptr: Pointer[WorkInfo, Self.work_origin],
        next_work: WorkInfo,
    ):
        self.work_info_ptr = work_info_ptr
        self.next_work = next_work

    @always_inline
    def __enter__(self) -> WorkInfo:
        return self.work_info_ptr[]

    @always_inline
    def __exit__(mut self):
        self.work_info_ptr[] = self.next_work


# =============================================================================
# WorkIteratorSplitK - Per-warp iterator encapsulating scheduler + pipeline state
# =============================================================================


struct WorkIteratorSplitK[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type=.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
    num_split_k: Int,
](Copyable, Iterable, Iterator, RegisterPassable):
    """Per-warp work iterator for split-K using __next__-style iteration.

    Usage:
        var work_iter = scheduler.work_iterator()
        for current in work_iter:
            scheduler.throttle_signal(ctx.is_first_cta_in_cluster)
            do_work(current)

    Parameters:
        num_stages: Number of CLC pipeline stages for work distribution.
        reduction_tile_shape: The `(BM, MMA_N, BK)` per-block reduction tile
            shape used for the split-K workspace layout.
        cluster_shape: Cluster tile counts as `(m, n, k)`.
        rasterize_order: Order CLC rasterizes tiles across the cluster grid.
        block_swizzle_size: Block swizzle factor for tile remapping, one of
            0, 1, 2, 4, or 8.
        num_split_k: Number of splits the K dimension is partitioned into.
    """

    comptime Element = WorkInfo

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]

    var scheduler: Self.SchedulerType
    var work_info: WorkInfo
    var consumer_state: PipelineState[Self.num_stages]
    var needs_fetch: Bool

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    @always_inline
    def __init__(out self, scheduler: Self.SchedulerType, work_info: WorkInfo):
        """Create work iterator with initial work_info.

        Args:
            scheduler: The split-K tile scheduler owning the CLC state.
            work_info: Initial work descriptor for the first output tile.
        """
        self.scheduler = scheduler
        self.work_info = work_info
        self.consumer_state = PipelineState[Self.num_stages]()
        self.needs_fetch = False

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> WorkInfo:
        """Return current work item, deferring fetch to next call.

        Raises:
            StopIteration: When there is no more work to process.
        """
        if self.needs_fetch:
            self.work_info = self.scheduler.fetch_next_work(
                self.work_info, self.consumer_state
            )
            self.consumer_state.step()
        if not self.work_info.is_valid():
            raise StopIteration()
        self.needs_fetch = True
        return self.work_info


# =============================================================================
# SchedulerWorkIteratorSplitK - For Scheduler warp (split-K variant)
# =============================================================================


struct SchedulerWorkIteratorSplitK[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type=.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
    num_split_k: Int,
](Copyable, Iterable, Iterator, RegisterPassable):
    """Work iterator for Scheduler warp (split-K) using __next__-style iteration.

    Usage:
        var sched_iter = scheduler.scheduler_iterator()
        for _ in sched_iter:
            sched_iter.signal_and_advance()
        sched_iter.drain()

    Parameters:
        num_stages: Number of CLC pipeline stages for work distribution.
        reduction_tile_shape: The `(BM, MMA_N, BK)` per-block reduction tile
            shape used for the split-K workspace layout.
        cluster_shape: Cluster tile counts as `(m, n, k)` (defaults to
            `(1, 1, 1)`).
        rasterize_order: Order CLC rasterizes tiles across the cluster
            grid (defaults to `RasterOrder.AlongM`).
        block_swizzle_size: Block swizzle factor for tile remapping, one
            of 0, 1, 2, 4, or 8 (defaults to 8).
        num_split_k: Number of splits the K dimension is partitioned into.
    """

    comptime Element = WorkInfo

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]
    comptime ThrottlePipeline = Self.SchedulerType.ThrottlePipeline

    var scheduler: Self.SchedulerType
    var work_info: WorkInfo
    var consumer_state: PipelineState[Self.num_stages]
    var producer_state: PipelineState[Self.num_stages]
    var throttle_pipeline: Self.ThrottlePipeline
    var needs_fetch: Bool

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    @always_inline
    def __init__(out self, scheduler: Self.SchedulerType, work_info: WorkInfo):
        """Create scheduler iterator. Throttle pipeline from scheduler.

        Args:
            scheduler: The split-K tile scheduler owning the CLC state.
            work_info: Initial work descriptor for the first output tile.
        """
        self.scheduler = scheduler
        self.work_info = work_info
        self.consumer_state = PipelineState[Self.num_stages]()
        self.producer_state = PipelineState[Self.num_stages](0, 1, 0)
        self.throttle_pipeline = scheduler.throttle_pipeline
        self.needs_fetch = False

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> WorkInfo:
        """Return current work item, deferring fetch to next call.

        Raises:
            StopIteration: When there is no more work to process.
        """
        if self.needs_fetch:
            self.work_info = self.scheduler.fetch_next_work(
                self.work_info, self.consumer_state
            )
            self.consumer_state.step()
        if not self.work_info.is_valid():
            raise StopIteration()
        self.needs_fetch = True
        return self.work_info

    @always_inline
    def signal_and_advance(mut self):
        """Signal CLC throttle consumer and advance to next work request."""
        self.throttle_pipeline.consumer_signal_and_step()
        self.producer_state = self.scheduler.advance_to_next_work(
            self.producer_state
        )

    @always_inline
    def drain(mut self):
        """Drain all pending CLC requests before kernel exit."""

        comptime for i in range(Self.num_stages):
            # Split-K wraps underlying scheduler, so access via scheduler.scheduler
            self.scheduler.scheduler.empty_mbar[
                self.producer_state.index()
            ].wait(self.producer_state.phase())
            self.producer_state.step()


struct TileScheduler[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type=.uint32] = Index[
        dtype=DType.uint32
    ](1, 1, 1),
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    block_swizzle_size: Int = 8,
    num_split_k: Int = 1,
](TrivialRegisterPassable):
    """Tile scheduler that partitions the K dimension into multiple splits.

    Wraps the B200 tile scheduler and remaps its work info into split-K
    coordinates, managing per-split workspace reductions and lock-based
    synchronization so that CTAs contributing to the same output tile
    accumulate correctly.

    Parameters:
        num_stages: Number of CLC pipeline stages for work distribution.
        reduction_tile_shape: The `(BM, MMA_N, BK)` per-block reduction tile
            shape used for the split-K workspace layout.
        cluster_shape: Cluster tile counts as `(m, n, k)` (defaults to
            `(1, 1, 1)`).
        rasterize_order: Order CLC rasterizes tiles across the cluster
            grid (defaults to `RasterOrder.AlongM`).
        block_swizzle_size: Block swizzle factor for tile remapping, one
            of 0, 1, 2, 4, or 8 (defaults to 8).
        num_split_k: Number of splits the K dimension is partitioned into
            (defaults to 1).
    """

    comptime UnderlyingScheduler = B200TileScheduler[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]
    comptime BM = Self.reduction_tile_shape[0]
    comptime MMA_N = Self.reduction_tile_shape[1]
    comptime BK = Self.reduction_tile_shape[2]
    comptime ROW_SIZE = Self.MMA_N if Self.BM == 128 else Self.MMA_N // 2
    comptime ThrottlePipeline = Self.UnderlyingScheduler.ThrottlePipeline

    # Typed barrier array aliases (delegate to underlying scheduler)
    comptime ClcResponseArray = Self.UnderlyingScheduler.ClcResponseArray
    comptime ClcBarrierArray = Self.UnderlyingScheduler.ClcBarrierArray
    comptime ThrottleBarrierArray = Self.UnderlyingScheduler.ThrottleBarrierArray

    @__allow_legacy_any_origin_fields
    var locks_ptr: UnsafePointer[Int32, MutAnyOrigin]
    var scheduler: Self.UnderlyingScheduler
    var total_k_tiles: UInt32
    var k_tiles_per_split: UInt32
    var throttle_pipeline: Self.ThrottlePipeline

    @staticmethod
    def init_throttle_barriers(
        storage_ptr: SMemPtr[SharedMemBarrier],
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize throttle pipeline barriers. Called once by elect_one thread.

        Args:
            storage_ptr: Pointer to shared memory backing the throttle
                barriers.
            producer_arv_count: Arrival count the producer waits for per
                stage.
            consumer_arv_count: Arrival count the consumer waits for per
                stage.
        """
        Self.UnderlyingScheduler.init_throttle_barriers(
            storage_ptr, producer_arv_count, consumer_arv_count
        )

    @always_inline
    def __init__(
        out self,
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        clc_response: Self.ClcResponseArray,
        clc_full: Self.ClcBarrierArray,
        clc_empty: Self.ClcBarrierArray,
        clc_throttle: Self.ThrottleBarrierArray,
        locks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    ):
        """Initialize from typed barrier arrays.

        Args:
            cluster_dim: Grid cluster dimensions as `(m, n, k)` used for
                fast division rasterization.
            mnk: Problem dimensions as `(M, N, K)`; `K` sets the total K
                tile count.
            clc_response: Shared memory array storing CLC response payloads.
            clc_full: Barriers signaled when CLC response data is ready.
            clc_empty: Barriers signaled when a response slot is available.
            clc_throttle: Barriers for the throttle pipeline pacing the
                scheduler against the load warp.
            locks_ptr: Pointer to the per-output-tile lock buffer used to
                synchronize split-K reductions across CTAs.
        """
        self.scheduler = Self.UnderlyingScheduler(
            cluster_dim,
            clc_response,
            clc_full,
            clc_empty,
            clc_throttle,
        )
        self.total_k_tiles = ceildiv(
            mnk[2], UInt32(Self.reduction_tile_shape[2])
        )
        self.k_tiles_per_split = ceildiv(
            self.total_k_tiles, UInt32(Self.num_split_k)
        )
        self.locks_ptr = locks_ptr.bitcast[Int32]()
        self.throttle_pipeline = Self.ThrottlePipeline(clc_throttle.ptr)

    @always_inline
    def convert_to_splitk_work_info(self, work_info: B200WorkInfo) -> WorkInfo:
        var current_k_start = work_info.k_start * self.k_tiles_per_split
        var remaining_k_tiles = self.total_k_tiles - current_k_start
        return WorkInfo(
            work_info.m,
            work_info.n,
            current_k_start,
            min(self.k_tiles_per_split, remaining_k_tiles),
            work_info.is_valid_tile,
        )

    @always_inline
    def initial_work_info(self) -> WorkInfo:
        return self.convert_to_splitk_work_info(
            self.scheduler.initial_work_info()
        )

    @always_inline
    def advance_to_next_work(
        self,
        mut clc_state: PipelineState[Self.num_stages],
    ) -> PipelineState[Self.num_stages]:
        return self.scheduler.advance_to_next_work(clc_state)

    @always_inline
    def fetch_next_work(
        self,
        work_info: WorkInfo,
        consumer_state: PipelineState[Self.num_stages],
    ) -> WorkInfo:
        var underlying_workinfo = B200WorkInfo(
            work_info.m, work_info.n, work_info.k_start, work_info.is_valid_tile
        )
        return self.convert_to_splitk_work_info(
            self.scheduler.fetch_next_work(underlying_workinfo, consumer_state)
        )

    # =========================================================================
    # CLC Throttle (Producer Side)
    # =========================================================================

    @always_inline
    def throttle_signal(mut self, is_first_cta_in_cluster: Bool):
        """Signal CLC throttle if this is the first CTA in cluster.

        Args:
            is_first_cta_in_cluster: Only first CTA signals to avoid duplicates.
        """
        if is_first_cta_in_cluster:
            self.scheduler.throttle_pipeline.producer_signal_and_step()

    # =========================================================================
    # Work Iteration Context Managers
    # =========================================================================

    @always_inline
    def wait_and_advance_work[
        work_origin: MutOrigin, //
    ](
        self,
        ref[work_origin] work_info: WorkInfo,
        mut consumer_state: PipelineState[Self.num_stages],
    ) -> WaitAndAdvanceContextSplitK[work_origin]:
        """Wait for next work from CLC and advance (Split-K).

        Encapsulates the CLC barrier wait (called on scheduler directly).

        Usage:
            with scheduler.wait_and_advance_work(work_info, state) as current:
                do_mma(current)
            # After: work_info updated to next value

        Parameters:
            work_origin: Memory origin of the `work_info` reference
                (inferred).

        Args:
            work_info: Reference to the current work descriptor, updated to
                the next work item on context exit.
            consumer_state: CLC consumer pipeline state, advanced after the
                fetch.
        """
        var next = self.fetch_next_work(work_info, consumer_state)
        consumer_state.step()
        return WaitAndAdvanceContextSplitK(Pointer(to=work_info), next)

    @always_inline
    def work_iterator(
        self,
    ) -> WorkIteratorSplitK[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Create a per-warp work iterator that owns work_info internally.
        Throttle pipeline is obtained from the scheduler.
        """
        return WorkIteratorSplitK(self, self.initial_work_info())

    @always_inline
    def scheduler_iterator(
        self,
    ) -> SchedulerWorkIteratorSplitK[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Create iterator for Scheduler warp (owns work_info and both states).
        Throttle pipeline is obtained from the scheduler.
        """
        return SchedulerWorkIteratorSplitK(self, self.initial_work_info())

    @always_inline
    def is_last_split(self, work_tile_info: WorkInfo) -> Bool:
        return work_tile_info.is_valid() and work_tile_info.is_final_split(
            self.total_k_tiles
        )

    @always_inline
    def output_tile_index(self, work_info: WorkInfo) -> UInt32:
        return work_info.m * UInt32(grid_dim.y) + work_info.n

    comptime WorkspaceTileLayout = static_row_major[Self.BM, Self.MMA_N]

    @always_inline
    def _get_workspace_tile[
        accum_type: DType, workspace_layout: TensorLayout
    ](
        self,
        reduction_workspace: TileTensor[
            accum_type, workspace_layout, MutAnyOrigin
        ],
        reduction_tile_idx: UInt32,
    ) -> TileTensor[accum_type, Self.WorkspaceTileLayout, MutAnyOrigin]:
        var offset = reduction_tile_idx * UInt32(Self.BM) * UInt32(Self.MMA_N)
        return TileTensor[accum_type, Self.WorkspaceTileLayout, MutAnyOrigin](
            reduction_workspace._storage + Int(offset),
            row_major[Self.BM, Self.MMA_N](),
        )

    @always_inline
    @staticmethod
    def _get_max_width_per_stage[max_width: Int]() -> Int:
        return min(max_width, Self.ROW_SIZE & -Self.ROW_SIZE)

    @always_inline
    @staticmethod
    def _get_widths_per_stage[max_width: Int]() -> Tuple[Array[Int, 4], Int]:
        """helper functions to decompose MMA_N into widths that are powers of two
        """
        var arr = Array[Int, 4](uninitialized=True)
        var current_width = Self.ROW_SIZE
        var first_width: Int
        var second_width: Int

        var i = 0
        while current_width > 0:
            first_width = min(max_width, prev_power_of_two(current_width))
            second_width = current_width - first_width
            arr[i] = first_width
            i += 1
            current_width = second_width

        return (arr^, i)

    @always_inline
    @staticmethod
    def _to_next_subtile[
        accum_type: DType,
        tile_layout: TensorLayout,
        /,
        *,
        widths: Array[Int, 4],
        curr_stage: Int,
    ](
        tensor: TileTensor[accum_type, tile_layout, MutAnyOrigin],
    ) -> TileTensor[
        accum_type,
        # Shape narrows to [height, stage_width], but stride is preserved
        # from the parent [parent_stride, 1] -- NOT row_major of the
        # narrowed shape. The sub-tile is a strided view into wider rows.
        _StridedLayout[
            tile_layout.static_shape[0],
            widths[curr_stage],
            tile_layout.static_stride[0],
        ],
        MutAnyOrigin,
    ]:
        @__parameter
        def _get_current_width(widths: Array[Int, 4], curr_stage: Int) -> Int:
            var width = 0
            for i in range(curr_stage):
                width += widths[i]
            return width

        comptime current_width = _get_current_width(widths, curr_stage)

        return TileTensor[
            accum_type,
            _StridedLayout[
                tile_layout.static_shape[0],
                widths[curr_stage],
                tile_layout.static_stride[0],
            ],
            MutAnyOrigin,
        ](
            tensor._storage + current_width,
            _strided_layout[
                tile_layout.static_shape[0],
                widths[curr_stage],
                tile_layout.static_stride[0],
            ](),
        )

    @always_inline
    def store_to_workspace[
        accum_type: DType,
        workspace_layout: TensorLayout,
        /,
        *,
        do_reduction: Bool = False,
        write_back: Bool = False,
    ](
        self,
        tmem: TmemAddress,
        reduction_workspace: TileTensor[
            accum_type, workspace_layout, MutAnyOrigin
        ],
        epilogue_thread_idx: Int,
        reduction_tile_idx: UInt32,
    ):
        # 128 is a magic number that is provided by the NVCC backend.
        # register size that is greater than that will not compile.
        comptime widths_per_stage = Self._get_widths_per_stage[128]()
        comptime widths = widths_per_stage[0]
        comptime num_widths = widths_per_stage[1]

        # TmemTensor for split-K reduction.
        # Use cta_group=2 to force is_lower_required=True - split-K always
        # needs both upper and lower fragments for the full reduction.
        comptime accum_layout = Layout.row_major(Self.BM, Self.ROW_SIZE)
        comptime AccumTmem = TmemTensor[accum_type, accum_layout, cta_group=2]

        var local_warp_id = ufloordiv(epilogue_thread_idx, WARP_SIZE)

        # workspace has layout (X, BM, MMA_N)
        var workspace_tile = self._get_workspace_tile(
            reduction_workspace, reduction_tile_idx
        )

        comptime REDUCTION_BM = Self.BM // 4 if Self.BM == 128 else Self.BM // 2
        comptime REDUCTION_BN = Self.MMA_N if Self.BM == 128 else Self.MMA_N // 2
        var warp_id_x = local_warp_id if Self.BM == 128 else umod(
            local_warp_id, 2
        )
        var warp_id_y = 0 if Self.BM == 128 else ufloordiv(local_warp_id, 2)

        var reduction_frag = workspace_tile.tile[REDUCTION_BM, REDUCTION_BN](
            Coord(warp_id_x, warp_id_y)
        )
        var reduction_upper = reduction_frag.tile[16, REDUCTION_BN](
            Coord(Idx[0], Idx[0])
        )
        var reduction_lower = reduction_frag.tile[16, REDUCTION_BN](
            Coord(Idx[1], Idx[0])
        )
        var stage_addr = tmem  # Track address for iteration

        comptime for stage in range(num_widths):
            comptime stage_width = widths[stage]
            comptime stage_rep = stage_width // 8

            var stage_tmem = AccumTmem(stage_addr)
            var frags = stage_tmem.load_fragments[stage_rep]()
            AccumTmem.wait_load()

            # Get workspace subtiles for this stage
            var ws_upper = (
                Self._to_next_subtile[widths=widths, curr_stage=stage](
                    reduction_upper
                )
                .vectorize[1, 2]()
                .distribute[row_major[8, 4]()](lane_id())
            )
            var ws_lower = (
                Self._to_next_subtile[widths=widths, curr_stage=stage](
                    reduction_lower
                )
                .vectorize[1, 2]()
                .distribute[row_major[8, 4]()](lane_id())
            )

            comptime num_m = type_of(ws_upper).static_shape[0]
            comptime num_n = type_of(ws_upper).static_shape[1]

            comptime for m in range(num_m):
                comptime for n in range(num_n):
                    comptime i = m * num_n + n

                    var v2_upper = rebind[type_of(ws_upper).ElementType](
                        SIMD[accum_type, 2](
                            frags.upper[2 * i], frags.upper[2 * i + 1]
                        )
                    )
                    var v2_lower = rebind[type_of(ws_lower).ElementType](
                        SIMD[accum_type, 2](
                            frags.lower[2 * i], frags.lower[2 * i + 1]
                        )
                    )

                    comptime if do_reduction:
                        v2_upper += ws_upper[m, n]
                        v2_lower += ws_lower[m, n]

                    comptime if write_back:
                        ws_upper[m, n] = v2_upper
                        ws_lower[m, n] = v2_lower
                    else:
                        frags.upper[2 * i] = v2_upper[0]
                        frags.upper[2 * i + 1] = v2_upper[1]
                        frags.lower[2 * i] = v2_lower[0]
                        frags.lower[2 * i + 1] = v2_lower[1]

            # Store modified fragments back to TMEM
            comptime if not write_back:
                stage_tmem.store_fragments[stage_rep](frags)
                AccumTmem.wait_store()

            stage_addr = stage_addr + stage_width

    @always_inline
    def reduction[
        accum_type: DType,
        workspace_layout: TensorLayout,
    ](
        self,
        reduction_workspace: TileTensor[
            accum_type, workspace_layout, MutAnyOrigin
        ],
        tmem: TmemAddress,
        epilogue_thread_idx: Int,
        work_info: WorkInfo,
    ) -> Bool:
        var reduction_tile_idx = self.output_tile_index(work_info)

        var lock_idx = reduction_tile_idx

        if not self.is_last_split(work_info):
            if work_info.k_start == 0:
                # first split don't wait and just write to workspace.
                self.store_to_workspace[do_reduction=False, write_back=True](
                    tmem,
                    reduction_workspace,
                    epilogue_thread_idx,
                    reduction_tile_idx,
                )
            else:
                Self.wait_eq(
                    self.locks_ptr,
                    0,
                    epilogue_thread_idx,
                    lock_idx,
                    work_info.k_start,
                )

                self.store_to_workspace[do_reduction=True, write_back=True](
                    tmem,
                    reduction_workspace,
                    epilogue_thread_idx,
                    reduction_tile_idx,
                )

            var increment = work_info.num_k_tiles + work_info.k_start

            Self.arrive_set(
                self.locks_ptr,
                0,
                epilogue_thread_idx,
                lock_idx,
                increment,
            )

            return False
        else:
            Self.wait_eq(
                self.locks_ptr,
                0,
                epilogue_thread_idx,
                lock_idx,
                work_info.k_start,
            )
            self.store_to_workspace[do_reduction=True, write_back=False](
                tmem,
                reduction_workspace,
                epilogue_thread_idx,
                reduction_tile_idx,
            )

            return True

    @always_inline
    @staticmethod
    def wait_eq(
        lock_ptr: UnsafePointer[Int32, MutAnyOrigin],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        val: UInt32,
    ):
        var sema = NamedBarrierSemaphore[Int32(WARPGROUP_SIZE), 4, 1](
            lock_ptr + lock_idx, barrier_group_thread_idx
        )
        sema.wait_eq(barrier_id, Int32(val))

    @staticmethod
    @always_inline
    def wait_lt(
        lock_ptr: UnsafePointer[Int32, MutAnyOrigin],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        count: UInt32,
    ):
        pass

    @staticmethod
    @always_inline
    def arrive_set(
        lock_ptr: UnsafePointer[Int32, MutAnyOrigin],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        val: UInt32,
    ):
        var sema = NamedBarrierSemaphore[Int32(WARPGROUP_SIZE), 4, 1](
            lock_ptr + lock_idx, barrier_group_thread_idx
        )
        sema.arrive_set(barrier_id, Int32(val))


@always_inline
def get_num_tiles(
    problem_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: IndexList[2],
) -> IndexList[2]:
    """Computes the number of output tiles aligned up to the cluster shape.

    Args:
        problem_shape: The (M, N, K) problem dimensions.
        block_tile_shape: The (BM, BN, BK) per-block tile dimensions.
        cluster_shape: The (cluster_m, cluster_n) cluster dimensions.

    Returns:
        The aligned (num_blocks_m, num_blocks_n) tile grid.
    """
    var num_block_m = ceildiv(problem_shape[0], block_tile_shape[0])
    var num_block_n = ceildiv(problem_shape[1], block_tile_shape[1])

    var problem_blocks_m = align_up(num_block_m, cluster_shape[0])
    var problem_blocks_n = align_up(num_block_n, cluster_shape[1])

    return Index(problem_blocks_m, problem_blocks_n)


@always_inline
def get_required_locks_buffer_size_bytes[
    accum_type: DType
](
    problem_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: IndexList[2],
) -> Int:
    """Computes the byte size of the lock buffer needed for split-K reduction.

    Parameters:
        accum_type: Element type of the split-K reduction accumulator.

    Args:
        problem_shape: The (M, N, K) problem dimensions.
        block_tile_shape: The (BM, BN, BK) per-block tile dimensions.
        cluster_shape: The (cluster_m, cluster_n) cluster dimensions.

    Returns:
        The number of bytes required for the per-output-tile lock buffer.
    """
    var problem_blocks = get_num_tiles(
        problem_shape, block_tile_shape, cluster_shape
    )
    var num_output_tiles = problem_blocks[0] * problem_blocks[1]

    var locks_workspace_bytes = num_output_tiles * size_of[Int32]()

    return locks_workspace_bytes
