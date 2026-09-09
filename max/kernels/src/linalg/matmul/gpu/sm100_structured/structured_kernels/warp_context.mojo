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
"""RAII warp context managers for SM100 matmul kernel.

MmaWarpContext: MMA warp - allocates TMEM, deallocates on exit
EpilogueWarpContext: Epilogue warp - consumes TMEM, signals completion on exit
"""

from structured_kernels.barriers import WarpGroupBarrier
from .tmem import TmemDeallocBarrier
from .config import OutputPipelineConfig
from .tile_pipeline import (
    MbarPtr,
    OutputTilePipeline,
    EpilogueKContext,
    MmaKStage,
    InputTilePipeline,
    TilePayload,
    MmaStage,
    EpilogueStage,
)
from .tmem import TmemAllocation


# =============================================================================
# Shared type aliases for warp contexts
# =============================================================================


struct _WarpContextTypes[
    opc: OutputPipelineConfig,
    mma_threads: Int,
    epilogue_threads: Int,
](TrivialRegisterPassable):
    """Shared type definitions for MMA and Epilogue warp contexts."""

    comptime Tmem = TmemAllocation[Self.opc.cta_group]
    comptime Pipeline = OutputTilePipeline[Self.opc]
    comptime Dealloc = TmemDeallocBarrier[Self.opc.cta_group]
    comptime Sync = WarpGroupBarrier[
        Self.mma_threads + Self.epilogue_threads, 1
    ]


# =============================================================================
# MmaWarpContext
# =============================================================================


struct MmaWarpContext[
    opc: OutputPipelineConfig,
    mma_threads: Int,
    epilogue_threads: Int,
](TrivialRegisterPassable):
    """MMA warp context - owns TMEM lifecycle and output pipeline.

    __enter__: Signals epilogue that TMEM is allocated
    __exit__: Waits for epilogue, deallocates TMEM

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
        mma_threads: Number of MMA threads.
        epilogue_threads: Number of epilogue threads.
    """

    comptime _Types = _WarpContextTypes[
        Self.opc, Self.mma_threads, Self.epilogue_threads
    ]
    comptime Tmem = Self._Types.Tmem
    comptime Pipeline = Self._Types.Pipeline
    comptime Dealloc = Self._Types.Dealloc
    comptime Sync = Self._Types.Sync

    var tmem: Self.Tmem
    var output_pipeline: Self.Pipeline
    var dealloc_barrier: Self.Dealloc

    @always_inline
    def __init__(
        out self,
        tmem: Self.Tmem,
        output_pipeline: Self.Pipeline,
        dealloc_barrier: Self.Dealloc,
    ):
        self.tmem = tmem
        self.output_pipeline = output_pipeline
        self.dealloc_barrier = dealloc_barrier

    @staticmethod
    @always_inline
    def create(
        tmem_addr_storage: Self.Tmem.SmemAddrStorage,
        accum_barriers_ptr: MbarPtr,
        dealloc_mbar: Self.Dealloc.BarrierStorage,
        mma_complete_mask: UInt16,
    ) -> Self:
        """Create MMA warp context with all necessary components.

        Allocates TMEM and creates output pipeline internally.

        Args:
            tmem_addr_storage: Shared storage for TMEM address communication.
            accum_barriers_ptr: Pointer to accumulator pipeline barriers.
            dealloc_mbar: Barrier for TMEM deallocation synchronization.
            mma_complete_mask: Multicast mask for MMA completion signaling.

        Returns:
            Fully initialized MmaWarpContext.
        """
        var tmem = Self.Tmem.allocate(tmem_addr_storage)
        var output_pipeline = Self.Pipeline(
            accum_barriers_ptr, tmem, mma_complete_mask
        )
        return Self(tmem, output_pipeline, Self.Dealloc(dealloc_mbar))

    @always_inline
    def __enter__(self) -> Self:
        Self.Sync.arrive()
        return self

    @always_inline
    def __exit__(self):
        self.dealloc_barrier.complete_dealloc(self.tmem)

    @always_inline
    def per_k_stage(
        mut self,
    ) -> MmaKStage[origin_of(self.output_pipeline), Self.opc]:
        """Get per-K stage for blockwise FP8 MMA loop.

        Returns a context manager that acquires an output stage and
        signals mma_arrive on exit.

        Example:
            for i in range(num_iters):
                with mma_ctx.per_k_stage() as mma_stage:
                    mma(input_tiles, mma_op, AccumTensor(mma_stage.tmem.offset()))
                # __exit__ signals mma_arrive automatically
        """
        return self.output_pipeline.per_k().produce()


# =============================================================================
# EpilogueWarpContext
# =============================================================================


struct EpilogueWarpContext[
    opc: OutputPipelineConfig,
    mma_threads: Int,
    epilogue_threads: Int,
](TrivialRegisterPassable):
    """Epilogue warp context - consumes TMEM data, signals completion.

    IMPORTANT: Call Sync.wait() BEFORE constructing to ensure TMEM address
    is visible from shared memory.

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
        mma_threads: Number of MMA threads.
        epilogue_threads: Number of epilogue threads.
    """

    comptime _Types = _WarpContextTypes[
        Self.opc, Self.mma_threads, Self.epilogue_threads
    ]
    comptime Tmem = Self._Types.Tmem
    comptime Pipeline = Self._Types.Pipeline
    comptime Dealloc = Self._Types.Dealloc
    comptime Sync = Self._Types.Sync

    var tmem: Self.Tmem
    var output_pipeline: Self.Pipeline
    var dealloc_barrier: Self.Dealloc

    @always_inline
    def __init__(
        out self,
        tmem: Self.Tmem,
        output_pipeline: Self.Pipeline,
        dealloc_barrier: Self.Dealloc,
    ):
        self.tmem = tmem
        self.output_pipeline = output_pipeline
        self.dealloc_barrier = dealloc_barrier

    @staticmethod
    @always_inline
    def create(
        tmem_addr_storage: Self.Tmem.SmemAddrStorage,
        accum_barriers_ptr: MbarPtr,
        dealloc_mbar: Self.Dealloc.BarrierStorage,
        mma_complete_mask: UInt16,
    ) -> Self:
        """Create Epilogue warp context with all necessary components.

        Reads TMEM address from shared memory and creates output pipeline.
        IMPORTANT: Call Sync.wait() BEFORE calling this to ensure TMEM
        address is visible.

        Args:
            tmem_addr_storage: Shared storage containing TMEM address.
            accum_barriers_ptr: Pointer to accumulator pipeline barriers.
            dealloc_mbar: Barrier for TMEM deallocation synchronization.
            mma_complete_mask: Multicast mask for MMA completion signaling.

        Returns:
            Fully initialized EpilogueWarpContext.
        """
        var tmem = Self.Tmem.from_shared(tmem_addr_storage)
        var output_pipeline = Self.Pipeline(
            accum_barriers_ptr, tmem, mma_complete_mask
        )
        return Self(tmem, output_pipeline, Self.Dealloc(dealloc_mbar))

    @always_inline
    def __enter__(self) -> Self:
        return self

    @always_inline
    def __exit__(self):
        self.dealloc_barrier.signal_complete()

    @always_inline
    def per_k_stage[
        input_origin: MutOrigin,
        Payload: TilePayload,
        num_group_stages: Int,
        k_group_size: Int,
    ](
        mut self,
        ref[input_origin] input_pipeline: InputTilePipeline[
            Payload, num_group_stages, k_group_size
        ],
    ) -> EpilogueKContext[
        origin_of(self.output_pipeline),
        origin_of(input_pipeline.pipeline),
        Self.opc,
        num_group_stages,
    ]:
        """Get per-K stage context for blockwise FP8 epilogue.

        Bundles output pipeline (MMA→Epilogue sync) and input pipeline
        (A-scales consumption) into a single context manager.

        Example:
            for k_iter in range(num_iters):
                with epi_ctx.per_k_stage(input_pipeline) as epi_stage:
                    accum.promote(epi_stage, ...)
                # Both pipelines signaled automatically

        Parameters:
            input_origin: Memory origin of the `input_pipeline` reference,
                controlling which warp group owns it.
            Payload: Tile payload type carried by the input pipeline (for
                example, `BlockwiseFP8TilePayload`).
            num_group_stages: Number of producer-consumer stages in the input
                pipeline.
            k_group_size: Number of K iterations grouped into each pipeline
                stage.

        Args:
            input_pipeline: The InputTilePipeline (extracts .pipeline internally).

        Returns:
            EpilogueKContext context manager that handles both pipelines.
        """
        return self.output_pipeline.per_k_epilogue(input_pipeline.pipeline)


# =============================================================================
# Unified Linear Types for Warp Contexts
# =============================================================================
#
# These types work both as linear types (direct use) and within context managers.
#
# Linear Type API (flat):
#     var mma = MmaWarp.create(...)  # allocate TMEM, arrive sync
#     for i in range(num_iters):
#         var stage = mma.acquire_k_stage_linear()
#         mma_op.mma(stage.tmem_offset())
#         stage^.release()
#     mma^.release()  # wait for epilogue, deallocate TMEM
#
# Context Manager API (scoped):
#     with MmaWarpContext.create(...) as mma:
#         for i in range(num_iters):
#             with mma.per_k_stage() as stage:
#                 mma_op.mma(stage.tmem)
#     # __exit__ waits for epilogue, deallocates TMEM


@explicit_destroy("Must call release() to deallocate TMEM")
struct MmaWarp[
    opc: OutputPipelineConfig,
    mma_threads: Int,
    epilogue_threads: Int,
](Deinitable where False):
    """Unified linear type for MMA warp TMEM lifecycle.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `create()` - allocates TMEM, signals sync barrier
    2. Use `output_pipeline` or `acquire_k_stage_linear()` for MMA stages
    3. Must call `release()` to wait for epilogue and deallocate (compiler-enforced)

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
        mma_threads: Number of MMA threads.
        epilogue_threads: Number of epilogue threads.
    """

    comptime _Types = _WarpContextTypes[
        Self.opc, Self.mma_threads, Self.epilogue_threads
    ]
    comptime Tmem = Self._Types.Tmem
    comptime Pipeline = Self._Types.Pipeline
    comptime Dealloc = Self._Types.Dealloc
    comptime Sync = Self._Types.Sync

    var tmem: Self.Tmem
    var output_pipeline: Self.Pipeline
    var dealloc_barrier: Self.Dealloc

    @always_inline
    def __init__(
        out self,
        tmem: Self.Tmem,
        output_pipeline: Self.Pipeline,
        dealloc_barrier: Self.Dealloc,
    ):
        self.tmem = tmem
        self.output_pipeline = output_pipeline
        self.dealloc_barrier = dealloc_barrier

    @staticmethod
    @always_inline
    def create(
        tmem_addr_storage: Self.Tmem.SmemAddrStorage,
        accum_barriers_ptr: MbarPtr,
        dealloc_mbar: Self.Dealloc.BarrierStorage,
        mma_complete_mask: UInt16,
    ) -> Self:
        """Create MMA warp with TMEM allocation.

        Allocates TMEM and signals the warp group sync barrier.

        Args:
            tmem_addr_storage: Shared storage for TMEM address communication.
            accum_barriers_ptr: Pointer to accumulator pipeline barriers.
            dealloc_mbar: Barrier for TMEM deallocation synchronization.
            mma_complete_mask: Multicast mask for MMA completion signaling.

        Returns:
            Fully initialized MmaWarp that must be released.
        """
        var tmem = Self.Tmem.allocate(tmem_addr_storage)
        var output_pipeline = Self.Pipeline(
            accum_barriers_ptr, tmem, mma_complete_mask
        )
        Self.Sync.arrive()  # Signal epilogue that TMEM is ready
        return Self(tmem, output_pipeline, Self.Dealloc(dealloc_mbar))

    @always_inline
    def per_k_stage(
        mut self,
    ) -> MmaKStage[origin_of(self.output_pipeline), Self.opc]:
        """Get per-K stage context manager (for compatibility).

        Prefer acquire_k_stage_linear() for flat code structure.
        """
        return self.output_pipeline.per_k().produce()

    @always_inline
    def acquire_k_stage_linear(
        mut self,
    ) -> MmaStage[origin_of(self.output_pipeline), Self.opc]:
        """Acquire a per-K stage using linear types.

        Waits for epilogue to free the stage, returns a linear handle.

        Usage:
            var stage = mma_handle.acquire_k_stage_linear()
            mma_op.mma(a, b, stage.tmem_offset())
            mma_op.commit(stage.mbar())
            stage^.release()
        """
        return self.output_pipeline.acquire_mma_linear()

    @always_inline
    def release(deinit self):
        """Wait for epilogue and deallocate TMEM.

        This is the only way to destroy this linear type.
        """
        self.dealloc_barrier.complete_dealloc(self.tmem)


@explicit_destroy("Must call release() to signal completion")
struct EpilogueWarp[
    opc: OutputPipelineConfig,
    mma_threads: Int,
    epilogue_threads: Int,
](Deinitable where False):
    """Unified linear type for epilogue warp lifecycle.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `create()` after Sync.wait() - reads TMEM address
    2. Use `output_pipeline` or `acquire_k_stage_linear()` for epilogue stages
    3. Must call `release()` to signal completion (compiler-enforced)

    IMPORTANT: Call Sync.wait() BEFORE create() to ensure TMEM address is visible.

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
        mma_threads: Number of MMA threads.
        epilogue_threads: Number of epilogue threads.
    """

    comptime _Types = _WarpContextTypes[
        Self.opc, Self.mma_threads, Self.epilogue_threads
    ]
    comptime Tmem = Self._Types.Tmem
    comptime Pipeline = Self._Types.Pipeline
    comptime Dealloc = Self._Types.Dealloc
    comptime Sync = Self._Types.Sync

    var tmem: Self.Tmem
    var output_pipeline: Self.Pipeline
    var dealloc_barrier: Self.Dealloc

    @always_inline
    def __init__(
        out self,
        tmem: Self.Tmem,
        output_pipeline: Self.Pipeline,
        dealloc_barrier: Self.Dealloc,
    ):
        self.tmem = tmem
        self.output_pipeline = output_pipeline
        self.dealloc_barrier = dealloc_barrier

    @staticmethod
    @always_inline
    def create(
        tmem_addr_storage: Self.Tmem.SmemAddrStorage,
        accum_barriers_ptr: MbarPtr,
        dealloc_mbar: Self.Dealloc.BarrierStorage,
        mma_complete_mask: UInt16,
    ) -> Self:
        """Create Epilogue warp.

        Reads TMEM address from shared memory. IMPORTANT: Call Sync.wait()
        BEFORE this to ensure the address is visible.

        Args:
            tmem_addr_storage: Shared storage containing TMEM address.
            accum_barriers_ptr: Pointer to accumulator pipeline barriers.
            dealloc_mbar: Barrier for TMEM deallocation synchronization.
            mma_complete_mask: Multicast mask for MMA completion signaling.

        Returns:
            Fully initialized EpilogueWarp that must be released.
        """
        var tmem = Self.Tmem.from_shared(tmem_addr_storage)
        var output_pipeline = Self.Pipeline(
            accum_barriers_ptr, tmem, mma_complete_mask
        )
        return Self(tmem, output_pipeline, Self.Dealloc(dealloc_mbar))

    @always_inline
    def per_k_stage[
        input_origin: MutOrigin,
        Payload: TilePayload,
        num_group_stages: Int,
        k_group_size: Int,
    ](
        mut self,
        ref[input_origin] input_pipeline: InputTilePipeline[
            Payload, num_group_stages, k_group_size
        ],
    ) -> EpilogueKContext[
        origin_of(self.output_pipeline),
        origin_of(input_pipeline.pipeline),
        Self.opc,
        num_group_stages,
    ]:
        """Get per-K stage context manager (for compatibility).

        Prefer acquire_k_stage_linear() for flat code structure.

        Parameters:
            input_origin: Memory origin of the `input_pipeline` reference,
                controlling which warp group owns it.
            Payload: Tile payload type carried by the input pipeline (for
                example, `BlockwiseFP8TilePayload`).
            num_group_stages: Number of producer-consumer stages in the input
                pipeline.
            k_group_size: Number of K iterations grouped into each pipeline
                stage.

        Args:
            input_pipeline: Input tile pipeline providing A-scales data for
                the epilogue.
        """
        return self.output_pipeline.per_k_epilogue(input_pipeline.pipeline)

    @always_inline
    def acquire_k_stage_linear(
        mut self,
    ) -> EpilogueStage[origin_of(self.output_pipeline), Self.opc]:
        """Acquire a per-K stage using linear types.

        Waits for MMA to complete the stage, returns a linear handle.

        Usage:
            var stage = epi_handle.acquire_k_stage_linear()
            process_tmem(stage.tmem())
            stage^.release()
        """
        return self.output_pipeline.acquire_epilogue_linear()

    @always_inline
    def release(deinit self):
        """Signal epilogue completion.

        This is the only way to destroy this linear type.
        """
        self.dealloc_barrier.signal_complete()
