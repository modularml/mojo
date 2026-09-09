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

"""Tile pipeline for SM100 producer-consumer synchronization.

Provides staged tile storage with producer-consumer barrier synchronization
for TMA-MMA pipeline coordination. Two complementary APIs offer either
automatic scoped cleanup (context managers) or compiler-enforced explicit
cleanup (linear types).

All tiles use TileTensor natively.

Key Abstractions
----------------
- InputTilePipeline[Payload]: Generic pipeline with payload abstraction
- OutputTilePipeline: TMEM accumulator stages for MMA→Epilogue pipeline

Type Architecture: Two-Type Design
-----------------------------------
Input pipeline access uses a two-type design that separates
**compiler-enforced linear types** from **ergonomic context wrappers**:

    ┌─────────────────────────────┐   ┌─────────────────────────────┐
    │   InputProducerStage        │   │   InputConsumerStage        │
    │   @explicit_destroy,Movable │   │   @explicit_destroy,Movable │
    │   release(deinit self)      │   │   release(deinit self)      │
    │   Compiler enforces release │   │   Compiler enforces release │
    └─────────────────────────────┘   └─────────────────────────────┘

    ┌─────────────────────────────┐   ┌─────────────────────────────┐
    │   ProducerTiles       │   │   ConsumerTiles       │
    │   TrivialRegisterPassable   │   │   TrivialRegisterPassable   │
    │   __enter__/__exit__        │   │   __enter__/__exit__        │
    │   Auto-releases on exit     │   │   Auto-releases on exit     │
    └─────────────────────────────┘   └─────────────────────────────┘

Both type pairs share the same accessor interface (payload, stage,
barrier/mbar, expect_bytes). They differ in ownership semantics:

- **Stage types** (`InputProducerStage`, `InputConsumerStage`):
  `@explicit_destroy` linear types. The compiler errors if you forget
  to call `release()`. Use for flat code with explicit resource management.

- **Context types** (`ProducerTiles`, `ConsumerTiles`):
  `TrivialRegisterPassable` wrappers. Auto-release via `__exit__` when
  used in a `with` block. Use for scoped, automatic resource management.

Acquire API
-----------
Role handles (`InputProducer`, `InputConsumer`) provide the primary API:

    acquire()          → returns Context type (for `with` blocks)
    acquire_stage()    → returns Stage type  (linear, compiler-enforced)
    acquire_if_needed  → returns Context type (try-acquire pattern)
    try_acquire()      → non-blocking readiness check

`InputTilePipeline` also exposes direct linear-type acquire methods:

    acquire_producer() → InputProducerStage (linear)
    acquire_consumer() → InputConsumerStage (linear)

Context Manager API (scoped, automatic)
---------------------------------------
Use `acquire()` for automatic cleanup via `with` blocks. The context
type's `__exit__` handles barrier signaling and stage advancement:

    with producer.acquire() as tiles:   # BLOCKS until consumer frees stage
        load_tiles(tiles)                # safe to write
                                         # EXIT: advances producer automatically

    with consumer.acquire() as tiles:   # BLOCKS until producer fills stage
        use_tiles(tiles)                 # safe to read
                                         # EXIT: advances consumer automatically

Linear Type API (flat, compiler-enforced)
-----------------------------------------
Use `acquire_stage()` for explicit control with compile-time safety.
The compiler errors if you forget to call `release()`:

    var tiles = producer.acquire_stage()       # BLOCKS until consumer frees stage
    load_tiles(tiles)                          # safe to write
    tiles^.release()                           # transfer + release (compiler enforces)

    var input = pipeline.acquire_consumer()    # BLOCKS until producer fills stage
    process(input)                             # safe to read
    input^.release()                           # transfer + release (compiler enforces)

The `^` transfer operator is required because `release(deinit self)`
consumes the value. Omitting it is a compile error.

When to Use Which
-----------------
- **Context manager** (`acquire`): Default choice. Most kernel code uses this.
  Scoped cleanup is safe, readable, and works well with nested pipelines.
- **Linear type** (`acquire_stage`): Use when context managers create
  excessive nesting, or when you need explicit control over release ordering.
  The compiler catches forgotten releases at compile time.

Example: TMA Load Warp (context manager)
-----------------------------------------
    with input_pipeline.producer() as producer:
        for current in load_iter:
            scheduler.throttle_signal(ctx.is_first_cta_in_cluster)
            for i in range(num_iters):
                with producer.acquire() as tiles:
                    tma_load(tiles.a_tile(), tiles.b_tile())
        producer.drain()

Example: MMA Warp (linear types, flat)
--------------------------------------
    var mma_handle = MmaHandle.create(...)
    for _ in mma_iter:
        for _ in range(num_iters):
            var mma_stage = mma_handle.acquire_k_stage_linear()
            var input_tiles = input_pipeline.acquire_consumer()
            mma(input_tiles, mma_op, ...)
            input_tiles^.release()
            mma_stage^.release()
    mma_handle^.release()

Example: Epilogue Warp (context manager)
----------------------------------------
    with epi_ctx:
        for current in epi_iter:
            with output_pipeline.consumer() as output_stage:
                write_output(output_stage)
"""

from layout.tma_async import SharedMemBarrier
from std.utils.index import IndexList
from .config import OutputPipelineConfig
from structured_kernels.pipeline import (
    ProducerConsumerPipeline,
    SM100_PIPELINE_WAIT_TICKS,
)
from .tmem import TmemAllocation, TmemStage

# SMemArray for barriers (non-tile arrays), SMemPtr for pointers
from linalg.structuring import SMemPtr, SMemArray

# TileTensor-based tile arrays for most tile storage (A, B)
from structured_kernels.tile_types import (
    SMemTileArray2D,
    SMemTileArray2DRowMajor,
    SMemTileArrayWithLayout,
    TilePayload,
    internal_sf_k_major,
)

comptime MbarPtr = SMemPtr[SharedMemBarrier]


# ============================================================================
# Tile Payloads - Data containers for pipeline tile arrays
# ============================================================================


struct StandardTilePayload[
    a_type: DType,
    b_type: DType,
    a_shape: IndexList[2],  # (BM, BK)
    b_shape: IndexList[2],  # (BN, BK)
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload):
    """Tile payload for standard matmul (A and B tiles).

    Uses explicit dimensions for tile arrays. The tiles are stored as TileTensor
    with row_major layout. TileTensors are passed directly to TMA/MMA.
    at TMA/MMA boundaries.

    Parameters:
        a_type: Element type of the A operand tiles.
        b_type: Element type of the B operand tiles.
        a_shape: A tile dimensions as `(BM, BK)`.
        b_shape: B tile dimensions as `(BN, BK)`.
        num_pipeline_stages: Number of pipeline buffer stages.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_shape[0],
        Self.a_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_shape[0],
        Self.b_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray

    @always_inline
    def __init__(out self, a_tiles: Self.ATileArray, b_tiles: Self.BTileArray):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles

    @always_inline
    def get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[Self.ATile, Self.BTile]:
        """Get A and B tiles at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (self.a_tiles[idx], self.b_tiles[idx])

    @always_inline
    def get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    def get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]


struct BlockScaledTilePayload[
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_shape: IndexList[2],  # (BM, BK)
    b_shape: IndexList[2],  # (BN, BK)
    sfa_shape: IndexList[2],  # Scale factor A dims
    sfb_shape: IndexList[2],  # Scale factor B dims
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload):
    """Tile payload for block-scaled matmul (A, B, SFA, SFB tiles).

    Parameters:
        a_type: Element type of the A operand tiles.
        b_type: Element type of the B operand tiles.
        sfa_type: Element type of the A scale factor (SFA) tiles.
        sfb_type: Element type of the B scale factor (SFB) tiles.
        a_shape: A tile dimensions as `(BM, BK)`.
        b_shape: B tile dimensions as `(BN, BK)`.
        sfa_shape: Dimensions of the A scale factor (SFA) tile.
        sfb_shape: Dimensions of the B scale factor (SFB) tile.
        num_pipeline_stages: Number of pipeline buffer stages.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_shape[0],
        Self.a_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_shape[0],
        Self.b_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    # SF tiles use internal_sf_k_major layout (matches tile_sf_layout_k_major).
    # MMA extracts layout directly from TileTensor type parameters.
    comptime sfa_layout = internal_sf_k_major[
        Self.sfa_shape[0], Self.sfa_shape[1]
    ]
    comptime sfb_layout = internal_sf_k_major[
        Self.sfb_shape[0], Self.sfb_shape[1]
    ]
    comptime SFATileArray = SMemTileArrayWithLayout[
        Self.sfa_type,
        Self.sfa_layout,
        Self.num_pipeline_stages,
        128,
    ]
    comptime SFBTileArray = SMemTileArrayWithLayout[
        Self.sfb_type,
        Self.sfb_layout,
        Self.num_pipeline_stages,
        128,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile
    comptime SFATile = Self.SFATileArray.Tile
    comptime SFBTile = Self.SFBTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var sfa_tiles: Self.SFATileArray
    var sfb_tiles: Self.SFBTileArray

    @always_inline
    def __init__(
        out self,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        sfa_tiles: Self.SFATileArray,
        sfb_tiles: Self.SFBTileArray,
    ):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles
        self.sfa_tiles = sfa_tiles
        self.sfb_tiles = sfb_tiles

    @always_inline
    def get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[
        Self.ATile, Self.BTile, Self.SFATile, Self.SFBTile
    ]:
        """Get A, B, SFA, SFB tiles at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (
            self.a_tiles[idx],
            self.b_tiles[idx],
            self.sfa_tiles[idx],
            self.sfb_tiles[idx],
        )

    @always_inline
    def get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index."""
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    def get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index."""
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    def get_sfa_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.SFATile:
        """Get SFA tile at the specified stage and k-group index."""
        return self.sfa_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    def get_sfb_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.SFBTile:
        """Get SFB tile at the specified stage and k-group index."""
        return self.sfb_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]


struct BlockwiseFP8TilePayload[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    a_shape: IndexList[2],  # (BM, BK)
    b_shape: IndexList[2],  # (BN, BK)
    a_scales_shape: IndexList[2],  # (1, BM)
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload, TrivialRegisterPassable):
    """Tile payload for blockwise FP8 matmul (A, B, A-scales tiles).

    Unlike BlockScaledTilePayload, this only stores A-scales in SMEM.
    B-scales are read directly from global memory during the epilogue phase.

    Parameters:
        a_type: Element type of the A operand tiles.
        b_type: Element type of the B operand tiles.
        a_scales_type: Element type of the A-scales tiles.
        a_shape: A tile dimensions as `(BM, BK)`.
        b_shape: B tile dimensions as `(BN, BK)`.
        a_scales_shape: A-scales tile dimensions as `(1, BM)`.
        num_pipeline_stages: Number of pipeline buffer stages.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_shape[0],
        Self.a_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_shape[0],
        Self.b_shape[1],
        Self.num_pipeline_stages,
        128,
    ]
    # A-scales are 1D vectors (1 x BM) - use row_major, NOT swizzled
    comptime AScalesTileArray = SMemTileArray2DRowMajor[
        Self.a_scales_type,
        Self.a_scales_shape[0],
        Self.a_scales_shape[1],
        Self.num_pipeline_stages,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile
    comptime AScalesTile = Self.AScalesTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var a_scales_tiles: Self.AScalesTileArray

    @always_inline
    def __init__(
        out self,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        a_scales_tiles: Self.AScalesTileArray,
    ):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles
        self.a_scales_tiles = a_scales_tiles

    @always_inline
    def get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[
        Self.ATile, Self.BTile, Self.AScalesTile
    ]:
        """Get A, B, A-scales tiles at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (
            self.a_tiles[idx],
            self.b_tiles[idx],
            self.a_scales_tiles[idx],
        )

    @always_inline
    def get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    def get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    def get_a_scales_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.AScalesTile:
        """Get A-scales tile at the specified stage and k-group index.

        Parameters:
            k_group_size: Number of K-group tiles stored per pipeline stage.

        Args:
            stage: Pipeline stage index into the circular buffer.
            k_idx: K-group index within the current pipeline stage.
        """
        return self.a_scales_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]


# ============================================================================
# InputTilePipeline - Generic pipeline parameterized by payload type
# ============================================================================


struct InputTilePipeline[
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Tile pipeline with configurable payload type.

    Separates synchronization from tile storage. The Payload parameter
    (e.g., StandardTilePayload or BlockScaledTilePayload) holds tile arrays.

    Parameters:
        Payload: Tile payload type holding the tile arrays.
        num_group_stages: Number of synchronization stages in the pipeline.
        k_group_size: Number of K-group tiles stored per synchronization stage.
    """

    comptime Pipeline = ProducerConsumerPipeline[Self.num_group_stages]
    comptime BarrierArray = SMemArray[
        SharedMemBarrier, Self.num_group_stages * 2
    ]

    var pipeline: Self.Pipeline
    var payload: Self.Payload

    @staticmethod
    @always_inline
    def init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread.

        Args:
            storage_ptr: Pointer to shared memory barrier storage.
            producer_arv_count: Number of producer threads expected to arrive
                at each barrier.
            consumer_arv_count: Number of consumer threads expected to arrive
                at each barrier.
        """
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    @always_inline
    def __init__(out self, barriers: Self.BarrierArray, payload: Self.Payload):
        """Initialize from typed barrier array and payload.

        Args:
            barriers: Shared memory barrier array for producer-consumer
                synchronization.
            payload: Tile payload holding the tile arrays.
        """
        self.pipeline = Self.Pipeline(barriers.ptr)
        self.payload = payload

    @always_inline
    def _acquire_producer(mut self) -> Tuple[UInt32, MbarPtr]:
        """Wait for slot availability and return (stage, barrier)."""
        # SM100 warp-specialized matmul: hardware-suspend the producer warp
        # while blocked instead of busy-spinning the wait loop (BRA/SYNCS).
        # See SM100_PIPELINE_WAIT_TICKS.
        self.pipeline.wait_consumer[ticks=SM100_PIPELINE_WAIT_TICKS]()
        var stage = self.pipeline.producer_stage()
        return (stage, self.pipeline.producer_mbar(stage))

    @always_inline
    def _release_producer(mut self):
        """Signal completion and advance producer stage."""
        self.pipeline.producer_step()

    @always_inline
    def _acquire_consumer(mut self) -> Tuple[UInt32, MbarPtr]:
        """Wait for data availability and return (stage, barrier)."""
        # SM100 warp-specialized matmul: hardware-suspend the consumer warp
        # while blocked instead of busy-spinning the wait loop (BRA/SYNCS).
        # See SM100_PIPELINE_WAIT_TICKS.
        self.pipeline.wait_producer[ticks=SM100_PIPELINE_WAIT_TICKS]()
        var stage = self.pipeline.consumer_stage()
        return (stage, self.pipeline.consumer_mbar(stage))

    @always_inline
    def _release_consumer(mut self):
        """Signal completion and advance consumer stage."""
        self.pipeline.consumer_step()

    # ========== Try-Acquire Pattern Methods ==========
    # These enable overlapping barrier checks with useful work.

    @always_inline
    def try_acquire_producer(self) -> Bool:
        """Non-blocking check if next producer stage is available.

        Returns:
            True if consumer has freed the stage, False otherwise.

        Example (TMA Load warp):
            ```
            var ready = pipeline.try_acquire_producer()
            # ... do other work while potentially waiting ...
            pipeline.wait_producer_if_needed(ready)
            var stage = pipeline.producer_stage()
            # ... load tiles ...
            ```
        """
        return self.pipeline.try_wait_consumer()

    @always_inline
    def try_acquire_consumer(self) -> Bool:
        """Non-blocking check if next consumer stage has data.

        Returns:
            True if producer has filled the stage, False otherwise.

        Example (MMA warp):
            ```
            var ready = pipeline.try_acquire_consumer()
            # ... do other work while potentially waiting ...
            pipeline.wait_consumer_if_needed(ready)
            var stage = pipeline.consumer_stage()
            # ... process tiles ...
            ```
        """
        return self.pipeline.try_wait_producer()

    @always_inline
    def wait_producer_if_needed(self, already_ready: Bool):
        """Conditionally wait for producer stage if not already ready.

        Args:
            already_ready: Result from try_acquire_consumer().
        """
        self.pipeline.wait_producer_if_needed(already_ready)

    @always_inline
    def wait_consumer_if_needed(self, already_ready: Bool):
        """Conditionally wait for consumer to free stage if not already ready.

        Args:
            already_ready: Result from try_acquire_producer().
        """
        self.pipeline.wait_consumer_if_needed(already_ready)

    @always_inline
    def producer_stage(self) -> UInt32:
        return self.pipeline.producer_stage()

    @always_inline
    def consumer_stage(self) -> UInt32:
        return self.pipeline.consumer_stage()

    @always_inline
    def producer_mbar(self, stage: UInt32) -> MbarPtr:
        return self.pipeline.producer_mbar(stage)

    @always_inline
    def consumer_mbar(self, stage: UInt32) -> MbarPtr:
        return self.pipeline.consumer_mbar(stage)

    @always_inline
    def producer[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputProducer[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Get producer view for TMA Load warp.

        Parameters:
            mut_origin: Origin of the pipeline reference.
        """
        return InputProducer(pipeline_ptr=Pointer(to=self))

    @always_inline
    def consumer[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputConsumer[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Get consumer view for MMA warp.

        Parameters:
            mut_origin: Origin of the pipeline reference.
        """
        return InputConsumer(pipeline_ptr=Pointer(to=self))

    # =========================================================================
    # Linear Type API - Compiler-enforced resource management
    # =========================================================================

    @always_inline
    def acquire_producer[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputProducerStage[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire a producer stage handle (linear type).

        Waits for the consumer to free the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var tiles = pipeline.acquire_producer()
            load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
            tiles^.release()  # Advances to next stage

        Parameters:
            mut_origin: Origin of the pipeline reference.

        Returns:
            An InputProducerStage handle that must be released.
        """
        var stage, barrier = self._acquire_producer()
        return InputProducerStage(
            pipeline_ptr=Pointer(to=self), stage=stage, barrier=barrier
        )

    @always_inline
    def acquire_consumer[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputConsumerStage[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire a consumer stage handle (linear type).

        Waits for the producer to fill the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var tiles = pipeline.acquire_consumer()
            process_tiles(tiles.payload(), tiles.stage())
            tiles^.release()  # Signals complete and advances

        Parameters:
            mut_origin: Origin of the pipeline reference.

        Returns:
            An InputConsumerStage handle that must be released.
        """
        var stage, mbar = self._acquire_consumer()
        return InputConsumerStage(
            pipeline_ptr=Pointer(to=self), stage=stage, mbar=mbar
        )

    @always_inline
    def drain_producer(mut self):
        """Drain pipeline to prevent CTA exit while peer is still working.

        Call this after all producer iterations are complete.
        This is the linear type equivalent of InputProducer.drain().
        """

        comptime for _ in range(Self.num_group_stages):
            self.pipeline.wait_consumer()
            self.pipeline.producer_step()


# ============================================================================
# InputProducerStage/InputConsumerStage - Linear types for tile access
# ============================================================================
#
# These are @explicit_destroy linear types: the compiler enforces that every
# instance is consumed by calling release(). Two usage patterns:
#
# 1. Linear Type API (flat, explicit):
#    var tiles = input_pipeline.acquire_producer()
#    load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
#    tiles^.release()  # Compiler enforces this call
#
# 2. Context Manager API (scoped, automatic) via wrapper types:
#    with producer.acquire() as tiles:
#        load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
#    # release() called automatically by ProducerTiles.__exit__
#
# ============================================================================


@explicit_destroy("Must call release()")
struct InputProducerStage[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](Deinitable where False, Movable):
    """Linear type handle for producer tile access.

    Compiler-enforced: must call release() to advance the producer stage.

    Usage (linear):
        var tiles = producer.acquire_stage()
        load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
        tiles^.release()  # Compiler enforces this call

    For scoped/automatic usage, use ProducerTiles via
    producer.acquire() instead.

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]
    var _stage: UInt32
    var _barrier: MbarPtr

    @always_inline
    def __init__(
        out self,
        pipeline_ptr: Pointer[Self.PipelineType, Self.origin],
        stage: UInt32,
        barrier: MbarPtr,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._barrier = barrier

    @always_inline
    def payload(self) -> Self.Payload:
        """Get the tile payload for direct access."""
        return self.pipeline_ptr[].payload

    @always_inline
    def stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage

    @always_inline
    def expect_bytes(self, num_bytes: Int):
        """Set expected bytes on the barrier for TMA loads.

        Args:
            num_bytes: Number of bytes the TMA load writes before signaling
                the barrier.
        """
        self._barrier[0].expect_bytes(Int32(num_bytes))

    @always_inline
    def barrier(self) -> MbarPtr:
        """Get the barrier pointer for TMA multicast loads."""
        return self._barrier

    @always_inline
    def release(deinit self):
        """Advance producer to next stage.

        This is the only way to destroy this linear type.
        The compiler will error if you don't call this.
        """
        self.pipeline_ptr[]._release_producer()


struct ProducerTiles[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Context manager for producing one input pipeline stage.

    Provides the same accessor interface as InputProducerStage
    (payload, stage, barrier, expect_bytes) but automatically releases
    the producer on scope exit.

    Usage:
        with producer.acquire() as tiles:
            tiles.expect_bytes(num_bytes)
            load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
        # release called automatically

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]
    var _stage: UInt32
    var _barrier: MbarPtr

    @always_inline
    def __init__(
        out self,
        pipeline_ptr: Pointer[Self.PipelineType, Self.origin],
        stage: UInt32,
        barrier: MbarPtr,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._barrier = barrier

    @always_inline
    def payload(self) -> Self.Payload:
        """Get the tile payload for direct access."""
        return self.pipeline_ptr[].payload

    @always_inline
    def stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage

    @always_inline
    def expect_bytes(self, num_bytes: Int):
        """Set expected bytes on the barrier for TMA loads.

        Args:
            num_bytes: Number of bytes the TMA load writes before signaling
                the barrier.
        """
        self._barrier[0].expect_bytes(Int32(num_bytes))

    @always_inline
    def barrier(self) -> MbarPtr:
        """Get the barrier pointer for TMA multicast loads."""
        return self._barrier

    @always_inline
    def __enter__(self) -> Self:
        return self

    @always_inline
    def __exit__(self):
        """Release the producer (advances to next stage)."""
        self.pipeline_ptr[]._release_producer()


@explicit_destroy("Must call release()")
struct InputConsumerStage[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](Deinitable where False, Movable):
    """Linear type handle for consumer tile access.

    Compiler-enforced: must call release() to signal consumption and
    advance the consumer stage.

    Usage (linear):
        var tiles = pipeline.acquire_consumer()
        process_tiles(tiles.payload(), tiles.stage())
        tiles^.release()  # Compiler enforces this call

    For scoped/automatic usage, use ConsumerTiles via
    consumer.acquire() instead.

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]
    var _stage: UInt32
    var _mbar: MbarPtr

    @always_inline
    def __init__(
        out self,
        pipeline_ptr: Pointer[Self.PipelineType, Self.origin],
        stage: UInt32,
        mbar: MbarPtr,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._mbar = mbar

    @always_inline
    def payload(self) -> Self.Payload:
        """Get the tile payload for direct access."""
        return self.pipeline_ptr[].payload

    @always_inline
    def stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage

    @always_inline
    def mbar(self) -> MbarPtr:
        """Get the barrier pointer."""
        return self._mbar

    @always_inline
    def release(deinit self):
        """Signal consumption and advance to next stage.

        This is the only way to destroy this linear type.
        The compiler will error if you don't call this.
        """
        self.pipeline_ptr[]._release_consumer()


struct ConsumerTiles[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Context manager for consuming one input pipeline stage.

    Provides the same accessor interface as InputConsumerStage
    (payload, stage, mbar) but automatically releases the consumer
    on scope exit.

    Usage:
        with consumer.acquire() as tiles:
            process_tiles(tiles.payload(), tiles.stage())
        # release called automatically

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]
    var _stage: UInt32
    var _mbar: MbarPtr

    @always_inline
    def __init__(
        out self,
        pipeline_ptr: Pointer[Self.PipelineType, Self.origin],
        stage: UInt32,
        mbar: MbarPtr,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._mbar = mbar

    @always_inline
    def payload(self) -> Self.Payload:
        """Get the tile payload for direct access."""
        return self.pipeline_ptr[].payload

    @always_inline
    def stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage

    @always_inline
    def mbar(self) -> MbarPtr:
        """Get the barrier pointer."""
        return self._mbar

    @always_inline
    def __enter__(self) -> Self:
        return self

    @always_inline
    def __exit__(self):
        """Release the consumer (signals and advances to next stage)."""
        self.pipeline_ptr[]._release_consumer()


# ============================================================================
# InputProducer/InputConsumer - Role handles with acquire() for InputTilePipeline
# ============================================================================


@fieldwise_init
struct InputProducer[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Producer view for TMA Load warp. Use acquire() to get stages.

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]

    @always_inline
    def __enter__(mut self) -> Self:
        return self

    @always_inline
    def __exit__(mut self):
        pass

    @always_inline
    def drain(mut self):
        """Drain pipeline to prevent CTA exit while peer is still working."""

        comptime for _ in range(Self.num_group_stages):
            self.pipeline_ptr[].pipeline.wait_consumer()
            self.pipeline_ptr[].pipeline.producer_step()

    @always_inline
    def acquire(
        mut self,
    ) -> ProducerTiles[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire next stage, waiting for slot availability.

        Returns a context manager for loading tiles.
        """
        var stage_idx, barrier = self.pipeline_ptr[]._acquire_producer()
        return ProducerTiles(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage_idx,
            barrier=barrier,
        )

    @always_inline
    def acquire_stage(
        mut self,
    ) -> InputProducerStage[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire next stage as a linear type, waiting for slot availability.

        Returns a compiler-enforced linear type that must be explicitly
        released via `tiles^.release()`.
        """
        var stage_idx, barrier = self.pipeline_ptr[]._acquire_producer()
        return InputProducerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage_idx,
            barrier=barrier,
        )

    @always_inline
    def try_acquire(mut self) -> Bool:
        """Non-blocking check if next producer stage is available.

        Returns:
            True if the stage is ready, False if waiting is needed.

        Use with acquire_if_needed() for the try-acquire pattern:
        ```
        var ready = producer.try_acquire()
        # ... do other work ...
        with producer.acquire_if_needed(ready) as tiles:
            load_tiles()
        ```
        """
        return self.pipeline_ptr[].try_acquire_producer()

    @always_inline
    def acquire_if_needed(
        mut self, already_ready: Bool
    ) -> ProducerTiles[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire stage, only waiting if not already ready.

        Args:
            already_ready: Result from try_acquire(). Skips wait if True.

        Returns:
            Context manager wrapping the producer stage.
        """
        self.pipeline_ptr[].wait_consumer_if_needed(already_ready)
        var stage_idx = self.pipeline_ptr[].producer_stage()
        var barrier = self.pipeline_ptr[].producer_mbar(stage_idx)
        return ProducerTiles(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage_idx,
            barrier=barrier,
        )


@fieldwise_init
struct InputConsumer[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Consumer view for MMA warp. Use acquire() to get stages.

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]

    @always_inline
    def __enter__(mut self) -> Self:
        return self

    @always_inline
    def __exit__(mut self):
        pass

    @always_inline
    def acquire(
        mut self,
    ) -> ConsumerTiles[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire next stage, waiting for tiles to be ready.

        Returns a context manager for processing tiles.
        """
        var stage_idx, mbar = self.pipeline_ptr[]._acquire_consumer()
        return ConsumerTiles(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, mbar=mbar
        )

    @always_inline
    def acquire_stage(
        mut self,
    ) -> InputConsumerStage[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire next stage as a linear type, waiting for tiles to be ready.

        Returns a compiler-enforced linear type that must be explicitly
        released via `tiles^.release()`.
        """
        var stage_idx, mbar = self.pipeline_ptr[]._acquire_consumer()
        return InputConsumerStage(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, mbar=mbar
        )

    @always_inline
    def try_acquire(mut self) -> Bool:
        """Non-blocking check if next consumer stage has data.

        Returns:
            True if the stage has data, False if waiting is needed.

        Use with acquire_if_needed() for the try-acquire pattern:
        ```
        var ready = consumer.try_acquire()
        # ... do other work ...
        with consumer.acquire_if_needed(ready) as tiles:
            process_tiles()
        ```
        """
        return self.pipeline_ptr[].try_acquire_consumer()

    @always_inline
    def acquire_if_needed(
        mut self, already_ready: Bool
    ) -> ConsumerTiles[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire stage, only waiting if not already ready.

        Args:
            already_ready: Result from try_acquire(). Skips wait if True.

        Returns:
            Context manager wrapping the consumer stage.
        """
        self.pipeline_ptr[].wait_producer_if_needed(already_ready)
        var stage_idx = self.pipeline_ptr[].consumer_stage()
        var mbar = self.pipeline_ptr[].consumer_mbar(stage_idx)
        return ConsumerTiles(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, mbar=mbar
        )


struct OutputStage[
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Acquired output stage with TMEM handle and pipeline reference.

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime num_stages = Self.opc.num_stages
    comptime stage_stride = Self.opc.stage_stride_cols
    comptime cta_group = Self.opc.cta_group

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]
    comptime Tmem = TmemStage[Self.opc]

    var index: UInt32
    var tmem: Self.Tmem
    var pipeline: Self.Pipeline

    @always_inline
    def __init__(
        out self,
        index: UInt32,
        tmem: Self.Tmem,
        pipeline: Self.Pipeline,
    ):
        self.index = index
        self.tmem = tmem
        self.pipeline = pipeline

    @staticmethod
    @always_inline
    def from_raw(
        pipeline: Self.Pipeline,
        stage_index: UInt32,
        tmem_offset: UInt32,
    ) -> Self:
        """Create OutputStage from raw pipeline, stage index, and TMEM offset.

        Useful when not using OutputTilePipeline's consumer() context manager.

        Args:
            pipeline: The ProducerConsumerPipeline for barrier signaling.
            stage_index: Current pipeline stage index.
            tmem_offset: Pre-computed TMEM offset for this stage.

        Returns:
            OutputStage with the given parameters.
        """
        var tmem = Self.Tmem.from_offset(Int(tmem_offset), Int(stage_index))
        return Self(stage_index, tmem, pipeline)


struct OutputTilePipeline[
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Pipeline for MMA→Epilogue TMEM stage synchronization.

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime num_stages = Self.opc.num_stages
    comptime stage_stride_cols = Self.opc.stage_stride_cols
    comptime cta_group = Self.opc.cta_group

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]
    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime Stage = OutputStage[Self.opc]

    var pipeline: Self.Pipeline
    var tmem: Self.Tmem
    var mma_complete_mask: UInt16

    @staticmethod
    @always_inline
    def init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread.

        Args:
            storage_ptr: Pointer to shared memory barrier storage.
            producer_arv_count: Number of producer threads expected to arrive
                at each barrier.
            consumer_arv_count: Number of consumer threads expected to arrive
                at each barrier.
        """
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    @always_inline
    def __init__(
        out self,
        barriers_ptr: MbarPtr,
        tmem: Self.Tmem,
        mma_complete_mask: UInt16,
    ):
        """Initialize from barrier pointer, TMEM allocation, and multicast mask.

        Args:
            barriers_ptr: Pointer to shared memory barrier storage.
            tmem: TMEM allocation backing the MMA accumulator stages.
            mma_complete_mask: Multicast mask used by `mma_arrive_multicast`
                on 2-SM configurations.
        """
        self.pipeline = Self.Pipeline(barriers_ptr)
        self.tmem = tmem
        self.mma_complete_mask = mma_complete_mask

    @always_inline
    def acquire_for_mma(self) -> Self.Stage:
        """Acquire stage for MMA, waiting for epilogue to finish."""
        var idx = self.pipeline.producer_stage()
        self.pipeline.wait_consumer()
        var tmem = Self.Stage.Tmem(self.tmem, Int(idx))
        return Self.Stage(idx, tmem, self.pipeline)

    @always_inline
    def release_from_mma(mut self, stage: Self.Stage):
        """Signal MMA completion using mma_arrive (1-SM) or multicast (2-SM).

        Args:
            stage: The acquired output stage to signal completion for.
        """
        from max.gpu.primitives.cluster import elect_one_sync
        from max.gpu.compute.arch.mma_nvidia_sm100 import (
            mma_arrive,
            mma_arrive_multicast,
        )

        if elect_one_sync():
            comptime if Self.cta_group == 1:
                mma_arrive[Self.cta_group](
                    self.pipeline.producer_mbar(stage.index)
                )
            else:
                mma_arrive_multicast[Self.cta_group](
                    self.pipeline.producer_mbar(stage.index),
                    self.mma_complete_mask,
                )
        self.pipeline.producer_step()

    @always_inline
    def acquire_for_epilogue(self) -> Self.Stage:
        """Acquire stage for epilogue, waiting for MMA to complete."""
        var idx = self.pipeline.consumer_stage()
        self.pipeline.wait_producer()
        var tmem = Self.Stage.Tmem(self.tmem, Int(idx))
        return Self.Stage(idx, tmem, self.pipeline)

    @always_inline
    def release_from_epilogue(mut self):
        """Signal epilogue completion, freeing stage for MMA reuse."""
        self.pipeline.consumer_step()

    @always_inline
    def producer[
        origin: MutOrigin, //
    ](ref[origin] self) -> OutputProducer[origin, Self.opc]:
        """Get producer view for MMA warp.

        Parameters:
            origin: Origin of the pipeline reference.
        """
        return OutputProducer(Pointer(to=self))

    @always_inline
    def consumer[
        origin: MutOrigin, //
    ](ref[origin] self) -> OutputConsumer[origin, Self.opc]:
        """Get consumer view for epilogue warp.

        Parameters:
            origin: Origin of the pipeline reference.
        """
        return OutputConsumer(Pointer(to=self))

    # =========================================================================
    # Linear Type API - Compiler-enforced resource management
    # =========================================================================

    @always_inline
    def acquire_mma_linear[
        origin: MutOrigin, //
    ](ref[origin] self) -> MmaStage[origin, Self.opc]:
        """Acquire a stage for MMA using linear types.

        Waits for the epilogue to free the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var stage = output_pipeline.acquire_mma_linear()
            mma_op.mma(a_tile, b_tile, stage.tmem_offset())
            mma_op.commit(stage.mbar())
            stage^.release()  # Signals mma_arrive and advances

        Parameters:
            origin: Origin of the pipeline reference.

        Returns:
            An MmaStage handle that must be released.
        """
        var stage = self.acquire_for_mma()
        return MmaStage(Pointer(to=self), stage)

    @always_inline
    def acquire_epilogue_linear[
        origin: MutOrigin, //
    ](ref[origin] self) -> EpilogueStage[origin, Self.opc]:
        """Acquire a stage for epilogue using linear types.

        Waits for MMA to complete the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var stage = output_pipeline.acquire_epilogue_linear()
            process_tmem(stage.tmem())
            stage^.release()  # Advances to next stage

        Parameters:
            origin: Origin of the pipeline reference.

        Returns:
            An EpilogueStage handle that must be released.
        """
        var stage = self.acquire_for_epilogue()
        return EpilogueStage(Pointer(to=self), stage)

    @always_inline
    def get_pipeline(self) -> Self.Pipeline:
        """Get underlying pipeline (used during barrier initialization)."""
        return self.pipeline

    @always_inline
    def per_k[
        origin: MutOrigin, //
    ](ref[origin] self) -> OutputKPipeline[origin, Self.opc]:
        """Get per-K-iteration view for kernels with per-K signaling.

        Unlike producer()/consumer() which signal once per tile (after all K
        iterations), this view signals after each K iteration. Use for kernels
        with per-K accumulation patterns (e.g., blockwise FP8).

        Parameters:
            origin: Origin of the pipeline reference.

        Returns:
            OutputKPipeline view that provides produce()/consume() context
            managers for per-K-iteration barrier signaling.
        """
        return OutputKPipeline(Pointer(to=self))

    @always_inline
    def per_k_epilogue[
        output_origin: MutOrigin,
        input_origin: MutOrigin,
        num_input_stages: Int,
    ](
        ref[output_origin] self,
        ref[input_origin] input_pipeline: ProducerConsumerPipeline[
            num_input_stages
        ],
    ) -> EpilogueKContext[
        output_origin,
        input_origin,
        Self.opc,
        num_input_stages,
    ]:
        """Get combined per-K epilogue context for blockwise FP8.

        Bundles output pipeline (MMA->Epilogue sync) and input pipeline
        (A-scales consumption) into a single context manager.

        Example:
            for k_iter in range(num_iters):
                with output_pipeline.per_k_epilogue(input_pipeline) as stage:
                    accum.promote(stage, ...)
            # Both pipelines signaled automatically

        Parameters:
            output_origin: Origin of the output pipeline reference.
            input_origin: Origin of the input pipeline reference.
            num_input_stages: Number of stages in the input (A-scales) pipeline.

        Args:
            input_pipeline: The input pipeline for A-scales consumption.

        Returns:
            EpilogueKContext context manager that handles both pipelines.
        """
        return EpilogueKContext(Pointer(to=self), Pointer(to=input_pipeline))


struct OutputProducer[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Producer view for MMA warp (output pipeline).

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime num_stages = Self.opc.num_stages

    comptime TilePipelineType = OutputTilePipeline[Self.opc]
    comptime Stage = OutputStage[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var stage: Self.Stage

    @always_inline
    def __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr
        # Placeholder stage - set properly in __enter__
        var placeholder_tmem = Self.Stage.Tmem(0, 0)
        # SAFETY: Placeholder pipeline; replaced by acquire_for_mma() in
        # __enter__ before any method accesses the mbar pointer.
        self.stage = Self.Stage(
            0,
            placeholder_tmem,
            ProducerConsumerPipeline[Self.num_stages](
                MbarPtr.unsafe_dangling()
            ),
        )

    @always_inline
    def __enter__(mut self) -> Self.Stage:
        self.stage = self.pipeline_ptr[].acquire_for_mma()
        return self.stage

    @always_inline
    def __exit__(mut self):
        self.pipeline_ptr[].release_from_mma(self.stage)


struct OutputConsumer[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Consumer view for epilogue warp (output pipeline).

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime TilePipelineType = OutputTilePipeline[Self.opc]
    comptime Stage = OutputStage[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]

    @always_inline
    def __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr

    @always_inline
    def __enter__(mut self) -> Self.Stage:
        return self.pipeline_ptr[].acquire_for_epilogue()

    @always_inline
    def __exit__(mut self):
        self.pipeline_ptr[].release_from_epilogue()


# =============================================================================
# Unified Linear Types for OutputTilePipeline
# =============================================================================
#
# These types work both as linear types (direct use) and within context managers.
#
# Linear Type API (flat):
#     var stage = output_pipeline.acquire_mma_linear()
#     mma_op.mma(a_tile, b_tile, stage.tmem_offset())
#     stage^.release()
#
# Context Manager API (scoped):
#     with output_pipeline.producer() as stage:
#         mma_op.mma(a_tile, b_tile, stage.tmem.offset())
#


@explicit_destroy("Must call release() to signal MMA completion and advance")
struct MmaStage[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](Deinitable where False):
    """Unified linear type handle for MMA stage in output pipeline.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `output_pipeline.acquire_mma_linear()` - waits for epilogue
    2. Use `tmem()`, `tmem_offset()`, `mbar()` for MMA operations
    3. Must call `release()` to signal mma_arrive and advance (compiler-enforced)

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime TilePipelineType = OutputTilePipeline[Self.opc]
    comptime Stage = OutputStage[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: Self.Stage

    @always_inline
    def __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: Self.Stage,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage

    @always_inline
    def tmem(self) -> Self.Stage.Tmem:
        """Get the TMEM stage handle."""
        return self._stage.tmem

    @always_inline
    def tmem_offset(self) -> Int:
        """Get the TMEM offset for MMA accumulator."""
        return self._stage.tmem.offset()

    @always_inline
    def index(self) -> UInt32:
        """Get the current stage index."""
        return self._stage.index

    @always_inline
    def mbar(self) -> MbarPtr:
        """Get the producer barrier for MMA commit."""
        return self.pipeline_ptr[].pipeline.producer_mbar(self._stage.index)

    @always_inline
    def release(deinit self):
        """Signal MMA completion and advance to next stage.

        This is the only way to destroy this linear type.
        Internally calls mma_arrive (1-SM) or mma_arrive_multicast (2-SM).
        """
        self.pipeline_ptr[].release_from_mma(self._stage)


@explicit_destroy("Must call release() to free stage for MMA reuse")
struct EpilogueStage[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](Deinitable where False):
    """Unified linear type handle for epilogue stage in output pipeline.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `output_pipeline.acquire_epilogue_linear()` - waits for MMA
    2. Use `tmem()`, `tmem_offset()` for reading MMA results
    3. Must call `release()` to advance (compiler-enforced)

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime TilePipelineType = OutputTilePipeline[Self.opc]
    comptime Stage = OutputStage[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: Self.Stage

    @always_inline
    def __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: Self.Stage,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage

    @always_inline
    def tmem(self) -> Self.Stage.Tmem:
        """Get the TMEM stage handle."""
        return self._stage.tmem

    @always_inline
    def tmem_offset(self) -> Int:
        """Get the TMEM offset for reading MMA results."""
        return self._stage.tmem.offset()

    @always_inline
    def index(self) -> UInt32:
        """Get the current stage index."""
        return self._stage.index

    @always_inline
    def release(deinit self):
        """Free stage for MMA reuse and advance to next stage.

        This is the only way to destroy this linear type.
        """
        self.pipeline_ptr[].release_from_epilogue()


# =============================================================================
# Per-K Output Pipeline Views
# =============================================================================
# These types provide per-K-iteration signaling for kernels that need to
# signal after each K iteration rather than once per tile (e.g., blockwise FP8).


struct OutputKPipeline[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Per-K-iteration view of OutputTilePipeline.

    Unlike standard producer()/consumer() which signal once per tile (after
    all K iterations), this view signals after each K iteration. Use for
    kernels with per-K accumulation patterns (e.g., blockwise FP8).

    Example (MMA warp):
        for i in range(num_iters):
            with mma_ctx.output_pipeline.per_k().produce() as stage:
                mma(stage.tmem, ...)
            # __exit__ signals mma_arrive for this K iteration

    Example (Epilogue warp):
        for k_iter in range(num_iters):
            with epi_ctx.output_pipeline.per_k().consume() as stage:
                promote(stage.tmem, ...)
            # __exit__ signals consumer_step for this K iteration

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime TilePipelineType = OutputTilePipeline[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]

    @always_inline
    def __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr

    @always_inline
    def produce(
        self,
    ) -> MmaKStage[Self.origin, Self.opc]:
        """Get MMA stage context manager for one K iteration.

        Returns:
            Context manager that acquires stage on enter and signals
            mma_arrive on exit.
        """
        return MmaKStage(self.pipeline_ptr)

    @always_inline
    def consume(
        self,
    ) -> PerKConsumerStage[Self.origin, Self.opc]:
        """Get consumer context manager for one K iteration.

        Returns:
            Context manager that waits for MMA on enter and signals
            consumer_step on exit.
        """
        return PerKConsumerStage(self.pipeline_ptr)


struct MmaKStage[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Per-K stage context for MMA warp in blockwise FP8.

    __enter__: Acquires stage, waits for epilogue to release previous stage
    __exit__: Signals mma_arrive to notify epilogue, advances producer stage

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime num_stages = Self.opc.num_stages

    comptime TilePipelineType = OutputTilePipeline[Self.opc]
    comptime Stage = OutputStage[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var stage: Self.Stage

    @always_inline
    def __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr
        # Placeholder stage - set properly in __enter__
        var placeholder_tmem = Self.Stage.Tmem(0, 0)
        # SAFETY: Placeholder pipeline; replaced by acquire_for_mma() in
        # __enter__ before any method accesses the mbar pointer.
        self.stage = Self.Stage(
            0,
            placeholder_tmem,
            ProducerConsumerPipeline[Self.num_stages](
                MbarPtr.unsafe_dangling()
            ),
        )

    @always_inline
    def __enter__(mut self) -> Self.Stage:
        self.stage = self.pipeline_ptr[].acquire_for_mma()
        return self.stage

    @always_inline
    def __exit__(mut self):
        self.pipeline_ptr[].release_from_mma(self.stage)


struct PerKConsumerStage[
    origin: MutOrigin,
    opc: OutputPipelineConfig,
](TrivialRegisterPassable):
    """Context manager for per-K epilogue consumption.

    __enter__: Acquires stage, waits for MMA to complete this K iteration
    __exit__: Signals consumer barrier to release stage for MMA reuse

    IMPORTANT: Unlike standard per-tile consumption, per-K consumption must
    signal the consumer barrier explicitly. The MMA warp waits on this barrier
    before each K iteration, so we must signal after each K iteration.

    Parameters:
        origin: Origin of the pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
    """

    comptime cta_group = Self.opc.cta_group
    comptime num_stages = Self.opc.num_stages

    comptime TilePipelineType = OutputTilePipeline[Self.opc]
    comptime Stage = OutputStage[Self.opc]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var stage: Self.Stage

    @always_inline
    def __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr
        # Placeholder stage - set properly in __enter__
        var placeholder_tmem = Self.Stage.Tmem(0, 0)
        # SAFETY: Placeholder pipeline; replaced by acquire_for_epilogue() in
        # __enter__ before any method accesses the mbar pointer.
        self.stage = Self.Stage(
            0,
            placeholder_tmem,
            ProducerConsumerPipeline[Self.num_stages](
                MbarPtr.unsafe_dangling()
            ),
        )

    @always_inline
    def __enter__(mut self) -> Self.Stage:
        self.stage = self.pipeline_ptr[].acquire_for_epilogue()
        return self.stage

    @always_inline
    def __exit__(mut self):
        # Signal the consumer barrier to tell MMA we're done with this stage.
        # This is critical for per-K synchronization - MMA waits on this
        # barrier before each K iteration.
        from max.gpu.sync import mbarrier_arrive, umma_arrive_leader_cta

        comptime if Self.cta_group == 1:
            _ = mbarrier_arrive(
                self.pipeline_ptr[].pipeline.consumer_mbar(self.stage.index)
            )
        else:
            umma_arrive_leader_cta(
                self.pipeline_ptr[].pipeline.consumer_mbar(self.stage.index)
            )

        self.pipeline_ptr[].release_from_epilogue()


# =============================================================================
# Epilogue Per-K Stage (for blockwise FP8)
# =============================================================================


struct EpilogueKStage[
    opc: OutputPipelineConfig,
    num_input_stages: Int,
](TrivialRegisterPassable):
    """Per-K stage for epilogue warp in blockwise FP8.

    Returned from `EpilogueKContext.__enter__()`. Bundles:
    - output_stage: TMEM access (offset for reading MMA results)
    - input_stage_index: Current A-scales stage
    - input_pipeline: For signaling A-scales consumption

    Parameters:
        opc: Output pipeline configuration (stages, stride, cta_group).
        num_input_stages: Number of stages in the input (A-scales) pipeline.
    """

    comptime OutputStageType = OutputStage[Self.opc]
    comptime InputPipelineType = ProducerConsumerPipeline[Self.num_input_stages]

    var output_stage: Self.OutputStageType
    var input_stage_index: UInt32
    var input_pipeline: Self.InputPipelineType

    @always_inline
    def __init__(
        out self,
        output_stage: Self.OutputStageType,
        input_stage_index: UInt32,
        input_pipeline: Self.InputPipelineType,
    ):
        self.output_stage = output_stage
        self.input_stage_index = input_stage_index
        self.input_pipeline = input_pipeline

    @always_inline
    def arrive_input(self):
        """Arrive on the input pipeline's consumer barrier.

        Use with lane-guarded patterns:
            if lane_id() < cluster_size:
                epi_stage.arrive_input()
        """
        _ = self.input_pipeline.consumer_mbar(self.input_stage_index)[
            0
        ].arrive()


# =============================================================================
# Epilogue Per-K Context Manager (for blockwise FP8)
# =============================================================================


struct EpilogueKContext[
    origin: MutOrigin,
    input_origin: MutOrigin,
    opc: OutputPipelineConfig,
    num_input_stages: Int,
](TrivialRegisterPassable):
    """Per-K context manager for epilogue warp in blockwise FP8.

    Bundles output pipeline (MMA→Epilogue sync) and input pipeline (A-scales)
    into a single context manager for clean per-K iteration handling.

    Example usage:
        for k_iter in range(num_iters):
            with epi_ctx.per_k_stage(input_pipeline) as epi_stage:
                accum.promote(epi_stage, ...)
            # __exit__ signals BOTH pipelines

    __enter__: Waits for MMA to complete this K iteration, returns EpilogueKStage
    __exit__: Signals both output consumer barrier AND input consumer_step

    Parameters:
        origin: Origin of the output pipeline reference.
        input_origin: Origin of the input pipeline reference.
        opc: Output pipeline configuration (stages, stride, cta_group).
        num_input_stages: Number of stages in the input (A-scales) pipeline.
    """

    comptime num_output_stages = Self.opc.num_stages
    comptime cta_group = Self.opc.cta_group

    comptime OutputPipelineType = OutputTilePipeline[Self.opc]
    comptime OutputStageType = OutputStage[Self.opc]
    comptime InputPipelineType = ProducerConsumerPipeline[Self.num_input_stages]

    # Combined stage type returned from __enter__
    comptime CombinedStageType = EpilogueKStage[Self.opc, Self.num_input_stages]

    var output_pipeline_ptr: Pointer[Self.OutputPipelineType, Self.origin]
    var input_pipeline_ptr: Pointer[Self.InputPipelineType, Self.input_origin]
    var output_stage: Self.OutputStageType
    var input_stage_index: UInt32

    @always_inline
    def __init__(
        out self,
        output_pipeline_ptr: Pointer[Self.OutputPipelineType, Self.origin],
        input_pipeline_ptr: Pointer[Self.InputPipelineType, Self.input_origin],
    ):
        self.output_pipeline_ptr = output_pipeline_ptr
        self.input_pipeline_ptr = input_pipeline_ptr
        self.input_stage_index = 0
        # Placeholder stage - set properly in __enter__
        var placeholder_tmem = Self.OutputStageType.Tmem(0, 0)
        # SAFETY: Placeholder pipeline; replaced by acquire_for_epilogue() in
        # __enter__ before any method accesses the mbar pointer.
        self.output_stage = Self.OutputStageType(
            0,
            placeholder_tmem,
            ProducerConsumerPipeline[Self.num_output_stages](
                MbarPtr.unsafe_dangling()
            ),
        )

    @always_inline
    def __enter__(mut self) -> Self.CombinedStageType:
        self.output_stage = self.output_pipeline_ptr[].acquire_for_epilogue()
        self.input_stage_index = self.input_pipeline_ptr[].consumer_stage()
        return Self.CombinedStageType(
            self.output_stage, self.input_stage_index, self.input_pipeline_ptr[]
        )

    @always_inline
    def __exit__(mut self):
        # Signal input pipeline consumer_step (for A-scales consumption)
        self.input_pipeline_ptr[].consumer_step()

        # Signal output pipeline consumer barrier (for MMA synchronization)
        from max.gpu.sync import mbarrier_arrive, umma_arrive_leader_cta

        comptime if Self.cta_group == 1:
            _ = mbarrier_arrive(
                self.output_pipeline_ptr[].pipeline.consumer_mbar(
                    self.output_stage.index
                )
            )
        else:
            umma_arrive_leader_cta(
                self.output_pipeline_ptr[].pipeline.consumer_mbar(
                    self.output_stage.index
                )
            )

        self.output_pipeline_ptr[].release_from_epilogue()
