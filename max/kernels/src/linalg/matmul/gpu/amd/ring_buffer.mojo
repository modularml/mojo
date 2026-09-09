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
"""
Ring Buffer implementation for producer-consumer synchronization in GPU kernels.

This ring buffer coordinates data transfer between producer warps (loading from global memory)
and consumer warps (performing computation) through shared memory tiles.

Key features:
- Configurable synchronization strategies via the SyncStrategy trait
- Pipeline stages for overlapping data transfer and computation
- Context managers for automatic acquire/release of tiles
- Phase-based synchronization to prevent data races
"""

from layout import Layout
from std.utils import StaticTuple

from .structured import SMemBuffer
from .ring_buffer_traits import (
    SyncStrategy,
    SingleCounterSync,
    SplitCounterSync,
)


# ===----------------------------------------------------------------------=== #
# Tile Context Managers
# ===----------------------------------------------------------------------=== #


struct ProducerTile[
    origin: MutOrigin,
    ring_buffer_type: type_of(RingBuffer),
    warps_processed_per_producer: Int,
](TrivialRegisterPassable):
    """Context manager for producer access to a single ring buffer tile.

    Parameters:
        origin: Memory origin for pointers to the underlying ring buffer.
        ring_buffer_type: The `RingBuffer` type accessed through this tile.
        warps_processed_per_producer: Number of warps each producer processes
            per pipeline stage.
    """

    comptime ProducerViewType = ProducerView[
        Self.origin,
        Self.ring_buffer_type,
        Self.warps_processed_per_producer,
    ]
    comptime ProducerViewPtrType = Pointer[Self.ProducerViewType, Self.origin]

    var producer_view_ptr: Self.ProducerViewPtrType
    var stage: Int
    var producer_iteration: Int  # Which iteration this producer is on
    var warp_tile_idx: Int

    @always_inline
    def __init__(
        out self,
        producer_view_ptr: Self.ProducerViewPtrType,
        stage: Int,
        producer_iteration: Int,
        warp_tile_idx: Int,
    ):
        self.producer_view_ptr = producer_view_ptr
        self.stage = stage
        self.producer_iteration = producer_iteration
        self.warp_tile_idx = warp_tile_idx

    @always_inline
    def __enter__(mut self) -> Self.ring_buffer_type.WarpTileTupleType:
        """Acquire the tile for use."""
        return self.producer_view_ptr[].acquire_tiles(
            self.stage, self.producer_iteration, self.warp_tile_idx
        )

    @always_inline
    def __exit__(mut self):
        """Release the tile back to consumers."""
        self.producer_view_ptr[].release_tiles(self.stage, self.warp_tile_idx)


struct ConsumerTile[
    origin: MutOrigin,
    ring_buffer_type: type_of(RingBuffer),
    warps_computed_per_consumer: Int,
](TrivialRegisterPassable):
    """Context manager for consumer access to a single ring buffer tile.

    Parameters:
        origin: Memory origin for pointers to the underlying ring buffer.
        ring_buffer_type: The `RingBuffer` type accessed through this tile.
        warps_computed_per_consumer: Number of warps each consumer computes
            per pipeline stage.
    """

    comptime ConsumerViewType = ConsumerView[
        Self.origin,
        Self.ring_buffer_type,
        Self.warps_computed_per_consumer,
    ]
    comptime ConsumerViewPtrType = Pointer[Self.ConsumerViewType, Self.origin]

    var consumer_view_ptr: Self.ConsumerViewPtrType
    var stage: Int
    var consumer_iteration: Int  # Which iteration this consumer is on
    var warp_tile_idx: Int

    @always_inline
    def __init__(
        out self,
        consumer_view_ptr: Self.ConsumerViewPtrType,
        stage: Int,
        consumer_iteration: Int,
        warp_tile_idx: Int,
    ):
        self.consumer_view_ptr = consumer_view_ptr
        self.stage = stage
        self.consumer_iteration = consumer_iteration
        self.warp_tile_idx = warp_tile_idx

    @always_inline
    def __enter__(mut self) -> Self.ring_buffer_type.WarpTileTupleType:
        """Acquire the tile for use."""
        return self.consumer_view_ptr[].acquire_tiles(
            self.stage, self.consumer_iteration, self.warp_tile_idx
        )

    @always_inline
    def __exit__(mut self):
        """Release the tile back to producers."""
        self.consumer_view_ptr[].release_tiles(self.stage, self.warp_tile_idx)


# ===----------------------------------------------------------------------=== #
# Producer and Consumer Views
# ===----------------------------------------------------------------------=== #


struct ProducerView[
    origin: MutOrigin,
    ring_buffer_type: type_of(RingBuffer),
    warps_processed_per_producer: Int,
](TrivialRegisterPassable):
    """Producer view of the unified ring buffer.

    Parameters:
        origin: Memory origin for pointers to the underlying ring buffer.
        ring_buffer_type: The `RingBuffer` type accessed through this view.
        warps_processed_per_producer: Number of warps each producer processes
            per pipeline stage.
    """

    comptime RingBufferPtrType = Pointer[Self.ring_buffer_type, Self.origin]

    var ring_buffer_ptr: Self.RingBufferPtrType
    var phases: StaticTuple[
        Int32,
        Self.ring_buffer_type.pipeline_stages
        * Self.warps_processed_per_producer,
    ]

    @always_inline
    def __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr
        self.phases = StaticTuple[
            Int32,
            Self.ring_buffer_type.pipeline_stages
            * Self.warps_processed_per_producer,
        ](
            fill=0
        )  # Producers start at phase 0

    @always_inline
    def __enter__(mut self) -> Self:
        """Context manager entry."""
        return self

    @always_inline
    def __exit__(mut self):
        """Context manager exit."""
        pass

    @always_inline
    def acquire_tiles(
        mut self,
        stage: Int,
        producer_iteration: Int,
        warp_tile_idx: Int,
    ) -> Self.ring_buffer_type.WarpTileTupleType:
        """Acquire tiles for writing by this producer.

        Args:
            stage: Pipeline stage to write to.
            producer_iteration: Which iteration this producer is on (`0` to
                `warps_processed_per_producer - 1`).
            warp_tile_idx: Which tile this producer is responsible for.
        """
        # Compute phase index based on stage and iteration
        var phase_idx = (
            stage * Self.warps_processed_per_producer + producer_iteration
        )
        var phase = self.phases[phase_idx]

        # Wait until consumers have finished with this tile
        self.ring_buffer_ptr[].wait_producer_acquire(
            warp_tile_idx, stage, phase
        )

        # Update phase for next pipeline cycle
        self.phases[
            phase_idx
        ] += self.ring_buffer_ptr[].get_producer_phase_increment()

        # Return the tiles from shared memory
        return self.ring_buffer_ptr[].get_tiles(stage, warp_tile_idx)

    @always_inline
    def release_tiles(mut self, stage: Int, warp_tile_idx: Int):
        """Signal to consumers that tile is ready.

        Args:
            stage: Pipeline stage of the tile to release.
            warp_tile_idx: Index of the warp tile within the stage to release.
        """
        self.ring_buffer_ptr[].signal_producer_release(warp_tile_idx, stage)

    comptime ProducerTileType = ProducerTile[
        Self.origin, Self.ring_buffer_type, Self.warps_processed_per_producer
    ]

    @always_inline
    def get_tile(
        mut self,
        stage: Int,
        warp_tile_idx: Int,
        producer_iteration: Int,
    ) -> Self.ProducerTileType:
        """Get a context manager for accessing a tile.

        Args:
            stage: Pipeline stage.
            warp_tile_idx: Which tile to access.
            producer_iteration: Current iteration of this producer.
        """
        return Self.ProducerTileType(
            rebind[Pointer[Self, Self.origin]](Pointer(to=self)),
            stage,
            producer_iteration,
            warp_tile_idx,
        )


struct ConsumerView[
    origin: MutOrigin,
    ring_buffer_type: type_of(RingBuffer),
    warps_computed_per_consumer: Int,
](TrivialRegisterPassable):
    """Consumer view of the unified ring buffer.

    Parameters:
        origin: Memory origin for pointers to the underlying ring buffer.
        ring_buffer_type: The `RingBuffer` type accessed through this view.
        warps_computed_per_consumer: Number of warps each consumer computes
            per pipeline stage.
    """

    comptime RingBufferPtrType = Pointer[Self.ring_buffer_type, Self.origin]

    var ring_buffer_ptr: Self.RingBufferPtrType
    var phases: StaticTuple[
        Int32,
        Self.ring_buffer_type.pipeline_stages
        * Self.warps_computed_per_consumer,
    ]

    @always_inline
    def __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr
        self.phases = StaticTuple[
            Int32,
            Self.ring_buffer_type.pipeline_stages
            * Self.warps_computed_per_consumer,
        ](
            fill=1
        )  # Consumers start at phase 1

    @always_inline
    def __enter__(mut self) -> Self:
        """Context manager entry."""
        return self

    @always_inline
    def __exit__(mut self):
        """Context manager exit."""
        pass

    @always_inline
    def acquire_tiles(
        mut self,
        stage: Int,
        consumer_iteration: Int,
        warp_tile_idx: Int,
    ) -> Self.ring_buffer_type.WarpTileTupleType:
        """Acquire tiles for reading by this consumer.

        Args:
            stage: Pipeline stage to read from.
            consumer_iteration: Which iteration this consumer is on (0 to warps_computed_per_consumer-1).
            warp_tile_idx: Which tile this consumer wants to read.
        """
        # Compute phase index based on stage and iteration
        var phase_idx = (
            stage * Self.warps_computed_per_consumer + consumer_iteration
        )
        var phase = self.phases[phase_idx]

        # Wait until producer has finished writing to this tile
        self.ring_buffer_ptr[].wait_consumer_acquire(
            warp_tile_idx, stage, phase
        )

        # Update phase for next pipeline cycle
        self.phases[
            phase_idx
        ] += self.ring_buffer_ptr[].get_consumer_phase_increment()

        # Return the tiles from shared memory
        return self.ring_buffer_ptr[].get_tiles(stage, warp_tile_idx)

    @always_inline
    def release_tiles(mut self, stage: Int, warp_tile_idx: Int):
        """Signal to producers that tile is free.

        Args:
            stage: Pipeline stage of the tile to release.
            warp_tile_idx: Index of the warp tile within the stage to release.
        """
        self.ring_buffer_ptr[].signal_consumer_release(warp_tile_idx, stage)

    comptime ConsumerTileType = ConsumerTile[
        Self.origin,
        Self.ring_buffer_type,
        Self.warps_computed_per_consumer,
    ]

    @always_inline
    def get_tile(
        mut self,
        stage: Int,
        consumer_iteration: Int,
        warp_tile_idx: Int,
    ) -> Self.ConsumerTileType:
        """Get a context manager for accessing a tile.

        Args:
            stage: Pipeline stage.
            consumer_iteration: Current iteration of this consumer.
            warp_tile_idx: Which tile to access.
        """
        return Self.ConsumerTileType(
            rebind[Pointer[Self, Self.origin]](Pointer(to=self)),
            stage,
            consumer_iteration,
            warp_tile_idx,
        )


# ===----------------------------------------------------------------------=== #
# Main Ring Buffer Implementation
# ===----------------------------------------------------------------------=== #


struct RingBuffer[
    dtype: DType,
    layout: Layout,
    pipeline_stages: Int,
    block_rows: Int,
    block_cols: Int,
    warp_rows: Int,
    warp_cols: Int,
    reads_per_warp_block: Int,
    tile_buffers: Int,
    sync_strategy_type: SyncStrategy,
]:
    """Ring buffer for coordinating producer-consumer warps in matrix multiplication.

    Parameters:
        dtype: Data type of elements.
        layout: Memory layout for shared memory tiles.
        pipeline_stages: Number of stages for software pipelining.
        block_rows: Number of rows in block-level tiles.
        block_cols: Number of columns in block-level tiles.
        warp_rows: Number of rows in warp-level tiles.
        warp_cols: Number of columns in warp-level tiles.
        reads_per_warp_block: How many consumer warps read each tile.
        tile_buffers: Number of separate tile buffers (usually 1).
        sync_strategy_type: Synchronization strategy (SingleCounterSync or SplitCounterSync).
    """

    comptime block_warps = Self.block_rows // Self.warp_rows
    comptime total_tiles = Self.block_warps * Self.pipeline_stages

    comptime SmemBufferType = SMemBuffer[
        Self.dtype,
        Self.layout,
        Self.pipeline_stages,
        Self.block_rows,
        Self.block_cols,
        Self.warp_rows,
        Self.warp_cols,
    ]
    comptime WarpTileType = Self.SmemBufferType.WarpTileType
    comptime SMemBuffersType = StaticTuple[
        Self.SmemBufferType, Self.tile_buffers
    ]
    comptime WarpTileTupleType = StaticTuple[
        Self.WarpTileType, Self.tile_buffers
    ]

    var smem_buffers: Self.SMemBuffersType
    var sync_strategy: Self.sync_strategy_type

    @always_inline
    def __init__(out self):
        comptime assert Self.total_tiles <= 32, (
            "total_tiles must be less than or equal to 32 for AMD atomic"
            " limitations"
        )

        var smem_buffer = Self.SmemBufferType()
        self.smem_buffers = StaticTuple[
            type_of(smem_buffer), Self.tile_buffers
        ](smem_buffer)

        # Initialize sync strategy based on type
        # We still need compile-time dispatch for the specific type
        self.sync_strategy = Self.sync_strategy_type()

    @always_inline
    def get_tiles(
        self, stage: Int, warp_tile_idx: Int
    ) -> Self.WarpTileTupleType:
        """Get tiles from shared memory.

        Args:
            stage: Pipeline stage to read tiles from.
            warp_tile_idx: Index of the warp tile within the stage to read.
        """
        var result = Self.WarpTileTupleType()

        comptime for i in range(Self.tile_buffers):
            var staged_smem_tile = self.smem_buffers[i].get_tile(stage)
            result[i] = staged_smem_tile.tile[Self.warp_rows, Self.warp_cols](
                warp_tile_idx, 0
            )
        return result

    @always_inline
    def producer[
        warps_processed_per_producer: Int
    ](
        mut self,
    ) -> ProducerView[
        origin_of(self),
        type_of(self),
        warps_processed_per_producer,
    ]:
        """Create a producer view of this ring buffer.

        Parameters:
            warps_processed_per_producer: Number of warps each producer processes
                per pipeline stage.
        """
        return ProducerView[
            origin_of(self),
            type_of(self),
            warps_processed_per_producer,
        ](Pointer(to=self))

    @always_inline
    def consumer[
        warps_computed_per_consumer: Int
    ](mut self) -> ConsumerView[
        origin_of(self),
        type_of(self),
        warps_computed_per_consumer,
    ]:
        """Create a consumer view of this ring buffer.

        Parameters:
            warps_computed_per_consumer: Number of warps each consumer
                computes per pipeline stage.
        """
        return ConsumerView[
            origin_of(self),
            type_of(self),
            warps_computed_per_consumer,
        ](Pointer(to=self))

    @always_inline
    def get_staged_idx(self, tile_idx: Int, stage: Int) -> Int:
        """Get the staged index for a tile and stage.

        Args:
            tile_idx: Index of the warp tile within the block.
            stage: Pipeline stage of the tile.
        """
        return self.sync_strategy.get_staged_idx(tile_idx, stage)

    @always_inline
    def wait_producer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        """Producer waits to acquire a tile.

        Args:
            tile_idx: Index of the warp tile within the block to acquire.
            stage: Pipeline stage of the tile to acquire.
            phase: Synchronization phase counter for this tile and stage.
        """
        self.sync_strategy.wait_producer_acquire(tile_idx, stage, phase)

    @always_inline
    def signal_producer_release(mut self, tile_idx: Int, stage: Int):
        """Producer signals it has released a tile.

        Args:
            tile_idx: Index of the warp tile within the block to release.
            stage: Pipeline stage of the tile to release.
        """
        self.sync_strategy.signal_producer_release(tile_idx, stage)

    @always_inline
    def wait_consumer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        """Consumer waits to acquire a tile.

        Args:
            tile_idx: Index of the warp tile within the block to acquire.
            stage: Pipeline stage of the tile to acquire.
            phase: Synchronization phase counter for this tile and stage.
        """
        self.sync_strategy.wait_consumer_acquire(tile_idx, stage, phase)

    @always_inline
    def signal_consumer_release(mut self, tile_idx: Int, stage: Int):
        """Consumer signals it has released a tile.

        Args:
            tile_idx: Index of the warp tile within the block to release.
            stage: Pipeline stage of the tile to release.
        """
        self.sync_strategy.signal_consumer_release(tile_idx, stage)

    @always_inline
    def get_producer_phase_increment(self) -> Int32:
        """Get the phase increment for producers."""
        return self.sync_strategy.get_producer_phase_increment()

    @always_inline
    def get_consumer_phase_increment(self) -> Int32:
        """Get the phase increment for consumers."""
        return self.sync_strategy.get_consumer_phase_increment()
