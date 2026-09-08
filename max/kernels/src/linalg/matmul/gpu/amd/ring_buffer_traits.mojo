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
Trait definitions and utilities for ring buffer synchronization strategies.

This module provides:
- SyncStrategy trait: Interface for producer-consumer synchronization protocols
- SingleCounterSync: Uses a single atomic counter per tile (original RingBuffer behavior)
- SplitCounterSync: Uses separate producer/consumer counters to reduce contention
- Atomic utility functions for thread-safe counter operations
"""

from std.math.uutils import umod
from max.gpu import thread_idx, WARP_SIZE
from linalg.structuring import SMemArray
from std.atomic import Atomic
from std.sys._assembly import inlined_assembly


# ===----------------------------------------------------------------------=== #
# Common Synchronization Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
def wait_for_counter(
    counter: UnsafePointer[mut=True, Int32, _, address_space=.SHARED],
    threshold: Int32,
):
    """Spin-wait until counter reaches threshold.

    Args:
        counter: Shared-memory atomic counter to poll.
        threshold: Value the counter must reach before returning.
    """
    while Atomic.load(counter) < threshold:
        inlined_assembly[
            "s_sleep 0", NoneType, constraints="", has_side_effect=True
        ]()


@always_inline
def increment_counter_if_first_thread(
    counter: UnsafePointer[mut=True, Int32, _, address_space=.SHARED],
    increment: Int32,
):
    """Atomically increment counter, but only from the first thread in warp.

    Args:
        counter: Shared-memory atomic counter to update.
        increment: Amount to add to the counter.
    """
    if umod(thread_idx.x, WARP_SIZE) == 0:
        _ = Atomic.fetch_add(counter, increment)


# ===----------------------------------------------------------------------=== #
# Core Trait Definitions
# ===----------------------------------------------------------------------=== #


trait SyncStrategy(TrivialRegisterPassable):
    """Interface for synchronization strategies between producers and consumers.

    All methods have the same signature regardless of the specific implementation,
    allowing the RingBuffer to be parameterized with any conforming strategy.

    Phase tracking ensures producers and consumers access different tiles:
    - Producers wait until consumers have finished with a tile (phase N)
    - Consumers wait until producers have filled a tile (phase N+1)
    """

    @always_inline
    def __init__(out self):
        """Initialize with internally allocated sync counter."""
        ...

    @always_inline
    def get_staged_idx(self, tile_idx: Int, stage: Int) -> Int:
        """Convert tile index and stage to a flat index in the counter arrays.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).

        Returns:
            Flat index for accessing synchronization counters.
        """
        ...

    @always_inline
    def wait_producer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        """Producer waits until it can write to the specified tile.

        Blocks until all consumers have finished reading from this tile
        (counter >= phase).

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
            phase: Counter threshold to wait for before acquiring the tile.
        """
        ...

    @always_inline
    def signal_producer_release(mut self, tile_idx: Int, stage: Int):
        """Producer signals that it has finished writing to the tile.

        Increments the appropriate counter to notify waiting consumers.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
        """
        ...

    @always_inline
    def wait_consumer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        """Consumer waits until it can read from the specified tile.

        Blocks until producer has finished writing to this tile
        (counter >= phase).

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
            phase: Counter threshold to wait for before acquiring the tile.
        """
        ...

    @always_inline
    def signal_consumer_release(mut self, tile_idx: Int, stage: Int):
        """Consumer signals that it has finished reading from the tile.

        Increments the appropriate counter to notify waiting producers.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
        """
        ...

    @always_inline
    def get_producer_phase_increment(self) -> Int32:
        """Returns how much to advance the producer phase after each acquisition.

        This determines when producers can reuse a tile after consumers finish.
        """
        ...

    @always_inline
    def get_consumer_phase_increment(self) -> Int32:
        """Returns how much to advance the consumer phase after each acquisition.

        This determines when consumers can read a tile after producers finish.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Sync Strategy Implementations
# ===----------------------------------------------------------------------=== #


struct SingleCounterSync[
    pipeline_stages: Int,
    block_rows: Int,
    warp_rows: Int,
    reads_per_warp_block: Int,
](SyncStrategy):
    """Single counter synchronization strategy.

    Uses one atomic counter per tile that tracks both producer and consumer progress.
    This is simpler but has higher contention as all warps compete for the same counter.

    Phase progression:
    - Each phase advances by (writes_per_warp_block + reads_per_warp_block)
    - Producers wait for phase N, increment counter by 1
    - Consumers wait for phase N+1, increment counter by 1

    Parameters:
        pipeline_stages: Number of pipeline stages in the ring buffer.
        block_rows: Total number of rows in the work block.
        warp_rows: Number of rows processed by a single warp.
        reads_per_warp_block: Number of consumer read operations per warp block.
    """

    comptime writes_per_warp_block = 1
    comptime block_warps = Self.block_rows // Self.warp_rows
    comptime total_tiles = Self.block_warps * Self.pipeline_stages
    comptime SyncCounterArray = SMemArray[Int32, Self.total_tiles]

    var sync_counter: Self.SyncCounterArray

    @always_inline
    def __init__(out self):
        """Initialize with internally allocated sync counter."""
        self.sync_counter = Self.SyncCounterArray.stack_allocation[
            alignment=32
        ]()

        comptime for i in range(Self.total_tiles):
            self.sync_counter[i][] = 0

    @always_inline
    def get_staged_idx(self, tile_idx: Int, stage: Int) -> Int:
        return tile_idx * Self.pipeline_stages + stage

    @always_inline
    def wait_producer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        wait_for_counter(self.sync_counter[staged_idx], phase)

    @always_inline
    def signal_producer_release(mut self, tile_idx: Int, stage: Int):
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        increment_counter_if_first_thread(
            self.sync_counter[staged_idx], Int32(1)
        )

    @always_inline
    def wait_consumer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        wait_for_counter(self.sync_counter[staged_idx], phase)

    @always_inline
    def signal_consumer_release(mut self, tile_idx: Int, stage: Int):
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        increment_counter_if_first_thread(
            self.sync_counter[staged_idx], Int32(1)
        )

    @always_inline
    def get_producer_phase_increment(self) -> Int32:
        return Int32(Self.writes_per_warp_block + Self.reads_per_warp_block)

    @always_inline
    def get_consumer_phase_increment(self) -> Int32:
        return Int32(Self.writes_per_warp_block + Self.reads_per_warp_block)


struct SplitCounterSync[
    pipeline_stages: Int,
    block_rows: Int,
    warp_rows: Int,
    reads_per_warp_block: Int,
](SyncStrategy):
    """Split counter synchronization strategy.

    Uses separate producer and consumer counters per tile to reduce atomic contention.
    Producers only write to producer counters, consumers only write to consumer counters.

    Phase progression:
    - Producer phase advances by reads_per_warp_block (waits for N consumers)
    - Consumer phase advances by writes_per_warp_block (waits for 1 producer)
    - This asymmetry reflects the 1-producer-to-N-consumers relationship

    Parameters:
        pipeline_stages: Number of pipeline stages in the ring buffer.
        block_rows: Total number of rows in the work block.
        warp_rows: Number of rows processed by a single warp.
        reads_per_warp_block: Number of consumer read operations per warp block.
    """

    comptime writes_per_warp_block = 1
    comptime block_warps = Self.block_rows // Self.warp_rows
    comptime total_tiles = Self.block_warps * Self.pipeline_stages

    comptime ProducerCounterArray = SMemArray[Int32, Self.total_tiles]
    comptime ConsumerCounterArray = SMemArray[Int32, Self.total_tiles]

    var producer_counters: Self.ProducerCounterArray
    var consumer_counters: Self.ConsumerCounterArray

    @always_inline
    def __init__(out self):
        """Initialize with internally allocated producer and consumer counters.
        """
        self.producer_counters = Self.ProducerCounterArray.stack_allocation[
            alignment=32
        ]()
        self.consumer_counters = Self.ConsumerCounterArray.stack_allocation[
            alignment=32
        ]()

        comptime for i in range(Self.total_tiles):
            self.producer_counters[i][] = 0
            self.consumer_counters[i][] = 0

    @always_inline
    def get_staged_idx(self, tile_idx: Int, stage: Int) -> Int:
        return tile_idx * Self.pipeline_stages + stage

    @always_inline
    def wait_producer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        """Producer waits on consumer counter.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
            phase: Counter threshold to wait for before acquiring the tile.
        """
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        wait_for_counter(self.consumer_counters[staged_idx], phase)

    @always_inline
    def signal_producer_release(mut self, tile_idx: Int, stage: Int):
        """Producer increments producer counter.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
        """
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        increment_counter_if_first_thread(
            self.producer_counters[staged_idx],
            Int32(Self.writes_per_warp_block),
        )

    @always_inline
    def wait_consumer_acquire(self, tile_idx: Int, stage: Int, phase: Int32):
        """Consumer waits on producer counter.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
            phase: Counter threshold to wait for before acquiring the tile.
        """
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        wait_for_counter(self.producer_counters[staged_idx], phase)

    @always_inline
    def signal_consumer_release(mut self, tile_idx: Int, stage: Int):
        """Consumer increments consumer counter by 1.

        Args:
            tile_idx: Index of the tile within a stage (0 to block_warps-1).
            stage: Pipeline stage (0 to pipeline_stages-1).
        """
        var staged_idx = self.get_staged_idx(tile_idx, stage)
        increment_counter_if_first_thread(
            self.consumer_counters[staged_idx], Int32(1)
        )

    @always_inline
    def get_producer_phase_increment(self) -> Int32:
        """Producer phase advances by reads_per_warp_block."""
        return Int32(Self.reads_per_warp_block)

    @always_inline
    def get_consumer_phase_increment(self) -> Int32:
        """Consumer phase advances by writes_per_warp_block."""
        return Int32(Self.writes_per_warp_block)
