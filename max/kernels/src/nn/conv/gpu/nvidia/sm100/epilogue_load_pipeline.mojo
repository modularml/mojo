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

"""Epilogue load pipeline types for SM100 Conv2D kernel.

This module provides runtime pipeline types for coordinating the epilogue
load warp with other kernel components:

- EpiLoadPipeline: Pipeline for EpilogueLoad warp → Epilogue warps transfer
- LoadOrderBarrier: Barrier for MainLoad → EpilogueLoad coordination

## Pipeline Architecture

The epilogue load warp (warp ID 7) pre-fetches source tensor C via TMA for
residual operations (D = Conv(A,B) + beta*C). This overlaps C loading with
MMA computation for better latency hiding.

```
MainLoad warp                 EpilogueLoad warp              Epilogue warps
    |                              |                              |
    |-- prologue loads --|         |                              |
    |                    |         |                              |
    |-- arrive() --------|-------->| wait()                       |
    |                              |                              |
    |-- steady-state     |         |-- TMA load C --|             |
    |                              |                |             |
    |                              |-- produce() ---|------------>| consume()
    |                              |                              |
```

## Usage

### Barrier Initialization

```mojo
if elect_one_thread:
    load_order_barrier.init(arrive_count=1)  # MainLoad arrives
    epi_load_pipeline.init_barriers(
        producer_arv_count=1,    # EpilogueLoad (TMA)
        consumer_arv_count=128,  # Epilogue warps (4 × 32)
    )
```

### MainLoad Warp (prologue/steady-state split)

```mojo
if WarpRole.is_main_load():
    # Issue prologue loads
    for _ in range(num_prologue_stages):
        load_input_tiles(...)

    # Signal epilogue load can start
    load_order_barrier.arrive()

    # Continue with steady-state loads
    for _ in range(remaining_stages):
        load_input_tiles(...)
```

### EpilogueLoad Warp

```mojo
if WarpRole.is_epilogue_load():
    # Wait for mainloop to start
    load_order_barrier.wait()

    with epi_load_pipeline.produce() as stage:
        # Load C tile via TMA
        tma_load(c_tile, stage.mbar(), ...)
```

### Epilogue Warps

```mojo
if WarpRole.is_epilogue():
    with epi_load_pipeline.consume() as c_stage:
        # C tile now in SMEM
        c_tile = smem.src_tiles()[c_stage.index()]
        tile_writer.write_with_residual(accum, c_tile, beta, ...)
```
"""

from layout.tma_async import SharedMemBarrier

from structured_kernels.pipeline import ProducerConsumerPipeline
from structured_kernels.pipeline_backend import MbarPtr


# =============================================================================
# EpiLoadPipeline - Pipeline for EpilogueLoad → Epilogue transfer
# =============================================================================


struct EpiLoadPipeline[num_stages: Int]:
    """Pipeline for epilogue load warp to epilogue store warps.

    Producer: Epilogue load warp (1 warp, 32 threads)
    Consumer: Epilogue store warps (4 warps, 128 threads)

    This pipeline synchronizes source tensor C loading with epilogue
    consumption for residual add operations (D = accum + beta * C).

    Parameters:
        num_stages: Number of pipeline stages (typically 2 for double-buffering).

    ## Barrier Layout in SMEM

    The pipeline uses 2 × num_stages barriers:
    - Full barriers [0..num_stages): Producer signals data ready
    - Empty barriers [num_stages..2*num_stages): Consumer signals stage free

    ## Arrive Counts

    - Producer arrive count: 1 (single TMA transaction per stage)
    - Consumer arrive count: 128 (all epilogue threads)
    """

    var pipeline: ProducerConsumerPipeline[Self.num_stages]

    @always_inline
    def __init__(out self, ptr: MbarPtr):
        """Initialize the epilogue load pipeline.

        Args:
            ptr: Pointer to shared memory barrier storage.
                 Requires 2 × num_stages SharedMemBarrier slots.
        """
        self.pipeline = ProducerConsumerPipeline[Self.num_stages](ptr)

    @always_inline
    def init_barriers(
        self,
        producer_arv_count: Int32 = 1,
        consumer_arv_count: Int32 = 128,
    ):
        """Initialize the pipeline barriers.

        Should be called by a single thread (elect_one_thread) during
        kernel initialization.

        Args:
            producer_arv_count: Arrive count for producer (default 1 for TMA).
            consumer_arv_count: Arrive count for consumer (default 128 for
                4 epilogue warps × 32 threads).
        """
        self.pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    # =========================================================================
    # Producer API (EpilogueLoad warp)
    # =========================================================================

    @always_inline
    def produce[
        origin: MutOrigin, //
    ](ref[origin] self) -> type_of(self.pipeline.produce()):
        """Produce one pipeline stage with encapsulated barriers.

        Usage:
            with epi_load_pipeline.produce() as stage:
                tma_load(c_tile, stage.mbar(), ...)
            # __exit__ advances producer stage

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            Context that waits for consumer on enter, advances on exit.
        """
        return self.pipeline.produce()

    @always_inline
    def acquire_producer[
        origin: MutOrigin, //
    ](ref[origin] self,) -> type_of(self.pipeline.acquire_producer()):
        """Acquire a producer stage handle.

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            ProducerStage handle that must be released.
        """
        return self.pipeline.acquire_producer()

    @always_inline
    def wait_consumer(self):
        """Wait for consumer to free the current stage."""
        self.pipeline.wait_consumer()

    @always_inline
    def producer_mbar(self) -> MbarPtr:
        """Get the producer barrier for the current stage.

        Returns:
            Barrier pointer for TMA arrive.
        """
        return rebind[MbarPtr](
            self.pipeline.producer_mbar(self.pipeline.producer_stage())
        )

    @always_inline
    def producer_step(mut self):
        """Advance producer to next stage."""
        self.pipeline.producer_step()

    # =========================================================================
    # Consumer API (Epilogue warps)
    # =========================================================================

    @always_inline
    def consume[
        origin: MutOrigin, //
    ](ref[origin] self) -> type_of(self.pipeline.consume()):
        """Consume one pipeline stage with encapsulated barriers.

        Usage:
            with epi_load_pipeline.consume() as stage:
                c_tile = smem.src_tiles()[stage.index()]
                # Use C tile for residual add
            # __exit__ signals consumption and advances

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            Context that waits for producer on enter, signals+advances on exit.
        """
        return self.pipeline.consume()

    @always_inline
    def consume_explicit[
        origin: MutOrigin, //
    ](ref[origin] self,) -> type_of(self.pipeline.consume_explicit()):
        """Consume with explicit barrier arrive.

        Use for lane-guarded signaling patterns.

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            Context that waits on enter, advances only on exit.
        """
        return self.pipeline.consume_explicit()

    @always_inline
    def acquire_consumer[
        origin: MutOrigin, //
    ](ref[origin] self,) -> type_of(self.pipeline.acquire_consumer()):
        """Acquire a consumer stage handle.

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            ConsumerStage handle that must be released.
        """
        return self.pipeline.acquire_consumer()

    @always_inline
    def wait_producer(self):
        """Wait for producer to fill the current stage."""
        self.pipeline.wait_producer()

    @always_inline
    def consumer_stage(self) -> UInt32:
        """Get the current consumer stage index."""
        return self.pipeline.consumer_stage()

    @always_inline
    def consumer_step(mut self):
        """Advance consumer to next stage."""
        self.pipeline.consumer_step()


# =============================================================================
# LoadOrderBarrier - For MainLoad → EpilogueLoad coordination
# =============================================================================


struct LoadOrderBarrier:
    """Barrier for coordinating mainloop load and epilogue load warps.

    This barrier implements a simple producer-consumer pattern where the
    mainloop load warp (producer) signals after completing prologue loads,
    and the epilogue load warp (consumer) waits before starting C loads.

    Protocol:
    1. Mainloop load warp issues prologue TMA loads
    2. Mainloop load warp calls arrive()
    3. Epilogue load warp calls wait() before starting
    4. Epilogue load warp can now issue TMA loads without contention

    This prevents TMA resource contention and ensures proper ordering.

    ## Phase Tracking

    The barrier uses a single phase bit that toggles per tile iteration.
    This allows proper synchronization across multiple output tiles.
    """

    var barrier: MbarPtr
    var phase: UInt32

    @always_inline
    def __init__(out self, ptr: MbarPtr, initial_phase: UInt32 = 0):
        """Initialize the load order barrier.

        Args:
            ptr: Pointer to shared memory barrier.
            initial_phase: Initial phase (default 0).
        """
        self.barrier = ptr
        self.phase = initial_phase

    @always_inline
    def init(self, arrive_count: Int32 = 1):
        """Initialize the barrier.

        Should be called by a single thread (elect_one_thread) during
        kernel initialization.

        Args:
            arrive_count: Number of arrives to expect (default 1 for
                single mainloop load warp).
        """
        self.barrier[0].init(arrive_count)

    @always_inline
    def arrive(self):
        """Signal that mainloop prologue loads are complete.

        Called by the mainloop load warp after issuing prologue TMA loads.
        """
        _ = self.barrier[0].arrive()

    @always_inline
    def wait(self):
        """Wait for mainloop to signal prologue completion.

        Called by the epilogue load warp before starting C loads.
        """
        self.barrier[0].wait(self.phase)

    @always_inline
    def step(mut self):
        """Toggle phase for next tile iteration.

        Called after both arrive and wait have completed to prepare
        for the next output tile's synchronization.
        """
        self.phase ^= 1

    @always_inline
    def arrive_and_step(mut self):
        """Arrive and advance phase in one call.

        Convenience method for mainloop load warp:
        ```
        load_order_barrier.arrive_and_step()
        ```
        """
        self.arrive()
        self.step()

    @always_inline
    def wait_and_step(mut self):
        """Wait and advance phase in one call.

        Convenience method for epilogue load warp:
        ```
        load_order_barrier.wait_and_step()
        ```
        """
        self.wait()
        self.step()
