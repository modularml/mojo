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
"""Producer-consumer pipeline utilities for SM100 structured kernels.

This module provides pipeline synchronization primitives for warp-specialized
GPU kernels, enabling efficient producer-consumer patterns between warps.

Key abstraction:
- ProducerConsumerPipeline: Low-level barrier management for N-stage pipelines
- ProducerStage / ConsumerStage: Unified stage handles (linear types)

## Unified Stage Types

ProducerStage and ConsumerStage are linear types (`@explicit_destroy`) that work
in both contexts:

1. **Linear Type API** (flat, explicit):
    var stage = pipeline.acquire_producer()
    # ... use stage.index(), stage.mbar() ...
    stage^.release()  # Compiler enforces this call

2. **Context Manager API** (scoped, automatic):
    with pipeline.produce() as stage:
        # ... use stage.index(), stage.mbar() ...
    # release() called automatically

The context managers store the stage internally and return a `ref` to it,
allowing access to the full stage API while managing lifetime automatically.

## API Examples

Producer side (e.g., MMA warp producing to epilogue):

    # Context manager:
    with pipeline.produce() as stage:
        mma_op.mma(a, b, tmem_offset)
        mma_op.commit(stage.mbar())
    # __exit__ calls stage^.release() -> producer_step()

    # Linear type:
    var stage = pipeline.acquire_producer()
    mma_op.mma(a, b, tmem_offset)
    mma_op.commit(stage.mbar())
    stage^.release()

Consumer side (e.g., epilogue consuming from MMA):

    # Context manager:
    with pipeline.consume() as stage:
        process(stage.index())
    # __exit__ calls stage^.release() -> arrive + consumer_step()

    # Linear type:
    var stage = pipeline.acquire_consumer()
    process(stage.index())
    stage^.release()  # Signal + advance

    # Explicit signaling:
    var stage = pipeline.acquire_consumer()
    if lane_id() < CLUSTER_SIZE:
        stage.arrive()
    stage^.release_without_signal()  # Advance only

Direct API (for special cases):
    pipeline.wait_producer() / wait_consumer()
    pipeline.producer_step() / consumer_step()
    pipeline.producer_mbar(stage) / consumer_mbar(stage)
"""

from std.sys import size_of

from .pipeline_backend import PipelineBackend, NvidiaMbarBackend

# SM100 (B200) warp-specialized pipeline backoff hint, in nanoseconds.
#
# `SharedMemBarrier.wait` lowers to the
# `LAB_WAIT: mbarrier.try_wait.parity ...; @P1 bra DONE; bra LAB_WAIT;`.
# Without a `ticks` operand `try_wait.parity` returns immediately, so the warp
# tight-spins (BRA/SYNCS per iteration). With a `ticks` operand the hardware
# *suspends* the warp for up to `ticks` ns (it wakes the instant the barrier
# flips, so this is a ceiling, not a fixed sleep) — the NANOSLEEP-equivalent
# idiom cuBLAS `nvjet` uses. Cuts the spin instructions when a warp would
# otherwise re-iterate the loop many times (near-full occupancy / many waves).
comptime SM100_PIPELINE_WAIT_TICKS = UInt32(0x989680)


struct ProducerConsumerPipeline[
    num_stages: Int, Backend: PipelineBackend = NvidiaMbarBackend[num_stages]
](TrivialRegisterPassable):
    """A producer-consumer pipeline that synchronizes producer and consumer
    warps through a pluggable hardware backend.

    Parameters:
        num_stages: The number of pipeline stages.
        Backend: The synchronization backend. Defaults to `NvidiaMbarBackend`
            (NVIDIA `mbarrier`); other backends (e.g. atomic counters for AMD)
            plug in without changing this struct.

    This struct is commonly used with warp specialization to pipeline operations
    between two warps/warpgroups with data dependencies.
    """

    # Hardware synchronization backend (mbarrier, atomic counters, ...).
    var backend: Self.Backend

    # The stage in pipeline, from 0 to num_stages-1
    var _consumer_stage: UInt32
    var _producer_stage: UInt32

    # The lap around the ring for each side, incremented on wrap-around.
    # Backends map it onto their substrate (NVIDIA uses the low bit as parity).
    var _consumer_phase: UInt32
    var _producer_phase: UInt32

    @always_inline
    def __init__(
        out self,
        ptr: UnsafePointer[
            Self.Backend.BarrierStorage,
            MutUntrackedOrigin,
            address_space=.SHARED,
        ],
    ):
        """Initialize the producer-consumer pipeline with default phases.

        Args:
            ptr: Pointer to shared memory barriers.
        """
        self.backend = Self.Backend.__init__[Self.num_stages](ptr)
        self._producer_stage = 0
        self._consumer_stage = 0
        # This ensures producer's wait_consumer() passes trivially at
        # the beginning when it tries to initialize data buffer.
        self._producer_phase = 1
        self._consumer_phase = 0

    @always_inline
    def wait_producer[ticks: Optional[UInt32] = None](self):
        """Consumer waits for producer.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns) forwarded to
                `SharedMemBarrier.wait`. `None` (default) preserves the
                immediate-return spin for every existing caller; pass
                `SM100_PIPELINE_WAIT_TICKS` to let the warp suspend instead
                of busy-spinning while blocked.
        """
        self.backend.wait_full[ticks=ticks](
            self._consumer_stage, self._consumer_phase
        )

    @always_inline
    def wait_consumer[ticks: Optional[UInt32] = None](self):
        """Producer waits for consumer.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns) forwarded to
                `SharedMemBarrier.wait`. See `wait_producer` for semantics.
        """
        self.backend.wait_empty[ticks=ticks](
            self._producer_stage, self._producer_phase
        )

    @always_inline
    def try_wait_producer(self) -> Bool:
        """Non-blocking check if producer data is ready.

        Returns:
            True if the producer has filled the current stage, False otherwise.

        Note:
            Use this with wait_producer_if_needed() for the try-acquire pattern:
            ```
            var ready = pipeline.try_wait_producer()
            # ... do other work ...
            pipeline.wait_producer_if_needed(ready)
            ```
        """
        return self.backend.try_full(self._consumer_stage, self._consumer_phase)

    @always_inline
    def try_wait_consumer(self) -> Bool:
        """Non-blocking check if consumer has freed the stage.

        Returns:
            True if the consumer has freed the current stage, False otherwise.

        Note:
            Use this with wait_consumer_if_needed() for the try-acquire pattern.
        """
        return self.backend.try_empty(
            self._producer_stage, self._producer_phase
        )

    @always_inline
    def wait_producer_if_needed(self, already_ready: Bool):
        """Conditionally wait for producer if not already ready.

        Args:
            already_ready: Result from try_wait_producer(). If True, skips waiting.
        """
        if not already_ready:
            self.wait_producer()

    @always_inline
    def wait_consumer_if_needed(self, already_ready: Bool):
        """Conditionally wait for consumer if not already ready.

        Args:
            already_ready: Result from try_wait_consumer(). If True, skips waiting.
        """
        if not already_ready:
            self.wait_consumer()

    @always_inline
    def producer_mbar(self, stage: UInt32) -> Self.Backend.Handle:
        """Get the producer barrier for a specific stage.

        Args:
            stage: The pipeline stage.

        Returns:
            The shared memory barrier that the producer signals.
        """
        return self.backend.full_handle(stage)

    @always_inline
    def consumer_mbar(self, stage: UInt32) -> Self.Backend.Handle:
        """Get the consumer barrier for a specific stage.

        Args:
            stage: The pipeline stage.

        Returns:
            The shared memory barrier that the consumer signals.
        """
        return self.backend.empty_handle(stage)

    @always_inline
    def producer_stage(self) -> UInt32:
        """Get the current producer stage index.

        Returns:
            The current stage index for the producer (0 to num_stages-1).
        """
        return self._producer_stage

    @always_inline
    def consumer_stage(self) -> UInt32:
        """Get the current consumer stage index.

        Returns:
            The current stage index for the consumer (0 to num_stages-1).
        """
        return self._consumer_stage

    @always_inline
    def consumer_step(mut self):
        """Advance the consumer to the next pipeline stage.

        Increments the consumer stage and wraps to 0 when reaching num_stages,
        incrementing the lap counter on wrap-around.
        Only advance the lap at end of pipeline because we assume all barriers
        are at the same consumer/producer phase before checked. Once checked,
        the execution moves to next barrier.
        """
        self._consumer_stage += 1

        if self._consumer_stage == UInt32(Self.num_stages):
            self._consumer_stage = 0
            self._consumer_phase += 1

    @always_inline
    def producer_step(mut self):
        """Advance the producer to the next pipeline stage.

        Increments the producer stage and wraps to 0 when reaching num_stages,
        incrementing the lap counter on wrap-around.
        """
        self._producer_stage += 1

        if self._producer_stage == UInt32(Self.num_stages):
            self._producer_stage = 0
            self._producer_phase += 1

    @staticmethod
    @always_inline
    def smem_bytes() -> UInt32:
        """Calculate the shared memory bytes required for pipeline barriers.

        Returns:
            The total number of bytes needed for all pipeline barriers
            (2 * num_stages barriers).
        """
        return UInt32(
            Self.Backend.storage_elems[Self.num_stages]()
            * size_of[Self.Backend.BarrierStorage]()
        )

    @always_inline
    def init_mbars(
        self, producer_arrive_count: Int32, consumer_arrive_count: Int32
    ):
        """
        Initialize the smem barriers for the producer and consumer.

        Args:
            producer_arrive_count: The number of threads that will arrive at the barrier marking data as produced.
            consumer_arrive_count: The number of threads that will arrive at the barrier marking data as consumed.

        This function must be called by a single thread and must be called before any the pipeline object is used.
        """
        self.backend.init_barriers[Self.num_stages](
            producer_arrive_count, consumer_arrive_count
        )

    @always_inline
    def producer_signal_and_step(mut self):
        """Wait for consumer, signal production, and advance stage.

        Combined operation for CLC throttling (Load warp):
        1. Wait for consumer to finish with current stage
        2. Signal that producer has new data
        3. Advance to next stage
        """
        self.wait_consumer()
        self.backend.arrive_full(self._producer_stage)
        self.producer_step()

    @always_inline
    def consumer_signal_and_step(mut self):
        """Wait for producer, signal consumption, and advance stage.

        Combined operation for CLC throttling (Scheduler warp):
        1. Wait for producer to have data ready
        2. Signal that consumer has consumed data
        3. Advance to next stage
        """
        self.wait_producer()
        self.backend.arrive_empty(self._consumer_stage)
        self.consumer_step()

    # =========================================================================
    # Context Manager API - Encapsulated barrier operations
    # =========================================================================

    @always_inline
    def produce[
        origin: MutOrigin, //
    ](ref[origin] self) -> ProduceContext[
        origin, Self.num_stages, Self.Backend
    ]:
        """Produce one pipeline stage with encapsulated barriers.

        Usage:
            with pipeline.produce() as stage:
                # stage.index() gives current stage
                # stage.mbar() gives barrier for signaling
                # __exit__ calls producer_step()

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            Context that waits for consumer on enter, advances on exit.
        """
        return ProduceContext(Pointer(to=self))

    @always_inline
    def consume[
        origin: MutOrigin, //
    ](ref[origin] self) -> ConsumeContext[
        origin, Self.num_stages, Self.Backend
    ]:
        """Consume one pipeline stage with encapsulated barriers.

        Usage:
            with pipeline.consume() as stage:
                # stage.index() gives current stage
                # __exit__ signals consumer done and advances

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            Context that waits for producer on enter, signals+advances on exit.
        """
        return ConsumeContext(Pointer(to=self))

    @always_inline
    def consume_explicit[
        origin: MutOrigin, //
    ](ref[origin] self) -> ExplicitConsumeContext[
        origin, Self.num_stages, Self.Backend
    ]:
        """Consume one pipeline stage with EXPLICIT barrier arrive.

        Use this for kernels requiring lane-guarded or specialized signaling.

        Usage:
            with pipeline.consume_explicit() as stage:
                # ... do work ...
                if lane_id() < CLUSTER_SIZE:
                    stage.arrive()  # Lane-guarded arrive
            # __exit__ only advances, does NOT arrive

        For specialized signaling (e.g., umma_arrive_leader_cta):
            with pipeline.consume_explicit() as stage:
                if cta_group == 1:
                    stage.arrive()
                else:
                    umma_arrive_leader_cta(stage.mbar())

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            Context that waits for producer on enter, advances only on exit.
        """
        return ExplicitConsumeContext(Pointer(to=self))

    # =========================================================================
    # Linear Type API - Compiler-enforced resource management
    # =========================================================================

    @always_inline
    def acquire_producer[
        origin: MutOrigin, //
    ](ref[origin] self) -> ProducerStage[origin, Self.num_stages, Self.Backend]:
        """Acquire a producer stage handle using linear types.

        Waits for the consumer to free the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var stage = pipeline.acquire_producer()
            # ... produce data, signal via stage.mbar() ...
            stage^.release()  # Advances to next stage

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            A ProducerStage handle that must be released.
        """
        self.wait_consumer()
        return ProducerStage(Pointer(to=self), self._producer_stage)

    @always_inline
    def acquire_consumer[
        origin: MutOrigin, //
    ](ref[origin] self) -> ConsumerStage[origin, Self.num_stages, Self.Backend]:
        """Acquire a consumer stage handle using linear types.

        Waits for the producer to fill the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var stage = pipeline.acquire_consumer()
            # ... consume data ...
            stage^.release()  # Signals complete and advances

        For explicit signaling:
            var stage = pipeline.acquire_consumer()
            # ... consume data ...
            if lane_id() < CLUSTER_SIZE:
                stage.arrive()
            stage^.release_without_signal()

        Parameters:
            origin: Origin of the mutable borrow of `self` (inferred).

        Returns:
            A ConsumerStage handle that must be released.
        """
        self.wait_producer()
        return ConsumerStage(Pointer(to=self), self._consumer_stage)


# =============================================================================
# Unified Stage Types - Work as both linear types and with context managers
# =============================================================================
#
# These types can be used in two ways:
#
# 1. Linear Type API (flat, explicit):
#    var stage = pipeline.acquire_producer()
#    # ... use stage.index(), stage.mbar() ...
#    stage^.release()  # Compiler enforces this call
#
# 2. Context Manager API (scoped, automatic):
#    with pipeline.produce() as stage:
#        # ... use stage.index(), stage.mbar() ...
#    # release() called automatically by context manager
#
# =============================================================================


@explicit_destroy("Must call release() to advance stage")
struct ProducerStage[
    pipeline_origin: MutOrigin,
    num_stages: Int,
    Backend: PipelineBackend = NvidiaMbarBackend[num_stages],
](Deinitable where False, Movable):
    """Unified handle for producing to a pipeline stage.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `pipeline.acquire_producer()` or context manager
    2. Use `index()` and `mbar()` for production
    3. Must call `release()` to advance stage (compiler-enforced)

    Parameters:
        pipeline_origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        Backend: The synchronization backend (see `ProducerConsumerPipeline`).
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages, Self.Backend],
        Self.pipeline_origin,
    ]
    var _index: UInt32

    @always_inline
    def __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages, Self.Backend],
            Self.pipeline_origin,
        ],
        index: UInt32,
    ):
        self.pipeline = pipeline
        self._index = index

    @always_inline
    def index(self) -> UInt32:
        """Get the current stage index."""
        return self._index

    @always_inline
    def mbar(self) -> Self.Backend.Handle:
        """Get the barrier to signal when production is complete.

        Caller is responsible for signaling via mma_arrive or similar.
        """
        return self.pipeline[].backend.full_handle(self._index)

    @always_inline
    def release(deinit self):
        """Advance producer to next stage.

        This is the only way to destroy this linear type.
        The compiler will error if you don't call this.
        """
        self.pipeline[].producer_step()


struct ProduceContext[
    pipeline_origin: MutOrigin,
    num_stages: Int,
    Backend: PipelineBackend = NvidiaMbarBackend[num_stages],
]:
    """Context manager for producing one pipeline stage.

    - __enter__: Waits for consumer to be ready, returns ref to stage
    - __exit__: Releases the stage (advances producer)

    Note: The actual production signal (mma_arrive) is kernel-specific
    and must be called by the user before exiting the context.

    Parameters:
        pipeline_origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        Backend: Pipeline synchronization backend (defaults to
            `NvidiaMbarBackend`).
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages, Self.Backend],
        Self.pipeline_origin,
    ]
    var _stage: Optional[
        ProducerStage[Self.pipeline_origin, Self.num_stages, Self.Backend]
    ]

    @always_inline
    def __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages, Self.Backend],
            Self.pipeline_origin,
        ],
    ):
        self.pipeline = pipeline
        self._stage = None

    @always_inline
    def __enter__(
        mut self,
    ) -> ref[self._stage.value()] ProducerStage[
        Self.pipeline_origin, Self.num_stages, Self.Backend
    ]:
        """Wait for consumer and return reference to stage."""
        self.pipeline[].wait_consumer()
        # `_stage` is empty here; destroy it before repopulating, since an
        # `Optional` of a non-implicitly-deletable stage can't drop implicitly.
        self._stage^.deinit_assert_empty()
        self._stage = ProducerStage(
            self.pipeline, self.pipeline[].producer_stage()
        )
        return self._stage.value()

    @always_inline
    def __exit__(mut self):
        """Release the stage (advances producer)."""
        self._stage.take().release()
        # take() already sets _stage to None

    @always_inline
    def __deinit__(deinit self):
        self._stage^.deinit_assert_empty()


@explicit_destroy("Must call release() or release_without_signal()")
struct ConsumerStage[
    pipeline_origin: MutOrigin,
    num_stages: Int,
    Backend: PipelineBackend = NvidiaMbarBackend[num_stages],
](Deinitable where False, Movable):
    """Unified handle for consuming from a pipeline stage.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `pipeline.acquire_consumer()` or context manager
    2. Use `index()` for consumption
    3. Must call `release()` to signal and advance (compiler-enforced)

    Two exit paths:
    - `release()`: Signal consumption complete + advance (normal path)
    - `release_without_signal()`: Advance only (for explicit signaling)

    Parameters:
        pipeline_origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        Backend: The synchronization backend (see `ProducerConsumerPipeline`).
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages, Self.Backend],
        Self.pipeline_origin,
    ]
    var _index: UInt32

    @always_inline
    def __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages, Self.Backend],
            Self.pipeline_origin,
        ],
        index: UInt32,
    ):
        self.pipeline = pipeline
        self._index = index

    @always_inline
    def index(self) -> UInt32:
        """Get the current stage index."""
        return self._index

    @always_inline
    def mbar(self) -> Self.Backend.Handle:
        """Get the barrier for manual signaling.

        Use this for specialized signaling patterns like umma_arrive_leader_cta.
        For standard usage, just call release().
        """
        return self.pipeline[].backend.empty_handle(self._index)

    @always_inline
    def arrive(self):
        """Manually arrive on the consumer barrier.

        Use for lane-guarded patterns:
            if lane_id() < CLUSTER_SIZE:
                stage.arrive()
            stage^.release_without_signal()
        """
        self.pipeline[].backend.arrive_empty(self._index)

    @always_inline
    def release(deinit self):
        """Signal consumption complete and advance to next stage.

        This is the standard exit path. Equivalent to:
            arrive()
            consumer_step()
        """
        self.pipeline[].backend.arrive_empty(self._index)
        self.pipeline[].consumer_step()

    @always_inline
    def release_without_signal(deinit self):
        """Advance to next stage WITHOUT signaling.

        Use when you've already signaled via arrive() or specialized APIs.
        """
        self.pipeline[].consumer_step()


struct ConsumeContext[
    pipeline_origin: MutOrigin,
    num_stages: Int,
    Backend: PipelineBackend = NvidiaMbarBackend[num_stages],
]:
    """Context manager for consuming one pipeline stage.

    - __enter__: Waits for producer to be ready, returns ref to stage
    - __exit__: Releases the stage (signals consumption + advances)

    Parameters:
        pipeline_origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        Backend: Pipeline synchronization backend (defaults to
            `NvidiaMbarBackend`).
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages, Self.Backend],
        Self.pipeline_origin,
    ]
    var _stage: Optional[
        ConsumerStage[Self.pipeline_origin, Self.num_stages, Self.Backend]
    ]

    @always_inline
    def __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages, Self.Backend],
            Self.pipeline_origin,
        ],
    ):
        self.pipeline = pipeline
        self._stage = None

    @always_inline
    def __enter__(
        mut self,
    ) -> ref[self._stage.value()] ConsumerStage[
        Self.pipeline_origin, Self.num_stages, Self.Backend
    ]:
        """Wait for producer and return reference to stage."""
        self.pipeline[].wait_producer()
        var stage_idx = self.pipeline[].consumer_stage()
        # `_stage` is empty here; destroy it before repopulating, since an
        # `Optional` of a non-implicitly-deletable stage can't drop implicitly.
        self._stage^.deinit_assert_empty()
        self._stage = ConsumerStage(self.pipeline, stage_idx)
        return self._stage.value()

    @always_inline
    def __exit__(mut self):
        """Release the stage (signals consumption + advances)."""
        self._stage.take().release()
        # take() already sets _stage to None

    @always_inline
    def __deinit__(deinit self):
        self._stage^.deinit_assert_empty()


struct ExplicitConsumeContext[
    pipeline_origin: MutOrigin,
    num_stages: Int,
    Backend: PipelineBackend = NvidiaMbarBackend[num_stages],
]:
    """Context manager for consuming with EXPLICIT barrier arrive.

    Use this when you need lane-guarded or specialized barrier signaling.

    - __enter__: Waits for producer to be ready, returns ref to stage with mbar
    - __exit__: Only advances stage counter, does NOT arrive on barrier

    The caller is responsible for calling arrive via stage.arrive() or stage.mbar():
        with pipeline.consume_explicit() as stage:
            # ... do work ...
            if lane_id() < CLUSTER_SIZE:
                stage.arrive()
        # __exit__ only calls consumer_step(), not arrive()

    Parameters:
        pipeline_origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        Backend: Pipeline synchronization backend (defaults to
            `NvidiaMbarBackend`).
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages, Self.Backend],
        Self.pipeline_origin,
    ]
    var _stage: Optional[
        ConsumerStage[Self.pipeline_origin, Self.num_stages, Self.Backend]
    ]

    @always_inline
    def __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages, Self.Backend],
            Self.pipeline_origin,
        ],
    ):
        self.pipeline = pipeline
        self._stage = None

    @always_inline
    def __enter__(
        mut self,
    ) -> ref[self._stage.value()] ConsumerStage[
        Self.pipeline_origin, Self.num_stages, Self.Backend
    ]:
        """Wait for producer and return reference to stage with barrier access.
        """
        self.pipeline[].wait_producer()
        var stage_idx = self.pipeline[].consumer_stage()
        # `_stage` is empty here; destroy it before repopulating, since an
        # `Optional` of a non-implicitly-deletable stage can't drop implicitly.
        self._stage^.deinit_assert_empty()
        self._stage = ConsumerStage(self.pipeline, stage_idx)
        return self._stage.value()

    @always_inline
    def __exit__(mut self):
        """Advance to next stage WITHOUT signaling barrier."""
        # Caller is responsible for signaling via stage.arrive() or stage.mbar()
        self._stage.take().release_without_signal()
        # take() already sets _stage to None

    @always_inline
    def __deinit__(deinit self):
        self._stage^.deinit_assert_empty()
