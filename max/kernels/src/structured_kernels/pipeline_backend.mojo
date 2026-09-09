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
"""Hardware synchronization backends for `ProducerConsumerPipeline`.

`ProducerConsumerPipeline` coordinates warp-specialized producer and consumer
warps through a ring of shared-memory slots. Each slot has two signals:

- `full`:  the producer raises it when a slot holds fresh data.
- `empty`: the consumer raises it when a slot has been drained and may be
           refilled.

The *protocol* (which slot, which lap around the ring, who waits on whom) is
hardware-neutral and lives in `ProducerConsumerPipeline`. The *mechanism* used
to raise and wait on a signal is hardware-specific and lives behind the
`PipelineBackend` trait defined here:

- `NvidiaMbarBackend` uses NVIDIA `mbarrier` objects (`SharedMemBarrier`).
- A future non-NVIDIA backend (see KERN-2625) will use shared-memory atomic
  counters, which have no phase/parity concept.

## Phase / lap convention

The pipeline hands each backend a monotonically increasing `phase` (the lap
number for a given slot, incremented each time the ring wraps). Backends map it
onto their substrate:

- `NvidiaMbarBackend` reduces it to the single parity bit `phase & 1` that
  `mbarrier.try_wait.parity` expects. The parity sequence is identical to the
  historical `^= 1` toggle, so this backend is bit-for-bit equivalent to the
  previous hardcoded implementation.
- A counter-based backend can derive the absolute threshold a slot must reach
  from `phase` directly, so it needs no per-slot mutable wait state.
"""

from layout.tma_async import SharedMemBarrier

comptime MbarPtr = UnsafePointer[
    SharedMemBarrier, MutUntrackedOrigin, address_space=.SHARED
]


trait PipelineBackend(TrivialRegisterPassable):
    """Hardware backend for `ProducerConsumerPipeline` slot signaling.

    A backend owns the per-slot `full`/`empty` signal storage and implements the
    four primitive operations the pipeline needs: wait on a signal, raise a
    signal, and their non-blocking `try_*` variants. It also exposes a per-slot
    `Handle` that callers use for substrate-specific extras (for example, setting
    the expected TMA transaction size on NVIDIA).

    Conforming types must be `TrivialRegisterPassable` so the pipeline stays a
    register-passed value, matching every existing call site.
    """

    # The shared-memory element type backing one signal slot.
    # NVIDIA: `SharedMemBarrier`. Counter backends: `Int32`.
    comptime BarrierStorage: AnyType

    # What `full_handle`/`empty_handle` return for one slot. Bound to
    # TrivialRegisterPassable (like the backend itself) so the @explicit_destroy
    # stage handles can store it as a field with no custom destroy semantics
    # to reason about generically.
    comptime Handle: TrivialRegisterPassable

    @always_inline
    def __init__[
        num_stages: Int
    ](
        out self,
        ptr: UnsafePointer[
            Self.BarrierStorage, MutUntrackedOrigin, address_space=.SHARED
        ],
    ):
        """Construct from the base pointer of the backing storage array.

        Parameters:
            num_stages: The number of pipeline stages (ring depth). Must
                match `Self.num_stages`.

        Args:
            ptr: Pointer to the first of `storage_elems(num_stages)` elements.
        """
        ...

    @staticmethod
    def storage_elems[num_stages: Int]() -> Int:
        """Return the number of `BarrierStorage` elements to reserve in SMEM.

        Parameters:
            num_stages: The number of pipeline stages. Must match
                `Self.num_stages`.

        Returns:
            The element count for the backing shared-memory array.
        """
        ...

    @always_inline
    def init_barriers[
        num_stages: Int
    ](self, producer_arrive_count: Int32, consumer_arrive_count: Int32,):
        """Initialize all `full`/`empty` signals for the ring.

        Must be called by a single thread before the pipeline is used.

        Parameters:
            num_stages: The number of pipeline stages (ring depth). Must
                match `Self.num_stages`.

        Args:
            producer_arrive_count: Threads that arrive to mark a slot full.
            consumer_arrive_count: Threads that arrive to mark a slot empty.
        """
        ...

    @always_inline
    def wait_full[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        """Block until the producer has filled `stage` on the given lap.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns). Honored by backends
                whose hardware supports it (NVIDIA); ignored otherwise.

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.
        """
        ...

    @always_inline
    def wait_empty[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        """Block until the consumer has drained `stage` on the given lap.

        Parameters:
            ticks: Optional hardware-suspend ceiling (ns). Honored by backends
                whose hardware supports it (NVIDIA); ignored otherwise.

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.
        """
        ...

    @always_inline
    def try_full(self, stage: UInt32, phase: UInt32) -> Bool:
        """Return whether the producer has filled `stage` (non-blocking).

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.

        Returns:
            True if the slot is full for this lap, False otherwise.
        """
        ...

    @always_inline
    def try_empty(self, stage: UInt32, phase: UInt32) -> Bool:
        """Return whether the consumer has drained `stage` (non-blocking).

        Args:
            stage: The slot index in the ring.
            phase: The monotonic lap number for this slot.

        Returns:
            True if the slot is empty for this lap, False otherwise.
        """
        ...

    @always_inline
    def arrive_full(self, stage: UInt32):
        """Raise the `full` signal for `stage` (producer side).

        Args:
            stage: The slot index in the ring.
        """
        ...

    @always_inline
    def arrive_empty(self, stage: UInt32):
        """Raise the `empty` signal for `stage` (consumer side).

        Args:
            stage: The slot index in the ring.
        """
        ...

    @always_inline
    def full_handle(self, stage: UInt32) -> Self.Handle:
        """Return the `full` signal handle for `stage`.

        Args:
            stage: The slot index in the ring.

        Returns:
            The backend-specific handle to the slot's `full` signal.
        """
        ...

    @always_inline
    def empty_handle(self, stage: UInt32) -> Self.Handle:
        """Return the `empty` signal handle for `stage`.

        Args:
            stage: The slot index in the ring.

        Returns:
            The backend-specific handle to the slot's `empty` signal.
        """
        ...


struct NvidiaMbarBackend[num_stages: Int](PipelineBackend):
    """`PipelineBackend` using NVIDIA `mbarrier` objects.

    This is the default backend and reproduces the historical, hardcoded
    `mbarrier` behavior exactly. The ring's `2 * num_stages` `SharedMemBarrier`
    objects are laid out as `full[0..num_stages)` followed by
    `empty[0..num_stages)`; `full` points at the first, `empty` at the second.

    Parameters:
        num_stages: The number of pipeline stages (ring depth) this backend
            instance is configured for.
    """

    comptime BarrierStorage = SharedMemBarrier
    comptime Handle = MbarPtr

    # Full implies data has been produced. Producer signals this barrier
    # and consumer waits on this barrier.
    var full: MbarPtr

    # Empty implies data has been consumed. Consumer signals this barrier
    # and producer waits on this barrier.
    var empty: MbarPtr

    @always_inline
    def __init__[passed_num_stages: Int](out self, ptr: MbarPtr):
        """Construct from the base pointer of the backing barrier array.

        Parameters:
            passed_num_stages: The number of pipeline stages (ring depth).
                Must match `Self.num_stages`.

        Args:
            ptr: Pointer to the first of `2 * num_stages` barriers.
        """
        comptime assert passed_num_stages == Self.num_stages, (
            "num_stages passed to NvidiaMbarBackend.__init__ must match"
            " NvidiaMbarBackend's own num_stages"
        )
        self.full = ptr
        self.empty = ptr + Self.num_stages

    @staticmethod
    @always_inline
    def storage_elems[passed_num_stages: Int]() -> Int:
        comptime assert passed_num_stages == Self.num_stages, (
            "num_stages passed to NvidiaMbarBackend.storage_elems must match"
            " NvidiaMbarBackend's own num_stages"
        )
        return 2 * Self.num_stages

    @always_inline
    def init_barriers[
        passed_num_stages: Int
    ](self, producer_arrive_count: Int32, consumer_arrive_count: Int32,):
        comptime assert passed_num_stages == Self.num_stages, (
            "num_stages passed to NvidiaMbarBackend.init_barriers must match"
            " NvidiaMbarBackend's own num_stages"
        )
        comptime for i in range(Self.num_stages):
            self.full[i].init(producer_arrive_count)
            self.empty[i].init(consumer_arrive_count)

    @always_inline
    def wait_full[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        # mbarrier tracks a single parity bit; `& 1` reduces the lap to it.
        self.full[stage].wait[ticks=ticks](phase & 1)

    @always_inline
    def wait_empty[
        ticks: Optional[UInt32] = None
    ](self, stage: UInt32, phase: UInt32):
        self.empty[stage].wait[ticks=ticks](phase & 1)

    @always_inline
    def try_full(self, stage: UInt32, phase: UInt32) -> Bool:
        return self.full[stage].try_wait(phase & 1)

    @always_inline
    def try_empty(self, stage: UInt32, phase: UInt32) -> Bool:
        return self.empty[stage].try_wait(phase & 1)

    @always_inline
    def arrive_full(self, stage: UInt32):
        _ = self.full[stage].arrive()

    @always_inline
    def arrive_empty(self, stage: UInt32):
        _ = self.empty[stage].arrive()

    @always_inline
    def full_handle(self, stage: UInt32) -> Self.Handle:
        return self.full + stage

    @always_inline
    def empty_handle(self, stage: UInt32) -> Self.Handle:
        return self.empty + stage
