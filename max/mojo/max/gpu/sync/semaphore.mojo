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

"""This module provides a device-wide semaphore implementation for NVIDIA GPUs.

The Semaphore struct enables inter-CTA (Cooperative Thread Array) synchronization
by providing atomic operations and memory barriers. It uses NVIDIA-specific intrinsics
to implement efficient thread synchronization.

Examples:

    ```mojo
    from max.gpu.sync import Semaphore

    def semaphore(lock: Pointer[mut=True, Int32, ...]):
        var thread_id = 0
        var sem = Semaphore(lock, thread_id)

        # Wait for a specific state
        sem.wait(0)

        # Release the semaphore
        sem.release(1)
    ```
"""

from std.atomic import Atomic, Ordering
from std.sys import is_nvidia_gpu, llvm_intrinsic

from .sync import MaxHardwareBarriers, barrier, named_barrier

comptime _device_atomic = Atomic[Int32, scope="device"]


@always_inline
def _barrier_and(state: Bool) -> Bool:
    comptime assert is_nvidia_gpu(), "target must be an nvidia GPU"
    return llvm_intrinsic["llvm.nvvm.barrier.cta.red.and.aligned.all", Bool](
        Int32(0), state
    )


struct Semaphore[origin: MutOrigin, //](TrivialRegisterPassable):
    """A device-wide semaphore implementation for GPUs.

    This struct provides atomic operations and memory barriers for inter-CTA synchronization.
    It uses a single thread per CTA to perform atomic operations on a shared lock variable.
    """

    var _lock: Pointer[Int32, Self.origin]
    """Pointer to the shared lock variable in global memory that all CTAs synchronize on"""

    var _wait_thread: Bool
    """Flag indicating if this thread should perform atomic operations (true for thread 0)"""

    var _state: Int32
    """Current state of the semaphore, used to track synchronization status"""

    @always_inline
    def __init__(out self, lock: Pointer[Int32, Self.origin], thread_id: Int):
        """Initialize a new Semaphore instance.

        Args:
            lock: Pointer to shared lock variable in global memory.
            thread_id: Thread ID within the CTA, used to determine if this thread
                      should perform atomic operations.
        """
        comptime assert is_nvidia_gpu(), "target must be cuda"
        self._lock = lock
        self._wait_thread = thread_id <= 0
        self._state = -1

    @always_inline
    def fetch(mut self):
        """Fetch the current state of the semaphore from global memory.

        Only the designated wait thread (thread 0) performs the actual load,
        using an acquire memory ordering to ensure proper synchronization.
        """
        if self._wait_thread:
            self._state = _device_atomic.load[ordering=Ordering.ACQUIRE](
                self._lock
            )

    @always_inline
    def state(self) -> Int32:
        """Get the current state of the semaphore.

        Returns:
            The current state value of the semaphore.
        """
        return self._state

    @always_inline
    def wait(mut self, status: Int = 0):
        """Wait until the semaphore reaches the specified state.

        Uses a barrier-based spin loop where all threads participate in checking
        the state. Only proceeds when the state matches the expected status.

        Args:
            status: The state value to wait for (defaults to 0).
        """
        while _barrier_and(self._state.eq(Int32(status)).select(False, True)):
            self.fetch()
        barrier()

    @always_inline
    def release(mut self, status: Int32 = 0):
        """Release the semaphore by setting it to the specified state.

        Ensures all threads have reached this point via a barrier before
        the designated thread updates the semaphore state.

        Args:
            status: The new state value to set (defaults to 0).
        """
        barrier()
        if self._wait_thread:
            _device_atomic.store[ordering=Ordering.RELEASE](self._lock, status)


struct NamedBarrierSemaphore[
    origin: MutOrigin,
    //,
    thread_count: Int32,
    id_offset: Int32,
    max_num_barriers: Int32,
](TrivialRegisterPassable):
    """A device-wide semaphore implementation for NVIDIA GPUs with named barriers.

    It's using an acquire-release logic instead of atomic instructions for inter-CTA synchronization with a shared lock variable.
    Please note that the memory barrier is for syncing warp groups within in a CTA.
    Cutlass reference implementation:
    https://github.com/NVIDIA/cutlass/blob/a1aaf2300a8fc3a8106a05436e1a2abad0930443/include/cutlass/arch/barrier.h.

    Parameters:
        origin: Origin of the shared lock pointer in global memory.
        thread_count: Number of threads participating in the barrier.
        id_offset: Offset for the barrier ID.
        max_num_barriers: Maximum number of named barriers to use.
    """

    var _lock: Pointer[Int32, Self.origin]
    """Pointer to the shared lock variable in global memory that all CTAs synchronize on"""

    var _wait_thread: Bool
    """Flag indicating if this thread should perform atomic operations (true for thread 0)"""

    var _state: Int32
    """Current state of the semaphore, used to track synchronization status"""

    @always_inline
    def __init__(out self, lock: Pointer[Int32, Self.origin], thread_id: Int):
        """Initialize a new Semaphore instance.

        Args:
            lock: Pointer to shared lock variable in global memory.
            thread_id: Thread ID within the CTA, used to determine if this thread
                      should perform atomic operations.
        """
        comptime assert is_nvidia_gpu(), "target must be cuda"
        comptime assert (
            Self.id_offset + Self.max_num_barriers < MaxHardwareBarriers
        ), "max number of barriers is " + String(MaxHardwareBarriers)
        self._lock = lock
        self._wait_thread = thread_id <= 0
        self._state = -1

    @always_inline
    def state(self) -> Int32:
        """Get the current state of the semaphore.

        Returns:
            The current state value of the semaphore.
        """
        return self._state

    @always_inline
    def wait_eq(mut self, id: Int32, status: Int32 = 0):
        """Waits until the semaphore state equals the specified status.

        Args:
            id: Barrier ID to use for synchronization.
            status: Expected status value to wait for.
        """
        if self._wait_thread:
            while self._state != status:
                self._state = _device_atomic.load[ordering=Ordering.ACQUIRE](
                    self._lock
                )

        named_barrier[Self.thread_count,](Self.id_offset + id)

    @always_inline
    def wait_lt(mut self, id: Int32, count: Int32 = 0):
        """Waits until the semaphore state is less than the specified count.

        Args:
            id: Barrier ID to use for synchronization.
            count: Count value to compare against.
        """
        if self._wait_thread:
            while self._state < count:
                self._state = _device_atomic.load[ordering=Ordering.ACQUIRE](
                    self._lock
                )

        named_barrier[Self.thread_count,](Self.id_offset + id)

    @always_inline
    def arrive_set(self, id: Int32, status: Int32 = 0):
        """Arrives at the barrier and sets the semaphore status.

        Args:
            id: Barrier ID to use for synchronization.
            status: Status value to set after arriving.
        """
        named_barrier[Self.thread_count,](Self.id_offset + id)

        if self._wait_thread:
            _device_atomic.store[ordering=Ordering.RELEASE](self._lock, status)
