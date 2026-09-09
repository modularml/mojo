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
"""Implements thread synchronization primitives including spin locks.

This module provides low-level locking mechanisms for thread synchronization,
including spin locks with blocking behavior and scoped lock guards for
automatic lock management. These primitives enable safe concurrent access to
shared resources in multi-threaded code.
"""

from std.atomic import Atomic
from std.ffi import external_call

# ===-----------------------------------------------------------------------===#
# SpinWaiter
# ===-----------------------------------------------------------------------===#


struct SpinWaiter(Defaultable):
    """A proxy for the C++ runtime's SpinWaiter type."""

    var storage: OpaquePointer[MutUntrackedOrigin]
    """Pointer to the underlying SpinWaiter instance."""

    def __init__(out self):
        """Initializes a SpinWaiter instance."""
        self.storage = external_call[
            "KGEN_CompilerRT_AsyncRT_InitializeSpinWaiter",
            OpaquePointer[MutUntrackedOrigin],
        ]()

    def __deinit__(deinit self):
        """Destroys the SpinWaiter instance."""
        external_call["KGEN_CompilerRT_AsyncRT_DestroySpinWaiter", NoneType](
            self.storage
        )

    def wait(self):
        """Blocks the current task for a duration determined by the underlying
        policy."""
        external_call["KGEN_CompilerRT_AsyncRT_SpinWaiter_Wait", NoneType](
            self.storage
        )


struct BlockingSpinLock(Defaultable):
    """A basic locking implementation that uses an integer to represent the
    owner of the lock."""

    comptime UNLOCKED: Int64 = -1
    """Non-zero means locked, -1 means unlocked."""

    var counter: Atomic[Int64]
    """The atomic counter implementing the spin lock."""

    def __init__(out self):
        """Default constructor."""

        self.counter = Atomic[Int64](Self.UNLOCKED)

    def lock(mut self, owner: Int):
        """Acquires the lock.

        Args:
            owner: The lock's owner (usually an address).
        """

        var expected = Int64(Self.UNLOCKED)
        var waiter = SpinWaiter()
        while not self.counter.compare_exchange(expected, Int64(owner)):
            # this should be yield
            waiter.wait()
            expected = Self.UNLOCKED

    def unlock(mut self, owner: Int) -> Bool:
        """Releases the lock.

        Args:
            owner: The lock's owner (usually an address).

        Returns:
            The successful release of the lock.
        """

        var expected = Int64(owner)
        if self.counter.load() != Int64(owner):
            # No one else can modify other than owner
            return False
        while not self.counter.compare_exchange(expected, Self.UNLOCKED):
            expected = Int64(owner)
        return True


struct BlockingScopedLock[origin: MutOrigin, //]:
    """A scope adapter for BlockingSpinLock."""

    comptime LockType = BlockingSpinLock
    """The type of the lock."""

    var lock: Pointer[Self.LockType, Self.origin]
    """The underlying lock instance."""

    def __init__(
        out self,
        lock: Pointer[Self.LockType, Self.origin],
    ):
        """Primary constructor.

        Args:
            lock: A pointer to the underlying lock.
        """

        self.lock = lock

    def __init__(out self, ref[Self.origin] lock: Self.LockType):
        """Secondary constructor.

        Args:
            lock: A mutable reference to the underlying lock.
        """

        self.lock = Pointer(to=lock)

    @no_inline
    def __enter__(mut self):
        """Acquire the lock on entry.
        This is done by setting the owner of the lock to own address."""
        var address = Pointer(to=self)
        self.lock[].lock(Int(address))

    @no_inline
    def __exit__(mut self):
        """Release the lock on exit.
        Reset the address on the underlying lock."""
        var address = Pointer(to=self)
        _ = self.lock[].unlock(Int(address))
