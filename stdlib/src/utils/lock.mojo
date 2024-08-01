# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

from memory import UnsafePointer
from os import Atomic
from time import sleep
from sys import external_call


# ===----------------------------------------------------------------------===#
# SpinWaiter
# ===----------------------------------------------------------------------===#


struct SpinWaiter:
    """A proxy for the C++ runtime's SpinWaiter type."""

    var storage: UnsafePointer[NoneType]
    """Pointer to the underlying SpinWaiter instance."""

    fn __init__(inout self: Self):
        """Initializes a SpinWaiter instance."""
        self.storage = external_call[
            "KGEN_CompilerRT_AsyncRT_InitializeSpinWaiter",
            UnsafePointer[NoneType],
        ]()

    fn __del__(owned self: Self):
        """Destroys the SpinWaiter instance."""
        external_call["KGEN_CompilerRT_AsyncRT_DestroySpinWaiter", NoneType](
            self.storage
        )

    fn wait(self: Self):
        """Blocks the current task for a duration determined by the underlying
        policy."""
        external_call["KGEN_CompilerRT_AsyncRT_SpinWaiter_Wait", NoneType](
            self.storage
        )


struct BlockingSpinLock:
    """A basic locking implementation that uses an integer to represent the
    owner of the lock."""

    alias UNLOCKED = -1
    """non-zero means locked, -1 means unlocked."""

    var counter: Atomic[DType.int64]
    """The atomic counter implementing the spin lock."""

    fn __init__(inout self: Self):
        """Default constructor."""

        self.counter = Atomic[DType.int64](Self.UNLOCKED)

    fn lock(inout self: Self, owner: Int):
        """Acquires the lock.

        Args:
            owner: The lock's owner (usually an address).
        """

        var expected = Int64(Self.UNLOCKED)
        var waiter = SpinWaiter()
        while not self.counter.compare_exchange_weak(expected, owner):
            # this should be yield
            waiter.wait()
            expected = Self.UNLOCKED

    fn unlock(inout self: Self, owner: Int) -> Bool:
        """Releases the lock.

        Args:
            owner: The lock's owner (usually an address).

        Returns:
            The successful release of the lock.
        """

        var expected = Int64(owner)
        if self.counter.load() != owner:
            # No one else can modify other than owner
            return False
        while not self.counter.compare_exchange_weak(expected, Self.UNLOCKED):
            expected = owner
        return True


struct BlockingScopedLock:
    """A scope adapter for BlockingSpinLock."""

    alias LockType = BlockingSpinLock
    """The type of the lock."""

    var lock: UnsafePointer[Self.LockType]
    """The underlying lock instance."""

    fn __init__(
        inout self,
        lock: UnsafePointer[Self.LockType],
    ):
        """Primary constructor.

        Args:
            lock: A pointer to the underlying lock.
        """

        self.lock = lock

    fn __init__(
        inout self,
        inout lock: Self.LockType,
    ):
        """Secondary constructor.

        Args:
            lock: A mutable reference to the underlying lock.
        """

        self.lock = UnsafePointer.address_of(lock)

    @no_inline
    fn __enter__(inout self):
        """Acquire the lock on entry.
        This is done by setting the owner of the lock to own address."""
        var address = UnsafePointer[Self].address_of(self)
        self.lock[].lock(int(address))

    @no_inline
    fn __exit__(inout self):
        """Release the lock on exit.
        Reset the address on the underlying lock."""
        var address = UnsafePointer[Self].address_of(self)
        _ = self.lock[].unlock(int(address))
