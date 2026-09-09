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

"""Defines non-blocking queue protocols and utility functions for producer-consumer patterns in MAX."""

from __future__ import annotations

import queue
import time
from typing import (
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

__all__ = [
    "MAXAsyncPullQueue",
    "MAXAsyncPushQueue",
    "MAXPullQueue",
    "MAXPushQueue",
    "drain_queue",
    "get_blocking",
]

_PushItemType = TypeVar("_PushItemType", contravariant=True)
_PullItemType = TypeVar("_PullItemType", covariant=True)


@runtime_checkable
class MAXPushQueue(Protocol, Generic[_PushItemType]):
    """Protocol for a minimal, non-blocking push queue interface in MAX.

    This protocol defines the contract for a queue that supports non-blocking
    put operations for adding items. It is generic over the item type and designed
    for scenarios where the caller must be immediately notified of success or failure
    rather than waiting for space to become available.

    The protocol is intended for producer-side queue operations where immediate
    feedback is critical for proper flow control and error handling.
    """

    def put_nowait(self, item: _PushItemType) -> None:
        """Attempt to put an item into the queue without blocking.

        This method is designed to immediately fail (typically by raising an exception)
        if the item cannot be added to the queue at the time of the call. Unlike the
        traditional 'put' method in many queue implementations, which may block until
        space becomes available or the transfer is completed - this method never waits.
        It is intended for use cases where the caller must be notified of failure to
        enqueue immediately, rather than waiting for space.

        Args:
            item: The item to be added to the queue.
        """
        ...


@runtime_checkable
class MAXPullQueue(Protocol, Generic[_PullItemType]):
    """Protocol for a minimal, non-blocking pull queue interface in MAX.

    This protocol defines the contract for a queue that supports non-blocking
    get operations for retrieving items. It is generic over the item type and designed
    for scenarios where the caller must be immediately notified if no items are available
    rather than waiting for items to arrive.

    The protocol is intended for consumer-side queue operations where immediate
    feedback about queue state is critical for proper flow control and error handling.
    """

    def get_nowait(self) -> _PullItemType:
        """Remove and return an item from the queue without blocking.

        This method is expected to raise `queue.Empty` if no item is available
        to retrieve from the queue.

        Returns:
            The item removed from the queue.

        Raises:
            queue.Empty: If the queue is empty and no item can be retrieved.
        """
        ...


@runtime_checkable
class MAXAsyncPushQueue(Protocol, Generic[_PushItemType]):
    """Protocol for an async push queue that supports both blocking and non-blocking put."""

    async def put(self, item: _PushItemType) -> None:
        """Send an item, awaiting until the queue is ready to accept it."""
        ...

    def put_nowait(self, item: _PushItemType) -> None:
        """Attempt to put an item without blocking; raises immediately on failure."""
        ...

    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        """Whether ``put`` can currently accept an item without blocking.

        A point-in-time check when ``timeout_s`` is 0: ``False`` means the queue
        is full (the consumer has stopped draining) or has no consumer connected
        yet. A positive ``timeout_s`` waits up to that long for the queue to
        become writable; ``None`` blocks indefinitely.
        """
        ...


@runtime_checkable
class MAXAsyncPullQueue(Protocol, Generic[_PullItemType]):
    """Protocol for an async pull queue that supports both blocking and non-blocking get."""

    async def get(self) -> _PullItemType:
        """Receive an item, awaiting until one is available."""
        ...

    def get_nowait(self) -> _PullItemType:
        """Remove and return an item without blocking; raises queue.Empty if unavailable."""
        ...


def drain_queue(
    pull_queue: MAXPullQueue[_PullItemType], max_items: int | None = None
) -> list[_PullItemType]:
    """Remove and return items from the queue without blocking.

    This method is expected to return an empty list if the queue is empty.
    If max_items is specified, at most that many items will be returned.

    Args:
        pull_queue: The queue to drain items from.
        max_items: Maximum number of items to return. If None, returns all items.

    Returns:
        List of items removed from the queue, limited by max_items if specified.
    """
    output: list[_PullItemType] = []
    while True:
        if max_items is not None and len(output) >= max_items:
            break

        try:
            output.append(pull_queue.get_nowait())
        except queue.Empty:
            break
    return output


def get_blocking(pull_queue: MAXPullQueue[_PullItemType]) -> _PullItemType:
    """Get the next item from the queue.

    If no item is available, this method will spin until one is,
    sleeping 1 ms between attempts (minimum latency floor of ~1 ms).
    """
    while True:
        try:
            return pull_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.001)
