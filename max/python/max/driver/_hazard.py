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

"""Host<->device hazard stamping for graph realization.

The binding owns the records, the waits, and the copy and fill sites. Graph
realization stamps from Python instead, because its written set comes from
graph metadata rather than from the call, and its failure arrives as a Python
exception.
"""

from __future__ import annotations

import traceback
from collections.abc import Sequence

from max._core.driver import (
    Buffer,
    DeviceQueue,
    HostHazardCompletion,
)
from max._core.driver import HostHazardError as HostHazardError

QueueKey = int


def render_producer_error(exc: BaseException) -> str:
    """Renders a failed producer's traceback to text.

    Text, not the exception: its frames reference the buffers being written,
    and ``Buffer`` is not GC-tracked, so the cycle would leak the allocation.
    """
    return "".join(traceback.format_exception(exc)).rstrip()


def stamp_device_write(
    written: Sequence[Buffer],
    read: Sequence[Buffer] = (),
    error: str | None = None,
) -> None:
    """Stamps one completion per distinct queue, grouped by each buffer's own.

    Not for copies: a cross-device copy is enqueued on the destination's queue,
    so a source stamped on its own is a silent WAR hole. Copy sites stamp in
    the binding.

    Args:
        written: Buffers the submission wrote.
        read: Buffers it only read.
        error: Rendered traceback when the submission failed after enqueuing.
            The buffers take a poisoned token, so a later host access raises.
    """
    groups: dict[QueueKey, tuple[DeviceQueue, list[Buffer], list[Buffer]]] = {}
    for buffers, writes_it in ((written, True), (read, False)):
        for buf in buffers:
            # The binding no-ops on these anyway; filtering here is what stops
            # an all-untracked group recording an event nothing will hold.
            if not buf._host_hazard_tracked:
                continue
            queue = buf.stream
            _, writes, reads = groups.setdefault(
                queue._device_context_ptr(), (queue, [], [])
            )
            (writes if writes_it else reads).append(buf)

    for queue, writes, reads in groups.values():
        completion = (
            HostHazardCompletion.poisoned(queue, error)
            if error is not None
            else HostHazardCompletion.record_on(queue)
        )
        # A mutated input is in both lists, and a write stamp clears the
        # reader set, so the reverse order would wipe the reader just added.
        for buf in writes:
            buf._stamp_write(completion)
        for buf in reads:
            buf._stamp_read(completion)
