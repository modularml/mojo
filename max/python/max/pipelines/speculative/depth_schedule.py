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
"""How many drafted tokens to verify, as a function of decode batch size.

For now we draft K tokens but only verify a subset of them depending on the schedule.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "DepthScheduleEntry",
    "build_depth_lookup",
    "normalize_depth_schedule",
]

#: One inclusive batch-size range and the draft depth to use across it.
DepthScheduleEntry = tuple[int, int, int]


def normalize_depth_schedule(
    entries: Sequence[Sequence[int]],
) -> list[DepthScheduleEntry]:
    """Validates a schedule and returns it sorted by batch size.

    Args:
        entries: Triples of ``(batch_start, batch_end, depth)``.

    Returns:
        The entries as tuples, sorted ascending by ``batch_start``.

    Raises:
        ValueError: If ``entries`` is empty, an entry is not a triple, a range
            is not positive and ascending, a depth is negative, ranges overlap,
            or the first range does not start at batch size 1. Starting at 1 is
            required so every runtime batch size resolves to a depth.
    """
    if not entries:
        raise ValueError(
            "A depth schedule must name at least one batch-size range, e.g. "
            "[(1, 16, 3), (17, 64, 1)]."
        )
    parsed: list[DepthScheduleEntry] = []
    for entry in entries:
        if len(entry) != 3:
            raise ValueError(
                "Each depth-schedule entry must be a "
                f"(batch_start, batch_end, depth) triple, got {tuple(entry)!r}."
            )
        start, end, depth = (int(entry[0]), int(entry[1]), int(entry[2]))
        if start <= 0 or end <= 0:
            raise ValueError(
                f"Batch-size range ({start}, {end}) must be positive."
            )
        if start > end:
            raise ValueError(
                f"Batch-size range ({start}, {end}) must have start <= end."
            )
        if depth < 0:
            raise ValueError(
                f"Draft depth {depth} for batch sizes {start}-{end} must be "
                "non-negative."
            )
        parsed.append((start, end, depth))

    parsed.sort(key=lambda entry: entry[0])

    previous_end = 0
    for start, end, _ in parsed:
        if start <= previous_end:
            raise ValueError(
                "Depth-schedule batch-size ranges must not overlap; "
                f"({start}, {end}) starts at or before {previous_end}."
            )
        previous_end = end

    if parsed[0][0] != 1:
        raise ValueError(
            "The first depth-schedule range must start at batch size 1 so "
            f"every runtime batch size has a depth; it starts at {parsed[0][0]}."
        )
    return parsed


def build_depth_lookup(
    schedule: Sequence[DepthScheduleEntry],
    *,
    max_batch_size: int,
    max_depth: int,
) -> list[int]:
    """Expands a schedule into a dense ``batch_size -> depth`` lookup.

    Index 0 is unused so a runtime batch size indexes the list directly. Gaps
    between configured ranges carry the previous range's depth forward, as does
    the tail past the final range, so the lookup is total over
    ``1..max_batch_size``.

    Args:
        schedule: A schedule already through
            :func:`normalize_depth_schedule`.
        max_batch_size: Largest decode batch size the lookup must cover.
        max_depth: Ceiling applied to every entry. A schedule cannot ask to
            verify more drafts than the step actually carries, which is the
            configured ``num_speculative_tokens``.

    Returns:
        A list of length ``max_batch_size + 1`` whose element ``i`` is the
        number of drafts to verify at batch size ``i``.

    Raises:
        ValueError: If ``max_batch_size`` or ``max_depth`` is not positive.
    """
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be > 0, got {max_batch_size}.")
    if max_depth <= 0:
        raise ValueError(f"max_depth must be > 0, got {max_depth}.")

    lookup = [0] * (max_batch_size + 1)
    next_batch_size = 1
    carried: int | None = None

    for start, end, depth in schedule:
        if start > next_batch_size and carried is not None:
            for batch_size in range(
                next_batch_size, min(start, max_batch_size + 1)
            ):
                lookup[batch_size] = min(max_depth, carried)
        for batch_size in range(
            max(start, next_batch_size), min(end, max_batch_size) + 1
        ):
            lookup[batch_size] = min(max_depth, depth)
        next_batch_size = max(next_batch_size, end + 1)
        carried = depth
        if next_batch_size > max_batch_size:
            break

    assert carried is not None, "normalize_depth_schedule rejects empty input"
    for batch_size in range(next_batch_size, max_batch_size + 1):
        lookup[batch_size] = min(max_depth, carried)
    return lookup
