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
"""The verify-width schedule: validation and the dense lookup."""

from __future__ import annotations

import pytest
from max.pipelines.speculative.depth_schedule import (
    build_depth_lookup,
    normalize_depth_schedule,
)


def test_sorts_by_batch_start() -> None:
    assert normalize_depth_schedule([(17, 64, 1), (1, 16, 3)]) == [
        (1, 16, 3),
        (17, 64, 1),
    ]


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ([], "at least one batch-size range"),
        ([(1, 16)], "triple"),
        # A batch size with two depths has no answer.
        ([(1, 16, 3), (10, 64, 1)], "must not overlap"),
        # Abutting on the same value is still an overlap.
        ([(1, 16, 3), (16, 64, 1)], "must not overlap"),
        # Every runtime batch size must resolve to a depth.
        ([(2, 16, 3)], "must start at batch size 1"),
        ([(1, 16, 3), (17, 16, 1)], "must have start <= end"),
    ],
)
def test_rejects_malformed_schedules(
    schedule: list[tuple[int, ...]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_depth_schedule(schedule)


def test_lookup_is_total_and_one_indexed() -> None:
    # Batch sizes 5-8 are unnamed, so they carry 3 forward rather than falling
    # to zero, as does the tail past the final range.
    lookup = build_depth_lookup(
        [(1, 4, 3), (9, 10, 1)], max_batch_size=12, max_depth=3
    )
    # Index 0 is unused so a runtime batch size indexes directly.
    assert lookup == [0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1]


def test_max_depth_clamps_every_entry() -> None:
    # A schedule cannot ask to verify more drafts than the step carries.
    lookup = build_depth_lookup(
        [(1, 4, 7), (5, 8, 1)], max_batch_size=8, max_depth=3
    )
    assert lookup[1:] == [3, 3, 3, 3, 1, 1, 1, 1]
