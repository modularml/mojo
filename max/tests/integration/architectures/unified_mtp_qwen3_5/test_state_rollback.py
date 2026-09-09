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

"""The accepted-row plan behind the Qwen3.5 speculative state rollback.

The plan decides which rows of the verify pass's per-token tensors the state
kernels re-run over, so an off-by-one here rewinds the recurrence to the wrong
length -- which produces a slightly wrong generation rather than an error. The
bit-exactness gate on the real kernels needs a GPU and lives in
``mach/docs/qwen38_27b/mtp-serving.md``; these are the arithmetic's own
invariants, including the prefill case where there are no drafts at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.pipelines.architectures.unified_mtp_qwen3_5.state_rollback import (
    accepted_row_plan,
)

CPU = DeviceRef.CPU()


def _plan(
    merged_offsets: list[int], num_accepted: list[int], num_draft_tokens: int
) -> tuple[np.ndarray, np.ndarray]:
    """Runs ``accepted_row_plan`` for one batch and returns its two outputs."""
    total_rows = merged_offsets[-1]
    types = [
        TensorType(DType.uint32, [len(merged_offsets)], device=CPU),
        TensorType(DType.int64, ["batch_size"], device=CPU),
        TensorType(DType.int64, [], device=CPU),
    ]
    with Graph("accepted_row_plan", input_types=types) as graph:
        offsets, accepted, k = (v.tensor for v in graph.inputs)
        rows, replay_offsets = accepted_row_plan(
            offsets, accepted, k, Dim(total_rows), CPU
        )
        graph.output(rows, replay_offsets)

    model = InferenceSession().load(graph)
    outputs = model.execute(
        Buffer.from_numpy(np.array(merged_offsets, dtype=np.uint32)),
        Buffer.from_numpy(np.array(num_accepted, dtype=np.int64)),
        Buffer.from_numpy(np.array(num_draft_tokens, dtype=np.int64)),
    )
    return outputs[0].to_numpy(), outputs[1].to_numpy()


@pytest.mark.parametrize(
    ("num_accepted", "expected_rows", "expected_offsets"),
    [
        # Every draft accepted: the replay covers the whole window.
        ([3], [0, 1, 2, 3], [0, 4]),
        ([2], [0, 1, 2], [0, 3]),
        ([1], [0, 1], [0, 2]),
        # Nothing accepted: only the token the previous step committed.
        ([0], [0], [0, 1]),
    ],
)
def test_a_decode_row_keeps_its_accepted_prefix(
    num_accepted: list[int],
    expected_rows: list[int],
    expected_offsets: list[int],
) -> None:
    rows, offsets = _plan([0, 4], num_accepted, 3)
    accepted_total = expected_offsets[-1]
    assert list(offsets) == expected_offsets
    assert list(rows[:accepted_total]) == expected_rows


def test_each_request_keeps_its_own_acceptance_length() -> None:
    # Three requests verifying four positions each, accepting 3 / 1 / 0 drafts.
    rows, offsets = _plan([0, 4, 8, 12], [3, 1, 0], 3)
    assert list(offsets) == [0, 4, 6, 7]
    assert list(rows[:7]) == [0, 1, 2, 3, 4, 5, 8]


def test_rows_past_the_accepted_total_stay_in_bounds() -> None:
    # The plan is deliberately not trimmed to the accepted total -- doing that
    # would need the total as a shape, which is a device-to-host sync. The
    # trailing entries are never read, but they must not index out of bounds.
    rows, offsets = _plan([0, 4, 8, 12], [0, 0, 0], 3)
    assert list(offsets) == [0, 1, 2, 3]
    assert len(rows) == 12
    assert rows.max() < 12
    assert rows.min() >= 0


def test_prefill_replays_the_whole_prompt() -> None:
    # With no drafts the accepted length is the merged length, so the same
    # expression that rewinds a decode step commits an entire prompt. Nothing
    # in the plan branches on the phase.
    rows, offsets = _plan([0, 5, 9], [0, 0], 0)
    assert list(offsets) == [0, 5, 9]
    assert list(rows) == list(range(9))
