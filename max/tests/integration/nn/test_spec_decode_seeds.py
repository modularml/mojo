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
"""Seed-family separation for sampled speculative decoding.

A request's per-execute seed is ``sampling_params.seed + len(ctx.tokens)``, so
it advances by however many tokens the last iteration committed -- 1 to
``K + 1`` under speculation, rather than always 1. Two invariants keep the
sampling streams apart under that advance:

1. No draft step key repeats across iterations, for any commit count a
   speculative iteration can produce.
2. No residual-recovery key ever equals a draft-proposal key, in either
   adjacent-iteration ordering.

The offsets come from the production helpers, so a call site reverting to
consecutive integers changes what these assert.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType
from max.nn.sampling.rejection_sampler import (
    _SEED_GOLDEN_GAMMA,
    _recovery_row_offset,
    _recovery_seed_rows,
    _seed_offset,
)

_U64 = 1 << 64

# Draft width in production recipes; the bound on how far one iteration can
# advance the base seed is num_speculative_tokens + 1 (drafts plus the bonus).
_MAX_DRAFT_STEPS = 8
_MAX_COMMITTED = _MAX_DRAFT_STEPS + 1
_MAX_BATCH = 64

_DRAFT_OFFSETS = [_seed_offset(step) for step in range(_MAX_DRAFT_STEPS)]
_RECOVERY_OFFSETS = [_recovery_row_offset(row) for row in range(_MAX_BATCH)]


def test_draft_offsets_are_distinct() -> None:
    assert len(set(_DRAFT_OFFSETS)) == len(_DRAFT_OFFSETS)


def test_step_zero_leaves_the_base_seed_alone() -> None:
    """Why ``_draft_step_seed`` may return the seed unchanged at step 0."""
    assert _seed_offset(0) == 0


def test_draft_offsets_survive_the_token_count_advance() -> None:
    """No step key repeats once the base seed advances by a commit count.

    This is the collision the spacing exists to remove: with consecutive
    integers, an iteration committing ``c`` tokens puts the next iteration's
    step ``s`` on this iteration's step ``s + c``.
    """
    draft = set(_DRAFT_OFFSETS)
    for committed in range(1, _MAX_COMMITTED + 1):
        for step, offset in enumerate(_DRAFT_OFFSETS):
            key = (committed + offset) % _U64
            assert key not in draft, (
                f"draft step {step} at commit advance {committed} reuses the "
                f"key of another step"
            )


def test_recovery_and_draft_offsets_never_meet() -> None:
    """Recovery and proposal draws must not share a key, in either ordering.

    Row ``b``'s recovery draw and draft step ``b``'s proposal draw hang off
    the same base seed. Walking the same small integers puts them on the same
    key whenever ``b == step``, which is what the recovery domain tag removes.
    Both adjacent-iteration orderings matter: a recovery key can be carried
    forward onto a later draft key, and a draft key onto a later recovery key.
    """
    draft = set(_DRAFT_OFFSETS)
    recovery = set(_RECOVERY_OFFSETS)
    assert not (draft & recovery)

    for committed in range(1, _MAX_COMMITTED + 1):
        for row, offset in enumerate(_RECOVERY_OFFSETS):
            assert (committed + offset) % _U64 not in draft, (
                f"recovery row {row} at commit advance {committed} shares a "
                f"key with a draft step"
            )
        for step, offset in enumerate(_DRAFT_OFFSETS):
            assert (committed + offset) % _U64 not in recovery, (
                f"draft step {step} at commit advance {committed} shares a "
                f"key with a recovery row"
            )


def test_recovery_offsets_are_distinct_per_row() -> None:
    assert len(set(_RECOVERY_OFFSETS)) == len(_RECOVERY_OFFSETS)


def test_gamma_is_odd() -> None:
    """Odd multipliers are bijective mod 2**64, so distinct steps stay distinct."""
    assert _SEED_GOLDEN_GAMMA % 2 == 1


def test_recovery_seed_rows_use_the_row_offsets(
    session: InferenceSession,
) -> None:
    """The graph the sampler builds must apply ``_recovery_row_offset``.

    The arithmetic above pins the offsets; this pins that the residual-recovery
    path actually spends them, so reverting its rows to consecutive integers
    fails here rather than passing silently.
    """
    batch, num_steps = 4, 3
    device = DeviceRef.CPU()

    with Graph(
        "recovery_seed_rows",
        input_types=[TensorType(DType.uint64, ["batch_size"], device=device)],
    ) as graph:
        seed = graph.inputs[0].tensor
        graph.output(
            _recovery_seed_rows(seed, Dim("batch_size"), Dim(num_steps), device)
        )

    base = np.arange(batch, dtype=np.uint64) * np.uint64(1_000_000)
    model = session.load(graph)
    rows = model(Buffer.from_numpy(base))[0].to_numpy()

    expected = np.array(
        [
            (int(base[row]) + _recovery_row_offset(row)) % _U64
            for row in range(batch)
            for _ in range(num_steps)
        ],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(rows, expected)
