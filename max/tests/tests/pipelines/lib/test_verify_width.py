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
"""How many carried drafts a step verifies, and the host mirror of that trim.

The width is a pure function of the batch, because two different actors shape
buffers from it -- the execute path builds the draft array, and the async
FSM callback fills the constrained-decoding bitmask -- and they have to agree
without communicating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pytest
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
    _host_mirror_realized_drafts,
    _verify_width_lookup,
)
from max.pipelines.speculative.config import (
    SpeculativeConfig,
    VerifyWidthRange,
)


@dataclass
class _Tokens:
    generated_length: int


@dataclass
class _Ctx:
    tokens: _Tokens


@dataclass
class _Inputs:
    batches: list[list[_Ctx]] = field(default_factory=list)

    @property
    def flat_batch(self) -> list[_Ctx]:
        return [ctx for batch in self.batches for ctx in batch]


def _decode_batches(*sizes: int) -> _Inputs:
    """Replica batches whose every row has already generated a token."""
    return _Inputs([[_Ctx(_Tokens(1)) for _ in range(n)] for n in sizes])


def _width(
    inputs: _Inputs, *, configured: int, lookup: list[int] | None
) -> int:
    pipeline = object.__new__(OverlapTextGenerationPipeline)
    spec_state = type("_S", (), {"num_speculative_tokens": configured})()
    pipeline._spec_decode_state = spec_state
    pipeline._width_lookup = lookup
    return OverlapTextGenerationPipeline._verify_width(
        pipeline, cast(Any, inputs)
    )


def test_no_schedule_verifies_every_carried_draft() -> None:
    assert _width(_decode_batches(8), configured=3, lookup=None) == 3


def test_schedule_selects_by_batch_size() -> None:
    # batch_size -> width; index 0 unused.
    lookup = [0, 3, 3, 3, 1, 1]
    assert _width(_decode_batches(2), configured=3, lookup=lookup) == 3
    assert _width(_decode_batches(5), configured=3, lookup=lookup) == 1
    # A batch size past the lookup tail saturates rather than indexing off it.
    assert _width(_decode_batches(999), configured=3, lookup=lookup) == 1


def test_a_row_with_no_generated_token_collapses_the_width() -> None:
    """A freshly prefilled row carries no proposals, so nothing is verified."""
    inputs = _Inputs([[_Ctx(_Tokens(1)), _Ctx(_Tokens(0))]])
    assert _width(inputs, configured=3, lookup=None) == 0


def test_empty_batch_does_not_index_off_the_lookup() -> None:
    assert _width(_Inputs([]), configured=3, lookup=[0, 3, 1]) == 3


def test_width_is_the_per_replica_maximum_not_the_total() -> None:
    """Under DP the schedule is indexed by per-replica decode batch size.

    Eight requests spread over four replicas is a per-replica batch of 2, which
    is a small batch on every GPU -- not a large one.
    """
    lookup = [0, 3, 3, 3, 1, 1, 1, 1, 1]
    inputs = _decode_batches(2, 2, 2, 2)
    assert len(inputs.flat_batch) == 8
    assert _width(inputs, configured=3, lookup=lookup) == 3


# ---------------------------------------------------------------------------
# the schedule as it arrives from SpeculativeConfig
# ---------------------------------------------------------------------------


def _config(
    schedule: list[tuple[int, int, int]] | None, *, method: str = "eagle"
) -> SpeculativeConfig:
    return SpeculativeConfig(
        speculative_method=cast(Any, method),
        num_speculative_tokens=3,
        num_speculative_tokens_per_batch_size=(
            None
            if schedule is None
            else [
                VerifyWidthRange(
                    batch_start=start, batch_end=end, num_tokens=count
                )
                for start, end, count in schedule
            ]
        ),
    )


def test_no_schedule_leaves_the_width_at_the_configured_depth() -> None:
    assert _verify_width_lookup(_config(None), 3, 8) is None
    # A schedule set on a pipeline that does not speculate at all.
    assert _verify_width_lookup(_config([(1, 8, 1)]), 0, 8) is None


def test_schedule_builds_a_dense_lookup() -> None:
    config = _config([(1, 2, 3), (3, 8, 1)])
    assert _verify_width_lookup(config, 3, 8) == [0, 3, 3, 1, 1, 1, 1, 1, 1]


def test_block_drafters_apply_the_schedule_too() -> None:
    """A block drafter's block width is baked into its checkpoint, but the
    target still only verifies a prefix of that block per the schedule --
    the draft itself always produces the full block regardless.
    """
    config = _config([(1, 2, 3), (3, 8, 1)], method="dflash")
    assert config.num_speculative_tokens_per_batch_size is not None
    assert _verify_width_lookup(config, 3, 8) == [0, 3, 3, 1, 1, 1, 1, 1, 1]


def test_config_rejects_a_schedule_with_a_gap_at_the_front() -> None:
    with pytest.raises(ValueError, match="must start at batch size 1"):
        _config([(2, 8, 1)])


# ---------------------------------------------------------------------------
# the host mirror of the device realize-scatter
# ---------------------------------------------------------------------------
#
# The mirror has to trim the previous step's proposals to this step's verify
# width exactly as the device graph does. The previous step always drafted the
# configured depth, so once a step verifies fewer than it drafted the two
# arrays stop having the same width -- and mirroring untrimmed raised
# "could not broadcast input array from shape (3,) into shape (1,)".

_MAGIC = -7


def test_mirror_trims_to_a_narrower_verify_width() -> None:
    """Drafted 3, verifying 1: the tail is dropped, not an error."""
    realized = _host_mirror_realized_drafts(
        np.full((2, 1), _MAGIC, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
    )
    np.testing.assert_array_equal(realized, np.array([[11], [21]]))


def test_mirror_leaves_equal_widths_unchanged() -> None:
    prev_next = np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64)
    realized = _host_mirror_realized_drafts(
        np.full((2, 3), _MAGIC, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        prev_next,
    )
    np.testing.assert_array_equal(realized, prev_next)


def test_mirror_of_a_zero_verify_width_is_an_empty_array() -> None:
    """A prefill->decode boundary step verifies nothing."""
    realized = _host_mirror_realized_drafts(
        np.zeros((2, 0), dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([[11, 12, 13], [21, 22, 23]], dtype=np.int64),
    )
    assert realized.shape == (2, 0)
