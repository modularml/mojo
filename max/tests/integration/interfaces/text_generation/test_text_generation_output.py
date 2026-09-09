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

from __future__ import annotations

import pytest
from max.pipelines.context import (
    GenerationStatus,
    LogProbabilities,
    TextGenerationOutput,
)
from max.pipelines.modeling.types import RequestID


def _log_probs(i: int) -> LogProbabilities:
    return LogProbabilities([float(i)], [{i: float(i)}])


def test_combine_many() -> None:
    request_id = RequestID("1")
    a = TextGenerationOutput(
        request_id=request_id,
        tokens=[1, 2],
        final_status=GenerationStatus.ACTIVE,
        log_probabilities=[_log_probs(1), _log_probs(2)],
    )
    b = TextGenerationOutput(
        request_id=request_id,
        tokens=[3, 4, 5],
        final_status=GenerationStatus.ACTIVE,
        log_probabilities=[_log_probs(3), _log_probs(4), _log_probs(5)],
    )
    c = TextGenerationOutput(
        request_id=request_id,
        tokens=[6],
        final_status=GenerationStatus.END_OF_SEQUENCE,
        log_probabilities=[_log_probs(6)],
    )

    expected = TextGenerationOutput(
        request_id=request_id,
        tokens=[1, 2, 3, 4, 5, 6],
        final_status=GenerationStatus.END_OF_SEQUENCE,
        log_probabilities=[
            _log_probs(1),
            _log_probs(2),
            _log_probs(3),
            _log_probs(4),
            _log_probs(5),
            _log_probs(6),
        ],
    )
    actual = TextGenerationOutput.merge([a, b, c])
    assert actual == expected


def test_combine_many_with_no_log_probabilities() -> None:
    request_id = RequestID("1")
    a = TextGenerationOutput(
        request_id=request_id,
        tokens=[1, 2],
        final_status=GenerationStatus.ACTIVE,
    )
    b = TextGenerationOutput(
        request_id=request_id,
        tokens=[3, 4, 5],
        final_status=GenerationStatus.ACTIVE,
    )
    c = TextGenerationOutput(
        request_id=request_id,
        tokens=[6],
        final_status=GenerationStatus.END_OF_SEQUENCE,
    )

    expected = TextGenerationOutput(
        request_id=request_id,
        tokens=[1, 2, 3, 4, 5, 6],
        final_status=GenerationStatus.END_OF_SEQUENCE,
        log_probabilities=None,
    )
    actual = TextGenerationOutput.merge([a, b, c])
    assert actual == expected


def test_combine_many_with_mixed_log_probabilities() -> None:
    request_id = RequestID("1")
    a = TextGenerationOutput(
        request_id=request_id,
        tokens=[1, 2],
        final_status=GenerationStatus.ACTIVE,
        log_probabilities=[_log_probs(1), _log_probs(2)],
    )
    b = TextGenerationOutput(
        request_id=request_id,
        tokens=[3, 4, 5],
        final_status=GenerationStatus.ACTIVE,
        log_probabilities=None,
    )
    with pytest.raises(ValueError):
        _ = TextGenerationOutput.merge([a, b])


def test_combine_empty() -> None:
    with pytest.raises(ValueError):
        _ = TextGenerationOutput.merge([])
