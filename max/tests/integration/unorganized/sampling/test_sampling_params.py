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
"""Tests for SamplingParams validation (no GPU required)."""

import pytest
from max.pipelines.context import SamplingParams


def test_sampling_top_k() -> None:
    """Test that SamplingParams accepts large top_k values.

    The arbitrary limit of 255 for top_k has been removed to support models
    like Kimi 2.5 that may request larger top_k values. The underlying Mojo
    kernel's ternary search algorithm supports arbitrary k values.
    """
    # Large top_k values should be accepted (previously raised ValueError)
    params = SamplingParams(top_k=257)
    assert params.top_k == 257

    # Very large top_k values should also be accepted
    params = SamplingParams(top_k=1000)
    assert params.top_k == 1000

    # top_k=0 should be converted to -1 (sample all tokens)
    params = SamplingParams(top_k=0)
    assert params.top_k == -1

    # top_k=-1 (sample all tokens) should still work
    params = SamplingParams(top_k=-1)
    assert params.top_k == -1

    # top_k < -1 should still be rejected
    with pytest.raises(ValueError, match="top_k must be -1 or greater than 0"):
        SamplingParams(top_k=-2)


def test_temperature_zero_sets_top_k_to_one() -> None:
    sampling_params = SamplingParams(temperature=0.0)
    assert sampling_params.top_k == 1


def test_temperature_non_finite_rejected() -> None:
    """Regression guard: NaN/Inf temperature must not reach the kernel."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="temperature must be in"):
            SamplingParams(temperature=bad)


def test_penalty_out_of_range_rejected() -> None:
    """Penalties outside [-2.0, 2.0], or non-finite, are rejected (400 not 500)."""
    for bad in (2.5, -2.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="frequency_penalty must be in"):
            SamplingParams(frequency_penalty=bad)
        with pytest.raises(ValueError, match="presence_penalty must be in"):
            SamplingParams(presence_penalty=bad)


def test_penalty_in_range_accepted() -> None:
    for ok in (-2.0, 0.0, 1.3, 2.0):
        assert SamplingParams(frequency_penalty=ok).frequency_penalty == ok
        assert SamplingParams(presence_penalty=ok).presence_penalty == ok


def test_repetition_penalty_non_finite_rejected() -> None:
    """``NaN``/``Inf`` repetition_penalty must not reach the kernel divide.

    ``NaN <= 0`` and ``inf <= 0`` are both False, so the bound check alone
    would let non-finite values through to ``logit / repetition_penalty``.
    """
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(
            ValueError, match="repetition_penalty must be a finite value"
        ):
            SamplingParams(repetition_penalty=bad)


def test_repetition_penalty_non_positive_rejected() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(
            ValueError, match="repetition_penalty must be a finite value"
        ):
            SamplingParams(repetition_penalty=bad)


def test_repetition_penalty_in_range_accepted() -> None:
    for ok in (0.5, 1.0, 1.3, 2.0):
        assert SamplingParams(repetition_penalty=ok).repetition_penalty == ok


def test_token_count_rejects_bool() -> None:
    """``bool`` is an ``int`` subclass; ``max_tokens=true`` must not coerce to 1."""
    with pytest.raises(
        ValueError, match="max_new_tokens must be a non-negative integer"
    ):
        SamplingParams(max_new_tokens=True)
    with pytest.raises(
        ValueError, match="min_new_tokens must be a non-negative integer"
    ):
        SamplingParams(min_new_tokens=True)


def test_token_count_rejects_negative() -> None:
    with pytest.raises(
        ValueError, match="max_new_tokens must be a non-negative integer"
    ):
        SamplingParams(max_new_tokens=-1)
    with pytest.raises(
        ValueError, match="min_new_tokens must be a non-negative integer"
    ):
        SamplingParams(min_new_tokens=-1)
