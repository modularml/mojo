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
# tests/integration/interfaces/test_eos_tracking.py
"""Unit tests for EOSTracker (EOS single-ID, sequence, and strings stop sequences)."""

import numpy as np
import pytest
from max.pipelines.context.eos_tracking import EOSTracker


def test_is_eos_single_token() -> None:
    t = EOSTracker(eos_token_ids={99})
    assert t.is_eos_from_tokens(np.array([99], dtype=np.int64)) is True
    assert t.is_eos_from_tokens(np.array([100], dtype=np.int64)) is False


def test_is_eos_sequence_suffix() -> None:
    t = EOSTracker(eos_sequences=[[30, 31]])
    assert t.is_eos_from_tokens(np.array([30, 31], dtype=np.int64)) is True
    assert t.is_eos_from_tokens(np.array([30], dtype=np.int64)) is False
    assert t.is_eos_from_tokens(np.array([1, 30, 31], dtype=np.int64)) is True
    assert t.is_eos_from_tokens(np.array([1, 30, 32], dtype=np.int64)) is False


def test_is_eos_empty() -> None:
    """checks empty eos sequences"""
    t = EOSTracker()
    assert t.is_eos_from_tokens(np.array([1, 2, 3], dtype=np.int64)) is False


def test_combined_eos_token_and_sequence() -> None:
    t = EOSTracker(eos_token_ids={99}, eos_sequences=[[30, 31]])
    assert (
        t.is_eos_from_tokens(np.array([10, 99], dtype=np.int64)) is True
    )  # single-ID path
    assert (
        t.is_eos_from_tokens(np.array([30, 31], dtype=np.int64)) is True
    )  # sequence path
    assert t.is_eos_from_tokens(np.array([10, 20], dtype=np.int64)) is False


def test_multiple_eos_sequences() -> None:
    t = EOSTracker(eos_sequences=[[5, 6], [7, 8, 9]])
    assert t.is_eos_from_tokens(np.array([1, 5, 6], dtype=np.int64)) is True
    assert t.is_eos_from_tokens(np.array([1, 7, 8, 9], dtype=np.int64)) is True
    assert t.is_eos_from_tokens(np.array([1, 5, 9], dtype=np.int64)) is False


def test_sequence_check_with_numpy_input() -> None:
    t = EOSTracker(eos_sequences=[[30, 31]])
    assert t.is_eos_from_tokens(np.array([1, 30, 31], dtype=np.int64)) is True


def test_eq_different_sequence_lengths() -> None:
    a = EOSTracker(eos_sequences=[[1, 2]])
    b = EOSTracker(eos_sequences=[[1, 2], [3, 4]])
    assert a != b


def test_last_token_numpy_scalar_matches_python_int_in_set() -> None:
    """Document that np.ndarray[-1] is np.integer while eos_token_ids holds Python int."""
    t = EOSTracker(eos_token_ids={128001})
    generated = np.array([1, 2, 128001], dtype=np.int64)
    last = generated[-1]
    assert isinstance(last, np.integer)
    assert t.is_eos_from_tokens(generated) is True


def test_eos_sequence_list_int_vs_int64_suffix_array_equal() -> None:
    """eos_sequences are list[list[int]]; generated suffix is int64 — np.array_equal must match."""
    t = EOSTracker(eos_sequences=[[1000000, 1000001]])
    generated = np.array([9, 1000000, 1000001], dtype=np.int64)
    assert t.is_eos_from_tokens(generated) is True


def test_is_eos_from_tokens_empty_array_raises() -> None:
    t = EOSTracker(eos_token_ids={1})
    with pytest.raises(ValueError, match="non-empty"):
        t.is_eos_from_tokens(np.array([], dtype=np.int64))


# --- first_eos_offset (multi-token committed span scan) ---


def test_first_eos_offset_none_when_span_does_not_end() -> None:
    t = EOSTracker(eos_token_ids={1})
    prior = np.array([99], dtype=np.int64)
    assert t.first_eos_offset(prior, [8, 9, 10]) is None


def test_first_eos_offset_single_id_interior() -> None:
    t = EOSTracker(eos_token_ids={1})
    prior = np.array([99], dtype=np.int64)
    # Span [8, 1, 9]: the single-id EOS at offset 1 terminates first.
    assert t.first_eos_offset(prior, [8, 1, 9]) == 1


def test_first_eos_offset_returns_earliest_eos() -> None:
    t = EOSTracker(eos_token_ids={1})
    prior = np.array([99], dtype=np.int64)
    # Two EOS in the span; the earliest (offset 0) wins.
    assert t.first_eos_offset(prior, [1, 2, 1]) == 0


def test_first_eos_offset_sequence_within_span() -> None:
    t = EOSTracker(eos_sequences=[[5, 6]])
    prior = np.array([99], dtype=np.int64)
    # Sequence [5, 6] completes at offset 1 (the token that closes it).
    assert t.first_eos_offset(prior, [5, 6, 8]) == 1


def test_first_eos_offset_sequence_straddling_boundary() -> None:
    t = EOSTracker(eos_sequences=[[5, 6]])
    # Prior ends in 5; the span's first token (6) completes the sequence.
    prior = np.array([99, 5], dtype=np.int64)
    assert t.first_eos_offset(prior, [6, 8]) == 0


def test_first_eos_offset_long_sequence_short_prior() -> None:
    # 4-token stop sequence: keep=3, but only 2 prior tokens exist, so the whole
    # prior must be retained for the straddling match (index must not go
    # negative and drop the earliest token).
    t = EOSTracker(eos_sequences=[[11, 12, 20, 21]])
    prior = np.array([11, 12], dtype=np.int64)
    # span 20, 21 completes [11, 12, 20, 21] at offset 1.
    assert t.first_eos_offset(prior, [20, 21, 99]) == 1


def test_first_eos_offset_empty_prior() -> None:
    t = EOSTracker(eos_token_ids={1})
    prior = np.array([], dtype=np.int64)
    assert t.first_eos_offset(prior, [8, 1]) == 1


# --- EOS Stop StringSequence (s) ---


def test_is_eos_from_string_none() -> None:
    t = EOSTracker()
    assert t.is_eos_from_string("abc") is None


def test_is_eos_from_string_list() -> None:
    stop = ["abc", "abcdef"]
    t = EOSTracker(eos_stop_strings=stop)
    assert t.is_eos_from_string("a") is None
    assert t.is_eos_from_string("b") is None
    assert t.is_eos_from_string("c") == "abc"


def test_is_eos_from_string_str() -> None:
    stop = ["abc"]
    t = EOSTracker(eos_stop_strings=stop)
    assert t.is_eos_from_string("all good here") is None
    assert t.is_eos_from_string("ab") is None
    assert t.is_eos_from_string("c") == "abc"


def test_is_eos_from_string_long_continuation() -> None:
    stop = ["abc"]
    t = EOSTracker(eos_stop_strings=stop)
    for c in "long continuation" * 1024:
        assert t.is_eos_from_string(c) is None
    assert t.is_eos_from_string("abc") == "abc"


def test_earliest_position_wins() -> None:
    t = EOSTracker(eos_stop_strings=["cd", "ab"])
    assert t.is_eos_from_string("abcd") == "ab"  # pos 0 beats pos 2


def test_single_call_match() -> None:
    t = EOSTracker(eos_stop_strings=["stop"])
    assert t.is_eos_from_string("Please") is None
    assert t.is_eos_from_string("stop") == "stop"


def test_empty_token() -> None:
    t = EOSTracker(eos_stop_strings=["ab"])
    assert t.is_eos_from_string("") is None
    assert t.is_eos_from_string("ab") == "ab"
