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
"""Comprehensive TokenBuffer integration tests covering the public API surface."""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.context.tokens import Range, TokenBuffer


def test_token_buffer__tokens_validation_during_init() -> None:
    # Token array must be one-dimensional
    with pytest.raises(ValueError):
        TokenBuffer(array=np.array([[1, 2, 3], [3, 4, 5]], dtype=np.int64))

    # Token array must be int64
    with pytest.raises(ValueError):
        TokenBuffer(array=np.array([1, 2, 3], dtype=np.int32))

    # Token array cannot be empty
    with pytest.raises(ValueError):
        TokenBuffer(array=np.array([], dtype=np.int64))

    tokens = np.array([1, 2, 3], dtype=np.int64)
    token_buffer = TokenBuffer(array=tokens)

    np.testing.assert_array_equal(token_buffer.all, tokens)
    np.testing.assert_array_equal(token_buffer.prompt, tokens)
    np.testing.assert_array_equal(token_buffer.active, tokens)
    np.testing.assert_array_equal(
        token_buffer.generated, np.array([], dtype=np.int64)
    )

    assert token_buffer.generated.size == 0
    assert token_buffer.prompt_length == 3
    assert len(token_buffer) == 3
    assert token_buffer.generated_length == 0
    assert token_buffer.active_length == 3
    assert token_buffer.processed_length == 0


def test_token_buffer__generated_and_completion_tracking() -> None:
    """Exercise token addition, generation tracking, and completion flow."""
    token_buffer = TokenBuffer(array=np.array([10, 11, 12], dtype=np.int64))

    assert token_buffer.processed_length == 0

    with pytest.raises(ValueError):
        _ = token_buffer.consume_recently_generated_tokens()

    token_buffer.advance_with_token(13)
    token_buffer.advance_with_token(14)

    assert token_buffer.processed_length == 4

    np.testing.assert_array_equal(
        token_buffer.generated, np.array([13, 14], dtype=np.int64)
    )
    assert token_buffer.generated_length == 2
    assert len(token_buffer) == 5
    np.testing.assert_array_equal(
        token_buffer[-2:], np.array([13, 14], dtype=np.int64)
    )
    assert token_buffer[-1] == 14

    completed = token_buffer.consume_recently_generated_tokens()
    np.testing.assert_array_equal(completed, np.array([13, 14], dtype=np.int64))

    assert token_buffer.processed_length == 4

    with pytest.raises(ValueError):
        _ = token_buffer.consume_recently_generated_tokens()

    token_buffer.advance_with_token(15)
    assert token_buffer.processed_length == 5
    np.testing.assert_array_equal(
        token_buffer.consume_recently_generated_tokens(),
        np.array([15], dtype=np.int64),
    )


def test_token_buffer__chunking_behaviors() -> None:
    """Validate maybe_chunk and chunk reset semantics."""
    token_buffer = TokenBuffer(
        array=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    )

    assert token_buffer.processed_length == 0

    # We cannot advance a chunk if it is not actively chunked.
    with pytest.raises(ValueError):
        token_buffer.advance_chunk()

    token_buffer.chunk(2)
    assert token_buffer.actively_chunked
    assert token_buffer.active_length == 2
    assert token_buffer.processed_length == 0
    np.testing.assert_array_equal(
        token_buffer.active, np.array([0, 1], dtype=np.int64)
    )

    # We can advance a chunk if it is actively chunked.
    token_buffer.advance_chunk()

    # This should have reset the chunking.
    assert not token_buffer.actively_chunked
    np.testing.assert_array_equal(
        token_buffer.active, np.array([2, 3, 4, 5], dtype=np.int64)
    )
    assert token_buffer.processed_length == 2

    token_buffer.chunk(3)
    assert token_buffer.actively_chunked
    assert token_buffer.active_length == 3
    assert token_buffer.processed_length == 2

    # If we are actively chunked, we cannot add a token.
    with pytest.raises(ValueError):
        token_buffer.advance_with_token(6)

    # When there are less tokens than the chunk size, the chunking should be disabled.
    no_chunk_buffer = TokenBuffer(array=np.array([0, 1, 2, 3], dtype=np.int64))

    assert no_chunk_buffer.processed_length == 0

    # When there are less tokens than the chunk size, the chunking should fail
    with pytest.raises(ValueError):
        assert no_chunk_buffer.chunk(10)

    assert not no_chunk_buffer.actively_chunked
    np.testing.assert_array_equal(
        no_chunk_buffer.active, np.array([0, 1, 2, 3], dtype=np.int64)
    )

    # If we pass a negative value this should also fail.
    with pytest.raises(ValueError):
        no_chunk_buffer.chunk(-1)


def test_token_buffer__rewind_and_skip_processing() -> None:
    """Test skip_processing and rewind_processing window management."""
    token_buffer = TokenBuffer(
        array=np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    )

    assert token_buffer.processed_length == 0

    token_buffer.skip_processing(3)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([4, 5, 6, 7], dtype=np.int64)
    )
    assert token_buffer.active_length == 4
    assert token_buffer.processed_length == 3

    with pytest.raises(ValueError):
        token_buffer.skip_processing(token_buffer.active_length + 1)

    token_buffer.rewind_processing(2)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([2, 3, 4, 5, 6, 7], dtype=np.int64)
    )
    assert token_buffer.active_length == 6
    assert token_buffer.processed_length == 1

    with pytest.raises(ValueError):
        token_buffer.rewind_processing(10)

    with pytest.raises(ValueError):
        token_buffer.skip_processing(-1)

    with pytest.raises(ValueError):
        token_buffer.rewind_processing(-1)


def test_token_buffer__reset_as_new_prompt() -> None:
    """Ensure reset_as_new_prompt promotes generated tokens to prompt."""
    token_buffer = TokenBuffer(array=np.array([9, 10], dtype=np.int64))
    token_buffer.advance_with_token(11)
    token_buffer.advance_with_token(12)
    assert token_buffer.generated_length == 2
    assert token_buffer.processed_length == 3

    token_buffer.reset_as_new_prompt()

    assert token_buffer.generated_length == 0
    assert token_buffer.prompt_length == len(token_buffer)
    np.testing.assert_array_equal(token_buffer.prompt, token_buffer.all)
    np.testing.assert_array_equal(
        token_buffer.generated, np.array([], dtype=np.int64)
    )
    np.testing.assert_array_equal(token_buffer.active, token_buffer.all)
    assert token_buffer.active_length == len(token_buffer)
    assert token_buffer.processed_length == 0

    with pytest.raises(ValueError):
        _ = token_buffer.consume_recently_generated_tokens()


def test_token_buffer__reset_as_new_prompt_deletes_last_generated_token() -> (
    None
):
    """Bool trailing deletion drops the last generated token before reset."""
    token_buffer = TokenBuffer(array=np.array([9, 10], dtype=np.int64))
    token_buffer.advance_with_token(11)
    token_buffer.advance_with_token(12)

    token_buffer.reset_as_new_prompt(delete_last_generated_token=True)

    assert token_buffer.generated_length == 0
    np.testing.assert_array_equal(
        token_buffer.all, np.array([9, 10, 11], dtype=np.int64)
    )
    assert token_buffer.prompt_length == 3

    prompt_only = TokenBuffer(array=np.array([9, 10], dtype=np.int64))
    with pytest.raises(ValueError):
        prompt_only.reset_as_new_prompt(delete_last_generated_token=True)


def test_token_buffer__getitem_access() -> None:
    """Test TokenBuffer.__getitem__ for integer and slice access."""
    token_buffer = TokenBuffer(
        array=np.array([10, 20, 30, 40, 50], dtype=np.int64)
    )

    assert token_buffer.processed_length == 0

    # Test positive integer indexing
    assert token_buffer[0] == 10
    assert token_buffer[2] == 30
    assert token_buffer[4] == 50

    # Test negative integer indexing
    assert token_buffer[-1] == 50
    assert token_buffer[-2] == 40
    assert token_buffer[-5] == 10

    # Test slice access
    np.testing.assert_array_equal(
        token_buffer[1:3], np.array([20, 30], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        token_buffer[:2], np.array([10, 20], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        token_buffer[2:], np.array([30, 40, 50], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        token_buffer[::2], np.array([10, 30, 50], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        token_buffer[::-1], np.array([50, 40, 30, 20, 10], dtype=np.int64)
    )

    # Test out-of-bounds access raises IndexError
    with pytest.raises(IndexError):
        _ = token_buffer[5]
    with pytest.raises(IndexError):
        _ = token_buffer[-6]
    with pytest.raises(IndexError):
        _ = token_buffer[100]

    # Test invalid index type raises TypeError
    with pytest.raises(TypeError):
        _ = token_buffer["invalid"]  # type: ignore[index]
    with pytest.raises(TypeError):
        _ = token_buffer[1.5]  # type: ignore[index]

    # Test that __getitem__ respects current_length after adding tokens
    token_buffer.advance_with_token(60)
    assert token_buffer[5] == 60
    assert token_buffer[-1] == 60
    np.testing.assert_array_equal(
        token_buffer[:], np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    )
    assert token_buffer.processed_length == 5


def test_range() -> None:
    """Verify Range enforces invariants and updates correctly."""
    r = Range(0, 5)
    assert len(r) == 5

    r.bump_start(2)
    assert r.start == 2
    assert len(r) == 3

    r.bump_start(-1)
    assert r.start == 1

    r.bump_end(3)
    assert r.end == 8

    r.bump_end(-4)
    assert r.end == 4

    with pytest.raises(ValueError):
        r.bump_start(10)

    with pytest.raises(ValueError):
        r.bump_end(-10)

    with pytest.raises(ValueError):
        Range(5, 4)

    with pytest.raises(ValueError):
        Range(-1, 3)


def test_token_buffer__apply_processing_offset_behaviour() -> None:
    """Verify apply_processing_offset windowing, reset semantics, and validation."""
    token_buffer = TokenBuffer(array=np.array([1, 2, 3, 4, 5], dtype=np.int64))

    np.testing.assert_array_equal(
        token_buffer.active, np.array([1, 2, 3, 4, 5], dtype=np.int64)
    )
    assert token_buffer.active_length == 5
    assert token_buffer.active_length == len(token_buffer.active)

    token_buffer.apply_processing_offset(2)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([3, 4, 5], dtype=np.int64)
    )
    assert token_buffer.active_length == 3
    assert token_buffer.active_length == len(token_buffer.active)

    token_buffer.advance_with_token(6)
    assert token_buffer.processed_length == 5
    np.testing.assert_array_equal(
        token_buffer.active, np.array([6], dtype=np.int64)
    )
    assert token_buffer.active_length == 1
    assert token_buffer.active_length == len(token_buffer.active)

    token_buffer.apply_processing_offset(-2)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([4, 5, 6], dtype=np.int64)
    )
    assert token_buffer.active_length == 3
    assert token_buffer.active_length == len(token_buffer.active)

    with pytest.raises(ValueError):
        token_buffer.apply_processing_offset(-10)

    with pytest.raises(ValueError):
        token_buffer.apply_processing_offset(token_buffer.active_length + 1)


def test_token_buffer__active_length_matches_active_with_offset() -> None:
    """Verify active_length always matches len(active) when processing offset is applied.

    This is critical for speculative decoding (EAGLE) which uses negative offsets
    to include additional tokens in the active window.
    """
    token_buffer = TokenBuffer(array=np.array([1, 2, 3, 4, 5], dtype=np.int64))

    # Skip some tokens to create a smaller active window
    token_buffer.skip_processing(3)
    assert token_buffer.active_length == 2
    assert token_buffer.active_length == len(token_buffer.active)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([4, 5], dtype=np.int64)
    )

    # Apply a negative offset to expand the active window backwards
    # This is the pattern used in EAGLE speculative decoding
    token_buffer.apply_processing_offset(-1)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([3, 4, 5], dtype=np.int64)
    )
    # active_length must match the actual length of active
    assert token_buffer.active_length == len(token_buffer.active)
    assert token_buffer.active_length == 3

    # Apply a positive offset to shrink the active window
    token_buffer.apply_processing_offset(1)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([5], dtype=np.int64)
    )
    assert token_buffer.active_length == len(token_buffer.active)
    assert token_buffer.active_length == 1

    # Reset offset to 0
    token_buffer.apply_processing_offset(0)
    np.testing.assert_array_equal(
        token_buffer.active, np.array([4, 5], dtype=np.int64)
    )
    assert token_buffer.active_length == len(token_buffer.active)
    assert token_buffer.active_length == 2


def test_expand_capacity_with_min_capacity() -> None:
    tokens = np.arange(4, dtype=np.int64)
    buf = TokenBuffer(array=tokens)
    assert buf.array.size == 4
    assert len(buf) == 4

    buf._expand_capacity(min_capacity=3)
    assert buf.array.size == 4

    buf._expand_capacity(min_capacity=5)
    assert buf.array.size >= 5
    np.testing.assert_array_equal(buf.array[:4], np.arange(4, dtype=np.int64))

    buf._current_length = 5
    assert len(buf) == 5

    buf._expand_capacity(min_capacity=100)
    assert buf.array.size >= 100
    np.testing.assert_array_equal(buf.array[:4], np.arange(4, dtype=np.int64))
