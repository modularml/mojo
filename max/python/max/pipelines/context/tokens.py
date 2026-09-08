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
"""Token management interfaces for MAX pipelines.

This module provides high-performance data structures for managing token sequences
during text generation. The primary components are:

- TokenSlice: Type alias for numpy arrays containing token IDs.
- TokenBuffer: Dynamic, resizable container for token sequences.

The TokenBuffer class provides efficient management of tokens for language model
inference, including prompt tokens and generated output tokens.

Example:

.. code-block:: python

    import numpy as np
    from max.pipelines.context.tokens import TokenBuffer

    # Create a token buffer with initial prompt tokens
    prompt_tokens = np.array([1, 2, 3, 4], dtype=np.int64)
    token_buffer = TokenBuffer(prompt_tokens)

    # Add generated tokens
    token_buffer.advance_with_token(5)
    token_buffer.advance_with_token(6)

    # Access token sequences
    all_tokens = token_buffer.all  # [1, 2, 3, 4, 5, 6]
    generated = token_buffer.generated  # [5, 6]
    prompt = token_buffer.prompt  # [1, 2, 3, 4]

.. invisible-code-block: python

    assert list(all_tokens) == [1, 2, 3, 4, 5, 6]
    assert list(generated) == [5, 6]
    assert list(prompt) == [1, 2, 3, 4]
"""

from __future__ import annotations

import dataclasses
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "ImageMetadata",
    "Range",
    "TokenBuffer",
    "TokenHashOverride",
    "TokenSlice",
]


@dataclasses.dataclass(frozen=False, slots=True)
class Range:
    """Represents a range with start and end indices.

    Args:
        start: The inclusive start index.
        end: The exclusive end index.
    """

    start: int
    end: int

    def __init__(self, start: int, end: int) -> None:
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        if start < 0:
            raise ValueError(f"start ({start}) must be non-negative")

        if end < 0:
            raise ValueError(f"end ({end}) must be non-negative")

        self.start = start
        self.end = end

    def __post_init__(self) -> None:
        """Validate that start <= end."""
        if self.start > self.end:
            raise ValueError(
                f"start ({self.start}) must be <= end ({self.end})"
            )

    def __len__(self) -> int:
        """Returns the number of elements in the range (``end - start``).

        Returns:
            The length of the range, which is always >= 0.
        """
        return max(0, self.end - self.start)

    def bump_start(self, amount: int) -> None:
        """Bump the start index by the given amount.

        Args:
            amount: The amount to bump the start index by.

        Raises:
            ValueError: If the new start index would exceed the end index.
        """
        new_start = self.start + amount
        if new_start < 0:
            raise ValueError(
                f"start ({self.start}) + amount ({amount}) must be >= 0"
            )

        if new_start > self.end:
            raise ValueError(
                f"start ({self.start}) + amount ({amount}) must be <= end ({self.end})"
            )

        self.start = new_start

    def bump_end(self, amount: int) -> None:
        """Bump the end index by the given amount.

        Args:
            amount: The amount to bump the end index by.

        """
        new_end = self.end + amount
        if new_end < 0:
            raise ValueError(
                f"end ({self.end}) + amount ({amount}) must be >= 0"
            )

        if new_end < self.start:
            raise ValueError(
                f"end ({self.end}) + amount ({amount}) must be >= start ({self.start})"
            )

        self.end = new_end

    def advance(self) -> None:
        """Advance the range to the next token."""
        self.start = self.end


TokenSlice: TypeAlias = npt.NDArray[np.int64]
"""Type alias for arrays containing token IDs as 64-bit integers.

A TokenSlice represents a contiguous sequence of token IDs, typically used
for representing portions of text that have been tokenized. This is the
fundamental data type for token-based operations in MAX pipelines.
"""


@dataclasses.dataclass(frozen=False, slots=True)
class TokenBuffer:
    """A dynamically resizable container for managing token sequences.

    :class:`TokenBuffer` provides efficient storage and access to token sequences
    during text generation. It maintains the prompt tokens (initial input) and
    generated tokens (model output) separately, while handling automatic memory
    management as new tokens are added.

    :class:`TokenBuffer` organizes tokens across three related views:

    1. The full stored sequence (``all``), split into ``prompt`` and ``generated`` tokens.
    2. The processing window (``active`` versus processed and pending tokens).
    3. The streaming window over newly generated tokens consumed by callers.

    The first diagram shows how prompt and generated tokens share a single
    backing array. Later diagrams explain how processing and streaming walk
    over that array during generation::

        +-------------------- all --------------------+
        +-----------------+---------------------------+
        |     prompt      |        generated          |
        +-----------------+---------------------------+
        0   prompt_length ^          generated_length ^
        0                                   len(self) ^

    This includes three attributes for accessing tokens:

    - ``all``: The slice of the array containing all valid tokens.
    - ``prompt``: The slice of the array containing the prompt tokens.
    - ``generated``: The slice of the array containing the generated tokens.

    Along with three attributes for tracking their lengths:
    - `prompt_length`: The number of tokens in the prompt.
    - `generated_length`: The number of tokens in the generated tokens.
    - `len(self)`: The total number of valid tokens in the buffer.

    Processing window (what the model will process next)::

        +-------------------------------- all  -------------------------+
        +-------------------+---------------------------+---------------+
        |     processed     |          active           |    pending    |
        +-------------------+---------------------------+---------------+
        0  processed_length ^             active_length ^
        0                              current_position ^
        0                                                     len(self) ^

    In the above, ``processed`` tracks tokens which has already been processed,
    ``active`` tracks tokens, which are scheduled to be processed in the next batch,
    and ``pending`` tracks tokens, which have not yet been processed, but are not
    actively scheduled to be processed in the next batch (this commonly
    occurs during chunked prefill).

    This includes one attribute for accessing tokens:

    - ``active``: The slice of the array containing the tokens scheduled
      for processing in the next batch.

    Along with three additional attributes for tracking their lengths:

    - `processed_length`: The number of tokens that have already been processed.
    - `active_length`: The number of tokens that is currently scheduled for
      processing in the next batch.
    - `current_position`: The global index marking the end of the current
      active processing window.

    This processing view is updated by methods such as ``rewind_processing``,
    ``skip_processing``, ``chunk``, and ``advance_chunk``/``advance_with_token``,
    which control how much of the existing sequence is reprocessed or advanced
    at each step.

    It also maintains a completion window over the generated tokens
    for completion streaming::

        +------------- generated -------------+
        +------------+------------------------+
        |  streamed  |  ready to stream next  |
        +------------+------------------------+
        |     (1)    |          (2)           |

    Generated tokens are conceptually split into:

    1. **streamed**: tokens that have already been returned to the caller.
    2. **read to stream**: the newest generated tokens that have not yet
       been returned.

    Each call to ``consume_recently_generated_tokens()`` returns the (2) region
    and advances the boundary between (1) and (2), so subsequent calls only
    see newly generated tokens.

    Together, these three views let :class:`TokenBuffer` support efficient prompt
    handling, chunked processing, and incremental streaming while exposing a small,
    consistent public API.
    """

    array: TokenSlice
    """In-place storage holding the prompt plus any generated tokens."""

    # Tracks the subset of tokens scheduled for the next round of processing.
    _processing_range: Range = dataclasses.field(
        default_factory=lambda: Range(0, 0), init=False
    )

    # Tracks the offset of the processing range.
    _processing_offset: int = dataclasses.field(default=0, init=False)

    # Marks which generated tokens have already been handed back to the caller.
    _completion_range: Range = dataclasses.field(
        default_factory=lambda: Range(0, 0), init=False
    )

    # Cached length of the prompt tokens.
    _prompt_length: int = dataclasses.field(default=0, init=False)

    # Number of valid tokens currently stored in `array`.
    _current_length: int = dataclasses.field(default=0, init=False)

    # Indicates whether chunking constraints are currently applied to the buffer.
    _actively_chunked: bool = dataclasses.field(default=False, init=False)

    # ============================================================================
    # Initialization and Setup
    # ============================================================================

    def __init__(self, array: TokenSlice) -> None:
        """Initialize a TokenBuffer with the given token array.

        Args:
            array: A 1D numpy array of int64 token IDs. Must be non-empty.

        Raises:
            ValueError: If the array is not 1-dimensional, not int64 dtype, or empty.
        """
        # Validate and set initial array
        if array.ndim != 1:
            raise ValueError(
                f"array must be one-dimensional: got shape {array.shape}"
            )

        if array.dtype != np.int64:
            raise ValueError(f"array must be int64: got dtype {array.dtype}")

        if len(array) == 0:
            raise ValueError(
                f"array must be non-empty: got length {len(array)}"
            )

        self.array = array

        # Initialize ranges
        self._processing_range = Range(0, len(array))
        self._processing_offset = 0
        self._completion_range = Range(len(array), len(array))

        # Set Initial Values
        self._prompt_length = len(array)
        self._current_length = len(array)

        self._actively_chunked = False

    def __post_init__(self) -> None:
        # This is called in post_init to ensure that the token array is always
        # writable even if this token buffer was created via deserialization.
        self._ensure_writeable()

    def _ensure_writeable(self) -> None:
        """Ensure that the underlying token array is writeable.

        This makes a private copy of the array if the current array is not writeable,
        allowing in-place manipulation of tokens for prompt or generation operations.
        """
        if not self.array.flags.writeable:
            self.array = self.array.copy()

    # ============================================================================
    # Token Access Properties
    # ============================================================================

    def __getitem__(self, index: int | slice) -> TokenSlice:
        """Retrieve token(s) from the buffer at the specified index or slice.

        Args:
            index: An integer or slice specifying the token(s) to access.

        Returns:
            The token (if index is int) or a slice of tokens (if index is slice).

        Raises:
            IndexError: If the integer index is out of bounds.
            TypeError: If the index is not an int or slice.
        """
        # If the index is an integer, handle single token lookup.
        if isinstance(index, int):
            key = index  # Use 'key' to follow original code logic.
            # Handle negative index by converting to positive (Python-style negative indexing)
            if key < 0:
                key = self._current_length + key
            # Check bounds - only allow access within current valid length
            if key < 0 or key >= self._current_length:
                raise IndexError(
                    f"Index {key} is out of bounds for array of length {self._current_length}"
                )
            # Return the token at the specified position
            return self.array[key]
        # If the index is a slice, allow for convenient sub-sequence extraction.
        elif isinstance(index, slice):
            # Apply slice to the valid portion of the buffer
            # This correctly handles all slice operations including negative steps
            return self.all[index]
        else:
            # Disallow attempts to index with objects other than int or slice
            raise TypeError(
                f"Index must be an integer or slice, got {type(index)}"
            )

    @property
    def all(self) -> TokenSlice:
        """Return every valid token currently stored (prompt + generated).

        Use this when downstream components need the full sequence for scoring,
        logging, or serialization.
        """
        return self.array[: self._current_length]

    @property
    def prompt(self) -> TokenSlice:
        """Return only the original prompt tokens.

        Helpful for echo suppression, prompt-side metrics, or offset
        calculations that should exclude generated output.
        """
        return self.array[: self._prompt_length]

    @property
    def generated(self) -> TokenSlice:
        """Return all tokens produced after the prompt.

        Use this slice for stop checks, repetition penalties, or any logic that
        should consider only newly generated content.
        """
        return self.array[self._prompt_length : self._current_length]

    @property
    def active(self) -> TokenSlice:
        """Return the tokens queued for the next processing step."""
        return self.array[
            self._processing_range.start
            + self._processing_offset : self._processing_range.end
        ]

    # ============================================================================
    # Length Properties
    # ============================================================================

    def __len__(self) -> int:
        """Return the total number of valid tokens in the buffer.

        This includes both prompt and generated tokens currently held in the buffer.

        Returns:
            The current total number of tokens.
        """
        return len(self.all)

    @property
    def prompt_length(self) -> int:
        """Number of tokens that belong to the prompt."""
        return self._prompt_length

    @property
    def processed_length(self) -> int:
        """Number of tokens that have already been processed."""
        return self._processing_range.start

    @property
    def generated_length(self) -> int:
        """Number of tokens generated after the prompt."""
        return self._current_length - self._prompt_length

    @property
    def active_length(self) -> int:
        """Count of tokens currently scheduled for processing."""
        return len(self._processing_range) - self._processing_offset

    @property
    def current_position(self) -> int:
        """Global index marking the end of the current active processing window.

        Equal to processed_length + active_length; represents the index of
        the next token to be processed, which may be less than the total length
        when processing is limited by chunking.
        """
        return self.processed_length + self.active_length

    # ============================================================================
    # Processing and Chunking
    # ============================================================================

    def apply_processing_offset(self, value: int) -> None:
        """Set the processing offset.

        Args:
            value: The new processing offset.
        """
        new_start = self._processing_range.start + value
        if new_start < 0:
            raise ValueError(
                f"processing range start ({self._processing_range.start}) + processing offset ({value}) must be >= 0"
            )

        if new_start > self._processing_range.end:
            raise ValueError(
                f"processing range start ({self._processing_range.start}) + processing offset ({value}) must be <= processing range end ({self._processing_range.end})"
            )

        self._processing_offset = value

    def rewind_processing(self, n: int) -> None:
        """Re-expose n earlier tokens so they can be processed again.

        Args:
            n: Number of tokens to move back into the active window.

        Raises:
            ValueError: If n is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative: got {n}")

        self._processing_range.bump_start(-n)

    def skip_processing(self, n: int) -> None:
        """Advance the active window start by n tokens.

        Args:
            n: Number of tokens to drop from the active window.

        Raises:
            ValueError: If n exceeds the number of available tokens to process,
                or if skipping n tokens would leave 0 active tokens.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative: got {n}")

        if n >= self.active_length:
            raise ValueError(
                f"Cannot skip {n} tokens: would leave {self.active_length - n} active tokens. "
                f"Must have at least 1 active token (current active_length={self.active_length})"
            )

        self._processing_range.bump_start(n)

    # ============================================================================
    # Chunking
    # ============================================================================

    def chunk(self, chunk_size: int) -> None:
        """Limit the upcoming processing step to at most n tokens.

        Args:
            chunk_size: Maximum number of tokens to process.

        Raises:
            ValueError: If chunk_size is not between 1 and the current number of active tokens.
        """
        if chunk_size <= 0 or chunk_size >= self.active_length:
            raise ValueError(
                f"chunk size must be between 1 and the current number of active tokens: got {chunk_size}"
            )

        self._processing_range.bump_end(chunk_size - self.active_length)
        self._actively_chunked = True

    @property
    def actively_chunked(self) -> bool:
        """Check if the buffer has active chunk limits applied.

        Returns:
            True if chunk limits are active, False otherwise.
        """
        return self._actively_chunked

    def advance_chunk(self) -> None:
        """Move to the next set of tokens after a limited chunk.

        Call this after `maybe_chunk` when you have finished working with the
        current `active` tokens and want the remaining tokens in the sequence
        to become active.

        Raises:
            ValueError: If called before `maybe_chunk` has limited the active
                tokens (that is, when no chunk is currently active).
        """
        # This error occurs if no chunk was set up with `maybe_chunk` first.
        if not self.actively_chunked:
            raise ValueError("Cannot advance chunk if not actively chunked.")

        # We should reset the processing offset to 0 when we advance the chunk
        if self._processing_offset != 0:
            self.apply_processing_offset(0)

        self._processing_range.advance()
        self._processing_range.end = self._current_length
        self._actively_chunked = False

    # ============================================================================
    # Token Addition and Management
    # ============================================================================

    def advance_with_token(self, token: int) -> None:
        """Add a new token to the buffer.

        Args:
            token: The token ID to add.
        """
        if self.actively_chunked:
            raise ValueError("Cannot add a token while actively chunked.")

        # Update processing ranges
        self._processing_range.advance()
        self._processing_range.bump_end(1)

        # Update completion ranges
        self._completion_range.bump_end(1)

        # Expand capacity if needed and add the token
        self._expand_capacity()
        self.array[self._current_length] = token

        # Update current length
        self._current_length += 1

        # Reset the processing offset to 0
        if self._processing_offset != 0:
            self.apply_processing_offset(0)

    def overwrite_last_token(self, token: int) -> None:
        """Overwrite the last token in the buffer."""
        self.array[self._current_length - 1] = token

    # ============================================================================
    # Completion Tracking
    # ============================================================================

    @property
    def has_outstanding_generated_tokens(self) -> bool:
        """Indicates whether there are generated tokens that have not yet been consumed.

        Returns:
            ``True`` if there are outstanding generated tokens to be streamed or processed, ``False`` otherwise.
        """
        return len(self._completion_range) > 0

    def consume_recently_generated_tokens(self) -> TokenSlice:
        """Return newly generated tokens since the last consumption.

        Returns:
            A slice containing tokens ready to stream to the caller.

        Raises:
            ValueError: If no new tokens are available.
        """
        if self._completion_range.start == self._completion_range.end:
            raise ValueError("No tokens have been generated yet.")

        # Validate that we're not consuming beyond current length
        if self._completion_range.end > self._current_length:
            raise ValueError(
                f"Internal error: completion range end ({self._completion_range.end}) "
                f"exceeds current length ({self._current_length})"
            )

        generated_tokens = self.array[
            self._completion_range.start : self._completion_range.end
        ]

        # Use bump_start to maintain validation instead of direct assignment
        amount = self._completion_range.end - self._completion_range.start
        self._completion_range.bump_start(amount)
        return generated_tokens

    # ============================================================================
    # State Management and Reset
    # ============================================================================

    def reset_as_new_prompt(
        self, delete_last_generated_token: bool = False
    ) -> None:
        """Treat the current sequence as a fresh prompt.

        Marks all existing tokens as prompt tokens so the next generation pass
        starts from this state.

        Args:
            delete_last_generated_token: If True, deletes the last generated
                token before resetting the buffer. This is useful when the
                last token is a placeholder future token.

        Raises:
            ValueError: If the buffer state is invalid.
        """
        # Validate current state before resetting
        if self._processing_range.end > self._current_length:
            raise ValueError(
                f"Internal error: processing range end ({self._processing_range.end}) "
                f"exceeds current length ({self._current_length})"
            )

        if delete_last_generated_token:
            if not self.generated_length:
                raise ValueError(
                    "Cannot delete the last generated token if there are no "
                    "generated tokens."
                )
            self._current_length -= 1

        # Reset ranges and make all current tokens the new prompt
        self._processing_range = Range(0, self._current_length)
        self._completion_range = Range(
            self._current_length, self._current_length
        )
        self._prompt_length = self._current_length
        self._actively_chunked = False

    # ============================================================================
    # Internal Memory Management
    # ============================================================================

    def _expand_capacity(self, min_capacity: int | None = None) -> None:
        """Expand the underlying array capacity when needed.

        Automatically called by add_token() when more space is required.
        Uses exponential growth for efficient amortized performance.

        Args:
            min_capacity: If provided, ensure the array can hold at least
                this many elements. Otherwise uses ``_current_length``.
        """
        target = (
            min_capacity if min_capacity is not None else self._current_length
        )
        if target >= self.array.size:
            new_size = self.array.size
            while new_size <= target:
                new_size *= 2
            new_array = np.empty(new_size, dtype=np.int64)
            np.copyto(
                new_array[: self._current_length],
                self.array[: self._current_length],
            )
            self.array = new_array

    # ============================================================================
    # Debugging and Introspection
    # ============================================================================

    def __repr__(self) -> str:
        """Return a concise debug representation of the buffer state.

        The representation summarizes:

        - Whether the buffer is currently actively chunked.
        - How many tokens are active for processing.
        - How many tokens belong to the prompt.
        - How many tokens have been generated.
        - How many tokens exist in total.

        A small preview of the underlying token array is included; for large
        buffers, only the beginning and end of the sequence are shown with an
        ellipsis to avoid overwhelming logs.
        """
        prompt_length = self.prompt_length
        generated_length = self.generated_length
        total_length = len(self)
        active_length = len(self._processing_range)

        # Build a compact preview of the token sequence.
        tokens = self.all
        max_preview = 16
        if total_length <= max_preview:
            preview_values = tokens.tolist()
        else:
            head_count = max_preview // 2
            tail_count = max_preview - head_count
            head = tokens[:head_count].tolist()
            tail = tokens[-tail_count:].tolist()
            preview_values = head + ["..."] + tail

        tokens_preview = "[" + ", ".join(str(v) for v in preview_values) + "]"

        return (
            f"{self.__class__.__name__}("
            f"  actively_chunked={self._actively_chunked}, "
            f"  active_tokens={active_length}, "
            f"  prompt_tokens={prompt_length}, "
            f"  generated_tokens={generated_length}, "
            f"  total_tokens={total_length}, "
            f"  tokens_preview={tokens_preview})"
        )


@dataclasses.dataclass(kw_only=True)
class ImageMetadata:
    """Metadata about an image in the prompt.

    Each image corresponds to a range in the text token array [start_idx, end_idx).
    """

    start_idx: int
    """Index of the first <vision_token_id> special token for the image"""

    end_idx: int
    """One after the index of the last <vision_token_id> special token for the image"""

    pixel_values: npt.NDArray[Any]
    """Pixel values for the image.

    Can be various dtypes depending on the vision model:

    - float32: Original precision
    - uint16: BFloat16 bits stored as uint16 (workaround for NumPy's lack of
      native bfloat16 support). Reinterpreted as bfloat16 on GPU.
    """

    image_hash: int | None = None
    """Hash of the image, for use in prefix caching"""

    num_embedding_rows: int | None = None
    """Embedding rows the encoder emits for this entry.

    ``None`` means every token in ``[start_idx, end_idx)`` is a placeholder,
    so the row count equals the span width. Set by tokenizers whose spans
    interleave placeholder runs with other tokens (e.g. video timestamp
    text)."""

    @property
    def embedding_rows(self) -> int:
        """Embedding rows for this entry (span width unless overridden)."""
        if self.num_embedding_rows is not None:
            return self.num_embedding_rows
        return self.end_idx - self.start_idx

    def __post_init__(self) -> None:
        if self.start_idx < 0:
            raise ValueError("Images must have a valid start index")
        if self.end_idx <= self.start_idx:
            raise ValueError(
                "Images must have a valid start and end index containing at least one <vision_token_id>"
            )

    def __repr__(self):
        return f"ImageMetadata(start_idx={self.start_idx}, end_idx={self.end_idx}, pixel_values={self.pixel_values.shape})"


@dataclasses.dataclass(kw_only=True)
class TokenHashOverride:
    """Content hash to use in place of a token when hashing KV-cache blocks.

    The token stream itself is unchanged. The override is applied only while
    computing block hashes, so content that is represented by placeholder tokens
    can participate in prefix-cache keys.
    """

    token_idx: int
    """Index of the token to replace while hashing."""

    token_hash: int
    """Hash value to use at ``token_idx`` while hashing."""

    source: str = "media"
    """Human-readable label describing where the hash override came from (e.g. "image", "video")."""

    def __post_init__(self) -> None:
        if self.token_idx < 0:
            raise ValueError(
                "Token hash overrides must have a valid token index"
            )
