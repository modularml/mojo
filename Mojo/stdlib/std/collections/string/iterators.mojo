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
"""Implements the iterators and related utilities for efficient string
operations."""

from std.collections.string._utf8 import (
    _utf8_first_byte_sequence_length,
    _is_utf8_continuation_byte,
)
from std.collections.string._grapheme_break import (
    _GraphemeBreakState,
    _find_safe_grapheme_start,
    _is_grapheme_break,
    _is_safe_ascii_for_grapheme,
    _reset_grapheme_state_to_other,
    GBP_PREPEND,
)
from std.collections.span import _SpanIter


struct CodepointSliceIter[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator, Sized):
    """Iterator for `StringSlice` over substring slices containing a single
    Unicode codepoint.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.
        forward: The iteration direction. `False` is backwards.

    The `forward` parameter only controls the behavior of the `__next__()`
    method used for normal iteration. Calls to `next()` will always take an
    element from the front of the iterator, and calls to `next_back()` will
    always take an element from the end.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this codepoint iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = StringSlice[Self.origin]
    """The element type yielded by iteration."""

    var _slice: StringSlice[Self.origin]

    # Note:
    #   Marked private since `StringSlice.codepoints()` is the intended public
    #   way to construct this type.
    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        self._slice = str_slice

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the `StringSlice` yielding individual characters.

        Returns:
            An iterator over the characters in the string slice.
        """
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> StringSlice[Self.origin]:
        """Get the next codepoint in the underlying string slice.

        This returns the next single-codepoint substring slice encoded in the
        underlying string, and advances the iterator state.

        If `forward` is set to `False`, this will return the next codepoint
        from the end of the string.

        This function will abort if this iterator has been exhausted.

        Returns:
            The next character in the string.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """

        # NOTE: This intentionally check if the length *in bytes* is greater
        # than zero, because checking the codepoint length requires a linear
        # scan of the string, which is needlessly expensive for this purpose.
        if self._slice.byte_length() <= 0:
            raise StopIteration()

        comptime if Self.forward:
            return self.next().value()
        else:
            return self.next_back().value()

    @always_inline
    def __len__(self) -> Int:
        """Returns the remaining length of this iterator in `Codepoint`s.

        The value returned from this method indicates the number of subsequent
        calls to `next()` that will return a value.

        Returns:
            Number of codepoints remaining in this iterator.
        """
        return self._slice.count_codepoints()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def peek_next(self) -> Optional[StringSlice[Self.origin]]:
        """Check what the next single-codepoint slice in this iterator is,
        without advancing the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The next codepoint slice in the underlying string, or None if the
            string is empty.

        **Examples:**

        `peek_next()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoint_slices()

        assert_equal(iter.peek_next().value(), "1")
        assert_equal(iter.peek_next().value(), "1")
        assert_equal(iter.peek_next().value(), "1")

        # A call to `next()` return the same value as `peek_next()` had,
        # but also advance the iterator.
        assert_equal(iter.next().value(), "1")

        # Later `peek_next()` calls will return the _new_ next character:
        assert_equal(iter.peek_next().value(), "2")
        ```
        """
        if self._slice.byte_length() > 0:
            # SAFETY: Will not read out of bounds because `_slice` is guaranteed
            #   to contain valid UTF-8.
            var curr_ptr = self._slice.as_bytes().unsafe_ptr()
            var byte_len = _utf8_first_byte_sequence_length(curr_ptr[])
            return StringSlice[Self.origin](
                unsafe_from_utf8=Span[Byte, Self.origin](
                    unsafe_ptr=curr_ptr, length=byte_len
                )
            )
        else:
            return None

    def peek_back(mut self) -> Optional[StringSlice[Self.origin]]:
        """Check what the last single-codepoint slice in this iterator is,
        without advancing the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The last codepoint slice in the underlying string, or None if the
            string is empty.

        **Examples:**

        `peek_back()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoint_slices()

        # Repeated calls to `peek_back()` return the same value.
        assert_equal(iter.peek_back().value(), "3")
        assert_equal(iter.peek_back().value(), "3")
        assert_equal(iter.peek_back().value(), "3")

        # A call to `next_back()` return the same value as `peek_back()` had,
        # but also advance the iterator.
        assert_equal(iter.next_back().value(), "3")

        # Later `peek_back()` calls will return the _new_ next character:
        assert_equal(iter.peek_back().value(), "2")
        ```
        """
        if self._slice.byte_length() > 0:
            var byte_len = 1
            var back_ptr = (
                self._slice.as_bytes()
                .unsafe_ptr()
                .unsafe_offset(self._slice.byte_length() - 1)
            )
            # SAFETY:
            #   Guaranteed not to go out of bounds because UTF-8
            #   guarantees there is always a "start" byte eventually before any
            #   continuation bytes.
            while _is_utf8_continuation_byte(back_ptr[]):
                byte_len += 1
                back_ptr = back_ptr.unsafe_offset(-1)

            return StringSlice[Self.origin](
                unsafe_from_utf8=Span[Byte, Self.origin](
                    unsafe_ptr=back_ptr, length=byte_len
                )
            )
        else:
            return None

    def next(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the next codepoint slice in the underlying string slice, or None
        if the iterator is empty.

        This returns the next single-codepoint substring encoded in the
        underlying string, and advances the iterator state.

        Returns:
            A character if the string is not empty, otherwise None.
        """
        var result: Optional[StringSlice[Self.origin]] = self.peek_next()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var slice_len = result.unsafe_value().byte_length()
            # Advance the pointer in _slice.
            self._slice._slice._data = self._slice._slice._data.unsafe_offset(
                slice_len
            )
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= slice_len

        return result

    def next_back(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the last single-codepoint slice in this iterator is, or None
        if the iterator is empty.

        This returns the last codepoint slice in this iterator, and advances
        the iterator state.

        Returns:
            The last codepoint slice in the underlying string, or None if the
            string is empty.
        """
        var result: Optional[StringSlice[Self.origin]] = self.peek_back()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var slice_len = result.unsafe_value().byte_length()
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= slice_len

        return result


struct CodepointsIter[mut: Bool, //, origin: Origin[mut=mut]](
    ImplicitlyCopyable, Iterable, Iterator, Sized
):
    """Iterator over the `Codepoint`s in a string slice, constructed by
    `StringSlice.codepoints()`.

    Parameters:
        mut: Mutability of the underlying string data.
        origin: Origin of the underlying string data.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this codepoint iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = Codepoint
    """The element type yielded by iteration."""

    var _slice: StringSlice[Self.origin]
    """String slice containing the bytes that have not been read yet.

    When this iterator advances, the pointer in `_slice` is advanced by the
    byte length of each read character, and the slice length is decremented by
    the same amount.
    """

    # Note:
    #   Marked private since `StringSlice.codepoints()` is the intended public
    #   way to construct this type.
    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        self._slice = str_slice

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(mut self) raises StopIteration -> Codepoint:
        """Get the next codepoint in the underlying string slice.

        This returns the next `Codepoint` encoded in the underlying string, and
        advances the iterator state.

        Returns:
            The next character in the string.

        Raises:
            StopIteration: If the iterator is exhausted.
        """
        if self._slice.byte_length() <= 0:
            raise StopIteration()
        return self.next().value()

    @always_inline
    def __len__(self) -> Int:
        """Returns the remaining length of this iterator in `Codepoint`s.

        The value returned from this method indicates the number of subsequent
        calls to `next()` that will return a value.

        Returns:
            Number of codepoints remaining in this iterator.
        """
        return self._slice.count_codepoints()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def peek_next(self) -> Optional[Codepoint]:
        """Check what the next codepoint in this iterator is, without advancing
        the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The next character in the underlying string, or None if the string
            is empty.

        # Examples

        `peek_next()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoints()

        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))
        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))
        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))

        # A call to `next()` return the same value as `peek_next()` had,
        # but also advance the iterator.
        assert_equal(iter.next().value(), Codepoint.ord("1"))

        # Later `peek_next()` calls will return the _new_ next character:
        assert_equal(iter.peek_next().value(), Codepoint.ord("2"))
        ```
        """
        if self._slice.byte_length() > 0:
            # SAFETY: Will not read out of bounds because `_slice` is guaranteed
            #   to contain valid UTF-8.
            var codepoint, _ = Codepoint.unsafe_decode_utf8_codepoint(
                self._slice._slice
            )
            return codepoint
        else:
            return None

    def next(mut self) -> Optional[Codepoint]:
        """Get the next codepoint in the underlying string slice, or None if
        the iterator is empty.

        This returns the next `Codepoint` encoded in the underlying string, and
        advances the iterator state.

        Returns:
            A character if the string is not empty, otherwise None.
        """
        var result: Optional[Codepoint] = self.peek_next()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var char_len = result.unsafe_value().utf8_byte_length()
            # Advance the pointer in _slice.
            self._slice._slice._data = self._slice._slice._data.unsafe_offset(
                char_len
            )
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= char_len

        return result


struct GraphemeSliceIter[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator, Sized):
    """Iterator over grapheme clusters in a string, yielding each cluster as a
    `StringSlice`.

    A grapheme cluster is what a user would typically think of as a single
    "character" on screen. This includes combining character sequences, emoji
    with modifiers, flag sequences, and other multi-codepoint grapheme clusters
    as defined by UAX #29.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.
        forward: The iteration direction. `False` is backwards.

    The `forward` parameter only controls the behavior of the `__next__()`
    method used for normal iteration. Calls to `next()` will always take an
    element from the front of the iterator, and calls to `next_back()` will
    always take an element from the end. Mixing `next()` and `next_back()`
    on the same iterator is supported: they share the remaining byte range
    but use independent state (forward iteration keeps incremental UAX #29
    state; reverse iteration caches a safe restart boundary). This is safe
    because forward priming only consults the codepoint at the start of the
    remaining range, and `next_back()` shrinks the range from the end without
    moving that start. A forward `next()` advances the front and invalidates
    the cached reverse safe-start so the next reverse call recomputes it.

    Note: `len()` is an O(n) operation that must scan all remaining bytes
    to count grapheme boundaries. Avoid calling it in a loop; prefer
    iterating with `for g in s.graphemes()` or calling `next()` until
    `None`.

    Note: Reverse iteration costs more per element than forward iteration.
    The UAX #29 state machine is forward-scanning, so `next_back()` backs
    up to a guaranteed grapheme boundary (typically a line break or the
    start of the string) and forward-scans from there. The safe boundary
    is cached across reverse calls (a forward `next()` invalidates the
    cache), so per-call cost is dominated by forward-scan length: small
    in text with frequent Control/CR/LF codepoints, growing with the
    distance back to such a codepoint in long runs without them.

    # TODO: Vectorize the existing scalar safe-ASCII fast path. Runs of
    # safe-ASCII bytes (U+0020..U+007E) are already skipped one-by-one
    # without entering the state machine; a SIMD check (e.g. `>= 0x20 &
    # <= 0x7E`) could extend a run by a whole vector width per iteration.

    Example:

    ```mojo
    from std.testing import assert_equal
    var text = String("cafe\\u{0301}")  # "café" with combining accent
    var count = 0
    for grapheme in text.graphemes():
        count += 1
    # count == 4: c, a, f, e + combining acute (2 codepoints, 1 grapheme)
    assert_equal(count, 4)
    ```
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = StringSlice[Self.origin]
    """The element type yielded by iteration."""

    var _slice: StringSlice[Self.origin]
    """Remaining bytes to iterate over."""

    var _state: _GraphemeBreakState
    """UAX #29 segmentation state for forward iteration."""

    var _state_primed: Bool
    """True if the state machine already processed the first codepoint of
    the next grapheme (from the previous break detection)."""

    var _back_safe_known: Bool
    """True if `_back_safe_start` holds a usable safe boundary for reverse
    iteration. Cleared whenever forward `next()` advances the front of the
    range; preserved across `next_back()` calls because shrinking from the
    end does not move the data pointer."""

    var _back_safe_start: Int
    """Cached byte offset in `_slice` at which a fresh forward UAX #29 scan
    can restart and reproduce the same boundaries it would from offset 0.
    Only meaningful when `_back_safe_known` is True."""

    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        """Construct from a string slice.

        Args:
            str_slice: The string slice to iterate over.
        """
        self._slice = str_slice
        self._state = _GraphemeBreakState()
        self._state_primed = False
        self._back_safe_known = False
        self._back_safe_start = 0

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Return an iterator over grapheme clusters.

        Returns:
            A copy of this iterator.
        """
        return self.copy()

    @always_inline
    def __next__(mut self) raises StopIteration -> StringSlice[Self.origin]:
        """Get the next grapheme cluster.

        If `forward` is set to `False`, this will return the next grapheme
        cluster from the end of the string.

        Returns:
            The next grapheme cluster as a `StringSlice`.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """
        if self._slice.byte_length() <= 0:
            raise StopIteration()
        comptime if Self.forward:
            return self.next().value()
        else:
            return self.next_back().value()

    def __len__(self) -> Int:
        """Return the number of remaining grapheme clusters.

        This is an O(n) operation that scans all remaining bytes to count
        grapheme cluster boundaries.

        Returns:
            The number of grapheme clusters remaining.
        """
        # The state machine reports a break at start-of-text (GB1) via its
        # initial `prev_gbp = GBP_CONTROL`, so every grapheme cluster --
        # including the first -- is signalled by a `True` return.
        var count = 0
        var state = _GraphemeBreakState()
        var remaining = self._slice
        var ptr = remaining.as_bytes().unsafe_ptr()
        var pos = 0
        var total = remaining.byte_length()

        while pos < total:
            # ASCII fast path. Safe-ASCII bytes all have GBP_OTHER. Two
            # consecutive safe-ASCII bytes always have a break between them
            # (GB999), and the first in such a run is a break start provided
            # the previous codepoint's GBP is not Prepend (GB9b). Runs of
            # safe-ASCII bytes are therefore one-grapheme-per-byte.
            if _is_safe_ascii_for_grapheme(ptr[unsafe_offset=pos]) and (
                state.prev_gbp != GBP_PREPEND
            ):
                var run_start = pos
                while pos < total and _is_safe_ascii_for_grapheme(
                    ptr[unsafe_offset=pos]
                ):
                    pos += 1
                count += pos - run_start
                _reset_grapheme_state_to_other(state)
                continue

            # Slow path: decode one codepoint and feed the state machine.
            var sub = Span[Byte, Self.origin](
                unsafe_ptr=ptr.unsafe_offset(pos), length=total - pos
            )
            var cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(sub)
            if _is_grapheme_break(state, cp.to_u32()):
                count += 1
            pos += num_bytes

        return count

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def remaining_byte_length(self) -> Int:
        """Returns the number of bytes not yet consumed by the iterator.

        This is O(1): it reports the size of the remaining range without
        scanning grapheme boundaries. Combined with the original byte length
        of the source slice, callers can compute how many bytes the iterator
        has produced so far without summing per-grapheme byte lengths.

        Returns:
            The byte length of the iterator's remaining range.
        """
        return self._slice.byte_length()

    def next(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the next grapheme cluster, or `None` if exhausted.

        Returns:
            The next grapheme cluster as a `StringSlice`, or `None`.
        """
        if self._slice.byte_length() <= 0:
            return None

        var start_ptr = self._slice.as_bytes().unsafe_ptr()
        var total_bytes = self._slice.byte_length()
        var consumed = 0

        # Decode the first codepoint of this grapheme cluster.
        var cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            self._slice._slice
        )

        if not self._state_primed:
            # First call, or state is not yet primed: feed this codepoint
            # to the state machine to establish the initial state.
            _ = _is_grapheme_break(self._state, cp.to_u32())
        # else: state was already updated for this codepoint when the
        # previous next() detected the break. Skip re-feeding.

        consumed += num_bytes

        # Continue consuming codepoints until we hit a break.
        var found_break = False
        while consumed < total_bytes:
            var remaining = Span[Byte, Self.origin](
                unsafe_ptr=self._slice.as_bytes()
                .unsafe_ptr()
                .unsafe_offset(consumed),
                length=total_bytes - consumed,
            )
            cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(remaining)

            if _is_grapheme_break(self._state, cp.to_u32()):
                # Found a break — the grapheme ends before this codepoint.
                # The state machine has already been updated with this
                # codepoint, so mark as primed for the next call.
                found_break = True
                break

            consumed += num_bytes

        self._state_primed = found_break

        # Advance the slice past this grapheme cluster. This moves the data
        # pointer, so any previously-cached reverse safe-start (an offset
        # relative to the old data pointer) is now stale.
        self._slice._slice._data = self._slice._slice._data.unsafe_offset(
            consumed
        )
        self._slice._slice._len -= consumed
        self._back_safe_known = False

        return StringSlice[Self.origin](
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=start_ptr, length=consumed
            )
        )

    def peek_back(mut self) -> Optional[StringSlice[Self.origin]]:
        """Return the last grapheme cluster without advancing the iterator.

        Repeated calls return the same value. The first reverse call (`peek_back`
        or `next_back`) does the backward scan to find a safe restart boundary
        and caches it; subsequent reverse calls reuse the cache and only pay
        for the forward scan from that boundary.

        Returns:
            The last grapheme cluster as a `StringSlice`, or `None` if the
            iterator is empty.
        """
        var total = self._slice.byte_length()
        if total <= 0:
            return None
        var grapheme_start = self._grapheme_start_of_last_cluster(total)
        return StringSlice[Self.origin](
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self._slice.as_bytes()
                .unsafe_ptr()
                .unsafe_offset(grapheme_start),
                length=total - grapheme_start,
            )
        )

    def next_back(mut self) -> Optional[StringSlice[Self.origin]]:
        """Get the last grapheme cluster in the underlying string, or `None`
        if the iterator is empty.

        This consumes one grapheme from the end of the remaining range. It
        does not share state with forward iteration (`next()`), so the two
        can be interleaved freely.

        The UAX #29 state machine is inherently forward-scanning, so
        `next_back()` backs up to a guaranteed grapheme boundary — a
        Control/CR/LF codepoint or the start of the string — and then
        forward-scans from that boundary. The safe boundary, once found,
        is cached and reused across subsequent reverse calls (a forward
        `next()` invalidates the cache because it moves the front pointer).
        Per-call cost is therefore dominated by the forward scan length:
        roughly proportional to the distance from the most recent Control/
        CR/LF codepoint to the cluster being returned. For text containing
        line breaks or whitespace this is small; for long runs without
        Control/CR/LF the forward scan extends back toward the start of
        the string and the per-call cost grows accordingly.

        Returns:
            The last grapheme cluster as a `StringSlice`, or `None`.
        """
        var total = self._slice.byte_length()
        if total <= 0:
            return None
        var grapheme_start = self._grapheme_start_of_last_cluster(total)
        var result = StringSlice[Self.origin](
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self._slice.as_bytes()
                .unsafe_ptr()
                .unsafe_offset(grapheme_start),
                length=total - grapheme_start,
            )
        )
        # Shrink the range from the end. Data pointer is unchanged, so the
        # cached `_back_safe_start` (if set) remains valid for future calls
        # — `_back_safe_start <= grapheme_start <= new total`.
        self._slice._slice._len = grapheme_start
        return result

    @doc_hidden
    def _grapheme_start_of_last_cluster(mut self, end: Int) -> Int:
        """Return the byte offset in `self._slice[:end]` where the last
        grapheme cluster begins, using the cached safe-start when available.

        Args:
            end: The end of the range to scan, in `(0, self._slice.byte_length()]`.

        Returns:
            The byte offset `<= end` that begins the last grapheme cluster.
        """
        var span = self._slice.as_bytes()
        # Invalidate the cache when the iter has shrunk down to (or past) the
        # cached safe-start: an empty forward-scan range would return that
        # offset as the boundary, producing a zero-length cluster and an
        # infinite loop.
        if not self._back_safe_known or self._back_safe_start >= end:
            self._back_safe_start = _find_safe_grapheme_start(span, end)
            self._back_safe_known = True
        var state = _GraphemeBreakState()
        var last_boundary = self._back_safe_start
        var pos = self._back_safe_start
        while pos < end:
            var cp, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
                span[pos:end]
            )
            if _is_grapheme_break(state, cp.to_u32()):
                last_boundary = pos
            pos += num_bytes
        return last_boundary


struct GraphemeIndicesIter[mut: Bool, //, origin: Origin[mut=mut]](
    ImplicitlyCopyable, Iterable, Iterator
):
    """Iterator over grapheme clusters paired with their starting byte offset.

    Each call to `__next__()` yields a `Tuple[Int, StringSlice[origin]]` where
    the first element is the byte offset (relative to the original string)
    at which the grapheme begins, and the second is the grapheme slice
    itself.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.

    Mirrors `str::grapheme_indices` from Rust's `unicode-segmentation` crate.
    Useful for text editors and UIs that need to map cursor byte positions
    back to grapheme boundaries.

    Example:

    ```mojo
    from std.testing import assert_equal
    var s = StringSlice("abc")
    var pairs = List[Tuple[Int, String]]()
    for off, g in s.grapheme_indices():
        pairs.append((off, String(g)))
    assert_equal(len(pairs), 3)
    assert_equal(pairs[0][0], 0)
    assert_equal(pairs[1][0], 1)
    assert_equal(pairs[2][0], 2)
    ```
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this grapheme indices iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = Tuple[Int, StringSlice[Self.origin]]
    """The element type yielded by iteration."""

    var _inner: GraphemeSliceIter[Self.origin]
    """Underlying grapheme slice iterator."""

    var _byte_offset: Int
    """Running byte offset of the next grapheme, relative to the original
    string."""

    @doc_hidden
    def __init__(out self, str_slice: StringSlice[Self.origin]):
        self._inner = GraphemeSliceIter(str_slice)
        self._byte_offset = 0

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the grapheme indices in the underlying string slice.

        Returns:
            An iterator yielding `(byte_offset, grapheme)` pairs.
        """
        return self.copy()

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> Tuple[Int, StringSlice[Self.origin]]:
        """Get the next `(byte_offset, grapheme)` pair.

        Returns:
            The byte offset at which the next grapheme starts and the
            grapheme slice itself.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """
        var g = self._inner.next()
        if not g:
            raise StopIteration()
        var offset = self._byte_offset
        self._byte_offset += g.unsafe_value().byte_length()
        return (offset, g.unsafe_value())


struct BytesIter[mut: Bool, //, origin: Origin[mut=mut]](
    ImplicitlyCopyable, Iterable, Iterator, Sized
):
    """Iterator over the raw UTF-8 bytes of a string slice, constructed by
    `StringSlice.bytes()`.

    Each call to `__next__()` yields the next `Byte` value from the underlying
    UTF-8 encoded data. Unlike `CodepointsIter` and `GraphemeSliceIter`, this
    iterator operates at the byte level and does not interpret multi-byte
    UTF-8 sequences as codepoints or grapheme clusters.

    Parameters:
        mut: Whether the underlying string data is mutable.
        origin: The origin of the underlying string data.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for this bytes iterator.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = Byte
    """The element type yielded by iteration."""

    var _iter: _SpanIter[Byte, Self.origin]
    """The underlying span iterator over the string's bytes."""

    @doc_hidden
    def __init__(out self, *, _slice: StringSlice[Self.origin]):
        self._iter = iter(_slice.as_bytes())

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterator over the underlying string's bytes.

        Returns:
            This iterator.
        """
        return self

    @always_inline
    def __next__(mut self) raises StopIteration -> Byte:
        """Get the next byte in the underlying string slice.

        Returns:
            The next byte in the string.

        Raises:
            `StopIteration` if the iterator has been exhausted.
        """
        return next(self._iter)

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        """Returns bounds `[lower, upper]` for the remaining iterator length.

        Returns:
            The lower and upper bound of this iterator.
        """
        return self._iter.bounds()

    @always_inline
    def __len__(self) -> Int:
        """Returns the remaining length of this iterator in bytes.

        Returns:
            Number of bytes remaining in this iterator.
        """
        return len(self._iter)
