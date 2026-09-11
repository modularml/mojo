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
"""Provides a compact set of non-negative integers backed by inline storage.

Optimized for space (1 bit per element) and speed (O(1) operations).
Offers set/clear/test/toggle and fast population count.

Example:
```mojo
from std.collections import BitSet

var bs = BitSet[128]()  # 128-bit set, all clear
bs.set(42)              # Mark value 42 as present.
if bs.test(42):         # Check membership.
    print("hit")        # Prints "hit".
bs.clear(42)            # Remove 42.
print(len(bs))          # Prints 0.
```
"""
# ---------------------------------------------------------------------------


from std.math import ceildiv, align_down
from std.sys import simd_width_of, bit_width_of

from std.algorithm import vectorize
from std.bit import log2_floor, pop_count
from std.format._utils import FormatStruct
from std.memory import pack_bits, unsafe_memcpy, unsafe_memset_zero

from std.collections.array import Array

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

comptime _WORD_BITS = bit_width_of[DType.int64]()
comptime _WORD_BITS_LOG2 = log2_floor(_WORD_BITS)


@always_inline
def _word_index(idx: Int) -> Int:
    """Computes the 0-based index of the 64-bit word containing bit `idx`."""
    return idx >> _WORD_BITS_LOG2


@always_inline
def _bit_mask(idx: Int) -> Int:
    """Returns a Int64 mask with only the bit corresponding to `idx` set."""
    return 1 << (idx & _WORD_BITS - 1)


@always_inline
def _range_mask(lo: Int, hi: Int) -> Int64:
    """Creates an Int64 mask with bits in `[lo, hi)` set within a single Int.

    Args:
        lo: The low bit position, inclusive (`0 <= lo < hi`).
        hi: The high bit position, exclusive (`lo < hi <= _WORD_BITS`).

    Returns:
        An Int64 mask with bits in `[lo, hi)` set within a single Int.
    """
    # `1 << _WORD_BITS` overflows Int64, so special-case a full-width span.
    var high = ~Int64(0) if hi == _WORD_BITS else (Int64(1) << Int64(hi)) - 1
    var low = (Int64(1) << Int64(lo)) - 1
    return high & ~low


@always_inline
def _check_index_bounds[operation_name: StaticString](idx: Int, max_size: Int):
    """Checks if the index is within bounds for a BitSet operation.

    Parameters:
        operation_name: The name of the operation for error reporting.

    Args:
        idx: The index to check.
        max_size: The maximum size of the BitSet.
    """
    debug_assert(
        idx < max_size,
        "BitSet index out of bounds when ",
        operation_name,
        " bit: ",
        idx,
        " >= ",
        max_size,
    )


# ===-----------------------------------------------------------------------===#
# BitSet
# ===-----------------------------------------------------------------------===#


struct BitSet[size: Int](Boolable, Copyable, Defaultable, Sized, Writable):
    """A grow-only set storing non-negative integers efficiently using bits.

    Parameters:
        size: The maximum number of bits the bitset can store.

    Each integer element is represented by a single bit within an array
    of 64-bit words (`Int64`). This structure is optimized for:

    *   **Compactness:** Uses 64 times less memory than `List[Bool]`.
    *   **Speed:** Offers O(1) time complexity for `set`, `clear`, `test`,
        and `toggle` operations (single word load/store).

    Ideal for applications like data-flow analysis, graph algorithms, or
    any task requiring dense sets of small integers where memory and
    lookup speed are critical.
    """

    comptime _words_size = max(1, ceildiv(Self.size, _WORD_BITS))
    var _words: Array[Int64, Self._words_size]  # Payload storage.

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #

    def __init__(out self):
        """Initializes an empty BitSet with zero capacity and size."""
        self._words = type_of(self._words)(fill=0)

    def __init__(init: SIMD[.bool, _], out self: BitSet[init.length]):
        """Initializes a BitSet with the given SIMD vector of booleans.

        Args:
            init: A SIMD vector of booleans to initialize the bitset with.
        """
        comptime assert (
            max(Int(init.length), _WORD_BITS) // _WORD_BITS == Self._words_size
        )
        self._words = type_of(self._words)(uninitialized=True)
        comptime step = min(Int(init.length), _WORD_BITS)

        comptime for i in range(Self._words_size):
            self._words.unsafe_get(i) = pack_bits(
                init.slice[step, offset=i * step]()
            ).cast[.int64]()

    def __init__[
        other_size: Int,
        //,
        *,
        truncate: Bool = False,
    ](out self, *, resized_from: BitSet[other_size]):
        """Initializes a BitSet by copying from a BitSet of a different size.

        if `truncate` is `False`, this constructor will `debug_assert` that no
        set bits are lost when narrowing. If `truncate` is `True`, the
        constructor will silently drop any set bits at or above the new size.
        When narrowing, bits at or above `size` are dropped; when widening, the
        new high bits are cleared.

        Parameters:
            other_size: The `size` of the source bitset.
            truncate: Whether or not to allow truncating set bits.

        Args:
            resized_from: The BitSet to copy from.
        """

        comptime if not truncate and Self.size < other_size:
            debug_assert(
                resized_from.test_range[False, lo=Self.size](),
                "Cast would truncate set bits.",
            )

        comptime if other_size == Self.size:
            # delegate to copy constructor if sizes match
            self._words = rebind[type_of(self._words)](
                resized_from._words
            ).copy()
        else:
            comptime words_to_copy = min(
                Self._words_size, resized_from._words_size
            )

            var values = resized_from.copy()
            comptime if resized_from.size < Self.size:
                values._zero_upper()

            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(self._words)
            )

            unsafe_memcpy(
                dest=self._words.unsafe_ptr(),
                src=values._words.unsafe_ptr(),
                count=words_to_copy,
            )

            comptime if Self._words_size > resized_from._words_size:
                unsafe_memset_zero(
                    self._words.unsafe_ptr().unsafe_offset(
                        resized_from._words_size
                    ),
                    Self._words_size - resized_from._words_size,
                )

            comptime if Self.size < resized_from.size:
                self._zero_upper()

    def __init__[
        other_size: Int,
        //,
        *,
    ](out self, *, resized_from: BitSet[other_size], truncate_set_bits: ()):
        """Initializes a BitSet by copying from a BitSet of a different size.

        When narrowing, bits at or above `size` are dropped; when widening, the
        new high bits are cleared.

        Parameters:
            other_size: The `size` of the source bitset.

        Args:
            resized_from: The BitSet to copy from.
            truncate_set_bits: A marker argument to disambiguate an overload.
        """

        self = Self.__init__[truncate=True](resized_from=resized_from)

    # --------------------------------------------------------------------- #
    # Capacity queries
    # --------------------------------------------------------------------- #

    @always_inline
    def __len__(self) -> Int:
        """Counts the total number of bits that are set to 1 in the bitset.

        Uses the efficient `pop_count` intrinsic for each underlying word.
        The complexity is proportional to the number of words used by the
        bitset's capacity (`_words_size`), not the logical size (`len`).

        Returns:
            The total count of set bits (population count).
        """
        var total = 0

        comptime for i in range(self._words_size):
            total += Int(pop_count(self._words.unsafe_get(i)))

        return total

    @always_inline
    def __bool__(self) -> Bool:
        """Checks if the bitset is non-empty (contains at least one set bit).

        Returns:
            True if at least one bit is set, False otherwise.
        """
        return len(self) != 0

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    @always_inline
    def _zero_upper(mut self):
        """Clears any bits in the last word that lie beyond the logical `size`.

        When `size` is not a multiple of 64, the last storage word has unused
        high bits. Operations like `set_all` and `toggle_all` write to the full
        word, leaving those bits set. This method masks them back to zero so
        that `__len__` (population count) and other word-level operations
        remain correct. When `size` is an exact multiple of 64, this method
        compiles away entirely.
        """
        comptime bits_in_last_word = Self.size % _WORD_BITS
        comptime if bits_in_last_word != 0:
            comptime mask = Int64((1 << bits_in_last_word) - 1)
            self._words.unsafe_get(Self._words_size - 1) &= mask

    # --------------------------------------------------------------------- #
    # Bit manipulation
    # --------------------------------------------------------------------- #

    @always_inline
    def set(mut self, idx: Int):
        """Sets the bit at the specified index `idx` to 1.

        If `idx` is greater than or equal to the current logical size,
        the logical size is updated. Aborts if `idx` is negative or
        greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to set (must be < `size`).
        """
        _check_index_bounds["set"](idx, Self.size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) |= Int64(_bit_mask(idx))

    @always_inline
    def clear(mut self, idx: Int):
        """Clears the bit at the specified index `idx` (sets it to 0).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`. Does not change the logical size.

        Args:
            idx: The non-negative index of the bit to clear (must be < `size`).
        """
        _check_index_bounds["clearing"](idx, Self.size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) &= Int64(~_bit_mask(idx))

    @always_inline
    def toggle(mut self, idx: Int):
        """Toggles (inverts) the bit at the specified index `idx`.

        If the bit becomes 1 and `idx` is greater than or equal to the
        current logical size, the logical size is updated. Aborts if `idx`
        is negative or greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to toggle (must be < `size`).
        """
        _check_index_bounds["toggling"](idx, Self.size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) ^= Int64(_bit_mask(idx))

    @always_inline
    def test(self, idx: Int) -> Bool:
        """Tests if the bit at the specified index `idx` is set (is 1).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`.

        Args:
            idx: The non-negative index of the bit to test (must be < `size`).

        Returns:
            True if the bit at `idx` is set, False otherwise.
        """
        _check_index_bounds["testing"](idx, Self.size)
        var w = _word_index(idx)
        return (self._words.unsafe_get(w) & Int64(_bit_mask(idx))) != 0

    def __contains__(self, idx: Int) -> Bool:
        """Returns `True` if the bit at `idx` is set.

        Enables `if idx in bitset:` syntax.

        Args:
            idx: The non-negative index to check (must be < `size`).

        Returns:
            True if the bit at `idx` is set, False otherwise.
        """
        return self.test(idx)

    @always_inline
    def test_range[
        bit_value: Bool,
        *,
        lo: Int = 0,
        hi: Int = Self.size,
        max_unroll: Int = 16,
    ](self) -> Bool:
        """Tests whether every bit in `[lo, hi)` equals `bit_value`.

        With `lo`/`hi` known at compile time, the word indices, bit offsets,
        and boundary masks all fold away. The two boundary words are masked
        and checked scalar; the interior full words are checked in SIMD
        batches. An empty range is vacuously true.

        Parameters:
            bit_value: The value every in-range bit must have (1 if True, else 0).
            lo: The low bit index, inclusive.
            hi: The high bit index, exclusive.
            max_unroll: The maximum unroll factor for SIMD loops.

        Returns:
            True if every bit in the range equals `bit_value`, False otherwise.
        """
        comptime assert (
            0 <= lo
        ), t"test_range: invalid range [lo, hi) = [{lo}, {hi})"
        comptime assert (
            lo <= hi
        ), t"test_range: invalid range [lo, hi) = [{lo}, {hi})"
        comptime assert (
            hi <= Self.size
        ), t"test_range: invalid range [lo, hi) = [{lo}, {hi})"
        comptime assert (
            max_unroll > 0
        ), t"test_range: max_unroll must be positive, got {max_unroll}"

        comptime if lo == hi:
            return True
        else:
            comptime word_lo = _word_index(lo)
            comptime word_hi = _word_index(hi - 1)
            comptime lo_off = lo & (_WORD_BITS - 1)
            comptime hi_off = ((hi - 1) & (_WORD_BITS - 1)) + 1

            # A "violation" is a bit that disagrees with the expected value: a 0
            # bit when checking for all-set, or a 1 bit when checking all-unset.
            @always_inline
            def _violations[
                width: Int, //
            ](word: SIMD[.int64, width]) -> SIMD[.int64, width]:
                comptime if bit_value:
                    return ~word
                else:
                    return word

            comptime if word_lo == word_hi:
                comptime mask = _range_mask(lo_off, hi_off)
                return (
                    _violations[width=1](self._words.unsafe_get(word_lo)) & mask
                ) == 0
            else:
                comptime lead = _range_mask(lo_off, _WORD_BITS)
                if (
                    _violations[width=1](self._words.unsafe_get(word_lo)) & lead
                ) != 0:
                    return False
                comptime trail = _range_mask(0, hi_off)
                if (
                    _violations[width=1](self._words.unsafe_get(word_hi))
                    & trail
                ) != 0:
                    return False

                # Interior words `(word_lo, word_hi)` are fully in range, so each
                # must be all-ones (all_set) or all-zeros. Accumulate violation
                # bits across SIMD batches, then handle the scalar tail.
                comptime simd_width = simd_width_of[Int64]()
                comptime interior_lo = word_lo + 1
                comptime n_batches = (word_hi - interior_lo) // simd_width

                comptime if n_batches > 0:
                    comptime unroll_factor = min(n_batches, max_unroll)
                    comptime unrolled = align_down(n_batches, unroll_factor)

                    def kernel[width: Int](batch: Int) {mut, imm self} -> Bool:
                        var vec = self._words.unsafe_ptr().unsafe_load[
                            width=width
                        ](interior_lo + batch * width)

                        return _violations[width=width](vec).reduce_or() != 0

                    for i in range(0, unrolled, unroll_factor):
                        comptime for j in range(unroll_factor):
                            if kernel[width=simd_width](i + j):
                                return False

                    comptime for i in range(unrolled, n_batches):
                        if kernel[width=simd_width](i):
                            return False

                comptime tail_lo = interior_lo + n_batches * simd_width

                comptime for i in range(tail_lo, word_hi):
                    if _violations[width=1](self._words.unsafe_get(i)) != 0:
                        return False

                return True

    def clear_all(mut self):
        """Clears all bits in the set, resetting the logical size to 0.

        The allocated storage capacity remains unchanged. Equivalent to
        re-initializing the set with `Self()`.
        """
        self = Self()

    def toggle_all(mut self):
        """Toggles (inverts) all bits in the set up to the compile-time `size`.
        """

        comptime for i in range(self._words_size):
            self._words.unsafe_get(i) ^= ~0
        self._zero_upper()

    def set_all(mut self):
        """Sets all bits in the set up to the compile-time `size`."""

        comptime for i in range(self._words_size):
            self._words.unsafe_get(i) = ~0
        self._zero_upper()

    # --------------------------------------------------------------------- #
    # Set operations
    # --------------------------------------------------------------------- #
    @always_inline
    @staticmethod
    def _vectorize_apply[](
        left: Self,
        right: Self,
        func: Some[
            def[
                simd_width: Int
            ](
                SIMD[.int64, simd_width],
                SIMD[.int64, simd_width],
            ) -> SIMD[
                .int64, simd_width
            ]
        ],
    ) -> Self:
        """Applies a vectorized binary operation between two bitsets.

        This internal utility function optimizes set operations by processing
        multiple words in parallel using SIMD instructions when possible. It
        applies the provided function to corresponding words from both bitsets
        and returns a new bitset with the results.

        The vectorized operation is applied to each word in the bitsets but only
        if the number of words in the bitsets is greater than or equal to the
        SIMD width.

        Args:
            left: The first bitset operand.
            right: The second bitset operand.
            func: A function that takes two SIMD vectors of UInt64 values and
                returns a SIMD vector with the result of the operation. The
                function should implement the desired set operation (e.g.,
                union, intersection).

        Returns:
            A new bitset containing the result of applying the function to each
            corresponding pair of words from the input bitsets.
        """
        comptime simd_width = simd_width_of[Int64]()
        var res = Self()

        # Define a vectorized operation that processes multiple words at once
        @always_inline
        def _intersect[
            simd_width: Int
        ](offset: Int) {mut res, imm left, imm right, imm func}:
            # Initialize SIMD vectors to hold multiple words from each bitset
            var left_vec = SIMD[.int64, simd_width]()
            var right_vec = SIMD[.int64, simd_width]()

            # Load a batch of words from both bitsets into SIMD vectors
            comptime for i in range(simd_width):
                left_vec[i] = left._words.unsafe_get(offset + i)
                right_vec[i] = right._words.unsafe_get(offset + i)

            # Apply the provided operation (union, intersection, etc.) to the
            # vectors
            var result_vec = func[simd_width](left_vec, right_vec)

            # Store the results back into the result bitset
            comptime for i in range(simd_width):
                res._words.unsafe_get(offset + i) = result_vec[i]

        # Choose between vectorized or scalar implementation based on word count
        comptime if Self._words_size >= simd_width:
            # If we have enough words, use SIMD vectorization for better
            # performance
            vectorize[simd_width](Self._words_size, _intersect)
        else:
            # For small bitsets, use a simple scalar implementation
            comptime for i in range(Self._words_size):
                res._words.unsafe_get(i) = Int64(
                    func[simd_width=1](
                        left._words.unsafe_get(i),
                        right._words.unsafe_get(i),
                    )
                )

        return res^

    def union(self, other: Self) -> Self:
        """Returns a new bitset that is the union of `self` and `other`.

        Args:
            other: The bitset to union with.

        Returns:
            A new bitset containing all elements from both sets.
        """

        @always_inline
        def _union[
            simd_width: Int
        ](
            left: SIMD[.int64, simd_width],
            right: SIMD[.int64, simd_width],
        ) -> SIMD[.int64, simd_width]:
            return left | right

        return Self._vectorize_apply(self, other, _union)

    def intersection(self, other: Self) -> Self:
        """Returns a new bitset that is the intersection of `self` and `other`.

        Args:
            other: The bitset to intersect with.

        Returns:
            A new bitset containing only the elements present in both sets.
        """

        @always_inline
        def _intersection[
            simd_width: Int
        ](
            left: SIMD[.int64, simd_width],
            right: SIMD[.int64, simd_width],
        ) -> SIMD[.int64, simd_width]:
            return left & right

        return Self._vectorize_apply(self, other, _intersection)

    def difference(self, other: Self) -> Self:
        """Returns a new bitset that is the difference of `self` and `other`.

        Args:
            other: The bitset to subtract from `self`.

        Returns:
            A new bitset containing elements from `self` that are not in `other`.
        """

        @always_inline
        def _difference[
            simd_width: Int
        ](
            left: SIMD[.int64, simd_width],
            right: SIMD[.int64, simd_width],
        ) -> SIMD[.int64, simd_width]:
            return left & ~right

        return Self._vectorize_apply(self, other, _difference)

    def symmetric_difference(self, other: Self) -> Self:
        """Returns a new bitset that is the symmetric difference of `self`
        and `other`.

        Args:
            other: The bitset to compute the symmetric difference with.

        Returns:
            A new bitset containing elements that are in exactly one of the
            two sets.
        """

        @always_inline
        def _xor[
            simd_width: Int
        ](
            left: SIMD[.int64, simd_width],
            right: SIMD[.int64, simd_width],
        ) -> SIMD[.int64, simd_width]:
            return left ^ right

        return Self._vectorize_apply(self, other, _xor)

    def __or__(self, other: Self) -> Self:
        """Returns the union of `self` and `other`.

        Args:
            other: The bitset to union with.

        Returns:
            A new bitset containing all elements from both sets.
        """
        return self.union(other)

    def __ior__(mut self, other: Self):
        """Updates `self` to be the union of `self` and `other`.

        Args:
            other: The bitset to union with.
        """
        self = self.union(other)

    def __and__(self, other: Self) -> Self:
        """Returns the intersection of `self` and `other`.

        Args:
            other: The bitset to intersect with.

        Returns:
            A new bitset containing only the elements present in both sets.
        """
        return self.intersection(other)

    def __iand__(mut self, other: Self):
        """Updates `self` to be the intersection of `self` and `other`.

        Args:
            other: The bitset to intersect with.
        """
        self = self.intersection(other)

    def __sub__(self, other: Self) -> Self:
        """Returns the difference of `self` and `other`.

        Args:
            other: The bitset to subtract from `self`.

        Returns:
            A new bitset containing elements from `self` that are not in
            `other`.
        """
        return self.difference(other)

    def __isub__(mut self, other: Self):
        """Updates `self` to be the difference of `self` and `other`.

        Args:
            other: The bitset to subtract from `self`.
        """
        self = self.difference(other)

    def __xor__(self, other: Self) -> Self:
        """Returns the symmetric difference of `self` and `other`.

        Args:
            other: The bitset to compute the symmetric difference with.

        Returns:
            A new bitset containing elements that are in exactly one of the
            two sets.
        """
        return self.symmetric_difference(other)

    def __ixor__(mut self, other: Self):
        """Updates `self` to be the symmetric difference of `self` and `other`.

        Args:
            other: The bitset to compute the symmetric difference with.
        """
        self = self.symmetric_difference(other)

    # --------------------------------------------------------------------- #
    # Representation helpers
    # --------------------------------------------------------------------- #

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the set bits to the given writer.
        Outputs the indices of the set bits in ascending order, enclosed in
        curly braces and separated by commas (e.g., "{1, 5, 42}"). Uses
        efficient bitwise operations to find set bits without iterating
        through every possible bit.

        Args:
            writer: The writer instance to output the representation to.
        """
        writer.write("{")
        var first = True

        # Iterate through words rather than individual bits
        for word_idx in range(self._words_size):
            var word = self._words.unsafe_get(word_idx)

            # Skip empty words entirely
            if word == 0:
                continue

            var bit_idx_base = word_idx << _WORD_BITS_LOG2

            # Process bits efficiently using bit manipulation
            while word != 0:
                # Find position of rightmost 1-bit using binary tricks
                # (x & -x) isolates the rightmost 1-bit
                var rightmost_bit = word & (~word + 1)

                # Calculate the position of the bit within the word
                var bit_pos = pop_count(rightmost_bit - 1)

                # Write the absolute bit index
                var abs_idx = Int64(bit_idx_base) + bit_pos

                # Skip bits that would be beyond the maximum size
                if abs_idx >= Int64(Self.size):
                    break

                if not first:
                    writer.write(", ")
                writer.write(abs_idx)
                first = False

                # Clear the rightmost bit
                word &= ~rightmost_bit

        writer.write("}")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the `BitSet` to the writer.

        Args:
            writer: The value to write to.
        """
        FormatStruct(writer, "BitSet").params(Self.size).fields(self)
