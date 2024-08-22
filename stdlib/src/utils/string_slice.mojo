# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

"""Implements the StringSlice type.

You can import these APIs from the `utils.string_slice` module. For example:

```mojo
from utils import StringSlice
```
"""

from bit import count_leading_zeros
from utils import Span
from collections.string import _isspace
from collections import List
from memory import memcmp
from sys import simdwidthof, bitwidthof

alias StaticString = StringSlice[ImmutableStaticLifetime]
"""An immutable static string slice."""


fn _utf8_byte_type(b: SIMD[DType.uint8, _], /) -> __type_of(b):
    """UTF-8 byte type.

    Returns:
        The byte type.

    Notes:

        - 0 -> ASCII byte.
        - 1 -> continuation byte.
        - 2 -> start of 2 byte long sequence.
        - 3 -> start of 3 byte long sequence.
        - 4 -> start of 4 byte long sequence.
    """
    return count_leading_zeros(~(b & UInt8(0b1111_0000)))


@value
struct _ProcessedUtfBytes[simd_size: Int]:
    """Contains information about the previous chunk of UTF-8 bytes.
    Notably necessary to verify continuation bytes.
    """

    var raw_bytes: SIMD[DType.uint8, simd_size]
    var high_nibbles: SIMD[DType.uint8, simd_size]
    var carried_continuations: SIMD[DType.uint8, simd_size]


fn _get_last_carried_continuation_check_vector[
    simd_size: Int
]() -> SIMD[DType.uint8, simd_size]:
    """Returns a vector (9, 9, 9, 9, ... 9, 9, 9, 1)."""
    result = SIMD[DType.uint8, simd_size](9)
    result[simd_size - 1] = 1
    return result


@always_inline
fn _mm_alignr_epi8[
    simd_size: Int, //, count: Int
](a: SIMD[DType.uint8, simd_size], b: SIMD[DType.uint8, simd_size]) -> SIMD[
    DType.uint8, simd_size
]:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_alignr_epi8.html .
    """
    # This can be one instruction with ssse3. We can maybe force the compiler to use it.
    var concatenated = a.join(b)
    var shifted = concatenated.rotate_left[count]()
    return shifted.slice[a.size, offset = a.size]()


@always_inline
fn _subtract_with_saturation[
    simd_size: Int, //, b: Int
](a: SIMD[DType.uint8, simd_size]) -> SIMD[DType.uint8, simd_size]:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_subs_epu8.html .
    """
    alias b_as_vector = SIMD[DType.uint8, simd_size](b)
    return max(a, b_as_vector) - b_as_vector


@always_inline
fn _count_nibbles[
    simd_size: Int, //
](
    bytes: SIMD[DType.uint8, simd_size],
    inout answer: _ProcessedUtfBytes[simd_size],
):
    answer.raw_bytes = bytes
    answer.high_nibbles = bytes >> 4


@always_inline
fn _check_smaller_than_0xF4[
    simd_size: Int, //
](
    current_bytes: SIMD[DType.uint8, simd_size],
    inout has_error: SIMD[DType.bool, simd_size],
):
    var bigger_than_0xF4 = current_bytes > SIMD[DType.uint8, simd_size](0xF4)
    has_error |= bigger_than_0xF4


@always_inline
fn _continuation_lengths[
    simd_size: Int, //
](high_nibbles: SIMD[DType.uint8, simd_size]) -> SIMD[DType.uint8, simd_size]:
    # The idea is to end up with this pattern:
    #                                                                     error here
    # Input:  0xxxxxxx, 110xxxxx, 10xxxxxx, 1110xxxx, 10xxxxxx, 10xxxxxx, 10xxxxxx, 1111xxxx, ...
    # Output: 1       , 2,      , 0       , 3,      , 0       , 0       , 0       , 4
    # fmt: off
    alias table_of_continuations = SIMD[DType.uint8, 16](
        1, 1, 1, 1, 1, 1, 1, 1,  # 0xxx (ASCII)
        0, 0, 0, 0,              # 10xx (continuation)
        2, 2,                    # 110x
        3,                       # 1110
        4,                       # 1111, next should be 0 (not checked here)
    )
    # fmt: on
    return table_of_continuations.dynamic_shuffle(high_nibbles)


@always_inline
fn _carry_continuations[
    simd_size: Int, //
](
    initial_lengths: SIMD[DType.uint8, simd_size],
    previous_carries: SIMD[DType.uint8, simd_size],
) -> SIMD[DType.uint8, simd_size]:
    var right1 = _subtract_with_saturation[1](
        _mm_alignr_epi8[simd_size - 1](initial_lengths, previous_carries)
    )
    # Input:           0xxxxxxx, 110xxxxx, 10xxxxxx, 1110xxxx, 10xxxxxx, 10xxxxxx, 10xxxxxx, 1111xxxx,
    # initial_lengths: 1       , 2,      , 0       , 3,      , 0       , 0       , 0       , 4
    # right1           ?       , 0       , 1       , 0       , 2,      , 0       , 0       , 0
    var sum = initial_lengths + right1
    # sum              ?       , 2       , 1       , 3       , 2,      , 0       , 0       , 4

    var right2 = _subtract_with_saturation[2](
        _mm_alignr_epi8[simd_size - 2](sum, previous_carries)
    )
    # right2           ?       , ?       , ?       , 0       , 0,      , 1       , 0       , 0
    return sum + right2
    # return           ?       , ?       , ?       , 3       , 2,      , 1       , 0       , 4


@always_inline
fn _check_continuations[
    simd_size: Int, //
](
    initial_lengths: SIMD[DType.uint8, simd_size],
    carries: SIMD[DType.uint8, simd_size],
    inout has_error: SIMD[DType.bool, simd_size],
):
    var overunder = (initial_lengths < carries) == (
        SIMD[DType.uint8, simd_size](0) < initial_lengths
    )
    has_error |= overunder


@always_inline
fn _check_first_continuation_max[
    simd_size: Int, //
](
    # When 0b11101101 is found, next byte must be no larger than 0b10011111
    # When 0b11110100 is found, next byte must be no larger than 0b10001111
    # Next byte must be continuation, ie sign bit is set, so signed < is ok
    current_bytes: SIMD[DType.uint8, simd_size],
    off1_current_bytes: SIMD[DType.uint8, simd_size],
    inout has_error: SIMD[DType.bool, simd_size],
):
    var bad_follow_ED = (
        SIMD[DType.uint8, simd_size](0b10011111) < current_bytes
    ) & (off1_current_bytes == SIMD[DType.uint8, simd_size](0b11101101))
    var bad_follow_F4 = (
        SIMD[DType.uint8, simd_size](0b10001111) < current_bytes
    ) & (off1_current_bytes == SIMD[DType.uint8, simd_size](0b11110100))

    has_error |= bad_follow_ED | bad_follow_F4


@always_inline
fn _check_overlong[
    simd_size: Int, //
](
    current_bytes: SIMD[DType.uint8, simd_size],
    off1_current_bytes: SIMD[DType.uint8, simd_size],
    hibits: SIMD[DType.uint8, simd_size],
    previous_hibits: SIMD[DType.uint8, simd_size],
    inout has_error: SIMD[DType.bool, simd_size],
):
    """Map off1_hibits => error condition.

    hibits     off1    cur
    C       => < C2 && true
    E       => < E1 && < A0
    F       => < F1 && < 90
    else      false && false
    """
    var off1_hibits = _mm_alignr_epi8[simd_size - 1](hibits, previous_hibits)
    # fmt: off
    alias table1 = SIMD[DType.uint8, 16](
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, # 0xxx
        0x80, 0x80,  # 10xx => false
        0xC2, 0x80,  # 110x
        0xE1,        # 1110
        0xF1,        # 1111
    )
    var initial_mins = table1.dynamic_shuffle(off1_hibits).cast[DType.int8]()
    var initial_under = off1_current_bytes.cast[DType.int8]() < initial_mins
    alias table2 = SIMD[DType.uint8, 16](
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, # 10xx => False
        127, 127,   # 110x => True
        0xA0,       # 1110
        0x90,       # 1111
    )
    # fmt: on
    var second_mins = table2.dynamic_shuffle(off1_hibits).cast[DType.int8]()
    var second_under = current_bytes.cast[DType.int8]() < second_mins
    has_error |= initial_under & second_under


fn _check_utf8_bytes[
    simd_size: Int
](
    current_bytes: SIMD[DType.uint8, simd_size],
    previous: _ProcessedUtfBytes[simd_size],
    inout has_error: SIMD[DType.bool, simd_size],
) -> _ProcessedUtfBytes[simd_size]:
    # No memory loads, just vectorized arithmetic.
    # Each instruction counts there for performance.
    # Benchmark every change. It's a very hot loop.
    var pb = _ProcessedUtfBytes[simd_size](
        SIMD[DType.uint8, simd_size](),
        SIMD[DType.uint8, simd_size](),
        SIMD[DType.uint8, simd_size](),
    )
    _count_nibbles(current_bytes, pb)
    _check_smaller_than_0xF4(current_bytes, has_error)

    var initial_lengths = _continuation_lengths(pb.high_nibbles)
    pb.carried_continuations = _carry_continuations(
        initial_lengths, previous.carried_continuations
    )

    _check_continuations(initial_lengths, pb.carried_continuations, has_error)
    var off1_current_bytes = _mm_alignr_epi8[simd_size - 1](
        pb.raw_bytes, previous.raw_bytes
    )
    _check_first_continuation_max(current_bytes, off1_current_bytes, has_error)
    _check_overlong(
        current_bytes,
        off1_current_bytes,
        pb.high_nibbles,
        previous.high_nibbles,
        has_error,
    )

    return pb


# no_inline adds a 10% speedup for some reason
@no_inline
fn _is_valid_utf8(ptr: UnsafePointer[UInt8], length: Int) -> Bool:
    """Verify that the bytes are valid UTF-8.

    Args:
        ptr: The pointer to the data.
        length: The length of the items pointed to.

    Returns:
        Whether the data is valid UTF-8.

    #### UTF-8 coding format
    [Table 3-7 page 94](http://www.unicode.org/versions/Unicode6.0.0/ch03.pdf).
    Well-Formed UTF-8 Byte Sequences

    Code Points        | First Byte | Second Byte | Third Byte | Fourth Byte |
    :----------        | :--------- | :---------- | :--------- | :---------- |
    U+0000..U+007F     | 00..7F     |             |            |             |
    U+0080..U+07FF     | C2..DF     | 80..BF      |            |             |
    U+0800..U+0FFF     | E0         | ***A0***..BF| 80..BF     |             |
    U+1000..U+CFFF     | E1..EC     | 80..BF      | 80..BF     |             |
    U+D000..U+D7FF     | ED         | 80..***9F***| 80..BF     |             |
    U+E000..U+FFFF     | EE..EF     | 80..BF      | 80..BF     |             |
    U+10000..U+3FFFF   | F0         | ***90***..BF| 80..BF     | 80..BF      |
    U+40000..U+FFFFF   | F1..F3     | 80..BF      | 80..BF     | 80..BF      |
    U+100000..U+10FFFF | F4         | 80..***8F***| 80..BF     | 80..BF      |
    .
    """
    # Reference for this algorithm:
    # https://arxiv.org/abs/2010.03090
    # https://lemire.me/blog/2018/10/19/validating-utf-8-bytes-using-only-0-45-cycles-per-byte-avx-edition/
    alias simd_size = sys.simdbytewidth()
    var i: Int = 0
    var has_error = SIMD[DType.bool, simd_size]()
    var previous = _ProcessedUtfBytes(
        SIMD[DType.uint8, simd_size](),
        SIMD[DType.uint8, simd_size](),
        SIMD[DType.uint8, simd_size](),
    )
    while i + simd_size <= length:
        var current_bytes = (ptr + i).load[width=simd_size]()
        previous = _check_utf8_bytes(current_bytes, previous, has_error)
        if has_error.reduce_or():
            return False
        i += simd_size

    # last incomplete chunk
    if i != length:
        var buffer = SIMD[DType.uint8, simd_size](0)
        for j in range(i, length):
            buffer[j - i] = (ptr + j)[]
        previous = _check_utf8_bytes(buffer, previous, has_error)
    else:
        # Just check that the last carried_continuations is 1 or 0
        alias base = _get_last_carried_continuation_check_vector[simd_size]()
        var comparison = previous.carried_continuations > base
        has_error |= comparison

    return not has_error.reduce_or()


fn _is_newline_start(
    ptr: UnsafePointer[UInt8], read_ahead: Int = 1
) -> (Bool, Int):
    """Returns if the first item in the pointer is the start of
    a newline sequence, and its length.
    """
    # TODO add line and paragraph separator as StringLiteral
    # once Unicode escape sequences are accepted
    alias ` ` = UInt8(ord(" "))
    var rn = "\r\n"
    var next_line = List[UInt8](0xC2, 0x85)
    """TODO: \\x85"""
    var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
    """TODO: \\u2028"""
    var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
    """TODO: \\u2029"""

    var val = _utf8_byte_type(ptr[0])
    if val == 0:
        if read_ahead > 1:
            if memcmp(ptr, rn.unsafe_ptr(), 2) == 0:
                return True, 2
            _ = rn
        return ptr[0] != ` ` and _isspace(ptr[0]), 1
    elif val == 2 and read_ahead > 1:
        var comp = memcmp(ptr, next_line.unsafe_ptr(), 2) == 0
        _ = next_line
        return comp, 2
    elif val == 3 and read_ahead > 2:
        var comp = (
            memcmp(ptr, unicode_line_sep.unsafe_ptr(), 3) == 0
            or memcmp(ptr, unicode_paragraph_sep.unsafe_ptr(), 3) == 0
        )
        _ = unicode_line_sep, unicode_paragraph_sep
        return comp, 3
    return False, 1


@value
struct _StringSliceIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for StringSlice

    Parameters:
        is_mutable: Whether the slice is mutable.
        lifetime: The lifetime of the underlying string data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var continuation_bytes: Int
    var ptr: UnsafePointer[UInt8]
    var length: Int

    fn __init__(
        inout self, *, unsafe_pointer: UnsafePointer[UInt8], length: Int
    ):
        self.index = 0 if forward else length
        self.ptr = unsafe_pointer
        self.length = length
        self.continuation_bytes = 0
        for i in range(length):
            if _utf8_byte_type(unsafe_pointer[i]) == 1:
                self.continuation_bytes += 1

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> StringSlice[lifetime]:
        @parameter
        if forward:
            var byte_len = 1
            if self.continuation_bytes > 0:
                var byte_type = _utf8_byte_type(self.ptr[self.index])
                if byte_type != 0:
                    byte_len = int(byte_type)
                    self.continuation_bytes -= byte_len - 1
            self.index += byte_len
            return StringSlice[lifetime](
                unsafe_from_utf8_ptr=self.ptr + (self.index - byte_len),
                len=byte_len,
            )
        else:
            var byte_len = 1
            if self.continuation_bytes > 0:
                var byte_type = _utf8_byte_type(self.ptr[self.index - 1])
                if byte_type != 0:
                    while byte_type == 1:
                        byte_len += 1
                        var b = self.ptr[self.index - byte_len]
                        byte_type = _utf8_byte_type(b)
                    self.continuation_bytes -= byte_len - 1
            self.index -= byte_len
            return StringSlice[lifetime](
                unsafe_from_utf8_ptr=self.ptr + self.index, len=byte_len
            )

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index - self.continuation_bytes
        else:
            return self.index - self.continuation_bytes


struct StringSlice[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
](Stringable, Sized, Formattable):
    """
    A non-owning view to encoded string data.

    TODO:
    The underlying string data is guaranteed to be encoded using UTF-8.

    Parameters:
        is_mutable: Whether the slice is mutable.
        lifetime: The lifetime of the underlying string data.
    """

    var _slice: Span[UInt8, lifetime]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    fn __init__(inout self, literal: StringLiteral):
        """Construct a new string slice from a string literal.

        Args:
            literal: The literal to construct this string slice from.
        """

        # Its not legal to try to mutate a StringLiteral. String literals are
        # static data.
        constrained[
            not is_mutable, "cannot create mutable StringSlice of StringLiteral"
        ]()

        # Since a StringLiteral has static lifetime, it will outlive
        # whatever arbitrary `lifetime` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StringLiteral is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        # TODO(#933): use when llvm intrinsics can be used at compile time
        # debug_assert(
        #     _is_valid_utf8(literal.unsafe_ptr(), literal._byte_length()),
        #     "StringLiteral doesn't have valid UTF-8 encoding",
        # )
        self = StringSlice[lifetime](
            unsafe_from_utf8_ptr=literal.unsafe_ptr(), len=literal.byte_length()
        )

    @always_inline
    fn __init__(inout self, *, owned unsafe_from_utf8: Span[UInt8, lifetime]):
        """
        Construct a new StringSlice from a sequence of UTF-8 encoded bytes.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8: A slice of bytes encoded in UTF-8.
        """

        self._slice = unsafe_from_utf8^

    fn __init__(inout self, *, unsafe_from_utf8_strref: StringRef):
        """
        Construct a new StringSlice from a StringRef pointing to UTF-8 encoded
        bytes.

        Safety:
            - `unsafe_from_utf8_strref` MUST point to data that is valid for
              `lifetime`.
            - `unsafe_from_utf8_strref` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8_strref: A StringRef of bytes encoded in UTF-8.
        """
        var strref = unsafe_from_utf8_strref

        var byte_slice = Span[UInt8, lifetime](
            unsafe_ptr=strref.unsafe_ptr(),
            len=len(strref),
        )

        self = Self(unsafe_from_utf8=byte_slice)

    @always_inline
    fn __init__(
        inout self,
        *,
        unsafe_from_utf8_ptr: UnsafePointer[UInt8],
        len: Int,
    ):
        """
        Construct a StringSlice from a pointer to a sequence of UTF-8 encoded
        bytes and a length.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to at least `len` bytes of valid
              UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` must point to data that is live for the
              duration of `lifetime`.

        Args:
            unsafe_from_utf8_ptr: A pointer to a sequence of bytes encoded in
              UTF-8.
            len: The number of bytes of encoded data.
        """
        var byte_slice = Span[UInt8, lifetime](
            unsafe_ptr=unsafe_from_utf8_ptr,
            len=len,
        )

        self._slice = byte_slice

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Gets this slice as a standard `String`.

        Returns:
            The string representation of the slice.
        """
        return String(str_slice=self)

    fn __len__(self) -> Int:
        """Nominally returns the _length in Unicode codepoints_ (not bytes!).

        Returns:
            The length in Unicode codepoints.
        """
        var unicode_length = self.byte_length()

        for i in range(unicode_length):
            if _utf8_byte_type(self._slice[i]) == 1:
                unicode_length -= 1

        return unicode_length

    fn format_to(self, inout writer: Formatter):
        """
        Formats this string slice to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        writer.write_str(str_slice=self)

    fn __bool__(self) -> Bool:
        """Check if a string slice is non-empty.

        Returns:
           True if a string slice is non-empty, False otherwise.
        """
        return len(self._slice) > 0

    fn __eq__(self, rhs: StringSlice) -> Bool:
        """Verify if a string slice is equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string slices are equal in length and contain the same elements, False otherwise.
        """
        if not self and not rhs:
            return True
        if len(self) != len(rhs):
            return False
        # same pointer and length, so equal
        if self._slice.unsafe_ptr() == rhs._slice.unsafe_ptr():
            return True
        for i in range(len(self)):
            if self._slice[i] != rhs._slice.unsafe_ptr()[i]:
                return False
        return True

    @always_inline
    fn __eq__(self, rhs: String) -> Bool:
        """Verify if a string slice is equal to a string.

        Args:
            rhs: The string to compare against.

        Returns:
            True if the string slice is equal to the input string in length and contain the same bytes, False otherwise.
        """
        return self == rhs.as_string_slice()

    @always_inline
    fn __eq__(self, rhs: StringLiteral) -> Bool:
        """Verify if a string slice is equal to a literal.

        Args:
            rhs: The literal to compare against.

        Returns:
            True if the string slice is equal to the input literal in length and contain the same bytes, False otherwise.
        """
        return self == rhs.as_string_slice()

    @always_inline
    fn __ne__(self, rhs: StringSlice) -> Bool:
        """Verify if span is not equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string slices are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: String) -> Bool:
        """Verify if span is not equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string and slice are not equal in length or contents, False otherwise.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Verify if span is not equal to a literal.

        Args:
            rhs: The string literal to compare against.

        Returns:
            True if the slice is not equal to the literal in length or contents, False otherwise.
        """
        return not self == rhs

    fn __iter__(ref [_]self) -> _StringSliceIter[__lifetime_of(self)]:
        """Iterate over elements of the string slice, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return _StringSliceIter[__lifetime_of(self)](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    fn __reversed__(
        ref [_]self,
    ) -> _StringSliceIter[__lifetime_of(self), False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return _StringSliceIter[__lifetime_of(self), forward=False](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn as_bytes_slice(self) -> Span[UInt8, lifetime]:
        """Get the sequence of encoded bytes as a slice of the underlying string.

        Returns:
            A slice containing the underlying sequence of encoded bytes.
        """
        return self._slice

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Gets a pointer to the first element of this string slice.

        Returns:
            A pointer pointing at the first element of this string slice.
        """

        return self._slice.unsafe_ptr()

    @always_inline
    fn byte_length(self) -> Int:
        """Get the length of this string slice in bytes.

        Returns:
            The length of this string slice in bytes.
        """

        return len(self.as_bytes_slice())

    @always_inline
    @deprecated("use byte_length() instead")
    fn _byte_length(self) -> Int:
        """Get the length of this string slice in bytes.

        Returns:
            The length of this string slice in bytes.
        """

        return len(self.as_bytes_slice())

    fn _strref_dangerous(self) -> StringRef:
        """Returns an inner pointer to the string as a StringRef.

        Safety:
            This functionality is extremely dangerous because Mojo eagerly
            releases strings.  Using this requires the use of the
            _strref_keepalive() method to keep the underlying string alive long
            enough.
        """
        return StringRef(self.unsafe_ptr(), self.byte_length())

    fn _strref_keepalive(self):
        """A no-op that keeps `self` alive through the call.  This
        can be carefully used with `_strref_dangerous()` to wield inner pointers
        without the string getting deallocated early.
        """
        pass

    fn _from_start(self, start: Int) -> Self:
        """Gets the `StringSlice` pointing to the substring after the specified slice start position.

        If start is negative, it is interpreted as the number of characters
        from the end of the string to start at.

        Args:
            start: Starting index of the slice.

        Returns:
            A `StringSlice` borrowed from the current string containing the
            characters of the slice starting at start.
        """

        var self_len = len(self)

        var abs_start: Int
        if start < 0:
            # Avoid out of bounds earlier than the start
            # len = 5, start = -3,  then abs_start == 2, i.e. a partial string
            # len = 5, start = -10, then abs_start == 0, i.e. the full string
            abs_start = max(self_len + start, 0)
        else:
            # Avoid out of bounds past the end
            # len = 5, start = 2,   then abs_start == 2, i.e. a partial string
            # len = 5, start = 8,   then abs_start == 5, i.e. an empty string
            abs_start = min(start, self_len)

        debug_assert(
            abs_start >= 0, "strref absolute start must be non-negative"
        )
        debug_assert(
            abs_start <= self_len,
            "strref absolute start must be less than source String len",
        )

        # TODO: We assumes the StringSlice only has ASCII.
        # When we support utf-8 slicing, we should drop self._slice[abs_start:]
        # and use something smarter.
        return StringSlice(unsafe_from_utf8=self._slice[abs_start:])

    fn find(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        if not substr:
            return 0

        if len(self) < len(substr) + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._from_start(start)

        var loc = stringref._memmem(
            haystack_str.unsafe_ptr(),
            len(haystack_str),
            substr.unsafe_ptr(),
            len(substr),
        )

        if not loc:
            return -1

        return int(loc) - int(self.unsafe_ptr())

    fn isspace(self) -> Bool:
        """Determines whether every character in the given StringSlice is a
        python whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\r\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole StringSlice is made up of whitespace characters
                listed above, otherwise False.
        """

        if self.byte_length() == 0:
            return False

        # TODO add line and paragraph separator as stringliteral
        # once Unicode escape sequences are accepted
        var next_line = List[UInt8](0xC2, 0x85)
        """TODO: \\x85"""
        var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
        """TODO: \\u2028"""
        var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
        """TODO: \\u2029"""

        for s in self:
            var no_null_len = s.byte_length()
            var ptr = s.unsafe_ptr()
            if no_null_len == 1 and _isspace(ptr[0]):
                continue
            elif (
                no_null_len == 2 and memcmp(ptr, next_line.unsafe_ptr(), 2) == 0
            ):
                continue
            elif no_null_len == 3 and (
                memcmp(ptr, unicode_line_sep.unsafe_ptr(), 3) == 0
                or memcmp(ptr, unicode_paragraph_sep.unsafe_ptr(), 3) == 0
            ):
                continue
            else:
                return False
        _ = next_line, unicode_line_sep, unicode_paragraph_sep
        return True

    fn splitlines(self, keepends: Bool = False) -> List[String]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\t\\n\\r\\r\\n\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        var output = List[String]()
        var length = self.byte_length()
        var current_offset = 0
        var ptr = self.unsafe_ptr()

        while current_offset < length:
            var eol_location = length - current_offset
            var eol_length = 0
            var curr_ptr = ptr.offset(current_offset)

            for i in range(current_offset, length):
                var read_ahead = 3 if i < length - 2 else (
                    2 if i < length - 1 else 1
                )
                var res = _is_newline_start(ptr.offset(i), read_ahead)
                if res[0]:
                    eol_location = i - current_offset
                    eol_length = res[1]
                    break

            var str_len: Int
            var end_of_string = False
            if current_offset >= length:
                end_of_string = True
                str_len = 0
            elif keepends:
                str_len = eol_location + eol_length
            else:
                str_len = eol_location

            output.append(
                String(Self(unsafe_from_utf8_ptr=curr_ptr, len=str_len))
            )

            if end_of_string:
                break
            current_offset += eol_location + eol_length

        return output^
