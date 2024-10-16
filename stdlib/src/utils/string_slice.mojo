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
from utils.span import Span, AsBytesRead, AsBytesWrite
from collections.string import _isspace
from collections import List, Optional
from memory import memcmp, UnsafePointer
from sys import simdwidthof, bitwidthof

alias StaticString = StringSlice[StaticConstantOrigin]
"""An immutable static string slice."""


fn _count_utf8_continuation_bytes(span: Span[UInt8]) -> Int:
    alias sizes = (256, 128, 64, 32, 16, 8)
    var ptr = span.unsafe_ptr()
    var num_bytes = len(span)
    var amnt: Int = 0
    var processed = 0

    @parameter
    for i in range(len(sizes)):
        alias s = sizes.get[i, Int]()

        @parameter
        if simdwidthof[DType.uint8]() >= s:
            var rest = num_bytes - processed
            for _ in range(rest // s):
                var vec = (ptr + processed).load[width=s]()
                var comp = (vec & 0b1100_0000) == 0b1000_0000
                amnt += int(comp.cast[DType.uint8]().reduce_add())
                processed += s

    for i in range(num_bytes - processed):
        amnt += int((ptr[processed + i] & 0b1100_0000) == 0b1000_0000)

    return amnt


@always_inline
fn _unicode_codepoint_utf8_byte_length(c: Int) -> Int:
    debug_assert(
        0 <= c <= 0x10FFFF, "Value: ", c, " is not a valid Unicode code point"
    )
    alias sizes = SIMD[DType.int32, 4](0, 0b0111_1111, 0b0111_1111_1111, 0xFFFF)
    return int((sizes < c).cast[DType.uint8]().reduce_add())


fn _shift_unicode_to_utf8(ptr: UnsafePointer[UInt8], c: Int, num_bytes: Int):
    """Shift unicode to utf8 representation.

    ### Unicode (represented as UInt32 BE) to UTF-8 conversion:
    - 1: 00000000 00000000 00000000 0aaaaaaa -> 0aaaaaaa
        - a
    - 2: 00000000 00000000 00000aaa aabbbbbb -> 110aaaaa 10bbbbbb
        - (a >> 6)  | 0b11000000, b         | 0b10000000
    - 3: 00000000 00000000 aaaabbbb bbcccccc -> 1110aaaa 10bbbbbb 10cccccc
        - (a >> 12) | 0b11100000, (b >> 6)  | 0b10000000, c        | 0b10000000
    - 4: 00000000 000aaabb bbbbcccc ccdddddd -> 11110aaa 10bbbbbb 10cccccc
    10dddddd
        - (a >> 18) | 0b11110000, (b >> 12) | 0b10000000, (c >> 6) | 0b10000000,
        d | 0b10000000
    """
    if num_bytes == 1:
        ptr[0] = UInt8(c)
        return

    var shift = 6 * (num_bytes - 1)
    var mask = UInt8(0xFF) >> (num_bytes + 1)
    var num_bytes_marker = UInt8(0xFF) << (8 - num_bytes)
    ptr[0] = ((c >> shift) & mask) | num_bytes_marker
    for i in range(1, num_bytes):
        shift -= 6
        ptr[i] = ((c >> shift) & 0b0011_1111) | 0b1000_0000


@always_inline
fn _utf8_first_byte_sequence_length(b: UInt8) -> Int:
    """Get the length of the sequence starting with given byte. Do note that
    this does not work correctly if given a continuation byte."""
    debug_assert(
        (b & 0b1100_0000) != 0b1000_0000,
        (
            "Function `_utf8_first_byte_sequence_length()` does not work"
            " correctly if given a continuation byte."
        ),
    )
    var flipped = ~b
    return int(count_leading_zeros(flipped) + (flipped >> 7))


@always_inline("nodebug")
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
    origin: Origin[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for StringSlice

    Parameters:
        is_mutable: Whether the slice is mutable.
        origin: The origin of the underlying string data.
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
        alias S = Span[UInt8, StaticConstantOrigin]
        var s = S(unsafe_ptr=self.ptr, len=self.length)
        self.continuation_bytes = _count_utf8_continuation_bytes(s)

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> StringSlice[origin]:
        @parameter
        if forward:
            var byte_len = 1
            if self.continuation_bytes > 0:
                var byte_type = _utf8_byte_type(self.ptr[self.index])
                if byte_type != 0:
                    byte_len = int(byte_type)
                    self.continuation_bytes -= byte_len - 1
            self.index += byte_len
            return StringSlice[origin](
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
            return StringSlice[origin](
                unsafe_from_utf8_ptr=self.ptr + self.index, len=byte_len
            )

    @always_inline
    fn __hasmore__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index - self.continuation_bytes
        else:
            return self.index - self.continuation_bytes


@value
struct StringSlice[is_mutable: Bool, //, origin: Origin[is_mutable].type,](
    Stringable,
    Sized,
    Formattable,
    CollectionElement,
    CollectionElementNew,
    Stringlike,
    AsBytesWrite,
):
    """A non-owning view to encoded string data.

    TODO:
    The underlying string data is guaranteed to be encoded using UTF-8.

    Parameters:
        is_mutable: Whether the slice is mutable.
        origin: The origin of the underlying string data.
    """

    var _slice: Span[UInt8, origin]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(
        inout self: StringSlice[StaticConstantOrigin], lit: StringLiteral
    ):
        """Construct a new string slice from a string literal.

        Args:
            lit: The literal to construct this string slice from.
        """
        # Since a StringLiteral has static origin, it will outlive
        # whatever arbitrary `origin` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StringLiteral is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        # TODO(#933): use when llvm intrinsics can be used at compile time
        # debug_assert(
        #     _is_valid_utf8(literal.unsafe_ptr(), literal.byte_length()),
        #     "StringLiteral doesn't have valid UTF-8 encoding",
        # )
        self = StaticString(
            unsafe_from_utf8_ptr=lit.unsafe_ptr(), len=lit.byte_length()
        )

    @always_inline
    fn __init__(inout self, *, owned unsafe_from_utf8: Span[UInt8, origin]):
        """Construct a new StringSlice from a sequence of UTF-8 encoded bytes.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8: A slice of bytes encoded in UTF-8.
        """

        self._slice = unsafe_from_utf8^

    fn __init__(inout self, *, unsafe_from_utf8_strref: StringRef):
        """Construct a new StringSlice from a StringRef pointing to UTF-8
        encoded bytes.

        Safety:
            - `unsafe_from_utf8_strref` MUST point to data that is valid for
              `origin`.
            - `unsafe_from_utf8_strref` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8_strref: A StringRef of bytes encoded in UTF-8.
        """
        var strref = unsafe_from_utf8_strref

        var byte_slice = Span[UInt8, origin](
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
        """Construct a StringSlice from a pointer to a sequence of UTF-8 encoded
        bytes and a length.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to at least `len` bytes of valid
              UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` must point to data that is live for the
              duration of `origin`.

        Args:
            unsafe_from_utf8_ptr: A pointer to a sequence of bytes encoded in
              UTF-8.
            len: The number of bytes of encoded data.
        """
        var byte_slice = Span[UInt8, origin](
            unsafe_ptr=unsafe_from_utf8_ptr,
            len=len,
        )

        self._slice = byte_slice

    @always_inline
    fn __init__(inout self, *, other: Self):
        """Explicitly construct a deep copy of the provided `StringSlice`.

        Args:
            other: The `StringSlice` to copy.
        """
        self._slice = other._slice

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
        var b_len = self.byte_length()
        alias S = Span[UInt8, StaticConstantOrigin]
        var s = S(unsafe_ptr=self.unsafe_ptr(), len=b_len)
        return b_len - _count_utf8_continuation_bytes(s)

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

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__(self, rhs: StringSlice) -> Bool:
        """Verify if a string slice is equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string slices are equal in length and contain the same
                elements, False otherwise.
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
            True if the string slice is equal to the input string in length and
                contain the same bytes, False otherwise.
        """
        return self == rhs.as_string_slice()

    @always_inline
    fn __eq__(self, rhs: StringLiteral) -> Bool:
        """Verify if a string slice is equal to a literal.

        Args:
            rhs: The literal to compare against.

        Returns:
            True if the string slice is equal to the input literal in length and
                contain the same bytes, False otherwise.
        """
        return self == rhs.as_string_slice()

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline
    fn __ne__(self, rhs: StringSlice) -> Bool:
        """Verify if span is not equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string slices are not equal in length or contents, False
                otherwise.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: String) -> Bool:
        """Verify if span is not equal to another string slice.

        Args:
            rhs: The string slice to compare against.

        Returns:
            True if the string and slice are not equal in length or contents,
                False otherwise.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Verify if span is not equal to a literal.

        Args:
            rhs: The string literal to compare against.

        Returns:
            True if the slice is not equal to the literal in length or contents,
                False otherwise.
        """
        return not self == rhs

    fn __iter__(ref [_]self) -> _StringSliceIter[__origin_of(self)]:
        """Iterate over the string unicode characters.

        Returns:
            An iterator of references to the string unicode characters.
        """
        return _StringSliceIter[__origin_of(self)](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    fn __reversed__(ref [_]self) -> _StringSliceIter[__origin_of(self), False]:
        """Iterate backwards over the string unicode characters.

        Returns:
            A reversed iterator of references to the string unicode characters.
        """
        return _StringSliceIter[__origin_of(self), forward=False](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn as_bytes(ref [_]self) -> Span[UInt8, origin]:
        """Returns a contiguous slice of bytes.

        Returns:
            A contiguous slice pointing to bytes.

        Notes:
            This does not include the trailing null terminator.
        """
        return self._slice

    @always_inline
    fn as_bytes_write[O: MutableOrigin, //](ref [O]self) -> Span[UInt8, O]:
        """Returns a mutable contiguous slice of the bytes.

        Parameters:
            O: The Origin of the bytes.

        Returns:
            A mutable contiguous slice pointing to the bytes.

        Notes:
            This does not include the trailing null terminator.
        """

        return Span[UInt8, O](
            unsafe_ptr=self.unsafe_ptr(), len=self.byte_length()
        )

    @always_inline
    fn as_bytes_read[O: ImmutableOrigin, //](ref [O]self) -> Span[UInt8, O]:
        """Returns an immutable contiguous slice of the bytes.

        Parameters:
            O: The Origin of the bytes.

        Returns:
            An immutable contiguous slice pointing to the bytes.

        Notes:
            This does not include the trailing null terminator.
        """

        return Span[UInt8, O](
            unsafe_ptr=self.unsafe_ptr(), len=self.byte_length()
        )

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

        return len(self.as_bytes())

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
        """Gets the `StringSlice` pointing to the substring after the specified
        slice start position.

        If start is negative, it is interpreted as the number of characters
        from the end of the string to start at.

        Args:
            start: Starting index of the slice.

        Returns:
            A `StringSlice` borrowed from the current string containing the
            characters of the slice starting at start.
        """

        var self_len = self.byte_length()

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

    fn find[T: Stringlike, //](self, substr: T, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Parameters:
            T: The type of the substring.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        var sub = substr.as_bytes_read()
        var sub_len = len(sub)
        if sub_len == 0:
            return 0

        var s_span = self.as_bytes_read()
        if len(s_span) < sub_len + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack = self._from_start(start).as_bytes_read()

        var loc = stringref._memmem(
            haystack.unsafe_ptr(), len(haystack), sub.unsafe_ptr(), sub_len
        )

        if not loc:
            return -1

        return int(loc) - int(s_span.unsafe_ptr())

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

    @always_inline
    fn split[
        T: Stringlike, O: ImmutableOrigin, //
    ](ref [O]self, sep: T, maxsplit: Int) -> List[StringSlice[O]]:
        """Split the string by a separator.

        Parameters:
            T: The type of the separator.
            O: The origin of the stringslice.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting with maxsplit
        _ = "1,2,3".split(",", maxsplit=1) # ['1', '2,3']
        # Splitting with starting or ending separators
        _ = ",1,2,3,".split(",", maxsplit=1) # ['', '1,2,3,']
        _ = "123".split("", maxsplit=1) # ['', '123']
        ```
        .
        """
        return _split_sl[has_maxsplit=True, has_sep=True](self, sep, maxsplit)

    @always_inline
    fn split[
        T: Stringlike, O: ImmutableOrigin, //
    ](ref [O]self, sep: T) -> List[StringSlice[O]]:
        """Split the string by a separator.

        Parameters:
            T: The type of the separator.
            O: The origin of the stringslice.

        Args:
            sep: The string to split on.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting a space
        _ = "hello world".split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = "hello,,world".split(",") # ["hello", "", "world"]
        # Splitting with starting or ending separators
        _ = ",1,2,3,".split(",") # ['', '1', '2', '3', '']
        _ = "123".split("") # ['', '1', '2', '3', '']
        ```
        .
        """
        return _split_sl[has_maxsplit=False, has_sep=True](self, sep, -1)

    @always_inline
    fn split[
        O: ImmutableOrigin, //
    ](ref [O]self, *, maxsplit: Int) -> List[StringSlice[O]]:
        """Split the string by every Whitespace separator.

        Parameters:
            O: The origin of the stringslice.

        Args:
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting with maxsplit
        _ = "1     2  3".split(maxsplit=1) # ['1', '2  3']
        ```
        .
        """
        return _split_sl[has_maxsplit=True, has_sep=False](self, None, maxsplit)

    @always_inline
    fn split[
        O: ImmutableOrigin, //
    ](ref [O]self, sep: NoneType = None) -> List[StringSlice[O]]:
        """Split the string by every Whitespace separator.

        Parameters:
            O: The origin of the stringslice.

        Args:
            sep: None.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = "      ".split() # []
        _ = "".split() # []
        # Splitting a string with leading, trailing, and middle whitespaces
        _ = "      hello    world     ".split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = (
            "hello \\t\\n\\r\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029world"
        ).split()  # ["hello", "world"]
        ```
        .
        """
        return _split_sl[has_maxsplit=False, has_sep=False](self, sep, -1)

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


# ===----------------------------------------------------------------------===#
# Utils
# ===----------------------------------------------------------------------===#


trait Stringlike(AsBytesRead, CollectionElement, CollectionElementNew):
    """Trait intended to be used only with `String`, `StringLiteral` and
    `StringSlice`."""

    fn find[T: Stringlike, //](self, substr: T, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Parameters:
            T: The type of the substring.

        Args:
            substr: The substring to find.
            start: The offset from which to find.

        Returns:
            The offset of `substr` relative to the beginning of the string.
        """
        ...

    fn __iter__(ref [_]self) -> _StringSliceIter[__origin_of(self)]:
        """Iterate over the string unicode characters.

        Returns:
            An iterator of references to the string unicode characters.
        """
        ...


fn _split[
    T0: Stringlike, //,
    has_maxsplit: Bool,
    has_sep: Bool,
    T1: Stringlike = StringLiteral,
](src_str: T0, sep: Optional[T1], maxsplit: Int) -> List[String]:
    var items: List[Span[Byte, __origin_of(src_str)]]

    @parameter
    if has_sep:
        items = _split_impl[has_maxsplit](src_str, sep.value(), maxsplit)
    else:
        items = _split_impl[has_maxsplit](src_str, maxsplit)
    var output = List[String](capacity=len(items))
    # TODO: some fancy custom allocator since we know the exact length of each
    for item in items:
        output.append(String(StringSlice(unsafe_from_utf8=item[])))
    return output^


fn _split_sl[
    O: ImmutableOrigin, //,
    has_maxsplit: Bool,
    has_sep: Bool,
    T: Stringlike = StringLiteral,
](ref [O]src_str: StringSlice, sep: Optional[T], maxsplit: Int) -> List[
    StringSlice[O]
]:
    var items: List[Span[Byte, O]]

    @parameter
    if has_sep:
        items = _split_impl[has_maxsplit](src_str, sep.value(), maxsplit)
    else:
        items = _split_impl[has_maxsplit](src_str, maxsplit)
    var output = List[StringSlice[O]](capacity=len(items))
    for item in items:
        output.append(StringSlice(unsafe_from_utf8=item[]))
    return output^


fn _split_impl[
    T0: Stringlike,
    T1: Stringlike,
    O: ImmutableOrigin, //,
    has_maxsplit: Bool,
](ref [O]src_str: T0, sep: T1, maxsplit: Int) -> List[Span[Byte, O]] as output:
    var sep_len = len(sep.as_bytes_read())
    if sep_len == 0:
        var iterator = src_str.__iter__()
        output = __type_of(output)(capacity=len(iterator) + 2)
        output.append(rebind[Span[Byte, O]]("".as_bytes_read()))
        for s in iterator:
            output.append(rebind[Span[Byte, O]](s.as_bytes_read()))
        output.append(rebind[Span[Byte, O]]("".as_bytes_read()))
        return

    alias prealloc = 16  # guessing, Python's implementation uses 12
    var amnt = prealloc

    @parameter
    if has_maxsplit:
        amnt = maxsplit + 1 if maxsplit < prealloc else prealloc
    output = __type_of(output)(capacity=amnt)
    var str_byte_len = len(src_str.as_bytes_read())
    var lhs = 0
    var rhs = 0
    var items = 0
    var ptr = src_str.as_bytes_read().unsafe_ptr()
    # var str_span = src_str.as_bytes_read() # FIXME(#3295)
    # var sep_span = sep.as_bytes_read() # FIXME(#3295)

    while lhs <= str_byte_len:
        rhs = src_str.find(sep, lhs)  # FIXME(#3295): use str_span and sep_span
        rhs += int(rhs == -1) * (str_byte_len + 1)  # if not found go to end

        @parameter
        if has_maxsplit:
            rhs += int(items == maxsplit) * (str_byte_len - rhs)
            items += 1

        output.append(Span[Byte, O](unsafe_ptr=ptr + lhs, len=rhs - lhs))
        lhs = rhs + sep_len


fn _split_impl[
    T: Stringlike, O: ImmutableOrigin, //, has_maxsplit: Bool
](ref [O]src_str: T, maxsplit: Int) -> List[Span[Byte, O]] as output:
    alias prealloc = 16  # guessing, Python's implementation uses 12
    var amnt = prealloc

    @parameter
    if has_maxsplit:
        amnt = maxsplit + 1 if maxsplit < prealloc else prealloc
    output = __type_of(output)(capacity=amnt)
    var str_byte_len = len(src_str.as_bytes_read())
    var lhs = 0
    var rhs = 0
    var items = 0
    var ptr = src_str.as_bytes_read().unsafe_ptr()
    alias S = StringSlice[StaticConstantOrigin]

    @always_inline("nodebug")
    fn _build_slice(p: UnsafePointer[Byte], start: Int, end: Int) -> S:
        return S(unsafe_from_utf8_ptr=p + start, len=end - start)

    while lhs <= str_byte_len:
        # Python adds all "whitespace chars" as one separator
        # if no separator was specified
        for s in _build_slice(ptr, lhs, str_byte_len):
            if not s.isspace():
                break
            lhs += s.byte_length()
        # if it went until the end of the String, then it should be sliced
        # until the start of the whitespace which was already appended
        if lhs == str_byte_len:
            break
        rhs = lhs + _utf8_first_byte_sequence_length(ptr[lhs])
        for s in _build_slice(ptr, rhs, str_byte_len):
            if s.isspace():
                break
            rhs += s.byte_length()

        @parameter
        if has_maxsplit:
            rhs += int(items == maxsplit) * (str_byte_len - rhs)
            items += 1

        output.append(Span[Byte, O](unsafe_ptr=ptr + lhs, len=rhs - lhs))
        lhs = rhs
