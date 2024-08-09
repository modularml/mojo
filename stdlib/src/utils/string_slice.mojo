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
from sys import simdwidthof

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


fn _validate_utf8_simd_slice[
    width: Int, remainder: Bool = False
](ptr: UnsafePointer[UInt8], length: Int, owned iter_len: Int) -> Int:
    """Internal method to validate utf8, use _is_valid_utf8.

    Parameters:
        width: The width of the SIMD vector to build for validation.
        remainder: Whether it is computing the remainder that doesn't fit in the
            SIMD vector.

    Args:
        ptr: Pointer to the data.
        length: The length of the items in the pointer.
        iter_len: The amount of items to still iterate through.

    Returns:
        The new amount of items to iterate through that don't fit in the
            specified width of SIMD vector. If -1 then it is invalid.
    """
    # TODO: implement a faster algorithm like https://github.com/cyb70289/utf8
    # and benchmark the difference.
    var idx = length - iter_len
    while iter_len >= width or remainder:
        var d: SIMD[DType.uint8, width]  # use a vector of the specified width

        @parameter
        if not remainder:
            d = ptr.offset(idx).simd_strided_load[DType.uint8, width](1)
        else:
            debug_assert(iter_len > -1, "iter_len must be > -1")
            d = SIMD[DType.uint8, width](0)
            for i in range(iter_len):
                d[i] = ptr[idx + i]

        var is_ascii = d < 0b1000_0000
        if is_ascii.reduce_and():  # skip all ASCII bytes

            @parameter
            if not remainder:
                idx += width
                iter_len -= width
                continue
            else:
                return 0
        elif is_ascii[0]:
            for i in range(1, width):
                if is_ascii[i]:
                    continue
                idx += i
                iter_len -= i
                break
            continue

        var byte_types = _utf8_byte_type(d)
        var first_byte_type = byte_types[0]

        # byte_type has to match against the amount of continuation bytes
        alias Vec = SIMD[DType.uint8, 4]
        alias n4_byte_types = Vec(4, 1, 1, 1)
        alias n3_byte_types = Vec(3, 1, 1, 0)
        alias n3_mask = Vec(0b111, 0b111, 0b111, 0)
        alias n2_byte_types = Vec(2, 1, 0, 0)
        alias n2_mask = Vec(0b111, 0b111, 0, 0)
        var byte_types_4 = byte_types.slice[4]()
        var valid_n4 = (byte_types_4 == n4_byte_types).reduce_and()
        var valid_n3 = ((byte_types_4 & n3_mask) == n3_byte_types).reduce_and()
        var valid_n2 = ((byte_types_4 & n2_mask) == n2_byte_types).reduce_and()
        if not (valid_n4 or valid_n3 or valid_n2):
            return -1

        # special unicode ranges
        var b0 = d[0]
        var b1 = d[1]
        if first_byte_type == 2 and b0 < UInt8(0b1100_0010):
            return -1
        elif b0 == 0xE0 and not (UInt8(0xA0) <= b1 <= UInt8(0xBF)):
            return -1
        elif b0 == 0xED and not (UInt8(0x80) <= b1 <= UInt8(0x9F)):
            return -1
        elif b0 == 0xF0 and not (UInt8(0x90) <= b1 <= UInt8(0xBF)):
            return -1
        elif b0 == 0xF4 and not (UInt8(0x80) <= b1 <= UInt8(0x8F)):
            return -1

        # amount of bytes evaluated
        idx += int(first_byte_type)
        iter_len -= int(first_byte_type)

        @parameter
        if remainder:
            break
    return iter_len


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

    var iter_len = length
    if iter_len >= 64 and simdwidthof[DType.uint8]() >= 64:
        iter_len = _validate_utf8_simd_slice[64](ptr, length, iter_len)
        if iter_len < 0:
            return False
    if iter_len >= 32 and simdwidthof[DType.uint8]() >= 32:
        iter_len = _validate_utf8_simd_slice[32](ptr, length, iter_len)
        if iter_len < 0:
            return False
    if iter_len >= 16 and simdwidthof[DType.uint8]() >= 16:
        iter_len = _validate_utf8_simd_slice[16](ptr, length, iter_len)
        if iter_len < 0:
            return False
    if iter_len >= 8:
        iter_len = _validate_utf8_simd_slice[8](ptr, length, iter_len)
        if iter_len < 0:
            return False
    if iter_len >= 4:
        iter_len = _validate_utf8_simd_slice[4](ptr, length, iter_len)
        if iter_len < 0:
            return False
    return _validate_utf8_simd_slice[4, True](ptr, length, iter_len) == 0


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
