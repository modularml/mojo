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

You can import these APIs from the `utils.string_slice` module.

Examples:

```mojo
from utils import StringSlice
```
"""

from bit import count_leading_zeros
from builtin.builtin_list import _lit_mut_cast
from utils import Span
from collections.string import _isspace, _atol, _atof
from collections import List, Optional
from memory import memcmp, UnsafePointer, memcpy
from sys import simdwidthof, bitwidthof
from sys.intrinsics import unlikely
from memory.memory import _memcmp_impl_unconstrained
from ._utf8_validation import _is_valid_utf8

alias StaticString = StringSlice[StaticConstantOrigin]
"""An immutable static string slice."""


fn _count_utf8_continuation_bytes(span: Span[Byte]) -> Int:
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


fn _unicode_codepoint_utf8_byte_length(c: Int) -> Int:
    debug_assert(
        0 <= c <= 0x10FFFF, "Value: ", c, " is not a valid Unicode code point"
    )
    alias sizes = SIMD[DType.int32, 4](0, 0b0111_1111, 0b0111_1111_1111, 0xFFFF)
    return int((sizes < c).cast[DType.uint8]().reduce_add())


@always_inline
fn _utf8_first_byte_sequence_length(b: Byte) -> Int:
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


@always_inline
fn _memrchr[
    type: DType
](
    source: UnsafePointer[Scalar[type]], char: Scalar[type], len: Int
) -> UnsafePointer[Scalar[type]]:
    if not len:
        return UnsafePointer[Scalar[type]]()
    for i in reversed(range(len)):
        if source[i] == char:
            return source + i
    return UnsafePointer[Scalar[type]]()


@always_inline
fn _memrmem[
    type: DType
](
    haystack: UnsafePointer[Scalar[type]],
    haystack_len: Int,
    needle: UnsafePointer[Scalar[type]],
    needle_len: Int,
) -> UnsafePointer[Scalar[type]]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return UnsafePointer[Scalar[type]]()
    if needle_len == 1:
        return _memrchr[type](haystack, needle[0], haystack_len)
    for i in reversed(range(haystack_len - needle_len + 1)):
        if haystack[i] != needle[0]:
            continue
        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i
    return UnsafePointer[Scalar[type]]()


@value
struct _StringSliceIter[
    is_mutable: Bool, //,
    origin: Origin[is_mutable].type,
    forward: Bool = True,
]:
    """Iterator for `StringSlice` over unicode characters.

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
        alias S = Span[Byte, StaticConstantOrigin]
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
    Writable,
    CollectionElement,
    CollectionElementNew,
    Hashable,
):
    """A non-owning view to encoded string data.

    Parameters:
        is_mutable: Whether the slice is mutable.
        origin: The origin of the underlying string data.

    Notes:
        TODO: The underlying string data is guaranteed to be encoded using
        UTF-8.
    """

    var _slice: Span[Byte, origin]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(inout self: StaticString, lit: StringLiteral):
        """Construct a new `StringSlice` from a `StringLiteral`.

        Args:
            lit: The literal to construct this `StringSlice` from.
        """
        # Since a StringLiteral has static origin, it will outlive
        # whatever arbitrary `origin` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StringLiteral is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        # FIXME: this gets practically stuck at compile time
        # debug_assert(
        #     _is_valid_utf8(lit.as_bytes()),
        #     "StringLiteral doesn't have valid UTF-8 encoding",
        # )
        self = StaticString(unsafe_from_utf8=lit.as_bytes())

    @always_inline
    fn __init__(inout self, *, owned unsafe_from_utf8: Span[Byte, origin]):
        """Construct a new `StringSlice` from a sequence of UTF-8 encoded bytes.

        Args:
            unsafe_from_utf8: A `Span[Byte]` encoded in UTF-8.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.
        """

        self._slice = unsafe_from_utf8^

    fn __init__(inout self, *, unsafe_from_utf8_strref: StringRef):
        """Construct a new StringSlice from a `StringRef` pointing to UTF-8
        encoded bytes.

        Args:
            unsafe_from_utf8_strref: A `StringRef` of bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_strref` MUST point to data that is valid for
              `origin`.
            - `unsafe_from_utf8_strref` MUST be valid UTF-8 encoded data.
        """

        var strref = unsafe_from_utf8_strref

        var byte_slice = Span[Byte, origin](
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
        """Construct a `StringSlice` from a pointer to a sequence of UTF-8
        encoded bytes and a length.

        Args:
            unsafe_from_utf8_ptr: A pointer to a sequence of bytes encoded in
              UTF-8.
            len: The number of bytes of encoded data.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to at least `len` bytes of valid
              UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` must point to data that is live for the
              duration of `origin`.
        """
        var byte_slice = Span[Byte, origin](
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

    fn __init__[
        O: ImmutableOrigin, //
    ](inout self: StringSlice[O], ref [O]value: String):
        """Construct an immutable StringSlice.

        Parameters:
            O: The immutable origin.

        Args:
            value: The string value.
        """

        debug_assert(
            _is_valid_utf8(value.as_bytes()), "value is not valid utf8"
        )
        self = StringSlice[O](unsafe_from_utf8=value.as_bytes())

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
        alias S = Span[Byte, StaticConstantOrigin]
        var s = S(unsafe_ptr=self.unsafe_ptr(), len=b_len)
        return b_len - _count_utf8_continuation_bytes(s)

    fn write_to[W: Writer](self, inout writer: W):
        """Formats this string slice to the provided `Writer`.

        Parameters:
            W: A type conforming to the `Writable` trait.

        Args:
            writer: The object to write to.
        """
        writer.write_bytes(self.as_bytes())

    fn __bool__(self) -> Bool:
        """Check if a string slice is non-empty.

        Returns:
           True if a string slice is non-empty, False otherwise.
        """
        return len(self._slice) > 0

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self._slice._data, self._slice._len)

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__(self, rhs: StringSlice) -> Bool:
        """Verify if a `StringSlice` is equal to another `StringSlice`.

        Args:
            rhs: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is equal to the input in length and contents.
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
        """Verify if a `StringSlice` is equal to a string.

        Args:
            rhs: The `String` to compare against.

        Returns:
            If the `StringSlice` is equal to the input in length and contents.
        """
        return self == rhs.as_string_slice()

    @always_inline
    fn __eq__(self, rhs: StringLiteral) -> Bool:
        """Verify if a `StringSlice` is equal to a literal.

        Args:
            rhs: The `StringLiteral` to compare against.

        Returns:
            If the `StringSlice` is equal to the input in length and contents.
        """
        return self == rhs.as_string_slice()

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline
    fn __ne__(self, rhs: StringSlice) -> Bool:
        """Verify if span is not equal to another `StringSlice`.

        Args:
            rhs: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is not equal to the input in length and
            contents.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: String) -> Bool:
        """Verify if span is not equal to another `StringSlice`.

        Args:
            rhs: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is not equal to the input in length and
            contents.
        """
        return not self == rhs

    @always_inline
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Verify if span is not equal to a `StringLiteral`.

        Args:
            rhs: The `StringLiteral` to compare against.

        Returns:
            If the `StringSlice` is not equal to the input in length and
            contents.
        """
        return not self == rhs

    @always_inline
    fn __lt__(self, rhs: StringSlice) -> Bool:
        """Verify if the `StringSlice` bytes are strictly less than the input in
        overlapping content.

        Args:
            rhs: The other `StringSlice` to compare against.

        Returns:
            If the `StringSlice` bytes are strictly less than the input in
            overlapping content.
        """
        var len1 = len(self)
        var len2 = len(rhs)
        return int(len1 < len2) > _memcmp_impl_unconstrained(
            self.unsafe_ptr(), rhs.unsafe_ptr(), min(len1, len2)
        )

    fn __iter__(self) -> _StringSliceIter[origin]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return _StringSliceIter[origin](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    fn __reversed__(self) -> _StringSliceIter[origin, False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return _StringSliceIter[origin, forward=False](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    fn __getitem__[IndexerType: Indexer](self, idx: IndexerType) -> String:
        """Gets the character at the specified position.

        Parameters:
            IndexerType: The inferred type of an indexer argument.

        Args:
            idx: The index value.

        Returns:
            A new string containing the character at the specified position.
        """
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time
        var buf = String._buffer_type(capacity=1)
        buf.append(self._slice[idx])
        buf.append(0)
        return String(buf^)

    fn __contains__(ref [_]self, substr: StringSlice[_]) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return self.find(substr) != -1

    @always_inline
    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return _atol(self)

    @always_inline
    fn __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.
        """
        return _atof(self)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn strip(self) -> StringSlice[origin]:
        """Gets a StringRef with leading and trailing whitespaces removed.
        This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A StringRef with leading and trailing whitespaces removed.

        Examples:

        ```mojo
        print("  mojo  ".strip()) # "mojo"
        ```
        .
        """
        # FIXME: this can already do full isspace support with iterator
        var start: Int = 0
        var end: Int = len(self)
        var ptr = self.unsafe_ptr()
        while start < end and _isspace(ptr[start]):
            start += 1
        while end > start and _isspace(ptr[end - 1]):
            end -= 1
        return StringSlice[origin](
            unsafe_from_utf8_ptr=ptr + start, len=end - start
        )

    @always_inline
    fn as_bytes(self) -> Span[Byte, origin]:
        """Get the sequence of encoded bytes of the underlying string.

        Returns:
            A slice containing the underlying sequence of encoded bytes.
        """
        return self._slice

    @always_inline
    fn unsafe_ptr[
        is_mutable: Bool = Self.is_mutable,
        origin: Origin[is_mutable]
        .type = _lit_mut_cast[Self.origin, is_mutable]
        .result,
    ](self) -> UnsafePointer[Byte, is_mutable=is_mutable, origin=origin]:
        """Gets a pointer to the first element of this string slice.

        Returns:
            A pointer pointing at the first element of this string slice.
        """
        return self._slice.unsafe_ptr[is_mutable, origin]()

    @always_inline
    fn byte_length(self) -> Int:
        """Get the length of this string slice in bytes.

        Returns:
            The length of this string slice in bytes.
        """

        return len(self.as_bytes())

    fn startswith(
        self, prefix: StringSlice[_], start: Int = 0, end: Int = -1
    ) -> Bool:
        """Verify if the `StringSlice` starts with the specified prefix between
        start and end positions.

        Args:
            prefix: The prefix to check.
            start: The start offset from which to check.
            end: The end offset from which to check.

        Returns:
            True if the `self[start:end]` is prefixed by the input prefix.
        """
        if end == -1:
            return self.find(prefix, start) == start
        return StringSlice[__origin_of(self)](
            unsafe_from_utf8_ptr=self.unsafe_ptr() + start, len=end - start
        ).startswith(prefix)

    fn endswith(
        self, suffix: StringSlice[_], start: Int = 0, end: Int = -1
    ) -> Bool:
        """Verify if the `StringSlice` end with the specified suffix between
        start and end positions.

        Args:
            suffix: The suffix to check.
            start: The start offset from which to check.
            end: The end offset from which to check.

        Returns:
            True if the `self[start:end]` is suffixed by the input suffix.
        """
        if len(suffix) > len(self):
            return False
        if end == -1:
            return self.rfind(suffix, start) + len(suffix) == len(self)
        return StringSlice[__origin_of(self)](
            unsafe_from_utf8_ptr=self.unsafe_ptr() + start, len=end - start
        ).endswith(suffix)

    fn _from_start(self, start: Int) -> Self:
        """Gets the `StringSlice` pointing to the substring after the specified
        slice start position. If start is negative, it is interpreted as the
        number of characters from the end of the string to start at.

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

    @always_inline
    fn format[*Ts: _CurlyEntryFormattable](self, *args: *Ts) raises -> String:
        """Format a template with `*args`.

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Representable`
                and `Stringable` (to be changed and made more flexible).

        Returns:
            The template with the given values substituted.

        Examples:

        ```mojo
        # Manual indexing:
        print("{0} {1} {0}".format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print("{} {}".format(True, "hello world")) # True hello world
        ```
        .
        """
        return _FormatCurlyEntry.format(self, args)

    fn find(ref [_]self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns `-1`.

        Args:
            substr: The substring to find.
            start: The offset from which to find.

        Returns:
            The offset of `substr` relative to the beginning of the string.
        """
        if not substr:
            return 0

        if self.byte_length() < substr.byte_length() + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._from_start(start)

        var loc = stringref._memmem(
            haystack_str.unsafe_ptr(),
            haystack_str.byte_length(),
            substr.unsafe_ptr(),
            substr.byte_length(),
        )

        if not loc:
            return -1

        return int(loc) - int(self.unsafe_ptr())

    fn rfind(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns `-1`.

        Args:
            substr: The substring to find.
            start: The offset from which to find.

        Returns:
            The offset of `substr` relative to the beginning of the string.
        """
        if not substr:
            return len(self)

        if len(self) < len(substr) + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._from_start(start)

        var loc = _memrmem(
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
        [universal separators:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

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

    fn isnewline[single_character: Bool = False](self) -> Bool:
        """Determines whether every character in the given StringSlice is a
        python newline character. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Parameters:
            single_character: Whether to evaluate the stringslice as a single
                unicode character (avoids overhead when already iterating).

        Returns:
            True if the whole StringSlice is made up of whitespace characters
                listed above, otherwise False.
        """

        fn _is_newline_char(s: StringSlice) -> Bool:
            # sorry for readability, but this has less overhead than memcmp
            # highly performance sensitive code, benchmark before touching
            alias `\t` = UInt8(ord("\t"))
            alias `\r` = UInt8(ord("\r"))
            alias `\n` = UInt8(ord("\n"))
            alias `\x1c` = UInt8(ord("\x1c"))
            alias `\x1e` = UInt8(ord("\x1e"))
            no_null_len = s.byte_length()
            ptr = s.unsafe_ptr()
            if no_null_len == 1:
                v = ptr[0]
                return `\t` <= v <= `\x1e` and not (`\r` < v < `\x1c`)
            elif no_null_len == 2:
                v0 = ptr[0]
                v1 = ptr[1]
                next_line = v0 == 0xC2 and v1 == 0x85  # next line: \x85
                r_n = v0 == `\r` and v1 == `\n`
                return next_line or r_n
            elif no_null_len == 3:
                # unicode line sep or paragraph sep: \u2028 , \u2029
                v2 = ptr[2]
                lastbyte = v2 == 0xA8 or v2 == 0xA9
                return ptr[0] == 0xE2 and ptr[1] == 0x80 and lastbyte
            return False

        @parameter
        if single_character:
            return _is_newline_char(self)
        else:
            for s in self:
                if not _is_newline_char(s):
                    return False
            return self.byte_length() != 0

    fn splitlines[
        O: ImmutableOrigin, //
    ](self: StringSlice[O], keepends: Bool = False) -> List[StringSlice[O]]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Parameters:
            O: The immutable origin.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """

        alias `\r` = UInt8(ord("\r"))
        alias `\n` = UInt8(ord("\n"))
        alias `\t` = UInt8(ord("\t"))
        alias `\x1c` = UInt8(ord("\x1c"))
        alias `\x1e` = UInt8(ord("\x1e"))
        output = List[StringSlice[O]](capacity=128)  # guessing
        ptr = self.unsafe_ptr()
        length = self.byte_length()
        offset = 0

        @always_inline
        @parameter
        fn _is_newline_char(p: UnsafePointer[Byte], l: Int, b0: Byte) -> Bool:
            # sorry for readability, but this has less overhead than memcmp
            # highly performance sensitive code, benchmark before touching
            if l == 1:
                return `\t` <= b0 <= `\x1e` and not (`\r` < b0 < `\x1c`)
            elif l == 2:
                return b0 == 0xC2 and p[1] == 0x85  # next line: \x85
            elif l == 3:
                # unicode line sep or paragraph sep: \u2028 , \u2029
                v2 = p[2]
                lastbyte = v2 == 0xA8 or v2 == 0xA9
                return b0 == 0xE2 and p[1] == 0x80 and lastbyte
            return False

        while offset < length:
            eol_start = offset
            eol_length = 0

            while eol_start < length:
                b0 = ptr[eol_start]
                char_len = _utf8_first_byte_sequence_length(b0)
                debug_assert(
                    eol_start + char_len <= length,
                    "corrupted sequence causing unsafe memory access",
                )
                isnewline = int(_is_newline_char(ptr + eol_start, char_len, b0))
                char_end = isnewline * (eol_start + char_len)
                next_idx = char_end * int(char_end < length)
                is_r_n = b0 == `\r` and next_idx != 0 and ptr[next_idx] == `\n`
                eol_length = isnewline * char_len + int(is_r_n)
                if unlikely(isnewline == 1):
                    break
                eol_start += char_len

            str_len = eol_start - offset + int(keepends) * eol_length
            s = StringSlice[O](unsafe_from_utf8_ptr=ptr + offset, len=str_len)
            output.append(s^)
            offset = eol_start + eol_length

        return output^


# ===----------------------------------------------------------------------===#
# Utils
# ===----------------------------------------------------------------------===#


trait Stringlike:
    """Trait intended to be used only with `String`, `StringLiteral` and
    `StringSlice`."""

    fn byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes.

        Notes:
            This does not include the trailing null terminator in the count.
        """
        ...

    fn unsafe_ptr[
        is_mutable: Bool, origin: Origin[is_mutable].type
    ](self) -> UnsafePointer[Byte, is_mutable=is_mutable, origin=origin]:
        """Get raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        ...


fn _to_string_list[
    T: CollectionElement, //,
    len_fn: fn (T) -> Int,
    unsafe_ptr_fn: fn (T) -> UnsafePointer[Byte],
](items: List[T]) -> List[String]:
    i_len = len(items)
    i_ptr = items.unsafe_ptr()
    out_ptr = UnsafePointer[String].alloc(i_len)

    for i in range(i_len):
        og_len = len_fn(i_ptr[i])
        f_len = og_len + 1  # null terminator
        p = UnsafePointer[Byte].alloc(f_len)
        og_ptr = unsafe_ptr_fn(i_ptr[i])
        memcpy(p, og_ptr, og_len)
        p[og_len] = 0  # null terminator
        buf = String._buffer_type(unsafe_pointer=p, size=f_len, capacity=f_len)
        (out_ptr + i).init_pointee_move(String(buf^))
    return List[String](unsafe_pointer=out_ptr, size=i_len, capacity=i_len)


@always_inline
fn _to_string_list[
    O: ImmutableOrigin, //
](items: List[StringSlice[O]]) -> List[String]:
    """Create a list of Strings **copying** the existing data.

    Parameters:
        O: The origin of the data.

    Args:
        items: The List of string slices.

    Returns:
        The list of created strings.
    """

    fn unsafe_ptr_fn(v: StringSlice[O]) -> UnsafePointer[Byte]:
        return v.unsafe_ptr()

    fn len_fn(v: StringSlice[O]) -> Int:
        return v.byte_length()

    return _to_string_list[len_fn, unsafe_ptr_fn](items)


@always_inline
fn _to_string_list[
    O: ImmutableOrigin, //
](items: List[Span[Byte, O]]) -> List[String]:
    """Create a list of Strings **copying** the existing data.

    Parameters:
        O: The origin of the data.

    Args:
        items: The List of Bytes.

    Returns:
        The list of created strings.
    """

    fn unsafe_ptr_fn(v: Span[Byte, O]) -> UnsafePointer[Byte]:
        return v.unsafe_ptr()

    fn len_fn(v: Span[Byte, O]) -> Int:
        return len(v)

    return _to_string_list[len_fn, unsafe_ptr_fn](items)


# ===----------------------------------------------------------------------===#
# Format method structures
# ===----------------------------------------------------------------------===#


trait _CurlyEntryFormattable(Stringable, Representable):
    """This trait is used by the `format()` method to support format specifiers.
    Currently, it is a composition of both `Stringable` and `Representable`
    traits i.e. a type to be formatted must implement both. In the future this
    will be less constrained.
    """

    ...


@value
struct _FormatCurlyEntry(CollectionElement, CollectionElementNew):
    """The struct that handles `Stringlike` formatting by curly braces entries.
    This is internal for the types: `String`, `StringLiteral` and `StringSlice`.
    """

    var first_curly: Int
    """The index of an opening brace around a substitution field."""
    var last_curly: Int
    """The index of a closing brace around a substitution field."""
    # TODO: ord("a") conversion flag not supported yet
    var conversion_flag: UInt8
    """The type of conversion for the entry: {ord("s"), ord("r")}."""
    var format_spec: Optional[_FormatSpec]
    """The format specifier."""
    # TODO: ord("a") conversion flag not supported yet
    alias supported_conversion_flags = SIMD[DType.uint8, 2](ord("s"), ord("r"))
    """Currently supported conversion flags: `__str__` and `__repr__`."""
    alias _FieldVariantType = Variant[String, Int, NoneType, Bool]
    """Purpose of the `Variant` `Self.field`:

    - `Int` for manual indexing: (value field contains `0`).
    - `NoneType` for automatic indexing: (value field contains `None`).
    - `String` for **kwargs indexing: (value field contains `foo`).
    - `Bool` for escaped curlies: (value field contains False for `{` or True
        for `}`).
    """
    var field: Self._FieldVariantType
    """Store the substitution field. See `Self._FieldVariantType` docstrings for
    more details."""
    alias _args_t = VariadicPack[element_trait=_CurlyEntryFormattable, *_]
    """Args types that are formattable by curly entry."""

    fn __init__(inout self, *, other: Self):
        self.first_curly = other.first_curly
        self.last_curly = other.last_curly
        self.conversion_flag = other.conversion_flag
        self.field = Self._FieldVariantType(other=other.field)
        self.format_spec = other.format_spec

    fn __init__(
        inout self,
        first_curly: Int,
        last_curly: Int,
        field: Self._FieldVariantType,
        conversion_flag: UInt8 = 0,
        format_spec: Optional[_FormatSpec] = None,
    ):
        self.first_curly = first_curly
        self.last_curly = last_curly
        self.field = field
        self.conversion_flag = conversion_flag
        self.format_spec = format_spec

    @always_inline
    fn is_escaped_brace(ref [_]self) -> Bool:
        return self.field.isa[Bool]()

    @always_inline
    fn is_kwargs_field(ref [_]self) -> Bool:
        return self.field.isa[String]()

    @always_inline
    fn is_automatic_indexing(ref [_]self) -> Bool:
        return self.field.isa[NoneType]()

    @always_inline
    fn is_manual_indexing(ref [_]self) -> Bool:
        return self.field.isa[Int]()

    @staticmethod
    fn format[T: Stringlike](fmt_src: T, args: Self._args_t) raises -> String:
        alias len_pos_args = __type_of(args).__len__()
        entries, size_estimation = Self._create_entries(fmt_src, len_pos_args)
        var fmt_len = fmt_src.byte_length()
        var buf = String._buffer_type(capacity=fmt_len + size_estimation)
        buf.size = 1
        buf.unsafe_set(0, 0)
        var res = String(buf^)
        var offset = 0
        var ptr = fmt_src.unsafe_ptr[False, __origin_of(fmt_src)]()
        alias S = StringSlice[StaticConstantOrigin]

        @always_inline("nodebug")
        fn _build_slice(p: UnsafePointer[UInt8], start: Int, end: Int) -> S:
            return S(unsafe_from_utf8_ptr=p + start, len=end - start)

        var auto_arg_index = 0
        for e in entries:
            debug_assert(offset < fmt_len, "offset >= fmt_src.byte_length()")
            res += _build_slice(ptr, offset, e[].first_curly)
            e[]._format_entry[len_pos_args](res, args, auto_arg_index)
            offset = e[].last_curly + 1

        res += _build_slice(ptr, offset, fmt_len)
        return res^

    @staticmethod
    fn _create_entries[
        T: Stringlike
    ](fmt_src: T, len_pos_args: Int) raises -> (List[Self], Int):
        """Returns a list of entries and its total estimated entry byte width.
        """
        var manual_indexing_count = 0
        var automatic_indexing_count = 0
        var raised_manual_index = Optional[Int](None)
        var raised_automatic_index = Optional[Int](None)
        var raised_kwarg_field = Optional[String](None)
        alias `}` = UInt8(ord("}"))
        alias `{` = UInt8(ord("{"))
        alias l_err = "there is a single curly { left unclosed or unescaped"
        alias r_err = "there is a single curly } left unclosed or unescaped"

        var entries = List[Self]()
        var start = Optional[Int](None)
        var skip_next = False
        var fmt_ptr = fmt_src.unsafe_ptr[False, __origin_of(fmt_src)]()
        var fmt_len = fmt_src.byte_length()
        var total_estimated_entry_byte_width = 0

        for i in range(fmt_len):
            if skip_next:
                skip_next = False
                continue
            if fmt_ptr[i] == `{`:
                if not start:
                    start = i
                    continue
                if i - start.value() != 1:
                    raise Error(l_err)
                # python escapes double curlies
                entries.append(Self(start.value(), i, field=False))
                start = None
                continue
            elif fmt_ptr[i] == `}`:
                if not start and (i + 1) < fmt_len:
                    # python escapes double curlies
                    if fmt_ptr[i + 1] == `}`:
                        entries.append(Self(i, i + 1, field=True))
                        total_estimated_entry_byte_width += 2
                        skip_next = True
                        continue
                elif not start:  # if it is not an escaped one, it is an error
                    raise Error(r_err)

                var start_value = start.value()
                var current_entry = Self(start_value, i, field=NoneType())

                if i - start_value != 1:
                    if current_entry._handle_field_and_break(
                        fmt_src,
                        len_pos_args,
                        i,
                        start_value,
                        automatic_indexing_count,
                        raised_automatic_index,
                        manual_indexing_count,
                        raised_manual_index,
                        raised_kwarg_field,
                        total_estimated_entry_byte_width,
                    ):
                        break
                else:  # automatic indexing
                    if automatic_indexing_count >= len_pos_args:
                        raised_automatic_index = automatic_indexing_count
                        break
                    automatic_indexing_count += 1
                    total_estimated_entry_byte_width += 8  # guessing
                entries.append(current_entry^)
                start = None

        if raised_automatic_index:
            raise Error("Automatic indexing require more args in *args")
        elif raised_kwarg_field:
            var val = raised_kwarg_field.value()
            raise Error("Index " + val + " not in kwargs")
        elif manual_indexing_count and automatic_indexing_count:
            raise Error("Cannot both use manual and automatic indexing")
        elif raised_manual_index:
            var val = str(raised_manual_index.value())
            raise Error("Index " + val + " not in *args")
        elif start:
            raise Error(l_err)
        return entries^, total_estimated_entry_byte_width

    fn _handle_field_and_break[
        T: Stringlike
    ](
        inout self,
        fmt_src: T,
        len_pos_args: Int,
        i: Int,
        start_value: Int,
        inout automatic_indexing_count: Int,
        inout raised_automatic_index: Optional[Int],
        inout manual_indexing_count: Int,
        inout raised_manual_index: Optional[Int],
        inout raised_kwarg_field: Optional[String],
        inout total_estimated_entry_byte_width: Int,
    ) raises -> Bool:
        alias S = StringSlice[StaticConstantOrigin]

        @always_inline("nodebug")
        fn _build_slice(p: UnsafePointer[UInt8], start: Int, end: Int) -> S:
            return S(unsafe_from_utf8_ptr=p + start, len=end - start)

        var field = _build_slice(
            fmt_src.unsafe_ptr[False, __origin_of(fmt_src)](),
            start_value + 1,
            i,
        )
        var field_ptr = field.unsafe_ptr()
        var field_len = i - (start_value + 1)
        var exclamation_index = -1
        var idx = 0
        while idx < field_len:
            if field_ptr[idx] == ord("!"):
                exclamation_index = idx
                break
            idx += 1
        var new_idx = exclamation_index + 1
        if exclamation_index != -1:
            if new_idx == field_len:
                raise Error("Empty conversion flag.")
            var conversion_flag = field_ptr[new_idx]
            if field_len - new_idx > 1 or (
                conversion_flag not in Self.supported_conversion_flags
            ):
                var f = String(_build_slice(field_ptr, new_idx, field_len))
                _ = field^
                raise Error('Conversion flag "' + f + '" not recognised.')
            self.conversion_flag = conversion_flag
            field = _build_slice(field_ptr, 0, exclamation_index)
        else:
            new_idx += 1

        var extra = int(new_idx < field_len)
        var fmt_field = _build_slice(field_ptr, new_idx + extra, field_len)
        self.format_spec = _FormatSpec.parse(fmt_field)
        var w = int(self.format_spec.value().width) if self.format_spec else 0
        # fully guessing the byte width here to be at least 8 bytes per entry
        # minus the length of the whole format specification
        total_estimated_entry_byte_width += 8 * int(w > 0) + w - (field_len + 2)

        if field.byte_length() == 0:
            # an empty field, so it's automatic indexing
            if automatic_indexing_count >= len_pos_args:
                raised_automatic_index = automatic_indexing_count
                return True
            automatic_indexing_count += 1
        else:
            try:
                # field is a number for manual indexing:
                var number = int(field)
                self.field = number
                if number >= len_pos_args or number < 0:
                    raised_manual_index = number
                    return True
                manual_indexing_count += 1
            except e:
                alias unexp = "Not the expected error from atol"
                debug_assert("not convertible to integer" in str(e), unexp)
                # field is a keyword for **kwargs:
                var f = str(field)
                self.field = f
                raised_kwarg_field = f
                return True
        return False

    fn _format_entry[
        len_pos_args: Int
    ](self, inout res: String, args: Self._args_t, inout auto_idx: Int) raises:
        # TODO(#3403 and/or #3252): this function should be able to use
        # Formatter syntax when the type implements it, since it will give great
        # performance benefits. This also needs to be able to check if the given
        # args[i] conforms to the trait needed by the conversion_flag to avoid
        # needing to constraint that every type needs to conform to every trait.
        alias `r` = UInt8(ord("r"))
        alias `s` = UInt8(ord("s"))
        # alias `a` = UInt8(ord("a")) # TODO

        @parameter
        fn _format(idx: Int) raises:
            @parameter
            for i in range(len_pos_args):
                if i == idx:
                    var type_impls_repr = True  # TODO
                    var type_impls_str = True  # TODO
                    var type_impls_formatter_repr = True  # TODO
                    var type_impls_formatter_str = True  # TODO
                    var flag = self.conversion_flag
                    var empty = flag == 0 and not self.format_spec

                    var data: String
                    if empty and type_impls_formatter_str:
                        data = str(args[i])  # TODO: use writer and return
                    elif empty and type_impls_str:
                        data = str(args[i])
                    elif flag == `s` and type_impls_formatter_str:
                        if empty:
                            # TODO: use writer and return
                            pass
                        data = str(args[i])
                    elif flag == `s` and type_impls_str:
                        data = str(args[i])
                    elif flag == `r` and type_impls_formatter_repr:
                        if empty:
                            # TODO: use writer and return
                            pass
                        data = repr(args[i])
                    elif flag == `r` and type_impls_repr:
                        data = repr(args[i])
                    elif self.format_spec:
                        self.format_spec.value().stringify(res, args[i])
                        return
                    else:
                        alias argnum = "Argument number: "
                        alias does_not = " does not implement the trait "
                        alias needed = "needed for conversion_flag: "
                        var flg = String(List[UInt8](flag, 0))
                        raise Error(argnum + str(i) + does_not + needed + flg)

                    if self.format_spec:
                        self.format_spec.value().format_string(res, data)
                    else:
                        res += data

        if self.is_escaped_brace():
            res += "}" if self.field[Bool] else "{"
        elif self.is_manual_indexing():
            _format(self.field[Int])
        elif self.is_automatic_indexing():
            _format(auto_idx)
            auto_idx += 1


@value
@register_passable("trivial")
struct _FormatSpec:
    """Store every field of the format specifier in a byte (e.g., ord("+") for
    sign). It is stored in a byte because every [format specifier](
    https://docs.python.org/3/library/string.html#formatspec) is an ASCII
    character.
    """

    var fill: UInt8
    """If a valid align value is specified, it can be preceded by a fill
    character that can be any character and defaults to a space if omitted.
    """
    var align: UInt8
    """The meaning of the various alignment options is as follows:

    | Option | Meaning|
    |:------:|:-------|
    |'<' | Forces the field to be left-aligned within the available space \
    (this is the default for most objects).|
    |'>' | Forces the field to be right-aligned within the available space \
    (this is the default for numbers).|
    |'=' | Forces the padding to be placed after the sign (if any) but before \
    the digits. This is used for printing fields in the form `+000000120`. This\
    alignment option is only valid for numeric types. It becomes the default\
    for numbers when `0` immediately precedes the field width.|
    |'^' | Forces the field to be centered within the available space.|
    """
    var sign: UInt8
    """The sign option is only valid for number types, and can be one of the
    following:

    | Option | Meaning|
    |:------:|:-------|
    |'+' | indicates that a sign should be used for both positive as well as\
    negative numbers.|
    |'-' | indicates that a sign should be used only for negative numbers (this\
    is the default behavior).|
    |space | indicates that a leading space should be used on positive numbers,\
    and a minus sign on negative numbers.|
    """
    var coerce_z: Bool
    """The 'z' option coerces negative zero floating-point values to positive
    zero after rounding to the format precision. This option is only valid for
    floating-point presentation types.
    """
    var alternate_form: Bool
    """The alternate form is defined differently for different types. This
    option is only valid for types that implement the trait `# TODO: define
    trait`. For integers, when binary, octal, or hexadecimal output is used,
    this option adds the respective prefix '0b', '0o', '0x', or '0X' to the
    output value. For float and complex the alternate form causes the result of
    the conversion to always contain a decimal-point character, even if no
    digits follow it.
    """
    var width: UInt8
    """A decimal integer defining the minimum total field width, including any
    prefixes, separators, and other formatting characters. If not specified,
    then the field width will be determined by the content. When no explicit
    alignment is given, preceding the width field by a zero ('0') character
    enables sign-aware zero-padding for numeric types. This is equivalent to a
    fill character of '0' with an alignment type of '='.
    """
    var grouping_option: UInt8
    """The ',' option signals the use of a comma for a thousands separator. For
    a locale aware separator, use the 'n' integer presentation type instead. The
    '_' option signals the use of an underscore for a thousands separator for
    floating-point presentation types and for integer presentation type 'd'. For
    integer presentation types 'b', 'o', 'x', and 'X', underscores will be
    inserted every 4 digits. For other presentation types, specifying this
    option is an error.
    """
    var precision: UInt8
    """The precision is a decimal integer indicating how many digits should be
    displayed after the decimal point for presentation types 'f' and 'F', or
    before and after the decimal point for presentation types 'g' or 'G'. For
    string presentation types the field indicates the maximum field size - in
    other words, how many characters will be used from the field content. The
    precision is not allowed for integer presentation types.
    """
    var type: UInt8
    """Determines how the data should be presented.

    The available integer presentation types are:

    | Option | Meaning|
    |:------:|:-------|
    |'b' |Binary format. Outputs the number in base 2.|
    |'c' |Character. Converts the integer to the corresponding unicode\
    character before printing.|
    |'d' |Decimal Integer. Outputs the number in base 10.|
    |'o' |Octal format. Outputs the number in base 8.|
    |'x' |Hex format. Outputs the number in base 16, using lower-case letters\
    for the digits above 9.|
    |'X' |Hex format. Outputs the number in base 16, using upper-case letters\
    for the digits above 9. In case '#' is specified, the prefix '0x' will be\
    upper-cased to '0X' as well.|
    |'n' |Number. This is the same as 'd', except that it uses the current\
    locale setting to insert the appropriate number separator characters.|
    |None | The same as 'd'.|

    In addition to the above presentation types, integers can be formatted with
    the floating-point presentation types listed below (except 'n' and None).
    When doing so, float() is used to convert the integer to a floating-point
    number before formatting.

    The available presentation types for float and Decimal values are:

    | Option | Meaning|
    |:------:|:-------|
    |'e' |Scientific notation. For a given precision p, formats the number in\
    scientific notation with the letter `e` separating the coefficient from the\
    exponent. The coefficient has one digit before and p digits after the\
    decimal point, for a total of p + 1 significant digits. With no precision\
    given, uses a precision of 6 digits after the decimal point for float, and\
    shows all coefficient digits for Decimal. If no digits follow the decimal\
    point, the decimal point is also removed unless the # option is used.|
    |'E' |Scientific notation. Same as 'e' except it uses an upper case `E` as\
    the separator character.|
    |'f' |Fixed-point notation. For a given precision p, formats the number as\
    a decimal number with exactly p digits following the decimal point. With no\
    precision given, uses a precision of 6 digits after the decimal point for\
    float, and uses a precision large enough to show all coefficient digits for\
    Decimal. If no digits follow the decimal point, the decimal point is also\
    removed unless the '#' option is used.|
    |'F' |Fixed-point notation. Same as 'f', but converts nan to NAN and inf to\
    INF.|
    |'g' |General format. For a given precision p >= 1, this rounds the number\
    to p significant digits and then formats the result in either fixed-point\
    format or in scientific notation, depending on its magnitude. A precision\
    of 0 is treated as equivalent to a precision of 1.\
    The precise rules are as follows: suppose that the result formatted with\
    presentation type 'e' and precision p-1 would have exponent exp. Then, if\
    m <= exp < p, where m is -4 for floats and -6 for Decimals, the number is\
    formatted with presentation type 'f' and precision p-1-exp. Otherwise, the\
    number is formatted with presentation type 'e' and precision p-1. In both\
    cases insignificant trailing zeros are removed from the significand, and\
    the decimal point is also removed if there are no remaining digits\
    following it, unless the '#' option is used.\
    With no precision given, uses a precision of 6 significant digits for\
    float. For Decimal, the coefficient of the result is formed from the\
    coefficient digits of the value; scientific notation is used for values\
    smaller than 1e-6 in absolute value and values where the place value of the\
    least significant digit is larger than 1, and fixed-point notation is used\
    otherwise.\
    Positive and negative infinity, positive and negative zero, and nans, are\
    formatted as inf, -inf, 0, -0 and nan respectively, regardless of the\
    precision.|
    |'G' |General format. Same as 'g' except switches to 'E' if the number gets\
    too large. The representations of infinity and NaN are uppercased, too.|
    |'n' |Number. This is the same as 'g', except that it uses the current\
    locale setting to insert the appropriate number separator characters.|
    |'%' |Percentage. Multiplies the number by 100 and displays in fixed ('f')\
    format, followed by a percent sign.|
    |None |For float this is like the 'g' type, except that when fixed-point\
    notation is used to format the result, it always includes at least one\
    digit past the decimal point, and switches to the scientific notation when\
    exp >= p - 1. When the precision is not specified, the latter will be as\
    large as needed to represent the given value faithfully.\
    For Decimal, this is the same as either 'g' or 'G' depending on the value\
    of context.capitals for the current decimal context.\
    The overall effect is to match the output of str() as altered by the other\
    format modifiers.|
    """

    fn __init__(
        inout self,
        fill: UInt8 = ord(" "),
        align: UInt8 = 0,
        sign: UInt8 = ord("-"),
        coerce_z: Bool = False,
        alternate_form: Bool = False,
        width: UInt8 = 0,
        grouping_option: UInt8 = 0,
        precision: UInt8 = 0,
        type: UInt8 = 0,
    ):
        """Construct a FormatSpec instance.

        Args:
            fill: Defaults to space.
            align: Defaults to `0` which is adjusted to the default for the arg
                type.
            sign: Defaults to `-`.
            coerce_z: Defaults to False.
            alternate_form: Defaults to False.
            width: Defaults to `0` which is adjusted to the default for the arg
                type.
            grouping_option: Defaults to `0` which is adjusted to the default for
                the arg type.
            precision: Defaults to `0` which is adjusted to the default for the
                arg type.
            type: Defaults to `0` which is adjusted to the default for the arg
                type.
        """
        self.fill = fill
        self.align = align
        self.sign = sign
        self.coerce_z = coerce_z
        self.alternate_form = alternate_form
        self.width = width
        self.grouping_option = grouping_option
        self.precision = precision
        self.type = type

    @staticmethod
    fn parse(fmt_str: StringSlice) -> Optional[Self]:
        """Parses the format spec string.

        Args:
            fmt_str: The StringSlice with the format spec.

        Returns:
            An instance of FormatSpec.
        """

        alias `:` = UInt8(ord(":"))
        var f_len = fmt_str.byte_length()
        var f_ptr = fmt_str.unsafe_ptr()
        var colon_idx = -1
        var idx = 0
        while idx < f_len:
            if f_ptr[idx] == `:`:
                exclamation_index = idx
                break
            idx += 1

        if colon_idx == -1:
            return None

        # TODO: Future implementation of format specifiers
        return None

    fn stringify[
        T: _CurlyEntryFormattable
    ](self, inout res: String, item: T) raises:
        """Stringify a type according to its format specification.

        Args:
            res: The resulting String.
            item: The item to stringify.
        """
        var type_implements_float = True  # TODO
        var type_implements_float_raising = True  # TODO
        var type_implements_int = True  # TODO
        var type_implements_int_raising = True  # TODO

        # TODO: transform to int/float depending on format spec and stringify
        # with hex/bin/oct etc.
        res += str(item)

    fn format_string(self, inout res: String, item: String) raises:
        """Transform a String according to its format specification.

        Args:
            res: The resulting String.
            item: The item to format.
        """

        # TODO: align, fill, etc.
        res += item
