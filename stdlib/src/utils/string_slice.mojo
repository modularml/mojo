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

from collections import List, Optional
from collections.string import _atof, _atol, _isspace
from sys import bitwidthof, simdwidthof
from sys.intrinsics import unlikely, likely

from bit import count_leading_zeros
from memory import UnsafePointer, memcmp, memcpy, Span
from memory.memory import _memcmp_impl_unconstrained

from utils.format import _CurlyEntryFormattable, _FormatCurlyEntry

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
        "Function does not work correctly if given a continuation byte.",
    )
    return int(count_leading_zeros(~b)) + int(b < 0b1000_0000)


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
    origin: Origin[is_mutable],
    forward: Bool = True,
]:
    """Iterator for `StringSlice` over unicode characters.

    Parameters:
        is_mutable: Whether the slice is mutable.
        origin: The origin of the underlying string data.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var ptr: UnsafePointer[Byte]
    var length: Int

    fn __init__(mut self, *, unsafe_pointer: UnsafePointer[Byte], length: Int):
        self.index = 0 if forward else length
        self.ptr = unsafe_pointer
        self.length = length

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> StringSlice[origin]:
        @parameter
        if forward:
            byte_len = _utf8_first_byte_sequence_length(self.ptr[self.index])
            i = self.index
            self.index += byte_len
            return StringSlice[origin](ptr=self.ptr + i, length=byte_len)
        else:
            byte_len = 1
            while _utf8_byte_type(self.ptr[self.index - byte_len]) == 1:
                byte_len += 1
            self.index -= byte_len
            return StringSlice[origin](
                ptr=self.ptr + self.index, length=byte_len
            )

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            remaining = self.length - self.index
            cont = _count_utf8_continuation_bytes(
                Span[Byte, ImmutableAnyOrigin](
                    ptr=self.ptr + self.index, length=remaining
                )
            )
            return remaining - cont
        else:
            return self.index - _count_utf8_continuation_bytes(
                Span[Byte, ImmutableAnyOrigin](ptr=self.ptr, length=self.index)
            )


@value
@register_passable("trivial")
struct StringSlice[is_mutable: Bool, //, origin: Origin[is_mutable]](
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
    @implicit
    fn __init__(out self: StaticString, lit: StringLiteral):
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
    fn __init__(out self, *, owned unsafe_from_utf8: Span[Byte, origin]):
        """Construct a new `StringSlice` from a sequence of UTF-8 encoded bytes.

        Args:
            unsafe_from_utf8: A `Span[Byte]` encoded in UTF-8.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.
        """

        self._slice = unsafe_from_utf8

    fn __init__(out self, *, unsafe_from_utf8_strref: StringRef):
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
            ptr=strref.unsafe_ptr(),
            length=len(strref),
        )

        self = Self(unsafe_from_utf8=byte_slice)

    @always_inline
    fn __init__(out self, *, ptr: UnsafePointer[Byte], length: Int):
        """Construct a `StringSlice` from a pointer to a sequence of UTF-8
        encoded bytes and a length.

        Args:
            ptr: A pointer to a sequence of bytes encoded in UTF-8.
            length: The number of bytes of encoded data.

        Safety:
            - `ptr` MUST point to at least `length` bytes of valid UTF-8 encoded
                data.
            - `ptr` must point to data that is live for the duration of
                `origin`.
        """
        self._slice = Span[Byte, origin](ptr=ptr, length=length)

    @always_inline
    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided `StringSlice`.

        Args:
            other: The `StringSlice` to copy.
        """
        self._slice = other._slice

    @implicit
    fn __init__[
        O: ImmutableOrigin, //
    ](mut self: StringSlice[O], ref [O]value: String):
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
        var s = S(ptr=self.unsafe_ptr(), length=b_len)
        return b_len - _count_utf8_continuation_bytes(s)

    fn write_to[W: Writer](self, mut writer: W):
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

    fn __contains__(ref self, substr: StringSlice[_]) -> Bool:
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

    fn __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n : The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """

        var len_self = self.byte_length()
        var count = len_self * n + 1
        var buf = String._buffer_type(capacity=count)
        buf.size = count
        var b_ptr = buf.unsafe_ptr()
        for i in range(n):
            memcpy(b_ptr + len_self * i, self.unsafe_ptr(), len_self)
        b_ptr[count - 1] = 0
        return String(buf^)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn strip(self, chars: StringSlice) -> Self:
        """Return a copy of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading or trailing characters.

        Examples:

        ```mojo
        print("himojohi".strip("hi")) # "mojo"
        ```
        .
        """

        return self.lstrip(chars).rstrip(chars)

    @always_inline
    fn strip(self) -> Self:
        """Return a copy of the string with leading and trailing whitespaces
        removed.

        Returns:
            A copy of the string with no leading or trailing whitespaces.

        Examples:

        ```mojo
        print("  mojo  ".strip()) # "mojo"
        ```
        .
        """
        return self.lstrip().rstrip()

    @always_inline
    fn rstrip(self, chars: StringSlice) -> Self:
        """Return a copy of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no trailing characters.

        Examples:

        ```mojo
        print("mojohi".strip("hi")) # "mojo"
        ```
        .
        """

        var r_idx = self.byte_length()
        while r_idx > 0 and self[r_idx - 1] in chars:
            r_idx -= 1

        return Self(unsafe_from_utf8=self.as_bytes()[:r_idx])

    @always_inline
    fn rstrip(self) -> Self:
        """Return a copy of the string with trailing whitespaces removed.

        Returns:
            A copy of the string with no trailing whitespaces.

        Examples:

        ```mojo
        print("mojo  ".strip()) # "mojo"
        ```
        .
        """
        var r_idx = self.byte_length()
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self.__reversed__():
        #     if not s.isspace():
        #         break
        #     r_idx -= 1
        while r_idx > 0 and _isspace(self.as_bytes()[r_idx - 1]):
            r_idx -= 1
        return Self(unsafe_from_utf8=self.as_bytes()[:r_idx])

    @always_inline
    fn lstrip(self, chars: StringSlice) -> Self:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.

        Examples:

        ```mojo
        print("himojo".strip("hi")) # "mojo"
        ```
        .
        """

        var l_idx = 0
        while l_idx < self.byte_length() and self[l_idx] in chars:
            l_idx += 1

        return Self(unsafe_from_utf8=self.as_bytes()[l_idx:])

    @always_inline
    fn lstrip(self) -> Self:
        """Return a copy of the string with leading whitespaces removed.

        Returns:
            A copy of the string with no leading whitespaces.

        Examples:

        ```mojo
        print("  mojo".strip()) # "mojo"
        ```
        .
        """
        var l_idx = 0
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self:
        #     if not s.isspace():
        #         break
        #     l_idx += 1
        while l_idx < self.byte_length() and _isspace(self.as_bytes()[l_idx]):
            l_idx += 1
        return Self(unsafe_from_utf8=self.as_bytes()[l_idx:])

    @always_inline
    fn as_bytes(self) -> Span[Byte, origin]:
        """Get the sequence of encoded bytes of the underlying string.

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
        return StringSlice[origin](
            ptr=self.unsafe_ptr() + start, length=end - start
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
        return StringSlice[origin](
            ptr=self.unsafe_ptr() + start, length=end - start
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

    fn find(ref self, substr: StringSlice, start: Int = 0) -> Int:
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

        var ptr = self.unsafe_ptr()
        var length = self.byte_length()

        @parameter
        if single_character:
            return length != 0 and _is_newline_char[include_r_n=True](
                ptr, 0, ptr[0], length
            )
        else:
            var offset = 0
            for s in self:
                var b_len = s.byte_length()
                if not _is_newline_char(ptr, offset, ptr[offset], b_len):
                    return False
                offset += b_len
            return length != 0

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

        # highly performance sensitive code, benchmark before touching
        alias `\r` = UInt8(ord("\r"))
        alias `\n` = UInt8(ord("\n"))

        output = List[StringSlice[O]](capacity=128)  # guessing
        var ptr = self.unsafe_ptr()
        var length = self.byte_length()
        var offset = 0

        while offset < length:
            var eol_start = offset
            var eol_length = 0

            while eol_start < length:
                var b0 = ptr[eol_start]
                var char_len = _utf8_first_byte_sequence_length(b0)
                debug_assert(
                    eol_start + char_len <= length,
                    "corrupted sequence causing unsafe memory access",
                )
                var isnewline = unlikely(
                    _is_newline_char(ptr, eol_start, b0, char_len)
                )
                var char_end = int(isnewline) * (eol_start + char_len)
                var next_idx = char_end * int(char_end < length)
                var is_r_n = b0 == `\r` and next_idx != 0 and ptr[
                    next_idx
                ] == `\n`
                eol_length = int(isnewline) * char_len + int(is_r_n)
                if isnewline:
                    break
                eol_start += char_len

            var str_len = eol_start - offset + int(keepends) * eol_length
            var s = StringSlice[O](ptr=ptr + offset, length=str_len)
            output.append(s)
            offset = eol_start + eol_length

        return output^


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


fn _to_string_list[
    T: CollectionElement,  # TODO(MOCO-1446): Make `T` parameter inferred
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
        buf = String._buffer_type(ptr=p, length=f_len, capacity=f_len)
        (out_ptr + i).init_pointee_move(String(buf^))
    return List[String](ptr=out_ptr, length=i_len, capacity=i_len)


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

    return _to_string_list[items.T, len_fn, unsafe_ptr_fn](items)


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

    return _to_string_list[items.T, len_fn, unsafe_ptr_fn](items)


@always_inline
fn _is_newline_char[
    include_r_n: Bool = False
](p: UnsafePointer[Byte], eol_start: Int, b0: Byte, char_len: Int) -> Bool:
    """Returns whether the char is a newline char.

    Safety:
        This assumes valid utf-8 is passed.
    """
    # highly performance sensitive code, benchmark before touching
    alias `\r` = UInt8(ord("\r"))
    alias `\n` = UInt8(ord("\n"))
    alias `\t` = UInt8(ord("\t"))
    alias `\x1c` = UInt8(ord("\x1c"))
    alias `\x1e` = UInt8(ord("\x1e"))

    # here it's actually faster to have branching due to the branch predictor
    # "realizing" that the char_len == 1 path is often taken. Using the likely
    # intrinsic is to make the machine code be ordered to optimize machine
    # instruction fetching, which is an optimization for the CPU front-end.
    if likely(char_len == 1):
        return `\t` <= b0 <= `\x1e` and not (`\r` < b0 < `\x1c`)
    elif char_len == 2:
        var b1 = p[eol_start + 1]
        var is_next_line = b0 == 0xC2 and b1 == 0x85  # unicode next line \x85

        @parameter
        if include_r_n:
            return is_next_line or (b0 == `\r` and b1 == `\n`)
        else:
            return is_next_line
    elif char_len == 3:  # unicode line sep or paragraph sep: \u2028 , \u2029
        var b1 = p[eol_start + 1]
        var b2 = p[eol_start + 2]
        return b0 == 0xE2 and b1 == 0x80 and (b2 == 0xA8 or b2 == 0xA9)
    return False
