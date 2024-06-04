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
"""Implements the StringRef class.
"""

from bit import countr_zero
from builtin.dtype import _uint_type_of_width
from builtin.string import _atol, _isspace
from memory import DTypePointer, UnsafePointer, memcmp


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _align_down(value: Int, alignment: Int) -> Int:
    return value._positive_div(alignment) * alignment


# ===----------------------------------------------------------------------===#
# StringRef
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct StringRef(
    Sized,
    IntableRaising,
    CollectionElement,
    Stringable,
    Hashable,
    Boolable,
    Comparable,
):
    """
    Represent a constant reference to a string, i.e. a sequence of characters
    and a length, which need not be null terminated.
    """

    # Fields
    var data: UnsafePointer[UInt8]
    """A pointer to the beginning of the string data being referenced."""
    var length: Int
    """The length of the string being referenced."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__() -> Self:
        """Construct a StringRef value with length zero.

        Returns:
            Constructed `StringRef` object.
        """
        return StringRef(UnsafePointer[UInt8](), 0)

    @always_inline
    fn __init__(str: StringLiteral) -> Self:
        """Construct a StringRef value given a constant string.

        Args:
            str: The input constant string.

        Returns:
            Constructed `StringRef` object.
        """
        return StringRef(str.unsafe_ptr(), len(str))

    # TODO: #2317 Drop support for this constructor when we have fully
    # transitioned to UInt8 as the main byte type.
    @always_inline
    fn __init__(ptr: DTypePointer[DType.int8], len: Int) -> Self:
        """Construct a StringRef value given a (potentially non-0 terminated
        string).

        The constructor takes a raw pointer and a length.

        Note that you should use the constructor from `DTypePointer[DType.uint8]` instead
        as we are now storing the bytes as UInt8.
        See https://github.com/modularml/mojo/issues/2317 for more information.

        Args:
            ptr: DTypePointer to the string.
            len: The length of the string.

        Returns:
            Constructed `StringRef` object.
        """
        var unsafe_ptr = UnsafePointer[Int8]._from_dtype_ptr(ptr)

        return Self {data: unsafe_ptr.bitcast[UInt8](), length: len}

    @always_inline
    fn __init__(ptr: DTypePointer[DType.uint8], len: Int) -> Self:
        """Construct a StringRef value given a (potentially non-0 terminated
        string).

        The constructor takes a raw pointer and a length.

        Args:
            ptr: DTypePointer to the string.
            len: The length of the string.

        Returns:
            Constructed `StringRef` object.
        """
        var unsafe_ptr = UnsafePointer[UInt8]._from_dtype_ptr(ptr)

        return Self {data: unsafe_ptr, length: len}

    @always_inline
    fn __init__(ptr: UnsafePointer[UInt8]) -> Self:
        """Construct a StringRef value given a null-terminated string.

        Args:
            ptr: UnsafePointer to the string.

        Returns:
            Constructed `StringRef` object.
        """

        return DTypePointer[DType.uint8](ptr)

    # TODO: #2317 Drop support for this constructor when we have fully
    # transitioned to UInt8 as the main byte type.
    @always_inline
    fn __init__(ptr: DTypePointer[DType.int8]) -> Self:
        """Construct a StringRef value given a null-terminated string.

        Note that you should use the constructor from `DTypePointer[DType.uint8]` instead
        as we are now storing the bytes as UInt8.
        See https://github.com/modularml/mojo/issues/2317 for more information.

        Args:
            ptr: DTypePointer to the string.

        Returns:
            Constructed `StringRef` object.
        """

        var len = 0
        while ptr.load(len):
            len += 1

        return StringRef(ptr, len)

    @always_inline
    fn __init__(ptr: DTypePointer[DType.uint8]) -> Self:
        """Construct a StringRef value given a null-terminated string.

        Args:
            ptr: DTypePointer to the string.

        Returns:
            Constructed `StringRef` object.
        """

        var len = 0
        while ptr.load(len):
            len += 1

        return StringRef(ptr.bitcast[DType.int8](), len)

    # ===-------------------------------------------------------------------===#
    # Helper methods for slicing
    # ===-------------------------------------------------------------------===#
    # TODO: Move to slice syntax like str_ref[:42]

    fn take_front(self, num_bytes: Int = 1) -> Self:
        """Return a StringRef equal to 'self' but with only the first
        `num_bytes` elements remaining.  If `num_bytes` is greater than the
        length of the string, the entire string is returned.

        Args:
          num_bytes: The number of bytes to include.

        Returns:
          A new slice that starts with those bytes.
        """
        debug_assert(num_bytes >= 0, "num_bytes must be non-negative")
        if num_bytes >= self.length:
            return self
        return Self(self.data, num_bytes)

    fn take_back(self, num_bytes: Int = 1) -> Self:
        """Return a StringRef equal to 'self' but with only the last
        `num_bytes` elements remaining.  If `num_bytes` is greater than the
        length of the string, the entire string is returned.

        Args:
          num_bytes: The number of bytes to include.

        Returns:
          A new slice that ends with those bytes.
        """
        debug_assert(num_bytes >= 0, "num_bytes must be non-negative")
        if num_bytes >= self.length:
            return self
        return Self(self.data + (self.length - num_bytes), num_bytes)

    fn drop_front(self, num_bytes: Int = 1) -> Self:
        """Return a StringRef equal to 'self' but with the first
        `num_bytes` elements skipped.  If `num_bytes` is greater than the
        length of the string, an empty StringRef is returned.

        Args:
          num_bytes: The number of bytes to drop.

        Returns:
          A new slice with those bytes skipped.
        """
        debug_assert(num_bytes >= 0, "num_bytes must be non-negative")
        if num_bytes >= self.length:
            return StringRef()
        return Self(self.data + num_bytes, self.length - num_bytes)

    fn drop_back(self, num_bytes: Int = 1) -> Self:
        """Return a StringRef equal to 'self' but with the last `num_bytes`
        elements skipped.  If `num_bytes` is greater than the
        length of the string, the entire string is returned.

        Args:
          num_bytes: The number of bytes to include.

        Returns:
          A new slice ends earlier than those bytes.
        """
        debug_assert(num_bytes >= 0, "num_bytes must be non-negative")
        if num_bytes >= self.length:
            return StringRef()
        return Self(self.data, self.length - num_bytes)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> StringRef:
        """Get the string value at the specified position.

        Args:
          idx: The index position.

        Returns:
          The character at the specified position.
        """
        return StringRef {data: self.data + idx, length: 1}

    @always_inline
    fn __eq__(self, rhs: StringRef) -> Bool:
        """Compares two strings are equal.

        Args:
          rhs: The other string.

        Returns:
          True if the strings match and False otherwise.
        """
        return not (self != rhs)

    fn __contains__(self, substr: StringRef) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return self.find(substr) != -1

    @always_inline
    fn __ne__(self, rhs: StringRef) -> Bool:
        """Compares two strings are not equal.

        Args:
          rhs: The other string.

        Returns:
          True if the strings do not match and False otherwise.
        """
        return len(self) != len(rhs) or self._memcmp(rhs, len(self))

    @always_inline
    fn __lt__(self, rhs: StringRef) -> Bool:
        """Compare this StringRef to the RHS using LT comparison.

        Args:
            rhs: The other StringRef to compare against.

        Returns:
            True if this string is strictly less than the RHS string and False
            otherwise.
        """
        var len1 = len(self)
        var len2 = len(rhs)
        return self._memcmp(rhs, min(len1, len2)) < int(len1 < len2)

    @always_inline
    fn __le__(self, rhs: StringRef) -> Bool:
        """Compare this StringRef to the RHS using LE comparison.

        Args:
            rhs: The other StringRef to compare against.

        Returns:
            True if this string is less than or equal to the RHS string and
            False otherwise.
        """
        return not (rhs < self)

    @always_inline
    fn __gt__(self, rhs: StringRef) -> Bool:
        """Compare this StringRef to the RHS using GT comparison.

        Args:
            rhs: The other StringRef to compare against.

        Returns:
            True if this string is strictly greater than the RHS string and
            False otherwise.
        """
        return rhs < self

    @always_inline
    fn __ge__(self, rhs: StringRef) -> Bool:
        """Compare this StringRef to the RHS using GE comparison.

        Args:
            rhs: The other StringRef to compare against.

        Returns:
            True if this string is greater than or equal to the RHS string and
            False otherwise.
        """
        return not (self < rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the string is empty or not.

        Returns:
          Returns True if the string is not empty and False otherwise.
        """
        return len(self) != 0

    fn __hash__(self) -> Int:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.data, self.length)

    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.

        For example, `int("19")` returns `19`. If the given string cannot be parsed
        as an integer value, an error is raised. For example, `int("hi")` raises an
        error.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return _atol(self)

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the string.

        Returns:
          The length of the string.
        """
        return self.length

    fn __str__(self) -> String:
        """Convert the string reference to a string.

        Returns:
            A new string.
        """
        return self

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    # Use a local memcmp rather than memory.memcpy to avoid indirect recursions.
    @always_inline("nodebug")
    fn _memcmp(self, other: StringRef, count: Int) -> Int:
        for i in range(count):
            var s1i = self.data[i]
            var s2i = other.data[i]
            if s1i == s2i:
                continue
            return 1 if s1i > s2i else -1
        return 0

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Retrieves  a pointer to the underlying memory.

        Prefer to use `as_uint8_ptr()` instead.

        Returns:
            The pointer to the underlying memory.
        """
        return self.data

    @always_inline
    fn empty(self) -> Bool:
        """Returns True if the StringRef has length = 0.

        Returns:
            Whether the stringref is empty.
        """
        return self.length == 0

    fn count(self, substr: StringRef) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        if not substr:
            return len(self) + 1

        var res = 0
        var offset = 0

        while True:
            var pos = self.find(substr, offset)
            if pos == -1:
                break
            res += 1

            offset = pos + len(substr)

        return res

    fn find(self, substr: StringRef, start: Int = 0) -> Int:
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

        var loc = _memmem(
            haystack_str.unsafe_ptr(),
            len(haystack_str),
            substr.unsafe_ptr(),
            len(substr),
        )

        if not loc:
            return -1

        return int(loc) - int(self.unsafe_ptr())

    fn rfind(self, substr: StringRef, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

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

    fn _from_start(self, start: Int) -> StringRef:
        """Gets the StringRef pointing to the substring after the specified slice start position.

        If start is negative, it is interpreted as the number of characters
        from the end of the string to start at.

        Args:
            start: Starting index of the slice.

        Returns:
            A StringRef borrowed from the current string containing the
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

        var data = self.data + abs_start
        var length = self_len - abs_start

        return StringRef(data, length)

    fn strip(self) -> StringRef:
        """Gets a StringRef with leading and trailing whitespaces removed.
        This only takes C spaces into account: " \\t\\n\\r\\f\\v".

        For example, `"  mojo  "` returns `"mojo"`.

        Returns:
            A StringRef with leading and trailing whitespaces removed.
        """
        var start: Int = 0
        var end: Int = len(self)
        var ptr = self.unsafe_ptr()
        while start < end and _isspace(ptr[start]):
            start += 1
        while end > start and _isspace(ptr[end - 1]):
            end -= 1
        return StringRef(ptr + start, end - start)

    fn startswith(
        self, prefix: StringRef, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the StringRef starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          prefix: The prefix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is prefixed by the input prefix.
        """
        if end == -1:
            return self.find(prefix, start) == start
        return StringRef(self.unsafe_ptr() + start, end - start).startswith(
            prefix
        )

    fn endswith(self, suffix: StringRef, start: Int = 0, end: Int = -1) -> Bool:
        """Checks if the StringRef end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          suffix: The suffix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is suffixed by the input suffix.
        """

        if end == -1:
            return self.rfind(suffix, start) + len(suffix) == len(self)
        return StringRef(self.unsafe_ptr() + start, end - start).endswith(
            suffix
        )


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _memchr[
    type: DType
](source: DTypePointer[type], char: Scalar[type], len: Int) -> DTypePointer[
    type
]:
    if not len:
        return DTypePointer[type]()
    alias bool_mask_width = simdwidthof[DType.bool]()
    var first_needle = SIMD[type, bool_mask_width](char)
    var vectorized_end = _align_down(len, bool_mask_width)

    for i in range(0, vectorized_end, bool_mask_width):
        var bool_mask = source.load[width=bool_mask_width](i) == first_needle
        var mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)
        if mask:
            return source + i + countr_zero(mask)

    for i in range(vectorized_end, len):
        if source[i] == char:
            return source + i
    return DTypePointer[type]()


@always_inline
fn _memmem[
    type: DType
](
    haystack: DTypePointer[type],
    haystack_len: Int,
    needle: DTypePointer[type],
    needle_len: Int,
) -> DTypePointer[type]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return DTypePointer[type]()
    if needle_len == 1:
        return _memchr[type](haystack, needle[0], haystack_len)

    alias bool_mask_width = simdwidthof[DType.bool]()
    var vectorized_end = _align_down(
        haystack_len - needle_len + 1, bool_mask_width
    )

    var first_needle = SIMD[type, bool_mask_width](needle[0])
    var last_needle = SIMD[type, bool_mask_width](needle[needle_len - 1])

    for i in range(0, vectorized_end, bool_mask_width):
        var first_block = haystack.load[width=bool_mask_width](i)
        var last_block = haystack.load[width=bool_mask_width](
            i + needle_len - 1
        )

        var eq_first = first_needle == first_block
        var eq_last = last_needle == last_block

        var bool_mask = eq_first & eq_last
        var mask = bitcast[_uint_type_of_width[bool_mask_width]()](bool_mask)

        while mask:
            var offset = i + countr_zero(mask)
            if memcmp(haystack + offset + 1, needle + 1, needle_len - 1) == 0:
                return haystack + offset
            mask = mask & (mask - 1)

    # remaining partial block compare using byte-by-byte
    #
    for i in range(vectorized_end, haystack_len - needle_len + 1):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i

    return DTypePointer[type]()


@always_inline
fn _memrchr[
    type: DType
](source: DTypePointer[type], char: Scalar[type], len: Int) -> DTypePointer[
    type
]:
    if not len:
        return DTypePointer[type]()
    for i in reversed(range(len)):
        if source[i] == char:
            return source + i
    return DTypePointer[type]()


@always_inline
fn _memrmem[
    type: DType
](
    haystack: DTypePointer[type],
    haystack_len: Int,
    needle: DTypePointer[type],
    needle_len: Int,
) -> DTypePointer[type]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return DTypePointer[type]()
    if needle_len == 1:
        return _memrchr[type](haystack, needle[0], haystack_len)
    for i in reversed(range(haystack_len - needle_len + 1)):
        if haystack[i] != needle[0]:
            continue
        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i
    return DTypePointer[type]()
