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

from collections.string import _atol, _isspace
from hashlib._hasher import _HashableWithHasher, _Hasher
from memory import UnsafePointer, bitcast
from memory.memory import _memcmp_impl_unconstrained
from utils import StringSlice
from sys.ffi import c_char

# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------===#
# StringRef
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct StringRef(
    Sized,
    IntableRaising,
    CollectionElement,
    CollectionElementNew,
    Stringable,
    Formattable,
    Hashable,
    _HashableWithHasher,
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
    fn __init__(inout self):
        """Construct a StringRef value with length zero."""
        self = StringRef(UnsafePointer[UInt8](), 0)

    @always_inline
    fn __init__(inout self, *, other: Self):
        """Copy the object.

        Args:
            other: The value to copy.
        """
        self.data = other.data
        self.length = other.length

    @always_inline
    fn __init__(inout self, str: StringLiteral):
        """Construct a StringRef value given a constant string.

        Args:
            str: The input constant string.
        """
        self = StringRef(str.unsafe_ptr(), len(str))

    @always_inline
    fn __init__(inout self, ptr: UnsafePointer[c_char], len: Int):
        """Construct a StringRef value given a (potentially non-0 terminated
        string).

        The constructor takes a raw pointer and a length.

        Note that you should use the constructor from `UnsafePointer[UInt8]` instead
        as we are now storing the bytes as UInt8.
        See https://github.com/modularml/mojo/issues/2317 for more information.

        Args:
            ptr: UnsafePointer to the string.
            len: The length of the string.
        """

        self.data = ptr.bitcast[UInt8]()
        self.length = len

    @always_inline
    fn __init__(inout self, ptr: UnsafePointer[UInt8]):
        """Construct a StringRef value given a null-terminated string.

        Args:
            ptr: UnsafePointer to the string.
        """

        var len = 0
        while ptr.load(len):
            len += 1

        self = StringRef(ptr, len)

    @always_inline
    fn __init__(inout self, ptr: UnsafePointer[c_char]):
        """Construct a StringRef value given a null-terminated string.

        Note that you should use the constructor from `UnsafePointer[UInt8]` instead
        as we are now storing the bytes as UInt8.
        See https://github.com/modularml/mojo/issues/2317 for more information.

        Args:
            ptr: UnsafePointer to the string.
        """

        var len = 0
        while ptr.load(len):
            len += 1

        self = StringRef(ptr, len)

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
        alias S = StringSlice[StaticConstantOrigin]
        var s_slice = S(unsafe_from_utf8_strref=self)
        return s_slice.find(S(unsafe_from_utf8_strref=substr)) != -1

    @always_inline
    fn __ne__(self, rhs: StringRef) -> Bool:
        """Compares two strings are not equal.

        Args:
          rhs: The other string.

        Returns:
          True if the strings do not match and False otherwise.
        """
        return len(self) != len(rhs) or _memcmp_impl_unconstrained(
            self.data, rhs.data, len(self)
        )

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
        return int(len1 < len2) > _memcmp_impl_unconstrained(
            self.data, rhs.data, min(len1, len2)
        )

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

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.data, self.length)

    fn __hash__[H: _Hasher](self, inout hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_bytes(self.data, self.length)

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

    @no_inline
    fn __str__(self) -> String:
        """Convert the string reference to a string.

        Returns:
            A new string.
        """
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this StringRef to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        # SAFETY:
        #   Safe because our use of this StringSlice does not outlive `self`.
        var str_slice = StringSlice[ImmutableAnyOrigin](
            unsafe_from_utf8_strref=self
        )

        writer.write_str(str_slice)

    fn __fspath__(self) -> String:
        """Return the file system path representation of the object.

        Returns:
          The file system path representation as a string.
        """
        return self.__str__()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Retrieves  a pointer to the underlying memory.

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
            alias S = StringSlice[StaticConstantOrigin]
            var pos = S(unsafe_from_utf8_strref=self).find(
                S(unsafe_from_utf8_strref=substr), offset
            )
            if pos == -1:
                break
            res += 1

            offset = pos + len(substr)

        return res

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

    # FIXME: look at string's split and copy it to StringSlice
    # Delete this after as it does not conform to Python's behavior and we don't
    # want to keep using StringRef. I'll do it in another PR, this is a reminder
    fn split(self, delimiter: StringRef) raises -> List[StringRef]:
        """Split the StringRef by a delimiter.

        Args:
            delimiter: The StringRef to split on.

        Returns:
            A List of StringRefs containing the input split by the delimiter.

        Raises:
            Error if an empty delimiter is specified.
        """
        if not delimiter:
            raise Error("empty delimiter not allowed to be passed to split.")

        var output = List[StringRef]()
        var ptr = self.unsafe_ptr()

        var current_offset = 0
        alias S = StringSlice[StaticConstantOrigin]
        var self_slice = S(unsafe_from_utf8_strref=self)
        var delim_slice = S(unsafe_from_utf8_strref=delimiter)
        while True:
            var loc = self_slice.find(delim_slice, current_offset)
            # delimiter not found, so add the search slice from where we're currently at
            if loc == -1:
                output.append(
                    StringRef(ptr + current_offset, len(self) - current_offset)
                )
                break

            # We found a delimiter, so add the preceding string slice
            output.append(StringRef(ptr + current_offset, loc - current_offset))

            # Advance our search offset past the delimiter
            current_offset = loc + len(delimiter)
        return output

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
            alias S = StringSlice[StaticConstantOrigin]
            var s = S(unsafe_from_utf8_strref=self)
            return s.find(S(unsafe_from_utf8_strref=prefix), start) == start
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
        if len(suffix) > len(self):
            return False
        if end == -1:
            alias S = StringSlice[StaticConstantOrigin]
            return S(unsafe_from_utf8_strref=self).rfind(
                S(unsafe_from_utf8_strref=suffix), start
            ) + len(suffix) == len(self)
        return StringRef(self.unsafe_ptr() + start, end - start).endswith(
            suffix
        )
