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
"""Implements the StringLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

from sys.ffi import c_char

from memory import memcpy, UnsafePointer
from collections import List
from hashlib._hasher import _HashableWithHasher, _Hasher
from utils import StringRef, Span, StringSlice, StaticString
from utils import Formattable, Formatter
from utils._visualizers import lldb_formatter_wrapping_type

from collections.string import _atol, _StringSliceIter

# ===----------------------------------------------------------------------===#
# StringLiteral
# ===----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct StringLiteral(
    Boolable,
    Comparable,
    CollectionElementNew,
    Formattable,
    IntableRaising,
    KeyElement,
    Representable,
    Sized,
    Stringable,
    FloatableRaising,
    _HashableWithHasher,
):
    """This type represents a string literal.

    String literals are all null-terminated for compatibility with C APIs, but
    this is subject to change. String literals store their length as an integer,
    and this does not include the null terminator.
    """

    # Fields
    alias type = __mlir_type.`!kgen.string`

    var value: Self.type
    """The underlying storage for the string literal."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(inout self, value: Self.type):
        """Create a string literal from a builtin string type.

        Args:
            value: The string value.
        """
        self.value = value

    @always_inline("nodebug")
    fn __init__(inout self, *, other: Self):
        """Copy constructor.

        Args:
            other: The string literal to copy.
        """
        self = other

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __add__(self, rhs: StringLiteral) -> StringLiteral:
        """Concatenate two string literals.

        Args:
            rhs: The string to concat.

        Returns:
            The concatenated string.
        """
        return __mlir_op.`pop.string.concat`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: StringLiteral) -> Bool:
        """Compare two string literals for equality.

        Args:
            rhs: The string to compare.

        Returns:
            True if they are equal.
        """
        return not (self != rhs)

    @always_inline("nodebug")
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Compare two string literals for inequality.

        Args:
            rhs: The string to compare.

        Returns:
            True if they are not equal.
        """
        return StringRef(self) != StringRef(rhs)

    @always_inline("nodebug")
    fn __lt__(self, rhs: StringLiteral) -> Bool:
        """Compare this StringLiteral to the RHS using LT comparison.

        Args:
            rhs: The other StringLiteral to compare against.

        Returns:
            True if this StringLiteral is strictly less than the RHS StringLiteral and False otherwise.
        """
        return StringRef(self) < StringRef(rhs)

    @always_inline("nodebug")
    fn __le__(self, rhs: StringLiteral) -> Bool:
        """Compare this StringLiteral to the RHS using LE comparison.

        Args:
            rhs: The other StringLiteral to compare against.

        Returns:
            True if this StringLiteral is less than or equal to the RHS StringLiteral and False otherwise.
        """
        return not (rhs < self)

    @always_inline("nodebug")
    fn __gt__(self, rhs: StringLiteral) -> Bool:
        """Compare this StringLiteral to the RHS using GT comparison.

        Args:
            rhs: The other StringLiteral to compare against.

        Returns:
            True if this StringLiteral is strictly greater than the RHS StringLiteral and False otherwise.
        """
        return rhs < self

    @always_inline("nodebug")
    fn __ge__(self, rhs: StringLiteral) -> Bool:
        """Compare this StringLiteral to the RHS using GE comparison.

        Args:
            rhs: The other StringLiteral to compare against.

        Returns:
            True if this StringLiteral is greater than or equal to the RHS StringLiteral and False otherwise.
        """
        return not (self < rhs)

    fn __contains__(self, substr: StringLiteral) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return substr in StringRef(self)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the string length.

        Returns:
            The length of this StringLiteral.
        """
        # TODO(MSTDL-160):
        #   Properly count Unicode codepoints instead of returning this length
        #   in bytes.
        return self.byte_length()

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Convert the string to a bool value.

        Returns:
            True if the string is not empty.
        """
        return len(self) != 0

    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.

        For example, `int("19")` returns `19`. If the given string cannot be parsed
        as an integer value, an error is raised. For example, `int("hi")` raises an
        error.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return _atol(self)

    fn __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value.

        If the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.
        """
        return atof(self)

    @no_inline
    fn __str__(self) -> String:
        """Convert the string literal to a string.

        Returns:
            A new string.
        """
        # TODO(MOCO-1224): We should be able to reuse this, but we have to
        # inline the string slice constructor to work around an elaborator
        # memory leak.
        # return self.as_string_slice()
        var string = String()
        var length = self.byte_length()
        var buffer = String._buffer_type()
        var new_capacity = length + 1
        buffer._realloc(new_capacity)
        buffer.size = new_capacity
        var data: UnsafePointer[UInt8] = self.unsafe_ptr()
        memcpy(buffer.data, data, length)
        (buffer.data + length).init_pointee_move(0)
        string._buffer = buffer^
        return string

    @no_inline
    fn __repr__(self) -> String:
        """Return a representation of the `StringLiteral` instance.

        You don't need to call this method directly, use `repr("...")` instead.

        Returns:
            A new representation of the string.
        """
        return self.__str__().__repr__()

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.unsafe_ptr(), len(self))

    fn __hash__[H: _Hasher](self, inout hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_bytes(self.unsafe_ptr(), self.byte_length())

    fn __fspath__(self) -> String:
        """Return the file system path representation of the object.

        Returns:
          The file system path representation as a string.
        """
        return self.__str__()

    fn __iter__(ref [_]self) -> _StringSliceIter[StaticConstantLifetime]:
        """Return an iterator over the string literal.

        Returns:
            An iterator over the string.
        """
        return _StringSliceIter[StaticConstantLifetime](
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
        return str(self)[idx]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this StringLiteral in bytes.

        Notes:
            This does not include the trailing null terminator in the count.
        """
        return __mlir_op.`pop.string.size`(self.value)

    @always_inline("nodebug")
    # FIXME(MSTDL-956): This should return a pointer with StaticConstantLifetime.
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Get raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        var ptr = UnsafePointer(__mlir_op.`pop.string.address`(self.value))

        # TODO(MSTDL-555):
        #   Remove bitcast after changing pop.string.address
        #   return type.
        return ptr.bitcast[UInt8]()

    @always_inline
    # FIXME(MSTDL-956): This should return a pointer with StaticConstantLifetime.
    fn unsafe_cstr_ptr(self) -> UnsafePointer[c_char]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be NUL terminated, and not null.

        Returns:
            The pointer to the underlying memory.
        """
        return self.unsafe_ptr().bitcast[c_char]()

    @always_inline
    fn as_string_slice(self) -> StaticString:
        """Returns a string slice of this static string literal.

        Returns:
            A string slice pointing to this static string literal.
        """

        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in StringLiteral so this is actually
        #   guaranteed to be valid.
        return StaticString(
            unsafe_from_utf8_ptr=self.unsafe_ptr(), len=self.byte_length()
        )

    @always_inline
    fn as_bytes(self) -> Span[UInt8, StaticConstantLifetime]:
        """
        Returns a contiguous Span of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        return Span[UInt8, StaticConstantLifetime](
            unsafe_ptr=self.unsafe_ptr(),
            len=self.byte_length(),
        )

    fn format_to(self, inout writer: Formatter):
        """
        Formats this string literal to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        writer.write_str(self.as_string_slice())

    fn find(self, substr: StringLiteral, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        return StringRef(self).find(substr, start=start)

    fn rfind(self, substr: StringLiteral, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        return StringRef(self).rfind(substr, start=start)

    fn replace(self, old: StringLiteral, new: StringLiteral) -> StringLiteral:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`. This operation only works in the param domain.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        return __mlir_op.`pop.string.replace`(self.value, old.value, new.value)

    fn join[T: StringableCollectionElement](self, elems: List[T, *_]) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            T: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        var result: String = ""
        var is_first = True

        for e in elems:
            if is_first:
                is_first = False
            else:
                result += self
            result += str(e[])

        return result

    fn split(self, sep: String, maxsplit: Int = -1) raises -> List[String]:
        """Split the string literal by a separator.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.
                Defaults to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting a space
        _ = "hello world".split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = "hello,,world".split(",") # ["hello", "", "world"]
        # Splitting with maxsplit
        _ = "1,2,3".split(",", 1) # ['1', '2,3']
        ```
        .
        """
        return str(self).split(sep, maxsplit)

    fn split(self, sep: NoneType = None, maxsplit: Int = -1) -> List[String]:
        """Split the string literal by every whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from string. Defaults
                to unlimited.

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
        _ = "hello \\t\\n\\r\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029world".split()
        # ["hello", "world"]
        ```
        .
        """
        return str(self).split(sep, maxsplit)

    fn splitlines(self, keepends: Bool = False) -> List[String]:
        """Split the string literal at line boundaries. This corresponds to Python's
        [universal newlines](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\t\\n\\r\\r\\n\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        return self.as_string_slice().splitlines(keepends)

    fn count(self, substr: String) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string literal.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        return str(self).count(substr)

    fn lower(self) -> String:
        """Returns a copy of the string literal with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        return str(self).lower()

    fn upper(self) -> String:
        """Returns a copy of the string literal with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        return str(self).upper()

    fn rjust(self, width: Int, fillchar: StringLiteral = " ") -> String:
        """Returns the string right justified in a string literal of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns right justified string, or self if width is not bigger than self length.
        """
        return str(self).rjust(width, fillchar)

    fn ljust(self, width: Int, fillchar: StringLiteral = " ") -> String:
        """Returns the string left justified in a string literal of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns left justified string, or self if width is not bigger than self length.
        """
        return str(self).ljust(width, fillchar)

    fn center(self, width: Int, fillchar: StringLiteral = " ") -> String:
        """Returns the string center justified in a string literal of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns center justified string, or self if width is not bigger than self length.
        """
        return str(self).center(width, fillchar)

    fn startswith(self, prefix: String, start: Int = 0, end: Int = -1) -> Bool:
        """Checks if the string literal starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          prefix: The prefix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is prefixed by the input prefix.
        """
        return str(self).startswith(prefix, start, end)

    fn endswith(self, suffix: String, start: Int = 0, end: Int = -1) -> Bool:
        """Checks if the string literal end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          suffix: The suffix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is suffixed by the input suffix.
        """
        return str(self).endswith(suffix, start, end)

    fn isdigit(self) -> Bool:
        """Returns True if all characters in the string literal are digits.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits else False.
        """
        return str(self).isdigit()

    fn isupper(self) -> Bool:
        """Returns True if all cased characters in the string literal are
        uppercase and there is at least one cased character.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all cased characters in the string literal are uppercase
            and there is at least one cased character, False otherwise.
        """
        return str(self).isupper()

    fn islower(self) -> Bool:
        """Returns True if all cased characters in the string literal
        are lowercase and there is at least one cased character.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all cased characters in the string literal are lowercase
            and there is at least one cased character, False otherwise.
        """
        return str(self).islower()

    fn strip(self) -> String:
        """Return a copy of the string literal with leading and trailing whitespaces
        removed.

        Returns:
            A string with no leading or trailing whitespaces.
        """
        return self.lstrip().rstrip()

    fn strip(self, chars: String) -> String:
        """Return a copy of the string literal with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A string with no leading or trailing characters.
        """

        return self.lstrip(chars).rstrip(chars)

    fn rstrip(self, chars: String) -> String:
        """Return a copy of the string literal with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A string with no trailing characters.
        """
        return str(self).rstrip(chars)

    fn rstrip(self) -> String:
        """Return a copy of the string with trailing whitespaces removed.

        Returns:
            A copy of the string with no trailing whitespaces.
        """
        return str(self).rstrip()

    fn lstrip(self, chars: String) -> String:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.
        """
        return str(self).lstrip(chars)

    fn lstrip(self) -> String:
        """Return a copy of the string with leading whitespaces removed.

        Returns:
            A copy of the string with no leading whitespaces.
        """
        return str(self).lstrip()
