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
"""Implements the StringLiteral struct.

These are Mojo built-ins, so you don't need to import them.
"""

from std.collections.string.format import _FormatUtils
from std.collections.string.string_span import (
    CodepointSliceIter,
    CodepointsIter,
    GraphemeSliceIter,
    StaticString,
)
from std.os import PathLike
from std.ffi import c_char, CStringSlice

# ===-----------------------------------------------------------------------===#
# StringLiteral
# ===-----------------------------------------------------------------------===#


@__nonmaterializable(String)
struct StringLiteral[value: __mlir_type.`!kgen.string`](
    Boolable,
    Defaultable,
    FloatableRaising,
    ImplicitlyCopyable,
    IntableRaising,
    PathLike,
    TrivialRegisterPassable,
    Writable,
):
    """This type represents a string literal.

    String literals are all null-terminated for compatibility with C APIs, but
    this is subject to change. String literals store their length as an integer,
    and this does not include the null terminator.

    Parameters:
        value: The underlying string value.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Constructor for any value."""
        pass

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __add__(
        self, rhs: StringLiteral
    ) -> StringLiteral[
        __mlir_attr[
            `#pop.string_concat<`,
            self.value,
            `,`,
            rhs.value,
            `> : !kgen.string`,
        ]
    ]:
        """Concatenate two string literals.

        Args:
            rhs: The string to concat.

        Returns:
            The concatenated string.
        """
        return {}

    def __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n : The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """
        return StringSlice(self) * n

    @always_inline("nodebug")
    def __eq__(self, rhs: StringSlice) -> Bool:
        """Compare two string literals for equality.

        Args:
            rhs: The string to compare.

        Returns:
            True if they are equal.
        """
        return not (self != rhs)

    @always_inline("nodebug")
    def __ne__(self, rhs: StringSlice) -> Bool:
        """Compare two string literals for inequality.

        Args:
            rhs: The string to compare.

        Returns:
            True if they are not equal.
        """
        return StringSlice(self) != rhs

    @always_inline("nodebug")
    def __lt__(self, rhs: StringSlice) -> Bool:
        """Compare this value to the RHS using lesser than (LT) comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this is strictly less than the RHS and False otherwise.
        """
        return StringSlice(self) < rhs

    @always_inline("nodebug")
    def __le__(self, rhs: StringSlice) -> Bool:
        """Compare this value to the RHS using lesser than or equal to (LE) comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this is less than or equal to the RHS and False otherwise.
        """
        return not (rhs < self)

    @always_inline("nodebug")
    def __gt__(self, rhs: StringSlice) -> Bool:
        """Compare this value to the RHS using greater than (GT) comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this is strictly greater than the RHS and False otherwise.
        """
        return rhs < self

    @always_inline("nodebug")
    def __ge__(self, rhs: StringSlice) -> Bool:
        """Compare this value to the RHS using greater than or equal to (GE) comparison.

        Args:
            rhs: The other value to compare against.

        Returns:
            True if this is greater than or equal to the RHS and False otherwise.
        """
        return not (self < rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __bool__(self) -> Bool:
        """Convert the string to a bool value.

        Returns:
            True if the string is not empty.
        """
        return self.byte_length() != 0

    @always_inline
    def __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.

        Returns:
            An integer value that represents the string.

        Raises:
            If the string cannot be parsed as a valid base-10 integer.
        """
        return Int(StringSlice(self))

    @always_inline
    def __float__(self) raises -> Float64:
        """Parses the string as a floating-point number and returns that value.

        Returns:
            A float value that represents the string.

        Raises:
            If the string cannot be parsed as a valid floating-point number.
        """
        return Float64(StringSlice(self))

    def __fspath__(self) -> String:
        """Return the file system path representation of the object.

        Returns:
          The file system path representation as a string.
        """
        return String(self)

    def __iter__(self) -> GraphemeSliceIter[ImmStaticOrigin]:
        """Iterate over the grapheme clusters in the string literal.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen, such as a base letter together with
        any combining marks. See `graphemes()` for the precise definition.

        To iterate by Unicode codepoint or by byte instead, use
        `codepoints()`/`codepoint_slices()` or `as_bytes()`.

        Returns:
            An iterator yielding each grapheme cluster as a `StringSlice`.
        """
        return self.graphemes()

    def __reversed__(self) -> GraphemeSliceIter[ImmStaticOrigin, False]:
        """Iterate backwards over the grapheme clusters in the string literal.

        See `graphemes()` for the definition of a grapheme cluster. Reverse
        iteration is more expensive per element than forward iteration; see
        `graphemes_reversed()` for details.

        Returns:
            A reverse iterator yielding each grapheme cluster as a
            `StringSlice`.
        """
        return self.graphemes_reversed()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes.

        Notes:
            This does not include the trailing null terminator in the count.
        """
        return Int(
            SIMDLength(mlir_value=__mlir_op.`pop.string.size`(self.value))
        )

    @always_inline
    def count_codepoints(self) -> Int:
        """Calculates the length in Unicode codepoints encoded in the
        UTF-8 representation of this string.

        This is an O(n) operation, where n is the length of the string, as it
        requires scanning the full string contents.

        Returns:
            The length in Unicode codepoints.

        Examples:

            Query the length of a string, in bytes and Unicode codepoints:

            ```mojo
            from std.testing import assert_equal

            var s = StringSlice("ನಮಸ್ಕಾರ")
            assert_equal(s.count_codepoints(), 7)
            assert_equal(s.byte_length(), 21)
            ```

            Strings containing only ASCII characters have the same byte and
            Unicode codepoint length:

            ```mojo
            from std.testing import assert_equal

            var s = StringSlice("abc")
            assert_equal(s.count_codepoints(), 3)
            assert_equal(s.byte_length(), 3)
            ```

            The character length of a string with visual combining characters is
            the length in Unicode codepoints, not grapheme clusters:

            ```mojo
            from std.testing import assert_equal

            var s = StringSlice("á")
            assert_equal(s.count_codepoints(), 2)
            assert_equal(s.byte_length(), 3)
            ```

        Notes:
            This method needs to traverse the whole string to count, so it has
            a performance hit compared to using the byte length.
        """
        return StringSlice(self).count_codepoints()

    @always_inline("nodebug")
    def ptr(
        self,
    ) -> Pointer[Byte, ImmStaticOrigin]:
        """Get a pointer to the underlying data.

        Returns:
            The pointer to the data.
        """
        var ptr = Pointer[_, ImmStaticOrigin](
            _mlir_value=__mlir_op.`pop.string.address`(self.value)
        )

        # TODO(MSTDL-555):
        #   Remove bitcast after changing pop.string.address
        #   return type.
        return ptr.unsafe_bitcast[Byte]()

    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=ptr)
    def unsafe_ptr(
        self,
    ) -> Pointer[Byte, ImmStaticOrigin]:
        """Get a pointer to the underlying data.

        Returns:
            The pointer to the data.
        """
        return self.ptr()

    @always_inline
    def as_c_string_slice(
        self,
    ) -> CStringSlice[ImmStaticOrigin]:
        """Return a `CStringSlice` to the underlying memory of the string.

        Returns:
            The `CStringSlice` of the string.
        """
        # Safety: StringLiteral is guaranteed to be nul-terminated.
        return CStringSlice(unsafe_from_ptr=self.ptr().unsafe_bitcast[c_char]())

    @always_inline("nodebug")
    def as_string_slice(self) -> StaticString:
        """Returns a string slice of this static string literal.

        Returns:
            A string slice pointing to this static string literal.
        """

        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in StringLiteral so this is actually
        #   guaranteed to be valid.
        return StaticString(
            unsafe_from_utf8=Span(
                unsafe_ptr=self.ptr(),
                length=self.byte_length(),
            )
        )

    @always_inline("nodebug")
    def as_bytes(self) -> Span[Byte, ImmStaticOrigin]:
        """
        Returns a contiguous Span of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        return Span[Byte, ImmStaticOrigin](
            unsafe_ptr=self.ptr(), length=self.byte_length()
        )

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this string literal to the provided Writer.

        Args:
            writer: The object to write to.
        """

        writer.write(StringSlice(self))

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the string literal.

        Args:
            writer: The value to write to.
        """
        StringSlice(self).write_repr_to(writer)

    def find(self, substr: StaticString, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        return StringSlice(self).find(substr, start=start)

    def rfind(self, substr: StaticString, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """
        return StringSlice(self).rfind(substr, start=start)

    def count(self, substr: StringSlice) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string literal.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        return StringSlice(self).count(substr)

    def lower(self) -> String:
        """Returns a copy of the string literal with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        return StringSlice(self).lower()

    def upper(self) -> String:
        """Returns a copy of the string literal with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        return StringSlice(self).upper()

    def ascii_rjust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string literal right justified in a string of specified width.

        Pads the string literal on the left with the specified fill character so
        that the total byte length of the resulting string equals `width`. If the
        original string literal is already longer than or equal to `width`,
        returns the string literal unchanged (as a `String`).

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A right-justified string of byte length `width`, or the original string
            literal (as a `String`) if its length is already greater than or
            equal to `width`.

        Examples:

        ```mojo
        var s = "hello"
        print(s.ascii_rjust(10))        # "     hello"
        print(s.ascii_rjust(10, "*"))   # "*****hello"
        print(s.ascii_rjust(3))         # "hello" (no padding)
        ```
        """
        return StringSlice(self).ascii_rjust(width, fillchar)

    def ascii_ljust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string literal left justified in a string of specified width.

        Pads the string literal on the right with the specified fill character so
        that the total byte length of the resulting string equals `width`. If the
        original string literal is already longer than or equal to `width`,
        returns the string literal unchanged (as a `String`).

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A left-justified string of (byte) length `width`, or the original string
            literal (as a `String`) if its length is already greater than or
            equal to `width`.

        Examples:

        ```mojo
        var s = "hello"
        print(s.ascii_ljust(10))        # "hello     "
        print(s.ascii_ljust(10, "*"))   # "hello*****"
        print(s.ascii_ljust(3))         # "hello" (no padding)
        ```
        """
        return StringSlice(self).ascii_ljust(width, fillchar)

    def ascii_center(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string literal center justified in a string of specified width.

        Pads the string literal on both sides with the specified fill character so
        that the total length of the resulting string equals `width`. If the
        padding needed is odd, the extra character goes on the right side. If the
        original string literal is already longer than or equal to `width`,
        returns the string literal unchanged (as a `String`).

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A center-justified string of length `width`, or the original string
            literal (as a `String`) if its length is already greater than or
            equal to `width`.

        Examples:

        ```mojo
        var s = "hello"
        print(s.ascii_center(10))        # "  hello   "
        print(s.ascii_center(11, "*"))   # "***hello***"
        print(s.ascii_center(3))         # "hello" (no padding)
        ```
        """
        return StringSlice(self).ascii_center(width, fillchar)

    def startswith(
        self, prefix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string literal starts with the specified prefix between
        start and end positions. Returns True if found and False otherwise.

        Args:
            prefix: The prefix to check.
            start: The start byte offset from which to check.
            end: The end byte offset from which to check.

        Returns:
            True if the `self[byte=start:end]` is prefixed by the input prefix.
        """
        return StringSlice(self).startswith(prefix, start, end)

    def endswith(
        self, suffix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string literal end with the specified suffix between
        start and end positions. Returns True if found and False otherwise.

        Args:
            suffix: The suffix to check.
            start: The start byte offset from which to check.
            end: The end byte offset from which to check.

        Returns:
            True if the `self[byte=start:end]` is suffixed by the input suffix.
        """
        return StringSlice(self).endswith(suffix, start, end)

    def is_ascii_digit(self) -> Bool:
        """Returns True if all characters in the string literal are ASCII digits.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits else False.
        """
        return StringSlice(self).is_ascii_digit()

    def isupper(self) -> Bool:
        """Returns True if all cased characters in the string literal are
        uppercase and there is at least one cased character.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all cased characters in the string literal are uppercase
            and there is at least one cased character, False otherwise.
        """
        return StringSlice(self).isupper()

    def islower(self) -> Bool:
        """Returns True if all cased characters in the string literal
        are lowercase and there is at least one cased character.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all cased characters in the string literal are lowercase
            and there is at least one cased character, False otherwise.
        """
        return StringSlice(self).islower()

    def strip(self) -> StaticString:
        """Returns a view of the string literal with leading and trailing
        whitespaces removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A `StaticString` into the literal with no leading or trailing
            whitespaces.
        """
        return self.lstrip().rstrip()

    def strip(self, chars: ImmStringSlice) -> StaticString:
        """Returns a view of the string literal with leading and trailing
        characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A `StaticString` into the literal with no leading or trailing
            characters.
        """

        return self.lstrip(chars).rstrip(chars)

    def rstrip(self, chars: ImmStringSlice) -> StaticString:
        """Returns a view of the string literal with trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A `StaticString` into the literal with no trailing characters.
        """
        return StringSlice(self).rstrip(chars)

    def rstrip(self) -> StaticString:
        """Returns a view of the string literal with trailing whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A `StaticString` into the literal with no trailing whitespaces.
        """
        return StringSlice(self).rstrip()

    def lstrip(self, chars: ImmStringSlice) -> StaticString:
        """Returns a view of the string literal with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A `StaticString` into the literal with no leading characters.
        """
        return StringSlice(self).lstrip(chars)

    def lstrip(self) -> StaticString:
        """Returns a view of the string literal with leading whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A `StaticString` into the literal with no leading whitespaces.
        """
        return StringSlice(self).lstrip()

    def codepoint_slices(self) -> CodepointSliceIter[ImmStaticOrigin]:
        """Iterate over the string's codepoints as immutable slices.

        Returns:
            An iterator of codepoint slices.
        """
        return StringSlice(self).codepoint_slices()

    def codepoint_slices_reversed(
        self,
    ) -> CodepointSliceIter[ImmStaticOrigin, False]:
        """Iterates backwards over the string literal, returning single-character slices.

        Each returned slice points to a single Unicode codepoint encoded in the
        underlying UTF-8 representation of this string literal, starting from the end
        and moving towards the beginning.

        Returns:
            A reversed iterator of references to the string literal elements.
        """
        return CodepointSliceIter[ImmStaticOrigin, False](StringSlice(self))

    def codepoints(self) -> CodepointsIter[ImmStaticOrigin]:
        """Iterate over the `Codepoint`s in this string constant.

        Returns:
            An iterator over successive `Codepoint` values.
        """
        return StringSlice(self).codepoints()

    def graphemes(self) -> GraphemeSliceIter[ImmStaticOrigin]:
        """Return an iterator over the grapheme clusters in this string.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen.

        Returns:
            An iterator yielding each grapheme cluster as a `StringSlice`.
        """
        return StringSlice(self).graphemes()

    def graphemes_reversed(
        self,
    ) -> GraphemeSliceIter[ImmStaticOrigin, False]:
        """Return an iterator over the grapheme clusters in this string,
        yielding them in reverse order.

        See `graphemes()` for the definition of a grapheme cluster.

        Returns:
            A reverse iterator yielding each grapheme cluster as a
            `StringSlice`.
        """
        return StringSlice(self).graphemes_reversed()

    def format[*Ts: Writable](self, *args: *Ts) -> String:
        """Produce a formatted string using the current string as a template.

        The template, or "format string" can contain literal text and/or
        replacement fields delimited with curly braces (`{}`). Returns a copy of
        the format string with the replacement fields replaced with string
        representations of the `args` arguments.

        For more information, see the discussion in the
        [`format` module](/docs/std/collections/string/format/).

        Parameters:
            Ts: The types of substitution values that implement `Writable`.

        Args:
            args: The substitution values.

        Returns:
            The template with the given values substituted.

        Example:

        ```mojo
        # Manual indexing:
        print("{0} {1} {0}".format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print("{} {}".format(True, "hello world")) # True hello world
        ```
        """
        var buffer = String()
        _FormatUtils.format_to_comptime[StaticString(Self())](buffer, *args)
        return buffer^

    def join[T: Copyable & Writable, //](self, elems: Span[T, _]) -> String:
        """Joins string elements using the current string as a delimiter.
        Defaults to writing to the stack if total bytes of `elems` is less than
        `buffer_size`, otherwise will allocate once to the heap and write
        directly into that. The `buffer_size` defaults to 4096 bytes to match
        the default page size on arm64 and x86-64.

        Parameters:
            T: The type of the elements. Must implement the `Copyable`,
                and `Writable` traits.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        # Materialize self
        var result = self
        return result.join(elems)

    @always_inline
    def split(self, sep: StringSlice) -> List[StaticString]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting a space
        _ = StringSlice("hello world").split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = StringSlice("hello,,world").split(",") # ["hello", "", "world"]
        # Splitting with starting or ending separators
        _ = StringSlice(",1,2,3,").split(",") # ['', '1', '2', '3', '']
        # Splitting with an empty separator
        _ = StringSlice("123").split("") # ['', '1', '2', '3', '']
        ```
        """
        return StringSlice(self).split(sep)

    @always_inline
    def split(self, sep: StringSlice, maxsplit: Int) -> List[StaticString]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:
        ```mojo
        # Splitting with maxsplit
        _ = StringSlice("1,2,3").split(",", maxsplit=1) # ['1', '2,3']
        # Splitting with starting or ending separators
        _ = StringSlice(",1,2,3,").split(",", maxsplit=1) # ['', '1,2,3,']
        # Splitting with an empty separator
        _ = StringSlice("123").split("", maxsplit=1) # ['', '123']
        ```
        """
        return StringSlice(self).split(sep, maxsplit=maxsplit)

    @always_inline
    def split(self, sep: NoneType = None) -> List[StaticString]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = StringSlice("      ").split() # []
        _ = StringSlice("").split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = StringSlice("      hello    world     ").split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = StringSlice(
            "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85world"
        ).split()  # ["hello", "world"]
        ```
        """
        return StringSlice(self).split(sep)

    @always_inline
    def split(
        self, sep: NoneType = None, *, maxsplit: Int
    ) -> List[StaticString]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:
        ```mojo
        # Splitting with maxsplit
        _ = StringSlice("1     2  3").split(maxsplit=1) # ['1', '2  3']
        ```
        """
        return StringSlice(self).split(sep, maxsplit=maxsplit)
