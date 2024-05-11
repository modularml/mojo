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

from memory import DTypePointer

from utils import StringRef
from utils._visualizers import lldb_formatter_wrapping_type
from utils._format import Formattable, Formatter

from .string import _atol

# ===----------------------------------------------------------------------===#
# StringLiteral
# ===----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct StringLiteral(
    Sized,
    IntableRaising,
    Stringable,
    Representable,
    KeyElement,
    Boolable,
    Formattable,
):
    """This type represents a string literal.

    String literals are all null-terminated for compatibility with C APIs, but
    this is subject to change. String literals store their length as an integer,
    and this does not include the null terminator.
    """

    alias type = __mlir_type.`!kgen.string`

    var value: Self.type
    """The underlying storage for the string literal."""

    @always_inline("nodebug")
    fn __init__(inout self, value: Self.type):
        """Create a string literal from a builtin string type.

        Args:
            value: The string value.
        """
        self.value = value

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the string length.

        Returns:
            The length of this StringLiteral.
        """
        return __mlir_op.`pop.string.size`(self.value)

    @always_inline("nodebug")
    fn unsafe_ptr(self) -> DTypePointer[DType.int8]:
        """Get raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        return __mlir_op.`pop.string.address`(self.value)

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Convert the string to a bool value.

        Returns:
            True if the string is not empty.
        """
        return len(self) != 0

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
        var length = len(self)
        if length != len(rhs):
            return False

        return _memcmp(self.unsafe_ptr(), rhs.unsafe_ptr(), length) == 0

    @always_inline("nodebug")
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Compare two string literals for inequality.

        Args:
            rhs: The string to compare.

        Returns:
            True if they are not equal.
        """
        return not self == rhs

    @always_inline("nodebug")
    fn __lt__(self, rhs: StringLiteral) -> Bool:
        """Compare this StringLiteral to the RHS using LT comparison.

        Args:
            rhs: The other StringLiteral to compare against.

        Returns:
            True if this StringLiteral is strictly less than the RHS StringLiteral and False otherwise.
        """
        var len1 = len(self)
        var len2 = len(rhs)

        if len1 < len2:
            return _memcmp(self.unsafe_ptr(), rhs.unsafe_ptr(), len1) <= 0
        else:
            return _memcmp(self.unsafe_ptr(), rhs.unsafe_ptr(), len2) < 0

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

    fn __hash__[H: Hasher](self, inout hasher: H):
        """Update hasher with this string literal value."""
        var size = len(self)
        size.__hash__(hasher)
        hasher._update_with_bytes(
            self.unsafe_ptr().bitcast[DType.uint8](), size
        )

    fn __str__(self) -> String:
        """Convert the string literal to a string.

        Returns:
            A new string.
        """
        return self

    fn __repr__(self) -> String:
        """Return a representation of the `StringLiteral` instance.

        You don't need to call this method directly, use `repr("...")` instead.

        Returns:
            A new representation of the string.
        """
        return self.__str__().__repr__()

    fn format_to(self, inout writer: Formatter):
        """
        Formats this string literal to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        # SAFETY:
        #   Safe because `self` is borrowed, so the lifetime of this
        #   StringRef extends beyond this function.
        writer.write_str(StringRef(self))

    fn __contains__(self, substr: StringLiteral) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return substr in StringRef(self)

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

    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.

        For example, `int("19")` returns `19`. If the given string cannot be parsed
        as an integer value, an error is raised. For example, `int("hi")` raises an
        error.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return _atol(self)


# Use a local memcmp rather than memory.memcpy to avoid #31139 and #25100.
@always_inline("nodebug")
fn _memcmp(
    s1: DTypePointer[DType.int8], s2: DTypePointer[DType.int8], count: Int
) -> Int:
    for i in range(count):
        var s1i = s1[i]
        var s2i = s2[i]
        if s1i == s2i:
            continue
        if s1i > s2i:
            return 1
        return -1
    return 0
