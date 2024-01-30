# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the StringLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

from debug.lldb import lldb_formatter_wrapping_type
from memory.unsafe import DTypePointer
from memory import memcmp
from collections.vector import CollectionElement

# ===----------------------------------------------------------------------===#
# StringLiteral
# ===----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct StringLiteral(Sized, Stringable, CollectionElement, Hashable, Boolable):
    """This type represents a string literal.

    String literals are all null-terminated for compatibility with C APIs, but
    this is subject to change. String literals store their length as an integer,
    and this does not include the null terminator.
    """

    alias type = __mlir_type.`!kgen.string`

    var value: Self.type
    """The underlying storage for the string literal."""

    @always_inline("nodebug")
    fn __init__(value: Self.type) -> Self:
        """Create a string literal from a builtin string type.

        Args:
            value: The string value.

        Returns:
            A string literal object.
        """
        return StringLiteral {value: value}

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the string length.

        Returns:
            The length of this StringLiteral.
        """
        return __mlir_op.`pop.string.size`(self.value)

    @always_inline("nodebug")
    fn data(self) -> DTypePointer[DType.int8]:
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
        let length = len(self)
        if length != len(rhs):
            return False

        return memcmp(self.data(), rhs.data(), length) == 0

    @always_inline("nodebug")
    fn __ne__(self, rhs: StringLiteral) -> Bool:
        """Compare two string literals for inequality.

        Args:
            rhs: The string to compare.

        Returns:
            True if they are not equal.
        """
        return not self == rhs

    fn __hash__(self) -> Int:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.data(), len(self))

    fn __str__(self) -> String:
        """Convert the string literal to a string.

        Returns:
            A new string.
        """
        return self
