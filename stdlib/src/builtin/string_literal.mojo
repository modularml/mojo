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

from sys.ffi import C_char

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
    Comparable,
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
    # Trait impelemntations
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
        return self._byte_length()

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

    fn __str__(self) -> String:
        """Convert the string literal to a string.

        Returns:
            A new string.
        """
        var string = String()
        var length: Int = __mlir_op.`pop.string.size`(self.value)
        var buffer = String._buffer_type()
        var new_capacity = length + 1
        buffer._realloc(new_capacity)
        buffer.size = new_capacity
        var uint8Ptr = __mlir_op.`pop.pointer.bitcast`[
            _type = __mlir_type.`!kgen.pointer<scalar<ui8>>`
        ](__mlir_op.`pop.string.address`(self.value))
        var data: DTypePointer[DType.uint8] = DTypePointer[DType.uint8](
            uint8Ptr
        )
        memcpy(DTypePointer(buffer.data), data, length)
        (buffer.data + length).init_pointee_move(0)
        string._buffer = buffer^
        return string

    fn __repr__(self) -> String:
        """Return a representation of the `StringLiteral` instance.

        You don't need to call this method directly, use `repr("...")` instead.

        Returns:
            A new representation of the string.
        """
        return self.__str__().__repr__()

    fn __hash__(self) -> Int:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.unsafe_ptr(), len(self))

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn _byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this StringLiteral in bytes.
        """
        return __mlir_op.`pop.string.size`(self.value)

    @always_inline("nodebug")
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Get raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        var ptr = DTypePointer[DType.int8](
            __mlir_op.`pop.string.address`(self.value)
        )

        # TODO(MSTDL-555):
        #   Remove bitcast after changing pop.string.address
        #   return type.
        return UnsafePointer[Int8]._from_dtype_ptr(ptr).bitcast[UInt8]()

    fn unsafe_cstr_ptr(self) -> UnsafePointer[C_char]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be NUL terminated, and not null.

        Returns:
            The pointer to the underlying memory.
        """
        return self.unsafe_ptr().bitcast[C_char]()

    @always_inline("nodebug")
    fn as_uint8_ptr(self) -> DTypePointer[DType.uint8]:
        """Get raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        return self.unsafe_ptr().bitcast[UInt8]()

    @always_inline
    fn as_string_slice(self) -> StringSlice[ImmutableStaticLifetime]:
        """Returns a string slice of this static string literal.

        Returns:
            A string slice pointing to this static string literal.
        """

        var bytes = self.as_bytes_slice()

        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in StringLiteral so this is actually
        #   guaranteed to be valid.
        return StringSlice[ImmutableStaticLifetime](unsafe_from_utf8=bytes)

    @always_inline
    fn as_bytes_slice(self) -> Span[UInt8, ImmutableStaticLifetime]:
        """
        Returns a contiguous slice of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        var ptr = self.unsafe_ptr()

        return Span[UInt8, ImmutableStaticLifetime](
            unsafe_ptr=ptr,
            len=self._byte_length(),
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
