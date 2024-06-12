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
"""Implements the UInt class.

These are Mojo built-ins, so you don't need to import them.
"""


@lldb_formatter_wrapping_type
@value
@register_passable("trivial")
struct UInt(Stringable, Representable):
    """This type represents an unsigned integer.

    An unsigned integer is represents a positive integral number.

    The size of this unsigned integer is platform-dependent.

    If you wish to use a fixed size unsigned integer, consider using
    `UInt8`, `UInt16`, `UInt32`, or `UInt64`.
    """

    var value: __mlir_type.index
    """The underlying storage for the integer value.

    Note that it is the same type as the `Int.value` field.
    MLIR doesn't differentiate between signed and unsigned integers
    when it comes to storing them with the index dialect. 
    The difference is in the operations that are performed on them,
    which have signed and unsigned variants.
    """

    @always_inline("nodebug")
    fn __init__(inout self):
        """Default constructor that produces zero."""
        self.value = __mlir_op.`index.constant`[value = __mlir_attr.`0:index`]()

    @always_inline("nodebug")
    fn __init__(inout self, value: __mlir_type.index):
        """Construct Int from the given index value.

        Args:
            value: The init value.
        """
        self.value = value

    @always_inline("nodebug")
    fn __init__(inout self, value: IntLiteral):
        """Construct Int from the given IntLiteral value.

        Args:
            value: The init value.
        """
        # TODO: Find a way to convert directly without using UInt64.
        # This is because the existing
        # __mlir_op.`kgen.int_literal.convert`
        # in IntLiteral.__as_mlir_index()
        # assumes that the index represents an signed integer.
        # We need a variant for unsigned integers.
        # Change when https://github.com/modularml/mojo/issues/2933 is fixed
        self.value = int(UInt64(value)).value

    @always_inline("nodebug")
    fn __lt__(self, rhs: UInt) -> Bool:
        """Compare this UInt to the RHS using LT comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is less-than the RHS UInt and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ult>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __le__(self, rhs: UInt) -> Bool:
        """Compare this UInt to the RHS using LE comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is less-or-equal than the RHS UInt and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ule>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __gt__(self, rhs: UInt) -> Bool:
        """Compare this UInt to the RHS using GT comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is greater-than the RHS UInt and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ugt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ge__(self, rhs: UInt) -> Bool:
        """Compare this UInt to the RHS using GE comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is greater-or-equal than the RHS UInt and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate uge>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Convert this UInt to a string.

        A small example.
        ```mojo
        var x = UInt(50)
        var x_as_string = str(x)  # x_as_string = "50"
        ```

        Returns:
            The string representation of this UInt.
        """
        return str(UInt64(self))

    @always_inline("nodebug")
    fn __repr__(self) -> String:
        """Convert this UInt to a string.

        A small example.
        ```mojo
        var x = UInt(50)
        var x_as_string = repr(x)  # x_as_string = "UInt(50)"
        ```

        Returns:
            The string representation of this UInt.
        """
        return "UInt(" + str(self) + ")"
