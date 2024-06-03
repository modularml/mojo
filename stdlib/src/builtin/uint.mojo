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
struct UInt:
    var value: __mlir_type.index


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
        self.value = int(UInt64(value)).value

    @always_inline("nodebug")
    fn __init__(inout self, value: __mlir_type.`!pop.scalar<ui64>`):
        """Construct Int from the given Int64 value.

        Args:
            value: The init value.
        """
        self.value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.index](
            __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
                value
            )
        )

    @always_inline("nodebug")
    fn __init__(inout self, value: __mlir_type.`!pop.scalar<index>`):
        """Construct Int from the given Index value.

        Args:
            value: The init value.
        """
        self.value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.index](
            __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
                value
            )
        )

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self.value
    
    
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using LT comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is less-than the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ult>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using LE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is less-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ule>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using EQ comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using NE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is non-equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using GT comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is greater-than the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ugt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using GE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is greater-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate uge>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Convert this Int to a string.

        Returns:
            The string representation of this Int.
        """
        return str(UInt64(self.value))