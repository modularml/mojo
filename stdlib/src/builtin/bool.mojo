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
"""Implements the Bool class.

These are Mojo built-ins, so you don't need to import them.
"""

from utils._visualizers import lldb_formatter_wrapping_type


trait Boolable:
    """The `Boolable` trait describes a type that can be converted to a bool.

    This trait requires the type to implement the `__bool__()` method. For
    example:

    ```mojo
    @value
    struct Foo(Boolable):
        var val: Bool

        fn __bool__(self) -> Bool:
            return self.val
    ```
    """

    fn __bool__(self) -> Bool:
        """Get the boolean representation of the value.

        Returns:
            The boolean representation of the value.
        """
        ...


@lldb_formatter_wrapping_type
@value
@register_passable("trivial")
struct Bool(
    Stringable, CollectionElement, Boolable, EqualityComparable, Intable
):
    """The primitive Bool scalar value used in Mojo."""

    var value: __mlir_type.`!pop.scalar<bool>`
    """The underlying storage of the boolean value."""

    @always_inline("nodebug")
    fn __init__(value: __mlir_type.i1) -> Bool:
        """Construct a Bool value given a __mlir_type.i1 value.

        Args:
            value: The initial __mlir_type.i1 value.

        Returns:
            The constructed Bool value.
        """
        return __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<bool>`
        ](value)

    @always_inline("nodebug")
    fn __init__[width: Int](value: SIMD[DType.bool, width]) -> Bool:
        """Construct a Bool value given a SIMD value.

        If there is more than a single element in the SIMD value, then value is
        reduced using the and operator.

        Parameters:
            width: SIMD width.

        Args:
            value: The initial SIMD value.

        Returns:
            The constructed Bool value.
        """
        return value.__bool__()

    @always_inline("nodebug")
    fn __init__[boolable: Boolable](value: boolable) -> Bool:
        """Implicitly convert a Boolable value to a Bool.

        Parameters:
            boolable: The Boolable type.

        Args:
            value: The boolable value.

        Returns:
            The constructed Bool value.
        """
        return value.__bool__()

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Convert to Bool.

        Returns:
            This value.
        """
        return self

    @always_inline("nodebug")
    fn __mlir_i1__(self) -> __mlir_type.i1:
        """Convert this Bool to __mlir_type.i1.

        This method is a special hook used by the compiler to test boolean
        objects in control flow conditions.  It should be implemented by Bool
        but not other general boolean convertible types (they should implement
        `__bool__` instead).

        Returns:
            The underlying value for the Bool.
        """
        return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i1](
            self.value
        )

    fn __str__(self) -> String:
        """Get the bool as a string.

        Returns:
            A string representation.
        """
        return "True" if self else "False"

    @always_inline("nodebug")
    fn __eq__(self, rhs: Bool) -> Bool:
        """Compare this Bool to RHS.

        Performs an equality comparison between the Bool value and the argument.
        This method gets invoked when a user uses the `==` infix operator.

        Args:
            rhs: The rhs value of the equality statement.

        Returns:
            True if the two values match and False otherwise.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: Bool) -> Bool:
        """Compare this Bool to RHS.

        Performs a non-equality comparison between the Bool value and the
        argument. This method gets invoked when a user uses the `!=` infix
        operator.

        Args:
            rhs: The rhs value of the non-equality statement.

        Returns:
            False if the two values do match and True otherwise.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __and__(self, rhs: Bool) -> Bool:
        """Compute `self & rhs`.

        Bitwise and's the Bool value with the argument. This method gets invoked
        when a user uses the `and` infix operator.

        Args:
            rhs: The rhs value of the and statement.

        Returns:
            `self & rhs`.
        """
        return __mlir_op.`pop.and`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __or__(self, rhs: Bool) -> Bool:
        """Compute `self | rhs`.

        Bitwise or's the Bool value with the argument. This method gets invoked
        when a user uses the `or` infix operator.

        Args:
            rhs: The rhs value of the or statement.

        Returns:
            `self | rhs`.
        """
        return __mlir_op.`pop.or`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __xor__(self, rhs: Bool) -> Bool:
        """Compute `self ^ rhs`.

        Bitwise Xor's the Bool value with the argument. This method gets invoked
        when a user uses the `^` infix operator.

        Args:
            rhs: The rhs value of the xor statement.

        Returns:
            `self ^ rhs`.
        """
        return __mlir_op.`pop.xor`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __invert__(self) -> Bool:
        """Inverts the Bool value.

        Returns:
            True if the object is false and False otherwise.
        """
        var true = __mlir_op.`kgen.param.constant`[
            _type = __mlir_type.`!pop.scalar<bool>`,
            value = __mlir_attr.`#pop.simd<true> : !pop.scalar<bool>`,
        ]()
        return __mlir_op.`pop.xor`(self.value, true)

    @always_inline("nodebug")
    fn __rand__(self, value: Bool) -> Bool:
        """Return `value & self`.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        return value & self

    @always_inline("nodebug")
    fn __ror__(self, value: Bool) -> Bool:
        """Return `value | self`.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        return value | self

    @always_inline("nodebug")
    fn __rxor__(self, value: Bool) -> Bool:
        """Return `value ^ self`.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        return value ^ self

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Convert this Bool to an integer.

        Returns:
            1 if the Bool is True, 0 otherwise.
        """
        return Int(
            __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
                self.value
            )
        )


@always_inline
fn bool(value: None) -> Bool:
    """Get the bool representation of the `None` type.

    Args:
        value: The object to get the bool representation of.

    Returns:
        The bool representation of the object.
    """
    return False


@always_inline
fn bool[T: Boolable](value: T) -> Bool:
    """Get the bool representation of the object.

    Parameters:
        T: The type of the object.

    Args:
        value: The object to get the bool representation of.

    Returns:
        The bool representation of the object.
    """
    return value.__bool__()
