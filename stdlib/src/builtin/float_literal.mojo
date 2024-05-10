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
"""Implements the FloatLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

from builtin._math import Ceilable, CeilDivable, Floorable

# ===----------------------------------------------------------------------===#
# FloatLiteral
# ===----------------------------------------------------------------------===#


@value
@nonmaterializable(Float64)
@register_passable("trivial")
struct FloatLiteral(
    Absable,
    Boolable,
    Ceilable,
    CeilDivable,
    Comparable,
    Floorable,
    Intable,
    Stringable,
):
    """Mojo floating point literal type."""

    alias fp_type = __mlir_type.`!kgen.float_literal`
    var value: Self.fp_type
    """The underlying storage for the floating point value."""

    # ===------------------------------------------------------------------===#
    # Constructors
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(inout self, value: Self.fp_type):
        """Create a FloatLiteral value from a kgen.float_literal value.

        Args:
            value: The float value.
        """
        self.value = value

    @always_inline("nodebug")
    fn __init__(inout self, value: IntLiteral):
        """Convert an IntLiteral to a FloatLiteral value.

        Args:
            value: The IntLiteral value.
        """
        self.value = __mlir_op.`kgen.int_literal.to_float_literal`(value.value)

    alias nan = Self(__mlir_attr.`#kgen.float_literal<nan>`)
    alias infinity = Self(__mlir_attr.`#kgen.float_literal<inf>`)
    alias negative_infinity = Self(__mlir_attr.`#kgen.float_literal<neg_inf>`)
    alias negative_zero = Self(__mlir_attr.`#kgen.float_literal<neg_zero>`)

    @always_inline("nodebug")
    fn is_nan(self) -> Bool:
        """Return whether the FloatLiteral is nan.

        Since `nan == nan` is False, this provides a way to check for nan-ness.

        Returns:
            True, if the value is nan, False otherwise.
        """
        return __mlir_op.`kgen.float_literal.isa`[
            special = __mlir_attr.`#kgen<float_literal.special_values nan>`
        ](self.value)

    @always_inline("nodebug")
    fn is_neg_zero(self) -> Bool:
        """Return whether the FloatLiteral is negative zero.

        Since `FloatLiteral.negative_zero == 0.0` is True, this provides a way
        to check if the FloatLiteral is negative zero.

        Returns:
            True, if the value is negative zero, False otherwise.
        """
        return __mlir_op.`kgen.float_literal.isa`[
            special = __mlir_attr.`#kgen<float_literal.special_values neg_zero>`
        ](self.value)

    @always_inline("nodebug")
    fn _is_normal(self) -> Bool:
        """Return whether the FloatLiteral is a normal (i.e. not special) value.

        Returns:
            True, if the value is a normal float, False otherwise.
        """
        return __mlir_op.`kgen.float_literal.isa`[
            special = __mlir_attr.`#kgen<float_literal.special_values normal>`
        ](self.value)

    # ===------------------------------------------------------------------===#
    # Conversion Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Get the float as a string.

        Returns:
            A string representation.
        """
        return str(Float64(self))

    @always_inline("nodebug")
    fn __int_literal__(self) -> IntLiteral:
        """Casts the floating point value to an IntLiteral. If there is a
        fractional component, then the value is truncated towards zero.

        Eg. `(4.5).__int_literal__()` returns `4`, and `(-3.7).__int_literal__()`
        returns `-3`.

        Returns:
            The value as an integer.
        """
        return IntLiteral(
            __mlir_op.`kgen.float_literal.to_int_literal`(self.value)
        )

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Converts the FloatLiteral value to an Int. If there is a fractional
        component, then the value is truncated towards zero.

        Eg. `(4.5).__int__()` returns `4`, and `(-3.7).__int__()` returns `-3`.

        Returns:
            The value as an integer.
        """
        return self.__int_literal__().__int__()

    # ===------------------------------------------------------------------===#
    # Unary Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """A FloatLiteral value is true if it is non-zero.

        Returns:
            True if non-zero.
        """
        return self != 0.0

    @always_inline("nodebug")
    fn __neg__(self) -> FloatLiteral:
        """Return the negation of the FloatLiteral value.

        Returns:
            The negated FloatLiteral value.
        """
        return self * Self(-1)

    @always_inline("nodebug")
    fn __abs__(self) -> Self:
        """Return the absolute value of the FloatLiteral.

        Returns:
            The absolute value.
        """
        if self > 0:
            return self
        return -self

    @always_inline("nodebug")
    fn __floor__(self) -> Self:
        """Return the floor value of the FloatLiteral.

        Returns:
            The floor value.
        """

        # Handle special values first.
        if not self._is_normal():
            return self

        # __int_literal__ rounds towards zero, so it's correct for integers and
        # positive values.
        var truncated: IntLiteral = self.__int_literal__()

        # Ensure this equality doesn't hit any implicit conversions.
        if self >= 0 or self.__eq__(Self(truncated)):
            return truncated
        return truncated - 1

    @always_inline("nodebug")
    fn __ceil__(self) -> Self:
        """Return the ceiling value of the FloatLiteral.

        Returns:
            The ceiling value.
        """

        # Handle special values first.
        if not self._is_normal():
            return self

        # __int_literal__ rounds towards zero, so it's correct for integers and
        # negative values.
        var truncated: IntLiteral = self.__int_literal__()

        # Ensure this equality doesn't hit any implicit conversions.
        if self <= 0 or self.__eq__(Self(truncated)):
            return truncated
        return truncated + 1

    # TODO: implement __round__

    # ===------------------------------------------------------------------===#
    # Arithmetic Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __add__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Add two FloatLiterals.

        Args:
            rhs: The value to add.

        Returns:
            The sum of the two values.
        """
        return __mlir_op.`kgen.float_literal.binop`[
            oper = __mlir_attr.`#kgen<float_literal.binop_kind add>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __sub__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Subtract two FloatLiterals.

        Args:
            rhs: The value to subtract.

        Returns:
            The difference of the two values.
        """
        return __mlir_op.`kgen.float_literal.binop`[
            oper = __mlir_attr.`#kgen<float_literal.binop_kind sub>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __mul__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Multiply two FloatLiterals.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the two values.
        """
        return __mlir_op.`kgen.float_literal.binop`[
            oper = __mlir_attr.`#kgen<float_literal.binop_kind mul>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Divide two FloatLiterals.

        Args:
            rhs: The value to divide.

        Returns:
            The quotient of the two values.
        """
        # TODO - Python raises an error on divide by 0.0 or -0.0
        return __mlir_op.`kgen.float_literal.binop`[
            oper = __mlir_attr.`#kgen<float_literal.binop_kind truediv>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Self) -> Self:
        """Returns self divided by rhs, rounded down to the nearest integer.

        Args:
            rhs: The divisor value.

        Returns:
            `floor(self / rhs)` value.
        """
        return self.__truediv__(rhs).__floor__()

    @always_inline("nodebug")
    fn __rfloordiv__(self, rhs: Self) -> Self:
        """Returns rhs divided by self, rounded down to the nearest integer.

        Args:
            rhs: The value to be divided by self.

        Returns:
            `floor(rhs / self)` value.
        """
        return rhs // self

    # TODO - maybe __mod__?
    # TODO - maybe __pow__?

    # ===------------------------------------------------------------------===#
    # In-place Arithmetic Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __iadd__(inout self, rhs: FloatLiteral):
        """In-place addition operator.

        Args:
            rhs: The value to add.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(inout self, rhs: FloatLiteral):
        """In-place subtraction operator.

        Args:
            rhs: The value to subtract.
        """
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(inout self, rhs: FloatLiteral):
        """In-place multiplication operator.

        Args:
            rhs: The value to multiply.
        """
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(inout self, rhs: FloatLiteral):
        """In-place division.

        Args:
            rhs: The value to divide.
        """
        self = self / rhs

    # ===------------------------------------------------------------------===#
    # Reversed Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __radd__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed addition operator.

        Args:
            rhs: The value to add.

        Returns:
            The sum of this and the given value.
        """
        return rhs + self

    @always_inline("nodebug")
    fn __rsub__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed subtraction operator.

        Args:
            rhs: The value to subtract from.

        Returns:
            The result of subtracting this from the given value.
        """
        return rhs - self

    @always_inline("nodebug")
    fn __rmul__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed multiplication operator.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the given number and this.
        """
        return rhs * self

    @always_inline("nodebug")
    fn __rtruediv__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed division.

        Args:
            rhs: The value to be divided by this.

        Returns:
            The result of dividing the given value by this.
        """
        return rhs / self

    # ===------------------------------------------------------------------===#
    # Comparison Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __eq__(self, rhs: FloatLiteral) -> Bool:
        """Compare for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return __mlir_op.`kgen.float_literal.cmp`[
            pred = __mlir_attr.`#kgen<float_literal.cmp_pred eq>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ne__(self, rhs: FloatLiteral) -> Bool:
        """Compare for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return __mlir_op.`kgen.float_literal.cmp`[
            pred = __mlir_attr.`#kgen<float_literal.cmp_pred ne>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __lt__(self, rhs: FloatLiteral) -> Bool:
        """Less than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than `rhs`.
        """
        return __mlir_op.`kgen.float_literal.cmp`[
            pred = __mlir_attr.`#kgen<float_literal.cmp_pred lt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __le__(self, rhs: FloatLiteral) -> Bool:
        """Less than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than or equal to `rhs`.
        """
        return __mlir_op.`kgen.float_literal.cmp`[
            pred = __mlir_attr.`#kgen<float_literal.cmp_pred le>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __gt__(self, rhs: FloatLiteral) -> Bool:
        """Greater than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than `rhs`.
        """
        return __mlir_op.`kgen.float_literal.cmp`[
            pred = __mlir_attr.`#kgen<float_literal.cmp_pred gt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ge__(self, rhs: FloatLiteral) -> Bool:
        """Greater than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than or equal to `rhs`.
        """
        return __mlir_op.`kgen.float_literal.cmp`[
            pred = __mlir_attr.`#kgen<float_literal.cmp_pred ge>`
        ](self.value, rhs.value)
