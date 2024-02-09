# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the FloatLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------===#
# FloatLiteral
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct FloatLiteral(Intable, Stringable, Boolable):
    """Mojo floating point literal type."""

    alias fp_type = __mlir_type.`!pop.scalar<f64>`
    var value: Self.fp_type
    """The underlying storage for the floating point value."""

    # ===------------------------------------------------------------------===#
    # Constructors
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(value: FloatLiteral) -> Self:
        """Forwarding constructor.

        Args:
           value: The double value.

        Returns:
           The value.
        """
        return value

    @always_inline("nodebug")
    fn __init__(value: __mlir_type.f64) -> Self:
        """Create a double value from a builtin MLIR f64 value.

        Args:
            value: The underlying MLIR value.

        Returns:
            A double value.
        """
        return __mlir_op.`pop.cast_from_builtin`[_type = Self.fp_type](value)

    @always_inline("nodebug")
    fn __init__(value: Int) -> Self:
        """Convert an integer to a double value.

        Args:
            value: The integer value.

        Returns:
            The integer value as a double.
        """
        let v0 = __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<index>`
        ](value.value)
        return __mlir_op.`pop.cast`[_type = Self.fp_type](v0)

    @always_inline("nodebug")
    fn __init__(value: IntLiteral) -> Self:
        """Convert an IntLiteral to a double value.

        Args:
            value: The IntLiteral value.

        Returns:
            The integer value as a double.
        """
        return Self(Int(value))

    # ===------------------------------------------------------------------===#
    # Conversion Operators
    # ===------------------------------------------------------------------===#

    fn __str__(self) -> String:
        """Get the float as a string.

        Returns:
            A string representation.
        """
        return Float64(self)

    @always_inline("nodebug")
    fn to_int(self) -> Int:
        """Casts to the floating point value to an Int. If there is a fractional
        component, then the value is truncated towards zero.

        Returns:
            The value as an integer.
        """
        return self.__int__()

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Casts to the floating point value to an Int. If there is a fractional
        component, then the value is truncated towards zero.

        Returns:
            The value as an integer.
        """
        return __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
            (self).value
        )

    # ===------------------------------------------------------------------===#
    # Unary Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """A double value is true if it is non-zero.

        Returns:
            True if non-zero.
        """
        return self != 0.0

    @always_inline("nodebug")
    fn __neg__(self) -> FloatLiteral:
        """Return the negation of the double value.

        Returns:
            The negated double value.
        """
        return __mlir_op.`pop.neg`(self.value)

    # ===------------------------------------------------------------------===#
    # Arithmetic Operators
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __add__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Add two doubles.

        Args:
            rhs: The value to add.

        Returns:
            The sum of the two values.
        """
        return __mlir_op.`pop.add`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __sub__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Subtract two doubles.

        Args:
            rhs: The value to subtract.

        Returns:
            The difference of the two values.
        """
        return __mlir_op.`pop.sub`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __mul__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Multiply two doubles.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the two values.
        """
        return __mlir_op.`pop.mul`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __truediv__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Divide two doubles.

        Args:
            rhs: The value to divide.

        Returns:
            The quotient of the two values.
        """
        return __mlir_op.`pop.div`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Divide two doubles and round towards negative infinity.

        Args:
            rhs: The value to divide.

        Returns:
            The quotient of the two values rounded towards negative infinity.
        """
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = "llvm.floor".value, _type = Self.fp_type
        ]((self / rhs).value)

    @always_inline("nodebug")
    fn __mod__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Compute the remainder of dividing by a value.

        Args:
            rhs: The divisor.

        Returns:
            The remainder of the division operation.
        """
        var remainder: FloatLiteral = __mlir_op.`pop.rem`(self.value, rhs.value)
        if (self < 0.0) ^ (rhs < 0.0):
            remainder += rhs
        return remainder

    @always_inline("nodebug")
    fn __pow__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Compute the power.

        Args:
            rhs: The exponent.

        Returns:
            The current value raised to the exponent.
        """
        let lhs = __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = "llvm.fabs".value, _type = Self.fp_type
        ](self.value)
        let result = __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = "llvm.pow".value, _type = Self.fp_type
        ](lhs, rhs.value)
        return __mlir_op.`pop.call_llvm_intrinsic`[
            intrin = "llvm.copysign".value, _type = Self.fp_type
        ](result, self.value)

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

    @always_inline("nodebug")
    fn __ifloordiv__(inout self, rhs: FloatLiteral):
        """In-place floor division.

        Args:
            rhs: The value to divide.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(inout self, rhs: FloatLiteral):
        """In-place remainder.

        Args:
            rhs: The divisor.
        """
        self = self % rhs

    @always_inline("nodebug")
    fn __ipow__(inout self, rhs: FloatLiteral):
        """In-place power.

        Args:
            rhs: The exponent.
        """
        self = self**rhs

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

    @always_inline("nodebug")
    fn __rfloordiv__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed floor division.

        Args:
            rhs: The value to be floor-divided by this.

        Returns:
            The result of dividing the given value by this, modulo any
            remainder.
        """
        return rhs // self

    @always_inline("nodebug")
    fn __rmod__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed remainder.

        Args:
            rhs: The divisor.

        Returns:
            The remainder after dividing the given value by this.
        """
        return rhs % self

    @always_inline("nodebug")
    fn __rpow__(self, rhs: FloatLiteral) -> FloatLiteral:
        """Reversed power.

        Args:
            rhs: The number to be raised to the power of this.

        Returns:
            The result of raising the given number by this value.
        """
        return rhs**self

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
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred eq>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __ne__(self, rhs: FloatLiteral) -> Bool:
        """Compare for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ne>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __lt__(self, rhs: FloatLiteral) -> Bool:
        """Less than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than `rhs`.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred lt>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __le__(self, rhs: FloatLiteral) -> Bool:
        """Less than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than or equal to `rhs`.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred le>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __gt__(self, rhs: FloatLiteral) -> Bool:
        """Greater than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than `rhs`.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred gt>`](
            self.value, rhs.value
        )

    @always_inline("nodebug")
    fn __ge__(self, rhs: FloatLiteral) -> Bool:
        """Greater than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than or equal to `rhs`.
        """
        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](
            self.value, rhs.value
        )
