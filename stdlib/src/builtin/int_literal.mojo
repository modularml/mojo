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
"""Implements the IntLiteral class."""

from math import Ceilable, CeilDivable, Floorable, Truncable


@value
@nonmaterializable(Int)
@register_passable("trivial")
struct IntLiteral(
    Absable,
    Ceilable,
    CeilDivable,
    Comparable,
    Floorable,
    ImplicitlyBoolable,
    Indexer,
    Intable,
    Roundable,
    Stringable,
    Truncable,
):
    """This type represents a static integer literal value with
    infinite precision.  They can't be materialized at runtime and
    must be lowered to other integer types (like Int), but allow for
    compile-time operations that would overflow on Int and other fixed
    precision integer types.
    """

    # Fields
    alias _mlir_type = __mlir_type.`!kgen.int_literal`

    var _value: Self._mlir_type
    """The underlying storage for the integer value."""

    alias _one = IntLiteral(
        __mlir_attr.`#kgen.int_literal<1> : !kgen.int_literal`
    )

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(out self):
        """Default constructor."""
        self._value = __mlir_attr.`#kgen.int_literal<0> : !kgen.int_literal`

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!kgen.int_literal`):
        """Construct IntLiteral from the given mlir !kgen.int_literal value.

        Args:
            value: The init value.
        """
        self._value = value

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using LT comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is less-than the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred lt>`
        ](self._value, rhs._value)

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using LE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is less-or-equal than the RHS IntLiteral and False
            otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred le>`
        ](self._value, rhs._value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using EQ comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is equal to the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred eq>`
        ](self._value, rhs._value)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using NE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is non-equal to the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred ne>`
        ](self._value, rhs._value)

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using GT comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is greater-than the RHS IntLiteral and False otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred gt>`
        ](self._value, rhs._value)

    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Compare this IntLiteral to the RHS using GE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is greater-or-equal than the RHS IntLiteral and False
            otherwise.
        """
        return __mlir_op.`kgen.int_literal.cmp`[
            pred = __mlir_attr.`#kgen<int_literal.cmp_pred ge>`
        ](self._value, rhs._value)

    @always_inline("nodebug")
    fn __pos__(self) -> Self:
        """Return +self.

        Returns:
            The +self value.
        """
        return self

    @always_inline("nodebug")
    fn __neg__(self) -> Self:
        """Return -self.

        Returns:
            The -self value.
        """
        return Self() - self

    @always_inline("nodebug")
    fn __divmod__(self, rhs: Self) -> Tuple[Self, Self]:
        """Return the quotient and remainder of the division of self by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The quotient and remainder of the division.
        """
        var quotient: Self = self.__floordiv__(rhs)
        var remainder: Self = self - (quotient * rhs)
        return quotient, remainder

    @always_inline("nodebug")
    fn __invert__(self) -> Self:
        """Return ~self.

        Returns:
            The ~self value.
        """
        return self ^ (Self() - Self._one)

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind add>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __sub__(self, rhs: Self) -> Self:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind sub>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __mul__(self, rhs: Self) -> Self:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind mul>`
            ](self._value, rhs._value)
        )

    # TODO: implement __pow__

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Self) -> Self:
        """Return `self // rhs`.

        Args:
            rhs: The value to divide with.

        Returns:
            `self // rhs` value.
        """
        if rhs == Self():
            # this should raise an exception.
            return Self()
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind floordiv>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __mod__(self, rhs: Self) -> Self:
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        if rhs == Self():
            # this should raise an exception.
            return Self()
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind mod>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __lshift__(self, rhs: Self) -> Self:
        """Return `self << rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self << rhs`.
        """
        if rhs < Self():
            # this should raise an exception.
            return Self()
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind lshift>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __rshift__(self, rhs: Self) -> Self:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        if rhs < Self():
            # this should raise an exception.
            return Self()
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind rshift>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __and__(self, rhs: Self) -> Self:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind and>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __xor__(self, rhs: Self) -> Self:
        """Return `self ^ rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind xor>`
            ](self._value, rhs._value)
        )

    @always_inline("nodebug")
    fn __or__(self, rhs: Self) -> Self:
        """Return `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return Self(
            __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind or>`
            ](self._value, rhs._value)
        )

    # ===----------------------------------------------------------------------===#
    # In place operations.
    # ===----------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __iadd__(inout self, rhs: Self):
        """Compute `self + rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(inout self, rhs: Self):
        """Compute `self - rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(inout self, rhs: Self):
        """Compute self*rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self * rhs

    @always_inline("nodebug")
    fn __ifloordiv__(inout self, rhs: Self):
        """Compute self//rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __ilshift__(inout self, rhs: Self):
        """Compute `self << rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(inout self, rhs: Self):
        """Compute `self >> rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self >> rhs

    @always_inline("nodebug")
    fn __iand__(inout self, rhs: Self):
        """Compute `self & rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(inout self, rhs: Self):
        """Compute `self ^ rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(inout self, rhs: Self):
        """Compute self|rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self | rhs

    # ===----------------------------------------------------------------------===#
    # Reversed operations
    # ===----------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __radd__(self, value: Self) -> Self:
        """Return `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        return self + value

    @always_inline("nodebug")
    fn __rsub__(self, value: Self) -> Self:
        """Return `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        return value - self

    @always_inline("nodebug")
    fn __rmul__(self, value: Self) -> Self:
        """Return `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        return self * value

    @always_inline("nodebug")
    fn __rfloordiv__(self, value: Self) -> Self:
        """Return `value // self`.

        Args:
            value: The other value.

        Returns:
            `value // self`.
        """
        return value // self

    @always_inline("nodebug")
    fn __rlshift__(self, value: Self) -> Self:
        """Return `value << self`.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        return value << self

    @always_inline("nodebug")
    fn __rrshift__(self, value: Self) -> Self:
        """Return `value >> self`.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        return value >> self

    @always_inline("nodebug")
    fn __rand__(self, value: Self) -> Self:
        """Return `value & self`.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        return value & self

    @always_inline("nodebug")
    fn __ror__(self, value: Self) -> Self:
        """Return `value | self`.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        return value | self

    @always_inline("nodebug")
    fn __rxor__(self, value: Self) -> Self:
        """Return `value ^ self`.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        return value ^ self

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Convert this IntLiteral to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self != Self()

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        """Convert this IntLiteral to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn __index__(self) -> Int:
        """Return self converted to an integer, if self is suitable for use as
        an index into a list.

        Returns:
            The corresponding Int value.
        """
        return self.__int__()

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Convert from IntLiteral to Int.

        Returns:
            The value as an integer of platform-specific width.
        """
        return Int(self.__as_mlir_index())

    @always_inline("nodebug")
    fn __uint__(self) -> UInt:
        """Convert from IntLiteral to UInt.

        Returns:
            The value as an unsigned integer of platform-specific width.
        """
        return __mlir_op.`kgen.int_literal.convert`[
            _type = __mlir_type.index, treatIndexAsUnsigned = __mlir_attr.unit
        ](self._value)

    @always_inline("nodebug")
    fn __abs__(self) -> Self:
        """Return the absolute value of the IntLiteral value.

        Returns:
            The absolute value.
        """
        if self >= 0:
            return self
        return -self

    @always_inline("nodebug")
    fn __ceil__(self) -> Self:
        """Return the ceiling of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("nodebug")
    fn __floor__(self) -> Self:
        """Return the floor of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("nodebug")
    fn __round__(self) -> Self:
        """Return the rounded value of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("nodebug")
    fn __trunc__(self) -> Self:
        """Return the truncated of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        """Return the rounded value of the IntLiteral value, which is itself.

        Args:
            ndigits: The number of digits to round to.

        Returns:
            The IntLiteral value itself if ndigits >= 0 else the rounded value.
        """
        if ndigits >= 0:
            return self
        alias one = __mlir_attr.`#kgen.int_literal<1> : !kgen.int_literal`
        alias ten = __mlir_attr.`#kgen.int_literal<10> : !kgen.int_literal`
        var multiplier = one
        # TODO: Use IntLiteral.__pow__() when it's implemented.
        for _ in range(-ndigits):
            multiplier = __mlir_op.`kgen.int_literal.binop`[
                oper = __mlir_attr.`#kgen<int_literal.binop_kind mul>`
            ](multiplier, ten)
        alias Pair = Tuple[Self, Self]
        var mod: IntLiteral = self % Self(multiplier)
        if mod * 2 >= multiplier:
            mod -= multiplier
        return self - mod

    @no_inline
    fn __str__(self) -> String:
        """Convert from IntLiteral to String.

        Returns:
            The value as a string.
        """
        return str(Int(self))

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return -(self // -denominator)

    # ===----------------------------------------------------------------------===#
    # Methods
    # ===----------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn _bit_width(self) -> IntLiteral:
        """Get the (signed) bit width of the IntLiteral.

        Returns:
            The bit width.
        """
        return __mlir_op.`kgen.int_literal.bit_width`(self._value)

    @always_inline("nodebug")
    fn __as_mlir_index(self) -> __mlir_type.index:
        """Convert from IntLiteral to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return __mlir_op.`kgen.int_literal.convert`[_type = __mlir_type.index](
            self._value
        )
