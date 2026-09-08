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
"""Implements the FloatLiteral class.

These are Mojo built-ins, so you don't need to import them.
"""

# ===-----------------------------------------------------------------------===#
# FloatLiteral
# ===-----------------------------------------------------------------------===#


@__nonmaterializable(Float64)
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`](
    Boolable,
    Defaultable,
    Floatable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """Mojo floating point literal type.

    Parameters:
        value: The underlying infinite precision floating point value.
    """

    # ===------------------------------------------------------------------===#
    # Constructors
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Create a FloatLiteral for any parameter value."""
        pass

    @always_inline("builtin")
    @implicit
    def __init__(
        _value: IntLiteral[_],
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<int_to_float_literal<`,
            _value.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Convert an IntLiteral to a FloatLiteral value.

        Args:
            _value: The IntLiteral value.
        """
        return {}

    comptime nan = FloatLiteral[__mlir_attr.`#pop.float_literal<nan>`]()
    """Not a number (NaN) value."""

    comptime infinity = FloatLiteral[__mlir_attr.`#pop.float_literal<inf>`]()
    """Positive infinity value."""

    comptime negative_infinity = FloatLiteral[
        __mlir_attr.`#pop.float_literal<neg_inf>`
    ]()
    """Negative infinity value."""

    comptime negative_zero = FloatLiteral[
        __mlir_attr.`#pop.float_literal<neg_zero>`
    ]()
    """Negative zero value."""

    @always_inline("builtin")
    def is_nan(self) -> Bool:
        """Return whether the FloatLiteral is nan.

        Since `nan == nan` is False, this provides a way to check for nan-ness.

        Returns:
            True, if the value is nan, False otherwise.
        """
        return __mlir_attr[`#pop<float_literal_isa<nan `, self.value, `>> : i1`]

    @always_inline("builtin")
    def is_neg_zero(self) -> Bool:
        """Return whether the FloatLiteral is negative zero.

        Since `FloatLiteral.negative_zero == 0.0` is True, this provides a way
        to check if the FloatLiteral is negative zero.

        Returns:
            True, if the value is negative zero, False otherwise.
        """
        return __mlir_attr[
            `#pop<float_literal_isa<neg_zero `, self.value, `>> : i1`
        ]

    @always_inline("builtin")
    def _is_normal(self) -> Bool:
        """Return whether the FloatLiteral is a normal (i.e. not special) value.

        Returns:
            True, if the value is a normal float, False otherwise.
        """
        return __mlir_attr[
            `#pop<float_literal_isa<normal `, self.value, `>> : i1`
        ]

    # ===------------------------------------------------------------------===#
    # Conversion Operators
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __int_literal__(
        self,
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<float_to_int_literal<`,
            Self.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Casts the floating point value to an IntLiteral. If there is a
        fractional component, then the value is truncated towards zero.

        Eg. `(4.5).__int_literal__()` returns `4`, and `(-3.7).__int_literal__()`
        returns `-3`.

        Returns:
            The value as an integer.
        """
        return {}

    @always_inline("builtin")
    def __int__(self) -> Int:
        """Converts the FloatLiteral value to an Int. If there is a fractional
        component, then the value is truncated towards zero.

        Eg. `(4.5).__int__()` returns `4`, and `(-3.7).__int__()` returns `-3`.

        Returns:
            The value as an integer.
        """
        return self.__int_literal__().__int__()

    @always_inline("nodebug")
    def __float__(self) -> Float64:
        """Converts the FloatLiteral to a concrete Float64.

        Returns:
            The Float value.
        """
        return Float64(self)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the FloatLiteral in string form.

        Args:
            writer: The Writer to write the value to.
        """
        Float64(self).write_to(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the FloatLiteral in repr form.

        Args:
            writer: The Writer to write the value to.
        """
        Float64(self).write_repr_to(writer)

    # ===------------------------------------------------------------------===#
    # Unary Operators
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        """A FloatLiteral value is true if it is non-zero.

        Returns:
            True if non-zero.
        """
        return self != 0.0

    @always_inline("builtin")
    def __neg__(self) -> type_of(self * -1):
        """Return the negation of the FloatLiteral value.

        Returns:
            The negated FloatLiteral value.
        """
        return {}

    # ===------------------------------------------------------------------===#
    # Arithmetic Operators
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __add__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<add `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Add two FloatLiterals.

        Args:
            rhs: The value to add.

        Returns:
            The sum of the two values.
        """
        return {}

    @always_inline("builtin")
    def __sub__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<sub `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Subtract two FloatLiterals.

        Args:
            rhs: The value to subtract.

        Returns:
            The difference of the two values.
        """
        return {}

    @always_inline("builtin")
    def __mul__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<mul `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Multiply two FloatLiterals.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the two values.
        """
        return {}

    @always_inline("builtin")
    def __truediv__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<truediv `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Divide two FloatLiterals.

        Args:
            rhs: The value to divide.

        Returns:
            The quotient of the two values.
        """
        # TODO - Python raises an error on divide by 0.0 or -0.0
        return {}

    @always_inline("builtin")
    def __floordiv__(
        self, rhs: FloatLiteral
    ) -> FloatLiteral[
        __mlir_attr[
            `#pop<float_literal_bin<floordiv `,
            Self.value,
            `,`,
            rhs.value,
            `>> : !pop.float_literal`,
        ]
    ]:
        """Returns self divided by rhs, rounded down to the nearest integer.

        Args:
            rhs: The divisor value.

        Returns:
            `floor(self / rhs)` value.
        """
        # TODO - Python raises an error on divide by 0.0 or -0.0
        return {}

    @always_inline("builtin")
    def __mod__(
        self, rhs: FloatLiteral
    ) -> type_of(self - (self.__floordiv__(rhs) * rhs)):
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        return {}

    @always_inline("builtin")
    def __ceildiv__(
        self, denominator: FloatLiteral
    ) -> type_of(-(self // -denominator)):
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return {}

    # TODO - maybe __pow__?

    # ===------------------------------------------------------------------===#
    # Reversed Operators, allowing things like "1 / 2.0" to work
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __radd__(self, rhs: FloatLiteral) -> type_of(rhs + self):
        """Reversed addition operator.

        Args:
            rhs: The value to add.

        Returns:
            The sum of this and the given value.
        """
        return {}

    @always_inline("builtin")
    def __rsub__(self, rhs: FloatLiteral) -> type_of(rhs - self):
        """Reversed subtraction operator.

        Args:
            rhs: The value to subtract from.

        Returns:
            The result of subtracting this from the given value.
        """
        return {}

    @always_inline("builtin")
    def __rmul__(self, rhs: FloatLiteral) -> type_of(rhs * self):
        """Reversed multiplication operator.

        Args:
            rhs: The value to multiply.

        Returns:
            The product of the given number and this.
        """
        return {}

    @always_inline("builtin")
    def __rmod__(self, rhs: FloatLiteral) -> type_of(rhs.__mod__(self)):
        """Return the remainder of rhs divided by self.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing rhs by self.
        """
        return {}

    @always_inline("builtin")
    def __rfloordiv__(self, rhs: FloatLiteral) -> type_of(rhs // self):
        """Returns rhs divided by self, rounded down to the nearest integer.

        Args:
            rhs: The value to be divided by self.

        Returns:
            `floor(rhs / self)` value.
        """
        return {}

    @always_inline("builtin")
    def __rtruediv__(self, rhs: FloatLiteral) -> type_of(rhs / self):
        """Reversed division.

        Args:
            rhs: The value to be divided by this.

        Returns:
            The result of dividing the given value by this.
        """
        return {}

    # ===------------------------------------------------------------------===#
    # Comparison Operators
    # ===------------------------------------------------------------------===#

    @always_inline("builtin")
    def __eq__(self, rhs: FloatLiteral) -> Bool:
        """Compare for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<eq `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    def __ne__(self, rhs: FloatLiteral) -> Bool:
        """Compare for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<ne `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    def __lt__(self, rhs: FloatLiteral) -> Bool:
        """Less than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<lt `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    def __le__(self, rhs: FloatLiteral) -> Bool:
        """Less than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than or equal to `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<le `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    def __gt__(self, rhs: FloatLiteral) -> Bool:
        """Greater than comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<gt `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )

    @always_inline("builtin")
    def __ge__(self, rhs: FloatLiteral) -> Bool:
        """Greater than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is greater than or equal to `rhs`.
        """
        return Bool(
            __mlir_attr[
                `#pop<float_literal_cmp<ge `,
                self.value,
                `,`,
                rhs.value,
                `>> : i1`,
            ]
        )
