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
"""Implements the IntLiteral class."""

from std.math import Ceilable, Floorable, Truncable


@__nonmaterializable(Int)
struct IntLiteral[value: __mlir_type.`!pop.int_literal`](
    Boolable,
    Ceilable,
    Defaultable,
    Floorable,
    Indexer,
    Intable,
    TrivialRegisterPassable,
    Truncable,
    Writable,
):
    """This type represents a static integer literal value with
    infinite precision.  This type is a compile-time construct which stores its
    value as a parameter.  It is typically materialized into other types (like
    `Int`) for use at runtime.  This compile-time representation allows for
    arbitrary precision constants that would overflow on Int and other fixed
    precision integer types.

    Parameters:
        value: The underlying integer value.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Constructor for any value."""
        pass

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __lt__(self, rhs: IntLiteral[_]) -> Bool:
        """Compare this IntLiteral to the RHS using LT comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is less-than the RHS IntLiteral and False otherwise.
        """
        return __mlir_attr[
            `#pop<int_literal_cmp<lt `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __le__(self, rhs: IntLiteral[_]) -> Bool:
        """Compare this IntLiteral to the RHS using LE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is less-or-equal than the RHS IntLiteral and False
            otherwise.
        """
        return __mlir_attr[
            `#pop<int_literal_cmp<le `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __eq__(self, rhs: IntLiteral[_]) -> Bool:
        """Compare this IntLiteral to the RHS using EQ comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is equal to the RHS IntLiteral and False otherwise.
        """
        return __mlir_attr[
            `#pop<int_literal_cmp<eq `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __ne__(self, rhs: IntLiteral[_]) -> Bool:
        """Compare this IntLiteral to the RHS using NE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is non-equal to the RHS IntLiteral and False otherwise.
        """
        return __mlir_attr[
            `#pop<int_literal_cmp<ne `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __gt__(self, rhs: IntLiteral[_]) -> Bool:
        """Compare this IntLiteral to the RHS using GT comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is greater-than the RHS IntLiteral and False otherwise.
        """
        return __mlir_attr[
            `#pop<int_literal_cmp<gt `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __ge__(self, rhs: IntLiteral[_]) -> Bool:
        """Compare this IntLiteral to the RHS using GE comparison.

        Args:
            rhs: The other IntLiteral to compare against.

        Returns:
            True if this IntLiteral is greater-or-equal than the RHS IntLiteral and False
            otherwise.
        """
        return __mlir_attr[
            `#pop<int_literal_cmp<ge `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]

    @always_inline("builtin")
    def __pos__(self) -> Self:
        """Return +self.

        Returns:
            The +self value.
        """
        return self

    @always_inline("builtin")
    def __neg__(self) -> type_of(0 - self):
        """Return -self.

        Returns:
            The -self value.
        """
        return 0 - self

    @always_inline("builtin")
    def __invert__(self) -> type_of(self ^ -1):
        """Return ~self.

        Returns:
            The ~self value.
        """
        return {}

    @always_inline("builtin")
    def __add__(
        self,
        rhs: IntLiteral[_],
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<add `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __add__(self, rhs: FloatLiteral) -> type_of(FloatLiteral(self) + rhs):
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __sub__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<sub `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __sub__(self, rhs: FloatLiteral) -> type_of(FloatLiteral(self) - rhs):
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __mul__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<mul `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __mul__(self, rhs: FloatLiteral) -> type_of(FloatLiteral(self) * rhs):
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __truediv__(
        self, rhs: FloatLiteral
    ) -> type_of(FloatLiteral(self) / rhs):
        """Return `self / rhs`.

        Args:
            rhs: The value to divide with.

        Returns:
            `self / rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __pow__(
        self, exp: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<pow `,
            self.value,
            `,`,
            exp.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return the value raised to the power of the given exponent.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        return {}

    @always_inline("builtin")
    def __floordiv__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<floordiv `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self // rhs`.

        Args:
            rhs: The value to divide with.

        Returns:
            `self // rhs` value.
        """
        return {}

    @always_inline("builtin")
    def __mod__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<mod `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        return {}

    @always_inline("builtin")
    def __lshift__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<lshift `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self << rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self << rhs`.
        """
        return {}

    @always_inline("builtin")
    def __rshift__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<rshift `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        return {}

    @always_inline("builtin")
    def __and__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<and `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return {}

    @always_inline("builtin")
    def __xor__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<xor `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self ^ rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        return {}

    @always_inline("builtin")
    def __or__(
        self, rhs: IntLiteral[_]
    ) -> IntLiteral[
        __mlir_attr[
            `#pop<int_literal_bin<or `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
    ]:
        """Return `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return {}

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        """Convert this IntLiteral to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self != 0

    @always_inline("builtin")
    def __int__(self) -> Int:
        """Convert from IntLiteral to Int.

        Returns:
            The value as an integer of platform-specific width.
        """
        return Int(SIMDLength(mlir_value=self.__mlir_index__()))

    @always_inline("builtin")
    def __ceil__(self) -> Self:
        """Return the ceiling of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("builtin")
    def __floor__(self) -> Self:
        """Return the floor of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @always_inline("builtin")
    def __trunc__(self) -> Self:
        """Return the truncated of the IntLiteral value, which is itself.

        Returns:
            The IntLiteral value itself.
        """
        return self

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the IntLiteral in string form.

        Args:
            writer: The Writer to write the value to.
        """
        Int(self).write_to(writer)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the IntLiteral in repr form.

        Args:
            writer: The Writer to write the value to.
        """
        Int(self).write_repr_to(writer)

    @always_inline("builtin")
    def __ceildiv__(
        self, denominator: IntLiteral
    ) -> type_of(-(self // -denominator)):
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return {}

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        """Convert from IntLiteral to index.

        Returns:
            The corresponding __mlir_type.index value, interpreting as signed.
        """
        return __mlir_attr[
            `#kgen.cast_to_builtin<#pop.int_literal_convert<`,
            self.value,
            `> : !kgen.scalar<index>> : index`,
        ]
