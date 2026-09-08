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
"""Implements `SIMDLength`, which is a wrapper around the MLIR `index` type."""

from std.utils._select import _select_register_value as select


struct SIMDLength(
    Comparable, Equatable, ImplicitlyCopyable, Indexer, TrivialRegisterPassable
):
    """Represents a type appropriate for the length of a simd vector.

    Note: Typically you should use Int instead."""

    var _mlir_value: __mlir_type.index

    @always_inline("builtin")
    def __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self._mlir_value

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self, *, mlir_value: __mlir_type.index):
        """Construct SIMDLength from the given index value.

        Args:
            mlir_value: The init value.
        """
        self._mlir_value = mlir_value

    @implicit
    @always_inline("builtin")
    def __init__(out self, value: IntLiteral, /):
        """Construct SIMDLength from the given IntLiteral value.

        Args:
            value: The init value.
        """
        self._mlir_value = value.__mlir_index__()

    @always_inline("nodebug")
    def __init__[T: Indexer](out self, value: T):
        """Construct a SIMDLength from the given Indexer.

        Parameters:
            T: The type of the init value.

        Args:
            value: The init value.
        """
        self._mlir_value = value.__mlir_index__()

    @implicit
    @always_inline("builtin")
    def __init__(out self, value: Int, /):
        """Construct a SIMDLength from the given Int.

        Args:
            value: The init value.
        """
        self._mlir_value = __mlir_op.`pop.cast_to_builtin`[
            _type=__mlir_type.index
        ](value._mlir_value)

    @always_inline("builtin")
    def __eq__(self, rhs: Self) -> Bool:
        """Compare this SIMDLength to the RHS using EQ comparison.

        Args:
            rhs: The other SIMDLength to compare against.

        Returns:
            True if this SIMDLength is equal to the RHS SIMDLength and False otherwise.
        """
        return Bool(
            value=__mlir_op.`index.cmp`[
                pred=__mlir_attr.`#index.cmp_predicate<eq>`
            ](self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __ne__(self, rhs: Self) -> Bool:
        """Compare this Int to the RHS using NE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is non-equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<ne>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __add__(self, rhs: Self) -> Self:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return Self(
            mlir_value=__mlir_op.`index.add`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __sub__(self, rhs: Self) -> Self:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return Self(
            mlir_value=__mlir_op.`index.sub`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __le__(self, rhs: Self) -> Bool:
        """Compare this SIMDLength to the RHS using LE comparison.

        Args:
            rhs: The other SIMDLength to compare against.

        Returns:
            True if this SIMDLength is less-or-equal than the RHS SIMDLength and False
            otherwise.
        """
        return Bool(
            value=__mlir_op.`index.cmp`[
                pred=__mlir_attr.`#index.cmp_predicate<sle>`
            ](self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __gt__(self, rhs: Self) -> Bool:
        """Compare this SIMDLength to the RHS using GT comparison.

        Args:
            rhs: The other SIMDLength to compare against.

        Returns:
            True if this SIMDLength is greater than the RHS SIMDLength and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<sgt>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __lt__(self, rhs: Self) -> Bool:
        """Compare this SIMDLength to the RHS using LT comparison.

        Args:
            rhs: The other SIMDLength to compare against.

        Returns:
            True if this SIMDLength is less-than the RHS SIMDLength and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<slt>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __ge__(self, rhs: Self) -> Bool:
        """Compare this SIMDLength to the RHS using GE comparison.

        Args:
            rhs: The other SIMDLength to compare against.

        Returns:
            True if this SIMDLength is greater-or-equal than the RHS SIMDLength and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred=__mlir_attr.`#index.cmp_predicate<sge>`
        ](self._mlir_value, rhs._mlir_value)

    @always_inline("builtin")
    def __neg__(self) -> Self:
        """Return -self.

        Returns:
            The -self value.
        """
        return self * -1

    @always_inline("builtin")
    def __and__(self, rhs: Self) -> Self:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return Self(
            mlir_value=__mlir_op.`index.and`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __truediv__(self, rhs: Self) -> Self:
        """Return the result of the division of `self` and `rhs`.

        Performs truncating division (toward zero) for integers.

        Args:
            rhs: The value to divide on.

        Returns:
            `self / rhs` value.
        """
        return Self(
            mlir_value=__mlir_op.`index.divs`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("builtin")
    def __mul__(self, rhs: Self) -> Self:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return Self(
            mlir_value=__mlir_op.`index.mul`(self._mlir_value, rhs._mlir_value)
        )

    @always_inline("nodebug")
    def __imul__(mut self, rhs: Self):
        """Compute self*rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self * rhs

    @always_inline("builtin")
    def __rshift__(self, rhs: Self) -> Self:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        return select(
            rhs < 0,
            0,
            Self(
                mlir_value=__mlir_op.`index.shrs`(
                    self._mlir_value, rhs._mlir_value
                )
            ),
        )

    @always_inline("nodebug")
    def __irshift__(mut self, rhs: Self):
        """Compute `self >> rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self >> rhs

    @always_inline("nodebug")
    def __pow__(self, exp: Self) -> Self:
        """Return the value raised to the power of the given exponent.

        Computes the power of an integer using the Russian Peasant Method.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        if exp < 0:
            # Not defined for Integers, this should raise an
            # exception.
            return 0
        var res: Int = 1
        var x = self
        var n = exp
        while n > 0:
            if n & 1 != 0:
                res *= x
            x *= x
            n >>= 1
        return res

    @always_inline("builtin")
    def is_power_of_two(self) -> Bool:
        """Check if the integer is a (non-zero) power of two.

        Returns:
            True if the integer is a power of two, False otherwise.
        """
        return (self & (self - 1) == 0) & (self > 0)
