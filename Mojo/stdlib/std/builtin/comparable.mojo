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
"""Implements comparison and equality traits for Mojo types."""

from std.builtin.constrained import _field_conforms_to_error
from std.builtin.range import _ZeroStartingRange
from std.reflection import reflect


trait Equatable:
    """A type which can be compared for equality with other instances of itself.

    The `Equatable` trait has a default implementation of `__eq__()` that uses
    reflection to compare all fields. This means simple structs can conform to
    `Equatable` without implementing any methods:

    ```mojo
    @fieldwise_init
    struct Point(Equatable):
        var x: Int
        var y: Int

    var p1 = Point(1, 2)
    var p2 = Point(1, 2)
    print(p1 == p2)  # True
    ```

    All fields must conform to `Equatable`. Override `__eq__()` for custom
    equality semantics.

    Note: The default implementation performs memberwise equality comparison.
    This may not be appropriate for types containing floating-point fields
    (due to NaN semantics) or types requiring custom equality logic.

    Note: The default reflection-based implementation iterates over all fields
    at compile time. For mutually recursive types (e.g., struct `A` has a field
    of type `List[B]` and struct `B` has a field of type `A`), this creates an
    infinite monomorphization cycle that causes the compiler to hang. To fix
    this, provide an explicit `__eq__()` implementation for at least one type
    in the cycle.
    """

    @always_inline
    def __eq__(self, other: Self, /) -> Bool:
        """Define whether two instances of the object are equal to each other.

        The default implementation uses reflection to compare all fields for
        equality. All fields must conform to `Equatable`.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are equal according to the type's definition
            of equality, False otherwise.
        """

        # Default implementation using reflection: compare all fields
        comptime r = reflect[Self]
        comptime names = r.field_names()
        comptime types = r.field_types()

        comptime for i in range(names.length):
            comptime T = types[i]
            comptime assert conforms_to(T, Equatable), _field_conforms_to_error[
                Parent=Self,
                FieldIndex=i,
                ParentConformsTo="Equatable",
            ]()
            if r.field_ref[i](self) != r.field_ref[i](other):
                return False
        return True

    @always_inline
    def __ne__(self, other: Self, /) -> Bool:
        """Define whether two instances of the object are not equal to each
        other.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are not equal according to the type's
            definition of equality, False otherwise.
        """
        return not self == other


trait Comparable(Equatable):
    """A type which can be compared for order with other instances of itself.

    Implementers of this trait must define the `__lt__` and `__eq__` methods.

    The default implementations of the default comparison methods can be
    potentially inefficient for types where comparison is expensive. For such
    types, it is recommended to override all the default implementations.
    """

    def __lt__(self, rhs: Self) -> Bool:
        """Define whether `self` is less than `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is less than `rhs`.
        """
        ...

    @always_inline
    def __gt__(self, rhs: Self) -> Bool:
        """Define whether `self` is greater than `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is greater than `rhs`.
        """
        return rhs < self

    @always_inline
    def __le__(self, rhs: Self) -> Bool:
        """Define whether `self` is less than or equal to `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is less than or equal to `rhs`.
        """
        return not rhs < self

    @always_inline
    def __ge__(self, rhs: Self) -> Bool:
        """Define whether `self` is greater than or equal to `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is greater than or equal to `rhs`.
        """
        return not self < rhs
