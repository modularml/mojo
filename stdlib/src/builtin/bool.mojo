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

from collections import List, Set

from utils._select import _select_register_value
from utils._visualizers import lldb_formatter_wrapping_type

# ===----------------------------------------------------------------------=== #
#  Boolable
# ===----------------------------------------------------------------------=== #


trait Boolable:
    """The `Boolable` trait describes a type that can be explicitly converted to
    a `Bool` or evaluated as a boolean expression in `if` or `while` conditions.

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


# ===----------------------------------------------------------------------=== #
#  ImplicitlyBoolable
# ===----------------------------------------------------------------------=== #


trait ImplicitlyBoolable(Boolable):
    """The `ImplicitlyBoolable` trait describes a type that can be implicitly
    converted to a `Bool`.

    Types conforming to this trait can be passed to a function that expects a
    `Bool` without explicitly converting to it. Accordingly, most types should
    conform to `Boolable` instead, since implicit conversions to `Bool` can have
    unintuitive consequences.

    This trait requires the type to implement the `__as_bool__()` method. For
    example:

    ```mojo
    @value
    struct Foo(ImplicitlyBoolable):
        var val: Bool

        fn __as_bool__(self) -> Bool:
            return self.val

        fn __bool__(self) -> Bool:
            return self.__as_bool__()
    ```
    """

    fn __as_bool__(self) -> Bool:
        """Get the boolean representation of the value.

        Returns:
            The boolean representation of the value.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  Bool
# ===----------------------------------------------------------------------=== #


@lldb_formatter_wrapping_type
@value
@register_passable("trivial")
struct Bool(
    CollectionElementNew,
    ComparableCollectionElement,
    Defaultable,
    ImplicitlyBoolable,
    Indexer,
    Intable,
    Representable,
    Stringable,
    Writable,
    Floatable,
):
    """The primitive Bool scalar value used in Mojo."""

    var value: __mlir_type.i1
    """The underlying storage of the boolean value."""

    @always_inline("nodebug")
    fn __init__(out self):
        """Construct a default, `False` Bool."""
        self = False

    @always_inline("nodebug")
    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self.value = other.value

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.i1):
        """Construct a Bool value given a __mlir_type.i1 value.

        Args:
            value: The initial __mlir_type.i1 value.
        """
        self.value = value

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<bool>`):
        """Construct a Bool value given a `!pop.scalar<bool>` value.

        Args:
            value: The initial value.
        """
        self.value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.i1](
            value
        )

    @always_inline("nodebug")
    @implicit
    fn __init__[T: ImplicitlyBoolable, //](mut self, value: T):
        """Convert an ImplicitlyBoolable value to a Bool.

        Parameters:
            T: The ImplicitlyBoolable type.

        Args:
            value: The boolable value.
        """
        self = value.__bool__()

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: SIMD[DType.bool, 1]):
        """Convert a scalar SIMD value to a Bool.

        Args:
            value: The scalar value.
        """
        self = value.__bool__()

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Convert to Bool.

        Returns:
            This value.
        """
        return self

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        """Convert to Bool.

        Returns:
            This value.
        """
        return self.__bool__()

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
        return self.value

    @always_inline("nodebug")
    fn _as_scalar_bool(self) -> __mlir_type.`!pop.scalar<bool>`:
        return __mlir_op.`pop.cast_from_builtin`[
            _type = __mlir_type.`!pop.scalar<bool>`
        ](self.value)

    @no_inline
    fn __str__(self) -> String:
        """Get the bool as a string.

        Returns `"True"` or `"False"`.

        Returns:
            A string representation.
        """
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this boolean to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("True" if self else "False")

    fn __repr__(self) -> String:
        """Get the bool as a string.

        Returns `"True"` or `"False"`.

        Returns:
            A string representation.
        """
        return str(self)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Convert this Bool to an integer.

        Returns:
            1 if the Bool is True, 0 otherwise.
        """
        return _select_register_value(self, Int(1), Int(0))

    @always_inline("nodebug")
    fn __float__(self) -> Float64:
        """Convert this Bool to a float.

        Returns:
            1.0 if True else 0.0 otherwise.
        """
        return _select_register_value(self, Float64(1.0), Float64(0.0))

    @always_inline("nodebug")
    fn __index__(self) -> Int:
        """Convert this Bool to an integer for indexing purposes.

        Returns:
            1 if the Bool is True, 0 otherwise.
        """
        return self.__int__()

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
            self._as_scalar_bool(), rhs._as_scalar_bool()
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
            self._as_scalar_bool(), rhs._as_scalar_bool()
        )

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using less-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is False and rhs is True.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred lt>`](
            self._as_scalar_bool(), rhs._as_scalar_bool()
        )

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using less-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is False and rhs is True or False.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred le>`](
            self._as_scalar_bool(), rhs._as_scalar_bool()
        )

    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using greater-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is True and rhs is False.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred gt>`](
            self._as_scalar_bool(), rhs._as_scalar_bool()
        )

    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using greater-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is True and rhs is True or False.
        """

        return __mlir_op.`pop.cmp`[pred = __mlir_attr.`#pop<cmp_pred ge>`](
            self._as_scalar_bool(), rhs._as_scalar_bool()
        )

    # ===-------------------------------------------------------------------===#
    # Bitwise operations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __invert__(self) -> Bool:
        """Inverts the Bool value.

        Returns:
            True if the object is false and False otherwise.
        """
        var true = __mlir_op.`index.bool.constant`[
            _type = __mlir_type.i1,
            value = __mlir_attr.`true : i1`,
        ]()
        return __mlir_op.`pop.xor`(self.value, true)

    @always_inline("nodebug")
    fn __and__(self, rhs: Bool) -> Bool:
        """Returns `self & rhs`.

        Bitwise and's the Bool value with the argument. This method gets invoked
        when a user uses the `and` infix operator.

        Args:
            rhs: The right hand side of the `and` statement.

        Returns:
            `self & rhs`.
        """
        return __mlir_op.`pop.and`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: Bool):
        """Computes `self & rhs` and store the result in `self`.

        Args:
            rhs: The right hand side of the `and` statement.
        """
        self = self & rhs

    @always_inline("nodebug")
    fn __rand__(self, lhs: Bool) -> Bool:
        """Returns `lhs & self`.

        Args:
            lhs: The left hand side of the `and` statement.

        Returns:
            `lhs & self`.
        """
        return lhs & self

    @always_inline("nodebug")
    fn __or__(self, rhs: Bool) -> Bool:
        """Returns `self | rhs`.

        Bitwise or's the Bool value with the argument. This method gets invoked
        when a user uses the `or` infix operator.

        Args:
            rhs: The right hand side of the `or` statement.

        Returns:
            `self | rhs`.
        """
        return __mlir_op.`pop.or`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Bool):
        """Computes `self | rhs` and store the result in `self`.

        Args:
            rhs: The right hand side of the `or` statement.
        """
        self = self | rhs

    @always_inline("nodebug")
    fn __ror__(self, lhs: Bool) -> Bool:
        """Returns `lhs | self`.

        Args:
            lhs: The left hand side of the `or` statement.

        Returns:
            `lhs | self`.
        """
        return lhs | self

    @always_inline("nodebug")
    fn __xor__(self, rhs: Bool) -> Bool:
        """Returns `self ^ rhs`.

        Bitwise Xor's the Bool value with the argument. This method gets invoked
        when a user uses the `^` infix operator.

        Args:
            rhs: The right hand side of the `xor` statement.

        Returns:
            `self ^ rhs`.
        """
        return __mlir_op.`pop.xor`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Bool):
        """Computes `self ^ rhs` and stores the result in `self`.

        Args:
            rhs: The right hand side of the `xor` statement.
        """
        self = self ^ rhs

    @always_inline("nodebug")
    fn __rxor__(self, lhs: Bool) -> Bool:
        """Returns `lhs ^ self`.

        Args:
            lhs: The left hand side of the `xor` statement.

        Returns:
            `lhs ^ self`.
        """
        return lhs ^ self

    @always_inline("nodebug")
    fn __neg__(self) -> Int:
        """Defines the unary `-` operation.

        Returns:
            0 for -False and -1 for -True.
        """
        return __mlir_op.`index.casts`[_type = __mlir_type.index](self.value)


# ===----------------------------------------------------------------------=== #
#  bool
# ===----------------------------------------------------------------------=== #


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
fn bool[T: Boolable, //](value: T) -> Bool:
    """Get the bool representation of the object.

    Parameters:
        T: The type of the object.

    Args:
        value: The object to get the bool representation of.

    Returns:
        The bool representation of the object.
    """
    return value.__bool__()


# ===----------------------------------------------------------------------=== #
#  any
# ===----------------------------------------------------------------------=== #


# TODO: Combine these into Iterators over Boolable elements


fn any[T: BoolableCollectionElement](list: List[T, *_]) -> Bool:
    """Checks if **any** element in the list is truthy.

    Parameters:
        T: The type of elements to check.

    Args:
        list: The list to check.

    Returns:
        `True` if **any** element in the list is truthy, `False` otherwise.
    """
    for item in list:
        if item[]:
            return True
    return False


fn any[T: BoolableKeyElement](set: Set[T]) -> Bool:
    """Checks if **any** element in the set is truthy.

    Parameters:
        T: The type of elements to check.

    Args:
        set: The set to check.

    Returns:
        `True` if **any** element in the set is truthy, `False` otherwise.
    """
    for item in set:
        if item[]:
            return True
    return False


fn any(value: SIMD) -> Bool:
    """Checks if **any** element in the simd vector is truthy.

    Args:
        value: The simd vector to check.

    Returns:
        `True` if **any** element in the simd vector is truthy, `False`
        otherwise.
    """
    return value.cast[DType.bool]().reduce_or()


# ===----------------------------------------------------------------------=== #
#  all
# ===----------------------------------------------------------------------=== #


# TODO: Combine these into Iterators over Boolable elements


fn all[T: BoolableCollectionElement](list: List[T, *_]) -> Bool:
    """Checks if **all** elements in the list are truthy.

    Parameters:
        T: The type of elements to check.

    Args:
        list: The list to check.

    Returns:
        `True` if **all** elements in the list are truthy, `False` otherwise.
    """
    for item in list:
        if not item[]:
            return False
    return True


fn all[T: BoolableKeyElement](set: Set[T]) -> Bool:
    """Checks if **all** elements in the set are truthy.

    Parameters:
        T: The type of elements to check.

    Args:
        set: The set to check.

    Returns:
        `True` if **all** elements in the set are truthy, `False` otherwise.
    """
    for item in set:
        if not item[]:
            return False
    return True


fn all(value: SIMD) -> Bool:
    """Checks if **all** elements in the simd vector are truthy.

    Args:
        value: The simd vector to check.

    Returns:
        `True` if **all** elements in the simd vector are truthy, `False`
        otherwise.
    """
    return value.cast[DType.bool]().reduce_and()
