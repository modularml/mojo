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
"""Implements the Bool class.

These are Mojo built-ins, so you don't need to import them.
"""

from std.collections import List, Set
from std.hashlib.hasher import Hasher

from std.python import (
    ConvertibleFromPython,
    Python,
    PythonObject,
)

from std.utils._select import _select_register_value as select
from std.utils._visualizers import lldb_formatter_wrapping_type

# ===----------------------------------------------------------------------=== #
#  Boolable
# ===----------------------------------------------------------------------=== #


trait Boolable:
    """The `Boolable` trait describes a type that can be explicitly converted to
    a `Bool` or evaluated as a boolean expression in `if` or `while` conditions.

    This trait requires the type to implement the `__bool__()` method. For
    example:

    ```mojo
    struct Foo(Boolable):
        var val: Bool

        def __bool__(self) -> Bool:
            return self.val
    ```
    """

    def __bool__(self) -> Bool:
        """Get the boolean representation of the value.

        Returns:
            The boolean representation of the value.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  Bool
# ===----------------------------------------------------------------------=== #


@stable(since="1.0")
@lldb_formatter_wrapping_type
struct Bool(
    Boolable,
    Comparable,
    ConvertibleFromPython,
    Defaultable,
    Floatable,
    Hashable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """The primitive Bool scalar value used in Mojo."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _mlir_value: __mlir_type.`!kgen.scalar<bool>`
    """The underlying storage of the boolean value."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime MIN: Bool = False
    """The minimum value of a Bool."""

    comptime MAX: Bool = True
    """The maximum value of a Bool."""

    # ===-------------------------------------------------------------------===#
    # Trivial bits for special functions.
    # ===-------------------------------------------------------------------===#

    comptime __del__is_trivial: Bool = True
    comptime __move_ctor_is_trivial: Bool = True
    comptime __copy_ctor_is_trivial: Bool = True

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __init__(out self):
        """Construct a default, `False` Bool."""
        self = False

    @doc_hidden
    @always_inline("builtin")
    @implicit
    def __init__(out self, value: __mlir_type.i1):
        """Construct a Bool value given a __mlir_type.i1 value.

        Args:
            value: The initial __mlir_type.i1 value.
        """
        self._mlir_value = __mlir_op.`pop.cast_from_builtin`[
            _type=__mlir_type.`!kgen.scalar<bool>`
        ](value)

    @doc_hidden
    @always_inline("builtin")
    @implicit
    def __init__(out self, mlir_value: __mlir_type.`!kgen.scalar<bool>`):
        """Construct a Bool value given a `!kgen.scalar<bool>` value.

        Args:
            mlir_value: The initial value.
        """
        self._mlir_value = mlir_value

    @always_inline("nodebug")
    def __init__[T: Boolable, //](out self, value: T):
        """Set the bool representation of the object.

        Parameters:
            T: The type of the object.

        Args:
            value: The object to get the bool representation of.
        """
        self = value.__bool__()

    @always_inline("builtin")
    def __init__(out self, value: None):
        """Set the bool representation of the `None` type to `False`.

        Args:
            value: The object to get the bool representation of.
        """
        self = False

    @always_inline("nodebug")
    @implicit
    def __init__(out self, value: Scalar[.bool]):
        """Convert a scalar SIMD value to a Bool.

        Args:
            value: The scalar value.
        """
        self = value.__bool__()

    @always_inline("builtin")
    def __bool__(self) -> Bool:
        """Convert to Bool.

        Returns:
            This value.
        """
        return self

    @doc_hidden
    @always_inline("builtin")
    def __mlir_bool__(self) -> __mlir_type.`!kgen.scalar<bool>`:
        """Convert this Bool to __mlir_type.`!kgen.scalar<bool>`.

        This method is a special hook used by the compiler to test boolean
        objects in control flow conditions.  It should be implemented by Bool
        but not other general boolean convertible types (they should implement
        `__bool__` instead).

        Returns:
            The underlying value for the Bool.
        """
        return self._mlir_value

    @doc_hidden
    @always_inline("builtin")
    def __mlir_i1__(self) -> __mlir_type.i1:
        """Convert this Bool to __mlir_type.i1.

        Returns:
            The underlying value for the Bool as an __mlir_type.i1.
        """
        return __mlir_op.`pop.cast_to_builtin`[_type=__mlir_type.i1](
            self._mlir_value
        )

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this boolean to the provided Writer.

        Args:
            writer: The object to write to.
        """

        # `write_string` rather than `write` so the literal reaches the writer
        # without becoming a `String`. The arms name `StaticString` because the
        # two literals have distinct types, and a conditional over the bare
        # literals unifies them through `String`.
        writer.write_string(
            StaticString("True") if self else StaticString("False")
        )

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr of this boolean to a writer.

        The repr of a boolean is the same as its string representation:
        `True` or `False`.

        Args:
            writer: The object to write to.
        """
        self.write_to(writer)

    @always_inline("builtin")
    def __int__(self) -> Int:
        """Convert this Bool to an integer.

        Returns:
            1 if the Bool is True, 0 otherwise.
        """
        return select[Int](self, 1, 0)

    @always_inline("nodebug")
    def __float__(self) -> Float64:
        """Convert this Bool to a float.

        Returns:
            1.0 if True else 0.0 otherwise.
        """
        return select[Float64](self, 1, 0)

    @always_inline("builtin")
    def __eq__(self, rhs: Bool) -> Bool:
        """Compare this Bool to RHS.

        Performs an equality comparison between the Bool value and the argument.
        This method gets invoked when a user uses the `==` infix operator.

        Args:
            rhs: The rhs value of the equality statement.

        Returns:
            True if the two values match and False otherwise.
        """
        return ~(self != rhs)

    @always_inline("builtin")
    def __ne__(self, rhs: Bool) -> Bool:
        """Compare this Bool to RHS.

        Performs a non-equality comparison between the Bool value and the
        argument. This method gets invoked when a user uses the `!=` infix
        operator.

        Args:
            rhs: The rhs value of the non-equality statement.

        Returns:
            False if the two values do match and True otherwise.
        """
        return self ^ rhs

    @always_inline("builtin")
    def __lt__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using less-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is False and rhs is True.
        """

        return ~self & rhs

    @always_inline("builtin")
    def __le__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using less-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is False and rhs is True or False.
        """

        return ~self | rhs

    @always_inline("builtin")
    def __gt__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using greater-than comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is True and rhs is False.
        """

        return rhs < self

    @always_inline("builtin")
    def __ge__(self, rhs: Self) -> Bool:
        """Compare this Bool to RHS using greater-than-or-equal comparison.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if self is True and rhs is True or False.
        """

        return rhs <= self

    # ===-------------------------------------------------------------------===#
    # Bitwise operations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    def __invert__(self) -> Bool:
        """Inverts the Bool value.

        Returns:
            True if the object is false and False otherwise.
        """
        return __mlir_op.`pop.simd.xor`(
            self._mlir_value,
            __mlir_attr.`#kgen.simd<true> : !kgen.scalar<bool>`,
        )

    @always_inline("builtin")
    def __and__(self, rhs: Bool) -> Bool:
        """Returns `self & rhs`.

        Bitwise and's the Bool value with the argument. This method gets invoked
        when a user uses the `and` infix operator.

        Args:
            rhs: The right hand side of the `and` statement.

        Returns:
            `self & rhs`.
        """
        return __mlir_op.`pop.simd.and`(self._mlir_value, rhs._mlir_value)

    @always_inline("nodebug")
    def __iand__(mut self, rhs: Bool):
        """Computes `self & rhs` and store the result in `self`.

        Args:
            rhs: The right hand side of the `and` statement.
        """
        self = self & rhs

    @always_inline("builtin")
    def __rand__(self, lhs: Bool) -> Bool:
        """Returns `lhs & self`.

        Args:
            lhs: The left hand side of the `and` statement.

        Returns:
            `lhs & self`.
        """
        return lhs & self

    @always_inline("builtin")
    def __or__(self, rhs: Bool) -> Bool:
        """Returns `self | rhs`.

        Bitwise or's the Bool value with the argument. This method gets invoked
        when a user uses the `or` infix operator.

        Args:
            rhs: The right hand side of the `or` statement.

        Returns:
            `self | rhs`.
        """
        return __mlir_op.`pop.simd.or`(self._mlir_value, rhs._mlir_value)

    @always_inline("nodebug")
    def __ior__(mut self, rhs: Bool):
        """Computes `self | rhs` and store the result in `self`.

        Args:
            rhs: The right hand side of the `or` statement.
        """
        self = self | rhs

    @always_inline("builtin")
    def __ror__(self, lhs: Bool) -> Bool:
        """Returns `lhs | self`.

        Args:
            lhs: The left hand side of the `or` statement.

        Returns:
            `lhs | self`.
        """
        return lhs | self

    @always_inline("builtin")
    def __xor__(self, rhs: Bool) -> Bool:
        """Returns `self ^ rhs`.

        Bitwise Xor's the Bool value with the argument. This method gets invoked
        when a user uses the `^` infix operator.

        Args:
            rhs: The right hand side of the `xor` statement.

        Returns:
            `self ^ rhs`.
        """
        return __mlir_op.`pop.simd.xor`(self._mlir_value, rhs._mlir_value)

    @always_inline("nodebug")
    def __ixor__(mut self, rhs: Bool):
        """Computes `self ^ rhs` and stores the result in `self`.

        Args:
            rhs: The right hand side of the `xor` statement.
        """
        self = self ^ rhs

    @always_inline("builtin")
    def __rxor__(self, lhs: Bool) -> Bool:
        """Returns `lhs ^ self`.

        Args:
            lhs: The left hand side of the `xor` statement.

        Returns:
            `lhs ^ self`.
        """
        return lhs ^ self

    def __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(Scalar[.bool](self))

    @doc_hidden
    def __init__(out self, *, py: PythonObject) raises:
        """Construct a `Bool` from a PythonObject.

        Args:
            py: The Python object to convert from.

        Raises:
            An error if conversion failed.
        """
        # TODO: return py.__bool__() when it no longer fails silently.
        return Python.is_true(py)


# ===----------------------------------------------------------------------=== #
#  any
# ===----------------------------------------------------------------------=== #


def any[
    IterableType: Iterable
](iterable: IterableType) -> Bool where conforms_to(
    IterableType.IteratorType[origin_of(iterable)].Element,
    Boolable & Deinitable,
):
    """Checks if **all** elements in the list are truthy.

    Parameters:
        IterableType: The type of the iterable containing `Boolable` items.

    Args:
        iterable: The iterable to check.

    Returns:
        `True` if **any** element in the list is truthy, `False` otherwise.
    """

    for var item0 in iterable:
        var item = item0^
        if item:
            return True
    return False


def any(value: SIMD) -> Bool:
    """Checks if **any** element in the simd vector is truthy.

    Args:
        value: The simd vector to check.

    Returns:
        `True` if **any** element in the simd vector is truthy, `False`
        otherwise.
    """
    return value.cast[.bool]().reduce_or()


# ===----------------------------------------------------------------------=== #
#  all
# ===----------------------------------------------------------------------=== #


def all[
    IterableType: Iterable
](iterable: IterableType) -> Bool where conforms_to(
    IterableType.IteratorType[origin_of(iterable)].Element,
    Boolable & Deinitable,
):
    """Checks if **all** elements in the list are truthy.

    Parameters:
        IterableType: The type of the iterable containing `Boolable` items.

    Args:
        iterable: The iterable to check.

    Returns:
        `True` if **all** elements in the iterable are truthy, `False` otherwise.
    """
    for var item0 in iterable:
        var item = item0^
        if not item:
            return False
    return True


def all(value: SIMD) -> Bool:
    """Checks if **all** elements in the simd vector are truthy.

    Args:
        value: The simd vector to check.

    Returns:
        `True` if **all** elements in the simd vector are truthy, `False`
        otherwise.
    """
    return value.cast[.bool]().reduce_and()
