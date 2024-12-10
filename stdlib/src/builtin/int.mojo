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
"""Implements the Int class.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import KeyElement
from collections.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
)
from hashlib._hasher import _HashableWithHasher, _Hasher
from hashlib.hash import _hash_simd
from math import Ceilable, CeilDivable, Floorable, Truncable
from sys import bitwidthof

from builtin.io import _snprintf
from memory import UnsafePointer
from python import Python, PythonObject
from python._cpython import Py_ssize_t

from utils import Writable, Writer
from utils._select import _select_register_value as select
from utils._visualizers import lldb_formatter_wrapping_type

# ===----------------------------------------------------------------------=== #
#  Indexer
# ===----------------------------------------------------------------------=== #


trait Indexer:
    """This trait denotes a type that can be used to index a container that
    handles integral index values.

    This solves the issue of being able to index data structures such as `List`
    with the various integral types without being too broad and allowing types
    that are coercible to `Int` (e.g. floating point values that have `__int__`
    method). In contrast to `Intable`, types conforming to `Indexer` must be
    convertible to `Int` in a lossless way.

    Note that types conforming to `Indexer` are implicitly convertible to `Int`.
    """

    fn __index__(self) -> Int:
        """Return the index value.

        Returns:
            The index value of the object.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  index
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn index[T: Indexer](idx: T, /) -> Int:
    """Returns the value of `__index__` for the given value.

    Parameters:
        T: A type conforming to the `Indexer` trait.

    Args:
        idx: The value.

    Returns:
        An `Int` representing the index value.
    """
    return idx.__index__()


# ===----------------------------------------------------------------------=== #
#  Intable
# ===----------------------------------------------------------------------=== #


trait Intable(CollectionElement):
    """The `Intable` trait describes a type that can be converted to an Int.

    Any type that conforms to `Intable` or
    [`IntableRaising`](/mojo/stdlib/builtin/int/IntableRaising) works with
    the built-in [`int()`](/mojo/stdlib/builtin/int/int-function) function.

    This trait requires the type to implement the `__int__()` method. For
    example:

    ```mojo
    @value
    struct Foo(Intable):
        var i: Int

        fn __int__(self) -> Int:
            return self.i
    ```

    Now you can use the `int()` function to convert a `Foo` to an
    `Int`:

    ```mojo
    %# from testing import assert_equal
    foo = Foo(42)
    assert_equal(int(foo), 42)
    ```

    **Note:** If the `__int__()` method can raise an error, use the
    [`IntableRaising`](/mojo/stdlib/builtin/int/intableraising) trait
    instead.
    """

    fn __int__(self) -> Int:
        """Get the integral representation of the value.

        Returns:
            The integral representation of the value.
        """
        ...


trait IntableRaising:
    """
    The `IntableRaising` trait describes a type can be converted to an Int, but
    the conversion might raise an error.

    Any type that conforms to [`Intable`](/mojo/stdlib/builtin/int/Intable)
    or `IntableRaising` works with the built-in
    [`int()`](/mojo/stdlib/builtin/int/int-function) function.

    This trait requires the type to implement the `__int__()` method, which can
    raise an error. For example:

    ```mojo
    @value
    struct Foo(IntableRaising):
        var i: Int

        fn __int__(self) raises -> Int:
            return self.i
    ```

    Now you can use the `int()` function to convert a `Foo` to an
    `Int`:

    ```mojo
    %# from testing import assert_equal
    foo = Foo(42)
    assert_equal(int(foo), 42)
    ```
    """

    fn __int__(self) raises -> Int:
        """Get the integral representation of the value.

        Returns:
            The integral representation of the type.

        Raises:
            If the type does not have an integral representation.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  IntLike
# ===----------------------------------------------------------------------=== #


trait IntLike(
    Absable,
    Ceilable,
    Floorable,
    Writable,
    Powable,
    Stringable,
    Truncable,
):
    """
    The `IntLike` trait is a tag for `Int` or `UInt`. This allows writing
    functions that works on either.
    """

    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  int
# ===----------------------------------------------------------------------=== #


@always_inline
fn int[T: Intable](value: T) -> Int:
    """Get the Int representation of the value.

    Parameters:
        T: The Intable type.

    Args:
        value: The object to get the integral representation of.

    Returns:
        The integral representation of the value.
    """
    return value.__int__()


@always_inline
fn int[T: IntableRaising](value: T) raises -> Int:
    """Get the Int representation of the value.

    Parameters:
        T: The Intable type.

    Args:
        value: The object to get the integral representation of.

    Returns:
        The integral representation of the value.

    Raises:
        If the type does not have an integral representation.
    """
    return value.__int__()


fn int(value: String, base: Int = 10) raises -> Int:
    """Parses and returns the given string as an integer in the given base.

    If base is set to 0, the string is parsed as an Integer literal, with the
    following considerations:
    - '0b' or '0B' prefix indicates binary (base 2)
    - '0o' or '0O' prefix indicates octal (base 8)
    - '0x' or '0X' prefix indicates hexadecimal (base 16)
    - Without a prefix, it's treated as decimal (base 10)

    Args:
        value: A string to be parsed as an integer in the given base.
        base: Base used for conversion, value must be between 2 and 36, or 0.

    Returns:
        An integer value that represents the string.

    Raises:
        If the given string cannot be parsed as an integer value or if an
        incorrect base is provided.

    Examples:
        >>> int("32")
        32
        >>> int("FF", 16)
        255
        >>> int("0xFF", 0)
        255
        >>> int("0b1010", 0)
        10

    Notes:
        This follows [Python's integer literals](
        https://docs.python.org/3/reference/lexical_analysis.html#integers).
    """
    return atol(value, base)


fn int(value: UInt) -> Int:
    """Get the Int representation of the value.

    Args:
        value: The object to get the integral representation of.

    Returns:
        The integral representation of the value.
    """
    return value.value


# ===----------------------------------------------------------------------=== #
#  Int
# ===----------------------------------------------------------------------=== #


@lldb_formatter_wrapping_type
@value
@register_passable("trivial")
struct Int(
    CeilDivable,
    Indexer,
    Intable,
    ImplicitlyBoolable,
    KeyElement,
    Roundable,
    IntLike,
    _HashableWithHasher,
):
    """This type represents an integer value."""

    # Fields
    var value: __mlir_type.index
    """The underlying storage for the integer value."""

    alias MAX = int(Scalar[DType.index].MAX)
    """Returns the maximum integer value."""

    alias MIN = int(Scalar[DType.index].MIN)
    """Returns the minimum value of type."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __init__(out self):
        """Default constructor that produces zero."""
        self.value = __mlir_op.`index.constant`[value = __mlir_attr.`0:index`]()

    fn __init__(out self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        """Construct Int from the given index value.

        Args:
            value: The init value.
        """
        self.value = value

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<si16>`):
        """Construct Int from the given Int16 value.

        Args:
            value: The init value.
        """
        self = Self(
            __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
                value
            )
        )

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<si32>`):
        """Construct Int from the given Int32 value.

        Args:
            value: The init value.
        """
        self = Self(
            __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
                value
            )
        )

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<si64>`):
        """Construct Int from the given Int64 value.

        Args:
            value: The init value.
        """
        self = Self(
            __mlir_op.`pop.cast`[_type = __mlir_type.`!pop.scalar<index>`](
                value
            )
        )

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<index>`):
        """Construct Int from the given Index value.

        Args:
            value: The init value.
        """
        self.value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.index](
            value
        )

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: IntLiteral):
        """Construct Int from the given IntLiteral value.

        Args:
            value: The init value.
        """
        self = value.__int__()

    @always_inline("nodebug")
    @implicit
    fn __init__[IndexerTy: Indexer](mut self, value: IndexerTy):
        """Construct Int from the given Indexer value.

        Parameters:
            IndexerTy: A type conforming to Indexer.

        Args:
            value: The init value.
        """
        self = value.__index__()

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: UInt):
        """Construct Int from the given UInt value.

        Args:
            value: The init value.
        """
        self = Self(value.value)

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __lt__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using LT comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is less-than the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate slt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __le__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using LE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is less-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sle>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using EQ comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using NE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is non-equal to the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __gt__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using GT comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is greater-than the RHS Int and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sgt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __ge__(self, rhs: Int) -> Bool:
        """Compare this Int to the RHS using GE comparison.

        Args:
            rhs: The other Int to compare against.

        Returns:
            True if this Int is greater-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate sge>`
        ](self.value, rhs.value)

    @always_inline("nodebug")
    fn __pos__(self) -> Int:
        """Return +self.

        Returns:
            The +self value.
        """
        return self

    @always_inline("nodebug")
    fn __neg__(self) -> Int:
        """Return -self.

        Returns:
            The -self value.
        """
        return __mlir_op.`index.mul`(
            self.value,
            __mlir_op.`index.constant`[value = __mlir_attr.`-1:index`](),
        )

    @always_inline("nodebug")
    fn __invert__(self) -> Int:
        """Return ~self.

        Returns:
            The ~self value.
        """
        return self ^ -1

    @always_inline("nodebug")
    fn __add__(self, rhs: Int) -> Int:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return __mlir_op.`index.add`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __sub__(self, rhs: Int) -> Int:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return __mlir_op.`index.sub`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __mul__(self, rhs: Int) -> Int:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return __mlir_op.`index.mul`(self.value, rhs.value)

    fn __truediv__(self, rhs: Int) -> Float64:
        """Return the floating point division of `self` and `rhs`.

        Args:
            rhs: The value to divide on.

        Returns:
            `float(self)/float(rhs)` value.
        """
        return Float64(self) / Float64(rhs)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: Int) -> Int:
        """Return the division of `self` and `rhs` rounded down to the nearest
        integer.

        Args:
            rhs: The value to divide on.

        Returns:
            `floor(self/rhs)` value.
        """
        # This should raise an exception
        var denominator = select(rhs == 0, 1, rhs)
        var div: Int = self._positive_div(denominator)

        var mod = self - div * rhs
        var div_mod = select(((rhs < 0) ^ (self < 0)) & mod, div - 1, div)
        div = select(self > 0 & rhs > 0, div, div_mod)
        div = select(rhs == 0, 0, div)
        return div

    @always_inline("nodebug")
    fn __mod__(self, rhs: Int) -> Int:
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        var denominator = select(rhs == 0, 1, rhs)
        var div: Int = self._positive_div(denominator)

        var mod = self - div * rhs
        var div_mod = select(((rhs < 0) ^ (self < 0)) & mod, mod + rhs, mod)
        mod = select(
            self > 0 & rhs > 0, self._positive_rem(denominator), div_mod
        )
        mod = select(rhs == 0, 0, mod)
        return mod

    @always_inline("nodebug")
    fn __divmod__(self, rhs: Int) -> Tuple[Int, Int]:
        """Computes both the quotient and remainder using integer division.

        Args:
            rhs: The value to divide on.

        Returns:
            The quotient and remainder as a `Tuple(self // rhs, self % rhs)`.
        """
        if rhs == 0:
            return 0, 0
        var div: Int = self._positive_div(rhs)
        if rhs > 0 & self > 0:
            return div, self._positive_rem(rhs)
        var mod = self - div * rhs
        if ((rhs < 0) ^ (self < 0)) & mod:
            return div - 1, mod + rhs
        return div, mod

    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
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

    @always_inline("nodebug")
    fn __lshift__(self, rhs: Int) -> Int:
        """Return `self << rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self << rhs`.
        """
        if rhs < 0:
            # this should raise an exception.
            return 0
        return __mlir_op.`index.shl`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __rshift__(self, rhs: Int) -> Int:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        if rhs < 0:
            # this should raise an exception.
            return 0
        return __mlir_op.`index.shrs`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __and__(self, rhs: Int) -> Int:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return __mlir_op.`index.and`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __xor__(self, rhs: Int) -> Int:
        """Return `self ^ rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        return __mlir_op.`index.xor`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __or__(self, rhs: Int) -> Int:
        """Return `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return __mlir_op.`index.or`(self.value, rhs.value)

    # ===-------------------------------------------------------------------===#
    # In place operations.
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: Int):
        """Compute `self + rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: Int):
        """Compute `self - rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: Int):
        """Compute self*rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self * rhs

    fn __itruediv__(mut self, rhs: Int):
        """Compute `self / rhs`, convert to int, and save the result in self.

        Since `floor(self / rhs)` is equivalent to `self // rhs`, this yields
        the same as `__ifloordiv__`.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: Int):
        """Compute `self // rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    fn __imod__(mut self, rhs: Int):
        """Compute `self % rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self % rhs

    @always_inline("nodebug")
    fn __ipow__(mut self, rhs: Int):
        """Compute `pow(self, rhs)` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self**rhs

    @always_inline("nodebug")
    fn __ilshift__(mut self, rhs: Int):
        """Compute `self << rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: Int):
        """Compute `self >> rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self >> rhs

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: Int):
        """Compute `self & rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: Int):
        """Compute `self ^ rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: Int):
        """Compute self|rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self | rhs

    # ===-------------------------------------------------------------------===#
    # Reversed operations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __radd__(self, value: Int) -> Int:
        """Return `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        return self + value

    @always_inline("nodebug")
    fn __rsub__(self, value: Int) -> Int:
        """Return `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        return value - self

    @always_inline("nodebug")
    fn __rmul__(self, value: Int) -> Int:
        """Return `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        return self * value

    @always_inline("nodebug")
    fn __rfloordiv__(self, value: Int) -> Int:
        """Return `value // self`.

        Args:
            value: The other value.

        Returns:
            `value // self`.
        """
        return value // self

    @always_inline("nodebug")
    fn __rmod__(self, value: Int) -> Int:
        """Return `value % self`.

        Args:
            value: The other value.

        Returns:
            `value % self`.
        """
        return value % self

    @always_inline("nodebug")
    fn __rpow__(self, value: Int) -> Int:
        """Return `pow(value,self)`.

        Args:
            value: The other value.

        Returns:
            `pow(value,self)`.
        """
        return value**self

    @always_inline("nodebug")
    fn __rlshift__(self, value: Int) -> Int:
        """Return `value << self`.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        return value << self

    @always_inline("nodebug")
    fn __rrshift__(self, value: Int) -> Int:
        """Return `value >> self`.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        return value >> self

    @always_inline("nodebug")
    fn __rand__(self, value: Int) -> Int:
        """Return `value & self`.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        return value & self

    @always_inline("nodebug")
    fn __ror__(self, value: Int) -> Int:
        """Return `value | self`.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        return value | self

    @always_inline("nodebug")
    fn __rxor__(self, value: Int) -> Int:
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
        """Convert this Int to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self != 0

    @always_inline("nodebug")
    fn __as_bool__(self) -> Bool:
        """Convert this Int to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self.__bool__()

    @always_inline("nodebug")
    fn __index__(self) -> Int:
        """Return self converted to an integer, if self is suitable for use as
        an index into a list.

        For Int type this is simply the value.

        Returns:
            The corresponding Int value.
        """
        return self

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Gets the integral value (this is an identity function for Int).

        Returns:
            The value as an integer.
        """
        return self

    @always_inline("nodebug")
    fn __abs__(self) -> Self:
        """Return the absolute value of the Int value.

        Returns:
            The absolute value.
        """
        return select(self < 0, -self, self)

    @always_inline("nodebug")
    fn __ceil__(self) -> Self:
        """Return the ceiling of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("nodebug")
    fn __floor__(self) -> Self:
        """Return the floor of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("nodebug")
    fn __round__(self) -> Self:
        """Return the rounded value of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("nodebug")
    fn __round__(self, ndigits: Int) -> Self:
        """Return the rounded value of the Int value, which is itself.

        Args:
            ndigits: The number of digits to round to.

        Returns:
            The Int value itself if ndigits >= 0 else the rounded value.
        """
        if ndigits >= 0:
            return self
        return self - (self % 10 ** -(ndigits))

    @always_inline("nodebug")
    fn __trunc__(self) -> Self:
        """Return the truncated Int value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @no_inline
    fn __str__(self) -> String:
        """Get the integer as a string.

        Returns:
            A string representation.
        """

        return String.write(self)

    @no_inline
    fn __repr__(self) -> String:
        """Get the integer as a string. Returns the same `String` as `__str__`.

        Returns:
            A string representation.
        """
        return str(self)

    fn __hash__(self) -> UInt:
        """Hash the int using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        # TODO(MOCO-636): switch to DType.index
        return _hash_simd(Scalar[DType.int64](self))

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with this int value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(Int64(self))

    @doc_private
    @staticmethod
    fn try_from_python(obj: PythonObject, out result: Self) raises:
        """Construct an `Int` from a Python integer value.

        Raises:
            An error if conversion failed.
        """

        result = Python.py_long_as_ssize_t(obj)

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return -(self // -denominator)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this integer to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write(Int64(self))

    fn write_padded[W: Writer](self, mut writer: W, width: Int):
        """Write the int right-aligned to a set padding.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
            width: The amount to pad to the left.
        """
        var int_width = self._decimal_digit_count()

        # TODO: Assumes user wants right-aligned content.
        if int_width < width:
            for _ in range(width - int_width):
                writer.write(" ")

        writer.write(self)

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self.value

    @always_inline("nodebug")
    fn _positive_div(self, rhs: Int) -> Int:
        """Return the division of `self` and `rhs` assuming that the arguments
        are both positive.

        Args:
            rhs: The value to divide on.

        Returns:
            The integer division of `self` and `rhs` .
        """
        return __mlir_op.`index.divs`(self.value, rhs.value)

    @always_inline("nodebug")
    fn _positive_rem(self, rhs: Int) -> Int:
        """Return the modulus of `self` and `rhs` assuming that the arguments
        are both positive.

        Args:
            rhs: The value to divide on.

        Returns:
            The integer modulus of `self` and `rhs` .
        """
        return __mlir_op.`index.rems`(self.value, rhs.value)

    fn _decimal_digit_count(self) -> Int:
        """
        Returns the number of decimal digits required to display this integer.

        Note that if this integer is negative, the returned count does not
        include space to store a leading minus character.

        Returns:
            A count of the number of decimal digits required to display this integer.

        Examples:

        ```mojo
        %# from testing import assert_equal
        assert_equal(Int(10)._decimal_digit_count(), 2)
        assert_equal(Int(-10)._decimal_digit_count(), 2)
        ```
        .
        """

        var n = abs(self)

        alias is_32bit_system = bitwidthof[DType.index]() == 32

        @parameter
        if is_32bit_system:
            return _calc_initial_buffer_size_int32(n)

        # The value only has low-bits.
        if n >> 32 == 0:
            return _calc_initial_buffer_size_int32(n)

        return _calc_initial_buffer_size_int64(n)
