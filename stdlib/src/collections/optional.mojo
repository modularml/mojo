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
"""Defines Optional, a type modeling a value which may or may not be present.

Optional values can be thought of as a type-safe nullable pattern.
Your value can take on a value or `None`, and you need to check
and explicitly extract the value to get it out.

```mojo
from collections import Optional
var a = Optional(1)
var b = Optional[Int](None)
if a:
    print(a.value())  # prints 1
if b:  # bool(b) is False, so no print
    print(b.value())
var c = a.or_else(2)
var d = b.or_else(2)
print(c)  # prints 1
print(d)  # prints 2
```
"""

from os import abort

from utils import Variant


# TODO(27780): NoneType can't currently conform to traits
@value
struct _NoneType(CollectionElement, CollectionElementNew):
    fn __init__(out self, *, other: Self):
        pass


# ===-----------------------------------------------------------------------===#
# Optional
# ===-----------------------------------------------------------------------===#


@value
struct Optional[T: CollectionElement](
    CollectionElement, CollectionElementNew, Boolable
):
    """A type modeling a value which may or may not be present.

    Optional values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    Currently T is required to be a `CollectionElement` so we can implement
    copy/move for Optional and allow it to be used in collections itself.

    ```mojo
    from collections import Optional
    var a = Optional(1)
    var b = Optional[Int](None)
    if a:
        print(a.value())  # prints 1
    if b:  # bool(b) is False, so no print
        print(b.value())
    var c = a.or_else(2)
    var d = b.or_else(2)
    print(c)  # prints 1
    print(d)  # prints 2
    ```

    Parameters:
        T: The type of value stored in the Optional.
    """

    # Fields
    # _NoneType comes first so its index is 0.
    # This means that Optionals that are 0-initialized will be None.
    alias _type = Variant[_NoneType, T]
    var _value: Self._type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Construct an empty Optional."""
        self._value = Self._type(_NoneType())

    @implicit
    fn __init__(out self, owned value: T):
        """Construct an Optional containing a value.

        Args:
            value: The value to store in the optional.
        """
        self._value = Self._type(value^)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initilaizer from a `NoneType`.
    @doc_private
    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self(value=NoneType(value))

    @implicit
    fn __init__(out self, value: NoneType):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self()

    fn __init__(out self, *, other: Self):
        """Copy construct an Optional.

        Args:
            other: The Optional to copy.
        """
        self.__copyinit__(other)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        It allows you to use the following syntax: `if my_optional is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.
        """
        return not self.__bool__()

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        It allows you to use the following syntax: `if my_optional is not None:`.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.
        """
        return self.__bool__()

    fn __eq__(self, rhs: NoneType) -> Bool:
        """Return `True` if a value is not present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `True` if a value is not present, `False` otherwise.
        """
        return self is None

    fn __eq__[
        T: EqualityComparableCollectionElement
    ](self: Optional[T], rhs: Optional[T]) -> Bool:
        """Return `True` if this is the same as another optional value, meaning
        both are absent, or both are present and have the same underlying value.

        Parameters:
            T: The type of the elements in the list. Must implement the
              traits `CollectionElement` and `EqualityComparable`.

        Args:
            rhs: The value to compare to.

        Returns:
            True if the values are the same.
        """
        if self:
            if rhs:
                return self.value() == rhs.value()
            return False
        return not rhs

    fn __ne__(self, rhs: NoneType) -> Bool:
        """Return `True` if a value is present.

        Args:
            rhs: The `None` value to compare to.

        Returns:
            `False` if a value is not present, `True` otherwise.
        """
        return self is not None

    fn __ne__[
        T: EqualityComparableCollectionElement
    ](self: Optional[T], rhs: Optional[T]) -> Bool:
        """Return `False` if this is the same as another optional value, meaning
        both are absent, or both are present and have the same underlying value.

        Parameters:
            T: The type of the elements in the list. Must implement the
              traits `CollectionElement` and `EqualityComparable`.

        Args:
            rhs: The value to compare to.

        Returns:
            False if the values are the same.
        """
        return not (self == rhs)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __bool__(self) -> Bool:
        """Return true if the Optional has a value.

        Returns:
            True if the optional has a value and False otherwise.
        """
        return not self._value.isa[_NoneType]()

    fn __invert__(self) -> Bool:
        """Return False if the optional has a value.

        Returns:
            False if the optional has a value and True otherwise.
        """
        return not self

    fn __str__[
        U: RepresentableCollectionElement, //
    ](self: Optional[U]) -> String:
        """Return the string representation of the value of the Optional.

        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the Optional.
        """
        var output = String()
        self.write_to(output)
        return output

    # TODO: Include the Parameter type in the string as well.
    fn __repr__[
        U: RepresentableCollectionElement, //
    ](self: Optional[U]) -> String:
        """Returns the verbose string representation of the Optional.

        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A verbose string representation of the Optional.
        """
        var output = String()
        output.write("Optional(")
        self.write_to(output)
        output.write(")")
        return output

    fn write_to[
        W: Writer, U: RepresentableCollectionElement, //
    ](self: Optional[U], mut writer: W):
        """Write Optional string representation to a `Writer`.

        Parameters:
            W: A type conforming to the Writable trait.
            U: The type of the elements in the list. Must implement the
              traits `Representable` and `CollectionElement`.

        Args:
            writer: The object to write to.
        """
        if self:
            writer.write(repr(self.value()))
        else:
            writer.write("None")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn value(ref self) -> ref [self._value] T:
        """Retrieve a reference to the value of the Optional.

        This check to see if the optional contains a value.
        If you call this without first verifying the optional with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), the program will abort

        Returns:
            A reference to the contained data of the option as a Pointer[T].
        """
        if not self.__bool__():
            abort(".value() on empty Optional")

        return self.unsafe_value()

    @always_inline
    fn unsafe_value(ref self) -> ref [self._value] T:
        """Unsafely retrieve a reference to the value of the Optional.

        This doesn't check to see if the optional contains a value.
        If you call this without first verifying the optional with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), you'll get garbage unsafe data out.

        Returns:
            A reference to the contained data of the option as a Pointer[T].
        """
        debug_assert(self.__bool__(), ".value() on empty Optional")
        return self._value.unsafe_get[T]()

    fn take(mut self) -> T:
        """Move the value out of the Optional.

        The caller takes ownership over the new value, which is moved
        out of the Optional, and the Optional is left in an empty state.

        This check to see if the optional contains a value.
        If you call this without first verifying the optional with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), you'll get garbage unsafe data out.

        Returns:
            The contained data of the option as an owned T value.
        """
        if not self.__bool__():
            abort(".take() on empty Optional")
        return self.unsafe_take()

    fn unsafe_take(mut self) -> T:
        """Unsafely move the value out of the Optional.

        The caller takes ownership over the new value, which is moved
        out of the Optional, and the Optional is left in an empty state.

        This check to see if the optional contains a value.
        If you call this without first verifying the optional with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), the program will abort!

        Returns:
            The contained data of the option as an owned T value.
        """
        debug_assert(self.__bool__(), ".unsafe_take() on empty Optional")
        return self._value.unsafe_replace[_NoneType, T](_NoneType())

    fn or_else(self, default: T) -> T:
        """Return the underlying value contained in the Optional or a default
        value if the Optional's underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Optional or a default value.
        """
        if self.__bool__():
            return self._value[T]
        return default


# ===-----------------------------------------------------------------------===#
# OptionalReg
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct OptionalReg[T: AnyTrivialRegType](Boolable):
    """A register-passable optional type.

    This struct optionally contains a value. It only works with trivial register
    passable types at the moment.

    Parameters:
        T: The type of value stored in the Optional.
    """

    # Fields
    alias _mlir_type = __mlir_type[`!kgen.variant<`, T, `, i1>`]
    var _value: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Create an optional with a value of None."""
        self = Self(None)

    @implicit
    fn __init__(out self, value: T):
        """Create an optional with a value.

        Args:
            value: The value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(0).value
        ](value)

    # TODO(MSTDL-715):
    #   This initializer should not be necessary, we should need
    #   only the initilaizer from a `NoneType`.
    @doc_private
    @implicit
    fn __init__(out self, value: NoneType._mlir_type):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self(value=NoneType(value))

    @implicit
    fn __init__(out self, value: NoneType):
        """Create an optional without a value from a None literal.

        Args:
            value: The None value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(1).value
        ](__mlir_attr.false)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has no value.

        It allows you to use the following syntax: `if my_optional is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has no value and False otherwise.
        """
        return not self.__bool__()

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the Optional has a value.

        It allows you to use the following syntax: `if my_optional is not None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Optional has a value and False otherwise.
        """
        return self.__bool__()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __bool__(self) -> Bool:
        """Return true if the optional has a value.

        Returns:
            True if the optional has a value and False otherwise.
        """
        return __mlir_op.`kgen.variant.is`[index = Int(0).value](self._value)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn value(self) -> T:
        """Get the optional value.

        Returns:
            The contained value.
        """
        return __mlir_op.`kgen.variant.get`[index = Int(0).value](self._value)

    fn or_else(self, default: T) -> T:
        """Return the underlying value contained in the Optional or a default
        value if the Optional's underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Optional or a default value.
        """
        if self:
            return self.value()
        return default
