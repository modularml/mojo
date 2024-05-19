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
"""Defines Result, a type modeling a value which may or may not be present.
With a `Result.err` field which gives the error returned by the function.

Result values can be thought of as a type-safe nullable pattern.
Your value can take on a value or `None`, and you need to check
and explicitly extract the value to get it out.
```mojo
from collections import Result
var a = Result(1)
var b = Result[Int]()
if a:
    print(a.value()[])  # prints 1
if b:  # bool(b) is False, so no print
    print(b.value()[])
var c = a.or_else(2)
var d = b.or_else(2)
print(c)  # prints 1
print(d)  # prints 2
```

And if more information about the returned Error is wanted it is available.
```mojo
from collections import Result
var a = Result(1)
var b = Result[Int](err=Error("something went wrong"))
var c = Result[Int](err=Error("error 1"))
var d = Result[Int](err=Error("error 2"))
if a:
    print(a.err)  # prints ""
if not b:
    print(b.err) # prints "something went wrong"

if c.err:
    print("c had an error")

# TODO: pattern matching
if d.err == "error 1":
    print("d had error 1")
elif d.err == "error 2":
    print("d had error 2")
```
.
"""

from utils import Variant


# TODO(27780): NoneType can't currently conform to traits
@value
struct _NoneType(CollectionElement):
    pass


# ===----------------------------------------------------------------------===#
# Result
# ===----------------------------------------------------------------------===#


@value
struct Result[T: CollectionElement](CollectionElement, Boolable):
    """A type modeling a value which may or may not be present.

    Result values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    Currently T is required to be a `CollectionElement` so we can implement
    copy/move for Result and allow it to be used in collections itself.

    ```mojo
    from collections import Result
    var a = Result(1)
    var b = Result[Int]()
    if a:
        print(a.value()[])  # prints 1
    if b:  # bool(b) is False, so no print
        print(b.value()[])
    var c = a.or_else(2)
    var d = b.or_else(2)
    print(c)  # prints 1
    print(d)  # prints 2
    ```

    And if more information about the returned Error is wanted it is available.
    ```mojo
    from collections import Result
    var a = Result(1)
    var b = Result[Int](err=Error("something went wrong"))
    var c = Result[Int](err=Error("error 1"))
    var d = Result[Int](err=Error("error 2"))
    if a:
        print(a.err)  # prints ""
    if not b:
        print(b.err) # prints "something went wrong"

    if c.err:
        print("c had an error")

    # TODO: pattern matching
    if d.err == "error 1":
        print("d had error 1")
    elif d.err == "error 2":
        print("d had error 2")
    ```

    Parameters:
        T: The type of value stored in the `Result`.
    """

    # _NoneType comes first so its index is 0.
    # This means that Results that are 0-initialized will be None.
    alias _type = Variant[_NoneType, T]
    var _value: Self._type
    var err: Error
    """The Error inside the `Result`."""

    fn __init__(inout self):
        """Create an empty `Result`."""
        self._value = Self._type(_NoneType())
        self.err = Error("Result value was not set")

    fn __init__(inout self, value: NoneType):
        """Create an empty `Result`.

        Args:
            value: Must be exactly `None`.
        """
        self = Self()

    fn __init__[A: CollectionElement](inout self, owned other: Result[A]):
        """Create a `Result` by transferring another `Result`'s Error.

        Parameters:
            A: The type of the value contained in other.

        Args:
            other: The other `Result`.
        """
        self = Self(err=other.err)

    fn __init__(inout self, owned value: T):
        """Create a `Result` containing a value.

        Args:
            value: The value to store in the `Result`.
        """
        self._value = Self._type(value^)
        self.err = Error()

    fn __init__(inout self, *, err: Error):
        """Create an empty `Result`.

        Args:
            err: Must be an `Error`.
        """
        self._value = Self._type(_NoneType())
        self.err = err

    @always_inline
    fn value(
        self: Reference[Self, _, _]
    ) -> Reference[T, self.is_mutable, self.lifetime]:
        """Retrieve a reference to the value of the `Result`.

        This check to see if the `Result` contains a value.
        If you call this without first verifying the `Result` with __bool__()
        eg. by `if my_result:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), the program will abort

        Returns:
            A reference to the contained data of the `Result` as a Reference[T].
        """
        if not self[]:
            abort(".value() on empty `Result`")

        return self[].unsafe_value()

    @always_inline
    fn unsafe_value(
        self: Reference[Self, _, _]
    ) -> Reference[T, self.is_mutable, self.lifetime]:
        """Unsafely retrieve a reference to the value of the `Result`.

        This doesn't check to see if the `Result` contains a value.
        If you call this without first verifying the `Result` with __bool__()
        eg. by `if my_result:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), you'll get garbage unsafe data out.

        Returns:
            A reference to the contained data of the `Result` as a Reference[T].
        """
        debug_assert(self[], ".value() on empty Result")
        return self[]._value[T]

    @always_inline
    fn _value_copy(self) -> T:
        """Unsafely retrieve the value out of the `Result`.

        Note: only used for Results when used in a parameter context
        due to compiler bugs.  In general, prefer using the public `Result.value()`
        function that returns a `Reference[T]`.
        """

        debug_assert(self, ".value() on empty Result")
        return self._value[T]

    fn take(inout self) -> T:
        """Move the value out of the `Result`.

        The caller takes ownership over the new value, which is moved
        out of the `Result`, and the `Result` is left in an empty state.

        This check to see if the `Result` contains a value.
        If you call this without first verifying the `Result` with __bool__()
        eg. by `if my_result:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), you'll get garbage unsafe data out.

        Returns:
            The contained data of the `Result` as an owned T value.
        """
        if not self:
            abort(".take() on empty `Result`")
        return self.unsafe_take()

    fn unsafe_take(inout self) -> T:
        """Unsafely move the value out of the `Result`.

        The caller takes ownership over the new value, which is moved
        out of the `Result`, and the `Result` is left in an empty state.

        This check to see if the `Result` contains a value.
        If you call this without first verifying the `Result` with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), the program will abort!

        Returns:
            The contained data of the option as an owned T value.
        """
        debug_assert(self, ".unsafe_take() on empty Result")
        return self._value.unsafe_take[T]()

    fn or_else(self, default: T) -> T:
        """Return the underlying value contained in the `Result` or a default
        value if the `Result`'s underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Result or a default value.
        """
        if self:
            return self._value[T]
        return default

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the `Result` has no value.

        It allows you to use the following syntax: `if my_result is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the `Result` has no value and False otherwise.
        """
        return not self

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the `Result` has a value.

        It allows you to use the following syntax: `if my_result is not None:`.

        Args:
            other: The value to compare to (None).

        Returns:
            True if the `Result` has a value and False otherwise.
        """
        return self

    fn __bool__(self) -> Bool:
        """Return true if the `Result` has a value.

        Returns:
            True if the `Result` has a value and False otherwise.
        """
        return not self._value.isa[_NoneType]()

    fn __invert__(self) -> Bool:
        """Return False if the `Result` has a value.

        Returns:
            False if the `Result` has a value and True otherwise.
        """
        return not self


# ===----------------------------------------------------------------------===#
# ResultReg
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct ResultReg[T: AnyRegType](Boolable):
    """A register-passable `ResultReg` type.

    This struct `ResultReg` contains a value. It only works with trivial register
    passable types at the moment.

    Parameters:
        T: The type of value stored in the `ResultReg`.
    """

    alias _mlir_type = __mlir_type[`!kgen.variant<`, T, `, i1>`]
    var _value: Self._mlir_type
    var err: ErrorReg
    """The Error inside the `ResultReg`."""

    fn __init__(inout self):
        """Create a `ResultReg` with a value of None."""
        self = Self(err=ErrorReg("Result value was not set"))

    fn __init__(inout self, value: NoneType):
        """Create a `ResultReg` without a value from a None literal.

        Args:
            value: The None value.
        """
        self = Self()

    fn __init__[A: CollectionElement](inout self, owned other: ResultReg[A]):
        """Create a `ResultReg` by transferring another `ResultReg`'s Error.

        Parameters:
            A: The type of the value contained in other.

        Args:
            other: The other `ResultReg`.
        """
        self = Self(err=other.err)

    fn __init__(inout self, value: T):
        """Create a `ResultReg` with a value.

        Args:
            value: The value.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(0).value
        ](value)
        self.err = ErrorReg()

    fn __init__(inout self, *, err: ErrorReg):
        """Create a `ResultReg` without a value from an `ErrorReg`.

        Args:
            err: The `ErrorReg`.
        """
        self._value = __mlir_op.`kgen.variant.create`[
            _type = Self._mlir_type, index = Int(1).value
        ](__mlir_attr.false)
        self.err = err

    @always_inline
    fn value(self) -> T:
        """Get the `Result` value.

        Returns:
            The contained value.
        """
        return __mlir_op.`kgen.variant.take`[index = Int(0).value](self._value)

    fn __is__(self, other: NoneType) -> Bool:
        """Return `True` if the `Result` has no value.

        It allows you to use the following syntax: `if my_result is None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the `ResultReg` has no value and False otherwise.
        """
        return not self

    fn __isnot__(self, other: NoneType) -> Bool:
        """Return `True` if the `ResultReg` has a value.

        It allows you to use the following syntax: `if my_result is not None:`

        Args:
            other: The value to compare to (None).

        Returns:
            True if the Result has a value and False otherwise.
        """
        return self

    fn __bool__(self) -> Bool:
        """Return true if the `ResultReg` has a value.

        Returns:
            True if the `ResultReg` has a value and False otherwise.
        """
        return __mlir_op.`kgen.variant.is`[index = Int(0).value](self._value)
