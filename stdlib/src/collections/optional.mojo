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
from collections.optional import Optional
var a = Optional(1)
var b = Optional[Int](None)
if a:
    print(a.value())  # prints 1
if b:  # b is False, so no print
    print(b.value())
var c = a.or_else(2)
var d = b.or_else(2)
print(c.value())  # prints 1
print(d.value())  # prints 2
```
"""

from utils.variant import Variant


# TODO(27780): NoneType can't currently conform to traits
@value
struct _NoneType(CollectionElement):
    pass


@value
struct Optional[T: CollectionElement](CollectionElement, Boolable):
    """A type modeling a value which may or may not be present.

    Optional values can be thought of as a type-safe nullable pattern.
    Your value can take on a value or `None`, and you need to check
    and explicitly extract the value to get it out.

    Currently T is required to be a `CollectionElement` so we can implement
    copy/move for Optional and allow it to be used in collections itself.

    ```mojo
    from collections.optional import Optional
    var a = Optional(1)
    var b = Optional[Int](None)
    if a:
        print(a.value())  # prints 1
    if b:  # b is False, so no print
        print(b.value())
    var c = a.or_else(2)
    var d = b.or_else(2)
    print(c.value())  # prints 1
    print(d.value())  # prints 2
    ```

    Parameters:
        T: The type of value stored in the Optional.
    """

    # _NoneType comes first so its index is 0.
    # This means that Optionals that are 0-initialized will be None.
    alias _type = Variant[_NoneType, T]
    var _value: Self._type

    fn __init__(inout self):
        """Construct an empty Optional."""
        self._value = Self._type(_NoneType())

    fn __init__(inout self, owned value: T):
        """Construct an Optional containing a value.

        Args:
            value: The value to store in the optional.
        """
        self._value = Self._type(value ^)

    fn __init__(inout self, value: NoneType):
        """Construct an empty Optional.

        Args:
            value: Must be exactly `None`.
        """
        self = Self()

    fn value(self) -> T:
        """Unsafely retrieve the value out of the Optional.

        This function currently creates a copy. Once we have lifetimes
        we'll be able to have it return a reference.

        This doesn't check to see if the optional contains a value.
        If you call this without first verifying the optional with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), you'll get garbage unsafe data out.

        Returns:
            The contained data of the option as a T value.
        """
        debug_assert(self.__bool__(), ".value() on empty Optional")
        return self._value.get[T]()[]

    fn take(owned self) -> T:
        """Unsafely move the value out of the Optional.

        The caller takes ownership over the new value, and the Optional is
        destroyed.

        This doesn't check to see if the optional contains a value.
        If you call this without first verifying the optional with __bool__()
        eg. by `if my_option:` or without otherwise knowing that it contains a
        value (for instance with `or_else`), you'll get garbage unsafe data out.

        Returns:
            The contained data of the option as an owned T value.
        """
        debug_assert(self.__bool__(), ".take() on empty Optional")
        return self._value.take[T]()

    fn or_else(self, default: T) -> T:
        """Return the underlying value contained in the Optional or a default value if the Optional's underlying value is not present.

        Args:
            default: The new value to use if no value was present.

        Returns:
            The underlying value contained in the Optional or a default value.
        """
        if self.__bool__():
            return self._value.get[T]()[]
        return default

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

    fn __and__[type: Boolable](self, other: type) -> Bool:
        """Return true if self has a value and the other value is coercible to
        True.

        Parameters:
            type: Type coercible to Bool.

        Args:
            other: Value to compare to.

        Returns:
            True if both inputs are True after boolean coercion.
        """
        return self.__bool__() and other.__bool__()

    fn __rand__[type: Boolable](self, other: type) -> Bool:
        """Return true if self has a value and the other value is coercible to
        True.

        Parameters:
            type: Type coercible to Bool.

        Args:
            other: Value to compare to.

        Returns:
            True if both inputs are True after boolean coercion.
        """
        return self.__bool__() and other.__bool__()

    fn __or__[type: Boolable](self, other: type) -> Bool:
        """Return true if self has a value or the other value is coercible to
        True.

        Parameters:
            type: Type coercible to Bool.

        Args:
            other: Value to compare to.

        Returns:
            True if either inputs is True after boolean coercion.
        """
        return self.__bool__() or other.__bool__()

    fn __ror__[type: Boolable](self, other: type) -> Bool:
        """Return true if self has a value or the other value is coercible to
        True.

        Parameters:
            type: Type coercible to Bool.

        Args:
            other: Value to compare to.

        Returns:
            True if either inputs is True after boolean coercion.
        """
        return self.__bool__() or other.__bool__()
