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
"""Defines `Copyable` and `ImplicitlyCopyable`, the traits for types whose value can be copied.

These are Mojo built-ins, so you don't need to import them.
"""

from std.builtin.rebind import downcast
from std.traits.movable import Movable


@stable(since="1.0")
trait Copyable(Movable):
    """The Copyable trait denotes a type whose value can be explicitly copied.

    Example implementing the `Copyable` trait on `Foo`, which requires the
    `def __init__(out self,*, copy: Self)` method:

    ```mojo
    struct Foo(Copyable):
        var s: String

        def __init__(out self, s: String):
            self.s = s

        def __init__(out self, *, copy: Self):
            print("copying value")
            self.s = copy.s
    ```

    You can now copy objects inside a generic function:

    ```mojo
    def copy_return[T: Copyable](foo: T) -> T:
        var copy = foo.copy()
        return copy^

    var foo = Foo("test")
    var res = copy_return(foo)
    ```

    ```plaintext
    copying value
    ```
    """

    @stable(since="1.0")
    def __init__(out self, *, copy: Self):
        """Create a new instance of the value by copying an existing one.

        Args:
            copy: The value to copy.
        """
        ...

    @always_inline
    @stable(since="1.0")
    def copy(self) -> Self:
        """Explicitly construct a copy of self, a convenience method for
        `Self(copy=self)` when the type is inconvenient to write out.

        Overriding this method is not allowed.

        Returns:
            A copy of this value.
        """
        return Self(copy=self)

    comptime __copy_ctor_is_trivial: Bool
    """A flag (often compiler generated) to indicate whether the implementation
    of the copy constructor is trivial.

    A copy constructor is considered to be trivial if:
    - The struct has a compiler-generated trivial copy constructor because all
      its fields have trivial copy constructors.

    In practice, it means the value can be copied by copying the bits from
    one location to another without side effects.
    """


# TODO(MOCO-4525): Remove `downcast`
comptime IsTriviallyCopyable[T: AnyType]: Bool = conforms_to(
    T, TrivialRegisterPassable
) or (conforms_to(T, Copyable) and downcast[T, Copyable].__copy_ctor_is_trivial)
"""Indicates whether `T` is `Copyable` with a trivial copy initializer.

A copy initializer is trivial when the compiler generates it and all of
`T`'s fields are themselves trivially copyable — the value can be copied by
duplicating its bits to a new location with no additional side effects.
Evaluates to `False` for non-`Copyable` types.

Parameters:
    T: The type to check.
"""


@stable(since="1.0")
trait ImplicitlyCopyable(Copyable):
    """A marker trait to permit compiler to insert implicit calls to the copy
    constructor in order to make a copy of the object when needed.

    Conforming a type to `ImplicitlyCopyable` gives the Mojo language permission
    to implicitly insert a call to that types copy constructor whenever a borrowed
    instance of the type is passed or assigned where an owned value is required.

    Types that are expensive to copy, or where implicit copying could mask a
    logic error, typically should not be `ImplicitlyCopyable`.

    The `ImplicitlyCopyable` trait is a marker trait, meaning that it does not
    require a type to provide any additional methods or associated aliases to
    conform to this trait. However, all `ImplicitlyCopyable` types are required
    to conform to `Copyable`, which ensures there is only one definition for the
    logic of how a type is copied.

    **Note:** `ImplicitlyCopyable` should only be used to mark structs that may
    be copied implicitly. It should not be used as a trait bound
    (`T: ImplicitlyCopyable`) on functions or types, except in special
    circumstances. Generic code that may perform copies should always use the
    more general `T: Copyable` bound. This ensures that generic code is usable
    with all types that are copyable, regardless of whether they opt-in to
    implicit copying.

    **Examples:**

    A type can opt-in to implicit copying by conforming to `ImplicitlyCopyable`
    (in the example below, the compiler also synthesizes a default field-wise
    copy constructor, as the user didn't provide a definition):

    ```mojo
    @fieldwise_init
    struct Point(ImplicitlyCopyable):
        var x: Int
        var y: Int

    def main():
        var p = Point(5, 10)

        # Perform an implicit copy of `p`.
        var p2 = p
    ```
    """

    pass
