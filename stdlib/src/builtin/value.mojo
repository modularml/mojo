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
"""Defines core value traits.

These are Mojo built-ins, so you don't need to import them.
"""


trait Movable:
    """The Movable trait denotes a type whose value can be moved.

    Implement the `Movable` trait on `Foo` which requires the `__moveinit__`
    method:

    ```mojo
    struct Foo(Movable):
        fn __init__(inout self):
            pass

        fn __moveinit__(inout self, owned existing: Self):
            print("moving")
    ```

    You can now use the ^ suffix to move the object instead of copying
    it inside generic functions:

    ```mojo
    fn return_foo[T: Movable](owned foo: T) -> T:
        return foo^

    var foo = Foo()
    var res = return_foo(foo^)
    ```

    ```plaintext
    moving
    ```
    """

    fn __moveinit__(inout self, owned existing: Self, /):
        """Create a new instance of the value by moving the value of another.

        Args:
            existing: The value to move.
        """
        ...


trait Copyable:
    """The Copyable trait denotes a type whose value can be copied.

    Example implementing the `Copyable` trait on `Foo` which requires the `__copyinit__`
    method:

    ```mojo
    struct Foo(Copyable):
        var s: String

        fn __init__(inout self, s: String):
            self.s = s

        fn __copyinit__(inout self, other: Self):
            print("copying value")
            self.s = other.s
    ```

    You can now copy objects inside a generic function:

    ```mojo
    fn copy_return[T: Copyable](foo: T) -> T:
        var copy = foo
        return copy

    var foo = Foo("test")
    var res = copy_return(foo)
    ```

    ```plaintext
    copying value
    ```
    """

    fn __copyinit__(inout self, existing: Self, /):
        """Create a new instance of the value by copying an existing one.

        Args:
            existing: The value to copy.
        """
        ...


trait CollectionElement(Copyable, Movable):
    """The CollectionElement trait denotes a trait composition
    of the `Copyable` and `Movable` traits.

    This is useful to have as a named entity since Mojo does not
    currently support anonymous trait compositions to constrain
    on `Copyable & Movable` in the parameter.
    """

    pass


trait StringableCollectionElement(CollectionElement, Stringable):
    """The StringableCollectionElement trait denotes a trait composition
    of the `CollectionElement` and `Stringable` traits.

    This is useful to have as a named entity since Mojo does not
    currently support anonymous trait compositions to constrain
    on `CollectionElement & Stringable` in the parameter.
    """

    pass


trait ComparableCollectionElement(CollectionElement, EqualityComparable):
    """
    This trait is a temporary solution to enable comparison of
    collection elements as utilized in the `index` and `count` methods of
    a list.
    This approach will be revised with the introduction of conditional trait
    conformances.
    """

    pass
