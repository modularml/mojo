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
"""Provide the `repr` function.

The functions and traits provided here are built-ins, so you don't need to import them.
"""


trait Representable:
    """A trait that describes a type that has a String representation.

    Any type that conforms to the `Representable` trait can be used with the
    `repr` function. Any conforming type must also implement the `__repr__` method.
    Here is an example:

    ```mojo
    @value
    struct Dog(Representable):
        var name: String
        var age: Int

        fn __repr__(self) -> String:
            return "Dog(name=" + repr(self.name) + ", age=" + repr(self.age) + ")"

    var dog = Dog("Rex", 5)
    print(repr(dog))
    # Dog(name='Rex', age=5)
    ```

    The method `__repr__` should compute the "official" string representation of a type.

    If at all possible, this should look like a valid Mojo expression
    that could be used to recreate a struct instance with the same
    value (given an appropriate environment).
    So a returned String of the form `module_name.SomeStruct(arg1=value1, arg2=value2)` is advised.
    If this is not possible, a string of the form `<...some useful description...>`
    should be returned.

    The return value must be a `String` instance.
    This is typically used for debugging, so it is important that the representation is information-rich and unambiguous.

    Note that when computing the string representation of a collection (`Dict`, `List`, `Set`, etc...),
    the `repr` function is called on each element, not the `str()` function.
    """

    fn __repr__(self) -> String:
        """Get the string representation of the type instance, if possible, compatible with Mojo syntax.

        Returns:
            The string representation of the instance.
        """
        pass


fn repr[T: Representable](value: T) -> String:
    """Returns the string representation of the given value.

    Args:
        value: The value to get the string representation of.

    Parameters:
        T: The type of `value`. Must implement the `Representable` trait.

    Returns:
        The string representation of the given value.
    """
    return value.__repr__()


fn repr(value: None) -> String:
    """Returns the string representation of `None`.

    Args:
        value: A `None` value.

    Returns:
        The string representation of `None`.
    """
    return "None"
