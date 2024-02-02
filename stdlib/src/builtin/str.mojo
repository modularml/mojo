# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides the `str` function.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------=== #
# Stringable
# ===----------------------------------------------------------------------=== #


trait Stringable:
    """
    The `Stringable` trait describes a type that can be converted to a
    [`String`](https://docs.modular.com/mojo/stdlib/builtin/string.html).

    Any type that conforms to `Stringable` or
    [`StringableRaising`](/mojo/stdlib/builtin/str.html#stringableraising) works
    with the built-in [`print()`](/mojo/stdlib/builtin/io.html#print) and
    [`str()`](/mojo/stdlib/builtin/str.html) functions.

    The `Stringable` trait requires the type to define the `__str__()` method.
    For example:

    ```mojo
    @value
    struct Foo(Stringable):
        var s: String

        fn __str__(self) -> String:
            return self.s
    ```

    Now you can pass an instance of `Foo` to the `str()` function to get back a
    `String`:

    ```mojo
    let foo = Foo("test")
    print(str(foo) == "test")
    ```

    ```plaintext
    True
    ```

    **Note:** If the `__str__()` method might raise an error, use the
    [`StringableRaising`](/mojo/stdlib/builtin/str.html#stringableraising)
    trait, instead.
    """

    fn __str__(self) -> String:
        """Get the string representation of the type.

        Returns:
            The string representation of the type.
        """
        ...


trait StringableRaising:
    """The StringableRaising trait describes a type that can be converted to a
    [`String`](https://docs.modular.com/mojo/stdlib/builtin/string.html).

    Any type that conforms to
    [`Stringable`](/mojo/stdlib/builtin/str.html#stringable) or
    `StringableRaising` works with the built-in
    [`print()`](/mojo/stdlib/builtin/io.html#print) and
    [`str()`](/mojo/stdlib/builtin/str.html#str) functions.

    The `StringableRaising` trait requires the type to define the `__str__()`
    method, which can raise an error. For example:

    ```mojo
    @value
    struct Foo(StringableRaising):
        var s: String

        fn __str__(self) raises -> String:
            if self.s == "":
                raise Error("Empty String")
            return self.s
    ```

    Now you can pass an instance of `Foo` to the `str()` function to get back a
    `String`:

    ```mojo
    fn main() raises:
        let foo = Foo("test")
        print(str(foo) == "test")
    ```

    ```plaintext
    True
    ```
    """

    fn __str__(self) raises -> String:
        """Get the string representation of the type.

        Returns:
            The string representation of the type.

        Raises:
            If there is an error when computing the string representation of the type.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  str
# ===----------------------------------------------------------------------=== #


@always_inline
fn str[T: Stringable](value: T) -> String:
    """Get the string representation of a value.

    Parameters:
        T: The type conforming to Stringable.

    Args:
        value: The object to get the string representation of.

    Returns:
        The string representation of the object.
    """
    return value.__str__()


@always_inline
fn str(value: None) -> String:
    """Get the string representation of the `None` type.

    Args:
        value: The object to get the string representation of.

    Returns:
        The string representation of the object.
    """
    return "None"


@always_inline
fn str[T: StringableRaising](value: T) raises -> String:
    """Get the string representation of a value.

    Parameters:
        T: The type conforming to Stringable.

    Args:
        value: The object to get the string representation of.

    Returns:
        The string representation of the object.

    Raises:
        If there is an error when computing the string representation of the type.
    """
    return value.__str__()
