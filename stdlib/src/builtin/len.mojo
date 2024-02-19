# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides the `len()` function and its associated traits.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------=== #
#  Sized
# ===----------------------------------------------------------------------=== #


trait Sized:
    """The `Sized` trait describes a type that has an integer length (such as a
    string or array).

    Any type that conforms to `Sized` or
    [`SizedRaising`](/mojo/stdlib/builtin/len.html#sizedraising) works with the
    built-in [`len()`](/mojo/stdlib/builtin/len.html#len) function.

    The `Sized` trait requires a type to implement the `__len__()`
    method. For example:

    ```mojo
    @value
    struct Foo(Sized):
        var length: Int

        fn __len__(self) -> Int:
            return self.length
    ```

    You can pass an instance of `Foo` to the `len()` function to get its
    length:

    ```mojo
    var foo = Foo(42)
    print(len(foo) == 42)
    ```

    ```plaintext
    True
    ```

    **Note:** If the `__len__()` method can raise an error, use the
    [`SizedRaising`](/mojo/stdlib/builtin/len.html#sizedraising) trait instead.

    """

    fn __len__(self) -> Int:
        """Get the length of the type.

        Returns:
            The length of the type.
        """
        ...


trait SizedRaising:
    """The `SizedRaising` trait describes a type that has an integer length,
    which might raise an error if the length can't be determined.

    Any type that conforms to [`Sized`](/mojo/stdlib/builtin/len.html#sized) or
    `SizedRaising` works with the built-in
    [`len()`](/mojo/stdlib/builtin/len.html#len) function.

    The `SizedRaising` trait requires a type to implement the `__len__()`
    method, which can raise an error. For example:

    ```mojo
    @value
    struct Foo(SizedRaising):
        var length: Int

        fn __len__(self) raises -> Int:
            if self.length < 0:
                raise Error("Length is negative")
            return self.length
    ```

    You can pass an instance of `Foo` to the `len()` function to get its
    length:

    ```mojo
    fn main() raises:
        var foo = Foo(42)
        print(len(foo) == 42)
    ```

    ```plaintext
    True
    ```
    """

    fn __len__(self) raises -> Int:
        """Get the length of the type.

        Returns:
            The length of the type.

        Raises:
            If the length cannot be computed.
        """
        ...


# ===----------------------------------------------------------------------=== #
#  len
# ===----------------------------------------------------------------------=== #


@always_inline
fn len[T: Sized](value: T) -> Int:
    """Get the length of a value.

    Parameters:
        T: The Sized type.

    Args:
        value: The object to get the length of.

    Returns:
        The length of the object.
    """
    return value.__len__()


@always_inline
fn len[T: SizedRaising](value: T) raises -> Int:
    """Get the length of a value.

    Parameters:
        T: The Sized type.

    Args:
        value: The object to get the length of.

    Returns:
        The length of the object.

    Raises:
        If the length cannot be computed.
    """
    return value.__len__()
