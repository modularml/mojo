# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines the `AnyType` trait.

These are Mojo built-ins, so you don't need to import them.
"""

# ===----------------------------------------------------------------------=== #
#  AnyType
# ===----------------------------------------------------------------------=== #


trait AnyType:
    """The AnyType trait describes a type that has a destructor.

    In Mojo, a type that provide a destructor indicates to the language that it
    is an object with a lifetime whose destructor needs to be called whenever
    an instance of the object reaches the end of its lifetime. Hence, only
    non-trivial types may have destructors.

    Any composition of types that have lifetimes is also an object with a
    lifetime, and the resultant type receives a destructor regardless of whether
    the user explicitly defines one.

    All types pessimistically require a destructor when used in generic
    functions. Hence, all Mojo traits are considered to inherit from
    AnyType, providing a default no-op destructor implementation for types
    that may need them.

    Example implementing the `AnyType` trait on `Foo` that frees the
    allocated memory:

    ```mojo
    @value
    struct Foo(AnyType):
        var p: Pointer[Int]
        var size: Int

        fn __init__(inout self, size: Int):
            self.p = Pointer[Int].alloc(size)
            self.size = size

        fn __del__(owned self):
            print("--freeing allocated memory--")
            self.p.free()
    ```
    """

    fn __del__(owned self, /):
        """Destroy the contained value.

        The destructor receives an owned value and is expected to perform any
        actions needed to end the lifetime of the object. In the simplest case,
        this is nothing, and the language treats the object as being dead at the
        end of this function.
        """
        ...
