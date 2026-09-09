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
"""Defines `Defaultable`, `RegisterPassable`, and `TrivialRegisterPassable`,
plus deprecated aliases for the value-semantics traits that used to live in
this module.

`Movable`, `Copyable`, and `ImplicitlyCopyable` have moved to `std.traits`.
They're still exported from the prelude, so most code doesn't need to
change; see the deprecated aliases below for the explicit import paths.
"""

import std.traits


@deprecated(
    "`Movable` has moved to `std.traits`. It's still exported from the"
    " prelude, so most code needs no changes; for the explicit import path"
    " use `std.traits.movable.Movable`."
)
comptime Movable = std.traits.Movable
"""Deprecated: The trait for types whose value can be moved.

This trait has moved to `std.traits.movable`. It's exported from the
prelude, so most code doesn't need to change. This alias will be removed in
a future version of Mojo."""


@deprecated(
    "`Copyable` has moved to `std.traits`. It's still exported from the"
    " prelude, so most code needs no changes; for the explicit import path"
    " use `std.traits.copyable.Copyable`."
)
comptime Copyable = std.traits.Copyable
"""Deprecated: The trait for types whose value can be explicitly copied.

This trait has moved to `std.traits.copyable`. It's exported from the
prelude, so most code doesn't need to change. This alias will be removed in
a future version of Mojo."""


@deprecated(
    "`ImplicitlyCopyable` has moved to `std.traits`. It's still exported"
    " from the prelude, so most code needs no changes; for the explicit"
    " import path use `std.traits.copyable.ImplicitlyCopyable`."
)
comptime ImplicitlyCopyable = std.traits.ImplicitlyCopyable
"""Deprecated: A marker trait to permit the compiler to insert implicit
copies.

This trait has moved to `std.traits.copyable`. It's exported from the
prelude, so most code doesn't need to change. This alias will be removed in
a future version of Mojo."""


def materialize[T: AnyType, //, value: T](out result: T):
    """Explicitly materialize a compile-time parameter into a run-time value.

    Parameters:
        T: The type of the value to materialize.
        value: The compile-time parameter value to materialize.

    Returns:
        The materialized run-time value.
    """
    __mlir_op.`lit.materialize_into`[value=value](
        __get_mvalue_as_litref(result)
    )


trait Defaultable:
    """The `Defaultable` trait describes a type with a default constructor.

    Implementing the `Defaultable` trait requires the type to define
    an `__init__` method with no arguments:

    ```mojo
    struct Foo(Defaultable):
        var s: String

        def __init__(out self):
            self.s = "default"
    ```

    You can now construct a generic `Defaultable` type:

    ```mojo
    def default_init[T: Defaultable]() -> T:
        return T()

    var foo = default_init[Foo]()
    print(foo.s)
    ```

    ```plaintext
    default
    ```
    """

    def __init__(out self):
        """Create a default instance of the value."""
        ...


trait TrivialRegisterPassable(
    std.traits.Deinitable,
    std.traits.ImplicitlyCopyable,
    std.traits.Movable,
    RegisterPassable,
):
    """A marker trait to denote the type to be register passable trivial.

     The compiler treats the type that conforms to this trait with the
     following constraints:

     - The type implicitly conforms to Copyable and the compiler synthesizes
       copy ctor that does a memcpy.
     - A trivial `__deinit__` member is synthesized by the compiler too,
       so the type can’t be a linear type.
     - All declared members are required to also conforms to this trait,
       since you can’t memcpy or trivially destroy a container if one
       of its stored members has a non-trivial copy constructor.
     - You are not allowed to define a custom copy ctor or `__deinit__`.


     ```mojo
    struct Foo(TrivialRegisterPassable):
        ...
     ```

    """

    pass


trait RegisterPassable(std.traits.Movable):
    """A marker trait to denote the type to be register passable.

     The compiler treats the type that conforms to this trait with the
     following constraints:

     - the value struct doesn’t have “identity” - you can’t take the
       address of self on read convention methods. This is allows
       the compiler to pass it in registers.

     - The type implicitly conforms to Movable and the compiler synthesizes
       a trivial move constructor. The compiler needs to be able to move around
       values of the type by loading and storing them. A custom
       move constructor is not allowed.

     - Compiler checks that any stored member (`var`s) also conforms to this
       trait. It wouldn’t be possible to provide identity for a contained
       member if the container doesn’t have identity.

     - The type can choose whether it wants to be Copyable or not.


     ```mojo
    struct Foo(RegisterPassable):
        ...
     ```

    """

    pass
