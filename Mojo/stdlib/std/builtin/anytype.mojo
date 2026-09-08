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
"""Existential-type helpers built on `AnyType`, plus deprecated aliases for
the traits that used to live in this module.

This module defines `Some` and `SomeTypeList`, aliases that let a function
signature express "any type conforming to trait `X`" without introducing an
explicit type parameter. It also keeps deprecated aliases for `AnyType` and
`Deinitable`, both of which are now defined in `std.traits`.
"""

from std.builtin.variadics import _MLIR
import std.traits


@deprecated(
    "`AnyType` has moved to `std.traits`. It's still exported from the"
    " prelude, so most code needs no changes; for the explicit import path"
    " use `std.traits.anytype.AnyType`."
)
comptime AnyType = std.traits.AnyType
"""Deprecated: The most basic trait that all Mojo types extend by default.

This trait has moved to `std.traits.anytype`. It's exported from the
prelude, so most code doesn't need to change. This alias will be removed in
a future version of Mojo."""


@deprecated(
    "`Deinitable` has moved to `std.traits`. It's still exported from the"
    " prelude, so most code needs no changes; for the explicit import path"
    " use `std.traits.deinitable.Deinitable`."
)
comptime Deinitable = std.traits.Deinitable
"""Deprecated: A trait for types that require lifetime management through
destructors.

This trait has moved to `std.traits.deinitable`. It's exported from the
prelude, so most code doesn't need to change. This alias will be removed in
a future version of Mojo."""


comptime __SomeImpl[Trait: type_of(std.traits.AnyType), T: Trait] = T
comptime __SomeTypeListImpl[
    Trait: type_of(std.traits.AnyType),
    values: _MLIR.KGENParamListType[Trait],
] = TypeList[Trait=Trait, values]()

comptime Some[Trait: type_of(std.traits.AnyType)] = __SomeImpl[Trait, ...]
"""An alias allowing users to tersely express that a function argument is an
instance of a type that implements a trait or trait composition.

For example, instead of writing

```mojo
def foo[T: Intable, //](x: T) -> Int:
    return x.__int__()
```

one can write:

```mojo
def foo(x: Some[Intable]) -> Int:
    return x.__int__()
```

Parameters:
    Trait: The trait or trait composition that the argument type must implement.
"""

comptime SomeTypeList[Trait: type_of(std.traits.AnyType)] = __SomeTypeListImpl[
    Trait, ...
]
"""An alias allowing users to tersely express that a function argument is a
list of types that implement a trait or trait composition. This is particularly
useful for variadic packs.

For example, instead of writing

```mojo
def foo[*arg_types: Copyable](*args: *arg_types) -> Int: ...
```

one can write:

```mojo
def foo(*args: *SomeTypeList[Copyable]) -> Int: ...
```

Parameters:
    Trait: The trait or trait composition that the argument types must implement.
"""
