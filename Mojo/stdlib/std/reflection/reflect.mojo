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
"""Provides the unified `reflect[T]` / `Reflected[T]` reflection API.

`reflect[T]` is a `comptime` alias for the `Reflected[T]` handle type, which
exposes type introspection through static methods. The handle has no runtime
state — `T` is carried entirely in the compile-time parameter — so all queries
are spelled as `reflect[T].method()` (no parens after `[T]`).

- `is_struct()` - whether `T` is a Mojo struct type.
- `field_count()` - number of fields.
- `field_names()` - `Array[StaticString, N]` of field names.
- `field_types()` - a `TypeList` of field types.
- `field_index[name]()` - index of the named field.
- `field[name]` - `Reflected[FieldT]` for the named field's type.
- `field_at[idx]` - `Reflected[FieldT]` for the field at index `idx`.
- `field_offset[name=...]()` / `field_offset[index=...]()` - byte offset.
- `field_ref[idx](s)` - reference to field at index `idx` in value `s`.

`reflect` is auto-imported via the prelude, so it is available without
an explicit import. `Reflected[T]` must be imported from `std.reflection`
when named in signatures.

Example:

```mojo
struct Point:
    var x: Int
    var y: Float64

def print_fields[T: AnyType]():
    comptime names = reflect[T].field_names()
    comptime for i in range(reflect[T].field_count()):
        print(materialize[names[i]]())

def main():
    print_fields[Point]()
```

The wrapped type is exposed as the `T` parameter, so the result of
`field[name]` can be used as a type directly:

```mojo
def main():
    comptime y_type = reflect[Point].field["y"]
    var v: y_type.T = 3.14  # y_type.T is Float64
```
"""

from std.builtin.variadics import ParameterList, TypeList
from std.sys.info import _TargetType, _current_target


# ===----------------------------------------------------------------------=== #
# Implementation primitives
# ===----------------------------------------------------------------------=== #
#
# These KGEN attributes implement `ContextuallyEvaluatedAttrInterface`, which
# allows them to be evaluated during elaboration after generic type parameters
# have been specialized to concrete types. This is what lets reflection work in
# generic code:
#
#   def foo[T: AnyType]():
#       comptime count = reflect[T].field_count()
# ===----------------------------------------------------------------------=== #


comptime _field_types_of[T: AnyType] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_types<`, T, `> : !kgen.param_list<`, AnyType, `>`
    ]
]

comptime _field_names_of[T: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_names<`, T, `> : !kgen.param_list<!kgen.string>`
    ]
]


# ===----------------------------------------------------------------------=== #
# `reflect` / `Reflected[T]`
# ===----------------------------------------------------------------------=== #
#
# `reflect[T]` is a `comptime` alias for `Reflected[T]`, not a function. The
# handle type has no runtime fields — only the compile-time parameter `T` —
# so all introspection answers come from the parameter alone. Every method is
# `@staticmethod`. Spelling access as `reflect[T].method()` (no parens after
# `[T]`) keeps the elaboration-time work elaboration-time and removes the
# zero-sized-instance ceremony the previous instance form required.
# ===----------------------------------------------------------------------=== #


comptime reflect[T: AnyType] = Reflected[T]
"""A compile-time alias for the reflection handle type of `T`.

Resolves to `Reflected[T]`, whose static methods expose introspection of `T`.
Use it as `reflect[T].method()` rather than constructing an instance.

`reflect` is auto-imported via the prelude.

Parameters:
    T: The type to introspect.

Example:
    ```mojo
    struct Point:
        var x: Int
        var y: Float64

    def main():
        print(reflect[Point].field_count())  # 2
    ```
"""


struct Reflected[T: AnyType]:
    """A compile-time reflection handle type for a Mojo type.

    `Reflected[T]` exposes compile-time introspection of `T` through static
    methods. It has no runtime fields — `T` lives entirely in the compile-time
    parameter — and is not constructible. Spell access as `reflect[T].method()`
    (preferred) or `Reflected[T].method()`.

    Member shape — when to use `@staticmethod` vs a `comptime` alias:

    - A member that returns a **type** (e.g. `Reflected[FieldT]`) is a
      `comptime` member alias and is spelled without `()`. This keeps it
      composable in type position: `reflect[T].field["x"].T` reads as a
      type. `field[name]` and `field_at[idx]` are the type-returning
      members today.
    - A member that returns a **value** (an `Int`, `StaticString`,
      `Array`, a `TypeList`, a typed `ref`, etc.) is an
      `@staticmethod` and is spelled with `()` — e.g.
      `reflect[T].field_count()`, `reflect[T].field_names()`,
      `reflect[T].field_index["x"]()`. The `()` at the call site signals
      "evaluate this comptime expression to a value." `field_types`
      returns a `TypeList` value (not a type) and so is a static method.

    When adding a new member: pick a `comptime` alias if the result will be
    used in type position, `@staticmethod` if it will be assigned to a
    `comptime` variable or compared at the call site.

    For best performance, assign the result of static methods that return
    type-level values (such as `field_names`, `field_types`, `field_count`)
    to `comptime` variables so the work happens at compile time.

    Parameters:
        T: The type being introspected. The wrapped type is exposed via this
            parameter, so `reflect[T].T` is `T`.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime if reflect[Point].is_struct():
                comptime names = reflect[Point].field_names()
                comptime for i in range(reflect[Point].field_count()):
                    print(materialize[names[i]]())
        ```
    """

    @staticmethod
    @always_inline("builtin")
    def is_struct() -> Bool:
        """Returns `True` if `T` is a Mojo struct type, `False` otherwise.

        This distinguishes Mojo struct types from MLIR primitive types (such as
        `__mlir_type.index` or `__mlir_type.i64`). The other reflection methods
        produce a compile error on non-struct types, so `is_struct` is useful
        as a `comptime if` guard when iterating over field types that may
        contain MLIR primitives.

        Returns:
            `True` if `T` is a Mojo struct type, `False` if it is an MLIR type.

        Example:
            ```mojo
            def process_type[T: AnyType]():
                comptime if reflect[T].is_struct():
                    print("struct with", reflect[T].field_count(), "fields")
                else:
                    print("non-struct:", reflect[T].name())
            ```
        """
        return __mlir_attr[`#kgen.is_struct_type<`, Self.T, `> : i1`]

    @staticmethod
    def name[*, qualified_builtins: Bool = False]() -> StaticString:
        """Returns the struct name of `T`.

        Parameters:
            qualified_builtins: Whether to print fully qualified builtin type
                names (e.g. `std.builtin.int.Int`) or shorten them
                (e.g. `Int`).

        Returns:
            Type name.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                print(reflect[Point].name())  # "Point" (or module-qualified if defined)
            ```
        """
        return StaticString(
            __mlir_attr[
                `#kgen.get_type_name<`,
                Self.T,
                `, `,
                qualified_builtins._mlir_value,
                `> : !kgen.string`,
            ]
        )

    @staticmethod
    def base_name() -> StaticString:
        """Returns the name of the base type of a parameterized type.

        For parameterized types like `List[Int]`, this returns `"List"`.
        For non-parameterized types, it returns the type's simple name.

        Unlike `name`, this method strips type parameters and returns only the
        unqualified base type name.

        Returns:
            The unqualified name of the base type as a `StaticString`.

        Example:
            ```mojo
            from std.collections import List, Dict

            def main():
                print(reflect[List[Int]].base_name())          # "List"
                print(reflect[Dict[String, Int]].base_name())  # "Dict"
                print(reflect[Int].base_name())                # "Int"
            ```
        """
        return StaticString(
            __mlir_attr[
                `#kgen.get_base_type_name<`,
                Self.T,
                `> : !kgen.string`,
            ]
        )

    @staticmethod
    @always_inline("builtin")
    def field_count() -> Int:
        """Returns the number of fields in struct `T`.

        Constraints:
            `T` must be a struct type.

        Returns:
            The number of fields in the struct.
        """
        return _field_types_of[Self.T]().length

    @staticmethod
    def field_types() -> _field_types_of[Self.T]:
        """Returns the types of all fields in struct `T` as a `TypeList`.

        For nested structs this returns the struct type itself, not its
        flattened fields.

        Constraints:
            `T` must be a struct type.

        Returns:
            A `TypeList` with one entry per field in the struct.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                comptime types = reflect[Point].field_types()
                comptime for i in range(reflect[Point].field_count()):
                    print(reflect[types[i]].name())
            ```
        """
        return {}

    @staticmethod
    def field_names() -> Array[StaticString, _field_types_of[Self.T]().length]:
        """Returns the names of all fields in struct `T`.

        Constraints:
            `T` must be a struct type.

        Returns:
            An `Array` of `StaticString`, one entry per field.
        """
        comptime count = _field_types_of[Self.T]().length
        comptime raw = _field_names_of[Self.T]()

        # Safety: uninitialized=True is safe because the comptime for loop
        # below initializes every element.
        var result = Array[StaticString, count](uninitialized=True)

        comptime for i in range(raw.size):
            result[i] = comptime (StaticString(raw[i]))

        return result^

    @staticmethod
    def field_index[name: StringLiteral]() -> Int:
        """Returns the index of the field with the given name in struct `T`.

        Note: `T` must be a concrete type, not a generic type parameter, when
        looking up by name.

        Parameters:
            name: The name of the field to look up.

        Returns:
            The zero-based index of the field.
        """
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_index_by_name<`,
                Self.T,
                `, `,
                str_value,
                `> : index`,
            ]
        )

    # `field` is a parametric `comptime` member alias rather than a
    # static method, so callers spell `reflect[T].field["y"]` (no
    # parens) and get back `Reflected[FieldT]` directly. The result is
    # itself a reflection handle type — fully composable.
    comptime field[name: StringLiteral] = Reflected[
        __mlir_attr[
            `#kgen.struct_field_type_by_name<`,
            Self.T,
            `, `,
            name.value,
            `> : `,
            AnyType,
        ]
    ]
    """A reflection handle type for the named field's type.

    The result is `Reflected[FieldT]`, so `reflect[T].field["x"].T` can
    be used in type position and `.name()`, `.field_count()`, etc. compose
    directly without an additional `()`.

    Note: `T` must be a concrete type, not a generic type parameter, when
    looking up by name.

    Parameters:
        name: The name of the field.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime y_type = reflect[Point].field["y"]
            var v: y_type.T = 3.14  # y_type.T is Float64
        ```
    """

    # A separate name (not an overload of `field`) because `comptime`
    # member aliases cannot be overloaded on parameter type.
    comptime field_at[idx: Int] = Reflected[_field_types_of[Self.T]()[idx]]
    """A reflection handle type for the type of the field at the given index.

    The by-index dual of `field[name]`. Unlike the by-name form, it works
    when only the field index is available, such as inside a `comptime for` over
    a struct's fields, and `T` may be a generic type parameter. The result is
    `Reflected[FieldT]`, so `reflect[T].field_at[idx].T` reads as a type.

    Parameters:
        idx: The zero-based index of the field.

    Constraints:
        `T` must be a struct type. `idx` must be in range `[0, field_count())`.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime y_type = reflect[Point].field_at[1]
            var v: y_type.T = 3.14  # y_type.T is Float64
        ```
    """

    # `nodebug` (not `builtin`) because the body emits a `lit.ref.struct.ger`
    # MLIR op, which is not legal inside a `@always_inline("builtin")` function.
    @staticmethod
    @always_inline("nodebug")
    def field_ref[
        idx: Int
    ](ref s: Self.T) -> ref[s] _field_types_of[Self.T]()[idx]:
        """Returns a reference to the field at the given index in `s`.

        Returns a reference rather than a copy, so this works with
        non-copyable field types and supports mutation through the result.

        Parameters:
            idx: The zero-based index of the field.

        Args:
            s: The struct value to access.

        Constraints:
            `T` must be a struct type. `idx` must be in range
            `[0, field_count())`.

        Returns:
            A reference to the field at the specified index, with the same
            mutability as `s`.

        Example:
            ```mojo
            @fieldwise_init
            struct Container:
                var id: Int
                var name: String

            def main():
                var c = Container(id=1, name="test")
                reflect[Container].field_ref[0](c) = 42  # mutates c.id
            ```
        """
        # Emit `lit.ref.struct.ger` with index-access form. The op accepts
        # StructType, ParamType, and ClosureType element types, so this works
        # for concrete structs, generic type parameters, and closures alike.
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.struct.ger`[
                index=idx.__mlir_index__(),
                _type=__mlir_type[
                    `!lit.ref<`,
                    _field_types_of[Self.T]()[idx],
                    `, `,
                    origin_of(s)._mlir_origin,
                    `>`,
                ],
            ](__get_mvalue_as_litref(s))
        )

    @staticmethod
    def field_offset[
        *,
        name: StringLiteral,
        target: _TargetType = _current_target(),
    ]() -> Int:
        """Returns the byte offset of the named field within struct `T`.

        Accounts for alignment padding between fields. Computed using the
        target's data layout.

        Parameters:
            name: The name of the field.
            target: The target architecture (defaults to the current target).

        Constraints:
            `T` must be a struct type with a field of the given name.

        Returns:
            The byte offset of the field from the start of the struct.

        Example:
            ```mojo
            struct Point:
                var x: Int      # offset 0
                var y: Float64  # offset 8

            def main():
                comptime x_off = reflect[Point].field_offset[name="x"]()  # 0
                comptime y_off = reflect[Point].field_offset[name="y"]()  # 8
            ```
        """
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_offset_by_name<`,
                Self.T,
                `, `,
                str_value,
                `, `,
                target,
                `> : index`,
            ]
        )

    @staticmethod
    def field_offset[
        *,
        index: Int,
        target: _TargetType = _current_target(),
    ]() -> Int:
        """Returns the byte offset of the field at the given index.

        Accounts for alignment padding between fields. Computed using the
        target's data layout.

        Parameters:
            index: The zero-based index of the field.
            target: The target architecture (defaults to the current target).

        Constraints:
            `T` must be a struct type. `index` must be in range
            `[0, field_count())`.

        Returns:
            The byte offset of the field from the start of the struct.
        """
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_offset_by_index<`,
                Self.T,
                `, `,
                index.__mlir_index__(),
                `, `,
                target,
                `> : index`,
            ]
        )
