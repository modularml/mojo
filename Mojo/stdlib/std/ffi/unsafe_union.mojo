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
"""Defines an untagged union type for C FFI interoperability.

This module provides a C-style union type that allows storing different types
in the same memory location without runtime type tracking. Unlike `Variant`,
`UnsafeUnion` does not maintain a discriminant field, making it suitable for:

- Interfacing with C libraries that use union types
- Low-level memory manipulation and type punning
- Situations where memory layout must exactly match C unions

Warning: Using `UnsafeUnion` is inherently unsafe. Reading a value as the wrong
type results in undefined behavior, just like in C. Prefer `Variant` for safe,
type-checked sum types.
"""

from std.builtin.rebind import downcast
from std.format._utils import FormatStruct, Named, TypeNames
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from std.sys import align_of, size_of


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def _all_types_unique[*Ts: AnyType]() -> Bool:
    """Check if all types in the variadic pack are unique.

    Returns True if no type appears more than once, False otherwise.
    """

    comptime for i in range(Ts.length):
        comptime for j in range(i + 1, Ts.length):
            if Ts[i] == Ts[j]:
                return False
    return True


@always_inline("nodebug")
def _check_union_types[*Ts: AnyType]():
    """Compile-time check that union types are valid.

    This function enforces the constraints:
    - At least one type must be provided
    - All types must be unique (no duplicates)
    - All types must have trivial copy, move, and destroy operations
    """
    comptime assert Ts.length > 0, "UnsafeUnion requires at least one type"
    comptime assert _all_types_unique[
        *Ts
    ](), "UnsafeUnion requires all types to be unique"
    comptime assert Ts.all[
        IsTriviallyDeinitable
    ](), "UnsafeUnion requires all types to have trivial destructors"
    comptime assert Ts.all[
        IsTriviallyCopyable
    ](), "UnsafeUnion requires all types to have trivial copy constructors"
    comptime assert Ts.all[
        IsTriviallyMovable
    ](), "UnsafeUnion requires all types to have trivial move constructors"


# ===----------------------------------------------------------------------=== #
# Union
# ===----------------------------------------------------------------------=== #


struct UnsafeUnion[*Ts: AnyType](ImplicitlyCopyable, Movable, Writable):
    """An untagged union that can store any one of its element types.

    `UnsafeUnion` is an untagged (non-discriminated) union, similar to `union`
    in C. It provides a way to store different types in the same memory
    location, where only one value is active at a time. The size of an
    `UnsafeUnion` is the size of its largest element type, and its alignment
    is the strictest alignment requirement among its elements.

    **Important**: Unlike `Variant`, `UnsafeUnion` does NOT track which type is
    currently stored. Reading a value as the wrong type is undefined behavior.
    This type is primarily intended for C FFI interoperability.

    **Type requirements**: All element types must have trivial copy, move, and
    destroy operations. This ensures safe bitwise operations and matches the
    semantics of C unions which don't have constructors or destructors.

    Example:

    ```mojo
    from std.ffi import UnsafeUnion

    # Define a union that can hold Int32 or Float32
    comptime IntOrFloat = UnsafeUnion[Int32, Float32]

    # Create with an integer
    var u = IntOrFloat(Int32(42))
    print(u.unsafe_get[Int32]())  # => 42

    # Type punning (reinterpreting bits)
    var u2 = IntOrFloat(Float32(1.0))
    print(u2.unsafe_get[Int32]())  # => 1065353216 (IEEE 754 bits of 1.0f)
    ```

    Example for C FFI:

    ```mojo
    from std.ffi import UnsafeUnion

    # Matches C: union { int32_t i; float f; }
    comptime CUnion = UnsafeUnion[Int32, Float32]

    def call_c_function(u: CUnion):
        # Pass to C code expecting union type
        ...
    ```

    Parameters:
        Ts: The possible types that this union can hold. Must have at least
            one type, and all types must be unique.

    Constraints:
        The type list must contain at least one type, all types must be
        distinct (no duplicates allowed), and all types must have trivial
        copy, move, and destroy operations.
    """

    # Union uses bitwise copy/move and has no destructor, so all operations
    # are trivial regardless of element types.
    comptime __del__is_trivial = True
    comptime __copy_ctor_is_trivial = True
    comptime __move_ctor_is_trivial = True

    # Use pop.union directly for C-compatible memory layout
    comptime _union_type = __mlir_type[
        `!pop.union<[rebind(:`,
        type_of(Self.Ts.values),
        ` `,
        Self.Ts.values,
        `)]>`,
    ]
    var _storage: Self._union_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self, *, unsafe_uninitialized: ()):
        """Unsafely create an uninitialized `UnsafeUnion`.

        The storage contains garbage data. You must call `unsafe_set` before
        reading any value.

        Args:
            unsafe_uninitialized: Marker argument indicating this initializer
                is unsafe.
        """
        _check_union_types[*Self.Ts]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    def __init__[T: Movable](out self, var value: T):
        """Create a union initialized with the given value.

        Parameters:
            T: The type of the value. Must be one of the union's element types.

        Args:
            value: The value to initialize the union with.

        Example:

        ```mojo
        from std.ffi import UnsafeUnion

        var u = UnsafeUnion[Int32, Float32](Int32(42))
        ```
        """
        _check_union_types[*Self.Ts]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        comptime assert Self._is_element[
            T
        ](), "type is not a union element type"
        self._get_ptr[T]().unsafe_write(value^)

    def __init__(out self, *, copy: Self):
        """Creates a bitwise copy of the union.

        Args:
            copy: The union to copy from.
        """
        # Bitwise copy of the raw storage
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        self._storage = copy._storage

    def __init__(out self, *, deinit move: Self):
        """Move initializer for the union.

        Args:
            move: The union to move from.
        """
        # Bitwise move of the raw storage
        self._storage = move._storage

    # Note: No __deinit__ - UnsafeUnion doesn't know what type is stored, so it
    # cannot call destructors. Users must manually manage destruction if needed.
    # For trivial types (integers, floats, pointers) this is fine.

    # ===-------------------------------------------------------------------===#
    # Internal methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def _get_ptr[
        origin: Origin, address_space: AddressSpace, //, T: AnyType
    ](ref[origin, address_space] self) -> Pointer[
        T, origin, address_space=address_space
    ]:
        """Get a pointer to the storage interpreted as type T."""
        comptime assert Self._is_element[
            T
        ](), "type is not a union element type"
        var ptr = Pointer(to=self._storage)._get_kgen_pointer()
        var typed_ptr = __mlir_op.`pop.union.bitcast`[
            _type=Pointer[T, origin, address_space=address_space]._mlir_type,
        ](ptr)
        return {_mlir_value = typed_ptr}

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def unsafe_get[
        origin: Origin, address_space: AddressSpace, //, T: AnyType
    ](ref[origin, address_space] self) -> ref[origin, address_space] T:
        """Get a reference to the stored value as type T.

        Parameters:
            T: The type to interpret the stored value as. Must be one of the
                union's element types.

        Returns:
            A reference to the storage interpreted as type T.

        Safety:
            Reading as the wrong type is undefined behavior.

        Example:

        ```mojo
        from std.ffi import UnsafeUnion

        var u = UnsafeUnion[Int32, Float32](Int32(42))
        ref val = u.unsafe_get[Int32]()
        print(val)  # => 42
        ```
        """
        return self._get_ptr[T]()[]

    def write_to(self, mut writer: Some[Writer]):
        """Writes a representation of the union to the writer.

        Since `UnsafeUnion` is untagged and doesn't track the stored type,
        this writes the union's type signature and size/alignment info rather
        than the stored value.

        Args:
            writer: The object to write to.

        Example:

        ```mojo
        from std.ffi import UnsafeUnion

        var u = UnsafeUnion[Int32, Float32](Int32(42))
        print(u)  # => UnsafeUnion[Int32, Float32](size=4, align=4)
        ```
        """
        FormatStruct(writer, "UnsafeUnion").params(
            TypeNames[*Self.Ts]()
        ).fields(
            Named("size", size_of[Self._union_type]()),
            Named("align", align_of[Self._union_type]()),
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the repr representation of the union to the writer.

        Args:
            writer: The object to write to.
        """
        self.write_to(writer)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def unsafe_take[T: Movable](mut self) -> T:
        """Move the stored value out of the union.

        This takes ownership of the stored value, leaving the union in an
        uninitialized state. The caller is responsible for ensuring that
        `T` matches the type that was actually stored.

        Parameters:
            T: The type to take out. Must be one of the union's element types
                and must be `Movable`.

        Returns:
            The stored value, moved out of the union.

        Safety:
            Taking as the wrong type is undefined behavior. After calling this,
            the union is uninitialized and you must call `unsafe_set` before
            reading again.

        Example:

        ```mojo
        from std.ffi import UnsafeUnion

        var u = UnsafeUnion[Int32, Float32](Int32(42))
        var val = u.unsafe_take[Int32]()  # val = 42, u is now uninitialized
        ```
        """
        comptime assert Self._is_element[
            T
        ](), "type is not a union element type"
        return self._get_ptr[T]().unsafe_take_pointee()

    @always_inline("nodebug")
    def unsafe_set[T: Movable](mut self, var value: T):
        """Set the union to hold the given value.

        This overwrites whatever was previously stored. Since all union element
        types must be trivial, no destructor needs to be called on the old
        value.

        Parameters:
            T: The type of the value. Must be one of the union's element types.

        Args:
            value: The value to store.

        Example:

        ```mojo
        from std.ffi import UnsafeUnion

        var u = UnsafeUnion[Int32, Float32](Int32(0))
        u.unsafe_set(Float32(3.14))
        ```
        """
        comptime assert Self._is_element[
            T
        ](), "type is not a union element type"
        self._get_ptr[T]().unsafe_write(value^)

    @always_inline("nodebug")
    def unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //, T: AnyType
    ](ref[origin, address_space] self) -> Pointer[
        T, origin, address_space=address_space
    ]:
        """Get a pointer to the storage interpreted as type T.

        This allows direct manipulation of the union's storage. Useful for
        C FFI where you need to pass a pointer to union members.

        Parameters:
            T: The type to interpret the storage as. Must be one of the
                union's element types.

        Returns:
            A pointer to the storage as type T.

        Safety:
            Interpreting as the wrong type is undefined behavior.

        Example:

        ```mojo
        from std.ffi import UnsafeUnion

        var u = UnsafeUnion[Int32, Float32](Int32(0))
        var ptr = u.unsafe_ptr[Int32]()
        ptr[] = 42
        ```
        """
        return self._get_ptr[T]()

    # ===-------------------------------------------------------------------===#
    # Static methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def _is_element[T: AnyType]() -> Bool:
        """Check if T is one of the union's element types.

        Returns True if found, False otherwise.
        """
        return Self.Ts.contains[T]()
