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
"""Implements `Pointer`, Mojo's primary pointer type for indirect memory
access.

`Pointer` is safe when constructed from and used to access an existing
value. It also provides `unsafe_`-prefixed methods for working with
dynamically-allocated or uninitialized memory; the caller is responsible
for tracking initialization state and memory ownership.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import Pointer
```
"""

from std.sys import (
    align_of,
    bit_width_of,
    is_apple_gpu,
    is_gpu,
    is_nvidia_gpu,
    size_of,
)
from std.sys.intrinsics import (
    gather,
    scatter,
    strided_load,
    strided_store,
)

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.format_int import _write_int
from std.builtin.simd import _simd_construction_checks
from std.format._utils import FormatStruct, Named, TypeNames
from std.reflection import reflect
from std.traits import IsTriviallyDeinitable, IsTriviallyMovable
from std.memory.address_space import AddressSpace
from std.memory import unsafe_memcpy
from std.memory.memory import _free
from std.memory import MaybeUninit
from std.memory._poison import _check_not_poison
from std._plugin import CurrentPlugin
from std.python import PythonObject
from std.utils._nicheable import (
    UnsafeSingleNicheable,
    UnsafeCustomNicheStorage,
    NicheStorageTraits,
)


@always_inline
def _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


# ===----------------------------------------------------------------------=== #
# Nicheable utilities
# ===----------------------------------------------------------------------=== #


struct _Null[type: AnyType = NoneType, address_space: AddressSpace = .GENERIC](
    Defaultable, Intable, TrivialRegisterPassable
):
    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    var address: Self._mlir_type

    @always_inline("builtin")
    def __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @always_inline("nodebug")
    def __int__(self) -> Int:
        return Int(mlir_value=__mlir_op.`pop.pointer_to_index`(self.address))


struct _PointerNicheStorage[
    type: AnyType,
    address_space: AddressSpace,
](NicheStorageTraits):
    """Custom niche backing for `Pointer` that lowers directly to
    `kgen.pointer` instead of `pop.array<1, kgen.pointer>`."""

    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.type,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    var address: Self._mlir_type

    @always_inline
    def __init__(out self):
        self.address = _Null[Self.type, Self.address_space]().address

    @always_inline
    def as_uninit[
        U: AnyType
    ](ref self) -> Pointer[MaybeUninit[U], origin_of(self)]:
        return (
            Pointer(to=self.address)
            .unsafe_bitcast[MaybeUninit[U]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


# ===-----------------------------------------------------------------------===#
# Pointer aliases
# ===-----------------------------------------------------------------------===#


@stable(since="1.0")
comptime MutPointer[
    T: AnyType,
    origin: MutOrigin,
    *,
    address_space: AddressSpace = .GENERIC,
] = Pointer[T, origin, address_space=address_space]
"""A mutable pointer.

Parameters:
    T: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""


@stable(since="1.0")
comptime ImmPointer[
    T: AnyType,
    origin: ImmOrigin,
    *,
    address_space: AddressSpace = .GENERIC,
] = Pointer[T, origin, address_space=address_space]
"""An immutable pointer.

Parameters:
    T: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime OptionalPointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
] = Optional[Pointer[T, origin, address_space=address_space]]
"""An optional (nullable) `Pointer`.

Parameters:
    mut: The mutability of the pointer.
    T: The type of the pointee.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime OpaquePointer[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
] = Pointer[NoneType, origin, address_space=address_space]
"""An opaque pointer, equivalent to the C `(const) void*` type.

Parameters:
    mut: Whether the pointer is mutable.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime MutOpaquePointer[
    origin: MutOrigin,
    *,
    address_space: AddressSpace = .GENERIC,
] = OpaquePointer[origin, address_space=address_space]
"""A mutable opaque pointer, equivalent to the C `void*` type.

Parameters:
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime ImmOpaquePointer[
    origin: ImmOrigin,
    *,
    address_space: AddressSpace = .GENERIC,
] = OpaquePointer[origin, address_space=address_space]
"""An immutable opaque pointer, equivalent to the C `const void*` type.

Parameters:
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""


# ===-----------------------------------------------------------------------===#
# Pointer
# ===-----------------------------------------------------------------------===#


@stable(since="1.0")
struct Pointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
](
    Comparable,
    DevicePassable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    UnsafeCustomNicheStorage,
    UnsafeSingleNicheable,
    Writable,
):
    """`Pointer` represents an indirect reference to one or more values
    of type `T` consecutively in memory, and can refer to uninitialized memory.

    Constructing a `Pointer` to an existing value (`Pointer(to=value)`) and
    dereferencing it (`ptr[]`) are safe operations: the pointer keeps the
    ownership linkage to the original value, so Mojo can track its lifetime.
    Working with dynamically-allocated or uninitialized memory is unsafe.
    Those operations are named with an `unsafe_` prefix, or take an
    `unsafe_`-prefixed keyword argument, and require the caller to track
    initialization state and memory ownership manually.

    Important things to know:

    - This pointer is non-nullable by design. To model a nullable pointer,
      use `Optional[Pointer[...]]`, which shares the same layout (the null
      address is the `None` niche) so it remains zero-overhead.
    - It does not own existing memory. `alloc()` returns an `Allocation`
      that owns heap-allocated memory; get a `Pointer` to it with
      `.unsafe_ptr()`, and release the memory by passing the `Allocation`
      to `dealloc()`.
    - For simple read/write access, use `ptr.unsafe_offset(i)[]` or
      `ptr[unsafe_offset=i]` where `i` is the offset size.
    - For SIMD operations on numeric data, use `Pointer[Scalar[.xxx]]`
      with `unsafe_load[dtype=DType.xxx]()` and
      `unsafe_store[dtype=DType.xxx]()`.

    Key APIs:

    - `unsafe_offset(i)`: Pointer arithmetic. Returns a new pointer shifted by
      `i` elements. No bounds checking.
    - `[]` or `[unsafe_offset=i]`: Dereference to a reference of the pointee
      (or at offset `i`). Only valid if the memory at that location is
      initialized.
    - `unsafe_load()`: Loads `width` elements starting at `offset` (default 0)
      as `SIMD[dtype, width]` from `Pointer[Scalar[dtype]]`. Pass
      `alignment` when data is not naturally aligned.
    - `unsafe_store()`: Stores `val: SIMD[dtype, width]` at `offset` into
      `Pointer[Scalar[dtype]]`. Requires a mutable pointer.
    - `unsafe_deinit_pointee()` / `unsafe_take_pointee()`:
      Explicitly end the lifetime of the current pointee, or move it out,
      taking ownership.
    - `unsafe_write()` / `unsafe_write_move_from()`:
      Initialize a pointee that is currently uninitialized, by moving an
      existing value into it (pass the argument as `copy=` to copy instead),
      or by moving from another pointee.
      Use these to manage lifecycles when working with uninitialized memory.

    For more information see [Using
    pointers](/docs/manual/pointers/using-pointers) in the Mojo Manual. For a
    comparison with other pointer types, see [Intro to
    pointers](/docs/manual/pointers/).

    Examples:

    Element-wise store and load (width = 1):

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    var allocation = alloc(Layout[Float32](count=4))
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        ptr.unsafe_store(i, Float32(i))
    var v = ptr.unsafe_load(2)
    print(v[0])  # => 2.0
    dealloc(allocation^)
    ```

    Vectorized store and load (width = 4):

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    var allocation = alloc(Layout[Int32](count=8))
    var ptr = allocation.unsafe_ptr()
    var vec = SIMD[.int32, 4](1, 2, 3, 4)
    ptr.unsafe_store(0, vec)
    var out = ptr.unsafe_load[width=4](0)
    print(out)  # => [1, 2, 3, 4]
    dealloc(allocation^)
    ```

    Pointer arithmetic and dereference:

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout

    var allocation = alloc(Layout[Int32](count=3))
    var ptr = allocation.unsafe_ptr()
    ptr.unsafe_offset(0)[] = 10  # offset by 0, then dereference to write
    ptr.unsafe_offset(1)[] = 20  # offset by 1, then dereference to write
    ptr[unsafe_offset=2] = 30  # equivalent offset/dereference via brackets
    var second = ptr[unsafe_offset=1]  # reads the element at index 1
    print(second, ptr[unsafe_offset=2])  # => 20 30
    dealloc(allocation^)
    ```

    Point to a value on the stack:

    ```mojo
    var foo: Int = 123
    var ptr = Pointer(to=foo)
    print(ptr[])  # => 123
    # Don't call `unsafe_free()` because the value was not heap-allocated
    # Mojo will destroy it when the `foo` lifetime ends
    ```

    Model a nullable pointer with `Optional`:

    `Pointer` is non-nullable by design, so nullability must be modeled
    explicitly with `Optional[Pointer[T, origin]]`. This keeps the same
    layout as `Optional` stores the null address as its `None` niche, so there
    is no overhead compared to a raw pointer.

    ```mojo
    from std.memory.alloc import alloc, dealloc, Layout, ThinAllocation
    from std.random import random_float64

    comptime layout = Layout[Int].single()

    # A field that may or may not point to a heap-allocated Int.
    var maybe_ptr: Optional[Pointer[Int, MutUntrackedOrigin]] = None

    # Maybe populate it later. `Optional` stores a raw pointer, so leak
    # the allocation and later pair it back with its layout to free it.
    if random_float64() > 0.5:
        maybe_ptr = alloc(layout).unsafe_leak()

    # Check for absence, then unwrap to use the pointer.
    if maybe_ptr:
        var ptr = maybe_ptr.value()
        ptr.write(42)
        print(ptr[])  # => 42
        dealloc(
            ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(
                layout
            )
        )
    ```

    If you instead need a non-null placeholder for a field that will be
    populated on demand (for example, a buffer that is allocated lazily),
    use `Pointer.unsafe_dangling()`. Note that `unsafe_dangling()` is
    not a null sentinel — it returns an aligned but dangling address, so
    types that lazily allocate must track initialization separately.

    Parameters:
        mut: Whether the origin is mutable.
        T: The type the pointer points to.
        origin: The origin of the memory being addressed.
        address_space: The address space associated with the `Pointer` allocated memory.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _mlir_lit_ref = __mlir_type[
        `!lit.ref<`,
        Self.T,
        `, `,
        Self.origin._mlir_origin,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]

    comptime _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        Self.T,
        `, `,
        Self.address_space._value._mlir_value,
        `>`,
    ]
    """The underlying pointer type."""

    comptime _with_origin[
        with_mut: Bool, //, with_origin: Origin[mut=with_mut]
    ] = Pointer[
        mut=with_mut, Self.T, with_origin, address_space=Self.address_space
    ]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _mlir_value: Self._mlir_type
    """The underlying pointer."""

    @always_inline("builtin")
    def _get_kgen_pointer(self) -> Self._mlir_type:
        """Returns the raw `kgen.pointer` MLIR value backing this pointer."""
        return self._mlir_value

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @always_inline("builtin")
    def __init__(out self, *, _mlir_value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            _mlir_value: The MLIR value of the pointer to construct with.
        """
        self._mlir_value = _mlir_value

    @doc_hidden
    @always_inline("nodebug")
    def __init__(out self, *, _mlir_value: Self._mlir_lit_ref):
        self = Self(_mlir_value=__mlir_op.`lit.ref.to_pointer`(_mlir_value))

    @always_inline
    def __init__(out self, *, unsafe_from_address: Int):
        """Create a pointer from a raw address.

        Args:
            unsafe_from_address: The raw address to create a pointer from.

        Safety:
            Creating a pointer from a raw address is inherently unsafe as the
            caller must ensure the address is valid before writing to it, and
            that the memory is initialized before reading from it. The caller
            must also ensure the pointer's origin and mutability is valid for
            the address, failure to do may result in undefined behavior.
        """
        # Some address spaces have pointers narrower than `Int` (for example,
        # AMDGPU `SHARED` is 32-bit); reject addresses that don't fit. The
        # guard also keeps `1 << bit_width_of[Self]()` from overflowing when
        # the pointer is as wide as `Int`.
        comptime if bit_width_of[Self]() < bit_width_of[Int]():
            debug_assert(
                UInt(unsafe_from_address)
                <= (UInt(1) << UInt(bit_width_of[Self]())) - 1,
                "address ",
                unsafe_from_address,
                " does not fit in this pointer's address space",
            )
        self = Pointer(to=unsafe_from_address).unsafe_bitcast[type_of(self)]()[]

    @always_inline
    @doc_hidden
    def __init__(out self, *, unsafe_from_address: IntLiteral):
        """Create a pointer from a raw address.

        This checks at compile time if the address is invalid and emits a compilation error.
        """
        comptime assert type_of(unsafe_from_address)() != 0, (
            "Pointer is non-nullable. To construct a null pointer, use"
            " Optional[Pointer] to model nullability."
        )
        comptime assert (
            type_of(unsafe_from_address)() > 0
        ), "Pointer's address cannot be negative."
        self = Self(unsafe_from_address=Int(unsafe_from_address))

    @always_inline("nodebug")
    def __init__(
        out self,
        *,
        ref[Self.origin, Self.address_space._value._mlir_value] to: Self.T,
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(
            _mlir_value=__mlir_op.`lit.ref.to_pointer`(
                __get_mvalue_as_litref(to)
            )
        )

    @always_inline("builtin")
    @implicit
    def __init__(
        other: Pointer,
        out self: Pointer[
            other.T, ImmOrigin(other.origin), address_space=other.address_space
        ],
    ):
        """Implicitly casts a mutable pointer to immutable.

        Args:
            other: The mutable pointer to cast from.
        """
        self._mlir_value = __mlir_op.`pop.pointer.bitcast`[
            _type=type_of(self)._mlir_type
        ](other._mlir_value)

    def __init__[
        U: Deinitable, //
    ](
        out self: Pointer[U, Self.origin],
        *,
        ref[Self.origin] unchecked_downcast_value: PythonObject,
    ):
        """Downcast a `PythonObject` known to contain a Mojo object to a pointer.

        This operation is only valid if the provided Python object contains
        an initialized Mojo object of matching type.

        Parameters:
            U: Pointee type that can be destroyed implicitly (without
              deinitializer arguments).

        Args:
            unchecked_downcast_value: The Python object to downcast from.
        """
        self = unchecked_downcast_value.unchecked_downcast_value_ptr[U]()

    # ===------------------------------------------------------------------===#
    # UnsafeNicheable
    # ===------------------------------------------------------------------===#

    @doc_hidden
    comptime NicheStorage: NicheStorageTraits = _PointerNicheStorage[
        Self.T, Self.address_space
    ]

    @staticmethod
    @always_inline
    @doc_hidden
    def write_niche(memory: MutPointer[MaybeUninit[Self], _]):
        memory.unsafe_bitcast[_Null[Self.T, Self.address_space]]().unsafe_write(
            {}
        )

    @staticmethod
    @always_inline
    @doc_hidden
    def isa_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> Bool:
        comptime NullType = _Null[Self.T, Self.address_space]
        comptime null_address = Int(NullType())
        return Int(memory.unsafe_bitcast[NullType]()[]) == null_address

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __getitem__(self) -> ref[Self.origin, Self.address_space] Self.T:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type=Self._mlir_lit_ref](
                self._mlir_value
            )
        )

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __getitem__[
        I: Indexer
    ](self, *, unsafe_offset: I) -> ref[Self.origin, Self.address_space] Self.T:
        """Return a reference to the underlying data, offset by the given index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            unsafe_offset: The offset index.

        Returns:
            An offset reference.

        Safety:

        - The pointer does not track the bounds of the memory it points
          into, so this does not check whether `unsafe_offset` (in elements
          of `T`) keeps the result within those bounds. `self` offset by
          `unsafe_offset` must point to a valid, initialized element of
          `T`; an out-of-bounds `unsafe_offset` is undefined behavior.
        """
        return self.unsafe_offset(unsafe_offset)[]

    @__unsafe_nested_origins_read_only
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(
        "positional `__getitem__` is deprecated, use `unsafe_offset=` instead"
    )
    def __getitem__[
        I: Indexer, //
    ](self, offset: I) -> ref[Self.origin, Self.address_space] Self.T:
        """Return a reference to the underlying data, offset by the given index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset reference.
        """
        return self[unsafe_offset=offset]

    @always_inline
    def _get_ref_with_unsafe_interior_origin[
        name: StringLiteral,
        base_origin: Origin,
    ](
        self,
    ) -> ref[
        base_origin._get_owned_interior[name], Self.address_space
    ] Self.T:
        """Returns a reference to the pointee with an interior origin.

        The returned reference uses an interior sub-origin derived from
        `base_origin` instead of this pointer's origin. Collections use this to
        vend element references whose lifetimes are tracked against the owning
        value.

        Parameters:
            name: A compile-time string that identifies the interior object.
            base_origin: The origin of the value that owns the interior storage.
        """
        comptime res_origin = base_origin._get_owned_interior[name]
        comptime ref_type = __mlir_type[
            `!lit.ref<`,
            Self.T,
            `, `,
            res_origin._mlir_origin,
            `, `,
            Self.address_space._value._mlir_value,
            `>`,
        ]
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type=ref_type](self._mlir_value)
        )

    @always_inline
    def _get_ref_with_unsafe_interior_origin[
        name: StringLiteral,
    ](self, ref base: Some[AnyType]) -> ref[
        origin_of(base)._get_owned_interior[name], Self.address_space
    ] Self.T:
        """Returns a reference to the pointee with an interior origin.

        The returned reference uses an interior sub-origin derived from
        `base` instead of this pointer's origin. Collections use this to vend
        element references whose lifetimes are tracked against the owning
        value.

        Parameters:
            name: A compile-time string that identifies the interior object.

        Args:
            base: The value whose origin the interior reference is derived
                from.

        Returns:
            A reference to this pointer's pointee with the interior origin.
        """
        comptime res_origin = origin_of(base)._get_owned_interior[name]
        comptime ref_type = __mlir_type[
            `!lit.ref<`,
            Self.T,
            `, `,
            res_origin._mlir_origin,
            `, `,
            Self.address_space._value._mlir_value,
            `>`,
        ]
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type=ref_type](self._mlir_value)
        )

    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_offset)
    def __add__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self.unsafe_offset(offset)

    @doc_hidden
    @always_inline
    @deprecated(use=unsafe_offset)
    def __sub__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self.unsafe_offset(-1 * index(offset))

    @doc_hidden
    @always_inline
    @deprecated(use=unsafe_offset)
    def __iadd__[I: Indexer, //](mut self, offset: I):
        """Add an offset to this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self.unsafe_offset(offset)

    @doc_hidden
    @always_inline
    @deprecated(use=unsafe_offset)
    def __isub__[I: Indexer, //](mut self, offset: I):
        """Subtract an offset from this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self.unsafe_offset(-1 * index(offset))

    @doc_hidden
    @__unsafe_nested_origins_read_only
    @always_inline
    def __sub__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Int:
        """Returns the signed distance from `rhs` to `self` in elements of
        `T` (not bytes), such that `rhs.unsafe_offset(result)` produces a
        pointer equal to `self`.

        This is the operator form of `offset_from()`. Its documentation is
        duplicated here because LSP hover shows this operator's own
        docstring rather than the method's; keep the two in sync.

        Constraints:
            The pointee type `T` must not be zero-sized.

        Args:
            rhs: The pointer to measure the distance from.

        Returns:
            The signed element distance from `rhs` to `self`.

        Safety:

        - Both pointers must point into the same allocation (or one past
          its end); the distance between pointers into unrelated
          allocations is not meaningful.
        - The byte distance between the two pointers must be an exact
          multiple of `size_of[T]()`. Violating this is undefined
          behavior.

        Examples:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var allocation = alloc(Layout[Int32](count=4))
        var ptr = allocation.unsafe_ptr()
        var end = ptr.unsafe_offset(3)
        print(end - ptr)  # => 3
        print(ptr - end)  # => -3
        dealloc(allocation^)
        ```
        """
        return self.offset_from(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __eq__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __ne__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __lt__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __le__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __gt__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __ge__(
        self,
        rhs: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @__unsafe_nested_origins_read_only
    @always_inline("nodebug")
    def __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @always_inline("builtin")
    def __merge_with__[
        other_type: type_of(
            Pointer[Self.T, origin=_, address_space=Self.address_space]
        ),
    ](self) -> Pointer[
        T=Self.T,
        origin=origin_of(Self.origin, other_type.origin),
        address_space=Self.address_space,
    ]:
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return {
            _mlir_value = self._mlir_value
        }  # allow kgen.pointer to convert.

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @unavailable(
        "Pointer is non-null by design, so Bool(ptr) is not"
        " meaningful. To model a null pointer, use"
        " `Optional[Pointer[...]]` and check with `Bool(opt_ptr)` / `!="
        " None`."
    )
    def __bool__(self) -> Bool:
        ...

    @always_inline
    def __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return Int(
            mlir_value=__mlir_op.`pop.pointer_to_index`(self._mlir_value)
        )

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """
        _write_int[radix=16](writer, Int(Int(self)), prefix="0x")

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the Pointer.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "Pointer").params(
            Named("mut", Self.mut),
            TypeNames[Self.T](),
            Named("address_space", Self.address_space),
        ).fields(self)

    # ===-------------------------------------------------------------------===#
    # DevicePassable
    # ===-------------------------------------------------------------------===#

    # Implementation of `DevicePassable`
    comptime device_type: AnyType = Self
    """DeviceBuffer dtypes are remapped to Pointer when passed to accelerator devices."""

    @staticmethod
    def _is_convertible_to_device_type[U: AnyType]() -> Bool:
        comptime if Self.mut:
            return TypeList.of[
                Self,
                Self._OriginCastType[MutAnyOrigin],
                Self._OriginCastType[MutUntrackedOrigin],
                Self._OriginCastType[ImmutAnyOrigin],
                Self._OriginCastType[ImmUntrackedOrigin],
            ]().contains[U]()
        else:
            return TypeList.of[
                Self,
                Self._OriginCastType[ImmutAnyOrigin],
                Self._OriginCastType[ImmUntrackedOrigin],
            ]().contains[U]()

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device dtype mapping from DeviceBuffer to the device's Pointer."""
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type name, for use in error messages when handing arguments
        to kernels.
        TODO: This will go away soon, when we get better error messages for
        kernel calls.

        Returns:
            This name of the type.
        """
        return String(
            "Pointer[",
            reflect[Self.T].name(),
            ", mut=",
            Self.mut,
            ", address_space=",
            Self.address_space,
            "]",
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def unsafe_offset[I: Indexer](self, offset: I, /) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.

        Safety:

        - The pointer does not track the bounds of the memory it points
          into, so this does not check whether `offset` (in elements of
          `T`) keeps the result within those bounds. Computing an
          out-of-bounds pointer is not itself unsafe, but dereferencing
          (loading from, storing to, or indexing) the result is undefined
          behavior unless it is back within bounds.
        """
        return {
            _mlir_value = __mlir_op.`pop.offset`(
                self._mlir_value, index(offset).__mlir_index__()
            )
        }

    # NOTE: `__sub__` duplicates this docstring so LSP hover on the `-`
    # operator shows the same documentation. Sync any docstring change here
    # over to `__sub__`.
    @__unsafe_nested_origins_read_only
    @always_inline
    def offset_from(
        self,
        other: Pointer[Self.T, _, address_space=Self.address_space],
    ) -> Int:
        """Returns the signed distance from `other` to `self` in elements of
        `T` (not bytes), such that `other.unsafe_offset(result)` produces a
        pointer equal to `self`.

        Constraints:
            The pointee type `T` must not be zero-sized.

        Args:
            other: The pointer to measure the distance from.

        Returns:
            The signed element distance from `other` to `self`.

        Safety:

        - Both pointers must point into the same allocation (or one past
          its end); the distance between pointers into unrelated
          allocations is not meaningful.
        - The byte distance between the two pointers must be an exact
          multiple of `size_of[T]()`. Violating this is undefined
          behavior.

        Examples:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var allocation = alloc(Layout[Int32](count=4))
        var ptr = allocation.unsafe_ptr()
        var end = ptr.unsafe_offset(3)
        print(end.offset_from(ptr))  # => 3
        print(ptr - end)  # => -3
        dealloc(allocation^)
        ```
        """
        comptime assert (
            size_of[Self.T]() > 0
        ), "offset_from() requires a non-zero-sized pointee type"
        var byte_diff = Int(self) - Int(other)
        debug_assert(
            byte_diff % size_of[Self.T]() == 0,
            "pointer difference is not a multiple of the element size",
        )
        return Int(
            mlir_value=__mlir_op.`pop.div`[isExact=__mlir_attr.`unit`](
                byte_diff._mlir_value, size_of[Self.T]()._mlir_value
            )
        )

    @always_inline
    @staticmethod
    def unsafe_dangling() -> Self:
        """Creates a new `Pointer` that is dangling, but well-aligned.

        This is useful for initializing types which lazily allocate.

        Note that the address of the returned pointer may potentially be that
        of a valid pointer, which means this must not be used as a "not yet
        initialized" sentinel value. Types that lazily allocate must track
        initialization by some other means.

        Returns:
            A dangling but well-aligned `Pointer`.

        Safety:

        - The returned pointer does not point to any valid storage. Reading
          from or writing through it is undefined behavior until it has been
          reassigned to point at real, live memory.

        Example:

        ```mojo
        var ptr = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()
        # Important: don't try to access the value of `ptr` without
        # initializing it first! The pointer is not null but isn't valid either!
        ```
        """
        comptime alignment = align_of[Self.T]()
        comptime if CurrentPlugin.unsafe_dangling_fn:
            comptime address = CurrentPlugin.unsafe_dangling_fn.value()[
                alignment
            ]()
            comptime assert (
                address != 0
            ), "Pointer cannot be constructed with address 0"
            return Self(unsafe_from_address=address)
        return Self(unsafe_from_address=alignment)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def swap_pointees[
        U: Movable
    ](self: MutPointer[U, _], other: MutPointer[U, _],):
        """Swap the values at the pointers.

        This function assumes that `self` and `other` _may_ overlap in memory.
        If that is not the case, or when references are available, you should
        use `builtin.swap` instead.

        Parameters:
            U: The type the pointers point to, which must be `Movable`.

        Args:
            other: The other pointer to swap with.

        Safety:
            - `self` and `other` must both point to valid, initialized instances
              of `T`.
        """

        comptime if IsTriviallyMovable[U]:
            # If `moveinit` is trivial, we can avoid the branch introduced from
            # checking if the pointers are equal by using temporary stack
            # values.
            #
            # Since `lhs` may overlap with `rhs` we need two temporary stack
            # values since we cannot call `unsafe_memcpy` with the potentially
            # overlapping pointers.
            #
            # Even if they are not overlapping, this also produces better llvm
            # code with only 2 loads and 2 stores. Whereas with only 1 temporary
            # and an unsafe_memcpy between the pointers it produces 3 load and
            # 3 stores.

            var self_tmp = MaybeUninit[U]()
            var other_tmp = MaybeUninit[U]()
            unsafe_memcpy(dest=self_tmp.unsafe_ptr(), src=self, count=1)
            unsafe_memcpy(dest=other_tmp.unsafe_ptr(), src=other, count=1)

            unsafe_memcpy(dest=self, src=other_tmp.unsafe_ptr(), count=1)
            unsafe_memcpy(dest=other, src=self_tmp.unsafe_ptr(), count=1)

            # Safety:
            # The memcpy moved the values from the temporary storage into
            # self & other, so the values are uninitialized.
            self_tmp^.unsafe_forget()
            other_tmp^.unsafe_forget()
        else:
            # If `moveinit` is NOT trivial, we need to check if the pointers are
            # the same to avoid undefined behavior when moving from rhs to lhs.
            if self == other:
                return
            var tmp = self.unsafe_take_pointee()
            self.unsafe_write_move_from(other)
            other.unsafe_write(tmp^)

    @always_inline("nodebug")
    def unsafe_as_noalias(self) -> Self:
        """Cast the pointer to a new pointer that is known not to locally alias
        any other pointer. In other words, the pointer transitively does not
        comptime any other memory value declared in the local function context.

        This information is relayed to the optimizer. If the pointer does
        locally alias another memory value, the behaviour is undefined.

        Returns:
            A noalias pointer.

        Safety:

        - The pointer must not locally alias any other pointer reachable in
          the current function context. The optimizer trusts this assertion
          without checking it, so reads and writes through an aliasing
          pointer become undefined behavior.
        """
        return {
            _mlir_value = __mlir_op.`pop.noalias_pointer_cast`(self._mlir_value)
        }

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: Pointer[Scalar[dtype], ...]) -> SIMD[dtype, width]:
        """Loads `width` elements from the value the pointer points to.

        Use `alignment` to specify minimal known alignment in bytes; pass a
        smaller value (such as 1) if loading from packed/unaligned memory. The
        `volatile`/`invariant` flags control reordering and common-subexpression
        elimination semantics for special cases.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var allocation = alloc(Layout[Int32](count=8))
        var p = allocation.unsafe_ptr()
        p.unsafe_store(0, SIMD[.int32, 4](1, 2, 3, 4))
        var v = p.unsafe_load[width=4]()
        print(v)  # => [1, 2, 3, 4]
        dealloc(allocation^)
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of the SIMD vector.
            width: The number of elements to load.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            invariant: Whether the load is from invariant memory.
            non_temporal: Whether the load has no temporal locality (streaming).

        Returns:
            The loaded SIMD vector.

        Safety:

        - The pointer does not track how many elements it points to, so
          `self` must point to `width` contiguous, initialized elements of
          `dtype`. This is not checked — passing a `width` larger than the
          number of valid elements reads past the end of them, which is
          undefined behavior.
        - The address must satisfy `alignment` bytes of alignment.
        """
        _simd_construction_checks[dtype, width]()
        comptime assert (
            alignment > 0
        ), "alignment must be a positive integer value"
        comptime assert (
            not volatile or volatile ^ invariant
        ), "both volatile and invariant cannot be set at the same time"

        comptime if is_nvidia_gpu() and size_of[
            dtype
        ]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[dtype, width]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(width):
                v[i] = __mlir_op.`pop.load`[
                    alignment=alignment.__mlir_index__(),
                    isVolatile=volatile.__mlir_i1__(),
                    isInvariant=invariant.__mlir_i1__(),
                    isNonTemporal=non_temporal.__mlir_i1__(),
                ](self.unsafe_offset(i)._mlir_value)
            comptime if dtype.is_floating_point():
                _check_not_poison[dtype, width](v)
            return v
        elif dtype == .bool and width > 1:
            # Bool (i1) is sub-byte, so a vector load of SIMD[bool, N]
            # packs bits. Load as uint8 and convert to bool so each
            # element occupies its own byte boundary.
            return rebind[SIMD[dtype, width]](
                self.unsafe_bitcast[UInt8]()
                .unsafe_load[
                    width=width,
                    alignment=alignment,
                    volatile=volatile,
                    invariant=invariant,
                    non_temporal=non_temporal,
                ]()
                .cast[.bool]()
            )

        var address = self.unsafe_bitcast[SIMD[dtype, width]]()._mlir_value

        var result = __mlir_op.`pop.load`[
            alignment=alignment.__mlir_index__(),
            isVolatile=volatile.__mlir_i1__(),
            isInvariant=invariant.__mlir_i1__(),
            isNonTemporal=non_temporal.__mlir_i1__(),
        ](address)
        comptime if dtype.is_floating_point():
            _check_not_poison[dtype, width](result)
        return result

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_load)
    def load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: Pointer[Scalar[dtype], ...]) -> SIMD[dtype, width]:
        return self.unsafe_load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: Pointer[Scalar[dtype], ...], offset: Scalar) -> SIMD[dtype, width]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be an integer.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.
            non_temporal: Whether the load has no temporal locality (streaming).

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.

        Safety:

        - `self` offset by `offset` elements must point to `width`
          contiguous, initialized elements of `dtype`. This is not checked —
          an out-of-bounds `offset`, or a `width` larger than the number of
          valid elements, is undefined behavior.
        - The resulting address must satisfy `alignment` bytes of alignment.
        """
        comptime assert offset.dtype.is_integral(), "offset must be an integer"
        return self.unsafe_offset(offset).unsafe_load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_load)
    def load[
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: Pointer[Scalar[dtype], ...], offset: Scalar,) -> SIMD[dtype, width]:
        return self.unsafe_load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_load[
        I: Indexer,
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: Pointer[Scalar[dtype], ...], offset: I) -> SIMD[dtype, width]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.
            non_temporal: Whether the load has no temporal locality (streaming).

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.

        Safety:

        - `self` offset by `offset` elements must point to `width`
          contiguous, initialized elements of `dtype`. This is not checked —
          an out-of-bounds `offset`, or a `width` larger than the number of
          valid elements, is undefined behavior.
        - The resulting address must satisfy `alignment` bytes of alignment.
        """
        return self.unsafe_offset(offset).unsafe_load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ]()

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_load)
    def load[
        I: Indexer,
        dtype: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[Self.mut](),
        non_temporal: Bool = False,
    ](self: Pointer[Scalar[dtype], ...], offset: I,) -> SIMD[dtype, width]:
        return self.unsafe_load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
            non_temporal=non_temporal,
        ](offset)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_store[
        I: Indexer,
        dtype: DType,
        //,
        width: SIMDLength = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: MutPointer[Scalar[dtype], ...],
        offset: I,
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            offset: The offset to store to.
            val: The value to store.

        Safety:

        - `self` offset by `offset` elements must point to writable memory
          for `width` contiguous elements of `dtype`. This is not checked —
          an out-of-bounds `offset`, or a `width` larger than the number of
          valid elements, is undefined behavior.
        - The resulting address must satisfy `alignment` bytes of alignment.
        """
        self.unsafe_offset(offset).unsafe_store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_store[
        dtype: DType,
        offset_type: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: MutPointer[Scalar[dtype], ...],
        offset: Scalar[offset_type],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            offset_type: The data type of the offset value.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            offset: The offset to store to.
            val: The value to store.

        Safety:

        - `self` offset by `offset` elements must point to writable memory
          for `width` contiguous elements of `dtype`. This is not checked —
          an out-of-bounds `offset`, or a `width` larger than the number of
          valid elements, is undefined behavior.
        - The resulting address must satisfy `alignment` bytes of alignment.
        """
        comptime assert offset_type.is_integral(), "offset must be integer"
        self.unsafe_offset(offset)._store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_store[
        dtype: DType,
        //,
        width: SIMDLength = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](self: MutPointer[Scalar[dtype], ...], val: SIMD[dtype, width]):
        """Stores a single element value `val` at element offset 0.

        Specify `alignment` when writing to packed/unaligned memory. Requires a
        mutable pointer. For writing at an element offset, use the overloads
        that accept an index or scalar offset.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var allocation = alloc(Layout[Float32](count=4))
        var p = allocation.unsafe_ptr()
        var vec = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
        p.unsafe_store(vec)
        var out = p.unsafe_load[width=4]()
        print(out)  # => [1.0, 2.0, 3.0, 4.0]
        dealloc(allocation^)
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The number of elements to store.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            non_temporal: Whether the store has no temporal locality (streaming).

        Args:
            val: The SIMD value to store.

        Safety:

        - `self` must point to writable memory for `width` contiguous
          elements of `dtype`. This is not checked — passing a `width`
          larger than the number of valid elements writes past the end of
          them, which is undefined behavior.
        - The address must satisfy `alignment` bytes of alignment.
        """
        self._store[
            alignment=alignment, volatile=volatile, non_temporal=non_temporal
        ](val)

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_store)
    def store[
        I: Indexer,
        dtype: DType,
        //,
        width: SIMDLength = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: MutPointer[Scalar[dtype], ...],
        offset: I,
        val: SIMD[dtype, width],
    ):
        self.unsafe_store[
            width,
            alignment=alignment,
            volatile=volatile,
            non_temporal=non_temporal,
        ](offset, val)

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_store)
    def store[
        dtype: DType,
        offset_type: DType,
        //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](
        self: MutPointer[Scalar[dtype], ...],
        offset: Scalar[offset_type],
        val: SIMD[dtype, width],
    ):
        self.unsafe_store[
            width,
            alignment=alignment,
            volatile=volatile,
            non_temporal=non_temporal,
        ](offset, val)

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_store)
    def store[
        dtype: DType,
        //,
        width: SIMDLength = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](self: MutPointer[Scalar[dtype], ...], val: SIMD[dtype, width]):
        self.unsafe_store[
            width,
            alignment=alignment,
            volatile=volatile,
            non_temporal=non_temporal,
        ](val)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def _store[
        dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        non_temporal: Bool = False,
    ](self: MutPointer[Scalar[dtype], ...], val: SIMD[dtype, width]):
        comptime assert width > 0, "width must be a positive integer value"
        comptime assert (
            alignment > 0
        ), "alignment must be a positive integer value"

        comptime if dtype == .bool and width > 1:
            # Bool (i1) is sub-byte, so a vector store of SIMD[bool, N]
            # packs bits. Cast to uint8 and store so each element
            # occupies its own byte boundary.
            self.unsafe_bitcast[UInt8]()._store[
                alignment=alignment,
                volatile=volatile,
                non_temporal=non_temporal,
            ](val.cast[.uint8]())
        else:
            __mlir_op.`pop.store`[
                alignment=alignment.__mlir_index__(),
                isVolatile=volatile.__mlir_i1__(),
                isNonTemporal=non_temporal.__mlir_i1__(),
            ](val, self.unsafe_bitcast[SIMD[dtype, width]]()._mlir_value)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_strided_load[
        dtype: DType, S: Intable, //, width: Int
    ](self: Pointer[Scalar[dtype], ...], stride: S) -> SIMD[dtype, width]:
        """Performs a strided load of the SIMD vector.

        Parameters:
            dtype: DType of returned SIMD value.
            S: The Intable type of the stride.
            width: The SIMD width.

        Args:
            stride: The stride between loads.

        Returns:
            A vector which is stride loaded.

        Safety:

        - This reads `width` elements from `self`, each `stride` elements
          apart. Every element read must be an initialized `dtype` value.
          This is not checked, so an out-of-bounds `stride` or `width` is
          undefined behavior.
        """
        return strided_load(self, Int(stride), SIMD[.bool, width](fill=True))

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_strided_load)
    def strided_load[
        dtype: DType, S: Intable, //, width: Int
    ](self: Pointer[Scalar[dtype], ...], stride: S,) -> SIMD[dtype, width]:
        return self.unsafe_strided_load[width=width](stride)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_strided_store[
        dtype: DType,
        S: Intable,
        //,
        width: SIMDLength = 1,
    ](
        self: MutPointer[Scalar[dtype], ...],
        val: SIMD[dtype, width],
        stride: S,
    ):
        """Performs a strided store of the SIMD vector.

        Parameters:
            dtype: DType of `val`, the SIMD value to store.
            S: The Intable type of the stride.
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
            stride: The stride between stores.

        Safety:

        - This writes the `width` elements of `val` to `self`, each `stride`
          elements apart. Every location written must be writable memory
          for `dtype`. This is not checked, so an out-of-bounds `stride` or
          `width` is undefined behavior.
        """
        strided_store(val, self, Int(stride), SIMD[.bool, width](fill=True))

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_strided_store)
    def strided_store[
        dtype: DType,
        S: Intable,
        //,
        width: SIMDLength = 1,
    ](
        self: MutPointer[Scalar[dtype], ...],
        val: SIMD[dtype, width],
        stride: S,
    ):
        self.unsafe_strided_store[width=width](val, stride)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_gather[
        dtype: DType,
        //,
        *,
        width: SIMDLength = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: Pointer[Scalar[dtype], ...],
        offset: SIMD[_, width],
        mask: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        """Gathers a SIMD vector from offsets of the current pointer.

        This method loads from memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        or takes from the `default` SIMD vector, depending on the values of
        the `mask` SIMD vector.

        If a mask element is `True`, the respective result element is given
        by the current pointer and the `offset` SIMD vector; otherwise, the
        result element is taken from the `default` SIMD vector.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            dtype: DType of the return SIMD.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to gather from.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to load from memory or to take from the
                `default` SIMD vector.
            default: The SIMD vector providing default values to be taken
                where the `mask` SIMD vector is `False`.

        Returns:
            The SIMD vector containing the gathered values.

        Safety:

        - This reads from `self` offset by each active lane's `offset`
          element (the lanes where `mask` is `True`). Each of those
          locations must be an initialized `dtype` element. This is not
          checked, so an out-of-bounds `offset` on an active lane is
          undefined behavior. Lanes where `mask` is `False` are never read,
          so their `offset` can be anything.
        - The resulting addresses must satisfy `alignment` bytes of
          alignment.
        """
        comptime assert (
            offset.dtype.is_integral()
        ), "offset type must be an integral type"
        comptime assert (
            alignment.is_power_of_two()
        ), "alignment must be a power of two integer value"

        comptime if is_apple_gpu():
            # `Int(self)` would erase the address space; on Apple AIR the
            # resulting GENERIC load silently reads zero (MOCO-3762).
            var result = default
            comptime for i in range(width):
                if mask[i]:
                    result[i] = self.unsafe_load[alignment=alignment](
                        Int(offset[i])
                    )
            return result

        var base = offset.cast[.int]().fma(
            SIMD[.int, width](size_of[dtype]()),
            SIMD[.int, width](Int(self)),
        )
        return gather[alignment=alignment](base, mask, default)

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_gather)
    def gather[
        dtype: DType,
        //,
        *,
        width: SIMDLength = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: Pointer[Scalar[dtype], ...],
        offset: SIMD[_, width],
        mask: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        return self.unsafe_gather[alignment=alignment](offset, mask, default)

    @__allow_legacy_custom_self_type
    @always_inline("nodebug")
    def unsafe_scatter[
        dtype: DType,
        //,
        *,
        width: SIMDLength = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: MutPointer[Scalar[dtype], ...],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
    ):
        """Scatters a SIMD vector into offsets of the current pointer.

        This method stores at memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        depending on the values of the `mask` SIMD vector.

        If a mask element is `True`, the respective element in the `val` SIMD
        vector is stored at the memory address defined by the current pointer
        and the `offset` SIMD vector; otherwise, no action is taken for that
        element in `val`.

        If the same offset is targeted multiple times, the values are stored
        in the order they appear in the `val` SIMD vector, from the first to
        the last element.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            dtype: DType of `value`, the result SIMD buffer.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to scatter into.
            val: The SIMD vector containing the values to be scattered.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to store at memory or not.

        Safety:

        - This writes to `self` offset by each active lane's `offset`
          element (the lanes where `mask` is `True`). Each of those
          locations must be writable memory for a `dtype` element. This is
          not checked, so an out-of-bounds `offset` on an active lane is
          undefined behavior. Lanes where `mask` is `False` are never
          written, so their `offset` can be anything.
        - The resulting addresses must satisfy `alignment` bytes of
          alignment.
        """
        comptime assert (
            offset.dtype.is_integral()
        ), "offset type must be an integral type"
        comptime assert (
            alignment.is_power_of_two()
        ), "alignment must be a power of two integer value"

        comptime if is_apple_gpu():
            # See `gather` for the address-space rationale (MOCO-3762).
            comptime for i in range(width):
                if mask[i]:
                    self.unsafe_store[alignment=alignment](
                        Int(offset[i]), val[i]
                    )
            return

        var base = offset.cast[.int]().fma(
            SIMD[.int, width](size_of[dtype]()),
            SIMD[.int, width](Int(self)),
        )
        scatter[alignment=alignment](val, base, mask)

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline("nodebug")
    @deprecated(use=unsafe_scatter)
    def scatter[
        dtype: DType,
        //,
        *,
        width: SIMDLength = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: MutPointer[Scalar[dtype], ...],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[.bool, width] = SIMD[.bool, width](fill=True),
    ):
        self.unsafe_scatter[alignment=alignment](offset, val, mask)

    @__allow_legacy_custom_self_type
    @doc_hidden
    @always_inline
    @deprecated(use=unsafe_free)
    def free(self: MutPointer[Self.T, ...]):
        """Free the memory referenced by the pointer."""
        _free(self)

    @__allow_legacy_custom_self_type
    @always_inline
    def unsafe_free(self: MutPointer[Self.T, ...]):
        """Frees the memory referenced by the pointer."""
        _free(self)

    @always_inline("builtin")
    def unsafe_bitcast[
        U: AnyType
    ](self) -> Pointer[U, Self.origin, address_space=Self.address_space]:
        """Bitcasts a Pointer to a different type.

        Parameters:
            U: The target type.

        Returns:
            A new Pointer object with the specified type and the same address,
            as the original Pointer.

        Safety:

        - This does not check that `U` is compatible with the pointee type
          in size, alignment, or bit layout. Reading or writing through the
          returned pointer is undefined behavior unless the memory it
          points to actually holds (or is being written as) a valid `U`.
        """
        return {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=Pointer[
                    U, Self.origin, address_space=Self.address_space
                ]._mlir_type,
            ](self._mlir_value)
        }

    @doc_hidden
    @always_inline("builtin")
    @deprecated(use=unsafe_bitcast)
    def bitcast[
        U: AnyType
    ](self) -> Pointer[U, Self.origin, address_space=Self.address_space]:
        return self.unsafe_bitcast[U]()

    comptime _OriginCastType[
        target_mut: Bool, //, target_origin: Origin[mut=target_mut]
    ] = Pointer[Self.T, target_origin, address_space=Self.address_space]

    @always_inline("nodebug")
    @deprecated(
        "`mut_cast` is deprecated in favor of explicitly specifying a"
        " mutability on the pointer type (`ImmPointer` or `MutPointer`). If"
        " a mutability cast is truly needed (this should almost always be"
        " avoided), use `unsafe_mut_cast` instead."
    )
    def mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of a pointer.

        This is a safe way to change the mutability of a pointer with an
        unbounded mutability. This function will emit a compile time error if
        you try to cast an immutable pointer to mutable.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.
        """
        comptime assert (
            target_mut == False or target_mut == Self.mut
        ), "Cannot safely cast an immutable pointer to mutable"
        return self.unsafe_mut_cast[target_mut]()

    @always_inline("builtin")
    def unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[Self.origin.unsafe_mut_cast[target_mut]()]:
        """Changes the mutability of a pointer.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.

        If you are unconditionally casting the mutability to `False`, use
        `as_imm` instead.
        If you are casting to mutable or a parameterized mutability, prefer
        using the safe `mut_cast` method instead.

        Safety:
            Casting the mutability of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior. Consider restricting
            types to their proper mutability at the function signature level.
            For example, taking a `Pointer[T, mut=True, ...]` as an
            argument over an unbound `Pointer[T, ...]` is preferred.
        """
        return {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=Self._OriginCastType[
                    Self.origin.unsafe_mut_cast[target_mut]()
                ]._mlir_type,
            ](self._mlir_value)
        }

    @always_inline("builtin")
    def unsafe_origin_cast[
        target_origin: Origin[mut=Self.mut]
    ](self) -> Self._OriginCastType[target_origin]:
        """Changes the origin of a pointer.

        Parameters:
            target_origin: Origin of the destination pointer.

        Returns:
            A pointer with the same type, mutability and address space as the
            original pointer, but with the newly specified origin.

        If you are unconditionally casting the origin to an `UnsafeAnyOrigin`,
        use `as_unsafe_any_origin` instead.

        Safety:
            Casting the origin of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior or unexpected variable
            destruction. Considering parameterizing the origin at the function
            level to avoid unnecessary casts.
        """
        return {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=Self._OriginCastType[target_origin]._mlir_type,
            ](self._mlir_value)
        }

    @always_inline("builtin")
    def as_imm(
        self,
    ) -> Self._OriginCastType[ImmOrigin(Self.origin)]:
        """Changes the mutability of a pointer to immutable.

        Unlike `unsafe_mut_cast`, this function is always safe to use as casting
        from (im)mutable to immutable is always safe.

        Returns:
            A pointer with the mutability set to immutable.
        """
        return self.unsafe_mut_cast[False]()

    @always_inline("builtin")
    def as_unsafe_any_origin(
        self,
    ) -> Self._OriginCastType[UnsafeAnyOrigin[mut=Self.mut]]:
        """Casts the origin of a pointer to `UnsafeAnyOrigin`.

        Returns:
            A pointer with the origin set to `UnsafeAnyOrigin`.

        Safety:

        It is **always** preferred to maintain a concrete origin values instead of
        using `UnsafeAnyOrigin`. Casting to `UnsafeAnyOrigin` is an inherently unsafe
        operation that will silently extend unrelated lifetimes and turn off
        exclusivity checking.
        """
        return {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=Self._OriginCastType[
                    UnsafeAnyOrigin[mut=Self.mut]
                ]._mlir_type,
            ](self._mlir_value)
        }

    @always_inline("builtin")
    def unsafe_address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> Pointer[Self.T, Self.origin, address_space=target_address_space]:
        """Casts an Pointer to a different address space.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new Pointer object with the same type and the same address,
            as the original Pointer and the new address space.

        Safety:

        - This does not check that the pointer's address is actually valid
          within `target_address_space`. Dereferencing the returned pointer
          is undefined behavior unless the address is one the target
          address space can legally access (for example, casting a
          `GENERIC` host pointer to a GPU-only address space and then
          dereferencing it).
        """
        return {
            _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                _type=Pointer[
                    Self.T, Self.origin, address_space=target_address_space
                ]._mlir_type,
            ](self._mlir_value)
        }

    @doc_hidden
    @always_inline("builtin")
    @deprecated(use=unsafe_address_space_cast)
    def address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> Pointer[Self.T, Self.origin, address_space=target_address_space]:
        return self.unsafe_address_space_cast[target_address_space]()

    @always_inline
    def unsafe_deinit_pointee(
        self,
    ) where (
        Self.mut
        and Self.address_space == .GENERIC
        and conforms_to(Self.T, Deinitable)
    ):
        """Destroys the pointed-to value.

        This is equivalent to `_ = self.unsafe_take_pointee()` but doesn't
        require `Movable` and is more efficient because it doesn't invoke a
        move constructor.

        Safety:

        - This runs the pointee's deinitializer and leaves the pointee
          memory uninitialized. Subsequent reads of this pointer are
          invalid until a new valid value is written using an
          `unsafe_write()` method.
        - `self` must point to a valid, initialized instance of `T`.
          Calling this on a pointer to uninitialized memory is undefined
          behavior.
        """
        var this = self.unsafe_address_space_cast[.GENERIC]()
        _ = __get_address_as_owned_value(this._mlir_value)

    @always_inline
    def unsafe_deinit_pointee_with(
        self, deinit_func: Some[def(var Self.T)], /
    ) where Self.mut and Self.address_space == .GENERIC:
        """Destroys the pointed-to value using a user-provided deinitializer
        function.

        This can be used to destroy non-`Deinitable` values in-place
        without moving.

        Args:
            deinit_func: A function that takes ownership of the pointee value
                for the purpose of deinitializing it.

        Safety:

        - This runs `deinit_func` on the pointee and leaves the pointee
          memory uninitialized. Subsequent reads of this pointer are
          invalid until a new valid value is written using an
          `unsafe_write()` method.
        - `self` must point to a valid, initialized instance of
          `Self.T`. Calling this on a pointer to uninitialized memory
          is undefined behavior.
        """
        var this = self.unsafe_address_space_cast[.GENERIC]()
        deinit_func(__get_address_as_owned_value(this._mlir_value))

    @always_inline
    def unsafe_take_pointee(
        self,
    ) -> Self.T where (
        Self.mut
        and Self.address_space == .GENERIC
        and conforms_to(Self.T, Movable)
    ):
        """Move the value at the pointer out, leaving it uninitialized.

        This performs a _consuming_ move, ending the origin of the value stored
        in this pointer memory location. Subsequent reads of this pointer are
        not valid. If a new valid value is stored using `unsafe_write()`, then
        reading from this pointer becomes valid again.

        Returns:
            The value at the pointer.

        Safety:

        - `self` must point to a valid, initialized instance of `Self.T`.
          Calling this on a pointer to uninitialized memory is undefined
          behavior.
        - This moves the pointee out without running its destructor and
          leaves the pointee memory uninitialized. Subsequent reads of this
          pointer are invalid until a new valid value is written using an
          `unsafe_write()` method.
        """
        return __get_address_as_owned_value(
            self.unsafe_address_space_cast[.GENERIC]()._mlir_value
        )

    @doc_hidden
    @always_inline
    @deprecated(use=unsafe_take_pointee)
    def take_pointee(
        self,
    ) -> Self.T where (
        Self.mut
        and Self.address_space == .GENERIC
        and conforms_to(Self.T, Movable)
    ):
        return self.unsafe_take_pointee()

    @always_inline
    def unsafe_write(
        self,
        *,
        init_with: Some[def() -> Self.T],
    ) where Self.mut and Self.address_space == .GENERIC:
        """Initializes the pointee in place using the value returned by `f`.

        The value returned by `init_with` is constructed directly into the pointer's
        memory location rather than being constructed elsewhere and then
        moved, so `Self.T` does not need to be `Movable`.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        @fieldwise_init
        struct Pinned(Movable where False):
            var value: Int

        var allocation = alloc(Layout[Pinned].single())
        var ptr = allocation.unsafe_ptr()
        ptr.unsafe_write(init_with=lambda () -> Pinned: Pinned(7))
        print(ptr[].value)  # => 7
        ptr.unsafe_deinit_pointee()
        dealloc(allocation^)
        ```

        Args:
            init_with: A function that constructs and returns the value to emplace.

        Safety:

        - `self` must point to writable memory for `Self.T` that does not
          currently hold a valid, live value. Writing into memory that
          already holds one overwrites it without running its destructor,
          leaking any resources it owned.
        """
        __get_address_as_uninit_lvalue(
            self.unsafe_address_space_cast[.GENERIC]()._mlir_value
        ) = init_with()

    @__allow_legacy_custom_self_type
    @always_inline
    def write[
        U: Movable, //
    ](self: Pointer[U, _], var value: U, /) where (
        type_of(self).mut and IsTriviallyDeinitable[U]
    ):
        """Write `value` into the pointer location, moving from `value`.

        Unlike `unsafe_write()`, this is safe to call even when the pointee
        already holds a live value: `U` is constrained to be trivially
        deinitializable, so there's no destructor to skip and overwriting
        a previous value can't leak a resource.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var allocation = alloc(Layout[Int].single())
        var ptr = allocation.unsafe_ptr()
        ptr.write(41)
        ptr.write(42)  # OK: overwriting is safe, `Int` has no destructor.
        print(ptr[])  # => 42
        dealloc(allocation^)
        ```

        Parameters:
            U: The type the pointer points to, which must be `Movable` and
                trivially deinitializable.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self._mlir_value) = value^

    @always_inline
    def unsafe_write(
        self, var value: Self.T, /
    ) where (
        Self.mut
        and Self.address_space == .GENERIC
        and conforms_to(Self.T, Movable)
    ):
        """Write `value` into the pointer location, moving from `value`.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not
        deinitialized before writing `value`.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var allocation = alloc(Layout[String].single())
        var ptr = allocation.unsafe_ptr()
        ptr.unsafe_write("foo")
        print(ptr[])  # => foo
        ptr.unsafe_deinit_pointee()
        dealloc(allocation^)
        ```

        If `Self.T` is trivially deinitializable (for example, `Int`), prefer
        `write()` instead: it overwrites safely, since there's no
        destructor that a previous value could leak.

        Args:
            value: The value to emplace.

        Safety:

        - `self` must point to writable memory for `Self.T` that does not
          currently hold a valid, live value. Writing into memory that
          already holds one overwrites it without running its destructor,
          leaking any resources it owned.
        """
        __get_address_as_uninit_lvalue(
            self.unsafe_address_space_cast[.GENERIC]()._mlir_value
        ) = (value^)

    @always_inline
    def unsafe_write(
        self, *, copy: Self.T
    ) where (
        Self.mut
        and Self.address_space == .GENERIC
        and conforms_to(Self.T, Copyable)
    ):
        """Write a copy of `copy` into the pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not deinitialized
        before writing the copy.

        When compared to calling the positional `unsafe_write()` overload with
        a value you copied yourself, this avoids an extra move on the callee
        side.

        Args:
            copy: The value to copy.

        Safety:

        - `self` must point to writable memory for `Self.T` that does not
          currently hold a valid, live value. Writing into memory that
          already holds one overwrites it without running its destructor,
          leaking any resources it owned.
        """
        __get_address_as_uninit_lvalue(
            self.unsafe_address_space_cast[.GENERIC]()._mlir_value
        ) = copy.copy()

    @always_inline
    def unsafe_write_move_from(
        self, src: Pointer[Self.T, _]
    ) where (
        Self.mut
        and Self.address_space == .GENERIC
        and type_of(src).mut
        and conforms_to(Self.T, Movable)
    ):
        """Moves the value `src` points to into the memory location pointed to
        by `self`.

        The `self` pointer memory location is assumed to contain uninitialized
        data prior to this assignment, and consequently the current contents of
        this pointer are not destructed before writing the value from the `src`
        pointer.

        Ownership of the value is logically transferred from `src` into `self`'s
        pointer location.

        After this call, the `src` pointee value should be treated as
        uninitialized data. Subsequent reads of or destructor calls on the `src`
        pointee value are invalid, unless and until a new valid value has been
        written into the `src` pointer's memory location using an
        `unsafe_write()` method.

        This transfers the value out of `src` and into `self` using at most one
        move constructor call.

        Example:

        ```mojo
        from std.memory.alloc import alloc, dealloc, Layout

        var a_allocation = alloc(Layout[String].single())
        var b_allocation = alloc(Layout[String].single())
        var a_ptr = a_allocation.unsafe_ptr()
        var b_ptr = b_allocation.unsafe_ptr()

        # Initialize A pointee
        a_ptr.unsafe_write("foo")

        # Perform the move
        b_ptr.unsafe_write_move_from(a_ptr)

        # Clean up
        b_ptr.unsafe_deinit_pointee()
        dealloc(a_allocation^)
        dealloc(b_allocation^)
        ```

        Safety:

        * `src` must point to a valid, initialized instance of `Self.T`.
        * `self` must point to writable memory for `Self.T`. The pointee
          contents of `self` should be uninitialized; if `self` was previously
          written with a valid value, that value will be overwritten and its
          destructor will NOT be run.

        Args:
            src: Source pointer that the value will be moved from.
        """
        __get_address_as_uninit_lvalue(
            self.unsafe_address_space_cast[.GENERIC]()._mlir_value
        ) = __get_address_as_owned_value(
            src.unsafe_address_space_cast[.GENERIC]()._mlir_value
        )
