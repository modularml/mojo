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
"""Defines a Variant type."""

from std.builtin.rebind import downcast
from std.format._utils import (
    FormatStruct,
    TypeNames,
)
from std.memory import MaybeUninit
from std.hashlib.hasher import Hasher
from std.reflection import call_location
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from ._nicheable import (
    UnsafeNicheable,
    NicheIndex,
    NicheStorageTraits,
    UnsafeCustomNicheStorage,
)
from std.os import abort
from std.sys import align_of, size_of

# ===----------------------------------------------------------------------=== #
# Variant Storages
# ===----------------------------------------------------------------------=== #

comptime _InvalidTypeIndex: Int = -1


@always_inline
def _get_type_index[T: AnyType, *Ts: AnyType]() -> Int:
    comptime for i in range(Ts.length):
        comptime if Ts[i] == T:
            return i
    return _InvalidTypeIndex


trait _VariantStorage(Copyable, Deinitable):
    """Internal storage backend for `Variant`.

    This trait abstracts over the two concrete storage strategies:

    - `_DefaultVariantStorage`: general discriminated-union storage backed by
      an MLIR `kgen.variant` allocation with an explicit integer discriminant.
    - `_NichedOptionalStorage`: niche-optimized storage for two-type variants
      where one type is `UnsafeNicheable` and the other is zero-sized; encodes
      the active type in an invalid bit pattern rather than a separate tag byte.
    """

    def __init__[U: Movable](out self, var value: U):
        """Initialize storage with a value of type `U`."""
        ...

    def __init__(out self, *, unsafe_uninitialized: ()):
        """Create storage whose active-type slot is left uninitialized.

        The caller must mark the active type with `unsafe_set_active` and then
        write a valid value into `unsafe_ptr` before the storage is read or
        destroyed."""
        ...

    def unsafe_set_active[U: AnyType](mut self):
        """Mark `U` as the active type without writing its value.

        Used together with `unsafe_uninitialized` and `unsafe_ptr` to populate
        storage in place. The caller must emplace a valid `U` (or, for the
        empty type of a niche-optimized variant, leave the encoded niche
        untouched) immediately afterwards."""
        ...

    def unwrap[U: Movable](deinit self) -> U:
        """Consume this storage and return the held value as type `U`."""
        return self.unsafe_ptr[U]().unsafe_take_pointee()

    def unsafe_discard(deinit self):
        """Consume this storage without reading or destroying the held value.

        Safety: the held value must already have been destroyed or moved out;
        the active slot is treated as uninitialized."""
        pass

    def isa[U: AnyType](self) -> Bool:
        """Return `True` if the currently active type is `U`."""
        ...

    def unsafe_ptr[U: AnyType](ref self) -> Pointer[U, origin_of(self)]:
        """Return a raw pointer to the stored data interpreted as type `U`.

        Safety: the caller must ensure `U` matches the active type."""
        ...


trait _NicheStorage(Defaultable, Deinitable):
    """Internal abstraction over niche backing storage backends."""

    def as_uninit[
        T: AnyType
    ](ref self) -> Pointer[MaybeUninit[T], origin_of(self)]:
        ...


struct _DefaultNicheStorage[T: AnyType](
    Defaultable, Movable where False, _NicheStorage
):
    """Default niche backing: stores the value in `MaybeUninit[T]`
    (lowers to `pop.array<1, T>`)."""

    var _memory: MaybeUninit[Self.T]

    @always_inline
    def __init__(out self):
        self._memory = {}

    def __deinit__(deinit self):
        self._memory^.unsafe_forget()

    @always_inline
    def as_uninit[
        U: AnyType
    ](ref self) -> Pointer[MaybeUninit[U], origin_of(self)]:
        comptime assert Self.T == U
        return (
            Pointer(to=self._memory)
            .unsafe_bitcast[MaybeUninit[U]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


struct _CustomNicheStorage[Storage: UnsafeCustomNicheStorage](
    Defaultable, _NicheStorage
):
    """Niche backing that delegates to the user-provided `Storage` type,
    allowing the nicheable type to control what MLIR type the storage lowers
    to."""

    var _memory: Self.Storage.NicheStorage

    @always_inline
    def __init__(out self):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    def as_uninit[
        T: AnyType
    ](ref self) -> Pointer[MaybeUninit[T], origin_of(self)]:
        comptime assert (
            size_of[Self.Storage.NicheStorage]() == size_of[MaybeUninit[T]]()
        ), "Custom storage must be the same size as Self"
        comptime assert (
            align_of[Self.Storage.NicheStorage]() == align_of[MaybeUninit[T]]()
        ), "Custom storage must have the the same alignment as Self"
        return (
            Pointer(to=self._memory)
            .unsafe_bitcast[MaybeUninit[T]]()
            .unsafe_origin_cast[origin_of(self)]()
        )


comptime _NicheStorageFor[T: AnyType] = _CustomNicheStorage[T] if conforms_to(
    T, UnsafeCustomNicheStorage
) else _DefaultNicheStorage[T]


struct _NichedOptionalStorage[
    T: UnsafeNicheable, EmptyType: TrivialRegisterPassable
](
    Copyable,
    RegisterPassable where conforms_to(T, RegisterPassable),
    _VariantStorage,
):
    """Optimized storage for two-type variants where one type is `UnsafeNicheable`
    and the other is zero-sized & `TrivialRegisterPassable` (e.g. `NoneType`).

    Instead of storing a discriminant tag, the niche of `T` (an invalid bit
    pattern) is repurposed to encode the "empty" state, eliminating the extra
    byte of overhead that `_DefaultVariantStorage` would require."""

    comptime __del__is_trivial = IsTriviallyDeinitable[Self.T]
    comptime __copy_ctor_is_trivial = IsTriviallyCopyable[Self.T]
    comptime __move_ctor_is_trivial = IsTriviallyMovable[Self.T]

    var _memory: _NicheStorageFor[Self.T]

    @staticmethod
    def _check[U: AnyType]():
        comptime assert U == Self.T or U == Self.EmptyType, "unexpected type"

    @always_inline
    def __init__(out self):
        comptime assert (
            Self.T.niche_count() > 0
        ), "UnsafeNicheable must specify at least 1 invalid bit pattern"
        self._memory = {}
        Self.T.write_niche[index=0](self._memory.as_uninit[Self.T]())

    @always_inline
    def __init__[U: Movable](out self, var value: U):
        Self._check[U]()
        comptime if U == Self.T:
            self._memory = {}
            self._memory.as_uninit[U]()[].unsafe_write(value^)
        else:
            # This is the empty "none" type. `U` is refined to
            # `TrivialRegisterPassable` above, so an explicit `^` transfer of
            # `value` is a no-op the compiler rejects; a plain discard suffices.
            comptime assert conforms_to(U, TrivialRegisterPassable)
            _ = value
            self = Self()

    @always_inline
    def __init__(out self, *, unsafe_uninitialized: ()):
        self._memory = {}

    @always_inline
    def unsafe_set_active[U: AnyType](mut self):
        Self._check[U]()
        comptime if U != Self.T:
            # The empty ("none") type is encoded by the niche; the nicheable
            # type becomes active simply by writing a valid value into it, so
            # only the empty case needs to stamp the niche here.
            Self.T.write_niche[index=0](self._memory.as_uninit[Self.T]())

    @always_inline
    def __init__(out self, *, deinit move: Self):
        comptime assert conforms_to(Self.T, Movable)
        if move.isa[Self.T]():
            self = Self(move.unsafe_ptr[Self.T]().unsafe_take_pointee())
        else:
            self = Self()

    @always_inline
    def __init__(out self, *, copy: Self):
        comptime assert conforms_to(Self.T, Copyable)
        if copy.isa[Self.T]():
            self = Self(copy.unsafe_ptr[Self.T]()[].copy())
        else:
            self = Self()

    @always_inline
    def __deinit__(deinit self):
        comptime assert conforms_to(Self.T, Deinitable)
        if self.isa[Self.T]():
            self._memory.as_uninit[
                Self.T
            ]()[].unsafe_ptr().unsafe_deinit_pointee()

    @always_inline
    def isa[U: AnyType](self) -> Bool:
        Self._check[U]()
        var niche = Self.T.classify_niche(self._memory.as_uninit[Self.T]())
        var is_some = niche == NicheIndex.NotANiche
        comptime if U == Self.T:
            return is_some
        else:
            return not is_some

    @always_inline
    def unsafe_ptr[U: AnyType](ref self) -> Pointer[U, origin_of(self)]:
        Self._check[U]()
        # The niche backing only has a slot for `Self.T`, so address that slot
        # and bitcast. When `U` is the zero-sized empty type, the slot's
        # location is still a valid `U` address.
        return (
            self._memory.as_uninit[Self.T]()
            .unsafe_bitcast[U]()
            .unsafe_origin_cast[origin_of(self)]()
        )


struct _DefaultVariantStorage[*Ts: AnyType](
    Copyable,
    RegisterPassable where Ts.all_conforms_to[RegisterPassable](),
    _VariantStorage,
):
    """General-purpose discriminated-union storage for `Variant`.

    Stores all possible types in a single MLIR `kgen.variant` allocation and
    tracks the active type via an integer discriminant. Used whenever the
    variant types do not qualify for the niche-optimized path."""

    comptime __del__is_trivial = Self.Ts.all[IsTriviallyDeinitable]()
    comptime __copy_ctor_is_trivial = Self.Ts.all[IsTriviallyCopyable]()
    comptime __move_ctor_is_trivial = Self.Ts.all[IsTriviallyMovable]()

    comptime _mlir_type = __mlir_type[
        `!kgen.variant<[rebind(:`,
        type_of(Self.Ts.values),
        ` `,
        Self.Ts.values,
        `)]>`,
    ]
    var _impl: Self._mlir_type

    @always_inline
    def __init__(out self, *, unsafe_uninitialized: ()):
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline
    def __init__[T: Movable](out self, var value: T):
        self = Self(unsafe_uninitialized=())
        self.get_discriminant() = UInt8(_get_type_index[T, *Self.Ts]())
        self.unsafe_ptr[T]().unsafe_write(value^)

    @always_inline
    def unsafe_set_active[T: AnyType](mut self):
        self.get_discriminant() = UInt8(_get_type_index[T, *Self.Ts]())

    @always_inline
    def __init__(out self, *, copy: Self):
        self = Self(unsafe_uninitialized=())
        self.get_discriminant() = copy.get_discriminant()

        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(T, Copyable)

            if self.get_discriminant() == UInt8(i):
                self.unsafe_ptr[T]().unsafe_write(copy=copy.unsafe_ptr[T]()[])
                return

    @always_inline
    def __init__(out self, *, deinit move: Self):
        self = Self(unsafe_uninitialized=())
        self.get_discriminant() = move.get_discriminant()

        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(T, Movable)

            if self.get_discriminant() == UInt8(i):
                self.unsafe_ptr[T]().unsafe_write_move_from(
                    move.unsafe_ptr[T]()
                )
                return

    @always_inline
    def __deinit__(deinit self):
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            comptime assert conforms_to(T, Deinitable)

            if self.get_discriminant() == UInt8(i):
                self.unsafe_ptr[T]().unsafe_deinit_pointee()
                return

    @always_inline("nodebug")
    def get_discriminant(ref self) -> ref[self] UInt8:
        var discr_ptr = __mlir_op.`pop.variant.discr_gep`[
            _type=__mlir_type.`!kgen.pointer<scalar<ui8>>`
        ](Pointer(to=self._impl)._get_kgen_pointer())
        return Pointer[_, origin_of(self)](
            _mlir_value=discr_ptr
        ).unsafe_bitcast[UInt8]()[]

    @always_inline("nodebug")
    def isa[T: AnyType](self) -> Bool:
        comptime discriminant = UInt8(_get_type_index[T, *Self.Ts]())
        return self.get_discriminant() == discriminant

    @always_inline("nodebug")
    def unsafe_ptr[T: AnyType](ref self) -> Pointer[T, origin_of(self)]:
        comptime idx = _get_type_index[T, *Self.Ts]()
        return {
            _mlir_value = __mlir_op.`pop.variant.bitcast`[
                _type=Pointer[T, origin_of(self)]._mlir_type,
                index=idx.__mlir_index__(),
            ](Pointer(to=self._impl)._get_kgen_pointer())
        }


# TODO(MOCO-3653): size_of[T]() == 0 does not work correctly in some cases when
# an `Optional` is used as a comptime parameter's field.
comptime _IsEmptyType[T: AnyType]: Bool = reflect[
    T
].field_count() == 0 and conforms_to(T, TrivialRegisterPassable)
"""True if `T` is a zero-sized, trivially passable type (i.e. carries no state,
like `NoneType`). Used to identify the "empty" type of a niche-optimized variant."""

comptime _IsNicheablePair[T: AnyType, U: AnyType]: Bool = conforms_to(
    T, UnsafeNicheable
) and _IsEmptyType[U]
"""True if `T` is `UnsafeNicheable` and `U` is an empty type. Called twice with
swapped args by `_IsNicheEligible` to handle either ordering."""

comptime _IsNicheEligible[*Ts: AnyType]: Bool = (Ts.length == 2) and (
    _IsNicheablePair[Ts[0], Ts[1]] or _IsNicheablePair[Ts[1], Ts[0]]
)
"""True if `Ts` qualifies for niche-optimized storage: exactly two types
where one is `UnsafeNicheable` and the other is an empty type."""

comptime _NichedStorageFor[*Ts: AnyType] = _NichedOptionalStorage[
    Ts[0],
    downcast[Ts[1], TrivialRegisterPassable],
] if conforms_to(Ts[0], UnsafeNicheable) else _NichedOptionalStorage[
    downcast[Ts[1], UnsafeNicheable],
    downcast[Ts[0], TrivialRegisterPassable],
]
"""Resolves to the concrete `_NichedOptionalStorage[T]` for the eligible type,
regardless of which position the `UnsafeNicheable` type occupies in `Ts`."""

comptime _VariantStorageFor[*Ts: AnyType] = _NichedStorageFor[
    *Ts
] if _IsNicheEligible[*Ts] else _DefaultVariantStorage[*Ts]
"""Selects the storage strategy for `Variant[*Ts]`: niche-optimized storage
when eligible, falling back to the general discriminant-tagged storage."""

# ===----------------------------------------------------------------------=== #
# Variant
# ===----------------------------------------------------------------------=== #


struct Variant[*Ts: AnyType](
    Copyable where Ts.all_conforms_to[Copyable](),
    Deinitable where Ts.all_conforms_to[Deinitable](),
    Equatable where Ts.all_conforms_to[Equatable](),
    Hashable where Ts.all_conforms_to[Hashable](),
    ImplicitlyCopyable where Ts.all_conforms_to[ImplicitlyCopyable](),
    Movable where Ts.all_conforms_to[Movable](),
    RegisterPassable where Ts.all_conforms_to[RegisterPassable](),
    Writable where Ts.all_conforms_to[Writable](),
):
    """A union that can hold a runtime-variant value from a set of predefined
    types.

    `Variant` is a discriminated union type, similar to `std::variant` in C++
    or `enum` in Rust. It can store exactly one value that can be any of the
    specified types, determined at runtime.

    The key feature is that the actual type stored in a `Variant` is determined
    at runtime, not compile time. This allows you to change what type a variant
    holds during program execution. Memory-wise, a variant only uses the space
    needed for the largest possible type plus a small discriminant field to
    track which type is currently active.

    Tips:

    - use `isa[T]()` to check what type a variant is
    - use `unsafe_unwrap[T]()` to take a value from the variant
    - use `[T]` to get a value out of a variant
        - This currently does an extra copy/move until we have origins
        - It also temporarily requires the value to be mutable
    - use `set[T](var new_value: T)` to reset the variant to a new value
    - use `is_type_supported[T]` to check if the variant permits the type `T`

    **Note**: Currently, variant operations require the variant to be
    mutable (`mut`), even for read operations.

    Example:

    ```mojo
    from std.utils import Variant
    import std.random as random

    comptime IntOrString = Variant[Int, String]

    def to_string(mut x: IntOrString) -> String:
        if x.isa[String]():
            return x[String]
        return String(x[Int])

    var an_int = IntOrString(4)
    var a_string = IntOrString("I'm a string!")
    var who_knows = IntOrString(0)
    # Randomly change who_knows to a string
    random.seed()
    if random.random_ui64(0, 1):
        who_knows.set[String]("I'm also a string!")

    print(a_string[String])      # => I'm a string!
    print(an_int[Int])           # => 4
    print(to_string(who_knows))  # Either 0 or "I'm also a string!"

    if who_knows.isa[String]():
        print("It's a String!")
    ```

    Example usage for error handling:

    ```mojo
    comptime Result = Variant[String, Error]

    def process_data(data: String) -> Result:
        if data.byte_length() == 0:
            return Result(Error("Empty data"))
        return Result(String("Processed: ", data))

    var result = process_data("Hello")
    if result.isa[String]():
        print("Success:", result[String])
    else:
        print("Error:", result[Error])
    ```

    Example usage in a `List` to create a heterogeneous list:

    ```mojo
    comptime MixedType = Variant[Int, Float64, String, Bool]

    var mixed_list = List[MixedType]()
    mixed_list.append(MixedType(42))
    mixed_list.append(MixedType(3.14))
    mixed_list.append(MixedType("hello"))
    mixed_list.append(MixedType(True))

    for item in mixed_list:
        if item.isa[String]():
            print("String:", item[String])
        elif item.isa[Int]():
            print("Integer:", item[Int])
        elif item.isa[Float64]():
            print("Float:", item[Float64])
        elif item.isa[Bool]():
            print("Boolean:", item[Bool])
    ```

    ## Layout

    The layout of `Variant` is not guaranteed and may change at any time. The
    implementation may apply niche optimizations (for example, encoding the
    discriminant inside spare bits of one of the types in `Ts`) that alter the
    resulting layout. Do not rely on `size_of[Variant[...]]()` or
    `align_of[Variant[...]]()` being stable across language versions. The only
    guarantee is that the size and alignment will be at least as large as those
    of the largest type among `Ts`.

    If you need to inspect the current size or alignment, use `size_of` and
    `align_of`, but treat the results as non-stable implementation details.

    Parameters:
        Ts: The possible types that this variant can hold. Types that
            implement `Copyable` enable copy semantics for the variant.
    """

    comptime _Storage: _VariantStorage = _VariantStorageFor[*Self.Ts]

    comptime __del__is_trivial = IsTriviallyDeinitable[Self._Storage]
    comptime __copy_ctor_is_trivial = IsTriviallyCopyable[Self._Storage]
    comptime __move_ctor_is_trivial = IsTriviallyMovable[Self._Storage]

    # Fields
    var _storage: Self._Storage

    @staticmethod
    def _check[T: AnyType]():
        comptime idx = _get_type_index[T, *Self.Ts]()
        comptime assert (
            idx != _InvalidTypeIndex
        ), "Type does not exist in Variant."

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @implicit
    def __init__[T: Movable](out self, var value: T):
        """Create a variant with one of the types.

        Parameters:
            T: The type to initialize the variant to. Generally this should
                be able to be inferred from the call type, eg. `Variant[Int, String](4)`.

        Args:
            value: The value to initialize the variant with.
        """
        Self._check[T]()
        self._storage = Self._Storage(value^)

    def __init__[T: AnyType, //, F: def() -> T](out self, *, init_with: F):
        """Create a variant holding a `T` produced in place by a closure.

        The value returned by `init_with` is constructed directly into the
        variant's storage without being moved, so this is the only way to
        store a value whose type is not `Movable`.

        The `init_with` keyword is required to disambiguate from the value
        constructor: a closure is itself a storable value, so a positional
        `Variant(f)` stores `f`, whereas `Variant(init_with=f)` calls `f` and
        stores its result.

        Parameters:
            T: The type to initialize the variant to. Must be one of the
                variant's type arguments.
            F: The type of the initializer closure.

        Args:
            init_with: A closure returning the value to store. Called exactly once.

        Examples:

        ```mojo
        from std.utils import Variant

        @fieldwise_init
        struct Pinned(Movable where False):
            var value: Int

        def make() -> Pinned:
            return Pinned(7)

        var v = Variant[Pinned, Int](init_with=make)
        print(v[Pinned].value)  # => 7
        ```
        """
        Self._check[T]()
        self._storage = Self._Storage(unsafe_uninitialized=())
        self._storage.unsafe_set_active[T]()
        self._storage.unsafe_ptr[T]().unsafe_write(init_with=init_with)

    def __deinit__(
        deinit self,
    ) where Self.Ts.all_conforms_to[Deinitable]():
        """Destroy the variant, running the destructor of the currently held value.

        Constraints:
            All types in `Ts` must conform to `Deinitable`.
        """
        self._storage^.__deinit__()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem_param__[
        T: AnyType
    ](ref self) -> ref[origin_of(self)._get_owned_interior["value"]] T:
        """Get the value out of the variant as a type-checked type.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            A reference to the internal data.
        """
        if not self.isa[T]():
            abort("get: wrong variant type", location=call_location())

        return self._storage.unsafe_ptr[
            T
        ]()._get_ref_with_unsafe_interior_origin["value", origin_of(self)]()

    @always_inline
    def __eq__(
        self, other: Self
    ) -> Bool where Self.Ts.all_conforms_to[Equatable]():
        """Compares two variants for equality.

        Two variants are equal if they hold the same type and the held
        values are equal.

        Args:
            other: The other variant to compare against.

        Returns:
            True if the variants hold the same type and equal values.
        """
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                if not other.isa[T]():
                    return False
                return self.unsafe_get[T]() == other.unsafe_get[T]()
        return False

    @always_inline
    def __ne__(
        self, other: Self
    ) -> Bool where Self.Ts.all_conforms_to[Equatable]():
        """Compares two variants for inequality.

        Args:
            other: The other variant to compare against.

        Returns:
            True if the variants hold different types or unequal values.
        """
        return not self == other

    def __hash__(
        self, mut hasher: Some[Hasher]
    ) where Self.Ts.all_conforms_to[Hashable]():
        """Hashes the variant using the given hasher.

        The hash incorporates both the type discriminant and the held
        value's hash, so variants holding different types are unlikely to
        collide.

        Args:
            hasher: The hasher instance.
        """
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                UInt8(i).__hash__(hasher)
                self.unsafe_get[T]().__hash__(hasher)
                return

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def _write_value_to[
        *, is_repr: Bool
    ](self, mut writer: Some[Writer]) where Self.Ts.all_conforms_to[Writable]():
        comptime for i in range(Self.Ts.length):
            comptime T = Self.Ts[i]
            if self.isa[T]():
                ref value = self.unsafe_get[T]()

                comptime if is_repr:
                    value.write_repr_to(writer)
                else:
                    value.write_to(writer)

                return

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where Self.Ts.all_conforms_to[Writable]():
        """Writes the currently held variant value to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self._write_value_to[is_repr=False](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where Self.Ts.all_conforms_to[Writable]():
        """Write the string representation of the Variant.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_field(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_value_to[is_repr=True](w)

        FormatStruct(writer, "Variant").params(TypeNames[*Self.Ts]()).fields(
            write_field
        )

    @always_inline
    def unwrap[T: Movable](deinit self) -> T:
        """Take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[T]():
            abort("taking the wrong type!")

        return self._storage^.unwrap[T]()

    @always_inline
    def _unsafe_unchecked_unwrap[T: Movable](deinit self) -> T:
        return self._storage^.unwrap[T]()

    @always_inline
    def unsafe_unwrap[T: Movable](deinit self) -> T:
        """Unsafely take the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            T: The type to take out.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        Self._check[T]()
        assert self.isa[T](), "taking wrong type"
        return self._storage^.unwrap[T]()

    @always_inline
    def replace[
        Tin: Movable & Deinitable,
        Tout: Movable,
    ](mut self, var value: Tin) -> Tout:
        """Replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, the program
        will abort!

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        if not self.isa[Tout]():
            abort("taking out the wrong type!")

        return self.unsafe_replace[Tin, Tout](value^)

    @always_inline
    def unsafe_replace[
        Tin: Movable, Tout: Movable
    ](mut self, var value: Tin) -> Tout:
        """Unsafely replace the current value of the variant with the provided type.

        The caller takes ownership of the underlying value.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        Parameters:
            Tin: The type to put in.
            Tout: The type to take out.

        Args:
            value: The value to put in.

        Returns:
            The underlying data to be taken out as an owned value.
        """
        assert self.isa[Tout](), "taking out the wrong type!"

        var x = self^.unsafe_unwrap[Tout]()
        self = Self(value^)
        return x^

    def set[
        T: Movable
    ](mut self, var value: T) where Self.Ts.all_conforms_to[Deinitable]():
        """Set the variant value.

        This will call the destructor on the old value, and update the variant's
        internal type and data to the new value.

        Parameters:
            T: The new variant type. Must be one of the Variant's type arguments.

        Args:
            value: The new value to set the variant to.
        """
        self = Self(value^)

    def set[T: AnyType, //, F: def() -> T](mut self, *, init_with: F):
        """Replace the variant's value with a `T` produced in place by a closure.

        Destroys the currently held value, then constructs the closure's return
        value directly into the variant's storage without moving it. This is the
        only way to replace the contents with a value whose type is not
        `Movable`.

        The `init_with` keyword is required to disambiguate from the value-taking
        `set`: a closure is itself a storable value, so a positional
        `set(f)` stores `f`, whereas `set(init_with=f)` calls `f` and stores its
        result.

        Parameters:
            T: The new variant type. Must be one of the variant's type
                arguments.
            F: The type of the initializer closure.

        Args:
            init_with: A closure returning the replacement value. Called exactly
                once.

        Constraints:
            All types in `Ts` must conform to `Deinitable`, since the
            outgoing value is destroyed in place.

        Examples:

        ```mojo
        from std.utils import Variant

        @fieldwise_init
        struct Pinned(Movable where False):
            var value: Int

        def make() -> Pinned:
            return Pinned(7)

        var v = Variant[Pinned, Int](0)
        v.set(init_with=make)
        print(v[Pinned].value)  # => 7
        ```
        """
        comptime assert Self.Ts.all_conforms_to[
            Deinitable
        ](), "Cannot replace in place when a type is not `Deinitable`"
        Self._check[T]()
        # Destroy-then-emplace is exception-safe only because `init_with` cannot
        # raise (closure types are not `raises`); a throw here would leave the
        # discriminant set with no value written for `__deinit__` to destroy.
        self._storage^.__deinit__()
        self._storage = Self._Storage(unsafe_uninitialized=())
        self._storage.unsafe_set_active[T]()
        self._storage.unsafe_ptr[T]().unsafe_write(
            init_with=lambda () {ref} -> T: init_with()
        )

    def isa[T: AnyType](self) -> Bool:
        """Check if the variant contains the required type.

        Parameters:
            T: The type to check.

        Returns:
            True if the variant contains the requested type.
        """
        Self._check[T]()
        return self._storage.isa[T]()

    @always_inline
    def _unsafe_unchecked_get[T: AnyType](ref self) -> ref[self] T:
        return self._storage.unsafe_ptr[T]().unsafe_origin_cast[
            origin_of(self)
        ]()[]

    def unsafe_get[T: AnyType](ref self) -> ref[self] T:
        """Get the value out of the variant as a type-checked type.

        This doesn't explicitly check that your value is of that type!
        If you haven't verified the type correctness at runtime, you'll get
        a type that _looks_ like your type, but has potentially unsafe
        and garbage member data.

        For now this has the limitations that it
            - requires the variant value to be mutable

        Parameters:
            T: The type of the value to get out.

        Returns:
            The internal data represented as a `Pointer[T]`.
        """
        Self._check[T]()
        assert self.isa[T](), "get: wrong variant type"
        return self._unsafe_unchecked_get[T]()

    # ===-------------------------------------------------------------------===#
    # In-place construction primitives
    # ===-------------------------------------------------------------------===#
    #
    # These internal primitives expose storage-level placement-new so a wrapper
    # that already knows which of the variant's types to store (for example
    # `Optional`, whose element floor is `AnyType`) can construct a
    # non-`Movable` value in place. They sidestep the `init_with=` ctor, whose stored
    # type is inferred from the closure return and so cannot be resolved when a
    # wrapper forwards an abstract closure through it. Use them together: build
    # with `unsafe_uninitialized`, mark the active type with
    # `_unsafe_set_active[T]`, then placement-new a `T` into `_unsafe_ptr[T]`
    # exactly once before any read or destroy. The constructor carries
    # `@doc_hidden` because a dunder can't be hidden by an underscore name.

    @doc_hidden
    def __init__(out self, *, unsafe_uninitialized: ()):
        """Create a variant whose active-type slot is left uninitialized.

        Args:
            unsafe_uninitialized: Tag to select this constructor.
        """
        self._storage = Self._Storage(unsafe_uninitialized=())

    def _unsafe_set_active[T: AnyType](mut self):
        """Mark `T` as the active type without writing its value.

        Parameters:
            T: The type to mark active. Must be one of the variant's types.
        """
        Self._check[T]()
        self._storage.unsafe_set_active[T]()

    def _unsafe_ptr[T: AnyType](ref self) -> Pointer[T, origin_of(self)]:
        """Return a raw pointer to the active slot interpreted as type `T`.

        Parameters:
            T: The type to interpret the slot as. Must be the active type.

        Returns:
            A pointer to the storage slot as a `Pointer[T]`.
        """
        Self._check[T]()
        return self._storage.unsafe_ptr[T]().unsafe_origin_cast[
            origin_of(self)
        ]()

    @staticmethod
    def is_type_supported[T: Movable]() -> Bool:
        """Check if a type can be used by the `Variant`.

        Parameters:
            T: The type of the value to check support for.

        Returns:
            `True` if type `T` is supported by the `Variant`.

        Example:

        ```mojo
        from std.utils import Variant

        def takes_variant(mut arg: Variant) raises:
            if arg.is_type_supported[Float64]():
                arg = Float64(1.5)

        def main() raises:
            var x = Variant[Int, Float64](1)
            takes_variant(x)
            if x.isa[Float64]():
                print(x[Float64]) # 1.5
        ```

        For example, the `Variant[Int, Bool]` permits `Int` and `Bool`.
        """
        return Self.Ts.contains[T]()

    def deinit_with[T: AnyType, F: def(var T)](deinit self, deinit_func: F, /):
        """Deinitialize a value contained in this Variant in-place using a caller
        provided destructor function.

        This method can be used to deinitialize types that do not conform to
        `Deinitable` in a `Variant` in-place.

        This method will abort if this variant does not current contain an
        element of the specified type `T`.

        Parameters:
            T: The element type the variant is expected to currently contain,
                and which will be deinitialized by `deinit_func`.
            F: The type of the caller-provided deinitializer function.

        Args:
            deinit_func: Caller-provided function for deinitializing
                an instance of `T`.
        """
        if not self.isa[T]():
            abort("Variant.deinit_with: wrong variant type")

        self._storage.unsafe_ptr[T]().unsafe_deinit_pointee_with(deinit_func)
        self._storage^.unsafe_discard()
