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
"""Implements `OwnedPointer`, a safe, single-ownership smart pointer.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import OwnedPointer
```
"""

from std.builtin.rebind import downcast
from std.format._utils import (
    Repr,
    FormatStruct,
    TypeNames,
)
from std.memory.alloc import (
    Allocation,
    ThinAllocation,
    alloc,
    dealloc,
    Layout,
)


@explicit_destroy(
    "Use `into_inner()` (for a `Movable` `T`) or `unsafe_take_allocation()`"
    " to consume an `OwnedPointer` whose element type is not"
    " `Deinitable`"
)
struct OwnedPointer[T: AnyType](
    Deinitable where conforms_to(T, Deinitable),
    RegisterPassable,
    Writable where conforms_to(T, Writable),
):
    """A safe, owning, smart pointer.

    This smart pointer is designed for cases where there is clear ownership
    of the underlying data, and restricts access to it through the origin
    system such that no more than one mutable alias for the underlying data
    may exist.

    For a comparison with other pointer types, see [Intro to
    pointers](/docs/manual/pointers/) in the Mojo Manual.

    Parameters:
        T: The type to be stored in the `OwnedPointer`. When `T` is not
            `Deinitable`, the `OwnedPointer` has no implicit
            destructor and must be consumed with `into_inner()` (for a
            `Movable` `T`) or `unsafe_take_allocation()`.
    """

    var _inner: ThinAllocation[Self.T]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__[_T: Movable](out self: OwnedPointer[_T], var value: _T):
        """Construct a new `OwnedPointer` by moving the passed value into a new backing allocation.

        Parameters:
            _T: The type of the data to store. It is restricted to `Movable` here to allow efficient move construction.

        Args:
            value: The value to move into the `OwnedPointer`.
        """
        self._inner = alloc(Layout[_T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(value^)

    def __init__[_T: Copyable](out self: OwnedPointer[_T], *, copy_value: _T):
        """Construct a new `OwnedPointer` by explicitly copying the passed value into a new backing allocation.

        Parameters:
            _T: The type of the data to store, which must be
               `Copyable`.

        Args:
            copy_value: The value to explicitly copy into the `OwnedPointer`.
        """
        self._inner = alloc(Layout[_T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(copy=copy_value)

    def __init__[
        _T: Copyable, U: NoneType = None
    ](out self: OwnedPointer[_T], value: _T):
        """Construct a new `OwnedPointer` by copying the passed value into a new backing allocation.

        Parameters:
            _T: The type of the data to store.
            U: A dummy type parameter, to lower the selection priority of this ctor.

        Args:
            value: The value to copy into the `OwnedPointer`.
        """
        self._inner = alloc(Layout[_T].single()).into_thin()
        self._inner.unsafe_ptr().unsafe_write(copy=value)

    def __init__[
        _T: Copyable
    ](out self: OwnedPointer[_T], *, other: OwnedPointer[_T]):
        """Construct a new `OwnedPointer` by explicitly copying the value from another `OwnedPointer`.

        Parameters:
            _T: The type of the data to store.

        Args:
            other: The `OwnedPointer` to copy.
        """
        self = OwnedPointer[_T](copy_value=other[])

    def __init__(
        out self,
        *,
        unsafe_from_raw_pointer: Pointer[Self.T, MutUntrackedOrigin],
    ):
        """Construct a new `OwnedPointer` by taking ownership of the provided `Pointer`.

        Args:
            unsafe_from_raw_pointer: The `Pointer` to take ownership of.

        Safety:

        This function is unsafe as the provided `Pointer` must be initialize with a single valid `T`
        initially allocated with this `OwnedPointer`'s backing allocator.
        This function is unsafe as other memory problems can arise such as a double-free if this function
        is called twice with the same pointer or a user manually deallocates the same data.

        After using this constructor, the `Pointer` is assumed to be owned by this `OwnedPointer`.
        In particular, the destructor method will call `T.__deinit__` and `dealloc`.
        """
        self._inner = ThinAllocation(unsafe_owned_ptr=unsafe_from_raw_pointer)

    def __init__(out self, *, unsafe_from_opaque_pointer: MutOpaquePointer[_]):
        """Construct a new `OwnedPointer` by taking ownership of the provided `Pointer`.

        Args:
            unsafe_from_opaque_pointer: The `OpaquePointer` to take ownership of.

        Safety:

        This function is unsafe as the provided `OpaquePointer` must be initialize with a single valid `T`
        initially allocated with this `OwnedPointer`'s backing allocator.
        This function is unsafe as other memory problems can arise such as a double-free if this function
        is called twice with the same pointer or a user manually deallocates the same data.

        After using this constructor, the `Pointer` is assumed to be owned by this `OwnedPointer`.
        In particular, the destructor method will call `T.__deinit__` and `dealloc`.
        """
        var ptr = unsafe_from_opaque_pointer.unsafe_bitcast[Self.T]()
        self = Self(
            unsafe_from_raw_pointer=ptr.unsafe_origin_cast[MutUntrackedOrigin]()
        )

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        """Destroy the `OwnedPointer`, running the destructor of its value.

        Constraints:
            `T` must be `Deinitable`. When it is not, the
            `OwnedPointer` has no implicit destructor and must be consumed
            with `into_inner()` (for a `Movable` `T`) or
            `unsafe_take_allocation()`.
        """
        self._inner.unsafe_ptr().unsafe_deinit_pointee()
        dealloc(self._inner^.unsafe_with_layout(Layout[Self.T].single()))

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    def __getitem__(
        ref[AddressSpace.GENERIC] self,
    ) -> ref[
        origin_of(self)._get_owned_interior["value"], AddressSpace.GENERIC
    ] Self.T:
        """Returns a reference to the pointers's underlying data with parametric mutability.

        Returns:
            A reference to the data underlying the `OwnedPointer`.
        """
        return self._inner.unsafe_ptr()._get_ref_with_unsafe_interior_origin[
            "value", origin_of(self)
        ]()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def ptr[origin: Origin, //](ref[origin] self) -> Pointer[Self.T, origin]:
        """Returns a pointer to the `OwnedPointer`'s contents.

        Parameters:
            origin: The origin of the pointer.

        Returns:
            A pointer to the `OwnedPointer`'s contents.
        """
        return (
            self._inner.unsafe_ptr()
            .unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
        )

    @doc_hidden
    @deprecated(use=ptr)
    def unsafe_ptr[
        mut: Bool,
        origin: Origin[mut=mut],
        //,
    ](ref[origin] self) -> Pointer[Self.T, origin]:
        """Returns the backing pointer for this `OwnedPointer`.

        Parameters:
            mut: Whether the pointer is mutable.
            origin: The origin of the pointer.

        Returns:
            A pointer to the backing allocation for this `OwnedPointer`.
        """
        return self.ptr()

    def into_inner(deinit self) -> Self.T where conforms_to(Self.T, Movable):
        """Move the value within the `OwnedPointer` out of it, consuming the
        `OwnedPointer` in the process.

        Returns:
            The data that is (was) backing the `OwnedPointer`.
        """
        var r = self._inner.unsafe_ptr().unsafe_take_pointee()
        dealloc(self._inner^.unsafe_with_layout(Layout[Self.T].single()))
        return r^

    def unsafe_take_allocation(deinit self) -> Allocation[Self.T]:
        """Take ownership of the heap allocation backing this `OwnedPointer`.

        Returns:
            The `Allocation` that owns the backing storage.

        Safety:

        The pointee is handed over still initialized, and deallocating the
        storage does not run its destructor. Destroy it yourself before
        deallocating if `T` needs it.

        The returned `Allocation` is an explicitly destroyed handle, so the
        compiler requires the caller to consume it: pass it to `dealloc`, or
        take the raw pointer with `unsafe_leak()`.
        """
        return self._inner^.unsafe_with_layout(Layout[Self.T].single())

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Formats this pointer's value to the provided Writer.

        Args:
            writer: The object to write to.

        Constraints:
            `T` must conform to Writable.
        """
        self[].write_to(writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write the string representation of the `OwnedPointer`.

        Args:
            writer: The object to write to.

        Constraints:
            `T` must conform to Writable.
        """
        FormatStruct(writer, "OwnedPointer").params(
            TypeNames[Self.T](),
        ).fields(Repr(self[]))
