# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
from memory import UnsafePointer, stack_allocation, memcpy


struct Boxed[T: AnyType, address_space: AddressSpace = AddressSpace.GENERIC]:
    """A safe, owning, smart pointer.

    This smart pointer is designed for cases where there is clear ownership
    of the underlying data, and restricts access to it through the lifetime
    system such that no more than one mutable alias for the underlying data
    may exist. Consider Boxed[T] over UnsafePointer[T] where possible.

    Parameters:
        T: The type to be stored in the Boxed[].
        address_space: The address space that the allocation behind this box resides in.
    """

    var _inner: UnsafePointer[T, address_space]

    fn __init__[
        T: Movable
    ](inout self: Boxed[T, AddressSpace.GENERIC], owned value: T):
        """Construct a new Boxed[] by moving the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store. It is restricted to `Movable` here to allow efficient move construction.

        Args:
            value: The value to move into the Boxed[].
        """
        self._inner = UnsafePointer[T, AddressSpace.GENERIC].alloc(1)
        self._inner.init_pointee_move(value^)

    fn __init__[
        T: ExplicitlyCopyable
    ](inout self: Boxed[T, AddressSpace.GENERIC], *, copy_value: T):
        """Construct a new Boxed[] by explicitly copying the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store.

        Args:
            copy_value: The value to explicitly copy into the Boxed[].
        """
        self._inner = UnsafePointer[T, address_space].alloc(1)
        self._inner.init_pointee_explicit_copy(copy_value)

    fn __init__[
        T: ExplicitlyCopyable
    ](
        inout self: Boxed[T, AddressSpace.GENERIC],
        *,
        copy_box: Boxed[T, AddressSpace.GENERIC],
    ):
        """Construct a new Boxed[] by explicitly copying the value from another Boxed[].

        Parameters:
            T: The type of the data to store.

        Args:
            copy_box: The Boxed[] to copy.
        """
        self.__init__(copy_value=copy_box[])

    fn __moveinit__(inout self, owned existing: Self):
        """Move this Boxed[].

        Args:
            existing: The value to move.
        """
        self._inner = existing._inner
        existing._inner = UnsafePointer[T, address_space]()

    fn __getitem__(
        ref [_, address_space._value.value]self
    ) -> ref [__lifetime_of(self._inner), address_space._value.value] T:
        """Returns a reference to the box's underlying data with parametric mutability.

        Returns:
            A reference to the data underlying the Boxed[].
        """
        # This should have a widening conversion here that allows
        # the mutable ref that is always (potentially unsafely)
        # returned from UnsafePointer to be guarded behind the
        # aliasing guarantees of the lifetime system here.
        # All of the magic happens above in the function signature
        var inner_not_null = self._inner.__bool__()

        debug_assert(
            inner_not_null,
            (
                "Box is horribly broken, and __getitem__ was called on a"
                " destroyed box"
            ),
        )

        return self._inner[]

    fn __del__(owned self: Boxed[T, AddressSpace.GENERIC]):
        """Destroy the Boxed[]."""
        # check that inner is non-null to accomodate take() and other
        # consuming end states
        if self._inner:
            (self._inner).destroy_pointee()
            self._inner.free()
            self._inner = UnsafePointer[T, AddressSpace.GENERIC]()

    fn unsafe_ptr(self) -> UnsafePointer[T, address_space]:
        """UNSAFE: returns the backing pointer for this Boxed[]

        Returns:
            An UnsafePointer to the backing allocation for this Boxed[].
        """
        return self._inner

    fn take[T: Movable](owned self: Boxed[T, AddressSpace.GENERIC]) -> T:
        """Move the value within the Boxed[] out of it, consuming the Boxed[] in the process.

        Parameters:
            T: The type of the data backing this Boxed[]. `take()` only exists for T: Movable
                since this consuming operation only makes sense for types that you want to avoid copying.
                For types that are Copy or ExplicitlyCopy but are not Movable, you can copy them through
                `__getitem__` as in `var v = some_box_var[]`.

        Returns:
            The data that is (was) backing the Boxed[].
        """
        var r = self._inner.take_pointee()
        self._inner.free()
        self._inner = UnsafePointer[T, AddressSpace.GENERIC]()

        return r^
