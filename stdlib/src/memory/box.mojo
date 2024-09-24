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


struct Box[T: AnyType]:
    """A safe, owning, smart pointer.

    This smart pointer is designed for cases where there is clear ownership
    of the underlying data, and restricts access to it through the lifetime
    system such that no more than one mutable alias for the underlying data
    may exist.

    Parameters:
        T: The type to be stored in the Box[].
    """

    var _inner: UnsafePointer[T, AddressSpace.GENERIC]

    fn __init__[T: Movable](inout self: Box[T], owned value: T):
        """Construct a new Box[] by moving the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store. It is restricted to `Movable` here to allow efficient move construction.

        Args:
            value: The value to move into the Box[].
        """
        self._inner = UnsafePointer[T].alloc(1)
        self._inner.init_pointee_move(value^)

    fn __init__[T: ExplicitlyCopyable](inout self: Box[T], *, copy_value: T):
        """Construct a new Box[] by explicitly copying the passed value into a new backing allocation.

        Parameters:
            T: The type of the data to store.

        Args:
            copy_value: The value to explicitly copy into the Box[].
        """
        self._inner = UnsafePointer[T].alloc(1)
        self._inner.init_pointee_explicit_copy(copy_value)

    # TODO: disambiguation and other niceties
    #    fn __init__[
    #        T: Copyable
    #    ](inout self: Box[T], value: T):
    #        self._inner = UnsafePointer[T].alloc(1)
    #        self._inner.init_pointee_copy(value)

    fn __init__[
        T: ExplicitlyCopyable
    ](inout self: Box[T], *, copy_box: Box[T],):
        """Construct a new Box[] by explicitly copying the value from another Box[].

        Parameters:
            T: The type of the data to store.

        Args:
            copy_box: The Box[] to copy.
        """
        self.__init__(copy_value=copy_box[])

    fn __moveinit__(inout self, owned existing: Self):
        """Move this Box[].

        Args:
            existing: The value to move.
        """
        self._inner = existing._inner
        existing._inner = UnsafePointer[T]()

    fn __getitem__(
        ref [_, AddressSpace.GENERIC._value.value]self
    ) -> ref [__lifetime_of(self._inner), AddressSpace.GENERIC._value.value] T:
        """Returns a reference to the box's underlying data with parametric mutability.

        Returns:
            A reference to the data underlying the Box[].
        """
        # This should have a widening conversion here that allows
        # the mutable ref that is always (potentially unsafely)
        # returned from UnsafePointer to be guarded behind the
        # aliasing guarantees of the lifetime system here.
        # All of the magic happens above in the function signature

        return self._inner[]

    fn __del__(owned self: Box[T]):
        """Destroy the Box[]."""
        self._inner.destroy_pointee()
        self._inner.free()

    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """UNSAFE: returns the backing pointer for this Box[].

        Returns:
            An UnsafePointer to the backing allocation for this Box[].
        """
        return self._inner

    fn take[T: Movable](owned self: Box[T]) -> T:
        """Move the value within the Box[] out of it, consuming the Box[] in the process.

        Parameters:
            T: The type of the data backing this Box[]. `take()` only exists for T: Movable
                since this consuming operation only makes sense for types that you want to avoid copying.
                For types that are Copy or ExplicitlyCopy but are not Movable, you can copy them through
                `__getitem__` as in `var v = some_box_var[]`.

        Returns:
            The data that is (was) backing the Box[].
        """
        var r = self._inner.take_pointee()
        self._inner.free()
        __mlir_op.`lit.ownership.mark_destroyed`(__get_mvalue_as_litref(self))

        return r^
