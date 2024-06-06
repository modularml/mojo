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


struct UnsafeMaybeUninitialized[ElementType: CollectionElement](
    CollectionElement
):
    """A memory location that may or may not be initialized.

    Note that the destructor is a no-op. If the memory was initialized, the caller
    is responsible for calling `assume_initialized_destroy` before the memory is
    deallocated.

    Every method in this struct is unsafe and the caller must know at all
    times if the memory is initialized or not. Calling a method
    that assumes the memory is initialized when it is not will result in
    undefined behavior.

    Parameters:
        ElementType: The type of the element to store.
    """

    alias type = __mlir_type[`!pop.array<1, `, Self.ElementType, `>`]
    var _array: Self.type

    @always_inline
    fn __init__(inout self):
        """The memory is now considered uninitialized."""
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

    @always_inline
    fn __init__(inout self, owned value: Self.ElementType):
        """The memory is now considered initialized.

        Args:
            value: The value to initialize the memory with.
        """
        self = Self()
        self.write(value^)

    @always_inline
    fn __copyinit__(inout self, other: Self):
        """Copy another object.

        This method should never be called as implicit copy should not
        be done on memory that may be uninitialized.

        Calling this method will actally cause the program to abort.

        If you wish to perform a copy, you should manually call the method
        `copy_from` instead.

        Args:
            other: The object to copy.
        """
        abort("You should never call __copyinit__ on MaybeUninitialized")
        self = Self()

    @always_inline
    fn copy_from(inout self, other: Self):
        """Copy another object.

        This function assumes that the current memory is uninitialized
        and the other object is initialized memory.

        Args:
            other: The object to copy.
        """
        self.unsafe_ptr().init_pointee_copy(other.assume_initialized())

    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        """Move another object.

        This method should never be called as implicit moves should not
        be done on memory that may be uninitialized.

        Calling this method will actally cause the program to abort.

        If you wish to perform a move, you should manually call the method
        `move_from` instead.

        Args:
            other: The object to move.
        """
        abort("You should never call __moveinit__ on MaybeUninitialized")
        self = Self()

    @always_inline
    fn move_from(inout self, inout other: Self):
        """Move another object.

        This function assumes that the current memory is uninitialized
        and the other object is initialized memory.

        After the function is called, the other object is considered uninitialized.

        Args:
            other: The object to move.
        """
        other.unsafe_ptr().move_pointee_into(self.unsafe_ptr())

    @always_inline
    fn write(inout self, owned value: Self.ElementType):
        """Write a value into an uninitialized memory location.

        Calling this method assumes that the memory is uninitialized.

        Args:
            value: The value to write.
        """
        self.unsafe_ptr().init_pointee_move(value^)

    @always_inline
    fn assume_initialized(
        ref [_]self: Self,
    ) -> ref [__lifetime_of(self)] Self.ElementType:
        """Returns a reference to the internal value.

        Calling this method assumes that the memory is initialized.

        Returns:
            A reference to the internal value.
        """
        return self.unsafe_ptr()[]

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Self.ElementType]:
        """Get a pointer to the underlying element.

        Note that this method does not assumes that the memory is initialized
        or not. It can always be called.

        Returns:
            A pointer to the underlying element.
        """
        return UnsafePointer.address_of(self._array).bitcast[Self.ElementType]()

    @always_inline
    fn assume_initialized_destroy(inout self):
        """Runs the destructor of the internal value.

        Calling this method assumes that the memory is initialized.

        """
        self.unsafe_ptr().destroy_pointee()

    @always_inline
    fn __del__(owned self):
        """This is a no-op.

        Calling this method assumes that the memory is uninitialized.
        If the memory was initialized, the caller should
        use `assume_initialized_destroy` before.
        """
        pass
