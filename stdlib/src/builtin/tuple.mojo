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
"""Implements the Tuple type.

These are Mojo built-ins, so you don't need to import them.
"""

from sys.intrinsics import _type_is_eq

from memory import UnsafePointer

from utils._visualizers import lldb_formatter_wrapping_type

# ===-----------------------------------------------------------------------===#
# Tuple
# ===-----------------------------------------------------------------------===#


@lldb_formatter_wrapping_type
struct Tuple[*element_types: CollectionElement](Sized, CollectionElement):
    """The type of a literal tuple expression.

    A tuple consists of zero or more values, separated by commas.

    Parameters:
        element_types: The elements type.
    """

    alias _mlir_type = __mlir_type[
        `!kgen.pack<:!kgen.variadic<`,
        CollectionElement,
        `> `,
        element_types,
        `>`,
    ]

    var storage: Self._mlir_type
    """The underlying storage for the tuple."""

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, owned *args: *element_types):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self = Self(storage=args^)

    @always_inline("nodebug")
    fn __init__(
        mut self,
        *,
        owned storage: VariadicPack[_, CollectionElement, *element_types],
    ):
        """Construct the tuple from a low-level internal representation.

        Args:
            storage: The variadic pack storage to construct from.
        """

        # Mark 'self.storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        # Move each element into the tuple storage.
        @parameter
        for i in range(Self.__len__()):
            UnsafePointer.address_of(storage[i]).move_pointee_into(
                UnsafePointer.address_of(self[i])
            )

        # Do not destroy the elements when 'storage' goes away.
        __mlir_op.`lit.ownership.mark_destroyed`(
            __get_mvalue_as_litref(storage)
        )

    fn __del__(owned self):
        """Destructor that destroys all of the elements."""

        # Run the destructor on each member, the destructor of !kgen.pack is
        # trivial and won't do anything.
        @parameter
        for i in range(Self.__len__()):
            UnsafePointer.address_of(self[i]).destroy_pointee()

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Copy construct the tuple.

        Args:
            existing: The value to copy from.
        """
        # Mark 'storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        for i in range(Self.__len__()):
            UnsafePointer.address_of(self[i]).init_pointee_copy(existing[i])

    @always_inline("nodebug")
    fn __moveinit__(out self, owned existing: Self):
        """Move construct the tuple.

        Args:
            existing: The value to move from.
        """
        # Mark 'storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        for i in range(Self.__len__()):
            UnsafePointer.address_of(existing[i]).move_pointee_into(
                UnsafePointer.address_of(self[i])
            )
        # Note: The destructor on `existing` is auto-disabled in a moveinit.

    @always_inline
    @staticmethod
    fn __len__() -> Int:
        """Return the number of elements in the tuple.

        Returns:
            The tuple length.
        """

        @parameter
        fn variadic_size(
            x: __mlir_type[`!kgen.variadic<`, CollectionElement, `>`]
        ) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        alias result = variadic_size(element_types)
        return result

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of elements in the tuple.

        Returns:
            The tuple length.
        """
        return Self.__len__()

    @always_inline("nodebug")
    fn __getitem__[idx: Int](ref self) -> ref [self] element_types[idx.value]:
        """Get a reference to an element in the tuple.

        Parameters:
            idx: The element to return.

        Returns:
            A reference to the specified element.
        """
        # Return a reference to an element at the specified index, propagating
        # mutability of self.
        var storage_kgen_ptr = UnsafePointer.address_of(self.storage).address

        # KGenPointer to the element.
        var elt_kgen_ptr = __mlir_op.`kgen.pack.gep`[index = idx.value](
            storage_kgen_ptr
        )
        # Use an immortal mut reference, which converts to self's origin.
        return UnsafePointer(elt_kgen_ptr)[]

    # TODO(#38268): Remove this method when references and parameter expressions
    # cooperate better.  We can't handle the use in test_simd without this.
    @always_inline("nodebug")
    fn get[i: Int, T: CollectionElement](ref self) -> ref [self] T:
        """Get a tuple element and rebind to the specified type.

        Parameters:
            i: The element index.
            T: The element type.

        Returns:
            The tuple element at the requested index.
        """
        return rebind[T](self[i])

    @always_inline("nodebug")
    fn __contains__[
        T: EqualityComparableCollectionElement
    ](self, value: T) -> Bool:
        """Return whether the tuple contains the specified value.

        For example:

        ```mojo
        var t = Tuple(True, 1, 2.5)
        if 1 in t:
            print("t contains 1")
        ```

        Args:
            value: The value to search for.

        Parameters:
            T: The type of the value.

        Returns:
            True if the value is in the tuple, False otherwise.
        """

        @parameter
        for i in range(len(VariadicList(element_types))):

            @parameter
            if _type_is_eq[element_types[i], T]():
                if self.get[i, T]() == value:
                    return True

        return False
