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

from memory.unsafe_pointer import initialize_pointee
from utils._visualizers import lldb_formatter_wrapping_type

# ===----------------------------------------------------------------------===#
# Tuple
# ===----------------------------------------------------------------------===#


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
        +element_types,
        `>`,
    ]

    var storage: Self._mlir_type
    """The underlying storage for the tuple."""

    @always_inline("nodebug")
    fn __init__(inout self, *args: *element_types):
        """Construct the tuple.

        Args:
            args: Initial values.
        """
        self = Self(storage=args)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        *,
        storage: VariadicPack[_, _, CollectionElement, element_types],
    ):
        """Construct the tuple from a low-level internal representation.

        Args:
            storage: The variadic pack storage to construct from.
        """
        # Mark 'storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        fn initialize_elt[idx: Int]():
            # TODO: We could be fancier and take the values out of an owned
            # pack. For now just keep everything simple and copy the element.
            initialize_pointee(
                UnsafePointer(self._refitem__[idx]()),
                storage.get_element[idx]()[],
            )

        unroll[initialize_elt, Self.__len__()]()

    fn __del__(owned self):
        """Destructor that destroys all of the elements."""

        # Run the destructor on each member, the destructor of !kgen.pack is
        # trivial and won't do anything.
        @parameter
        fn destroy_elt[idx: Int]():
            destroy_pointee(UnsafePointer(self._refitem__[idx]()))

        unroll[destroy_elt, Self.__len__()]()

    @always_inline("nodebug")
    fn __copyinit__(inout self, existing: Self):
        """Copy construct the tuple.

        Args:
            existing: The value to copy from.
        """
        # Mark 'storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        fn initialize_elt[idx: Int]():
            var existing_elt_ptr = UnsafePointer(existing._refitem__[idx]())

            initialize_pointee(
                UnsafePointer(self._refitem__[idx]()),
                __get_address_as_owned_value(existing_elt_ptr.value),
            )

        unroll[initialize_elt, Self.__len__()]()

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        """Move construct the tuple.

        Args:
            existing: The value to move from.
        """
        # Mark 'storage' as being initialized so we can work on it.
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self.storage)
        )

        @parameter
        fn initialize_elt[idx: Int]():
            initialize_pointee(
                UnsafePointer(self._refitem__[idx]()),
                existing._refitem__[idx]()[],
            )

        unroll[initialize_elt, Self.__len__()]()

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

    # TODO: Mojo's small brain can't handle a __refitem__ like this yet.
    @always_inline("nodebug")
    fn _refitem__[
        idx: Int,
        mutability: __mlir_type.i1,
        self_life: AnyLifetime[mutability].type,
    ](self_lit: Reference[Self, mutability, self_life]._mlir_type) -> Reference[
        element_types[idx.value], mutability, self_life
    ]:
        # Return a reference to an element at the specified index, propagating
        # mutability of self.
        var storage_kgen_ptr = Reference(
            Reference(self_lit)[].storage
        ).get_legacy_pointer().address

        # Pointer to the element.
        var elt_kgen_ptr = __mlir_op.`kgen.pack.gep`[index = idx.value](
            storage_kgen_ptr
        )
        # Convert to an immortal mut reference, which conforms to self_life.
        return UnsafePointer(elt_kgen_ptr)[]

    # TODO: Remove the get methods in favor of __refitem__ some day.  This will
    # be annoying if we don't have autoderef though.
    @always_inline("nodebug")
    fn get[i: Int, T: CollectionElement](self) -> T:
        """Get a tuple element and rebind to the specified type.

        Parameters:
            i: The element index.
            T: The element type.

        Returns:
            The tuple element at the requested index.
        """
        return rebind[T](self.get[i]())

    @always_inline("nodebug")
    fn get[i: Int](self) -> element_types[i.value]:
        """Get a tuple element.

        Parameters:
            i: The element index.

        Returns:
            The tuple element at the requested index.
        """
        return self._refitem__[i]()[]

    @staticmethod
    fn _offset[i: Int]() -> Int:
        constrained[i >= 0, "index must be positive"]()

        @parameter
        if i == 0:
            return 0
        else:
            return _align_up(
                Self._offset[i - 1]()
                + _align_up(
                    sizeof[element_types[i - 1]](),
                    alignof[element_types[i - 1]](),
                ),
                alignof[element_types[i]](),
            )


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@always_inline
fn _align_up(value: Int, alignment: Int) -> Int:
    var div_ceil = (value + alignment - 1)._positive_div(alignment)
    return div_ceil * alignment
