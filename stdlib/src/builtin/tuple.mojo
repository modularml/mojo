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

from utils._visualizers import lldb_formatter_wrapping_type

from memory.unsafe_pointer import (
    initialize_pointee_move,
    initialize_pointee_copy,
)

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
            initialize_pointee_copy(
                UnsafePointer(self[idx]),
                storage.get_element[idx]()[],
            )

        unroll[initialize_elt, Self.__len__()]()

    fn __del__(owned self):
        """Destructor that destroys all of the elements."""

        # Run the destructor on each member, the destructor of !kgen.pack is
        # trivial and won't do anything.
        @parameter
        fn destroy_elt[idx: Int]():
            destroy_pointee(UnsafePointer(self[idx]))

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
            initialize_pointee_copy(UnsafePointer(self[idx]), existing[idx])

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
            var existing_elt_ptr = UnsafePointer(existing[idx]).address
            initialize_pointee_move(
                UnsafePointer(self[idx]),
                __get_address_as_owned_value(existing_elt_ptr),
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

    @always_inline("nodebug")
    fn __refitem__[
        idx: Int,
        mutability: __mlir_type.i1,
        self_life: AnyLifetime[mutability].type,
    ](self_lit: Reference[Self, mutability, self_life]._mlir_type) -> Reference[
        element_types[idx.value], mutability, self_life
    ]:
        # Return a reference to an element at the specified index, propagating
        # mutability of self.
        var storage_kgen_ptr = UnsafePointer.address_of(
            Reference(self_lit)[].storage
        ).address

        # KGenPointer to the element.
        var elt_kgen_ptr = __mlir_op.`kgen.pack.gep`[index = idx.value](
            storage_kgen_ptr
        )
        # Convert to an immortal mut reference, which conforms to self_life.
        return UnsafePointer(elt_kgen_ptr)[]

    # TODO(#38268): Remove this method when references and parameter expressions
    # cooperate better.  We can't handle the use in test_simd without this.
    @always_inline("nodebug")
    fn get[i: Int, T: CollectionElement](self) -> T:
        """Get a tuple element and rebind to the specified type.

        Parameters:
            i: The element index.
            T: The element type.

        Returns:
            The tuple element at the requested index.
        """
        return rebind[T](self[i])
