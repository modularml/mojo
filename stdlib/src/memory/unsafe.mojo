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
"""Implements types that work with unsafe pointers.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import Pointer
```
"""


from sys import (
    alignof,
    bitwidthof,
    simdwidthof,
    sizeof,
    triple_is_nvidia_cuda,
)
from sys.intrinsics import PrefetchOptions, _mlirtype_is_eq
from sys.intrinsics import prefetch as _prefetch
from sys.intrinsics import gather, scatter, strided_load, strided_store
from bit import is_power_of_two

from .memory import _free, _malloc
from .reference import AddressSpace

# ===----------------------------------------------------------------------===#
# bitcast
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bitcast[
    new_type: DType, new_width: Int, src_type: DType, src_width: Int
](val: SIMD[src_type, src_width]) -> SIMD[new_type, new_width]:
    """Bitcasts a SIMD value to another SIMD value.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        new_type: The target type.
        new_width: The target width.
        src_type: The source type.
        src_width: The source width.

    Args:
        val: The source value.

    Returns:
        A new SIMD value with the specified type and width with a bitcopy of the
        source SIMD value.
    """
    constrained[
        bitwidthof[SIMD[src_type, src_width]]()
        == bitwidthof[SIMD[new_type, new_width]](),
        "the source and destination types must have the same bitwidth",
    ]()

    @parameter
    if new_type == src_type:
        return rebind[SIMD[new_type, new_width]](val)
    return __mlir_op.`pop.bitcast`[
        _type = __mlir_type[
            `!pop.simd<`, new_width.value, `, `, new_type.value, `>`
        ]
    ](val.value)


@always_inline("nodebug")
fn bitcast[
    new_type: DType, src_type: DType
](val: SIMD[src_type, 1]) -> SIMD[new_type, 1]:
    """Bitcasts a SIMD value to another SIMD value.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        new_type: The target type.
        src_type: The source type.

    Args:
        val: The source value.

    Returns:
        A new SIMD value with the specified type and width with a bitcopy of the
        source SIMD value.
    """
    constrained[
        bitwidthof[SIMD[src_type, 1]]() == bitwidthof[SIMD[new_type, 1]](),
        "the source and destination types must have the same bitwidth",
    ]()

    return bitcast[new_type, 1, src_type, 1](val)


@always_inline("nodebug")
fn bitcast[
    new_type: DType, src_width: Int
](val: SIMD[DType.bool, src_width]) -> Scalar[new_type]:
    """Packs a SIMD bool into an integer.

    Constraints:
        The bitwidth of the two types must be the same.

    Parameters:
        new_type: The target type.
        src_width: The source width.

    Args:
        val: The source value.

    Returns:
        A new integer scalar which has the same bitwidth as the bool vector.
    """
    constrained[
        src_width == bitwidthof[Scalar[new_type]](),
        "the source and destination types must have the same bitwidth",
    ]()

    return __mlir_op.`pop.bitcast`[
        _type = __mlir_type[`!pop.scalar<`, new_type.value, `>`]
    ](val.value)


# ===----------------------------------------------------------------------===#
# LegacyPointer
# ===----------------------------------------------------------------------===#

alias Pointer = LegacyPointer


@value
@register_passable("trivial")
struct LegacyPointer[
    type: AnyTrivialRegType, address_space: AddressSpace = AddressSpace.GENERIC
](Boolable, CollectionElement, Intable, Stringable, EqualityComparable):
    """Defines a LegacyPointer struct that contains the address of a register passable
    type.

    Parameters:
        type: Type of the underlying data.
        address_space: The address space the pointer is in.
    """

    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`, type, `,`, address_space._value.value, `>`
    ]

    var address: Self._mlir_type
    """The pointed-to address."""

    alias _ref_type = Reference[type, MutableStaticLifetime, address_space]

    @always_inline("nodebug")
    fn __init__() -> Self:
        """Constructs a null LegacyPointer from the value of pop.pointer type.

        Returns:
            Constructed LegacyPointer object.
        """
        return __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @always_inline("nodebug")
    fn __init__(address: Self._mlir_type) -> Self:
        """Constructs a LegacyPointer from the address.

        Args:
            address: The input pointer address.

        Returns:
            Constructed LegacyPointer object.
        """
        return Self {address: address}

    @always_inline("nodebug")
    fn __init__(value: Scalar[DType.address]) -> Self:
        """Constructs a LegacyPointer from the value of scalar address.

        Args:
            value: The input pointer index.

        Returns:
            Constructed LegacyPointer object.
        """
        var address = __mlir_op.`pop.index_to_pointer`[_type = Self._mlir_type](
            value.cast[DType.index]().value
        )
        return Self {address: address}

    @always_inline("nodebug")
    fn __init__(*, address: Int) -> Self:
        """Constructs a Pointer from an address in an integer.

        Args:
            address: The input address.

        Returns:
            Constructed Pointer object.
        """
        return __mlir_op.`pop.index_to_pointer`[_type = Self._mlir_type](
            Scalar[DType.index](address).value
        )

    fn __str__(self) -> String:
        """Format this pointer as a hexadecimal string.

        Returns:
            A String containing the hexadecimal representation of the memory
            location destination of this pointer.
        """
        return hex(int(self))

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Checks if the LegacyPointer is null.

        Returns:
            Returns False if the LegacyPointer is null and True otherwise.
        """
        return self != Self()

    @staticmethod
    @always_inline("nodebug")
    fn address_of(ref [_, address_space._value.value]arg: type) -> Self:
        """Gets the address of the argument.

        Args:
            arg: The value to get the address of.

        Returns:
            A LegacyPointer struct which contains the address of the argument.
        """
        # Work around AnyTrivialRegType vs AnyType.
        return __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            UnsafePointer.address_of(arg).address
        )

    @always_inline("nodebug")
    fn __getitem__(
        self,
    ) -> ref [MutableStaticLifetime, address_space._value.value] type:
        """Enable subscript syntax `ptr[]` to access the element.

        Returns:
            The reference for the Mojo compiler to use.
        """
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type = Self._ref_type._mlir_type](
                self.address
            )
        )

    @always_inline("nodebug")
    fn __getitem__(
        self, offset: Int
    ) -> ref [MutableStaticLifetime, address_space._value.value] type:
        """Enable subscript syntax `ptr[idx]` to access the element.

        Args:
            offset: The offset to load from.

        Returns:
            The reference for the Mojo compiler to use.
        """
        return (self + offset)[]

    # ===------------------------------------------------------------------=== #
    # Load/Store
    # ===------------------------------------------------------------------=== #

    alias _default_alignment = alignof[type]() if triple_is_nvidia_cuda() else 1

    @always_inline("nodebug")
    fn load[*, alignment: Int = Self._default_alignment](self) -> type:
        """Loads the value the LegacyPointer object points to.

        Constraints:
            The alignment must be a positive integer value.

        Parameters:
            alignment: The minimal alignment of the address.

        Returns:
            The loaded value.
        """
        return self.load[alignment=alignment](0)

    @always_inline("nodebug")
    fn load[
        T: Intable, *, alignment: Int = Self._default_alignment
    ](self, offset: T) -> type:
        """Loads the value the LegacyPointer object points to with the given offset.

        Constraints:
            The alignment must be a positive integer value.

        Parameters:
            T: The Intable type of the offset.
            alignment: The minimal alignment of the address.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()
        return __mlir_op.`pop.load`[alignment = alignment.value](
            self.offset(offset).address
        )

    @always_inline("nodebug")
    fn store[
        T: Intable, /, *, alignment: Int = Self._default_alignment
    ](self, offset: T, value: type):
        """Stores the specified value to the location the LegacyPointer object points
        to with the given offset.

        Constraints:
            The alignment must be a positive integer value.

        Parameters:
            T: The Intable type of the offset.
            alignment: The minimal alignment of the address.

        Args:
            offset: The offset to store to.
            value: The value to store.
        """
        self.offset(offset).store[alignment=alignment](value)

    @always_inline("nodebug")
    fn store[*, alignment: Int = Self._default_alignment](self, value: type):
        """Stores the specified value to the location the LegacyPointer object points
        to.

        Constraints:
            The alignment value must be a positive integer.

        Parameters:
            alignment: The minimal alignment of the address.

        Args:
            value: The value to store.
        """
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()
        __mlir_op.`pop.store`[alignment = alignment.value](value, self.address)

    @always_inline("nodebug")
    fn nt_store(self, value: type):
        """Stores a value using non-temporal store.

        The address must be properly aligned, 64B for avx512, 32B for avx2, and
        16B for avx.

        Args:
            value: The value to store.
        """
        # Store a simd value into the pointer. The address must be properly
        # aligned, 64B for avx512, 32B for avx2, and 16B for avx.
        __mlir_op.`pop.store`[
            alignment = int(8 * simdwidthof[type]()).value,
            nonTemporal = __mlir_attr.unit,
        ](value, self.address)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return __mlir_op.`pop.pointer_to_index`[
            _type = __mlir_type.`!pop.scalar<index>`
        ](self.address)

    # ===------------------------------------------------------------------=== #
    # Allocate/Free
    # ===------------------------------------------------------------------=== #

    @staticmethod
    @always_inline
    fn alloc(count: Int, /, *, alignment: Int = alignof[type]()) -> Self:
        """Heap-allocates a number of element of the specified type using
        the specified alignment.

        Args:
            count: The number of elements to allocate (note that this is not
              the bytecount).
            alignment: The alignment used for the allocation.

        Returns:
            A new LegacyPointer object which has been allocated on the heap.
        """
        return _malloc[type, address_space=address_space](
            count * sizeof[type](), alignment=alignment
        )

    @always_inline
    fn free(self):
        """Frees the heap allocated memory."""
        return _free(self)

    # ===------------------------------------------------------------------=== #
    # Casting
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn bitcast[
        new_type: AnyTrivialRegType = type,
        /,
        address_space: AddressSpace = Self.address_space,
    ](self) -> LegacyPointer[new_type, address_space]:
        """Bitcasts a LegacyPointer to a different type.

        Parameters:
            new_type: The target type.
            address_space: The address space of the result.

        Returns:
            A new LegacyPointer object with the specified type and the same address,
            as the original LegacyPointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = LegacyPointer[new_type, address_space]._mlir_type,
        ](self.address)

    # ===------------------------------------------------------------------=== #
    # Comparisons
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return int(self) == int(rhs)

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return int(self) != int(rhs)

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.


        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return int(self) < int(rhs)

    # ===------------------------------------------------------------------=== #
    # Pointer Arithmetic
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn offset[T: Intable](self, idx: T) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            idx: The offset.

        Returns:
            The new LegacyPointer shifted by the offset.
        """
        # Returns a new pointer shifted by the specified offset.
        return __mlir_op.`pop.offset`(self.address, int(idx).value)

    @always_inline("nodebug")
    fn __add__[T: Intable](self, rhs: T) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.

        Returns:
            The new LegacyPointer shifted by the offset.
        """
        return self.offset(rhs)

    @always_inline("nodebug")
    fn __sub__[T: Intable](self, rhs: T) -> Self:
        """Returns a new pointer shifted back by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.

        Returns:
            The new LegacyPointer shifted back by the offset.
        """
        return self.offset(-int(rhs))

    @always_inline("nodebug")
    fn __iadd__[T: Intable](inout self, rhs: T):
        """Shifts the current pointer by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__[T: Intable](inout self, rhs: T):
        """Shifts back the current pointer by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.
        """
        self = self - rhs


# ===----------------------------------------------------------------------===#
# DTypePointer
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct DTypePointer[
    type: DType, address_space: AddressSpace = AddressSpace.GENERIC
](Boolable, CollectionElement, Intable, Stringable, EqualityComparable):
    """Defines a `DTypePointer` struct that contains an address of the given
    dtype.

    Parameters:
        type: DType of the underlying data.
        address_space: The address space the pointer is in.
    """

    alias element_type = Scalar[type]
    alias _pointer_type = Pointer[Scalar[type], address_space]
    var address: Self._pointer_type
    """The pointed-to address."""

    @always_inline("nodebug")
    fn __init__(inout self):
        """Constructs a null `DTypePointer` from the given type."""

        self.address = Self._pointer_type()

    @always_inline("nodebug")
    fn __init__(
        inout self,
        value: __mlir_type[
            `!kgen.pointer<scalar<`,
            type.value,
            `>,`,
            address_space._value.value,
            `>`,
        ],
    ):
        """Constructs a `DTypePointer` from a scalar pointer of the same type.

        Args:
            value: The scalar pointer.
        """
        self = Pointer[
            __mlir_type[`!pop.scalar<`, type.value, `>`], address_space
        ](value).bitcast[Scalar[type]]()

    @always_inline("nodebug")
    fn __init__(inout self, value: Pointer[Scalar[type], address_space]):
        """Constructs a `DTypePointer` from a scalar pointer of the same type.

        Args:
            value: The scalar pointer.
        """
        self.address = value

    @always_inline("nodebug")
    fn __init__(inout self, other: UnsafePointer[Scalar[type], address_space]):
        """Constructs a `DTypePointer` from a scalar pointer of the same type.

        Args:
            other: The scalar pointer.
        """
        self.address = other.address

    @always_inline("nodebug")
    fn __init__(inout self, value: Scalar[DType.address]):
        """Constructs a `DTypePointer` from the value of scalar address.

        Args:
            value: The input pointer index.
        """
        var address = __mlir_op.`pop.index_to_pointer`[
            _type = Self._pointer_type._mlir_type
        ](value.cast[DType.index]().value)
        self.address = address

    @always_inline
    fn __init__(inout self, *, address: Int):
        """Constructs a `DTypePointer` from an integer address.

        Args:
            address: The input address.
        """
        self.address = Self._pointer_type(address=address)

    fn __str__(self) -> String:
        """Format this pointer as a hexadecimal string.

        Returns:
            A String containing the hexadecimal representation of the memory location
            destination of this pointer.
        """
        return str(self.address)

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Checks if the DTypePointer is *null*.

        Returns:
            Returns False if the DTypePointer is *null* and True otherwise.
        """
        return self.address.__bool__()

    @staticmethod
    @always_inline("nodebug")
    fn address_of(ref [_, address_space._value.value]arg: Scalar[type]) -> Self:
        """Gets the address of the argument.

        Args:
            arg: The value to get the address of.

        Returns:
            A DTypePointer struct which contains the address of the argument.
        """
        return LegacyPointer.address_of(arg)

    @always_inline("nodebug")
    fn __getitem__(self, offset: Int) -> Scalar[type]:
        """Loads a single element (SIMD of size 1) from the pointer at the
        specified index.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.load(offset)

    @always_inline("nodebug")
    fn __setitem__(self, offset: Int, val: Scalar[type]):
        """Stores a single element value at the given offset.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        return self.store(offset, val)

    # ===------------------------------------------------------------------=== #
    # Comparisons
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.address == rhs.address

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.address != rhs.address

    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return self.address < rhs.address

    # ===------------------------------------------------------------------=== #
    # Allocate/Free
    # ===------------------------------------------------------------------=== #

    @staticmethod
    @always_inline
    fn alloc(count: Int, /, *, alignment: Int = alignof[type]()) -> Self:
        """Heap-allocates a number of element of the specified type using
        the specified alignment.

        Args:
            count: The number of elements to allocate (note that this is not
              the bytecount).
            alignment: The alignment used for the allocation.

        Returns:
            A new `DTypePointer` object which has been allocated on the heap.
        """
        return _malloc[Self.element_type, address_space=address_space](
            count * sizeof[type](), alignment=alignment
        )

    @always_inline
    fn free(self):
        """Frees the heap allocates memory."""
        _free(self)

    # ===------------------------------------------------------------------=== #
    # Casting
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn bitcast[
        new_type: DType = type,
        /,
        address_space: AddressSpace = Self.address_space,
    ](self) -> DTypePointer[new_type, address_space]:
        """Bitcasts `DTypePointer` to a different dtype.

        Parameters:
            new_type: The target dtype.
            address_space: The address space of the result.

        Returns:
            A new `DTypePointer` object with the specified dtype and the same
            address, as the original `DTypePointer`.
        """
        return self.address.bitcast[SIMD[new_type, 1], address_space]()

    @always_inline("nodebug")
    fn _as_scalar_pointer(self) -> Pointer[Scalar[type], address_space]:
        """Converts the `DTypePointer` to a scalar pointer of the same dtype.

        Returns:
            A `Pointer` to a scalar of the same dtype.
        """
        return self.address

    # ===------------------------------------------------------------------=== #
    # Load/Store
    # ===------------------------------------------------------------------=== #

    alias _default_alignment = alignof[
        Scalar[type]
    ]() if triple_is_nvidia_cuda() else 1

    @always_inline
    fn prefetch[params: PrefetchOptions](self):
        # Prefetch at the underlying address.
        """Prefetches memory at the underlying address.

        Parameters:
            params: Prefetch options (see `PrefetchOptions` for details).
        """
        _prefetch[params](self)

    @always_inline("nodebug")
    fn load[
        *, width: Int = 1, alignment: Int = Self._default_alignment
    ](self) -> SIMD[type, width]:
        """Loads the value the Pointer object points to.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Returns:
            The loaded value.
        """
        return self.load[width=width, alignment=alignment](0)

    @always_inline("nodebug")
    fn load[
        T: Intable, *, width: Int = 1, alignment: Int = Self._default_alignment
    ](self, offset: T) -> SIMD[type, width]:
        """Loads the value the Pointer object points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            T: The Intable type of the offset.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """

        @parameter
        if triple_is_nvidia_cuda() and sizeof[type]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[type, width]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(width):
                v[i] = self.address.offset(int(offset) + i).load[
                    alignment=alignment
                ]()
            return v

        return (
            self.address.offset(offset)
            .bitcast[SIMD[type, width]]()
            .load[alignment=alignment]()
        )

    @always_inline("nodebug")
    fn store[
        T: Intable,
        /,
        *,
        width: Int = 1,
        alignment: Int = Self._default_alignment,
    ](self, offset: T, val: SIMD[type, width]):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            T: The Intable type of the offset.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        self.offset(offset).store[width=width, alignment=alignment](val)

    @always_inline("nodebug")
    fn store[
        *, width: Int = 1, alignment: Int = Self._default_alignment
    ](self, val: SIMD[type, width]):
        """Stores a single element value.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            val: The value to store.
        """
        constrained[width > 0, "width must be a positive integer value"]()
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()
        self.address.bitcast[SIMD[type, width]]().store[alignment=alignment](
            val
        )

    @always_inline("nodebug")
    fn simd_nt_store[
        width: Int, T: Intable
    ](self, offset: T, val: SIMD[type, width]):
        """Stores a SIMD vector using non-temporal store.

        Parameters:
            width: The SIMD width.
            T: The Intable type of the offset.

        Args:
            offset: The offset to store to.
            val: The SIMD value to store.
        """
        self.offset(offset).simd_nt_store[width](val)

    @always_inline("nodebug")
    fn simd_strided_load[
        width: Int, T: Intable
    ](self, stride: T) -> SIMD[type, width]:
        """Performs a strided load of the SIMD vector.

        Parameters:
            width: The SIMD width.
            T: The Intable type of the stride.

        Args:
            stride: The stride between loads.

        Returns:
            A vector which is stride loaded.
        """
        return strided_load[type, width](
            self, int(stride), SIMD[DType.bool, width](1)
        )

    @always_inline("nodebug")
    fn simd_strided_store[
        width: Int, T: Intable
    ](self, val: SIMD[type, width], stride: T):
        """Performs a strided store of the SIMD vector.

        Parameters:
            width: The SIMD width.
            T: The Intable type of the stride.

        Args:
            val: The SIMD value to store.
            stride: The stride between stores.
        """
        strided_store(val, self, int(stride), True)

    @always_inline("nodebug")
    fn simd_nt_store[width: Int](self, val: SIMD[type, width]):
        """Stores a SIMD vector using non-temporal store.

        The address must be properly aligned, 64B for avx512, 32B for avx2, and
        16B for avx.

        Parameters:
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
        """
        # Store a simd value into the pointer. The address must be properly
        # aligned, 64B for avx512, 32B for avx2, and 16B for avx.
        self.address.bitcast[SIMD[type, width]]().nt_store(val)

    # ===------------------------------------------------------------------=== #
    # Gather/Scatter
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn gather[
        *, width: Int = 1, alignment: Int = Self._default_alignment
    ](self, offset: SIMD[_, width]) -> SIMD[type, width]:
        """Gathers a SIMD vector from offsets of the current pointer.

        This method loads from memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to gather from.

        Returns:
            The SIMD vector containing the gathered values.
        """
        var mask = SIMD[DType.bool, width](True)
        var default = SIMD[type, width]()
        return self.gather[width=width, alignment=alignment](
            offset, mask, default
        )

    @always_inline("nodebug")
    fn gather[
        *, width: Int = 1, alignment: Int = Self._default_alignment
    ](
        self,
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width],
        default: SIMD[type, width],
    ) -> SIMD[type, width]:
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
        """
        constrained[
            offset.type.is_integral(),
            "offset type must be an integral type",
        ]()
        constrained[
            is_power_of_two(alignment),
            "alignment must be a power of two integer value",
        ]()

        var base = offset.cast[DType.index]().fma(sizeof[type](), int(self))
        return gather(base.cast[DType.address](), mask, default, alignment)

    @always_inline("nodebug")
    fn scatter[
        *, width: Int = 1, alignment: Int = Self._default_alignment
    ](self, offset: SIMD[_, width], val: SIMD[type, width]):
        """Scatters a SIMD vector into offsets of the current pointer.

        This method stores at memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector.

        If the same offset is targeted multiple times, the values are stored
        in the order they appear in the `val` SIMD vector, from the first to
        the last element.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to scatter into.
            val: The SIMD vector containing the values to be scattered.
        """
        var mask = SIMD[DType.bool, width](True)
        self.scatter[width=width, alignment=alignment](offset, val, mask)

    @always_inline("nodebug")
    fn scatter[
        *, width: Int = 1, alignment: Int = Self._default_alignment
    ](
        self,
        offset: SIMD[_, width],
        val: SIMD[type, width],
        mask: SIMD[DType.bool, width],
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
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to scatter into.
            val: The SIMD vector containing the values to be scattered.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to store at memory or not.
        """
        constrained[
            offset.type.is_integral(),
            "offset type must be an integral type",
        ]()
        constrained[
            is_power_of_two(alignment),
            "alignment must be a power of two integer value",
        ]()

        var base = offset.cast[DType.index]().fma(sizeof[type](), int(self))
        scatter(val, base.cast[DType.address](), mask, alignment)

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return int(self.address)

    @always_inline
    fn is_aligned[alignment: Int](self) -> Bool:
        """Checks if the pointer is aligned.

        Parameters:
            alignment: The minimal desired alignment.

        Returns:
            `True` if the pointer is at least `alignment`-aligned or `False`
            otherwise.
        """
        constrained[
            is_power_of_two(alignment), "alignment must be a power of 2."
        ]()
        return int(self) % alignment == 0

    # ===------------------------------------------------------------------=== #
    # Pointer Arithmetic
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn offset[T: Intable](self, idx: T) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed DTypePointer.
        """
        return self.address.offset(idx)

    @always_inline("nodebug")
    fn __add__[T: Intable](self, rhs: T) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.

        Returns:
            The new DTypePointer shifted by the offset.
        """
        return self.offset(rhs)

    @always_inline("nodebug")
    fn __sub__[T: Intable](self, rhs: T) -> Self:
        """Returns a new pointer shifted back by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.

        Returns:
            The new DTypePointer shifted by the offset.
        """
        return self.offset(-int(rhs))

    @always_inline("nodebug")
    fn __iadd__[T: Intable](inout self, rhs: T):
        """Shifts the current pointer by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__[T: Intable](inout self, rhs: T):
        """Shifts back the current pointer by the specified offset.

        Parameters:
            T: The Intable type of the offset.

        Args:
            rhs: The offset.
        """
        self = self - rhs
