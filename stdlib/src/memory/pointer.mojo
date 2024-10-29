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
"""Implements the Pointer type.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import Pointer
```
"""

from documentation import doc_private
from collections import Optional
from .unsafe_pointer import _default_alignment

# ===----------------------------------------------------------------------===#
# AddressSpace
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _GPUAddressSpace(EqualityComparable):
    var _value: Int

    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    alias GENERIC = AddressSpace(0)
    """Generic address space."""
    alias GLOBAL = AddressSpace(1)
    """Global address space."""
    alias CONSTANT = AddressSpace(2)
    """Constant address space."""
    alias SHARED = AddressSpace(3)
    """Shared address space."""
    alias PARAM = AddressSpace(4)
    """Param address space."""
    alias LOCAL = AddressSpace(5)
    """Local address space."""

    @always_inline("nodebug")
    fn __init__(inout self, value: Int):
        self._value = value

    @always_inline("nodebug")
    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """The True if the two address spaces are equal and False otherwise.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self is other

    @always_inline("nodebug")
    fn __eq__(self, other: AddressSpace) -> Bool:
        """The True if the two address spaces are equal and False otherwise.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self is other

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """True if the two address spaces are inequal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are inequal and False otherwise.
        """
        return self is not other

    @always_inline("nodebug")
    fn __ne__(self, other: AddressSpace) -> Bool:
        """True if the two address spaces are inequal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are inequal and False otherwise.
        """
        return self is not other

    @always_inline("nodebug")
    fn __is__(self, other: Self) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() == other.value()

    @always_inline("nodebug")
    fn __is__(self, other: AddressSpace) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() == other.value()

    @always_inline("nodebug")
    fn __isnot__(self, other: Self) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() != other.value()

    @always_inline("nodebug")
    fn __isnot__(self, other: AddressSpace) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() != other.value()


@value
@register_passable("trivial")
struct AddressSpace(EqualityComparable):
    """Address space of the pointer."""

    var _value: Int

    alias GENERIC = AddressSpace(0)
    """Generic address space."""

    @always_inline("nodebug")
    fn __init__(inout self, value: Int):
        """Initializes the address space from the underlying integral value.

        Args:
            value: The address space value.
        """
        self._value = value

    @always_inline("nodebug")
    fn __init__(inout self, value: _GPUAddressSpace):
        """Initializes the address space from the underlying integral value.

        Args:
          value: The address space value.
        """
        self._value = int(value)

    @always_inline("nodebug")
    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self._value.value

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self is other

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """True if the two address spaces are inequal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are inequal and False otherwise.
        """
        return self is not other

    @always_inline("nodebug")
    fn __is__(self, other: Self) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() == other.value()

    @always_inline("nodebug")
    fn __isnot__(self, other: Self) -> Bool:
        """True if the two address spaces are equal and False otherwise.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() != other.value()


# ===----------------------------------------------------------------------===#
# Pointer
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Pointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
](CollectionElementNew, Stringable):
    """Defines a non-nullable safe pointer.

    Parameters:
        is_mutable: Whether the pointee data may be mutated through this.
        type: Type of the underlying data.
        origin: The origin of the pointer.
        address_space: The address space of the pointee data.
    """

    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        type,
        `, `,
        origin,
        `, `,
        address_space._value.value,
        `>`,
    ]

    var _mlir_value: Self._mlir_type
    """The underlying MLIR representation."""
    var _flags: UInt8
    """Bitwise flags for the pointer.
    
    #### Bits:

    - 0: is_register_passable.
    - 1: is_allocated.
    - 2: unset.
    - 3: unset.
    - 4: unset.
    - 5: unset.
    - 6: unset.
    - 7: unset.
    """

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @doc_private
    @always_inline("nodebug")
    fn __init__(
        inout self,
        *,
        _mlir_value: Self._mlir_type,
        is_allocated: Bool,
        register_passable: Bool = False,
    ):
        """Constructs a Pointer from its MLIR prepresentation.

        Args:
            _mlir_value: The MLIR representation of the pointer.
            is_allocated: Whether the pointer's memory is allocated.
            register_passable: Whether the pointer is allocated on registers.
        """
        self._mlir_value = _mlir_value
        self._flags = (UInt8(register_passable) << 7) | (
            UInt8(is_allocated) << 6
        )

    @staticmethod
    @always_inline("nodebug")
    fn address_of(ref [origin, address_space._value.value]value: type) -> Self:
        """Constructs a Pointer from a reference to a value.

        Args:
            value: The value to get the address of.

        Returns:
            The result Pointer.
        """
        # TODO(#3581): this should make register_passable = is_trivial(type)
        return Pointer(
            _mlir_value=__get_mvalue_as_litref(value), is_allocated=True
        )

    fn __init__(inout self, *, other: Self):
        """Constructs a copy from another Pointer **(not the data)**.

        Args:
            other: The `Pointer` to copy.
        """
        self._mlir_value = other._mlir_value
        self._flags = other._flags

    @doc_private
    @always_inline("nodebug")
    fn __init__[
        O: MutableOrigin
    ](
        inout self: Pointer[type, O, address_space],
        *,
        unsafe_ptr: UnsafePointer[type, address_space, _, O],
        is_allocated: Bool = True,
        register_passable: Bool = False,
    ):
        """Constructs a Pointer from its MLIR prepresentation.

        Args:
            unsafe_ptr: The UnsafePointer.
            is_allocated: Whether the pointer's memory is allocated.
            register_passable: Whether the pointer is allocated on registers.
        """
        self = __type_of(self)(
            _mlir_value=__mlir_op.`lit.ref.from_pointer`[
                _type = __type_of(self)._mlir_type
            ](unsafe_ptr.address),
            is_allocated=is_allocated,
            register_passable=register_passable,
        )

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [origin, address_space._value.value] type:
        """Enable subscript syntax `ptr[]` to access the element.

        Returns:
            A reference to the underlying value in memory.
        """
        return __get_litref_as_mvalue(self._mlir_value)

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Pointer[type, _, address_space]) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return UnsafePointer.address_of(self[]) == UnsafePointer.address_of(
            rhs[]
        )

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Pointer[type, _, address_space]) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the Pointer.

        Returns:
            The string representation of the Pointer.
        """
        return str(UnsafePointer.address_of(self[]))

    @staticmethod
    @always_inline
    fn alloc[O: MutableOrigin](count: Int) -> Pointer[type, O, address_space]:
        """Allocate an array with specified or default alignment.

        Parameters:
            O: The origin of the Pointer.

        Args:
            count: The number of elements in the array.

        Returns:
            The pointer to the newly allocated array.
        """
        return Pointer[type, O, address_space](
            unsafe_ptr=UnsafePointer[type, address_space].alloc(count),
            is_allocated=True,
        )

    @staticmethod
    @always_inline
    fn alloc[
        count: Int,
        /,
        O: MutableOrigin,
        *,
        stack_alloc_limit: Int = 1 * 2**20,
        name: Optional[StringLiteral] = None,
    ]() -> Pointer[type, O, address_space]:
        """Allocate an array on the stack with specified or default alignment.

        Parameters:
            count: The number of elements in the array.
            O: The origin of the Pointer.
            stack_alloc_limit: The limit of bytes to allocate on the stack
                (default 1 MiB).
            name: The name of the global variable (only honored in certain
                cases).

        Returns:
            The pointer to the newly allocated array.
        """
        return Pointer[type, O, address_space](
            unsafe_ptr=UnsafePointer[type, address_space].alloc[count](),
            is_allocated=True,
            register_passable=True,
        )

    fn unsafe_free[
        O: MutableOrigin
    ](inout self: Pointer[type, O, address_space]):
        """Free the memory referenced by the pointer.

        Safety:
            Pointer is not reference counted, so any dereferencing of another
            pointer to this same address that was copied before the free is
            **not safe**.
        """

        @parameter
        if address_space is AddressSpace.GENERIC:
            if self._flags & 0b1100_0000 == 0b0100_0000:
                p = __mlir_op.`lit.ref.to_pointer`(self._mlir_value)
                alias UP = UnsafePointer[
                    type, AddressSpace.GENERIC, _default_alignment[type](), O
                ]
                UP(rebind[UP._mlir_type](p)).free()
                self._flags &= 0b0011_1111

    fn bitcast[
        T: AnyType = Self.type
    ](self) -> Pointer[T, origin, address_space] as output:
        """Bitcasts a `Pointer` to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new `Pointer` object with the specified type and the same address,
            as the original `Pointer`.
        """
        output = rebind[__type_of(output)](
            __mlir_op.`pop.pointer.bitcast`[
                _type = UnsafePointer[
                    T, address_space, _default_alignment[type]()
                ]._mlir_type,
            ](
                rebind[Pointer[type, MutableAnyOrigin]](self)
                .unsafe_ptr()
                .address
            )
        )

    fn unsafe_ptr[
        O: MutableOrigin, //
    ](self: Pointer[type, O]) -> UnsafePointer[
        type, address_space, _default_alignment[type](), O
    ] as output:
        """Get a raw pointer to the underlying data.

        Returns:
            The raw pointer to the data.
        """
        p = __mlir_op.`lit.ref.to_pointer`(self._mlir_value)
        output = __type_of(output)(rebind[__type_of(output)._mlir_type](p))
