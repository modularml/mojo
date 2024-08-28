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
"""Implements the Reference type.

You can import these APIs from the `memory` package. For example:

```mojo
from memory.reference import Reference
```
"""

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
# Reference
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Reference[
    is_mutable: Bool, //,
    type: AnyType,
    lifetime: AnyLifetime[is_mutable].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
](CollectionElementNew, Stringable):
    """Defines a non-nullable safe reference.

    Parameters:
        is_mutable: Whether the referenced data may be mutated through this.
        type: Type of the underlying data.
        lifetime: The lifetime of the reference.
        address_space: The address space of the referenced data.
    """

    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        type,
        `, `,
        lifetime,
        `, `,
        address_space._value.value,
        `>`,
    ]

    var value: Self._mlir_type
    """The underlying MLIR reference."""

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __init__(
        inout self, ref [lifetime, address_space._value.value]value: type
    ):
        """Constructs a Reference from a value reference.

        Args:
            value: The value reference.
        """
        self.value = __get_mvalue_as_litref(value)

    fn __init__(inout self, *, other: Self):
        """Constructs a copy from another Reference.

        Note that this does **not** copy the underlying data.

        Args:
            other: The `Reference` to copy.
        """
        self.value = other.value

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [lifetime, address_space._value.value] type:
        """Enable subscript syntax `ref[]` to access the element.

        Returns:
            The MLIR reference for the Mojo compiler to use.
        """
        return __get_litref_as_mvalue(self.value)

    @always_inline("nodebug")
    fn __eq__(self, rhs: Reference[type, _, address_space]) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return UnsafePointer(
            __mlir_op.`lit.ref.to_pointer`(self.value)
        ) == UnsafePointer(__mlir_op.`lit.ref.to_pointer`(rhs.value))

    @always_inline("nodebug")
    fn __ne__(self, rhs: Reference[type, _, address_space]) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the Reference.

        Returns:
            The string representation of the Reference.
        """
        return str(UnsafePointer(__mlir_op.`lit.ref.to_pointer`(self.value)))
