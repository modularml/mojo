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


# ===-----------------------------------------------------------------------===#
# AddressSpace
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _GPUAddressSpace(EqualityComparable):
    var _value: Int

    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    # And https://llvm.org/docs/AMDGPUUsage.html#address-spaces
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
    @implicit
    fn __init__(out self, value: Int):
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
struct AddressSpace(EqualityComparable, Stringable, Writable):
    """Address space of the pointer."""

    var _value: Int

    alias GENERIC = AddressSpace(0)
    """Generic address space."""

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        """Initializes the address space from the underlying integral value.

        Args:
          value: The address space value.
        """
        self._value = value

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: _GPUAddressSpace):
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

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Gets a string representation of the AddressSpace.

        Returns:
            The string representation of the AddressSpace.
        """
        return String.write(self)

    @always_inline("nodebug")
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats the address space to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        if self is AddressSpace.GENERIC:
            writer.write("AddressSpace.GENERIC")
        else:
            writer.write("AddressSpace(", self.value(), ")")


# ===-----------------------------------------------------------------------===#
# Pointer
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct Pointer[
    is_mutable: Bool, //,
    type: AnyType,
    origin: Origin[is_mutable],
    address_space: AddressSpace = AddressSpace.GENERIC,
](CollectionElementNew, Stringable):
    """Defines a non-nullable safe pointer.

    For a comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/) in the Mojo Manual.

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
        origin._mlir_origin,
        `, `,
        address_space._value.value,
        `>`,
    ]

    var _value: Self._mlir_type
    """The underlying MLIR representation."""

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @doc_private
    @always_inline("nodebug")
    fn __init__(out self, *, _mlir_value: Self._mlir_type):
        """Constructs a Pointer from its MLIR prepresentation.

        Args:
             _mlir_value: The MLIR representation of the pointer.
        """
        self._value = _mlir_value

    @staticmethod
    @always_inline("nodebug")
    fn address_of(ref [origin, address_space]value: type) -> Self:
        """Constructs a Pointer from a reference to a value.

        Args:
            value: The value to get the address of.

        Returns:
            The result Pointer.
        """
        return Pointer(_mlir_value=__get_mvalue_as_litref(value))

    fn __init__(out self, *, other: Self):
        """Constructs a copy from another Pointer.

        Note that this does **not** copy the underlying data.

        Args:
            other: The `Pointer` to copy.
        """
        self._value = other._value

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [origin, address_space] type:
        """Enable subscript syntax `ptr[]` to access the element.

        Returns:
            A reference to the underlying value in memory.
        """
        return __get_litref_as_mvalue(self._value)

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
