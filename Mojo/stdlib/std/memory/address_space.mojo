# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Implements `AddressSpace`, the memory address space of a pointer.

You can import this API from the `memory` package. For example:

```mojo
from std.memory import AddressSpace
```
"""

from std._plugin import CurrentPlugin


struct AddressSpace(
    Equatable,
    ImplicitlyCopyable,
    Intable,
    TrivialRegisterPassable,
    Writable,
):
    """Address space of the pointer.

    This type represents memory address spaces for both CPU and GPU targets.
    On CPUs, typically only GENERIC is used. On GPUs (NVIDIA/AMD), various
    address spaces provide access to different memory regions with different
    performance characteristics.
    """

    var _value: SIMDLength

    # CPU address space
    comptime GENERIC = AddressSpace(
        __mlir_attr[`#lit.struct<{_mlir_value = 0}> : `, SIMDLength]
    )
    """Generic address space. Used for CPU memory and default GPU memory."""

    # GPU address spaces
    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    # And https://llvm.org/docs/AMDGPUUsage.html#address-spaces
    comptime GLOBAL = AddressSpace(
        __mlir_attr[`#lit.struct<{_mlir_value = 1}> : `, SIMDLength]
    )
    """Global GPU memory address space."""
    comptime SHARED = AddressSpace(3)
    """Shared GPU memory address space (per thread block/workgroup)."""
    comptime CONSTANT = AddressSpace(4)
    """Constant GPU memory address space (read-only)."""
    comptime LOCAL = AddressSpace(5)
    """Local GPU memory address space (per thread, private)."""
    comptime SHARED_CLUSTER = AddressSpace(7)
    """Shared cluster GPU memory address space (NVIDIA-specific)."""
    comptime BUFFER_RESOURCE = AddressSpace(8)
    """Buffer resource GPU memory address space (AMD-specific)."""

    @always_inline("nodebug")
    @staticmethod
    def __getattr_param__[name: StaticString]() -> AddressSpace:
        """Resolves a target-specific named address space.

        The address spaces above (`GENERIC`, `GLOBAL`, `SHARED`, ...) are the
        built-in GPU set. Accessing any *other* name as `AddressSpace.<NAME>`
        routes here, which consults the active `PluginHooks` backend. This
        keeps the set of valid names open and target-extensible — for example
        an accelerator backend can provide scratchpad-style spaces that do not
        exist on GPUs — instead of a fixed portable enum.

        Parameters:
            name: The address-space name being accessed.

        Returns:
            The `AddressSpace` the active backend defines for `name`.

        Constraints:
            `name` must be defined by the active `PluginHooks` backend;
            otherwise this is a compile-time error.
        """
        comptime if CurrentPlugin.address_space_fn[name]:
            return CurrentPlugin.address_space_fn[name].value()
        else:
            comptime assert False, "unknown address space: '" + name + "'"

    @always_inline("builtin")
    def __init__(out self, value: SIMDLength):
        """Initializes the address space from the underlying integral value.

        Args:
          value: The address space value.
        """
        self._value = value

    @always_inline("builtin")
    def value(self) -> SIMDLength:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("builtin")
    def __int__(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return Int(self._value)

    @always_inline("builtin")
    def __eq__(self, other: Self) -> Bool:
        """Checks if the two address spaces are equal.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self._value == other._value

    @always_inline("nodebug")
    def write_to(self, mut writer: Some[Writer]):
        """Formats the address space to the provided Writer.

        Args:
            writer: The object to write to.
        """
        if self == .GENERIC:
            writer.write("AddressSpace.GENERIC")
        elif self == .GLOBAL:
            writer.write("AddressSpace.GLOBAL")
        elif self == .SHARED:
            writer.write("AddressSpace.SHARED")
        elif self == .CONSTANT:
            writer.write("AddressSpace.CONSTANT")
        elif self == .LOCAL:
            writer.write("AddressSpace.LOCAL")
        elif self == .SHARED_CLUSTER:
            writer.write("AddressSpace.SHARED_CLUSTER")
        else:
            writer.write("AddressSpace(", Int(self.value()), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the AddressSpace.

        Args:
            writer: The object to write to.
        """
        self.write_to(writer)
