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
"""Defines `UnsafePointer` as a backward-compatible name for `Pointer`.

`UnsafePointer` is a `comptime` alias for `Pointer`, kept for code written
before the two pointer types were unified. Prefer `Pointer` directly in new
code; see `std.memory.pointer` for the type's docstring and its `unsafe_`
method surface.
"""

from std.collections import OptionalReg
from std.memory.address_space import AddressSpace
from std.memory.pointer import OptionalPointer, Pointer

# ===----------------------------------------------------------------------=== #
# unsafe_cast
# ===----------------------------------------------------------------------=== #


@always_inline
@doc_hidden
def unsafe_cast[
    from_mut: Bool,
    from_type: AnyType,
    from_origin: Origin[mut=from_mut],
    from_address_space: AddressSpace,
    mut: Bool = from_mut,
    //,
    *,
    Type: AnyType = from_type,
    origin: Origin[mut=mut] = from_origin,
    address_space: AddressSpace = from_address_space,
](
    pointer: OptionalPointer[
        from_type, from_origin, address_space=from_address_space
    ],
    out result: OptionalPointer[Type, origin, address_space=address_space],
):
    result = Pointer(to=pointer).unsafe_bitcast[type_of(result)]()[]


@always_inline
@doc_hidden
def unsafe_cast[
    from_mut: Bool,
    from_type: AnyType,
    from_origin: Origin[mut=from_mut],
    from_address_space: AddressSpace,
    mut: Bool = from_mut,
    //,
    *,
    Type: AnyType = from_type,
    origin: Origin[mut=mut] = from_origin,
    address_space: AddressSpace = from_address_space,
](
    pointer: OptionalReg[
        Pointer[from_type, from_origin, address_space=from_address_space]
    ],
    out result: OptionalReg[Pointer[Type, origin, address_space=address_space]],
):
    result = Pointer(to=pointer).unsafe_bitcast[type_of(result)]()[]


@always_inline("nodebug")
@doc_hidden
def pointer_to_int(pointer: OptionalPointer[...]) -> Int:
    return Pointer(to=pointer).unsafe_bitcast[Int]()[]


# ===----------------------------------------------------------------------=== #
# UnsafePointer alias
# ===----------------------------------------------------------------------=== #


@deprecated(use=Pointer)
comptime UnsafePointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = .GENERIC,
] = Pointer[T, origin, address_space=address_space]
"""An indirect reference to one or more values of `T` consecutively in
memory, and can refer to uninitialized memory.

Parameters:
    mut: Whether the pointer is mutable.
    T: The type the pointer points to.
    origin: The origin of the memory being addressed.
    address_space: The address space of the pointer.
"""


comptime _UnsafeDanglingPluginHookFnType = def[alignment: Int]() thin -> Int
"""Plugin-hook signature for `PluginHooks.unsafe_dangling_fn`; keep in sync with `Pointer.unsafe_dangling`."""
