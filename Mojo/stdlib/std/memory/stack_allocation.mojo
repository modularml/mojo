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
"""Defines the `unsafe_stack_allocation` function for stack-based memory
allocation.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import unsafe_stack_allocation
```
"""

from std.collections.string.string_span import _get_kgen_string
from std.sys import align_of, is_gpu
from std._plugin import CurrentPlugin


@always_inline
def unsafe_stack_allocation[
    count: Int,
    dtype: DType,
    /,
    alignment: Int = align_of[dtype](),
    address_space: AddressSpace = .GENERIC,
]() -> Pointer[Scalar[dtype], MutUntrackedOrigin, address_space=address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        dtype: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return unsafe_stack_allocation[
        count, Scalar[dtype], alignment=alignment, address_space=address_space
    ]()


comptime _StackAllocationPluginHookFnType[address_space: AddressSpace] = def[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString],
    alignment: Int,
]() thin -> Pointer[type, MutUntrackedOrigin, address_space=address_space]
"""Plugin-hook signature for `PluginHooks.stack_allocation_fn`; keep in sync with `unsafe_stack_allocation`."""


@always_inline
def unsafe_stack_allocation[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString] = None,
    alignment: Int = align_of[type](),
    address_space: AddressSpace = .GENERIC,
]() -> Pointer[type, MutUntrackedOrigin, address_space=address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        name: The name of the global variable (only honored in certain cases).
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    comptime if is_gpu():
        # On NVGPU, SHARED and CONSTANT address spaces lower to global memory.

        comptime global_name = name.value() if name else "_global_alloc"

        comptime if address_space == .SHARED:
            return {
                _mlir_value = __mlir_op.`pop.global_alloc`[
                    name=_get_kgen_string[global_name](),
                    count=count.__mlir_index__(),
                    memoryType=__mlir_attr.`#pop.global_alloc_addr_space<gpu_shared>`,
                    _type=Pointer[
                        type, MutUntrackedOrigin, address_space=address_space
                    ]._mlir_type,
                    alignment=alignment.__mlir_index__(),
                ]()
            }
        elif address_space == .CONSTANT:
            # No need to annotation this global_alloc because constants in
            # GPU shared memory won't prevent llvm module splitting to
            # happen since they are immutables.
            return {
                _mlir_value = __mlir_op.`pop.global_alloc`[
                    name=_get_kgen_string[global_name](),
                    count=count.__mlir_index__(),
                    _type=Pointer[
                        type, MutUntrackedOrigin, address_space=address_space
                    ]._mlir_type,
                    alignment=alignment.__mlir_index__(),
                ]()
            }

        # MSTDL-797: The NVPTX backend requires that `alloca` instructions may
        # only have generic address spaces. When allocating LOCAL memory,
        # addrspacecast the resulting pointer.
        elif address_space == .LOCAL:
            var generic_ptr = __mlir_op.`pop.stack_allocation`[
                count=count.__mlir_index__(),
                _type=Pointer[type, MutUntrackedOrigin]._mlir_type,
                alignment=alignment.__mlir_index__(),
            ]()
            return {
                _mlir_value = __mlir_op.`pop.pointer.bitcast`[
                    _type=Pointer[
                        type, MutUntrackedOrigin, address_space=address_space
                    ]._mlir_type
                ](generic_ptr)
            }

    elif CurrentPlugin.stack_allocation_fn[address_space]:
        return comptime (
            CurrentPlugin.stack_allocation_fn[address_space].value()
        )[count, type, name=name, alignment=alignment]()

    # Perform a stack allocation of the requested size, alignment, and type.
    return {
        _mlir_value = __mlir_op.`pop.stack_allocation`[
            count=count.__mlir_index__(),
            _type=Pointer[
                type, MutUntrackedOrigin, address_space=address_space
            ]._mlir_type,
            alignment=alignment.__mlir_index__(),
        ]()
    }


@always_inline
def stack_allocation[
    count: Int,
    dtype: DType,
    /,
    alignment: Int = align_of[dtype](),
    address_space: AddressSpace = .GENERIC,
]() -> Pointer[Scalar[dtype], MutUntrackedOrigin, address_space=address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        dtype: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return unsafe_stack_allocation[
        count,
        Scalar[dtype],
        alignment=alignment,
        address_space=address_space,
    ]()


@always_inline
def stack_allocation[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString] = None,
    alignment: Int = align_of[type](),
    address_space: AddressSpace = .GENERIC,
]() -> Pointer[type, MutUntrackedOrigin, address_space=address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        name: The name of the global variable (only honored in certain cases).
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return unsafe_stack_allocation[
        count,
        type,
        name=name,
        alignment=alignment,
        address_space=address_space,
    ]()
