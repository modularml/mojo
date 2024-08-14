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
"""Implements a foreign functions interface (FFI)."""

from os import abort
from memory import UnsafePointer

from utils import StringRef

from .info import os_is_linux, os_is_windows
from .intrinsics import _mlirtype_is_eq
from builtin.builtin_list import _LITRefPackHelper

alias C_char = Int8
"""C `char` type."""


struct RTLD:
    """Enumeration of the RTLD flags used during dynamic library loading."""

    alias LAZY = 1
    """Load library lazily (defer function resolution until needed).
    """
    alias NOW = 2
    """Load library immediately (resolve all symbols on load)."""
    alias LOCAL = 4
    """Make symbols not available for symbol resolution of subsequently loaded
    libraries."""
    alias GLOBAL = 256 if os_is_linux() else 8
    """Make symbols available for symbol resolution of subsequently loaded
    libraries."""


alias DEFAULT_RTLD = RTLD.NOW | RTLD.GLOBAL


@value
@register_passable("trivial")
struct DLHandle(CollectionElement, CollectionElementNew, Boolable):
    """Represents a dynamically linked library that can be loaded and unloaded.

    The library is loaded on initialization and unloaded by `close`.
    """

    var handle: UnsafePointer[Int8]
    """The handle to the dynamic library."""

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn __init__(inout self, path: String, flags: Int = DEFAULT_RTLD):
        """Initialize a DLHandle object by loading the dynamic library at the
        given path.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.
        """

        @parameter
        if not os_is_windows():
            var handle = external_call["dlopen", UnsafePointer[Int8]](
                path.unsafe_cstr_ptr(), flags
            )
            if handle == UnsafePointer[Int8]():
                var error_message = external_call[
                    "dlerror", UnsafePointer[UInt8]
                ]()
                abort("dlopen failed: " + String(error_message))
            self.handle = handle
        else:
            self.handle = UnsafePointer[Int8]()

    fn __init__(inout self, *, other: Self):
        """Copy the object.

        Args:
            other: The value to copy.
        """
        self = other

    fn check_symbol(self, name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        constrained[
            not os_is_windows(),
            "Checking dynamic library symbol is not supported on Windows",
        ]()

        var opaque_function_ptr = external_call["dlsym", UnsafePointer[Int8]](
            self.handle.address, name.unsafe_cstr_ptr()
        )
        if opaque_function_ptr:
            return True

        return False

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn close(inout self):
        """Delete the DLHandle object unloading the associated dynamic library.
        """

        @parameter
        if not os_is_windows():
            _ = external_call["dlclose", Int](self.handle)
            self.handle = UnsafePointer[Int8]()

    fn __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
          True if the DLHandle is not null and False otherwise.
        """
        return self.handle.__bool__()

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn get_function[
        result_type: AnyTrivialRegType
    ](self, name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """

        return self._get_function[result_type](name.unsafe_cstr_ptr())

    @always_inline
    fn _get_function[
        result_type: AnyTrivialRegType
    ](self, name: UnsafePointer[C_char]) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        debug_assert(self.handle, "Dylib handle is null")

        @parameter
        if not os_is_windows():
            var opaque_function_ptr = external_call[
                "dlsym", UnsafePointer[Int8]
            ](self.handle.address, name)
            var result = UnsafePointer.address_of(opaque_function_ptr).bitcast[
                result_type
            ]()[]
            _ = opaque_function_ptr
            return result
        else:
            return abort[result_type]("get_function isn't supported on windows")

    @always_inline
    fn _get_function[
        func_name: StringLiteral, result_type: AnyTrivialRegType
    ](self) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            func_name:The name of the function to get the handle for.
            result_type: The type of the function pointer to return.

        Returns:
            A handle to the function.
        """

        return self._get_function[result_type](func_name.unsafe_cstr_ptr())


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_global[
    name: StringLiteral,
    init_fn: fn (UnsafePointer[NoneType]) -> UnsafePointer[NoneType],
    destroy_fn: fn (UnsafePointer[NoneType]) -> None,
](
    payload: UnsafePointer[NoneType] = UnsafePointer[NoneType]()
) -> UnsafePointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrCreate", UnsafePointer[NoneType]
    ](StringRef(name), payload, init_fn, destroy_fn)


@always_inline
fn _get_global_or_null[name: StringLiteral]() -> UnsafePointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrNull", UnsafePointer[NoneType]
    ](name.unsafe_ptr(), name.byte_length())


@always_inline
fn _get_dylib[
    name: StringLiteral,
    init_fn: fn (UnsafePointer[NoneType]) -> UnsafePointer[NoneType],
    destroy_fn: fn (UnsafePointer[NoneType]) -> None,
](payload: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) -> DLHandle:
    var ptr = _get_global[name, init_fn, destroy_fn](payload).bitcast[
        DLHandle
    ]()
    return ptr[]


@always_inline
fn _get_dylib_function[
    name: StringLiteral,
    func_name: StringLiteral,
    init_fn: fn (UnsafePointer[NoneType]) -> UnsafePointer[NoneType],
    destroy_fn: fn (UnsafePointer[NoneType]) -> None,
    result_type: AnyTrivialRegType,
](payload: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) -> result_type:
    alias func_cache_name = name + "/" + func_name
    var func_ptr = _get_global_or_null[func_cache_name]()
    if func_ptr:
        var result = UnsafePointer.address_of(func_ptr).bitcast[result_type]()[]
        _ = func_ptr
        return result

    var dylib = _get_dylib[name, init_fn, destroy_fn](payload)
    var new_func = dylib._get_function[func_name, result_type]()
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringRef(func_cache_name),
        UnsafePointer.address_of(new_func).bitcast[UnsafePointer[NoneType]]()[],
    )

    return new_func


# ===----------------------------------------------------------------------===#
# external_call
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral, type: AnyTrivialRegType, *types: AnyType
](*arguments: *types) -> type:
    """Calls an external function.

    Args:
      arguments: The arguments to pass to the external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      types: The argument types.

    Returns:
      The external call result.
    """

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = _LITRefPackHelper(arguments._value).get_loaded_kgen_pack()

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            loaded_pack
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            loaded_pack
        )


# ===----------------------------------------------------------------------===#
# _external_call_const
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _external_call_const[
    callee: StringLiteral, type: AnyTrivialRegType, *types: AnyType
](*arguments: *types) -> type:
    """Mark the external function call as having no observable effects to the
    program state. This allows the compiler to optimize away successive calls
    to the same function.

    Args:
      arguments: The arguments to pass to the external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      types: The argument types.

    Returns:
      The external call result.
    """

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = _LITRefPackHelper(arguments._value).get_loaded_kgen_pack()

    return __mlir_op.`pop.external_call`[
        func = callee.value,
        resAttrs = __mlir_attr.`[{llvm.noundef}]`,
        funcAttrs = __mlir_attr.`["willreturn"]`,
        memory = __mlir_attr[
            `#llvm.memory_effects<other = none, `,
            `argMem = none, `,
            `inaccessibleMem = none>`,
        ],
        _type=type,
    ](loaded_pack)
