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

from memory import DTypePointer, LegacyPointer

from utils import StringRef

from .info import os_is_linux, os_is_windows
from .intrinsics import _mlirtype_is_eq

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
struct DLHandle(CollectionElement, Boolable):
    """Represents a dynamically linked library that can be loaded and unloaded.

    The library is loaded on initialization and unloaded by `close`.
    """

    var handle: DTypePointer[DType.int8]
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
            self.handle = external_call["dlopen", DTypePointer[DType.int8]](
                path.unsafe_ptr(), flags
            )
        else:
            self.handle = DTypePointer[DType.int8]()

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn close(inout self):
        """Delete the DLHandle object unloading the associated dynamic library.
        """

        @parameter
        if not os_is_windows():
            _ = external_call["dlclose", Int](self.handle)
            self.handle = DTypePointer[DType.int8]()

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

        @parameter
        if not os_is_windows():
            var opaque_function_ptr = external_call[
                "dlsym", DTypePointer[DType.int8]
            ](self.handle.address, name)
            return UnsafePointer.address_of(opaque_function_ptr).bitcast[
                result_type
            ]()[]
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

        return self._get_function[result_type](
            func_name.unsafe_ptr().bitcast[C_char]()
        )


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_global[
    name: StringLiteral,
    init_fn: fn (LegacyPointer[NoneType]) -> LegacyPointer[NoneType],
    destroy_fn: fn (LegacyPointer[NoneType]) -> None,
](
    payload: LegacyPointer[NoneType] = LegacyPointer[NoneType]()
) -> LegacyPointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrCreate", LegacyPointer[NoneType]
    ](StringRef(name), payload, init_fn, destroy_fn)


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
    ](StringRef(name))


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
        return UnsafePointer.address_of(func_ptr).bitcast[result_type]()[]

    var dylib = _get_dylib[name, init_fn, destroy_fn](payload)
    var new_func = dylib._get_function[func_name, result_type]()
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringRef(func_cache_name),
        UnsafePointer.address_of(new_func).bitcast[Pointer[NoneType]]()[],
    )

    return new_func


# ===----------------------------------------------------------------------===#
# external_call
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn external_call[callee: StringLiteral, type: AnyTrivialRegType]() -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None]()
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type]()


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral, type: AnyTrivialRegType, T0: AnyTrivialRegType
](arg0: T0) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.

    Args:
      arg0: The first argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](arg0)
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
](arg0: T0, arg1: T1) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
](arg0: T0, arg1: T1, arg2: T2) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3, arg4: T4) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3, arg4: T4, arg5: T5) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4, arg5
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4, arg5
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
](arg0: T0, arg1: T1, arg2: T2, arg3: T3, arg4: T4, arg5: T5, arg6: T6) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.
      T9: The tenth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.
      arg9: The tenth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
    T10: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
    arg10: T10,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.
      T9: The tenth argument type.
      T10: The eleventh argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.
      arg9: The tenth argument.
      arg10: The eleventh argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
    T10: AnyTrivialRegType,
    T11: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
    arg10: T10,
    arg11: T11,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.
      T9: The tenth argument type.
      T10: The eleventh argument type.
      T11: The twelfth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.
      arg9: The tenth argument.
      arg10: The eleventh argument.
      arg11: The twelfth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
    T10: AnyTrivialRegType,
    T11: AnyTrivialRegType,
    T12: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
    arg10: T10,
    arg11: T11,
    arg12: T12,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.
      T9: The tenth argument type.
      T10: The eleventh argument type.
      T11: The twelfth argument type.
      T12: The thirteenth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.
      arg9: The tenth argument.
      arg10: The eleventh argument.
      arg11: The twelfth argument.
      arg12: The thirteenth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
    T10: AnyTrivialRegType,
    T11: AnyTrivialRegType,
    T12: AnyTrivialRegType,
    T13: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
    arg10: T10,
    arg11: T11,
    arg12: T12,
    arg13: T13,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.
      T9: The tenth argument type.
      T10: The eleventh argument type.
      T11: The twelfth argument type.
      T12: The thirteenth argument type.
      T13: The fourteenth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.
      arg9: The tenth argument.
      arg10: The eleventh argument.
      arg11: The twelfth argument.
      arg12: The thirteenth argument.
      arg13: The fourteenth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
        )


@always_inline("nodebug")
fn external_call[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
    T2: AnyTrivialRegType,
    T3: AnyTrivialRegType,
    T4: AnyTrivialRegType,
    T5: AnyTrivialRegType,
    T6: AnyTrivialRegType,
    T7: AnyTrivialRegType,
    T8: AnyTrivialRegType,
    T9: AnyTrivialRegType,
    T10: AnyTrivialRegType,
    T11: AnyTrivialRegType,
    T12: AnyTrivialRegType,
    T13: AnyTrivialRegType,
    T14: AnyTrivialRegType,
](
    arg0: T0,
    arg1: T1,
    arg2: T2,
    arg3: T3,
    arg4: T4,
    arg5: T5,
    arg6: T6,
    arg7: T7,
    arg8: T8,
    arg9: T9,
    arg10: T10,
    arg11: T11,
    arg12: T12,
    arg13: T13,
    arg14: T14,
) -> type:
    """Calls an external function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.
      T2: The third argument type.
      T3: The fourth argument type.
      T4: The fifth argument type.
      T5: The sixth argument type.
      T6: The seventh argument type.
      T7: The eighth argument type.
      T8: The ninth argument type.
      T9: The tenth argument type.
      T10: The eleventh argument type.
      T11: The twelfth argument type.
      T12: The thirteenth argument type.
      T13: The fourteenth argument type.
      T14: The fifteenth argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.
      arg2: The third argument.
      arg3: The fourth argument.
      arg4: The fifth argument.
      arg5: The sixth argument.
      arg6: The seventh argument.
      arg7: The eighth argument.
      arg8: The ninth argument.
      arg9: The tenth argument.
      arg10: The eleventh argument.
      arg11: The twelfth argument.
      arg12: The thirteenth argument.
      arg13: The fourteenth argument.
      arg14: The fifteenth argument.

    Returns:
      The external call result.
    """

    @parameter
    if _mlirtype_is_eq[type, NoneType]():
        __mlir_op.`pop.external_call`[func = callee.value, _type=None](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
            arg14,
        )
        return rebind[type](None)
    else:
        return __mlir_op.`pop.external_call`[func = callee.value, _type=type](
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
            arg14,
        )


# ===----------------------------------------------------------------------===#
# _external_call_const
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _external_call_const[
    callee: StringLiteral, type: AnyTrivialRegType
]() -> type:
    """Mark the external function call as having no observable effects to the
    program state. This allows the compiler to optimize away successive calls
    to the same function.

    Parameters:
      callee: The name of the external function.
      type: The return type.

    Returns:
      The external call result.
    """
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
    ]()


@always_inline("nodebug")
fn _external_call_const[
    callee: StringLiteral, type: AnyTrivialRegType, T0: AnyTrivialRegType
](arg0: T0) -> type:
    """Mark the external function call as having no observable effects to the
    program state. This allows the compiler to optimize away successive calls
    to the same function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.

    Args:
      arg0: The first argument.

    Returns:
      The external call result.
    """
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
    ](arg0)


@always_inline("nodebug")
fn _external_call_const[
    callee: StringLiteral,
    type: AnyTrivialRegType,
    T0: AnyTrivialRegType,
    T1: AnyTrivialRegType,
](arg0: T0, arg1: T1) -> type:
    """Mark the external function call as having no observable effects to the
    program state. This allows the compiler to optimize away successive calls
    to the same function.

    Parameters:
      callee: The name of the external function.
      type: The return type.
      T0: The first argument type.
      T1: The second argument type.

    Args:
      arg0: The first argument.
      arg1: The second argument.

    Returns:
      The external call result.
    """
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
    ](arg0, arg1)
