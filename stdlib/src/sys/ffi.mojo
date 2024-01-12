# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements a foreign functions interface (FFI)."""

from sys import external_call
from sys.info import os_is_linux, os_is_windows

from memory.unsafe import DTypePointer, Pointer


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
@register_passable
struct DLHandle:
    """Represents a dynamically linked library that can be loaded and unloaded.

    The library is loaded on initialization and unloaded on deletion of the object.
    """

    var handle: DTypePointer[DType.int8]
    """The handle to the dynamic library."""

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn __init__(path: String, flags: Int = DEFAULT_RTLD) -> Self:
        """Initialize a DLHandle object by loading the dynamic library at the
        given path.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.

        Returns:
            The constructed handle object.
        """

        @parameter
        if not os_is_windows():
            return Self {
                handle: external_call["dlopen", DTypePointer[DType.int8]](
                    path._as_ptr(), flags
                )
            }
        else:
            return Self {handle: DTypePointer[DType.int8]()}

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn _del_old(inout self):
        """Delete the DLHandle object unloading the associated dynamic library.
        """

        @parameter
        if not os_is_windows():
            _ = external_call["dlclose", Int](self.handle)
            self.handle = DTypePointer[DType.int8].get_null()

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn get_function[
        result_type: AnyRegType
    ](self, name: StringRef) -> result_type:
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
            ](self.handle.address, name.data)
            return (
                Pointer(__get_lvalue_as_address(opaque_function_ptr))
                .bitcast[result_type]()
                .load()
            )
        else:
            return Pointer[result_type].get_null().load()


# ===----------------------------------------------------------------------===#
# Library Load
# ===----------------------------------------------------------------------===#


@always_inline
fn _get_global[
    name: StringLiteral,
    init_fn: fn (Pointer[NoneType]) -> Pointer[NoneType],
    destroy_fn: fn (Pointer[NoneType]) -> None,
](payload: Pointer[NoneType] = Pointer[NoneType]()) -> Pointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrCreate", Pointer[NoneType]
    ](StringRef(name), payload, init_fn, destroy_fn)


@always_inline
fn _get_global_or_null[name: StringLiteral]() -> Pointer[NoneType]:
    return external_call["KGEN_CompilerRT_GetGlobalOrNull", Pointer[NoneType]](
        StringRef(name)
    )


@always_inline
fn _get_dylib[
    name: StringLiteral,
    init_fn: fn (Pointer[NoneType]) -> Pointer[NoneType],
    destroy_fn: fn (Pointer[NoneType]) -> None,
](payload: Pointer[NoneType] = Pointer[NoneType]()) -> DLHandle:
    let ptr = _get_global[name, init_fn, destroy_fn](payload).bitcast[
        DLHandle
    ]()
    return __get_address_as_lvalue(ptr.address)


@always_inline
fn _get_dylib_function[
    name: StringLiteral,
    init_fn: fn (Pointer[NoneType]) -> Pointer[NoneType],
    destroy_fn: fn (Pointer[NoneType]) -> None,
    result_type: AnyRegType,
](
    fn_name: StringRef, payload: Pointer[NoneType] = Pointer[NoneType]()
) -> result_type:
    return _get_dylib_function[result_type](
        _get_dylib[name, init_fn, destroy_fn](payload), fn_name
    )


@always_inline
fn _get_dylib_function[
    result_type: AnyRegType
](dylib: DLHandle, name: StringRef) -> result_type:
    return dylib.get_function[result_type](name)
