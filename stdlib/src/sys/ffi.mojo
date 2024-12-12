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
from sys._libc import dlclose, dlerror, dlopen, dlsym

from memory import UnsafePointer

from utils import StringRef

from .info import is_64bit, os_is_linux, os_is_macos, os_is_windows
from .intrinsics import _mlirtype_is_eq

# ===-----------------------------------------------------------------------===#
# Primitive C type aliases
# ===-----------------------------------------------------------------------===#

alias c_char = Int8
"""C `char` type."""

alias c_int = Int32
"""C `int` type.

The C `int` type is typically a signed 32-bit integer on commonly used targets
today.
"""

alias c_uint = UInt32
"""C `unsigned int` type."""

alias c_long = Scalar[_c_long_dtype()]
"""C `long` type.

The C `long` type is typically a signed 64-bit integer on macOS and Linux, and a
32-bit integer on Windows."""

alias c_long_long = Scalar[_c_long_long_dtype()]
"""C `long long` type.

The C `long long` type is typically a signed 64-bit integer on commonly used
targets today."""

alias c_size_t = UInt
"""C `size_t` type."""

alias c_ssize_t = Int
"""C `ssize_t` type."""

alias OpaquePointer = UnsafePointer[NoneType]
"""An opaque pointer, equivalent to the C `void*` type."""


fn _c_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and (os_is_macos() or os_is_linux()):
        # LP64
        return DType.int64
    elif is_64bit() and os_is_windows():
        # LLP64
        return DType.int32
    else:
        constrained[False, "size of C `long` is unknown on this target"]()
        return abort[DType]()


fn _c_long_long_dtype() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    @parameter
    if is_64bit() and (os_is_macos() or os_is_linux() or os_is_windows()):
        # On a 64-bit CPU, `long long` is *always* 64 bits in every OS's data
        # model.
        return DType.int64
    else:
        constrained[False, "size of C `long long` is unknown on this target"]()
        return abort[DType]()


# ===-----------------------------------------------------------------------===#
# Dynamic Library Loading
# ===-----------------------------------------------------------------------===#


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

    var handle: OpaquePointer
    """The handle to the dynamic library."""

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn __init__(out self, path: String, flags: Int = DEFAULT_RTLD):
        """Initialize a DLHandle object by loading the dynamic library at the
        given path.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.
        """

        @parameter
        if not os_is_windows():
            var handle = dlopen(path.unsafe_cstr_ptr(), flags)
            if handle == OpaquePointer():
                var error_message = dlerror()
                abort("dlopen failed: " + String(error_message))
            self.handle = handle
        else:
            self.handle = OpaquePointer()

    fn __init__(out self, *, other: Self):
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

        var opaque_function_ptr: OpaquePointer = dlsym(
            self.handle,
            name.unsafe_cstr_ptr(),
        )

        return bool(opaque_function_ptr)

    # TODO(#15590): Implement support for windows and remove the always_inline.
    @always_inline
    fn close(mut self):
        """Delete the DLHandle object unloading the associated dynamic library.
        """

        @parameter
        if not os_is_windows():
            _ = dlclose(self.handle)
            self.handle = OpaquePointer()

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
    ](self, name: UnsafePointer[c_char]) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        var opaque_function_ptr = self.get_symbol[NoneType](name)

        var result = UnsafePointer.address_of(opaque_function_ptr).bitcast[
            result_type
        ]()[]
        _ = opaque_function_ptr
        return result

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

    fn get_symbol[
        result_type: AnyType,
    ](self, name: StringLiteral) -> UnsafePointer[result_type]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            A pointer to the symbol.
        """
        return self.get_symbol[result_type](name.unsafe_cstr_ptr())

    fn get_symbol[
        result_type: AnyType
    ](self, name: UnsafePointer[Int8]) -> UnsafePointer[result_type]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            A pointer to the symbol.
        """
        debug_assert(self.handle, "Dylib handle is null")

        @parameter
        if os_is_windows():
            return abort[UnsafePointer[result_type]](
                "get_symbol isn't supported on windows"
            )

        # To check for `dlsym()` results that are _validly_ NULL, we do the
        # dance described in https://man7.org/linux/man-pages/man3/dlsym.3.html:
        #
        # > In unusual cases (see NOTES) the value of the symbol could
        # > actually be NULL.  Therefore, a NULL return from dlsym() need not
        # > indicate an error.  The correct way to distinguish an error from
        # > a symbol whose value is NULL is to call dlerror(3) to clear any
        # > old error conditions, then call dlsym(), and then call dlerror(3)
        # > again, saving its return value into a variable, and check whether
        # > this saved value is not NULL.

        var res = dlsym[result_type](self.handle, name)

        if not res:
            # Clear any potential unrelated error that pre-dates the `dlsym`
            # call above.
            _ = dlerror()

            # Redo the `dlsym` call
            res = dlsym[result_type](self.handle, name)

            debug_assert(not res, "dlsym unexpectedly returned non-NULL result")

            # Check if an error occurred during the 2nd `dlsym` call.
            var err = dlerror()

            if err:
                abort("dlsym failed: " + String(err))

        return res

    @always_inline
    fn call[
        name: StringLiteral,
        return_type: AnyTrivialRegType = NoneType,
        *T: AnyType,
    ](self, *args: *T) -> return_type:
        """Call a function with any amount of arguments.

        Parameters:
            name: The name of the function.
            return_type: The return type of the function.
            T: The types of `args`.

        Args:
            args: The arguments.

        Returns:
            The result.
        """
        return self.call[name, return_type](args)

    fn call[
        name: StringLiteral, return_type: AnyTrivialRegType = NoneType
    ](self, args: VariadicPack[element_trait=AnyType]) -> return_type:
        """Call a function with any amount of arguments.

        Parameters:
            name: The name of the function.
            return_type: The return type of the function.

        Args:
            args: The arguments.

        Returns:
            The result.
        """

        debug_assert(self.check_symbol(name), "symbol not found: " + name)
        var v = args.get_loaded_kgen_pack()
        return self.get_function[fn (__type_of(v)) -> return_type](name)(v)


@always_inline
fn _get_dylib[
    name: StringLiteral,
    init_fn: fn (OpaquePointer) -> OpaquePointer,
    destroy_fn: fn (OpaquePointer) -> None,
](payload: OpaquePointer = OpaquePointer()) -> DLHandle:
    var ptr = _get_global[name, init_fn, destroy_fn](payload).bitcast[
        DLHandle
    ]()
    return ptr[]


@always_inline
fn _get_dylib_function[
    name: StringLiteral,
    func_name: StringLiteral,
    init_fn: fn (OpaquePointer) -> OpaquePointer,
    destroy_fn: fn (OpaquePointer) -> None,
    result_type: AnyTrivialRegType,
](payload: OpaquePointer = OpaquePointer()) -> result_type:
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
        UnsafePointer.address_of(new_func).bitcast[OpaquePointer]()[],
    )

    return new_func


# ===-----------------------------------------------------------------------===#
# Globals
# ===-----------------------------------------------------------------------===#


struct _Global[
    name: StringLiteral,
    storage_type: Movable,
    init_fn: fn () -> storage_type,
]:
    @staticmethod
    fn _init_wrapper(payload: OpaquePointer) -> OpaquePointer:
        # Struct-based globals don't get to take arguments to their initializer.
        debug_assert(not payload)

        # Heap allocate space to store this "global"
        var ptr = UnsafePointer[storage_type].alloc(1)

        # TODO:
        #   Any way to avoid the move, e.g. by calling this function
        #   with the ABI destination result pointer already set to `ptr`?
        ptr.init_pointee_move(init_fn())

        return ptr.bitcast[NoneType]()

    @staticmethod
    fn _deinit_wrapper(self_: OpaquePointer):
        var ptr: UnsafePointer[storage_type] = self_.bitcast[storage_type]()

        # Deinitialize and deallocate the global
        ptr.destroy_pointee()
        ptr.free()

    @staticmethod
    fn get_or_create_ptr() -> UnsafePointer[storage_type]:
        return _get_global[
            name, Self._init_wrapper, Self._deinit_wrapper
        ]().bitcast[storage_type]()


@always_inline
fn _get_global[
    name: StringLiteral,
    init_fn: fn (OpaquePointer) -> OpaquePointer,
    destroy_fn: fn (OpaquePointer) -> None,
](payload: OpaquePointer = OpaquePointer()) -> OpaquePointer:
    return external_call["KGEN_CompilerRT_GetGlobalOrCreate", OpaquePointer](
        StringRef(name), payload, init_fn, destroy_fn
    )


@always_inline
fn _get_global_or_null[name: StringLiteral]() -> OpaquePointer:
    return external_call["KGEN_CompilerRT_GetGlobalOrNull", OpaquePointer](
        name.unsafe_ptr(), name.byte_length()
    )


# ===-----------------------------------------------------------------------===#
# external_call
# ===-----------------------------------------------------------------------===#


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
    var loaded_pack = arguments.get_loaded_kgen_pack()

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


# ===-----------------------------------------------------------------------===#
# _external_call_const
# ===-----------------------------------------------------------------------===#


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
    var loaded_pack = arguments.get_loaded_kgen_pack()

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
