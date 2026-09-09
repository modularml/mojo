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
"""Foreign function interface (FFI) for calling C code and loading libraries.

This module provides tools for interfacing Mojo with C libraries and other
foreign code. It includes:

- **C type aliases**: `c_int`, `c_char`, `c_long`, `c_size_t`, etc. for
  portable type definitions that match C's type sizes on each platform.
- **Dynamic library loading**: `OwnedDLHandle` for loading shared libraries
  at runtime and calling their functions.
- **External function calls**: `external_call()` for calling C functions
  by name with compile-time resolution.
- **String interop**: `CStringSlice` for working with null-terminated C strings.

Example:

```mojo
from std.ffi import c_int, external_call

def get_random() -> c_int:
    return external_call["rand", c_int]()
```

For loading dynamic libraries:

```mojo
from std.ffi import OwnedDLHandle

def main() raises:
    var lib = OwnedDLHandle("libm.so")
    var sqrt = lib.get_function[Float64]("sqrt")
    print(sqrt(4.0))  # 2.0
```
"""

from std.collections.optional import OptionalReg
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.os import PathLike as stdPathLike, abort
from std.pathlib import Path
from std.sys._libc import dlclose, dlerror, dlopen, dlsym
from std.sys._libc_errno import ErrNo, get_errno, set_errno

from std.memory import OwnedPointer, Pointer
from std.memory.alloc import dealloc, ThinAllocation
from std.memory.unsafe_pointer import unsafe_cast

from std.sys.info import CompilationTarget, is_32bit, is_64bit, size_of
from .cstring import CStringSlice
from .unsafe_union import UnsafeUnion

# ===-----------------------------------------------------------------------===#
# Primitive C type aliases
# ===-----------------------------------------------------------------------===#

comptime c_char = Int8
"""C `char` type."""

comptime c_uchar = UInt8
"""C `unsigned char` type."""

comptime c_int = Int32
"""C `int` type.

The C `int` type is typically a signed 32-bit integer on commonly used targets
today.
"""

comptime c_uint = UInt32
"""C `unsigned int` type."""

comptime c_short = Int16
"""C `short` type."""

comptime c_ushort = UInt16
"""C `unsigned short` type."""

comptime c_long = Scalar[_c_long_dtype()]
"""C `long` type.

The C `long` type is typically a signed 64-bit integer on macOS and Linux, and a
32-bit integer on Windows."""

comptime c_long_long = Scalar[_c_long_long_dtype()]
"""C `long long` type.

The C `long long` type is typically a signed 64-bit integer on commonly used
targets today."""

comptime c_ulong = Scalar[_c_long_dtype[unsigned=True]()]
"""C `unsigned long` type.

The C `unsigned long` type is typically a 64-bit integer on commonly used
targets today."""

comptime c_ulong_long = Scalar[_c_long_long_dtype[unsigned=True]()]
"""C `unsigned long long` type.

The C `unsigned long long` type is typically a 64-bit integer on commonly used
targets today."""


comptime c_size_t = UInt
"""C `size_t` type."""

comptime c_ssize_t = Int
"""C `ssize_t` type."""

comptime c_float = Float32
"""C `float` type."""

comptime c_double = Float64
"""C `double` type."""

comptime c_pid_t = Int
"""C `pid_t` type."""

comptime MAX_PATH = _get_max_path()
"""Maximum path length for the current platform."""


def _get_max_path() -> Int:
    comptime if CompilationTarget.is_linux():
        return 4096
    elif CompilationTarget.is_macos():
        return 1024
    # Default POSIX limit
    else:
        return 256


def _c_long_dtype[unsigned: Bool = False]() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models

    comptime if is_64bit() and (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ):
        # LP64: long is 64-bit on 64-bit systems (e.g. x86_64 or aarch64)
        return DType.uint64 if unsigned else DType.int64
    elif is_32bit():
        # ILP32: long is 32-bit on 32-bit systems (e.g. x86 or RISC-V 32bit)
        return DType.uint32 if unsigned else DType.int32
    else:
        comptime assert False, "size of C `long` is unknown on this target"


def _c_long_long_dtype[unsigned: Bool = False]() -> DType:
    # https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
    # `long long` is 64 bits on all common platforms (LP64, LLP64, ILP32).

    comptime assert (
        is_64bit() or is_32bit()
    ), "size of C `long long` is unknown on this target"
    return DType.uint64 if unsigned else DType.int64


# ===-----------------------------------------------------------------------===#
# Dynamic Library Loading
# ===-----------------------------------------------------------------------===#


struct RTLD:
    """Enumeration of the RTLD flags used during dynamic library loading."""

    comptime LAZY = 1
    """Load library lazily (defer function resolution until needed).
    """
    comptime NOW = 2
    """Load library immediately (resolve all symbols on load)."""
    comptime LOCAL = 0 if CompilationTarget.is_linux() else 4
    """Make symbols not available for symbol resolution of subsequently loaded
    libraries."""
    comptime GLOBAL = 256 if CompilationTarget.is_linux() else 8
    """Make symbols available for symbol resolution of subsequently loaded
    libraries."""
    comptime NODELETE = 4096 if CompilationTarget.is_linux() else 128
    """Do not delete the library when the process exits."""


comptime DEFAULT_RTLD = RTLD.NOW | RTLD.GLOBAL
"""Default runtime linker flags for dynamic library loading."""


@fieldwise_init
struct _DLCallable[
    return_type: RegisterPassable,
    origin: ImmOrigin,
](TrivialRegisterPassable):
    """A callable proxy returned from `OwnedDLHandle.get_function`.

    Holds a raw function pointer resolved via `dlsym` together with an
    immutable borrow of the originating `OwnedDLHandle` (via `origin`),
    so the library cannot be `dlclose`d until after the pointer is invoked.

    Parameters:
        return_type: The return type of the underlying C function.
        origin: The origin of the `OwnedDLHandle` that this callable borrows
            from, preventing ASAP destruction of the handle across the call.

    Notes:
        Each call reinterprets the symbol as a C function pointer whose
        signature is the call's argument types and forwards the arguments
        individually, using the C ABI (`abi("C")`).
    """

    var _opaque: Pointer[NoneType, MutUntrackedOrigin]
    """The raw function pointer resolved via `dlsym`, stored opaquely."""

    var _lib: Pointer[OwnedDLHandle, Self.origin]
    """An immutable borrow of the owning handle. Its presence forces the
    compiler to keep the handle alive for the lifetime of this callable."""

    @always_inline
    def __call__[*T: AnyType](self, *args: *T) -> Self.return_type:
        """Invokes the underlying C function with the given arguments.

        Parameters:
            T: The types of the arguments.

        Args:
            args: The arguments to forward to the underlying function.

        Returns:
            The return value of the C function.
        """
        # Spelling the callee type over the argument pack `T`, rather than as
        # a single kgen-pack aggregate, is what makes codegen expand the pack
        # and apply the C calling convention per argument.
        #
        # The bitcast goes via `Pointer(to=self._opaque)` — taking the address
        # of the field, reinterpreting it as pointing to a function-pointer
        # type, then loading — because a `Pointer[NoneType]` value
        # cannot be directly reinterpreted as a function-pointer value
        # (`.unsafe_bitcast` only changes the pointee type).
        var typed_fn = Pointer(to=self._opaque).unsafe_bitcast[
            def(* a: * T) thin abi("C") -> Self.return_type
        ]()[]
        # The `_lib` field's origin parameter keeps `OwnedDLHandle` borrowed
        # for the lifetime of `self`, which spans this entire method — so
        # `dlclose` cannot run before `typed_fn` returns.
        return typed_fn(*args)


struct OwnedDLHandle(Movable):
    """Represents an owned handle to a dynamically linked library with RAII
    semantics.

    `OwnedDLHandle` owns the library handle and automatically calls `dlclose()`
    when the object is destroyed. This prevents resource leaks and double-free
    bugs.

    Example usage:
    ```mojo
    from std.ffi import OwnedDLHandle

    def main() raises:
        var lib = OwnedDLHandle("libm.so")
        var sqrt = lib.get_function[Float64]("sqrt")
        print(sqrt(4.0))  # Prints: 2.0
        # Library automatically closed when lib goes out of scope
    ```
    """

    var _handle: _DLHandle

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self, flags: Int = DEFAULT_RTLD) raises:
        """Initialize an owned handle to all global symbols in the current
        process.

        Args:
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(nullptr, flags)` fails.
        """
        self._handle = _DLHandle(flags)

    def __init__[
        PathLike: stdPathLike, //
    ](out self, path: PathLike, flags: Int = DEFAULT_RTLD) raises:
        """Initialize an OwnedDLHandle by loading the dynamic library at the
        given path.

        Parameters:
            PathLike: The type conforming to the `os.PathLike` trait.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(path, flags)` fails.
        """
        self._handle = _DLHandle(path, flags)

    @doc_hidden
    @always_inline
    def __init__(out self, *, unsafe_uninitialized: Bool):
        self._handle = _DLHandle({})

    def __deinit__(deinit self):
        """Unload the associated dynamic library.

        This automatically calls `dlclose()` on the underlying library handle.
        """
        self._handle.close()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def borrow(self) -> _DLHandle:
        """Returns a non-owning reference to this handle.

        The returned `_DLHandle` does not own the library and should not be
        used after this `OwnedDLHandle` is destroyed.

        Returns:
            A non-owning reference to the library handle.
        """
        return self._handle

    def __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
            `True` if the handle is not null and `False` otherwise.
        """
        return self._handle.__bool__()

    def check_symbol(self, var name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        return self._handle.check_symbol(name)

    def get_function[
        return_type: RegisterPassable = NoneType,
    ](ref self, var name: String) raises -> _DLCallable[
        return_type, origin_of(self)
    ]:
        """Returns a callable for the function with the given name in the
        dynamic library.

        The returned callable carries an immutable borrow of `self`, so the
        library cannot be `dlclose`d until after the callable is invoked.
        This prevents the dangling-function-pointer crash that would occur
        if the raw function pointer were returned directly and ASAP
        destruction ran `dlclose` between `dlsym` and the call.

        Argument forwarding uses the C ABI (see the `_DLCallable` notes).

        Missing symbols raise `Error("symbol not found: ...")` rather than
        aborting the process, so callers can probe for optional symbols.

        Example:
        ```mojo
        from std.ffi import OwnedDLHandle

        var lib = OwnedDLHandle("libm.so")
        var sqrt = lib.get_function[Float64]("sqrt")
        print(sqrt(4.0))  # 2.0
        ```

        Parameters:
            return_type: The return type of the underlying C function.
                Defaults to `NoneType` for void-returning functions.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A callable proxy that forwards to the resolved function and
            keeps the owning handle alive for the duration of each call.

        Raises:
            If the symbol cannot be resolved in the dynamic library.
        """
        # The type parameter used to be the full function pointer type
        # (e.g. `def(Float64) abi("C") -> Float64`). It is now just the
        # return type, matching `OwnedDLHandle.call`. Catch the old shape
        # at compile time to give users a clear migration pointer instead
        # of a confusing "function is not called" warning + silent no-op.
        comptime assert not __fn_type_is_cabi[return_type](), (
            "OwnedDLHandle.get_function now takes the return type only,"
            " not the full function type. For scalar args/returns, change"
            ' `get_function[def(Arg) abi("C") -> Ret](name)` to'
            " `get_function[Ret](name)`."
        )
        var ptr = self._handle.get_symbol[NoneType](
            cstr_name=name.as_c_string_slice()
        )
        if not ptr:
            raise Error(t"symbol not found: {name}")
        return _DLCallable[return_type, origin_of(self)](
            ptr.unsafe_value().unsafe_origin_cast[MutUntrackedOrigin](),
            Pointer(to=self),
        )

    @always_inline
    def _get_function[
        func_name: StaticString, result_type: TrivialRegisterPassable
    ](self) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            func_name: The name of the function to get the handle for.
            result_type: The type of the function pointer to return.

        Returns:
            A handle to the function.
        """
        return self._handle._get_function[func_name, result_type]()

    @always_inline
    def _get_function[
        result_type: TrivialRegisterPassable
    ](self, *, cstr_name: CStringSlice[_]) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            cstr_name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        return self._handle._get_function[result_type](cstr_name=cstr_name)

    def get_symbol[
        mut: Bool, origin: Origin[mut=mut], //, result_type: AnyType
    ](ref[origin] self, name: StringSlice) -> Optional[
        Pointer[result_type, origin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        The returned pointer borrows `self`, so the library cannot be
        `dlclose`d while the pointer is live. Its mutability follows the
        handle's: a symbol resolved through an immutable handle is read-only.

        Example:
        ```mojo
        from std.ffi import OwnedDLHandle, c_int

        var lib = OwnedDLHandle("libcounters.so")
        var counter = lib.get_symbol[c_int]("live_connections")
        if counter:
            print(counter.value()[])
        ```

        Parameters:
            mut: The mutability of self.
            origin: The origin of self.
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        var name_copy = String(name)
        return self.get_symbol[result_type](
            cstr_name=name_copy.as_c_string_slice()
        )

    def get_symbol[
        mut: Bool, origin: Origin[mut=mut], //, result_type: AnyType
    ](ref[origin] self, *, cstr_name: CStringSlice[_]) -> Optional[
        Pointer[result_type, origin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        See the `name` overload for how the returned pointer's origin and
        mutability relate to the handle.

        Parameters:
            mut: The mutability of self.
            origin: The origin of self.
            result_type: The type of the symbol to return.

        Args:
            cstr_name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        var res = self._handle.get_symbol[result_type](cstr_name=cstr_name)
        if not res:
            return None
        # `_DLHandle` is non-owning, so it can only hand back an untracked
        # origin. Re-tie the pointer to this owning handle, which is the borrow
        # `_DLHandle` has no way to express.
        return (
            res.unsafe_value()
            .unsafe_mut_cast[mut]()
            .unsafe_origin_cast[origin]()
        )

    @always_inline
    def call[
        name: StaticString,
        return_type: RegisterPassable = NoneType,
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
        return self._handle.call[name, return_type](*args)


def __fn_type_is_cabi[T: AnyType]() -> Bool:
    """Returns `True` if `T` is a function pointer type with the `abi("C")` effect.

    This is used to enforce that `DLHandle.get_function` is called with an
    explicit C-ABI function pointer type.  A plain Mojo function type (without
    `abi("C")`) returns `False`.

    Parameters:
        T: The type to check.

    Returns:
        `True` if `T` has the `abi("C")` effect, `False` otherwise.
    """
    return __mlir_attr[
        `#kgen.fn_type_is_cabi<`,
        T,
        `> : i1`,
    ]


@fieldwise_init
struct _DLHandle(Boolable, ImplicitlyCopyable, RegisterPassable):
    """Represents a non-owning reference to a dynamically linked library.

    `_DLHandle` is a lightweight, trivially copyable reference to a dynamic
    library. It does not own the library handle and multiple copies can safely
    reference the same library.

    For automatic resource management with RAII semantics, use `OwnedDLHandle`
    instead, which automatically calls `dlclose()` when destroyed.

    Notes:
        If you manually call `close()` on a `_DLHandle`, be careful not to use
        any copies of that handle afterward, as they will reference a closed
        library. For safer usage, prefer `OwnedDLHandle`.
    """

    var handle: OptionalPointer[NoneType, MutUntrackedOrigin]
    """The handle to the dynamic library."""

    @always_inline
    def __init__(out self, flags: Int = DEFAULT_RTLD) raises:
        """Initialize a dynamic library handle to all global symbols in the
        current process.

        Args:
            flags: The flags to load the dynamic library.

        Notes:
            On POSIX-compatible operating systems, this performs
            `dlopen(nullptr, flags)`.

        Raises:
            If `dlopen(nullptr, flags)` fails.
        """
        self = Self._dlopen(
            OptionalPointer[c_char, ImmUntrackedOrigin](), flags
        )

    def __init__[
        PathLike: stdPathLike, //
    ](out self, path: PathLike, flags: Int = DEFAULT_RTLD) raises:
        """Initialize a DLHandle object by loading the dynamic library at the
        given path.

        Parameters:
            PathLike: The type conforming to the `os.PathLike` trait.

        Args:
            path: The path to the dynamic library file.
            flags: The flags to load the dynamic library.

        Raises:
            If `dlopen(path, flags)` fails.
        """

        var fspath = path.__fspath__()
        var file = (
            fspath.as_c_string_slice()
            .ptr()
            .as_imm()
            .unsafe_origin_cast[ImmUntrackedOrigin]()
        )
        var handle = Self._dlopen(file, flags)
        # `file` carries an untracked origin, so nothing ties it back to
        # `fspath`. Without this, the compiler could destroy `fspath` before
        # `dlopen` copies the filename, freeing the buffer out from under it.
        _ = fspath^
        self = handle

    @staticmethod
    def _dlopen(
        file: OptionalPointer[mut=False, c_char, ImmUntrackedOrigin], flags: Int
    ) raises -> _DLHandle:
        var handle = dlopen(file, Int32(flags))
        if not handle:
            var error_message = dlerror()
            var message = StringSlice(
                unsafe_from_utf8=CStringSlice(
                    unsafe_from_ptr=error_message.value().as_imm()
                )
            ) if error_message else {}
            raise Error("dlopen failed: ", message)
        return _DLHandle(handle)

    def check_symbol(self, var name: String) -> Bool:
        """Check that the symbol exists in the dynamic library.

        Args:
            name: The symbol to check.

        Returns:
            `True` if the symbol exists.
        """
        var opaque_function_ptr = dlsym(
            self.handle,
            name.as_c_string_slice().ptr(),
        )

        return Bool(opaque_function_ptr)

    def close(mut self):
        """Unload the associated dynamic library.

        Warning:
            Since `DLHandle` is trivially copyable, multiple copies of this
            handle may exist. After calling `close()`, all copies will reference
            an invalid library handle. For safer resource management, prefer
            using `OwnedDLHandle` which automatically manages the library
            lifetime.
        """
        _ = dlclose(self.handle)
        self.handle = {}

    def __bool__(self) -> Bool:
        """Checks if the handle is valid.

        Returns:
          True if the DLHandle is not null and False otherwise.
        """
        return Bool(self.handle)

    def get_function[
        result_type: TrivialRegisterPassable
    ](self, var name: String) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The C-ABI function pointer type to return.

        Args:
            name: The name of the function to get the handle for.

        Returns:
            A handle to the function.

        Constraints:
            `result_type` must be a function pointer type annotated with
            `abi("C")` (e.g. `def(Float64) thin abi("C") -> Float64`). Using a
            plain Mojo function type causes silent ABI corruption for struct
            arguments and return values.
        """
        comptime assert __fn_type_is_cabi[result_type](), (
            'result_type must be a C-ABI function pointer type: use abi("C") on'
            ' the function type, e.g. `def(Float64) thin abi("C") -> Float64`'
        )

        return self._get_function[result_type](
            cstr_name=name.as_c_string_slice()
        )

    @always_inline
    def _get_function[
        func_name: StaticString, result_type: TrivialRegisterPassable
    ](self) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            func_name:The name of the function to get the handle for.
            result_type: The type of the function pointer to return.

        Returns:
            A handle to the function.
        """
        # Force unique the func_name so we know that it is nul-terminated.
        comptime func_name_literal = get_static_string[func_name]()
        return self._get_function[result_type](
            cstr_name=func_name_literal.as_c_string_slice(),
        )

    @always_inline
    def _get_function[
        result_type: TrivialRegisterPassable
    ](self, *, cstr_name: CStringSlice[_]) -> result_type:
        """Returns a handle to the function with the given name in the dynamic
        library.

        Parameters:
            result_type: The type of the function pointer to return.

        Args:
            cstr_name: The name of the function to get the handle for.

        Returns:
            A handle to the function.
        """
        var opaque_function_ptr = self.get_symbol[NoneType](cstr_name=cstr_name)

        if not opaque_function_ptr:
            abort(t"symbol not found: {cstr_name}")

        return Pointer(to=opaque_function_ptr.value()).unsafe_bitcast[
            result_type
        ]()[]

    def get_symbol[
        result_type: AnyType,
    ](self, name: StringSlice) -> Optional[
        Pointer[result_type, MutUntrackedOrigin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        var name_copy = String(name)
        return self.get_symbol[result_type](
            cstr_name=name_copy.as_c_string_slice()
        )

    def get_symbol[
        result_type: AnyType
    ](self, *, cstr_name: CStringSlice[_]) -> Optional[
        Pointer[result_type, MutUntrackedOrigin]
    ]:
        """Returns a pointer to the symbol with the given name in the dynamic
        library, or `None` if the symbol is not found.

        Parameters:
            result_type: The type of the symbol to return.

        Args:
            cstr_name: The name of the symbol to get the handle for.

        Returns:
            An optional pointer to the symbol, or `None` if not found.
        """
        debug_assert(
            Bool(self.handle),
            "Dylib handle is null when loading symbol: ",
            cstr_name,
        )

        # Follow the dance described in
        # https://man7.org/linux/man-pages/man3/dlsym.3.html to distinguish
        # a symbol that was not found from a symbol whose value is NULL:
        #
        # 1. Clear any old error with dlerror()
        # 2. Call dlsym()
        # 3. Call dlerror() again — if it returns non-NULL, an error occurred

        # Clear any pre-existing error.
        _ = dlerror()

        var res: Optional[Pointer[result_type, MutUntrackedOrigin]] = dlsym[
            result_type
        ](self.handle, cstr_name.ptr())

        if not res:
            # Result is NULL — check if it's an error or a valid NULL symbol.
            var err = dlerror()
            if err:
                # Symbol lookup failed.
                return None

            # Symbol is validly NULL (unusual but possible per dlsym docs).
            # Abort rather than returning a null pointer wrapped in Some,
            # which would be misleading. Callers who need to handle NULL
            # symbols should specify a nullable pointer as the result_type.
            abort(t"symbol resolved to NULL: {cstr_name}")

        return res.value()

    @always_inline
    def call[
        name: StaticString,
        return_type: RegisterPassable = NoneType,
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

        def _check_symbol() {self} -> Bool:
            return self.check_symbol(String(name))

        debug_assert(_check_symbol, "symbol not found: ", name)
        # The callee type is spelled over the argument pack so that codegen
        # applies the C calling convention per argument, as in
        # `_DLCallable.__call__`. The call is synchronous, so the handle stays
        # alive across it without a borrow.
        return self._get_function[
            name, def(* a: * T) thin abi("C") -> return_type
        ]()(*args)


@always_inline
def _get_dylib_function[
    dylib_global: _Global[StorageType=OwnedDLHandle, ...],
    func_name: StaticString,
    result_type: TrivialRegisterPassable,
]() raises -> result_type:
    var func_cache_name = String(t"{dylib_global.name}/{func_name}")
    var func_ptr = _get_global_or_null(func_cache_name)
    if func_ptr:
        return Pointer(to=func_ptr).unsafe_bitcast[result_type]()[]

    var dylib = dylib_global.get_or_create_ptr()[].borrow()
    var new_func = dylib._get_function[func_name, result_type]()

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(func_cache_name),
        Pointer(to=new_func).unsafe_bitcast[
            OpaquePointer[MutUntrackedOrigin]
        ]()[],
    )

    return new_func


def _try_find_dylib[
    name: StaticString = ""
](paths: List[Path]) raises -> OwnedDLHandle:
    """Try to load a dynamically linked library given a list of possible paths.

    Parameters:
        name: Optional name for the library to be used in error messages.

    Args:
        paths: A list of paths or library names to pass to the OwnedDLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.

    Raises:
        If the library could not be loaded from any of the provided paths.
    """
    comptime dylib_name = name if name != "" else "dynamic library"
    for path in paths:
        # If we are given a library name like libfoo.so, pass it directly to
        # dlopen(), which will invoke the system linker to find the library.
        # We can't check the existence of the path ahead of time, we have to
        # call the function and check for an error.
        try:
            return OwnedDLHandle(String(path))
        except:
            # If the call to DLOpen fails, we should just try the next path
            # in the list. It's only a fatal error if the library cannot be
            # loaded from any of the paths provided.
            pass

    raise Error("Failed to load ", dylib_name, " from ", " or ".join(paths))


def _try_find_dylib[
    name: StaticString = ""
](*paths: Path) raises -> OwnedDLHandle:
    """Load a dynamically linked library given a variadic list of possible names.
    """
    # Convert the variadic pack to a list.
    return _try_find_dylib[name](
        List(length=len(paths), fill_with=lambda (i: Int) -> Path: paths[i])
    )


def _find_dylib[
    name: StaticString = "", abort_on_failure: Bool = True
](paths: List[Path]) -> OwnedDLHandle:
    """Load a dynamically linked library given a list of possible paths or names.

    If the library is not found, the function will abort.

    Parameters:
        name: Optional name for the library to be used in error messages.
        abort_on_failure: If set, then the function will abort the program if
           the library is not found. Otherwise, we return a null OwnedDLHandle
           on failure.

    Args:
        paths: A list of paths or library names to pass to the OwnedDLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.
    """
    try:
        return _try_find_dylib[name](paths)
    except e:
        comptime if abort_on_failure:
            abort(String(e))
        else:
            return OwnedDLHandle(unsafe_uninitialized=True)


def _find_dylib[
    msg: def() thin -> String, abort_on_failure: Bool = True
](paths: List[Path]) -> OwnedDLHandle:
    """Load a dynamically linked library given a list of possible paths or names.

    If the library is not found, the function will abort.

    Parameters:
        msg: A function that produces the error message to use if the
             library cannot be found.
        abort_on_failure: If set, then the function will abort the program if
           the library is not found. Otherwise, we return a null OwnedDLHandle
           on failure.

    Args:
        paths: A list of paths or library names to pass to the OwnedDLHandle
               constructor.

    Returns:
        A handle to the loaded dynamic library.
    """
    try:
        return _try_find_dylib(paths)
    except e:
        comptime if abort_on_failure:
            abort[prefix="ERROR:"](msg())
        else:
            return OwnedDLHandle(unsafe_uninitialized=True)


def _find_dylib[name: StaticString = ""](*paths: Path) -> OwnedDLHandle:
    """Load a dynamically linked library given a variadic list of possible names.
    """
    # Convert the variadic pack to a list.
    var paths_list = List[Path]()
    for path in paths:
        paths_list.append(path)
    return _find_dylib[name](paths_list)


# ===-----------------------------------------------------------------------===#
# Globals
# ===-----------------------------------------------------------------------===#


# NOTE: This is vending shared mutable pointers to the client without locking.
# This is not guaranteeing any sort of thread safety.
struct _Global[
    StorageType: Movable,
    //,
    name: StaticString,
    init_fn: def() thin -> StorageType,
    on_error_msg: Optional[def() thin -> Error] = None,
](Defaultable):
    comptime ResultType = Pointer[Self.StorageType, MutUntrackedOrigin]

    def __init__(out self):
        pass

    @staticmethod
    def _init_wrapper() -> OptionalPointer[NoneType, UntrackedOrigin[mut=True]]:
        # Heap allocate space to store this "global"
        # TODO:
        #   Any way to avoid the move, e.g. by calling this function
        #   with the ABI destination result pointer already set to `ptr`?
        var ptr = OwnedPointer(Self.init_fn())

        var storage = ptr^.unsafe_take_allocation().unsafe_leak()
        var opaque = storage.unsafe_bitcast[NoneType]()
        return opaque

    @staticmethod
    def _deinit_wrapper(
        opaque_ptr: OptionalPointer[NoneType, UntrackedOrigin[mut=True]]
    ):
        # Deinitialize and deallocate the storage.
        if opaque_ptr:
            dealloc(
                ThinAllocation(
                    unsafe_owned_ptr=opaque_ptr.unsafe_value().unsafe_bitcast[
                        Self.StorageType
                    ]()
                ).unsafe_with_layout({count = 1})
            )

    @staticmethod
    def get_or_create_ptr() raises -> Self.ResultType:
        var ptr = _get_global[
            Self.name, Self._init_wrapper, Self._deinit_wrapper
        ]()

        comptime if Self.on_error_msg:
            if not ptr:
                raise Self.on_error_msg.value()()

        return unsafe_cast[Type=Self.StorageType](ptr).value()

    # Currently known values for get_or_create_indexed_ptr. See
    # NUM_INDEXED_GLOBALS in CompilerRT.
    # 0: Python runtime context
    # 1: GPU comm P2P availability cache
    # 2: Intentionally unused (reserved for prototyping / future use)
    comptime _python_idx = 0
    comptime _gpu_comm_p2p_idx = 1
    comptime _unused = 2  # Intentionally unused (enabled for prototyping).

    # This accesses a well-known global with a fixed index rather than using a
    # name to unique the value.  The index table is above.
    @staticmethod
    def get_or_create_indexed_ptr(idx: Int) raises -> Self.ResultType:
        var ptr = external_call[
            "KGEN_CompilerRT_GetOrCreateGlobalIndexed",
            OptionalPointer[NoneType, UntrackedOrigin[mut=True]],
        ](
            idx,
            Self._init_wrapper,
            Self._deinit_wrapper,
        )

        comptime if Self.on_error_msg:
            if not ptr:
                raise Self.on_error_msg.value()()

        return unsafe_cast[Type=Self.StorageType](ptr).value()


@always_inline
def _get_global[
    name: StaticString,
    init_fn: def() thin -> OptionalPointer[NoneType, UntrackedOrigin[mut=True]],
    destroy_fn: def(
        OptionalPointer[NoneType, UntrackedOrigin[mut=True]]
    ) thin -> None,
]() -> OptionalPointer[NoneType, UntrackedOrigin[mut=True]]:
    return external_call[
        "KGEN_CompilerRT_GetOrCreateGlobal",
        OptionalPointer[NoneType, UntrackedOrigin[mut=True]],
    ](
        name,
        init_fn,
        destroy_fn,
    )


@always_inline
def _get_global_or_null(
    name: StringSlice,
) -> OptionalPointer[NoneType, UntrackedOrigin[mut=True]]:
    return external_call[
        "KGEN_CompilerRT_GetGlobalOrNull",
        OptionalPointer[NoneType, UntrackedOrigin[mut=True]],
    ](name.as_bytes().unsafe_ptr(), name.byte_length())


# ===-----------------------------------------------------------------------===#
# external_call
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def external_call[
    callee: StaticString,
    return_type: RegisterPassable,
    *types: AnyType,
    num_fixed_args: Optional[Int] = None,
](*args: *types) -> return_type:
    """Calls an external function.

    By default every argument is a fixed argument of a non-variadic callee.
    Pass `num_fixed_args` to call a C variadic function such as `open()` or
    `snprintf()`, giving how many of the first arguments are fixed arguments of
    the callee; the arguments after those are passed as variadic arguments.
    This matters because ABIs such as AAPCS on ARM64 macOS pass variadic
    arguments differently from fixed ones.

    Examples:

    ```mojo
    from std.ffi import c_char, c_int, external_call

    # int open(const char *path, int oflag, ...);
    var path_str = path
    var fd = external_call["open", c_int, num_fixed_args=2](
        path_str.as_c_string_slice(), c_int(flags), c_int(0o666)
    )
    ```

    Args:
        args: The arguments to pass to the external function.

    Parameters:
        callee: The name of the external function.
        return_type: The return type.
        types: The argument types.
        num_fixed_args: The number of fixed arguments of a C variadic callee,
            or `None` for a non-variadic callee. A count of `0` declares a
            callee whose every argument is variadic, which `None` cannot
            express. An argument pack is flattened into one argument per
            element, so the count must cover the elements rather than the
            pack, and the fixed arguments must precede any pack.

    Constraints:
        `num_fixed_args` must not be negative.

    Returns:
        The external call result.
    """

    comptime assert num_fixed_args.or_else(0) >= 0, (
        "`num_fixed_args` counts the fixed arguments of the C variadic callee,"
        " so it cannot be negative; use `None` for a non-variadic callee"
    )

    comptime if types.contains[String]():
        comptime assert False, (
            "Passing a `String` to `external_call` is never correct. Instead,"
            " first call `as_c_string_slice()` to pass a C-FFI compatible"
            " `CStringSlice` type (synonymous with C's `const char*`). Example"
            ' `external_call["foo", NoneType]("some'
            ' string".as_c_string_slice())'
        )

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()
    comptime callee_kgen_string = _get_kgen_string[callee]()

    # The presence of the `numFixedArgs` attribute — not its value — is what
    # declares the callee variadic, so `None` maps to the attribute being
    # absent while `num_fixed_args=0` maps to a count of zero, which the op
    # reads as a callee whose every argument is variadic. Hence the duplicated
    # calls. A negative count takes the attribute-free path too, since the
    # assert above has already failed the build and emitting the attribute
    # would bury that message under the op verifier's error.
    comptime declares_variadic = num_fixed_args.or_else(-1) >= 0
    comptime fixed_args = num_fixed_args.or_else(0)

    comptime if return_type == NoneType:
        comptime if declares_variadic:
            __mlir_op.`pop.external_call`[
                func=callee_kgen_string,
                numFixedArgs=fixed_args.__mlir_index__(),
                _type=None,
            ](loaded_pack)
        else:
            __mlir_op.`pop.external_call`[func=callee_kgen_string, _type=None](
                loaded_pack
            )
        return rebind_var[return_type](NoneType())
    else:
        comptime if declares_variadic:
            return __mlir_op.`pop.external_call`[
                func=callee_kgen_string,
                numFixedArgs=fixed_args.__mlir_index__(),
                _type=return_type,
            ](loaded_pack)
        else:
            return __mlir_op.`pop.external_call`[
                func=callee_kgen_string,
                _type=return_type,
            ](loaded_pack)


# ===-----------------------------------------------------------------------===#
# _external_call_const
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def _external_call_const[
    callee: StaticString,
    return_type: TrivialRegisterPassable,
    *types: AnyType,
](*args: *types) -> return_type:
    """Mark the external function call as having no observable effects to the
    program state. This allows the compiler to optimize away successive calls
    to the same function.

    Args:
      args: The arguments to pass to the external function.

    Parameters:
      callee: The name of the external function.
      return_type: The return type.
      types: The argument types.

    Returns:
      The external call result.
    """

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()

    return __mlir_op.`pop.external_call`[
        func=_get_kgen_string[callee](),
        resAttrs=__mlir_attr.`[{llvm.noundef}]`,
        funcAttrs=__mlir_attr.`["willreturn"]`,
        memory=__mlir_attr[
            `#llvm.memory_effects<other = none, `,
            `argMem = none, `,
            `inaccessibleMem = none, `,
            `errnoMem = none, `,
            `targetMem0 = none, `,
            `targetMem1 = none>`,
        ],
        _type=return_type,
    ](loaded_pack)
