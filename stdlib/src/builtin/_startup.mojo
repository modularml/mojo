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
"""Implements functionality to start a mojo execution."""

from sys import external_call


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


fn _init_global_runtime(
    ignored: UnsafePointer[NoneType],
) -> UnsafePointer[NoneType]:
    """Initialize the global runtime. This is a singleton that handle the common
    case where the runtime has the same number of threads as the number of cores.
    """
    return external_call[
        "KGEN_CompilerRT_LLCL_CreateRuntime", UnsafePointer[NoneType]
    ](0)


fn _destroy_global_runtime(ptr: UnsafePointer[NoneType]):
    """Destroy the global runtime if ever used."""
    external_call["KGEN_CompilerRT_LLCL_DestroyRuntime", NoneType](ptr)


@always_inline
fn _get_current_or_global_runtime() -> UnsafePointer[NoneType]:
    """Returns the current runtime, or returns the Mojo singleton global
    runtime, creating it if it does not already exist. When Mojo is used within
    the Modular Execution Engine the current runtime will be that already
    constructed by the execution engine. If the user has already manually
    constructed a runtime and added tasks to it, the current runtime for those
    tasks will be that runtime. Otherwise, the singleton runtime is used, which
    is created with number of threads equal to the number of cores.
    """
    var current_runtime = external_call[
        "KGEN_CompilerRT_LLCL_GetCurrentRuntime", UnsafePointer[NoneType]
    ]()
    if current_runtime:
        return current_runtime
    return _get_global[
        "Runtime", _init_global_runtime, _destroy_global_runtime
    ]()


fn __wrap_and_execute_main[
    main_func: fn () -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for non-raising main function"""

    # Initialize the global runtime.
    _ = _get_current_or_global_runtime()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Call into the user main function.
    main_func()

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


fn __wrap_and_execute_raising_main[
    main_func: fn () raises -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for a raising main function"""

    # Initialize the global runtime.
    _ = _get_current_or_global_runtime()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Call into the user main function.
    try:
        main_func()
    except e:
        print("Unhandled exception caught during execution:", e)
        return 1

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


fn __wrap_and_execute_object_raising_main[
    main_func: fn () raises -> object
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for a raising main function that
    returns an object"""

    fn wrapped_main() raises:
        _ = main_func()

    return __wrap_and_execute_raising_main[wrapped_main](argc, argv)


# A prototype of the main entry point, used by the compiled when synthesizing
# main.
fn __mojo_main_prototype(
    argc: Int32, argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`]
) -> Int32:
    return 0
