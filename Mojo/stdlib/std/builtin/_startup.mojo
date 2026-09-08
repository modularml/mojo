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
"""Implements functionality to start a mojo execution."""

from std.ffi import external_call, _get_global
from std.sys.compile import SanitizeAddress
from std.sys import stderr


def _init_global_runtime() -> (
    OptionalPointer[NoneType, UntrackedOrigin[mut=True]]
):
    return external_call[
        "KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice",
        OptionalPointer[NoneType, UntrackedOrigin[mut=True]],
    ]()


def _destroy_global_runtime(
    ptr: OptionalPointer[NoneType, UntrackedOrigin[mut=True]]
):
    """Destroy the global runtime if ever used."""
    external_call["KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice", NoneType](ptr)


@always_inline
def _ensure_runtime_init():
    var current_runtime = external_call[
        "KGEN_CompilerRT_AsyncRT_GetCurrentCPUDevice",
        OptionalPointer[NoneType, UntrackedOrigin[mut=True]],
    ]()
    if current_runtime:
        return
    _ = _get_global["Runtime", _init_global_runtime, _destroy_global_runtime]()


def __wrap_and_execute_main[
    main_func: def() thin -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for non-raising main function."""

    # Initialize the global runtime.
    _ensure_runtime_init()

    comptime if SanitizeAddress:
        external_call["KGEN_CompilerRT_SetAsanAllocators", NoneType]()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Initialize signal handler for SIGSEGV  SIGABRT that will print a stack
    # trace unless the `max-debug.stack-trace-on-crash` Config key is
    # disabled.  This functionality is gated because otherwise an extra signal
    # handler will be registered when the user runs code with a sanitizer
    # enabled, which would lead to duplicate stack traces being printed.
    external_call["KGEN_CompilerRT_PrintStackTraceOnFault", NoneType]()

    # Call into the user main function.
    main_func()

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


def __wrap_and_execute_raising_main[
    main_func: def() thin raises -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for a raising main function."""

    # Initialize the global runtime.
    _ensure_runtime_init()

    comptime if SanitizeAddress:
        external_call["KGEN_CompilerRT_SetAsanAllocators", NoneType]()

    # Initialize the mojo argv with those provided.
    external_call["KGEN_CompilerRT_SetArgV", NoneType](argc, argv)

    # Initialize signal handler for SIGSEGV  SIGABRT that will print a stack
    # trace unless the `max-debug.stack-trace-on-crash` Config key is
    # disabled.  This functionality is gated because otherwise an extra signal
    # handler will be registered when the user runs code with a sanitizer
    # enabled, which would lead to duplicate stack traces being printed.
    external_call["KGEN_CompilerRT_PrintStackTraceOnFault", NoneType]()

    # Call into the user main function.
    try:
        main_func()
    except e:
        var stack_trace = e.get_stack_trace()
        if stack_trace:
            print(stack_trace.value(), file=stderr)
        else:
            print(
                (
                    "stack trace was not collected. Enable stack trace"
                    " collection with environment variable"
                    " `MODULAR_DEBUG=stack-trace-on-error`"
                ),
                file=stderr,
            )
        print("Unhandled exception caught during execution:", e, file=stderr)
        return 1

    # Delete any globals we have allocated.
    external_call["KGEN_CompilerRT_DestroyGlobals", NoneType]()

    # Return OK.
    return 0


# A prototype of the main entry point, used by the compiled when synthesizing
# main.
def __mojo_main_prototype(
    argc: Int32, argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`]
) -> Int32:
    return 0
