# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements functionality to start a mojo execution."""

from sys import external_call


fn __wrap_and_execute_main[
    main_func: fn () -> None
](
    argc: Int32,
    argv: __mlir_type[`!kgen.pointer<!kgen.pointer<scalar<ui8>>>`],
) -> Int32:
    """Define a C-ABI compatible entry point for non-raising main function"""

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
