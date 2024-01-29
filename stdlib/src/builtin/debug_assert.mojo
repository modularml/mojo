# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements a debug assert.

These are Mojo built-ins, so you don't need to import them.
"""


from sys._build import is_kernels_debug_build
from sys.param_env import is_defined

from debug import trap


@always_inline
fn debug_assert(cond: Bool, msg: StringLiteral):
    """Asserts that the condition is true.

    The `debug_assert` is similar to `assert` in C++. It is a no-op in release
    builds unless MOJO_ENABLE_ASSERTIONS is defined.

    Right now, users of the mojo-sdk must explicitly specify `-D MOJO_ENABLE_ASSERTIONS`
    to enable assertions.  It is not sufficient to compile programs with `-debug-level full`
    for enabling assertions in the library.

    Args:
        cond: The bool value to assert.
        msg: The message to display on failure.
    """
    _debug_assert_impl(cond, msg)


@always_inline
fn debug_assert[boolable: Boolable](cond: boolable, msg: StringLiteral):
    """Asserts that the condition is true.

    The `debug_assert` is similar to `assert` in C++. It is a no-op in release
    builds unless MOJO_ENABLE_ASSERTIONS is defined.

    Right now, users of the mojo-sdk must explicitly specify `-D MOJO_ENABLE_ASSERTIONS`
    to enable assertions.  It is not sufficient to compile programs with `-debug-level full`
    for enabling assertions in the library.

    Parameters:
        boolable: The trait of the conditional.

    Args:
        cond: The bool value to assert.
        msg: The message to display on failure.
    """
    _debug_assert_impl(cond, msg)


@always_inline
fn _debug_assert_impl[boolable: Boolable](cond: boolable, msg: StringLiteral):
    """Asserts that the condition is true."""

    # Print an error and fail.
    alias err = is_kernels_debug_build() or is_defined[
        "MOJO_ENABLE_ASSERTIONS"
    ]()

    # Print a warning, but do not fail (useful for testing assert behavior).
    alias warn = is_defined["ASSERT_WARNING"]()

    @parameter
    if err or warn:
        if cond.__bool__():
            return

        @parameter
        if err:
            print("Assert Error:", msg)
            trap()
        else:
            print("Assert Warning:", msg)
