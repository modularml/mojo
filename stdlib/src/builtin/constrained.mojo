# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements compile time contraints.

These are Mojo built-ins, so you don't need to import them.
"""


@always_inline("nodebug")
fn constrained[cond: Bool, msg: StringLiteral = "param assertion failed"]():
    """Compile time checks that the condition is true.

    The `constrained` is similar to `static_assert` in C++ and is used to
    introduce constraints on the enclosing function. In Mojo, the assert places
    a constraint on the function. The message is displayed when the assertion
    fails.

    Parameters:
        cond: The bool value to assert.
        msg: The message to display on failure.
    """
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(), message = msg.value
    ]()
    return
