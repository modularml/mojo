# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the trap functions.
"""


@always_inline("nodebug")
fn trap():
    """Calls a target dependent trap instruction. If the target does not have a
    trap instruction, this intrinsic will be lowered to a call of the abort()
    function."""
    __mlir_op.`llvm.intr.trap`()


@always_inline("nodebug")
fn trap[T: Stringable](message: T):
    """Prints a message before calling a target dependent trap instruction.
    If the target does not have a trap instruction, this intrinsic will be
    lowered to a call of the abort() function."""
    print(message)
    __mlir_op.`llvm.intr.trap`()


@always_inline("nodebug")
fn debug_trap():
    """Cause an execution trap with the intention of requesting the attention
    of a debugger."""
    __mlir_op.`llvm.intr.debugtrap`()
