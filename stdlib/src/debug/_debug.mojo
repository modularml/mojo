# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the abort functions.
"""

from sys.info import triple_is_nvidia_cuda


@always_inline("nodebug")
fn abort():
    """Calls a target dependent trap instruction. If the target does not have a
    trap instruction, this intrinsic will be lowered to a call of the abort()
    function."""

    __mlir_op.`llvm.intr.trap`()


@always_inline("nodebug")
fn abort[T: Stringable](message: T):
    """Prints a message before calling a target dependent trap instruction.
    If the target does not have a trap instruction, this intrinsic will be
    lowered to a call of the abort() function."""

    @parameter
    if not triple_is_nvidia_cuda():
        print(message)
    __mlir_op.`llvm.intr.trap`()
