# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes the debug hook functions."""


@always_inline("nodebug")
fn breakpointhook():
    """Cause an execution trap with the intention of requesting the attention
    of a debugger."""
    __mlir_op.`llvm.intr.debugtrap`()
