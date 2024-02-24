# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes the builtin breakpoint function."""

from sys import breakpointhook


@always_inline("nodebug")
fn breakpoint():
    """Cause an execution trap with the intention of requesting the attention
    of a debugger."""
    breakpointhook()
