# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides utilities for enhancing visualizations of variables in LLDB and the
Mojo REPL.

These utilities automate the interaction with the LLDB data formatting system.
"""


fn lldb_formatter_wrapping_type():
    """Replace the visualization of the decorated struct with the one of its
    first field. If the decorated struct has no fields, an empty variable is displayed instead.
    """
    return
