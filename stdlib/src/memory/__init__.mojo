# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the memory package."""

from .memory import (
    AddressSpace,
    DTypePointer,
    memcmp,
    memcpy,
    parallel_memcpy,
    memset_zero,
    memset,
    Pointer,
    stack_allocation,
)
