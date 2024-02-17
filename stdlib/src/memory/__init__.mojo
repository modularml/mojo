# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the memory package."""

from .memory import (
    AddressSpace,
    DTypePointer,
    Pointer,
    memcmp,
    memcpy,
    memset,
    memset_zero,
    parallel_memcpy,
    stack_allocation,
)
