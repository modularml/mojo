# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements support functions for working with Windows."""

from sys import external_call
from sys.info import os_is_windows


@always_inline
fn _assert_windows():
    constrained[os_is_windows(), "This operation is only valid in Windows."]()


@always_inline
fn reset_last_error():
    """Resets any existing error code previous operations."""
    _assert_windows()
    external_call["SetLastError", NoneType](0)


@always_inline
fn get_last_error_code() -> Int:
    """
    Returns the error code from last operation.

    Returns:
      The error code from last compvared operation. Returns 0
      if operation was success.
    """
    _assert_windows()
    return external_call["GetLastError", Int]()


@always_inline
fn last_operation_succeeded() -> Bool:
    """
    Returns if the last compvared operation was a success or not.

    Returns:
      Returns True on success, else returns False.
    """
    _assert_windows()
    return get_last_error_code() == 0
