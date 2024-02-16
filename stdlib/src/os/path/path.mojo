# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the os.path operations.

You can import these APIs from the `os.path` package. For example:

```mojo
from os.path import isdir
```
"""

from .._macos import _stat as _stat_macos, _lstat as _lstat_macos
from .._linux import _stat as _stat_linux, _lstat as _lstat_linux
from ..stat import _S_ISDIR, _S_ISLNK
from sys.info import os_is_windows, os_is_linux, os_is_macos


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #
fn _constrain_unix():
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()


@always_inline
fn _get_stat_st_mode(path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return int(_stat_macos(path).st_mode)
    else:
        return int(_stat_linux(path).st_mode)


@always_inline
fn _get_lstat_st_mode(path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return int(_lstat_macos(path).st_mode)
    else:
        return int(_lstat_linux(path).st_mode)


# ===----------------------------------------------------------------------=== #
# isdir
# ===----------------------------------------------------------------------=== #
fn isdir(path: String) -> Bool:
    """Return True if path is an existing directory. This follows
    symbolic links, so both islink() and isdir() can be true for the same path.

    Args:
      path: The path to the directory.

    Returns:
      True if the path is a directory or a link to a directory and
      False otherwise.
    """
    _constrain_unix()
    try:
        let st_mode = _get_stat_st_mode(path)
        return _S_ISDIR(st_mode) or _S_ISLNK(st_mode)
    except:
        return False


# ===----------------------------------------------------------------------=== #
# islink
# ===----------------------------------------------------------------------=== #
fn islink(path: String) -> Bool:
    """Return True if path refers to an existing directory entry that is a
    symbolic link.

    Args:
      path: The path to the directory.

    Returns:
      True if the path is a link to a directory and False otherwise.
    """
    _constrain_unix()
    try:
        return _S_ISLNK(_get_lstat_st_mode(path))
    except:
        return False
