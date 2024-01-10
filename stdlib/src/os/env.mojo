# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements basic routines for working with the OS.

You can import these APIs from the `os` package. For example:

```mojo
from os import setenv
```
"""

from sys import external_call
from sys.info import os_is_linux, os_is_macos

from memory.unsafe import DTypePointer


fn setenv(name: StringRef, value: StringRef, overwrite: Bool = True) -> Bool:
    """Changes or adds an environment variable.

    Constraints:
      The function only works on macOS or Linux and returns False otherwise.

    Args:
      name: The name of the environment variable.
      value: The value of the environment variable.
      overwrite: If an environment variable with the given name already exists,
        its value is not changed unless `overwrite` is True.

    Returns:
      False if the name is empty or contains an `=` character. In any other
      case, True is returned.
    """
    alias os_is_supported = os_is_linux() or os_is_macos()
    if not os_is_supported:
        return False

    let status = external_call["setenv", Int32](
        name.data, value.data, Int32(1 if overwrite else 0)
    )
    return status == 0


fn getenv(name: StringRef, default: StringRef) -> StringRef:
    """Returns the value of the given environment variable.

    Constraints:
      The function only works on macOS or Linux and returns an empty string
      otherwise.

    Args:
      name: The name of the environment variable.
      default: The default value to return if the environment variable
        doesn't exist.

    Returns:
      The value of the environment variable.
    """
    alias os_is_supported = os_is_linux() or os_is_macos()

    if not os_is_supported:
        return default

    let ptr = external_call["getenv", DTypePointer[DType.int8]](name.data)
    if not ptr:
        return default
    return StringRef(ptr)


fn getenv(name: StringRef) -> StringRef:
    """Returns the value of the given environment variable. If the
    environment variable is not found, then an empty string is returned.

    Constraints:
      The function only works on macOS or Linux and returns an empty string
      otherwise.

    Args:
      name: The name of the environment variable.

    Returns:
      The value of the environment variable.
    """
    return getenv(name, "")
