# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements PathLike trait.

You can import the trait from the `os` package. For example:

```mojo
from os import PathLike
```
"""


trait PathLike:
    """A trait representing file system paths."""

    fn __fspath__(self) -> String:
        """Return the file system path representation of the object.

        Returns:
          The file system path representation as a string.
        """
        ...
