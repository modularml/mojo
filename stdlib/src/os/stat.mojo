# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the file system stat operations.

You can import these APIs from the `os` package. For example:

```mojo
from os import stat
```
"""


alias __S_IFMT = 0o0170000  # These bits determine file type
alias __S_IFDIR = 0o040000  # Directory
alias __S_IFCHR = 0o020000  # Character device
alias __S_IFBLK = 0o060000  # Block device
alias __S_IFREG = 0o0100000  # Regular file
alias __S_IFIFO = 0o010000  # FIFO
alias __S_IFLNK = 0o0120000  # Symbolic link
alias __S_IFSOCK = 0o0140000  # Socket


fn _S_ISLNK[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFLNK


fn _S_ISREG[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFREG


fn _S_ISDIR[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFDIR


fn _S_ISCHR[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFCHR


fn _S_ISBLK[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFBLK


fn _S_ISFIFO[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFIFO


fn _S_ISSOCK[intable: Intable](m: intable) -> Bool:
    return (int(m) & __S_IFMT) == __S_IFSOCK
