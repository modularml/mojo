# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements the stat module."""

alias S_IFMT = 0o0170000
"""Bits that determine the file type."""

alias S_IFDIR = 0o040000
"""Bits that determine the directory."""

alias S_IFCHR = 0o020000
"""Bits that determine the char device."""

alias S_IFBLK = 0o060000
"""Bits that determine the block device."""

alias S_IFREG = 0o0100000
"""Bits that determine the regular file."""

alias S_IFIFO = 0o010000
"""Bits that determine the fifo."""

alias S_IFLNK = 0o0120000
"""Bits that determine the symlink."""

alias S_IFSOCK = 0o0140000
"""Bits that determine the socket."""


fn S_ISLNK[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a symlink.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a symlink and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFLNK


fn S_ISREG[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a regular file.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a regular file and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFREG


fn S_ISDIR[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a directory.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a directory and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFDIR


fn S_ISCHR[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a character device.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a character device and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFCHR


fn S_ISBLK[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a block device.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a block device and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFBLK


fn S_ISFIFO[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a fifo.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a fifo and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFIFO


fn S_ISSOCK[intable: Intable](mode: intable) -> Bool:
    """
    Returns True if the mode is a socket.

    Parameters:
      intable: A type conforming to Intable.

    Args:
      mode: The file mode.

    Returns:
      True if the mode is a socket and False otherwise.
    """
    return (int(mode) & S_IFMT) == S_IFSOCK
