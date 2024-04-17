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
"""Implements the file system stat operations.

You can import these APIs from the `os` package. For example:

```mojo
from os import stat
```
"""

from sys import has_neon, os_is_linux, os_is_macos, os_is_windows
from time.time import _CTimeSpec

from . import PathLike
from ._linux_aarch64 import _lstat as _lstat_linux_arm
from ._linux_aarch64 import _stat as _stat_linux_arm
from ._linux_x86 import _lstat as _lstat_linux_x86
from ._linux_x86 import _stat as _stat_linux_x86
from ._macos import _lstat as _lstat_macos
from ._macos import _stat as _stat_macos


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #
fn _constrain_unix():
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()


# ===----------------------------------------------------------------------=== #
# stat_result
# ===----------------------------------------------------------------------=== #


@value
struct stat_result(Stringable):
    """Object whose fields correspond  to the members of the stat structure."""

    var st_mode: Int
    """File mode: file type and file mode bits (permissions)."""

    var st_ino: Int
    """Platform dependent, but if non-zero, uniquely identifies the file for
    a given value of st_dev."""

    var st_dev: Int
    """Identifier of the device on which this file resides."""

    var st_nlink: Int
    """Number of hard links."""

    var st_uid: Int
    """User identifier of the file owner."""

    var st_gid: Int
    """Group identifier of the file owner."""

    var st_size: Int
    """Size of the file in bytes, if it is a regular file or a symbolic link."""

    var st_atimespec: _CTimeSpec
    """Time of file most recent access."""

    var st_mtimespec: _CTimeSpec
    """Time of file most recent modification."""

    var st_ctimespec: _CTimeSpec
    """Time of file most recent change."""

    var st_birthtimespec: _CTimeSpec
    """Time of file creation."""

    var st_blocks: Int
    """Number of 512-byte blocks allocated for file."""

    var st_blksize: Int
    """Preferred blocksize for efficient file system I/O."""

    var st_rdev: Int
    """Type of device if an inode device."""

    var st_flags: Int
    """User defined flags for file."""

    fn __init__(
        inout self,
        /,
        *,
        st_mode: Int,
        st_ino: Int,
        st_dev: Int,
        st_nlink: Int,
        st_uid: Int,
        st_gid: Int,
        st_size: Int,
        st_atimespec: _CTimeSpec,
        st_mtimespec: _CTimeSpec,
        st_ctimespec: _CTimeSpec,
        st_birthtimespec: _CTimeSpec,
        st_blocks: Int,
        st_blksize: Int,
        st_rdev: Int,
        st_flags: Int,
    ):
        """Initialize the stat_result structure.

        Args:
          st_mode: File mode: file type and file mode bits (permissions).
          st_ino: Uniquely identifier for a file.
          st_dev: Identifier of the device on which this file resides.
          st_nlink: Number of hard links.
          st_uid: User identifier of the file owner.
          st_gid: Group identifier of the file owner.
          st_size: Size of the file (bytes), if it is a file or a symlink.
          st_atimespec: Time of file most recent access.
          st_mtimespec: Time of file most recent modification.
          st_ctimespec: Time of file most recent change.
          st_birthtimespec: Time of file creation.
          st_blocks: Number of 512-byte blocks allocated for file.
          st_blksize: Preferred blocksize for efficient file system I/O.
          st_rdev: Type of device if an inode device.
          st_flags: User defined flags for file.
        """
        self.st_mode = st_mode
        self.st_ino = st_ino
        self.st_dev = st_dev
        self.st_nlink = st_nlink
        self.st_uid = st_uid
        self.st_gid = st_gid
        self.st_size = st_size
        self.st_atimespec = st_atimespec
        self.st_mtimespec = st_mtimespec
        self.st_ctimespec = st_ctimespec
        self.st_birthtimespec = st_birthtimespec
        self.st_blocks = st_blocks
        self.st_blksize = st_blksize
        self.st_rdev = st_rdev
        self.st_flags = st_flags

    fn __str__(self) -> String:
        """Constructs a string representation of stat_result.

        Returns:
          A string representation of stat_result.
        """
        var res = String("os.stat_result(")
        res += "st_mode=" + str(self.st_mode)
        res += ", st_ino=" + str(self.st_ino)
        res += ", st_dev=" + str(self.st_dev)
        res += ", st_nlink=" + str(self.st_nlink)
        res += ", st_uid=" + str(self.st_uid)
        res += ", st_gid=" + str(self.st_gid)
        res += ", st_size=" + str(self.st_size)
        res += ", st_atime=" + str(self.st_atimespec)
        res += ", st_mtime=" + str(self.st_mtimespec)
        res += ", st_ctime=" + str(self.st_ctimespec)
        res += ", st_birthtime=" + str(self.st_birthtimespec)
        res += ", st_blocks=" + str(self.st_blocks)
        res += ", st_blksize=" + str(self.st_blksize)
        res += ", st_rdev=" + str(self.st_rdev)
        res += ", st_flags=" + str(self.st_flags)
        return res + ")"

    fn __repr__(self) -> String:
        """Constructs a representation of stat_result.

        Returns:
          A representation of stat_result.
        """
        return str(self)


# ===----------------------------------------------------------------------=== #
# stat
# ===----------------------------------------------------------------------=== #
fn stat(path: String) raises -> stat_result:
    """Get the status of a file or a file descriptor.

    Args:
      path: The path to the directory.

    Returns:
      Returns the stat_result on the path.
    """
    _constrain_unix()

    @parameter
    if os_is_macos():
        return _stat_macos(path)._to_stat_result()
    elif has_neon():
        return _stat_linux_arm(path)._to_stat_result()
    else:
        return _stat_linux_x86(path)._to_stat_result()


fn stat[pathlike: os.PathLike](path: pathlike) raises -> stat_result:
    """Get the status of a file or a file descriptor.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns the stat_result on the path.
    """
    return stat(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# lstat
# ===----------------------------------------------------------------------=== #
fn lstat(path: String) raises -> stat_result:
    """Get the status of a file or a file descriptor (similar to stat, but does
    not follow symlinks).

    Args:
      path: The path to the directory.

    Returns:
      Returns the stat_result on the path.
    """
    _constrain_unix()

    @parameter
    if os_is_macos():
        return _lstat_macos(path)._to_stat_result()
    elif has_neon():
        return _lstat_linux_arm(path)._to_stat_result()
    else:
        return _lstat_linux_x86(path)._to_stat_result()


fn lstat[pathlike: os.PathLike](path: pathlike) raises -> stat_result:
    """Get the status of a file or a file descriptor (similar to stat, but does
    not follow symlinks).

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns the stat_result on the path.
    """
    return lstat(path.__fspath__())
