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
"""Implements the os.path operations.

You can import these APIs from the `os.path` package. For example:

```mojo
from os.path import isdir
```
"""

from collections import List
from stat import S_ISDIR, S_ISLNK, S_ISREG
from sys import has_neon, os_is_linux, os_is_macos, os_is_windows

from .. import PathLike
from .._linux_aarch64 import _lstat as _lstat_linux_arm
from .._linux_aarch64 import _stat as _stat_linux_arm
from .._linux_x86 import _lstat as _lstat_linux_x86
from .._linux_x86 import _stat as _stat_linux_x86
from .._macos import _lstat as _lstat_macos
from .._macos import _stat as _stat_macos
from ..fstat import stat
from ..os import sep
from ..env import getenv
from pwd import getpwuid


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
    elif has_neon():
        return int(_stat_linux_arm(path).st_mode)
    else:
        return int(_stat_linux_x86(path).st_mode)


@always_inline
fn _get_lstat_st_mode(path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return int(_lstat_macos(path).st_mode)
    elif has_neon():
        return int(_lstat_linux_arm(path).st_mode)
    else:
        return int(_lstat_linux_x86(path).st_mode)


# ===----------------------------------------------------------------------=== #
# expanduser
# ===----------------------------------------------------------------------=== #


fn _user_home_path(path: String) -> String:
    @parameter
    if os_is_windows():
        return getenv("USERPROFILE")
    else:
        var user_end = path.find(sep, 1)
        if user_end < 0:
            user_end = len(path)
        # Special POSIX syntax for ~[user-name]/path
        if len(path) > 1 and user_end > 1:
            try:
                return pwd.getpwnam(path[1:user_end]).pw_dir
            except:
                return ""
        else:
            var user_home = getenv("HOME")
            # Fallback to password database if `HOME` not set
            if not user_home:
                try:
                    user_home = pwd.getpwuid(getuid()).pw_dir
                except:
                    return ""
            return user_home


fn expanduser[PathLike: os.PathLike, //](path: PathLike) raises -> String:
    """Expands a tilde "~" prefix in `path` to the user's home directory.

    For example, `~/folder` becomes `/home/current_user/folder`. On macOS and
    Linux a path starting with `~user/` will expand to the specified user's home
    directory, so `~user/folder` becomes `/home/user/folder`.

    If the home directory cannot be determined, or the `path` is not prefixed
    with "~", the original path is returned unchanged.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path that is being expanded.

    Returns:
        The expanded path.
    """
    var fspath = path.__fspath__()
    if not fspath.startswith("~"):
        return fspath
    var userhome = _user_home_path(fspath)
    if not userhome:
        return fspath
    var path_split = fspath.split(os.sep, 1)
    # If there is a properly formatted separator, return expanded fspath.
    if len(path_split) == 2:
        return os.path.join(userhome, path_split[1])
    # Path was a single `~` character, return home path
    return userhome


# ===----------------------------------------------------------------------=== #
# isdir
# ===----------------------------------------------------------------------=== #
fn isdir[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path is an existing directory. This follows
    symbolic links, so both islink() and isdir() can be true for the same path.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        True if the path is a directory or a link to a directory and
        False otherwise.
    """
    _constrain_unix()
    var fspath = path.__fspath__()
    try:
        var st_mode = _get_stat_st_mode(fspath)
        if S_ISDIR(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISDIR(_get_lstat_st_mode(fspath))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# isfile
# ===----------------------------------------------------------------------=== #


fn isfile[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Test whether a path is a regular file.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path is a regular file.
    """
    _constrain_unix()
    var fspath = path.__fspath__()
    try:
        var st_mode = _get_stat_st_mode(fspath)
        if S_ISREG(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISREG(_get_lstat_st_mode(fspath))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# islink
# ===----------------------------------------------------------------------=== #
fn islink[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path refers to an existing directory entry that is a
    symbolic link.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        True if the path is a link to a directory and False otherwise.
    """
    _constrain_unix()
    try:
        return S_ISLNK(_get_lstat_st_mode(path.__fspath__()))
    except:
        return False


# ===----------------------------------------------------------------------=== #
# dirname
# ===----------------------------------------------------------------------=== #


fn dirname[PathLike: os.PathLike, //](path: PathLike) -> String:
    """Returns the directory component of a pathname.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to a file.

    Returns:
        The directory component of a pathname.
    """
    var fspath = path.__fspath__()
    alias sep = str(os.sep)
    var i = fspath.rfind(sep) + 1
    var head = fspath[:i]
    if head and head != sep * len(head):
        return head.rstrip(sep)
    return head


# ===----------------------------------------------------------------------=== #
# exists
# ===----------------------------------------------------------------------=== #


fn exists[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path exists.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path exists and is not a broken symbolic link.
    """
    _constrain_unix()
    try:
        _ = _get_stat_st_mode(path.__fspath__())
        return True
    except:
        return False


# ===----------------------------------------------------------------------=== #
# lexists
# ===----------------------------------------------------------------------=== #


fn lexists[PathLike: os.PathLike, //](path: PathLike) -> Bool:
    """Return True if path exists or is a broken symlink.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the directory.

    Returns:
        Returns True if the path exists or is a broken symbolic link.
    """
    _constrain_unix()
    try:
        _ = _get_lstat_st_mode(path.__fspath__())
        return True
    except:
        return False


# ===----------------------------------------------------------------------=== #
# getsize
# ===----------------------------------------------------------------------=== #


fn getsize[PathLike: os.PathLike, //](path: PathLike) raises -> Int:
    """Return the size, in bytes, of the specified path.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to the file.

    Returns:
        The size of the path in bytes.
    """
    return stat(path.__fspath__()).st_size


# ===----------------------------------------------------------------------=== #
# join
# ===----------------------------------------------------------------------=== #


fn join(path: String, *paths: String) -> String:
    """Join two or more pathname components, inserting '/' as needed.
    If any component is an absolute path, all previous path components
    will be discarded.  An empty last part will result in a path that
    ends with a separator.

    Args:
        path: The path to join.
        paths: The paths to join.

    Returns:
        The joined path.
    """
    var joined_path = path

    for cur_path in paths:
        if cur_path[].startswith(sep):
            joined_path = cur_path[]
        elif not joined_path or path.endswith(sep):
            joined_path += cur_path[]
        else:
            joined_path += sep + cur_path[]

    return joined_path


# ===----------------------------------------------------------------------=== #
# split
# ===----------------------------------------------------------------------=== #


def split[PathLike: os.PathLike, //](path: PathLike) -> (String, String):
    """
    Split a given pathname into two components: head and tail. This is useful
    for separating the directory path from the filename. If the input path ends
    with a separator, the tail component will be empty. If there is no separator
    in the path, the head component will be empty, and the entire path will be
    considered the tail. Trailing separators in the head are stripped unless the
    head is the root directory.

    Parameters:
        PathLike: The type conforming to the os.PathLike trait.

    Args:
        path: The path to be split.

    Returns:
        A tuple containing two strings: (head, tail).
    """
    fspath = path.__fspath__()
    i = fspath.rfind(os.sep) + 1
    head, tail = fspath[:i], fspath[i:]
    if head and head != str(os.sep) * len(head):
        head = head.rstrip(sep)
    return head, tail


# TODO uncomment this when unpacking is supported
# fn join[PathLike: os.PathLike](path: PathLike, *paths: PathLike) -> String:
#     """Join two or more pathname components, inserting '/' as needed.
#     If any component is an absolute path, all previous path components
#     will be discarded.  An empty last part will result in a path that
#     ends with a separator.

#     Parameters:
#       PathLike: The type conforming to the os.PathLike trait.

#     Args:
#       path: The path to join.
#       paths: The paths to join.

#     Returns:
#       The joined path.
#     """
#     var paths_str= List[String]()

#     for cur_path in paths:
#         paths_str.append(cur_path[].__fspath__())

#     return join(path.__fspath__(), *paths_str)
