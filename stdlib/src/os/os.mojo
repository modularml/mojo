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
"""Implements os methods.

You can import a method from the `os` package. For example:

```mojo
from os import listdir
```
"""

from collections import InlineArray, List
from sys import external_call, is_gpu, os_is_linux, os_is_windows
from sys.ffi import OpaquePointer, c_char

from memory import UnsafePointer

from utils import StringRef

from .path import isdir, split
from .pathlike import PathLike

# TODO move this to a more accurate location once nt/posix like modules are in stdlib
alias sep = "\\" if os_is_windows() else "/"


# ===----------------------------------------------------------------------=== #
# SEEK Constants
# ===----------------------------------------------------------------------=== #


alias SEEK_SET: UInt8 = 0
"""Seek from the beginning of the file."""
alias SEEK_CUR: UInt8 = 1
"""Seek from the current position."""
alias SEEK_END: UInt8 = 2
"""Seek from the end of the file."""


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


@value
struct _dirent_linux:
    alias MAX_NAME_SIZE = 256
    var d_ino: Int64
    """File serial number."""
    var d_off: Int64
    """Seek offset value."""
    var d_reclen: Int16
    """Length of the record."""
    var d_type: Int8
    """Type of file."""
    var name: InlineArray[c_char, Self.MAX_NAME_SIZE]
    """Name of entry."""


@value
struct _dirent_macos:
    alias MAX_NAME_SIZE = 1024
    var d_ino: Int64
    """File serial number."""
    var d_off: Int64
    """Seek offset value."""
    var d_reclen: Int16
    """Length of the record."""
    var d_namlen: Int16
    """Length of the name."""
    var d_type: Int8
    """Type of file."""
    var name: InlineArray[c_char, Self.MAX_NAME_SIZE]
    """Name of entry."""


fn _strnlen(ptr: UnsafePointer[c_char], max: Int) -> Int:
    var offset = 0
    while offset < max and ptr[offset]:
        offset += 1
    return offset


struct _DirHandle:
    """Handle to an open directory descriptor opened via opendir."""

    var _handle: OpaquePointer

    fn __init__(out self, path: String) raises:
        """Construct the _DirHandle using the path provided.

        Args:
          path: The path to open.
        """
        constrained[
            not os_is_windows(), "operation is only available on unix systems"
        ]()

        if not isdir(path):
            raise "the directory '" + path + "' does not exist"

        self._handle = external_call["opendir", OpaquePointer](
            path.unsafe_ptr()
        )

        if not self._handle:
            raise "unable to open the directory '" + path + "'"

    fn __del__(owned self):
        """Closes the handle opened via popen."""
        _ = external_call["closedir", Int32](self._handle)

    fn list(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
          A string containing the output of running the command.
        """

        @parameter
        if os_is_linux():
            return self._list_linux()
        else:
            return self._list_macos()

    fn _list_linux(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
          A string containing the output of running the command.
        """
        var res = List[String]()

        while True:
            var ep = external_call["readdir", UnsafePointer[_dirent_linux]](
                self._handle
            )
            if not ep:
                break
            var name = ep.take_pointee().name
            var name_ptr = name.unsafe_ptr()
            var name_str = StringRef(
                name_ptr, _strnlen(name_ptr, _dirent_linux.MAX_NAME_SIZE)
            )
            if name_str == "." or name_str == "..":
                continue
            res.append(name_str)
            _ = name^

        return res

    fn _list_macos(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
          A string containing the output of running the command.
        """
        var res = List[String]()

        while True:
            var ep = external_call["readdir", UnsafePointer[_dirent_macos]](
                self._handle
            )
            if not ep:
                break
            var name = ep.take_pointee().name
            var name_ptr = name.unsafe_ptr()
            var name_str = StringRef(
                name_ptr, _strnlen(name_ptr, _dirent_macos.MAX_NAME_SIZE)
            )
            if name_str == "." or name_str == "..":
                continue
            res.append(name_str)
            _ = name^

        return res


# ===----------------------------------------------------------------------=== #
# getuid
# ===----------------------------------------------------------------------=== #
fn getuid() -> Int:
    """Retrieve the user ID of the calling process.

    Returns:
        The user ID of the calling process.

    Constraints:
        This function is constrained to run on Linux or macOS operating systems only.
    """
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()
    return int(external_call["getuid", UInt32]())


# ===----------------------------------------------------------------------=== #
# listdir
# ===----------------------------------------------------------------------=== #


fn listdir[PathLike: os.PathLike](path: PathLike) raises -> List[String]:
    """Gets the list of entries contained in the path provided.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns the list of entries in the path provided.
    """
    var dir = _DirHandle(path.__fspath__())
    return dir.list()


# ===----------------------------------------------------------------------=== #
# abort
# ===----------------------------------------------------------------------=== #


@no_inline
fn abort[result: AnyType = NoneType._mlir_type]() -> result:
    """Calls a target dependent trap instruction if available.

    Parameters:
        result: The result type.

    Returns:
        A null result type.
    """

    __mlir_op.`llvm.intr.trap`()

    # We need to satisfy the noreturn checker.
    while True:
        pass


@no_inline
fn abort[
    result: AnyType = NoneType._mlir_type, *, W: Writable
](message: W) -> result:
    """Calls a target dependent trap instruction if available.

    Parameters:
        result: The result type.
        W: The Writable type.

    Args:
        message: The message to include when aborting.

    Returns:
        A null result type.
    """

    @parameter
    if not is_gpu():
        print(message, flush=True)

    return abort[result]()


# ===----------------------------------------------------------------------=== #
# remove/unlink
# ===----------------------------------------------------------------------=== #
fn remove[PathLike: os.PathLike](path: PathLike) raises:
    """Removes the specified file.
    If the path is a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file.

    """
    var fspath = path.__fspath__()
    var error = external_call["unlink", Int32](fspath.unsafe_ptr())

    if error != 0:
        # TODO get error message, the following code prints it
        # var error_str = String("Something went wrong")
        # _ = external_call["perror", UnsafePointer[NoneType]](error_str.unsafe_ptr())
        # _ = error_str
        raise Error("Can not remove file: " + fspath)


fn unlink[PathLike: os.PathLike](path: PathLike) raises:
    """Removes the specified file.
    If the path is a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file.

    """
    remove(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# mkdir/rmdir
# ===----------------------------------------------------------------------=== #


fn mkdir[PathLike: os.PathLike](path: PathLike, mode: Int = 0o777) raises:
    """Creates a directory at the specified path.
    If the directory can not be created an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
      mode: The mode to create the directory with.
    """

    var fspath = path.__fspath__()
    var error = external_call["mkdir", Int32](fspath.unsafe_ptr(), mode)
    if error != 0:
        raise Error("Can not create directory: " + fspath)


def makedirs[
    PathLike: os.PathLike
](path: PathLike, mode: Int = 0o777, exist_ok: Bool = False) -> None:
    """Creates a specified leaf directory along with any necessary intermediate
    directories that don't already exist.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
      mode: The mode to create the directory with.
      exist_ok: Ignore error if `True` and path exists (default `False`).
    """
    head, tail = split(path)
    if not tail:
        head, tail = split(head)
    if head and tail and not os.path.exists(head):
        try:
            makedirs(head, exist_ok=exist_ok)
        except:
            # Defeats race condition when another thread created the path
            pass
        # xxx/newdir/. exists if xxx/newdir exists
        if tail == ".":
            return None
    try:
        mkdir(path, mode)
    except e:
        if not exist_ok:
            raise str(
                e
            ) + "\nset `makedirs(path, exist_ok=True)` to allow existing dirs"
        if not os.path.isdir(path):
            raise "path not created: " + path.__fspath__() + "\n" + str(e)


fn rmdir[PathLike: os.PathLike](path: PathLike) raises:
    """Removes the specified directory.
    If the path is not a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
    """
    var fspath = path.__fspath__()
    var error = external_call["rmdir", Int32](fspath.unsafe_ptr())
    if error != 0:
        raise Error("Can not remove directory: " + fspath)


def removedirs[PathLike: os.PathLike](path: PathLike) -> None:
    """Remove a leaf directory and all empty intermediate ones. Directories
    corresponding to rightmost path segments will be pruned away until either
    the whole path is consumed or an error occurs.  Errors during this latter
    phase are ignored, which occur when a directory was not empty.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
    """
    rmdir(path)
    head, tail = os.path.split(path)
    if not tail:
        head, tail = os.path.split(head)
    while head and tail:
        try:
            rmdir(head)
        except:
            break
        head, tail = os.path.split(head)
